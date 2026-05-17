/**
 * ==============================================================================
 * WESTERN GHATS FRONTIER RISK DIGITAL TWIN
 * Predictive Modeling of 2030 Forest Loss using Random Forest
 * ==============================================================================
 */

// 1. Define Region of Interest: Western Ghats Boundary (Approximate Bounding Box)
// Note: In a production environment, import the exact WGESA UNESCO shapefile.
var roi = ee.Geometry.Polygon([
  [[73.0, 8.0], [78.0, 8.0], [78.0, 21.0], [73.0, 21.0], [73.0, 8.0]]
]);

// 2. Temporal Variables
var t1Start = '2005-01-01'; var t1End = '2005-12-31'; // Landsat 5 (Historical Baseline)
var t2Start = '2023-01-01'; var t2End = '2023-12-31'; // Landsat 9 (Current State)

// 3. Cloud Masking Functions (Scaling and Masking Bitwise operations)
function maskL5Clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBitMask = 1 << 3;
  var shadowBitMask = 1 << 4;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(shadowBitMask).eq(0));
  return image.updateMask(mask).multiply(0.0000275).add(-0.2)
    .copyProperties(image, ["system:time_start"]);
}

function maskL9Clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBitMask = 1 << 3;
  var shadowBitMask = 1 << 4;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(shadowBitMask).eq(0));
  return image.updateMask(mask).multiply(0.0000275).add(-0.2)
    .copyProperties(image, ["system:time_start"]);
}

// 4. Data Acquisition & Preprocessing
// Landsat 5 (T1) and Landsat 9 (T2) Median Composites
var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(roi).filterDate(t1Start, t1End).map(maskL5Clouds).median().clip(roi);

var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(roi).filterDate(t2Start, t2End).map(maskL9Clouds).median().clip(roi);

// Calculate NDVI for both temporal states
var ndviT1 = l5.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI');
var ndviT2 = l9.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');

// 5. Generate Proxy Historical Deforestation (Training Target)
// Thresholding: Forest at T1 (NDVI > 0.6) transitioning to Non-Forest at T2 (NDVI < 0.4)
var forestT1 = ndviT1.gt(0.6);
var nonForestT2 = ndviT2.lt(0.4);
var deforestation = forestT1.and(nonForestT2).rename('loss');
var stableForest = forestT1.and(ndviT2.gt(0.6)).rename('stable');

// Combine into a single training image (1 = loss transition, 0 = stable forest)
var trainingTarget = ee.Image(1).updateMask(deforestation)
  .unmask(ee.Image(0).updateMask(stableForest)).rename('class');

// 6. Assemble Predictor Variables at T2 (Current State)
// Topography (Elevation and Slope)
var srtm = ee.Image('CGIAR/SRTM90_V4').clip(roi);
var elevation = srtm.rename('elevation');
var slope = ee.Terrain.slope(srtm).rename('slope');

// Infrastructure: Distance to Human Settlements using GHSL Settlement Model Grid
// Using JRC Global Human Settlement Layer (built-in GEE dataset — no special access required)
// as a reliable proxy for infrastructure/road proximity.
// SMOD classes: 1=rural, 2=low-density cluster, 3=urban cluster, 4-6=urban centres
var ghslSmod = ee.ImageCollection('JRC/GHSL/P2016/SMOD_POP_GLOBE_V1').first().clip(roi);
// Any pixel with settlement class >= 2 indicates human presence / infrastructure nearby
var settled = ghslSmod.gte(2).unmask(0);
// Compute Euclidean distance (pixels) then scale to approximate metres at ~1 km native resolution
var distToRoad = settled.fastDistanceTransform(256, 'pixels', 'squared_euclidean')
  .sqrt().multiply(1000).rename('distance_to_road').clip(roi);

// Stack Predictors: Reflectance bands, NDVI, Topography, and Infrastructure Proximity
var predictors = l9.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
  .addBands([ndviT2, elevation, slope, distToRoad]);

// 7. Training the Random Forest Classifier
var trainingData = predictors.addBands(trainingTarget);
var sample = trainingData.stratifiedSample({
  numPoints: 3000,
  classBand: 'class',
  region: roi,
  scale: 30,
  seed: 42
});

// Train the classifier in PROBABILITY mode to predict continuous 2030 risk
var rfClassifier = ee.Classifier.smileRandomForest({numberOfTrees: 100})
  .setOutputMode('PROBABILITY')
  .train({
    features: sample,
    classProperty: 'class',
    inputProperties: predictors.bandNames()
  });

// 8. Apply Model: 2030 Deforestation Probability Map
var risk2030 = predictors.classify(rfClassifier).rename('probability');
// Isolate the risk solely to areas that are currently forested at T2
var futureRiskMap = risk2030.updateMask(ndviT2.gt(0.6));

// 9. User Interface (UI) & Dynamic Area Calculation
var panel = ui.Panel({style: {width: '380px', padding: '15px'}});
var title = ui.Label('2030 Frontier Risk Digital Twin', {fontSize: '22px', fontWeight: 'bold'});
var instructions = ui.Label('Adjust the slider to establish the probability threshold. The viewport and areal calculations will update dynamically to highlight forests at critical risk of clearing by 2030.');
var areaLabel = ui.Label('Calculating area...', {fontSize: '18px', color: '#FF00FF', fontWeight: 'bold'});

Map.add(panel);
panel.add(title).add(instructions).add(areaLabel);

// UI Callback Function for Dynamic Risk Thresholding
var updateRisk = function(threshold) {
  // Extract pixels exceeding the user-defined probability threshold
  var highRisk = futureRiskMap.gte(threshold);
  var riskMasked = futureRiskMap.updateMask(highRisk);

  // Update Visualization Layers
  Map.layers().reset();
  Map.centerObject(roi, 6);
  // True color composite for baseline context
  Map.addLayer(l9, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.15}, 'Landsat 9 Context');
  // Gradient overlay for risk severity
  Map.addLayer(riskMasked, {min: threshold, max: 1, palette: ['yellow', 'orange', '#FF00FF']}, '2030 High Risk Zones');

  // Dynamic Areal Calculation using ee.Image.pixelArea()
  var pixelArea = ee.Image.pixelArea();
  var riskAreaImg = highRisk.multiply(pixelArea);

  riskAreaImg.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 100, // Processed at 100m scale for UI responsiveness
    maxPixels: 1e11
  }).evaluate(function(result) {
    // Guard against null result (e.g. no pixels exceed the threshold, or pipeline error)
    if (!result || result['probability'] === undefined || result['probability'] === null) {
      areaLabel.setValue('No high-risk forest areas at this threshold.');
      return;
    }
    // Convert square metres to hectares
    var hectares = result['probability'] / 10000;
    areaLabel.setValue('At-Risk Forest Area: ' + hectares.toFixed(2) + ' hectares');
  });
};

// Instantiate the ui.Slider widget
var slider = ui.Slider({
  min: 0.5,
  max: 0.95,
  value: 0.75,
  step: 0.05,
  onChange: updateRisk,
  style: {width: '320px'}
});
panel.add(ui.Label('Model Probability Threshold:')).add(slider);

// Initialize Default Application State
updateRisk(0.75);
