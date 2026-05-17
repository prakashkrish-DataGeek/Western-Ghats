// =============================================================================
// Script 01: Landsat Multi-Temporal NDVI Export
//            2030 Frontier Risk Digital Twin — Western Ghats, India
// =============================================================================
// Objective : Export annual dry-season NDVI composites (2006–2026) for the
//             Western Ghats using Landsat 5, 8, and 9 surface reflectance.
//             Combined with terrain (SRTM) and settlement proximity (GHSL),
//             these form the feature stack for a Random Forest deforestation
//             risk model in Python.
//
// Landsat missions used:
//   2006–2012  → Landsat 5 TM (LT05) — SR_B4 (NIR), SR_B3 (Red)
//   2013–2022  → Landsat 8 OLI (LC08) — SR_B5 (NIR), SR_B4 (Red)
//   2023–2026  → Landsat 9 OLI-2 (LC09) — SR_B5 (NIR), SR_B4 (Red)
//
// KEY DESIGN: all three missions are band-harmonised to common names
// (NIR, Red) and merged into a single collection BEFORE the per-year
// loop. This avoids the "No band named SR_B4" error caused by calling
// .normalizedDifference() on an empty image when a mission has no data
// for a given year.
//
// Dry-season window: January–April (minimises monsoon cloud cover).
//
// Outputs (Google Drive → folder "WesternGhats_Risk"):
//   (a) WG_Landsat_Annual_NDVI_2006_2026.csv  ← primary Python input
//       Columns: year, longitude, latitude, ndvi, n_images
//   (b) WG_Terrain.csv
//       Columns: longitude, latitude, elevation, slope, dist_settlement
//
// Pipeline:
//   GEE (this script) → data/WG_Landsat_Annual_NDVI_2006_2026.csv
//                      → data/WG_Terrain.csv
//                      → python/wg_01_process_landsat.py
//                      → python/wg_02_rf_risk_map.py
//                      → web/wg_frontier_risk_2030.html
// =============================================================================

// ── 1. REGION OF INTEREST ─────────────────────────────────────────────────────
var roi = ee.Geometry.Rectangle([73.0, 8.0, 78.0, 21.0]);

Map.centerObject(roi, 6);
Map.addLayer(roi, {color: '00FF88'}, 'Western Ghats ROI', true);

// ── 2. TEMPORAL CONFIGURATION ─────────────────────────────────────────────────
var YEARS = ee.List.sequence(2006, 2026);  // 21 annual composites
print('Export years:', YEARS);
print('Dry-season window: Jan 1 – Apr 30 each year');

// ── 3. BAND-HARMONISED PREP FUNCTIONS ────────────────────────────────────────
// Each function: (a) cloud-masks, (b) applies scale factors, and
// (c) selects + renames bands to the common names NIR and Red.
// This allows all three missions to be merged into one collection.

function prepL5(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask)
    .multiply(0.0000275).add(-0.2)
    .select(['SR_B4', 'SR_B3'], ['NIR', 'Red'])   // L5: B4=NIR, B3=Red
    .copyProperties(image, ['system:time_start']);
}

function prepL8(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask)
    .multiply(0.0000275).add(-0.2)
    .select(['SR_B5', 'SR_B4'], ['NIR', 'Red'])   // L8: B5=NIR, B4=Red
    .copyProperties(image, ['system:time_start']);
}

function prepL9(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask)
    .multiply(0.0000275).add(-0.2)
    .select(['SR_B5', 'SR_B4'], ['NIR', 'Red'])   // L9: B5=NIR, B4=Red
    .copyProperties(image, ['system:time_start']);
}

// ── 4. BUILD MERGED HARMONISED COLLECTION ────────────────────────────────────
// Load all three missions over the ROI, apply prep functions, then merge.
// Filtering by date happens later (per year), so GEE only fetches what's needed.

print('Loading and harmonising Landsat 5 / 8 / 9 collections...');

var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(roi)
  .map(prepL5);

var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi)
  .map(prepL8);

var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(roi)
  .map(prepL9);

// Single merged collection with unified NIR / Red bands
var allLandsat = l5.merge(l8).merge(l9);

print('Merged collection size (all years, all passes):', allLandsat.size());

// ── 5. PER-YEAR NDVI COMPOSITE AND SAMPLING ──────────────────────────────────
// For each year: filter merged collection to Jan–Apr, compute median NDVI,
// sample at 5 km grid, and tag each feature with the year.

var makeAnnualNDVI = function(year) {
  year = ee.Number(year);
  var start = ee.Date.fromYMD(year, 1, 1);
  var end   = ee.Date.fromYMD(year, 4, 30);

  var col   = allLandsat.filterDate(start, end);
  var count = col.size();

  // Guard: if no images exist for this year-window, fill with NoData (-9999)
  var ndvi = ee.Image(
    ee.Algorithms.If(
      count.gt(0),
      col.median().normalizedDifference(['NIR', 'Red']).rename('NDVI'),
      ee.Image.constant(-9999).rename('NDVI')
    )
  ).clip(roi);

  // Add pixel lon/lat bands for explicit coordinate export
  var imgWithCoords = ndvi.addBands(ee.Image.pixelLonLat());

  // Sample at 5 km grid (keeps CSV size manageable for the full WG extent)
  var sampled = imgWithCoords.sample({
    region    : roi,
    scale     : 5000,
    projection: 'EPSG:4326',
    seed      : 42,
    geometries: false
  });

  // Attach year and image count as feature properties
  return sampled.map(function(f) {
    return f.set({
      year     : year,
      ndvi     : f.get('NDVI'),
      n_images : count,
      longitude: f.get('longitude'),
      latitude : f.get('latitude')
    });
  });
};

// ── 6. MAP OVER ALL YEARS AND MERGE ──────────────────────────────────────────
print('Building annual NDVI composites for 2006–2026...');

var allYears = ee.FeatureCollection(
  YEARS.map(makeAnnualNDVI)
).flatten();

print('Total sampled features (all years):', allYears.size());
print('First feature sample:', allYears.first());

// ── 7. VISUALISE 2023 COMPOSITE ──────────────────────────────────────────────
var ndvi2023 = allLandsat
  .filterDate('2023-01-01', '2023-04-30')
  .median()
  .normalizedDifference(['NIR', 'Red'])
  .clip(roi);

Map.addLayer(
  ndvi2023,
  {min: -0.1, max: 0.85, palette: ['white', 'yellow', 'green', 'darkgreen']},
  'NDVI 2023 (dry season)',
  true
);

// ── 8. TERRAIN FEATURES ───────────────────────────────────────────────────────
var srtm      = ee.Image('CGIAR/SRTM90_V4').clip(roi);
var elevation = srtm.rename('elevation');
var slope     = ee.Terrain.slope(srtm).rename('slope');

// GHSL Settlement Model Grid — distance to nearest settled area (class ≥ 2)
var ghslSmod    = ee.ImageCollection('JRC/GHSL/P2016/SMOD_POP_GLOBE_V1').first().clip(roi);
var settled     = ghslSmod.gte(2).unmask(0);
var distSettled = settled
  .fastDistanceTransform(512, 'pixels', 'squared_euclidean')
  .sqrt().multiply(1000)          // pixels → approximate metres
  .rename('dist_settlement');

var terrainStack   = elevation.addBands([slope, distSettled])
                              .addBands(ee.Image.pixelLonLat());
var terrainSampled = terrainStack.sample({
  region    : roi,
  scale     : 5000,
  projection: 'EPSG:4326',
  seed      : 42,
  geometries: false
}).map(function(f) {
  return f.set({
    longitude       : f.get('longitude'),
    latitude        : f.get('latitude'),
    elevation       : f.get('elevation'),
    slope           : f.get('slope'),
    dist_settlement : f.get('dist_settlement')
  });
});

print('Terrain features (first sample):', terrainSampled.first());

// ── 9. QA CHART — NDVI TREND AT SILENT VALLEY ────────────────────────────────
// Representative old-growth forest point (Silent Valley NP, Kerala)
var silentValley = ee.Geometry.Point([76.45, 11.07]);

var ndviChart = ui.Chart.image.series({
  imageCollection: allLandsat
    .filterDate('2006-01-01', '2026-04-30')
    .map(function(img) {
      return img.normalizedDifference(['NIR', 'Red'])
                .rename('NDVI')
                .set('system:time_start', img.get('system:time_start'));
    }),
  region : silentValley,
  reducer: ee.Reducer.mean(),
  scale  : 30
}).setOptions({
  title    : 'NDVI Trend — Silent Valley NP (2006–2026, Jan–Apr)',
  hAxis    : {title: 'Date'},
  vAxis    : {title: 'NDVI', minValue: 0, maxValue: 1},
  lineWidth: 2,
  pointSize: 3,
  colors   : ['#22c55e']
});
print(ndviChart);

// ── 10. EXPORTS ───────────────────────────────────────────────────────────────

// (a) Annual NDVI time series — primary Python input
Export.table.toDrive({
  collection : allYears,
  description: 'WG_Landsat_Annual_NDVI_2006_2026',
  folder     : 'WesternGhats_Risk',
  fileFormat : 'CSV',
  selectors  : ['year', 'longitude', 'latitude', 'ndvi', 'n_images']
});

// (b) Terrain features
Export.table.toDrive({
  collection : terrainSampled,
  description: 'WG_Terrain',
  folder     : 'WesternGhats_Risk',
  fileFormat : 'CSV',
  selectors  : ['longitude', 'latitude', 'elevation', 'slope', 'dist_settlement']
});

// (c) 2023 NDVI raster — visual QA
Export.image.toDrive({
  image         : ndvi2023,
  description   : 'WG_NDVI_2023_DrySeason',
  folder        : 'WesternGhats_Risk',
  fileNamePrefix: 'wg_ndvi_2023',
  region        : roi,
  scale         : 500,
  crs           : 'EPSG:4326',
  maxPixels     : 1e10
});

print('');
print('══════════════════════════════════════════════════════');
print('EXPORTS QUEUED — check the Tasks panel (top right)');
print('  1. WG_Landsat_Annual_NDVI_2006_2026.csv');
print('  2. WG_Terrain.csv');
print('  3. wg_ndvi_2023.tif  (visual QA raster)');
print('');
print('After download, place CSVs in data/ and run:');
print('  python python/wg_01_process_landsat.py \\');
print('    --ndvi    data/WG_Landsat_Annual_NDVI_2006_2026.csv \\');
print('    --terrain data/WG_Terrain.csv');
print('══════════════════════════════════════════════════════');
