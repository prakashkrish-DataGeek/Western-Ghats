"""
wg_02_rf_risk_map.py
====================
Train a Random Forest classifier on historical Western Ghats deforestation
data and generate an interactive Folium map showing:

  Layer 1 — HeatMapWithTime: annual NDVI change 2006–2026 (20 frames)
  Layer 2 — 2030 Frontier Risk overlay: RF-predicted deforestation probability
  Layer 3 — Western Ghats boundary polygon
  Layer 4 — Key landmarks (hotspots, cities, protected areas)
  Legend, title bar, minimap, fullscreen control

Output:
    web/wg_frontier_risk_2030.html  (~self-contained, publishable)

Usage:
    python python/wg_02_rf_risk_map.py
    python python/wg_02_rf_risk_map.py --input data/wg_features.csv
    python python/wg_02_rf_risk_map.py --input data/wg_features.csv \\
                                        --output web/wg_frontier_risk_2030.html
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime, MiniMap, Fullscreen, MeasureControl, LocateControl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR  = 'data'
WEB_DIR   = 'web'
IN_CSV    = os.path.join(DATA_DIR, 'wg_features.csv')
OUT_HTML  = os.path.join(WEB_DIR, 'wg_frontier_risk_2030.html')

WG_CENTER = [14.0, 75.5]   # [lat, lon] — central Western Ghats
WG_ZOOM   = 6

# RF predictor columns (annual NDVI + temporal statistics + terrain)
PREDICTOR_COLS = [
    'ndvi_2006', 'ndvi_2010', 'ndvi_2015', 'ndvi_2020', 'ndvi_2023',
    'ndvi_trend', 'ndvi_std', 'ndvi_min', 'n_loss_years',
    'elevation', 'slope', 'dist_settlement'
]

NDVI_FOREST_THRESH = 0.50   # Must still be forested in 2023 for prediction

YEARS = list(range(2006, 2027))

# ── Western Ghats boundary (simplified 16-point polygon) ─────────────────────
# Approximated from official WGESA UNESCO boundary.
# Replace with geopandas shapefile for publication-grade mapping.
# Format: [[lat, lon], ...]
WG_BOUNDARY = [
    [21.0, 73.8],   # NW — Gujarat coast
    [20.2, 73.5],
    [19.0, 73.2],
    [17.5, 73.5],
    [16.0, 73.8],
    [15.0, 74.0],
    [14.0, 74.2],
    [13.0, 74.8],
    [12.0, 75.5],
    [11.0, 76.2],
    [10.0, 77.0],
    [ 9.0, 77.2],
    [ 8.2, 77.4],   # SE — Kanyakumari
    [ 8.1, 76.5],
    [ 8.5, 75.2],
    [ 9.5, 74.5],
    [11.0, 74.2],
    [12.5, 73.8],
    [14.0, 73.7],
    [16.0, 73.5],
    [18.0, 73.3],
    [20.0, 73.1],
    [21.0, 73.8],   # close ring
]

# ── Key landmarks: [lat, lon, emoji, tooltip] ─────────────────────────────────
LANDMARKS = [
    [11.07, 76.45, '🌿', 'Silent Valley National Park — pristine tropical forest'],
    [10.10, 77.10, '🍃', 'Munnar — tea plantation frontier risk zone'],
    [12.35, 75.75, '☕', 'Coorg / Kodagu — coffee expansion hotspot'],
    [11.40, 76.70, '🏙️', 'Nilgiris / Ooty — urban fringe pressure'],
    [15.30, 74.05, '⛏️', 'Goa mining belt — habitat fragmentation'],
    [16.50, 74.20, '🚜', 'Northern WG Maharashtra — agricultural encroachment'],
    [10.50, 76.00, '🚪', 'Palakkad Gap — natural opening, high agricultural pressure'],
    [ 9.50, 77.25, '🐘', 'Anamalai Tiger Reserve'],
    [14.00, 74.90, '🌊', 'Sharavathi Valley — hydropower impact zone'],
    [12.00, 75.70, '🦁', 'Kudremukh National Park — core biodiversity zone'],
]

# ── Colour gradients ──────────────────────────────────────────────────────────
# NDVI HeatMapWithTime: dark (low NDVI) → green (high NDVI)
NDVI_GRADIENT = {
    0.00: '#1a0a00',
    0.20: '#5c3300',
    0.35: '#a0522d',
    0.50: '#d4a820',
    0.65: '#5dbf20',
    0.80: '#228b22',
    1.00: '#004d00',
}

# Risk heatmap: yellow → orange → digital magenta
RISK_GRADIENT = {
    0.00: '#00000000',   # transparent for low risk
    0.40: '#ffff00',
    0.65: '#ff8800',
    0.85: '#ff3300',
    1.00: '#ff00ff',     # Digital Magenta — extreme risk
}

YEAR_LABELS = [str(y) for y in YEARS]


# ── Random Forest pipeline ─────────────────────────────────────────────────────

def train_random_forest(df: pd.DataFrame):
    """
    Train a Random Forest classifier on labelled pixels and return the model
    along with 2030 deforestation probability predictions for all forested pixels.

    Returns:
        rf          : trained RandomForestClassifier
        df_pred     : DataFrame with all forested pixels + 'risk_2030' column
        metrics     : dict of evaluation metrics
    """
    # ── Resolve available predictor columns ────────────────────────────────────
    available_predictors = [c for c in PREDICTOR_COLS if c in df.columns]
    missing_cols = set(PREDICTOR_COLS) - set(df.columns)
    if missing_cols:
        print(f"  Note: Some predictor columns not found, using subset: {missing_cols}")

    # ── Training subset (labelled pixels only) ────────────────────────────────
    df_labelled = df[df['class'].notna()].copy()
    X = df_labelled[available_predictors].fillna(df[available_predictors].median())
    y = df_labelled['class'].astype(int)

    print(f"  Training samples: {len(X):,}  "
          f"(class=1: {y.sum():,}, class=0: {(y==0).sum():,})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 12,
        min_samples_leaf = 5,
        class_weight     = 'balanced',
        random_state     = 42,
        n_jobs           = -1
    )
    rf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_prob_test = rf.predict_proba(X_test)[:, 1]
    y_pred_test = rf.predict(X_test)
    auc   = roc_auc_score(y_test, y_prob_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)

    metrics = {
        'auc'      : auc,
        'accuracy' : report['accuracy'],
        'precision': report['1']['precision'],
        'recall'   : report['1']['recall'],
        'f1'       : report['1']['f1-score'],
    }

    print(f"\n  ── Model Evaluation ──────────────────────────────")
    print(f"  AUC-ROC   : {auc:.3f}")
    print(f"  Accuracy  : {report['accuracy']:.3f}")
    print(f"  Precision : {report['1']['precision']:.3f}  (deforestation class)")
    print(f"  Recall    : {report['1']['recall']:.3f}")
    print(f"  F1 Score  : {report['1']['f1-score']:.3f}")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.Series(rf.feature_importances_, index=available_predictors)
    importance = importance.sort_values(ascending=False)
    print(f"\n  ── Feature Importance (top 8) ────────────────────")
    for feat, imp in importance.head(8).items():
        bar = '▓' * int(imp * 100)
        print(f"  {feat:<22} {imp:.3f}  {bar}")

    # ── Predict 2030 risk for all currently forested pixels ───────────────────
    forested_mask = df['predict_mask'] == 1
    df_forest = df[forested_mask].copy()
    X_pred = df_forest[available_predictors].fillna(df[available_predictors].median())
    df_forest['risk_2030'] = rf.predict_proba(X_pred)[:, 1]

    print(f"\n  ── 2030 Risk Predictions ─────────────────────────")
    print(f"  Forested pixels predicted: {len(df_forest):,}")
    print(f"  Risk p > 0.50: {(df_forest['risk_2030'] > 0.50).sum():,} pixels "
          f"({(df_forest['risk_2030'] > 0.50).mean()*100:.1f}%)")
    print(f"  Risk p > 0.75: {(df_forest['risk_2030'] > 0.75).sum():,} pixels "
          f"({(df_forest['risk_2030'] > 0.75).mean()*100:.1f}%)")
    print(f"  Mean risk     : {df_forest['risk_2030'].mean():.3f}")

    return rf, df_forest, metrics


# ── Data preparation ───────────────────────────────────────────────────────────

def build_ndvi_heatmap_data(df: pd.DataFrame):
    """
    Build HeatMapWithTime input for annual NDVI frames (2006–2026).
    NDVI is scaled to [0,1] for the gradient renderer.
    Only pixels with meaningful forest cover are included (NDVI > 0.3).

    Returns:
        data_list   : list of 21 lists [[lat, lon, intensity], ...]
        year_labels : list of year strings
    """
    data_list = []
    ndvi_year_cols = sorted(
        [c for c in df.columns if c.startswith('ndvi_20') and len(c)==9]
    )

    for col in ndvi_year_cols:
        subset = df[df[col] > 0.30][['latitude', 'longitude', col]].copy()
        # Normalise NDVI [0.3, 0.9] → [0, 1] for the gradient
        subset['intensity'] = ((subset[col] - 0.30) / 0.60).clip(0, 1)
        pts = subset[['latitude', 'longitude', 'intensity']].values.tolist()
        data_list.append(pts)

    return data_list, [col.split('_')[1] for col in ndvi_year_cols]


def build_risk_heatmap_data(df_risk: pd.DataFrame, threshold: float = 0.40):
    """
    Build static heatmap data for the 2030 risk layer.
    Only pixels with risk > threshold are included.
    """
    subset = df_risk[df_risk['risk_2030'] >= threshold].copy()
    subset['intensity'] = ((subset['risk_2030'] - threshold) / (1.0 - threshold)).clip(0, 1)
    return subset[['latitude', 'longitude', 'intensity']].values.tolist()


# ── HTML helpers ───────────────────────────────────────────────────────────────

def legend_html(metrics: dict) -> str:
    auc_str = f"{metrics.get('auc', 0):.3f}"
    acc_str = f"{metrics.get('accuracy', 0)*100:.1f}%"
    return f"""
<div id="wg-legend" style="
    position: fixed;
    bottom: 60px; left: 12px;
    z-index: 1000;
    background: rgba(8, 12, 24, 0.94);
    color: #dde8f0;
    padding: 15px 17px;
    border-radius: 10px;
    border: 1px solid rgba(100, 200, 100, 0.30);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.65);
    max-width: 240px;
    line-height: 1.55;
">
  <div style="font-size:13px;font-weight:700;color:#80ee80;margin-bottom:9px;">
    🌿 2030 Frontier Risk — Western Ghats
  </div>

  <!-- Risk gradient bar -->
  <div style="font-size:10px;color:#88aa88;margin-bottom:3px;">
    2030 Deforestation Risk
  </div>
  <div style="
    width:100%;height:10px;
    background:linear-gradient(to right,#ffff00,#ff8800,#ff3300,#ff00ff);
    border-radius:4px;margin-bottom:4px;">
  </div>
  <div style="display:flex;justify-content:space-between;
              font-size:9px;color:#99aabb;margin-bottom:10px;">
    <span>Low (40%)</span><span>High</span><span>Extreme (100%)</span>
  </div>

  <!-- NDVI gradient bar -->
  <div style="font-size:10px;color:#88aa88;margin-bottom:3px;">
    Annual NDVI (time slider)
  </div>
  <div style="
    width:100%;height:8px;
    background:linear-gradient(to right,#5c3300,#d4a820,#228b22,#004d00);
    border-radius:4px;margin-bottom:10px;">
  </div>

  <div style="border-top:1px solid rgba(100,200,100,0.20);padding-top:8px;">
    <div>🌍 <b>Region:</b> Western Ghats, India</div>
    <div>📅 <b>Training:</b> Landsat 2006–2015</div>
    <div>🎯 <b>Prediction:</b> 2030 forest loss</div>
    <div>🤖 <b>Model:</b> Random Forest (200 trees)</div>
  </div>

  <div style="margin-top:8px;padding-top:6px;
              border-top:1px solid rgba(100,200,100,0.20);">
    <div>📊 AUC-ROC : <b style="color:#80ee80">{auc_str}</b></div>
    <div>✅ Accuracy : <b style="color:#80ee80">{acc_str}</b></div>
  </div>

  <div style="margin-top:8px;padding-top:6px;
              border-top:1px solid rgba(100,200,100,0.20);
              font-size:9px;color:#e09090;">
    ⚠️ Probability model — not a definitive forecast.<br>
    High values indicate statistically elevated risk<br>
    based on historical deforestation patterns.
  </div>

  <div style="margin-top:7px;font-size:9px;color:#4a6878;">
    by Prakash Krishnamachari · 2026
  </div>
</div>
"""


def title_html() -> str:
    return """
<div style="
    position: fixed;
    top: 10px; left: 50%; transform: translateX(-50%);
    z-index: 1000;
    background: rgba(6, 14, 20, 0.92);
    color: #e0f0e8;
    padding: 10px 24px;
    border-radius: 9px;
    border: 1px solid rgba(80, 200, 100, 0.45);
    font-family: 'Segoe UI', Arial, sans-serif;
    text-align: center;
    pointer-events: none;
    box-shadow: 0 2px 14px rgba(0,120,60,0.40);
    white-space: nowrap;
">
  <div style="font-size:15px;font-weight:700;letter-spacing:0.5px;">
    🌿 WESTERN GHATS FRONTIER RISK DIGITAL TWIN
  </div>
  <div style="font-size:10px;color:#80c8a0;margin-top:3px;">
    Random Forest · Landsat 2006–2026 · 2030 Deforestation Risk Prediction
  </div>
</div>
"""


# ── Map builder ────────────────────────────────────────────────────────────────

def build_map(df: pd.DataFrame, df_risk: pd.DataFrame, metrics: dict) -> folium.Map:
    """Construct and return the full Folium Western Ghats risk map."""

    # ── Base map ──────────────────────────────────────────────────────────────
    m = folium.Map(
        location     = WG_CENTER,
        zoom_start   = WG_ZOOM,
        tiles        = None,
        control_scale= True,
        prefer_canvas= True
    )

    # ── Basemap tiles ─────────────────────────────────────────────────────────
    folium.TileLayer(
        'CartoDB dark_matter',
        name='🌑 Dark Basemap (default)',
        attr='© CartoDB, © OpenStreetMap',
        show=True
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            'https://server.arcgisonline.com/ArcGIS/rest/services/'
            'World_Imagery/MapServer/tile/{z}/{y}/{x}'
        ),
        name='🛰️ Esri Satellite',
        attr='© Esri, Maxar, GeoEye, USGS',
        show=False
    ).add_to(m)

    folium.TileLayer('OpenStreetMap', name='🗺️ OpenStreetMap', show=False).add_to(m)

    # ── Layer 1: 2030 Risk Heatmap (static, always on) ───────────────────────
    print("  Building 2030 RF risk heatmap...")
    risk_pts = build_risk_heatmap_data(df_risk, threshold=0.40)
    risk_layer = folium.FeatureGroup(name='🔥 2030 Frontier Risk (RF)', show=True)
    from folium.plugins import HeatMap
    HeatMap(
        data      = risk_pts,
        name      = '2030 Risk',
        min_opacity = 0.40,
        max_opacity = 0.90,
        radius    = 18,
        blur      = 12,
        gradient  = {0.0: '#ffff00', 0.4: '#ff8800', 0.7: '#ff3300', 1.0: '#ff00ff'},
    ).add_to(risk_layer)
    risk_layer.add_to(m)

    # ── Layer 2: NDVI HeatMapWithTime (2006–2026 annual) ─────────────────────
    print("  Building NDVI time slider (21 annual frames)...")
    ndvi_data, year_labels = build_ndvi_heatmap_data(df)

    HeatMapWithTime(
        data          = ndvi_data,
        index         = year_labels,
        name          = '🌿 NDVI Historical Change (2006–2026)',
        auto_play     = False,
        max_opacity   = 0.80,
        min_opacity   = 0.00,
        radius        = 16,
        blur          = 0.80,
        gradient      = NDVI_GRADIENT,
        display_index = True,
        min_speed     = 0.5,
        max_speed     = 10,
        speed_step    = 0.5,
        position      = 'bottomright'
    ).add_to(m)

    # ── Layer 3: Western Ghats boundary ──────────────────────────────────────
    boundary_layer = folium.FeatureGroup(name='🟩 Western Ghats Boundary', show=True)
    folium.Polygon(
        locations = WG_BOUNDARY,
        color     = '#00FF88',
        weight    = 2.5,
        opacity   = 0.85,
        fill      = False,
        tooltip   = 'Western Ghats UNESCO World Heritage Site (~160,000 km²) · India'
    ).add_to(boundary_layer)
    boundary_layer.add_to(m)

    # ── Layer 4: Landmarks ────────────────────────────────────────────────────
    landmarks_layer = folium.FeatureGroup(name='📍 Key Sites & Risk Zones', show=True)
    for lat, lon, emoji, tip in LANDMARKS:
        folium.Marker(
            location = [lat, lon],
            tooltip  = tip,
            popup    = folium.Popup(
                f'<div style="font-family:sans-serif;font-size:12px;">'
                f'{emoji} <b>{tip}</b></div>',
                max_width=260
            ),
            icon = folium.DivIcon(
                html=(f'<div style="font-size:20px;'
                      f'text-shadow:0 0 6px #000,0 0 12px #000;">{emoji}</div>'),
                icon_size   = (28, 28),
                icon_anchor = (14, 14)
            )
        ).add_to(landmarks_layer)
    landmarks_layer.add_to(m)

    # ── Plugins ──────────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    MiniMap(tile_layer='CartoDB dark_matter', toggle_display=True,
            position='bottomleft').add_to(m)
    MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    Fullscreen(position='topright').add_to(m)

    # ── Overlays ──────────────────────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element(legend_html(metrics)))
    m.get_root().html.add_child(folium.Element(title_html()))

    return m


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train RF model and generate Western Ghats 2030 risk Folium map'
    )
    parser.add_argument('--input',  type=str, default=IN_CSV,
                        help=f'Feature CSV from wg_01 (default: {IN_CSV})')
    parser.add_argument('--output', type=str, default=OUT_HTML,
                        help=f'Output HTML path (default: {OUT_HTML})')
    args = parser.parse_args()

    print("=" * 60)
    print("Western Ghats 2030 Frontier Risk — RF Model + Map Builder")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"ERROR: Feature CSV not found at {args.input}")
        print("Run first: python python/wg_01_process_landsat.py --demo")
        return

    print(f"\n[Load] {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} pixels | columns: {df.shape[1]}")

    # ── Train RF and predict ──────────────────────────────────────────────────
    print("\n[Model] Training Random Forest classifier...")
    rf, df_risk, metrics = train_random_forest(df)

    # Save risk predictions
    risk_out = os.path.join(DATA_DIR, 'wg_risk_predictions.csv')
    df_risk[['longitude', 'latitude', 'risk_2030']].to_csv(risk_out, index=False)
    print(f"\n✓ Risk predictions saved → {risk_out}")

    # ── Build map ─────────────────────────────────────────────────────────────
    print("\n[Map] Constructing Folium map...")
    m = build_map(df, df_risk, metrics)

    # ── Save map ──────────────────────────────────────────────────────────────
    os.makedirs(WEB_DIR, exist_ok=True)
    m.save(args.output)

    size_kb  = os.path.getsize(args.output) / 1024
    n_high   = (df_risk['risk_2030'] > 0.75).sum()
    n_forest = len(df_risk)

    print(f"\n✓ Map saved → {args.output}")
    print(f"  File size           : {size_kb:.0f} KB")
    print(f"  Forested pixels     : {n_forest:,}")
    print(f"  High-risk (p > 0.75): {n_high:,} pixels "
          f"({n_high/n_forest*100:.1f}% of forested area)")
    print(f"\nTo deploy on GitHub Pages:")
    print(f"  1. Copy {args.output} to your repo web/ folder")
    print(f"  2. Update web/index.html iframe src to point to this file")
    print(f"  3. Push and enable GitHub Pages from the /web folder")


if __name__ == '__main__':
    main()
