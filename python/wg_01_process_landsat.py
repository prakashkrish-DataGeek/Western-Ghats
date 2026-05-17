"""
wg_01_process_landsat.py
========================
Process GEE-exported Landsat NDVI and terrain data for the Western Ghats
2030 Frontier Risk Digital Twin.

Workflow:
  1. Load GEE-exported annual NDVI CSV (2006–2026) + terrain CSV
     — or generate synthetic demo data (no GEE required)
  2. Pivot to wide format: one row per pixel, one column per year
  3. Compute per-pixel temporal features:
       ndvi_trend      (linear regression slope, NDVI/year)
       ndvi_std        (temporal standard deviation)
       ndvi_min        (minimum observed NDVI)
       n_loss_years    (years where NDVI dropped below 0.35)
  4. Create binary training labels from historical transitions:
       class=1  Forest→Deforested  (ndvi_2006 > 0.50 AND ndvi_2015 < 0.35)
       class=0  Stable Forest      (ndvi_2006 > 0.50 AND ndvi_2015 > 0.50)
       class=NaN  Other / not used for training
  5. Export processed feature CSV → data/wg_features.csv

Input (GEE exports):
    data/WG_Landsat_Annual_NDVI_2006_2026.csv
        Columns: year, longitude, latitude, ndvi, n_images
    data/WG_Terrain.csv
        Columns: longitude, latitude, elevation, slope, dist_settlement

Output:
    data/wg_features.csv
    Columns: longitude, latitude,
             ndvi_2006 … ndvi_2026,   (annual NDVI, 21 columns)
             ndvi_trend, ndvi_std, ndvi_min, n_loss_years,
             elevation, slope, dist_settlement,
             class  (0/1/NaN)

Usage:
    # Real GEE data:
    python python/wg_01_process_landsat.py \\
        --ndvi    data/WG_Landsat_Annual_NDVI_2006_2026.csv \\
        --terrain data/WG_Terrain.csv

    # Synthetic demo (generates realistic Western Ghats data):
    python python/wg_01_process_landsat.py --demo

    # Custom output path:
    python python/wg_01_process_landsat.py --demo --output data/wg_features.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = 'data'
OUT_CSV    = os.path.join(DATA_DIR, 'wg_features.csv')

YEARS      = list(range(2006, 2027))   # 21 years
LABEL_T1   = 2006    # baseline forest year
LABEL_T2   = 2015    # deforestation detection year (midpoint label window)
LABEL_T3   = 2023    # current state (for prediction masking)

NDVI_FOREST_THRESH  = 0.50   # NDVI > this → classified as forest
NDVI_NONFOREST_THRESH = 0.35 # NDVI < this → classified as deforested
NDVI_LOSS_THRESH    = 0.35   # annual threshold for counting loss events
NDVI_VALID_RANGE    = (-0.2, 1.0)

# Western Ghats bounding box [lon_min, lat_min, lon_max, lat_max]
WG_BBOX = {
    'lon_min': 73.0, 'lon_max': 78.0,
    'lat_min':  8.0, 'lat_max': 21.0
}
GRID_SPACING = 0.10   # ~11 km grid for demo (fast generation)


# ── Demo data generator ────────────────────────────────────────────────────────

def generate_demo_data() -> tuple:
    """
    Generate synthetic but physically realistic Landsat NDVI and terrain data
    for the Western Ghats.

    Spatial structure:
      • High-elevation western slopes: dense evergreen forest (NDVI 0.72–0.85)
      • Eastern rain-shadow slopes:   mixed deciduous (NDVI 0.50–0.65)
      • Low-lying coastal:            mixed/agriculture (NDVI 0.30–0.55)
      • Palakkad Gap (76°E, 10.5°N):  natural opening, agriculture
      • Urban fringe pixels:          low NDVI, rapid decline

    Temporal structure (2006–2026):
      • Background: slow NDVI decline (−0.002/yr) from climate/land pressure
      • Protected core: slight increase (+0.001/yr) from better management
      • Deforestation hotspots (5 zones): accelerated linear decline

    Deforestation hotspots:
      Zone                        Rate (NDVI/yr)
      ──────────────────────────────────────────
      Coorg/Kodagu coffee belt    −0.018
      Munnar tea-plantation fringe−0.015
      Northern WG (Maharashtra)   −0.022
      Goa mining belt             −0.020
      Nilgiris urban fringe       −0.016
    """
    np.random.seed(42)

    # ── Spatial grid ──────────────────────────────────────────────────────────
    lons = np.arange(WG_BBOX['lon_min'] + GRID_SPACING / 2,
                     WG_BBOX['lon_max'], GRID_SPACING)
    lats = np.arange(WG_BBOX['lat_min'] + GRID_SPACING / 2,
                     WG_BBOX['lat_max'], GRID_SPACING)
    lon_g, lat_g = np.meshgrid(lons, lats)
    n_pts  = lon_g.size
    lons_f = lon_g.flatten()
    lats_f = lat_g.flatten()

    print(f"  Demo grid: {len(lons)} × {len(lats)} = {n_pts:,} pixels")
    print(f"  Years: {YEARS[0]} → {YEARS[-1]}  ({len(YEARS)} annual composites)")

    # ── Spatial NDVI baseline (2006) ──────────────────────────────────────────
    lon_norm = (lons_f - WG_BBOX['lon_min']) / (WG_BBOX['lon_max'] - WG_BBOX['lon_min'])
    lat_norm = (lats_f - WG_BBOX['lat_min']) / (WG_BBOX['lat_max'] - WG_BBOX['lat_min'])

    # Synthetic elevation proxy (higher near western coast)
    elev_proxy = 1200.0 - 1100.0 * lon_norm + 200.0 * lat_norm
    elev_proxy = np.clip(elev_proxy, 50, 2500)

    # Baseline NDVI driven by elevation and latitude (southern WG more forested)
    ndvi_base = 0.72 + 0.08 * (elev_proxy / 2500) + 0.05 * lat_norm
    ndvi_base = np.clip(ndvi_base, 0.30, 0.87)

    # Palakkad Gap: natural low-forest corridor (~76°E, 10.5°N)
    palakkad_dist = np.sqrt((lons_f - 76.0)**2 + (lats_f - 10.5)**2)
    palakkad_mask = palakkad_dist < 0.6
    ndvi_base[palakkad_mask] -= 0.25 * np.exp(-palakkad_dist[palakkad_mask] / 0.3)

    # Coastal lowlands: lower canopy
    coastal_mask = lons_f < 74.5
    ndvi_base[coastal_mask] -= 0.08 * (74.5 - lons_f[coastal_mask]) / 1.5

    # Eastern rain-shadow: drier vegetation
    rainshadow_mask = lons_f > 77.0
    ndvi_base[rainshadow_mask] -= 0.12

    ndvi_base = np.clip(ndvi_base, 0.10, 0.88)

    # ── Synthetic terrain features ─────────────────────────────────────────────
    terrain_elev  = elev_proxy + np.random.normal(0, 30, n_pts)
    terrain_slope = 5.0 + 15.0 * (elev_proxy / 2500) + np.random.normal(0, 2, n_pts)
    terrain_slope = np.clip(terrain_slope, 0, 45)

    # Distance to settlement: closer to eastern edge and cities = smaller
    dist_settle = 8000.0 + 25000.0 * lon_norm + np.random.normal(0, 3000, n_pts)
    dist_settle = np.abs(dist_settle)

    # ── Deforestation hotspot catalogue ──────────────────────────────────────
    # (lon_center, lat_center, radius_deg, ndvi_trend_per_year)
    hotspots = [
        (75.75, 12.35, 0.45, -0.018),   # Coorg / Kodagu coffee belt
        (77.05, 10.10, 0.35, -0.015),   # Munnar tea-plantation fringe
        (74.20, 16.50, 0.55, -0.022),   # Northern WG — Maharashtra pressure
        (74.05, 15.30, 0.40, -0.020),   # Goa mining & resort belt
        (76.70, 11.40, 0.38, -0.016),   # Nilgiris urban fringe (Ooty)
    ]

    # Background slow decline (non-protected land pressure)
    base_trend = -0.0015 + 0.001 * lat_norm  # southern slightly stable

    # ── Build annual NDVI time series ─────────────────────────────────────────
    ndvi_rows = []
    for yr in YEARS:
        t = yr - 2006   # time step (0 = baseline)

        # Seasonal noise (dry-season composites are more stable but not perfect)
        noise = np.random.normal(0, 0.015, n_pts)

        # Annual NDVI: baseline + background trend + noise
        ndvi = ndvi_base + base_trend * t + noise

        # Apply hotspot deforestation trends
        for hs_lon, hs_lat, hs_r, hs_trend in hotspots:
            dist = np.sqrt((lons_f - hs_lon)**2 + (lats_f - hs_lat)**2)
            hs_m = dist < hs_r
            weight = np.exp(-0.5 * (dist[hs_m] / (hs_r * 0.5))**2)
            ndvi[hs_m] += hs_trend * t * weight

        ndvi = np.clip(ndvi, 0.05, 0.92)

        for i in range(n_pts):
            ndvi_rows.append({
                'year'     : yr,
                'longitude': round(float(lons_f[i]), 3),
                'latitude' : round(float(lats_f[i]), 3),
                'ndvi'     : round(float(ndvi[i]), 4),
                'n_images' : np.random.randint(3, 8)
            })

    df_ndvi = pd.DataFrame(ndvi_rows)
    print(f"  NDVI rows generated: {len(df_ndvi):,}")

    # ── Terrain dataframe ─────────────────────────────────────────────────────
    terrain_rows = [{
        'longitude'      : round(float(lons_f[i]), 3),
        'latitude'       : round(float(lats_f[i]), 3),
        'elevation'      : round(float(terrain_elev[i]), 1),
        'slope'          : round(float(terrain_slope[i]), 2),
        'dist_settlement': round(float(dist_settle[i]), 0)
    } for i in range(n_pts)]
    df_terrain = pd.DataFrame(terrain_rows)

    # Save raw demo files
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_ndvi    = os.path.join(DATA_DIR, 'wg_ndvi_raw_demo.csv')
    raw_terrain = os.path.join(DATA_DIR, 'wg_terrain_raw_demo.csv')
    df_ndvi.to_csv(raw_ndvi, index=False)
    df_terrain.to_csv(raw_terrain, index=False)
    print(f"  Demo NDVI saved   : {raw_ndvi}")
    print(f"  Demo terrain saved: {raw_terrain}")

    return df_ndvi, df_terrain


# ── GEE export loaders ─────────────────────────────────────────────────────────

def load_gee_ndvi(csv_path: str) -> pd.DataFrame:
    """Load and validate the GEE annual NDVI export."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded: {csv_path}  ({len(df):,} rows, columns: {list(df.columns)})")

    required = {'year', 'longitude', 'latitude', 'ndvi'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"NDVI CSV missing columns: {missing}. "
                         "Ensure the GEE export uses selectors: "
                         "['year','longitude','latitude','ndvi','n_images']")

    df['year'] = df['year'].astype(int)
    print(f"  Year range  : {df['year'].min()} → {df['year'].max()}")
    print(f"  Unique years: {df['year'].nunique()}")
    print(f"  Grid pixels : {df.groupby(['longitude','latitude']).ngroups:,}")
    return df


def load_gee_terrain(csv_path: str) -> pd.DataFrame:
    """Load and validate the GEE terrain export."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded: {csv_path}  ({len(df):,} rows)")
    required = {'longitude', 'latitude', 'elevation', 'slope', 'dist_settlement'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Terrain CSV missing columns: {missing}")
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically implausible NDVI values."""
    n_before = len(df)
    df = df[df['ndvi'].between(*NDVI_VALID_RANGE)].copy()
    df.dropna(subset=['longitude', 'latitude', 'year', 'ndvi'], inplace=True)
    n_after = len(df)
    if n_before > n_after:
        print(f"  Cleaned {n_before - n_after:,} invalid rows "
              f"({(n_before - n_after)/n_before*100:.1f}%)")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def compute_features(df_ndvi: pd.DataFrame, df_terrain: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot NDVI time series to wide format and compute temporal features.

    Per-pixel features:
      ndvi_{year}      Annual dry-season NDVI (21 columns)
      ndvi_trend       Linear slope (NDVI/year) via OLS regression
      ndvi_std         Temporal standard deviation
      ndvi_min         Minimum observed NDVI over 2006–2026
      n_loss_years     Count of years where NDVI < NDVI_LOSS_THRESH
    """
    print("  Pivoting to wide format...")
    df_wide = df_ndvi.pivot_table(
        index  = ['longitude', 'latitude'],
        columns= 'year',
        values = 'ndvi',
        aggfunc= 'mean'
    ).reset_index()

    # Rename year columns: 2006 → ndvi_2006
    df_wide.columns.name = None
    ndvi_cols = [c for c in df_wide.columns if isinstance(c, int)]
    rename_map = {yr: f'ndvi_{yr}' for yr in ndvi_cols}
    df_wide.rename(columns=rename_map, inplace=True)
    ncols = [rename_map[y] for y in ndvi_cols]

    print(f"  Wide table: {len(df_wide):,} pixels × {len(ncols)} NDVI years")

    # ── Temporal statistics ────────────────────────────────────────────────────
    ndvi_matrix = df_wide[ncols].values
    years_arr   = np.array([int(c.split('_')[1]) for c in ncols], dtype=float)

    # Linear trend (slope) per pixel
    print("  Computing linear NDVI trends (OLS)...")
    slopes = np.full(len(df_wide), np.nan)
    for i in range(len(df_wide)):
        row = ndvi_matrix[i]
        valid = ~np.isnan(row)
        if valid.sum() >= 5:   # need at least 5 observations for a trend
            sl, *_ = stats.linregress(years_arr[valid], row[valid])
            slopes[i] = sl

    df_wide['ndvi_trend']    = slopes
    df_wide['ndvi_std']      = np.nanstd(ndvi_matrix, axis=1)
    df_wide['ndvi_min']      = np.nanmin(ndvi_matrix, axis=1)
    df_wide['n_loss_years']  = (ndvi_matrix < NDVI_LOSS_THRESH).sum(axis=1)

    # ── Merge terrain ──────────────────────────────────────────────────────────
    print("  Merging terrain features...")
    df_terrain_clean = df_terrain.copy()
    # Round coordinates to match grid precision
    df_wide['longitude']        = df_wide['longitude'].round(3)
    df_wide['latitude']         = df_wide['latitude'].round(3)
    df_terrain_clean['longitude'] = df_terrain_clean['longitude'].round(3)
    df_terrain_clean['latitude']  = df_terrain_clean['latitude'].round(3)

    df_out = df_wide.merge(
        df_terrain_clean[['longitude', 'latitude', 'elevation', 'slope', 'dist_settlement']],
        on=['longitude', 'latitude'],
        how='left'
    )

    # Fill missing terrain with regional medians (fallback)
    for col in ['elevation', 'slope', 'dist_settlement']:
        df_out[col].fillna(df_out[col].median(), inplace=True)

    return df_out


# ── Training label generation ──────────────────────────────────────────────────

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate binary training labels from historical forest transition.

    Logic:
      T1 = ndvi_2006  (historical baseline)
      T2 = ndvi_2015  (mid-period transition check)

      class = 1  →  Forest→Deforested:
                     ndvi_2006 > NDVI_FOREST_THRESH
                     AND ndvi_2015 < NDVI_NONFOREST_THRESH

      class = 0  →  Stable Forest:
                     ndvi_2006 > NDVI_FOREST_THRESH
                     AND ndvi_2015 > NDVI_FOREST_THRESH

      class = NaN → All other pixels (not used for training)

    Future risk mask:
      Pixels with ndvi_2023 > NDVI_FOREST_THRESH are still forested and
      eligible for 2030 risk prediction (set as predict_mask=True).
    """
    df = df.copy()
    df['class'] = np.nan

    t1_col = f'ndvi_{LABEL_T1}'
    t2_col = f'ndvi_{LABEL_T2}'
    t3_col = f'ndvi_{LABEL_T3}'

    for col in [t1_col, t2_col, t3_col]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' — check year range in GEE export")

    # Deforested class (1)
    deforested = (df[t1_col] > NDVI_FOREST_THRESH) & (df[t2_col] < NDVI_NONFOREST_THRESH)
    df.loc[deforested, 'class'] = 1

    # Stable forest class (0)
    stable = (df[t1_col] > NDVI_FOREST_THRESH) & (df[t2_col] > NDVI_FOREST_THRESH)
    df.loc[stable, 'class'] = 0

    # Prediction eligibility: still forested at T3 (2023)
    df['predict_mask'] = (df[t3_col] > NDVI_FOREST_THRESH).astype(int)

    n_def    = deforested.sum()
    n_stable = stable.sum()
    n_pred   = df['predict_mask'].sum()

    print(f"\n  Training label breakdown:")
    print(f"    class=1 (deforested)   : {n_def:,} pixels  ({n_def/len(df)*100:.1f}%)")
    print(f"    class=0 (stable forest): {n_stable:,} pixels  ({n_stable/len(df)*100:.1f}%)")
    print(f"    class=NaN (not labelled): {len(df)-n_def-n_stable:,} pixels")
    print(f"    Eligible for 2030 pred : {n_pred:,} pixels")

    return df


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    n_pixels = len(df)
    n_forest = (df.get('predict_mask', pd.Series(dtype=int)) == 1).sum()

    print("\n" + "=" * 60)
    print("Feature Engineering Summary")
    print("=" * 60)
    print(f"  Total pixels          : {n_pixels:,}")
    print(f"  Currently forested    : {n_forest:,}  (ndvi_2023 > {NDVI_FOREST_THRESH})")
    print(f"  Training pixels       : {df['class'].notna().sum():,}")
    print(f"  Class balance         : {(df['class']==1).sum():,} loss / {(df['class']==0).sum():,} stable")
    print()
    ndvi_cols = [c for c in df.columns if c.startswith('ndvi_20') and len(c) == 9]
    if ndvi_cols:
        mean_2006 = df.get('ndvi_2006', pd.Series()).mean()
        mean_2023 = df.get('ndvi_2023', pd.Series()).mean()
        print(f"  Mean NDVI 2006        : {mean_2006:.3f}")
        print(f"  Mean NDVI 2023        : {mean_2023:.3f}")
        print(f"  Mean NDVI change      : {mean_2023 - mean_2006:+.3f}")
    print(f"  Mean NDVI trend       : {df['ndvi_trend'].mean():.5f} NDVI/yr")
    print(f"  Mean elevation        : {df['elevation'].mean():.0f} m")
    print(f"  Mean dist_settlement  : {df['dist_settlement'].mean():.0f} m")

    # Year-by-year mean NDVI drift
    ndvi_cols_sorted = sorted([c for c in df.columns if c.startswith('ndvi_20') and len(c)==9])
    print(f"\n  Mean annual NDVI (selected years):")
    step = max(1, len(ndvi_cols_sorted)//7)
    for col in ndvi_cols_sorted[::step]:
        yr  = col.split('_')[1]
        val = df[col].mean()
        bar = '█' * int(val * 30)
        print(f"    {yr}: {val:.3f}  {bar}")
    print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Process Landsat NDVI data for Western Ghats frontier risk model'
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--demo', action='store_true',
                   help='Generate synthetic demo data (no GEE required)')
    g.add_argument('--ndvi', type=str,
                   help='Path to GEE NDVI CSV (WG_Landsat_Annual_NDVI_2006_2026.csv)')
    parser.add_argument('--terrain', type=str,
                        help='Path to GEE terrain CSV (WG_Terrain.csv) '
                             '[required unless --demo]')
    parser.add_argument('--output', type=str, default=OUT_CSV,
                        help=f'Output feature CSV path (default: {OUT_CSV})')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)

    if args.demo:
        print("DEMO MODE — generating synthetic Western Ghats Landsat data")
        print("=" * 60)
        df_ndvi, df_terrain = generate_demo_data()
    else:
        if not args.terrain:
            parser.error("--terrain is required when not using --demo")
        print(f"Loading GEE NDVI export  : {args.ndvi}")
        print(f"Loading GEE terrain export: {args.terrain}")
        print("=" * 60)
        df_ndvi   = load_gee_ndvi(args.ndvi)
        df_terrain = load_gee_terrain(args.terrain)

    print("\n[Clean] Removing invalid NDVI values...")
    df_ndvi = clean_ndvi(df_ndvi)

    print("\n[Features] Computing temporal features and merging terrain...")
    df_features = compute_features(df_ndvi, df_terrain)

    print("\n[Labels] Generating training labels from historical transitions...")
    df_features = create_labels(df_features)

    print_summary(df_features)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_features.to_csv(args.output, index=False)
    print(f"\n✓ Feature table saved → {args.output}")
    print(f"  Shape: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")
    print(f"\nNext step:")
    print(f"  python python/wg_02_rf_risk_map.py --input {args.output}")


if __name__ == '__main__':
    main()
