"""Validation script: compare modedetect package output against the original pipeline.

Run this on the server where the data lives. Proceeds in 3 levels:
  Level 1 — Feature comparison (are the pre-generated features reproducible?)
  Level 2 — Classification comparison (train both ways, compare F1)
  Level 3 — Confusion matrix comparison (same error patterns?)

Requires:
  - Pre-generated feature parquets in DATA_Feature_sets/
  - Raw GPS CSVs in gps_dataset/
  - trips_dataset.csv
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — loaded from .env file (see .env.example)
# ─────────────────────────────────────────────────────────────────────

load_dotenv()

CHECKPOINT_DIR = Path("checkpoints/")
DATA_FEATURES = Path(os.environ.get("DATA_FEATURES_DIR", "DATA_Feature_sets/"))
GPS_DIR = Path(os.environ["GPS_DIR"])
TRIPS_PATH = Path(os.environ["TRIPS_PATH"])
GTFS_DIR = Path(os.environ["GTFS_DIR"]) if os.environ.get("GTFS_DIR") else None
OSM_ROUTES_DIR = Path(os.environ["OSM_ROUTES_DIR"]) if os.environ.get("OSM_ROUTES_DIR") else None

# ─────────────────────────────────────────────────────────────────────
# ORIGINAL PIPELINE: Mode filtering and speed thresholds
# ─────────────────────────────────────────────────────────────────────

MODE_REMAP = {
    "PRIV_CAR_DRIVER": "CAR",
    "PRIV_CAR_PASSENGER": "CAR",
    "LIGHT_COMM_VEHICLE": "CAR",
    "TAXI": "CAR",
    "TRAIN_EXPRESS": "TRAIN",
}
MODE_DROP = ["OTHER", "PLANE", "ON_DEMAND", "ELECT_BIKE", "ELECT_SCOOTER", "TWO_WHEELER"]

SPEED_THRESHOLDS = {
    "WALKING": 6.5, "BIKE": 10, "CAR": 39, "BUS": 14,
    "TRAMWAY": 19.5, "SUBWAY": 11.1, "TRAIN": 61,
}


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _remap_and_drop(df, mode_col="MAIN_MODE"):
    """Apply mode consolidation and drop rare modes."""
    df = df.copy()
    df[mode_col] = df[mode_col].replace(MODE_REMAP)
    df = df[~df[mode_col].isin(MODE_DROP)]
    return df


def _print_cm(cm, labels, signed=False):
    """Pretty-print a confusion matrix."""
    fmt = "{:>+6d}" if signed else "{:>6d}"
    header = "  " + " " * 12 + "".join(f"{l:>9s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"  {label:<12}" + "".join(fmt.format(int(v)) for v in cm[i])
        print(row)


# ─────────────────────────────────────────────────────────────────────
# Load and filter pre-generated features (original pipeline output)
# ─────────────────────────────────────────────────────────────────────

def load_original_features():
    """Load pre-generated feature parquets and apply the original notebook's filtering."""
    print("=" * 60)
    print("Loading pre-generated features from DATA_Feature_sets/")
    print("=" * 60)

    df_stats = pd.read_parquet(DATA_FEATURES / "base_motion_feature_set.parquet")
    df_pct = pd.read_parquet(DATA_FEATURES / "motion_feature_percentile_map.parquet")
    df_stops = pd.read_parquet(DATA_FEATURES / "stop_feature_set_1000m.parquet")
    df_routes = pd.read_parquet(DATA_FEATURES / "route_match_feature_set.parquet")

    print(f"  stats: {df_stats.shape}, pct: {df_pct.shape}, stops: {df_stops.shape}, routes: {df_routes.shape}")

    # Print column names for debugging
    print(f"\n  stats columns: {list(df_stats.columns[:10])}...")
    print(f"  pct columns: {list(df_pct.columns[:10])}...")
    print(f"  stops columns: {list(df_stops.columns[:10])}...")
    print(f"  routes columns: {list(df_routes.columns[:10])}...")

    # Identify key column name (TRIP_KEY or KEY)
    trip_key_col = "TRIP_KEY" if "TRIP_KEY" in df_stats.columns else "KEY"
    mode_col = "MAIN_MODE" if "MAIN_MODE" in df_stats.columns else "main_mode"
    multi_col = "MULTI_MODAL" if "MULTI_MODAL" in df_stats.columns else None

    print(f"\n  Using trip key: '{trip_key_col}', mode: '{mode_col}', multi: '{multi_col}'")

    # Filter: single-modal only
    if multi_col and multi_col in df_stats.columns:
        df_stats = df_stats[df_stats[multi_col] == False].copy()
        if multi_col in df_pct.columns:
            df_pct = df_pct[df_pct[multi_col] == False].copy()
        if multi_col in df_stops.columns:
            df_stops = df_stops[df_stops[multi_col] == False].copy()
    print(f"  After single-modal filter: {len(df_stats)} trips")

    # Mode remap + drop
    df_stats = _remap_and_drop(df_stats, mode_col)
    df_pct = _remap_and_drop(df_pct, mode_col) if mode_col in df_pct.columns else df_pct
    df_stops = _remap_and_drop(df_stops, mode_col) if mode_col in df_stops.columns else df_stops

    # Temporal filter (>180s)
    time_col = "TOTAL_TIME" if "TOTAL_TIME" in df_stats.columns else None
    if time_col:
        df_stats = df_stats[df_stats[time_col] > 180]
        print(f"  After temporal filter: {len(df_stats)} trips")

    # Speed filter on average speed
    speed_avg_col = "SPEED_AVG" if "SPEED_AVG" in df_stats.columns else None
    if speed_avg_col:
        per_row_thr = df_stats[mode_col].map(SPEED_THRESHOLDS)
        df_stats = df_stats[~(df_stats[speed_avg_col] > per_row_thr)]
        print(f"  After avg speed filter: {len(df_stats)} trips")

    # Sync keys across all datasets
    keep_keys = set(df_stats[trip_key_col])
    df_pct = df_pct[df_pct[trip_key_col].isin(keep_keys)]
    df_stops = df_stops[df_stops[trip_key_col].isin(keep_keys)]

    # Speed filter on P85
    if "SPEED_P85" in df_pct.columns and mode_col in df_pct.columns:
        per_row_thr_p85 = df_pct[mode_col].map(SPEED_THRESHOLDS)
        df_pct = df_pct[~(df_pct["SPEED_P85"] > per_row_thr_p85)]
        keep_keys_p = set(df_pct[trip_key_col])
        df_stats = df_stats[df_stats[trip_key_col].isin(keep_keys_p)]
        df_stops = df_stops[df_stops[trip_key_col].isin(keep_keys_p)]
        print(f"  After P85 speed filter: {len(df_stats)} trips")

    return df_stats, df_pct, df_stops, df_routes, trip_key_col, mode_col


def build_original_feature_matrix(df_stats, df_pct, df_stops, df_routes, trip_key_col, mode_col):
    """Merge features exactly as the original notebook does for the final model."""
    print("\n  Building original feature matrix...")

    # Motion features
    motion_cols = [
        trip_key_col, mode_col, "TOTAL_TIME",
        "AVG_PT_DIST_WGS84", "MED_PT_DIST_WGS84", "DIST_TOTAL_WGS84", "GPS_POINTS",
        "SPEED_AVG", "SPEED_MED", "SPEED_STD",
        "ACC_AVG", "ACC_MED", "ACC_STD",
        "JERK_AVG", "JERK_MED", "JERK_STD",
        "BC_AVG_WGS84", "BC_MED_WGS84", "BC_STD_WGS84",
        "BR_AVG_WGS84", "BR_MED_WGS84", "BR_STD_WGS84",
    ]
    df_motion = df_stats[[c for c in motion_cols if c in df_stats.columns]].copy()

    # Optimal percentiles (from original notebook's JS divergence selection)
    optimal_pct_cols = [trip_key_col, "ACC_P95", "BC_WGS84_P50", "BR_WGS84_P50", "JERK_P95", "SPEED_P95"]
    available_pct = [c for c in optimal_pct_cols if c in df_pct.columns]
    df_opt_pct = df_pct[available_pct].copy()

    merged = df_motion.merge(df_opt_pct, on=trip_key_col, how="left", validate="one_to_one")

    # Stop features (basic + optimal percentiles)
    stop_basic_cols = [
        trip_key_col, "STOPS_NUM_AVG", "STOPS_NUM_MAX", "STOPS_NUM_MIN", "STOPS_MINDIST_AVG",
        "DURATION_TRAM", "DURATION_METRO", "DURATION_RAIL", "DURATION_BUS",
        "MIN_DIST_TRAM", "MIN_DIST_METRO", "MIN_DIST_RAIL", "MIN_DIST_BUS",
    ]
    stop_pct_cols = [
        trip_key_col, "STOPS_CLOSEST_BUS_P95", "STOPS_CLOSEST_METRO_P95",
        "STOPS_CLOSEST_RAIL_P20", "STOPS_CLOSEST_TRAM_P15",
        "STOPS_MINDIST_P95", "STOPS_NUM_P5",
    ]

    available_stop_basic = [c for c in stop_basic_cols if c in df_stops.columns]
    available_stop_pct = [c for c in stop_pct_cols if c in df_stops.columns]

    if len(available_stop_basic) > 1:  # more than just TRIP_KEY
        df_stop_basic = df_stops[available_stop_basic].copy()
        merged = merged.merge(df_stop_basic, on=trip_key_col, how="left", validate="one_to_one")

    if len(available_stop_pct) > 1:
        df_stop_pct = df_stops[available_stop_pct].copy()
        merged = merged.merge(df_stop_pct, on=trip_key_col, how="left", validate="one_to_one")

    # Route matching
    route_match_cols = ["bus_match_pct", "tram_match_pct", "train_match_pct", "subway_match_pct"]
    route_key = trip_key_col if trip_key_col in df_routes.columns else "TRIP_KEY"
    available_route = [route_key] + [c for c in route_match_cols if c in df_routes.columns]

    if len(available_route) > 1:
        df_rt = df_routes[available_route].copy()
        if route_key != trip_key_col:
            df_rt = df_rt.rename(columns={route_key: trip_key_col})
        merged = merged.merge(df_rt, on=trip_key_col, how="left")
        for c in route_match_cols:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0)

    # Fill NaN for stop/route features (trips not covered by those datasets)
    fill_cols = [c for c in merged.columns if c not in [trip_key_col, mode_col]]
    n_any_nan = merged[fill_cols].isna().any(axis=1).sum()
    if n_any_nan > 0:
        print(f"  {n_any_nan} rows have NaN in some feature columns — filling with 0")
        merged[fill_cols] = merged[fill_cols].fillna(0)

    print(f"  Final merged shape: {merged.shape}")
    print(f"  Feature columns: {[c for c in merged.columns if c not in [trip_key_col, mode_col]]}")
    return merged


def train_original_model(merged, trip_key_col, mode_col):
    """Train XGBoost exactly like the original notebook (single 80/20 split)."""
    print("\n  Training original-style XGBoost (80/20 split)...")

    X = merged.drop(columns=[trip_key_col, mode_col])
    le = LabelEncoder()
    y = le.fit_transform(merged[mode_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        tree_method="hist",
        n_jobs=-1,
        random_state=0,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  Original-style model: macro F1 = {f1:.4f}")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Features: {X.shape[1]}")

    return clf, le, X_test, y_test, y_pred, list(X.columns)


# ─────────────────────────────────────────────────────────────────────
# MODEDETECT PIPELINE: Run on the same raw data
# ─────────────────────────────────────────────────────────────────────

def link_gps_to_trips(gps_df, trips_df):
    """Assign trip KEY to each GPS point by matching person ID + time windows.

    Replicates the logic from 1_Intial_Setup_Motion_Features/link_netmob25_gps.py
    using per-person groupby + vectorized NumPy searchsorted for speed.
    """
    print("  Linking GPS points to trips via person ID + time windows...")

    # Build start/end times from trips (using Date_EMG as authoritative date)
    trips = trips_df.copy()
    trips["start_time"] = pd.to_datetime(
        trips["Date_EMG"].astype(str) + " " + trips["Time_O"].astype(str),
        format="mixed", dayfirst=False, errors="coerce",
    )
    trips["end_time"] = pd.to_datetime(
        trips["Date_EMG"].astype(str) + " " + trips["Time_D"].astype(str),
        format="mixed", dayfirst=False, errors="coerce",
    )
    trips = trips.dropna(subset=["start_time", "end_time"])

    # Fix trips crossing midnight
    mask = trips["end_time"] < trips["start_time"]
    trips.loc[mask, "end_time"] += pd.Timedelta(days=1)

    # Parse GPS timestamps
    gps_df = gps_df.copy()
    gps_df["timestamp"] = pd.to_datetime(
        gps_df["LOCAL DATETIME"], format="mixed", dayfirst=False, errors="coerce",
    )
    n_before = len(gps_df)
    gps_df = gps_df.dropna(subset=["timestamp"])
    if len(gps_df) < n_before:
        print(f"  Dropped {n_before - len(gps_df)} rows with unparseable timestamps")

    # Pre-index trips by person ID for fast lookup
    trips_by_person = trips.groupby("ID")[["KEY", "start_time", "end_time"]].apply(
        lambda x: x.values.tolist()
    ).to_dict()

    # Process per person using groupby (much faster than filtering full DF)
    gps_df["KEY"] = pd.NA
    t0 = time.time()
    n_linked = 0
    n_persons = 0

    for pid, grp_idx in gps_df.groupby("FILE_NAME", sort=False).groups.items():
        pid_trips = trips_by_person.get(pid)
        if not pid_trips:
            continue
        n_persons += 1
        ts = gps_df.loc[grp_idx, "timestamp"]
        for key, start, end in pid_trips:
            match_mask = (ts >= start) & (ts <= end)
            matched_idx = grp_idx[match_mask]
            if len(matched_idx) > 0:
                gps_df.loc[matched_idx, "KEY"] = key
                n_linked += len(matched_idx)

    # Drop unlinked points
    n_total = len(gps_df)
    gps_df = gps_df[gps_df["KEY"].notna()].copy()

    print(f"  Linked {n_linked:,} of {n_total:,} GPS points ({n_linked/n_total:.1%})")
    print(f"  Matched {n_persons} persons with trips")
    print(f"  Unique trips with GPS data: {gps_df['KEY'].nunique()}")
    print(f"  Linking took {time.time() - t0:.1f}s")

    return gps_df


def load_raw_gps(max_files=None):
    """Load raw GPS CSVs and trips dataset, with trip key linking."""
    print("\n  Loading raw GPS CSVs...")
    csv_files = sorted(GPS_DIR.glob("*.csv"))
    if max_files:
        csv_files = csv_files[:max_files]

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            df["FILE_NAME"] = f.stem
            dfs.append(df)
        except pd.errors.EmptyDataError:
            continue
    gps_df = pd.concat(dfs, ignore_index=True)

    trips_df = pd.read_csv(TRIPS_PATH)
    print(f"  GPS: {gps_df.shape} from {len(csv_files)} files")
    print(f"  Trips: {trips_df.shape}")
    print(f"  GPS columns: {list(gps_df.columns)}")
    print(f"  Trips columns: {list(trips_df.columns[:15])}...")

    # Link GPS points to trips
    gps_df = link_gps_to_trips(gps_df, trips_df)

    return gps_df, trips_df


def run_modedetect_pipeline(gps_df, trips_df):
    """Run the modedetect package on the raw GPS data.

    Expects gps_df to already have KEY and timestamp columns
    (from link_gps_to_trips).
    """
    print()
    print("=" * 60)
    print("Running modedetect pipeline...")
    print("=" * 60)

    from modedetect import ModeDetector, ModeDetectConfig

    config = ModeDetectConfig(
        col_latitude="LATITUDE",
        col_longitude="LONGITUDE",
        col_timestamp="timestamp",
        col_trip_key="KEY",
        col_speed_kmh="SPEED",
        col_main_mode="Main_Mode",
        gtfs_dir=GTFS_DIR,
        osm_routes_dir=OSM_ROUTES_DIR,
        gap_offset_s=13.0,
        gap_threshold_s=300.0,
        min_trip_duration_s=180,
        min_gps_points=10,
        stop_search_radius_m=1000,
        route_search_radius_m=20,
        route_densify_spacing_m=10,
        n_folds=5,
        random_state=0,
    )

    print(f"  GPS shape: {gps_df.shape}, columns: {list(gps_df.columns)}")
    print(f"  Trips shape: {trips_df.shape}")

    model = ModeDetector(config)

    t0 = time.time()
    model.fit(gps_df, trips_df, checkpoint_dir=CHECKPOINT_DIR / "fit_steps")
    t_fit = time.time() - t0
    print(f"  fit() completed in {t_fit:.1f}s")

    model.save("models/netmob25_modedetect.zip")
    print("  Model saved to models/netmob25_modedetect.zip")

    return model


# ─────────────────────────────────────────────────────────────────────
# LEVEL 1: Feature comparison
# ─────────────────────────────────────────────────────────────────────

def _compare_table(orig_sub, new_sub, col_mapping, label="Feature"):
    """Print a comparison table for aligned DataFrames."""
    print(f"\n  {label:<30} {'Max Diff':>12} {'Mean Diff':>12} {'Correlation':>12}")
    print("  " + "-" * 68)
    for orig_col, new_col in col_mapping.items():
        if orig_col in orig_sub.columns and new_col in new_sub.columns:
            aligned = orig_sub[[orig_col]].join(new_sub[[new_col]], how="inner",
                                                  lsuffix="_orig", rsuffix="_new")
            diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            print(f"  {orig_col:<30} {diff.max():>12.4f} {diff.mean():>12.4f} {corr:>12.4f}")
        elif orig_col in orig_sub.columns:
            print(f"  {orig_col:<30}  (missing in modedetect as '{new_col}')")
        elif new_col in new_sub.columns:
            print(f"  {orig_col:<30}  (missing in original)")


def level1_compare_features(gps_df, trips_df, df_stats_orig, df_pct_orig,
                            df_stops_orig, df_routes_orig, trip_key_col, mode_col):
    """Compare feature values produced by modedetect vs original pipeline."""
    print()
    print("=" * 60)
    print("LEVEL 1: Feature value comparison")
    print("=" * 60)

    from modedetect.config import ModeDetectConfig
    from modedetect.features.motion import extract_point_features
    from modedetect.features.percentiles import compute_percentile_map
    from modedetect.features.statistics import compute_trip_statistics

    config = ModeDetectConfig(
        col_latitude="LATITUDE",
        col_longitude="LONGITUDE",
        col_timestamp="timestamp",
        col_trip_key="KEY",
        col_speed_kmh="SPEED",
        col_main_mode="Main_Mode",
        gap_offset_s=13.0,
        gap_threshold_s=300.0,
    )

    # ── Load modedetect features from checkpoints ──────────────────────
    ckpt_dir = CHECKPOINT_DIR / "fit_steps"
    stats_ckpt = ckpt_dir / "02_stats.parquet"
    pct_ckpt = ckpt_dir / "02_pct_map.parquet"
    stop_ckpt = ckpt_dir / "03_stop_features_1000m.parquet"
    route_ckpt = ckpt_dir / "04_route_features.parquet"

    t0 = time.time()
    if stats_ckpt.exists() and pct_ckpt.exists():
        print("  Loading modedetect features from fit checkpoints...")
        new_stats = pd.read_parquet(stats_ckpt)
        new_pct = pd.read_parquet(pct_ckpt)
    else:
        print("  Extracting motion features via modedetect (no checkpoint found)...")
        gps_feat = extract_point_features(gps_df, config)
        new_stats = compute_trip_statistics(gps_feat, config, trips_df=trips_df)
        new_pct = compute_percentile_map(gps_feat, config)

    new_stops = pd.read_parquet(stop_ckpt) if stop_ckpt.exists() else None
    new_routes = pd.read_parquet(route_ckpt) if route_ckpt.exists() else None
    print(f"  Feature loading took {time.time() - t0:.1f}s")

    print(f"  modedetect stats: {new_stats.shape}, pct: {new_pct.shape}")
    if new_stops is not None:
        print(f"  modedetect stops: {new_stops.shape}")
    if new_routes is not None:
        print(f"  modedetect routes: {new_routes.shape}")

    # Align column names
    new_stats = new_stats.rename(columns={"trip_key": trip_key_col, "main_mode": mode_col})
    new_pct = new_pct.rename(columns={"trip_key": trip_key_col})
    if new_stops is not None:
        new_stops = new_stops.rename(columns={"trip_key": trip_key_col})
    if new_routes is not None:
        new_routes = new_routes.rename(columns={"trip_key": trip_key_col})

    # ── 1A: Trip Statistics ────────────────────────────────────────────
    print("\n  --- 1A: Trip Statistics ---")
    common_keys = set(df_stats_orig[trip_key_col]) & set(new_stats[trip_key_col])
    print(f"  Common trips: {len(common_keys)} "
          f"(orig: {len(df_stats_orig)}, modedetect: {len(new_stats)})")

    if len(common_keys) == 0:
        print("  WARNING: No common trips found!")
        return

    stats_mapping = {
        "TOTAL_TIME": "TOTAL_TIME",
        "GPS_POINTS": "GPS_POINTS",
        "SPEED_AVG": "SPEED_AVG",
        "SPEED_MED": "SPEED_MED",
        "SPEED_STD": "SPEED_STD",
        "ACC_AVG": "ACC_AVG",
        "ACC_MED": "ACC_MED",
        "ACC_STD": "ACC_STD",
        "JERK_AVG": "JERK_AVG",
        "JERK_MED": "JERK_MED",
        "JERK_STD": "JERK_STD",
        "BC_AVG_WGS84": "BC_WGS84_AVG",
        "BC_MED_WGS84": "BC_WGS84_MED",
        "BC_STD_WGS84": "BC_WGS84_STD",
        "BR_AVG_WGS84": "BR_WGS84_AVG",
        "BR_MED_WGS84": "BR_WGS84_MED",
        "BR_STD_WGS84": "BR_WGS84_STD",
        "DIST_TOTAL_WGS84": "DIST_TOTAL_WGS84",
    }

    orig_sub = df_stats_orig[df_stats_orig[trip_key_col].isin(common_keys)].set_index(trip_key_col)
    new_sub = new_stats[new_stats[trip_key_col].isin(common_keys)].set_index(trip_key_col)
    _compare_table(orig_sub, new_sub, stats_mapping, label="Feature")

    # ── 1B: Percentile Map (ALL columns) ──────────────────────────────
    print("\n  --- 1B: Percentile Map ---")
    common_pct_keys = set(df_pct_orig[trip_key_col]) & set(new_pct[trip_key_col])
    print(f"  Common trips: {len(common_pct_keys)}")

    if len(common_pct_keys) > 0:
        pct_cols = sorted(c for c in df_pct_orig.columns
                          if re.search(r"_P\d+$", c) and c in new_pct.columns)
        if pct_cols:
            orig_pct_sub = df_pct_orig[df_pct_orig[trip_key_col].isin(common_pct_keys)].set_index(trip_key_col)
            new_pct_sub = new_pct[new_pct[trip_key_col].isin(common_pct_keys)].set_index(trip_key_col)
            pct_mapping = {c: c for c in pct_cols}
            _compare_table(orig_pct_sub, new_pct_sub, pct_mapping, label="Percentile")

            # Show columns only in original or only in modedetect
            orig_only = sorted(c for c in df_pct_orig.columns
                               if re.search(r"_P\d+$", c) and c not in new_pct.columns)
            new_only = sorted(c for c in new_pct.columns
                              if re.search(r"_P\d+$", c) and c not in df_pct_orig.columns)
            if orig_only:
                print(f"\n  Percentile cols only in original ({len(orig_only)}): {orig_only[:10]}...")
            if new_only:
                print(f"  Percentile cols only in modedetect ({len(new_only)}): {new_only[:10]}...")

    # ── 1C: Stop Proximity Features ───────────────────────────────────
    print("\n  --- 1C: Stop Proximity Features ---")
    if new_stops is not None and df_stops_orig is not None:
        common_stop_keys = set(df_stops_orig[trip_key_col]) & set(new_stops[trip_key_col])
        print(f"  Common trips: {len(common_stop_keys)} "
              f"(orig: {len(df_stops_orig)}, modedetect: {len(new_stops)})")

        if len(common_stop_keys) > 0:
            orig_stop_sub = df_stops_orig[df_stops_orig[trip_key_col].isin(common_stop_keys)].set_index(trip_key_col)
            new_stop_sub = new_stops[new_stops[trip_key_col].isin(common_stop_keys)].set_index(trip_key_col)

            # Find all numeric columns that exist in both
            stop_compare = {}
            for col in orig_stop_sub.columns:
                if col in new_stop_sub.columns and orig_stop_sub[col].dtype in ("float64", "float32", "int64", "int32"):
                    stop_compare[col] = col
            # Also check common names with different case/format
            name_map = {
                "STOPS_NUM_AVG": "STOPS_NUM_AVG",
                "STOPS_NUM_MAX": "STOPS_NUM_MAX",
                "STOPS_NUM_MIN": "STOPS_NUM_MIN",
                "STOPS_MINDIST_AVG": "STOPS_MINDIST_AVG",
                "MIN_DIST_TRAM": "MIN_DIST_TRAM",
                "MIN_DIST_METRO": "MIN_DIST_METRO",
                "MIN_DIST_RAIL": "MIN_DIST_RAIL",
                "MIN_DIST_BUS": "MIN_DIST_BUS",
                "DURATION_TRAM": "DURATION_TRAM",
                "DURATION_METRO": "DURATION_METRO",
                "DURATION_RAIL": "DURATION_RAIL",
                "DURATION_BUS": "DURATION_BUS",
            }
            for orig_c, new_c in name_map.items():
                if orig_c in orig_stop_sub.columns and new_c in new_stop_sub.columns:
                    stop_compare[orig_c] = new_c

            # Add stop percentile columns
            for col in sorted(orig_stop_sub.columns):
                if re.search(r"_P\d+$", col) and col in new_stop_sub.columns:
                    stop_compare[col] = col

            _compare_table(orig_stop_sub, new_stop_sub, stop_compare, label="Stop Feature")

            # Show columns only in one side
            orig_stop_only = sorted(c for c in orig_stop_sub.select_dtypes("number").columns
                                    if c not in new_stop_sub.columns)
            new_stop_only = sorted(c for c in new_stop_sub.select_dtypes("number").columns
                                   if c not in orig_stop_sub.columns)
            if orig_stop_only:
                print(f"\n  Stop cols only in original ({len(orig_stop_only)}): {orig_stop_only[:15]}")
            if new_stop_only:
                print(f"  Stop cols only in modedetect ({len(new_stop_only)}): {new_stop_only[:15]}")
    else:
        print("  Skipped (checkpoint not available)")

    # ── 1D: Route Matching Features ───────────────────────────────────
    print("\n  --- 1D: Route Matching Features ---")
    if new_routes is not None and df_routes_orig is not None:
        common_route_keys = set(df_routes_orig[trip_key_col]) & set(new_routes[trip_key_col])
        print(f"  Common trips: {len(common_route_keys)} "
              f"(orig: {len(df_routes_orig)}, modedetect: {len(new_routes)})")

        if len(common_route_keys) > 0:
            orig_route_sub = df_routes_orig[df_routes_orig[trip_key_col].isin(common_route_keys)].set_index(trip_key_col)
            new_route_sub = new_routes[new_routes[trip_key_col].isin(common_route_keys)].set_index(trip_key_col)

            route_compare = {}
            for col in orig_route_sub.columns:
                if col in new_route_sub.columns and orig_route_sub[col].dtype in ("float64", "float32"):
                    route_compare[col] = col

            _compare_table(orig_route_sub, new_route_sub, route_compare, label="Route Feature")

            orig_route_only = sorted(c for c in orig_route_sub.select_dtypes("number").columns
                                     if c not in new_route_sub.columns)
            new_route_only = sorted(c for c in new_route_sub.select_dtypes("number").columns
                                    if c not in orig_route_sub.columns)
            if orig_route_only:
                print(f"\n  Route cols only in original: {orig_route_only}")
            if new_route_only:
                print(f"  Route cols only in modedetect: {new_route_only}")
    else:
        print("  Skipped (checkpoint not available)")


# ─────────────────────────────────────────────────────────────────────
# LEVEL 2: Classification comparison
# ─────────────────────────────────────────────────────────────────────

def level2_compare_classification(merged_orig, orig_clf, orig_le, orig_feature_cols,
                                  trip_key_col, mode_col, model_new):
    """Compare classification performance: original-style vs modedetect."""
    print()
    print("=" * 60)
    print("LEVEL 2: Classification comparison")
    print("=" * 60)

    # Original model: evaluate on its own test split
    X_orig = merged_orig[orig_feature_cols]
    y_true = merged_orig[mode_col].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_true)

    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_encoded, test_size=0.2, stratify=y_encoded, random_state=0
    )
    y_test_labels = le.inverse_transform(y_test)

    y_pred_orig = orig_clf.predict(X_test.values)
    y_pred_orig_labels = le.inverse_transform(y_pred_orig)
    f1_orig = f1_score(y_test, y_pred_orig, average="macro")

    # modedetect model: use its internal encoder and feature list
    # Note: modedetect was trained on the SAME raw data but with its own
    # feature extraction and K-fold CV, so exact numbers will differ.
    print(f"\n  Original-style model (80/20 split): macro F1 = {f1_orig:.4f}")
    print(f"  modedetect model was trained with K-fold CV on raw GPS data")
    print(f"  (Different training methodology → expect some difference)")

    if hasattr(model_new, '_model') and model_new._model is not None:
        print(f"\n  modedetect model classes: {list(model_new._encoder.classes_)}")
        print(f"  modedetect features: {len(model_new._feature_list)}")
        print(f"  Original features: {len(orig_feature_cols)}")

        # Show feature overlap
        orig_set = set(orig_feature_cols)
        new_set = set(model_new._feature_list)
        common = orig_set & new_set
        only_orig = orig_set - new_set
        only_new = new_set - orig_set
        print(f"\n  Feature overlap: {len(common)} common, "
              f"{len(only_orig)} only in original, {len(only_new)} only in modedetect")
        if only_orig:
            print(f"  Only in original: {sorted(only_orig)}")
        if only_new:
            print(f"  Only in modedetect: {sorted(only_new)}")
    else:
        print("  WARNING: modedetect model not trained (no _model attribute)")

    return f1_orig


# ─────────────────────────────────────────────────────────────────────
# LEVEL 3: Confusion matrix comparison
# ─────────────────────────────────────────────────────────────────────

def level3_compare_confusion(merged_orig, orig_clf, orig_feature_cols,
                             trip_key_col, mode_col):
    """Show confusion matrix and per-class F1 for the original-style model."""
    print()
    print("=" * 60)
    print("LEVEL 3: Per-class F1 and confusion matrix")
    print("=" * 60)

    X = merged_orig[orig_feature_cols]
    y_true = merged_orig[mode_col].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_true)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=0
    )
    y_test_labels = le.inverse_transform(y_test)
    modes = sorted(le.classes_)

    # Original-style model
    y_pred = orig_clf.predict(X_test.values)
    y_pred_labels = le.inverse_transform(y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n  Original-style model: macro F1 = {f1:.4f}")

    # Per-class F1
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    print(f"\n  {'Mode':<12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>8}")
    print("  " + "-" * 48)
    for mode in modes:
        r = report.get(mode, {})
        print(f"  {mode:<12} {r.get('f1-score', 0):>8.4f} {r.get('precision', 0):>10.4f} "
              f"{r.get('recall', 0):>8.4f} {int(r.get('support', 0)):>8}")

    # Confusion matrix
    print(f"\n  --- Confusion Matrix ---")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=modes)
    _print_cm(cm, modes)

    return f1


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    import logging
    import pickle
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("MODEDETECT VALIDATION PIPELINE")
    print("=" * 60)

    # ── Check paths ──
    if not DATA_FEATURES.exists():
        print(f"ERROR: Feature directory not found: {DATA_FEATURES}")
        print("  Make sure you're running from the repo root (Mode-Detection-from-GPS/)")
        sys.exit(1)
    if not GPS_DIR.exists():
        print(f"ERROR: GPS directory not found: {GPS_DIR}")
        sys.exit(1)
    if not TRIPS_PATH.exists():
        print(f"ERROR: Trips file not found: {TRIPS_PATH}")
        sys.exit(1)

    # ── Step 1: Load and process original features ──
    df_stats, df_pct, df_stops, df_routes, trip_key_col, mode_col = load_original_features()
    merged_orig = build_original_feature_matrix(df_stats, df_pct, df_stops, df_routes, trip_key_col, mode_col)
    orig_clf, orig_le, X_test, y_test, y_pred, feature_cols = train_original_model(merged_orig, trip_key_col, mode_col)

    # ── Step 2: Load raw GPS (with checkpoint) ──
    gps_ckpt = CHECKPOINT_DIR / "gps_linked.parquet"
    if gps_ckpt.exists():
        print(f"\n  Loading GPS from checkpoint: {gps_ckpt}")
        gps_df = pd.read_parquet(gps_ckpt)
        trips_df = pd.read_csv(TRIPS_PATH)
        print(f"  GPS: {gps_df.shape}, Trips: {trips_df.shape}")
    else:
        gps_df, trips_df = load_raw_gps()
        print(f"  Saving GPS checkpoint to {gps_ckpt}...")
        gps_df.to_parquet(gps_ckpt, index=False)

    # ── Step 3: Run modedetect (with checkpoint) ──
    model_ckpt = CHECKPOINT_DIR / "modedetect_model.zip"
    if model_ckpt.exists():
        print(f"\n  Loading modedetect model from checkpoint: {model_ckpt}")
        from modedetect import ModeDetector
        model_new = ModeDetector.load(model_ckpt)
    else:
        model_new = run_modedetect_pipeline(gps_df, trips_df)
        import shutil
        if Path("models/netmob25_modedetect.zip").exists():
            shutil.copy("models/netmob25_modedetect.zip", model_ckpt)

    # ── Level 1: Feature comparison ──
    print("\n  Using loaded GPS data for Level 1 comparison...")
    level1_compare_features(gps_df, trips_df, df_stats, df_pct, df_stops, df_routes,
                            trip_key_col, mode_col)

    # ── Level 2: Classification comparison ──
    level2_compare_classification(merged_orig, orig_clf, orig_le, feature_cols,
                                  trip_key_col, mode_col, model_new)

    # ── Level 3: Confusion matrices ──
    level3_compare_confusion(merged_orig, orig_clf, feature_cols, trip_key_col, mode_col)

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
