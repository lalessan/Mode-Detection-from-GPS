"""Validation script: compare modedetect package output against the original pipeline.

Run this on the server where the data lives. Proceeds in 3 levels:
  Level 1 — Feature comparison (are feature values the same?)
  Level 2 — Prediction comparison (do both models agree?)
  Level 3 — Confusion matrix comparison (same error patterns?)

Requires:
  - Original pre-generated feature parquets / CSVs
  - Original trained model (xgb_final_classifier.json)
  - The raw GPS parquets (split_motion_features_061125/)
  - trips_dataset.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — Update these paths to match your server layout
# ─────────────────────────────────────────────────────────────────────

# Original pipeline outputs
ORIG_STATS_PATH = Path("/home/s232534/data/stats_071125/statistical_summary_071125.csv")
ORIG_PERCENTILE_PATH = Path("/home/s232534/data/percentile_pipeline/percentiles_071125.parquet")
ORIG_STOPS_1000M_PATH = Path("/home/s232534/data/stops_analysis_071125/stops_statistical_summary_enriched.parquet")
ORIG_STOPS_250M_PATH = Path("/home/s232534/data/stops_analysis_250m_050126/stops_statistical_summary_250m_050126.csv")
ORIG_ROUTE_MATCH_PATH = Path("/home/s232534/scripts/map_matching/map_matching_interpolated_percentages.csv")
ORIG_MODEL_PATH = Path("/home/s232534/scripts/xgb_final_classifier.json")

# Pre-generated feature sets (alternative to the above)
DATA_FEATURES = Path("Mode-Detection-from-GPS/DATA_Feature_sets/")

# Raw GPS data (for modedetect pipeline)
GPS_DIR = Path("/home/s232534/data/split_motion_features_061125/")
TRIPS_PATH = Path("/home/lauale/netmob2025/data/trips_dataset.csv")

# GTFS and OSM (set to None to skip those features)
GTFS_DIR = Path("/home/s232534/data/gtfs_parquet/relevant/pmbt/")  # or None
OSM_ROUTES_DIR = Path("/home/s232534/scripts/map_matching/osm_routes/")  # or None

# ─────────────────────────────────────────────────────────────────────
# ORIGINAL PIPELINE: Replicate the exact notebook filtering
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


def load_original_features():
    """Load and filter features exactly as the original notebook does."""
    print("=" * 60)
    print("Loading original pipeline features...")
    print("=" * 60)

    # --- Load ---
    df_stats = pd.read_csv(ORIG_STATS_PATH)
    df_pct = pd.read_parquet(ORIG_PERCENTILE_PATH)
    df_stops_1000 = pd.read_parquet(ORIG_STOPS_1000M_PATH)
    df_routes = pd.read_csv(ORIG_ROUTE_MATCH_PATH)

    print(f"  stats: {df_stats.shape}, pct: {df_pct.shape}, stops: {df_stops_1000.shape}, routes: {df_routes.shape}")

    # --- Filter: single-modal only ---
    df_stats = df_stats[df_stats["MULTI_MODAL"] == False].copy()
    df_pct = df_pct[df_pct["MULTI_MODAL"] == False].copy()
    df_stops_1000 = df_stops_1000[df_stops_1000["MULTI_MODAL"] == False].copy()

    # --- Mode remap + drop ---
    for df in [df_stats, df_pct, df_stops_1000]:
        df["MAIN_MODE"] = df["MAIN_MODE"].replace(MODE_REMAP)
        df.drop(df[df["MAIN_MODE"].isin(MODE_DROP)].index, inplace=True)

    # --- Temporal filter (>180s) ---
    df_stats = df_stats[df_stats["TOTAL_TIME"] > 180]
    print(f"  After temporal filter: {len(df_stats)} trips")

    # --- Speed filter on average speed ---
    per_row_thr = df_stats["MAIN_MODE"].map(SPEED_THRESHOLDS)
    df_stats = df_stats[~(df_stats["SPEED_AVG"] > per_row_thr)]
    print(f"  After avg speed filter: {len(df_stats)} trips")

    # --- Sync keys ---
    keep_keys = set(df_stats["TRIP_KEY"])
    df_pct = df_pct[df_pct["TRIP_KEY"].isin(keep_keys)]
    df_stops_1000 = df_stops_1000[df_stops_1000["TRIP_KEY"].isin(keep_keys)]

    # --- Speed filter on P85 ---
    per_row_thr_p85 = df_pct["MAIN_MODE"].map(SPEED_THRESHOLDS)
    df_pct = df_pct[~(df_pct["SPEED_P85"] > per_row_thr_p85)]
    keep_keys_p = set(df_pct["TRIP_KEY"])
    df_stats = df_stats[df_stats["TRIP_KEY"].isin(keep_keys_p)]
    df_stops_1000 = df_stops_1000[df_stops_1000["TRIP_KEY"].isin(keep_keys_p)]
    print(f"  After P85 speed filter: {len(df_stats)} trips")

    return df_stats, df_pct, df_stops_1000, df_routes


def build_original_feature_matrix(df_stats, df_pct, df_stops_1000, df_routes):
    """Merge features exactly as the original notebook does for the final model."""
    # Motion features
    motion_cols = [
        "TRIP_KEY", "MAIN_MODE", "TOTAL_TIME",
        "AVG_PT_DIST_WGS84", "MED_PT_DIST_WGS84", "DIST_TOTAL_WGS84", "GPS_POINTS",
        "SPEED_AVG", "SPEED_MED", "SPEED_STD",
        "ACC_AVG", "ACC_MED", "ACC_STD",
        "JERK_AVG", "JERK_MED", "JERK_STD",
        "BC_AVG_WGS84", "BC_MED_WGS84", "BC_STD_WGS84",
        "BR_AVG_WGS84", "BR_MED_WGS84", "BR_STD_WGS84",
    ]
    df_motion = df_stats[[c for c in motion_cols if c in df_stats.columns]].copy()

    # Optimal percentiles
    optimal_pct_cols = ["TRIP_KEY", "ACC_P95", "BC_WGS84_P50", "BR_WGS84_P50", "JERK_P95", "SPEED_P95"]
    df_opt_pct = df_pct[[c for c in optimal_pct_cols if c in df_pct.columns]].copy()

    merged = df_motion.merge(df_opt_pct, on="TRIP_KEY", how="left", validate="one_to_one")

    # Stop features (1000m basic + optimal percentiles)
    stop_basic_cols = [
        "TRIP_KEY", "STOPS_NUM_AVG", "STOPS_NUM_MAX", "STOPS_NUM_MIN", "STOPS_MINDIST_AVG",
        "DURATION_TRAM", "DURATION_METRO", "DURATION_RAIL", "DURATION_BUS",
        "MIN_DIST_TRAM", "MIN_DIST_METRO", "MIN_DIST_RAIL", "MIN_DIST_BUS",
    ]
    stop_pct_cols = [
        "TRIP_KEY", "STOPS_CLOSEST_BUS_P95", "STOPS_CLOSEST_METRO_P95",
        "STOPS_CLOSEST_RAIL_P20", "STOPS_CLOSEST_TRAM_P15",
        "STOPS_MINDIST_P95", "STOPS_NUM_P5",
    ]

    df_stop_basic = df_stops_1000[[c for c in stop_basic_cols if c in df_stops_1000.columns]].copy()
    df_stop_pct = df_stops_1000[[c for c in stop_pct_cols if c in df_stops_1000.columns]].copy()

    merged = merged.merge(df_stop_basic, on="TRIP_KEY", how="left", validate="one_to_one")
    merged = merged.merge(df_stop_pct, on="TRIP_KEY", how="left", validate="one_to_one")

    # Route matching
    route_cols = ["TRIP_KEY", "bus_match_pct", "tram_match_pct", "train_match_pct", "subway_match_pct"]
    df_rt = df_routes[[c for c in route_cols if c in df_routes.columns]].copy()
    merged = merged.merge(df_rt, on="TRIP_KEY", how="left", validate="one_to_one")
    for c in ["bus_match_pct", "tram_match_pct", "train_match_pct", "subway_match_pct"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    print(f"  Final merged shape: {merged.shape}")
    return merged


def train_original_model(merged):
    """Train XGBoost exactly like the original notebook (single 80/20 split)."""
    from sklearn.model_selection import train_test_split

    X = merged.drop(columns=["TRIP_KEY", "MAIN_MODE"])
    le = LabelEncoder()
    y = le.fit_transform(merged["MAIN_MODE"])

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

    return clf, le, X, y, X_test, y_test, y_pred, list(X.columns)


# ─────────────────────────────────────────────────────────────────────
# MODEDETECT PIPELINE: Run on the same raw data
# ─────────────────────────────────────────────────────────────────────

def run_modedetect_pipeline():
    """Run the modedetect package on the raw GPS data."""
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
        gtfs_dir=GTFS_DIR if GTFS_DIR and GTFS_DIR.exists() else None,
        osm_routes_dir=OSM_ROUTES_DIR if OSM_ROUTES_DIR and OSM_ROUTES_DIR.exists() else None,
        gap_offset_s=13.0,
        gap_threshold_s=300.0,   # original pipeline doesn't drop trips by gap
        min_trip_duration_s=180,
        min_gps_points=10,
        stop_search_radius_m=250,
        route_search_radius_m=20,
        route_densify_spacing_m=10,
        n_folds=5,
        random_state=0,
    )

    # Load GPS data
    print("  Loading GPS parquets...")
    parquets = sorted(GPS_DIR.glob("*.parquet"))
    gps_df = pd.concat([pd.read_parquet(f) for f in parquets], ignore_index=True)
    print(f"  GPS shape: {gps_df.shape}")

    # Load trips
    trips_df = pd.read_csv(TRIPS_PATH)
    print(f"  Trips shape: {trips_df.shape}")

    # Fit
    model = ModeDetector(config)
    model.fit(gps_df, trips_df)
    model.save("models/netmob25_modedetect.zip")

    return model


# ─────────────────────────────────────────────────────────────────────
# LEVEL 1: Feature comparison
# ─────────────────────────────────────────────────────────────────────

def level1_compare_features(gps_df, trips_df, df_stats_orig, df_pct_orig):
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

    # Extract features via modedetect
    print("  Extracting motion features via modedetect...")
    gps_feat = extract_point_features(gps_df, config)
    new_stats = compute_trip_statistics(gps_feat, config, trips_df=trips_df)
    new_pct = compute_percentile_map(gps_feat, config)

    # Align column names for comparison
    # modedetect uses 'trip_key', original uses 'TRIP_KEY'
    new_stats = new_stats.rename(columns={"trip_key": "TRIP_KEY", "main_mode": "MAIN_MODE"})
    new_pct = new_pct.rename(columns={"trip_key": "TRIP_KEY"})

    # Compare trip statistics
    print("\n  --- Trip Statistics Comparison ---")
    common_keys = set(df_stats_orig["TRIP_KEY"]) & set(new_stats["TRIP_KEY"])
    print(f"  Common trips: {len(common_keys)}")

    compare_cols = [
        "TOTAL_TIME", "GPS_POINTS", "SPEED_AVG", "SPEED_MED", "SPEED_STD",
        "ACC_AVG", "ACC_MED", "ACC_STD", "JERK_AVG", "JERK_MED", "JERK_STD",
    ]

    # Map modedetect column names to original names
    col_mapping = {
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
    }

    orig_sub = df_stats_orig[df_stats_orig["TRIP_KEY"].isin(common_keys)].set_index("TRIP_KEY")
    new_sub = new_stats[new_stats["TRIP_KEY"].isin(common_keys)].set_index("TRIP_KEY")

    print(f"\n  {'Feature':<20} {'Max Diff':>12} {'Mean Diff':>12} {'Correlation':>12}")
    print("  " + "-" * 58)
    for orig_col, new_col in col_mapping.items():
        if orig_col in orig_sub.columns and new_col in new_sub.columns:
            aligned = orig_sub[[orig_col]].join(new_sub[[new_col]], lsuffix="_orig", rsuffix="_new")
            diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            print(f"  {orig_col:<20} {diff.max():>12.4f} {diff.mean():>12.4f} {corr:>12.4f}")

    # Compare percentile maps
    print("\n  --- Percentile Map Comparison ---")
    common_pct_keys = set(df_pct_orig["TRIP_KEY"]) & set(new_pct["TRIP_KEY"])
    print(f"  Common trips: {len(common_pct_keys)}")

    pct_cols = [c for c in df_pct_orig.columns if "_P" in c and c in new_pct.columns]
    if pct_cols:
        orig_pct_sub = df_pct_orig[df_pct_orig["TRIP_KEY"].isin(common_pct_keys)].set_index("TRIP_KEY")
        new_pct_sub = new_pct[new_pct["TRIP_KEY"].isin(common_pct_keys)].set_index("TRIP_KEY")

        print(f"\n  {'Pct Column':<20} {'Max Diff':>12} {'Mean Diff':>12} {'Correlation':>12}")
        print("  " + "-" * 58)
        for col in pct_cols[:10]:  # Show first 10
            if col in orig_pct_sub.columns and col in new_pct_sub.columns:
                aligned = orig_pct_sub[[col]].join(new_pct_sub[[col]], lsuffix="_orig", rsuffix="_new")
                diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                print(f"  {col:<20} {diff.max():>12.4f} {diff.mean():>12.4f} {corr:>12.4f}")
        if len(pct_cols) > 10:
            print(f"  ... ({len(pct_cols) - 10} more columns)")


# ─────────────────────────────────────────────────────────────────────
# LEVEL 2: Prediction comparison
# ─────────────────────────────────────────────────────────────────────

def level2_compare_predictions(
    merged_orig, orig_clf, orig_le, orig_feature_cols,
    model_new,
):
    """Compare predictions from original model vs modedetect model."""
    print()
    print("=" * 60)
    print("LEVEL 2: Prediction comparison")
    print("=" * 60)

    # Original model predictions on full dataset
    X_orig = merged_orig[orig_feature_cols]
    y_orig_encoded = orig_clf.predict(X_orig)
    y_orig_labels = orig_le.inverse_transform(y_orig_encoded)

    # modedetect model predictions on the same feature matrix
    # Align columns to modedetect's expected feature list
    X_new = X_orig.reindex(columns=model_new._feature_list, fill_value=0)
    y_new_encoded = model_new._model.predict(X_new.values)
    y_new_labels = model_new._encoder.inverse_transform(y_new_encoded)

    # Compare
    agreement = (y_orig_labels == y_new_labels).mean()
    print(f"  Prediction agreement: {agreement:.1%}")

    # Per-class agreement
    modes = sorted(set(y_orig_labels))
    print(f"\n  {'Mode':<12} {'Agreement':>10} {'Orig Count':>12} {'New Count':>12}")
    print("  " + "-" * 48)
    for mode in modes:
        mask = y_orig_labels == mode
        mode_agree = (y_orig_labels[mask] == y_new_labels[mask]).mean()
        orig_count = (y_orig_labels == mode).sum()
        new_count = (y_new_labels == mode).sum()
        print(f"  {mode:<12} {mode_agree:>10.1%} {orig_count:>12} {new_count:>12}")

    # Load saved original model for comparison too
    if ORIG_MODEL_PATH.exists():
        print("\n  --- Also comparing with saved xgb_final_classifier.json ---")
        saved_clf = XGBClassifier()
        saved_clf.load_model(str(ORIG_MODEL_PATH))
        y_saved = saved_clf.predict(X_orig.values)
        y_saved_labels = orig_le.inverse_transform(y_saved)
        saved_agree = (y_orig_labels == y_saved_labels).mean()
        print(f"  Agreement (retrained vs saved .json): {saved_agree:.1%}")

    return y_orig_labels, y_new_labels


# ─────────────────────────────────────────────────────────────────────
# LEVEL 3: Confusion matrix comparison
# ─────────────────────────────────────────────────────────────────────

def level3_compare_confusion_matrices(
    merged_orig, orig_clf, orig_le, orig_feature_cols,
    model_new,
):
    """Compare confusion matrices and per-class F1 from both models."""
    print()
    print("=" * 60)
    print("LEVEL 3: Confusion matrix and F1 comparison")
    print("=" * 60)

    from sklearn.model_selection import train_test_split

    # Use the same test split as original notebook
    X = merged_orig[orig_feature_cols]
    y_true = merged_orig["MAIN_MODE"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_true)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=0
    )
    y_test_labels = le.inverse_transform(y_test)

    # --- Original model ---
    y_pred_orig = orig_clf.predict(X_test.values)
    y_pred_orig_labels = le.inverse_transform(y_pred_orig)
    f1_orig = f1_score(y_test, y_pred_orig, average="macro")

    # --- modedetect model ---
    X_test_new = X_test.reindex(columns=model_new._feature_list, fill_value=0)
    y_pred_new = model_new._model.predict(X_test_new.values)
    y_pred_new_labels = model_new._encoder.inverse_transform(y_pred_new)
    f1_new = f1_score(y_test_labels, y_pred_new_labels, average="macro")

    print(f"\n  Macro F1 — Original: {f1_orig:.4f}  |  modedetect: {f1_new:.4f}  |  Diff: {f1_new - f1_orig:+.4f}")

    # Per-class comparison
    print("\n  --- Per-class F1 scores ---")
    report_orig = classification_report(y_test_labels, y_pred_orig_labels, output_dict=True)
    report_new = classification_report(y_test_labels, y_pred_new_labels, output_dict=True)

    modes = sorted(set(y_test_labels))
    print(f"\n  {'Mode':<12} {'F1 (orig)':>10} {'F1 (new)':>10} {'Diff':>10}")
    print("  " + "-" * 44)
    for mode in modes:
        f1_o = report_orig.get(mode, {}).get("f1-score", 0)
        f1_n = report_new.get(mode, {}).get("f1-score", 0)
        print(f"  {mode:<12} {f1_o:>10.4f} {f1_n:>10.4f} {f1_n - f1_o:>+10.4f}")

    # Confusion matrices
    print("\n  --- Confusion Matrix (Original) ---")
    cm_orig = confusion_matrix(y_test_labels, y_pred_orig_labels, labels=modes)
    _print_cm(cm_orig, modes)

    print("\n  --- Confusion Matrix (modedetect) ---")
    cm_new = confusion_matrix(y_test_labels, y_pred_new_labels, labels=modes)
    _print_cm(cm_new, modes)

    # Matrix difference
    print("\n  --- Difference (modedetect - original) ---")
    _print_cm(cm_new - cm_orig, modes, signed=True)


def _print_cm(cm, labels, signed=False):
    """Pretty-print a confusion matrix."""
    fmt = "{:>+5d}" if signed else "{:>5d}"
    header = "  " + " " * 12 + "".join(f"{l:>8s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"  {label:<12}" + "".join(fmt.format(int(v)) for v in cm[i])
        print(row)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    # ── Step 1: Load and process original features ──
    df_stats, df_pct, df_stops, df_routes = load_original_features()
    merged_orig = build_original_feature_matrix(df_stats, df_pct, df_stops, df_routes)
    orig_clf, orig_le, X_full, y_full, X_test, y_test, y_pred, feature_cols = train_original_model(merged_orig)

    # ── Step 2: Run modedetect pipeline ──
    model_new = run_modedetect_pipeline()

    # ── Step 3: Load GPS data for Level 1 comparison ──
    print("\n  Loading GPS for feature comparison...")
    parquets = sorted(GPS_DIR.glob("*.parquet"))
    # Load a sample (first 50 files) to keep Level 1 tractable
    sample_files = parquets[:50]
    gps_df = pd.concat([pd.read_parquet(f) for f in sample_files], ignore_index=True)
    trips_df = pd.read_csv(TRIPS_PATH)

    # ── Level 1 ──
    level1_compare_features(gps_df, trips_df, df_stats, df_pct)

    # ── Level 2 ──
    level2_compare_predictions(merged_orig, orig_clf, orig_le, feature_cols, model_new)

    # ── Level 3 ──
    level3_compare_confusion_matrices(merged_orig, orig_clf, orig_le, feature_cols, model_new)

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
