"""Quick test: compare bearing features on a few trips before full rerun."""

import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

GPS_DIR = Path(os.environ["GPS_DIR"])
DATA_FEATURES = Path(os.environ.get("DATA_FEATURES_DIR", "DATA_Feature_sets/"))

from modedetect.config import ModeDetectConfig
from modedetect.features.motion import extract_point_features
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

# Load original stats for comparison
print("Loading original stats...")
df_orig = pd.read_parquet(DATA_FEATURES / "base_motion_feature_set.parquet")

# Pick a few trips from the first GPS file
first_file = sorted(GPS_DIR.glob("*.parquet"))[0]
print(f"Loading GPS from {first_file.name}...")
gps_df = pd.read_parquet(first_file)
gps_df["timestamp"] = pd.to_datetime(gps_df["LOCAL DATETIME"])

# Take first 10 trips
trip_keys = gps_df["KEY"].dropna().unique()[:10]
gps_sample = gps_df[gps_df["KEY"].isin(trip_keys)].copy()
print(f"Testing on {len(trip_keys)} trips, {len(gps_sample)} GPS points")

# Extract features via modedetect
print("Extracting motion features...")
gps_feat = extract_point_features(gps_sample, config)
new_stats = compute_trip_statistics(gps_feat, config)

# Compare bearing features
orig_sub = df_orig[df_orig["TRIP_KEY"].isin(trip_keys)].set_index("TRIP_KEY")
new_sub = new_stats.set_index("trip_key")

common = orig_sub.index.intersection(new_sub.index)
print(f"\nCommon trips: {len(common)}")

if len(common) == 0:
    print("No common trips — check column names")
else:
    compare = [
        ("SPEED_AVG", "SPEED_AVG"),
        ("SPEED_MED", "SPEED_MED"),
        ("ACC_AVG", "ACC_AVG"),
        ("BC_AVG_WGS84", "BC_WGS84_AVG"),
        ("BC_MED_WGS84", "BC_WGS84_MED"),
        ("BC_STD_WGS84", "BC_WGS84_STD"),
        ("BR_AVG_WGS84", "BR_WGS84_AVG"),
        ("BR_MED_WGS84", "BR_WGS84_MED"),
        ("BR_STD_WGS84", "BR_WGS84_STD"),
    ]

    print(f"\n{'Feature':<22} {'Orig':>12} {'Modedetect':>12} {'Diff':>12}")
    print("-" * 60)
    for orig_col, new_col in compare:
        if orig_col in orig_sub.columns and new_col in new_sub.columns:
            for tk in common[:5]:  # Show first 5 trips
                o = orig_sub.loc[tk, orig_col]
                n = new_sub.loc[tk, new_col]
                d = abs(o - n) if pd.notna(o) and pd.notna(n) else float('nan')
                print(f"  {tk[:20]:<20} {orig_col:<15} {o:>10.4f} {n:>10.4f} {d:>10.4f}")
            # Summary across all common
            o_vals = orig_sub.loc[common, orig_col]
            n_vals = new_sub.loc[common, new_col]
            diff = (o_vals - n_vals).abs()
            corr = o_vals.corr(n_vals)
            print(f"  {'ALL':<20} {orig_col:<15} max={diff.max():.4f} mean={diff.mean():.4f} corr={corr:.4f}")
            print()
