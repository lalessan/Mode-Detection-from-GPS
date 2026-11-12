"""Compare stop features at 1000m radius against student's pre-generated features."""

import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DATA_FEATURES = Path(os.environ.get("DATA_FEATURES_DIR", "DATA_Feature_sets/"))
CHECKPOINT_DIR = Path("checkpoints/")
GTFS_DIR = Path(os.environ["GTFS_DIR"]) if os.environ.get("GTFS_DIR") else None

from modedetect.config import ModeDetectConfig
from modedetect.features.stops import build_stop_balltree, compute_stop_features, load_gtfs_stops

config = ModeDetectConfig(
    col_latitude="LATITUDE",
    col_longitude="LONGITUDE",
    col_timestamp="timestamp",
    col_trip_key="KEY",
    col_speed_kmh="SPEED",
    col_main_mode="Main_Mode",
    stop_search_radius_m=1000,  # Match student's 1000m
    n_jobs=-1,  # Use all cores for BallTree queries
)

# Load student's 1000m stop features
print("Loading student's 1000m stop features...")
df_orig = pd.read_parquet(DATA_FEATURES / "stop_feature_set_1000m.parquet")
print(f"  Shape: {df_orig.shape}, columns: {len(df_orig.columns)}")

# Load checkpointed GPS
gps_ckpt = CHECKPOINT_DIR / "fit_steps" / "01_gps_motion.parquet"
if not gps_ckpt.exists():
    print(f"ERROR: Need {gps_ckpt}")
    exit(1)

print(f"Loading GPS from {gps_ckpt}...")
gps_df = pd.read_parquet(gps_ckpt)
print(f"  GPS: {gps_df.shape}")

# Build GTFS stop BallTree
print(f"Loading GTFS stops from {GTFS_DIR}...")
stops = load_gtfs_stops(GTFS_DIR)
tree, stops_df = build_stop_balltree(stops)
print(f"  Stops: {len(stops_df)}")

# Compute stop features at 1000m (with checkpoint)
stop_ckpt = CHECKPOINT_DIR / "fit_steps" / "03_stop_features_1000m.parquet"
if stop_ckpt.exists():
    print(f"Loading 1000m stop features from checkpoint: {stop_ckpt}")
    new_stops = pd.read_parquet(stop_ckpt)
else:
    print(f"Computing stop features (radius={config.stop_search_radius_m}m)...")
    t0 = time.time()
    new_stops = compute_stop_features(gps_df, tree, stops_df, config)
    print(f"  Done in {time.time() - t0:.1f}s")
    new_stops.to_parquet(stop_ckpt, index=False)
    print(f"  Saved checkpoint: {stop_ckpt}")
print(f"  Shape: {new_stops.shape}")

# Compare
trip_key_col = "TRIP_KEY"
new_stops = new_stops.rename(columns={"trip_key": trip_key_col})

common_keys = set(df_orig[trip_key_col]) & set(new_stops[trip_key_col])
print(f"\nCommon trips: {len(common_keys)} (orig: {len(df_orig)}, modedetect: {len(new_stops)})")

orig_sub = df_orig[df_orig[trip_key_col].isin(common_keys)].set_index(trip_key_col)
new_sub = new_stops[new_stops[trip_key_col].isin(common_keys)].set_index(trip_key_col)

# Find all comparable numeric columns
compare = {}
for col in orig_sub.columns:
    if col in new_sub.columns and orig_sub[col].dtype in ("float64", "float32", "int64", "int32"):
        compare[col] = col

print(f"\n{'Feature':<30} {'Max Diff':>12} {'Mean Diff':>12} {'Correlation':>12}")
print("-" * 68)
for col in sorted(compare.keys()):
    o = orig_sub[col].astype(float)
    n = new_sub[col].astype(float)
    aligned = pd.DataFrame({"o": o, "n": n}).dropna()
    if len(aligned) == 0:
        continue
    diff = (aligned["o"] - aligned["n"]).abs()
    corr = aligned["o"].corr(aligned["n"])
    print(f"{col:<30} {diff.max():>12.4f} {diff.mean():>12.4f} {corr:>12.4f}")
