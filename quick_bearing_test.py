"""Quick test: compare bearing features on a few trips before full rerun."""

import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DATA_FEATURES = Path(os.environ.get("DATA_FEATURES_DIR", "DATA_Feature_sets/"))
CHECKPOINT_DIR = Path("checkpoints/")

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

# Load linked GPS from checkpoint (already has KEY column)
gps_ckpt = CHECKPOINT_DIR / "gps_linked.parquet"
if not gps_ckpt.exists():
    print(f"ERROR: Need {gps_ckpt} — run validate_pipeline.py once first")
    exit(1)

print(f"Loading GPS from {gps_ckpt}...")
gps_all = pd.read_parquet(gps_ckpt)
print(f"  Full GPS: {gps_all.shape}")

# Sample: pick 50 random trips
all_keys = gps_all["KEY"].dropna().unique()
rng = np.random.default_rng(42)
sample_keys = rng.choice(all_keys, size=min(50, len(all_keys)), replace=False)
gps_sample = gps_all[gps_all["KEY"].isin(sample_keys)].copy()
print(f"  Sample: {len(sample_keys)} trips, {len(gps_sample)} GPS points")

# Extract features via modedetect
print("Extracting motion features...")
gps_feat = extract_point_features(gps_sample, config)
new_stats = compute_trip_statistics(gps_feat, config)

# Compare
orig_sub = df_orig[df_orig["TRIP_KEY"].isin(sample_keys)].set_index("TRIP_KEY")
new_sub = new_stats.set_index("trip_key")
common = orig_sub.index.intersection(new_sub.index)
print(f"\nCommon trips: {len(common)}")

if len(common) == 0:
    print("No common trips found!")
    print(f"  Orig keys sample: {list(orig_sub.index[:5])}")
    print(f"  New keys sample: {list(new_sub.index[:5])}")
    exit(1)

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
    ("DIST_TOTAL_WGS84", "DIST_TOTAL_WGS84"),
]

print(f"\n{'Feature':<22} {'Max Diff':>12} {'Mean Diff':>12} {'Correlation':>12}")
print("-" * 60)
for orig_col, new_col in compare:
    if orig_col in orig_sub.columns and new_col in new_sub.columns:
        o_vals = orig_sub.loc[common, orig_col].astype(float)
        n_vals = new_sub.loc[common, new_col].astype(float)
        diff = (o_vals - n_vals).abs()
        corr = o_vals.corr(n_vals)
        print(f"{orig_col:<22} {diff.max():>12.4f} {diff.mean():>12.4f} {corr:>12.4f}")

# Show per-trip detail for worst bearing feature
print("\n--- Per-trip detail for BC_AVG_WGS84 (first 10 common trips) ---")
print(f"{'Trip key':<25} {'Original':>12} {'Modedetect':>12} {'Diff':>12}")
print("-" * 63)
for tk in list(common)[:10]:
    o = orig_sub.loc[tk, "BC_AVG_WGS84"]
    n = new_sub.loc[tk, "BC_WGS84_AVG"]
    d = abs(float(o) - float(n))
    print(f"{str(tk)[:25]:<25} {float(o):>12.4f} {float(n):>12.4f} {d:>12.4f}")
