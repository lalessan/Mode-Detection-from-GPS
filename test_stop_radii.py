"""Test which stop search radius is most discriminative for each stop type.

For each radius, computes MIN_DIST and DURATION per stop type, then measures
how well each feature separates its "target" mode from others.

Discriminability = ratio of (mean for non-target modes) / (mean for target mode).
Higher = better separation. E.g., for MIN_DIST_METRO, target mode is SUBWAY.

Usage:
    python test_stop_radii.py --sample 0.1   # 10% of trips (fast)
    python test_stop_radii.py                 # all trips
"""
import argparse
import time

import numpy as np
import pandas as pd

from modedetect import ModeDetectConfig
from modedetect.features.stops import (
    build_stop_balltree,
    compute_stop_features,
    load_gtfs_stops,
)

GTFS_DIR = next(
    p for p in [
        __import__("pathlib").Path("gtfs/"),
        __import__("pathlib").Path("/home/lauale/ModeDetection/gtfs/"),
    ] if p.exists()
)

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=float, default=1.0, help="Fraction of trips (0.0-1.0)")
args = parser.parse_args()

print("Loading GTFS stops...")
stops = load_gtfs_stops(GTFS_DIR)
tree, stops_df = build_stop_balltree(stops)

print("Loading GPS data...")
gps = pd.read_parquet("checkpoints/fit_steps/01_gps_motion.parquet")

if args.sample < 1.0:
    trips = gps["KEY"].unique()
    n = int(len(trips) * args.sample)
    sampled = pd.Series(trips).sample(n, random_state=42).values
    gps = gps[gps["KEY"].isin(sampled)]
    print(f"Sampled {n} trips ({len(gps)} GPS points)")
else:
    print(f"Using all trips ({len(gps)} GPS points)")

trips_meta = pd.read_parquet(
    "checkpoints/fit_steps/02_stats.parquet",
    columns=["trip_key", "Main_Mode"],
).drop_duplicates("trip_key")

# ── Configuration ──────────────────────────────────────────────────

RADII = [50, 100, 250, 500, 750, 1000]

# Which feature → which mode it should discriminate
FEATURE_TARGETS = {
    "MIN_DIST_BUS": "BUS",
    "DURATION_BUS": "BUS",
    "MIN_DIST_METRO": "SUBWAY",
    "DURATION_METRO": "SUBWAY",
    "MIN_DIST_RAIL": "TRAIN",
    "DURATION_RAIL": "TRAIN",
    "MIN_DIST_TRAM": "TRAMWAY",
    "DURATION_TRAM": "TRAMWAY",
}

KEY_FEATURES = list(FEATURE_TARGETS.keys())

# ── Run for each radius ───────────────────────────────────────────

results = []

print("\n" + "=" * 70)
print("Stop Feature Discriminability at Different Radii")
print("=" * 70)

for radius in RADII:
    print(f"\n  --- Radius = {radius}m ---")
    t0 = time.time()

    # Use uniform radius for this test (no per-mode filtering)
    cfg = ModeDetectConfig(
        col_latitude="LATITUDE", col_longitude="LONGITUDE",
        col_timestamp="timestamp", col_trip_key="KEY",
        col_speed_kmh="SPEED", col_main_mode="Main_Mode",
        gtfs_dir=GTFS_DIR,
        stop_search_radius_m=radius,
        metro_stop_radius_m=radius,  # same as default for this test
        stop_quantiles=(),  # skip percentiles for speed
    )
    sf = compute_stop_features(gps, tree, stops_df, cfg)
    elapsed = time.time() - t0
    print(f"  Computed in {elapsed:.0f}s")

    m = sf.merge(trips_meta, on="trip_key")
    available = [c for c in KEY_FEATURES if c in m.columns]
    if not available:
        print("  No key features available")
        continue

    # Compute per-mode means
    means = m.groupby("Main_Mode")[available].mean()
    print(f"\n  Mean feature values by mode (radius={radius}m):")
    print(means.round(1).to_string())

    # Compute discriminability for each feature
    print(f"\n  Discriminability (target_mean / others_mean):")
    for feat in available:
        target_mode = FEATURE_TARGETS[feat]
        if target_mode not in means.index:
            continue

        target_val = means.loc[target_mode, feat]
        other_vals = means.loc[means.index != target_mode, feat]
        others_mean = other_vals.mean()

        if feat.startswith("MIN_DIST"):
            # For distance: lower = closer to stops = better
            # Discriminability = others / target (higher = target is closer)
            if target_val > 0:
                ratio = others_mean / target_val
            else:
                ratio = float("inf")
        else:
            # For duration: higher = more time near stops = better
            # Discriminability = target / others (higher = target spends more time)
            if others_mean > 0:
                ratio = target_val / others_mean
            else:
                ratio = float("inf")

        results.append({
            "radius": radius,
            "feature": feat,
            "target_mode": target_mode,
            "target_mean": round(target_val, 1),
            "others_mean": round(others_mean, 1),
            "ratio": round(ratio, 2),
        })
        print(f"    {feat:20s} ({target_mode:7s}): target={target_val:7.1f}, others={others_mean:7.1f}, ratio={ratio:.2f}")

# ── Summary: best radius per feature ──────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: Best radius per feature (highest discriminability ratio)")
print("=" * 70)

df = pd.DataFrame(results)
for feat in KEY_FEATURES:
    feat_df = df[df["feature"] == feat].sort_values("ratio", ascending=False)
    if feat_df.empty:
        continue
    best = feat_df.iloc[0]
    print(f"  {feat:20s} → best radius = {int(best['radius']):4d}m  (ratio={best['ratio']:.2f}, "
          f"target={best['target_mean']}, others={best['others_mean']})")
    # Show all radii for this feature
    for _, row in feat_df.iterrows():
        marker = " <-- best" if row["radius"] == best["radius"] else ""
        print(f"      {int(row['radius']):4d}m: ratio={row['ratio']:.2f}{marker}")

# ── Recommended config ────────────────────────────────────────────

print("\n" + "=" * 70)
print("RECOMMENDED PER-MODE RADII")
print("=" * 70)

# For each stop type, pick the radius that maximizes the average ratio
# of its two features (MIN_DIST + DURATION)
stop_types = {"BUS": "bus", "SUBWAY": "metro", "TRAIN": "rail", "TRAMWAY": "tram"}
for mode, stop_name in stop_types.items():
    feats = [f"MIN_DIST_{stop_name.upper()}", f"DURATION_{stop_name.upper()}"]
    mode_df = df[df["feature"].isin(feats)]
    if mode_df.empty:
        continue
    avg_ratio = mode_df.groupby("radius")["ratio"].mean().sort_values(ascending=False)
    best_r = avg_ratio.index[0]
    print(f"  {mode:8s} ({stop_name:5s} stops): best radius = {int(best_r):4d}m  (avg ratio={avg_ratio.iloc[0]:.2f})")
    for r, v in avg_ratio.items():
        marker = " <-- best" if r == best_r else ""
        print(f"      {int(r):4d}m: avg_ratio={v:.2f}{marker}")
