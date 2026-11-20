"""Quick test: check stop-related feature discriminability at different radii.

Tests both:
  1. Bus route sequence features (BUS_SEQ_STOPS) — which radius best separates BUS from CAR?
  2. General stop features (MIN_DIST, DURATION) — which radius best separates each mode?

Usage:
    python test_bus_seq.py --sample 0.1   # 10% of trips
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

# ── Part 1: Bus route sequence features at different radii ──────────

STOP_RADII = [30, 50, 70]
KEY_FEATURES = ["MIN_DIST_BUS", "DURATION_BUS", "MIN_DIST_RAIL", "DURATION_RAIL",
                "MIN_DIST_METRO", "DURATION_METRO", "MIN_DIST_TRAM", "DURATION_TRAM"]

print("\n" + "=" * 70)
print("PART 2: General Stop Features at Different Radii")
print("=" * 70)

for radius in STOP_RADII:
    print(f"\n  --- Radius = {radius}m ---")
    t0 = time.time()
    cfg = ModeDetectConfig(
        col_latitude="LATITUDE", col_longitude="LONGITUDE",
        col_timestamp="timestamp", col_trip_key="KEY",
        col_speed_kmh="SPEED", col_main_mode="Main_Mode",
        gtfs_dir=GTFS_DIR,
        stop_search_radius_m=radius,
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

    means = m.groupby("Main_Mode")[available].mean()
    print(f"\n  Mean feature values by mode (radius={radius}m):")
    print(means.round(1).to_string())
