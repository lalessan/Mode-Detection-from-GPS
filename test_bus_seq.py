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
    compute_bus_route_sequence_features,
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

BUS_SEQ_RADII = [20, 30, 50, 75, 100, 150]

print("\n" + "=" * 70)
print("PART 1: Bus Route Sequence Features (BUS_SEQ_STOPS)")
print("=" * 70)

summary_rows = []
for radius in BUS_SEQ_RADII:
    t0 = time.time()
    cfg = ModeDetectConfig(
        col_latitude="LATITUDE", col_longitude="LONGITUDE",
        col_timestamp="timestamp", col_trip_key="KEY",
        col_speed_kmh="SPEED", col_main_mode="Main_Mode",
        gtfs_dir=GTFS_DIR,
    )
    f = compute_bus_route_sequence_features(gps, tree, stops_df, cfg, radius_m=radius)
    elapsed = time.time() - t0

    if f is None:
        print(f"  {radius}m: No results")
        continue

    m = f.merge(trips_meta, on="trip_key")
    means = m.groupby("Main_Mode")["BUS_SEQ_STOPS"].mean()
    bus_val = means.get("BUS", 0)
    car_val = means.get("CAR", 1)
    ratio = bus_val / max(car_val, 0.01)

    summary_rows.append({
        "radius": radius, "BUS": bus_val, "CAR": car_val, "BIKE": means.get("BIKE", 0),
        "WALKING": means.get("WALKING", 0), "SUBWAY": means.get("SUBWAY", 0),
        "TRAIN": means.get("TRAIN", 0), "BUS/CAR": ratio, "time_s": elapsed,
    })

print(f"\n  {'Radius':>6}  {'BUS':>7}  {'CAR':>7}  {'BIKE':>7}  {'WALK':>7}  {'SUBWAY':>7}  {'TRAIN':>7}  {'BUS/CAR':>8}  {'Time':>6}")
print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}")
for r in summary_rows:
    print(f"  {r['radius']:>5}m  {r['BUS']:>7.1f}  {r['CAR']:>7.1f}  {r['BIKE']:>7.1f}  {r['WALKING']:>7.1f}  {r['SUBWAY']:>7.1f}  {r['TRAIN']:>7.1f}  {r['BUS/CAR']:>7.2f}x  {r['time_s']:>5.0f}s")

# ── Part 2: General stop features at different radii ────────────────

STOP_RADII = [100, 250, 500, 1000]
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
