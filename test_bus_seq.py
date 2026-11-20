"""Quick test: check bus route sequence feature values by mode.

Usage:
    python test_bus_seq.py [--sample FRAC]

Example:
    python test_bus_seq.py --sample 0.1   # 10% of trips (~3min instead of 14)
    python test_bus_seq.py                 # all trips
"""
import argparse
import time

import pandas as pd

from modedetect import ModeDetectConfig
from modedetect.features.stops import (
    build_stop_balltree,
    compute_bus_route_sequence_features,
    load_gtfs_stops,
)

GTFS_DIR = next(
    p for p in [
        __import__("pathlib").Path("gtfs/"),
        __import__("pathlib").Path("/home/lauale/ModeDetection/gtfs/"),
    ] if p.exists()
)

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=float, default=1.0, help="Fraction of trips to use (0.0-1.0)")
args = parser.parse_args()

cfg = ModeDetectConfig(
    col_latitude="LATITUDE", col_longitude="LONGITUDE",
    col_timestamp="timestamp", col_trip_key="KEY",
    col_speed_kmh="SPEED", col_main_mode="Main_Mode",
    gtfs_dir=GTFS_DIR,
)

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

print("Computing bus route sequence features...")
t0 = time.time()
f = compute_bus_route_sequence_features(gps, tree, stops_df, cfg)
print(f"Done in {time.time() - t0:.1f}s")

print(f"\n{f.describe()}\n")

trips_meta = pd.read_parquet(
    "checkpoints/fit_steps/02_stats.parquet",
    columns=["trip_key", "Main_Mode"],
).drop_duplicates("trip_key")
m = f.merge(trips_meta, on="trip_key")

print("Mean by mode:")
print(m.groupby("Main_Mode")[["BUS_SEQ_STOPS", "BUS_SEQ_ROUTES"]].mean().round(2))
print("\nMedian by mode:")
print(m.groupby("Main_Mode")[["BUS_SEQ_STOPS", "BUS_SEQ_ROUTES"]].median())
