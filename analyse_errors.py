"""Analyse misclassified trips to understand what features are missing.

Loads OOF predictions, finds the biggest confusion pairs, and compares
feature values of correctly vs incorrectly classified trips.

Usage:
    python analyse_errors.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints/")

# ── Load data ──────────────────────────────────────────────────────

print("Loading OOF predictions...")
oof = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "05_oof_predictions.parquet")
print(f"  {len(oof)} trips, columns: {list(oof.columns)}")

# Identify true and predicted mode columns
if "true_label" in oof.columns and "pred_label" in oof.columns:
    true_col, pred_col = "true_label", "pred_label"
elif "true_mode" in oof.columns and "predicted_mode" in oof.columns:
    true_col, pred_col = "true_mode", "predicted_mode"
elif "Main_Mode" in oof.columns and "predicted_mode" in oof.columns:
    true_col, pred_col = "Main_Mode", "predicted_mode"
else:
    print(f"  Available columns: {list(oof.columns)}")
    raise ValueError("Cannot find true/predicted mode columns")

print(f"  True: '{true_col}', Predicted: '{pred_col}'")

# Load features
print("Loading features...")
stats = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "02_stats.parquet")
stops = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "03_stop_features.parquet")

bus_seq_path = CHECKPOINT_DIR / "fit_steps" / "03_bus_route_features.parquet"
bus_seq = pd.read_parquet(bus_seq_path) if bus_seq_path.exists() else None

routes_path = CHECKPOINT_DIR / "fit_steps" / "04_route_features.parquet"
routes = pd.read_parquet(routes_path) if routes_path.exists() else None

# Merge features with OOF predictions
feats = stats.copy()
for extra, name in [(stops, "stops"), (bus_seq, "bus_seq"), (routes, "routes")]:
    if extra is not None:
        merge_on = ["trip_key"]
        if "segment" in extra.columns and "segment" in feats.columns:
            merge_on.append("segment")
        feats = feats.merge(extra, on=merge_on, how="left", suffixes=("", f"_{name}"))
        print(f"  Merged {name}: {feats.shape}")

oof_feats = oof.merge(feats, on="trip_key", how="left")
print(f"  OOF + features: {oof_feats.shape}")

# ── Find biggest confusion pairs ──────────────────────────────────

correct = oof_feats[true_col] == oof_feats[pred_col]
print(f"\n  Overall accuracy: {correct.mean():.1%} ({correct.sum()}/{len(correct)})")

# Build confusion counts
conf = oof_feats.groupby([true_col, pred_col]).size().reset_index(name="count")
conf = conf[conf[true_col] != conf[pred_col]].sort_values("count", ascending=False)

print(f"\n{'='*70}")
print("TOP 10 CONFUSION PAIRS (true → predicted, count)")
print(f"{'='*70}")
for _, row in conf.head(10).iterrows():
    total_true = (oof_feats[true_col] == row[true_col]).sum()
    pct = row["count"] / total_true * 100
    print(f"  {row[true_col]:>8s} → {row[pred_col]:<8s}  n={row['count']:5d}  ({pct:.1f}% of {row[true_col]})")

# ── Analyse each confusion pair ───────────────────────────────────

KEY_FEATURES = [
    "SPEED_AVG", "SPEED_MED", "SPEED_MAX", "SPEED_STD",
    "TOTAL_TIME", "DIST_TOTAL_WGS84", "GPS_POINTS",
    "MIN_DIST_BUS", "DURATION_BUS",
    "MIN_DIST_METRO", "DURATION_METRO",
    "MIN_DIST_RAIL", "DURATION_RAIL",
    "MIN_DIST_TRAM", "DURATION_TRAM",
    "BUS_SEQ_STOPS", "BUS_SEQ_ROUTES",
    "bus_match_pct", "tram_match_pct", "train_match_pct", "subway_match_pct",
    "STOPS_NUM_AVG", "STOP_FRACTION", "STRAIGHTNESS",
]

# Filter to features that exist
KEY_FEATURES = [f for f in KEY_FEATURES if f in oof_feats.columns]

print(f"\n{'='*70}")
print("DETAILED ANALYSIS: What distinguishes correct vs misclassified trips?")
print(f"{'='*70}")

for _, row in conf.head(6).iterrows():
    true_mode = row[true_col]
    pred_mode = row[pred_col]
    n_confused = row["count"]

    # Get the confused trips and correctly classified trips of the same true mode
    mask_true = oof_feats[true_col] == true_mode
    mask_correct = mask_true & (oof_feats[pred_col] == true_mode)
    mask_confused = mask_true & (oof_feats[pred_col] == pred_mode)

    correct_trips = oof_feats[mask_correct]
    confused_trips = oof_feats[mask_confused]

    # Also get correctly classified trips of the predicted mode (what does pred_mode look like?)
    mask_pred_correct = (oof_feats[true_col] == pred_mode) & (oof_feats[pred_col] == pred_mode)
    pred_correct = oof_feats[mask_pred_correct]

    print(f"\n{'─'*70}")
    print(f"  {true_mode} → {pred_mode}  (n={n_confused}, "
          f"correctly classified {true_mode}: {len(correct_trips)})")
    print(f"{'─'*70}")

    print(f"\n  {'Feature':<25s} {'Correct '+true_mode:>15s} {'Confused→'+pred_mode:>15s} {'Actual '+pred_mode:>15s}  Direction")
    print(f"  {'─'*90}")

    for feat in KEY_FEATURES:
        val_correct = correct_trips[feat].median()
        val_confused = confused_trips[feat].median()
        val_pred = pred_correct[feat].median()

        # Is the confused trip more like the predicted mode?
        dist_to_true = abs(val_confused - val_correct)
        dist_to_pred = abs(val_confused - val_pred)

        if dist_to_true < 0.01 and dist_to_pred < 0.01:
            direction = "  (same)"
        elif dist_to_pred < dist_to_true:
            direction = f"  ← looks like {pred_mode}"
        else:
            direction = f"  ← looks like {true_mode}"

        print(f"  {feat:<25s} {val_correct:>15.1f} {val_confused:>15.1f} {val_pred:>15.1f}{direction}")

    # Show a few example trips
    print(f"\n  Example confused trips (first 5):")
    example_cols = ["trip_key", "SPEED_AVG", "TOTAL_TIME", "DIST_TOTAL_WGS84", "GPS_POINTS"]
    example_cols = [c for c in example_cols if c in confused_trips.columns]
    if "bus_match_pct" in confused_trips.columns:
        example_cols.append("bus_match_pct")
    if "subway_match_pct" in confused_trips.columns:
        example_cols.append("subway_match_pct")
    if "train_match_pct" in confused_trips.columns:
        example_cols.append("train_match_pct")
    print(confused_trips[example_cols].head(5).to_string(index=False))

# ── SUBWAY deep dive ──────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUBWAY DEEP DIVE")
print(f"{'='*70}")

subway_trips = oof_feats[oof_feats[true_col] == "SUBWAY"]
subway_correct = subway_trips[subway_trips[pred_col] == "SUBWAY"]
subway_as_walk = subway_trips[subway_trips[pred_col] == "WALKING"]
subway_as_bus = subway_trips[subway_trips[pred_col] == "BUS"]

print(f"\n  Total SUBWAY trips: {len(subway_trips)}")
print(f"  Correctly classified: {len(subway_correct)} ({len(subway_correct)/len(subway_trips):.1%})")
print(f"  Misclassified as WALKING: {len(subway_as_walk)} ({len(subway_as_walk)/len(subway_trips):.1%})")
print(f"  Misclassified as BUS: {len(subway_as_bus)} ({len(subway_as_bus)/len(subway_trips):.1%})")

print(f"\n  Feature comparison (median):")
print(f"  {'Feature':<25s} {'Correct SUB':>12s} {'SUB→WALK':>12s} {'SUB→BUS':>12s} {'True WALK':>12s}")
print(f"  {'─'*75}")

walk_correct = oof_feats[(oof_feats[true_col] == "WALKING") & (oof_feats[pred_col] == "WALKING")]

for feat in KEY_FEATURES:
    v1 = subway_correct[feat].median()
    v2 = subway_as_walk[feat].median() if len(subway_as_walk) > 0 else float("nan")
    v3 = subway_as_bus[feat].median() if len(subway_as_bus) > 0 else float("nan")
    v4 = walk_correct[feat].median()
    print(f"  {feat:<25s} {v1:>12.1f} {v2:>12.1f} {v3:>12.1f} {v4:>12.1f}")

# Speed distribution for SUBWAY trips
print(f"\n  Speed distribution (SPEED_AVG, m/s):")
for label, subset in [("Correct SUBWAY", subway_correct),
                       ("SUBWAY→WALKING", subway_as_walk),
                       ("SUBWAY→BUS", subway_as_bus)]:
    if len(subset) == 0:
        continue
    speeds = subset["SPEED_AVG"]
    print(f"    {label:<20s}: P10={speeds.quantile(0.1):.1f}, P25={speeds.quantile(0.25):.1f}, "
          f"P50={speeds.median():.1f}, P75={speeds.quantile(0.75):.1f}, P90={speeds.quantile(0.9):.1f}")

print(f"\n  Duration distribution (TOTAL_TIME, seconds):")
for label, subset in [("Correct SUBWAY", subway_correct),
                       ("SUBWAY→WALKING", subway_as_walk),
                       ("SUBWAY→BUS", subway_as_bus)]:
    if len(subset) == 0:
        continue
    times = subset["TOTAL_TIME"]
    print(f"    {label:<20s}: P10={times.quantile(0.1):.0f}, P25={times.quantile(0.25):.0f}, "
          f"P50={times.median():.0f}, P75={times.quantile(0.75):.0f}, P90={times.quantile(0.9):.0f}")

# GPS gaps — are SUBWAY trips losing signal underground?
if "GAP_MAX" in oof_feats.columns:
    print(f"\n  GPS gaps (GAP_MAX, seconds):")
    for label, subset in [("Correct SUBWAY", subway_correct),
                           ("SUBWAY→WALKING", subway_as_walk),
                           ("True WALKING", walk_correct),
                           ("True BUS", oof_feats[(oof_feats[true_col] == "BUS") & (oof_feats[pred_col] == "BUS")])]:
        if len(subset) == 0:
            continue
        gaps = subset["GAP_MAX"]
        print(f"    {label:<20s}: P10={gaps.quantile(0.1):.0f}, P50={gaps.median():.0f}, "
              f"P75={gaps.quantile(0.75):.0f}, P90={gaps.quantile(0.9):.0f}, max={gaps.max():.0f}")

if "subway_match_pct" in oof_feats.columns:
    print(f"\n  subway_match_pct distribution:")
    for label, subset in [("Correct SUBWAY", subway_correct),
                           ("SUBWAY→WALKING", subway_as_walk),
                           ("SUBWAY→BUS", subway_as_bus),
                           ("True WALKING", walk_correct),
                           ("True BUS", oof_feats[(oof_feats[true_col] == "BUS") & (oof_feats[pred_col] == "BUS")])]:
        if len(subset) == 0:
            continue
        vals = subset["subway_match_pct"]
        print(f"    {label:<20s}: P10={vals.quantile(0.1):.3f}, P50={vals.median():.3f}, "
              f"P75={vals.quantile(0.75):.3f}, P90={vals.quantile(0.9):.3f}")

# ── CAR → WALKING deep dive ──────────────────────────────────────

print(f"\n{'='*70}")
print("CAR → WALKING DEEP DIVE")
print(f"{'='*70}")

car_trips = oof_feats[oof_feats[true_col] == "CAR"]
car_correct = car_trips[car_trips[pred_col] == "CAR"]
car_as_walk = car_trips[car_trips[pred_col] == "WALKING"]
walk_correct = oof_feats[(oof_feats[true_col] == "WALKING") & (oof_feats[pred_col] == "WALKING")]

print(f"\n  Total CAR trips: {len(car_trips)}")
print(f"  Correctly classified: {len(car_correct)} ({len(car_correct)/len(car_trips):.1%})")
print(f"  Misclassified as WALKING: {len(car_as_walk)} ({len(car_as_walk)/len(car_trips):.1%})")

print(f"\n  Feature comparison (median):")
print(f"  {'Feature':<25s} {'Correct CAR':>12s} {'CAR→WALK':>12s} {'True WALK':>12s}")
print(f"  {'─'*65}")

for feat in KEY_FEATURES:
    v1 = car_correct[feat].median()
    v2 = car_as_walk[feat].median() if len(car_as_walk) > 0 else float("nan")
    v3 = walk_correct[feat].median()
    print(f"  {feat:<25s} {v1:>12.1f} {v2:>12.1f} {v3:>12.1f}")

for feat_name, feat_col in [("SPEED_AVG (m/s)", "SPEED_AVG"),
                             ("DIST_TOTAL_WGS84 (m)", "DIST_TOTAL_WGS84"),
                             ("TOTAL_TIME (s)", "TOTAL_TIME"),
                             ("GPS_POINTS", "GPS_POINTS")]:
    if feat_col not in oof_feats.columns:
        continue
    print(f"\n  {feat_name} distribution:")
    for label, subset in [("Correct CAR", car_correct),
                           ("CAR→WALKING", car_as_walk),
                           ("True WALKING", walk_correct)]:
        if len(subset) == 0:
            continue
        vals = subset[feat_col]
        print(f"    {label:<20s}: P10={vals.quantile(0.1):.1f}, P25={vals.quantile(0.25):.1f}, "
              f"P50={vals.median():.1f}, P75={vals.quantile(0.75):.1f}, P90={vals.quantile(0.9):.1f}")

# What threshold would catch them?
if "DIST_TOTAL_WGS84" in oof_feats.columns and len(car_as_walk) > 0:
    print(f"\n  Distance thresholds — how many CAR→WALK trips would be filtered:")
    for thresh in [100, 200, 500, 1000, 2000]:
        n_below = (car_as_walk["DIST_TOTAL_WGS84"] < thresh).sum()
        pct = n_below / len(car_as_walk) * 100
        # Also check how many correct CAR trips we'd lose
        n_correct_lost = (car_correct["DIST_TOTAL_WGS84"] < thresh).sum()
        pct_correct = n_correct_lost / len(car_correct) * 100
        print(f"    < {thresh:>5d}m: {n_below:>4d}/{len(car_as_walk)} CAR→WALK ({pct:.1f}%), "
              f"would also drop {n_correct_lost} correct CAR ({pct_correct:.1f}%)")

if "SPEED_AVG" in oof_feats.columns and len(car_as_walk) > 0:
    print(f"\n  Speed thresholds — how many CAR→WALK trips would be filtered:")
    for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
        n_below = (car_as_walk["SPEED_AVG"] < thresh).sum()
        pct = n_below / len(car_as_walk) * 100
        n_correct_lost = (car_correct["SPEED_AVG"] < thresh).sum()
        pct_correct = n_correct_lost / len(car_correct) * 100
        print(f"    < {thresh:.1f} m/s: {n_below:>4d}/{len(car_as_walk)} CAR→WALK ({pct:.1f}%), "
              f"would also drop {n_correct_lost} correct CAR ({pct_correct:.1f}%)")

print(f"\n{'='*70}")
print("Done. Use these insights to identify what features or signals are missing.")
print(f"{'='*70}")
