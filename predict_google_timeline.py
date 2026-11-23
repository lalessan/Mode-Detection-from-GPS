"""Run the trained modedetect model on Google Timeline location history.

Uses Google's own activity recognition to segment trips (split on STILL
periods) and as ground truth labels to evaluate our model's predictions.

Usage:
    python predict_google_timeline.py --json "Location History.json" \
        --model models/netmob25_modedetect.zip

    # Filter to a date range or bounding box:
    python predict_google_timeline.py --json "Location History.json" \
        --model models/netmob25_modedetect.zip \
        --start 2019-01-01 --end 2019-12-31

    # Generate maps for visual inspection:
    python predict_google_timeline.py --json "Location History.json" \
        --model models/netmob25_modedetect.zip --maps timeline_maps/
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Google activity → our mode mapping ────────────────────────────
# Google types: STILL, IN_VEHICLE, ON_FOOT, WALKING, RUNNING,
#   ON_BICYCLE, TILTING, UNKNOWN, IN_ROAD_VEHICLE, IN_RAIL_VEHICLE,
#   IN_FOUR_WHEELER_VEHICLE, IN_CAR, IN_TWO_WHEELER_VEHICLE, IN_BUS,
#   EXITING_VEHICLE

GOOGLE_TO_MODE = {
    "IN_VEHICLE": "CAR",
    "IN_ROAD_VEHICLE": "CAR",
    "IN_FOUR_WHEELER_VEHICLE": "CAR",
    "IN_CAR": "CAR",
    "IN_TWO_WHEELER_VEHICLE": "CAR",
    "IN_BUS": "BUS",
    "IN_RAIL_VEHICLE": "TRAIN",
    "ON_FOOT": "WALKING",
    "WALKING": "WALKING",
    "RUNNING": "WALKING",
    "ON_BICYCLE": "BIKE",
}
# STILL, TILTING, UNKNOWN, EXITING_VEHICLE → not a transport mode


def parse_google_timeline(json_path: str, start: str = None, end: str = None,
                          lat_range: tuple = None, lon_range: tuple = None) -> pd.DataFrame:
    """Parse Google Timeline JSON into a GPS DataFrame with activity labels."""
    logger.info("Loading %s ...", json_path)
    with open(json_path) as f:
        data = json.load(f)

    locations = data["locations"]
    logger.info("Raw records: %d", len(locations))

    rows = []
    for rec in locations:
        ts_ms = int(rec.get("timestampMs", 0))
        lat = rec.get("latitudeE7", 0) / 1e7
        lon = rec.get("longitudeE7", 0) / 1e7
        acc = rec.get("accuracy", np.nan)

        # Extract top activity
        google_activity = None
        google_confidence = 0
        if "activity" in rec:
            acts = rec["activity"][0]["activity"]
            top = max(acts, key=lambda a: a["confidence"])
            google_activity = top["type"]
            google_confidence = top["confidence"]

        rows.append((ts_ms, lat, lon, acc, google_activity, google_confidence))

    df = pd.DataFrame(rows, columns=[
        "ts_ms", "latitude", "longitude", "accuracy",
        "google_activity", "google_confidence",
    ])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter by date range
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    # Filter by bounding box
    if lat_range:
        df = df[(df["latitude"] >= lat_range[0]) & (df["latitude"] <= lat_range[1])]
    if lon_range:
        df = df[(df["longitude"] >= lon_range[0]) & (df["longitude"] <= lon_range[1])]

    # Map Google activity to our modes
    df["google_mode"] = df["google_activity"].map(GOOGLE_TO_MODE)

    logger.info("After filtering: %d records (%s to %s)",
                len(df), df["timestamp"].min(), df["timestamp"].max())
    logger.info("Google activity coverage: %d/%d (%.1f%%) have activity labels",
                df["google_activity"].notna().sum(), len(df),
                df["google_activity"].notna().mean() * 100)
    return df


def segment_trips_by_still(df: pd.DataFrame, still_min_duration_s: float = 300.0,
                           gap_max_s: float = 600.0,
                           min_points: int = 10,
                           min_duration_s: float = 180.0) -> pd.DataFrame:
    """Segment GPS stream into trips by splitting on STILL periods.

    A STILL period is a run of consecutive records where Google's top activity
    is STILL and the total duration exceeds still_min_duration_s. Points during
    STILL periods are excluded (they are stops, not trips).

    Also splits on large time gaps (> gap_max_s) regardless of activity, to
    handle periods with no data.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill google_activity for records without activity labels
    # (44% have labels; between labeled points, the activity likely persists)
    df["_activity_filled"] = df["google_activity"].ffill()

    # Mark STILL records
    is_still = df["_activity_filled"] == "STILL"

    # Find STILL runs and filter to those lasting > threshold
    still_run = (is_still != is_still.shift()).cumsum()
    still_run_duration = df.groupby(still_run)["timestamp"].transform(
        lambda s: (s.max() - s.min()).total_seconds()
    )
    # A record is part of a "real stop" if it's STILL and the run is long enough
    is_stop = is_still & (still_run_duration >= still_min_duration_s)

    # Also check for time gaps
    dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
    is_gap = dt > gap_max_s

    # Trip boundaries: either end of a stop period or a time gap
    boundary = is_gap | (is_stop.shift(fill_value=True) & ~is_stop)
    trip_id = boundary.cumsum()

    # Assign trip keys and remove stop records
    df["trip_key"] = "trip_" + trip_id.astype(str)
    df = df[~is_stop].reset_index(drop=True)

    # Filter short/sparse trips
    trip_stats = df.groupby("trip_key").agg(
        n_points=("timestamp", "count"),
        duration=("timestamp", lambda s: (s.max() - s.min()).total_seconds()),
    )
    valid = trip_stats[(trip_stats["n_points"] >= min_points) &
                       (trip_stats["duration"] >= min_duration_s)].index
    df = df[df["trip_key"].isin(valid)].reset_index(drop=True)

    df.drop(columns=["_activity_filled"], inplace=True)

    n_trips = df["trip_key"].nunique()
    logger.info("Segmented into %d trips (still_dur=%.0fs, gap=%.0fs, min_pts=%d)",
                n_trips, still_min_duration_s, gap_max_s, min_points)
    return df


def google_trip_label(trip_df: pd.DataFrame) -> str | None:
    """Determine Google's label for a trip: dominant non-STILL activity."""
    modes = trip_df["google_mode"].dropna()
    if len(modes) == 0:
        return None
    return modes.value_counts().index[0]


def main():
    parser = argparse.ArgumentParser(description="Predict transport modes from Google Timeline")
    parser.add_argument("--json", required=True, help="Path to Location History.json")
    parser.add_argument("--model", required=True, help="Path to trained modedetect model .zip")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--lat-range", nargs=2, type=float, default=None, help="Lat min max")
    parser.add_argument("--lon-range", nargs=2, type=float, default=None, help="Lon min max")
    parser.add_argument("--output", default="google_timeline_predictions.csv", help="Output CSV path")
    parser.add_argument("--maps", default=None, help="Directory to save trip maps (optional)")
    args = parser.parse_args()

    # ── Parse and segment ──
    gps = parse_google_timeline(
        args.json, start=args.start, end=args.end,
        lat_range=args.lat_range, lon_range=args.lon_range,
    )
    if len(gps) == 0:
        print("No GPS data after filtering.")
        return

    gps = segment_trips_by_still(gps)
    if gps["trip_key"].nunique() == 0:
        print("No valid trips found.")
        return

    # ── Google ground truth per trip ──
    google_labels = gps.groupby("trip_key").apply(google_trip_label).reset_index()
    google_labels.columns = ["trip_key", "google_label"]

    # ── Load model and predict ──
    from modedetect import ModeDetector
    model = ModeDetector.load(args.model)
    logger.info("Model loaded: %d features, classes: %s",
                len(model._feature_list), list(model._encoder.classes_))

    # Whole-trip prediction (not segmented — compare trip-level)
    preds = model.predict(gps)

    # Add confidence
    proba_df = model.predict_proba(gps)
    proba_cols = [c for c in proba_df.columns if c not in ("trip_key", "segment")]
    preds["confidence"] = proba_df[proba_cols].max(axis=1).values

    # Merge with Google labels
    preds = preds.merge(google_labels, on="trip_key", how="left")

    # Enrich with trip stats
    trip_stats = gps.groupby("trip_key").agg(
        start_time=("timestamp", "first"),
        end_time=("timestamp", "last"),
        n_points=("timestamp", "count"),
        duration_s=("timestamp", lambda s: (s.max() - s.min()).total_seconds()),
        lat_start=("latitude", "first"),
        lon_start=("longitude", "first"),
        lat_end=("latitude", "last"),
        lon_end=("longitude", "last"),
    )
    preds = preds.merge(trip_stats, on="trip_key", how="left")

    # Save
    preds.to_csv(args.output, index=False)
    logger.info("Saved %d predictions to %s", len(preds), args.output)

    # ── Print summary ──
    n_trips = len(preds)
    print(f"\n{'='*70}")
    print(f"GOOGLE TIMELINE PREDICTIONS ({n_trips} trips)")
    print(f"{'='*70}")

    # Mode distribution: our model
    print(f"\n  Our model predictions:")
    mode_counts = preds["predicted_mode"].value_counts()
    for mode, count in mode_counts.items():
        pct = count / n_trips * 100
        med_conf = preds[preds["predicted_mode"] == mode]["confidence"].median()
        print(f"    {mode:<12} {count:>6} ({pct:5.1f}%)  median conf: {med_conf:.3f}")

    # Google label distribution
    has_label = preds["google_label"].notna()
    n_labeled = has_label.sum()
    print(f"\n  Google labels (available for {n_labeled}/{n_trips} trips, {n_labeled/n_trips*100:.1f}%):")
    if n_labeled > 0:
        gl_counts = preds.loc[has_label, "google_label"].value_counts()
        for mode, count in gl_counts.items():
            print(f"    {mode:<12} {count:>6} ({count/n_labeled*100:5.1f}%)")

    # ── Agreement analysis ──
    labeled = preds[has_label].copy()
    if len(labeled) > 0:
        agree = (labeled["predicted_mode"] == labeled["google_label"]).mean()
        print(f"\n  --- Agreement: Our model vs Google ({len(labeled)} trips) ---")
        print(f"  Overall agreement: {agree*100:.1f}%")

        # Per Google-mode agreement
        print(f"\n  {'Google label':<12} {'N trips':>8} {'Agreement':>10} {'Our top prediction':>30}")
        print("  " + "-" * 65)
        for gl in sorted(labeled["google_label"].unique()):
            gl_mask = labeled["google_label"] == gl
            gl_trips = gl_mask.sum()
            gl_agree = (labeled.loc[gl_mask, "predicted_mode"] == gl).mean()
            top_pred = labeled.loc[gl_mask, "predicted_mode"].value_counts()
            top_str = ", ".join(f"{m}({c})" for m, c in top_pred.head(3).items())
            print(f"  {gl:<12} {gl_trips:>8} {gl_agree:>9.1%} {top_str:>30}")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report, f1_score
        # Only use modes that exist in both
        common_modes = sorted(set(labeled["google_label"]) & set(labeled["predicted_mode"]))
        both_common = labeled[labeled["google_label"].isin(common_modes) &
                              labeled["predicted_mode"].isin(common_modes)]
        if len(both_common) > 0:
            f1 = f1_score(both_common["google_label"], both_common["predicted_mode"],
                          average="macro", labels=common_modes)
            print(f"\n  Macro F1 (common modes): {f1:.4f}")

            print(f"\n  Confusion matrix (rows=Google, cols=ours):")
            cm = confusion_matrix(both_common["google_label"], both_common["predicted_mode"],
                                  labels=common_modes)
            # Print as percentages
            header = "  " + " " * 12 + "".join(f"{m:>10}" for m in common_modes)
            print(header)
            for i, mode in enumerate(common_modes):
                row_sum = cm[i].sum()
                if row_sum == 0:
                    continue
                pcts = [f"{cm[i][j]/row_sum*100:>9.1f}%" for j in range(len(common_modes))]
                print(f"  {mode:<12}" + "".join(pcts))

        # Agreement by confidence
        print(f"\n  --- Agreement by confidence threshold ---")
        print(f"  {'Threshold':>10} {'Kept':>7} {'Coverage':>9} {'Agreement':>10}")
        print("  " + "-" * 40)
        for t in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            mask = labeled["confidence"] >= t
            n = mask.sum()
            if n == 0:
                continue
            cov = n / len(labeled)
            ag = (labeled.loc[mask, "predicted_mode"] == labeled.loc[mask, "google_label"]).mean()
            print(f"  {t:>10.2f} {n:>7} {cov:>8.1%} {ag:>9.1%}")

    # Confidence stats
    print(f"\n  Confidence distribution:")
    print(f"    Mean: {preds['confidence'].mean():.3f}, Median: {preds['confidence'].median():.3f}")
    print(f"    <0.5: {(preds['confidence'] < 0.5).mean()*100:.1f}%, "
          f"<0.7: {(preds['confidence'] < 0.7).mean()*100:.1f}%")

    # ── Maps ──
    if args.maps:
        _generate_maps(gps, preds, args.maps)


def _generate_maps(gps: pd.DataFrame, preds: pd.DataFrame, map_dir: str):
    """Generate folium maps for interesting trips."""
    import folium

    out = Path(map_dir)
    out.mkdir(exist_ok=True, parents=True)

    # Pick trips: disagreements, high-conf, low-conf, random
    has_label = preds["google_label"].notna()
    labeled = preds[has_label]

    selected = []
    # Disagreements (most interesting)
    disagree = labeled[labeled["predicted_mode"] != labeled["google_label"]]
    if len(disagree) > 0:
        selected.append(("disagree", disagree.sample(n=min(10, len(disagree)), random_state=42)))
    # High confidence agreements
    agree = labeled[labeled["predicted_mode"] == labeled["google_label"]]
    if len(agree) > 0:
        selected.append(("agree_high", agree.nlargest(5, "confidence")))
    # Low confidence
    selected.append(("low_conf", preds.nsmallest(5, "confidence")))

    maps_index = []
    for group, subset in selected:
        for _, row in subset.iterrows():
            trip_key = row["trip_key"]
            trip_gps = gps[gps["trip_key"] == trip_key].sort_values("timestamp")
            if len(trip_gps) < 2:
                continue

            lats = trip_gps["latitude"].values
            lons = trip_gps["longitude"].values
            m = folium.Map(
                location=[np.median(lats), np.median(lons)], zoom_start=14,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
            )

            # Color by speed
            from modedetect._geo import haversine_m
            dists = haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
            dts = np.diff(trip_gps["timestamp"].values) / np.timedelta64(1, "s")
            dts[dts == 0] = 1
            speeds = np.append(dists / dts, 0)

            coords = list(zip(lats, lons))
            for i in range(len(coords) - 1):
                t = min(speeds[i] / 15.0, 1.0)
                r, b = int(255 * t), int(255 * (1 - t))
                color = f"#{r:02x}00{b:02x}"
                folium.PolyLine([coords[i], coords[i+1]], color=color, weight=3, opacity=0.6).add_to(m)

            folium.Marker(coords[0], popup="START", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(coords[-1], popup="END", icon=folium.Icon(color="red")).add_to(m)

            our_mode = row["predicted_mode"]
            conf = row["confidence"]
            gl = row.get("google_label", "N/A")
            dur = row.get("duration_s", 0)
            start = row.get("start_time", "")
            match_str = "AGREE" if our_mode == gl else "DISAGREE"
            title_html = f'''
            <div style="position:fixed; top:10px; left:60px; z-index:9999;
                        background:white; padding:10px; border:2px solid black;
                        font-size:14px; font-family:monospace;">
                <b>{trip_key}</b> [{match_str}]<br>
                Our model: <b>{our_mode}</b> (conf: {conf:.3f})<br>
                Google: <b>{gl}</b><br>
                {start}<br>
                Duration: {dur/60:.0f} min, Points: {len(trip_gps)}
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

            fname = f"{group}_{our_mode}_vs_{gl}_{trip_key}.html"
            m.save(str(out / fname))
            maps_index.append((group, trip_key, our_mode, gl, conf, fname))

    # Index page
    index = "<html><head><style>body{font-family:sans-serif;max-width:900px;margin:40px auto}</style></head><body>\n"
    index += "<h1>Google Timeline: Our Model vs Google Labels</h1>\n"
    current_group = None
    for group, tk, our, gl, conf, fname in maps_index:
        if group != current_group:
            labels = {"disagree": "Disagreements (our model vs Google)",
                      "agree_high": "Agreements (high confidence)",
                      "low_conf": "Low confidence predictions"}
            index += f"\n<h2>{labels.get(group, group)}</h2>\n<ul>\n"
            current_group = group
        match = "==" if our == gl else "!="
        index += f'<li><a href="{fname}">{tk}: ours={our} {match} google={gl} (conf={conf:.3f})</a></li>\n'
    index += "</ul></body></html>"
    (out / "index.html").write_text(index)
    logger.info("Maps saved to %s/ (%d maps)", out, len(maps_index))


if __name__ == "__main__":
    main()
