"""Run the trained modedetect model on Google Timeline location history.

Proper pipeline:
  1. Parse raw GPS (lat, lon, time) from Google Timeline JSON
  2. Run infostop for stop detection → trips = movement between stops
  3. Run model.predict_proba(gps, segment=True) for multi-modal prediction
  4. For each predicted segment, look up Google's activity label as ground truth
  5. Compare segment-level predictions vs Google labels

Usage:
    # Basic — motion features only (no GTFS/OSM needed):
    python predict_google_timeline.py --json "Location History.json" \
        --model models/netmob25_modedetect.zip --city copenhagen

    # With auto-downloaded GTFS + OSM for full feature set:
    python predict_google_timeline.py --json "Location History.json" \
        --model models/netmob25_modedetect.zip --city copenhagen --download-data

    # Generate maps for visual inspection:
    python predict_google_timeline.py --json "Location History.json" \
        --model models/netmob25_modedetect.zip --city copenhagen --maps timeline_maps/
"""
import argparse
import json
import logging
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── City presets ──────────────────────────────────────────────────
# bbox: (lat_min, lat_max, lon_min, lon_max)
# gtfs_urls: direct download URLs for GTFS zips (most agencies publish stable URLs)
# osm_pbf_url: Geofabrik extract URL

CITY_PRESETS = {
    "copenhagen": {
        "bbox": (55.55, 55.80, 12.30, 12.70),
        "gtfs_urls": [
            # Rejseplanen (all Danish public transport)
            "https://www.rejseplanen.info/labs/GTFS.zip",
        ],
        "osm_pbf_url": "https://download.geofabrik.de/europe/denmark-latest.osm.pbf",
    },
    "paris": {
        "bbox": (48.80, 48.92, 2.22, 2.47),
        "gtfs_urls": [
            # Ile-de-France Mobilites (RATP + SNCF Transilien + buses)
            "https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/offre-horaires-tc-gtfs-idfm/files/a925e164271e4bca93433756d6a340d1",
        ],
        "osm_pbf_url": "https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf",
    },
    "london": {
        "bbox": (51.40, 51.60, -0.25, 0.05),
        "gtfs_urls": [
            # TfL (requires API key, so use transitfeeds mirror)
            "https://transitfeeds.com/p/transport-for-london/982/latest/download",
        ],
        "osm_pbf_url": "https://download.geofabrik.de/europe/great-britain/england-latest.osm.pbf",
    },
    "berlin": {
        "bbox": (52.40, 52.60, 13.20, 13.55),
        "gtfs_urls": [
            "https://transitfeeds.com/p/verkehrsverbund-berlin-brandenburg/213/latest/download",
        ],
        "osm_pbf_url": "https://download.geofabrik.de/europe/germany/berlin-latest.osm.pbf",
    },
    "rome": {
        "bbox": (41.80, 41.97, 12.40, 12.58),
        "gtfs_urls": [
            "https://transitfeeds.com/p/atac/2056/latest/download",
        ],
        "osm_pbf_url": "https://download.geofabrik.de/europe/italy/centro-latest.osm.pbf",
    },
    "amsterdam": {
        "bbox": (52.30, 52.43, 4.80, 5.00),
        "gtfs_urls": [
            "https://transitfeeds.com/p/ov/814/latest/download",
        ],
        "osm_pbf_url": "https://download.geofabrik.de/europe/netherlands-latest.osm.pbf",
    },
}


def download_gtfs(city_preset: dict, out_dir: Path) -> Path | None:
    """Download GTFS data for a city.

    The model's load_gtfs_stops() expects .zip files inside the directory,
    so we save the downloaded zip as-is (not extracted).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for .zip files)
    if list(out_dir.glob("*.zip")):
        logger.info("GTFS already downloaded in %s", out_dir)
        return out_dir

    for url in city_preset.get("gtfs_urls", []):
        logger.info("Downloading GTFS from %s ...", url)
        try:
            resp = requests.get(url, timeout=120, allow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to download GTFS: %s", e)
            continue

        # Verify it's a valid zip and save it (not extracted)
        try:
            zipfile.ZipFile(io.BytesIO(resp.content))  # validate
            zip_path = out_dir / "gtfs.zip"
            zip_path.write_bytes(resp.content)
            logger.info("GTFS saved to %s (%.1f MB)", zip_path, len(resp.content) / 1e6)
            return out_dir
        except zipfile.BadZipFile:
            logger.warning("Downloaded file is not a valid zip")
            continue

    logger.warning("Could not download GTFS. Running without stop features.")
    return None


def download_osm_pbf(city_preset: dict, out_dir: Path) -> Path | None:
    """Download OSM PBF extract for the region."""
    out_dir.mkdir(parents=True, exist_ok=True)

    url = city_preset.get("osm_pbf_url")
    if not url:
        return None

    fname = url.split("/")[-1]
    pbf_path = out_dir / fname

    if pbf_path.exists():
        logger.info("OSM PBF already downloaded: %s", pbf_path)
        return pbf_path

    logger.info("Downloading OSM PBF from %s ...", url)
    logger.info("(This may take a few minutes for large regions)")
    try:
        resp = requests.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        with open(pbf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = pbf_path.stat().st_size / 1e6
        logger.info("OSM PBF downloaded: %s (%.0f MB)", pbf_path, size_mb)
        return pbf_path
    except Exception as e:
        logger.warning("Failed to download OSM PBF: %s", e)
        return None

# ── Google activity → our mode mapping ────────────────────────────
# Google types: STILL, IN_VEHICLE, ON_FOOT, WALKING, RUNNING,
#   ON_BICYCLE, TILTING, UNKNOWN, IN_ROAD_VEHICLE, IN_RAIL_VEHICLE,
#   IN_FOUR_WHEELER_VEHICLE, IN_CAR, IN_TWO_WHEELER_VEHICLE, IN_BUS,
#   EXITING_VEHICLE

# Simple mapping (used only for parsing/grouping, NOT for evaluation)
GOOGLE_TO_MODE = {
    "IN_VEHICLE": "CAR",
    "IN_ROAD_VEHICLE": "CAR",
    "IN_FOUR_WHEELER_VEHICLE": "CAR",
    "IN_CAR": "CAR",
    "IN_BUS": "BUS",
    "IN_RAIL_VEHICLE": "TRAIN",
    "ON_FOOT": "WALKING",
    "WALKING": "WALKING",
    "ON_BICYCLE": "BIKE",
}

# ── Flexible evaluation matching ──────────────────────────────────
# For each Google activity type, which of OUR predictions count as correct?
GOOGLE_ACCEPTS = {
    "IN_VEHICLE": {"CAR", "BUS"},               # generic vehicle → CAR or BUS both ok
    "IN_ROAD_VEHICLE": {"CAR", "BUS"},
    "IN_FOUR_WHEELER_VEHICLE": {"CAR", "BUS"},
    "IN_CAR": {"CAR"},
    "IN_BUS": {"BUS"},
    "IN_RAIL_VEHICLE": {"TRAIN", "SUBWAY", "TRAMWAY"},  # any rail mode ok
    "ON_FOOT": {"WALKING"},
    "WALKING": {"WALKING"},
    "ON_BICYCLE": {"BIKE"},
}

# Excluded from evaluation entirely
GOOGLE_EXCLUDE = {
    "IN_TWO_WHEELER_VEHICLE",  # no matching mode in our model
    "RUNNING",                  # ambiguous
    "STILL", "TILTING", "UNKNOWN", "EXITING_VEHICLE",  # not transport modes
}


def is_match(google_activity: str, predicted_mode: str) -> bool:
    """Check if a prediction is correct given the Google raw activity type."""
    accepted = GOOGLE_ACCEPTS.get(google_activity)
    if accepted is None:
        return False
    return predicted_mode in accepted


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


def segment_trips_by_infostop(df: pd.DataFrame,
                               r1: float = 30.0,
                               min_staying_time: float = 300.0,
                               min_points: int = 10,
                               min_duration_s: float = 180.0,
                               gap_max_s: float = 300.0,
                               max_range_threshold_m: float = 150.0) -> pd.DataFrame:
    """Segment GPS stream into trips using infostop stop detection.

    infostop detects stationary locations (stops) from raw GPS coordinates.
    Trips are the movement periods between consecutive stops.

    Filters are aligned with ModeDetectConfig defaults:
    - min_points=10, min_duration_s=180, gap_max_s=300, max_range=150m

    Parameters
    ----------
    df : GPS DataFrame with latitude, longitude, timestamp columns.
    r1 : Spatial radius for stop detection (meters). Default 30m (wider than
         infostop's default 10m because Google Timeline has coarser resolution).
    min_staying_time : Minimum duration at a stop to count (seconds). Default 300s.
    min_points : Minimum GPS points per trip to keep.
    min_duration_s : Minimum trip duration (seconds) to keep.

    Returns
    -------
    DataFrame with trip_key column added, stop points excluded.
    """
    import infostop

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove invalid coordinates (same as model's remove_invalid_coordinates)
    valid = (
        df["latitude"].between(-90, 90) &
        df["longitude"].between(-180, 180) &
        df["latitude"].notna() & df["longitude"].notna() &
        ~((df["latitude"] == 0) & (df["longitude"] == 0))
    )
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        df = df[valid].reset_index(drop=True)
        logger.info("Removed %d invalid coordinates", n_invalid)

    # Prepare input: [lat, lon, timestamp_seconds]
    ts_seconds = (df["timestamp"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    coords = np.column_stack([
        df["latitude"].values,
        df["longitude"].values,
        ts_seconds.values,
    ])

    logger.info("Running infostop on %d points (r1=%.0fm, min_stay=%.0fs)...",
                len(coords), r1, min_staying_time)

    model = infostop.Infostop(
        r1=r1,
        r2=r1,
        min_staying_time=min_staying_time,
        min_size=2,
        distance_metric="haversine",
    )
    labels = model.fit_predict(coords)

    # labels: -1 = moving, >=0 = stop location ID
    n_stops = len(set(labels[labels >= 0]))
    n_moving = (labels == -1).sum()
    logger.info("infostop: %d stop locations found, %d/%d points are moving",
                n_stops, n_moving, len(labels))

    df["_stop_label"] = labels

    # ── Validate infostop stops vs Google STILL ──
    if "google_activity" in df.columns:
        has_google = df["google_activity"].notna()
        is_stop = labels >= 0
        google_still = df["google_activity"] == "STILL"

        # Only compare points where Google has a label
        g = has_google
        n_google = g.sum()
        if n_google > 0:
            # Infostop says stop & Google says STILL
            tp = (is_stop & google_still & g).sum()
            # Infostop says stop but Google says not STILL
            fp = (is_stop & ~google_still & g).sum()
            # Infostop says moving but Google says STILL
            fn = (~is_stop & google_still & g).sum()
            # Both say moving / not still
            tn = (~is_stop & ~google_still & g).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            logger.info("Stop validation (infostop vs Google STILL, %d labeled points):", n_google)
            logger.info("  Precision: %.1f%% (of infostop stops, %.0f%% are Google STILL)",
                        precision * 100, precision * 100)
            logger.info("  Recall:    %.1f%% (of Google STILL, %.0f%% are infostop stops)",
                        recall * 100, recall * 100)
            logger.info("  F1:        %.3f", f1)
            logger.info("  Confusion: TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)

            # Also show what Google labels infostop's stops have
            stop_google_acts = df.loc[is_stop & has_google, "google_activity"].value_counts()
            logger.info("  Google labels at infostop stops: %s",
                        ", ".join(f"{act}({n})" for act, n in stop_google_acts.head(5).items()))
            move_google_acts = df.loc[~is_stop & has_google, "google_activity"].value_counts()
            logger.info("  Google labels at infostop moving: %s",
                        ", ".join(f"{act}({n})" for act, n in move_google_acts.head(5).items()))

    # Trips = consecutive runs of -1 (moving) points.
    # Each time we transition from stop (>=0) to moving (-1) starts a new trip.
    is_moving = labels == -1

    # Assign trip IDs: each contiguous block of moving points is a trip.
    # Also split on large time gaps within moving segments.
    dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
    is_time_gap = dt > gap_max_s

    # Create trip boundaries: start of a moving segment, or time gap within moving
    boundary = (is_moving & ~np.roll(is_moving, 1)) | (is_moving & is_time_gap)
    boundary[0] = is_moving[0]  # first point
    trip_id = boundary.cumsum()

    # Only keep moving points
    df["trip_key"] = "trip_" + trip_id.astype(str)
    df = df[is_moving].reset_index(drop=True)

    # Filter short/sparse trips and stationary trips (GPS drift)
    from modedetect._geo import haversine_m as _hav

    trip_stats = df.groupby("trip_key").agg(
        n_points=("timestamp", "count"),
        duration=("timestamp", lambda s: (s.max() - s.min()).total_seconds()),
        lat_mean=("latitude", "mean"),
        lon_mean=("longitude", "mean"),
    )

    # Max range: furthest GPS point from trip centroid (same as training filter)
    centroids = df.merge(
        trip_stats[["lat_mean", "lon_mean"]],
        on="trip_key",
    )
    centroids["_dist_to_centroid"] = _hav(
        centroids["latitude"].values, centroids["longitude"].values,
        centroids["lat_mean"].values, centroids["lon_mean"].values,
    )
    max_range = centroids.groupby("trip_key")["_dist_to_centroid"].max()
    trip_stats["max_range_m"] = max_range

    # Keep trips with enough points, duration, and movement
    valid = trip_stats[
        (trip_stats["n_points"] >= min_points) &
        (trip_stats["duration"] >= min_duration_s) &
        (trip_stats["max_range_m"] >= max_range_threshold_m)
    ].index
    n_stationary = ((trip_stats["max_range_m"] < max_range_threshold_m) &
                    (trip_stats["n_points"] >= min_points)).sum()
    df = df[df["trip_key"].isin(valid)].reset_index(drop=True)
    if n_stationary > 0:
        logger.info("Dropped %d stationary trips (max_range < %.0fm)", n_stationary, max_range_threshold_m)

    df.drop(columns=["_stop_label"], inplace=True)

    n_trips = df["trip_key"].nunique()
    logger.info("Segmented into %d trips (min_pts=%d, min_dur=%.0fs)",
                n_trips, min_points, min_duration_s)
    return df


def filter_unreliable_trips(df: pd.DataFrame,
                            max_jump_m: float = 2000.0,
                            max_speed_kmh: float = 500.0,
                            min_points: int = 10,
                            max_median_interval_s: float = 120.0,
                            jumpback_dist_m: float = 2000.0,
                            jumpback_return_m: float = 500.0) -> pd.DataFrame:
    """Remove trips with GPS quality issues.

    Filters:
    1. Max jump: any consecutive pair > max_jump_m apart
    2. Max speed: any point-to-point speed > max_speed_kmh
    3. Jump-back: trajectory goes > jumpback_dist_m away then returns
       within jumpback_return_m of a previous position (cell tower bouncing)
    4. Min points: fewer than min_points GPS points
    5. Sparse sampling: median inter-point interval > max_median_interval_s
    """
    from modedetect._geo import haversine_m as _hav

    n_before = df["trip_key"].nunique()
    bad_keys = set()
    reasons = {"max_jump": 0, "max_speed": 0, "jumpback": 0,
               "min_points": 0, "sparse": 0}

    for tk, trip in df.groupby("trip_key"):
        trip = trip.sort_values("timestamp")
        n = len(trip)

        # Min points
        if n < min_points:
            bad_keys.add(tk)
            reasons["min_points"] += 1
            continue

        lats = trip["latitude"].values
        lons = trip["longitude"].values
        ts = trip["timestamp"].values

        # Consecutive distances and speeds
        dists = _hav(lats[:-1], lons[:-1], lats[1:], lons[1:])
        dt = np.diff(ts.astype("int64") / 1e9)  # seconds

        # Max jump distance
        if dists.max() > max_jump_m:
            bad_keys.add(tk)
            reasons["max_jump"] += 1
            continue

        # Max speed
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds_kmh = np.where(dt > 0, dists / dt * 3.6, 0)
        if speeds_kmh.max() > max_speed_kmh:
            bad_keys.add(tk)
            reasons["max_speed"] += 1
            continue

        # Sparse sampling
        if np.median(dt) > max_median_interval_s:
            bad_keys.add(tk)
            reasons["sparse"] += 1
            continue

        # Jump-back detection: check if any point is far from a previous
        # point but a later point returns close to the earlier one
        has_jumpback = False
        for i in range(1, n - 1):
            d_from_prev = _hav(lats[i - 1], lons[i - 1], lats[i], lons[i])
            if d_from_prev > jumpback_dist_m:
                # Check if any subsequent point returns near point i-1
                d_returns = _hav(lats[i - 1], lons[i - 1],
                                 lats[i + 1:], lons[i + 1:])
                if d_returns.min() < jumpback_return_m:
                    has_jumpback = True
                    break
        if has_jumpback:
            bad_keys.add(tk)
            reasons["jumpback"] += 1
            continue

    df = df[~df["trip_key"].isin(bad_keys)].reset_index(drop=True)
    n_after = df["trip_key"].nunique()
    n_removed = n_before - n_after

    logger.info("Trip quality filter: removed %d/%d trips (kept %d)",
                n_removed, n_before, n_after)
    for reason, count in sorted(reasons.items()):
        if count > 0:
            logger.info("  %s: %d trips", reason, count)

    return df


def main():
    city_names = ", ".join(sorted(CITY_PRESETS.keys()))
    parser = argparse.ArgumentParser(
        description="Predict transport modes from Google Timeline (infostop + multi-modal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", required=True, help="Path to Location History.json")
    parser.add_argument("--model", required=True, help="Path to trained modedetect model .zip")
    parser.add_argument("--city", default=None,
                        help=f"City preset for bbox + GTFS/OSM. Available: {city_names}")
    parser.add_argument("--download-data", action="store_true",
                        help="Auto-download GTFS + OSM for the city (requires internet)")
    parser.add_argument("--gtfs-dir", default=None, help="Path to local GTFS directory")
    parser.add_argument("--osm-dir", default=None, help="Path to local OSM routes directory")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--lat-range", nargs=2, type=float, default=None, help="Lat min max")
    parser.add_argument("--lon-range", nargs=2, type=float, default=None, help="Lon min max")
    parser.add_argument("--output", default="google_timeline_predictions.csv", help="Output CSV path")
    parser.add_argument("--maps", default=None, help="Directory to save trip maps (optional)")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoints and re-run from scratch")
    args = parser.parse_args()

    # ── Checkpoint directory ──
    ckpt_dir = Path(f"checkpoints_timeline_{args.city}") if args.city else Path("checkpoints_timeline")
    ckpt_dir.mkdir(exist_ok=True)

    if args.clean:
        import shutil
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            ckpt_dir.mkdir()
            logger.info("Deleted checkpoints in %s", ckpt_dir)

    # ── Resolve city preset ──
    preset = None
    if args.city:
        args.city = args.city.lower()
        if args.city not in CITY_PRESETS:
            print(f"Unknown city '{args.city}'. Available: {city_names}")
            return
        preset = CITY_PRESETS[args.city]
        # Apply bbox if not explicitly overridden
        if args.lat_range is None and args.lon_range is None:
            bbox = preset["bbox"]
            args.lat_range = [bbox[0], bbox[1]]
            args.lon_range = [bbox[2], bbox[3]]
            logger.info("Using %s bbox: lat [%.2f, %.2f], lon [%.2f, %.2f]",
                        args.city, *bbox)

    # ── GTFS + OSM data: auto-detect existing, download if requested ──
    data_dir = Path(f"data_{args.city}") if args.city else Path("data_timeline")
    gtfs_dir = Path(args.gtfs_dir) if args.gtfs_dir else None
    osm_routes_dir = Path(args.osm_dir) if args.osm_dir else None

    # Auto-detect existing data for this city
    if preset and gtfs_dir is None:
        candidate = data_dir / "gtfs"
        if candidate.exists() and list(candidate.glob("*.zip")):
            gtfs_dir = candidate
            logger.info("Auto-detected GTFS data: %s", gtfs_dir)
    if preset and osm_routes_dir is None:
        candidate = data_dir / "osm_routes"
        if candidate.exists():
            osm_routes_dir = candidate
            logger.info("Auto-detected OSM routes: %s", osm_routes_dir)

    # Download missing data if requested
    if args.download_data and preset:
        if gtfs_dir is None:
            gtfs_dir = download_gtfs(preset, data_dir / "gtfs")
        if osm_routes_dir is None:
            pbf_path = download_osm_pbf(preset, data_dir / "osm")
            if pbf_path is not None:
                osm_routes_dir = data_dir / "osm_routes"
    elif args.download_data and not preset:
        logger.warning("--download-data requires --city. Skipping data download.")

    # ── Parse and segment with infostop (with checkpointing) ──
    gps_ckpt = ckpt_dir / "gps_infostop.parquet"

    if gps_ckpt.exists():
        logger.info("Loading infostop-segmented GPS from checkpoint: %s", gps_ckpt)
        gps = pd.read_parquet(gps_ckpt)
        logger.info("Loaded %d GPS points, %d trips", len(gps), gps["trip_key"].nunique())
    else:
        gps = parse_google_timeline(
            args.json, start=args.start, end=args.end,
            lat_range=args.lat_range, lon_range=args.lon_range,
        )
        if len(gps) == 0:
            print("No GPS data after filtering.")
            return

        gps = segment_trips_by_infostop(gps)
        if gps["trip_key"].nunique() == 0:
            print("No valid trips found.")
            return

        # Quality filter: remove unreliable trips
        gps = filter_unreliable_trips(gps)
        if gps["trip_key"].nunique() == 0:
            print("No valid trips after quality filter.")
            return

        # Save checkpoint
        gps.to_parquet(gps_ckpt, index=False)
        logger.info("Saved GPS checkpoint: %s (%d trips)", gps_ckpt, gps["trip_key"].nunique())

    # ── Load model ──
    from modedetect import ModeDetector, ModeDetectConfig
    model = ModeDetector.load(args.model)

    # Configure GTFS/OSM if available
    if gtfs_dir and Path(gtfs_dir).exists():
        model.config = ModeDetectConfig(
            **{**model.config.__dict__, "gtfs_dir": Path(gtfs_dir)}
        )
        logger.info("GTFS configured: %s", gtfs_dir)

    if osm_routes_dir and Path(osm_routes_dir).exists():
        model.config = ModeDetectConfig(
            **{**model.config.__dict__, "osm_routes_dir": Path(osm_routes_dir)}
        )
        logger.info("OSM routes configured: %s", osm_routes_dir)
    elif args.download_data and preset and preset.get("osm_pbf_url"):
        # PBF downloaded but routes not yet extracted — extract now
        pbf_path = data_dir / "osm" / preset["osm_pbf_url"].split("/")[-1]
        routes_out = data_dir / "osm_routes"
        if pbf_path.exists() and not routes_out.exists():
            model.config = ModeDetectConfig(
                **{**model.config.__dict__, "osm_pbf_path": pbf_path}
            )
            # Override column names before prepare_osm (needs col_latitude/col_longitude)
            model.config.col_latitude = "latitude"
            model.config.col_longitude = "longitude"
            logger.info("Extracting OSM routes from %s ...", pbf_path)
            model.prepare_osm(gps, output_dir=routes_out)
            logger.info("OSM routes extracted to %s", routes_out)
        elif routes_out.exists():
            model.config = ModeDetectConfig(
                **{**model.config.__dict__, "osm_routes_dir": routes_out}
            )
            logger.info("OSM routes already extracted: %s", routes_out)

    # Override column names to match our Google Timeline GPS format
    model.config.col_latitude = "latitude"
    model.config.col_longitude = "longitude"
    model.config.col_timestamp = "timestamp"
    model.config.col_trip_key = "trip_key"
    model.config.col_speed_kmh = None  # compute from coords

    has_gtfs = model.config.gtfs_dir is not None
    has_osm = model.config.osm_routes_dir is not None or getattr(model.config, "osm_pbf_path", None) is not None
    feature_mode = "full" if (has_gtfs and has_osm) else \
                   "motion + stops" if has_gtfs else \
                   "motion + routes" if has_osm else "motion only"
    logger.info("Model loaded: %d features, classes: %s, feature mode: %s",
                len(model._feature_list), list(model._encoder.classes_), feature_mode)

    # ── Multi-modal predict with segment=True (with checkpointing) ──
    preds_ckpt = ckpt_dir / "predictions_segmented.parquet"

    gps_filtered_ckpt = ckpt_dir / "gps_filtered.parquet"

    if preds_ckpt.exists():
        logger.info("Loading predictions from checkpoint: %s", preds_ckpt)
        seg_preds = pd.read_parquet(preds_ckpt)
        if gps_filtered_ckpt.exists():
            gps = pd.read_parquet(gps_filtered_ckpt)
            logger.info("Loaded filtered GPS: %d points", len(gps))
    else:
        logger.info("Running multi-modal prediction (segment=True) on %d trips...",
                    gps["trip_key"].nunique())
        # predict_proba with segment=True: runs PELT segmentation + per-segment classification
        proba_df = model.predict_proba(gps, segment=True)
        proba_cols = [c for c in proba_df.columns if c not in ("trip_key", "segment")]
        seg_preds = proba_df[["trip_key", "segment"]].copy()
        seg_preds["predicted_mode"] = proba_df[proba_cols].idxmax(axis=1).values
        seg_preds["confidence"] = proba_df[proba_cols].max(axis=1).values

        # Get GPS→segment mapping (filtered by model: speed outliers removed)
        seg_gps = model.segmented_gps()

        # Use filtered GPS for maps and future analysis
        gps = seg_gps
        gps.to_parquet(gps_filtered_ckpt, index=False)
        logger.info("Saved filtered GPS: %d points (%d removed by model)",
                    len(gps), len(pd.read_parquet(ckpt_dir / "gps_infostop.parquet")) - len(gps))

        # Compute Google labels per segment
        has_mode = seg_gps["google_mode"].notna()
        labeled_gps = seg_gps[has_mode]

        # Majority vote for mapped mode and raw activity per segment
        google_labels = labeled_gps.groupby(["trip_key", "segment"]).agg(
            google_label=("google_mode", lambda s: s.value_counts().index[0]),
            google_activity_raw=("google_activity", lambda s: s.value_counts().index[0]),
            google_n_labeled=("google_mode", "count"),
            google_median_conf=("google_confidence", "median"),
        ).reset_index()

        seg_preds = seg_preds.merge(google_labels, on=["trip_key", "segment"], how="left")

        seg_preds.to_parquet(preds_ckpt, index=False)
        logger.info("Saved predictions checkpoint: %s", preds_ckpt)

    # ── Build trip-level summary ──
    trip_summary = seg_preds.groupby("trip_key").agg(
        n_segments=("segment", "nunique"),
        predicted_modes=("predicted_mode", lambda s: " → ".join(s.values)),
        predicted_mode_set=("predicted_mode", lambda s: ",".join(sorted(s.unique()))),
        dominant_predicted=("predicted_mode", lambda s: s.value_counts().index[0]),
        min_confidence=("confidence", "min"),
        mean_confidence=("confidence", "mean"),
        # Google trip-level: dominant across all segments
        google_label=("google_label", lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None),
        google_activity_raw=("google_activity_raw", lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None),
        google_median_conf=("google_median_conf", "median"),
        google_mode_set=("google_label", lambda s: ",".join(sorted(s.dropna().unique())) if s.notna().any() else None),
    ).reset_index()

    # Enrich with trip stats from GPS
    trip_stats = gps.groupby("trip_key").agg(
        start_time=("timestamp", "first"),
        end_time=("timestamp", "last"),
        n_points=("timestamp", "count"),
        duration_s=("timestamp", lambda s: (s.max() - s.min()).total_seconds()),
    )
    trip_summary = trip_summary.merge(trip_stats, on="trip_key", how="left")

    # Save CSV output
    trip_summary.to_csv(args.output, index=False)
    logger.info("Saved %d trip predictions to %s", len(trip_summary), args.output)

    # ── Print summary ──
    n_trips = len(trip_summary)
    n_segs = len(seg_preds)
    multi_modal = (trip_summary["n_segments"] > 1).sum()

    print(f"\n{'='*70}")
    print(f"GOOGLE TIMELINE — INFOSTOP + MULTI-MODAL PREDICTION")
    print(f"{'='*70}")
    print(f"\n  {n_trips} trips, {n_segs} segments ({multi_modal} multi-modal trips, "
          f"avg {trip_summary['n_segments'].mean():.1f} segs/trip)")

    # Mode distribution: our model (dominant per trip)
    print(f"\n  Our model predictions (dominant mode per trip):")
    mode_counts = trip_summary["dominant_predicted"].value_counts()
    for mode, count in mode_counts.items():
        pct = count / n_trips * 100
        med_conf = trip_summary[trip_summary["dominant_predicted"] == mode]["mean_confidence"].median()
        print(f"    {mode:<12} {count:>6} ({pct:5.1f}%)  median conf: {med_conf:.3f}")

    # ── Segment-level agreement (flexible matching) ──
    # Exclude segments with Google types not in GOOGLE_ACCEPTS
    seg_has_raw = seg_preds["google_activity_raw"].notna()
    seg_evaluable = seg_preds[
        seg_has_raw & ~seg_preds["google_activity_raw"].isin(GOOGLE_EXCLUDE)
    ].copy()
    seg_evaluable["_match"] = seg_evaluable.apply(
        lambda r: is_match(r["google_activity_raw"], r["predicted_mode"]), axis=1
    )
    n_seg_eval = len(seg_evaluable)
    n_seg_excluded = seg_has_raw.sum() - n_seg_eval
    print(f"\n  === SEGMENT-LEVEL ANALYSIS ({n_seg_eval}/{n_segs} segments evaluable, "
          f"{n_seg_excluded} excluded [TWO_WHEELER/RUNNING/etc]) ===")

    if n_seg_eval > 0:
        seg_agree = seg_evaluable["_match"].mean()
        print(f"  Segment agreement (flexible): {seg_agree*100:.1f}%")

        # Confusion matrix: rows = raw Google types, cols = our predicted modes
        raw_types = sorted(seg_evaluable["google_activity_raw"].unique())
        pred_modes = sorted(seg_evaluable["predicted_mode"].unique())
        print(f"\n  Confusion matrix (rows=Google raw type, cols=our prediction):")
        header = "  " + " " * 25 + "".join(f"{m:>9}" for m in pred_modes) + "   match%"
        print(header)
        print("  " + "-" * (25 + 9 * len(pred_modes) + 9))
        for gt in raw_types:
            gt_mask = seg_evaluable["google_activity_raw"] == gt
            gt_n = gt_mask.sum()
            gt_match = seg_evaluable.loc[gt_mask, "_match"].mean()
            accepts = GOOGLE_ACCEPTS.get(gt, set())
            counts = []
            for pm in pred_modes:
                c = ((seg_evaluable["google_activity_raw"] == gt) &
                     (seg_evaluable["predicted_mode"] == pm)).sum()
                # Mark accepted predictions
                marker = "*" if pm in accepts else " "
                counts.append(f"{c:>8}{marker}")
            print(f"  {gt:<25}" + "".join(counts) + f"  {gt_match:>5.1%}")
        print(f"\n  (* = accepted as correct match)")

        # Per raw Google type breakdown
        print(f"\n  {'Google type':<25} {'N segs':>8} {'Match%':>8} {'Accepted predictions':>30}")
        print("  " + "-" * 75)
        for gt in raw_types:
            gt_mask = seg_evaluable["google_activity_raw"] == gt
            gt_n = gt_mask.sum()
            gt_match = seg_evaluable.loc[gt_mask, "_match"].mean()
            top_pred = seg_evaluable.loc[gt_mask, "predicted_mode"].value_counts()
            top_str = ", ".join(f"{m}({c})" for m, c in top_pred.head(4).items())
            print(f"  {gt:<25} {gt_n:>8} {gt_match:>7.1%} {top_str:>30}")

        # Segment agreement by our confidence
        print(f"\n  Segment agreement by OUR confidence:")
        print(f"  {'Threshold':>10} {'Kept':>7} {'Coverage':>9} {'Agreement':>10}")
        print("  " + "-" * 40)
        for t in [0.0, 0.5, 0.7, 0.8, 0.9]:
            mask = seg_evaluable["confidence"] >= t
            n = mask.sum()
            if n == 0:
                continue
            cov = n / n_seg_eval
            ag = seg_evaluable.loc[mask, "_match"].mean()
            print(f"  {t:>10.2f} {n:>7} {cov:>8.1%} {ag:>9.1%}")

    # ── Trip-level analysis ──
    print(f"\n  === TRIP-LEVEL ANALYSIS ===")

    # Google label distribution (raw types)
    has_raw = trip_summary["google_activity_raw"].notna()
    has_label = trip_summary["google_label"].notna()
    n_labeled = has_label.sum()

    print(f"\n  Google raw activity distribution ({n_labeled}/{n_trips} trips with labels):")
    if has_raw.any():
        raw_counts = trip_summary.loc[has_raw, "google_activity_raw"].value_counts()
        for act, count in raw_counts.items():
            excluded = " [EXCLUDED]" if act in GOOGLE_EXCLUDE else ""
            med_gc = trip_summary.loc[has_raw & (trip_summary["google_activity_raw"] == act), "google_median_conf"].median()
            print(f"    {act:<30} {count:>6} ({count/has_raw.sum()*100:5.1f}%)  median conf: {med_gc:.0f}{excluded}")

        # Google confidence distribution
        gc = trip_summary.loc[has_label, "google_median_conf"]
        print(f"\n  Google confidence distribution (median per trip):")
        print(f"    Mean: {gc.mean():.0f}, Median: {gc.median():.0f}, "
              f"P25: {gc.quantile(0.25):.0f}, P75: {gc.quantile(0.75):.0f}")

    # ── Agreement analysis (flexible matching, excluding certain types) ──
    evaluable = trip_summary[
        has_raw & ~trip_summary["google_activity_raw"].isin(GOOGLE_EXCLUDE)
    ].copy()
    evaluable["_match"] = evaluable.apply(
        lambda r: is_match(r["google_activity_raw"], r["dominant_predicted"]), axis=1
    )
    n_eval = len(evaluable)
    n_excluded_trips = has_raw.sum() - n_eval

    if n_eval > 0:
        agree = evaluable["_match"].mean()
        print(f"\n  --- Agreement: Our model vs Google ({n_eval} evaluable trips, "
              f"{n_excluded_trips} excluded) ---")
        print(f"  Overall agreement (flexible): {agree*100:.1f}%")

        # Confusion matrix: rows = raw Google types, cols = our predicted modes
        raw_types = sorted(evaluable["google_activity_raw"].unique())
        pred_modes = sorted(evaluable["dominant_predicted"].unique())
        print(f"\n  Confusion matrix (rows=Google raw type, cols=our prediction):")
        header = "  " + " " * 25 + "".join(f"{m:>9}" for m in pred_modes) + "   match%"
        print(header)
        print("  " + "-" * (25 + 9 * len(pred_modes) + 9))
        for gt in raw_types:
            gt_mask = evaluable["google_activity_raw"] == gt
            gt_n = gt_mask.sum()
            gt_match = evaluable.loc[gt_mask, "_match"].mean()
            accepts = GOOGLE_ACCEPTS.get(gt, set())
            counts = []
            for pm in pred_modes:
                c = ((evaluable["google_activity_raw"] == gt) &
                     (evaluable["dominant_predicted"] == pm)).sum()
                marker = "*" if pm in accepts else " "
                counts.append(f"{c:>8}{marker}")
            print(f"  {gt:<25}" + "".join(counts) + f"  {gt_match:>5.1%}")
        print(f"\n  (* = accepted as correct match)")

        # Per raw Google type breakdown
        print(f"\n  {'Google type':<25} {'N trips':>8} {'Match%':>8} {'Our predictions':>30}")
        print("  " + "-" * 75)
        for gt in raw_types:
            gt_mask = evaluable["google_activity_raw"] == gt
            gt_n = gt_mask.sum()
            gt_match = evaluable.loc[gt_mask, "_match"].mean()
            top_pred = evaluable.loc[gt_mask, "dominant_predicted"].value_counts()
            top_str = ", ".join(f"{m}({c})" for m, c in top_pred.head(4).items())
            print(f"  {gt:<25} {gt_n:>8} {gt_match:>7.1%} {top_str:>30}")

        # Agreement by OUR confidence
        print(f"\n  --- Agreement by OUR confidence threshold ---")
        print(f"  {'Threshold':>10} {'Kept':>7} {'Coverage':>9} {'Agreement':>10}")
        print("  " + "-" * 40)
        for t in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            mask = evaluable["mean_confidence"] >= t
            n = mask.sum()
            if n == 0:
                continue
            cov = n / n_eval
            ag = evaluable.loc[mask, "_match"].mean()
            print(f"  {t:>10.2f} {n:>7} {cov:>8.1%} {ag:>9.1%}")

        # Agreement by GOOGLE confidence
        print(f"\n  --- Agreement by GOOGLE confidence threshold ---")
        print(f"  {'Threshold':>10} {'Kept':>7} {'Coverage':>9} {'Agreement':>10}")
        print("  " + "-" * 40)
        for t in [0, 30, 50, 70, 80, 90]:
            mask = evaluable["google_median_conf"] >= t
            n = mask.sum()
            if n == 0:
                continue
            cov = n / n_eval
            ag = evaluable.loc[mask, "_match"].mean()
            print(f"  {t:>10} {n:>7} {cov:>8.1%} {ag:>9.1%}")

        # Agreement when BOTH are confident
        print(f"\n  --- Agreement when both confident (ours >= 0.7 AND Google >= 70) ---")
        both_conf = evaluable[(evaluable["mean_confidence"] >= 0.7) & (evaluable["google_median_conf"] >= 70)]
        if len(both_conf) > 0:
            ag_both = both_conf["_match"].mean()
            print(f"  {len(both_conf)} trips ({len(both_conf)/n_eval*100:.1f}% coverage), "
                  f"agreement: {ag_both*100:.1f}%")

        # What does Google say about trips we predict as public transport?
        print(f"\n  --- Our public transport predictions vs Google ---")
        pt_modes = ["SUBWAY", "TRAIN", "BUS", "TRAMWAY"]
        for pm in pt_modes:
            pm_mask = evaluable["dominant_predicted"] == pm
            if pm_mask.sum() == 0:
                continue
            gl_dist = evaluable.loc[pm_mask, "google_activity_raw"].value_counts()
            gl_str = ", ".join(f"{m}({c})" for m, c in gl_dist.head(4).items())
            med_conf = evaluable.loc[pm_mask, "mean_confidence"].median()
            n_match = evaluable.loc[pm_mask, "_match"].sum()
            print(f"  We say {pm:<8} ({pm_mask.sum():>4} trips, med conf {med_conf:.3f}, "
                  f"{n_match} match): Google says {gl_str}")

    # ── Verifiable-only analysis ──
    # Exclude trips where we predict modes Google can't label (SUBWAY, TRAIN, TRAMWAY)
    # These are unverifiable, not necessarily wrong.
    if n_eval > 0:
        unverifiable_modes = {"SUBWAY", "TRAIN", "TRAMWAY"}
        verifiable = evaluable[~evaluable["dominant_predicted"].isin(unverifiable_modes)].copy()
        n_unverifiable = n_eval - len(verifiable)

        print(f"\n  === VERIFIABLE ONLY (excluding our SUBWAY/TRAIN/TRAMWAY predictions) ===")
        print(f"  {len(verifiable)}/{n_eval} trips ({n_unverifiable} unverifiable excluded)")

        if len(verifiable) > 0:
            v_agree = verifiable["_match"].mean()
            print(f"  Agreement: {v_agree*100:.1f}%")

            # Confusion matrix
            v_raw_types = sorted(verifiable["google_activity_raw"].unique())
            v_pred_modes = sorted(verifiable["dominant_predicted"].unique())
            print(f"\n  Confusion matrix (rows=Google raw type, cols=our prediction):")
            header = "  " + " " * 25 + "".join(f"{m:>9}" for m in v_pred_modes) + "   match%"
            print(header)
            print("  " + "-" * (25 + 9 * len(v_pred_modes) + 9))
            for gt in v_raw_types:
                gt_mask = verifiable["google_activity_raw"] == gt
                gt_n = gt_mask.sum()
                gt_match = verifiable.loc[gt_mask, "_match"].mean()
                accepts = GOOGLE_ACCEPTS.get(gt, set())
                counts = []
                for pm in v_pred_modes:
                    c = ((verifiable["google_activity_raw"] == gt) &
                         (verifiable["dominant_predicted"] == pm)).sum()
                    marker = "*" if pm in accepts else " "
                    counts.append(f"{c:>8}{marker}")
                print(f"  {gt:<25}" + "".join(counts) + f"  {gt_match:>5.1%}")

            # Agreement by confidence
            print(f"\n  Agreement by OUR confidence (verifiable):")
            print(f"  {'Threshold':>10} {'Kept':>7} {'Coverage':>9} {'Agreement':>10}")
            print("  " + "-" * 40)
            for t in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
                mask = verifiable["mean_confidence"] >= t
                n = mask.sum()
                if n == 0:
                    continue
                cov = n / len(verifiable)
                ag = verifiable.loc[mask, "_match"].mean()
                print(f"  {t:>10.2f} {n:>7} {cov:>8.1%} {ag:>9.1%}")

    # Confidence stats
    print(f"\n  Confidence distribution:")
    print(f"    Mean: {trip_summary['mean_confidence'].mean():.3f}, "
          f"Median: {trip_summary['mean_confidence'].median():.3f}")
    print(f"    <0.5: {(trip_summary['mean_confidence'] < 0.5).mean()*100:.1f}%, "
          f"<0.7: {(trip_summary['mean_confidence'] < 0.7).mean()*100:.1f}%")

    # ── Multi-modal trip details ──
    if multi_modal > 0:
        print(f"\n  --- Multi-modal trips ({multi_modal} trips) ---")
        multi_trips = trip_summary[trip_summary["n_segments"] > 1]
        multi_labeled = multi_trips[multi_trips["google_label"].notna()]
        if len(multi_labeled) > 0:
            print(f"  Labeled: {len(multi_labeled)}")
            # Show predicted mode sequences for multi-modal trips
            print(f"  Top predicted sequences:")
            seq_counts = multi_labeled["predicted_modes"].value_counts().head(10)
            for seq, count in seq_counts.items():
                print(f"    {seq:<40} {count:>4}")

            # Mode set comparison: does our set of modes overlap with Google's?
            multi_w_google_set = multi_labeled[multi_labeled["google_mode_set"].notna()]
            if len(multi_w_google_set) > 0:
                overlaps = []
                for _, row in multi_w_google_set.iterrows():
                    our_set = set(row["predicted_mode_set"].split(","))
                    gl_set = set(row["google_mode_set"].split(","))
                    overlaps.append(len(our_set & gl_set) / max(len(our_set | gl_set), 1))
                print(f"  Mode set Jaccard (our modes vs Google modes): "
                      f"{np.mean(overlaps):.3f} (n={len(overlaps)})")

    # ── Maps ──
    if args.maps:
        _generate_maps(gps, seg_preds, trip_summary, args.maps)


def _generate_maps(gps: pd.DataFrame, seg_preds: pd.DataFrame,
                    trip_summary: pd.DataFrame, map_dir: str):
    """Generate folium maps for interesting trips with segment coloring."""
    import folium

    out = Path(map_dir)
    out.mkdir(exist_ok=True, parents=True)

    MODE_COLORS = {
        "WALKING": "#2ecc71", "BIKE": "#3498db", "CAR": "#e74c3c",
        "BUS": "#f39c12", "SUBWAY": "#9b59b6", "TRAIN": "#1abc9c",
        "TRAMWAY": "#e67e22",
    }

    labeled = trip_summary[trip_summary["google_label"].notna()]

    selected = []
    # Top confident predictions per mode (most useful for inspection)
    for mode in ["SUBWAY", "TRAIN", "BUS", "CAR", "BIKE", "WALKING"]:
        mode_trips = labeled[labeled["dominant_predicted"] == mode]
        if len(mode_trips) == 0:
            continue
        top = mode_trips.nlargest(min(10, len(mode_trips)), "mean_confidence")
        selected.append((f"top_{mode}", top["trip_key"].values))

    # Multi-modal trips
    multi_labeled = labeled[labeled["n_segments"] > 1]
    if len(multi_labeled) > 0:
        sample = multi_labeled.sample(n=min(10, len(multi_labeled)), random_state=42)
        selected.append(("multi_modal", sample["trip_key"].values))

    maps_index = []
    for group, trip_keys in selected:
        for tk in trip_keys:
            trip_gps = gps[gps["trip_key"] == tk].sort_values("timestamp")
            if len(trip_gps) < 2:
                continue

            trip_seg_preds = seg_preds[seg_preds["trip_key"] == tk].sort_values("segment")
            trip_info = trip_summary[trip_summary["trip_key"] == tk].iloc[0]

            lats = trip_gps["latitude"].values
            lons = trip_gps["longitude"].values
            m = folium.Map(
                location=[np.median(lats), np.median(lons)], zoom_start=14,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
            )

            # Color GPS trace by predicted segment mode
            # Need to reconstruct which points belong to which segment
            n_segs = len(trip_seg_preds)
            n_pts = len(trip_gps)
            if n_segs == 1:
                # All points = one segment
                color = MODE_COLORS.get(trip_seg_preds.iloc[0]["predicted_mode"], "#999")
                coords = list(zip(lats, lons))
                folium.PolyLine(coords, color=color, weight=4, opacity=0.8).add_to(m)
            else:
                # Split points roughly by segment (equal splits as approximation)
                pts_per_seg = n_pts // n_segs
                for seg_idx, (_, srow) in enumerate(trip_seg_preds.iterrows()):
                    start_i = seg_idx * pts_per_seg
                    end_i = (seg_idx + 1) * pts_per_seg if seg_idx < n_segs - 1 else n_pts
                    seg_lats = lats[start_i:end_i + 1]  # overlap for continuity
                    seg_lons = lons[start_i:end_i + 1]
                    color = MODE_COLORS.get(srow["predicted_mode"], "#999")
                    if len(seg_lats) >= 2:
                        coords = list(zip(seg_lats, seg_lons))
                        folium.PolyLine(coords, color=color, weight=4, opacity=0.8).add_to(m)

            folium.Marker([lats[0], lons[0]], popup="START", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker([lats[-1], lons[-1]], popup="END", icon=folium.Icon(color="red")).add_to(m)

            # Build info panel
            modes_str = trip_info["predicted_modes"]
            confs_str = " / ".join(f"{c:.2f}" for c in trip_seg_preds["confidence"].values)
            gl = trip_info.get("google_activity_raw", trip_info.get("google_label", "N/A"))
            gl_mapped = trip_info.get("google_label", "N/A")
            dur = trip_info.get("duration_s", 0)
            start_time = trip_info.get("start_time", "")

            # Legend
            legend_items = "".join(
                f'<span style="color:{c}">■</span> {m} '
                for m, c in MODE_COLORS.items()
            )

            title_html = f'''
            <div style="position:fixed; top:10px; left:60px; z-index:9999;
                        background:white; padding:10px; border:2px solid black;
                        font-size:13px; font-family:monospace; max-width:500px;">
                <b>{tk}</b> [{group}]<br>
                Our model: <b>{modes_str}</b><br>
                Confidence: {confs_str}<br>
                Google: <b>{gl}</b> ({gl_mapped})<br>
                {start_time}<br>
                Duration: {dur/60:.0f} min, Points: {n_pts}, Segments: {n_segs}<br>
                <small>{legend_items}</small>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

            fname = f"{group}_{tk}.html"
            m.save(str(out / fname))
            maps_index.append((group, tk, modes_str, gl,
                             trip_seg_preds["confidence"].min(), fname))

    # Index page
    index = "<html><head><style>body{font-family:sans-serif;max-width:900px;margin:40px auto}</style></head><body>\n"
    index += "<h1>Google Timeline: Infostop + Multi-Modal Prediction</h1>\n"
    current_group = None
    for group, tk, our, gl, conf, fname in maps_index:
        if group != current_group:
            labels = {
                "top_SUBWAY": "Top SUBWAY predictions (by confidence)",
                "top_TRAIN": "Top TRAIN predictions (by confidence)",
                "top_BUS": "Top BUS predictions (by confidence)",
                "top_CAR": "Top CAR predictions (by confidence)",
                "top_BIKE": "Top BIKE predictions (by confidence)",
                "top_WALKING": "Top WALKING predictions (by confidence)",
                "multi_modal": "Multi-modal trips (random sample)",
            }
            index += f"\n<h2>{labels.get(group, group)}</h2>\n<ul>\n"
            current_group = group
        index += f'<li><a href="{fname}">{tk}: ours={our} | google={gl} (min_conf={conf:.3f})</a></li>\n'
    index += "</ul></body></html>"
    (out / "index.html").write_text(index)
    logger.info("Maps saved to %s/ (%d maps)", out, len(maps_index))


if __name__ == "__main__":
    main()
