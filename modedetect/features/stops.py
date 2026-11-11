"""GTFS stop proximity features using BallTree spatial indexing.

Computes per-GPS-point nearest stop distances, then aggregates to
trip-level (or segment-level) statistics: min distance per mode,
duration near stops, and percentile distributions.

Refactored from:
  3_Stop_Proximity/nearest_stops.py
  3_Stop_Proximity/generate_statistical_summary_stops.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

if TYPE_CHECKING:
    from ..config import ModeDetectConfig

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0

# GTFS route type mapping
ROUTE_TYPE_NAMES = {0: "tram", 1: "metro", 2: "rail", 3: "bus"}


# ── BallTree construction ────────────────────────────────────────────


def build_stop_balltree(
    stops_df: pd.DataFrame,
    lat_col: str = "stop_lat",
    lon_col: str = "stop_lon",
) -> tuple[BallTree, pd.DataFrame]:
    """Build a BallTree from GTFS stop locations.

    Parameters
    ----------
    stops_df : DataFrame
        Must have at minimum ``stop_lat``, ``stop_lon``, ``stop_id`` columns.
        Optionally ``route_type`` or ``route_type_set`` for modal classification.

    Returns
    -------
    tree : BallTree
    stops_df : DataFrame (aligned index for mapping query results back)
    """
    stops = stops_df.reset_index(drop=True)
    coords_rad = np.radians(
        np.column_stack([stops[lat_col].values, stops[lon_col].values])
    )
    tree = BallTree(coords_rad, metric="haversine")
    return tree, stops


# ── Per-point nearest stop query ─────────────────────────────────────


def query_nearest_stops(
    tree: BallTree,
    stops_df: pd.DataFrame,
    gps_lat: np.ndarray,
    gps_lon: np.ndarray,
    search_radius_m: int = 250,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list]]:
    """Query the BallTree for stops within search_radius_m of each GPS point.

    Returns
    -------
    stop_ids_per_point : list of arrays of stop_id strings
    dists_per_point    : list of arrays of distances in metres
    route_types_per_point : list of lists (route type sets per hit)
    """
    gps_rad = np.radians(np.column_stack([gps_lat, gps_lon]))
    rad = search_radius_m / EARTH_RADIUS_M

    idx_arrays, dist_arrays = tree.query_radius(gps_rad, r=rad, return_distance=True, sort_results=True)

    # Convert radians to metres
    dist_arrays_m = [d * EARTH_RADIUS_M for d in dist_arrays]

    # Map indices to stop_ids
    stop_id_values = stops_df["stop_id"].astype(str).values
    has_route_type = "route_type_set" in stops_df.columns

    stop_ids_per_point = []
    route_types_per_point = []
    for idx_arr in idx_arrays:
        idx_arr = np.asarray(idx_arr, dtype=np.int64)
        if len(idx_arr) == 0:
            stop_ids_per_point.append(np.array([], dtype=str))
            route_types_per_point.append([])
        else:
            stop_ids_per_point.append(stop_id_values[idx_arr])
            if has_route_type:
                route_types_per_point.append(stops_df["route_type_set"].iloc[idx_arr].tolist())
            else:
                route_types_per_point.append([[] for _ in idx_arr])

    return stop_ids_per_point, dist_arrays_m, route_types_per_point


# ── Trip-level aggregation ───────────────────────────────────────────


def compute_stop_features(
    gps_df: pd.DataFrame,
    tree: BallTree,
    stops_df: pd.DataFrame,
    config: ModeDetectConfig,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Compute trip-level stop proximity features.

    For each trip (or segment), queries the BallTree and computes:
    - min distance to nearest stop per transit mode (tram/metro/rail/bus)
    - number of stops within radius (avg, min, max)
    - overall min distance stats
    - duration (fraction of points) near each transit mode's stops

    Parameters
    ----------
    gps_df : DataFrame with point-level GPS data (must have lat/lon and trip_key)
    tree : prebuilt BallTree from ``build_stop_balltree``
    stops_df : aligned stops DataFrame from ``build_stop_balltree``
    config : ModeDetectConfig
    segment_col : optional segment column for segment-level aggregation
    """
    key_col = config.col_trip_key
    group_cols = [key_col] if segment_col is None else [key_col, segment_col]

    rows = []
    for group_key, grp in gps_df.groupby(group_cols, sort=False, observed=True):
        if isinstance(group_key, str):
            trip_key = group_key
            seg_id = None
        else:
            trip_key = group_key[0]
            seg_id = group_key[1] if len(group_key) > 1 else None

        lat = grp[config.col_latitude].astype("float64").values
        lon = grp[config.col_longitude].astype("float64").values

        stop_ids, dists_m, route_types = query_nearest_stops(
            tree, stops_df, lat, lon, config.stop_search_radius_m
        )

        row = {"trip_key": trip_key}
        if seg_id is not None:
            row["segment"] = seg_id

        # Aggregate counts
        counts = [len(d) for d in dists_m]
        row["STOPS_NUM_AVG"] = float(np.mean(counts)) if counts else 0.0
        row["STOPS_NUM_MAX"] = int(np.max(counts)) if counts else 0
        row["STOPS_NUM_MIN"] = int(np.min(counts)) if counts else 0

        # Aggregate min distance per point, then stats
        min_dists = [float(np.min(d)) if len(d) > 0 else np.nan for d in dists_m]
        row["STOPS_MINDIST_AVG"] = float(np.nanmean(min_dists)) if min_dists else np.nan

        # Per-mode min distances and duration
        for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
            mode_min_dists = []
            near_count = 0
            for i, rt_list in enumerate(route_types):
                # Find stops of this mode type
                mode_dists_this_point = []
                for j, rt in enumerate(rt_list):
                    if isinstance(rt, (list, tuple, set, np.ndarray)):
                        if mode_code in rt:
                            mode_dists_this_point.append(dists_m[i][j])
                    elif rt == mode_code:
                        mode_dists_this_point.append(dists_m[i][j])

                if mode_dists_this_point:
                    mode_min_dists.append(min(mode_dists_this_point))
                    near_count += 1

            row[f"MIN_DIST_{mode_name.upper()}"] = (
                float(np.min(mode_min_dists)) if mode_min_dists else np.nan
            )
            row[f"DURATION_{mode_name.upper()}"] = near_count / len(grp) if len(grp) > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ── GTFS loading helpers ─────────────────────────────────────────────


def load_gtfs_stops(gtfs_dir: Path) -> pd.DataFrame:
    """Load and merge stop data from GTFS zip files in a directory.

    Returns a DataFrame with ``stop_id``, ``stop_lat``, ``stop_lon``,
    and ``route_type_set`` columns.
    """
    import zipfile

    all_stops = []
    all_stop_times = []
    all_routes = []
    all_trips = []

    for zf_path in sorted(gtfs_dir.glob("*.zip")):
        with zipfile.ZipFile(zf_path) as zf:
            names = zf.namelist()

            if "stops.txt" in names:
                with zf.open("stops.txt") as f:
                    all_stops.append(pd.read_csv(f))

            if "stop_times.txt" in names:
                with zf.open("stop_times.txt") as f:
                    all_stop_times.append(pd.read_csv(f, usecols=["trip_id", "stop_id"]))

            if "routes.txt" in names:
                with zf.open("routes.txt") as f:
                    all_routes.append(pd.read_csv(f))

            if "trips.txt" in names:
                with zf.open("trips.txt") as f:
                    all_trips.append(pd.read_csv(f, usecols=["trip_id", "route_id"]))

    if not all_stops:
        raise FileNotFoundError(f"No stops.txt found in GTFS zips at {gtfs_dir}")

    stops = pd.concat(all_stops, ignore_index=True).drop_duplicates(subset=["stop_id"])

    # Build stop → route_type mapping if possible
    if all_stop_times and all_routes and all_trips:
        st = pd.concat(all_stop_times, ignore_index=True).drop_duplicates()
        routes = pd.concat(all_routes, ignore_index=True).drop_duplicates(subset=["route_id"])
        trips = pd.concat(all_trips, ignore_index=True).drop_duplicates()

        stop_routes = st.merge(trips, on="trip_id").merge(routes[["route_id", "route_type"]], on="route_id")
        route_type_sets = stop_routes.groupby("stop_id")["route_type"].apply(
            lambda x: list(set(x))
        ).reset_index()
        route_type_sets.columns = ["stop_id", "route_type_set"]

        stops = stops.merge(route_type_sets, on="stop_id", how="left")
        stops["route_type_set"] = stops["route_type_set"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    else:
        stops["route_type_set"] = [[] for _ in range(len(stops))]

    required = {"stop_id", "stop_lat", "stop_lon"}
    if not required.issubset(stops.columns):
        raise ValueError(f"GTFS stops missing columns: {required - set(stops.columns)}")

    return stops[["stop_id", "stop_lat", "stop_lon", "route_type_set"]]
