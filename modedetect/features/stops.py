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

    Uses fully vectorized aggregation (no per-trip Python loop).

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
    n_points = len(gps_df)

    # ── Chunked BallTree query (avoids hanging on huge arrays) ─────────
    from tqdm import tqdm

    all_lat = gps_df[config.col_latitude].astype("float64").values
    all_lon = gps_df[config.col_longitude].astype("float64").values
    gps_rad = np.radians(np.column_stack([all_lat, all_lon]))
    rad = config.stop_search_radius_m / EARTH_RADIUS_M

    CHUNK = 500_000
    n_chunks = (n_points + CHUNK - 1) // CHUNK
    all_idx = [None] * n_points
    all_dist = [None] * n_points

    for c in tqdm(range(n_chunks), desc="BallTree query", unit="chunk"):
        lo = c * CHUNK
        hi = min(lo + CHUNK, n_points)
        chunk_idx, chunk_dist = tree.query_radius(
            gps_rad[lo:hi], r=rad, return_distance=True, sort_results=True,
        )
        for j in range(hi - lo):
            all_idx[lo + j] = chunk_idx[j]
            all_dist[lo + j] = chunk_dist[j]

    logger.info("BallTree query complete. Building flat index...")

    # ── Per-point counts and min distances (vectorized) ───────────────
    counts = np.array([len(a) for a in all_idx], dtype=np.int32)
    min_dist_per_point = np.full(n_points, np.nan)
    has_hits = counts > 0
    for i in np.where(has_hits)[0]:
        min_dist_per_point[i] = all_dist[i].min() * EARTH_RADIUS_M

    # ── Build trip-key array for each GPS point (positional) ──────────
    trip_keys = gps_df[key_col].values
    if segment_col is not None:
        seg_ids = gps_df[segment_col].values

    # ── Basic stop stats via pandas groupby ───────────────────────────
    logger.info("Aggregating basic stop stats per trip...")
    agg_df = pd.DataFrame({
        "_key": trip_keys,
        "_count": counts,
        "_min_dist": min_dist_per_point,
    })
    if segment_col is not None:
        agg_df["_seg"] = seg_ids
        gb_cols = ["_key", "_seg"]
    else:
        gb_cols = ["_key"]

    grouped = agg_df.groupby(gb_cols, sort=False, observed=True)
    basic = grouped.agg(
        STOPS_NUM_AVG=("_count", "mean"),
        STOPS_NUM_MAX=("_count", "max"),
        STOPS_NUM_MIN=("_count", "min"),
        STOPS_MINDIST_AVG=("_min_dist", "mean"),
    ).reset_index()

    # Rename key columns back
    basic = basic.rename(columns={"_key": "trip_key"})
    if segment_col is not None:
        basic = basic.rename(columns={"_seg": "segment"})

    # ── Per-mode features via exploded flat table ─────────────────────
    has_route_type = "route_type_set" in stops_df.columns

    if has_route_type:
        # Pre-compute boolean mask per mode for each stop
        rt_values = stops_df["route_type_set"].values
        mode_masks = {}  # mode_code → bool array over stops
        for mode_code in ROUTE_TYPE_NAMES:
            mask = np.zeros(len(stops_df), dtype=bool)
            for si, rt in enumerate(rt_values):
                if isinstance(rt, (list, tuple, set, np.ndarray)):
                    if mode_code in rt:
                        mask[si] = True
                elif rt == mode_code:
                    mask[si] = True
            mode_masks[mode_code] = mask

        # Explode: (gps_point_pos, stop_idx, dist_m)
        logger.info("Exploding %d point-stop pairs for mode aggregation...",
                     int(counts.sum()))
        hit_positions = np.where(has_hits)[0]
        gps_pos_list = []
        stop_idx_list = []
        dist_m_list = []
        for i in hit_positions:
            n = len(all_idx[i])
            gps_pos_list.append(np.full(n, i, dtype=np.int64))
            stop_idx_list.append(all_idx[i].astype(np.int64))
            dist_m_list.append(all_dist[i] * EARTH_RADIUS_M)

        if gps_pos_list:
            flat_gps_pos = np.concatenate(gps_pos_list)
            flat_stop_idx = np.concatenate(stop_idx_list)
            flat_dist_m = np.concatenate(dist_m_list)
        else:
            flat_gps_pos = np.array([], dtype=np.int64)
            flat_stop_idx = np.array([], dtype=np.int64)
            flat_dist_m = np.array([], dtype=np.float64)

        # Map gps_pos → trip key (and segment)
        flat_keys = trip_keys[flat_gps_pos]

        flat_df = pd.DataFrame({
            "_key": flat_keys,
            "_gps_pos": flat_gps_pos,
            "_stop_idx": flat_stop_idx,
            "_dist_m": flat_dist_m,
        })
        if segment_col is not None:
            flat_df["_seg"] = seg_ids[flat_gps_pos]

        # Count points per trip (for duration denominator)
        merge_cols = ["trip_key"] if segment_col is None else ["trip_key", "segment"]
        points_per_trip = basic[merge_cols].copy()
        points_per_trip["_n_points"] = grouped.size().values

        for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
            col_min = f"MIN_DIST_{mode_name.upper()}"
            col_dur = f"DURATION_{mode_name.upper()}"

            # Filter to stops of this mode
            mask = mode_masks[mode_code][flat_stop_idx]
            mode_df = flat_df[mask].copy()

            if len(mode_df) == 0:
                basic[col_min] = np.nan
                basic[col_dur] = 0.0
                continue

            # Rename for groupby
            mode_df = mode_df.rename(columns={"_key": "trip_key"})
            if segment_col is not None:
                mode_df = mode_df.rename(columns={"_seg": "segment"})

            # MIN_DIST: min distance across all points and stops of this mode per trip
            mode_min = mode_df.groupby(merge_cols, sort=False, observed=True)["_dist_m"].min()
            mode_min.name = col_min
            basic = basic.merge(mode_min.reset_index(), on=merge_cols, how="left")

            # DURATION: number of unique GPS points near this mode / total points
            mode_pts = mode_df.groupby(merge_cols, sort=False, observed=True)["_gps_pos"].nunique()
            mode_pts.name = "_near_pts"
            dur_df = mode_pts.reset_index().merge(points_per_trip, on=merge_cols, how="left")
            dur_df[col_dur] = dur_df["_near_pts"] / dur_df["_n_points"]
            basic = basic.merge(dur_df[merge_cols + [col_dur]], on=merge_cols, how="left")
            basic[col_dur] = basic[col_dur].fillna(0.0)
    else:
        # No route type info — fill with NaN/0
        for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
            basic[f"MIN_DIST_{mode_name.upper()}"] = np.nan
            basic[f"DURATION_{mode_name.upper()}"] = 0.0

    # ── Percentile features ───────────────────────────────────────────
    quantiles = config.stop_quantiles  # e.g. (5, 10, 15, ..., 95)
    if quantiles:
        logger.info("Computing stop distance percentiles (P%d–P%d)...",
                     quantiles[0], quantiles[-1])

        q_fracs = [q / 100.0 for q in quantiles]

        # STOPS_MINDIST_Pq: percentile of min_dist_per_point per trip
        # STOPS_NUM_Pq:     percentile of stop count per trip
        pct_df = pd.DataFrame({**{c: agg_df[c] for c in gb_cols},
                                "_min_dist": min_dist_per_point,
                                "_count": counts.astype("float64")})
        pct_grouped = pct_df.groupby(gb_cols, sort=False, observed=True)

        mindist_pcts = pct_grouped["_min_dist"].quantile(q_fracs)
        num_pcts = pct_grouped["_count"].quantile(q_fracs)

        # Unstack the quantile level to columns
        mindist_wide = mindist_pcts.unstack(level=-1)
        mindist_wide.columns = [f"STOPS_MINDIST_P{int(q*100)}" for q in mindist_wide.columns]

        num_wide = num_pcts.unstack(level=-1)
        num_wide.columns = [f"STOPS_NUM_P{int(q*100)}" for q in num_wide.columns]

        pct_result = mindist_wide.join(num_wide).reset_index()
        pct_result = pct_result.rename(columns={"_key": "trip_key"})
        if segment_col is not None:
            pct_result = pct_result.rename(columns={"_seg": "segment"})

        merge_cols_local = ["trip_key"] if segment_col is None else ["trip_key", "segment"]
        basic = basic.merge(pct_result, on=merge_cols_local, how="left")

        # STOPS_CLOSEST_{MODE}_Pq: percentile of closest-stop-of-mode distance
        if has_route_type and len(flat_gps_pos) > 0:
            for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
                mask = mode_masks[mode_code][flat_stop_idx]
                mode_df = flat_df[mask]

                if len(mode_df) == 0:
                    for q in quantiles:
                        basic[f"STOPS_CLOSEST_{mode_name.upper()}_P{q}"] = np.nan
                    continue

                # Closest stop of this mode per GPS point per trip
                closest_gb = mode_df.groupby(gb_cols + ["_gps_pos"], sort=False,
                                              observed=True)["_dist_m"].min().reset_index()

                # Percentiles across GPS points per trip
                closest_pcts = closest_gb.groupby(
                    gb_cols, sort=False, observed=True
                )["_dist_m"].quantile(q_fracs)
                closest_wide = closest_pcts.unstack(level=-1)
                closest_wide.columns = [
                    f"STOPS_CLOSEST_{mode_name.upper()}_P{int(q*100)}"
                    for q in closest_wide.columns
                ]
                closest_wide = closest_wide.reset_index()
                closest_wide = closest_wide.rename(columns={"_key": "trip_key"})
                if segment_col is not None:
                    closest_wide = closest_wide.rename(columns={"_seg": "segment"})

                basic = basic.merge(closest_wide, on=merge_cols_local, how="left")
        else:
            # No route type info — fill percentile columns with NaN
            for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
                for q in quantiles:
                    basic[f"STOPS_CLOSEST_{mode_name.upper()}_P{q}"] = np.nan

    logger.info("Stop feature aggregation complete: %d trips, %d columns",
                len(basic), len(basic.columns))
    return basic


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
