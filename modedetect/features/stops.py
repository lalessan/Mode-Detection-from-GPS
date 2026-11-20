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


def _process_chunk(
    gps_rad_chunk: np.ndarray,
    tree: BallTree,
    rad: float,
    mode_mask_arrays: dict[int, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    """Process a single chunk of GPS points against the BallTree.

    Returns per-point summaries for the chunk.
    """
    n = len(gps_rad_chunk)
    counts = np.zeros(n, dtype=np.int32)
    min_dist = np.full(n, np.nan)
    mode_min_dist_top5 = {mc: np.full(n, np.nan) for mc in ROUTE_TYPE_NAMES}
    mode_has_stop_top5 = {mc: np.zeros(n, dtype=bool) for mc in ROUTE_TYPE_NAMES}
    mode_min_dist = {mc: np.full(n, np.nan) for mc in ROUTE_TYPE_NAMES}

    chunk_idx, chunk_dist = tree.query_radius(
        gps_rad_chunk, r=rad, return_distance=True, sort_results=True,
    )
    for j in range(n):
        idx_arr = chunk_idx[j]
        if len(idx_arr) == 0:
            continue

        dist_m = chunk_dist[j] * EARTH_RADIUS_M
        counts[j] = len(idx_arr)
        min_dist[j] = dist_m.min()

        if mode_mask_arrays is not None:
            top5_idx = idx_arr[:5]
            top5_dist = dist_m[:5]

            for mode_code in ROUTE_TYPE_NAMES:
                all_mode_hits = mode_mask_arrays[mode_code][idx_arr]
                if all_mode_hits.any():
                    mode_min_dist[mode_code][j] = dist_m[all_mode_hits].min()

                top5_mode_hits = mode_mask_arrays[mode_code][top5_idx]
                if top5_mode_hits.any():
                    mode_has_stop_top5[mode_code][j] = True
                    mode_min_dist_top5[mode_code][j] = top5_dist[top5_mode_hits].min()

    return counts, min_dist, mode_min_dist_top5, mode_has_stop_top5, mode_min_dist


def _query_balltree_parallel(
    gps_rad: np.ndarray,
    tree: BallTree,
    rad: float,
    mode_mask_arrays: dict[int, np.ndarray] | None,
    n_jobs: int = 1,
    chunk_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    """Run chunked BallTree queries, optionally in parallel with multiprocessing."""
    from tqdm import tqdm

    n_points = len(gps_rad)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    # Build chunk slices
    chunks = []
    for c in range(n_chunks):
        lo = c * chunk_size
        hi = min(lo + chunk_size, n_points)
        chunks.append((lo, hi))

    # Allocate output arrays
    counts = np.zeros(n_points, dtype=np.int32)
    min_dist = np.full(n_points, np.nan)
    mode_min_dist_top5 = {mc: np.full(n_points, np.nan) for mc in ROUTE_TYPE_NAMES}
    mode_has_stop_top5 = {mc: np.zeros(n_points, dtype=bool) for mc in ROUTE_TYPE_NAMES}
    mode_min_dist = {mc: np.full(n_points, np.nan) for mc in ROUTE_TYPE_NAMES}

    effective_jobs = min(n_jobs, n_chunks)

    if effective_jobs <= 1:
        # Sequential
        for lo, hi in tqdm(chunks, desc="BallTree query", unit="chunk"):
            c_counts, c_min, c_top5, c_has, c_all = _process_chunk(
                gps_rad[lo:hi], tree, rad, mode_mask_arrays,
            )
            counts[lo:hi] = c_counts
            min_dist[lo:hi] = c_min
            for mc in ROUTE_TYPE_NAMES:
                mode_min_dist_top5[mc][lo:hi] = c_top5[mc]
                mode_has_stop_top5[mc][lo:hi] = c_has[mc]
                mode_min_dist[mc][lo:hi] = c_all[mc]
    else:
        # Parallel with multiprocessing (not threading — avoids GIL/deadlock)
        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info("Using %d processes for BallTree query...", effective_jobs)
        futures = {}
        with ProcessPoolExecutor(max_workers=effective_jobs) as pool:
            for lo, hi in chunks:
                fut = pool.submit(
                    _process_chunk, gps_rad[lo:hi], tree, rad, mode_mask_arrays,
                )
                futures[fut] = (lo, hi)

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="BallTree query", unit="chunk"):
                lo, hi = futures[fut]
                c_counts, c_min, c_top5, c_has, c_all = fut.result()
                counts[lo:hi] = c_counts
                min_dist[lo:hi] = c_min
                for mc in ROUTE_TYPE_NAMES:
                    mode_min_dist_top5[mc][lo:hi] = c_top5[mc]
                    mode_has_stop_top5[mc][lo:hi] = c_has[mc]
                    mode_min_dist[mc][lo:hi] = c_all[mc]

    return counts, min_dist, mode_min_dist_top5, mode_has_stop_top5, mode_min_dist


def compute_stop_features(
    gps_df: pd.DataFrame,
    tree: BallTree,
    stops_df: pd.DataFrame,
    config: ModeDetectConfig,
    segment_col: str | None = None,
    point_cache_path: Path | str | None = None,
) -> pd.DataFrame:
    """Compute trip-level stop proximity features.

    Parameters
    ----------
    gps_df : DataFrame with point-level GPS data (must have lat/lon and trip_key)
    tree : prebuilt BallTree from ``build_stop_balltree``
    stops_df : aligned stops DataFrame from ``build_stop_balltree``
    config : ModeDetectConfig
    segment_col : optional segment column for segment-level aggregation
    point_cache_path : optional path to cache per-point BallTree results (.npz).
        If the file exists, the BallTree query is skipped entirely and
        only the (fast) aggregation runs. Saves ~20 min on re-runs.
    """
    key_col = config.col_trip_key
    n_points = len(gps_df)

    # ── Compute dt (time delta in seconds) per GPS point ─────────────────
    ts_col = config.col_timestamp
    trip_keys_arr = gps_df[key_col].values
    timestamps = pd.to_datetime(gps_df[ts_col]).values
    dt_seconds = np.full(n_points, np.nan)
    trip_boundaries = np.where(trip_keys_arr[1:] != trip_keys_arr[:-1])[0] + 1
    raw_dt = (timestamps[1:] - timestamps[:-1]).astype("timedelta64[ns]").astype(np.float64) / 1e9
    dt_seconds[1:] = raw_dt
    # Skip first point of each trip (no dt) AND last point (student uses range(1, len-1))
    dt_seconds[0] = np.nan
    dt_seconds[trip_boundaries] = np.nan
    # Last point of each trip = index before each boundary, plus the very last index
    trip_last_points = trip_boundaries - 1
    dt_seconds[trip_last_points] = np.nan
    dt_seconds[n_points - 1] = np.nan

    # ── BallTree query (or load from cache) ──────────────────────────────
    has_route_type = stops_df is not None and "route_type_set" in stops_df.columns
    cache_path = Path(point_cache_path) if point_cache_path is not None else None

    cache_loaded = False
    if cache_path is not None and cache_path.exists():
        logger.info("Loading per-point BallTree cache from %s", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        if len(data["counts"]) == n_points:
            counts = data["counts"]
            min_dist = data["min_dist"]
            mode_min_dist_top5 = {mc: data[f"mdist_top5_{mc}"] for mc in ROUTE_TYPE_NAMES}
            mode_has_stop_top5 = {mc: data[f"mhas_top5_{mc}"] for mc in ROUTE_TYPE_NAMES}
            mode_min_dist = {mc: data[f"mdist_all_{mc}"] for mc in ROUTE_TYPE_NAMES}
            has_route_type = True
            cache_loaded = True
        else:
            logger.warning("Cache size mismatch (%d vs %d points) — recomputing.",
                          len(data["counts"]), n_points)

    if not cache_loaded:
        mode_mask_arrays = None
        if has_route_type:
            mode_mask_arrays = {}
            rt_values = stops_df["route_type_set"].values
            for mode_code in ROUTE_TYPE_NAMES:
                mask = np.zeros(len(stops_df), dtype=bool)
                for si, rt in enumerate(rt_values):
                    if isinstance(rt, (list, tuple, set, np.ndarray)):
                        if mode_code in rt:
                            mask[si] = True
                    elif rt == mode_code:
                        mask[si] = True
                mode_mask_arrays[mode_code] = mask

        all_lat = gps_df[config.col_latitude].astype("float64").values
        all_lon = gps_df[config.col_longitude].astype("float64").values
        gps_rad = np.radians(np.column_stack([all_lat, all_lon]))
        rad = config.stop_search_radius_m / EARTH_RADIUS_M

        counts, min_dist, mode_min_dist_top5, mode_has_stop_top5, mode_min_dist = \
            _query_balltree_parallel(
                gps_rad, tree, rad, mode_mask_arrays,
                n_jobs=config.n_jobs, chunk_size=500_000,
            )

        # Save cache
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = {"counts": counts, "min_dist": min_dist}
            for mc in ROUTE_TYPE_NAMES:
                save_dict[f"mdist_top5_{mc}"] = mode_min_dist_top5[mc]
                save_dict[f"mhas_top5_{mc}"] = mode_has_stop_top5[mc]
                save_dict[f"mdist_all_{mc}"] = mode_min_dist[mc]
            np.savez(cache_path, **save_dict)
            logger.info("Saved per-point BallTree cache to %s", cache_path)

    logger.info("BallTree query complete. Aggregating to trip level...")

    # ── Build trip-key / segment arrays ────────────────────────────────
    trip_keys = gps_df[key_col].values
    if segment_col is not None:
        seg_ids = gps_df[segment_col].values

    agg_df = pd.DataFrame({
        "_key": trip_keys, "_count": counts, "_min_dist": min_dist,
        "_dt": dt_seconds,
    })
    if segment_col is not None:
        agg_df["_seg"] = seg_ids
        gb_cols = ["_key", "_seg"]
    else:
        gb_cols = ["_key"]

    # ── Basic stop stats ──────────────────────────────────────────────
    grouped = agg_df.groupby(gb_cols, sort=False, observed=True)
    basic = grouped.agg(
        STOPS_NUM_AVG=("_count", "mean"),
        STOPS_NUM_MAX=("_count", "max"),
        STOPS_NUM_MIN=("_count", "min"),
        STOPS_MINDIST_AVG=("_min_dist", "mean"),
    ).reset_index()

    basic = basic.rename(columns={"_key": "trip_key"})
    if segment_col is not None:
        basic = basic.rename(columns={"_seg": "segment"})

    merge_cols = ["trip_key"] if segment_col is None else ["trip_key", "segment"]

    # ── Per-mode features ─────────────────────────────────────────────
    if has_route_type:
        for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
            col_min = f"MIN_DIST_{mode_name.upper()}"
            col_dur = f"DURATION_{mode_name.upper()}"

            # Add per-mode columns to agg_df for groupby
            mode_dist_col = f"_mdist_{mode_code}"
            mode_dur_col = f"_mdur_{mode_code}"
            # MIN_DIST uses top-5 nearest stops (matching student's approach)
            agg_df[mode_dist_col] = mode_min_dist_top5[mode_code]
            # DURATION: dt (seconds) for points near this mode's stops (top 5)
            mode_dt = np.where(mode_has_stop_top5[mode_code], dt_seconds, 0.0)
            # NaN dt → 0 (first point of trip, or last point — no contribution)
            mode_dt = np.nan_to_num(mode_dt, nan=0.0)
            agg_df[mode_dur_col] = mode_dt

            mode_grouped = agg_df.groupby(gb_cols, sort=False, observed=True)

            # MIN_DIST: min across all points per trip (top-5 only)
            mode_min_series = mode_grouped[mode_dist_col].min()
            mode_min_series.name = col_min
            basic = basic.merge(mode_min_series.reset_index().rename(
                columns={"_key": "trip_key"} | ({"_seg": "segment"} if segment_col else {})
            ), on=merge_cols, how="left")

            # DURATION: sum of seconds near this mode's stops
            mode_dur_sum = mode_grouped[mode_dur_col].sum()
            mode_dur_sum.name = col_dur
            basic = basic.merge(mode_dur_sum.reset_index().rename(
                columns={"_key": "trip_key"} | ({"_seg": "segment"} if segment_col else {})
            ), on=merge_cols, how="left")
            basic[col_dur] = basic[col_dur].fillna(0.0)
    else:
        for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
            basic[f"MIN_DIST_{mode_name.upper()}"] = np.nan
            basic[f"DURATION_{mode_name.upper()}"] = 0.0

    # ── Percentile features ───────────────────────────────────────────
    quantiles = config.stop_quantiles
    if quantiles:
        logger.info("Computing stop distance percentiles (P%d–P%d)...",
                     quantiles[0], quantiles[-1])

        q_fracs = [q / 100.0 for q in quantiles]

        # STOPS_MINDIST_Pq and STOPS_NUM_Pq: percentiles of per-point values
        pct_df = pd.DataFrame({
            **{c: agg_df[c] for c in gb_cols},
            "_min_dist": min_dist,
            "_count": counts.astype("float64"),
        })
        pct_grouped = pct_df.groupby(gb_cols, sort=False, observed=True)

        mindist_pcts = pct_grouped["_min_dist"].quantile(q_fracs)
        num_pcts = pct_grouped["_count"].quantile(q_fracs)

        mindist_wide = mindist_pcts.unstack(level=-1)
        mindist_wide.columns = [f"STOPS_MINDIST_P{int(q*100)}" for q in mindist_wide.columns]

        num_wide = num_pcts.unstack(level=-1)
        num_wide.columns = [f"STOPS_NUM_P{int(q*100)}" for q in num_wide.columns]

        pct_result = mindist_wide.join(num_wide).reset_index()
        pct_result = pct_result.rename(columns={"_key": "trip_key"})
        if segment_col is not None:
            pct_result = pct_result.rename(columns={"_seg": "segment"})
        basic = basic.merge(pct_result, on=merge_cols, how="left")

        # STOPS_CLOSEST_{MODE}_Pq: percentiles of per-point closest-mode distance
        if has_route_type:
            for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
                mode_dist_arr = mode_min_dist[mode_code]

                # Only include points that had a stop of this mode
                mode_pct_df = pd.DataFrame({
                    **{c: agg_df[c] for c in gb_cols},
                    "_dist": mode_dist_arr,
                })
                # Keep only points with a mode stop (non-NaN)
                mode_pct_df = mode_pct_df.dropna(subset=["_dist"])

                if len(mode_pct_df) == 0:
                    for q in quantiles:
                        basic[f"STOPS_CLOSEST_{mode_name.upper()}_P{q}"] = np.nan
                    continue

                mode_pct_grouped = mode_pct_df.groupby(gb_cols, sort=False, observed=True)
                closest_pcts = mode_pct_grouped["_dist"].quantile(q_fracs)
                closest_wide = closest_pcts.unstack(level=-1)
                closest_wide.columns = [
                    f"STOPS_CLOSEST_{mode_name.upper()}_P{int(q*100)}"
                    for q in closest_wide.columns
                ]
                closest_wide = closest_wide.reset_index()
                closest_wide = closest_wide.rename(columns={"_key": "trip_key"})
                if segment_col is not None:
                    closest_wide = closest_wide.rename(columns={"_seg": "segment"})
                basic = basic.merge(closest_wide, on=merge_cols, how="left")
        else:
            for mode_code, mode_name in ROUTE_TYPE_NAMES.items():
                for q in quantiles:
                    basic[f"STOPS_CLOSEST_{mode_name.upper()}_P{q}"] = np.nan

    logger.info("Stop feature aggregation complete: %d trips, %d columns",
                len(basic), len(basic.columns))
    return basic


# ── Bus route sequence features ──────────────────────────────────────


def compute_bus_route_sequence_features(
    gps_df: pd.DataFrame,
    tree: BallTree | None,
    stops_df: pd.DataFrame | None,
    config: ModeDetectConfig,
    segment_col: str | None = None,
) -> pd.DataFrame | None:
    """Count how many unique stops from the same bus route each trip/segment visits.

    A bus passenger passes many stops of the same route in sequence;
    a car driver passes near random stops from many different routes.

    Fully vectorized: flattens BallTree hits into COO arrays, filters to bus
    stops, explodes route IDs, then uses pandas groupby for aggregation.

    Returns DataFrame with ``trip_key`` (and optional ``segment``) plus:

    - ``BUS_SEQ_STOPS``: max unique stops from a single bus route
    - ``BUS_SEQ_ROUTES``: number of bus routes with ≥2 stops visited
    """
    key_col = config.col_trip_key
    group_cols = [key_col] if segment_col is None else [key_col, segment_col]
    n_points = len(gps_df)

    # ── Compute flat COO arrays of (point_idx, stop_idx) ──────────────
    if tree is None or stops_df is None or "bus_route_id_set" not in stops_df.columns:
        return None

    # Build bus stop mask (stops that serve at least one bus route)
    bus_route_ids_arr = stops_df["bus_route_id_set"].values
    is_bus_stop = np.array([
        bool(r) if isinstance(r, (list, set)) else False
        for r in bus_route_ids_arr
    ])
    if not is_bus_stop.any():
        return None

    logger.info("Computing bus route sequence features (BallTree query on %d points)...",
                n_points)
    all_lat = gps_df[config.col_latitude].astype("float64").values
    all_lon = gps_df[config.col_longitude].astype("float64").values
    gps_rad = np.radians(np.column_stack([all_lat, all_lon]))
    rad = config.stop_search_radius_m / EARTH_RADIUS_M

    # Chunked BallTree query with progress bar
    from tqdm import tqdm
    chunk_size = 500_000
    point_indices = []
    stop_indices = []
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    for chunk_i in tqdm(range(n_chunks), desc="Bus route seq (BallTree)",
                        unit="chunk"):
        start = chunk_i * chunk_size
        end = min(start + chunk_size, n_points)
        chunk_rad = gps_rad[start:end]

        idx_arrays = tree.query_radius(chunk_rad, r=rad)

        for j, idx_arr in enumerate(idx_arrays):
            if len(idx_arr) == 0:
                continue
            bus_mask = is_bus_stop[idx_arr]
            if bus_mask.any():
                bus_hits = idx_arr[bus_mask]
                point_indices.append(np.full(len(bus_hits), start + j, dtype=np.int32))
                stop_indices.append(bus_hits.astype(np.int32))

    if point_indices:
        hit_point_idx = np.concatenate(point_indices)
        hit_stop_idx = np.concatenate(stop_indices)
    else:
        hit_point_idx = np.array([], dtype=np.int32)
        hit_stop_idx = np.array([], dtype=np.int32)

    if len(hit_point_idx) == 0:
        # No bus stop hits at all — return zeros
        result = gps_df.groupby(group_cols, sort=False, observed=True).size().reset_index(name="_n")
        result = result.drop(columns=["_n"])
        result["BUS_SEQ_STOPS"] = 0
        result["BUS_SEQ_ROUTES"] = 0
        if key_col != "trip_key" and key_col in result.columns:
            result = result.rename(columns={key_col: "trip_key"})
        if segment_col is not None and segment_col in result.columns and segment_col != "segment":
            result = result.rename(columns={segment_col: "segment"})
        return result

    # ── Explode stop → route_id mapping into flat arrays ──────────────
    # For each (point, stop) hit, expand to (point, stop, route_id) rows.
    # Build a flat lookup: for each stop index, its bus route IDs.
    if stops_df is not None and "bus_route_id_set" in stops_df.columns:
        bus_route_ids_arr = stops_df["bus_route_id_set"].values
    else:
        # Cache-only path: we need a stop→route_id mapping.
        # Without stops_df, we can't explode. Load it from GTFS.
        logger.warning("No stops_df available — cannot compute bus seq features from cache alone.")
        return None

    # Build flat exploded arrays: for each hit, one row per route_id
    unique_stops_in_hits = np.unique(hit_stop_idx)
    # Pre-build: stop_idx → list of integer route_id codes
    # Encode route_id strings as integers for speed
    all_route_ids = set()
    for si in unique_stops_in_hits:
        br = bus_route_ids_arr[si]
        if isinstance(br, (list, set)):
            all_route_ids.update(br)
    route_id_to_code = {rid: i for i, rid in enumerate(sorted(all_route_ids))}

    # For each stop, its route codes
    stop_route_codes = {}
    for si in unique_stops_in_hits:
        br = bus_route_ids_arr[si]
        if isinstance(br, (list, set)) and br:
            stop_route_codes[si] = [route_id_to_code[r] for r in br]

    # ── Sort hits by stop index for fast O(1) slicing per stop ────────
    sort_order = np.argsort(hit_stop_idx)
    sorted_stop_idx = hit_stop_idx[sort_order]
    sorted_point_idx = hit_point_idx[sort_order]

    # Map point → group once
    group_ids = gps_df.groupby(group_cols, sort=False, observed=True).ngroup().values

    # For each stop: find unique groups that visited it, then expand by routes.
    # This deduplicates at group level BEFORE exploding by route, reducing
    # ~1.6B hit rows to ~millions of (group, route, stop) triples.
    rows_group = []
    rows_route = []
    rows_stop = []

    for si in tqdm(unique_stops_in_hits, desc="Bus route seq (aggregate)",
                   disable=len(unique_stops_in_hits) < 100):
        codes = stop_route_codes.get(int(si))
        if not codes:
            continue
        left = np.searchsorted(sorted_stop_idx, si, side="left")
        right = np.searchsorted(sorted_stop_idx, si, side="right")
        pts = sorted_point_idx[left:right]
        unique_groups = np.unique(group_ids[pts])
        n_g = len(unique_groups)
        for rc in codes:
            rows_group.append(unique_groups)
            rows_route.append(np.full(n_g, rc, dtype=np.int32))
            rows_stop.append(np.full(n_g, int(si), dtype=np.int32))

    agg_df = pd.DataFrame({
        "group_id": np.concatenate(rows_group),
        "route_code": np.concatenate(rows_route),
        "stop_idx": np.concatenate(rows_stop),
    })

    logger.info("Bus route sequence: %d aggregation rows from %d hits, %d routes",
                len(agg_df), len(hit_point_idx), len(route_id_to_code))

    # ── Aggregate: per (group, route), count unique stops ─────────────
    route_stop_counts = agg_df.groupby(["group_id", "route_code"])["stop_idx"].nunique()

    # Per group: max stops from any single route, and count of routes with ≥2 stops
    group_stats = route_stop_counts.groupby("group_id").agg(
        BUS_SEQ_STOPS="max",
        BUS_SEQ_ROUTES=lambda x: (x >= 2).sum(),
    )

    # ── Build output aligned to groupby order ─────────────────────────
    result = gps_df.groupby(group_cols, sort=False, observed=True).ngroup()
    result_df = gps_df.groupby(group_cols, sort=False, observed=True).size().reset_index(name="_n")
    result_df = result_df.drop(columns=["_n"])
    # Group IDs in order of first appearance
    ordered_gids = gps_df.groupby(group_cols, sort=False, observed=True).ngroup().drop_duplicates().values

    result_df["BUS_SEQ_STOPS"] = [
        int(group_stats.loc[gid, "BUS_SEQ_STOPS"]) if gid in group_stats.index else 0
        for gid in ordered_gids
    ]
    result_df["BUS_SEQ_ROUTES"] = [
        int(group_stats.loc[gid, "BUS_SEQ_ROUTES"]) if gid in group_stats.index else 0
        for gid in ordered_gids
    ]

    if key_col != "trip_key" and key_col in result_df.columns:
        result_df = result_df.rename(columns={key_col: "trip_key"})
    if segment_col is not None and segment_col in result_df.columns and segment_col != "segment":
        result_df = result_df.rename(columns={segment_col: "segment"})

    logger.info("Bus route sequence features: %d groups, max BUS_SEQ_STOPS=%d",
                len(result_df), result_df["BUS_SEQ_STOPS"].max() if len(result_df) > 0 else 0)
    return result_df


# ── GTFS loading helpers ─────────────────────────────────────────────


def load_gtfs_stops(gtfs_dir: Path) -> pd.DataFrame:
    """Load and merge stop data from GTFS zip files in a directory.

    Returns a DataFrame with ``stop_id``, ``stop_lat``, ``stop_lon``,
    ``route_type_set``, and ``bus_route_id_set`` columns.
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

    # Build stop → route_type and stop → bus_route_id mappings
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

        # Bus route IDs per stop (route_type == 3)
        bus_routes = stop_routes[stop_routes["route_type"] == 3]
        if not bus_routes.empty:
            bus_route_sets = bus_routes.groupby("stop_id")["route_id"].apply(
                lambda x: list(set(x.astype(str)))
            ).reset_index()
            bus_route_sets.columns = ["stop_id", "bus_route_id_set"]
            stops = stops.merge(bus_route_sets, on="stop_id", how="left")
        else:
            stops["bus_route_id_set"] = np.nan
        stops["bus_route_id_set"] = stops["bus_route_id_set"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    else:
        stops["route_type_set"] = [[] for _ in range(len(stops))]
        stops["bus_route_id_set"] = [[] for _ in range(len(stops))]

    required = {"stop_id", "stop_lat", "stop_lon"}
    if not required.issubset(stops.columns):
        raise ValueError(f"GTFS stops missing columns: {required - set(stops.columns)}")

    return stops[["stop_id", "stop_lat", "stop_lon", "route_type_set", "bus_route_id_set"]]
