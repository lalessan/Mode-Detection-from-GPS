"""Trip-level (or segment-level) statistical aggregation of motion features.

Produces one row per trip (or per segment) with mean, median, std, 85th percentile
of all point-level motion features, plus total time, distance, and GPS point count.

Refactored from: 1_Intial_Setup_Motion_Features/statistical_summary_motionfeatures.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config import ModeDetectConfig

logger = logging.getLogger(__name__)

# Features to aggregate and their output prefixes
_AGG_FEATURES = [
    ("speed_ms", "SPEED"),
    ("acc_ms2", "ACC"),
    ("jerk_ms3", "JERK"),
    ("bearing_change_deg_wgs84", "BC_WGS84"),
    ("bearing_rate_deg_wgs84", "BR_WGS84"),
]


def compute_trip_statistics(
    gps_df: pd.DataFrame,
    config: ModeDetectConfig,
    trips_df: pd.DataFrame | None = None,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate point-level features into one row per trip (or per segment).

    Uses vectorized groupby operations instead of per-group Python loops.

    Parameters
    ----------
    gps_df : DataFrame
        Point-level GPS data with motion features already computed
        (output of ``features.motion.extract_point_features``).
    config : ModeDetectConfig
    trips_df : DataFrame, optional
        If provided (during fit), ``main_mode`` is joined from this table.
    segment_col : str, optional
        If provided, aggregation is done per (trip_key, segment_col) pair
        instead of per trip_key alone.
    """
    key_col = config.col_trip_key
    group_cols = [key_col] if segment_col is None else [key_col, segment_col]

    # Drop rows with NaN trip key
    gps_df = gps_df.dropna(subset=[key_col])

    # Ensure timestamp is datetime
    ts_col = config.col_timestamp
    gps_df[ts_col] = pd.to_datetime(gps_df[ts_col])

    grouped = gps_df.groupby(group_cols, sort=False, observed=True)

    # ── Basic aggregations (fully vectorized) ────────────────────────
    logger.info("Computing trip statistics (vectorized)...")

    # GPS_POINTS
    df_stats = grouped.size().reset_index(name="GPS_POINTS")

    # TOTAL_TIME: max(ts) - min(ts) in seconds
    ts_agg = grouped[ts_col].agg(["min", "max"])
    df_stats["TOTAL_TIME"] = (ts_agg["max"] - ts_agg["min"]).dt.total_seconds().values

    # Distance features
    if "dist_bp_wgs84" in gps_df.columns:
        dist_agg = grouped["dist_bp_wgs84"].agg(["mean", "median", "sum"])
        df_stats["AVG_PT_DIST_WGS84"] = dist_agg["mean"].values
        df_stats["MED_PT_DIST_WGS84"] = dist_agg["median"].values
        df_stats["DIST_TOTAL_WGS84"] = dist_agg["sum"].values
    else:
        df_stats["AVG_PT_DIST_WGS84"] = np.nan
        df_stats["MED_PT_DIST_WGS84"] = np.nan
        df_stats["DIST_TOTAL_WGS84"] = np.nan

    # SPEED_MAX
    if "speed_ms" in gps_df.columns:
        df_stats["SPEED_MAX"] = grouped["speed_ms"].max().values

    # ── Standard agg features: mean, median, std ─────────────────────
    for src_col, prefix in _AGG_FEATURES:
        if src_col in gps_df.columns:
            col_agg = grouped[src_col].agg(["mean", "median", "std"])
            df_stats[f"{prefix}_AVG"] = col_agg["mean"].values
            df_stats[f"{prefix}_MED"] = col_agg["median"].values
            df_stats[f"{prefix}_STD"] = col_agg["std"].values
        else:
            df_stats[f"{prefix}_AVG"] = np.nan
            df_stats[f"{prefix}_MED"] = np.nan
            df_stats[f"{prefix}_STD"] = np.nan

    # ── Gap features (vectorized via diff within groups) ─────────────
    dt_seconds = gps_df.sort_values([key_col, ts_col]).groupby(
        group_cols, sort=False, observed=True
    )[ts_col].diff().dt.total_seconds()

    # Assign group keys for aggregation
    dt_df = pd.DataFrame({
        "dt": dt_seconds.values,
    })
    for i, col in enumerate(group_cols):
        dt_df[col] = gps_df.sort_values([key_col, ts_col])[col].values
    dt_df = dt_df.dropna(subset=["dt"])

    if not dt_df.empty:
        dt_grouped = dt_df.groupby(group_cols, sort=False, observed=True)["dt"]
        gap_agg = dt_grouped.agg(["max", "mean", "std"])
        # Merge on group_cols
        df_stats = df_stats.merge(
            gap_agg.reset_index().rename(columns={"max": "GAP_MAX", "mean": "GAP_MEAN", "std": "DT_STD"}),
            on=group_cols, how="left",
        )
    else:
        df_stats["GAP_MAX"] = np.nan
        df_stats["GAP_MEAN"] = np.nan
        df_stats["DT_STD"] = np.nan

    # ── Straightness (vectorized: first/last point per group) ────────
    lat_col = config.col_latitude
    lon_col = config.col_longitude

    if lat_col in gps_df.columns and lon_col in gps_df.columns and "dist_bp_wgs84" in gps_df.columns:
        first_pts = grouped.agg({lat_col: "first", lon_col: "first"}).rename(
            columns={lat_col: "lat1", lon_col: "lon1"}
        )
        last_pts = grouped.agg({lat_col: "last", lon_col: "last"}).rename(
            columns={lat_col: "lat2", lon_col: "lon2"}
        )
        endpoints = first_pts.join(last_pts)

        from .._geo import haversine_m
        straight_dist = haversine_m(
            endpoints["lat1"].astype("float64").values,
            endpoints["lon1"].astype("float64").values,
            endpoints["lat2"].astype("float64").values,
            endpoints["lon2"].astype("float64").values,
        )
        # Join dist_total from df_stats
        dist_total = df_stats.set_index(group_cols)["DIST_TOTAL_WGS84"].reindex(endpoints.index).values
        with np.errstate(divide="ignore", invalid="ignore"):
            straightness = np.where(
                (dist_total > 0) & ~np.isnan(dist_total),
                np.minimum(straight_dist / dist_total, 1.0),
                np.nan,
            )
        # Also NaN for single-point groups
        n_points = df_stats.set_index(group_cols)["GPS_POINTS"].reindex(endpoints.index).values
        straightness = np.where(n_points >= 2, straightness, np.nan)
        df_stats["STRAIGHTNESS"] = straightness
    else:
        df_stats["STRAIGHTNESS"] = np.nan

    # ── Stop features (vectorized) ───────────────────────────────────
    if "speed_ms" in gps_df.columns:
        speed_numeric = pd.to_numeric(gps_df["speed_ms"], errors="coerce")
        is_stopped = (speed_numeric < 1.0)
        stopped_counts = is_stopped.groupby(
            [gps_df[c] for c in group_cols], sort=False, observed=True
        ).sum()
        total_counts = is_stopped.groupby(
            [gps_df[c] for c in group_cols], sort=False, observed=True
        ).count()
        stop_frac = (stopped_counts / total_counts).reset_index(name="STOP_FRACTION")
        df_stats = df_stats.merge(stop_frac, on=group_cols, how="left")

        # Brief stop rate — needs contiguous run detection, done per group
        # but using a faster numpy approach
        df_stats["BRIEF_STOP_RATE"] = _batch_brief_stop_rate(
            gps_df, group_cols, speed_numeric, ts_col
        )
    else:
        df_stats["STOP_FRACTION"] = np.nan
        df_stats["BRIEF_STOP_RATE"] = np.nan

    # ── Velocity entropy (per-group, but using fast numpy) ───────────
    if "speed_ms" in gps_df.columns:
        df_stats["VELOCITY_ENTROPY"] = _batch_velocity_entropy(
            gps_df, grouped, group_cols
        )
    else:
        df_stats["VELOCITY_ENTROPY"] = np.nan

    # ── Time features (vectorized via midpoint timestamp) ────────────
    # Use the midpoint index of each group
    mid_ts = grouped[ts_col].apply(lambda s: s.iloc[len(s) // 2])
    mid_ts = pd.to_datetime(mid_ts)
    if hasattr(mid_ts, "reset_index"):
        mid_ts = mid_ts.reset_index(name="_mid_ts") if isinstance(mid_ts, pd.Series) else mid_ts
        if "_mid_ts" not in mid_ts.columns:
            mid_ts = mid_ts.rename(columns={ts_col: "_mid_ts"})

    hour_frac = mid_ts["_mid_ts"].dt.hour + mid_ts["_mid_ts"].dt.minute / 60.0
    dow = mid_ts["_mid_ts"].dt.dayofweek

    df_stats["TOD_SIN"] = np.sin(2 * np.pi * hour_frac.values / 24.0)
    df_stats["TOD_COS"] = np.cos(2 * np.pi * hour_frac.values / 24.0)
    df_stats["DOW_SIN"] = np.sin(2 * np.pi * dow.values / 7.0)
    df_stats["DOW_COS"] = np.cos(2 * np.pi * dow.values / 7.0)

    # ── Rename key column and add segment ────────────────────────────
    if key_col != "trip_key":
        df_stats = df_stats.rename(columns={key_col: "trip_key"})
    if segment_col is not None and segment_col in df_stats.columns and segment_col != "segment":
        df_stats = df_stats.rename(columns={segment_col: "segment"})

    # Optionally join labels from trips_df
    if trips_df is not None and config.col_main_mode in trips_df.columns:
        mode_map = trips_df.set_index(config.col_trip_key)[config.col_main_mode].to_dict()
        df_stats[config.col_main_mode] = df_stats["trip_key"].map(mode_map)

    logger.info("Trip statistics: %d groups computed", len(df_stats))
    return df_stats


# ── Batch helpers ───────────────────────────────────────────────────


def _batch_brief_stop_rate(
    gps_df: pd.DataFrame,
    group_cols: list[str],
    speed_numeric: pd.Series,
    ts_col: str,
) -> np.ndarray:
    """Compute BRIEF_STOP_RATE per group using numpy run-length encoding."""
    is_stopped = (speed_numeric < 1.0).values
    ts_vals = gps_df[ts_col].values  # datetime64

    # Build group indices
    group_ids = gps_df.groupby(group_cols, sort=False, observed=True).ngroup().values
    n_groups = group_ids.max() + 1 if len(group_ids) > 0 else 0

    n_stops = np.zeros(n_groups, dtype=np.int64)
    n_brief = np.zeros(n_groups, dtype=np.int64)

    if len(is_stopped) == 0:
        return np.zeros(n_groups, dtype=np.float64)

    # Detect transitions: where group changes or stop state changes
    group_change = np.empty(len(group_ids), dtype=bool)
    group_change[0] = True
    group_change[1:] = group_ids[1:] != group_ids[:-1]

    # Walk through once, tracking stop runs
    in_stop = False
    stop_start = 0
    for i in range(len(is_stopped)):
        if group_change[i]:
            # Close previous stop run if any
            if in_stop and i > 0:
                gid = group_ids[i - 1]
                n_stops[gid] += 1
                dur = (ts_vals[i - 1] - ts_vals[stop_start]) / np.timedelta64(1, "s")
                if 30 <= dur <= 120:
                    n_brief[gid] += 1
            in_stop = False

        if is_stopped[i] and not in_stop:
            in_stop = True
            stop_start = i
        elif not is_stopped[i] and in_stop:
            in_stop = False
            gid = group_ids[i]
            n_stops[gid] += 1
            dur = (ts_vals[i - 1] - ts_vals[stop_start]) / np.timedelta64(1, "s")
            if 30 <= dur <= 120:
                n_brief[gid] += 1

    # Close final run
    if in_stop:
        gid = group_ids[-1]
        n_stops[gid] += 1
        dur = (ts_vals[-1] - ts_vals[stop_start]) / np.timedelta64(1, "s")
        if 30 <= dur <= 120:
            n_brief[gid] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(n_stops > 0, n_brief / n_stops, 0.0)

    # Map back to df_stats order (which uses the same groupby)
    group_order = gps_df.groupby(group_cols, sort=False, observed=True).ngroup()
    unique_gids = group_order.drop_duplicates().values
    return rates[unique_gids]


def _batch_velocity_entropy(
    gps_df: pd.DataFrame,
    grouped,
    group_cols: list[str],
    n_bins: int = 10,
) -> np.ndarray:
    """Compute Shannon entropy of speed distribution per group."""
    def _entropy(speed):
        speed = pd.to_numeric(speed, errors="coerce").dropna()
        if len(speed) < 2:
            return np.nan
        counts, _ = np.histogram(speed, bins=n_bins, range=(0, speed.max() + 1e-9))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    if "speed_ms" not in gps_df.columns:
        return np.full(len(grouped), np.nan)

    return grouped["speed_ms"].apply(_entropy).values
