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

    from tqdm import tqdm

    groups = gps_df.groupby(group_cols, sort=False, observed=True)
    rows = []
    for group_key, grp in tqdm(groups, desc="Trip statistics", unit="trip"):
        if isinstance(group_key, str):
            trip_key = group_key
            seg_id = None
        else:
            trip_key = group_key[0]
            seg_id = group_key[1] if len(group_key) > 1 else None

        if pd.isna(trip_key):
            continue

        row = {
            "trip_key": trip_key,
            "TOTAL_TIME": _total_time(grp[config.col_timestamp]),
            "AVG_PT_DIST_WGS84": _safe_mean(grp.get("dist_bp_wgs84")),
            "MED_PT_DIST_WGS84": _safe_median(grp.get("dist_bp_wgs84")),
            "DIST_TOTAL_WGS84": _safe_sum(grp.get("dist_bp_wgs84")),
            "GPS_POINTS": len(grp),
        }

        # Extra features: straightness, gaps, speed max, dt std
        dist_total = row["DIST_TOTAL_WGS84"]
        row["STRAIGHTNESS"] = _straightness(grp, config, dist_total)
        row["SPEED_MAX"] = _safe_max(grp.get("speed_ms"))
        _add_gap_features(row, grp[config.col_timestamp])

        # Paper-inspired features: stop fraction, brief stops, velocity entropy, time/day
        speed_series = grp.get("speed_ms")
        _add_stop_features(row, speed_series, grp[config.col_timestamp])
        row["VELOCITY_ENTROPY"] = _velocity_entropy(speed_series)
        _add_time_features(row, grp[config.col_timestamp])

        if seg_id is not None:
            row["segment"] = seg_id

        for src_col, prefix in _AGG_FEATURES:
            series = grp.get(src_col)
            row[f"{prefix}_AVG"] = _safe_mean(series)
            row[f"{prefix}_MED"] = _safe_median(series)
            row[f"{prefix}_STD"] = _safe_std(series)

        rows.append(row)

    df_stats = pd.DataFrame(rows)

    # Optionally join labels from trips_df
    if trips_df is not None and config.col_main_mode in trips_df.columns:
        mode_map = trips_df.set_index(config.col_trip_key)[config.col_main_mode].to_dict()
        df_stats[config.col_main_mode] = df_stats["trip_key"].map(mode_map)

    return df_stats


# ── Helpers ──────────────────────────────────────────────────────────


def _straightness(grp: pd.DataFrame, config, dist_total: float) -> float:
    """Ratio of straight-line distance to total distance travelled."""
    if dist_total <= 0 or len(grp) < 2 or np.isnan(dist_total):
        return np.nan
    lat = grp[config.col_latitude].astype("float64")
    lon = grp[config.col_longitude].astype("float64")
    from .._geo import haversine_m
    straight = haversine_m(
        lat.iloc[0], lon.iloc[0], lat.iloc[-1], lon.iloc[-1],
    )
    return float(min(straight / dist_total, 1.0))


def _safe_max(s: pd.Series | None) -> float:
    if s is None:
        return np.nan
    s = pd.to_numeric(s, errors="coerce")
    return float(s.max()) if not s.empty and not s.isna().all() else np.nan


def _add_gap_features(row: dict, ts: pd.Series) -> None:
    """Add GAP_MAX, GAP_MEAN, DT_STD to row dict."""
    ts = pd.to_datetime(ts)
    if len(ts) < 2:
        row["GAP_MAX"] = np.nan
        row["GAP_MEAN"] = np.nan
        row["DT_STD"] = np.nan
        return
    dt = ts.sort_values().diff().dt.total_seconds().iloc[1:]
    row["GAP_MAX"] = float(dt.max()) if not dt.empty else np.nan
    row["GAP_MEAN"] = float(dt.mean()) if not dt.empty else np.nan
    row["DT_STD"] = float(dt.std()) if not dt.empty else np.nan


def _add_stop_features(row: dict, speed: pd.Series | None, ts: pd.Series) -> None:
    """Add STOP_FRACTION and BRIEF_STOP_RATE."""
    if speed is None or speed.empty:
        row["STOP_FRACTION"] = np.nan
        row["BRIEF_STOP_RATE"] = np.nan
        return
    speed = pd.to_numeric(speed, errors="coerce")
    # Stop fraction: % of points with speed < 1 m/s
    row["STOP_FRACTION"] = float((speed < 1.0).sum()) / len(speed)

    # Brief stop rate: fraction of stops lasting 30-120 seconds
    # A "stop" is a contiguous run of points with speed < 1 m/s
    ts = pd.to_datetime(ts)
    is_stopped = (speed < 1.0).values
    if not is_stopped.any() or len(ts) < 2:
        row["BRIEF_STOP_RATE"] = 0.0
        return
    # Find contiguous stop runs
    n_stops = 0
    n_brief = 0
    in_stop = False
    stop_start = 0
    ts_vals = ts.values
    for i in range(len(is_stopped)):
        if is_stopped[i] and not in_stop:
            in_stop = True
            stop_start = i
        elif not is_stopped[i] and in_stop:
            in_stop = False
            n_stops += 1
            dur = (ts_vals[i - 1] - ts_vals[stop_start]) / np.timedelta64(1, "s")
            if 30 <= dur <= 120:
                n_brief += 1
    if in_stop:
        n_stops += 1
        dur = (ts_vals[-1] - ts_vals[stop_start]) / np.timedelta64(1, "s")
        if 30 <= dur <= 120:
            n_brief += 1
    row["BRIEF_STOP_RATE"] = float(n_brief) / max(n_stops, 1)


def _velocity_entropy(speed: pd.Series | None, n_bins: int = 10) -> float:
    """Shannon entropy of the speed distribution (binned)."""
    if speed is None or speed.empty:
        return np.nan
    speed = pd.to_numeric(speed, errors="coerce").dropna()
    if len(speed) < 2:
        return np.nan
    counts, _ = np.histogram(speed, bins=n_bins, range=(0, speed.max() + 1e-9))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _add_time_features(row: dict, ts: pd.Series) -> None:
    """Add sin/cos encoded time-of-day and day-of-week."""
    ts = pd.to_datetime(ts)
    if ts.empty:
        row["TOD_SIN"] = np.nan
        row["TOD_COS"] = np.nan
        row["DOW_SIN"] = np.nan
        row["DOW_COS"] = np.nan
        return
    # Use midpoint of the trip
    mid_ts = ts.iloc[len(ts) // 2]
    hour_frac = mid_ts.hour + mid_ts.minute / 60.0
    row["TOD_SIN"] = float(np.sin(2 * np.pi * hour_frac / 24.0))
    row["TOD_COS"] = float(np.cos(2 * np.pi * hour_frac / 24.0))
    dow = mid_ts.dayofweek  # 0=Monday
    row["DOW_SIN"] = float(np.sin(2 * np.pi * dow / 7.0))
    row["DOW_COS"] = float(np.cos(2 * np.pi * dow / 7.0))


def _safe_mean(s: pd.Series | None) -> float:
    if s is None:
        return np.nan
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if not s.empty and not s.isna().all() else np.nan


def _safe_median(s: pd.Series | None) -> float:
    if s is None:
        return np.nan
    s = pd.to_numeric(s, errors="coerce")
    return float(s.median()) if not s.empty and not s.isna().all() else np.nan


def _safe_std(s: pd.Series | None) -> float:
    if s is None:
        return np.nan
    s = pd.to_numeric(s, errors="coerce")
    return float(s.std()) if not s.empty and not s.isna().all() else np.nan


def _safe_sum(s: pd.Series | None) -> float:
    if s is None:
        return np.nan
    return float(s.sum(min_count=1))


def _total_time(ts: pd.Series) -> float:
    ts = pd.to_datetime(ts)
    if ts.empty:
        return 0.0
    return (ts.max() - ts.min()).total_seconds()
