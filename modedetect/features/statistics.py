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

    rows = []
    for group_key, grp in gps_df.groupby(group_cols, sort=False, observed=True):
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
        df_stats["main_mode"] = df_stats["trip_key"].map(mode_map)

    return df_stats


# ── Helpers ──────────────────────────────────────────────────────────


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
