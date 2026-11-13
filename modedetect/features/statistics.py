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

from .._geo import haversine_m

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

        total_dist = _safe_sum(grp.get("dist_bp_wgs84"))
        total_time = _total_time(grp[config.col_timestamp])

        row = {
            "trip_key": trip_key,
            "TOTAL_TIME": total_time,
            "AVG_PT_DIST_WGS84": _safe_mean(grp.get("dist_bp_wgs84")),
            "MED_PT_DIST_WGS84": _safe_median(grp.get("dist_bp_wgs84")),
            "DIST_TOTAL_WGS84": total_dist,
            "GPS_POINTS": len(grp),
        }

        if seg_id is not None:
            row["segment"] = seg_id

        for src_col, prefix in _AGG_FEATURES:
            series = grp.get(src_col)
            row[f"{prefix}_AVG"] = _safe_mean(series)
            row[f"{prefix}_MED"] = _safe_median(series)
            row[f"{prefix}_STD"] = _safe_std(series)

        # ── Straightness index: direct distance / total distance ──
        # Low for walking (meandering), high for car/train (direct routes)
        lat = grp[config.col_latitude].astype("float64")
        lon = grp[config.col_longitude].astype("float64")
        if len(grp) >= 2 and total_dist and total_dist > 0:
            direct_dist = haversine_m(
                np.array([lat.iloc[0]]), np.array([lon.iloc[0]]),
                np.array([lat.iloc[-1]]), np.array([lon.iloc[-1]]),
            )[0]
            row["STRAIGHTNESS"] = direct_dist / total_dist if np.isfinite(direct_dist) else np.nan
        else:
            row["STRAIGHTNESS"] = np.nan

        # ── GPS signal quality features ──
        # Helps identify subway (GPS dropouts underground) vs walking
        ts = pd.to_datetime(grp[config.col_timestamp])
        if len(ts) >= 2:
            dt_arr = ts.diff().dt.total_seconds().iloc[1:].values
            dt_valid = dt_arr[np.isfinite(dt_arr)]
            if len(dt_valid) > 0:
                median_dt = np.median(dt_valid)
                # Gaps: intervals much larger than the median sampling rate
                gap_threshold = max(median_dt * 3, 30.0)  # at least 30s
                gaps = dt_valid[dt_valid > gap_threshold]
                row["GAP_COUNT"] = len(gaps)
                row["GAP_FRAC"] = len(gaps) / len(dt_valid)
                row["GAP_DURATION_MEAN"] = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
                row["GAP_DURATION_TOTAL"] = float(np.sum(gaps)) if len(gaps) > 0 else 0.0
                row["DT_STD"] = float(np.std(dt_valid))
                row["SPEED_MAX"] = float(grp["speed_ms"].max()) if "speed_ms" in grp.columns else np.nan
            else:
                row.update({"GAP_COUNT": 0, "GAP_FRAC": 0.0, "GAP_DURATION_MEAN": 0.0,
                            "GAP_DURATION_TOTAL": 0.0, "DT_STD": np.nan, "SPEED_MAX": np.nan})
        else:
            row.update({"GAP_COUNT": 0, "GAP_FRAC": 0.0, "GAP_DURATION_MEAN": 0.0,
                        "GAP_DURATION_TOTAL": 0.0, "DT_STD": np.nan, "SPEED_MAX": np.nan})

        rows.append(row)

    df_stats = pd.DataFrame(rows)

    # Optionally join labels from trips_df
    if trips_df is not None and config.col_main_mode in trips_df.columns:
        mode_map = trips_df.set_index(config.col_trip_key)[config.col_main_mode].to_dict()
        df_stats[config.col_main_mode] = df_stats["trip_key"].map(mode_map)

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
