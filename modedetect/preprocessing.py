"""Data cleaning and preprocessing.

Handles: coordinate cleaning, mode remapping, temporal gap handling,
speed outlier removal, and trip filtering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import ModeDetectConfig

logger = logging.getLogger(__name__)


# ── Coordinate cleaning ──────────────────────────────────────────────


def remove_invalid_coordinates(df: pd.DataFrame, config: ModeDetectConfig) -> pd.DataFrame:
    """Remove GPS points with lat/lon outside valid ranges. Warns with count."""
    lat = df[config.col_latitude]
    lon = df[config.col_longitude]
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    # Also drop NaN coords
    valid = valid & lat.notna() & lon.notna()
    n_removed = int((~valid).sum())
    if n_removed > 0:
        logger.warning("Removed %d GPS points with invalid coordinates.", n_removed)
    return df.loc[valid].copy()


# ── Mode remapping ───────────────────────────────────────────────────


def remap_and_drop_modes(
    df: pd.DataFrame,
    config: ModeDetectConfig,
) -> pd.DataFrame:
    """Apply mode consolidation and remove unwanted modes.

    Operates on the column named by config.col_main_mode.
    """
    col = config.col_main_mode
    out = df.copy()
    out[col] = out[col].replace(config.mode_remap)
    if config.modes_to_drop:
        mask = ~out[col].isin(config.modes_to_drop)
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            logger.info("Dropped %d rows with modes in modes_to_drop.", n_dropped)
        out = out.loc[mask]
    return out


# ── Temporal gap handling ────────────────────────────────────────────


def drop_trips_with_gaps(
    df: pd.DataFrame,
    gaps: dict[str, list[int]],
    config: ModeDetectConfig,
) -> pd.DataFrame:
    """For fit(): drop entire trips that have temporal gaps."""
    if not gaps:
        return df
    keys_to_drop = set(gaps.keys())
    mask = ~df[config.col_trip_key].isin(keys_to_drop)
    logger.info("Dropped %d trips with temporal gaps > %.0fs.", len(keys_to_drop), config.gap_threshold_s)
    return df.loc[mask].copy()


def split_trips_at_gaps(
    df: pd.DataFrame,
    gaps: dict[str, list[int]],
    config: ModeDetectConfig,
) -> pd.DataFrame:
    """For predict(): split trips at temporal gaps into sub-trips.

    New trip keys are formatted as '{original_key}__{segment_index}'.
    Trips without gaps are left unchanged.
    """
    if not gaps:
        return df

    frames = []
    keys_with_gaps = set(gaps.keys())

    # Pass through trips without gaps
    frames.append(df.loc[~df[config.col_trip_key].isin(keys_with_gaps)])

    for key, gap_positions in gaps.items():
        trip = df.loc[df[config.col_trip_key] == key].sort_values(config.col_timestamp)
        # gap_positions are indices within the sorted trip where gaps occur
        split_points = [0] + gap_positions + [len(trip)]
        for seg_idx in range(len(split_points) - 1):
            start = split_points[seg_idx]
            end = split_points[seg_idx + 1]
            segment = trip.iloc[start:end].copy()
            segment[config.col_trip_key] = f"{key}__{seg_idx}"
            frames.append(segment)

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Split %d trips at temporal gaps into %d sub-trips.",
        len(gaps),
        sum(len(pos) + 1 for pos in gaps.values()),
    )
    return result


# ── Speed outlier filtering ──────────────────────────────────────────


def compute_speed_thresholds(
    df: pd.DataFrame,
    config: ModeDetectConfig,
    speed_col: str = "speed_ms",
) -> dict[str, float]:
    """Compute 99th percentile speed per mode from training data.

    Returns a dict like {'WALKING': 6.2, 'CAR': 38.5, ...} and
    also the global max across all modes (stored in artifact for predict).
    """
    thresholds = {}
    for mode, grp in df.groupby(config.col_main_mode):
        if speed_col in grp.columns:
            p99 = float(grp[speed_col].quantile(0.99))
            thresholds[mode] = p99
    return thresholds


def filter_speed_outliers_fit(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    config: ModeDetectConfig,
    speed_col: str = "speed_ms",
) -> pd.DataFrame:
    """For fit(): remove entire trips where any point exceeds the per-mode threshold."""
    per_row_threshold = df[config.col_main_mode].map(thresholds)
    exceeds = df[speed_col] > per_row_threshold
    # Find trip keys with any offending point
    bad_keys = set(df.loc[exceeds.fillna(False), config.col_trip_key].unique())
    if bad_keys:
        logger.info("Removed %d trips with speed outliers exceeding per-mode 99th percentile.", len(bad_keys))
        return df.loc[~df[config.col_trip_key].isin(bad_keys)].copy()
    return df


def filter_speed_outliers_predict(
    df: pd.DataFrame,
    max_speed_ms: float,
    config: ModeDetectConfig,
    speed_col: str = "speed_ms",
) -> pd.DataFrame:
    """For predict(): remove individual GPS points exceeding the speed ceiling."""
    if speed_col not in df.columns:
        return df
    exceeds = df[speed_col] > max_speed_ms
    n_removed = int(exceeds.sum())
    if n_removed > 0:
        logger.info("Removed %d GPS points exceeding speed ceiling of %.1f m/s.", n_removed, max_speed_ms)
    return df.loc[~exceeds].copy()


# ── Trip filtering (point count, duration) ───────────────────────────


def drop_short_trips(
    df: pd.DataFrame,
    min_points: int,
    config: ModeDetectConfig,
) -> pd.DataFrame:
    """Drop trips with fewer than min_points GPS points."""
    counts = df.groupby(config.col_trip_key).size()
    keep_keys = counts[counts >= min_points].index
    n_dropped = int(counts.shape[0] - len(keep_keys))
    if n_dropped > 0:
        logger.info("Dropped %d trips with fewer than %d GPS points.", n_dropped, min_points)
    return df.loc[df[config.col_trip_key].isin(keep_keys)].copy()


def drop_brief_trips(
    df: pd.DataFrame,
    min_duration_s: int,
    config: ModeDetectConfig,
) -> pd.DataFrame:
    """Drop trips shorter than min_duration_s seconds."""
    grouped = df.groupby(config.col_trip_key)[config.col_timestamp]
    durations = grouped.max() - grouped.min()
    keep_keys = durations[durations >= pd.Timedelta(seconds=min_duration_s)].index
    n_dropped = int(durations.shape[0] - len(keep_keys))
    if n_dropped > 0:
        logger.info("Dropped %d trips shorter than %ds.", n_dropped, min_duration_s)
    return df.loc[df[config.col_trip_key].isin(keep_keys)].copy()


# ── Detect multi-modal labels ────────────────────────────────────────


def detect_multimodal_columns(trips_df: pd.DataFrame, config: ModeDetectConfig) -> bool:
    """Auto-detect whether trips_df contains multi-modal label information.

    Supports:
      - Separate columns: mode_1, mode_2, ... (or Mode_1, Mode_2, ...)
      - A single list/set column: 'modes'
    """
    cols = set(trips_df.columns)

    # Check for a list-type 'modes' column
    if "modes" in cols:
        sample = trips_df["modes"].dropna().head(5)
        if sample.apply(lambda x: isinstance(x, (list, set, tuple))).all():
            return True

    # Check for numbered mode columns (mode_1, mode_2, ...)
    mode_cols = sorted(c for c in cols if c.lower().startswith("mode_") and c.split("_")[-1].isdigit())
    if mode_cols:
        # Multi-modal if any trip has a non-null value in mode_2+
        extra_cols = [c for c in mode_cols if int(c.split("_")[-1]) >= 2]
        if extra_cols and trips_df[extra_cols].notna().any().any():
            return True

    return False
