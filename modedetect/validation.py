"""Input data validators.

Three layers:
  1. Structural checks  – hard failures (ModeDetectError)
  2. Quality checks     – warnings via logging
  3. External data      – warn if GTFS/OSM missing or misconfigured

All error messages are actionable: they tell the user what is wrong and
how to fix it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import ModeDetectConfig

logger = logging.getLogger(__name__)


class ModeDetectError(Exception):
    """Raised when input data cannot be processed."""


# ── 1. Structural checks (hard fail) ────────────────────────────────


def validate_gps_schema(df: pd.DataFrame, config: ModeDetectConfig) -> None:
    """Verify that the GPS DataFrame has the required columns."""
    required = {config.col_latitude, config.col_longitude, config.col_timestamp, config.col_trip_key}
    present = set(df.columns)
    missing = required - present
    if missing:
        raise ModeDetectError(
            f"GPS data missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(required)}. "
            f"Adjust config.col_* to match your column names."
        )


def validate_trips_schema(df: pd.DataFrame, config: ModeDetectConfig) -> None:
    """Verify that the trips DataFrame has the required columns (fit only)."""
    required = {config.col_trip_key, config.col_main_mode}
    present = set(df.columns)
    missing = required - present
    if missing:
        # Suggest close matches
        hints = []
        for col in missing:
            close = [c for c in present if col.lower() in c.lower() or c.lower() in col.lower()]
            if close:
                hints.append(f"Did you mean '{close[0]}' for '{col}'?")
        hint_str = " ".join(hints) if hints else f"Set config.col_main_mode and config.col_trip_key to match your data."
        raise ModeDetectError(
            f"Trips data missing required columns: {sorted(missing)}. {hint_str}"
        )


def validate_gps_values(df: pd.DataFrame, config: ModeDetectConfig) -> None:
    """Check that lat/lon values are within valid ranges.

    Out-of-range points are not an error — they are reported so the caller
    can decide to remove them (see preprocessing.remove_invalid_coordinates).
    """
    lat = df[config.col_latitude]
    lon = df[config.col_longitude]

    bad_lat = ~lat.between(-90, 90) & lat.notna()
    bad_lon = ~lon.between(-180, 180) & lon.notna()
    bad = bad_lat | bad_lon
    n_bad = int(bad.sum())

    if n_bad > 0:
        first = df.loc[bad.idxmax()]
        logger.warning(
            "%d GPS points have coordinates outside valid range. "
            "First invalid: lat=%.4f, lon=%.4f at %s='%s'. "
            "These points will be removed.",
            n_bad,
            first[config.col_latitude],
            first[config.col_longitude],
            config.col_trip_key,
            first[config.col_trip_key],
        )

    return n_bad


def validate_timestamps(df: pd.DataFrame, config: ModeDetectConfig) -> None:
    """Ensure timestamps are parseable as datetime."""
    ts = df[config.col_timestamp]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        try:
            pd.to_datetime(ts.head(100))
        except (ValueError, TypeError) as exc:
            raise ModeDetectError(
                f"Column '{config.col_timestamp}' cannot be parsed as datetime. "
                f"First value: '{ts.iloc[0]}'. Error: {exc}"
            ) from exc


def validate_key_overlap(
    gps_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    config: ModeDetectConfig,
) -> None:
    """Check that GPS and trips DataFrames share trip keys."""
    gps_keys = set(gps_df[config.col_trip_key].dropna().unique())
    trip_keys = set(trips_df[config.col_trip_key].dropna().unique())
    overlap = gps_keys & trip_keys

    if not overlap:
        raise ModeDetectError(
            f"No matching trip keys between GPS data ({len(gps_keys)} unique keys) "
            f"and trips data ({len(trip_keys)} unique keys). "
            f"Check that both use the same format in column '{config.col_trip_key}'."
        )

    if len(overlap) < min(len(gps_keys), len(trip_keys)) * 0.5:
        logger.warning(
            "Only %d of %d GPS trip keys match the trips data (%d keys). "
            "%.0f%% of data may be unused.",
            len(overlap),
            len(gps_keys),
            len(trip_keys),
            (1 - len(overlap) / len(gps_keys)) * 100,
        )


# ── 2. Quality checks (warnings) ────────────────────────────────────


def warn_short_trips(df: pd.DataFrame, config: ModeDetectConfig) -> list[str]:
    """Identify trips with too few GPS points. Returns list of trip keys to drop."""
    counts = df.groupby(config.col_trip_key).size()
    short = counts[counts < config.min_gps_points]
    if len(short) > 0:
        logger.warning(
            "%d trips have fewer than %d GPS points and will be dropped.",
            len(short),
            config.min_gps_points,
        )
    return short.index.tolist()


def warn_brief_trips(df: pd.DataFrame, config: ModeDetectConfig) -> list[str]:
    """Identify trips shorter than min_trip_duration_s. Returns list of trip keys to drop."""
    grouped = df.groupby(config.col_trip_key)[config.col_timestamp]
    durations = grouped.max() - grouped.min()
    brief = durations[durations < pd.Timedelta(seconds=config.min_trip_duration_s)]
    if len(brief) > 0:
        logger.warning(
            "%d trips are shorter than %ds and will be dropped.",
            len(brief),
            config.min_trip_duration_s,
        )
    return brief.index.tolist()


def warn_temporal_gaps(df: pd.DataFrame, config: ModeDetectConfig) -> dict[str, list[int]]:
    """Find trips with large temporal gaps. Returns {trip_key: [gap_indices]}.

    These trips will be dropped during fit() or split during predict().
    """
    trips_with_gaps = {}
    for key, grp in df.groupby(config.col_trip_key):
        ts = grp[config.col_timestamp].sort_values()
        dt = ts.diff().dt.total_seconds()
        gap_mask = dt > config.gap_threshold_s
        if gap_mask.any():
            gap_positions = list(np.where(gap_mask.values)[0])
            trips_with_gaps[key] = gap_positions

    if trips_with_gaps:
        logger.warning(
            "%d trips have temporal gaps > %.0fs.",
            len(trips_with_gaps),
            config.gap_threshold_s,
        )
    return trips_with_gaps


def warn_high_nan_features(df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.5) -> None:
    """Warn about columns with high NaN rates after feature extraction."""
    for col in feature_cols:
        if col not in df.columns:
            continue
        nan_pct = df[col].isna().mean()
        if nan_pct > threshold:
            logger.warning(
                "%.0f%% of '%s' values are NaN — possible GPS quality issue.",
                nan_pct * 100,
                col,
            )


# ── 3. External data checks ─────────────────────────────────────────


def validate_gtfs(config: ModeDetectConfig) -> bool:
    """Check GTFS directory. Returns True if usable, False otherwise."""
    if config.gtfs_dir is None:
        logger.info("No GTFS directory configured — stop proximity features will be skipped.")
        return False

    gtfs_path = config.gtfs_dir
    if not gtfs_path.exists():
        logger.warning("GTFS directory '%s' does not exist. Stop proximity features unavailable.", gtfs_path)
        return False

    zips = list(gtfs_path.glob("*.zip"))
    if not zips:
        logger.warning("GTFS directory '%s' contains no .zip files. Stop proximity features unavailable.", gtfs_path)
        return False

    return True


def validate_osm(config: ModeDetectConfig) -> bool:
    """Check OSM route data. Returns True if usable, False otherwise."""
    if config.osm_routes_dir is not None:
        d = config.osm_routes_dir
        if not d.exists():
            logger.warning("OSM routes directory '%s' does not exist. Route features unavailable.", d)
            return False
        expected = {"bus.parquet", "tram.parquet", "train.parquet", "subway.parquet"}
        found = {f.name for f in d.glob("*.parquet")}
        missing = expected - found
        if missing:
            logger.warning(
                "OSM routes directory '%s' missing files: %s. Run ModeDetector.prepare_osm() first.",
                d,
                sorted(missing),
            )
            return False
        return True

    if config.osm_pbf_path is not None:
        if not config.osm_pbf_path.exists():
            logger.warning("OSM PBF file '%s' does not exist. Route features unavailable.", config.osm_pbf_path)
            return False
        return True

    logger.info("No OSM data configured — route matching features will be skipped.")
    return False


# ── Convenience: run all validations ─────────────────────────────────


def validate_fit_inputs(
    gps_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    config: ModeDetectConfig,
) -> None:
    """Run all structural validations needed before fit(). Raises on failure."""
    validate_gps_schema(gps_df, config)
    validate_trips_schema(trips_df, config)
    validate_timestamps(gps_df, config)
    validate_gps_values(gps_df, config)
    validate_key_overlap(gps_df, trips_df, config)


def validate_predict_inputs(
    gps_df: pd.DataFrame,
    config: ModeDetectConfig,
) -> None:
    """Run all structural validations needed before predict(). Raises on failure."""
    validate_gps_schema(gps_df, config)
    validate_timestamps(gps_df, config)
    validate_gps_values(gps_df, config)
