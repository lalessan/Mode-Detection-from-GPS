"""Point-level motion feature extraction.

Computes speed, acceleration, jerk, bearing (spherical + WGS84),
bearing rate, bearing change, and inter-point distances from raw GPS points.

Refactored from: 1_Intial_Setup_Motion_Features/extract_motion_features_gps_data.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._geo import bearing_spherical, bearing_wgs84, haversine_m, shortest_turn_deg

if TYPE_CHECKING:
    from ..config import ModeDetectConfig

logger = logging.getLogger(__name__)


def extract_point_features(
    df: pd.DataFrame,
    config: ModeDetectConfig,
) -> pd.DataFrame:
    """Compute all point-level motion features for every trip in *df*.

    Expects columns named by ``config.col_*``.  Adds internal columns:
    ``speed_ms``, ``acc_ms2``, ``jerk_ms3``, ``bearing_deg``,
    ``bearing_deg_wgs84``, ``bearing_change_deg_wgs84``,
    ``bearing_rate_deg_wgs84``, ``dist_bp_wgs84``, ``dt``.

    Returns a copy — the input DataFrame is never mutated.
    """
    out = df.copy()

    # Ensure timestamp is datetime
    out[config.col_timestamp] = pd.to_datetime(out[config.col_timestamp])

    # Sort within each trip
    out.sort_values([config.col_trip_key, config.col_timestamp], kind="mergesort", inplace=True)

    # Speed: use provided column or compute from coordinates
    if config.col_speed_kmh and config.col_speed_kmh in out.columns:
        out["speed_ms"] = out[config.col_speed_kmh].astype("float64") / 3.6
    else:
        out["speed_ms"] = _compute_speed_from_coords(out, config)

    # Radians for bearing calculations
    out["_lat_rad"] = np.radians(out[config.col_latitude].astype("float64"))
    out["_lon_rad"] = np.radians(out[config.col_longitude].astype("float64"))

    # Per-trip, per-segment feature computation
    out = _compute_per_trip(out, config)

    # Clean up temporary columns
    out.drop(columns=["_lat_rad", "_lon_rad", "_segment"], errors="ignore", inplace=True)

    return out


def _compute_speed_from_coords(df: pd.DataFrame, config: ModeDetectConfig) -> pd.Series:
    """Fallback: compute speed from consecutive lat/lon + timestamps."""
    lat = df[config.col_latitude].astype("float64")
    lon = df[config.col_longitude].astype("float64")
    ts = df[config.col_timestamp]

    dist = haversine_m(lat.shift(1).values, lon.shift(1).values, lat.values, lon.values)
    dt = ts.diff().dt.total_seconds().values

    speed = np.divide(dist, dt, out=np.full(len(dist), np.nan), where=(dt > 0) & np.isfinite(dt))
    return pd.Series(speed, index=df.index)


def _compute_per_trip(df: pd.DataFrame, config: ModeDetectConfig) -> pd.DataFrame:
    """Compute dt, acceleration, jerk, bearing, bearing rate per trip+segment."""
    frames = []

    for _key, trip in df.groupby(config.col_trip_key, sort=False):
        trip = trip.copy()

        # Time deltas
        dt = trip[config.col_timestamp].diff().dt.total_seconds()
        dt_vals = dt.to_numpy(dtype="float64", copy=True)
        dt_vals[~np.isfinite(dt_vals)] = np.nan

        trip["dt"] = dt_vals

        # Gap detection → segment assignment
        median_dt = np.nanmedian(dt_vals) if np.isfinite(dt_vals).any() else 0
        max_gap = median_dt + config.gap_offset_s
        trip["_segment"] = (trip["dt"] > max_gap).cumsum()

        # Per-segment computations
        seg_frames = []
        for _seg, seg in trip.groupby("_segment", sort=False):
            seg = _compute_segment_features(seg, config)
            seg_frames.append(seg)

        frames.append(pd.concat(seg_frames, ignore_index=False))

    return pd.concat(frames, ignore_index=False)


def _compute_segment_features(seg: pd.DataFrame, config: ModeDetectConfig) -> pd.DataFrame:
    """Compute features within a contiguous GPS segment (no gaps)."""
    seg = seg.copy()
    dt = seg["dt"].values

    # ── Acceleration ──
    dv = seg["speed_ms"].diff().values
    seg["acc_ms2"] = np.divide(dv, dt, out=np.full(len(dv), np.nan), where=(dt > 0) & ~np.isnan(dt))

    # ── Jerk ──
    da = pd.Series(seg["acc_ms2"].values).diff().values
    seg["jerk_ms3"] = np.divide(da, dt, out=np.full(len(da), np.nan), where=(dt > 0) & ~np.isnan(dt))

    # ── Bearing (spherical) ──
    lat0 = seg["_lat_rad"].shift(1).values
    lon0 = seg["_lon_rad"].shift(1).values
    lat1 = seg["_lat_rad"].values
    lon1 = seg["_lon_rad"].values
    seg["bearing_deg"] = bearing_spherical(lat0, lon0, lat1, lon1)

    # ── Bearing (WGS84) + distance ──
    lat0_d = seg[config.col_latitude].shift(1).astype("float64").values
    lon0_d = seg[config.col_longitude].shift(1).astype("float64").values
    lat1_d = seg[config.col_latitude].astype("float64").values
    lon1_d = seg[config.col_longitude].astype("float64").values

    fwd_az, dist_wgs84 = bearing_wgs84(lon0_d, lat0_d, lon1_d, lat1_d)
    seg["bearing_deg_wgs84"] = fwd_az
    seg["dist_bp"] = haversine_m(lat0_d, lon0_d, lat1_d, lon1_d)
    seg["dist_bp_wgs84"] = dist_wgs84

    # ── Null out stacked coordinates and first row ──
    stacked = (
        np.isclose(lon0_d, lon1_d, equal_nan=False) & np.isclose(lat0_d, lat1_d, equal_nan=False)
    )
    first_mask = np.zeros(len(seg), dtype=bool)
    first_mask[0] = True
    null_mask = stacked | first_mask

    for col in ("bearing_deg", "bearing_deg_wgs84", "dist_bp", "dist_bp_wgs84"):
        seg.loc[seg.index[null_mask], col] = np.nan

    # ── Bearing change & rate ──
    for bearing_col, bc_col, br_col in [
        ("bearing_deg", "bearing_change_deg", "bearing_rate_deg"),
        ("bearing_deg_wgs84", "bearing_change_deg_wgs84", "bearing_rate_deg_wgs84"),
    ]:
        b = seg[bearing_col].values
        b_next = np.roll(b, -1)
        b_next[-1] = np.nan
        delta = shortest_turn_deg(b_next - b)
        seg[bc_col] = np.abs(delta)

        rate = np.divide(
            np.abs(delta), dt,
            out=np.full(len(dt), np.nan),
            where=np.isfinite(dt) & (dt > 0),
        )
        seg[br_col] = rate

    return seg
