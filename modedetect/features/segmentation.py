"""PELT change-point detection for trip segmentation.

Wraps the ``ruptures`` library to detect mode-switch boundaries within
multi-modal trips. Supports speed-only, multivariate, and route-signal
based segmentation.

Refactored from: 5_Segmentation/signal_segmentation.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from ..config import ModeDetectConfig

logger = logging.getLogger(__name__)


def segment_trip_speed(
    df: pd.DataFrame,
    config: ModeDetectConfig,
    speed_col: str = "speed_ms",
) -> list[int]:
    """Run PELT on the speed signal of a single trip.

    Returns a list of breakpoint indices (last index = len(df)).
    """
    signal = df[speed_col].fillna(0).values
    return _run_pelt(signal, len(df), config)


def segment_trip_multivariate(
    df: pd.DataFrame,
    config: ModeDetectConfig,
    feature_cols: list[str] | None = None,
    weights: np.ndarray | None = None,
) -> list[int]:
    """Run PELT on a multivariate signal (speed + bearing rate by default).

    Parameters
    ----------
    feature_cols : columns to use (default: speed_ms, acc_ms2, bearing_rate_deg_wgs84)
    weights : per-feature weights after standardisation
    """
    if feature_cols is None:
        feature_cols = ["speed_ms", "acc_ms2", "bearing_rate_deg_wgs84"]
    if weights is None:
        weights = np.array([1.0, 0.75, 0.5])

    X = df[feature_cols].fillna(0).to_numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    signal = X_std * weights

    return _run_pelt(signal, len(df), config)


def segment_trip_signals(
    df: pd.DataFrame,
    route_signals: dict[str, np.ndarray],
    config: ModeDetectConfig,
) -> list[int]:
    """Run PELT using speed + bearing rate + per-mode binary route signals.

    Parameters
    ----------
    route_signals : dict mapping mode name → binary signal array (0/1),
        e.g. {"bus": arr, "train": arr, "subway": arr, "tram": arr}.
        Each array must have length == len(df).

    Unlike the original code, this does NOT use ground-truth main_mode
    for adaptive weighting — all signals are weighted equally.
    This makes it usable during prediction (no labels required).
    """
    cont_cols = ["speed_ms", "bearing_rate_deg_wgs84"]
    X_cont = df[cont_cols].fillna(0).to_numpy()

    scaler = StandardScaler()
    X_cont_std = scaler.fit_transform(X_cont)

    signal_arrays = []
    for mode in ("bus", "train", "subway", "tram"):
        arr = route_signals.get(mode)
        if arr is None:
            arr = np.zeros(len(df), dtype=np.int8)
        signal_arrays.append(arr.reshape(-1, 1))

    X_bin = np.hstack(signal_arrays)
    signal = np.hstack([X_cont_std, X_bin])

    # Equal weights (mode-agnostic — no label leakage)
    weights = np.array([1.0, 1.0, 0.75, 0.75, 0.75, 0.75])
    signal = signal * weights

    return _run_pelt(signal, len(df), config)


def assign_segments(
    df: pd.DataFrame,
    breakpoints: list[int],
    segment_col: str = "segment",
) -> pd.DataFrame:
    """Add a segment label column to a trip DataFrame from PELT breakpoints."""
    out = df.copy()
    arr = np.zeros(len(out), dtype=int)
    prev = 0
    for i, bp in enumerate(breakpoints):
        arr[prev:bp] = i
        prev = bp
    out[segment_col] = arr
    return out


# ── Internal ─────────────────────────────────────────────────────────


def _run_pelt(
    signal: np.ndarray,
    n_points: int,
    config: ModeDetectConfig,
) -> list[int]:
    """Run PELT with adaptive sizing and penalty fallback."""
    import ruptures as rpt

    min_size = min(config.pelt_min_size, max(2, n_points // 2))
    jump = min(config.pelt_jump, max(1, n_points // 5))

    algo = rpt.Pelt(model=config.pelt_model, min_size=min_size, jump=jump).fit(signal)

    # Try primary penalty, then fallbacks
    penalties = [config.pelt_penalty] + config.pelt_fallback_penalties
    for pen in penalties:
        breakpoints = algo.predict(pen=pen)
        if len(breakpoints) > 1:  # at least one real breakpoint + end marker
            return breakpoints

    # No breakpoints found even with lowest penalty — return just the end
    return [n_points]
