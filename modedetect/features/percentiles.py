"""Percentile map computation and Jensen-Shannon optimal percentile selection.

- ``compute_percentile_map``: for each trip, compute P50–P95 of motion features.
- ``find_optimal_percentiles``: (fit-only) use JS divergence across mode pairs
  to select the single best percentile per feature.

Refactored from:
  2_Optimal_Percentiles/percentile_map.py
  2_Optimal_Percentiles/percentile_calculate.py
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config import ModeDetectConfig

logger = logging.getLogger(__name__)

# Source column → output prefix (matching the original pipeline)
_FEATURES = {
    "speed_ms": "SPEED",
    "acc_ms2": "ACC",
    "jerk_ms3": "JERK",
    "bearing_change_deg_wgs84": "BC_WGS84",
    "bearing_rate_deg_wgs84": "BR_WGS84",
}


# ── Public API ───────────────────────────────────────────────────────


def compute_percentile_map(
    gps_df: pd.DataFrame,
    config: ModeDetectConfig,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Compute percentile distributions of motion features per trip (or per segment).

    Returns a DataFrame with one row per trip (or trip+segment), with columns
    like ``SPEED_P50``, ``SPEED_P55``, ..., ``BR_WGS84_P95``.
    """
    key_col = config.col_trip_key
    group_cols = [key_col] if segment_col is None else [key_col, segment_col]
    quantile_fracs = [p / 100.0 for p in config.percentile_levels]

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

        row = {"trip_key": trip_key}
        if seg_id is not None:
            row["segment"] = seg_id

        for src_col, prefix in _FEATURES.items():
            if src_col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[src_col], errors="coerce")
            if vals.dropna().empty:
                for p in config.percentile_levels:
                    row[f"{prefix}_P{p}"] = np.nan
            else:
                qs = vals.quantile(quantile_fracs).to_dict()
                for p in config.percentile_levels:
                    row[f"{prefix}_P{p}"] = float(qs.get(p / 100.0, np.nan))

        rows.append(row)

    return pd.DataFrame(rows)


def find_optimal_percentiles(
    percentile_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    config: ModeDetectConfig,
    mode_col: str | None = None,
    aggregation: str = "mean",
) -> list[str]:
    """Select the single best percentile column per feature using JS divergence.

    This is a **fit-only** operation: it requires labelled data to compute
    mode-pair divergences.

    Returns a list of column names like ``['SPEED_P95', 'ACC_P95', 'BC_WGS84_P50', ...]``.
    """
    if mode_col is None:
        mode_col = config.col_main_mode

    # Merge labels onto percentile data
    df = percentile_df.merge(
        stats_df[["trip_key", mode_col]].drop_duplicates(),
        on="trip_key",
        how="inner",
    )

    # Drop rows with NaN in key bearing percentiles
    drop_subset = [c for c in df.columns if c.startswith(("BC_WGS84_P", "BR_WGS84_P")) and c.endswith("50")]
    if drop_subset:
        df = df.dropna(subset=drop_subset)

    modes = df[mode_col].dropna().unique()
    if len(modes) < 2:
        logger.warning("Fewer than 2 modes in data — cannot compute JS divergence. Returning empty.")
        return []

    # Identify all percentile columns
    pct_cols = [c for c in df.columns if any(c.startswith(f"{pfx}_P") for pfx in _FEATURES.values())]

    js_records = []
    for col in pct_cols:
        edges = _make_pooled_edges(df[col].dropna().values)
        if edges is None:
            continue

        histograms = {}
        for m in modes:
            vals = df.loc[df[mode_col] == m, col].dropna().values
            if len(vals) == 0:
                continue
            histograms[m] = _histogram_normalised(vals, edges)

        for a, b in combinations(histograms.keys(), 2):
            js = _jensen_shannon(histograms[a], histograms[b])
            js_records.append({"col": col, "mode_a": a, "mode_b": b, "js": js})

    if not js_records:
        logger.warning("No JS divergence values computed. Returning empty.")
        return []

    df_js = pd.DataFrame(js_records)
    df_agg = df_js.groupby("col")["js"].agg(["mean", "median"]).reset_index()
    df_agg["feature"] = df_agg["col"].str.split("_P").str[0]

    best = df_agg.loc[df_agg.groupby("feature")[aggregation].idxmax(), "col"].tolist()

    logger.info("Optimal percentiles selected: %s", best)
    return best


# ── Internal helpers ─────────────────────────────────────────────────


def _kullback_leibler(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kullback_leibler(p, m) + 0.5 * _kullback_leibler(q, m)


def _make_pooled_edges(
    values: np.ndarray,
    bins: int = 50,
    trim: tuple[float, float] = (0.01, 0.99),
) -> np.ndarray | None:
    if len(values) == 0:
        return None
    lo, hi = np.quantile(values, trim)
    if hi <= lo:
        return None
    return np.linspace(lo, hi, bins + 1)


def _histogram_normalised(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges, density=False)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return counts / total
