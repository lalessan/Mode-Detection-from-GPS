"""Post-processing heuristics for segment-level predictions.

Applied after XGBoost classification to fix common confusions,
particularly BUS segments misclassified as CAR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import ModeDetectConfig

logger = logging.getLogger(__name__)


def postprocess_segments(
    predictions: pd.DataFrame,
    features_df: pd.DataFrame,
    config: ModeDetectConfig,
    proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Apply post-processing heuristics to segment predictions.

    Parameters
    ----------
    predictions : DataFrame with trip_key, segment, predicted_mode
    features_df : DataFrame with trip_key, segment, plus feature columns
    config : ModeDetectConfig
    proba : (n_segments, n_classes) probability matrix from XGBoost (optional)
    class_names : list of class names matching proba columns (optional)

    Returns
    -------
    DataFrame with same columns as predictions, possibly with corrected modes
        and renumbered segments after merging.
    """
    if not config.postprocess_enabled:
        return predictions

    result = predictions.copy()
    n_before = len(result)

    # Merge features needed for heuristics
    feat_cols = []
    for col in ["bus_match_pct", "BUS_SEQ_STOPS", "BUS_SEQ_ROUTES",
                 "DURATION_BUS", "MIN_DIST_BUS", "SPEED_AVG", "GPS_POINTS",
                 "TOTAL_TIME",
                 "METRO_DIST_EP_MAX_M", "TRAIN_DIST_EP_MAX_M", "TRAM_DIST_EP_MAX_M"]:
        if col in features_df.columns:
            feat_cols.append(col)

    if not feat_cols:
        logger.info("Post-processing: no relevant features available, skipping.")
        return result

    merge_keys = ["trip_key", "segment"]
    df = result.merge(
        features_df[merge_keys + feat_cols],
        on=merge_keys, how="left",
    )

    n_reclassified = 0

    # ── Rule 0: Station endpoint constraint ────────────────────────
    # A SUBWAY/TRAIN/TRAM segment must start AND end near a station.
    # If not, reassign to the next-best class from predict_proba.
    EP_THRESHOLDS = {
        "SUBWAY":  ("METRO_DIST_EP_MAX_M", 500.0),
        "TRAIN":   ("TRAIN_DIST_EP_MAX_M", 1000.0),
        "TRAMWAY": ("TRAM_DIST_EP_MAX_M",  300.0),
    }
    n_rule0 = 0
    for mode, (ep_col, max_dist) in EP_THRESHOLDS.items():
        if ep_col not in df.columns:
            continue
        mode_mask = df["predicted_mode"] == mode
        too_far = df[ep_col] > max_dist
        override_mask = mode_mask & too_far & df[ep_col].notna()
        n_override = override_mask.sum()
        if n_override > 0 and proba is not None and class_names is not None:
            # Reassign to next-best class (exclude the current mode)
            mode_idx = class_names.index(mode)
            for iloc_pos in np.where(override_mask.values)[0]:
                p = proba[iloc_pos].copy()
                p[mode_idx] = 0  # exclude current mode
                next_best = class_names[np.argmax(p)]
                df.iloc[iloc_pos, df.columns.get_loc("predicted_mode")] = next_best
            n_rule0 += n_override
        elif n_override > 0:
            # No proba available — fall back to WALKING
            df.loc[override_mask, "predicted_mode"] = "WALKING"
            n_rule0 += n_override

    n_reclassified += n_rule0
    if n_rule0 > 0:
        logger.info("Post-processing rule 0 (endpoint constraint): %d segments overridden", n_rule0)

    # ── Rule 1: CAR → BUS based on bus route evidence ────────────────
    car_mask = df["predicted_mode"] == "CAR"
    speed_ok = True  # default if no speed column
    if "SPEED_AVG" in df.columns:
        speed_ok = df["SPEED_AVG"] <= 25.0  # buses don't go > 90 km/h

    bus_evidence = np.zeros(len(df), dtype=bool)

    # Strong evidence: many stops from same bus route
    if "BUS_SEQ_STOPS" in df.columns:
        bus_evidence |= (df["BUS_SEQ_STOPS"] >= 4)

    # Combined evidence: OSM route match + bus stop sequence
    if "bus_match_pct" in df.columns and "BUS_SEQ_STOPS" in df.columns:
        bus_evidence |= (df["bus_match_pct"] >= 0.3) & (df["BUS_SEQ_STOPS"] >= 3)

    # Combined evidence: OSM route match + time near bus stops
    if "bus_match_pct" in df.columns and "DURATION_BUS" in df.columns:
        bus_evidence |= (df["bus_match_pct"] >= 0.4) & (df["DURATION_BUS"] >= 60)

    reclassify_mask = car_mask & bus_evidence & speed_ok
    n_rule1 = reclassify_mask.sum()
    df.loc[reclassify_mask, "predicted_mode"] = "BUS"
    n_reclassified += n_rule1

    # ── Rule 2: Sandwich repair (BUS-CAR-BUS → BUS-BUS-BUS) ─────────
    n_rule2 = 0
    for trip_key, trip_df in df.groupby("trip_key", sort=False):
        if len(trip_df) < 3:
            continue
        trip_idx = trip_df.index.tolist()
        modes = trip_df["predicted_mode"].values

        for j in range(1, len(modes) - 1):
            if modes[j] != "CAR":
                continue
            if modes[j - 1] != "BUS" or modes[j + 1] != "BUS":
                continue
            # Check segment is short
            idx = trip_idx[j]
            gps_pts = df.loc[idx, "GPS_POINTS"] if "GPS_POINTS" in df.columns else 999
            duration = df.loc[idx, "TOTAL_TIME"] if "TOTAL_TIME" in df.columns else 999
            if gps_pts < 50 or duration < 120:
                # Check some bus evidence
                has_bus_signal = False
                if "bus_match_pct" in df.columns and df.loc[idx, "bus_match_pct"] > 0:
                    has_bus_signal = True
                if "BUS_SEQ_STOPS" in df.columns and df.loc[idx, "BUS_SEQ_STOPS"] >= 1:
                    has_bus_signal = True
                if has_bus_signal:
                    df.loc[idx, "predicted_mode"] = "BUS"
                    modes[j] = "BUS"
                    n_rule2 += 1

    n_reclassified += n_rule2

    # ── Rule 3: Merge adjacent same-mode segments ────────────────────
    result = df[["trip_key", "segment", "predicted_mode"]].copy()
    new_segments = []
    for trip_key, trip_df in result.groupby("trip_key", sort=False):
        modes = trip_df["predicted_mode"].values
        seg_id = 0
        segs = [0]
        for i in range(1, len(modes)):
            if modes[i] != modes[i - 1]:
                seg_id += 1
            segs.append(seg_id)
        new_segments.extend(segs)
    result["segment"] = new_segments

    if n_reclassified > 0:
        logger.info("Post-processing: %d segments reclassified (rule1=%d, rule2=%d)",
                     n_reclassified, n_rule1, n_rule2)

    return result
