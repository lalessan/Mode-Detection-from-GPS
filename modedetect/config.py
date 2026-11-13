"""Central configuration for the modedetect pipeline.

All parameters that are currently hardcoded across the original scripts
are collected here in a single dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_mode_remap() -> dict[str, str]:
    """NetMob25 defaults: consolidate car variants and express trains."""
    return {
        "PRIV_CAR_DRIVER": "CAR",
        "PRIV_CAR_PASSENGER": "CAR",
        "LIGHT_COMM_VEHICLE": "CAR",
        "TAXI": "CAR",
        "TRAIN_EXPRESS": "TRAIN",
    }


def _default_mode_drop() -> list[str]:
    """NetMob25 defaults: modes too rare or ambiguous to classify."""
    return ["OTHER", "PLANE", "ON_DEMAND", "ELECT_BIKE", "ELECT_SCOOTER", "TWO_WHEELER"]


def _default_percentile_levels() -> tuple[int, ...]:
    return (50, 55, 60, 65, 70, 75, 80, 85, 90, 95)


def _default_stop_quantiles() -> tuple[int, ...]:
    return tuple(range(5, 100, 5))  # 5, 10, 15, ..., 95


@dataclass
class ModeDetectConfig:
    """All tuneable parameters for the modedetect pipeline.

    Users only *need* to set paths; everything else has sensible defaults.
    """

    # ── Column mapping (adapt to different datasets) ──────────────────
    col_latitude: str = "latitude"
    col_longitude: str = "longitude"
    col_timestamp: str = "timestamp"
    col_trip_key: str = "trip_key"
    col_speed_kmh: str | None = "speed_kmh"   # None → compute from coords
    col_main_mode: str = "main_mode"           # only needed for fit()

    # ── Mode consolidation ────────────────────────────────────────────
    mode_remap: dict[str, str] = field(default_factory=_default_mode_remap)
    modes_to_drop: list[str] = field(default_factory=_default_mode_drop)

    # ── GPS quality / validation ──────────────────────────────────────
    min_gps_points: int = 10
    min_trip_duration_s: int = 180
    gap_threshold_s: float = 300.0  # gaps > this → split (predict) or drop (fit)

    # ── Speed outlier filtering ───────────────────────────────────────
    #   During fit(): per-mode 99th percentile thresholds are computed
    #   from the data and stored in the artifact.
    #   During predict(): max across all training modes is used.
    #   max_speed_ms overrides the learned threshold if set.
    max_speed_ms: float | None = None

    # ── Motion feature extraction ─────────────────────────────────────
    gap_offset_s: float = 13.0  # added to median dt to define intra-trip gaps

    # ── Percentile maps ───────────────────────────────────────────────
    percentile_levels: tuple[int, ...] = field(default_factory=_default_percentile_levels)

    # ── Stop proximity (GTFS) ─────────────────────────────────────────
    gtfs_dir: Path | None = None
    stop_search_radius_m: int = 250
    stop_quantiles: tuple[int, ...] = field(default_factory=_default_stop_quantiles)

    # ── Route matching (OSM) ──────────────────────────────────────────
    osm_pbf_path: Path | None = None
    osm_routes_dir: Path | None = None  # pre-extracted parquet files
    route_search_radius_m: int = 20
    route_densify_spacing_m: int = 10

    # ── Segmentation (PELT) ───────────────────────────────────────────
    pelt_model: str = "rbf"
    pelt_min_size: int = 30
    pelt_jump: int = 10
    pelt_penalty: int = 20
    pelt_fallback_penalties: list[int] = field(default_factory=lambda: [15, 10, 5])

    # ── XGBoost classifier ────────────────────────────────────────────
    xgb_params: dict | None = None  # extra kwargs passed to XGBClassifier
    feature_selection: bool = True   # forward feature selection before training
    class_weight: str | None = "balanced"  # "balanced" = inverse freq, None = uniform
    hyperparam_tuning: bool = False  # random search over XGBoost params before K-fold
    tuning_n_iter: int = 30          # number of random combos to try
    test_size: float = 0.2
    random_state: int = 0
    n_folds: int = 5

    # ── Parallelism ───────────────────────────────────────────────────
    n_jobs: int = 1  # 1 = single-process; -1 = all cores
