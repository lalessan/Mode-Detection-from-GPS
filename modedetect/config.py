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
    max_range_threshold_m: float = 150.0  # non-walking trips whose GPS never goes further than this from centroid are dropped
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
    stop_search_radius_m: int = 50       # default / fallback for overall stats
    # Per-mode radii (empirically optimized via test_stop_radii.py):
    bus_stop_radius_m: int = 250         # bus(3) — MIN_DIST best at 250m
    metro_stop_radius_m: int = 750       # metro(1) — underground GPS noise
    rail_stop_radius_m: int = 50         # rail(2) — DURATION dominates at 50m
    tram_stop_radius_m: int = 50         # tram(0) — DURATION dominates at 50m
    bus_seq_radius_m: int = 30           # tighter radius for bus route sequence feature
    stop_quantiles: tuple[int, ...] = field(default_factory=_default_stop_quantiles)

    # ── Route matching (OSM) ──────────────────────────────────────────
    osm_pbf_path: Path | None = None
    osm_routes_dir: Path | None = None  # pre-extracted parquet files
    route_search_radius_m: int = 20
    route_densify_spacing_m: int = 10

    # ── Segmentation (PELT) ───────────────────────────────────────────
    pelt_model: str = "l2"
    pelt_min_size: int = 30
    pelt_jump: int = 20
    pelt_penalty: int = 20
    pelt_fallback_penalties: list[int] = field(default_factory=lambda: [15, 10, 5])
    speed_only_segmentation: bool = False  # True = skip route signals in PELT (much faster)

    # ── XGBoost classifier ────────────────────────────────────────────
    xgb_params: dict | None = None  # extra kwargs passed to XGBClassifier
    feature_selection: bool = False  # forward feature selection before training
    class_weight: str | None = "balanced"  # "balanced" = inverse freq, None = uniform
    hyperparam_tuning: bool = False  # random search over XGBoost params before K-fold
    tuning_n_iter: int = 30          # number of random combos to try
    test_size: float = 0.2
    random_state: int = 0
    n_folds: int = 5

    # ── Post-processing ────────────────────────────────────────────────
    postprocess_enabled: bool = False  # apply heuristic fixes after segment classification

    # ── Hierarchical model ───────────────────────────────────────────
    hierarchical: bool = False  # True = 3-stage model (non-motorized/motorized → subclass)

    # ── Parallelism ───────────────────────────────────────────────────
    n_jobs: int = 1  # 1 = single-process; -1 = all cores


# ── Hierarchical mode groupings ──────────────────────────────────────
NONMOTORIZED_MODES = {"WALKING", "BIKE"}
MOTORIZED_MODES = {"BUS", "CAR", "SUBWAY", "TRAIN", "TRAMWAY"}


def is_proximity_feature(col: str) -> bool:
    """Return True if the column is a proximity/infrastructure feature."""
    PROXIMITY_EXACT = {
        "bus_match_pct", "tram_match_pct", "train_match_pct", "subway_match_pct",
    }
    PROXIMITY_PREFIXES = (
        "STOPS_NUM_", "STOPS_MINDIST_", "STOPS_CLOSEST_",
        "MIN_DIST_", "DURATION_TRAM", "DURATION_METRO", "DURATION_RAIL", "DURATION_BUS",
        "BUS_SEQ_",
        "METRO_DIST_", "TRAIN_DIST_", "TRAM_DIST_",
    )
    return col in PROXIMITY_EXACT or any(col.startswith(p) for p in PROXIMITY_PREFIXES)
