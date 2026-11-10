"""Model artifact bundle: save and load as a single .zip file.

Bundle contents:
  model.ubj              – XGBoost model (Universal Binary JSON)
  label_encoder.joblib   – sklearn LabelEncoder
  feature_list.json      – ordered list of feature column names
  config.json            – full ModeDetectConfig as JSON
  metadata.json          – training timestamp, package version, dataset stats
  optimal_percentiles.json – learned optimal percentile columns
  speed_thresholds.json  – per-mode 99th-pct speed thresholds from training
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .config import ModeDetectConfig


def save_bundle(
    path: str | Path,
    model: XGBClassifier,
    encoder: LabelEncoder,
    feature_list: list[str],
    config: ModeDetectConfig,
    optimal_percentiles: list[str],
    optimal_stop_percentiles: list[str],
    speed_thresholds: dict[str, float],
    is_multimodal: bool,
    dataset_stats: dict | None = None,
) -> Path:
    """Save all model artifacts into a single .zip file.

    Returns the resolved path to the created zip file.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".zip")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # XGBoost model
        model_path = tmp / "model.ubj"
        model.save_model(str(model_path))

        # Label encoder
        encoder_path = tmp / "label_encoder.joblib"
        joblib.dump(encoder, encoder_path)

        # Feature list
        _write_json(tmp / "feature_list.json", feature_list)

        # Config (convert Path objects to strings for JSON)
        config_dict = _config_to_serializable(config)
        _write_json(tmp / "config.json", config_dict)

        # Learned artifacts
        _write_json(tmp / "optimal_percentiles.json", {
            "motion": optimal_percentiles,
            "stops": optimal_stop_percentiles,
        })
        _write_json(tmp / "speed_thresholds.json", speed_thresholds)

        # Metadata
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_multimodal": is_multimodal,
            "n_features": len(feature_list),
            "n_classes": len(encoder.classes_),
            "classes": encoder.classes_.tolist(),
        }
        if dataset_stats:
            metadata["dataset_stats"] = dataset_stats
        _write_json(tmp / "metadata.json", metadata)

        # Pack into zip
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in tmp.iterdir():
                zf.write(fpath, fpath.name)

    return path


def load_bundle(path: str | Path) -> dict:
    """Load a model artifact bundle from a .zip file.

    Returns a dict with keys:
      model, encoder, feature_list, config, optimal_percentiles,
      optimal_stop_percentiles, speed_thresholds, is_multimodal, metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp)

        # XGBoost model
        model = XGBClassifier()
        model.load_model(str(tmp / "model.ubj"))

        # Label encoder
        encoder = joblib.load(tmp / "label_encoder.joblib")

        # Feature list
        feature_list = _read_json(tmp / "feature_list.json")

        # Config
        config_dict = _read_json(tmp / "config.json")
        config = _config_from_serializable(config_dict)

        # Learned artifacts
        pct_data = _read_json(tmp / "optimal_percentiles.json")
        optimal_percentiles = pct_data.get("motion", [])
        optimal_stop_percentiles = pct_data.get("stops", [])

        speed_thresholds = _read_json(tmp / "speed_thresholds.json")

        # Metadata
        metadata = _read_json(tmp / "metadata.json")

    return {
        "model": model,
        "encoder": encoder,
        "feature_list": feature_list,
        "config": config,
        "optimal_percentiles": optimal_percentiles,
        "optimal_stop_percentiles": optimal_stop_percentiles,
        "speed_thresholds": speed_thresholds,
        "is_multimodal": metadata.get("is_multimodal", False),
        "metadata": metadata,
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _write_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _config_to_serializable(config: ModeDetectConfig) -> dict:
    """Convert config dataclass to a JSON-serializable dict."""
    d = asdict(config)
    # Convert Path objects to strings
    for key, val in d.items():
        if isinstance(val, Path):
            d[key] = str(val)
    # Convert tuples to lists (JSON has no tuple type)
    for key, val in d.items():
        if isinstance(val, tuple):
            d[key] = list(val)
    return d


def _config_from_serializable(d: dict) -> ModeDetectConfig:
    """Reconstruct a ModeDetectConfig from a deserialized dict."""
    # Convert path strings back to Path objects
    path_fields = {"gtfs_dir", "osm_pbf_path", "osm_routes_dir"}
    for key in path_fields:
        if key in d and d[key] is not None:
            d[key] = Path(d[key])

    # Convert lists back to tuples where the dataclass expects tuples
    tuple_fields = {"percentile_levels", "stop_quantiles"}
    for key in tuple_fields:
        if key in d and isinstance(d[key], list):
            d[key] = tuple(d[key])

    # Filter out unknown keys (forward compatibility)
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ModeDetectConfig)}
    d = {k: v for k, v in d.items() if k in valid_fields}

    return ModeDetectConfig(**d)
