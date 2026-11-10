"""ModeDetector — scikit-learn-inspired API for transport mode detection.

Usage::

    from modedetect import ModeDetector, ModeDetectConfig

    model = ModeDetector(ModeDetectConfig())
    model.fit(gps_df, trips_df)
    model.save("models/paris_v1.zip")

    model = ModeDetector.load("models/paris_v1.zip")
    predictions = model.predict(new_gps_df)
    probas = model.predict_proba(new_gps_df)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from . import artifacts
from .config import ModeDetectConfig
from .features import routes as feat_routes
from .features import segmentation as feat_seg
from .features.motion import extract_point_features
from .features.percentiles import compute_percentile_map, find_optimal_percentiles
from .features.statistics import compute_trip_statistics
from .preprocessing import (
    compute_speed_thresholds,
    detect_multimodal_columns,
    drop_brief_trips,
    drop_short_trips,
    drop_trips_with_gaps,
    filter_speed_outliers_fit,
    filter_speed_outliers_predict,
    remap_and_drop_modes,
    remove_invalid_coordinates,
    split_trips_at_gaps,
)
from .validation import (
    ModeDetectError,
    validate_fit_inputs,
    validate_gtfs,
    validate_osm,
    validate_predict_inputs,
    warn_high_nan_features,
    warn_temporal_gaps,
)

logger = logging.getLogger(__name__)


class ModeDetector:
    """Transport mode classifier from GPS traces.

    Follows the scikit-learn pattern: ``fit()`` → ``predict()`` / ``predict_proba()``,
    with ``save()`` / ``load()`` for persistence.
    """

    def __init__(self, config: ModeDetectConfig | None = None):
        self.config = config or ModeDetectConfig()

        # Fitted state (populated by fit() or load())
        self._model: XGBClassifier | None = None
        self._encoder: LabelEncoder | None = None
        self._feature_list: list[str] | None = None
        self._optimal_percentiles: list[str] = []
        self._optimal_stop_percentiles: list[str] = []
        self._speed_thresholds: dict[str, float] = {}
        self._max_speed_ms: float | None = None
        self._is_multimodal: bool = False

        # Lazy-loaded external resources
        self._stop_tree = None
        self._stop_df: pd.DataFrame | None = None
        self._route_balltrees: dict[str, feat_routes.RouteBallTree] = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._feature_list is not None

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, gps_df: pd.DataFrame, trips_df: pd.DataFrame) -> ModeDetector:
        """Train the mode detector from labelled GPS data.

        Parameters
        ----------
        gps_df : DataFrame
            GPS points with columns matching ``config.col_*``.
        trips_df : DataFrame
            Trip metadata with ``trip_key`` and ``main_mode``.
            Optionally ``mode_1``, ``mode_2``, ... for multi-modal.
        """
        cfg = self.config
        validate_fit_inputs(gps_df, trips_df, cfg)

        # Auto-detect multi-modal
        self._is_multimodal = detect_multimodal_columns(trips_df, cfg)
        if self._is_multimodal:
            logger.info("Multi-modal labels detected — segmentation will be enabled.")

        # ── Clean GPS data ────────────────────────────────────────────
        gps = remove_invalid_coordinates(gps_df, cfg)
        gaps = warn_temporal_gaps(gps, cfg)
        gps = drop_trips_with_gaps(gps, gaps, cfg)
        gps = drop_short_trips(gps, cfg.min_gps_points, cfg)
        gps = drop_brief_trips(gps, cfg.min_trip_duration_s, cfg)

        # ── Motion features ───────────────────────────────────────────
        logger.info("Extracting point-level motion features...")
        gps = extract_point_features(gps, cfg)

        # ── Speed outlier filtering (per-mode, needs labels) ──────────
        # Join labels temporarily for threshold computation
        mode_map = trips_df.set_index(cfg.col_trip_key)[cfg.col_main_mode].to_dict()
        gps["_mode_tmp"] = gps[cfg.col_trip_key].map(mode_map)

        # Remap modes on the temp column for threshold computation
        gps["_mode_tmp"] = gps["_mode_tmp"].replace(cfg.mode_remap)

        self._speed_thresholds = compute_speed_thresholds(
            gps.rename(columns={"_mode_tmp": cfg.col_main_mode}), cfg
        )
        self._max_speed_ms = max(self._speed_thresholds.values()) if self._speed_thresholds else None

        gps = filter_speed_outliers_fit(
            gps.rename(columns={"_mode_tmp": cfg.col_main_mode}),
            self._speed_thresholds, cfg,
        )
        gps.drop(columns=[cfg.col_main_mode], errors="ignore", inplace=True)

        # ── Trip-level statistics ─────────────────────────────────────
        logger.info("Computing trip-level statistics...")
        stats = compute_trip_statistics(gps, cfg, trips_df=trips_df)

        # ── Remap/drop modes in stats ─────────────────────────────────
        stats = remap_and_drop_modes(stats, cfg)

        # ── Percentile map ────────────────────────────────────────────
        logger.info("Computing percentile maps...")
        pct_map = compute_percentile_map(gps, cfg)

        # ── Optimal percentile selection (JS divergence) ──────────────
        logger.info("Selecting optimal percentiles via JS divergence...")
        self._optimal_percentiles = find_optimal_percentiles(pct_map, stats, cfg)

        # ── Stop proximity features ───────────────────────────────────
        stop_features = None
        if validate_gtfs(cfg):
            logger.info("Computing stop proximity features...")
            stop_features = self._compute_stop_features_fit(gps, stats, cfg)

        # ── Route matching features ───────────────────────────────────
        route_features = None
        if validate_osm(cfg):
            logger.info("Computing route matching features...")
            route_features = self._compute_route_features(gps, cfg)

        # ── Merge all feature sets ────────────────────────────────────
        logger.info("Merging feature sets...")
        merged = self._merge_features(stats, pct_map, stop_features, route_features)

        # ── Train XGBoost ─────────────────────────────────────────────
        logger.info("Training XGBoost classifier...")
        self._train_model(merged, cfg)

        logger.info(
            "Training complete: %d features, %d classes (%s)",
            len(self._feature_list),
            len(self._encoder.classes_),
            list(self._encoder.classes_),
        )
        return self

    # ── predict ───────────────────────────────────────────────────────

    def predict(
        self,
        gps_df: pd.DataFrame,
        segment: bool = False,
    ) -> pd.DataFrame:
        """Predict transport mode for unlabelled GPS data.

        Parameters
        ----------
        gps_df : DataFrame with GPS points.
        segment : if True, run PELT segmentation and return per-segment predictions.

        Returns
        -------
        DataFrame with columns ``trip_key``, ``predicted_mode``
        (and ``segment`` if segment=True).
        """
        self._check_fitted()
        cfg = self.config
        validate_predict_inputs(gps_df, cfg)

        gps = self._prepare_predict_gps(gps_df)

        if segment:
            return self._predict_segmented(gps, cfg)
        else:
            return self._predict_whole_trip(gps, cfg)

    def predict_proba(
        self,
        gps_df: pd.DataFrame,
        segment: bool = False,
    ) -> pd.DataFrame:
        """Predict class probabilities for unlabelled GPS data.

        Returns DataFrame with ``trip_key`` + one column per class.
        """
        self._check_fitted()
        cfg = self.config
        validate_predict_inputs(gps_df, cfg)

        gps = self._prepare_predict_gps(gps_df)

        if segment:
            features_df = self._extract_segmented_features(gps, cfg)
            id_cols = ["trip_key", "segment"]
        else:
            features_df = self._extract_trip_features(gps, cfg)
            id_cols = ["trip_key"]

        X = features_df.reindex(columns=self._feature_list, fill_value=0)
        proba = self._model.predict_proba(X)

        result = features_df[id_cols].copy()
        for i, cls in enumerate(self._encoder.classes_):
            result[cls] = proba[:, i]
        return result

    # ── save / load ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save the trained model as a .zip artifact bundle."""
        self._check_fitted()
        return artifacts.save_bundle(
            path=path,
            model=self._model,
            encoder=self._encoder,
            feature_list=self._feature_list,
            config=self.config,
            optimal_percentiles=self._optimal_percentiles,
            optimal_stop_percentiles=self._optimal_stop_percentiles,
            speed_thresholds=self._speed_thresholds,
            is_multimodal=self._is_multimodal,
        )

    @classmethod
    def load(cls, path: str | Path) -> ModeDetector:
        """Load a trained model from a .zip artifact bundle."""
        bundle = artifacts.load_bundle(path)

        detector = cls(config=bundle["config"])
        detector._model = bundle["model"]
        detector._encoder = bundle["encoder"]
        detector._feature_list = bundle["feature_list"]
        detector._optimal_percentiles = bundle["optimal_percentiles"]
        detector._optimal_stop_percentiles = bundle["optimal_stop_percentiles"]
        detector._speed_thresholds = bundle["speed_thresholds"]
        detector._is_multimodal = bundle["is_multimodal"]

        # Compute max speed from thresholds
        if detector._speed_thresholds:
            detector._max_speed_ms = max(detector._speed_thresholds.values())

        logger.info(
            "Loaded model: %d features, %d classes, multimodal=%s",
            len(detector._feature_list),
            len(detector._encoder.classes_),
            detector._is_multimodal,
        )
        return detector

    # ── OSM download helper ───────────────────────────────────────────

    def prepare_osm(
        self,
        gps_df: pd.DataFrame,
        output_dir: str | Path = "osm_routes",
        max_area_km2: float = 50_000,
    ) -> Path:
        """Download and extract OSM routes for the region covered by GPS data.

        Uses the 90th-percentile bounding box of GPS coordinates.
        Requires ``pyrosm`` (install with ``pip install modedetect[osm]``).
        """
        cfg = self.config
        lat = gps_df[cfg.col_latitude].astype("float64")
        lon = gps_df[cfg.col_longitude].astype("float64")

        # 90% bounding box (trim outliers)
        lat_lo, lat_hi = lat.quantile(0.05), lat.quantile(0.95)
        lon_lo, lon_hi = lon.quantile(0.05), lon.quantile(0.95)

        # Rough area estimate (degrees → km at mid-latitude)
        mid_lat = (lat_lo + lat_hi) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(mid_lat))
        area_km2 = abs(lat_hi - lat_lo) * km_per_deg_lat * abs(lon_hi - lon_lo) * km_per_deg_lon

        if area_km2 > max_area_km2:
            raise ModeDetectError(
                f"GPS bounding box covers ~{area_km2:.0f} km² which exceeds the "
                f"safety limit of {max_area_km2:.0f} km². Provide a smaller dataset "
                f"or set a higher max_area_km2."
            )

        logger.info(
            "GPS 90%% bounding box: [%.3f, %.3f] to [%.3f, %.3f] (~%.0f km²)",
            lat_lo, lon_lo, lat_hi, lon_hi, area_km2,
        )

        # Download from Overpass / Geofabrik would go here.
        # For now, require user to provide PBF and we just extract.
        if cfg.osm_pbf_path is None:
            raise ModeDetectError(
                "Set config.osm_pbf_path to an OSM PBF file. "
                "Download one from https://download.geofabrik.de/ for your region."
            )

        out = Path(output_dir)
        feat_routes.extract_osm_routes(cfg.osm_pbf_path, out)
        self.config = ModeDetectConfig(
            **{**{k: v for k, v in self.config.__dict__.items()}, "osm_routes_dir": out}
        )
        return out

    # ── Internal: prediction helpers ──────────────────────────────────

    def _prepare_predict_gps(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare GPS data for prediction."""
        cfg = self.config
        gps = remove_invalid_coordinates(gps_df, cfg)

        # Split at gaps (prediction mode)
        gaps = warn_temporal_gaps(gps, cfg)
        gps = split_trips_at_gaps(gps, gaps, cfg)
        gps = drop_short_trips(gps, cfg.min_gps_points, cfg)
        gps = drop_brief_trips(gps, cfg.min_trip_duration_s, cfg)

        # Motion features
        gps = extract_point_features(gps, cfg)

        # Speed outlier removal (prediction: remove individual points)
        max_speed = cfg.max_speed_ms or self._max_speed_ms
        if max_speed is not None:
            gps = filter_speed_outliers_predict(gps, max_speed, cfg)

        return gps

    def _predict_whole_trip(self, gps: pd.DataFrame, cfg: ModeDetectConfig) -> pd.DataFrame:
        """Predict one mode per trip."""
        features_df = self._extract_trip_features(gps, cfg)
        X = features_df.reindex(columns=self._feature_list, fill_value=0)

        y_pred = self._model.predict(X)
        labels = self._encoder.inverse_transform(y_pred)

        return pd.DataFrame({"trip_key": features_df["trip_key"].values, "predicted_mode": labels})

    def _predict_segmented(self, gps: pd.DataFrame, cfg: ModeDetectConfig) -> pd.DataFrame:
        """Segment trips with PELT, then predict per segment."""
        features_df = self._extract_segmented_features(gps, cfg)
        X = features_df.reindex(columns=self._feature_list, fill_value=0)

        y_pred = self._model.predict(X)
        labels = self._encoder.inverse_transform(y_pred)

        return pd.DataFrame({
            "trip_key": features_df["trip_key"].values,
            "segment": features_df["segment"].values,
            "predicted_mode": labels,
        })

    # ── Internal: feature extraction ──────────────────────────────────

    def _extract_trip_features(self, gps: pd.DataFrame, cfg: ModeDetectConfig) -> pd.DataFrame:
        """Extract and merge all trip-level features."""
        stats = compute_trip_statistics(gps, cfg)
        pct_map = compute_percentile_map(gps, cfg)

        stop_features = None
        if self._ensure_stop_resources(cfg):
            from .features.stops import compute_stop_features
            stop_features = compute_stop_features(gps, self._stop_tree, self._stop_df, cfg)

        route_features = None
        if self._ensure_route_resources(cfg):
            route_features = feat_routes.compute_route_features(gps, self._route_balltrees, cfg)

        return self._merge_features(stats, pct_map, stop_features, route_features)

    def _extract_segmented_features(self, gps: pd.DataFrame, cfg: ModeDetectConfig) -> pd.DataFrame:
        """Segment each trip, then extract per-segment features."""
        # Load route resources for signal-based segmentation
        self._ensure_route_resources(cfg)

        all_gps_segmented = []
        for trip_key, trip_df in gps.groupby(cfg.col_trip_key, sort=False):
            trip_df = trip_df.sort_values(cfg.col_timestamp)

            # Generate route signals if available
            route_signals = {}
            if self._route_balltrees:
                lat = trip_df[cfg.col_latitude].astype("float64").values
                lon = trip_df[cfg.col_longitude].astype("float64").values
                for mode_name, rbt in self._route_balltrees.items():
                    if mode_name in feat_routes.TRANSIT_MODES:
                        route_signals[mode_name] = feat_routes.generate_route_signal(
                            rbt, lat, lon, cfg.route_search_radius_m,
                        )

            # Choose segmentation strategy
            if route_signals:
                breakpoints = feat_seg.segment_trip_signals(trip_df, route_signals, cfg)
            else:
                breakpoints = feat_seg.segment_trip_speed(trip_df, cfg)

            segmented = feat_seg.assign_segments(trip_df, breakpoints)
            all_gps_segmented.append(segmented)

        gps_seg = pd.concat(all_gps_segmented, ignore_index=True)

        # Extract features at segment level
        stats = compute_trip_statistics(gps_seg, cfg, segment_col="segment")
        pct_map = compute_percentile_map(gps_seg, cfg, segment_col="segment")

        stop_features = None
        if self._ensure_stop_resources(cfg):
            from .features.stops import compute_stop_features
            stop_features = compute_stop_features(
                gps_seg, self._stop_tree, self._stop_df, cfg, segment_col="segment",
            )

        route_features = None
        if self._route_balltrees:
            route_features = feat_routes.compute_route_features(
                gps_seg, self._route_balltrees, cfg, segment_col="segment",
            )

        return self._merge_features(stats, pct_map, stop_features, route_features)

    # ── Internal: training ────────────────────────────────────────────

    def _train_model(self, merged: pd.DataFrame, cfg: ModeDetectConfig) -> None:
        """Train XGBoost via stratified K-fold, keep best model by macro F1."""
        mode_col = cfg.col_main_mode
        if mode_col not in merged.columns:
            raise ModeDetectError("Merged feature table is missing the mode label column.")

        # Drop rows with missing labels
        merged = merged.dropna(subset=[mode_col])

        # Encode labels
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(merged[mode_col].values)

        # Feature matrix
        exclude = {"trip_key", mode_col, "segment", "main_mode"}
        feature_cols = [c for c in merged.columns if c not in exclude]
        self._feature_list = feature_cols
        X = merged[feature_cols].values

        # XGBoost params
        n_classes = len(self._encoder.classes_)
        xgb_kwargs = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "tree_method": "hist",
            "random_state": cfg.random_state,
            "eval_metric": "mlogloss",
        }
        # n_jobs
        n_jobs = cfg.n_jobs if cfg.n_jobs != -1 else os.cpu_count()
        xgb_kwargs["n_jobs"] = n_jobs

        if cfg.xgb_params:
            xgb_kwargs.update(cfg.xgb_params)

        # Stratified K-fold
        from sklearn.metrics import f1_score

        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
        best_f1 = -1.0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = XGBClassifier(**xgb_kwargs)
            clf.fit(X[train_idx], y[train_idx])

            y_pred = clf.predict(X[val_idx])
            f1 = f1_score(y[val_idx], y_pred, average="macro")

            logger.info("Fold %d/%d — macro F1: %.4f", fold + 1, cfg.n_folds, f1)

            if f1 > best_f1:
                best_f1 = f1
                best_model = clf

        self._model = best_model
        logger.info("Best fold macro F1: %.4f", best_f1)

    # ── Internal: feature merging ─────────────────────────────────────

    def _merge_features(
        self,
        stats: pd.DataFrame,
        pct_map: pd.DataFrame,
        stop_features: pd.DataFrame | None,
        route_features: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Merge all feature DataFrames on trip_key (+ segment if present)."""
        has_segment = "segment" in stats.columns
        merge_on = ["trip_key", "segment"] if has_segment else ["trip_key"]

        # Select only optimal percentile columns (if known) plus trip_key/segment
        if self._optimal_percentiles:
            keep_cols = merge_on + [c for c in self._optimal_percentiles if c in pct_map.columns]
            pct_map = pct_map[keep_cols]

        merged = stats.merge(pct_map, on=merge_on, how="left")

        if stop_features is not None:
            merged = merged.merge(stop_features, on=merge_on, how="left")

        if route_features is not None:
            merged = merged.merge(route_features, on=merge_on, how="left")

        return merged

    # ── Internal: external resource management ────────────────────────

    def _ensure_stop_resources(self, cfg: ModeDetectConfig) -> bool:
        """Lazy-load GTFS stops and BallTree. Returns True if available."""
        if self._stop_tree is not None:
            return True
        if not validate_gtfs(cfg):
            return False

        from .features.stops import build_stop_balltree, load_gtfs_stops

        try:
            stops = load_gtfs_stops(cfg.gtfs_dir)
            self._stop_tree, self._stop_df = build_stop_balltree(stops)
            return True
        except Exception as exc:
            logger.warning("Failed to load GTFS stops: %s", exc)
            return False

    def _ensure_route_resources(self, cfg: ModeDetectConfig) -> bool:
        """Lazy-load OSM route BallTrees. Returns True if available."""
        if self._route_balltrees:
            return True
        if not validate_osm(cfg):
            return False

        routes_dir = cfg.osm_routes_dir
        if routes_dir is None and cfg.osm_pbf_path is not None:
            logger.warning("OSM PBF provided but not extracted. Run .prepare_osm() first.")
            return False

        for mode in feat_routes.TRANSIT_MODES:
            pq = routes_dir / f"{mode}.parquet"
            if pq.exists():
                self._route_balltrees[mode] = feat_routes.build_route_balltree(
                    pq, densify_spacing_m=cfg.route_densify_spacing_m,
                )

        return bool(self._route_balltrees)

    def _compute_stop_features_fit(
        self, gps: pd.DataFrame, stats: pd.DataFrame, cfg: ModeDetectConfig,
    ) -> pd.DataFrame | None:
        """Compute stop features during fit (with optimal percentile selection)."""
        if not self._ensure_stop_resources(cfg):
            return None

        from .features.stops import compute_stop_features

        stop_feats = compute_stop_features(gps, self._stop_tree, self._stop_df, cfg)

        # TODO: add JS divergence for stop percentile selection
        # For now, return all stop features
        return stop_feats

    def _compute_route_features(
        self, gps: pd.DataFrame, cfg: ModeDetectConfig,
    ) -> pd.DataFrame | None:
        """Compute route matching features."""
        if not self._ensure_route_resources(cfg):
            return None
        return feat_routes.compute_route_features(gps, self._route_balltrees, cfg)

    # ── Checks ────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ModeDetectError(
                "ModeDetector is not fitted. Call .fit() first or load a saved model with ModeDetector.load()."
            )
