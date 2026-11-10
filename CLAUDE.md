# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis project: **Multimodal Transport Mode Detection from GPS Traces**, built on the NetMob25 Paris GPS challenge dataset. The pipeline classifies transport modes (walk, bike, car, bus, metro, train, tram) from raw GPS traces and handles both single-modal and multi-modal (mode-switching) trips.

## Environment Setup

```bash
conda env create -f Mode-Detection-from-GPS/environment.yml
conda activate gps
```

Python 3.12. Key libraries: pandas, xgboost, scikit-learn, ruptures, geopandas, pyproj, pyarrow, pyrosm, shapely.

## Pipeline Architecture

The numbered folders form a strictly sequential pipeline — each stage produces Parquet outputs consumed by the next:

```
1_Intial_Setup_Motion_Features   Raw GPS → per-person parquets with speed/acc/jerk/bearing features → trip-level stats
        ↓
2_Optimal_Percentiles            Percentile distributions (P50–P95) of motion features per trip; JS divergence to find optimal percentile per feature
        ↓
3_Stop_Proximity                 GTFS loading → BallTree nearest-stop search (250m radius) → hive-partitioned Parquet dataset → trip-level stop stats
        ↓
4_Route_Matching                 OSM PBF → 7 modal route networks → BallTree match percentages + binary route signals per GPS point
        ↓
5_Segmentation                   PELT change-point detection (ruptures) on speed / multivariate / route signals → segment boundaries CSV
        ↓
6_Single_Modal_Classification    XGBoost classifier (multi:softprob); forward feature selection; notebook runs full train/eval pipeline
        ↓
7_Multi_Modal_Classification     Re-aggregates all features at segment level → applies single-modal classifier per segment → Jaccard scoring
```

**Key notebooks** (the main entry points for running classification):
- `6_Single_Modal_Classification/building_singlemodal_classifiers.ipynb` — full single-modal pipeline
- `7_Multi_Modal_Classification/building_multimodal_classification.ipynb` — full multi-modal pipeline

## Important Patterns

- **Data format**: Parquet throughout. All intermediate data stored as `.parquet` via PyArrow; some stages also emit `.csv` for inspection.
- **Concurrency**: `concurrent.futures.ProcessPoolExecutor` with `tqdm` progress bars is the standard parallelism pattern across all batch scripts.
- **Spatial indexing**: `sklearn.neighbors.BallTree` with haversine metric used for both stop proximity (folder 3) and route matching (folder 4).
- **All data paths are hardcoded** at the top of each script as `Path()` objects pointing to `/home/s232534/data/...` or `/home/s232534/scripts/...`. These must be updated when running on a different machine.
- **Folder 7 scripts accept CLI args** (`--segments`, `--out-stem`) via argparse to allow different segmentation boundary files.
- **Mode consolidation**: Multiple car types → `CAR`, `TRAIN_EXPRESS` → `TRAIN`. Dropped modes: `OTHER`, `PLANE`, `ON_DEMAND`, `ELECT_BIKE`, `ELECT_SCOOTER`, `TWO_WHEELER`. This logic lives in `percentile_calculate.py` and `exploration/functions.py`.

## Shared Utilities

`exploration/functions.py` — imported as `from exploration import functions as fn`. Contains:
- `haversine_m()` — vectorized Haversine distance
- `remap_and_drop_modes_df()` — mode consolidation
- `filter_motion_data_speed_avg()` / `filter_motion_data_speed_percentile()` — outlier filtering
- `plot_hist_mode_2x2()` — 2x2 histogram overlay by mode
- Various trip loading and visualization helpers

## Pre-generated Feature Sets

`DATA_Feature_sets/` contains ready-to-use parquet files (outputs of folders 1–4) that can be loaded directly in the classification notebooks without re-running upstream pipeline stages.

## `modedetect` Package (scikit-learn-style API)

A refactored, installable Python package wrapping the entire pipeline behind a single class:

```python
from modedetect import ModeDetector, ModeDetectConfig

model = ModeDetector(ModeDetectConfig(gtfs_dir=Path("gtfs/"), osm_routes_dir=Path("osm/")))
model.fit(gps_df, trips_df)           # train on labelled data
model.save("model.zip")               # save artifact bundle

model = ModeDetector.load("model.zip")
model.predict(new_gps_df)             # whole-trip predictions
model.predict(new_gps_df, segment=True)  # multi-modal (per-segment)
model.predict_proba(new_gps_df)       # class probabilities
```

Package structure:
```
modedetect/
├── config.py          # ModeDetectConfig — all parameters in one place
├── detector.py        # ModeDetector class (fit/predict/predict_proba/save/load)
├── validation.py      # Input validators with actionable error messages
├── preprocessing.py   # Coord cleaning, mode remap, gap handling, speed outliers
├── artifacts.py       # Save/load model bundle as .zip
├── _geo.py            # Haversine, bearing, geodesic helpers
├── _types.py          # Column name constants
└── features/
    ├── motion.py      # Point-level: speed, acc, jerk, bearing
    ├── statistics.py  # Trip/segment-level aggregation
    ├── percentiles.py # Percentile maps + JS divergence selection
    ├── stops.py       # GTFS BallTree stop proximity
    ├── routes.py      # OSM BallTree route matching
    └── segmentation.py # PELT change-point detection
```

Key design decisions:
- **GTFS and OSM are optional** — if not configured, those features are skipped.
- **fit() needs labels; predict() does not** — label-dependent steps (JS divergence, per-mode speed thresholds) run only during training and results are stored in the artifact.
- **Multi-modal auto-detected** from trips_df columns during fit; enabled via `segment=True` in predict.
- **Speed outlier handling**: fit removes entire trips; predict removes individual points. Thresholds are 99th percentile per mode (learned from data).
- **Gap handling**: fit drops trips with gaps > threshold; predict splits them into sub-trips.
