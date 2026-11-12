# modedetect

A Python package for **transport mode detection from GPS traces**. Given raw GPS data and trip metadata, it classifies trips into transport modes (walk, bike, car, bus, metro, train, tram) using motion features, stop proximity, route matching, and an XGBoost classifier.

Built as a scikit-learn-style API: `fit()` to train, `predict()` to classify, `save()`/`load()` to persist.

## Installation

From the repository root:

```bash
pip install -e .
```

For OSM route extraction (optional):

```bash
pip install -e ".[osm]"
```

## Quick start

### Training a model

```python
from pathlib import Path
import pandas as pd
from modedetect import ModeDetector, ModeDetectConfig

# 1. Configure — map your column names and set optional data paths
config = ModeDetectConfig(
    col_latitude="LATITUDE",
    col_longitude="LONGITUDE",
    col_timestamp="timestamp",       # must be datetime-parseable
    col_trip_key="KEY",              # unique trip identifier
    col_main_mode="Main_Mode",      # ground-truth label (only needed for fit)
    gtfs_dir=Path("data/gtfs/"),    # optional: directory with GTFS .zip files
    osm_routes_dir=Path("data/osm_routes/"),  # optional: pre-extracted route parquets
)

# 2. Load your data
gps_df = pd.read_parquet("data/gps_points.parquet")
trips_df = pd.read_csv("data/trips.csv")

# 3. Train
model = ModeDetector(config)
model.fit(gps_df, trips_df)

# 4. Save the trained model
model.save("models/my_model.zip")
```

### Making predictions

```python
from modedetect import ModeDetector

# Load a trained model
model = ModeDetector.load("models/my_model.zip")

# Predict one mode per trip
predictions = model.predict(new_gps_df)
# Returns: DataFrame with columns [trip_key, predicted_mode]

# Predict class probabilities
probas = model.predict_proba(new_gps_df)
# Returns: DataFrame with columns [trip_key, WALKING, BIKE, CAR, BUS, ...]

# Multi-modal: segment trips and predict per segment
segment_predictions = model.predict(new_gps_df, segment=True)
# Returns: DataFrame with columns [trip_key, segment, predicted_mode]
```

## Input data format

### GPS DataFrame

Each row is a GPS point. Required columns (names configurable via `ModeDetectConfig`):

| Column | Description | Example |
|--------|-------------|---------|
| `latitude` | WGS84 latitude | `48.8566` |
| `longitude` | WGS84 longitude | `2.3522` |
| `timestamp` | Date/time of the fix (datetime or parseable string) | `2023-01-15 08:30:00` |
| `trip_key` | Identifier linking the point to a trip | `"49_1414"` |

Optional:

| Column | Description |
|--------|-------------|
| `speed_kmh` | GPS-reported speed in km/h. If absent, speed is computed from coordinates. |

### Trips DataFrame (training only)

Each row is a trip. Required columns:

| Column | Description | Example |
|--------|-------------|---------|
| `trip_key` | Must match the GPS data's trip key column | `"49_1414"` |
| `main_mode` | Ground-truth transport mode label | `"CAR"` |

For multi-modal trips, include additional columns `mode_1`, `mode_2`, ... to list the sequence of modes. When these are present, `fit()` automatically enables segmentation support.

## Configuration

All parameters are set through `ModeDetectConfig`. The defaults work well for urban GPS data at ~1 Hz sampling rate. Only column names and data paths typically need to be changed.

```python
from modedetect import ModeDetectConfig

config = ModeDetectConfig(
    # -- Column mapping (adapt to your dataset) --
    col_latitude="latitude",
    col_longitude="longitude",
    col_timestamp="timestamp",
    col_trip_key="trip_key",
    col_speed_kmh="speed_kmh",       # set to None to compute from coordinates
    col_main_mode="main_mode",       # ground-truth label column in trips_df

    # -- External data (optional, improves accuracy) --
    gtfs_dir=None,                   # path to directory containing GTFS .zip files
    osm_routes_dir=None,             # path to pre-extracted OSM route parquets

    # -- GPS quality --
    min_gps_points=10,               # trips with fewer points are dropped
    min_trip_duration_s=180,         # trips shorter than 3 min are dropped
    gap_threshold_s=300,             # gaps >5 min: drop trip (fit) or split (predict)

    # -- Mode consolidation --
    mode_remap={                     # merge similar modes before training
        "PRIV_CAR_DRIVER": "CAR",
        "PRIV_CAR_PASSENGER": "CAR",
        "TAXI": "CAR",
        "TRAIN_EXPRESS": "TRAIN",
    },
    modes_to_drop=["OTHER", "PLANE"],  # remove rare/ambiguous modes

    # -- Stop proximity (GTFS) --
    stop_search_radius_m=250,        # search radius around each GPS point

    # -- Route matching (OSM) --
    route_search_radius_m=20,        # match radius to route geometries
    route_densify_spacing_m=10,      # interpolation spacing on route lines

    # -- Classifier --
    n_folds=5,                       # stratified K-fold cross-validation
    random_state=0,
    xgb_params=None,                 # extra kwargs passed to XGBClassifier
)
```

## External data (GTFS and OSM)

GTFS and OSM data are **optional**. Without them, the model uses only motion features (speed, acceleration, jerk, bearing). Adding stop proximity and route matching features improves accuracy, especially for distinguishing between bus, tram, metro, and train.

### GTFS (stop proximity)

Place one or more GTFS `.zip` files in a directory and point `config.gtfs_dir` to it. The package reads `stops.txt`, `routes.txt`, `trips.txt`, and `stop_times.txt` from each zip to build a stop location database with route type information.

```
data/gtfs/
  IDFM-gtfs.zip        # one or more standard GTFS feeds
```

### OSM (route matching)

Pre-extract route networks from an OSM PBF file into parquet format. The package expects these files:

```
data/osm_routes/
  bus.parquet
  tram.parquet
  train.parquet
  subway.parquet
```

To generate them from a PBF file (requires `pyrosm`):

```python
from modedetect.features.routes import extract_osm_routes
from pathlib import Path

extract_osm_routes(Path("ile-de-france-latest.osm.pbf"), Path("data/osm_routes/"))
```

## Checkpointing

For large datasets, `fit()` supports checkpointing to save intermediate results. If the process is interrupted, the next run resumes from the last completed step.

```python
model.fit(gps_df, trips_df, checkpoint_dir="checkpoints/")
```

This saves parquet files for each pipeline stage:
- `01_gps_motion.parquet` -- GPS cleaning + motion feature extraction
- `02_stats.parquet`, `02_pct_map.parquet` -- trip statistics + percentile maps
- `03_stop_features.parquet` -- stop proximity features
- `04_route_features.parquet` -- route matching features

Delete the checkpoint directory to force a full rerun.

## Features computed

The pipeline extracts the following feature groups per trip (or per segment):

**Motion features** (from GPS coordinates):
- Speed: mean, median, std
- Acceleration: mean, median, std
- Jerk: mean, median, std
- Bearing change: mean, median, std
- Bearing rate: mean, median, std
- Total distance, total time, number of GPS points

**Percentile features** (selected automatically via Jensen-Shannon divergence):
- Optimal percentiles (P50-P95) of speed, acceleration, jerk, and bearing features
- Only the most discriminative percentiles are kept per feature

**Stop proximity features** (requires GTFS):
- Number of nearby stops: mean, min, max
- Minimum distance to nearest stop
- Per-mode minimum distances (bus, tram, metro, rail)
- Per-mode duration near stops (fraction of trip within search radius)

**Route matching features** (requires OSM):
- Match percentage per transit mode (bus, tram, train, subway)
- Fraction of GPS points falling near each route network

## Pipeline overview

```
GPS points + Trip labels
        |
        v
  [1] Clean GPS data (remove invalid coords, drop gaps/short trips)
        |
        v
  [2] Extract point-level motion features (speed, acc, jerk, bearing)
        |
        v
  [3] Filter speed outliers (per-mode 99th percentile thresholds)
        |
        v
  [4] Aggregate to trip-level statistics (mean, median, std)
        |
        v
  [5] Compute percentile maps + select optimal percentiles (JS divergence)
        |
        v
  [6] Compute stop proximity features (GTFS BallTree, optional)
        |
        v
  [7] Compute route match features (OSM BallTree, optional)
        |
        v
  [8] Merge all features + train XGBoost (stratified K-fold CV)
        |
        v
  Trained model (.zip artifact)
```

## Package structure

```
modedetect/
  __init__.py          # Public API: ModeDetector, ModeDetectConfig, ModeDetectError
  config.py            # ModeDetectConfig dataclass — all parameters in one place
  detector.py          # ModeDetector class (fit/predict/predict_proba/save/load)
  validation.py        # Input validators with actionable error messages
  preprocessing.py     # Coord cleaning, mode remap, gap handling, speed outliers
  artifacts.py         # Save/load model bundle as .zip
  _geo.py              # Haversine, bearing, geodesic helpers
  _types.py            # Column name constants
  features/
    motion.py          # Point-level: speed, acceleration, jerk, bearing
    statistics.py      # Trip/segment-level aggregation (mean, median, std)
    percentiles.py     # Percentile maps + JS divergence selection
    stops.py           # GTFS BallTree stop proximity
    routes.py          # OSM BallTree route matching
    segmentation.py    # PELT change-point detection (for multi-modal trips)
```

## Saving and loading models

Trained models are saved as `.zip` files containing the XGBoost model, label encoder, feature list, configuration, and learned thresholds. This is a self-contained artifact -- no external data is needed to make predictions.

```python
# Save after training
model.save("models/paris_v1.zip")

# Load for inference (no GTFS/OSM needed)
model = ModeDetector.load("models/paris_v1.zip")
predictions = model.predict(new_gps_df)
```

Note: for prediction, GTFS and OSM data paths should be configured if the model was trained with stop/route features. If they are not available at prediction time, those features will be filled with zeros.

## Error handling

The package raises `ModeDetectError` with actionable messages when inputs are invalid:

```python
from modedetect import ModeDetectError

try:
    model.fit(gps_df, trips_df)
except ModeDetectError as e:
    print(e)  # e.g. "GPS data missing required columns: ['latitude']. Adjust config.col_* ..."
```

Common issues:
- **Column name mismatch**: set `config.col_*` to match your DataFrame column names
- **No matching trip keys**: ensure the trip key column has the same values in both GPS and trips DataFrames
- **Timestamps not parseable**: convert to datetime before passing, or use a format pandas can parse
