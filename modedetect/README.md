# modedetect

A Python package for **transport mode detection from GPS traces**. Given timestamped GPS trajectories grouped into trips, it classifies each trip's transport mode (walk, bike, car, bus, metro, train, tram).

The package follows a scikit-learn-style API: `fit()` to train on labelled data, `predict()` to classify new trips, and `save()`/`load()` to persist trained models.

## Installation

From the repository root:

```bash
pip install -e .
```

For OSM route extraction (optional):

```bash
pip install -e ".[osm]"
```

## What you need

### Required: GPS traces with trip identifiers

The minimum input is a **GPS DataFrame** -- one row per recorded GPS point, with at least a latitude, longitude, timestamp, and a trip identifier. Most GPS datasets already look like this:

| lat | lon | time | trip_id |
|-----|-----|------|---------|
| 48.857 | 2.352 | 2023-01-15 08:30:00 | trip_42 |
| 48.858 | 2.354 | 2023-01-15 08:30:05 | trip_42 |
| ... | ... | ... | ... |

For **training**, you also need a **trips DataFrame** with one row per trip and a ground-truth mode label:

| trip_id | mode |
|---------|------|
| trip_42 | CAR |
| trip_43 | BUS |

For **prediction**, only the GPS DataFrame is needed.

For multi-modal trips (e.g. walk then bus then walk), you can add columns `mode_1`, `mode_2`, ... to the trips DataFrame listing the sequence of modes; the package will then automatically enable segmentation during training.

Your column names will probably differ from the defaults. You tell the package which columns to use through `ModeDetectConfig`:

```python
config = ModeDetectConfig(
    col_latitude="LATITUDE",       # name of the latitude column in your GPS data
    col_longitude="LONGITUDE",     # name of the longitude column
    col_timestamp="timestamp",     # name of the timestamp column (must be datetime-parseable)
    col_trip_key="KEY",            # name of the column linking GPS points to trips
    col_main_mode="Main_Mode",    # name of the mode label column in trips_df (training only)
    col_speed_kmh="SPEED",        # name of a GPS speed column, or None to compute from coords
)
```

### Optional: GTFS and OSM data (for transit features)

Without any external data, the model classifies trips using **motion features** alone -- statistics derived from speed, acceleration, jerk, and bearing changes. This works well for distinguishing walk, bike, and car, but transit modes (bus, tram, metro, train) can look similar in motion alone.

To improve transit mode accuracy, you can provide two additional data sources:

- **GTFS feeds** (stop proximity features): Standard transit schedule files that contain stop locations. The package measures how close each GPS point is to bus stops, tram stops, metro stations, etc. A trip that consistently passes near metro stations is more likely to be a metro trip. Point `config.gtfs_dir` to a directory containing one or more GTFS `.zip` files.

- **OSM route geometries** (route matching features): OpenStreetMap data containing the physical paths of transit routes. The package checks what fraction of a trip's GPS points fall along a bus route, tram line, train track, etc. Point `config.osm_routes_dir` to a directory containing pre-extracted route parquets (see [External data setup](#external-data-setup) below).

Both are optional. You can use one, both, or neither.

## Quick start

### Training a model

```python
from pathlib import Path
import pandas as pd
from modedetect import ModeDetector, ModeDetectConfig

# Configure column names and optional data paths
config = ModeDetectConfig(
    col_latitude="LATITUDE",
    col_longitude="LONGITUDE",
    col_timestamp="timestamp",
    col_trip_key="KEY",
    col_main_mode="Main_Mode",
    gtfs_dir=Path("data/gtfs/"),              # optional
    osm_routes_dir=Path("data/osm_routes/"),  # optional
)

# Load your data
gps_df = pd.read_parquet("data/gps_points.parquet")
trips_df = pd.read_csv("data/trips.csv")

# Train
model = ModeDetector(config)
model.fit(gps_df, trips_df)

# Save the trained model
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

| Column | Default name | Description | Example |
|--------|-------------|-------------|---------|
| Latitude | `latitude` | WGS84 latitude in degrees | `48.8566` |
| Longitude | `longitude` | WGS84 longitude in degrees | `2.3522` |
| Timestamp | `timestamp` | Date/time of the GPS fix (datetime or parseable string) | `2023-01-15 08:30:00` |
| Trip key | `trip_key` | Identifier linking this point to a trip | `"49_1414"` |

Optional:

| Column | Default name | Description |
|--------|-------------|-------------|
| Speed | `speed_kmh` | GPS-reported speed in km/h. If not provided (set `col_speed_kmh=None`), speed is computed from successive coordinates and timestamps. |

### Trips DataFrame (training only)

Each row is a trip. Required columns:

| Column | Default name | Description | Example |
|--------|-------------|-------------|---------|
| Trip key | `trip_key` | Must match the GPS data's trip key column | `"49_1414"` |
| Mode label | `main_mode` | Ground-truth transport mode | `"CAR"` |

For multi-modal trips, include additional columns `mode_1`, `mode_2`, ... listing the sequence of modes used during the trip. When these columns are present, `fit()` automatically enables segmentation support.

## Configuration

All parameters are set through `ModeDetectConfig`. The defaults work for urban GPS data at ~1 Hz sampling rate. You typically only need to set column names and data paths.

```python
from modedetect import ModeDetectConfig

config = ModeDetectConfig(
    # -- Column mapping (match your DataFrame column names) --
    col_latitude="latitude",
    col_longitude="longitude",
    col_timestamp="timestamp",
    col_trip_key="trip_key",
    col_speed_kmh="speed_kmh",       # set to None to compute from coordinates
    col_main_mode="main_mode",       # ground-truth label column in trips_df

    # -- External data (optional, improves transit mode accuracy) --
    gtfs_dir=None,                   # path to directory containing GTFS .zip files
    osm_routes_dir=None,             # path to pre-extracted OSM route parquets

    # -- GPS quality filters --
    min_gps_points=10,               # drop trips with fewer GPS points
    min_trip_duration_s=180,         # drop trips shorter than 3 minutes
    gap_threshold_s=300,             # temporal gaps > 5 min: drop (fit) or split (predict)

    # -- Mode consolidation (applied before training) --
    mode_remap={                     # merge similar mode labels
        "PRIV_CAR_DRIVER": "CAR",
        "PRIV_CAR_PASSENGER": "CAR",
        "TAXI": "CAR",
        "TRAIN_EXPRESS": "TRAIN",
    },
    modes_to_drop=["OTHER", "PLANE"],  # remove rare/ambiguous modes

    # -- Stop proximity (requires GTFS) --
    stop_search_radius_m=250,        # search radius around each GPS point (metres)

    # -- Route matching (requires OSM) --
    route_search_radius_m=20,        # max distance from GPS point to route (metres)
    route_densify_spacing_m=10,      # interpolation spacing on route geometries (metres)

    # -- Classifier --
    n_folds=5,                       # stratified K-fold cross-validation during training
    random_state=0,                  # for reproducibility
    xgb_params=None,                 # extra kwargs passed to XGBClassifier
)
```

## External data setup

GTFS and OSM data are **optional**. Without them, the model uses only motion features. Adding them improves accuracy for transit modes.

### GTFS (stop proximity)

Place one or more standard GTFS `.zip` files in a directory. The package reads `stops.txt`, `routes.txt`, `trips.txt`, and `stop_times.txt` to build a stop location database with route type information (bus, tram, metro, rail).

```
data/gtfs/
  IDFM-gtfs.zip        # standard GTFS feed(s) for your region
```

GTFS feeds are published by transit agencies. For example, the Ile-de-France feed is available from [data.iledefrance-mobilites.fr](https://data.iledefrance-mobilites.fr).

### OSM (route matching)

The package expects pre-extracted route network parquet files:

```
data/osm_routes/
  bus.parquet
  tram.parquet
  train.parquet
  subway.parquet
```

To generate these from an OpenStreetMap PBF file (requires `pip install modedetect[osm]`):

```python
from modedetect.features.routes import extract_osm_routes
from pathlib import Path

extract_osm_routes(Path("ile-de-france-latest.osm.pbf"), Path("data/osm_routes/"))
```

Regional PBF files can be downloaded from [download.geofabrik.de](https://download.geofabrik.de).

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

Trained models are saved as `.zip` files containing the XGBoost model, label encoder, feature list, configuration, and learned thresholds.

```python
# Save after training
model.save("models/paris_v1.zip")

# Load for inference
model = ModeDetector.load("models/paris_v1.zip")
predictions = model.predict(new_gps_df)
```

The `.zip` artifact stores everything needed for the classifier itself. However, if the model was trained with GTFS/OSM features, the same external data should be available at prediction time so the package can compute those features for new trips. If the data is not available, those features are filled with zeros and predictions rely on motion features alone.

### What is stored in the artifact

| File | Contents |
|------|----------|
| `model.ubj` | XGBoost decision trees (split thresholds and leaf weights) |
| `label_encoder.joblib` | Mode label mapping (e.g. `CAR -> 0, BUS -> 1`) |
| `feature_list.json` | Ordered list of feature column names |
| `config.json` | Configuration parameters (paths are stripped, see below) |
| `speed_thresholds.json` | Per-mode 99th percentile speed thresholds |
| `optimal_percentiles.json` | Which percentile features were selected |
| `metadata.json` | Timestamp, class list, feature count |

**No GPS coordinates, trip identifiers, timestamps, or individual-level data are stored.** The XGBoost model contains only split thresholds on aggregate feature statistics (e.g. "if SPEED_AVG < 12.3, go left") learned over the full training set, not individual trips.

### Path privacy

Local file paths (GTFS directory, OSM directory) are **automatically stripped** from the saved artifact so the `.zip` is safe to share. When loading a model, paths are restored from a per-machine config file at `~/.modedetect.json`.

Create this file on each machine where you use the package:

```json
{
    "gtfs_dir": "/path/to/your/gtfs",
    "osm_routes_dir": "/path/to/your/osm_routes"
}
```

This file lives in your home directory, is not part of the repository, and is never included in the model artifact. Each machine can have different paths.

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
