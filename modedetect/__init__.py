"""modedetect — Transport mode detection from GPS traces.

Usage::

    from modedetect import ModeDetector, ModeDetectConfig

    config = ModeDetectConfig(gtfs_dir=Path("data/gtfs/"))
    model = ModeDetector(config)
    model.fit(gps_df, trips_df)
    model.save("models/paris_v1.zip")

    model = ModeDetector.load("models/paris_v1.zip")
    predictions = model.predict(new_gps_df)
"""

from .config import ModeDetectConfig
from .detector import ModeDetector
from .validation import ModeDetectError

__all__ = ["ModeDetector", "ModeDetectConfig", "ModeDetectError"]
