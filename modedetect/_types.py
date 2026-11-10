"""Column name constants and internal schema definitions."""

# --- Required GPS columns (user-facing names after schema mapping) ---
GPS_REQUIRED_COLS = {"latitude", "longitude", "timestamp", "trip_key"}
GPS_OPTIONAL_COLS = {"speed_kmh"}  # if absent, computed from coordinates

# --- Required trips columns ---
TRIPS_REQUIRED_COLS = {"trip_key", "main_mode"}
TRIPS_MULTIMODAL_COLS_PREFIX = "mode_"  # mode_1, mode_2, ... OR a list column "modes"

# --- Internal column names (produced by feature extraction) ---
# Point-level motion features
SPEED_MS = "speed_ms"
ACC_MS2 = "acc_ms2"
JERK_MS3 = "jerk_ms3"
BEARING_DEG = "bearing_deg"
BEARING_DEG_WGS84 = "bearing_deg_wgs84"
BEARING_CHANGE_DEG = "bearing_change_deg"
BEARING_CHANGE_DEG_WGS84 = "bearing_change_deg_wgs84"
BEARING_RATE_DEG = "bearing_rate_deg"
BEARING_RATE_DEG_WGS84 = "bearing_rate_deg_wgs84"
DIST_BP = "dist_bp"
DIST_BP_WGS84 = "dist_bp_wgs84"
DT = "dt"

# Trip-level statistics (prefixes)
STAT_SUFFIXES = ("avg", "med", "std", "p85")

# Percentile map features and their source columns
PERCENTILE_FEATURES = {
    SPEED_MS: "SPEED",
    ACC_MS2: "ACC",
    JERK_MS3: "JERK",
    BEARING_CHANGE_DEG_WGS84: "BC_WGS84",
    BEARING_RATE_DEG_WGS84: "BR_WGS84",
}

# Stop proximity route types (from GTFS)
STOP_ROUTE_TYPES = {"tram": 0, "metro": 1, "rail": 2, "bus": 3}

# Route matching modal networks
ROUTE_NETWORKS = ("walking", "cycling", "driving", "bus", "subway", "train", "tram")
