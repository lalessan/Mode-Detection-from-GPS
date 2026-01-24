import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import pyproj
from exploration import functions as fn
from tqdm import tqdm

# Paths
GPS_DIR = Path("/home/s232534/data/split") # <- Loading in our already labelled split GPS data
TRIPS_PATH = Path("/home/s232534/data/stats/statistical_summary_valid.csv") # <- Use our filtered stat summary as our KEY table | This can be replaced with any key table for example the raw NetMob25 trips.csv.
OUT_DIR = Path("/home/s232534/data/split_motion_features_061125")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load our keys dataset ---
df_keys = pd.read_csv(TRIPS_PATH)

# Remapping dtypes:
DTYPE_MAP_GPS = {
    "VALID": "string", "LATITUDE":"float64","LONGITUDE":"float64",
    "SPEED":"float32","KEY":"string"
}

USECOLS = ["timestamp","UTC DATETIME","LOCAL DATETIME","VALID",
           "LATITUDE","LONGITUDE","SPEED","KEY"]

# Worker/process function that gets distributed among the CPU cores.
def process_one(csv_path: Path) -> None:
    # --- Load GPS ---
    try:
        df_gps = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPE_MAP_GPS,
        parse_dates=["timestamp","UTC DATETIME","LOCAL DATETIME"],
        engine="pyarrow",
)
    except pd.errors.EmptyDataError:
        print(f"Skipping {csv_path.name}: empty file")
        return
    if df_gps.empty:
        print(f"Skipping {csv_path.name}: no rows")
        return
    
    # Safety filtering -> sort by timeline and remove duplicates as a JUST-IN-CASE:
    df_gps = df_gps.sort_values(["KEY","timestamp"], kind="mergesort") # <- More stable sort cant really find too much clear documentation on the sorting algo but mergesort is stable (which means it selects the sorting alg to the data.)
    df_gps = df_gps[~df_gps["timestamp"].duplicated(keep="first")]#.copy() # Drop duplicates and only keep the first one that we see
    
    # Additions of some needed additional columns:
    # Speed is in km/h there for we add another col for m/s -> 1km/h = 1000m / 3600 seconds.
    df_gps["SPEED_MS"] = df_gps["SPEED"] / 3.6
    
    df_gps["LAT_RAD"] = np.radians(df_gps["LATITUDE"].to_numpy())
    df_gps["LON_RAD"] = np.radians(df_gps["LONGITUDE"].to_numpy())
    
    # Calc acceleration and add a new col.
    df_acc = calculate_acceleration(df_gps, gap_offset=13)
    df_acc_bearing = calculate_bearing_rate(df_acc)
    
    # Parquet instead:
    out_path = OUT_DIR / f"{csv_path.stem}_motion.parquet"
    df_acc_bearing.to_parquet(out_path, index=False)

def calculate_acceleration(df: pd.DataFrame, gap_offset: int = 13):
    """Take a raw GPS DataFrame and calculate the acceleration from the doppler shifted speed col 
    and from the time difference between two points. Using acceleration also calculate the Jerk.

    Args:
        df (pd.DataFrame): _description_
        gap_offset (int, optional): _description_. Defaults to 13.

    Returns:
        DataFrame: Including the new col of Acceleration and Jerk.
    """
    # Split out by KEY as we do not want to calculate the acceleration over two different trips.
    temp_list_trips = []
    for _, df_trip in df.groupby("KEY", sort=False):
        df_trip['dt'] = df_trip['timestamp'].diff().dt.total_seconds()#.fillna(0) # <- had trouble with filling NaN so doing this way and then catching divide inf.
        
        dt_vals = df_trip['dt'].to_numpy(dtype='float64', copy=True)
        dt_vals[~np.isfinite(dt_vals)] = np.nan
        MAX_GAP = (np.nanmedian(dt_vals) + gap_offset) if np.isfinite(dt_vals).any() else gap_offset
        
        df_trip['segment'] = (df_trip['dt'] > MAX_GAP).cumsum()
        
        segments = []
        for _, segment in df_trip.groupby('segment', sort=False):
            dv = segment["SPEED_MS"].diff() # NaN at first
            dt = segment["dt"]
            acc = np.divide(dv, dt, out=np.full(len(dv), np.nan), where=(dt > 0) & ~np.isnan(dt)) # <- new addition we no longer fill 0's anywhere so now we check for NaNs!
            segment['ACC_MS2'] = acc
            
            # Calculate jerk here:
            da = segment['ACC_MS2'].diff()
            jerk = np.divide(da, dt, out=np.full(len(da), np.nan), where=(dt > 0) & ~np.isnan(dt))
            segment['JERK_MS3'] = jerk
            
            segments.append(segment)
            
        temp_list_trips.append(pd.concat(segments, ignore_index = True))
     
    rebuilt_df = pd.concat(temp_list_trips, ignore_index=True)   
     
    return rebuilt_df
    
def calculate_bearing_rate(df: pd.DataFrame):
    # NOTE: This operation requires the df to already be checked and labelled for segments. Which happens in fn calculate_acceleration()
    segments = []
    for _, df_trip in df.groupby("KEY", sort=False):
        for _, segment in df_trip.groupby('segment', sort=False):
            lat0 = segment['LAT_RAD'].shift(1) # <- shift adds a NaN at the top of the col, hence moving index 0 to 1 which is [i - 1].
            lon0 = segment['LON_RAD'].shift(1)
            lat1 = segment['LAT_RAD']
            lon1 = segment['LON_RAD']
            
            # Segment bearing
            # Taken from: https://mapscaping.com/how-to-calculate-bearing-between-two-coordinates/
            dlon = lon1 - lon0
            x = np.sin(dlon) * np.cos(lat1)
            y = np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)
            bearing_deg = np.degrees(np.arctan2(x, y))
            bearing_deg = (bearing_deg + 360) % 360
            segment['BEARING_DEG'] = bearing_deg 
            
            # TEST USING PYPROJ.
            #NOTE: Geod requires values to be in DEGREES and not RADIANS
            # Using the Geod function should net us more accurate results since instead of projecting to a 
            # perfect sphere we are projecting to a ellipsoid Earth shape using WGS84 projection.
            global GEOD # grab the init worker global geod.
            # Convert from rad -> degrees
            lon0_d, lat0_d, lon1_d, lat1_d = np.degrees((lon0, lat0, lon1, lat1))
            
            fwd_azimuth,back_azimuth,distance = GEOD.inv(lon0_d, lat0_d, lon1_d, lat1_d)
            
            # Clamp it like bearing deg
            fwd_azimuth = (fwd_azimuth + 360) % 360
            segment['BEARING_DEG_WGS84'] = fwd_azimuth
            
            # NOTE: HERE WE ALSO HANDLE ADDING DISTANCE BETWEEN POINTS FOR WGS84 PROJECTION AND USING NORMAL HAVERSINE DISTANCE (LINEAR)
            segment['DIST_BP'] = fn.haversine_m(lat0_d, lon0_d, lat1_d, lon1_d)
            segment['DIST_BP_WGS84'] = distance
            
            # Handle stacked coordinates (where we have the exact same coords for a number of entries but speed has values...)
            # Our bearing output will give us incorrect readings because there is no angle to the next coords if they're ontop of each other.
            # IMPORTANT: This must be done BEFORE calculating bearing rates!
            mask_stacked_coords = (lon0.eq(lon1)) & (lat0.eq(lat1))
            segment.loc[mask_stacked_coords, ['BEARING_DEG', 'BEARING_DEG_WGS84', 'DIST_BP','DIST_BP_WGS84']] = np.nan
            
            mask_first = segment.index == segment.index.min()
            segment.loc[mask_first, ['BEARING_DEG', 'BEARING_DEG_WGS84', 'DIST_BP','DIST_BP_WGS84']] = np.nan
            
            # Ammendment of Bearing Rate -> to be what Turning Rate is... 
            def _shortest_turn_deg(delta):
                return (delta + 180.0) % 360.0 - 180.0
            
            dt = segment['dt']

            # BEARING_CHANGE_RATE_DEG
            beta_next = segment['BEARING_DEG'].shift(-1)
            d_beta = _shortest_turn_deg(beta_next - segment['BEARING_DEG'])  # deg
            rate = np.divide(np.abs(d_beta), dt, out=np.full(len(dt), np.nan), where=np.isfinite(dt) & (dt > 0))
            segment['BEARING_RATE_DEG'] = rate  # deg/s (unsigned)

            # BEARING_CHANGE_RATE_DEG_WGS84
            beta_next_wgs84 = segment['BEARING_DEG_WGS84'].shift(-1)
            d_beta_w = _shortest_turn_deg(beta_next_wgs84 - segment['BEARING_DEG_WGS84'])  # deg
            rate_w = np.divide(np.abs(d_beta_w), dt, out=np.full(len(dt), np.nan), where=np.isfinite(dt) & (dt > 0))
            segment['BEARING_RATE_DEG_WGS84'] = rate_w  # deg/s (unsigned)
            
            # BEARING CHANGE DEG (no time)
            segment['BEARING_CHANGE_DEG'] = np.abs((beta_next - segment['BEARING_DEG'] + 180.0) % 360.0 - 180.0)
            segment['BEARING_CHANGE_DEG_WGS84'] = np.abs((beta_next_wgs84 - segment['BEARING_DEG_WGS84'] + 180.0) % 360.0 - 180.0)
            
            segments.append(segment)
            
    
    return pd.concat(segments, ignore_index = True)

def _init_worker():
    # runs once per process rather than running it once per loop 
    global GEOD
    GEOD = pyproj.Geod(ellps="WGS84")
    
def main(limit: int | None = None, workers: int = 1, id_prefix: str = "") -> None:
    # Collect file paths from our valid gps split data.
    csvs = sorted([GPS_DIR / filename for filename in df_keys["FILE_NAME"]]) 

    if limit is not None:
        csvs = csvs[:limit]

    if not csvs:
        print("No CSVs found to process.")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        list(tqdm(ex.map(process_one, csvs, chunksize=16),total=len(csvs), desc="Processing", unit="file"))

if __name__ == "__main__":
    main(limit=None, workers=os.cpu_count(), id_prefix="")
