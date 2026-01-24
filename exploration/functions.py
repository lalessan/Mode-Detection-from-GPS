import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000  # rad Earth meters
    
    # Convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # distance in meters

def avg_distance_between_points(p1s: list[tuple[float, float]] | np.ndarray,
                                p2s: list[tuple[float, float]] | np.ndarray) -> float:
    """
    Average haversine distance between corresponding points in p1s and p2s.
    Returns np.nan if there are fewer than 1 segments to compute.
    """
    p1s = np.asarray(p1s)
    p2s = np.asarray(p2s)

    # Handle empty or malformed inputs gracefully
    if p1s.size == 0 or p2s.size == 0:
        return float('nan')
    if p1s.ndim == 1:
        # Not enough dimensions (likely zero points)
        return float('nan')
    if p2s.ndim == 1:
        return float('nan')

    distances = haversine_m(p1s[:, 0], p1s[:, 1], p2s[:, 0], p2s[:, 1])
    if distances.size == 0:
        return float('nan')
    return float(np.nanmean(distances))

def gps_trace_length(p1s: list[tuple[float, float]] | np.ndarray,
                     p2s: list[tuple[float, float]] | np.ndarray) -> float:
    """
    Total length of a path computed by summing haversine distances between
    consecutive points. Returns 0.0 if there are fewer than 1 segments.
    """
    p1s = np.asarray(p1s)
    p2s = np.asarray(p2s)

    if p1s.size == 0 or p2s.size == 0:
        return 0.0
    if p1s.ndim == 1 or p2s.ndim == 1:
        return 0.0

    distances = haversine_m(p1s[:, 0], p1s[:, 1], p2s[:, 0], p2s[:, 1])
    if distances.size == 0:
        return 0.0
    return float(np.nansum(distances))

# def avg_distance_between_points_df(df, trip_id):
#     """
#     Given a DataFrame with 'trip_num', 'LATITUDE', and 'LONGITUDE' columns,
#     return the average distance between consecutive points for that trip_id.
#     Essentially a helper function.
#     """
#     trip_df = df[df['trip_num'] == trip_id]
#     # if len(trip_df) < 2:
#     #     return np.nan  # Not enough points to compute a distance
    
#     coords_start = list(zip(trip_df['LATITUDE'].values[:-1], trip_df['LONGITUDE'].values[:-1]))
#     coords_end = list(zip(trip_df['LATITUDE'].values[1:], trip_df['LONGITUDE'].values[1:]))
    
#     return avg_distance_between_points(coords_start, coords_end)

def avg_distance_between_points_df(df: pd.DataFrame, trip_id) -> float:
    """
    Given a DataFrame with 'trip_num', 'LATITUDE', and 'LONGITUDE' columns,
    return the average distance between consecutive points for that trip_id.
    Essentially a helper function.
    """
    df_gps_id = df[df['KEY'] == trip_id].copy()

    # Drop rows with missing coordinates
    df_gps_id = df_gps_id.dropna(subset=['LATITUDE', 'LONGITUDE'])

    if df_gps_id.shape[0] < 2:
        return float('nan')

    coords_start = list(zip(df_gps_id['LATITUDE'].values[:-1], df_gps_id['LONGITUDE'].values[:-1]))
    coords_end = list(zip(df_gps_id['LATITUDE'].values[1:], df_gps_id['LONGITUDE'].values[1:]))

    return avg_distance_between_points(coords_start, coords_end)

def gps_trace_length_df(df: pd.DataFrame, trip_id) -> float:
    """
    Given the df containing the GPS route, return the length of the route,
    taken by summing distances between consecutive points for the given trip_id.
    Returns 0.0 if fewer than 2 valid GPS records are present.
    """

    trip_df = df[df['KEY'] == trip_id].copy()
    trip_df = trip_df.dropna(subset=['LATITUDE', 'LONGITUDE'])

    if trip_df.shape[0] < 2:
        return 0.0

    coords_start = list(zip(trip_df['LATITUDE'].values[:-1], trip_df['LONGITUDE'].values[:-1]))
    coords_end = list(zip(trip_df['LATITUDE'].values[1:], trip_df['LONGITUDE'].values[1:]))

    return gps_trace_length(coords_start, coords_end)


def filter_motion_data_speed_avg(df_stats: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """_summary_

    Args:
        df_stats (pd.DataFrame): _description_
        thresholds (dict | None, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Easy access to thresholds can also be a defined dict.
    if thresholds is None:
        threshold = {
            "WALKING": 10,
            "BIKE": 28,
            "CAR": 42,
            "BUS": 42,        
            "TRAMWAY": 42,    
            "SUBWAY": 13,
            "TRAIN": 90,      
            "ELECT_SCOOTER": 22,  
            "ON_DEMAND": 22,      
            "TWO_WHEELER": 40,
            "ELECT_BIKE": 28,    
        }
    else:
        threshold = thresholds
    
    df_stats_slim = df_stats.copy()
    
    # # --- WALKING ---
    # walk_filter = ((df_stats_slim['MAIN_MODE']=='WALKING') & (df_stats_slim['SPEED_AVG'] > threshold['WALKING']))
    # print(f"At WALKING threshold {threshold['WALKING']}; ", walk_filter.sum(), " are above.")

    # df_stats_slim = df_stats_slim[~walk_filter] # <- Attach the fix to the DF:
    
    # # --- BIKING ---
    # bike_filter = ((df_stats_slim['MAIN_MODE']=='BIKE') & (df_stats_slim['SPEED_AVG'] > threshold['BIKE'])) # <- 28m/s = ~100 kmph which is VERY unlikely in Paris but maybe possible.
    # print(f"At BIKE threshold {threshold['BIKE']}; ", bike_filter.sum(), " are above.")

    # df_stats_slim = df_stats_slim[~bike_filter] 

    # # --- CAR/BUS/TRAM ---
    # car_filter = ((df_stats_slim['MAIN_MODE']=='CAR') & (df_stats_slim['SPEED_AVG'] > threshold['CAR'])) 
    # bus_filter = ((df_stats_slim['MAIN_MODE']=='BUS') & (df_stats_slim['SPEED_AVG'] > threshold['BUS'])) 
    # tram_filter = ((df_stats_slim['MAIN_MODE']=='TRAMWAY') & (df_stats_slim['SPEED_AVG'] > threshold['TRAMWAY']))
     
    # print(f"At CAR threshold {threshold['CAR']}; ", car_filter.sum(), " are above.")
    # print(f"At BUS threshold {threshold['BUS']}; ", bus_filter.sum(), " are above.")
    # print(f"At TRAMWAY threshold {threshold['TRAMWAY']}; ", tram_filter.sum(), " are above.")
    
    # df_stats_slim = df_stats_slim[~car_filter] 
    # df_stats_slim = df_stats_slim[~bus_filter] 
    # df_stats_slim = df_stats_slim[~tram_filter] 
    
    # # --- METRO ---
    # metro_filter = ((df_stats_slim['MAIN_MODE']=='SUBWAY') & (df_stats_slim['SPEED_AVG'] > threshold['SUBWAY']))
    # print(f"At SUBWAY threshold {threshold['SUBWAY']}; ", metro_filter.sum(), " are above.")

    # df_stats_slim = df_stats_slim[~metro_filter] 
    
    # # --- TRAINS ---
    # train_filter = ((df_stats_slim['MAIN_MODE']=='TRAIN') & (df_stats_slim['SPEED_AVG'] > threshold['TRAIN'])) 
    
    # print(f"At TRAIN threshold {threshold['TRAIN']}; ", train_filter.sum(), " are above.")
    # df_stats_slim = df_stats_slim[~train_filter] 
    
    # # --- ONDEMAND/TWOWHEELER/ELECTSCCOTER ---
    # elecscooter_filter = ((df_stats_slim['MAIN_MODE']=='ELECT_SCOOTER') & (df_stats_slim['SPEED_AVG'] > threshold['ELECT_SCOOTER'])) # <- 
    # ondemand_filter = ((df_stats_slim['MAIN_MODE']=='ON_DEMAND') & (df_stats_slim['SPEED_AVG'] > threshold['ON_DEMAND'])) # <- 
    # twowheeler_filter = ((df_stats_slim['MAIN_MODE']=='TWO_WHEELER') & (df_stats_slim['SPEED_AVG'] > threshold['TWO_WHEELER'])) # <- set this to ~150 kmph because it's a motorbike

    # print(f"At ELECT_SCOOTER threshold {threshold['ELECT_SCOOTER']}; ", elecscooter_filter.sum(), " are above.")
    # print(f"At ON_DEMAND threshold {threshold['ON_DEMAND']}; ", ondemand_filter.sum(), " are above.")
    # print(f"At TWO_WHEELER threshold {threshold['TWO_WHEELER']}; ", twowheeler_filter.sum(), " are above.")
    
    # df_stats_slim = df_stats_slim[~elecscooter_filter] 
    # df_stats_slim = df_stats_slim[~ondemand_filter] 
    # df_stats_slim = df_stats_slim[~twowheeler_filter] 
    
    # # --- ELECT BIKE ---
    # elect_bike_filter = ((df_stats_slim['MAIN_MODE']=='ELECT_BIKE') & (df_stats_slim['SPEED_AVG'] > threshold['ELECT_BIKE'])) # <- 28m/s = ~100 kmph which is VERY unlikely in Paris but maybe possible.
    # print(f"At ELECT_BIKE threshold {threshold['ELECT_BIKE']}; ", elect_bike_filter.sum(), " are above.")

    # df_stats_slim = df_stats_slim[~elect_bike_filter] 
    
    # return df_stats_slim
    
    # Fixed version: was getting some odd data misalignment and lucky I caught it otherwise might have led to some hard to trace trouble...
    #               fixed below is a "one-move" operation.
    
    # Add a sanity check to make sure that the tresholding main mode key exists in the df
    present_modes = pd.Index(df_stats_slim['MAIN_MODE'].dropna().unique())
    thr_keys = pd.Index(threshold.keys())
    unused_thresholds = thr_keys[~thr_keys.isin(present_modes)]
    if len(unused_thresholds):
        print(f"Note: thresholds not found in MAIN_MODE: {unused_thresholds.tolist()}")
    
    # -- THRESHOLDING
    per_row_thr = df_stats_slim['MAIN_MODE'].map(threshold)
    exceed = df_stats_slim['SPEED_AVG'] > per_row_thr  # NaN threshold -> False

    # Print counts per mode (preserving dict insertion order)
    for mode, thr in threshold.items():
        cnt = int(((df_stats_slim['MAIN_MODE'] == mode) & exceed).sum())
        print(f"At {mode} threshold {thr}; {cnt} are above.")

    # Filter once to avoid index misalignment
    return df_stats_slim.loc[~exceed]

def filter_motion_data_speed_percentile(df_stats: pd.DataFrame, thresholds: dict | None = None, percentile: str = 'SPEED_P50') -> pd.DataFrame:
    """_summary_

    Args:
        df_stats (pd.DataFrame): _description_
        thresholds (dict | None, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Easy access to thresholds can also be a defined dict.
    if thresholds is None:
        threshold = {
            "WALKING": 10,
            "BIKE": 28,
            "CAR": 42,
            "BUS": 42,        
            "TRAMWAY": 42,    
            "SUBWAY": 13,
            "TRAIN": 90,      
            "ELECT_SCOOTER": 22,  
            "ON_DEMAND": 22,      
            "TWO_WHEELER": 40,
            "ELECT_BIKE": 28,    
        }
    else:
        threshold = thresholds
    
    df_stats_slim = df_stats.copy()
    
    # Fixed version: was getting some odd data misalignment and lucky I caught it otherwise might have led to some hard to trace trouble...
    #               fixed below is a "one-move" operation.
    
    # Add a sanity check to make sure that the tresholding main mode key exists in the df
    present_modes = pd.Index(df_stats_slim['MAIN_MODE'].dropna().unique())
    thr_keys = pd.Index(threshold.keys())
    unused_thresholds = thr_keys[~thr_keys.isin(present_modes)]
    if len(unused_thresholds):
        print(f"Note: thresholds not found in MAIN_MODE: {unused_thresholds.tolist()}")
    
    # -- THRESHOLDING
    per_row_thr = df_stats_slim['MAIN_MODE'].map(threshold)
    exceed = df_stats_slim[percentile] > per_row_thr  # NaN threshold -> False

    # Print counts per mode (preserving dict insertion order)
    for mode, thr in threshold.items():
        cnt = int(((df_stats_slim['MAIN_MODE'] == mode) & exceed).sum())
        print(f"At {mode} threshold {thr}; {cnt} are above.")

    return df_stats_slim.loc[~exceed]

def remap_and_drop_modes_df(df: pd.DataFrame, mode_remap_dict: dict, modes_to_drop: list[str] = None) -> pd.DataFrame:
    df_filtered = df.copy()

    # Retain old as a fallback
    df_filtered['MAIN_MODE_ORIG'] = df_filtered['MAIN_MODE']

    # Remap to MAIN_MODE
    df_filtered['MAIN_MODE'] = df_filtered['MAIN_MODE'].replace(mode_remap_dict)

    # The categories we'd like to remove as they will not be "measurable" for us.
    if modes_to_drop is not None:
        df_filtered = df_filtered.loc[~df_filtered['MAIN_MODE'].isin(modes_to_drop)]
    
    return df_filtered

def filter_temporal_data(df, col: str = 'TOTAL_TIME', time_thres: int = 300) -> pd.DataFrame:
    df_copy = df.copy()
    
    if col not in df_copy.columns:
        # Soft fail TODO:maybe it's bad to do this?
        Warning(f"Note: column '{col}' not found. Returning input unchanged.")
        return df_copy
    else:
        mask = df_copy[col] > time_thres
        df_filtered = df_copy[mask]
        
        print(f"Filtered by {col} > {time_thres}. Dropped {int((~mask).sum())} rows.")

        return df_filtered
    
# Some useful data loading functions:
def filter_mode_random(df_stats: pd.DataFrame, mode: str = 'SUBWAY', single_modal: bool = True, min_trip_dur: int = None) -> pd.DataFrame:
    df_stat_mode = df_stats[df_stats["MAIN_MODE"] == mode].copy()
    
    # Modality
    df_stat_mode = df_stat_mode[df_stat_mode["MULTI_MODAL"] == (not single_modal)]
    
    if min_trip_dur is not None:
        df_stat_mode = df_stat_mode[df_stat_mode["TOTAL_TIME"] >= min_trip_dur]
    
    # Randonised Key
    random_key = np.random.choice(df_stat_mode['TRIP_KEY'].values)
    print(f"Trip key being sampled = {random_key}")
    
    # Load the GPS trace based on the key
    trip_id = re.search(r"\d+_\d+", random_key).group(0) # Pulls tge actual ID out of the trip key
    
    df = pd.read_parquet(f"/home/s232534/data/split_motion_features_061125/{trip_id}_s_motion.parquet") # Hard coded atm
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for that single trip
    df_trip = df[df['KEY'] == random_key]
    
    return df_trip

def load_trip_by_key(key: str):
    trip_id = re.search(r"\d+_\d+", key).group(0) # Pulls tge actual ID out of the trip key
    
    df = pd.read_parquet(f"/home/s232534/data/split_motion_features_061125/{trip_id}_s_motion.parquet") # Hard coded atm
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for that single trip
    df_trip = df[df['KEY'] == key]
    
    return df_trip
    
# --------------------------------------------
# --------- VISUALISATION FUNCTIONS ----------
# --------------------------------------------

def plot_hist_mode_2x2(df: pd.DataFrame, modes: list = ['TRAMWAY', 'SUBWAY', 'TRAIN' ,'BUS'], cols: list[str, str, str, str] = ['bus_match_pct','tram_match_pct','train_match_pct','subway_match_pct'], bin_num: int = 50, PROB_DENSITY: bool = True):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False) 
    
    for mode in modes:
        axs[0,0].hist(df[df['MAIN_MODE'] == mode][cols[0]], bins=bin_num, alpha=0.6, density=PROB_DENSITY, stacked=True, histtype='step', label=mode)
    axs[0,0].set_xlabel(f"Distribution of mode: {cols[0]}")
    axs[0,0].set_ylabel('Density')
    #axs[0,0].legend()
        
    for mode in modes:
        axs[0,1].hist(df[df['MAIN_MODE'] == mode][cols[1]], bins=bin_num, alpha=0.6, density=PROB_DENSITY, stacked=True, histtype='step', label=mode)
    axs[0,1].set_xlabel(f"Distribution of mode: {cols[1]}")
    axs[0,1].set_ylabel('Density')
    axs[0,1].legend()

    for mode in modes:
        axs[1,0].hist(df[df['MAIN_MODE'] == mode][cols[2]], bins=bin_num, alpha=0.6, density=PROB_DENSITY, stacked=True, histtype='step', label=mode)
    axs[1,0].set_xlabel(f"Distribution of mode: {cols[2]}")
    axs[1,0].set_ylabel('Density')
    #axs[1,0].legend()

    for mode in modes:
        axs[1,1].hist(df[df['MAIN_MODE'] == mode][cols[3]], bins=bin_num, alpha=0.6, density=PROB_DENSITY, stacked=True, histtype='step', label=mode)
    axs[1,1].set_xlabel(f"Distribution of mode: {cols[3]}")
    axs[1,1].set_ylabel('Density')
    #axs[1,1].legend()
    
    # Limit the x-axis to zoom a bit
    #axs[1,1].set_xlim(0, 6000)

    fig.suptitle(f'Distributions of _______________')
    plt.tight_layout()
    #plt.show()
    
    return fig, axs

def visualise_temporal_motion_features(df: pd.DataFrame, figsize=(12,10)) -> tuple[plt.Figure, plt.axes]:
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=figsize)
    
    axes[0].plot(df['timestamp'], df['SPEED_MS'], color='C0')
    axes[0].axhline(0, ls='--', lw=1, c='grey')
    axes[0].set_ylabel('Speed \n(m/s)')

    axes[1].plot(df['timestamp'], df['ACC_MS2'], color='C1')
    axes[1].axhline(0, ls='--', lw=1, c='grey')
    axes[1].set_ylabel('Acceleration \n(m/s²)')
    #axes[1].set_xlabel('Time')

    axes[2].plot(df['timestamp'], df['JERK_MS3'], color='C2')
    axes[2].axhline(0, ls='--', lw=1, c='grey')
    axes[2].set_ylabel('Jerk \n(m/s³)')
    #axes[2].set_xlabel('Time')

    axes[3].plot(df['timestamp'], df['BEARING_DEG_WGS84'], color='C3')
    axes[3].axhline(0, ls='--', lw=1, c='grey')
    axes[3].set_ylabel('Bearing \n(degrees)')
    #axes[3].set_xlabel('Time')
    axes[3].set_yticks(np.arange(0, 360, 45))

    axes[4].plot(df['timestamp'], df['BEARING_CHANGE_DEG_WGS84'], color='C4')
    axes[4].axhline(0, ls='--', lw=1, c='grey')
    axes[4].set_ylabel('BC \n(degrees)')
    #axes[4].set_xlabel('Time')
    axes[4].set_yticks(np.arange(0, 360, 45))

    axes[5].plot(df['timestamp'], df['BEARING_RATE_DEG_WGS84'], color='C5')
    axes[5].axhline(0, ls='--', lw=1, c='grey')
    axes[5].set_ylabel('BCR \n(degrees/s)')
    axes[5].set_xlabel('Time')
    # axes[5].set_yticks(np.arange(0, 360, 30))

    # plt.suptitle(f"Temporal Motion Features Signals for Trip: {trip_key}", fontsize=14)
    # plt.tight_layout()
    
    return fig, axes