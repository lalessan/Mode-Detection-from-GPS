import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow.dataset as ds
import numpy as np
import ast
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--segments", type=str, default="/home/s232534/scripts/segmentation_pelt/trip_segment_boundaries_pelt.csv")
parser.add_argument("--out-stem", type=str, default=None, help="Base filename for outputs",)
args, _ = parser.parse_known_args()

SEGMENTS_PATH = Path(args.segments)
out_stem = args.out_stem or f"features_stops_segmented__{SEGMENTS_PATH.stem}"

# Paths
GPS_DIR = Path("/home/s232534/data/split_motion_features_061125")
NEAREST_STOPS_DS = Path("/home/s232534/scripts/gtfs_functions/nearest_stops_ds")
TRIPS_PATH = Path("/home/lauale/netmob2025/data/trips_dataset.csv")

OUT_DIR = Path("/home/s232534/scripts/segmentation_pelt/multimodal_classification/Features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load stuff
df_trips = pd.read_csv(TRIPS_PATH)
dataset = ds.dataset(NEAREST_STOPS_DS, format="parquet", partitioning="hive")

df_segments = pd.read_csv(SEGMENTS_PATH)
df_segments["SEGMENTS_BOUNDS"] = df_segments["SEGMENTS_BOUNDS"].apply(ast.literal_eval) #evaluate as list[int]

# quantile spread.
QUANTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  

def process_one_file(parquet_path: Path) -> pd.DataFrame:
    """Process a single GPS parquet file and merge with stops data."""
    
    # Get user ID from filename
    cur_id = parquet_path.stem.removesuffix("_s_motion")
    
    try:
        #df_gps = pd.read_parquet(parquet_path, engine="pyarrow")
        # Only load in the key col as that's the only thing we need to correctly filter the stops.
        df_gps = pd.read_parquet(parquet_path, engine="pyarrow", columns=["KEY", "dt", "timestamp"])
    except Exception as e:
        print(f"Skipping {parquet_path.name}: {e}")
        return pd.DataFrame()

    if df_gps.empty:
        print(f"Skipping {parquet_path.name}: no rows")
        return pd.DataFrame()
    
    # Set correct dtypes
    df_gps["KEY"] = df_gps["KEY"].astype("string")
    
    # Load corresponding stops data
    try:
        # Try to find matching file_id in the dataset
        tbl_stops = dataset.to_table(filter=(ds.field("file_id") == cur_id))
        
        if tbl_stops.num_rows == 0:
            print(f"No stops data found for {cur_id}")
            return pd.DataFrame()
            
        df_stops = tbl_stops.to_pandas()
        
    except Exception as e:
        print(f"Error loading stops for {cur_id}: {e}")
        return pd.DataFrame()
    
    # Merge the two:
    df_merged = df_gps.merge(
        df_stops[['pt_idx', 'nearest_stops', 'count_within_radius', 'min_dist_m', 'file_id']], 
        left_index=True,
        right_on='pt_idx',
        how='left'
    )

    df_trips_inside = df_trips[df_trips['ID'] == cur_id].copy()
    
    # print(f"GPS rows: {len(df_gps)}, Stops rows: {len(df_stops)}, Merged: {len(df_merged)}")
    # print(f"Trips found: {df_merged['KEY'].nunique()} / {df_gps['KEY'].nunique()}")
    
    # Functions:
    def _calc_min_dist_mode(df_trip: pd.DataFrame) -> dict:
        stops_col = df_trip['nearest_stops'].values
        
        mode_dists = {0: [], 1: [], 2: [], 3: []}
    
        for stops_array in stops_col:
            if isinstance(stops_array, (list, np.ndarray)) and len(stops_array) > 0: # <- check type otherwise we run into len comapre issue when all NaNs
                top_stops = stops_array[:5]
            
                for stop in top_stops:
                    dist = stop['dist_m']
                    for rt in stop['r_type']:
                        if rt > 3:
                            continue
                        mode_dists[rt].append(dist)
                        
        return {
            'min_dist_0_tram': np.min(mode_dists[0]) if mode_dists[0] else np.nan,
            'min_dist_1_metro': np.min(mode_dists[1]) if mode_dists[1] else np.nan,
            'min_dist_2_rail': np.min(mode_dists[2]) if mode_dists[2] else np.nan,
            'min_dist_3_bus': np.min(mode_dists[3]) if mode_dists[3] else np.nan,
        }
                        
            
    def _calc_duration_near_mode(df_trip: pd.DataFrame) -> dict:
        stops_col = df_trip['nearest_stops'].values
        dt_values = df_trip['dt'].values
        
        mode_durations = {0: 0, 1: 0, 2: 0, 3: 0} 
        
        # Skip first and last row (dt is NaN)
        for i in range(1, len(stops_col) - 1):
            if isinstance(stops_col[i], (list, np.ndarray)) and len(stops_col[i]) > 0:
                # Get closest stop (first in list)
                #closest_stop = stops_col[i][0]
                closest_stops = stops_col[i][:5] # <- top 5 stops instead.
                time_delta = dt_values[i]
                
                # Add duration for each route type to our dict 
                unique_modes = set()
                for stop in closest_stops:
                    for rt in stop['r_type']:
                        if rt <= 3:
                            unique_modes.add(rt)

                for rt in unique_modes:
                    # Chaange in semnatics here a little since we are taking top 5 nearest stops.
                    # We only append time with this def "How long was I near at least one bus/tram/train etc stop?"
                    mode_durations[rt] += time_delta    
        return {
            'duration_near_0_tram_sec': mode_durations[0] if mode_durations[0] > 0 else np.nan,
            'duration_near_1_metro_sec': mode_durations[1] if mode_durations[1] > 0 else np.nan,
            'duration_near_2_rail_sec': mode_durations[2] if mode_durations[2] > 0 else np.nan,
            'duration_near_3_bus_sec': mode_durations[3] if mode_durations[3] > 0 else np.nan,
        }
    
    def _calc_stop_mode_over_journey(df_trip: pd.DataFrame) -> dict:
        stops_col = df_trip['nearest_stops'].values
        
        mode_dists = {0: [], 1: [], 2: [], 3: []}
    
        for stops_array in stops_col:
            if isinstance(stops_array, (list, np.ndarray)) and len(stops_array) > 0: # <- check type otherwise we run into len comapre issue when all NaNs
                rt_0 = False
                rt_1 = False
                rt_2 = False
                rt_3 = False
                
                for stop in stops_array:
                    dist = stop['dist_m']

                    if rt_0 and rt_1 and rt_2 and rt_3:
                        rt_0 = False
                        rt_1 = False
                        rt_2 = False
                        rt_3 = False
                        break
                    
                    for rt in stop['r_type']:
                        if rt > 3:
                            continue
                        
                        if rt == 0 and not rt_0:
                            rt_0 = True
                            mode_dists[rt].append(dist)
                        elif rt == 1 and not rt_1:
                            rt_1 = True
                            mode_dists[rt].append(dist)
                        elif rt == 2 and not rt_2:
                            rt_2 = True
                            mode_dists[rt].append(dist)
                        elif rt == 3 and not rt_3:
                            rt_3 = True
                            mode_dists[rt].append(dist)
              
        return {
            'list_first_dist_0_tram': mode_dists[0],
            'list_first_dist_1_metro': mode_dists[1],
            'list_first_dist_2_rail': mode_dists[2],
            'list_first_dist_3_bus': mode_dists[3],
        }
    
    trip_stats = []
    for trip_key, df_trip in df_merged.groupby("KEY", sort=False, observed=True):
        if pd.isna(trip_key):
            continue
        
        # Reset index to avoid slice deprication warning...
        df_trip = df_trip.reset_index(drop=True)
        
        # Mode data just like in the previous stat...
        trip_row = df_trips_inside[df_trips_inside['KEY'] == trip_key].head(1)
        
        if trip_row.empty:
            main_mode = pd.NA
            multimodal = pd.NA
        else:
            main_mode = trip_row["Main_Mode"].astype("string").iloc[0]
            extra_modes = trip_row[["Mode_2", "Mode_3", "Mode_4", "Mode_5"]].notna().any(axis=1).iloc[0]
            multimodal = (trip_row["Main_Mode"].iloc[0] != trip_row["Mode_1"].iloc[0]) or extra_modes
        
        previous_point = 0
        df_segments_trip = df_segments[df_segments['KEY'] == trip_key]
        segment_boundaries = df_segments_trip['SEGMENTS_BOUNDS'].iloc[0]
        for i, point in enumerate(segment_boundaries):
            df_trip_segment = df_trip[previous_point:point] # <-This is the segment of a trip
            previous_point = point
            
            # Calculations:
            dict_min_dist_per_mode = _calc_min_dist_mode(df_trip_segment)
            dict_duration_per_mode = _calc_duration_near_mode(df_trip_segment)
            dict_list_first_mode_stop = _calc_stop_mode_over_journey(df_trip_segment)
            
            trip_stats.append({
                "ID": cur_id,
                "FILE_NAME": parquet_path.name,
                "TRIP_KEY": trip_key,
                "SEGMENT": i,
                "MAIN_MODE": main_mode,
                "MULTI_MODAL": multimodal,
                
                # Stop statistics:
                "STOPS_NUM_AVG": df_trip_segment['count_within_radius'].mean(),
                "STOPS_NUM_MAX": df_trip_segment['count_within_radius'].max(),
                "STOPS_NUM_MIN": df_trip_segment['count_within_radius'].min(),
                
                "STOPS_MINDIST_AVG": df_trip_segment['min_dist_m'].mean(),
                
                "DURATION_TRAM": dict_duration_per_mode['duration_near_0_tram_sec'],
                "DURATION_METRO": dict_duration_per_mode['duration_near_1_metro_sec'],
                "DURATION_RAIL": dict_duration_per_mode['duration_near_2_rail_sec'],
                "DURATION_BUS": dict_duration_per_mode['duration_near_3_bus_sec'],
                
                "MIN_DIST_TRAM": dict_min_dist_per_mode['min_dist_0_tram'],
                "MIN_DIST_METRO": dict_min_dist_per_mode['min_dist_1_metro'],
                "MIN_DIST_RAIL": dict_min_dist_per_mode['min_dist_2_rail'],
                "MIN_DIST_BUS": dict_min_dist_per_mode['min_dist_3_bus'],
                
                # Quantile comprehension quite cool you can do this.
                **{f"STOPS_MINDIST_P{q}": df_trip_segment['min_dist_m'].quantile(q/100) for q in QUANTILES},
                **{f"STOPS_NUM_P{q}": df_trip_segment['count_within_radius'].quantile(q/100) for q in QUANTILES},
                
                **{
                    f"STOPS_CLOSEST_TRAM_P{q}":
                        (pd.Series(dict_list_first_mode_stop['list_first_dist_0_tram'], dtype="float64").quantile(q/100)
                        if dict_list_first_mode_stop['list_first_dist_0_tram'] else np.nan)
                    for q in QUANTILES
                },
                **{
                    f"STOPS_CLOSEST_METRO_P{q}":
                        (pd.Series(dict_list_first_mode_stop['list_first_dist_1_metro'], dtype="float64").quantile(q/100)
                        if dict_list_first_mode_stop['list_first_dist_1_metro'] else np.nan)
                    for q in QUANTILES
                },
                **{
                    f"STOPS_CLOSEST_RAIL_P{q}":
                        (pd.Series(dict_list_first_mode_stop['list_first_dist_2_rail'], dtype="float64").quantile(q/100)
                        if dict_list_first_mode_stop['list_first_dist_2_rail'] else np.nan)
                    for q in QUANTILES
                },
                **{
                    f"STOPS_CLOSEST_BUS_P{q}":
                        (pd.Series(dict_list_first_mode_stop['list_first_dist_3_bus'], dtype="float64").quantile(q/100)
                        if dict_list_first_mode_stop['list_first_dist_3_bus'] else np.nan)
                    for q in QUANTILES
                },
            })
    return pd.DataFrame(trip_stats) 

def main(limit: int | None = None, workers: int = 1) -> None:
    parquets = sorted(GPS_DIR.glob("*.parquet"))
    
    if limit is not None:
        parquets = parquets[:limit]

    if not parquets:
        print("No Parquet files found to process.")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    
    if workers == 1:
        results = [process_one_file(p) for p in tqdm(parquets, desc="Processing", unit="File")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(process_one_file, parquets), 
                              total=len(parquets), desc="Processing", unit="file"))
    
    # Merge all stats into one file - trying to do it all in ram rather than save out single files.
    df_all = pd.concat([r for r in results if not r.empty], ignore_index=True)
    
    df_all.to_parquet(OUT_DIR / out_stem, index=False, engine="pyarrow", compression="snappy")
    #df_all.to_csv(OUT_DIR / "stops_statistical_summary_enriched.csv", index=False)


if __name__ == "__main__":
    print("Using: ", os.cpu_count(), " CPUs...")
    main(limit=None, workers=os.cpu_count() or 4)
