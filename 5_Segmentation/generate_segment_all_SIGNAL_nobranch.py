import sys
sys.path.insert(0, "/home/s232534/scripts")

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

import signal_segmentation
from map_matching import routes
from typing import Dict

GPS_DIR = Path("/home/s232534/data/split_motion_features_061125")
STATS_PATH = Path("/home/s232534/data/stats_071125/statistical_summary_071125.csv")

OUT_DIR = Path("/home/s232534/scripts/segmentation_pelt/routesignal2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = "trip_segment_boundaries_pelt_SIGNAL-030126.csv"

# Load the heavy stuff:
print("Loading route parquet files...")
bus_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/bus_routes.parquet")
tram_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/tram_routes.parquet")
train_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/train_routes.parquet")
subway_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/subway_routes.parquet")

# Interpolation: Can be commented out for performance reasons.
# Since the point spacing in line geometries is very uneven, this means at times the fast_match might not match a GPS to a route, 
# even though the linegeometry would be crossing its path, this is discussed in the thesis
print("Interpolating routes...")
bus_routes = routes.add_densified_geometry(bus_routes, spacing_m=10, overwrite_geom=True)
tram_routes = routes.add_densified_geometry(tram_routes, spacing_m=10, overwrite_geom=True)
train_routes = routes.add_densified_geometry(train_routes, spacing_m=10, overwrite_geom=True)
subway_routes = routes.add_densified_geometry(subway_routes, spacing_m=10, overwrite_geom=True)

print("Generating BallTree for bus network...")
bus_arr = routes.bind_route_geometry_to_id(bus_routes)
BALLTREE_BUS = routes.generate_balltree_obj(bus_arr)

print("Generating BallTree for tram network...")
tram_arr = routes.bind_route_geometry_to_id(tram_routes)
BALLTREE_TRAM = routes.generate_balltree_obj(tram_arr)

print("Generating BallTree for train network...")
train_arr = routes.bind_route_geometry_to_id(train_routes)
BALLTREE_TRAIN = routes.generate_balltree_obj(train_arr)

print("Generating BallTree for subway network...")
subway_arr = routes.bind_route_geometry_to_id(subway_routes)
BALLTREE_SUBWAY = routes.generate_balltree_obj(subway_arr)

# Load the stats df:
df_stats = pd.read_csv(STATS_PATH, usecols=["TRIP_KEY", "MULTI_MODAL", "MAIN_MODE"])

def generate(csv_path: Path) -> None:
    # import warnings
    # warnings.filterwarnings("error", category=RuntimeWarning)
    # Imports per worker
    import pandas as pd
    from map_matching import routes
    import signal_segmentation
    
    global BALLTREE_BUS, BALLTREE_TRAM, BALLTREE_TRAIN, BALLTREE_SUBWAY, bus_arr, tram_arr, train_arr, subway_arr, df_stats
    
    try:
        df_gps = pd.read_parquet(csv_path, engine="pyarrow")
    except pd.errors.EmptyDataError:
        print(f"Skipping {csv_path.name}: empty file")
        return

    if df_gps.empty:
        print(f"Skipping {csv_path.name}: no rows")
        return

    if 'LOCAL DATETIME' not in df_gps.columns:
        print(f"Skipping {csv_path.name}: missing LOCAL DATETIME")
        return
    
    rows = []
    for key in df_gps['KEY'].unique():
        df_stats_lim = df_stats[df_stats['TRIP_KEY'] == key]
        
        df_lim = df_gps[df_gps['KEY'] == key]
        
        # Return what transport array we should use to generate signal
        # This will be done by reading the ground truth. So in this state it is only really useful for single-modal
        main_mode = df_stats_lim['MAIN_MODE'].values[0]
        
        # Amount of points check to catch mean(0) errors
        if df_lim.empty or len(df_lim) < 10:
            # too short/empty to segment
            breakpoints = [len(df_lim)]
            
            # So we catch early and return.
            rows.append(
            {
                "KEY": key,                
                "MAIN_MODE": main_mode,   
                "MULTI_MODAL": df_stats_lim['MULTI_MODAL'].values[0],  
                "SEGMENTS_BOUNDS": breakpoints,  
            })
            continue
        
        # flag: bool = False
        # breakpoints = None
        
        # branching check
        # if not main_mode in ("BUS", "TRAIN", "TRAIN_EXPRESS", "SUBWAY", "TRAMWAY"):
        #     flag = True
        
        # if flag:
        #     breakpoints = signal_segmentation.run_pelt(df_lim, metric='sb', visualise=False, pelt_tuning=(30, 10, 20))
       
        #     if len(breakpoints) == 1:
        #         # For reability we just hardcode 3 possible penalty values, for future robustness this should just be while loop.
        #         # We reduce and try smaller and smaller, and if still none are found we just return as what we have which will just be the last point index.
        #         pen_reduced = [15, 10, 5]
        #         for pen_r in pen_reduced:
        #             breakpoints = signal_segmentation.run_pelt(df_lim, metric='sb', visualise=False, pelt_tuning=(30, 10, pen_r))
        #             if len(breakpoints) > 1:
        #                 break
        # else:   
        
        matched_mode_bus = routes.fast_match(bus_arr, df_lim, BALLTREE_BUS, 20)
        signal_bus = routes.generate_route_signal(matched_mode_bus, bus_arr, df_lim, BALLTREE_BUS, 20, verbose=False)

        matched_mode_tram = routes.fast_match(tram_arr, df_lim, BALLTREE_TRAM, 20)
        signal_tram = routes.generate_route_signal(matched_mode_tram, tram_arr, df_lim, BALLTREE_TRAM, 20, verbose=False)

        matched_mode_train = routes.fast_match(train_arr, df_lim, BALLTREE_TRAIN, 20)
        signal_train = routes.generate_route_signal(matched_mode_train, train_arr, df_lim, BALLTREE_TRAIN, 20, verbose=False)

        matched_mode_subway = routes.fast_match(subway_arr, df_lim, BALLTREE_SUBWAY, 20)
        signal_subway = routes.generate_route_signal(matched_mode_subway, subway_arr, df_lim, BALLTREE_SUBWAY, 20, verbose=False)
        
        signal_arrs = (signal_bus, signal_train, signal_subway, signal_tram)

        breakpoints = signal_segmentation.run_pelt_signal_all_modes(df_lim, signal_arrs, main_mode, visualise=False, pelt_tuning=(30, 10, 20))
        
        if len(breakpoints) == 1:
            # For reability we just hardcode 3 possible penalty values, for future robustness this should just be while loop.
            # We reduce and try smaller and smaller, and if still none are found we just return as what we have which will just be the last point index.
            pen_reduced = [15, 10, 5]
            for pen_r in pen_reduced:
                breakpoints = signal_segmentation.run_pelt_signal_all_modes(df_lim, signal_arrs, main_mode, visualise=False, pelt_tuning=(30, 10, pen_r))
                if len(breakpoints) > 1:
                    break
                    
        rows.append(
        {
            "KEY": key,                
            "MAIN_MODE": main_mode,   
            "MULTI_MODAL": df_stats_lim['MULTI_MODAL'].values[0],  
            "SEGMENTS_BOUNDS": breakpoints,  
        })
    
    return pd.DataFrame(rows)
    # r = pd.DataFrame(rows)

    # PATH_OUTPUT = OUT_DIR / f"seg_{csv_path.stem}.parquet"
    # r.to_parquet(PATH_OUTPUT, index=False, engine="pyarrow", compression="snappy")
        
def main(limit: int | None = None, workers: int = 1) -> None:
    csvs = sorted(GPS_DIR.glob("*.parquet")) # Changed to parquet.
    
    if limit is not None:
        csvs = csvs[:limit]

    if not csvs:
        print("No Parquet files found to process.")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    # if workers == 1:
    #     for p in tqdm(csvs, desc="Processing", unit="File"):
    #         generate(p)
            
    # else:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(tqdm(ex.map(generate, csvs), total=len(csvs), desc="Processing", unit="file"))
    
    # concat
    df_all = pd.concat(
        [r for r in results if r is not None and not r.empty],
        ignore_index=True
    )
    # TODO: Saving out as a CSV means the list gets serialised as string... if we save as parquet PyArrow has a list object.
    df_all.to_csv(OUT_DIR / OUT_FILE, index=False)
    
if __name__ == "__main__":
    #main(limit=None, workers=os.cpu_count() or 4)
    main(limit=None, workers=24) # Issue with workers dying trying to see if going under CPU max will solve