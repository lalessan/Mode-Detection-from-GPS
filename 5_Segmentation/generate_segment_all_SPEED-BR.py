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

OUT_DIR = Path("/home/s232534/scripts/segmentation_pelt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the stats df:
df_stats = pd.read_csv(STATS_PATH, usecols=["TRIP_KEY", "MULTI_MODAL", "MAIN_MODE"])

def generate(csv_path: Path) -> None:
    # import warnings
    # warnings.filterwarnings("error", category=RuntimeWarning)
    # Imports per worker
    import pandas as pd
    from map_matching import routes
    import signal_segmentation
    
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
        
        # Run at our default values.
        breakpoints = signal_segmentation.run_pelt(df_lim, metric='sb', visualise=False, pelt_tuning=(30, 10, 20))
       
        if len(breakpoints) == 1:
            # For reability we just hardcode 3 possible penalty values, for future robustness this should just be while loop.
            # We reduce and try smaller and smaller, and if still none are found we just return as what we have which will just be the last point index.
            pen_reduced = [15, 10, 5]
            for pen_r in pen_reduced:
                breakpoints = signal_segmentation.run_pelt(df_lim, metric='sb', visualise=False, pelt_tuning=(30, 10, pen_r))
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
    
    if workers == 1:
        for p in tqdm(csvs, desc="Processing", unit="File"):
            generate(p)
            
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(generate, csvs), total=len(csvs), desc="Processing", unit="file"))
    
    # concat
    df_all = pd.concat(
        [r for r in results if r is not None and not r.empty],
        ignore_index=True
    )
    # TODO: Saving out as a CSV means the list gets serialised as string... if we save as parquet PyArrow has a list object.
    df_all.to_csv(OUT_DIR / "trip_segment_boundaries_pelt_SPEED-BR.csv", index=False)
    
if __name__ == "__main__":
    #main(limit=None, workers=os.cpu_count() or 4)
    main(limit=None, workers=50) # Issue with workers dying trying to see if going under CPU max will solve