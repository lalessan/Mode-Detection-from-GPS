import os, psutil
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import numpy as np
from tqdm import tqdm

# from gtfs_functions import load_gtfs
# from gtfs_functions import nearest_stops

import load_gtfs, nearest_stops

# SET VARS
SEARCH_RADIUS: int = 250 
OUT_DIR = "/home/s232534/scripts/gtfs_functions/nearest_stops_ds_250m"

print("Building GTFS dict and lookup objects...")
DICT_GTFS = nearest_stops.generate_GTFS_dict()         
PMBT_OBJECT, _ = load_gtfs.build_lookup_objects()      
print("Done building heavy objects!")

# mem check
process = psutil.Process(os.getpid())
print(f"Resident Set Size: {process.memory_info().rss / 1e6:.1f} MB")
print(f"Virtual Memory Size: {process.memory_info().vms / 1e6:.1f} MB")

# Function for each worker/core - generates a Parquet database of nearest stop per-point statistics. 
# NOTE: Remember to update the OUT_DIR
def generate(csv_path: Path) -> None:
    global DICT_GTFS, PMBT_OBJECT
    import pandas as pd
    import geopandas as gpd
    import nearest_stops
    
    # Load the GPS data
    try:
        df_gps = pd.read_parquet(csv_path, engine="pyarrow")
        # turn into a geodataframe
        df_gps = gpd.GeoDataFrame(df_gps, geometry=gpd.points_from_xy(df_gps.LONGITUDE, df_gps.LATITUDE), crs="EPSG:4326")
        
    except pd.errors.EmptyDataError:
        print(f"Skipping {csv_path.name}: empty file")
        return

    if df_gps.empty:
        print(f"Skipping {csv_path.name}: no rows")
        return
    
    # Fill the database.
    nearest_stops.feature_nearest_stops_to_parquet_dataset(df_gps, DICT_GTFS, PMBT_OBJECT, out_dir=OUT_DIR, radius_default_m=SEARCH_RADIUS)
    
def main(limit: int | None = None, workers: int = 1) -> None:
    GPS_DIR = Path("/home/s232534/data/split_motion_features_061125")
    
    csvs = sorted(GPS_DIR.glob("*.parquet")) # Changed to parquet.
    
    if limit is not None:
        csvs = csvs[:limit]

    if not csvs:
        print("No Parquet files found to process.")
        return
    
    if workers == 1:
        #for p in csvs:
        for p in tqdm(csvs, desc="Processing", unit="File"):
            generate(p)
            
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(generate, csvs), total=len(csvs), desc="Processing", unit="file"))
            
if __name__ == "__main__":
    print("Using: ", os.cpu_count(), " CPUs...")
    main(limit=None, workers=os.cpu_count())