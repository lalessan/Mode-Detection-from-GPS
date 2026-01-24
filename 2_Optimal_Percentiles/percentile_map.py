import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import numpy as np
from tqdm import tqdm

GPS_DIR = Path("/home/s232534/data/split_motion_features_061125")

DF_STATS_PATH = Path("/home/s232534/data/stats_071125/statistical_summary_071125.csv")
OUT_DIR = Path("/home/s232534/data/percentile_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_stats_pregen = pd.read_csv(DF_STATS_PATH)

# Create lookup dictionaries once at module level
trip_lookup = df_stats_pregen.set_index('TRIP_KEY')[['MAIN_MODE', 'MULTI_MODAL']].to_dict('index')

# Verbalised list of the quantiles we want to test for. 
QUANTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  

FEATURES = {
    "SPEED_MS": "SPEED",
    "ACC_MS2": "ACC",
    "JERK_MS3": "JERK",
    "BEARING_CHANGE_DEG": "BC",
    "BEARING_CHANGE_DEG_WGS84": "BC_WGS84",
    "BEARING_RATE_DEG": "BR",
    "BEARING_RATE_DEG_WGS84": "BR_WGS84",
    # "BEARING_DEG": "B",
    # "BEARING_DEG_WGS84": "B_WGS84",
}

def _add_quantiles(row_dict, df_trip, col, prefix):
    # Do it in one operation for all rows
    qs = df_trip[col].quantile([p/100.0 for p in QUANTILES]).to_dict()
    
    # Generate the cols to be readable:
    for p in QUANTILES:
        row_dict[f"{prefix}_P{p}"] = float(qs.get(p/100.0, np.nan))

def generate(csv_path: Path) -> None:
    # Set for the current ID we are investigating
    cur_id = csv_path.stem.removesuffix("_s_motion")
    
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

    # Dtypes:
    df_gps["SPEED_MS"] = df_gps["SPEED_MS"].astype("float64") 
    df_gps["ACC_MS2"] = df_gps["ACC_MS2"].astype("float64") 
    df_gps["JERK_MS3"] = df_gps["JERK_MS3"].astype("float64") 
    
    df_gps["BEARING_RATE_DEG"] = df_gps["BEARING_RATE_DEG"].astype("float64") 
    df_gps["BEARING_RATE_DEG_WGS84"] = df_gps["BEARING_RATE_DEG_WGS84"].astype("float64") 
 
    df_gps["BEARING_DEG"] = df_gps["BEARING_DEG"].astype("float64") 
    df_gps["BEARING_DEG_WGS84"] = df_gps["BEARING_DEG_WGS84"].astype("float64") 
    
    df_gps["BEARING_CHANGE_DEG"] = df_gps["BEARING_CHANGE_DEG"].astype("float64") 
    df_gps["BEARING_CHANGE_DEG_WGS84"] = df_gps["BEARING_CHANGE_DEG_WGS84"].astype("float64") 

    trip_stats = []
    #for trip_ID in df_gps["KEY"].dropna().unique():
    for trip_ID, df_trip in df_gps.groupby("KEY", sort=False, observed=True):
        if pd.isna(trip_ID):
            print(f"Warning: {trip_ID} is NaN.") 
            continue

        # Fast dictionary lookup - using already pre-generate stats csv (it's a bit of a shortcut)
        trip_info = trip_lookup.get(trip_ID, {})
        main_mode = trip_info.get('MAIN_MODE', pd.NA)
        multimodal = trip_info.get('MULTI_MODAL', pd.NA)

        # Isolate the trip:
        #df_trip = df_gps[df_gps["KEY"] == trip_ID]
        
        row = ({
            "ID": cur_id,
            "FILE_NAME": csv_path.name,
            "TRIP_KEY": trip_ID,
            "MAIN_MODE": main_mode,
            "MULTI_MODAL": multimodal,

        })
        
        for col, prefix in FEATURES.items():
            _add_quantiles(row, df_trip, col, prefix)
            
        trip_stats.append(row)
        
    # OUTPUT to Pyarrow table
    #df_stats = pd.DataFrame(trip_stats)
    table = pa.Table.from_pylist(trip_stats)
    
    # Swap to parquet:
    PATH_OUTPUT = OUT_DIR / f"percentile_{csv_path.stem}.parquet"
    #df_stats.to_parquet(PATH_OUTPUT, index=False, engine="pyarrow", compression="snappy")
    pa.parquet.write_table(table, PATH_OUTPUT, compression='snappy')
   
    
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
        #for p in csvs:
        for p in tqdm(csvs, desc="Processing", unit="File"):
            generate(p)
            
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            #list(ex.map(process_one, csvs))
            list(tqdm(ex.map(generate, csvs), total=len(csvs), desc="Processing", unit="file"))
    
    # Merge all stats into one parquet/csv:
    stats_parts = list(OUT_DIR.glob("percentile_*.parquet"))
    if stats_parts:

        df_all = (pd.concat((pd.read_parquet(p, engine="pyarrow") for p in stats_parts), ignore_index=True).drop_duplicates())
        df_all.to_parquet(OUT_DIR / "percentiles_071125.parquet", index=False, engine="pyarrow", compression="snappy")
        
        # Convert to CSV for easier readability :)
        df_all.to_csv(OUT_DIR / "percentiles_071125.csv", index=False)
        
        for p in stats_parts:
            p.unlink(missing_ok=True)
            
if __name__ == "__main__":
    print("Using: ", os.cpu_count(), " CPUs...")
    main(limit=None, workers=os.cpu_count())