import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import ast
import pandas as pd
import pyarrow as pa
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--segments", type=str, default="/home/s232534/scripts/segmentation_pelt/trip_segment_boundaries_pelt.csv")
parser.add_argument("--out-stem", type=str, default=None, help="Base filename for outputs",)
args, _ = parser.parse_known_args()

SEGMENTS_PATH = Path(args.segments)
out_stem = args.out_stem or f"percentile_map_segmented__{SEGMENTS_PATH.stem}"

GPS_DIR = Path("/home/s232534/data/split_motion_features_061125")
DF_STATS_PATH = Path("/home/s232534/data/stats_071125/statistical_summary_071125.csv")

OUT_DIR = Path("/home/s232534/scripts/segmentation_pelt/multimodal_classification/Features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_stats_pregen = pd.read_csv(DF_STATS_PATH)

# Segments addition
#df_segments = pd.read_csv("/home/s232534/scripts/segmentation_pelt/trip_segment_boundaries_pelt.csv")
df_segments = pd.read_csv(SEGMENTS_PATH)
df_segments["SEGMENTS_BOUNDS"] = df_segments["SEGMENTS_BOUNDS"].apply(ast.literal_eval) #evaluate as list[int]

# Create lookup dictionaries once at module level
trip_lookup = df_stats_pregen.set_index('TRIP_KEY')[['MAIN_MODE', 'MULTI_MODAL']].to_dict('index')

# Verbalised list of the quantiles we want to test for. 
QUANTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  

FEATURES = {
    "SPEED_MS": "SPEED",
    "ACC_MS2": "ACC",
    "JERK_MS3": "JERK",
    "BEARING_CHANGE_DEG_WGS84": "BC_WGS84",
    "BEARING_RATE_DEG_WGS84": "BR_WGS84",
}

# Helper function
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
    for trip_ID, df_trip in df_gps.groupby("KEY", sort=False, observed=True):
        if pd.isna(trip_ID):
            print(f"Warning: {trip_ID} is NaN.") 
            continue
            
        # Fast dictionary lookup - using already pre-generate stats csv (it's a bit of a shortcut)
        trip_info = trip_lookup.get(trip_ID, {})
        main_mode = trip_info.get('MAIN_MODE', pd.NA)
        multimodal = trip_info.get('MULTI_MODAL', pd.NA)
        
        previous_point = 0
        df_segments_trip = df_segments[df_segments['KEY'] == trip_ID]
        segment_boundaries = df_segments_trip['SEGMENTS_BOUNDS'].iloc[0]
        for i, point in enumerate(segment_boundaries):
            df_trip_segment = df_trip[previous_point:point]
            previous_point = point
            
            
            row = ({
                "ID": cur_id,
                "FILE_NAME": csv_path.name,
                "TRIP_KEY": trip_ID,
                "SEGMENT": i,
                "MAIN_MODE": main_mode,
                "MULTI_MODAL": multimodal,
            })
            
            for col, prefix in FEATURES.items():
                _add_quantiles(row, df_trip_segment, col, prefix)
                
            trip_stats.append(row)
        
    # OUTPUT to Pyarrow table
    return pa.Table.from_pylist(trip_stats)
   
    
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
    
    #df_all = pd.concat([r for r in results if r is not None and not r.empty], ignore_index=True)
    # df_all = pd.concat([r for r in results if r is not None and not r.empty], ignore_index=True)
    # df_all.to_parquet(OUT_DIR / "features_motion_segmented.parquet", index=False, engine="pyarrow", compression="snappy")
    # df_all.to_csv(OUT_DIR / "features_motion_segmented.csv", index=False)    
    
    tables = [t for t in results if isinstance(t, pa.Table) and t.num_rows > 0]
    if not tables:
        print("No tables produced.")
        return

    combined = pa.concat_tables(tables) 
    pq.write_table(combined, OUT_DIR / out_stem, compression="snappy")
            
if __name__ == "__main__":
    print("Using: ", os.cpu_count(), " CPUs...")
    main(limit=None, workers=os.cpu_count())