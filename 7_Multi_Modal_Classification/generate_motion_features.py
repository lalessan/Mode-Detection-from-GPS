import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from tqdm import tqdm

import ast

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--segments", type=str, default="/home/s232534/scripts/segmentation_pelt/trip_segment_boundaries_pelt.csv")
parser.add_argument("--out-stem", type=str, default=None, help="Base filename (no extension) for outputs",)
args, _ = parser.parse_known_args()

SEGMENTS_PATH = Path(args.segments)
out_stem = args.out_stem or f"features_motion_segmented__{SEGMENTS_PATH.stem}"

# Paths
GPS_DIR = Path("/home/s232534/data/split_motion_features_061125") # NOTE: confirm we are ingesting the latest enriched GPS
#MISSING_PATH = Path("/home/s232534/data/scripts/missing_trips.csv")
TRIPS_PATH = Path("/home/lauale/netmob2025/data/trips_dataset.csv")
OUT_DIR = Path("/home/s232534/scripts/segmentation_pelt/multimodal_classification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load dfs ---
df_trips = pd.read_csv(TRIPS_PATH)
#df_segments = pd.read_csv("/home/s232534/scripts/segmentation_pelt/trip_segment_boundaries_pelt.csv")

df_segments = pd.read_csv(SEGMENTS_PATH)
df_segments["SEGMENTS_BOUNDS"] = df_segments["SEGMENTS_BOUNDS"].apply(ast.literal_eval) #evaluate as list[int]
#df_missing = pd.read_csv(MISSING_PATH)


def generate(csv_path: Path) -> None:
    # Set for the current ID we are investigating
    #cur_id = csv_path.stem
    cur_id = csv_path.stem.removesuffix("_s_motion")
    df_trips_inside = df_trips[df_trips['ID'] == cur_id].copy()
    
    try:
        #df_gps = pd.read_csv(csv_path)
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
    
    # Set correct df dtypes:
    df_gps['UTC DATETIME'] = pd.to_datetime(df_gps['UTC DATETIME'], errors='coerce')
    df_gps['LOCAL DATETIME'] = pd.to_datetime(df_gps['LOCAL DATETIME'], errors='coerce')
    
    df_trips_inside['Date_O'] = pd.to_datetime(df_trips_inside['Date_O'], errors='coerce')
    df_trips_inside['Date_D'] = pd.to_datetime(df_trips_inside['Date_D'], errors='coerce')
    df_trips_inside['Date_EMG'] = pd.to_datetime(df_trips_inside['Date_EMG'], errors='coerce')
    
    df_gps["LATITUDE"] = df_gps["LATITUDE"].astype("float64") 
    df_gps["LONGITUDE"] = df_gps["LONGITUDE"].astype("float64") 
    
    df_gps["SPEED"] = df_gps["SPEED"].astype("float64") 
    df_gps["SPEED_MS"] = df_gps["SPEED_MS"].astype("float64") 
    df_gps["ACC_MS2"] = df_gps["ACC_MS2"].astype("float64") 
    df_gps["JERK_MS3"] = df_gps["JERK_MS3"].astype("float64") 
    
    df_gps["BEARING_DEG"] = df_gps["BEARING_DEG"].astype("float64") 
    df_gps["BEARING_DEG_WGS84"] = df_gps["BEARING_DEG_WGS84"].astype("float64") 
    
    df_gps["BEARING_RATE_DEG"] = df_gps["BEARING_RATE_DEG"].astype("float64") 
    df_gps["BEARING_RATE_DEG_WGS84"] = df_gps["BEARING_RATE_DEG_WGS84"].astype("float64") 
 
    df_gps["BEARING_CHANGE_DEG"] = df_gps["BEARING_CHANGE_DEG"].astype("float64") 
    df_gps["BEARING_CHANGE_DEG_WGS84"] = df_gps["BEARING_CHANGE_DEG_WGS84"].astype("float64") 
 
    # df_gps["TURNING_RATE_DEG_S"] = df_gps["TURNING_RATE_DEG_S"].astype("float64") 
    # df_gps["TURNING_RATE_DEG_S_WGS84"] = df_gps["TURNING_RATE_DEG_S_WGS84"].astype("float64") 
    
    
    df_gps["DIST_BP"] = df_gps["DIST_BP"].astype("float64") 
    df_gps["DIST_BP_WGS84"] = df_gps["DIST_BP_WGS84"].astype("float64") 
    
    df_gps["KEY"] = df_gps["KEY"].astype("string") 
    
    
    # Safeties
    def _safe_series(x: pd.Series) -> pd.Series:
        return pd.to_numeric(x, errors="coerce")

    def smean(x) -> float:
        s = _safe_series(x)
        
        if not s.empty and not s.isnull().all():
            return float(s.mean())
        else:
            #print(trip_ID, " is empty/NaN and cannot compute mean.")
            return np.nan
            
    def smedian(x) -> float:
        s = _safe_series(x)
        
        if not s.empty and not s.isnull().all():
            return float(s.median())
        else:
            #print(trip_ID, " is empty/NaN and cannot compute.")
            return np.nan

    def sstd(x) -> float:
        s = _safe_series(x)
        
        if not s.empty and not s.isnull().all():
            return float(s.std())
        else:
            #print(trip_ID, " is empty/NaN and cannot compute.")
            return np.nan

    def squantile(x, q: float) -> float:
        s = _safe_series(x)
        
        if not s.empty and not s.isnull().all():
            return float(s.quantile(q))
        else:
            #print(trip_ID, " is empty/NaN and cannot compute.")
            return np.nan
    
    def _total_time(t1, t2):
            delta = t2 - t1
            return delta.total_seconds()
        
    trip_stats = []
    trip_empty = []
    #for trip_ID in df_gps["KEY"].dropna().unique():
    for trip_ID in np.unique(df_gps['KEY']):
    #for trip_ID, df_trip in df_gps.groupby("KEY", sort=False, observed=True):
        if pd.isna(trip_ID):
            print(f"Warning: {trip_ID} is NaN.") 
            continue
        
        # Grab main mode and check for empties
        mm_series = df_trips_inside.loc[df_trips_inside['KEY'] == trip_ID, 'Main_Mode']
        main_mode = mm_series.astype('string').iloc[0] if not mm_series.empty else pd.NA
        
        # TODO: bit scuffed and repetative code... 
        trip_row = df_trips_inside.loc[df_trips_inside['KEY'] == trip_ID].head(1)  # expect 1 row

        if trip_row.empty:
            # Check for empties
            main_mode = pd.NA
            multimodal = pd.NA
        else:
            main_mode = trip_row["Main_Mode"].astype("string").iloc[0]

        extra_modes = trip_row[["Mode_2","Mode_3","Mode_4","Mode_5"]].notna().any(axis=1).iloc[0]

        if (trip_row["Main_Mode"].iloc[0] == trip_row["Mode_1"].iloc[0]) and not extra_modes:
            multimodal = False
        else:
            multimodal = True
        
        # We split again now into the segments per each trip
        previous_point = 0
        df_segments_trip = df_segments[df_segments['KEY'] == trip_ID]
        segment_boundaries = df_segments_trip['SEGMENTS_BOUNDS'].iloc[0]
        for i, point in enumerate(segment_boundaries):
            # Isolate the trip:
            df_trip = df_gps[df_gps["KEY"] == trip_ID]
            
            df_trip_segment = df_trip[previous_point:point] # <-This is the segment of a trip
            previous_point = point
            
            # Build empty trip report
            s_sp = _safe_series(df_trip_segment["SPEED_MS"])
            s_ac = _safe_series(df_trip_segment["ACC_MS2"])
            if (s_sp.empty or s_sp.isnull().all()) or (s_ac.empty or s_ac.isnull().all()):
                trip_empty.append(trip_ID)
            
            trip_stats.append({
                "ID": cur_id,
                "FILE_NAME": csv_path.name,
                "TRIP_KEY": trip_ID,
                "SEGMENT": i,
                "MAIN_MODE": main_mode,
                "MULTI_MODAL": multimodal,
                "TOTAL_TIME": _total_time(df_trip_segment['timestamp'].min(), df_trip_segment['timestamp'].max()),

                # "AVG_PT_DIST": smean(df_trip["DIST_BP"]),
                # "MED_PT_DIST": smedian(df_trip["DIST_BP"]),
                "AVG_PT_DIST_WGS84": smean(df_trip_segment["DIST_BP_WGS84"]),
                "MED_PT_DIST_WGS84": smedian(df_trip_segment["DIST_BP_WGS84"]),
                # "DIST_TOTAL": df_trip["DIST_BP"].sum(min_count=1),
                "DIST_TOTAL_WGS84": df_trip_segment["DIST_BP_WGS84"].sum(min_count=1),
                "GPS_POINTS": len(df_trip_segment),

                "SPEED_AVG": smean(df_trip_segment["SPEED_MS"]),
                "SPEED_MED": smedian(df_trip_segment["SPEED_MS"]),
                "SPEED_85P": squantile(df_trip_segment["SPEED_MS"], 0.85),
                "SPEED_STD": sstd(df_trip_segment["SPEED_MS"]),

                "ACC_AVG": smean(df_trip_segment["ACC_MS2"]),
                "ACC_MED": smedian(df_trip_segment["ACC_MS2"]),
                "ACC_85P": squantile(df_trip_segment["ACC_MS2"], 0.85),
                "ACC_STD": sstd(df_trip_segment["ACC_MS2"]),

                "JERK_AVG": smean(df_trip_segment["JERK_MS3"]),
                "JERK_MED": smedian(df_trip_segment["JERK_MS3"]),
                "JERK_85P": squantile(df_trip_segment["JERK_MS3"], 0.85),
                "JERK_STD": sstd(df_trip_segment["JERK_MS3"]),

                # "B_AVG": smean(df_trip["BEARING_DEG"]),
                # "B_MED": smedian(df_trip["BEARING_DEG"]),
                # "B_85P": squantile(df_trip["BEARING_DEG"], 0.85),
                # "B_STD": sstd(df_trip["BEARING_DEG"]),
                
                "B_AVG_WGS84": smean(df_trip_segment["BEARING_DEG_WGS84"]),
                "B_MED_WGS84": smedian(df_trip_segment["BEARING_DEG_WGS84"]),
                "B_85P_WGS84": squantile(df_trip_segment["BEARING_DEG_WGS84"], 0.85),
                "B_STD_WGS84": sstd(df_trip_segment["BEARING_DEG_WGS84"]),

                # "BC_AVG": smean(df_trip["BEARING_CHANGE_DEG"]),
                # "BC_MED": smedian(df_trip["BEARING_CHANGE_DEG"]),
                # "BC_85P": squantile(df_trip["BEARING_CHANGE_DEG"], 0.85),
                # "BC_STD": sstd(df_trip["BEARING_CHANGE_DEG"]),
                
                "BC_AVG_WGS84": smean(df_trip_segment["BEARING_CHANGE_DEG_WGS84"]),
                "BC_MED_WGS84": smedian(df_trip_segment["BEARING_CHANGE_DEG_WGS84"]),
                "BC_85P_WGS84": squantile(df_trip_segment["BEARING_CHANGE_DEG_WGS84"], 0.85),
                "BC_STD_WGS84": sstd(df_trip_segment["BEARING_CHANGE_DEG_WGS84"]),
                
                # "BR_AVG": smean(df_trip["BEARING_RATE_DEG"]),
                # "BR_MED": smedian(df_trip["BEARING_RATE_DEG"]),
                # "BR_85P": squantile(df_trip["BEARING_RATE_DEG"], 0.85),
                # "BR_STD": sstd(df_trip["BEARING_RATE_DEG"]),

                "BR_AVG_WGS84": smean(df_trip_segment["BEARING_RATE_DEG_WGS84"]),
                "BR_MED_WGS84": smedian(df_trip_segment["BEARING_RATE_DEG_WGS84"]),
                "BR_85P_WGS84": squantile(df_trip_segment["BEARING_RATE_DEG_WGS84"], 0.85),
                "BR_STD_WGS84": sstd(df_trip_segment["BEARING_RATE_DEG_WGS84"]),
            })
        
    # OUTPUT
    return pd.DataFrame(trip_stats)
    
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
    
    #df_all = pd.concat([r for r in results if r and not r.empty], ignore_index=True)
    df_all = pd.concat([r for r in results if r is not None and not r.empty], ignore_index=True)
    #df_empty = pd.concat([r[1] for r in results if r and not r[1].empty], ignore_index=True)
    
    df_all.to_parquet(OUT_DIR / out_stem, index=False, engine="pyarrow", compression="snappy")
    #df_all.to_csv(OUT_DIR / out_stem, index=False)
    
    #df_empty.to_csv(OUT_DIR / "empty_trips_071125.csv", index=False)
    
    
if __name__ == "__main__":
    main(limit=None, workers=os.cpu_count() or 4)