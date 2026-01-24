import sys
sys.path.insert(0, "/home/s232534/scripts")

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd
import geopandas as gpd
import numpy as np
import ast
from tqdm import tqdm

from map_matching import routes
from typing import Dict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--segments", type=str, default="/home/s232534/scripts/segmentation_pelt/trip_segment_boundaries_pelt.csv")
parser.add_argument("--out-stem", type=str, default=None, help="Base filename for outputs",)
args, _ = parser.parse_known_args()

SEGMENTS_PATH = Path(args.segments)
out_stem = args.out_stem or f"features_route_match_segmented__{SEGMENTS_PATH.stem}"

GPS_DIR = Path("/home/s232534/data/split_motion_features_061125")
STATS_PATH = Path("/home/s232534/data/stats_071125/statistical_summary_071125.csv")

OUT_DIR = Path("/home/s232534/scripts/segmentation_pelt/multimodal_classification/Features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the heavy stuff:
print("Loading route parquet files...")
bus_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/bus_routes.parquet")
tram_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/tram_routes.parquet")
train_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/train_routes.parquet")
subway_routes = gpd.read_parquet("/home/s232534/scripts/map_matching/osm_routes/subway_routes.parquet")

print("Loading segment boundaries...")
df_segments = pd.read_csv(SEGMENTS_PATH)
df_segments["SEGMENTS_BOUNDS"] = df_segments["SEGMENTS_BOUNDS"].apply(ast.literal_eval)

# Interpolation: Can be commented out for performance reasons.
# Since the point spacing in line geometries is very uneven, this means at times the fast_match might not match a GPS to a route, 
# even though the linegeometry would be crossing its path
#TODO: To avoid recomputing if the GPS and OSM data is not changing we can simply save out the densified geometry
# bus_routes = routes.add_densified_geometry(bus_routes, spacing_m=10, overwrite_geom=True)
# tram_routes = routes.add_densified_geometry(tram_routes, spacing_m=10, overwrite_geom=True)
# train_routes = routes.add_densified_geometry(train_routes, spacing_m=10, overwrite_geom=True)
# subway_routes = routes.add_densified_geometry(subway_routes, spacing_m=10, overwrite_geom=True)
print("Multi-threading: Interpolating routes...")

items = [
    ("bus", bus_routes),
    ("tram", tram_routes),
    ("train", train_routes),
    ("subway", subway_routes),
]

def densify(item):
    name, gdf = item
    return name, routes.add_densified_geometry(gdf, spacing_m=10, overwrite_geom=True)

with ThreadPoolExecutor(max_workers=4) as ex:
    out = dict(ex.map(densify, items))

bus_routes    = out["bus"]
tram_routes   = out["tram"]
train_routes  = out["train"]
subway_routes = out["subway"]

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

def _return_top_percentage(routes_gpd: tuple[np.array, np.array, np.array, np.array], balltreesobjs, gps_trace: pd.DataFrame, search_radius: int = 20) -> Dict:
    top_pct_dict: dict = {}
    
    (bus_arr, tram_arr, train_arr, subway_arr) = routes_gpd
    (BALLTREE_BUS, BALLTREE_TRAM, BALLTREE_TRAIN, BALLTREE_SUBWAY) = balltreesobjs
    
    matched_bus: Dict = routes.fast_match(bus_arr, gps_trace, BALLTREE_BUS, search_radius)
    matched_tram: Dict = routes.fast_match(tram_arr, gps_trace, BALLTREE_TRAM, search_radius)
    matched_train: Dict = routes.fast_match(train_arr, gps_trace, BALLTREE_TRAIN, search_radius)
    matched_subway: Dict = routes.fast_match(subway_arr, gps_trace, BALLTREE_SUBWAY, search_radius)
    
    # TODO: repeat code; turn into a loop if time.
    if not matched_bus:
        top_pct_dict['bus'] = np.nan
    else:
        top_pct_dict['bus'] = max(matched_bus.values(), key=lambda v: v[1])[1]
        
    if not matched_tram:
        top_pct_dict['tram'] = np.nan
    else:
        top_pct_dict['tram'] = max(matched_tram.values(), key=lambda v: v[1])[1]
        
    if not matched_train:
        top_pct_dict['train'] = np.nan
    else:
        top_pct_dict['train'] = max(matched_train.values(), key=lambda v: v[1])[1]
        
    if not matched_subway:
        top_pct_dict['subway'] = np.nan
    else:
        top_pct_dict['subway'] = max(matched_subway.values(), key=lambda v: v[1])[1]
        
    return top_pct_dict
    

def generate(csv_path: Path) -> None:
    # Imports per worker
    import pandas as pd
    from map_matching import routes
    
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
        
        previous_point = 0
        df_segments_trip = df_segments[df_segments['KEY'] == key]
        segment_boundaries = df_segments_trip['SEGMENTS_BOUNDS'].iloc[0]
        df_lim = df_gps[df_gps['KEY'] == key]
        
        for i, point in enumerate(segment_boundaries):
            df_trip_segment = df_lim[previous_point:point] # <-This is the segment of a trip
            previous_point = point
            top_percentage_dict = _return_top_percentage((bus_arr, tram_arr, train_arr, subway_arr), (BALLTREE_BUS, BALLTREE_TRAM, BALLTREE_TRAIN, BALLTREE_SUBWAY), df_trip_segment, 20)
            
            rows.append(
            {
                "KEY": key,   
                "SEGMENT": i,         
                "bus_match_pct": top_percentage_dict['bus'],      
                "tram_match_pct": top_percentage_dict['tram'],      
                "train_match_pct": top_percentage_dict['train'],      
                "subway_match_pct": top_percentage_dict['subway'],      
                "MAIN_MODE": df_stats_lim['MAIN_MODE'].values[0],   
                "MULTI_MODAL": df_stats_lim['MULTI_MODAL'].values[0],  
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
    #df_all.to_csv(OUT_DIR / "features.csv", index=False)
    df_all.to_parquet(OUT_DIR / out_stem, index=False, engine="pyarrow", compression="snappy")
    
    
if __name__ == "__main__":
    main(limit=None, workers=os.cpu_count() or 4)