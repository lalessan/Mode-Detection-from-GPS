import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent)) # So that we can load the GTFSFUNCTIONS folder

import geopandas as gpd
import pyarrow as pa
import re 
import pandas as pd
from shapely.geometry import Polygon
from pyproj import Geod
import numpy as np
from sklearn.neighbors import BallTree
#from gtfs_functions 
import load_gtfs

def _geodesic_point_buffer(lat: float, lon: float, geod: Geod, radius_m: float, n_points: int = 15):
    """_summary_

    Args:
        lat (float): coordinates
        lon (float): coordinates
        geod (Geod): Projection object set by pyproj library.
        radius_m (float): Radius of the buffer zone.
        n_points (int, optional): How many points comprise the circle buffer. Defaults to 15.

    Returns:
        shapely.geometry.Polygon: A Polygon sphere of set point resolution around the provided Lat/Lon coords.
    """
    
    # Create a geodesic circle around (lat, lon) - n_points is the resolution of the circle
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    lons, lats, _ = geod.fwd(np.full_like(azimuths, lon),
                             np.full_like(azimuths, lat),
                             azimuths,
                             np.full_like(azimuths, radius_m))
    return Polygon(zip(lons, lats))

def generate_polygon_buffers(stops_gdf: gpd.GeoDataFrame, geod: Geod, buffer_size: int = 1000, buffer_resolution: int = 15) -> None:
    """_summary_

    Args:
        stops_gdf (gpd.GeoDataFrame): Our dataframe containing our stop locations.
        geod (Geod): Projection object set by pyproj library.
        buffer_size (int, optional): Radius of the buffer zone. Defaults to 1000.
        buffer_resolution (int, optional): How many points comprise the circle buffer. Defaults to 15.
    """
    # For memory efficiency instead of copying and returning a new df we append the provided gdf with a new col for polygons
    stops_gdf["buffer_geom"] = stops_gdf.apply(lambda r: _geodesic_point_buffer(r.stop_lat, r.stop_lon, geod, buffer_size, buffer_resolution), axis=1)   
    stops_gdf = stops_gdf.set_geometry("buffer_geom")

def _dist_to_stop(row, geod: Geod):
    if pd.isna(row.get("stop_lon")):
        return np.nan
    
    _, _, d = geod.inv(row.geometry.x, row.geometry.y, row.stop_lon, row.stop_lat) # <- Geod.inv expects lon, lat ordering! There is also another function that performs this cross-check
    return d

def calculate_stop_distance_sjoin(stops_gdf: gpd.GeoDataFrame, gps_trace_df: gpd.GeoDataFrame, geod: Geod, keep_one: bool = False):
    """Appends our GPS trace dataframe with distances to the nearest stops based on defined circular buffer zones. Uses GeoPandas sjoin
    which is a Spatial Index method (efficient) -> https://geopandas.org/en/stable/docs/user_guide/spatial_indexing.html
    
    Note: We append instead of returning a new dataframe for memory efficiency which is required when processing large amounts of traces.

    Args:
        stops_gdf (gpd.GeoDataFrame): _description_
        gps_trace_df (gpd.GeoDataFrame): _description_
        geod (Geod): _description_
        keep_one (bool, optional): _description_. Defaults to False.
        
    Returns:
        If keep_one = False then: we return the hits df with all hits.
        else: it just appeands the gps_trace gdf with the nearest single stop.
    """
    # Spatial join: which GPS points fall inside which stop buffers?
    hits = gpd.sjoin(
        gps_trace_df[["geometry"]],           # left = GPS points
        stops_gdf[["buffer_geom", "stop_lon", "stop_lat", "stop_id"]].set_geometry("buffer_geom").set_crs(4326, allow_override=True),
        how="left",
        predicate="within",
    )
    
    hits["dist_m"] = hits.apply(lambda row: _dist_to_stop(row, geod), axis=1)
    
    # Change dtype due to PyArrow mismatch
    # hits["dist_m"]  = hits["dist_m"].astype("float32")
    # hits["stop_lon"] = hits["stop_lon"].astype("float32")
    # hits["stop_lat"] = hits["stop_lat"].astype("float32")
    # hits["stop_id"]  = hits["stop_id"].astype("category")
    
    if(keep_one):
        nearest = (
            hits.sort_values("dist_m")
            .groupby(level=0, as_index=True)
            .first()
        )
        # This only keeps the first nearest point (simple and might be enough.)
        gps_trace_df["nearest_stop_lon"]   = nearest["stop_lon"]
        gps_trace_df["nearest_stop_lat"]   = nearest["stop_lat"]
        gps_trace_df["nearest_stop_dist_m"] = nearest["dist_m"]
        gps_trace_df["nearest_stop_id"] = nearest["stop_id"]
        #return gps_trace_df
    
    else:
        # This way we can keep all hits to the GPS point.
        # agg = hits.groupby(level=0).agg({
        #     "stop_lon": list, "stop_lat": list, "stop_id": list, "dist_m": list
        # }).rename(columns={
        #     "stop_lon": "stop_lons",
        #     "stop_lat": "stop_lats",
        #     "stop_id": "stop_ids",
        #     "dist_m": "stop_dists_m",
        # })
        # gps_trace_df.join(agg)
        
        # agg = (hits.groupby(level=0)
        #   .agg(stop_lons=("stop_lon", list),
        #        stop_lats=("stop_lat", list),
        #        stop_ids =("stop_id",  list),
        #        stop_dists_m=("dist_m", list)))

        # # convert to Arrow-backed list columns
        # def to_list(s, pa_type):
        #     return pd.array(s, dtype=pd.ArrowDtype(pa.list_(pa_type)))

        # agg["stop_lons"]     = to_list(agg["stop_lons"], pa.float32())
        # agg["stop_lats"]     = to_list(agg["stop_lats"], pa.float32())
        # agg["stop_dists_m"]  = to_list(agg["stop_dists_m"], pa.float32())
        # agg["stop_ids"]      = to_list(agg["stop_ids"], pa.string()) 

        # gps_trace_df = gps_trace_df.join(agg)
        #return gps_trace_df.join(agg)
        
        return hits
    
# --------------- BallTree methods (faster) ---------
def generate_balltree(gdf_stops: gpd.GeoDataFrame):
    # Grab our stop lat/lon values and convert them from deg -> rad.
    stops_latlon = np.column_stack([gdf_stops["stop_lat"].values, gdf_stops["stop_lon"].values])
    stops_rad = np.radians(stops_latlon)
    
    # Construct the balltree to be used to queery
    tree = BallTree(stops_rad, metric="haversine")
    
    return tree

# https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b/
def calculate_stop_distance_balltree(tree: BallTree, gdf_gps_trace: gpd.GeoDataFrame, search_radius: int=1000) -> tuple[list, list]:
    EARTH_RADIUS_M = 6371000.0
    
    # GPS trace deg -> rad
    gps_lat = gdf_gps_trace.geometry.y.values
    gps_lon = gdf_gps_trace.geometry.x.values
    gps_rad = np.radians(np.column_stack([gps_lat, gps_lon]))
    
    # Queery the tree
    rad = search_radius / EARTH_RADIUS_M 
    idxs_list, dists_list = tree.query_radius(gps_rad, r=rad, return_distance=True, sort_results=True)
    
    # Scale radians to meters.
    for dists in dists_list:  
        dists *= EARTH_RADIUS_M
    
    return idxs_list, dists_list

def generate_GTFS_dict(balltree: bool = True) -> dict:
    # Generate all GTFS objects:
    zip_paths = load_gtfs.list_gtfs_zip_paths(keyword="PMBT")
    
    # Build a dictionary that is 
    dict_gtfs_balltree: dict = {}
    BALLTREE = None
    
    for zip_path in zip_paths:
        GTFS_DATA = load_gtfs.get_store(zip_path)
        STOPS: gpd.GeoDataFrame = load_gtfs.generate_stop_modality_parented(GTFS_DATA)
        if (balltree):
            BALLTREE = generate_balltree(STOPS)
        
        # Change zip_path as key to zip_name as a key!
        zip_name = Path(zip_path).name
        dict_gtfs_balltree.update({
            f'{zip_name}': {
                'gtfs_store': GTFS_DATA,
                'stops_gdf': STOPS,
                'balltree': BALLTREE
            }  
        })
        
    return dict_gtfs_balltree

# ------------ SAVING into Parquet for later access and filtering.
# This requires loading the correct GTFS data for the correct times of the GPS
        
def feature_nearest_stops_to_parquet(gdf_gps: gpd.GeoDataFrame, lookup: load_gtfs.DateRangeLookup, method: str = "balltree"):
    
    zip_paths = load_gtfs.return_zip_paths(gdf_gps, lookup)
    
    if (method == "sjoin"):
        geod = Geod(ellps="WGS84")
    
    #TODO: Huge fix needed as doing it per zippath will continually do it multiple times for the same GPS trace.
    #      This needs to look at the date ranges of the GPS trace itself.
    for zip_path in zip_paths:
        GTFS_DATA = load_gtfs.get_store(zip_path)
        
        stops_modality_parented: gpd.GeoDataFrame = load_gtfs.generate_stop_modality_parented(GTFS_DATA)
        #stops_modality_parented = stops_modality_parented.reset_index(drop=True)
        

        if (method == "balltree"):
            BALLTREE = generate_balltree(stops_modality_parented)
            
            index_arrays, dist_arrays_m = calculate_stop_distance_balltree(BALLTREE ,gdf_gps)
            
            # Convert the indices to actual stop IDs
            stop_ids = []
            for stop_idx_array in index_arrays:
                #stop_ids.append(stops_modality_parented['stop_id'][stop_idx_array]) # <- the stops
                stop_id_values = stops_modality_parented['stop_id'].astype(str).to_numpy() 
                stop_ids.append(stop_id_values[stop_idx_array].tolist())
                
            temp_df = pd.DataFrame({
                "gps_key":    gdf_gps["KEY"].astype(str).to_numpy(),  
                "stop_ids":   stop_ids,                           
                "distances":  dist_arrays_m,                          
            })
            
            schema = pa.schema([
                pa.field("gps_key", pa.string()),
                pa.field("stop_ids", pa.list_(pa.string())),
                pa.field("distances", pa.list_(pa.float32())),
            ])

            filename = str(gdf_gps["KEY"].iloc[0]).split("-", 1)[0] #takes the very first key and splits out the GPS code/filename.
  
            table = pa.Table.from_pandas(temp_df, schema=schema, preserve_index=False)
            pa.parquet.write_table(table, f"{filename}_balltree_nearest_stops.parquet")
        
        elif (method == "sjoin"):
            generate_polygon_buffers(stops_modality_parented, geod, 1000, 15) # TODO: this method should return a df copy.
            
            hits = calculate_stop_distance_sjoin(stops_modality_parented, gdf_gps, geod)

            nearest_hits = (hits.sort_values("dist_m").groupby(level=0, as_index=True).first())
        
def feature_nearest_stops_to_parquet_v2(gdf_gps: gpd.GeoDataFrame, dict_precalc: dict, lookup: load_gtfs.DateRangeLookup, method: str = "balltree"):
    
    zip_paths = load_gtfs.return_zip_paths(gdf_gps, lookup)
    
    #TODO: Huge fix needed as doing it per zippath will continually do it multiple times for the same GPS trace.
    #      This needs to look at the date ranges of the GPS trace itself.
    for zip_path in zip_paths:
        print(zip_path)
        dict_precalc_zip = dict_precalc[zip_path]
        
        GTFS_DATA = dict_precalc_zip['gtfs_store']
        STOPS = dict_precalc_zip['stops_gdf']
        BALLTREE = dict_precalc_zip['balltree']

        if (method == "balltree"):
            index_arrays, dist_arrays_m = calculate_stop_distance_balltree(BALLTREE ,gdf_gps)
            
            index_arrays = [np.asarray(a, dtype=np.int64) for a in index_arrays]
            stop_id_values = STOPS['stop_id'].astype(str).to_numpy()

            # 3) Flatten Gather split
            lengths = np.fromiter((len(a) for a in index_arrays), dtype=np.int64)
            if lengths.sum() == 0:
                stop_ids = [[] for _ in index_arrays]
            else:
                big_idx  = np.concatenate(index_arrays) # flat indices, shape (total_hits,)
                gathered = stop_id_values[big_idx] # vectorised lookup -> shape (total_hits,)
                splits   = lengths.cumsum()[:-1]  # split points
                stop_ids = [arr.tolist() for arr in np.split(gathered, splits)]
                
            temp_df = pd.DataFrame({
                "gps_key":    gdf_gps["KEY"].astype(str).to_numpy(),  
                "stop_ids":   stop_ids,                           
                "distances":  dist_arrays_m,      
            })
            
            schema = pa.schema([
                pa.field("gps_key", pa.string()),
                pa.field("stop_ids", pa.list_(pa.string())),
                pa.field("distances", pa.list_(pa.float32())),
            ])

            filename = str(gdf_gps["KEY"].iloc[0]).split("-", 1)[0] #takes the very first key and splits out the GPS code/filename.
  
            table = pa.Table.from_pandas(temp_df, schema=schema, preserve_index=False)
            pa.parquet.write_table(table, f"{filename}_balltree_nearest_stops2.parquet")


def parse_file_id_from_key(key: str) -> str:
    """Robustly extract 'GPS file/person id' from KEY col.
    Examples: '42_0197-monday-44851-1'  -> '42_0197'
                   '10_3077_s_acc-...'       -> '10_3077_s_acc'
    Strategy: take everything up to first '-' if present, else whole string.

    Args:
        key (str): _description_

    Returns:
        str: _description_
    """
    
    if "-" in key:
        return key.split("-", 1)[0]
    return key

def parse_snap_id_from_zip(zip_path: str) -> int:
    """Return compact validity range as YYYYMMDDYYYYMMDD from filenames like:
    ..._2022-12-29_to_2023-01-30_...
    Returns 0 if not found.

    Args:
        zip_path (str): zip like: f-u09-paris~metro~bus~tram_2022-12_2022-12-29_to_2023-01-30_20e14e72_parquet.zip

    Returns:
        int: snap_id = YYYYMMDDYYYYMMDD 
    """
    m = re.search(r'([12]\d{3})-(\d{2})-(\d{2})_to_([12]\d{3})-(\d{2})-(\d{2})', zip_path)
    if not m:
        return 0
    y1, mo1, d1, y2, mo2, d2 = m.groups()
    return int(f"{y1}{mo1}{d1}{y2}{mo2}{d2}")

def zip_nearest(stops_list, dists_list, r_list):
    """Zip parallel arrays into list[{'stop_id':..., 'dist_m':...}] per row.
        Ensures we retain the sync between the stop_id and it's distance, which could be a danger when running 2xArrays.
    Args:
        stops_list (_type_): _description_
        dists_list (_type_): _description_

    Returns:
        list[dict]: _description_
    """
    n = min(len(stops_list or []), len(dists_list or []), len(r_list or [])) # <- should be the exact same length per array pair but check to be sure
    return [{"stop_id": str(stops_list[i]),
             "dist_m": float(dists_list[i]),
             "r_type": r_list[i]} for i in range(n)] 

# -----STRUCT------
NEAREST_FIELD = pa.list_(pa.struct([
    pa.field("stop_id", pa.string()),
    pa.field("dist_m", pa.float64()),
    pa.field("r_type", pa.list_(pa.int8()))
]))

def feature_nearest_stops_to_parquet_dataset(
    # Didn't know I can split out function input like this very nice!
    gdf_gps: gpd.GeoDataFrame,
    dict_precalc: dict[str, dict],
    lookup,                         
    method: str = "balltree",
    out_dir: str = "nearest_stops_ds",
    radius_default_m: int = 1000,
):
    """_summary_

    Args:
        gdf_gps (gpd.GeoDataFrame): _description_
        dict_precalc (dict[str, dict]): _description_
        lookup (_type_): _description_
        method (str, optional): _description_. Defaults to "balltree".
        out_dir (str, optional): _description_. Defaults to "nearest_stops_ds".
        radius_default_m (int, optional): _description_. Defaults to 1000.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    gps = gdf_gps.copy()
    gps['timestamp'] = pd.to_datetime(gps['timestamp'], errors="coerce")
    
    zip_paths, zip_path_segments = load_gtfs.return_zip_paths(gdf_gps, lookup, True)
    
    if (len(zip_paths) > 1):
        gps['gtfs_zip'] = pd.NA # <- Init with NaN
        
        for zip_path_segment in zip_path_segments:
            mask, change_idx, zip_path_linked = zip_path_segment # TODO: Maybe there is some faster way using the change_idx
            gps['gtfs_zip'] = np.where(mask, zip_path_linked, gps['gtfs_zip'])
            
            # Generate a snap ID for the zip_path -> this seems to be common practice but I wonder if we could just use the zip path itself?
    else:   
        # We just populate all with one zip_path :)
        _, _, zip_path_linked = zip_path_segments[0]
        gps['gtfs_zip'] = zip_path_linked
        
    # Using KEY grab the ID of the GPS file -> which is something like 10_377 etc..
    # uses the helper function to perform this action.
    if "KEY" not in gps.columns:
        raise ValueError("Expected 'KEY' column to derive file_id.")
    #gps["file_id"] = gps["KEY"].astype(str).map(parse_file_id_from_key)
    gps_file_id: str = parse_file_id_from_key(gps["KEY"].values[0]) # TODO: This might fail if it's not cast as .astype('str) trying for now.
    gps["file_id"] = gps_file_id

    # Efficiency -> avoid reset_index.
    gps["pt_idx"] = np.arange(len(gps), dtype=np.int32)
    
    gps["year"]  = gps['timestamp'].dt.year
    gps["month"] = gps['timestamp'].dt.month
    
    batches = []
    
    # ---ARROW TABLE DEF----
    def to_arrow_table(df_part: pd.DataFrame) -> pa.Table:
        #nearest_struct_col = [zip_nearest(s, d) for s, d in zip(df_part["stop_ids"], df_part["dists_m"])]
        nearest_struct_col = [zip_nearest(s, d, t) for s, d, t in zip(df_part["stop_ids"], df_part["dists_m"], df_part["route_types"])]

        # Calcs:
        min_d = [(min(d) if len(d)>0 else None) for d in df_part["dists_m"]]
        count_d = [len(d) for d in df_part["dists_m"]]
        #gps_file_id: str = parse_file_id_from_key(df_part["KEY"].values[0])
        
        return pa.table({
            "year":          pa.array(df_part["year"].astype("int16").to_numpy(), type=pa.int16()),
            "month":         pa.array(df_part["month"].astype("int8").to_numpy(), type=pa.int8()),
            "file_id":       pa.array(df_part["file_id"].astype(str).to_numpy(), type=pa.string()),
            "pt_idx":        pa.array(df_part["pt_idx"].astype("int32").to_numpy(),type=pa.int32()),
            "timestamp":     pa.array(df_part["timestamp"].to_numpy(), type=pa.timestamp("ns")),
            "nearest_stops": pa.array(nearest_struct_col, type=NEAREST_FIELD),
            #"route_types": pa.array(route_types, type=pa.list_(pa.string())), # TODO: Down the chain need to make sure this is int
            #"radius_m":      pa.array(np.full(len(df_part), 1000, dtype=np.int16), type=pa.int16()),
            "min_dist_m":    pa.array(np.array(min_d, dtype="float32")), 
            "count_within_radius": pa.array(np.array(count_d, dtype="int16")), 
            "gtfs_zip":      pa.array(df_part["gtfs_zip"].astype(str).to_numpy(), type=pa.string()),
            #"method":        pa.array(np.full(len(df_part), f"{method}_haversine"), type=pa.string()),
        })
        
    for zip_name, df_grp in gps.groupby("gtfs_zip", sort=False):
        precalc = dict_precalc[zip_name]
        if not precalc:
            # skip if we don't have a prepared BallTree for this snapshot
            continue

        stops_gdf = precalc["stops_gdf"]
        balltree  = precalc["balltree"]
        
        # Query the balltree for the points that fall into the stop radius and their distance.
        index_arrays, dist_arrays_m = calculate_stop_distance_balltree(balltree, df_grp, search_radius=radius_default_m)
        
        # Convert the stop index to the actual stop_id by flat listing (computationally faster than row by row)
        stop_id_values = stops_gdf["stop_id"].astype(str).to_numpy()
        route_type_values = stops_gdf["route_type_set"].to_numpy()
        
        
        index_arrays = [np.asarray(a, dtype=np.int64) for a in index_arrays]
        lengths = np.fromiter((len(a) for a in index_arrays), dtype=np.int64)
        if lengths.sum() == 0:
            stop_ids = [[] for _ in index_arrays]
            route_types = [[] for _ in index_arrays] # If we find no stops for the entire GPS trip
        else:
            flat_idx = np.concatenate(index_arrays)
            splits = lengths.cumsum()[:-1]
            
            gathered_stop = stop_id_values[flat_idx]
            gathered_route = route_type_values[flat_idx]
            
            stop_ids = [arr.tolist() for arr in np.split(gathered_stop, splits)]
            route_types = [arr.tolist() for arr in np.split(gathered_route, splits)]
        
        # Building up a DF
        df_out = pd.DataFrame({
            "year":      df_grp["year"].values[0],
            "month":     df_grp["month"].values[0],
            "file_id":   df_grp["file_id"].values[0],
            "pt_idx":    df_grp["pt_idx"].to_numpy(),
            "timestamp": df_grp["timestamp"].to_numpy(),
            "stop_ids":  stop_ids,
            "route_types": route_types,
            "dists_m":   [np.asarray(x, dtype="float32").tolist() for x in dist_arrays_m],
            "gtfs_zip": zip_name,
        })
        batches.append(to_arrow_table(df_out))
        
    if not batches:
        ValueError("Something went wrong when pushing DF's to ArrowTables by batch.")
        
    big_table = pa.concat_tables(batches, promote_options='default')
    
    # Write the dataset here :)
    pa.dataset.write_dataset(
        big_table,
        base_dir=out_dir,
        format="parquet",
        #partitioning=["year", "month", "file_id"],  # monthly slices + per file
        partitioning=pa.dataset.partitioning(
            schema=pa.schema([
                pa.field("year", pa.int16()),
                pa.field("month", pa.int8()),
                pa.field("file_id", pa.string()),
            ]),flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=1_000_000, # bit arbitary but trying to see how it works
        max_rows_per_group=128_000,   
        min_rows_per_group=64_000,
        basename_template="part-{i}.parquet",
    )

def build_stops_lookup_dict() -> dict:
    zip_paths = load_gtfs.list_gtfs_zip_paths(keyword="PMBT")
    
    concat_stopids_df = None
    
    dict_gtfs: dict = {}
    
    diffs_test = []
    for zip_path in zip_paths:
        GTFS_DATA = load_gtfs.get_store(zip_path)
        STOPS: gpd.GeoDataFrame = load_gtfs.generate_stop_modality_parented(GTFS_DATA)
        
        # Read the stop parent names and compare if we have already seen it before...
        #stop_ids = STOPS['parent_station']
        
        if concat_stopids_df is None:
            concat_stopids_df = STOPS.copy()
        else:
            #symdiff = pd.Index(concat_stopids_df['stop_id']).symmetric_difference(STOPS['stop_id'])
            #diffs_test.append(symdiff)

            missing_ids = set(STOPS['stop_id']) - set(concat_stopids_df['stop_id'])
            to_add = STOPS[STOPS['stop_id'].isin(missing_ids)]
            if not to_add.empty:
                concat_stopids_df = pd.concat([concat_stopids_df, to_add], ignore_index=True)
        
        
        # Change zip_path as key to zip_name as a key!
        # zip_name = Path(zip_path).name
        # dict_gtfs.update({
        #     f'{zip_name}': {
        #         'gtfs_store': GTFS_DATA,
        #         'stops_gdf': STOPS,
        #     }  
        # })
        
    return concat_stopids_df

