import numpy as np
import geopandas as gpd
import pandas as pd

from sklearn.neighbors import BallTree
from shapely.geometry import LineString, MultiLineString, Point
#from pyrosm import OSM
from pathlib import Path
from typing import Dict, Union, TypedDict

from pyrosm import OSM

def _split_rail_routes(rail_routes: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split OSM rail routes into subway, rail, and tram/light rail groups.

    Args:
        rail_routes (gpd.GeoDataFrame): Rail routes with a 'route' column
            containing values like 'subway', 'railway', 'train', 'tram', and 'light_rail'.
    """
    train = rail_routes[rail_routes["route"] == "train"]
    railway = rail_routes[rail_routes["route"] == "railway"]
    
    tram = rail_routes[rail_routes["route"] == "tram"]
    light_rail = rail_routes[rail_routes["route"] == "light_rail"]
    
    subway = rail_routes[rail_routes["route"] == "subway"]
    
    if len(train) == 0 or len(railway) == 0 or len(tram) == 0 or len(light_rail) == 0 or len(subway) == 0:
        print("Warning: returning a GeoDataFrame with no entries, check if your Rail Routes GeoDataFrame contains the correct routes.")
    
    # Concat
    train_and_rail = pd.concat([train, railway], ignore_index=True)
    tram_and_lightrail = pd.concat([tram, light_rail], ignore_index=True)
    
    return subway, train_and_rail, tram_and_lightrail

def filter_osm_pbf(pbf_path: Path, out_path: Path) -> None:
    """Extract walking, cycling, driving, bus and rail transit networks from an OSM
    PBF file using pyrosm and save them as Parquet files under the given output
    directory.

    Args:
        pbf_path (Path): Path to the input OSM .pbf file.
        out_path (Path): Directory where the filtered Parquet files are written.
    """
    in_pbf = Path(pbf_path)
    osm = OSM(str(in_pbf))
    
    # Filter
    print("Filtering walking routes from OSM pbf file...")
    walking_network = osm.get_network(network_type="walking")
    print("Filtering cycling routes from OSM pbf file...")
    cycling_network = osm.get_network(network_type="cycling")
    print("Filtering driving routes from OSM pbf file...")
    driving_network = osm.get_network(network_type="driving")
    
    print("Filtering bus routes from OSM pbf file...")
    bus_routes = osm.get_data_by_custom_criteria(
        custom_filter={"route": ["bus"]},
        filter_type="keep",
        keep_ways=True,
        keep_relations=True,
        tags_as_columns=["route", "ref", "name", "operator"]
    )
    
    print("Filtering rail routes from OSM pbf file...")
    rail_routes = osm.get_data_by_custom_criteria(
        custom_filter={"route": ["subway", "railway", "train", "tram", "light_rail"]},
        filter_type="keep",
        keep_ways=True,
        keep_relations=True,
        tags_as_columns=["route", "ref", "name", "operator"]
    )
    
    # Split the rail routes:
    subway_routes, train_routes, tram_routes = _split_rail_routes(rail_routes)
        
    # Save out
    print(f"Saving routes to parquet file to {out_path}")
    subway_routes.to_parquet(out_path / "subway_routes.parquet")
    train_routes.to_parquet(out_path / "train_routes.parquet")
    tram_routes.to_parquet(out_path / "tram_routes.parquet")
    bus_routes.to_parquet(out_path / "bus_routes.parquet")
    
    driving_network.to_parquet(out_path / "driving_routes.parquet")
    walking_network.to_parquet(out_path / "walking_routes.parquet")
    cycling_network.to_parquet(out_path / "cycling_routes.parquet")
    print("Complete!")
    
def bind_route_geometry_to_id(routes: gpd.GeoDataFrame) -> np.ndarray:
    """Return an array of (geometry, id) pairs for routes."""
    route_geom = zip(routes["geometry"], routes["id"])
    return np.array(list(route_geom), dtype=object)

class BallTreeObj(TypedDict):
    balltree: BallTree
    route_idx: np.ndarray

def generate_balltree_obj(route_geom_arr: np.array) -> BallTreeObj:
    """Build a BallTree over route point coordinates extracted from route geometries.

    Args:
        route_geom_arr (np.array): Array of shape (N, 2) where the first column
            contains LineString or MultiLineString geometries and the second
            column contains their route identifiers.

    Returns:
        BallTree: BallTree built on all route points in latitude/longitude
            (radians) using the haversine metric.
    """
    
    # Here we merge all our route points into a single array 
    # -> while retaining a reference to what route it is by keeping it's index as a third val in the tuple.
    rows = []
    for i, geom in enumerate(route_geom_arr[:, 0]):  
        if isinstance(geom, LineString):
            for x, y in geom.coords:
                rows.append((y, x, i))           
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                for x, y in line.coords:
                    rows.append((y, x, i))

    rows = np.array(rows)  # shape (N, 3)
    
    latlon_for_balltree = np.radians(rows[:, :2].astype(float)) # lat, lon in radians
    route_idx = rows[:, 2].astype(int)
    
    # Generate / compute the BallTree for all route points
    balltree = BallTree(latlon_for_balltree, metric='haversine')
    
    return {"balltree": balltree, "route_idx": route_idx}
    
def fast_match(routes: Union[gpd.GeoDataFrame, np.ndarray], gps_trace: pd.DataFrame, balltreeobj: BallTreeObj, search_radius_m: int = 20, _return_rid_per_point: bool = False) -> Dict:
    """
    Match a GPS trace to nearby OSM routes using a prebuilt BallTree.

    This function queries a BallTree of route points around each GPS point
    in the trace, aggregates which routes are hit within the given search
    radius, and returns simple frequency statistics per route as a dictionary.

    Args:
        routes: Either a GeoDataFrame of routes with 'geometry' and 'id'
            columns, or a prebound numpy array of shape (N, 2) with
            (geometry, id) pairs. Prebound np array can be calculated with bind_route_geometry_to_id().
        gps_trace: DataFrame containing at least 'LATITUDE' and 'LONGITUDE'
            columns for the GPS points of a single trip.
        balltreeobj: A BallTreeObj that contains the BallTree itself and the route index
            array, that maps back the routeID to indices.
            Precomputed BallTree built on route point coordinates
            (lat/lon in radians) using the haversine metric.
        search_radius_m: Search radius in meters around each GPS point used
            when querying the BallTree.

    Returns:
        Dict mapping route IDs to a (count, percentage) tuple, where:
            - count is the number of GPS points that matched that route
              (after de-duplicating multiple hits of the same route per point),
            - percentage is count divided by the total number of GPS
              points in the trace.
    """
    
    # Immutable
    EARTH_RADIUS_M = 6371000.0
    
    if isinstance(routes, gpd.GeoDataFrame):
        # treat as GeoDataFrame
        route_geom_arr = bind_route_geometry_to_id(routes)
    elif isinstance(routes, np.ndarray):
        # treat as already prepared (geometry, id) array
        route_geom_arr = routes
    else:
        raise TypeError(f"Unsupported type for routes: {type(routes)}")
    
    # Take only the route geometry and bind to its ID.
    # route_geom = zip(routes['geometry'], routes['id'])
    # route_geom_arr = np.array(list(route_geom), dtype=object) # TODO: This can probably be made nicer

    # Split the balltreeobj into its route indices and actual balltree:
    balltree = balltreeobj["balltree"]
    route_idx = balltreeobj["route_idx"]
    
    gps_deg = gps_trace[['LATITUDE', 'LONGITUDE']].to_numpy()
    gps_rad = np.radians(gps_deg) 

    # Define the search radius per route point and queery the BallTree
    rad = search_radius_m / EARTH_RADIUS_M # search meter radius for checking intersections
    (ind, dist_rad) = balltree.query_radius(gps_rad, r=rad, return_distance=True, sort_results=True) # dist in radians
    
    
    # We use this as a way to only find the GPS points that crossed a part of a route (ind[non_empty_mask])
    non_empty_mask = np.array([len(a) > 0 for a in ind]) 
        
    if len(ind[non_empty_mask]) == 0: # Check if any matches found...if none we just return None
        return None
    
    # Due to tight route clustering we can have a GPS point capture multiple of the same route ID.
    # Therefore, we append the unique ID's per GPS point to avoid double counting the route.
    route_indices_per_point = []
    for index_arr in ind[non_empty_mask]:
        route_indices_per_point.append(np.unique([route_idx[index_arr]])) # <- Append only the unique indices found per GPS point.
    
    # With our list of lists containing all unique per GPS point indices found, we flatten and then return 
    # an array that links the indices to the corresponding route ID and it's relevant multi/linestring geometry. 
    # NOTE: If there are memory issues we can simplify this and not store the linestrings.
    route_indices_flat: np.array = np.concat(route_indices_per_point)
    route_ids_linestring: np.array = route_geom_arr[route_indices_flat, :] # <- take the linestrings as well so we can see how long they are

    # Frequency checks
    unique_ids, counts = np.unique(route_ids_linestring[:, 1], return_counts=True)

    # Build the return 
    # Removing this distance computation for speed
    # TODO: If this becomes relevant in the future -> precompute this with BallTree and pass it.
    # g = gpd.GeoSeries(route_ids_linestring[:, 0], crs="EPSG:4326") # TODO: not entirely efficient as we are calculating for the same Linestring many times, smarter would be to do this calc on the original list.
    # g_m = g.to_crs(2154) # Lambert-93: could just use degree * size of earth here but using Lambert we have more accurate projection and hence more accurate dist.
    # lengths_m = g_m.length

    gps_total_arr = np.full(len(counts), len(gps_rad), dtype='int32') # used to find the total to use for percentage calc.
    percentages = counts / gps_total_arr

    # We are left with a dict that is ID of route that is near our GPS journey
    # Format: routeID: amount of GPS points that fall near it, percentage of those points to total GPS length, length of the bus route itself!
    #dict_freq: Dict = dict(zip(unique_ids, zip(counts, np.round(percentages, 3), np.round(lengths_m, 0))))
    dict_freq: Dict = dict(zip(unique_ids, zip(counts, np.round(percentages, 3))))
    
    if _return_rid_per_point:
        # rid_per_point = []
        # for idices_single in route_indices_per_point:
        #     rids = route_geom_arr[idices_single, 1]
        #     rid_per_point.append(rids)
        # return dict_freq, rid_per_point
    
        #FIX: route_indices_per_point is only matched we therefore lose the correct index order
        rid_per_point = []
        for index_arr in ind:  # IMPORTANT: iterate over all points, not masked
            if len(index_arr) == 0:
                rid_per_point.append(np.array([], dtype=route_geom_arr[:,1].dtype))
                continue

            # map balltree point indices -> route indices -> route ids
            route_indices = np.unique(route_idx[index_arr])
            rids = route_geom_arr[route_indices, 1]
            rid_per_point.append(rids)

        return dict_freq, rid_per_point
    else: 
        return dict_freq

def return_top_route_ids(freq_dict: Dict, over_pct: float) -> list:
    """Return route IDs whose matched GPS percentage exceeds a given threshold.

    Args:
        freq_dict: Dictionary mapping route IDs to (count, percentage) tuples,
            as returned by fast_match().
        over_pct: Minimum percentage (0-1) of GPS points for a route to be
            included in the result.

    Returns:
        List of route IDs with percentage strictly greater than over_pct.
    """
    return [rid for rid, (_, pct) in freq_dict.items() if pct > over_pct]

def visualise_top_route_ids(gps_trace: pd.DataFrame, routes: gpd.GeoDataFrame, top_route_ids: list):
    top_routes = routes[routes['id'].isin(top_route_ids)]

    gdf_trace = gpd.GeoDataFrame(gps_trace, geometry=[Point(x, y) for x, y in zip(gps_trace['LONGITUDE'], gps_trace['LATITUDE'])], crs="EPSG:4326")

    ax = top_routes.plot(figsize=(15, 15), color="blue", linewidth=2, alpha=0.2)
    gdf_trace.plot(ax=ax, color="red", markersize=5)

    ax.set_title("GPS trace with matched routes >%")
    
def generate_route_signal(freq_dict: Dict, mode_arr, gps_trace, balltreemode, search_radius, verbose: bool = False) -> np.array:
    # Take the most matched dict
    
    # Max ID:
    if not freq_dict:  # catch None and {} and if return None so downstream we can not use this signal
        if verbose:
            print("No matched route (freq_dict is None/empty)")
        return np.zeros(len(gps_trace), dtype=np.int8)
        #return None
    
    max_rid = max(freq_dict, key=lambda rid: freq_dict[rid][1])
    
    # Max percentage:
    if verbose:
        max_pct = freq_dict[max_rid]
        
        print(f"ID: {max_rid} with a match percentage of {max_pct} is being used.")
    
    # Signal 0 or 1, 0 being no point on route of the top route, while 1 being route point in range.
    signal_arr = np.zeros(len(gps_trace), dtype=np.int8) # init arr with no signal.
    
    # Re-Run fast_match now also retrieving the per point data, NOTE: this could be passed directly
    # but I will contain it inside the function.
    
    _, rid_per_point = fast_match(mode_arr, gps_trace, balltreemode, search_radius, _return_rid_per_point=True)
    
    # Then we need to filter through per point and only generate = 1 when we see the max_rid.
    for i, rid_single in enumerate(rid_per_point):
        if max_rid in rid_single:
            signal_arr[i] = 1
    
    return signal_arr
    
# ----- POINT INTERPOLATION -----

def densify_line(line: LineString, spacing: float) -> LineString:
    length = line.length
    if length == 0:
        return line

    # number of segments --- spacing <= desired spacing
    n_seg = max(1, int(np.ceil(length / spacing)))
    dists = np.linspace(0, length, n_seg + 1)
    pts = [line.interpolate(d) for d in dists]
    
    return LineString(pts)

def densify_geom(geom, spacing: float):
    """
    Densify LineString / MultiLineString geometry.
    Expects CRS in *meters* (metric).
    """
    if geom.is_empty:
        return geom

    if isinstance(geom, LineString):
        return densify_line(geom, spacing)

    if isinstance(geom, MultiLineString):
        parts = []
        for part in geom.geoms:
            dense = densify_line(part, spacing)
            parts.append(dense)
        return MultiLineString(parts)

    # Fallback if it is neither...just return the same geom
    return geom

def add_densified_geometry(routes_gdf: gpd.GeoDataFrame, spacing_m: int, metric_crs=3857, overwrite_geom: bool = False):
    """
    routes_gdf: GeoDataFrame with OSM route geometries (EPSG:4326 as per OSM data).
    spacing_m : desired spacing between points in meters.
    
    Returns: new GeoDataFrame with an extra 'geometry_dense' column.
    """
    routes = routes_gdf.copy() # We ensure a copy of the original routes df. This might not be mem efficient...

    # project to metric CRS
    routes_metric = routes.to_crs(metric_crs)

    # densify geometry while in metric CRS
    routes_metric["geometry_dense"] = routes_metric["geometry"].apply(lambda g: densify_geom(g, spacing_m))
    if overwrite_geom:
        # project densified geometry back to original CRS while overwriting the original geom col
        routes["geometry"] = routes_metric["geometry_dense"].to_crs(routes_gdf.crs) # <- Could overwrite orig col to save memory.
    else:
        #routes_metric["geometry_dense"] = routes_metric["geometry"].apply(lambda g: densify_geom(g, spacing_m))

        # project densified geometry back to original CRS into a new col
        routes["geometry_dense"] = routes_metric["geometry_dense"].to_crs(routes_gdf.crs) # <- Could overwrite orig col to save memory.

    return routes
