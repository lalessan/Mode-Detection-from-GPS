import zipfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import date, datetime, timedelta
from bisect import bisect_right
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from functools import lru_cache
from typing import Dict, Iterable
import os

# NOTE: Remember to update these.
# ---- LOCATIONS ----
FOLDER_DIR_PMBT = Path(r"/home/s232534/data/gtfs_parquet/relevant/pmbt")
FOLDER_DIR_STR = Path(r"/home/s232534/data/gtfs_parquet/relevant/str")
#FILE_LOCATION = Path(r"C:\Users\Mikus\Documents\DTU\Thesis\ThesisGPS\gtfs_parquet\relevant\f-u09-paris~metro~bus~tram_2022-10_2022-10-31_to_2022-11-29_a2023650_parquet.zip")

# --- Functionality for grabbing the right dataset for the dates you need.
class DateRangeLookup:
    """
    Fast O(log n) lookup of a value by date falling in a (start,end) inclusive range.
    Ranges: iterable of (start_date, end_date, value)
    Dates must be datetime.date (not datetime).
    """
    
    def __init__(self, ranges):
        cleaned = []
        # start_date, end_date, value
        for s, e, v in ranges:
            if isinstance(s, datetime): 
                s = s.date()
            if isinstance(e, datetime): 
                e = e.date()
            if s > e:
                raise ValueError(f"Start > end: {s} {e}")
            
            cleaned.append((s, e, v))
            
        cleaned.sort(key=lambda x: x[0])
        
        # overlap check 
        # TODO: This is quite inefficient in the loops it is doing... but probs doesn't matter for this.
        for i in range(1, len(cleaned)):
            # prev_start_date, end_date, value
            # current_start_date, ...end_date, value
            ps, pe, pv = cleaned[i-1]
            cs, ce, cv = cleaned[i]
            if cs <= pe:  # overlap or touching-inclusive
                # give the boundary day to the later range
                new_pe = min(pe, cs - timedelta(days=1))
                if new_pe < ps:
                    raise ValueError(f"Range {i-1} would become empty after trim: {(ps,pe)} vs {(cs,ce)}")
                cleaned[i-1] = (ps, new_pe, pv)
                        
        self._starts = [r[0] for r in cleaned]
        self._ends   = [r[1] for r in cleaned]
        self._values = [r[2] for r in cleaned]

    def get(self, d: date):
        if isinstance(d, datetime):
            d = d.date()
        i = bisect_right(self._starts, d) - 1
        if i >= 0 and d <= self._ends[i]:
            return self._values[i]
        raise KeyError(d)

    def __contains__(self, d):
        try:
            self.get(d)
            return True
        except KeyError:
            return False

def _construct_date_lookup(folder_director: Path = FOLDER_DIR_PMBT):
    """
    Build a DateRangeLookup from calendar.parquet (GTFS). 
    """
    zip_files = list(folder_director.glob("*.zip"))
    # Sanity check if we are looking in the wrong folder!
    if not zip_files:
        raise FileNotFoundError(f"No zip files in {folder_director}")
    
    all_ranges = []
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as z:
            if "calendar_dates.parquet" not in z.namelist():
                continue
            with z.open("calendar_dates.parquet") as f:
                #calendar = pd.read_parquet(f, columns=["date"])
                calendar = pd.read_parquet(f)
        # convert GTFS yyyymmdd to date
        
        if 'date' not in calendar.columns:                 
            print("DATE COL MISSING OR INCORRECTLY LABELLED!")
        calendar['date'] = pd.to_datetime(calendar['date'], format='%Y%m%d', errors='coerce')
        calendar.sort_values('date', ascending=True)
        start_date = calendar['date'].min()
        end_date   = calendar['date'].max()
        print("Loaded frame: ",start_date, end_date)
        all_ranges.append((start_date, end_date, str(zip_path)))
    
    if not all_ranges:
        raise ValueError("No calendar.parquet data found in any zip.")
    return DateRangeLookup(all_ranges)

def build_lookup_objects() -> tuple[DateRangeLookup, DateRangeLookup]:
    """Allows you to build the lookup table once and then pass the object to return zip path
    function, saving rebuild time (important when running this function in large loops).

    Returns:
        tuple[DateRangeLookup(pmbt), DateRangeLookup(reg)]: The objects that have loaded the GTFS data (calendar_dates.parquet).
    """
    lookup_pmbt: DateRangeLookup = _construct_date_lookup(FOLDER_DIR_PMBT)
    lookup_str: DateRangeLookup = _construct_date_lookup(FOLDER_DIR_STR)
    
    return (lookup_pmbt, lookup_str)

def return_zip_path(date_input: datetime, lookup_object: DateRangeLookup) -> str:
    """Allows for split parquet database functionality. 
    By calling a date you load in only the GTFS data that is relevant.

    Args:
        date_input (datetime): The date we are interested in accessing GTFS data for.
        lookup_object (DateRangeLookup): The pre-built object that has ingested the GTFS datasets and extracted the date ranges.

    Returns:
        str: The zip path of the GTFS parquet zip file that contains that date.
    """
    # lookup_pmbt: DateRangeLookup = _construct_date_lookup(FOLDER_DIR_PMBT)
    # lookup_str: DateRangeLookup = _construct_date_lookup(FOLDER_DIR_STR)
    
    try: 
        zip_path: str = lookup_object.get(date_input)
    except KeyError:
        print("No range cover for: ", date_input)
    
    return zip_path

def return_zip_paths(gdf_gps: gpd.GeoDataFrame, lookup: DateRangeLookup, return_change_row: bool = False) -> list[str]:
    """Function that returns ALL zip paths required to load the required range of GTFS data from a GPS trace geodataframe/dataframe.
    Args:
        gdf_gps (gpd.GeoDataFrame): GPS trace dataframe
        lookup (load_gtfs.DateRangeLookup): Our generate DateRangeLookup object.

    Returns:
        list[str]: list of all GTFS parquet files we need to load to service the GPS file.
    """
    dates = pd.to_datetime(gdf_gps["timestamp"], errors='coerce').dt.date
    dates_unique = dates.unique()
    
    #NOTE: for the NetMob25 dataset the possible change point will only be two GTFS datasets at max because
    #      one person only has data for one week and the GTFS data splits are 1 month.
    #      This might be a problem if we start using data that does not follow this rule and hence
    #      we will need to consider more than 2 GTFS partitions so we would need a mask that can handle something like
    #      TTTFFT -> FFFTTF -> FFFFFT
    zip_paths = []
    masks: tuple = []
    for dt in dates_unique:
        zip_path = return_zip_path(dt, lookup)
        
        if (zip_path not in zip_paths):
            zip_paths.append(zip_path)
            
            if (return_change_row):
                mask = (dates == dt)
                # Fix for unsafe reference size1 + 1 reference:
                idx = np.flatnonzero(mask.to_numpy())
                end_excl = min(idx[-1] + 1, len(mask))
                masks.append((mask, end_excl, Path(zip_path).name))
    if (len(zip_paths) > 2):
        ValueError("There should at max be 2 zip_paths for a GPS dataframe!")
        
    if (return_change_row):
        return zip_paths, masks
    else:
        return zip_paths

def list_gtfs_zip_paths(folder: str | Path = None, keyword: str = None) -> list[Path]:
    """
    Return all GTFS parquet zip paths in the given folder.
    """
    if (keyword == None):
        folder = Path(folder)
        return sorted(folder.glob("*.zip"))
    elif (keyword == "PMBT"):
        folder = Path(FOLDER_DIR_PMBT)
        return sorted(folder.glob("*.zip"))
    elif (keyword == "STR"):
        folder = Path(FOLDER_DIR_STR)
        return sorted(folder.glob("*.zip"))
    else:
        KeyError("Not a valid GTFS store folder.")
        
# ---------- Functionality for storing and reading the actual data ---------
# Usage:
# store = get_store(Path("/data/gtfs/paris_gtfs_2024-10-01.zip"))
# stops = store["stops"]          # loaded once, reused thousands of times
# routes = store["routes"]
# warmup:
# store.preload(["stops", "trips", "stop_times"])
############################

# TABLES are all the possible datatypes in the GTFS data.
TABLES: tuple[str, ...] = ("agency", "calendar", "calendar_dates", "pathways", "routes", "stops", "stop_extensions", "stop_times", "transfers", "trips",)

class GTFSStore:
    """
    Lazily loads Parquet members from a GTFS zip and caches them as DataFrames.
    """
    def __init__(self, zip_path: Path, tables: Iterable[str] = TABLES):
        self.zip_path = Path(zip_path)
        self.tables = set(tables)
        self._dfs: Dict[str, pd.DataFrame] = {}

    def _load_parquet_from_zip(self, name: str) -> pd.DataFrame:
        """_summary_

        Args:
            name (str): _description_

        Raises:
            KeyError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        member = f"{name}.parquet"
        
        with zipfile.ZipFile(self.zip_path) as zf:
            try:
                raw = zf.read(member)  # bytes (compressed -> decompressed)
            except KeyError as e:
                raise KeyError(f"Missing member in zip: {member}") from e

        # read via Arrow buffer (no temp files)
        buf = pa.py_buffer(raw)
        table = pq.read_table(pa.BufferReader(buf))
        return table.to_pandas(types_mapper=pd.ArrowDtype)

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Unknown table: {name!r}. Valid: {sorted(self.tables)}")
        if name not in self._dfs:
            self._dfs[name] = self._load_parquet_from_zip(name)
        return self._dfs[name]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.get(name)

    def preload(self, names: Iterable[str] = TABLES) -> None:
        """Optional: load a subset/all upfront to avoid first-hit latency.
        Note super useful anymore since using much more efficient parquet format.
        Kept for legacy/future.
        """
        for n in names:
            _ = self.get(n)

def _zip_fingerprint(zip_path: Path) -> tuple[int, int]:
    """
    Cache invalidation: if the zip changes (mtime/size), rebuild the store.
    """
    s = os.stat(zip_path)
    return int(s.st_mtime), s.st_size

@lru_cache(maxsize=None)
def get_store(zip_path: Path) -> GTFSStore:
    # Include the fingerprint in the cache key by *calling* it here.
    _ = _zip_fingerprint(Path(zip_path))
    return GTFSStore(zip_path)
    
def clear_store_cache() -> None:
    """Call this if the GTFS zip is replaced and you need fresh tables."""
    get_store.cache_clear()
    
    
# -------- Functionality for joining dfs to find modality -------
# NOTE: DEPRECATED -> Use generate_stop_modality_parented() instead!!!
# TODO: Pretty slow to run ~10 seconds for a month of data. Consider using polars and doing groupby + unique).
def generate_stop_modality(gtfs: GTFSStore):
    stops = gtfs["stops"].copy()
    
    stop_times = gtfs["stop_times"][["stop_id", "trip_id"]]
    trips = gtfs["trips"][["trip_id", "route_id"]]
    routes = gtfs["routes"][["route_id", "route_type"]]
    
    # Join chain
    st_trips = stop_times.merge(trips, on="trip_id", how="left", validate="many_to_one")
    st_trips_routes = st_trips.merge(routes, on="route_id", how="left", validate="many_to_one")
    # Aggregate: collect distinct route_type per stop
    stop_route_info = (
        st_trips_routes[["stop_id", "route_type", "route_id"]]
        .dropna(subset=["route_id"])
        .drop_duplicates()
    )
    
    agg = (
        stop_route_info
        .groupby("stop_id")
        .agg(
            route_type_set=("route_type", lambda s: ",".join(str(v) for v in sorted(set(s)))),
            route_id_set=("route_id",   lambda s: ",".join(str(v) for v in sorted(set(s)))),
        )
        .reset_index()
    )

    # Attach to stops (left join keeps all stops)
    stops_with_modality = stops.merge(agg, on="stop_id", how="left")

    return stops_with_modality

def generate_stop_modality_parented(gtfs: GTFSStore) -> pd.DataFrame:
    """This is the more accurate function that correctly derives stop modality by using the parent station linkage.

    Args:
        gtfs (GTFSStore): Takes the GTFS object as input

    Returns:
        pd/gpd.DataFrame: A gdf containing stops with the type of associated modality.
    """
    stops = gtfs["stops"].copy()

    stop_times = gtfs["stop_times"][["stop_id", "trip_id"]]
    trips = gtfs["trips"][["trip_id", "route_id"]]
    routes = gtfs["routes"][["route_id", "route_type"]]

    # Join chain: stop_id -> trip_id -> route_id -> route_type
    st_trips = stop_times.merge(trips, on="trip_id", how="left", validate="many_to_one")
    st_trips_routes = st_trips.merge(routes, on="route_id", how="left", validate="many_to_one")

    # Unique stop-route pairs (drop trips duplication)
    stop_route_info = (
        st_trips_routes[["stop_id", "route_type", "route_id"]]
        .dropna(subset=["route_id"])
        .drop_duplicates()
    )

    # Map each stop to its station_key = parent_station (if present) else itself
    if "parent_station" in stops.columns:
        parent = stops["parent_station"].replace("", pd.NA)
    else:
        parent = pd.Series(pd.NA, index=stops.index, dtype="object")
    station_key = parent.fillna(stops["stop_id"])

    parent_map = pd.DataFrame({"stop_id": stops["stop_id"], "station_key": station_key})

    # Attach station_key to stop-route pairs
    stop_route_info = stop_route_info.merge(
        parent_map, on="stop_id", how="left", validate="many_to_one"
    )
    
    route_type_agg = (
        stop_route_info
        .groupby("station_key")["route_type"]
        .apply(lambda s: sorted(pd.unique(s.dropna()).astype(int).tolist()))
        .reset_index(name="route_type_set")
    )
    
    route_id_agg = (
        stop_route_info
        .groupby("station_key")["route_id"]
        .apply(lambda s: sorted(pd.unique(s.dropna()).tolist()))
        .reset_index(name="route_id_set")
    )
    
    agg = route_type_agg.merge(route_id_agg, on="station_key")

    # Join back to stops on station_key
    stops_with_modality = (
        stops.assign(station_key=station_key)
             .merge(agg, on="station_key", how="left")
             .drop(columns="station_key")
    )

    return stops_with_modality