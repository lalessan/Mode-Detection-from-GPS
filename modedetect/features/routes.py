"""OSM route matching features using BallTree spatial indexing.

Extracts modal route networks from OSM PBF files, builds BallTrees,
and computes per-trip match percentages and binary route signals.

Refactored from: 4_Route_Matching/routes.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from sklearn.neighbors import BallTree

if TYPE_CHECKING:
    from ..config import ModeDetectConfig

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0

# Transit modes whose match percentage becomes a feature
TRANSIT_MODES = ("bus", "tram", "train", "subway")


# ── OSM extraction ───────────────────────────────────────────────────


def extract_osm_routes(pbf_path: Path, out_dir: Path) -> None:
    """Extract modal route networks from an OSM PBF and save as parquet.

    Requires the optional ``pyrosm`` dependency (``pip install modedetect[osm]``).
    """
    try:
        from pyrosm import OSM
    except ImportError as exc:
        raise ImportError(
            "pyrosm is required for OSM extraction. Install with: pip install modedetect[osm]"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    osm = OSM(str(pbf_path))

    logger.info("Extracting route networks from %s ...", pbf_path.name)

    for net_type, fname in [("walking", "walking"), ("cycling", "cycling"), ("driving", "driving")]:
        gdf = osm.get_network(network_type=net_type)
        gdf.to_parquet(out_dir / f"{fname}.parquet")

    bus = osm.get_data_by_custom_criteria(
        custom_filter={"route": ["bus"]},
        filter_type="keep", keep_ways=True, keep_relations=True,
        tags_as_columns=["route", "ref", "name", "operator"],
    )
    bus.to_parquet(out_dir / "bus.parquet")

    rail = osm.get_data_by_custom_criteria(
        custom_filter={"route": ["subway", "railway", "train", "tram", "light_rail"]},
        filter_type="keep", keep_ways=True, keep_relations=True,
        tags_as_columns=["route", "ref", "name", "operator"],
    )

    # Split rail into sub-networks
    for label, filter_vals in [
        ("subway", ["subway"]),
        ("train", ["train", "railway"]),
        ("tram", ["tram", "light_rail"]),
    ]:
        subset = rail[rail["route"].isin(filter_vals)]
        subset.to_parquet(out_dir / f"{label}.parquet")

    logger.info("Route parquets saved to %s", out_dir)


# ── BallTree construction ────────────────────────────────────────────


class RouteBallTree:
    """Wraps a BallTree over route geometry points with route-ID tracking."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        rows = []
        geom_id_pairs = list(zip(gdf["geometry"], gdf["id"] if "id" in gdf.columns else range(len(gdf))))

        for i, (geom, _rid) in enumerate(geom_id_pairs):
            if isinstance(geom, LineString):
                for x, y in geom.coords:
                    rows.append((y, x, i))
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    for x, y in line.coords:
                        rows.append((y, x, i))

        if not rows:
            self.tree = None
            self.route_idx = np.array([], dtype=int)
            self._geom_id = np.array(geom_id_pairs, dtype=object)
            return

        arr = np.array(rows)
        coords_rad = np.radians(arr[:, :2].astype(float))
        self.tree = BallTree(coords_rad, metric="haversine")
        self.route_idx = arr[:, 2].astype(int)
        self._geom_id = np.array(geom_id_pairs, dtype=object)

    @property
    def is_empty(self) -> bool:
        return self.tree is None


def build_route_balltree(
    parquet_path: Path,
    densify_spacing_m: int | None = None,
) -> RouteBallTree:
    """Load a route parquet and build a RouteBallTree.

    Parameters
    ----------
    parquet_path : path to a GeoDataFrame parquet
    densify_spacing_m : if set, densify geometries to this spacing before building
    """
    gdf = gpd.read_parquet(parquet_path)

    if densify_spacing_m is not None and densify_spacing_m > 0:
        gdf = _densify_gdf(gdf, densify_spacing_m)

    return RouteBallTree(gdf)


def _densify_gdf(gdf: gpd.GeoDataFrame, spacing_m: int) -> gpd.GeoDataFrame:
    """Densify route geometries by interpolating points at regular spacing."""
    gdf = gdf.copy()
    metric = gdf.to_crs(3857)

    def _densify_geom(geom):
        if geom.is_empty:
            return geom
        if isinstance(geom, LineString):
            return _densify_line(geom, spacing_m)
        if isinstance(geom, MultiLineString):
            return MultiLineString([_densify_line(part, spacing_m) for part in geom.geoms])
        return geom

    metric["geometry"] = metric["geometry"].apply(_densify_geom)
    gdf["geometry"] = metric["geometry"].to_crs(gdf.crs)
    return gdf


def _densify_line(line: LineString, spacing: float) -> LineString:
    length = line.length
    if length == 0:
        return line
    n_seg = max(1, int(np.ceil(length / spacing)))
    dists = np.linspace(0, length, n_seg + 1)
    return LineString([line.interpolate(d) for d in dists])


# ── Route matching ───────────────────────────────────────────────────


def fast_match(
    rbt: RouteBallTree,
    gps_lat: np.ndarray,
    gps_lon: np.ndarray,
    search_radius_m: int = 20,
) -> dict[int, tuple[int, float]] | None:
    """Match GPS points against a RouteBallTree.

    Returns dict mapping route index → (hit_count, hit_percentage),
    or None if no matches found.
    """
    if rbt.is_empty:
        return None

    gps_rad = np.radians(np.column_stack([gps_lat, gps_lon]))
    rad = search_radius_m / EARTH_RADIUS_M
    ind, _ = rbt.tree.query_radius(gps_rad, r=rad, return_distance=True, sort_results=True)

    non_empty_mask = np.array([len(a) > 0 for a in ind])
    if not non_empty_mask.any():
        return None

    route_indices_per_point = [np.unique(rbt.route_idx[idx]) for idx in ind[non_empty_mask]]
    flat = np.concatenate(route_indices_per_point)

    unique_idx, counts = np.unique(flat, return_counts=True)
    total = len(gps_rad)

    return {int(idx): (int(c), round(c / total, 4)) for idx, c in zip(unique_idx, counts)}


def generate_route_signal(
    rbt: RouteBallTree,
    gps_lat: np.ndarray,
    gps_lon: np.ndarray,
    search_radius_m: int = 20,
) -> np.ndarray:
    """Generate a binary 0/1 signal: 1 where GPS is near the best-matching route.

    Returns array of length len(gps_lat) with dtype int8.
    """
    freq = fast_match(rbt, gps_lat, gps_lon, search_radius_m)
    signal = np.zeros(len(gps_lat), dtype=np.int8)

    if freq is None or len(freq) == 0:
        return signal

    # Find the route with the highest match percentage
    best_idx = max(freq, key=lambda k: freq[k][1])

    # Re-query to get per-point route IDs
    gps_rad = np.radians(np.column_stack([gps_lat, gps_lon]))
    rad = search_radius_m / EARTH_RADIUS_M
    ind, _ = rbt.tree.query_radius(gps_rad, r=rad, return_distance=True, sort_results=True)

    for i, idx_arr in enumerate(ind):
        if len(idx_arr) > 0:
            point_routes = np.unique(rbt.route_idx[idx_arr])
            if best_idx in point_routes:
                signal[i] = 1

    return signal


# ── Trip-level feature aggregation ───────────────────────────────────


def compute_route_features(
    gps_df: pd.DataFrame,
    route_balltrees: dict[str, RouteBallTree],
    config: ModeDetectConfig,
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Compute per-trip (or per-segment) route match percentage features.

    Uses batch BallTree queries (one per mode) and vectorized aggregation
    instead of per-trip loops.

    Parameters
    ----------
    gps_df : GPS DataFrame with lat/lon and trip_key
    route_balltrees : dict mapping mode name → RouteBallTree (e.g. {"bus": rbt, "tram": rbt, ...})
    config : ModeDetectConfig
    segment_col : optional segment column

    Returns DataFrame with columns like ``bus_match_pct``, ``tram_match_pct``, etc.
    """
    from tqdm import tqdm

    key_col = config.col_trip_key
    n_points = len(gps_df)

    # Prepare GPS coordinates once
    all_lat = gps_df[config.col_latitude].astype("float64").values
    all_lon = gps_df[config.col_longitude].astype("float64").values
    gps_rad = np.radians(np.column_stack([all_lat, all_lon]))
    rad = config.route_search_radius_m / EARTH_RADIUS_M

    # Build trip key + segment arrays
    trip_keys = gps_df[key_col].values
    merge_cols = [key_col]
    if segment_col is not None:
        seg_ids = gps_df[segment_col].values
        merge_cols = [key_col, segment_col]

    # Build the result DataFrame with one row per trip/segment
    if segment_col is not None:
        base_df = pd.DataFrame({key_col: trip_keys, segment_col: seg_ids})
    else:
        base_df = pd.DataFrame({key_col: trip_keys})
    points_per_trip = base_df.groupby(merge_cols, sort=False, observed=True).size()
    result = points_per_trip.reset_index(name="_n_points")

    CHUNK = 500_000
    n_chunks = (n_points + CHUNK - 1) // CHUNK

    for mode_name in TRANSIT_MODES:
        col = f"{mode_name}_match_pct"
        rbt = route_balltrees.get(mode_name)

        if rbt is None or rbt.is_empty:
            result[col] = 0.0
            continue

        # ── Chunked BallTree query for this mode ──────────────────────
        logger.info("Batch route query: %s (%d chunks)...", mode_name, n_chunks)
        all_route_ids = [None] * n_points  # route indices per GPS point

        for c in tqdm(range(n_chunks), desc=f"Route query ({mode_name})", unit="chunk"):
            lo = c * CHUNK
            hi = min(lo + CHUNK, n_points)
            ind = rbt.tree.query_radius(gps_rad[lo:hi], r=rad)
            for j in range(hi - lo):
                if len(ind[j]) > 0:
                    all_route_ids[lo + j] = np.unique(rbt.route_idx[ind[j]])

        # ── Explode to (gps_pos, route_idx) pairs ────────────────────
        gps_pos_list = []
        route_idx_list = []
        for i in range(n_points):
            if all_route_ids[i] is not None:
                n = len(all_route_ids[i])
                gps_pos_list.append(np.full(n, i, dtype=np.int64))
                route_idx_list.append(all_route_ids[i])

        if not gps_pos_list:
            result[col] = 0.0
            continue

        flat_gps_pos = np.concatenate(gps_pos_list)
        flat_route_idx = np.concatenate(route_idx_list)

        # ── Per-trip, per-route: count matching GPS points ────────────
        flat_df = pd.DataFrame({
            key_col: trip_keys[flat_gps_pos],
            "_route": flat_route_idx,
            "_gps_pos": flat_gps_pos,
        })
        if segment_col is not None:
            flat_df[segment_col] = seg_ids[flat_gps_pos]

        # Count unique GPS points per trip per route
        hits = flat_df.groupby(
            merge_cols + ["_route"], sort=False, observed=True
        )["_gps_pos"].nunique().reset_index(name="_hits")

        # Join with trip point counts to get percentages
        hits = hits.merge(result[merge_cols + ["_n_points"]], on=merge_cols, how="left")
        hits["_pct"] = hits["_hits"] / hits["_n_points"]

        # Best route per trip = max match percentage
        best = hits.groupby(merge_cols, sort=False, observed=True)["_pct"].max()
        best.name = col

        result = result.merge(best.reset_index(), on=merge_cols, how="left")
        result[col] = result[col].fillna(0.0)

    # Clean up and rename
    result = result.drop(columns=["_n_points"])
    result = result.rename(columns={key_col: "trip_key"})
    if segment_col is not None:
        result = result.rename(columns={segment_col: "segment"})

    logger.info("Route feature aggregation complete: %d trips", len(result))
    return result
