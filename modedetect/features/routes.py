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

    Parameters
    ----------
    gps_df : GPS DataFrame with lat/lon and trip_key
    route_balltrees : dict mapping mode name → RouteBallTree (e.g. {"bus": rbt, "tram": rbt, ...})
    config : ModeDetectConfig
    segment_col : optional segment column

    Returns DataFrame with columns like ``bus_match_pct``, ``tram_match_pct``, etc.
    """
    key_col = config.col_trip_key
    group_cols = [key_col] if segment_col is None else [key_col, segment_col]

    rows = []
    for group_key, grp in gps_df.groupby(group_cols, sort=False, observed=True):
        if isinstance(group_key, str):
            trip_key = group_key
            seg_id = None
        else:
            trip_key = group_key[0]
            seg_id = group_key[1] if len(group_key) > 1 else None

        lat = grp[config.col_latitude].astype("float64").values
        lon = grp[config.col_longitude].astype("float64").values

        row = {"trip_key": trip_key}
        if seg_id is not None:
            row["segment"] = seg_id

        for mode_name in TRANSIT_MODES:
            rbt = route_balltrees.get(mode_name)
            if rbt is None or rbt.is_empty:
                row[f"{mode_name}_match_pct"] = 0.0
                continue

            freq = fast_match(rbt, lat, lon, config.route_search_radius_m)
            if freq is None:
                row[f"{mode_name}_match_pct"] = 0.0
            else:
                best_pct = max(pct for _, pct in freq.values())
                row[f"{mode_name}_match_pct"] = best_pct

        rows.append(row)

    return pd.DataFrame(rows)
