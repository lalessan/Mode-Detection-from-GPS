"""Low-level geodesic and bearing helpers.

All functions are vectorized (accept numpy arrays) and stateless.
"""

import numpy as np
import pyproj

# Module-level Geod instance (WGS84 ellipsoid)
_GEOD = pyproj.Geod(ellps="WGS84")

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in metres. Inputs in **degrees**."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing_spherical(
    lat1_rad: np.ndarray,
    lon1_rad: np.ndarray,
    lat2_rad: np.ndarray,
    lon2_rad: np.ndarray,
) -> np.ndarray:
    """Forward bearing in degrees [0, 360). Inputs in **radians**."""
    dlon = lon2_rad - lon1_rad
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def bearing_wgs84(
    lon1_deg: np.ndarray,
    lat1_deg: np.ndarray,
    lon2_deg: np.ndarray,
    lat2_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward azimuth and distance (m) on WGS84 ellipsoid. Inputs in **degrees**.

    Returns
    -------
    fwd_azimuth : ndarray  (degrees, [0, 360))
    distance_m  : ndarray
    """
    fwd, _back, dist = _GEOD.inv(lon1_deg, lat1_deg, lon2_deg, lat2_deg)
    fwd = (np.asarray(fwd, dtype=np.float64) + 360) % 360
    return fwd, np.asarray(dist, dtype=np.float64)


def shortest_turn_deg(delta: np.ndarray) -> np.ndarray:
    """Signed shortest turn in [-180, 180) from a bearing difference."""
    return (delta + 180.0) % 360.0 - 180.0


def trace_length(
    lat: np.ndarray,
    lon: np.ndarray,
) -> float:
    """Total haversine length of a GPS trace (degrees input). Returns metres."""
    if len(lat) < 2:
        return 0.0
    dists = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.nansum(dists))
