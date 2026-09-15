"""Coordinate utilities used by the STEP02 baselines."""

from __future__ import annotations

import numpy as np

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def llh_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """Convert WGS84 geodetic latitude/longitude/height to ECEF meters."""
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * np.cos(lon)
    y = (n + h_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def enu_axes(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local east, north, up unit vectors in ECEF coordinates."""
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=float)
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        dtype=float,
    )
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=float)
    return east, north, up
