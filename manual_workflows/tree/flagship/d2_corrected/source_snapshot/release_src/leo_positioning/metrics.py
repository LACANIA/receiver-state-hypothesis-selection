"""Metrics for STEP02/STEP03 positioning baseline experiments."""

from __future__ import annotations

import numpy as np

from .coordinates import enu_axes


def position_errors(
    estimate_ecef_m: np.ndarray,
    truth_ecef_m: np.ndarray,
    truth_lat_deg: float,
    truth_lon_deg: float,
) -> tuple[float, float]:
    diff = np.asarray(estimate_ecef_m, dtype=float).reshape(3) - np.asarray(truth_ecef_m, dtype=float).reshape(3)
    err_3d = float(np.linalg.norm(diff))
    east, north, _up = enu_axes(truth_lat_deg, truth_lon_deg)
    e = float(east @ diff)
    n = float(north @ diff)
    horizontal = float(np.sqrt(e * e + n * n))
    return err_3d, horizontal


def residual_metrics(residual_mps: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(residual_mps, dtype=float)
    if residual.size == 0 or not np.all(np.isfinite(residual)):
        return np.nan, np.nan
    rmse = float(np.sqrt(np.mean(residual * residual)))
    median_abs = float(np.median(np.abs(residual)))
    return rmse, median_abs


def finite_or_penalty(values: np.ndarray, penalty: float = 1e7) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, float(penalty))
