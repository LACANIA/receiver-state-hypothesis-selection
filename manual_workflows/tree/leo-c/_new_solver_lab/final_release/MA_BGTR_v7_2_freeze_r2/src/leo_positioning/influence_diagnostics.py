"""Influence and validation diagnostics for GIR-TR and MA-BGTR-v3."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .geometry_weights import GeometryWeightDiagnostics


def robust_sigma_mad(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    if residual.size == 0 or not np.all(np.isfinite(residual)):
        return np.nan
    med = float(np.median(residual))
    return float(1.4826 * np.median(np.abs(residual - med)) + 1e-9)


def validation_residual_metrics(residual: np.ndarray, trim_fraction: float = 0.10, c_scale_mps: float = 2.0) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    if residual.size == 0 or not np.all(np.isfinite(residual)):
        return {
            "raw_validation_rmse_mps": np.nan,
            "trimmed_validation_rmse_mps": np.nan,
            "inlier_validation_rmse_mps": np.nan,
            "robust_validation_cost": np.nan,
            "tail_ratio": np.nan,
            "mad_ratio": np.nan,
        }
    raw = float(np.sqrt(np.mean(residual * residual)))
    n_trim = int(np.floor(len(residual) * float(trim_fraction)))
    ordered = residual[np.argsort(np.abs(residual))]
    trimmed = ordered[: max(len(ordered) - n_trim, 1)]
    trimmed_rmse = float(np.sqrt(np.mean(trimmed * trimmed)))
    sigma = robust_sigma_mad(residual)
    standardized = np.abs((residual - float(np.median(residual))) / max(sigma, 1e-9))
    inlier = residual[standardized <= 3.0]
    if inlier.size == 0:
        inlier = trimmed
    inlier_rmse = float(np.sqrt(np.mean(inlier * inlier)))
    scaled = residual / max(float(c_scale_mps), 1e-9)
    robust_cost = float(0.5 * c_scale_mps * c_scale_mps * np.mean(np.log1p(scaled * scaled)))
    return {
        "raw_validation_rmse_mps": raw,
        "trimmed_validation_rmse_mps": trimmed_rmse,
        "inlier_validation_rmse_mps": inlier_rmse,
        "robust_validation_cost": robust_cost,
        "tail_ratio": float(np.mean(standardized > 3.0)),
        "mad_ratio": float(raw / max(sigma, 1e-9)),
    }


def diagnostics_dict(diag: GeometryWeightDiagnostics | None) -> dict[str, float]:
    if diag is None:
        keys = [
            "robust_weight_min",
            "robust_weight_median",
            "robust_weight_max",
            "geometry_floor_min",
            "geometry_floor_median",
            "geometry_floor_max",
            "floor_active_fraction",
            "hard_outlier_fraction",
            "effective_weight_entropy",
            "information_retention_ratio",
            "leverage_max",
            "leverage_median",
            "weight_geometry_correlation",
        ]
        return {key: np.nan for key in keys}
    return {key: float(value) for key, value in asdict(diag).items()}


def summarize_downweighted_satellites(
    satellite_number: np.ndarray | None,
    weights: np.ndarray,
    max_items: int = 5,
    threshold: float = 0.2,
) -> str:
    if satellite_number is None:
        return ""
    sats = np.asarray(satellite_number)
    weights = np.asarray(weights, dtype=float)
    if sats.size != weights.size or sats.size == 0:
        return ""
    rows: list[tuple[Any, int, float]] = []
    for sat in np.unique(sats):
        mask = sats == sat
        frac = float(np.mean(weights[mask] < threshold))
        rows.append((sat, int(np.sum(mask)), frac))
    rows.sort(key=lambda item: item[2], reverse=True)
    return ";".join(f"{sat}:{count}:{frac:.2f}" for sat, count, frac in rows[:max_items])
