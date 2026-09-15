"""Geometry-preserving robust weights for GIR-TR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryWeightDiagnostics:
    robust_weight_min: float
    robust_weight_median: float
    robust_weight_max: float
    geometry_floor_min: float
    geometry_floor_median: float
    geometry_floor_max: float
    floor_active_fraction: float
    hard_outlier_fraction: float
    effective_weight_entropy: float
    information_retention_ratio: float
    leverage_max: float
    leverage_median: float
    weight_geometry_correlation: float


def cauchy_weight(residual: np.ndarray, c_scale_mps: float) -> np.ndarray:
    scaled = np.asarray(residual, dtype=float) / max(float(c_scale_mps), 1e-9)
    return 1.0 / (1.0 + scaled * scaled)


def leverage_scores(jacobian: np.ndarray, weights: np.ndarray | None = None, damping: float = 1e-6) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float)
    if jac.size == 0:
        return np.zeros(jac.shape[0], dtype=float)
    w = np.ones(jac.shape[0], dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    normal = jac.T @ (w[:, None] * jac)
    scale = max(float(np.trace(normal) / max(normal.shape[0], 1)), 1.0)
    normal = normal + float(damping) * scale * np.eye(normal.shape[0])
    try:
        inv_normal = np.linalg.pinv(normal)
        values = np.einsum("ij,jk,ik->i", jac, inv_normal, w[:, None] * jac)
    except Exception:
        values = np.sum(jac * jac, axis=1)
    return np.maximum(values, 0.0)


def dopt_proxy(jacobian: np.ndarray) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float)
    if jac.size == 0:
        return np.zeros(jac.shape[0], dtype=float)
    return np.sum(jac * jac, axis=1)


def normalized_geometry_contribution(jacobian: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    lev = leverage_scores(jacobian, weights=weights)
    info = dopt_proxy(jacobian)
    def norm(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return np.zeros_like(arr)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        maxv = float(np.max(arr))
        return arr / max(maxv, 1e-12)

    combined = 0.5 * norm(lev) + 0.5 * norm(info)
    return np.clip(combined, 0.0, 1.0)


def effective_weight_entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0 or w.size == 0:
        return 0.0
    p = w / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)) / max(np.log(len(w)), 1e-12))


def build_geometry_preserving_weights(
    residual: np.ndarray,
    jacobian: np.ndarray,
    c_scale_mps: float,
    floor_base: float = 0.02,
    floor_gain: float = 0.20,
    hard_outlier_threshold_mps: float = 10.0,
    hard_outlier_cap: float = 0.05,
) -> tuple[np.ndarray, GeometryWeightDiagnostics, np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=float)
    jac = np.asarray(jacobian, dtype=float)
    robust = cauchy_weight(residual, c_scale_mps)
    geometry = normalized_geometry_contribution(jac, weights=robust)
    floor = float(floor_base) + float(floor_gain) * geometry
    weights = np.maximum(robust, floor)
    hard_mask = np.abs(residual) > float(hard_outlier_threshold_mps)
    weights[hard_mask] = np.minimum(weights[hard_mask], float(hard_outlier_cap))
    floor_active = weights > robust + 1e-12
    full_trace = float(np.trace(jac.T @ jac))
    weighted_trace = float(np.trace(jac.T @ (weights[:, None] * jac)))
    retention = weighted_trace / max(full_trace, 1e-30)
    lev = leverage_scores(jac, weights=weights)
    if np.std(weights) > 0.0 and np.std(geometry) > 0.0:
        corr = float(np.corrcoef(weights, geometry)[0, 1])
    else:
        corr = np.nan
    diag = GeometryWeightDiagnostics(
        robust_weight_min=float(np.min(robust)) if robust.size else np.nan,
        robust_weight_median=float(np.median(robust)) if robust.size else np.nan,
        robust_weight_max=float(np.max(robust)) if robust.size else np.nan,
        geometry_floor_min=float(np.min(floor)) if floor.size else np.nan,
        geometry_floor_median=float(np.median(floor)) if floor.size else np.nan,
        geometry_floor_max=float(np.max(floor)) if floor.size else np.nan,
        floor_active_fraction=float(np.mean(floor_active)) if floor_active.size else np.nan,
        hard_outlier_fraction=float(np.mean(hard_mask)) if hard_mask.size else np.nan,
        effective_weight_entropy=effective_weight_entropy(weights),
        information_retention_ratio=float(retention),
        leverage_max=float(np.max(lev)) if lev.size else np.nan,
        leverage_median=float(np.median(lev)) if lev.size else np.nan,
        weight_geometry_correlation=corr,
    )
    return weights, diag, robust, floor
