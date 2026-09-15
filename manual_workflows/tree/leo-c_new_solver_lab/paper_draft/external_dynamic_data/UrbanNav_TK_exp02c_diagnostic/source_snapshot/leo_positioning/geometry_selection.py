"""Geometry-preserving observation selection diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .uncertainty_diagnostics import residual_jacobian_for_row


def _condition(jacobian: np.ndarray, weights: np.ndarray | None = None) -> float:
    if jacobian.size == 0:
        return np.nan
    if weights is None:
        normal = jacobian.T @ jacobian
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        normal = jacobian.T @ (w[:, None] * jacobian)
    try:
        return float(np.linalg.cond(normal))
    except Exception:
        return np.inf


def _trace_info(jacobian: np.ndarray, weights: np.ndarray | None = None) -> float:
    if jacobian.size == 0:
        return np.nan
    if weights is None:
        return float(np.trace(jacobian.T @ jacobian))
    w = np.asarray(weights, dtype=float).reshape(-1)
    return float(np.trace(jacobian.T @ (w[:, None] * jacobian)))


def compute_geometry_selection_diagnostics(row: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    base = {
        "residual_trim_info_retention": np.nan,
        "geometry_preserved_info_retention": np.nan,
        "residual_trim_condition_change": np.nan,
        "geometry_preserved_condition_change": np.nan,
        "geometry_critical_outlier_count": 0,
        "geometry_selection_recommendation": "diagnostic_unavailable",
    }
    try:
        residual, jacobian, _weights, _pred, _family = residual_jacobian_for_row(row, obs)
        n = len(residual)
        if n < 8:
            base["geometry_selection_recommendation"] = "too_few_observations"
            return base
        full_trace = max(_trace_info(jacobian), 1e-12)
        full_condition = _condition(jacobian)
        abs_res = np.abs(residual)
        trim_count = max(1, int(np.ceil(0.10 * n)))
        trim_idx = np.argsort(-abs_res)[:trim_count]
        keep = np.ones(n, dtype=bool)
        keep[trim_idx] = False
        trim_trace = _trace_info(jacobian[keep])
        trim_condition = _condition(jacobian[keep])

        geom_contribution = np.sum(jacobian * jacobian, axis=1)
        geom_norm = geom_contribution / max(float(np.max(geom_contribution)), 1e-12)
        critical_threshold = float(np.quantile(geom_norm, 0.75))
        critical = np.zeros(n, dtype=bool)
        critical[trim_idx] = geom_norm[trim_idx] >= critical_threshold
        weights = np.ones(n, dtype=float)
        for idx in trim_idx:
            if critical[idx]:
                weights[idx] = 0.70
            else:
                weights[idx] = 0.20
        geom_trace = _trace_info(jacobian, weights)
        geom_condition = _condition(jacobian, weights)
        critical_count = int(np.sum(critical))
        trim_retention = float(trim_trace / full_trace) if np.isfinite(trim_trace) else np.nan
        geom_retention = float(geom_trace / full_trace) if np.isfinite(geom_trace) else np.nan
        recommendation = "geometry_preserved_weighting"
        if critical_count == 0 and np.isfinite(trim_retention) and trim_retention > 0.85:
            recommendation = "residual_trim_low_geometry_risk"
        elif np.isfinite(geom_retention) and geom_retention < 0.70:
            recommendation = "geometry_fragile_do_not_trim_aggressively"
        base.update(
            {
                "residual_trim_info_retention": trim_retention,
                "geometry_preserved_info_retention": geom_retention,
                "residual_trim_condition_change": float(trim_condition / full_condition) if np.isfinite(trim_condition) and np.isfinite(full_condition) and full_condition > 0.0 else np.nan,
                "geometry_preserved_condition_change": float(geom_condition / full_condition) if np.isfinite(geom_condition) and np.isfinite(full_condition) and full_condition > 0.0 else np.nan,
                "geometry_critical_outlier_count": critical_count,
                "geometry_selection_recommendation": recommendation,
            }
        )
    except Exception as exc:  # noqa: BLE001
        base["geometry_selection_recommendation"] = f"failed: {exc}"
    return base
