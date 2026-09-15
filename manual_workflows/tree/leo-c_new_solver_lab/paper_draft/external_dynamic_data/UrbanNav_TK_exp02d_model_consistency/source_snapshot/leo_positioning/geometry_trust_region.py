"""Geometry-aware trust-region utilities for BE-GTR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryDiagnostics:
    singular_values: np.ndarray
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    weak_direction_count: int
    lambda_geo: float
    trust_radius_mixed: float


@dataclass
class TrustRegionStep:
    delta: np.ndarray
    predicted_reduction: float
    diagnostics: GeometryDiagnostics


def spectral_diagnostics(
    h_red: np.ndarray,
    weak_threshold_ratio: float = 1e-6,
    lambda_geo_base: float = 1.0,
    trust_radius_mixed: float = 100_000.0,
) -> GeometryDiagnostics:
    h_red = np.asarray(h_red, dtype=float)
    try:
        singular_values = np.linalg.eigvalsh(0.5 * (h_red + h_red.T))
    except Exception:
        singular_values = np.linalg.svd(h_red, compute_uv=False)
    singular_values = np.maximum(np.asarray(singular_values, dtype=float), 0.0)
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    min_sv = float(np.min(singular_values)) if singular_values.size else 0.0
    threshold = max(max_sv * float(weak_threshold_ratio), 1e-12)
    weak_count = int(np.sum(singular_values < threshold))
    cond = float(max_sv / max(min_sv, 1e-18)) if max_sv > 0.0 else np.inf
    lambda_geo = float(lambda_geo_base * (1.0 + weak_count))
    radius = float(trust_radius_mixed / (1.0 + 0.5 * weak_count))
    return GeometryDiagnostics(
        singular_values=singular_values,
        min_singular_value=min_sv,
        max_singular_value=max_sv,
        condition_number=cond,
        weak_direction_count=weak_count,
        lambda_geo=lambda_geo,
        trust_radius_mixed=radius,
    )


def geometry_regularized_step(
    h_red: np.ndarray,
    g_red: np.ndarray,
    lambda_lm: float,
    trust_radius_mixed: float,
    weak_threshold_ratio: float = 1e-6,
    lambda_geo_base: float = 1.0,
) -> TrustRegionStep:
    h_red = np.asarray(h_red, dtype=float)
    g_red = np.asarray(g_red, dtype=float)
    sym_h = 0.5 * (h_red + h_red.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(sym_h)
    except np.linalg.LinAlgError:
        u, s, _vh = np.linalg.svd(sym_h)
        eigvals, eigvecs = s, u
    eigvals = np.maximum(eigvals, 0.0)
    max_sv = float(np.max(eigvals)) if eigvals.size else 0.0
    threshold = max(max_sv * weak_threshold_ratio, 1e-12)
    weak = eigvals < threshold
    weak_count = int(np.sum(weak))
    geo_penalty = np.zeros_like(eigvals)
    if eigvals.size:
        geo_penalty[weak] = (threshold - eigvals[weak]) + threshold
    lambda_geo = lambda_geo_base * (1.0 + weak_count)
    diag_scale = np.diag(np.maximum(np.diag(sym_h), 1.0))
    geo_matrix = eigvecs @ np.diag(geo_penalty) @ eigvecs.T
    h_reg = sym_h + float(lambda_lm) * diag_scale + lambda_geo * geo_matrix
    try:
        delta = np.linalg.solve(h_reg, -g_red)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(h_reg, -g_red, rcond=None)[0]
    diag = spectral_diagnostics(sym_h, weak_threshold_ratio, lambda_geo_base, trust_radius_mixed)
    if np.linalg.norm(delta) > diag.trust_radius_mixed:
        delta = delta * (diag.trust_radius_mixed / max(np.linalg.norm(delta), 1e-15))
    predicted = float(-(g_red @ delta) - 0.5 * (delta @ (sym_h @ delta)))
    return TrustRegionStep(delta=delta, predicted_reduction=predicted, diagnostics=diag)


def update_trust_radius(current_radius: float, rho: float, weak_direction_count: int) -> float:
    if rho < 0.25:
        return max(0.25 * current_radius, 1e-3)
    if rho > 0.75 and weak_direction_count == 0:
        return min(2.0 * current_radius, 1_000_000.0)
    if rho > 0.75:
        return min(1.25 * current_radius, 250_000.0)
    return current_radius
