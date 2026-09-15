"""Observability diagnostics for BE-GTR reduced geometry systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_LABELS = ["p0_x", "p0_y", "p0_z", "v_x", "v_y", "v_z"]


@dataclass
class ObservabilitySummary:
    reduced_condition_number: float
    min_singular_value: float
    max_singular_value: float
    weak_direction_count: int
    top_correlation_pair: str
    top_abs_correlation: float
    covariance_diag_p0_mean: float
    covariance_diag_v_mean: float
    geometry_status: str


def summarize_observability(
    h_red: np.ndarray,
    weak_threshold_ratio: float = 1e-6,
) -> ObservabilitySummary:
    h_red = np.asarray(h_red, dtype=float)
    sym_h = 0.5 * (h_red + h_red.T)
    try:
        singular_values = np.linalg.eigvalsh(sym_h)
    except Exception:
        singular_values = np.linalg.svd(sym_h, compute_uv=False)
    singular_values = np.maximum(singular_values, 0.0)
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    min_sv = float(np.min(singular_values)) if singular_values.size else 0.0
    cond = float(max_sv / max(min_sv, 1e-18)) if max_sv > 0.0 else np.inf
    threshold = max(max_sv * weak_threshold_ratio, 1e-12)
    weak_count = int(np.sum(singular_values < threshold))
    reg = max(max_sv * 1e-10, 1e-9)
    try:
        covariance = np.linalg.pinv(sym_h + reg * np.eye(sym_h.shape[0]))
    except Exception:
        covariance = np.full_like(sym_h, np.nan)
    diag = np.diag(covariance) if covariance.ndim == 2 else np.full(6, np.nan)
    p0_mean = float(np.nanmean(diag[:3]))
    v_mean = float(np.nanmean(diag[3:6]))
    top_pair = ""
    top_abs = np.nan
    if covariance.shape == (6, 6) and np.all(np.isfinite(np.diag(covariance))):
        denom = np.sqrt(np.maximum(np.outer(np.diag(covariance), np.diag(covariance)), 1e-30))
        corr = covariance / denom
        np.fill_diagonal(corr, 0.0)
        idx = np.unravel_index(int(np.nanargmax(np.abs(corr))), corr.shape)
        top_pair = f"{STATE_LABELS[idx[0]]}-{STATE_LABELS[idx[1]]}"
        top_abs = float(abs(corr[idx]))
    if weak_count >= 3 or cond > 1e14:
        status = "poor"
    elif weak_count >= 1 or cond > 1e10:
        status = "weak"
    else:
        status = "ok"
    return ObservabilitySummary(
        reduced_condition_number=cond,
        min_singular_value=min_sv,
        max_singular_value=max_sv,
        weak_direction_count=weak_count,
        top_correlation_pair=top_pair,
        top_abs_correlation=top_abs,
        covariance_diag_p0_mean=p0_mean,
        covariance_diag_v_mean=v_mean,
        geometry_status=status,
    )


def summary_to_dict(summary: ObservabilitySummary) -> dict[str, float | int | str]:
    return {
        "reduced_condition_number": float(summary.reduced_condition_number),
        "min_singular_value": float(summary.min_singular_value),
        "max_singular_value": float(summary.max_singular_value),
        "weak_direction_count": int(summary.weak_direction_count),
        "top_correlation_pair": summary.top_correlation_pair,
        "top_abs_correlation": float(summary.top_abs_correlation),
        "covariance_diag_p0_mean": float(summary.covariance_diag_p0_mean),
        "covariance_diag_v_mean": float(summary.covariance_diag_v_mean),
        "geometry_status": summary.geometry_status,
    }
