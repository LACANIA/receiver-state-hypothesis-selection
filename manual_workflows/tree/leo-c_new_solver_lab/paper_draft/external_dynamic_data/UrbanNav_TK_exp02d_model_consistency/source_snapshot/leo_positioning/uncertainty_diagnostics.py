"""Uncertainty proxies for MA-BGTR-v5 candidate rows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import cauchy_weights, residuals_and_jacobian
from .trajectory_models import (
    FULL_CTD_CONFIG,
    NO_BDOT_CONFIG,
    NO_BIAS_CONFIG,
    TrajectoryModelConfig,
    pack_state,
    residuals_and_jacobian_ctd,
)


ROBUST_MODELS = {
    "M7_robust_ctd_full",
    "M8_robust_be_b0_only",
    "M9_robust_be_full",
    "M10_gir_tr_fixed_scale",
    "M11_gir_tr_mad_scale",
    "M13_robust_ctd_full_plus_gir_refine",
}


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _final_position(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            _as_float(row.get("final_ecef_x_m")),
            _as_float(row.get("final_ecef_y_m")),
            _as_float(row.get("final_ecef_z_m")),
        ],
        dtype=float,
    )


def _velocity(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            _as_float(row.get("estimated_vx_mps"), 0.0),
            _as_float(row.get("estimated_vy_mps"), 0.0),
            _as_float(row.get("estimated_vz_mps"), 0.0),
        ],
        dtype=float,
    )


def _dynamic_config(model: str) -> TrajectoryModelConfig:
    if model == "M3_ctd_no_drift":
        return NO_BDOT_CONFIG
    if model == "M4_ctd_no_bias":
        return NO_BIAS_CONFIG
    return FULL_CTD_CONFIG


def dynamic_state_from_row(row: dict[str, Any], obs: dict[str, Any], config: TrajectoryModelConfig | None = None) -> np.ndarray:
    """Reconstruct a trajectory state from persisted STEP09 row fields.

    STEP09 stores the final trajectory position. For dynamic diagnostics we
    recover p0 by backing out the estimated constant velocity over the observed
    time span.
    """
    cfg = config or _dynamic_config(str(row.get("candidate_model", "")))
    p_final = _final_position(row)
    v = _velocity(row)
    if not np.all(np.isfinite(p_final)):
        raise ValueError("candidate row has no finite final ECEF position")
    t0 = float(obs.get("t0_s", np.min(obs["time_s"])))
    tau_end = float(np.max(obs["time_s"]) - t0)
    p0 = p_final - tau_end * v
    b0 = _as_float(row.get("beta0_estimated_mps"), 0.0)
    bdot = _as_float(row.get("beta_dot_estimated_mps2"), 0.0)
    return pack_state(p0, v, b0, bdot, cfg)


def residual_jacobian_for_row(row: dict[str, Any], obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return residual, Jacobian, weights, prediction, and state family."""
    model = str(row.get("candidate_model", ""))
    meas = np.asarray(obs["meas_mps"], dtype=float)
    sat_pos = np.asarray(obs["sat_pos_m"], dtype=float)
    sat_vel = np.asarray(obs["sat_vel_mps"], dtype=float)

    if model in {"M0_static_position", "M1_static_position_bias"}:
        p = _final_position(row)
        if not np.all(np.isfinite(p)):
            raise ValueError("static candidate row has no finite final ECEF position")
        with_bias = model == "M1_static_position_bias"
        if with_bias:
            state = np.r_[p, _as_float(row.get("beta0_estimated_mps"), 0.0)]
        else:
            state = p
        residual, jacobian, pred = residuals_and_jacobian(state, sat_pos, sat_vel, meas, with_bias=with_bias)
        weights = np.ones(len(residual), dtype=float)
        return residual, jacobian, weights, pred, "static"

    cfg = _dynamic_config(model)
    theta = dynamic_state_from_row(row, obs, cfg)
    t0 = float(obs.get("t0_s", np.min(obs["time_s"])))
    residual, jacobian, pred = residuals_and_jacobian_ctd(theta, obs["time_s"], sat_pos, sat_vel, meas, t0, cfg)
    weights = cauchy_weights(residual, 2.0) if model in ROBUST_MODELS else np.ones(len(residual), dtype=float)
    return residual, jacobian, weights, pred, "trajectory"


def covariance_diagnostics_from_jacobian(
    residual: np.ndarray,
    jacobian: np.ndarray,
    weights: np.ndarray,
    family: str,
    tau_end_s: float = 0.0,
) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if jacobian.ndim != 2 or jacobian.size == 0 or len(residual) != jacobian.shape[0]:
        raise ValueError("invalid residual/Jacobian shape")
    normal = jacobian.T @ (weights[:, None] * jacobian)
    singular_values = np.linalg.svd(normal, compute_uv=False)
    finite_s = singular_values[np.isfinite(singular_values)]
    max_s = float(np.max(finite_s)) if finite_s.size else np.nan
    min_s = float(np.min(finite_s)) if finite_s.size else np.nan
    cond = float(max_s / min_s) if np.isfinite(max_s) and np.isfinite(min_s) and min_s > 0.0 else np.inf
    rank_tol = max(max_s if np.isfinite(max_s) else 1.0, 1.0) * 1e-10
    effective_rank = int(np.sum(singular_values > rank_tol)) if finite_s.size else 0
    covariance_pinv_used = False
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(normal, rcond=1e-10)
        covariance_pinv_used = True
    dof = max(len(residual) - jacobian.shape[1], 1)
    sigma2 = float(np.sum(weights * residual * residual) / dof)
    covariance = covariance * max(sigma2, 1e-12)

    if family == "trajectory" and covariance.shape[0] >= 6:
        cov_p0 = covariance[:3, :3]
        cov_v = covariance[3:6, 3:6]
        cov_pv = covariance[:3, 3:6]
        cov_final = cov_p0 + tau_end_s * (cov_pv + cov_pv.T) + (tau_end_s**2) * cov_v
        pos_cov_trace = float(max(np.trace(cov_final), 0.0))
        vel_cov_trace = float(max(np.trace(cov_v), 0.0))
        bias_cov_trace = float(max(np.trace(covariance[6:, 6:]), 0.0)) if covariance.shape[0] > 6 else np.nan
    else:
        pos_cov_trace = float(max(np.trace(covariance[:3, :3]), 0.0)) if covariance.shape[0] >= 3 else np.nan
        vel_cov_trace = np.nan
        bias_cov_trace = float(max(np.trace(covariance[3:, 3:]), 0.0)) if covariance.shape[0] > 3 else np.nan

    return {
        "normal_condition_number": cond,
        "covariance_trace": float(np.trace(covariance)) if covariance.size else np.nan,
        "position_cov_trace": pos_cov_trace,
        "position_cov_sqrt_trace_m": float(np.sqrt(pos_cov_trace)) if np.isfinite(pos_cov_trace) else np.nan,
        "velocity_cov_trace": vel_cov_trace,
        "velocity_cov_sqrt_trace_mps": float(np.sqrt(vel_cov_trace)) if np.isfinite(vel_cov_trace) else np.nan,
        "bias_cov_trace": bias_cov_trace,
        "min_singular_value": min_s,
        "max_singular_value": max_s,
        "effective_rank": effective_rank,
        "covariance_available": bool(np.all(np.isfinite(covariance))),
        "covariance_pinv_used": bool(covariance_pinv_used or not np.isfinite(cond)),
    }


def compute_uncertainty_diagnostics(row: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    base = {
        "normal_condition_number": np.nan,
        "covariance_trace": np.nan,
        "position_cov_trace": np.nan,
        "position_cov_sqrt_trace_m": np.nan,
        "velocity_cov_trace": np.nan,
        "velocity_cov_sqrt_trace_mps": np.nan,
        "bias_cov_trace": np.nan,
        "min_singular_value": np.nan,
        "max_singular_value": np.nan,
        "effective_rank": 0,
        "covariance_available": False,
        "covariance_pinv_used": False,
    }
    try:
        residual, jacobian, weights, _pred, family = residual_jacobian_for_row(row, obs)
        tau_end = float(np.max(obs["time_s"]) - float(obs.get("t0_s", np.min(obs["time_s"]))))
        base.update(covariance_diagnostics_from_jacobian(residual, jacobian, weights, family, tau_end))
    except Exception:
        pass
    return base
