"""Ephemeris covariance approximations for Doppler risk scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from .uncertainty_diagnostics import _velocity, dynamic_state_from_row
from .trajectory_models import FULL_CTD_CONFIG, unpack_state


def _scenario_sigmas(obs: dict[str, Any]) -> tuple[float, float]:
    cfg = obs.get("scenario_config", {}) or {}
    return float(cfg.get("sat_pos_sigma_m", 0.0) or 0.0), float(cfg.get("sat_vel_sigma_mps", 0.0) or 0.0)


def _trajectory_positions_from_row(row: dict[str, Any], obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = str(row.get("candidate_model", ""))
    time_s = np.asarray(obs["time_s"], dtype=float)
    if model in {"M0_static_position", "M1_static_position_bias"}:
        p = np.array(
            [float(row.get("final_ecef_x_m", np.nan)), float(row.get("final_ecef_y_m", np.nan)), float(row.get("final_ecef_z_m", np.nan))],
            dtype=float,
        )
        if not np.all(np.isfinite(p)):
            raise ValueError("missing final ECEF")
        return np.repeat(p[None, :], len(time_s), axis=0), np.zeros(3, dtype=float)
    theta = dynamic_state_from_row(row, obs, FULL_CTD_CONFIG)
    p0, velocity, _b0, _bdot = unpack_state(theta, FULL_CTD_CONFIG)
    t0 = float(obs.get("t0_s", np.min(time_s)))
    tau = time_s - t0
    return p0[None, :] + tau[:, None] * velocity[None, :], velocity


def ephemeris_variance_for_row(row: dict[str, Any], obs: dict[str, Any]) -> np.ndarray:
    pos_sigma_m, vel_sigma_mps = _scenario_sigmas(obs)
    n = len(obs["time_s"])
    if pos_sigma_m <= 0.0 and vel_sigma_mps <= 0.0:
        return np.zeros(n, dtype=float)
    receiver_pos, receiver_vel = _trajectory_positions_from_row(row, obs)
    sat_pos = np.asarray(obs["sat_pos_m"], dtype=float)
    sat_vel = np.asarray(obs["sat_vel_mps"], dtype=float)
    dp = receiver_pos - sat_pos
    rho = np.maximum(np.linalg.norm(dp, axis=1), 1e-9)
    u = dp / rho[:, None]
    w = receiver_vel[None, :] - sat_vel
    dp_dot_w = np.sum(dp * w, axis=1)
    # pred = u^T(v-vs). Grad wrt sat position is -A w.
    a_w = w / rho[:, None] - dp * (dp_dot_w / (rho**3))[:, None]
    grad_sat_pos = -a_w
    grad_sat_vel = -u
    return (pos_sigma_m**2) * np.sum(grad_sat_pos * grad_sat_pos, axis=1) + (vel_sigma_mps**2) * np.sum(grad_sat_vel * grad_sat_vel, axis=1)


def compute_ephemeris_diagnostics(row: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    try:
        r_eph = ephemeris_variance_for_row(row, obs)
        residual_rmse = float(row.get("full_residual_rmse_mps", row.get("residual_rmse_mps", np.nan)))
        meas_var = max(residual_rmse * residual_rmse if np.isfinite(residual_rmse) else 0.04, 0.04)
        weights = 1.0 / (meas_var + r_eph)
        base_weight = 1.0 / meas_var
        effect = float(np.median(weights / base_weight)) if len(weights) else np.nan
        median = float(np.median(r_eph)) if len(r_eph) else np.nan
        p95 = float(np.quantile(r_eph, 0.95)) if len(r_eph) else np.nan
        status = "nonzero_ephemeris_covariance" if (median > 0.0 or p95 > 0.0) else "zero_or_not_configured"
    except Exception as exc:  # noqa: BLE001
        median = np.nan
        p95 = np.nan
        effect = np.nan
        status = f"failed: {exc}"
    return {
        "median_R_eph": median,
        "p95_R_eph": p95,
        "ephemeris_weight_effect": effect,
        "ephemeris_sensitivity_status": status,
    }
