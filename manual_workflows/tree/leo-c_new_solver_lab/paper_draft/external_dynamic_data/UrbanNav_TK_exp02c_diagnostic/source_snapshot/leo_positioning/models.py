"""Doppler residual models and Jacobians.

Residual convention:
    r_i = meas_i - pred_i

Position-only prediction:
    pred_i(p) = ((p - p_s_i)^T (v_r - v_s_i)) / ||p - p_s_i||

Position-bias prediction:
    pred_i(p, b) = pred_i(p) + b

With this residual definition, the position Jacobian row is:
    J_p = - (v_r - v_s_i)^T [I/rho - (dp dp^T)/rho^3]

For a common Doppler bias b in m/s:
    J_b = -1
"""

from __future__ import annotations

import numpy as np


def predict_range_rate(
    position_m: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    receiver_vel_mps: np.ndarray | None = None,
    bias_mps: float = 0.0,
) -> np.ndarray:
    p = np.asarray(position_m, dtype=float).reshape(3)
    vr = np.zeros(3, dtype=float) if receiver_vel_mps is None else np.asarray(receiver_vel_mps, dtype=float).reshape(3)
    dp = p[None, :] - sat_pos_m
    dv = vr[None, :] - sat_vel_mps
    rho = np.linalg.norm(dp, axis=1)
    rho = np.maximum(rho, 1e-9)
    return np.sum(dp * dv, axis=1) / rho + float(bias_mps)


def residuals_and_jacobian(
    state: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    with_bias: bool,
    receiver_vel_mps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(state, dtype=float)
    p = x[:3]
    bias = float(x[3]) if with_bias else 0.0
    vr = np.zeros(3, dtype=float) if receiver_vel_mps is None else np.asarray(receiver_vel_mps, dtype=float).reshape(3)

    dp = p[None, :] - sat_pos_m
    dv = vr[None, :] - sat_vel_mps
    rho = np.linalg.norm(dp, axis=1)
    rho = np.maximum(rho, 1e-9)
    dot = np.sum(dp * dv, axis=1)
    pred = dot / rho + bias
    residual = meas_mps - pred

    grad_pred = dv / rho[:, None] - dp * (dot / (rho**3))[:, None]
    jac_pos = -grad_pred
    if with_bias:
        jac = np.column_stack([jac_pos, -np.ones(len(residual), dtype=float)])
    else:
        jac = jac_pos
    return residual, jac, pred


def cauchy_weights(residual: np.ndarray, c_scale_mps: float = 2.0) -> np.ndarray:
    scaled = residual / float(c_scale_mps)
    return 1.0 / (1.0 + scaled * scaled)


def objective(residual: np.ndarray, robust: bool, c_scale_mps: float = 2.0) -> float:
    residual = np.asarray(residual, dtype=float)
    if robust:
        scaled = residual / float(c_scale_mps)
        return float(0.5 * c_scale_mps * c_scale_mps * np.sum(np.log1p(scaled * scaled)))
    return float(0.5 * np.sum(residual * residual))


def normal_matrix_condition(jacobian: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Condition number of the weighted normal matrix used by a solver."""
    if weights is None:
        normal = jacobian.T @ jacobian
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        normal = jacobian.T @ (w[:, None] * jacobian)
    return float(np.linalg.cond(normal))
