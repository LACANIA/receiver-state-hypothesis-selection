"""Clock-drift-aware trajectory Doppler models.

The full CTD state is:
    theta = [p0_x, p0_y, p0_z, v_x, v_y, v_z, b0, bdot]

For observation i:
    tau_i = t_i - t0
    p_i = p0 + v * tau_i
    b_i = b0 + bdot * tau_i
    u_i = (p_i - p_s_i) / ||p_i - p_s_i||
    pred_i = u_i^T (v - v_s_i) + b_i
    r_i = meas_i - pred_i

The analytic Jacobian returned here is the residual Jacobian. Therefore all
prediction derivatives have a negative sign in the residual Jacobian:
    J_p0 = -A_i w_i
    J_v = -(u_i + tau_i A_i w_i)
    J_b0 = -1
    J_bdot = -tau_i
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import cauchy_weights


@dataclass(frozen=True)
class TrajectoryModelConfig:
    estimate_b0: bool = True
    estimate_bdot: bool = True
    fixed_b0_mps: float = 0.0
    fixed_bdot_mps2: float = 0.0


FULL_CTD_CONFIG = TrajectoryModelConfig(True, True, 0.0, 0.0)
NO_BDOT_CONFIG = TrajectoryModelConfig(True, False, 0.0, 0.0)
NO_BIAS_CONFIG = TrajectoryModelConfig(False, False, 0.0, 0.0)


def state_dim(config: TrajectoryModelConfig) -> int:
    return 6 + int(config.estimate_b0) + int(config.estimate_bdot)


def pack_state(
    p0_m: np.ndarray,
    velocity_mps: np.ndarray,
    b0_mps: float = 0.0,
    bdot_mps2: float = 0.0,
    config: TrajectoryModelConfig = FULL_CTD_CONFIG,
) -> np.ndarray:
    values = [*np.asarray(p0_m, dtype=float).reshape(3), *np.asarray(velocity_mps, dtype=float).reshape(3)]
    if config.estimate_b0:
        values.append(float(b0_mps))
    if config.estimate_bdot:
        values.append(float(bdot_mps2))
    return np.array(values, dtype=float)


def unpack_state(theta: np.ndarray, config: TrajectoryModelConfig = FULL_CTD_CONFIG) -> tuple[np.ndarray, np.ndarray, float, float]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != state_dim(config):
        raise ValueError(f"Expected state dimension {state_dim(config)}, got {theta.size}.")
    p0 = theta[:3]
    velocity = theta[3:6]
    idx = 6
    if config.estimate_b0:
        b0 = float(theta[idx])
        idx += 1
    else:
        b0 = float(config.fixed_b0_mps)
    if config.estimate_bdot:
        bdot = float(theta[idx])
    else:
        bdot = float(config.fixed_bdot_mps2)
    return p0, velocity, b0, bdot


def trajectory_positions(theta: np.ndarray, time_s: np.ndarray, t0_s: float, config: TrajectoryModelConfig) -> np.ndarray:
    p0, velocity, _b0, _bdot = unpack_state(theta, config)
    tau = np.asarray(time_s, dtype=float) - float(t0_s)
    return p0[None, :] + tau[:, None] * velocity[None, :]


def trajectory_bias(theta: np.ndarray, time_s: np.ndarray, t0_s: float, config: TrajectoryModelConfig) -> np.ndarray:
    _p0, _velocity, b0, bdot = unpack_state(theta, config)
    tau = np.asarray(time_s, dtype=float) - float(t0_s)
    return b0 + bdot * tau


def predict_ctd(
    theta: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    t0_s: float,
    config: TrajectoryModelConfig = FULL_CTD_CONFIG,
) -> np.ndarray:
    p0, velocity, b0, bdot = unpack_state(theta, config)
    tau = np.asarray(time_s, dtype=float) - float(t0_s)
    p_i = p0[None, :] + tau[:, None] * velocity[None, :]
    dp = p_i - np.asarray(sat_pos_m, dtype=float)
    rho = np.maximum(np.linalg.norm(dp, axis=1), 1e-9)
    u = dp / rho[:, None]
    w = velocity[None, :] - np.asarray(sat_vel_mps, dtype=float)
    return np.sum(u * w, axis=1) + b0 + bdot * tau


def residuals_and_jacobian_ctd(
    theta: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    config: TrajectoryModelConfig = FULL_CTD_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0, velocity, b0, bdot = unpack_state(theta, config)
    tau = np.asarray(time_s, dtype=float) - float(t0_s)
    sat_pos = np.asarray(sat_pos_m, dtype=float)
    sat_vel = np.asarray(sat_vel_mps, dtype=float)
    p_i = p0[None, :] + tau[:, None] * velocity[None, :]
    dp = p_i - sat_pos
    rho = np.maximum(np.linalg.norm(dp, axis=1), 1e-9)
    u = dp / rho[:, None]
    w = velocity[None, :] - sat_vel
    dp_dot_w = np.sum(dp * w, axis=1)
    aw = w / rho[:, None] - dp * (dp_dot_w / (rho**3))[:, None]
    pred = np.sum(u * w, axis=1) + b0 + bdot * tau
    residual = np.asarray(meas_mps, dtype=float) - pred

    columns = [-aw, -(u + tau[:, None] * aw)]
    if config.estimate_b0:
        columns.append(-np.ones((len(residual), 1), dtype=float))
    if config.estimate_bdot:
        columns.append(-tau[:, None])
    jacobian = np.column_stack(columns)
    return residual, jacobian, pred


def ctd_objective(
    residual: np.ndarray,
    robust: bool = False,
    c_scale_mps: float = 2.0,
    prior_residual: np.ndarray | None = None,
) -> float:
    residual = np.asarray(residual, dtype=float)
    if robust:
        scaled = residual / float(c_scale_mps)
        cost = 0.5 * c_scale_mps * c_scale_mps * np.sum(np.log1p(scaled * scaled))
    else:
        cost = 0.5 * np.sum(residual * residual)
    if prior_residual is not None:
        pr = np.asarray(prior_residual, dtype=float)
        cost += 0.5 * np.sum(pr * pr)
    return float(cost)


def weighted_normal_condition(jacobian: np.ndarray, residual: np.ndarray, robust: bool, c_scale_mps: float = 2.0) -> float:
    weights = cauchy_weights(residual, c_scale_mps) if robust else np.ones(len(residual), dtype=float)
    normal = jacobian.T @ (weights[:, None] * jacobian)
    return float(np.linalg.cond(normal))


def append_optional_prior(
    residual: np.ndarray,
    jacobian: np.ndarray,
    theta: np.ndarray,
    prior: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a diagonal Gaussian prior if requested.

    STEP04 defaults this to disabled. The hook is here so later experiments can
    add weak motion or clock priors without changing solver code.
    """
    if not prior:
        return residual, jacobian
    mean = np.asarray(prior["mean"], dtype=float)
    sigma = np.asarray(prior["sigma"], dtype=float)
    if mean.shape != theta.shape or sigma.shape != theta.shape:
        raise ValueError("Prior mean/sigma must match theta shape.")
    prior_residual = (theta - mean) / sigma
    prior_jacobian = np.diag(1.0 / sigma)
    return np.r_[residual, prior_residual], np.vstack([jacobian, prior_jacobian])
