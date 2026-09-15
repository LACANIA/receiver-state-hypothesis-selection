"""Numerical checks for Doppler residual Jacobians."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import residuals_and_jacobian
from .be_gtr_solver import evaluate_be_gtr, h_and_jacobian_x
from .trajectory_models import FULL_CTD_CONFIG, pack_state, residuals_and_jacobian_ctd
from .variable_projection import beta_prior_profiles, bias_design_matrix, joint_linear_beta_solution


def _central_difference_jacobian(
    state: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    with_bias: bool,
    position_step_m: float = 1.0,
    bias_step_mps: float = 1e-5,
) -> np.ndarray:
    state = np.asarray(state, dtype=float).copy()
    n_state = 4 if with_bias else 3
    jac = np.zeros((len(meas_mps), n_state), dtype=float)
    for j in range(n_state):
        step = bias_step_mps if with_bias and j == 3 else position_step_m
        plus = state.copy()
        minus = state.copy()
        plus[j] += step
        minus[j] -= step
        r_plus, _j_plus, _ = residuals_and_jacobian(plus, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias)
        r_minus, _j_minus, _ = residuals_and_jacobian(minus, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias)
        jac[:, j] = (r_plus - r_minus) / (2.0 * step)
    return jac


def _compare_jacobian(
    state: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    with_bias: bool,
) -> dict[str, Any]:
    _residual, analytic, _pred = residuals_and_jacobian(
        state, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias
    )
    numerical = _central_difference_jacobian(state, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias)
    diff = analytic - numerical
    max_abs_error = float(np.max(np.abs(diff)))
    relative_error = float(np.linalg.norm(diff) / max(np.linalg.norm(numerical), 1e-15))
    return {
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
        "passed": bool(relative_error <= 1e-5),
        "sample_count": int(len(meas_mps)),
    }


def check_jacobians(obs: dict[str, Any], seed: int = 20260628, sample_count: int = 5) -> dict[str, Any]:
    """Check analytic residual Jacobians against central differences."""
    rng = np.random.default_rng(seed)
    n_obs = len(obs["meas_mps"])
    if n_obs < sample_count:
        raise ValueError(f"Need at least {sample_count} observations for Jacobian validation, got {n_obs}.")
    indices = np.sort(rng.choice(n_obs, size=sample_count, replace=False))
    sat_pos = obs["sat_pos_m"][indices]
    sat_vel = obs["sat_vel_mps"][indices]
    meas = obs["meas_mps"][indices]
    p_gt = np.asarray(obs["p_gt_ecef_m"], dtype=float)
    state_pos = p_gt.copy()
    state_bias = np.r_[p_gt, 0.25]
    position_only = _compare_jacobian(state_pos, sat_pos, sat_vel, meas, with_bias=False)
    position_bias = _compare_jacobian(state_bias, sat_pos, sat_vel, meas, with_bias=True)
    passed = bool(position_only["passed"] and position_bias["passed"])
    return {
        "seed": int(seed),
        "indices": [int(i) for i in indices],
        "position_only": position_only,
        "position_bias": position_bias,
        "passed": passed,
    }


def _central_difference_ctd(
    theta: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).copy()
    steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-5, 1e-7], dtype=float)
    jac = np.zeros((len(meas_mps), theta.size), dtype=float)
    for j, step in enumerate(steps):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        r_plus, _j_plus, _ = residuals_and_jacobian_ctd(
            plus, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, FULL_CTD_CONFIG
        )
        r_minus, _j_minus, _ = residuals_and_jacobian_ctd(
            minus, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, FULL_CTD_CONFIG
        )
        jac[:, j] = (r_plus - r_minus) / (2.0 * step)
    return jac


def check_ctd_jacobian(obs: dict[str, Any], seed: int = 20260628, sample_count: int = 12) -> dict[str, Any]:
    """Check the 8-D CTD residual Jacobian against central differences."""
    rng = np.random.default_rng(seed)
    n_obs = len(obs["meas_mps"])
    if n_obs < sample_count:
        raise ValueError(f"Need at least {sample_count} observations for CTD Jacobian validation, got {n_obs}.")
    indices = np.sort(rng.choice(n_obs, size=sample_count, replace=False))
    time_s = np.asarray(obs["time_s"], dtype=float)[indices]
    sat_pos = obs["sat_pos_m"][indices]
    sat_vel = obs["sat_vel_mps"][indices]
    meas = obs["meas_mps"][indices]
    t0_s = float(np.min(np.asarray(obs["time_s"], dtype=float)))
    p0 = np.asarray(obs.get("p0_true_m", obs["p_gt_ecef_m"]), dtype=float)
    v = np.asarray(obs.get("v_true_mps", np.array([3.0, -2.0, 1.0])), dtype=float)
    theta = pack_state(p0 + np.array([10.0, -20.0, 5.0]), v + np.array([0.2, -0.1, 0.05]), 0.3, 0.01)
    _residual, analytic, _pred = residuals_and_jacobian_ctd(
        theta, time_s, sat_pos, sat_vel, meas, t0_s, FULL_CTD_CONFIG
    )
    numerical = _central_difference_ctd(theta, time_s, sat_pos, sat_vel, meas, t0_s)
    diff = analytic - numerical
    max_abs_error = float(np.max(np.abs(diff)))
    mean_abs_error = float(np.mean(np.abs(diff)))
    relative_error = float(np.linalg.norm(diff) / max(np.linalg.norm(numerical), 1e-15))
    return {
        "seed": int(seed),
        "indices": [int(i) for i in indices],
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_error": relative_error,
        "passed": bool(relative_error <= 1e-5),
    }


def _central_difference_h_x(
    x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    t0_s: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=float)
    jac = np.zeros((len(time_s), 6), dtype=float)
    for j, step in enumerate(steps):
        plus = x.copy()
        minus = x.copy()
        plus[j] += step
        minus[j] -= step
        h_plus, _ = h_and_jacobian_x(plus, time_s, sat_pos_m, sat_vel_mps, t0_s)
        h_minus, _ = h_and_jacobian_x(minus, time_s, sat_pos_m, sat_vel_mps, t0_s)
        # h_and_jacobian_x returns residual Jacobian, so numerical residual
        # Jacobian for a = y - h is -dh/dx.
        jac[:, j] = -(h_plus - h_minus) / (2.0 * step)
    return jac


def _central_difference_reduced(
    x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    weights: np.ndarray,
    profile_name: str,
    projection_mode: str = "full",
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    profile = beta_prior_profiles()[profile_name]
    steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=float)
    jac = np.zeros((len(time_s), 6), dtype=float)
    for j, step in enumerate(steps):
        plus = x.copy()
        minus = x.copy()
        plus[j] += step
        minus[j] -= step
        eval_plus = evaluate_be_gtr(
            plus,
            time_s,
            sat_pos_m,
            sat_vel_mps,
            meas_mps,
            t0_s,
            weights,
            profile,
            projection_mode=projection_mode,
            include_prior_cost=True,
        )
        eval_minus = evaluate_be_gtr(
            minus,
            time_s,
            sat_pos_m,
            sat_vel_mps,
            meas_mps,
            t0_s,
            weights,
            profile,
            projection_mode=projection_mode,
            include_prior_cost=True,
        )
        jac[:, j] = (eval_plus.reduced_residual - eval_minus.reduced_residual) / (2.0 * step)
    return jac


def check_be_gtr_projection(
    obs: dict[str, Any],
    seed: int = 20260628,
    sample_count: int = 12,
    profile_name: str = "B0_none",
) -> dict[str, Any]:
    """Run BE-GTR variable projection and projected-Jacobian validations."""
    rng = np.random.default_rng(seed)
    n_obs = len(obs["meas_mps"])
    if n_obs < sample_count:
        raise ValueError(f"Need at least {sample_count} observations, got {n_obs}.")
    indices = np.sort(rng.choice(n_obs, size=sample_count, replace=False))
    time_s = np.asarray(obs["time_s"], dtype=float)[indices]
    sat_pos = obs["sat_pos_m"][indices]
    sat_vel = obs["sat_vel_mps"][indices]
    meas = obs["meas_mps"][indices]
    t0_s = float(np.min(np.asarray(obs["time_s"], dtype=float)))
    p0 = np.asarray(obs.get("p0_true_m", obs["p_gt_ecef_m"]), dtype=float)
    v = np.asarray(obs.get("v_true_mps", np.zeros(3)), dtype=float)
    x = np.r_[p0 + np.array([10.0, -20.0, 5.0]), v + np.array([0.2, -0.1, 0.05])]
    weights = np.ones(len(time_s), dtype=float)
    profile = beta_prior_profiles()[profile_name]

    _h, analytic_jx = h_and_jacobian_x(x, time_s, sat_pos, sat_vel, t0_s)
    numerical_jx = _central_difference_h_x(x, time_s, sat_pos, sat_vel, t0_s)
    jx_diff = analytic_jx - numerical_jx
    jx_max = float(np.max(np.abs(jx_diff)))
    jx_mean = float(np.mean(np.abs(jx_diff)))
    jx_rel = float(np.linalg.norm(jx_diff) / max(np.linalg.norm(numerical_jx), 1e-15))

    evaluation = evaluate_be_gtr(x, time_s, sat_pos, sat_vel, meas, t0_s, weights, profile)
    b = bias_design_matrix(time_s, t0_s)
    joint_beta = joint_linear_beta_solution(evaluation.residual_no_bias, b, weights, profile)
    beta_diff = float(np.linalg.norm(joint_beta - evaluation.beta))

    numerical_jred = _central_difference_reduced(x, time_s, sat_pos, sat_vel, meas, t0_s, weights, profile_name)
    red_diff = evaluation.reduced_jacobian - numerical_jred
    red_max = float(np.max(np.abs(red_diff)))
    red_mean = float(np.mean(np.abs(red_diff)))
    red_rel = float(np.linalg.norm(red_diff) / max(np.linalg.norm(numerical_jred), 1e-15))

    ortho = b.T @ (weights * evaluation.reduced_residual)
    ortho_inf = float(np.linalg.norm(ortho, ord=np.inf))
    return {
        "seed": int(seed),
        "indices": [int(i) for i in indices],
        "profile_name": profile_name,
        "h_jacobian": {
            "max_abs_error": jx_max,
            "mean_abs_error": jx_mean,
            "relative_error": jx_rel,
            "passed": bool(jx_rel <= 1e-5),
        },
        "beta_equivalence": {
            "joint_vs_projected_beta_difference_norm": beta_diff,
            "passed": bool(beta_diff <= 1e-8),
        },
        "reduced_jacobian": {
            "max_abs_error": red_max,
            "mean_abs_error": red_mean,
            "relative_error": red_rel,
            "passed": bool(red_rel <= 1e-5),
        },
        "weighted_orthogonality": {
            "projection_weighted_orthogonality_error": ortho_inf,
            "passed": bool(ortho_inf <= 1e-7),
        },
        "beta_normal_condition_number": float(evaluation.projection.beta_normal_condition),
        "passed": bool(jx_rel <= 1e-5 and beta_diff <= 1e-8 and red_rel <= 1e-5 and ortho_inf <= 1e-7),
    }


def _central_difference_objective(
    x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    weights: np.ndarray,
    profile_name: str,
    projection_mode: str,
) -> np.ndarray:
    profile = beta_prior_profiles()[profile_name]
    x = np.asarray(x, dtype=float).copy()
    steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=float)
    grad = np.zeros(6, dtype=float)
    for j, step in enumerate(steps):
        plus = x.copy()
        minus = x.copy()
        plus[j] += step
        minus[j] -= step
        eval_plus = evaluate_be_gtr(
            plus,
            time_s,
            sat_pos_m,
            sat_vel_mps,
            meas_mps,
            t0_s,
            weights,
            profile,
            projection_mode=projection_mode,
            include_prior_cost=True,
        )
        eval_minus = evaluate_be_gtr(
            minus,
            time_s,
            sat_pos_m,
            sat_vel_mps,
            meas_mps,
            t0_s,
            weights,
            profile,
            projection_mode=projection_mode,
            include_prior_cost=True,
        )
        grad[j] = (eval_plus.objective - eval_minus.objective) / (2.0 * step)
    return grad


def check_step05b_projection_repair(
    obs: dict[str, Any],
    seed: int = 20260628,
    sample_count: int = 12,
) -> dict[str, Any]:
    """Validate b0-only/full projection and constrained reduced objective."""
    rng = np.random.default_rng(seed)
    n_obs = len(obs["meas_mps"])
    if n_obs < sample_count:
        raise ValueError(f"Need at least {sample_count} observations, got {n_obs}.")
    indices = np.sort(rng.choice(n_obs, size=sample_count, replace=False))
    time_s = np.asarray(obs["time_s"], dtype=float)[indices]
    sat_pos = obs["sat_pos_m"][indices]
    sat_vel = obs["sat_vel_mps"][indices]
    meas = obs["meas_mps"][indices]
    t0_s = float(np.min(np.asarray(obs["time_s"], dtype=float)))
    p0 = np.asarray(obs.get("p0_true_m", obs["p_gt_ecef_m"]), dtype=float)
    v = np.asarray(obs.get("v_true_mps", np.zeros(3)), dtype=float)
    x = np.r_[p0 + np.array([10.0, -20.0, 5.0]), v + np.array([0.2, -0.1, 0.05])]
    weights = np.ones(len(time_s), dtype=float)
    out: dict[str, Any] = {"seed": int(seed), "indices": [int(i) for i in indices]}

    for mode in ["b0_only", "full"]:
        profile_name = "B1_weak" if mode == "full" else "B1_weak"
        profile = beta_prior_profiles()[profile_name]
        evaluation = evaluate_be_gtr(
            x,
            time_s,
            sat_pos,
            sat_vel,
            meas,
            t0_s,
            weights,
            profile,
            projection_mode=mode,
            include_prior_cost=True,
        )
        numerical = _central_difference_reduced(
            x, time_s, sat_pos, sat_vel, meas, t0_s, weights, profile_name, projection_mode=mode
        )
        diff = evaluation.reduced_jacobian - numerical
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(numerical), 1e-15))
        out[f"{mode}_reduced_jacobian"] = {
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "relative_error": rel,
            "passed": bool(rel <= 1e-5),
        }

    profile = beta_prior_profiles()["B1_weak"]
    evaluation = evaluate_be_gtr(
        x,
        time_s,
        sat_pos,
        sat_vel,
        meas,
        t0_s,
        weights,
        profile,
        projection_mode="full",
        include_prior_cost=True,
    )
    _h, j_x = h_and_jacobian_x(x, time_s, sat_pos, sat_vel, t0_s)
    analytic_grad = j_x.T @ (weights * evaluation.reduced_residual)
    numerical_grad = _central_difference_objective(
        x, time_s, sat_pos, sat_vel, meas, t0_s, weights, "B1_weak", "full"
    )
    grad_diff = analytic_grad - numerical_grad
    grad_rel = float(np.linalg.norm(grad_diff) / max(np.linalg.norm(numerical_grad), 1e-15))
    out["constrained_full_objective_gradient"] = {
        "max_abs_error": float(np.max(np.abs(grad_diff))),
        "mean_abs_error": float(np.mean(np.abs(grad_diff))),
        "relative_error": grad_rel,
        "passed": bool(grad_rel <= 1e-5),
    }
    out["passed"] = bool(
        out["b0_only_reduced_jacobian"]["passed"]
        and out["full_reduced_jacobian"]["passed"]
        and out["constrained_full_objective_gradient"]["passed"]
    )
    return out
