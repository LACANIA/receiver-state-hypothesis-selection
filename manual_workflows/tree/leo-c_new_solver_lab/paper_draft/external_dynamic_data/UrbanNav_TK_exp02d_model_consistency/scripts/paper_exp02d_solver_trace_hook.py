"""Read-only tracing helpers for PAPER-EXP02D.

These routines duplicate frozen LM arithmetic only to expose initialization,
the first trial/accepted step, and fixed-parameter diagnostic ablations. They
never replace the frozen candidate result and are not part of M0--M14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Array = np.ndarray


def _solve_linear(lhs: Array, rhs: Array) -> Array:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _rmse(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values))) if values.size else float("nan")


def trace_ctd_first_accept(
    initial_state: Array,
    time_s: Array,
    sat_pos_m: Array,
    sat_vel_mps: Array,
    meas_mps: Array,
    t0_s: float,
    config: Any,
    residual_jacobian: Callable[..., tuple[Array, Array, Array]],
    objective: Callable[..., float],
    cauchy_weights: Callable[..., Array],
    robust: bool = False,
    max_iter: int = 80,
    lambda0: float = 1e-3,
    c_scale_mps: float = 2.0,
) -> dict[str, Any]:
    """Replay frozen CTD LM arithmetic until its first accepted update."""
    theta = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    actual_initial = theta.copy()
    lam = float(lambda0)
    r0, j0, _ = residual_jacobian(theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config)
    current_obj = float(objective(r0, robust=robust, c_scale_mps=c_scale_mps))
    first_trial_state: Array | None = None
    first_trial_step_norm = float("nan")
    first_trial_objective = float("nan")
    first_trial_accepted = False
    first_accepted_state: Array | None = None
    first_accepted_step_norm = float("nan")
    first_accepted_iteration: int | None = None

    for iteration in range(1, int(max_iter) + 1):
        residual, jacobian, _ = residual_jacobian(
            theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config
        )
        weights = cauchy_weights(residual, c_scale_mps) if robust else np.ones(len(residual), dtype=float)
        normal = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * residual)
        diag_scale = np.diag(np.maximum(np.diag(normal), 1.0))
        delta = _solve_linear(normal + lam * diag_scale, -gradient)
        candidate = theta + delta
        candidate_residual, _candidate_jacobian, _ = residual_jacobian(
            candidate, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config
        )
        candidate_obj = float(objective(candidate_residual, robust=robust, c_scale_mps=c_scale_mps))
        accepted = bool(np.isfinite(candidate_obj) and candidate_obj < current_obj)
        if first_trial_state is None:
            first_trial_state = candidate.copy()
            first_trial_step_norm = float(np.linalg.norm(delta))
            first_trial_objective = candidate_obj
            first_trial_accepted = accepted
        if accepted:
            theta = candidate
            current_obj = candidate_obj
            first_accepted_state = theta.copy()
            first_accepted_step_norm = float(np.linalg.norm(delta))
            first_accepted_iteration = iteration
            break
        lam = min(lam * 10.0, 1e12)

    return {
        "hook_scope": "diagnostic_only_read_only_replay",
        "actual_initial_state": actual_initial,
        "first_objective": float(objective(r0, robust=robust, c_scale_mps=c_scale_mps)),
        "first_residual_rmse_mps": _rmse(r0),
        "first_jacobian_frobenius_norm": float(np.linalg.norm(j0)),
        "first_trial_state": first_trial_state,
        "first_trial_step_norm": first_trial_step_norm,
        "first_trial_objective": first_trial_objective,
        "first_trial_accepted": first_trial_accepted,
        "first_accepted_state": first_accepted_state,
        "first_accepted_step_norm": first_accepted_step_norm,
        "first_accepted_iteration": first_accepted_iteration,
        "initial_state_reset_detected": False,
    }


def trace_static_first_accept(
    initial_state: Array,
    sat_pos_m: Array,
    sat_vel_mps: Array,
    meas_mps: Array,
    with_bias: bool,
    residual_jacobian: Callable[..., tuple[Array, Array, Array]],
    objective: Callable[..., float],
    max_iter: int = 50,
    lambda0: float = 1e-3,
) -> dict[str, Any]:
    """Replay frozen static LM arithmetic until its first accepted update."""
    state = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    actual_initial = state.copy()
    lam = float(lambda0)
    r0, j0, _ = residual_jacobian(state, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias)
    current_obj = float(objective(r0, robust=False))
    first_trial_state: Array | None = None
    first_trial_step_norm = float("nan")
    first_trial_objective = float("nan")
    first_trial_accepted = False
    first_accepted_state: Array | None = None
    first_accepted_step_norm = float("nan")
    first_accepted_iteration: int | None = None

    for iteration in range(1, int(max_iter) + 1):
        residual, jacobian, _ = residual_jacobian(
            state, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias
        )
        lhs = jacobian.T @ jacobian + lam * np.eye(jacobian.shape[1])
        delta = _solve_linear(lhs, -jacobian.T @ residual)
        candidate = state + delta
        candidate_residual, _candidate_jacobian, _ = residual_jacobian(
            candidate, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias
        )
        candidate_obj = float(objective(candidate_residual, robust=False))
        accepted = bool(np.isfinite(candidate_obj) and candidate_obj < current_obj)
        if first_trial_state is None:
            first_trial_state = candidate.copy()
            first_trial_step_norm = float(np.linalg.norm(delta))
            first_trial_objective = candidate_obj
            first_trial_accepted = accepted
        if accepted:
            state = candidate
            current_obj = candidate_obj
            first_accepted_state = state.copy()
            first_accepted_step_norm = float(np.linalg.norm(delta))
            first_accepted_iteration = iteration
            break
        lam = min(lam * 10.0, 1e12)

    return {
        "hook_scope": "diagnostic_only_read_only_replay",
        "actual_initial_state": actual_initial,
        "first_objective": float(objective(r0, robust=False)),
        "first_residual_rmse_mps": _rmse(r0),
        "first_jacobian_frobenius_norm": float(np.linalg.norm(j0)),
        "first_trial_state": first_trial_state,
        "first_trial_step_norm": first_trial_step_norm,
        "first_trial_objective": first_trial_objective,
        "first_trial_accepted": first_trial_accepted,
        "first_accepted_state": first_accepted_state,
        "first_accepted_step_norm": first_accepted_step_norm,
        "first_accepted_iteration": first_accepted_iteration,
        "initial_state_reset_detected": False,
    }


@dataclass
class RestrictedLmResult:
    state: Array
    success: bool
    converged: bool
    iterations: int
    accepted_steps: int
    initial_objective: float
    final_objective: float
    condition_number: float
    failure_reason: str


def solve_restricted_ctd_diagnostic(
    initial_full_state: Array,
    active_indices: list[int],
    time_s: Array,
    sat_pos_m: Array,
    sat_vel_mps: Array,
    meas_mps: Array,
    t0_s: float,
    full_config: Any,
    residual_jacobian: Callable[..., tuple[Array, Array, Array]],
    objective: Callable[..., float],
    max_iter: int = 80,
    lambda0: float = 1e-3,
) -> RestrictedLmResult:
    """Fixed-parameter LM ablation using frozen full-CTD residual/Jacobian."""
    theta = np.asarray(initial_full_state, dtype=float).reshape(8).copy()
    active = np.asarray(active_indices, dtype=int)
    lam = float(lambda0)
    residual0, jacobian0, _ = residual_jacobian(
        theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, full_config
    )
    current_obj = float(objective(residual0, robust=False))
    initial_obj = current_obj
    accepted_steps = 0
    converged = False
    iteration = 0

    for iteration in range(1, int(max_iter) + 1):
        residual, jacobian_full, _ = residual_jacobian(
            theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, full_config
        )
        jacobian = jacobian_full[:, active]
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        diag_scale = np.diag(np.maximum(np.diag(normal), 1.0))
        delta_active = _solve_linear(normal + lam * diag_scale, -gradient)
        if not np.all(np.isfinite(delta_active)):
            return RestrictedLmResult(
                theta, False, False, iteration, accepted_steps, initial_obj, current_obj, float("nan"), "non_finite_delta"
            )
        delta = np.zeros(8, dtype=float)
        delta[active] = delta_active
        candidate = theta + delta
        candidate_residual, _candidate_jacobian, _ = residual_jacobian(
            candidate, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, full_config
        )
        candidate_obj = float(objective(candidate_residual, robust=False))
        if np.isfinite(candidate_obj) and candidate_obj < current_obj:
            theta = candidate
            current_obj = candidate_obj
            accepted_steps += 1
            lam = max(lam / 3.0, 1e-12)
            pos_ok = True if not np.intersect1d(active, np.arange(0, 3)).size else np.linalg.norm(delta[:3]) < 1e-4
            vel_ok = True if not np.intersect1d(active, np.arange(3, 6)).size else np.linalg.norm(delta[3:6]) < 1e-6
            clock_ok = True if not np.intersect1d(active, np.arange(6, 8)).size else np.linalg.norm(delta[6:8]) < 1e-8
            if pos_ok and vel_ok and clock_ok:
                converged = True
                break
        else:
            lam = min(lam * 10.0, 1e12)

    residual_final, jacobian_full, _ = residual_jacobian(
        theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, full_config
    )
    jacobian = jacobian_full[:, active]
    try:
        condition = float(np.linalg.cond(jacobian.T @ jacobian))
    except Exception:
        condition = float("nan")
    success = bool(accepted_steps > 0 or _rmse(residual_final) <= 1e-8)
    reason = "" if success else "no_accepted_lm_step"
    return RestrictedLmResult(
        theta, success, converged, iteration, accepted_steps, initial_obj, current_obj, condition, reason
    )


def finite_difference_jacobian(
    theta: Array,
    step_sizes: Array,
    residual_function: Callable[[Array], Array],
) -> Array:
    """Central finite-difference residual Jacobian for implementation audit."""
    theta = np.asarray(theta, dtype=float).copy()
    steps = np.asarray(step_sizes, dtype=float)
    columns: list[Array] = []
    for index, step in enumerate(steps):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((residual_function(plus) - residual_function(minus)) / (2.0 * step))
    return np.column_stack(columns)
