"""Numpy implementations of Doppler baseline solvers used in STEP03."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import cauchy_weights, objective, residuals_and_jacobian


@dataclass
class SolverResult:
    state: np.ndarray
    success: bool
    iterations: int
    initial_objective: float
    final_objective: float
    converged: bool
    failure_reason: str = ""
    accepted_steps: int = 0


def _solve_linear(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _finite_failure(state: np.ndarray, cost: float) -> str:
    if not np.all(np.isfinite(state)):
        return "non_finite_state"
    if not np.isfinite(cost):
        return "non_finite_cost"
    return ""


def solve_gn_position_only(
    initial_position_m: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    max_iter: int = 50,
    tol_m: float = 1e-4,
) -> SolverResult:
    """Gauss-Newton for a 3-D position-only state.

    The model returns the residual Jacobian J = dr/dx. With additive updates,
    the local model is r(x + delta) ~= r + J delta, so the normal equation is
    (J^T J) delta = -J^T r.
    """
    x = np.asarray(initial_position_m, dtype=float).reshape(3).copy()
    r0, _j0, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=False)
    initial_obj = objective(r0, robust=False)
    final_obj = initial_obj
    accepted = 0

    for iteration in range(1, max_iter + 1):
        r, jac, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=False)
        try:
            delta = _solve_linear(jac.T @ jac, -jac.T @ r)
        except Exception as exc:  # noqa: BLE001 - numeric failure is part of the experiment
            return SolverResult(x, False, iteration, initial_obj, final_obj, False, f"linear_solve_failed: {exc}")
        if not np.all(np.isfinite(delta)):
            return SolverResult(x, False, iteration, initial_obj, final_obj, False, "non_finite_delta", accepted)

        x = x + delta
        accepted += 1
        r_new, _j_new, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=False)
        final_obj = objective(r_new, robust=False)
        failure = _finite_failure(x, final_obj)
        if failure:
            return SolverResult(x, False, iteration, initial_obj, final_obj, False, failure, accepted)
        if np.linalg.norm(delta) < tol_m:
            return SolverResult(x, True, iteration, initial_obj, final_obj, True, accepted_steps=accepted)

    return SolverResult(x, True, max_iter, initial_obj, final_obj, False, accepted_steps=accepted)


def solve_lm(
    initial_state: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    with_bias: bool,
    robust: bool,
    max_iter: int = 50,
    tol_m: float = 1e-4,
    lambda0: float = 1e-3,
    c_scale_mps: float = 2.0,
) -> SolverResult:
    """Levenberg-Marquardt with optional Cauchy IRLS weights."""
    x = np.asarray(initial_state, dtype=float).copy()
    lam = float(lambda0)
    r0, _j0, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias)
    current_obj = objective(r0, robust=robust, c_scale_mps=c_scale_mps)
    initial_obj = current_obj
    accepted = 0

    for iteration in range(1, max_iter + 1):
        r, jac, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias)
        weights = cauchy_weights(r, c_scale_mps=c_scale_mps) if robust else np.ones_like(r)
        lhs = jac.T @ (weights[:, None] * jac) + lam * np.eye(jac.shape[1])
        rhs = -jac.T @ (weights * r)
        try:
            delta = _solve_linear(lhs, rhs)
        except Exception as exc:  # noqa: BLE001
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, f"linear_solve_failed: {exc}", accepted)
        if not np.all(np.isfinite(delta)):
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, "non_finite_delta", accepted)

        candidate = x + delta
        r_candidate, _j_candidate, _ = residuals_and_jacobian(
            candidate, sat_pos_m, sat_vel_mps, meas_mps, with_bias=with_bias
        )
        candidate_obj = objective(r_candidate, robust=robust, c_scale_mps=c_scale_mps)
        if np.isfinite(candidate_obj) and candidate_obj < current_obj:
            x = candidate
            current_obj = candidate_obj
            lam = max(lam / 3.0, 1e-12)
            accepted += 1
            bias_step_ok = (not with_bias) or abs(float(delta[3])) < 1e-8
            if np.linalg.norm(delta[:3]) < tol_m and bias_step_ok:
                return SolverResult(x, True, iteration, initial_obj, current_obj, True, accepted_steps=accepted)
        else:
            lam = min(lam * 10.0, 1e12)

        failure = _finite_failure(x, current_obj)
        if failure:
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, failure, accepted)

    if accepted == 0:
        return SolverResult(x, False, max_iter, initial_obj, current_obj, False, "no_accepted_lm_step", accepted)
    return SolverResult(x, True, max_iter, initial_obj, current_obj, False, accepted_steps=accepted)


def _dogleg_step(gn_step: np.ndarray, cauchy_step: np.ndarray, trust_radius: float) -> tuple[np.ndarray, bool]:
    gn_norm = float(np.linalg.norm(gn_step))
    if gn_norm <= trust_radius:
        return gn_step, False

    cauchy_norm = float(np.linalg.norm(cauchy_step))
    if cauchy_norm >= trust_radius:
        if cauchy_norm <= 0.0:
            return np.zeros_like(cauchy_step), True
        return (trust_radius / cauchy_norm) * cauchy_step, True

    segment = gn_step - cauchy_step
    a = float(segment @ segment)
    b = float(2.0 * (cauchy_step @ segment))
    c = float(cauchy_step @ cauchy_step - trust_radius * trust_radius)
    disc = max(b * b - 4.0 * a * c, 0.0)
    tau = (-b + np.sqrt(disc)) / (2.0 * a) if a > 0.0 else 0.0
    return cauchy_step + tau * segment, True


def solve_dogleg_position_only(
    initial_position_m: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    max_iter: int = 80,
    tol_m: float = 1e-4,
    initial_trust_radius_m: float = 100_000.0,
    max_trust_radius_m: float = 10_000_000.0,
    eta: float = 0.10,
) -> SolverResult:
    """Standard Dog-Leg trust-region baseline for a 3-D position-only state."""
    x = np.asarray(initial_position_m, dtype=float).reshape(3).copy()
    r0, _j0, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=False)
    current_obj = objective(r0, robust=False)
    initial_obj = current_obj
    trust_radius = float(initial_trust_radius_m)
    accepted = 0

    for iteration in range(1, max_iter + 1):
        r, jac, _ = residuals_and_jacobian(x, sat_pos_m, sat_vel_mps, meas_mps, with_bias=False)
        gradient = jac.T @ r
        normal = jac.T @ jac
        try:
            gn_step = _solve_linear(normal, -gradient)
        except Exception as exc:  # noqa: BLE001
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, f"gn_step_failed: {exc}", accepted)

        g_norm_sq = float(gradient @ gradient)
        denom = float(gradient @ (normal @ gradient))
        if g_norm_sq <= 0.0:
            return SolverResult(x, True, iteration, initial_obj, current_obj, True, accepted_steps=accepted)
        if denom <= 0.0 or not np.isfinite(denom):
            cauchy_step = -gradient
        else:
            cauchy_step = -(g_norm_sq / denom) * gradient

        step, hit_boundary = _dogleg_step(gn_step, cauchy_step, trust_radius)
        if not np.all(np.isfinite(step)):
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, "non_finite_dogleg_step", accepted)

        predicted_reduction = float(-(gradient @ step) - 0.5 * (step @ (normal @ step)))
        candidate = x + step
        r_candidate, _j_candidate, _ = residuals_and_jacobian(
            candidate, sat_pos_m, sat_vel_mps, meas_mps, with_bias=False
        )
        candidate_obj = objective(r_candidate, robust=False)
        actual_reduction = current_obj - candidate_obj
        if predicted_reduction <= 0.0 or not np.isfinite(predicted_reduction):
            rho = -np.inf
        else:
            rho = actual_reduction / predicted_reduction

        if rho < 0.25:
            trust_radius = max(0.25 * trust_radius, 1e-6)
        elif rho > 0.75 and hit_boundary:
            trust_radius = min(2.0 * trust_radius, max_trust_radius_m)

        if rho > eta and np.isfinite(candidate_obj):
            x = candidate
            current_obj = candidate_obj
            accepted += 1
            if np.linalg.norm(step) < tol_m:
                return SolverResult(x, True, iteration, initial_obj, current_obj, True, accepted_steps=accepted)

        failure = _finite_failure(x, current_obj)
        if failure:
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, failure, accepted)
        if trust_radius < 1e-3 and accepted == 0:
            return SolverResult(x, False, iteration, initial_obj, current_obj, False, "trust_region_collapsed", accepted)

    if accepted == 0:
        return SolverResult(x, False, max_iter, initial_obj, current_obj, False, "no_accepted_dogleg_step", accepted)
    return SolverResult(x, True, max_iter, initial_obj, current_obj, False, accepted_steps=accepted)
