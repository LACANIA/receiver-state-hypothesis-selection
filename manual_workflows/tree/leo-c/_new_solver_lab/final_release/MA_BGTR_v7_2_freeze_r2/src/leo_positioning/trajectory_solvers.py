"""Trajectory Doppler LM and fixed-lag solver prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import cauchy_weights
from .trajectory_models import (
    FULL_CTD_CONFIG,
    TrajectoryModelConfig,
    append_optional_prior,
    ctd_objective,
    residuals_and_jacobian_ctd,
    trajectory_bias,
    trajectory_positions,
    unpack_state,
    weighted_normal_condition,
)


@dataclass
class TrajectorySolverResult:
    state: np.ndarray
    success: bool
    iterations: int
    accepted_steps: int
    initial_objective: float
    final_objective: float
    converged: bool
    condition_number: float
    failure_reason: str = ""


@dataclass
class FixedLagWindowResult:
    start_time_s: float
    end_time_s: float
    observation_count: int
    state: np.ndarray
    result: TrajectorySolverResult
    residual_rmse_mps: float
    condition_number: float


@dataclass
class FixedLagResult:
    windows: list[FixedLagWindowResult]
    success: bool
    converged: bool
    failure_reason: str
    runtime_ms: float


def _solve_linear(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def solve_ctd_lm(
    initial_state: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    config: TrajectoryModelConfig = FULL_CTD_CONFIG,
    robust: bool = False,
    prior: dict[str, Any] | None = None,
    max_iter: int = 100,
    lambda0: float = 1e-3,
    c_scale_mps: float = 2.0,
    position_tol_m: float = 1e-4,
    velocity_tol_mps: float = 1e-6,
    bias_tol: float = 1e-8,
) -> TrajectorySolverResult:
    """Clock-drift-aware trajectory LM with optional Cauchy IRLS."""
    theta = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    lam = float(lambda0)
    r0, j0, _ = residuals_and_jacobian_ctd(theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config)
    r0_aug, _j0_aug = append_optional_prior(r0, j0, theta, prior)
    current_obj = ctd_objective(r0_aug, robust=robust, c_scale_mps=c_scale_mps)
    initial_obj = current_obj
    accepted = 0
    condition = np.nan

    for iteration in range(1, max_iter + 1):
        r, jac, _ = residuals_and_jacobian_ctd(theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config)
        r_aug, jac_aug = append_optional_prior(r, jac, theta, prior)
        weights = cauchy_weights(r_aug, c_scale_mps) if robust else np.ones(len(r_aug), dtype=float)
        normal = jac_aug.T @ (weights[:, None] * jac_aug)
        gradient = jac_aug.T @ (weights * r_aug)
        diag_scale = np.diag(np.maximum(np.diag(normal), 1.0))
        lhs = normal + lam * diag_scale
        rhs = -gradient
        try:
            delta = _solve_linear(lhs, rhs)
        except Exception as exc:  # noqa: BLE001
            return TrajectorySolverResult(
                theta, False, iteration, accepted, initial_obj, current_obj, False, condition, f"linear_solve_failed: {exc}"
            )
        if not np.all(np.isfinite(delta)):
            return TrajectorySolverResult(
                theta, False, iteration, accepted, initial_obj, current_obj, False, condition, "non_finite_delta"
            )

        candidate = theta + delta
        r_candidate, j_candidate, _ = residuals_and_jacobian_ctd(
            candidate, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config
        )
        r_candidate_aug, _j_candidate_aug = append_optional_prior(r_candidate, j_candidate, candidate, prior)
        candidate_obj = ctd_objective(r_candidate_aug, robust=robust, c_scale_mps=c_scale_mps)

        if np.isfinite(candidate_obj) and candidate_obj < current_obj:
            theta = candidate
            current_obj = candidate_obj
            lam = max(lam / 3.0, 1e-12)
            accepted += 1
            pos_ok = np.linalg.norm(delta[:3]) < position_tol_m
            vel_ok = np.linalg.norm(delta[3:6]) < velocity_tol_mps
            clock_ok = True if delta.size <= 6 else np.linalg.norm(delta[6:]) < bias_tol
            if pos_ok and vel_ok and clock_ok:
                r_final, j_final, _ = residuals_and_jacobian_ctd(
                    theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config
                )
                try:
                    condition = weighted_normal_condition(j_final, r_final, robust, c_scale_mps)
                except Exception:
                    condition = np.nan
                return TrajectorySolverResult(
                    theta, True, iteration, accepted, initial_obj, current_obj, True, condition
                )
        else:
            lam = min(lam * 10.0, 1e12)

        if not np.all(np.isfinite(theta)) or not np.isfinite(current_obj):
            return TrajectorySolverResult(
                theta, False, iteration, accepted, initial_obj, current_obj, False, condition, "non_finite_state_or_cost"
            )

    r_final, j_final, _ = residuals_and_jacobian_ctd(theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, config)
    try:
        condition = weighted_normal_condition(j_final, r_final, robust, c_scale_mps)
    except Exception:
        condition = np.nan
    if accepted == 0:
        return TrajectorySolverResult(
            theta, False, max_iter, accepted, initial_obj, current_obj, False, condition, "no_accepted_lm_step"
        )
    return TrajectorySolverResult(theta, True, max_iter, accepted, initial_obj, current_obj, False, condition)


def _window_slices(time_s: np.ndarray, window_duration_s: float) -> list[tuple[float, float, np.ndarray]]:
    t = np.asarray(time_s, dtype=float)
    order = np.argsort(t)
    t_sorted = t[order]
    t_min = float(t_sorted[0])
    t_max = float(t_sorted[-1])
    step = float(window_duration_s) * 0.5
    starts: list[float] = []
    current = t_min
    while current <= t_max + 1e-9:
        starts.append(current)
        current += step
    windows: list[tuple[float, float, np.ndarray]] = []
    for start in starts:
        end = start + float(window_duration_s)
        if start >= t_max and windows:
            continue
        mask = (t >= start) & (t <= end)
        if int(np.sum(mask)) >= 8:
            windows.append((start, min(end, t_max), mask))
    if not windows:
        windows.append((t_min, t_max, np.ones(len(t), dtype=bool)))
    return windows


def propagate_state_to_start(
    theta: np.ndarray,
    old_t0_s: float,
    new_t0_s: float,
    config: TrajectoryModelConfig = FULL_CTD_CONFIG,
) -> np.ndarray:
    p0, velocity, b0, bdot = unpack_state(theta, config)
    dt = float(new_t0_s) - float(old_t0_s)
    return np.r_[p0 + velocity * dt, velocity, b0 + bdot * dt, bdot]


def solve_fixed_lag_ctd(
    initial_state: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    window_duration_s: float,
    robust: bool = False,
    c_scale_mps: float = 2.0,
) -> FixedLagResult:
    """Run a simple overlapping fixed-lag CTD-LM prototype."""
    import time as _time

    start_clock = _time.perf_counter()
    t = np.asarray(time_s, dtype=float)
    windows = _window_slices(t, float(window_duration_s))
    theta_start = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    current_t0 = float(windows[0][0])
    results: list[FixedLagWindowResult] = []
    reasons: list[str] = []

    for idx, (start_s, end_s, mask) in enumerate(windows):
        if idx == 0:
            local_initial = theta_start
        else:
            local_initial = propagate_state_to_start(results[-1].state, current_t0, start_s, FULL_CTD_CONFIG)
            current_t0 = start_s
        local_result = solve_ctd_lm(
            local_initial,
            t[mask],
            sat_pos_m[mask],
            sat_vel_mps[mask],
            meas_mps[mask],
            t0_s=start_s,
            config=FULL_CTD_CONFIG,
            robust=robust,
            c_scale_mps=c_scale_mps,
            max_iter=80,
        )
        residual, jac, _ = residuals_and_jacobian_ctd(
            local_result.state, t[mask], sat_pos_m[mask], sat_vel_mps[mask], meas_mps[mask], start_s, FULL_CTD_CONFIG
        )
        rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size else np.nan
        try:
            cond = weighted_normal_condition(jac, residual, robust, c_scale_mps)
        except Exception:
            cond = np.nan
        if local_result.failure_reason:
            reasons.append(f"window_{idx}:{local_result.failure_reason}")
        results.append(
            FixedLagWindowResult(
                start_time_s=float(start_s),
                end_time_s=float(end_s),
                observation_count=int(np.sum(mask)),
                state=local_result.state.copy(),
                result=local_result,
                residual_rmse_mps=rmse,
                condition_number=cond,
            )
        )

    runtime_ms = (_time.perf_counter() - start_clock) * 1000.0
    success = bool(results and all(w.result.success for w in results))
    converged = bool(results and all(w.result.converged for w in results))
    return FixedLagResult(results, success, converged, ";".join(reasons), runtime_ms)


def estimate_window_end_position(window: FixedLagWindowResult) -> np.ndarray:
    return trajectory_positions(
        window.state,
        np.array([window.end_time_s], dtype=float),
        window.start_time_s,
        FULL_CTD_CONFIG,
    )[0]


def estimate_window_start_bias(window: FixedLagWindowResult) -> float:
    return float(trajectory_bias(window.state, np.array([window.start_time_s]), window.start_time_s, FULL_CTD_CONFIG)[0])
