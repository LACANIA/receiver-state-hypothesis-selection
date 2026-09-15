"""Bias-Eliminated Geometry-aware Trust-Region Doppler Solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry_trust_region import geometry_regularized_step, update_trust_radius
from .observability import ObservabilitySummary, summarize_observability
from .robust_schedule import graduated_cauchy_schedule, weights_for_scale
from .variable_projection import (
    BetaPriorProfile,
    ProjectionResult,
    bias_design_matrix,
    expand_beta,
    project_residual_and_jacobian,
)


@dataclass
class BeGtrEvaluation:
    residual_no_bias: np.ndarray
    reduced_residual: np.ndarray
    reduced_jacobian: np.ndarray
    beta: np.ndarray
    h_nonbias: np.ndarray
    weights: np.ndarray
    projection: ProjectionResult
    h_red: np.ndarray
    g_red: np.ndarray
    objective: float
    observability: ObservabilitySummary


@dataclass
class BeGtrResult:
    state_x: np.ndarray
    beta: np.ndarray
    numerical_success: bool
    converged: bool
    iterations: int
    accepted_steps: int
    rejected_steps: int
    final_objective: float
    initial_objective: float
    residual: np.ndarray
    reduced_residual: np.ndarray
    condition_number: float
    beta_normal_condition: float
    min_singular_value: float
    max_singular_value: float
    weak_direction_count: int
    top_correlation_pair: str
    top_abs_correlation: float
    covariance_diag_p0_mean: float
    covariance_diag_v_mean: float
    geometry_status: str
    beta_regularized: bool
    weighted_orthogonality_error: float
    failure_reason: str = ""


def h_and_jacobian_x(
    x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    t0_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return h(x) and residual Jacobian J_x = -dh/dx for x=[p0,v]."""
    x = np.asarray(x, dtype=float).reshape(6)
    p0 = x[:3]
    velocity = x[3:6]
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
    h = np.sum(u * w, axis=1)
    j_x = np.column_stack([-aw, -(u + tau[:, None] * aw)])
    return h, j_x


def evaluate_be_gtr(
    x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    weights: np.ndarray,
    beta_profile: BetaPriorProfile,
    projection_mode: str = "full",
    include_prior_cost: bool = False,
) -> BeGtrEvaluation:
    h, j_x = h_and_jacobian_x(x, time_s, sat_pos_m, sat_vel_mps, t0_s)
    a = np.asarray(meas_mps, dtype=float) - h
    b = bias_design_matrix(time_s, t0_s, projection_mode=projection_mode)
    mode_profile = beta_profile.for_mode(projection_mode)
    projection = project_residual_and_jacobian(a, j_x, b, weights, mode_profile)
    reduced = projection.reduced_residual
    j_red = projection.projected_jacobian
    h_red = j_red.T @ (weights[:, None] * j_red)
    g_red = j_red.T @ (weights * reduced)
    if include_prior_cost:
        objective = projection.reduced_cost
    else:
        objective = 0.5 * float(np.sum(weights * reduced * reduced))
    observability = summarize_observability(h_red)
    return BeGtrEvaluation(
        residual_no_bias=a,
        reduced_residual=reduced,
        reduced_jacobian=j_red,
        beta=projection.beta,
        h_nonbias=h,
        weights=weights,
        projection=projection,
        h_red=h_red,
        g_red=g_red,
        objective=objective,
        observability=observability,
    )


def solve_be_gtr(
    initial_x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    beta_profile: BetaPriorProfile,
    robust: bool = False,
    max_iter_per_scale: int = 30,
    lambda_lm0: float = 1e-3,
    trust_radius0: float = 100_000.0,
    c_scale_schedule: list[float] | None = None,
    projection_mode: str = "full",
    include_prior_cost: bool = False,
) -> BeGtrResult:
    x = np.asarray(initial_x, dtype=float).reshape(6).copy()
    total_iterations = 0
    accepted = 0
    rejected = 0
    failure_reasons: list[str] = []
    scale_schedule = c_scale_schedule if c_scale_schedule is not None else graduated_cauchy_schedule(robust)
    weights = np.ones(len(meas_mps), dtype=float)
    first_eval = evaluate_be_gtr(
        x,
        time_s,
        sat_pos_m,
        sat_vel_mps,
        meas_mps,
        t0_s,
        weights,
        beta_profile,
        projection_mode=projection_mode,
        include_prior_cost=include_prior_cost,
    )
    initial_objective = first_eval.objective
    current_eval = first_eval

    for scale in scale_schedule:
        lam = float(lambda_lm0)
        trust_radius = float(trust_radius0)
        # Recompute robust weights from the current reduced residual at each graduated stage.
        if np.isfinite(scale):
            weights = weights_for_scale(current_eval.reduced_residual, scale)
            current_eval = evaluate_be_gtr(
                x,
                time_s,
                sat_pos_m,
                sat_vel_mps,
                meas_mps,
                t0_s,
                weights,
                beta_profile,
                projection_mode=projection_mode,
                include_prior_cost=include_prior_cost,
            )
        for _iteration in range(max_iter_per_scale):
            total_iterations += 1
            step = geometry_regularized_step(
                current_eval.h_red,
                current_eval.g_red,
                lambda_lm=lam,
                trust_radius_mixed=trust_radius,
                weak_threshold_ratio=1e-6,
                lambda_geo_base=1.0,
            )
            delta = step.delta
            if not np.all(np.isfinite(delta)):
                failure_reasons.append("non_finite_delta")
                break
            candidate_x = x + delta
            candidate_eval = evaluate_be_gtr(
                candidate_x,
                time_s,
                sat_pos_m,
                sat_vel_mps,
                meas_mps,
                t0_s,
                weights,
                beta_profile,
                projection_mode=projection_mode,
                include_prior_cost=include_prior_cost,
            )
            actual_reduction = current_eval.objective - candidate_eval.objective
            predicted = step.predicted_reduction
            rho = actual_reduction / predicted if predicted > 0.0 and np.isfinite(predicted) else -np.inf
            trust_radius = update_trust_radius(trust_radius, rho, step.diagnostics.weak_direction_count)
            if rho > 0.05 and np.isfinite(candidate_eval.objective):
                x = candidate_x
                current_eval = candidate_eval
                accepted += 1
                lam = max(lam / 3.0, 1e-12)
                if np.linalg.norm(delta[:3]) < 1e-4 and np.linalg.norm(delta[3:]) < 1e-6:
                    break
            else:
                rejected += 1
                lam = min(lam * 10.0, 1e12)
            if not np.all(np.isfinite(x)):
                failure_reasons.append("non_finite_state")
                break

    final_eval = current_eval
    b_final = bias_design_matrix(time_s, t0_s, projection_mode=projection_mode)
    full_residual = final_eval.residual_no_bias - b_final @ final_eval.beta
    converged = bool(accepted > 0 and np.linalg.norm(final_eval.g_red) < max(1e-3, 1e-6 * len(meas_mps)))
    numerical_success = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(final_eval.beta)) and np.isfinite(final_eval.objective))
    if accepted == 0:
        failure_reasons.append("no_accepted_trust_region_step")
        numerical_success = False
    obs = final_eval.observability
    return BeGtrResult(
        state_x=x,
        beta=expand_beta(final_eval.beta, projection_mode),
        numerical_success=numerical_success,
        converged=converged,
        iterations=total_iterations,
        accepted_steps=accepted,
        rejected_steps=rejected,
        final_objective=final_eval.objective,
        initial_objective=initial_objective,
        residual=full_residual,
        reduced_residual=final_eval.reduced_residual,
        condition_number=obs.reduced_condition_number,
        beta_normal_condition=final_eval.projection.beta_normal_condition,
        min_singular_value=obs.min_singular_value,
        max_singular_value=obs.max_singular_value,
        weak_direction_count=obs.weak_direction_count,
        top_correlation_pair=obs.top_correlation_pair,
        top_abs_correlation=obs.top_abs_correlation,
        covariance_diag_p0_mean=obs.covariance_diag_p0_mean,
        covariance_diag_v_mean=obs.covariance_diag_v_mean,
        geometry_status=obs.geometry_status,
        beta_regularized=final_eval.projection.regularized,
        weighted_orthogonality_error=final_eval.projection.weighted_orthogonality_error,
        failure_reason=";".join(dict.fromkeys(failure_reasons)),
    )


def result_diagnostics(result: BeGtrResult) -> dict[str, Any]:
    return {
        "reduced_condition_number": float(result.condition_number),
        "beta_normal_condition_number": float(result.beta_normal_condition),
        "min_singular_value": float(result.min_singular_value),
        "max_singular_value": float(result.max_singular_value),
        "weak_direction_count": int(result.weak_direction_count),
        "top_correlation_pair": result.top_correlation_pair,
        "top_abs_correlation": float(result.top_abs_correlation),
        "covariance_diag_p0_mean": float(result.covariance_diag_p0_mean),
        "covariance_diag_v_mean": float(result.covariance_diag_v_mean),
        "geometry_status": result.geometry_status,
        "projection_weighted_orthogonality_error": float(result.weighted_orthogonality_error),
    }
