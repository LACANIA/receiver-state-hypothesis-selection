"""Candidate model runners for MA-BGTR."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .be_gtr_solver import solve_be_gtr
from .gir_tr_solver import gir_positions, initial_full_state, solve_gir_tr
from .influence_diagnostics import diagnostics_dict, summarize_downweighted_satellites
from .metrics import residual_metrics
from .models import cauchy_weights, normal_matrix_condition, residuals_and_jacobian
from .projection_diagnostics import compute_projection_diagnostics, diagnostics_to_dict
from .quality import physical_plausibility, quality_gate
from .solvers import solve_lm
from .trajectory_models import (
    FULL_CTD_CONFIG,
    NO_BDOT_CONFIG,
    NO_BIAS_CONFIG,
    TrajectoryModelConfig,
    pack_state,
    residuals_and_jacobian_ctd,
    trajectory_positions,
    unpack_state,
    weighted_normal_condition,
)
from .trajectory_solvers import solve_ctd_lm
from .variable_projection import beta_prior_profiles


CANDIDATE_MODEL_LIST = [
    "M0_static_position",
    "M1_static_position_bias",
    "M2_ctd_full",
    "M3_ctd_no_drift",
    "M4_ctd_no_bias",
    "M5_be_b0_only",
    "M6_be_full",
    "M7_robust_ctd_full",
    "M8_robust_be_b0_only",
    "M9_robust_be_full",
    "M10_gir_tr_fixed_scale",
    "M11_gir_tr_mad_scale",
]


MODEL_DOF = {
    "M0_static_position": 3,
    "M1_static_position_bias": 4,
    "M2_ctd_full": 8,
    "M3_ctd_no_drift": 7,
    "M4_ctd_no_bias": 6,
    "M5_be_b0_only": 7,
    "M6_be_full": 8,
    "M7_robust_ctd_full": 8,
    "M8_robust_be_b0_only": 7,
    "M9_robust_be_full": 8,
    "M10_gir_tr_fixed_scale": 8,
    "M11_gir_tr_mad_scale": 8,
    "M12_ctd_full_plus_gir_refine": 8,
    "M13_robust_ctd_full_plus_gir_refine": 8,
    "M14_static_plus_gir_refine": 8,
}


def truth_positions(obs: dict[str, Any]) -> np.ndarray:
    if "truth_positions_m" in obs:
        return np.asarray(obs["truth_positions_m"], dtype=float)
    tau = np.asarray(obs["time_s"], dtype=float) - float(obs.get("t0_s", np.min(obs["time_s"])))
    return np.asarray(obs["p0_true_m"], dtype=float)[None, :] + tau[:, None] * np.asarray(obs["v_true_mps"], dtype=float)[None, :]


def trajectory_errors(estimated_positions: np.ndarray, true_positions: np.ndarray) -> tuple[float, float, float]:
    errors = np.linalg.norm(np.asarray(estimated_positions) - np.asarray(true_positions), axis=1)
    return float(errors[-1]), float(np.mean(errors)), float(np.max(errors))


def robust_cost_per_dof(residual: np.ndarray, model_dof: int, c_scale_mps: float = 2.0) -> float:
    residual = np.asarray(residual, dtype=float)
    scaled = residual / float(c_scale_mps)
    cost = 0.5 * c_scale_mps * c_scale_mps * np.sum(np.log1p(scaled * scaled))
    return float(cost / max(len(residual) - int(model_dof), 1))


def base_candidate_row(
    dataset_type: str,
    scenario: str,
    candidate_model: str,
    position_init_label: str,
    velocity_init_label: str,
    beta_prior_profile: str,
    p0_init: np.ndarray,
    v_init: np.ndarray,
    obs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_type": dataset_type,
        "scenario": scenario,
        "candidate_model": candidate_model,
        "position_init_label": position_init_label,
        "velocity_init_label": velocity_init_label,
        "beta_prior_profile": beta_prior_profile,
        "model_dof": MODEL_DOF[candidate_model],
    }


def failure_row(row: dict[str, Any], runtime_ms: float, reason: str) -> dict[str, Any]:
    for key in [
        "residual_rmse_mps",
        "robust_cost_per_dof",
        "condition_number",
        "retention_trace_ratio",
        "subspace_coherence_max",
        "rank_loss",
        "estimated_speed_mps",
        "beta0_estimated_mps",
        "beta_dot_estimated_mps2",
        "final_position_error_m",
        "mean_position_error_m",
        "max_position_error_m",
        "velocity_error_mps",
        "beta0_error_mps",
        "beta_dot_error_mps2",
        "runtime_ms",
    ]:
        row[key] = np.nan
    row.update(
        {
            "runtime_ms": float(runtime_ms),
            "numerical_success": False,
            "converged": False,
            "quality_pass": False,
            "physical_plausible": False,
            "failure_reason": reason,
        }
    )
    return row


def finalize_candidate(
    row: dict[str, Any],
    obs: dict[str, Any],
    estimated_positions: np.ndarray,
    velocity_est: np.ndarray,
    beta0: float,
    bdot: float,
    residual: np.ndarray,
    condition_number: float,
    numerical_success: bool,
    converged: bool,
    runtime_ms: float,
    failure_reason: str,
    projection_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rmse, _med = residual_metrics(residual)
    physical, physical_reason = physical_plausibility(row["dataset_type"], float(np.linalg.norm(velocity_est)), beta0, bdot)
    quality, quality_reason = quality_gate(row["dataset_type"], numerical_success, physical, None)
    reasons = ";".join(dict.fromkeys([r for r in [failure_reason, physical_reason, quality_reason] if r]))
    row.update(
        {
            "numerical_success": bool(numerical_success),
            "converged": bool(converged),
            "quality_pass": bool(quality),
            "physical_plausible": bool(physical),
            "residual_rmse_mps": rmse,
            "robust_cost_per_dof": robust_cost_per_dof(residual, int(row["model_dof"])),
            "condition_number": float(condition_number) if np.isfinite(condition_number) else np.nan,
            "retention_trace_ratio": np.nan,
            "subspace_coherence_max": np.nan,
            "rank_loss": np.nan,
            "estimated_speed_mps": float(np.linalg.norm(velocity_est)),
            "beta0_estimated_mps": float(beta0) if np.isfinite(beta0) else np.nan,
            "beta_dot_estimated_mps2": float(bdot) if np.isfinite(bdot) else np.nan,
            "final_ecef_x_m": float(np.asarray(estimated_positions)[-1, 0]),
            "final_ecef_y_m": float(np.asarray(estimated_positions)[-1, 1]),
            "final_ecef_z_m": float(np.asarray(estimated_positions)[-1, 2]),
            "estimated_vx_mps": float(np.asarray(velocity_est)[0]),
            "estimated_vy_mps": float(np.asarray(velocity_est)[1]),
            "estimated_vz_mps": float(np.asarray(velocity_est)[2]),
            "runtime_ms": float(runtime_ms),
            "failure_reason": reasons,
        }
    )
    if projection_diag:
        row.update(
            {
                "retention_trace_ratio": projection_diag.get("retention_trace_ratio", np.nan),
                "subspace_coherence_max": projection_diag.get("subspace_coherence_max", np.nan),
                "rank_loss": projection_diag.get("rank_loss", np.nan),
                "condition_number": projection_diag.get("reduced_condition_number", row["condition_number"]),
            }
        )
    return row


def run_static_candidate(dataset_type: str, scenario: str, model: str, obs: dict[str, Any], pos_label: str, p0: np.ndarray, vel_label: str, v0: np.ndarray, profile: str) -> dict[str, Any]:
    row = base_candidate_row(dataset_type, scenario, model, pos_label, vel_label, profile, p0, v0, obs)
    t0 = time.perf_counter()
    try:
        with_bias = model == "M1_static_position_bias"
        initial = np.r_[p0, 0.0] if with_bias else p0
        result = solve_lm(initial, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=with_bias, robust=False)
        runtime = (time.perf_counter() - t0) * 1000.0
        residual, jac, _ = residuals_and_jacobian(result.state, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=with_bias)
        p_est = result.state[:3]
        beta0 = float(result.state[3]) if with_bias and len(result.state) >= 4 else np.nan
        condition = normal_matrix_condition(jac)
        positions = np.repeat(p_est[None, :], len(obs["time_s"]), axis=0)
        return finalize_candidate(row, obs, positions, np.zeros(3), beta0, np.nan, residual, condition, bool(result.success), bool(result.converged), runtime, result.failure_reason)
    except Exception as exc:  # noqa: BLE001
        return failure_row(row, (time.perf_counter() - t0) * 1000.0, f"exception: {exc}")


def _ctd_config_for_model(model: str) -> tuple[TrajectoryModelConfig, bool]:
    if model == "M2_ctd_full":
        return FULL_CTD_CONFIG, False
    if model == "M3_ctd_no_drift":
        return NO_BDOT_CONFIG, False
    if model == "M4_ctd_no_bias":
        return NO_BIAS_CONFIG, False
    if model == "M7_robust_ctd_full":
        return FULL_CTD_CONFIG, True
    raise ValueError(model)


def run_ctd_candidate(dataset_type: str, scenario: str, model: str, obs: dict[str, Any], pos_label: str, p0: np.ndarray, vel_label: str, v0: np.ndarray, profile: str) -> dict[str, Any]:
    row = base_candidate_row(dataset_type, scenario, model, pos_label, vel_label, profile, p0, v0, obs)
    t0 = time.perf_counter()
    try:
        config, robust = _ctd_config_for_model(model)
        t_start = float(obs.get("t0_s", np.min(obs["time_s"])))
        theta0 = pack_state(p0, v0, 0.0, 0.0, config)
        result = solve_ctd_lm(theta0, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t_start, config, robust=robust, max_iter=80)
        runtime = (time.perf_counter() - t0) * 1000.0
        residual, jac, _ = residuals_and_jacobian_ctd(result.state, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t_start, config)
        p_est, v_est, b0, bdot = unpack_state(result.state, config)
        positions = trajectory_positions(result.state, obs["time_s"], t_start, config)
        condition = weighted_normal_condition(jac, residual, robust)
        return finalize_candidate(row, obs, positions, v_est, b0, bdot, residual, condition, bool(result.success), bool(result.converged), runtime, result.failure_reason)
    except Exception as exc:  # noqa: BLE001
        return failure_row(row, (time.perf_counter() - t0) * 1000.0, f"exception: {exc}")


def run_be_candidate(dataset_type: str, scenario: str, model: str, obs: dict[str, Any], pos_label: str, p0: np.ndarray, vel_label: str, v0: np.ndarray, profile: str) -> dict[str, Any]:
    row = base_candidate_row(dataset_type, scenario, model, pos_label, vel_label, profile, p0, v0, obs)
    t0 = time.perf_counter()
    try:
        projection_mode = "full" if model in {"M6_be_full", "M9_robust_be_full"} else "b0_only"
        robust = model in {"M8_robust_be_b0_only", "M9_robust_be_full"}
        t_start = float(obs.get("t0_s", np.min(obs["time_s"])))
        x0 = np.r_[p0, v0]
        beta_profile = beta_prior_profiles()[profile]
        result = solve_be_gtr(
            x0,
            obs["time_s"],
            obs["sat_pos_m"],
            obs["sat_vel_mps"],
            obs["meas_mps"],
            t_start,
            beta_profile,
            robust=robust,
            max_iter_per_scale=12,
            projection_mode=projection_mode,
            include_prior_cost=True,
        )
        runtime = (time.perf_counter() - t0) * 1000.0
        tau = np.asarray(obs["time_s"], dtype=float) - t_start
        positions = result.state_x[:3][None, :] + tau[:, None] * result.state_x[3:6][None, :]
        diag = diagnostics_to_dict(
            compute_projection_diagnostics(
                x0,
                obs["time_s"],
                obs["sat_pos_m"],
                obs["sat_vel_mps"],
                obs["meas_mps"],
                t_start,
                np.ones(len(obs["meas_mps"])),
                "full" if projection_mode == "full" else "b0_only",
                profile,
            )
        )
        return finalize_candidate(
            row,
            obs,
            positions,
            result.state_x[3:6],
            float(result.beta[0]),
            float(result.beta[1]),
            result.residual,
            result.condition_number,
            bool(result.numerical_success),
            bool(result.converged),
            runtime,
            result.failure_reason,
            diag,
        )
    except Exception as exc:  # noqa: BLE001
        return failure_row(row, (time.perf_counter() - t0) * 1000.0, f"exception: {exc}")


def run_gir_candidate(dataset_type: str, scenario: str, model: str, obs: dict[str, Any], pos_label: str, p0: np.ndarray, vel_label: str, v0: np.ndarray, profile: str) -> dict[str, Any]:
    row = base_candidate_row(dataset_type, scenario, model, pos_label, vel_label, profile, p0, v0, obs)
    t0 = time.perf_counter()
    try:
        scale_mode = "mad" if model == "M11_gir_tr_mad_scale" else "fixed"
        t_start = float(obs.get("t0_s", np.min(obs["time_s"])))
        theta0 = initial_full_state(p0, v0)
        result = solve_gir_tr(
            theta0,
            obs["time_s"],
            obs["sat_pos_m"],
            obs["sat_vel_mps"],
            obs["meas_mps"],
            t_start,
            scale_mode=scale_mode,
            dataset_type=dataset_type,
        )
        runtime = (time.perf_counter() - t0) * 1000.0
        positions = gir_positions(result, obs["time_s"], t_start)
        p_est, v_est, b0, bdot = unpack_state(result.state, FULL_CTD_CONFIG)
        out = finalize_candidate(
            row,
            obs,
            positions,
            v_est,
            b0,
            bdot,
            result.residual,
            result.condition_number,
            bool(result.numerical_success),
            bool(result.converged),
            runtime,
            result.failure_reason,
        )
        out.update(diagnostics_dict(result.diagnostics))
        out.update(
            {
                "accepted_steps": int(result.accepted_steps),
                "rejected_steps": int(result.rejected_steps),
                "lambda_final": float(result.lambda_final),
                "trust_radius_final": float(result.trust_radius_final),
                "top_downweighted_satellites": summarize_downweighted_satellites(obs.get("satellite_number"), result.weights),
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        return failure_row(row, (time.perf_counter() - t0) * 1000.0, f"exception: {exc}")


def run_candidate_model(
    model: str,
    dataset_type: str,
    scenario: str,
    obs: dict[str, Any],
    position_init_label: str,
    p0_init: np.ndarray,
    velocity_init_label: str,
    v_init: np.ndarray,
    beta_prior_profile: str,
) -> dict[str, Any]:
    if model in {"M0_static_position", "M1_static_position_bias"}:
        return run_static_candidate(dataset_type, scenario, model, obs, position_init_label, p0_init, velocity_init_label, v_init, beta_prior_profile)
    if model in {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full"}:
        return run_ctd_candidate(dataset_type, scenario, model, obs, position_init_label, p0_init, velocity_init_label, v_init, beta_prior_profile)
    if model in {"M5_be_b0_only", "M6_be_full", "M8_robust_be_b0_only", "M9_robust_be_full"}:
        return run_be_candidate(dataset_type, scenario, model, obs, position_init_label, p0_init, velocity_init_label, v_init, beta_prior_profile)
    if model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}:
        return run_gir_candidate(dataset_type, scenario, model, obs, position_init_label, p0_init, velocity_init_label, v_init, beta_prior_profile)
    raise ValueError(f"Unknown candidate model: {model}")


def run_all_candidates(
    dataset_type: str,
    scenario: str,
    obs: dict[str, Any],
    position_init_label: str,
    p0_init: np.ndarray,
    velocity_init_label: str,
    v_init: np.ndarray,
    beta_prior_profile: str,
    candidate_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    models = candidate_models or CANDIDATE_MODEL_LIST
    return [
        run_candidate_model(model, dataset_type, scenario, obs, position_init_label, p0_init, velocity_init_label, v_init, beta_prior_profile)
        for model in models
    ]
