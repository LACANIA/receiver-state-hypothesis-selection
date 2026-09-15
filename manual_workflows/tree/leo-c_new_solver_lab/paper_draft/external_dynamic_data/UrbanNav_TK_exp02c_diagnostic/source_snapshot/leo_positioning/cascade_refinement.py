"""Cascade initialization and GIR-TR local refinement for MA-BGTR-v4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gir_tr_solver import GirTrResult, solve_gir_tr
from .influence_diagnostics import diagnostics_dict, validation_residual_metrics
from .models import residuals_and_jacobian
from .quality import physical_plausibility
from .solvers import solve_lm
from .trajectory_models import FULL_CTD_CONFIG, pack_state, residuals_and_jacobian_ctd, trajectory_positions, unpack_state, weighted_normal_condition
from .trajectory_solvers import solve_ctd_lm


@dataclass
class CascadeRun:
    refine_model: str
    base_model: str
    base_state: np.ndarray
    refined_state: np.ndarray
    base_residual: np.ndarray
    refined_residual: np.ndarray
    base_condition_number: float
    refined_condition_number: float
    gir_result: GirTrResult
    failure_reason: str = ""


def base_model_for_refine(refine_model: str) -> str:
    if refine_model == "M12_ctd_full_plus_gir_refine":
        return "M2_ctd_full"
    if refine_model == "M13_robust_ctd_full_plus_gir_refine":
        return "M7_robust_ctd_full"
    if refine_model == "M14_static_plus_gir_refine":
        return "M0_static_position"
    raise ValueError(f"Unknown cascade model: {refine_model}")


def _fit_base_ctd(base_model: str, obs: dict, p0_init: np.ndarray, v_init: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    robust = base_model == "M7_robust_ctd_full"
    t0_s = float(obs.get("t0_s", np.min(obs["time_s"])))
    theta0 = pack_state(p0_init, v_init, 0.0, 0.0, FULL_CTD_CONFIG)
    result = solve_ctd_lm(theta0, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t0_s, FULL_CTD_CONFIG, robust=robust, max_iter=80)
    residual, jac, _pred = residuals_and_jacobian_ctd(result.state, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t0_s, FULL_CTD_CONFIG)
    condition = weighted_normal_condition(jac, residual, robust)
    return result.state.copy(), residual, condition, result.failure_reason


def _fit_base_static(obs: dict, p0_init: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    result = solve_lm(p0_init, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=False, robust=False)
    residual, jac, _pred = residuals_and_jacobian(result.state, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=False)
    theta = pack_state(result.state[:3], np.zeros(3), 0.0, 0.0, FULL_CTD_CONFIG)
    try:
        condition = float(np.linalg.cond(jac.T @ jac))
    except Exception:
        condition = np.inf
    return theta, residual, condition, result.failure_reason


def run_cascade_refinement(
    refine_model: str,
    obs: dict,
    p0_init: np.ndarray,
    v_init: np.ndarray,
    dataset_type: str,
    scale_mode: str = "fixed",
) -> CascadeRun:
    base_model = base_model_for_refine(refine_model)
    t0_s = float(obs.get("t0_s", np.min(obs["time_s"])))
    if base_model in {"M2_ctd_full", "M7_robust_ctd_full"}:
        base_state, base_residual, base_condition, base_reason = _fit_base_ctd(base_model, obs, p0_init, v_init)
    else:
        base_state, base_residual, base_condition, base_reason = _fit_base_static(obs, p0_init)
    gir = solve_gir_tr(
        base_state,
        obs["time_s"],
        obs["sat_pos_m"],
        obs["sat_vel_mps"],
        obs["meas_mps"],
        t0_s,
        scale_mode=scale_mode,
        dataset_type=dataset_type,
        trust_radius0=10_000.0,
        max_iter_per_scale=6,
    )
    refined_residual, _jac, _pred = residuals_and_jacobian_ctd(
        gir.state, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t0_s, FULL_CTD_CONFIG
    )
    return CascadeRun(
        refine_model=refine_model,
        base_model=base_model,
        base_state=base_state,
        refined_state=gir.state.copy(),
        base_residual=base_residual,
        refined_residual=refined_residual,
        base_condition_number=float(base_condition),
        refined_condition_number=float(gir.condition_number),
        gir_result=gir,
        failure_reason=";".join(r for r in [base_reason, gir.failure_reason] if r),
    )


def evaluate_full_state(theta: np.ndarray, obs: dict) -> tuple[np.ndarray, np.ndarray, float]:
    t0_s = float(obs.get("t0_s", np.min(obs["time_s"])))
    residual, jac, _pred = residuals_and_jacobian_ctd(theta, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t0_s, FULL_CTD_CONFIG)
    try:
        condition = float(np.linalg.cond(jac.T @ jac))
    except Exception:
        condition = np.inf
    return residual, jac, condition


def cascade_positions(theta: np.ndarray, obs: dict) -> np.ndarray:
    return trajectory_positions(theta, obs["time_s"], float(obs.get("t0_s", np.min(obs["time_s"]))), FULL_CTD_CONFIG)


def refine_gate(
    dataset_type: str,
    base_metrics: dict[str, float],
    refined_metrics: dict[str, float],
    base_condition: float,
    refined_condition: float,
    refined_state: np.ndarray,
    information_retention_ratio: float,
    accepted_steps: int,
) -> tuple[bool, str]:
    reasons: list[str] = []
    raw_ratio = refined_metrics["raw_validation_rmse_mps"] / max(base_metrics["raw_validation_rmse_mps"], 1e-9)
    trimmed_gain = (base_metrics["trimmed_validation_rmse_mps"] - refined_metrics["trimmed_validation_rmse_mps"]) / max(base_metrics["trimmed_validation_rmse_mps"], 1e-9)
    inlier_gain = (base_metrics["inlier_validation_rmse_mps"] - refined_metrics["inlier_validation_rmse_mps"]) / max(base_metrics["inlier_validation_rmse_mps"], 1e-9)
    cond_ratio = refined_condition / max(base_condition, 1e-9) if np.isfinite(refined_condition) and np.isfinite(base_condition) else np.inf
    _p, v, b0, bdot = unpack_state(refined_state, FULL_CTD_CONFIG)
    physical, physical_reason = physical_plausibility(dataset_type, float(np.linalg.norm(v)), float(b0), float(bdot))
    if raw_ratio > 1.01:
        reasons.append("raw_validation_worse_gt_1pct")
    if trimmed_gain < 0.02 and inlier_gain < 0.02:
        reasons.append("trimmed_inlier_gain_lt_2pct")
    if cond_ratio > 10.0:
        reasons.append("condition_worse_gt_10x")
    if not physical:
        reasons.append(f"physical:{physical_reason}")
    if not np.isfinite(information_retention_ratio) or information_retention_ratio < 0.5:
        reasons.append("information_retention_lt_0_5")
    if int(accepted_steps) <= 0:
        reasons.append("gir_no_accepted_step")
    if dataset_type == "real":
        if float(np.linalg.norm(v)) >= 2.0:
            reasons.append("real_speed_ge_2mps")
        if abs(float(bdot)) >= 0.02:
            reasons.append("real_bdot_ge_0_02")
    return len(reasons) == 0, ";".join(reasons)


def refinement_metrics(run: CascadeRun, obs: dict) -> dict[str, float | str | bool]:
    base_metrics = validation_residual_metrics(run.base_residual)
    refined_metrics = validation_residual_metrics(run.refined_residual)
    diag = diagnostics_dict(run.gir_result.diagnostics)
    passed, reason = refine_gate(
        "synthetic",
        base_metrics,
        refined_metrics,
        run.base_condition_number,
        run.refined_condition_number,
        run.refined_state,
        float(diag.get("information_retention_ratio", np.nan)),
        run.gir_result.accepted_steps,
    )
    return {
        "base_model": run.base_model,
        "base_raw_validation_rmse_mps": base_metrics["raw_validation_rmse_mps"],
        "refined_raw_validation_rmse_mps": refined_metrics["raw_validation_rmse_mps"],
        "base_trimmed_validation_rmse_mps": base_metrics["trimmed_validation_rmse_mps"],
        "refined_trimmed_validation_rmse_mps": refined_metrics["trimmed_validation_rmse_mps"],
        "base_inlier_validation_rmse_mps": base_metrics["inlier_validation_rmse_mps"],
        "refined_inlier_validation_rmse_mps": refined_metrics["inlier_validation_rmse_mps"],
        "base_condition_number": run.base_condition_number,
        "refined_condition_number": run.refined_condition_number,
        "refine_gate_pass": bool(passed),
        "refine_gate_reason": reason,
    }
