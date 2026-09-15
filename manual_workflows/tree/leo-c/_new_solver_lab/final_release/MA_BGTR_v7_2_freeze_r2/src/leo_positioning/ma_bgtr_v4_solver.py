"""MA-BGTR-v4: cascade refinement plus state consistency gates."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .cascade_refinement import cascade_positions, evaluate_full_state, refine_gate, run_cascade_refinement
from .consistency_gates import apply_consistency_gates, compute_consistency_diagnostics
from .influence_diagnostics import diagnostics_dict, validation_residual_metrics
from .ma_bgtr_v3_solver import V3_CANDIDATE_MODEL_LIST, fit_candidate_v3
from .ma_bgtr_v2_solver import evaluate_fit_residual
from .model_candidates import base_candidate_row, finalize_candidate, run_candidate_model
from .model_scoring import V4ScoringConfig, compute_score_v4
from .robust_gates import apply_robust_hard_gate
from .trajectory_models import FULL_CTD_CONFIG, residuals_and_jacobian_ctd, unpack_state
from .validation_split import blocked_train_validation


V4_CANDIDATE_MODEL_LIST = [
    *V3_CANDIDATE_MODEL_LIST,
    "M12_ctd_full_plus_gir_refine",
    "M13_robust_ctd_full_plus_gir_refine",
    "M14_static_plus_gir_refine",
]


def _add_common_defaults(row: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "robust_gate_pass": True,
        "robust_gate_reason": "",
        "trimmed_gain_vs_ctd": 0.0,
        "inlier_gain_vs_ctd": 0.0,
        "full_rmse_ratio_vs_ctd": np.nan,
        "consistency_gate_pass": True,
        "consistency_penalty": 0.0,
        "refine_gate_pass": True,
        "refine_gate_reason": "",
        "base_model_for_refine": "",
    }
    for key, value in defaults.items():
        row.setdefault(key, value)
    return row


def _validation_for_standard(model: str, full_row: dict[str, Any], train_obs: dict[str, Any], val_obs: dict[str, Any], p0: np.ndarray, v0: np.ndarray, profile: str, dataset_type: str) -> dict[str, Any]:
    train_fit = fit_candidate_v3(model, train_obs, p0, v0, profile, dataset_type)
    val_residual = evaluate_fit_residual(train_fit, val_obs)
    train_residual = evaluate_fit_residual(train_fit, train_obs)
    row = dict(full_row)
    row.update(validation_residual_metrics(val_residual, trim_fraction=0.10, c_scale_mps=2.0))
    row["train_residual_rmse_mps"] = validation_residual_metrics(train_residual)["raw_validation_rmse_mps"]
    row["full_residual_rmse_mps"] = float(row.get("residual_rmse_mps", np.nan))
    if model in {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full"}:
        row["information_retention_ratio"] = 1.0
    row.setdefault("information_retention_ratio", np.nan)
    if not train_fit.success:
        row["numerical_success"] = False
        row["quality_pass"] = False
        row["failure_reason"] = ";".join(r for r in [str(row.get("failure_reason", "")), train_fit.failure_reason] if r)
    return _add_common_defaults(row)


def _cascade_full_row(model: str, obs: dict[str, Any], p0: np.ndarray, v0: np.ndarray, dataset_type: str, scenario: str, pos_label: str, vel_label: str, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    run = run_cascade_refinement(model, obs, p0, v0, dataset_type)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    row = base_candidate_row(dataset_type, scenario, model, pos_label, vel_label, profile, p0, v0, obs)
    positions = cascade_positions(run.refined_state, obs)
    _p, v, b0, bdot = unpack_state(run.refined_state, FULL_CTD_CONFIG)
    full_row = finalize_candidate(
        row,
        obs,
        positions,
        v,
        b0,
        bdot,
        run.refined_residual,
        run.refined_condition_number,
        bool(run.gir_result.numerical_success),
        bool(run.gir_result.converged),
        runtime_ms,
        run.failure_reason,
    )
    full_row.update(diagnostics_dict(run.gir_result.diagnostics))
    full_row.update(
        {
            "accepted_steps": int(run.gir_result.accepted_steps),
            "rejected_steps": int(run.gir_result.rejected_steps),
            "base_model_for_refine": run.base_model,
        }
    )
    diag = {
        "dataset_type": dataset_type,
        "scenario": scenario,
        "position_init_label": pos_label,
        "velocity_init_label": vel_label,
        "beta_prior_profile": profile,
        "refine_model": model,
        "base_model": run.base_model,
        "base_condition_number": run.base_condition_number,
        "refined_condition_number": run.refined_condition_number,
        "base_position_error_m": np.nan,
        "refined_position_error_m": full_row.get("final_position_error_m", np.nan),
    }
    return full_row, diag


def _validation_for_cascade(model: str, full_row: dict[str, Any], diag: dict[str, Any], train_obs: dict[str, Any], val_obs: dict[str, Any], p0: np.ndarray, v0: np.ndarray, dataset_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    train_run = run_cascade_refinement(model, train_obs, p0, v0, dataset_type)
    base_val_residual, _base_j, base_val_condition = evaluate_full_state(train_run.base_state, val_obs)
    refined_val_residual, _ref_j, refined_val_condition = evaluate_full_state(train_run.refined_state, val_obs)
    base_metrics = validation_residual_metrics(base_val_residual, trim_fraction=0.10, c_scale_mps=2.0)
    refined_metrics = validation_residual_metrics(refined_val_residual, trim_fraction=0.10, c_scale_mps=2.0)
    info = float(diagnostics_dict(train_run.gir_result.diagnostics).get("information_retention_ratio", np.nan))
    passed, reason = refine_gate(
        dataset_type,
        base_metrics,
        refined_metrics,
        base_val_condition,
        refined_val_condition,
        train_run.refined_state,
        info,
        train_run.gir_result.accepted_steps,
    )
    row = dict(full_row)
    row.update(refined_metrics)
    row["train_residual_rmse_mps"] = validation_residual_metrics(train_run.refined_residual)["raw_validation_rmse_mps"]
    row["full_residual_rmse_mps"] = float(row.get("residual_rmse_mps", np.nan))
    row["information_retention_ratio"] = info
    row["refine_gate_pass"] = bool(passed)
    row["refine_gate_reason"] = reason
    row["base_model_for_refine"] = train_run.base_model
    row = _add_common_defaults(row)
    diag.update(
        {
            "base_trimmed_validation_rmse_mps": base_metrics["trimmed_validation_rmse_mps"],
            "refined_trimmed_validation_rmse_mps": refined_metrics["trimmed_validation_rmse_mps"],
            "base_inlier_validation_rmse_mps": base_metrics["inlier_validation_rmse_mps"],
            "refined_inlier_validation_rmse_mps": refined_metrics["inlier_validation_rmse_mps"],
            "base_condition_number": base_val_condition,
            "refined_condition_number": refined_val_condition,
            "refine_gate_pass": bool(passed),
            "refine_gate_reason": reason,
        }
    )
    return row, diag


def build_v4_candidate_rows_for_job(
    observations: dict[str, Any],
    initial_guess: dict[str, Any],
    candidate_models: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models = candidate_models or V4_CANDIDATE_MODEL_LIST
    train_obs, val_obs, _split = blocked_train_validation(observations, train_fraction=0.70)
    dataset_type = str(initial_guess["dataset_type"])
    scenario = str(initial_guess["scenario"])
    pos_label = str(initial_guess["position_init_label"])
    vel_label = str(initial_guess["velocity_init_label"])
    profile = str(initial_guess["beta_prior_profile"])
    p0 = np.asarray(initial_guess["p0_init"], dtype=float)
    v0 = np.asarray(initial_guess["v_init"], dtype=float)

    rows: list[dict[str, Any]] = []
    refine_diags: list[dict[str, Any]] = []
    for model in models:
        if model in {"M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine", "M14_static_plus_gir_refine"}:
            full_row, diag = _cascade_full_row(model, observations, p0, v0, dataset_type, scenario, pos_label, vel_label, profile)
            row, diag = _validation_for_cascade(model, full_row, diag, train_obs, val_obs, p0, v0, dataset_type)
            rows.append(row)
            refine_diags.append(diag)
        else:
            full_row = run_candidate_model(model, dataset_type, scenario, observations, pos_label, p0, vel_label, v0, profile)
            rows.append(_validation_for_standard(model, full_row, train_obs, val_obs, p0, v0, profile, dataset_type))
    rows, robust_diags = apply_robust_hard_gate(rows)
    return rows, robust_diags, refine_diags


def apply_consistency_and_score(all_rows: list[dict[str, Any]], n_val: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    consistency = compute_consistency_diagnostics(all_rows)
    rows = apply_consistency_gates(all_rows, consistency)
    scored: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged.update(compute_score_v4(merged, n_val, V4ScoringConfig()))
        scored.append(merged)
    selected_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in scored:
        key = (
            str(row.get("dataset_type")),
            str(row.get("scenario")),
            str(row.get("position_init_label")),
            str(row.get("velocity_init_label")),
            str(row.get("beta_prior_profile")),
        )
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        selectable = sorted([r for r in group if bool(r.get("selectable_v4", False))], key=lambda r: float(r["total_score_v4"]))
        if selectable:
            selected = selectable[0]
            low_quality = False
            reason = "lowest_v4_score_after_robust_refine_consistency_gates"
        else:
            finite = sorted([r for r in group if np.isfinite(float(r.get("total_score_v4", np.inf)))], key=lambda r: float(r["total_score_v4"]))
            selected = finite[0] if finite else group[0]
            low_quality = True
            reason = "no_selectable_candidate; lowest_reported_v4_score"
        runner = next((r for r in selectable[1:] if r.get("candidate_model") != selected.get("candidate_model")), None)
        if runner is None:
            sorted_all = sorted(group, key=lambda r: float(r.get("total_score_v4", 1e9)))
            runner = next((r for r in sorted_all if r.get("candidate_model") != selected.get("candidate_model")), selected)
        selected_error = float(selected.get("final_position_error_m", np.nan))
        finite_errors = [
            float(r.get("final_position_error_m", 1e7)) if np.isfinite(float(r.get("final_position_error_m", np.nan))) else 1e7
            for r in group
        ]
        oracle_error = min(finite_errors)
        ctd_error = next((float(r.get("final_position_error_m", np.nan)) for r in group if r["candidate_model"] == "M2_ctd_full"), np.nan)
        static_error = next((float(r.get("final_position_error_m", np.nan)) for r in group if r["candidate_model"] == "M0_static_position"), np.nan)
        selected_rows.append(
            {
                "dataset_type": key[0],
                "scenario": key[1],
                "position_init_label": key[2],
                "velocity_init_label": key[3],
                "beta_prior_profile": key[4],
                "selected_model_v4": selected.get("candidate_model"),
                "selected_score_v4": selected.get("total_score_v4"),
                "selected_low_quality": low_quality,
                "selection_reason_v4": reason,
                "true_speed_mps": selected.get("true_speed_mps", np.nan),
                "true_b0_mps": selected.get("true_b0_mps", np.nan),
                "true_bdot_mps2": selected.get("true_bdot_mps2", np.nan),
                "final_position_error_m": selected.get("final_position_error_m", np.nan),
                "mean_position_error_m": selected.get("mean_position_error_m", np.nan),
                "max_position_error_m": selected.get("max_position_error_m", np.nan),
                "velocity_error_mps": selected.get("velocity_error_mps", np.nan),
                "estimated_speed_mps": selected.get("estimated_speed_mps", np.nan),
                "beta0_estimated_mps": selected.get("beta0_estimated_mps", np.nan),
                "beta_dot_estimated_mps2": selected.get("beta_dot_estimated_mps2", np.nan),
                "beta0_error_mps": selected.get("beta0_error_mps", np.nan),
                "beta_dot_error_mps2": selected.get("beta_dot_error_mps2", np.nan),
                "raw_validation_rmse_mps": selected.get("raw_validation_rmse_mps", np.nan),
                "trimmed_validation_rmse_mps": selected.get("trimmed_validation_rmse_mps", np.nan),
                "inlier_validation_rmse_mps": selected.get("inlier_validation_rmse_mps", np.nan),
                "full_residual_rmse_mps": selected.get("full_residual_rmse_mps", np.nan),
                "information_retention_ratio": selected.get("information_retention_ratio", np.nan),
                "robust_gate_pass": selected.get("robust_gate_pass", True),
                "consistency_gate_pass": selected.get("consistency_gate_pass", True),
                "refine_gate_pass": selected.get("refine_gate_pass", True),
                "quality_pass": selected.get("quality_pass", False),
                "physical_plausible": selected.get("physical_plausible", False),
                "runtime_ms_total": sum(float(r.get("runtime_ms", 0.0)) for r in group if np.isfinite(float(r.get("runtime_ms", np.nan)))),
                "oracle_best_position_error_m": oracle_error,
                "selected_minus_oracle_error_m": selected_error - oracle_error if np.isfinite(selected_error) else np.nan,
                "selected_beats_ctd": bool(np.isfinite(selected_error) and np.isfinite(ctd_error) and selected_error <= ctd_error),
                "selected_beats_static_lm": bool(np.isfinite(selected_error) and np.isfinite(static_error) and selected_error <= static_error),
            }
        )
        score_rows.append(
            {
                "dataset_type": key[0],
                "scenario": key[1],
                "position_init_label": key[2],
                "velocity_init_label": key[3],
                "beta_prior_profile": key[4],
                "selected_model_v4": selected.get("candidate_model"),
                "runner_up_model_v4": runner.get("candidate_model"),
                "selected_total_score_v4": selected.get("total_score_v4"),
                "runner_up_total_score_v4": runner.get("total_score_v4"),
                "score_margin": float(runner.get("total_score_v4", 0.0)) - float(selected.get("total_score_v4", 0.0)),
                "selected_validation_component": selected.get("validation_component"),
                "selected_robust_component": selected.get("robust_component"),
                "selected_complexity_penalty": selected.get("complexity_penalty"),
                "selected_condition_penalty": selected.get("condition_penalty"),
                "selected_physical_penalty": selected.get("physical_penalty"),
                "selected_robust_gate_penalty": selected.get("robust_gate_penalty"),
                "selected_consistency_penalty": selected.get("consistency_penalty"),
                "selected_refine_penalty": selected.get("refine_penalty"),
                "selected_real_conservative_penalty": selected.get("real_conservative_penalty"),
            }
        )
    return scored, selected_rows, score_rows, consistency
