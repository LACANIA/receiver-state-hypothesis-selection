from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

CANDIDATES = [
    "M0_static_position", "M1_static_position_bias", "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias",
    "M5_be_b0_only", "M6_be_full", "M7_robust_ctd_full", "M8_robust_be_b0_only", "M9_robust_be_full",
    "M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale", "M12_ctd_full_plus_gir_refine",
    "M13_robust_ctd_full_plus_gir_refine", "M14_static_plus_gir_refine",
]

MODEL_DOF = {
    "M0_static_position": 3, "M1_static_position_bias": 4, "M2_ctd_full": 8, "M3_ctd_no_drift": 7,
    "M4_ctd_no_bias": 6, "M5_be_b0_only": 7, "M6_be_full": 8, "M7_robust_ctd_full": 8,
    "M8_robust_be_b0_only": 7, "M9_robust_be_full": 8, "M10_gir_tr_fixed_scale": 8,
    "M11_gir_tr_mad_scale": 8, "M12_ctd_full_plus_gir_refine": 8,
    "M13_robust_ctd_full_plus_gir_refine": 8, "M14_static_plus_gir_refine": 8,
}

MODEL_FAMILY = {
    "M0_static_position": "static", "M1_static_position_bias": "static", "M2_ctd_full": "core_ctd",
    "M3_ctd_no_drift": "core_ctd", "M4_ctd_no_bias": "core_ctd", "M5_be_b0_only": "be_gtr",
    "M6_be_full": "be_gtr", "M7_robust_ctd_full": "robust", "M8_robust_be_b0_only": "robust",
    "M9_robust_be_full": "robust", "M10_gir_tr_fixed_scale": "direct_gir",
    "M11_gir_tr_mad_scale": "direct_gir", "M12_ctd_full_plus_gir_refine": "cascaded_refinement",
    "M13_robust_ctd_full_plus_gir_refine": "cascaded_refinement",
    "M14_static_plus_gir_refine": "cascaded_refinement",
}

NO_TRUTH_COLUMNS = [
    "evaluation_dataset", "group_id", "evidence_mode", "candidate_model", "candidate_family", "dataset_type",
    "scenario", "position_init_label", "velocity_init_label", "beta_prior_profile", "numerical_success", "converged",
    "quality_pass", "physical_plausible", "state_finite", "residual_metrics_finite", "base_filter_pass",
    "base_filter_reason", "fixed_numeric_valid", "selectable", "full_residual_rmse_mps", "raw_validation_rmse_mps",
    "trimmed_validation_rmse_mps", "inlier_validation_rmse_mps", "robust_validation_cost", "condition_number",
    "position_covariance_proxy_m", "position_cov_sqrt_trace_m", "bootstrap_spread_m", "position_bootstrap_spread_m",
    "estimated_speed_mps", "beta0_estimated_mps", "beta_dot_estimated_mps2", "information_retention_ratio",
    "geometry_retention_ratio", "geometry_preserved_info_retention", "observation_count", "model_dof",
    "retention_trace_ratio", "rank_loss", "severe_risk_veto", "severe_risk_reason", "outlier_evidence",
    "burst_outlier_evidence", "tail_ratio", "mad_ratio", "refine_gate_pass", "refine_gate_reason",
    "base_model_for_refine", "robust_gate_pass", "robust_gate_reason", "consistency_gate_pass",
    "consistency_penalty", "candidate_pool_protocol", "execution_mode",
]

def scalar(value: np.ndarray | Any) -> Any:
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value

def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default

def truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default

def state_finite_from_row(row: pd.Series) -> bool:
    if not truthy(row.get("numerical_success"), False):
        return False
    model = str(row.get("candidate_model", ""))
    if not np.isfinite(safe_float(row.get("estimated_speed_mps"))):
        return False
    if model != "M0_static_position" and not np.isfinite(safe_float(row.get("beta0_estimated_mps"))):
        return False
    if model not in {"M0_static_position", "M1_static_position_bias"} and not np.isfinite(safe_float(row.get("beta_dot_estimated_mps2"))):
        return False
    for column in ["final_ecef_x_m", "final_ecef_y_m", "final_ecef_z_m"]:
        if column in row.index and not np.isfinite(safe_float(row.get(column))):
            return False
    return True

def load_method_observation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        public_anchor = np.asarray(payload["public_receiver_anchor_ecef_m"], dtype=float)
        time_s = np.asarray(payload["time_s"], dtype=float)
        n = len(time_s)
        scenario_config = json.loads(str(scalar(payload["scenario_config_json"])))
        metadata = json.loads(str(scalar(payload["observable_metadata_json"])))
        obs = {
            "lat_deg": float(scalar(payload["lat_deg"])),
            "lon_deg": float(scalar(payload["lon_deg"])),
            "height_m": float(scalar(payload["height_m"])),
            "p_gt_ecef_m": public_anchor.copy(),
            "sat_pos_m": np.asarray(payload["sat_pos_m"], dtype=float),
            "sat_vel_mps": np.asarray(payload["sat_vel_mps"], dtype=float),
            "meas_mps": np.asarray(payload["meas_mps"], dtype=float),
            "time_s": time_s,
            "satellite_number": np.asarray(payload["satellite_number"], dtype=np.int64),
            "row_index": np.asarray(payload["row_index"], dtype=np.int64),
            "t0_s": float(scalar(payload["t0_s"])),
            "p0_true_m": public_anchor.copy(),
            "v_true_mps": np.zeros(3, dtype=float),
            "b0_true_mps": np.nan,
            "bdot_true_mps2": np.nan,
            "scenario": str(scalar(payload["scenario"])),
            "scenario_config": scenario_config,
            "truth_positions_m": np.full((n, 3), np.nan, dtype=float),
            "confidence_values": np.asarray(payload["confidence_values"], dtype=float),
        }
        public = {
            "batch_id": str(scalar(payload["batch_id"])),
            "block_id": str(scalar(payload["block_id"])),
            "scenario": str(scalar(payload["scenario"])),
            "holdout_family": str(scalar(payload["holdout_family"])),
            "source_holdout_id": str(scalar(payload["source_holdout_id"])),
            "realization_index": int(scalar(payload["realization_index"])),
            "sigma_mps": float(scalar(payload["sigma_mps"])),
            "observation_count": n,
            "metadata": metadata,
        }
    return obs, public

def normalized_candidate(row: pd.Series, public: dict[str, Any]) -> dict[str, Any]:
    model = str(row.get("candidate_model", ""))
    full_residual = safe_float(row.get("full_residual_rmse_mps", row.get("residual_rmse_mps", np.nan)))
    return {
        "evaluation_dataset": "D2_CROSS_EPOCH",
        "group_id": public["batch_id"],
        "evidence_mode": "observable_only",
        "candidate_model": model,
        "candidate_family": MODEL_FAMILY.get(model, "unknown"),
        "dataset_type": "synthetic",
        "scenario": public["scenario"],
        "position_init_label": str(row.get("position_init_label", "east_10km")),
        "velocity_init_label": str(row.get("velocity_init_label", "zero_velocity")),
        "beta_prior_profile": str(row.get("beta_prior_profile", "B0_none")),
        "numerical_success": truthy(row.get("numerical_success"), False),
        "converged": truthy(row.get("converged"), False),
        "quality_pass": truthy(row.get("quality_pass"), False),
        "physical_plausible": truthy(row.get("physical_plausible"), False),
        "state_finite": state_finite_from_row(row),
        "residual_metrics_finite": bool(
            np.isfinite(full_residual)
            and np.isfinite(safe_float(row.get("raw_validation_rmse_mps")))
            and np.isfinite(safe_float(row.get("trimmed_validation_rmse_mps")))
            and np.isfinite(safe_float(row.get("inlier_validation_rmse_mps")))
        ),
        "full_residual_rmse_mps": full_residual,
        "raw_validation_rmse_mps": safe_float(row.get("raw_validation_rmse_mps")),
        "trimmed_validation_rmse_mps": safe_float(row.get("trimmed_validation_rmse_mps")),
        "inlier_validation_rmse_mps": safe_float(row.get("inlier_validation_rmse_mps")),
        "robust_validation_cost": safe_float(row.get("robust_validation_cost")),
        "condition_number": safe_float(row.get("condition_number")),
        "position_covariance_proxy_m": safe_float(row.get("position_cov_sqrt_trace_m")),
        "position_cov_sqrt_trace_m": safe_float(row.get("position_cov_sqrt_trace_m")),
        "bootstrap_spread_m": safe_float(row.get("position_bootstrap_spread_m")),
        "position_bootstrap_spread_m": safe_float(row.get("position_bootstrap_spread_m")),
        "estimated_speed_mps": safe_float(row.get("estimated_speed_mps")),
        "beta0_estimated_mps": safe_float(row.get("beta0_estimated_mps")),
        "beta_dot_estimated_mps2": safe_float(row.get("beta_dot_estimated_mps2")),
        "information_retention_ratio": safe_float(row.get("information_retention_ratio")),
        "geometry_retention_ratio": safe_float(row.get("geometry_preserved_info_retention")),
        "geometry_preserved_info_retention": safe_float(row.get("geometry_preserved_info_retention")),
        "observation_count": int(public["observation_count"]),
        "model_dof": int(safe_float(row.get("model_dof"), MODEL_DOF.get(model, -1))),
        "retention_trace_ratio": safe_float(row.get("retention_trace_ratio")),
        "rank_loss": safe_float(row.get("rank_loss")),
        "severe_risk_veto": truthy(row.get("severe_risk_veto"), False),
        "severe_risk_reason": str(row.get("severe_risk_reason", "")),
        "outlier_evidence": truthy(row.get("outlier_evidence"), False),
        "burst_outlier_evidence": truthy(row.get("burst_outlier_evidence"), False),
        "tail_ratio": safe_float(row.get("tail_ratio")),
        "mad_ratio": safe_float(row.get("mad_ratio")),
        "refine_gate_pass": truthy(row.get("refine_gate_pass"), True),
        "refine_gate_reason": str(row.get("refine_gate_reason", "")),
        "base_model_for_refine": str(row.get("base_model_for_refine", "")),
        "robust_gate_pass": truthy(row.get("robust_gate_pass"), True),
        "robust_gate_reason": str(row.get("robust_gate_reason", "")),
        "consistency_gate_pass": truthy(row.get("consistency_gate_pass"), True),
        "consistency_penalty": safe_float(row.get("consistency_penalty")),
        "candidate_pool_protocol": str(row.get("candidate_pool_protocol", "FULL_POOL_M0_M14_V1")),
        "execution_mode": str(row.get("execution_mode", "solver_rerun")),
    }

def apply_common_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    from leo_positioning.family_evidence import STATIC_MODELS, best_candidate
    from leo_positioning.hierarchical_gates import HierarchicalGateConfig, _base_filter

    result = frame.copy()
    result["base_filter_pass"] = False
    result["base_filter_reason"] = ""
    for _, group in result.groupby(["evaluation_dataset", "group_id"], sort=False):
        best_static = best_candidate(group, STATIC_MODELS)
        eligible, reasons = _base_filter(group, best_static, HierarchicalGateConfig())
        result.loc[group.index, "base_filter_reason"] = [reasons.get(int(index), "") for index in group.index]
        result.loc[eligible.index, "base_filter_pass"] = True
    result["fixed_numeric_valid"] = (
        result["numerical_success"].map(truthy)
        & result["state_finite"].map(truthy)
        & result["residual_metrics_finite"].map(truthy)
    )
    result["selectable"] = result["fixed_numeric_valid"] & result["base_filter_pass"].map(truthy)
    return result

def forbidden_candidate_columns(columns: list[str]) -> list[str]:
    forbidden: list[str] = []
    for column in columns:
        lower = column.lower()
        if (
            "truth" in lower
            or lower.startswith("true_")
            or "oracle" in lower
            or "position_error" in lower
            or "velocity_error" in lower
            or "beta0_error" in lower
            or "beta_dot_error" in lower
            or "p0_init_error" in lower
            or "v_init_error" in lower
        ):
            forbidden.append(column)
    return sorted(forbidden)

def selected_output_row(batch_id: str, model: str, full_safe: pd.DataFrame, status: str, reason: str, low_quality: bool) -> dict[str, Any]:
    match = full_safe.loc[(full_safe["batch_id"].astype(str) == batch_id) & (full_safe["candidate_model"].astype(str) == model)]
    if len(match) != 1:
        raise RuntimeError("d2_execution_blocked_candidate_identity")
    row = match.iloc[0]
    return {
        "batch_id": batch_id,
        "selected_candidate": model,
        "status": status,
        "selection_reason": reason,
        "low_quality": bool(low_quality),
        "numerical_success": truthy(row.get("numerical_success"), False),
        "final_ecef_x_m": safe_float(row.get("final_ecef_x_m")),
        "final_ecef_y_m": safe_float(row.get("final_ecef_y_m")),
        "final_ecef_z_m": safe_float(row.get("final_ecef_z_m")),
        "estimated_speed_mps": safe_float(row.get("estimated_speed_mps")),
        "beta0_estimated_mps": safe_float(row.get("beta0_estimated_mps")),
        "beta_dot_estimated_mps2": safe_float(row.get("beta_dot_estimated_mps2")),
    }

def fixed_selection(batch_id: str, model: str, full_safe: pd.DataFrame) -> dict[str, Any]:
    match = full_safe.loc[(full_safe["batch_id"].astype(str) == batch_id) & (full_safe["candidate_model"].astype(str) == model)]
    if len(match) != 1:
        raise RuntimeError("d2_execution_blocked_candidate_identity")
    row = match.iloc[0]
    status = "selected" if truthy(row.get("fixed_numeric_valid"), False) else "explicit_failure"
    return selected_output_row(batch_id, model, full_safe, status, "fixed_candidate_no_fallback", status != "selected")

def information_criterion_selection(batch_id: str, method: str, sigma0: float, group: pd.DataFrame, full_safe: pd.DataFrame) -> dict[str, Any]:
    compatible = group.loc[group["selectable"].map(truthy) & group["candidate_model"].str.extract(r"^M(\d+)", expand=False).astype(int).between(0, 4)].copy()
    compatible["candidate_order"] = compatible["candidate_model"].str.extract(r"^M(\d+)", expand=False).astype(int)
    compatible["chi2"] = compatible["observation_count"].astype(float) * np.square(compatible["full_residual_rmse_mps"].astype(float)) / (sigma0 * sigma0)
    compatible["AIC"] = compatible["chi2"] + 2.0 * compatible["model_dof"].astype(float)
    denominator = compatible["observation_count"].astype(float) - compatible["model_dof"].astype(float) - 1.0
    compatible["FSC"] = compatible["AIC"] + 2.0 * compatible["model_dof"].astype(float) * (compatible["model_dof"].astype(float) + 1.0) / denominator
    compatible.loc[denominator <= 0.0, "FSC"] = np.nan
    compatible["BIC"] = compatible["chi2"] + compatible["model_dof"].astype(float) * np.log(compatible["observation_count"].astype(float))
    valid = compatible.loc[np.isfinite(compatible[method])].copy()
    if valid.empty:
        return {
            "batch_id": batch_id, "selected_candidate": "", "status": "explicit_failure",
            "selection_reason": f"{method}_no_valid_M0_M4_candidate", "low_quality": True,
            "criterion_value": np.nan, "tie_count": 0, "numerical_success": False,
            "final_ecef_x_m": np.nan, "final_ecef_y_m": np.nan, "final_ecef_z_m": np.nan,
            "estimated_speed_mps": np.nan, "beta0_estimated_mps": np.nan, "beta_dot_estimated_mps2": np.nan,
        }
    ordered = valid.sort_values([method, "model_dof", "candidate_order"], kind="mergesort")
    selected = ordered.iloc[0]
    tie_count = int(np.sum(np.isclose(ordered[method].to_numpy(float), float(selected[method]), rtol=0.0, atol=0.0)))
    result = selected_output_row(batch_id, str(selected["candidate_model"]), full_safe, "selected", f"{method}_known_Sigma0_M0_M4", False)
    result["criterion_value"] = float(selected[method])
    result["tie_count"] = tie_count
    return result
