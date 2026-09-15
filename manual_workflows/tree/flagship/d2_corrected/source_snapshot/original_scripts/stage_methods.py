from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from d2_common import (
    EXEC_ROOT,
    PROTOCOL_ROOT,
    RELEASE_ROOT,
    RELEASE_RUNTIME,
    SELECTOR_REPLAY,
    SELECTOR_SPEC,
    import_module,
    load_json,
    plain_path,
    sha256_file,
    utc_now_z,
    write_json,
)


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


def main() -> None:
    started = time.perf_counter()
    observation_lock = load_json(EXEC_ROOT / "D2_OBSERVATION_LOCK.json")
    if not observation_lock["observation_locked"] or observation_lock["truth_only_field_leakage_count"] != 0:
        raise RuntimeError("d2_execution_blocked_observation_schema")
    if sha256_file(EXEC_ROOT / "D2_OBSERVATION_MANIFEST.csv") != observation_lock["observation_manifest_sha256"]:
        raise RuntimeError("d2_execution_blocked_protected_asset_change")
    release_src = RELEASE_ROOT / "src"
    if str(release_src) not in sys.path:
        sys.path.insert(0, str(release_src))
    rr = import_module(RELEASE_RUNTIME, "d2_release_runtime")
    replay = import_module(SELECTOR_REPLAY, "d2_publication_selector")
    spec = load_json(SELECTOR_SPEC)
    if list(rr.V4_CANDIDATE_MODEL_LIST) != CANDIDATES:
        raise RuntimeError("d2_execution_blocked_candidate_identity")

    candidate_dir = EXEC_ROOT / "candidate_outputs"
    method_dir = EXEC_ROOT / "method_outputs"
    baseline_dir = EXEC_ROOT / "baseline_outputs"
    runtime_dir = EXEC_ROOT / "runtime"
    per_batch_dir = candidate_dir / "per_batch"
    for directory in [candidate_dir, method_dir, baseline_dir, runtime_dir, per_batch_dir]:
        directory.mkdir(exist_ok=False)

    observation_manifest = pd.read_csv(EXEC_ROOT / "D2_OBSERVATION_MANIFEST.csv").sort_values("batch_id")
    all_full_rows: list[pd.DataFrame] = []
    all_no_truth_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    for index, record in enumerate(observation_manifest.itertuples(index=False), start=1):
        batch_started = time.perf_counter()
        observation_path = Path("\\\\?\\" + str(record.observation_file))
        if sha256_file(observation_path) != str(record.observation_sha256):
            raise RuntimeError("d2_execution_blocked_protected_asset_change")
        obs, public = load_method_observation(observation_path)
        frame = rr.run_actual_candidates(
            "synthetic",
            public["scenario"],
            obs,
            public["metadata"],
            candidate_models=CANDIDATES,
            position_init_label="east_10km",
            velocity_init_label="zero_velocity",
            beta_prior_profile="B0_none",
        )
        if len(frame) != 15 or set(frame["candidate_model"].astype(str)) != set(CANDIDATES):
            raise RuntimeError("d2_execution_blocked_candidate_identity")
        normalized = pd.DataFrame([normalized_candidate(row, public) for _, row in frame.iterrows()])
        normalized = apply_common_eligibility(normalized)
        no_truth = normalized.loc[:, NO_TRUTH_COLUMNS].copy()
        if list(no_truth.columns) != NO_TRUTH_COLUMNS or len(no_truth) != 15:
            raise RuntimeError("d2_execution_blocked_candidate_identity")
        if forbidden_candidate_columns(list(no_truth.columns)):
            raise RuntimeError("d2_execution_blocked_truth_leakage")

        full_safe = frame.copy()
        drop_columns = forbidden_candidate_columns(list(full_safe.columns))
        full_safe = full_safe.drop(columns=drop_columns, errors="ignore")
        full_safe["batch_id"] = public["batch_id"]
        full_safe["block_id"] = public["block_id"]
        full_safe["holdout_family"] = public["holdout_family"]
        full_safe["source_holdout_id"] = public["source_holdout_id"]
        full_safe["realization_index"] = public["realization_index"]
        eligibility_columns = normalized.set_index("candidate_model")[["state_finite", "residual_metrics_finite", "base_filter_pass", "base_filter_reason", "fixed_numeric_valid", "selectable"]]
        for column in eligibility_columns.columns:
            full_safe[column] = full_safe["candidate_model"].map(eligibility_columns[column])
        if forbidden_candidate_columns(list(full_safe.columns)):
            raise RuntimeError("d2_execution_blocked_truth_leakage")
        per_batch_path = per_batch_dir / f"{public['batch_id']}_CANDIDATES.csv"
        full_safe.to_csv(per_batch_path, index=False)
        all_full_rows.append(full_safe)
        all_no_truth_rows.append(no_truth)
        elapsed = time.perf_counter() - batch_started
        runtime_rows.append(
            {
                "batch_id": public["batch_id"],
                "candidate_wall_seconds": elapsed,
                "sum_candidate_runtime_seconds": float(pd.to_numeric(frame.get("runtime_seconds", np.nan), errors="coerce").sum()),
                "candidate_record_count": len(frame),
            }
        )
        if index % 12 == 0 or index == len(observation_manifest):
            print(json.dumps({"stage": "candidate_execution", "batches_completed": index, "batches_total": len(observation_manifest), "candidate_records": index * 15, "truth_reads": 0}), flush=True)

    full_safe = pd.concat(all_full_rows, ignore_index=True)
    no_truth = pd.concat(all_no_truth_rows, ignore_index=True)
    full_path = candidate_dir / "D2_ALL_CANDIDATE_RECORDS.csv"
    no_truth_path = candidate_dir / "D2_CANDIDATE_RECORDS_NO_TRUTH.csv"
    full_safe.to_csv(full_path, index=False)
    no_truth.to_csv(no_truth_path, index=False)
    if len(full_safe) != len(observation_manifest) * 15 or len(no_truth) != len(full_safe):
        raise RuntimeError("d2_execution_blocked_candidate_identity")
    leakage = forbidden_candidate_columns(list(full_safe.columns)) + forbidden_candidate_columns(list(no_truth.columns))
    if leakage:
        raise RuntimeError("d2_execution_blocked_truth_leakage")

    egshs_rows: list[dict[str, Any]] = []
    fixed_rows = {"M0": [], "M2": [], "M12": []}
    ic_rows = {"AIC": [], "FSC": [], "BIC": []}
    selector_runtime: list[float] = []
    comparator_runtime: list[float] = []
    sigma_map = observation_manifest.set_index("batch_id")["sigma_mps"].astype(float).to_dict()
    for batch_id, group in no_truth.groupby("group_id", sort=True):
        select_started = time.perf_counter()
        selected = replay.select_group(group.copy(), spec)
        egshs_rows.append(
            selected_output_row(
                str(batch_id),
                str(selected["selected_candidate"]),
                full_safe,
                str(selected["status"]),
                str(selected["selection_reason"]),
                bool(selected["low_quality"]),
            )
            | {"near_tie_members": ";".join(str(item) for item in selected["near_tie_members"])}
        )
        selector_runtime.append(time.perf_counter() - select_started)
        comparator_started = time.perf_counter()
        fixed_rows["M0"].append(fixed_selection(str(batch_id), "M0_static_position", full_safe))
        fixed_rows["M2"].append(fixed_selection(str(batch_id), "M2_ctd_full", full_safe))
        fixed_rows["M12"].append(fixed_selection(str(batch_id), "M12_ctd_full_plus_gir_refine", full_safe))
        for method in ["AIC", "FSC", "BIC"]:
            ic_rows[method].append(information_criterion_selection(str(batch_id), method, float(sigma_map[str(batch_id)]), group, full_safe))
        comparator_runtime.append(time.perf_counter() - comparator_started)

    outputs = {
        "EGSHS": (method_dir / "D2_EGSHS_SELECTIONS.csv", pd.DataFrame(egshs_rows)),
        "M0": (method_dir / "D2_ALWAYS_M0.csv", pd.DataFrame(fixed_rows["M0"])),
        "M2": (method_dir / "D2_ALWAYS_M2.csv", pd.DataFrame(fixed_rows["M2"])),
        "M12": (method_dir / "D2_ALWAYS_M12.csv", pd.DataFrame(fixed_rows["M12"])),
        "AIC": (baseline_dir / "D2_AIC_SELECTIONS.csv", pd.DataFrame(ic_rows["AIC"])),
        "FSC": (baseline_dir / "D2_FSC_KNOWN_SIGMA0_SELECTIONS.csv", pd.DataFrame(ic_rows["FSC"])),
        "BIC": (baseline_dir / "D2_BIC_SELECTIONS.csv", pd.DataFrame(ic_rows["BIC"])),
    }
    expected = len(observation_manifest)
    for name, (path, frame) in outputs.items():
        if len(frame) != expected or frame["batch_id"].nunique() != expected:
            raise RuntimeError("d2_execution_blocked_selector_identity" if name == "EGSHS" else "d2_execution_blocked_comparator_identity")
        frame.sort_values("batch_id").to_csv(path, index=False)

    runtime_frame = pd.DataFrame(runtime_rows).sort_values("batch_id")
    runtime_frame["EGSHS_seconds"] = selector_runtime
    runtime_frame["comparators_seconds"] = comparator_runtime
    runtime_frame.to_csv(runtime_dir / "D2_RUNTIME_BY_BATCH.csv", index=False)
    batch_manifest = pd.read_csv(EXEC_ROOT / "D2_BATCH_MANIFEST.csv")
    batch_manifest["candidate_status"] = "COMPLETE_15_RECORDS"
    egshs_status = outputs["EGSHS"][1].set_index("batch_id")["status"].astype(str).to_dict()
    batch_manifest["selector_status"] = batch_manifest["batch_id"].map(egshs_status)
    batch_manifest.to_csv(EXEC_ROOT / "D2_BATCH_MANIFEST.csv", index=False)
    write_json(
        EXEC_ROOT / "D2_METHOD_STAGE_SUMMARY.json",
        {
            "completed_utc": utc_now_z(),
            "eligible_batch_count": expected,
            "candidate_record_count": len(full_safe),
            "candidate_truth_only_field_count": 0,
            "EGSHS_selection_count": len(outputs["EGSHS"][1]),
            "ALWAYS_M0_count": len(outputs["M0"][1]),
            "ALWAYS_M2_count": len(outputs["M2"][1]),
            "ALWAYS_M12_count": len(outputs["M12"][1]),
            "AIC_count": len(outputs["AIC"][1]),
            "FSC_count": len(outputs["FSC"][1]),
            "BIC_count": len(outputs["BIC"][1]),
            "B1_B2_B3_status": "skipped_not_predeclared_for_direct_D2_execution",
            "C2_calls": 0,
            "CA120_calls": 0,
            "CTRV120_calls": 0,
            "C3_calls": 0,
            "C4_calls": 0,
            "F1_F7_calls": 0,
            "bridge_score_calls": 0,
            "truth_reads": 0,
            "runtime_seconds": time.perf_counter() - started,
            "output_hashes": {name: sha256_file(path) for name, (path, _frame) in outputs.items()},
            "candidate_output_sha256": sha256_file(full_path),
            "candidate_selector_input_sha256": sha256_file(no_truth_path),
        },
    )
    print(json.dumps({"stage": "methods", "batches": expected, "candidate_records": len(full_safe), "truth_reads": 0, "runtime_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
