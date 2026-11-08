"""PAPER-EXP03A frozen-candidate selection-strategy comparison.

This script never executes a Doppler candidate solver.  ``prepare`` builds a
truth-free strategy input and freezes the analysis protocol.  ``run`` first
selects model identifiers from that truth-free table, then joins evaluation
truth for retrospective error and oracle analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

try:
    from scipy import stats
except Exception:  # pragma: no cover - recorded as unavailable in outputs
    stats = None


ROOT = Path(r"C:\Users\32377\Desktop\leo-c")
HIST_LAB = ROOT / "_new_solver_lab"
LAB = Path(r"C:\Users\32377\Desktop\leo-c_new_solver_lab")
OUT = LAB / "paper_draft" / "selection_strategy_baselines" / "PAPER_EXP03A"
DIRS = {name: OUT / name for name in ["protocol", "inputs", "outputs", "reports", "scripts", "figures", "logs", "source_snapshot"]}

EXP01 = HIST_LAB / "paper_draft" / "holdout_experiments"
TECH18_OUTPUTS = HIST_LAB / "outputs"
RELEASE = HIST_LAB / "final_release" / "MA_BGTR_v7_2_freeze_r2"
TECH18_FINAL = HIST_LAB / "final_release"
URBAN = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02a"
MANUSCRIPT = LAB / "paper_draft" / "Acta_Astronautica_MA_BGTR" / "manuscript_v0_8_urbannav_limit"
EXP02B = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02b_window_diagnostic"
EXP02C = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02c_diagnostic"
EXP02D = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02d_model_consistency"

PROTOCOL_PATH = DIRS["protocol"] / "PAPER_EXP03A_PROTOCOL.json"
PROTOCOL_HASH_PATH = DIRS["protocol"] / "PAPER_EXP03A_PROTOCOL_SHA256.txt"
AUDIT_PATH = DIRS["protocol"] / "PAPER_EXP03A_INPUT_AND_FREEZE_AUDIT.csv"
PROTECTION_BASELINE_PATH = DIRS["protocol"] / "PAPER_EXP03A_PROTECTION_BASELINE.json"
TRUTH_INPUT_PATH = DIRS["inputs"] / "PAPER_EXP03A_POST_SELECTION_TRUTH_LOOKUP.csv"
REFERENCE_SELECTION_PATH = DIRS["inputs"] / "PAPER_EXP03A_REFERENCE_SELECTIONS.csv"
STANDARDIZED_PATH = DIRS["outputs"] / "PAPER_EXP03A_STANDARDIZED_CANDIDATES.csv"
NO_TRUTH_PATH = DIRS["outputs"] / "PAPER_EXP03A_STRATEGY_INPUT_NO_TRUTH.csv"
PRE_TRUTH_SELECTION_PATH = DIRS["inputs"] / "PAPER_EXP03A_PRE_TRUTH_SELECTION_KEYS.csv"
LOG_PATH = DIRS["logs"] / "PAPER_EXP03A_RUN.log"

CANDIDATES = [
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
    "M12_ctd_full_plus_gir_refine",
    "M13_robust_ctd_full_plus_gir_refine",
    "M14_static_plus_gir_refine",
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

MODEL_FAMILY = {
    "M0_static_position": "static",
    "M1_static_position_bias": "static",
    "M2_ctd_full": "core_ctd",
    "M3_ctd_no_drift": "core_ctd",
    "M4_ctd_no_bias": "core_ctd",
    "M5_be_b0_only": "be_gtr",
    "M6_be_full": "be_gtr",
    "M7_robust_ctd_full": "robust",
    "M8_robust_be_b0_only": "robust",
    "M9_robust_be_full": "robust",
    "M10_gir_tr_fixed_scale": "direct_gir",
    "M11_gir_tr_mad_scale": "direct_gir",
    "M12_ctd_full_plus_gir_refine": "cascaded_refinement",
    "M13_robust_ctd_full_plus_gir_refine": "cascaded_refinement",
    "M14_static_plus_gir_refine": "cascaded_refinement",
}

STRATEGIES = {
    "STR-01": "MA-BGTR frozen",
    "STR-02": "Fixed M2",
    "STR-03": "Fixed M13",
    "STR-04": "Minimum raw validation RMSE",
    "STR-05": "Minimum trimmed validation RMSE",
    "STR-06": "AICc nested Gaussian",
    "STR-07": "BIC nested Gaussian",
    "STR-08": "Simple rank score",
    "STR-09": "Generic residual Pareto",
    "STR-10": "Gaussian pseudo-AICc full pool",
    "STR-11": "Gaussian pseudo-BIC full pool",
}

TRUTH_FORBIDDEN_COLUMNS = {
    "final_position_error_m",
    "mean_position_error_m",
    "mean_trajectory_position_error_m",
    "median_trajectory_position_error_m",
    "p95_trajectory_position_error_m",
    "max_trajectory_position_error_m",
    "velocity_error_mps",
    "mean_velocity_error_mps",
    "oracle_model",
    "oracle_identity",
    "expected_model",
    "holdout_family",
    "family_or_scenario",
    "route",
    "motion_type",
    "perturbation_profile",
    "true_speed_mps",
    "true_b0_mps",
    "true_bdot_mps2",
    "qatar_outlier_ratio",
    "dropout_ratio",
}

SOURCE_EVIDENCE_COLUMNS = [
    "dataset_type",
    "scenario",
    "position_init_label",
    "velocity_init_label",
    "beta_prior_profile",
    "candidate_model",
    "model_dof",
    "numerical_success",
    "converged",
    "quality_pass",
    "physical_plausible",
    "residual_rmse_mps",
    "full_residual_rmse_mps",
    "raw_validation_rmse_mps",
    "trimmed_validation_rmse_mps",
    "inlier_validation_rmse_mps",
    "robust_validation_cost",
    "condition_number",
    "position_cov_sqrt_trace_m",
    "position_bootstrap_spread_m",
    "estimated_speed_mps",
    "beta0_estimated_mps",
    "beta_dot_estimated_mps2",
    "information_retention_ratio",
    "geometry_preserved_info_retention",
    "retention_trace_ratio",
    "rank_loss",
    "severe_risk_veto",
    "severe_risk_reason",
    "outlier_evidence",
    "burst_outlier_evidence",
    "tail_ratio",
    "mad_ratio",
    "refine_gate_pass",
    "refine_gate_reason",
    "base_model_for_refine",
    "robust_gate_pass",
    "robust_gate_reason",
    "consistency_gate_pass",
    "consistency_penalty",
    "candidate_pool_protocol",
    "execution_mode",
]


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, default=json_default, allow_nan=False) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(safe_bool).astype(bool)


def row_count(path: Path) -> int:
    if path.suffix.lower() != ".csv":
        return 1
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "size_bytes": 0, "metadata_digest": ""}
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix())
    digest = hashlib.sha256()
    size = 0
    latest = 0
    for item in files:
        stat = item.stat()
        rel = item.relative_to(path).as_posix()
        size += stat.st_size
        latest = max(latest, stat.st_mtime_ns)
        digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(files),
        "size_bytes": size,
        "latest_mtime_ns": latest,
        "metadata_digest": digest.hexdigest(),
    }


def require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing required {label}: {path}")
    return path


def source_paths() -> dict[str, Path]:
    return {
        "exp01_candidates": require(EXP01 / "PAPER_EXP01_ACTUAL_CANDIDATES.csv", "EXP01 candidates"),
        "exp01_selected": require(EXP01 / "PAPER_EXP01_ACTUAL_SELECTED.csv", "EXP01 selected"),
        "exp01_family_aggregate": require(EXP01 / "PAPER_EXP01_FAMILY_AGGREGATE.csv", "EXP01 aggregate"),
        "exp01_protocol": require(EXP01 / "PAPER_EXP01_HOLDOUT_PROTOCOL.json", "EXP01 protocol"),
        "exp01_truth_audit": require(EXP01 / "PAPER_EXP01_TRUTH_ISOLATION_AUDIT.csv", "EXP01 truth audit"),
        "exp01_scenarios": require(EXP01 / "PAPER_EXP01_HOLDOUT_SCENARIOS.csv", "EXP01 scenarios"),
        "exp01_execution": require(EXP01 / "PAPER_EXP01_EXECUTION_AUDIT.csv", "EXP01 execution audit"),
        "tech18_candidates": require(TECH18_OUTPUTS / "tech18_actual_candidate_results.csv", "TECH18 candidates"),
        "tech18_selected": require(TECH18_OUTPUTS / "tech18_actual_selected_results.csv", "TECH18 selected"),
        "tech18_reference": require(TECH18_FINAL / "TECH18_FULL_POOL_FREEZE_REFERENCE.csv", "TECH18 reference"),
        "tech18_decision": require(TECH18_FINAL / "TECH18_PROJECT_COMPLETION_DECISION.json", "TECH18 decision"),
        "urban_candidates": require(URBAN / "outputs" / "PAPER_EXP02A_CANDIDATE_RESULTS.csv", "UrbanNav candidates"),
        "urban_selected": require(URBAN / "outputs" / "PAPER_EXP02A_SELECTED_RESULTS.csv", "UrbanNav selected"),
        "urban_observation_inventory": require(URBAN / "observations" / "PAPER_EXP02A_OBSERVATION_INVENTORY.csv", "UrbanNav observation inventory"),
        "selector": require(RELEASE / "src" / "leo_positioning" / "ma_bgtr_v71_solver.py", "frozen selector"),
        "freeze_policy": require(RELEASE / "src" / "leo_positioning" / "selector_freeze_policy.py", "freeze policy"),
        "v72_shell": require(RELEASE / "src" / "leo_positioning" / "ma_bgtr_v72_solver.py", "v7.2 shell"),
        "hierarchical_gates": require(RELEASE / "src" / "leo_positioning" / "hierarchical_gates.py", "hierarchical gates"),
        "family_evidence": require(RELEASE / "src" / "leo_positioning" / "family_evidence.py", "family evidence"),
        "bias_gate": require(RELEASE / "src" / "leo_positioning" / "bias_gate_fix.py", "bias gate"),
        "robust_gates": require(RELEASE / "src" / "leo_positioning" / "robust_gates.py", "robust gates"),
        "risk_veto": require(RELEASE / "src" / "leo_positioning" / "risk_veto.py", "risk veto"),
    }


def protected_tree_paths() -> dict[str, Path]:
    return {
        "frozen_release": RELEASE,
        "paper_exp01": EXP01,
        "urban_exp02a": URBAN,
        "urban_exp02b": EXP02B,
        "urban_exp02c": EXP02C,
        "urban_exp02d": EXP02D,
        "manuscript_v0_8": MANUSCRIPT,
    }


def load_selector_modules() -> tuple[Any, Any, Any, Any]:
    src = RELEASE / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from leo_positioning.family_evidence import STATIC_MODELS, best_candidate
    from leo_positioning.hierarchical_gates import HierarchicalGateConfig, _base_filter
    from leo_positioning.ma_bgtr_v71_solver import select_group_v71
    from leo_positioning.ma_bgtr_v72_solver import select_v72_results_from_v71

    return (select_group_v71, select_v72_results_from_v71, (STATIC_MODELS, best_candidate), (HierarchicalGateConfig, _base_filter))


def validate_full_pool(frame: pd.DataFrame, group_column: str, expected_runs: int, label: str) -> None:
    groups = frame.groupby(group_column, sort=False)
    if groups.ngroups != expected_runs:
        raise RuntimeError(f"{label}: expected {expected_runs} runs, found {groups.ngroups}")
    expected = set(CANDIDATES)
    errors: list[str] = []
    for key, group in groups:
        models = set(group["candidate_model"].astype(str))
        if len(group) != len(CANDIDATES) or models != expected:
            errors.append(str(key))
    if errors:
        raise RuntimeError(f"{label}: incomplete candidate pool in {len(errors)} runs; first={errors[:3]}")


def observation_count_from_csv(path_text: Any) -> int:
    path = Path(str(path_text))
    if not path.is_file():
        return 0
    return row_count(path)


def source_value(row: pd.Series, column: str, default: Any = np.nan) -> Any:
    return row[column] if column in row.index else default


def state_finite_from_row(row: pd.Series) -> bool:
    if not safe_bool(source_value(row, "numerical_success", False)):
        return False
    model = str(source_value(row, "candidate_model", ""))
    if not np.isfinite(safe_float(source_value(row, "estimated_speed_mps", np.nan))):
        return False
    if model != "M0_static_position" and not np.isfinite(safe_float(source_value(row, "beta0_estimated_mps", np.nan))):
        return False
    if model not in {"M0_static_position", "M1_static_position_bias"} and not np.isfinite(
        safe_float(source_value(row, "beta_dot_estimated_mps2", np.nan))
    ):
        return False
    ecef_columns = ["final_ecef_x_m", "final_ecef_y_m", "final_ecef_z_m"]
    if all(column in row.index for column in ecef_columns):
        if not all(np.isfinite(safe_float(row[column])) for column in ecef_columns):
            return False
    return True


def normalized_candidate(
    row: pd.Series,
    *,
    evaluation_dataset: str,
    group_id: str,
    family_or_scenario: str,
    realization_or_run_id: str,
    evidence_mode: str,
    selector_dataset_type: str,
    selector_scenario: str,
    observation_count: int,
    source_file: Path,
    source_row_id: int,
    final_error: float,
    mean_trajectory_error: float,
) -> dict[str, Any]:
    model = str(source_value(row, "candidate_model", ""))
    full_residual = safe_float(source_value(row, "full_residual_rmse_mps", source_value(row, "residual_rmse_mps", np.nan)))
    out: dict[str, Any] = {
        "evaluation_dataset": evaluation_dataset,
        "group_id": group_id,
        "family_or_scenario": family_or_scenario,
        "realization_or_run_id": realization_or_run_id,
        "evidence_mode": evidence_mode,
        "candidate_model": model,
        "candidate_family": MODEL_FAMILY.get(model, "unknown"),
        "dataset_type": selector_dataset_type,
        "scenario": selector_scenario,
        "position_init_label": str(source_value(row, "position_init_label", "east_10km")),
        "velocity_init_label": str(source_value(row, "velocity_init_label", "zero_velocity")),
        "beta_prior_profile": str(source_value(row, "beta_prior_profile", "B0_none")),
        "numerical_success": safe_bool(source_value(row, "numerical_success", False)),
        "converged": safe_bool(source_value(row, "converged", False)),
        "quality_pass": safe_bool(source_value(row, "quality_pass", False)),
        "physical_plausible": safe_bool(source_value(row, "physical_plausible", False)),
        "full_residual_rmse_mps": full_residual,
        "raw_validation_rmse_mps": safe_float(source_value(row, "raw_validation_rmse_mps", np.nan)),
        "trimmed_validation_rmse_mps": safe_float(source_value(row, "trimmed_validation_rmse_mps", np.nan)),
        "inlier_validation_rmse_mps": safe_float(source_value(row, "inlier_validation_rmse_mps", np.nan)),
        "robust_validation_cost": safe_float(source_value(row, "robust_validation_cost", np.nan)),
        "condition_number": safe_float(source_value(row, "condition_number", np.nan)),
        "position_covariance_proxy_m": safe_float(source_value(row, "position_cov_sqrt_trace_m", np.nan)),
        "position_cov_sqrt_trace_m": safe_float(source_value(row, "position_cov_sqrt_trace_m", np.nan)),
        "bootstrap_spread_m": safe_float(source_value(row, "position_bootstrap_spread_m", np.nan)),
        "position_bootstrap_spread_m": safe_float(source_value(row, "position_bootstrap_spread_m", np.nan)),
        "estimated_speed_mps": safe_float(source_value(row, "estimated_speed_mps", np.nan)),
        "beta0_estimated_mps": safe_float(source_value(row, "beta0_estimated_mps", np.nan)),
        "beta_dot_estimated_mps2": safe_float(source_value(row, "beta_dot_estimated_mps2", np.nan)),
        "information_retention_ratio": safe_float(source_value(row, "information_retention_ratio", np.nan)),
        "geometry_retention_ratio": safe_float(source_value(row, "geometry_preserved_info_retention", np.nan)),
        "geometry_preserved_info_retention": safe_float(source_value(row, "geometry_preserved_info_retention", np.nan)),
        "observation_count": int(observation_count),
        "model_dof": int(safe_float(source_value(row, "model_dof", MODEL_DOF.get(model, np.nan)), MODEL_DOF.get(model, -1))),
        "state_finite": state_finite_from_row(row),
        "residual_metrics_finite": bool(
            np.isfinite(full_residual)
            and np.isfinite(safe_float(source_value(row, "raw_validation_rmse_mps", np.nan)))
            and np.isfinite(safe_float(source_value(row, "trimmed_validation_rmse_mps", np.nan)))
            and np.isfinite(safe_float(source_value(row, "inlier_validation_rmse_mps", np.nan)))
        ),
        "retention_trace_ratio": safe_float(source_value(row, "retention_trace_ratio", np.nan)),
        "rank_loss": safe_float(source_value(row, "rank_loss", np.nan)),
        "severe_risk_veto": safe_bool(source_value(row, "severe_risk_veto", False)),
        "severe_risk_reason": str(source_value(row, "severe_risk_reason", "")),
        "outlier_evidence": safe_bool(source_value(row, "outlier_evidence", False)),
        "burst_outlier_evidence": safe_bool(source_value(row, "burst_outlier_evidence", False)),
        "tail_ratio": safe_float(source_value(row, "tail_ratio", np.nan)),
        "mad_ratio": safe_float(source_value(row, "mad_ratio", np.nan)),
        "refine_gate_pass": safe_bool(source_value(row, "refine_gate_pass", True)),
        "refine_gate_reason": str(source_value(row, "refine_gate_reason", "")),
        "base_model_for_refine": str(source_value(row, "base_model_for_refine", "")),
        "robust_gate_pass": safe_bool(source_value(row, "robust_gate_pass", True)),
        "robust_gate_reason": str(source_value(row, "robust_gate_reason", "")),
        "consistency_gate_pass": safe_bool(source_value(row, "consistency_gate_pass", True)),
        "consistency_penalty": safe_float(source_value(row, "consistency_penalty", np.nan)),
        "candidate_pool_protocol": str(source_value(row, "candidate_pool_protocol", "FULL_POOL_M0_M14_V1")),
        "execution_mode": str(source_value(row, "execution_mode", "solver_rerun")),
        "final_position_error_m": final_error,
        "mean_trajectory_position_error_m": mean_trajectory_error,
        "post_selection_only_truth_field": True,
        "source_file": str(source_file),
        "source_row_id": int(source_row_id),
    }
    return out


def standardize_inputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_pool = set(CANDIDATES)
    standardized: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    exp_candidates = pd.read_csv(paths["exp01_candidates"])
    exp_candidates = exp_candidates.loc[
        (exp_candidates["evaluation_scope"].astype(str) == "primary")
        & (exp_candidates["position_init_label"].astype(str) == "east_10km")
        & (exp_candidates["velocity_init_label"].astype(str) == "zero_velocity")
        & (exp_candidates["beta_prior_profile"].astype(str) == "B0_none")
    ].copy()
    validate_full_pool(exp_candidates, "holdout_id", 60, "PAPER-EXP01 primary")
    family_counts = exp_candidates[["holdout_id", "holdout_family"]].drop_duplicates()["holdout_family"].value_counts().to_dict()
    if set(family_counts.values()) != {10} or len(family_counts) != 6:
        raise RuntimeError(f"PAPER-EXP01 family counts are not six-by-ten: {family_counts}")
    if set(exp_candidates["candidate_pool_protocol"].astype(str)) != {"FULL_POOL_M0_M14_V1"}:
        raise RuntimeError("PAPER-EXP01 candidate protocol mismatch")
    exp_selected = pd.read_csv(paths["exp01_selected"])
    exp_selected = exp_selected.loc[
        (exp_selected["evaluation_scope"].astype(str) == "primary")
        & (exp_selected["selector_mode"].astype(str) == "observable_only")
    ].copy()
    if len(exp_selected) != 60 or exp_selected["holdout_id"].nunique() != 60:
        raise RuntimeError("PAPER-EXP01 observable-only selected reference is not 60 runs")
    exp_exec = pd.read_csv(paths["exp01_execution"])
    exp_exec = exp_exec.loc[exp_exec["evaluation_scope"].astype(str) == "primary"].drop_duplicates("batch_id")
    count_map = exp_exec.set_index("batch_id")["actual_observation_count"].astype(int).to_dict()
    ref_map = exp_selected.set_index("holdout_id")["selected_model"].astype(str).to_dict()
    group_map = {key: f"A{idx:04d}" for idx, key in enumerate(sorted(exp_candidates["holdout_id"].astype(str).unique()))}
    for source_index, row in exp_candidates.reset_index(drop=False).iterrows():
        holdout_id = str(row["holdout_id"])
        group_id = group_map[holdout_id]
        batch_id = str(row["batch_id"])
        standardized.append(
            normalized_candidate(
                row,
                evaluation_dataset="PAPER_EXP01_HOLDOUT",
                group_id=group_id,
                family_or_scenario=str(row["holdout_family"]),
                realization_or_run_id=holdout_id,
                evidence_mode="observable_only",
                selector_dataset_type=str(row["dataset_type"]),
                selector_scenario=group_id,
                observation_count=int(count_map[batch_id]),
                source_file=paths["exp01_candidates"],
                source_row_id=int(row["index"]),
                final_error=safe_float(row["final_position_error_m"]),
                mean_trajectory_error=np.nan,
            )
        )
    for holdout_id, group_id in group_map.items():
        subset = exp_candidates.loc[exp_candidates["holdout_id"].astype(str) == holdout_id]
        first = subset.iloc[0]
        truth_rows.extend(
            {
                "evaluation_dataset": "PAPER_EXP01_HOLDOUT",
                "group_id": group_id,
                "family_or_scenario": str(first["holdout_family"]),
                "realization_or_run_id": holdout_id,
                "route": "",
                "motion_type": "",
                "perturbation_profile": "",
                "candidate_model": str(row["candidate_model"]),
                "final_position_error_m": safe_float(row["final_position_error_m"]),
                "mean_trajectory_position_error_m": np.nan,
            }
            for _, row in subset.iterrows()
        )
        reference_rows.append(
            {
                "evaluation_dataset": "PAPER_EXP01_HOLDOUT",
                "group_id": group_id,
                "reference_selected_model": ref_map[holdout_id],
                "reference_source": str(paths["exp01_selected"]),
                "reference_mode": "observable_only",
            }
        )
    details["exp01"] = {"candidate_rows": len(exp_candidates), "runs": 60, "family_counts": family_counts}

    tech_candidates = pd.read_csv(paths["tech18_candidates"])
    validate_full_pool(tech_candidates, "scenario", 9, "TECH18 core")
    if set(tech_candidates["candidate_pool_protocol"].astype(str)) != {"FULL_POOL_M0_M14_V1"}:
        raise RuntimeError("TECH18 candidate protocol mismatch")
    tech_selected = pd.read_csv(paths["tech18_selected"])
    if len(tech_selected) != 9 or not bool_series(tech_selected["model_exact_match"]).all():
        raise RuntimeError("TECH18 selected acceptance reference is incomplete")
    tech_ref_map = tech_selected.set_index("scenario")["selected_model_actual"].astype(str).to_dict()
    tech_scenarios = list(tech_selected["scenario"].astype(str))
    group_map = {key: f"B{idx:04d}" for idx, key in enumerate(tech_scenarios)}
    observation_counts = {
        scenario: observation_count_from_csv(group.iloc[0].get("source_observation_file", ""))
        for scenario, group in tech_candidates.groupby("scenario", sort=False)
    }
    if any(value <= 0 for value in observation_counts.values()):
        raise RuntimeError(f"TECH18 observation counts unavailable: {observation_counts}")
    for source_index, row in tech_candidates.reset_index(drop=False).iterrows():
        scenario = str(row["scenario"])
        group_id = group_map[scenario]
        standardized.append(
            normalized_candidate(
                row,
                evaluation_dataset="TECH18_CORE",
                group_id=group_id,
                family_or_scenario=scenario,
                realization_or_run_id=scenario,
                evidence_mode="observable_only",
                selector_dataset_type=str(row["dataset_type"]),
                selector_scenario=group_id,
                observation_count=observation_counts[scenario],
                source_file=paths["tech18_candidates"],
                source_row_id=int(row["index"]),
                final_error=safe_float(row["final_position_error_m"]),
                mean_trajectory_error=np.nan,
            )
        )
    for scenario, group_id in group_map.items():
        subset = tech_candidates.loc[tech_candidates["scenario"].astype(str) == scenario]
        truth_rows.extend(
            {
                "evaluation_dataset": "TECH18_CORE",
                "group_id": group_id,
                "family_or_scenario": scenario,
                "realization_or_run_id": scenario,
                "route": "",
                "motion_type": "",
                "perturbation_profile": "",
                "candidate_model": str(row["candidate_model"]),
                "final_position_error_m": safe_float(row["final_position_error_m"]),
                "mean_trajectory_position_error_m": np.nan,
            }
            for _, row in subset.iterrows()
        )
        reference_rows.append(
            {
                "evaluation_dataset": "TECH18_CORE",
                "group_id": group_id,
                "reference_selected_model": tech_ref_map[scenario],
                "reference_source": str(paths["tech18_selected"]),
                "reference_mode": "observable_only",
            }
        )
    details["tech18"] = {"candidate_rows": len(tech_candidates), "runs": 9, "observation_counts": observation_counts}

    urban_candidates = pd.read_csv(paths["urban_candidates"])
    validate_full_pool(urban_candidates, "run_id", 72, "UrbanNav EXP02A")
    urban_selected = pd.read_csv(paths["urban_selected"])
    urban_selected = urban_selected.loc[urban_selected["evidence_mode"].astype(str) == "observable-only"].copy()
    if len(urban_selected) != 72 or urban_selected["run_id"].nunique() != 72:
        raise RuntimeError("UrbanNav observable-only selected reference is not 72 runs")
    urban_inventory = pd.read_csv(paths["urban_observation_inventory"])
    count_map = urban_inventory.set_index("run_id")["used_by_solver_count"].astype(int).to_dict()
    ref_map = urban_selected.set_index("run_id")["selected_model"].astype(str).to_dict()
    group_map = {key: f"C{idx:04d}" for idx, key in enumerate(sorted(urban_candidates["run_id"].astype(str).unique()))}
    for source_index, row in urban_candidates.reset_index(drop=False).iterrows():
        run_id = str(row["run_id"])
        group_id = group_map[run_id]
        if run_id not in count_map:
            raise RuntimeError(f"UrbanNav observation count missing for {run_id}")
        row = row.copy()
        row["dataset_type"] = "urban_real_trajectory_controlled"
        row["scenario"] = group_id
        row["position_init_label"] = "east_10km"
        row["velocity_init_label"] = "zero_velocity"
        row["beta_prior_profile"] = "B0_none"
        row["model_dof"] = MODEL_DOF[str(row["candidate_model"])]
        row["candidate_pool_protocol"] = "FULL_POOL_M0_M14_V1"
        standardized.append(
            normalized_candidate(
                row,
                evaluation_dataset="URBANNAV_EXP02A",
                group_id=group_id,
                family_or_scenario=str(row["motion_type"]),
                realization_or_run_id=run_id,
                evidence_mode="observable_only",
                selector_dataset_type="urban_real_trajectory_controlled",
                selector_scenario=group_id,
                observation_count=int(count_map[run_id]),
                source_file=paths["urban_candidates"],
                source_row_id=int(row["index"]),
                final_error=safe_float(row["final_position_error_m"]),
                mean_trajectory_error=safe_float(row["mean_trajectory_position_error_m"]),
            )
        )
    for run_id, group_id in group_map.items():
        subset = urban_candidates.loc[urban_candidates["run_id"].astype(str) == run_id]
        first = subset.iloc[0]
        truth_rows.extend(
            {
                "evaluation_dataset": "URBANNAV_EXP02A",
                "group_id": group_id,
                "family_or_scenario": str(first["motion_type"]),
                "realization_or_run_id": run_id,
                "route": str(first["route"]),
                "motion_type": str(first["motion_type"]),
                "perturbation_profile": str(first["perturbation_profile"]),
                "candidate_model": str(row["candidate_model"]),
                "final_position_error_m": safe_float(row["final_position_error_m"]),
                "mean_trajectory_position_error_m": safe_float(row["mean_trajectory_position_error_m"]),
            }
            for _, row in subset.iterrows()
        )
        reference_rows.append(
            {
                "evaluation_dataset": "URBANNAV_EXP02A",
                "group_id": group_id,
                "reference_selected_model": ref_map[run_id],
                "reference_source": str(paths["urban_selected"]),
                "reference_mode": "observable-only",
            }
        )
    details["urban"] = {"candidate_rows": len(urban_candidates), "runs": 72}

    standardized_frame = pd.DataFrame(standardized)
    if set(standardized_frame["candidate_model"].astype(str)) != expected_pool:
        raise RuntimeError("standardized candidate model set mismatch")
    truth_frame = pd.DataFrame(truth_rows)
    references = pd.DataFrame(reference_rows)
    return standardized_frame, truth_frame, references, details


def apply_common_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    _, _, family_helpers, gate_helpers = load_selector_modules()
    static_models, best_candidate = family_helpers
    config_type, base_filter = gate_helpers
    result = frame.copy()
    result["base_filter_pass"] = False
    result["base_filter_reason"] = ""
    for _, group in result.groupby(["evaluation_dataset", "group_id"], sort=False):
        best_static = best_candidate(group, static_models)
        eligible, reasons = base_filter(group, best_static, config_type())
        result.loc[group.index, "base_filter_reason"] = [reasons.get(int(index), "") for index in group.index]
        result.loc[eligible.index, "base_filter_pass"] = True
    result["fixed_numeric_valid"] = (
        bool_series(result["numerical_success"])
        & bool_series(result["state_finite"])
        & bool_series(result["residual_metrics_finite"])
    )
    result["selectable"] = result["fixed_numeric_valid"] & bool_series(result["base_filter_pass"])
    return result


def no_truth_frame(standardized: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "evaluation_dataset",
        "group_id",
        "evidence_mode",
        "candidate_model",
        "candidate_family",
        "dataset_type",
        "scenario",
        "position_init_label",
        "velocity_init_label",
        "beta_prior_profile",
        "numerical_success",
        "converged",
        "quality_pass",
        "physical_plausible",
        "state_finite",
        "residual_metrics_finite",
        "base_filter_pass",
        "base_filter_reason",
        "fixed_numeric_valid",
        "selectable",
        "full_residual_rmse_mps",
        "raw_validation_rmse_mps",
        "trimmed_validation_rmse_mps",
        "inlier_validation_rmse_mps",
        "robust_validation_cost",
        "condition_number",
        "position_covariance_proxy_m",
        "position_cov_sqrt_trace_m",
        "bootstrap_spread_m",
        "position_bootstrap_spread_m",
        "estimated_speed_mps",
        "beta0_estimated_mps",
        "beta_dot_estimated_mps2",
        "information_retention_ratio",
        "geometry_retention_ratio",
        "geometry_preserved_info_retention",
        "observation_count",
        "model_dof",
        "retention_trace_ratio",
        "rank_loss",
        "severe_risk_veto",
        "severe_risk_reason",
        "outlier_evidence",
        "burst_outlier_evidence",
        "tail_ratio",
        "mad_ratio",
        "refine_gate_pass",
        "refine_gate_reason",
        "base_model_for_refine",
        "robust_gate_pass",
        "robust_gate_reason",
        "consistency_gate_pass",
        "consistency_penalty",
        "candidate_pool_protocol",
        "execution_mode",
    ]
    out = standardized.loc[:, keep].copy()
    leaked = sorted(TRUTH_FORBIDDEN_COLUMNS.intersection(out.columns))
    if leaked:
        raise RuntimeError(f"truth columns leaked into strategy input: {leaked}")
    if out["group_id"].astype(str).str.contains(r"H[0-5]|S[0-9]|Q[0-9]|ODA|SHI|iridium", case=False, regex=True).any():
        raise RuntimeError("opaque group identifiers unexpectedly contain scenario/family labels")
    return out


def strategy_definitions() -> pd.DataFrame:
    rows = [
        ("STR-01", "MA-BGTR frozen", "M0-M14; frozen hierarchy", "Frozen source selector after truth scrubbing", "Frozen v7.1 tie/risk rules", "Frozen base filter", "Deployable reference; v7.2 does not alter selected model"),
        ("STR-02", "Fixed M2", "M2 only", "Always M2 when numerically valid", "None", "Numeric validity only; no fallback", "Fixed-model baseline"),
        ("STR-03", "Fixed M13", "M13 only", "Always M13 when numerically valid", "None", "Numeric validity only; no fallback", "Fixed robust-refine baseline"),
        ("STR-04", "Minimum raw validation RMSE", "Common eligible M0-M14", "Minimum raw validation RMSE", "trimmed; DOF; model ID", "Common deployable filter", "Residual-only baseline"),
        ("STR-05", "Minimum trimmed validation RMSE", "Common eligible M0-M14", "Minimum trimmed validation RMSE", "raw; inlier; DOF; model ID", "Common deployable filter", "Residual-only baseline"),
        ("STR-06", "AICc nested Gaussian", "M0-M4", "n*ln(RSS/n)+2k+2k(k+1)/(n-k-1)", "criterion; model ID", "Common deployable filter", "Strict only for ordinary Gaussian nested models"),
        ("STR-07", "BIC nested Gaussian", "M0-M4", "n*ln(RSS/n)+k*ln(n)", "criterion; model ID", "Common deployable filter", "Strict only for ordinary Gaussian nested models"),
        ("STR-08", "Simple rank score", "Common eligible M0-M14", "0.25 raw + 0.40 trimmed + 0.25 inlier + 0.05 condition + 0.05 DOF percentile ranks", "trimmed; raw; DOF; model ID", "Common deployable filter", "Weights pre-frozen; missing condition renormalized per row"),
        ("STR-09", "Generic residual Pareto", "Common eligible M0-M14", "Pareto(raw, trimmed, inlier, condition, DOF)", "median residual rank; condition; DOF; model ID", "Common deployable filter", "Generic, not historical v6"),
        ("STR-10", "Gaussian pseudo-AICc full pool", "Common eligible M0-M14 with known DOF", "Gaussian RSS AICc approximation", "criterion; model ID", "Common deployable filter", "Not strict likelihood AICc for Robust/BE/GIR"),
        ("STR-11", "Gaussian pseudo-BIC full pool", "Common eligible M0-M14 with known DOF", "Gaussian RSS BIC approximation", "criterion; model ID", "Common deployable filter", "Not strict likelihood BIC for Robust/BE/GIR"),
        ("STR-ORACLE", "Post-selection oracle", "All finite candidate errors", "Minimum evaluation error after all deployable selections", "model ID", "Post-selection only", "Non-deployable upper bound"),
    ]
    return pd.DataFrame(rows, columns=["strategy_id", "strategy_name", "candidate_scope", "selection_rule", "tie_break_rule", "filter_rule", "limitation"])


def build_protocol(paths: dict[str, Path], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": "PAPER-EXP03A",
        "protocol_version": "PAPER_EXP03A_FROZEN_CANDIDATE_SELECTION_V1",
        "created_local": datetime.now().astimezone().isoformat(),
        "created_before_strategy_selection": True,
        "solver_execution_prohibited": True,
        "solver_rerun_count": 0,
        "candidate_pool_protocol": "FULL_POOL_M0_M14_V1",
        "algorithm_version": "MA-BGTR-v7.1",
        "freeze_policy_version": "v7.2",
        "datasets": {
            "primary": {"name": "PAPER_EXP01_HOLDOUT", "runs": 60, "mode": "observable_only", "families": 6},
            "mechanism": {"name": "TECH18_CORE", "runs": 9},
            "stress": {"name": "URBANNAV_EXP02A", "runs": 72, "mode": "observable-only", "claim_role": "supplementary model-out-of-family stress"},
        },
        "source_files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "prepared_artifacts": {
            "standardized_candidates": {"path": str(STANDARDIZED_PATH), "sha256": sha256_file(STANDARDIZED_PATH)},
            "strategy_input_no_truth": {"path": str(NO_TRUTH_PATH), "sha256": sha256_file(NO_TRUTH_PATH)},
            "post_selection_truth_lookup": {"path": str(TRUTH_INPUT_PATH), "sha256": sha256_file(TRUTH_INPUT_PATH)},
            "reference_selections_for_recovery_audit": {"path": str(REFERENCE_SELECTION_PATH), "sha256": sha256_file(REFERENCE_SELECTION_PATH)},
        },
        "input_details": details,
        "truth_isolation": {
            "selection_input": str(NO_TRUTH_PATH),
            "post_selection_truth_lookup": str(TRUTH_INPUT_PATH),
            "forbidden_columns": sorted(TRUTH_FORBIDDEN_COLUMNS),
            "opaque_group_ids": True,
            "truth_join_timing": "Only after all deployable strategy model identifiers have been written to PRE_TRUTH_SELECTION_KEYS.",
        },
        "common_filter": {
            "applies_to": ["STR-01 frozen internal filtering", "STR-04", "STR-05", "STR-06", "STR-07", "STR-08", "STR-09", "STR-10", "STR-11"],
            "requirements": ["numerical_success", "finite reported state proxy", "finite residual metrics", "physical_plausible", "quality_pass", "no severe risk veto", "frozen base-filter model restrictions"],
            "quality_pass_is_hard": True,
            "frozen_source_basis": "hierarchical_gates._base_filter",
            "fixed_strategy_exception": "STR-02/03 require only numerical/state/residual validity and never fall back.",
            "urban_archival_caution": "EXP02A exported table omits covariance/bootstrap/information-retention/BE-retention fields; missing evidence remains missing and is never reconstructed from selected outcomes.",
        },
        "model_dof": MODEL_DOF,
        "strategies": strategy_definitions().to_dict(orient="records"),
        "aicc_formula": "RSS=n*full_residual_rmse_mps^2; AIC=n*ln(RSS/n)+2k; AICc=AIC+2k(k+1)/(n-k-1); invalid when n<=k+1",
        "bic_formula": "RSS=n*full_residual_rmse_mps^2; BIC=n*ln(RSS/n)+k*ln(n)",
        "simple_score_weights": {"raw": 0.25, "trimmed": 0.40, "inlier": 0.25, "condition": 0.05, "dof": 0.05},
        "simple_score_missing_condition": "Remove condition term for that row and renormalize remaining fixed weights; record missing_condition.",
        "pareto_rule": "Strict domination over raw, trimmed, inlier, condition, DOF; frontier tie by median residual percentile rank, condition rank, DOF, model ID.",
        "primary_metric": "final_position_error_m for PAPER_EXP01/TECH18; mean_trajectory_position_error_m for UrbanNav",
        "failure_penalty": "Per-run maximum finite candidate evaluation error; if none finite, global maximum finite error.",
        "catastrophic_thresholds_m": [500.0, 1000.0, 2000.0],
        "bootstrap": {"resamples": 5000, "seed": 20260726, "interval": "percentile 95%"},
        "paired_statistics": {"wilcoxon": "two-sided", "holm": "all ten deployable baselines", "difference": "baseline_error_minus_ma_bgtr_error; positive favors MA-BGTR", "mcnemar": "exact binomial on discordant catastrophe pairs"},
        "best_simple_reporting_rule": "Lowest primary p95, then >1 km rate, then median, then higher coverage, then strategy ID.",
        "decision_rules": {
            "strategy_baseline_supports_ma_bgtr": [
                "No deployable simple strategy weakly dominates MA-BGTR on median, p95, >1 km rate, and coverage with at least one strict improvement.",
                "MA-BGTR ranks in the top three for both p95 and >1 km rate.",
                "Against at least one fixed M2/M13 baseline, MA-BGTR improves p95 or >1 km rate while median is no worse than 110% of that fixed baseline.",
                "MA-BGTR has nonzero coverage and <100% catastrophe rate in H0 and at least two dynamic families.",
            ],
            "strategy_baseline_does_not_support_ma_bgtr": [
                "A simpler strategy has coverage>=, median<=, p95<=, and >1 km rate<= MA-BGTR, with at least two strict improvements.",
                "No critical-family protection advantage: MA-BGTR does not achieve lower p95 or lower >1 km rate in H0 plus at least two dynamic families.",
            ],
            "strategy_baseline_partial_support": "All other mixed outcomes with no decisive support/non-support condition.",
        },
        "prohibited_adaptations": ["solver rerun", "weight tuning", "threshold changes", "scenario-name model mapping", "truth or oracle in deployable selection", "dropping failed candidates/runs"],
        "prohibited_claims": ["UrbanNav is a positive dynamic LEO benchmark", "oracle is deployable", "AICc is strict for Robust/BE/GIR", "statistical significance proves deployment readiness"],
    }


def input_audit_rows(paths: dict[str, Path], details: dict[str, Any]) -> pd.DataFrame:
    source_groups = {
        "exp01_candidates": ("PAPER_EXP01", "FULL_POOL_M0_M14_V1", 60, 60, True),
        "exp01_selected": ("PAPER_EXP01", "FULL_POOL_M0_M14_V1", 60, 60, True),
        "exp01_family_aggregate": ("PAPER_EXP01", "", 60, 60, True),
        "exp01_protocol": ("PAPER_EXP01", "FULL_POOL_M0_M14_V1", 60, 60, True),
        "exp01_truth_audit": ("PAPER_EXP01", "", 60, 60, True),
        "exp01_scenarios": ("PAPER_EXP01", "FULL_POOL_M0_M14_V1", 60, 60, True),
        "exp01_execution": ("PAPER_EXP01", "", 60, 60, True),
        "tech18_candidates": ("TECH18", "FULL_POOL_M0_M14_V1", 9, 9, True),
        "tech18_selected": ("TECH18", "FULL_POOL_M0_M14_V1", 9, 9, True),
        "tech18_reference": ("TECH18", "FULL_POOL_M0_M14_V1", 9, 9, True),
        "tech18_decision": ("TECH18", "FULL_POOL_M0_M14_V1", 9, 9, True),
        "urban_candidates": ("URBANNAV_EXP02A", "FULL_POOL_M0_M14_V1", 72, 72, True),
        "urban_selected": ("URBANNAV_EXP02A", "FULL_POOL_M0_M14_V1", 72, 72, True),
        "urban_observation_inventory": ("URBANNAV_EXP02A", "", 72, 72, True),
        "selector": ("FROZEN_SOURCE", "", 0, 0, True),
        "freeze_policy": ("FROZEN_SOURCE", "", 0, 0, True),
        "v72_shell": ("FROZEN_SOURCE", "", 0, 0, True),
        "hierarchical_gates": ("FROZEN_SOURCE", "", 0, 0, True),
        "family_evidence": ("FROZEN_SOURCE", "", 0, 0, True),
        "bias_gate": ("FROZEN_SOURCE", "", 0, 0, True),
        "robust_gates": ("FROZEN_SOURCE", "", 0, 0, True),
        "risk_veto": ("FROZEN_SOURCE", "", 0, 0, True),
    }
    rows = []
    for key, path in paths.items():
        group, pool, expected_runs, actual_runs, complete = source_groups[key]
        rows.append(
            {
                "source_group": group,
                "source_path": str(path),
                "file_exists": path.is_file(),
                "row_count": row_count(path),
                "size_bytes": path.stat().st_size,
                "sha256_before": sha256_file(path),
                "sha256_after": "pending",
                "unchanged": "pending",
                "candidate_pool": pool,
                "expected_runs": expected_runs,
                "actual_runs": actual_runs,
                "complete_pool_verified": complete,
                "usable": True,
                "rejection_reason": "",
            }
        )
    return pd.DataFrame(rows)


def prepare() -> None:
    configure_logging()
    logging.info("Preparing PAPER-EXP03A truth-isolated inputs and frozen protocol")
    if PROTOCOL_PATH.exists() or PROTOCOL_HASH_PATH.exists():
        raise RuntimeError("protocol already exists; refusing to overwrite a frozen protocol")
    paths = source_paths()
    decision = json.loads(paths["tech18_decision"].read_text(encoding="utf-8"))
    if decision.get("technical_phase_decision") != "project_technical_phase_complete":
        raise RuntimeError("TECH18 technical decision is not complete")
    if decision.get("candidate_pool_protocol") != "FULL_POOL_M0_M14_V1":
        raise RuntimeError("TECH18 candidate protocol is not FULL_POOL_M0_M14_V1")

    baseline = {
        "created_local": datetime.now().astimezone().isoformat(),
        "critical_file_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "protected_trees": {name: tree_snapshot(path) for name, path in protected_tree_paths().items()},
        "solver_rerun_count": 0,
    }
    write_json(PROTECTION_BASELINE_PATH, baseline)

    snapshot_dir = DIRS["source_snapshot"] / "leo_positioning"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for key in ["selector", "freeze_policy", "v72_shell", "hierarchical_gates", "family_evidence", "bias_gate", "robust_gates", "risk_veto"]:
        shutil.copy2(paths[key], snapshot_dir / paths[key].name)

    standardized, truth, references, details = standardize_inputs(paths)
    standardized = apply_common_eligibility(standardized)
    strategy_input = no_truth_frame(standardized)

    if len(strategy_input) != 2115:
        raise RuntimeError(f"unexpected standardized candidate rows: {len(strategy_input)}")
    if strategy_input.groupby(["evaluation_dataset", "group_id"])["candidate_model"].nunique().ne(15).any():
        raise RuntimeError("at least one standardized group is not full M0-M14")

    standardized.to_csv(STANDARDIZED_PATH, index=False)
    strategy_input.to_csv(NO_TRUTH_PATH, index=False)
    truth.to_csv(TRUTH_INPUT_PATH, index=False)
    references.to_csv(REFERENCE_SELECTION_PATH, index=False)
    strategy_definitions().to_csv(DIRS["protocol"] / "PAPER_EXP03A_STRATEGY_DEFINITIONS.csv", index=False)

    isolation_rows = [
        {"audit_item": "strategy_input_has_no_forbidden_truth_columns", "status": not bool(TRUTH_FORBIDDEN_COLUMNS.intersection(strategy_input.columns)), "evidence": ";".join(sorted(strategy_input.columns)), "action_required": ""},
        {"audit_item": "opaque_group_ids_hide_family_and_scenario", "status": True, "evidence": f"{strategy_input['group_id'].nunique()} opaque groups", "action_required": ""},
        {"audit_item": "post_selection_truth_stored_separately", "status": TRUTH_INPUT_PATH.is_file(), "evidence": str(TRUTH_INPUT_PATH), "action_required": ""},
        {"audit_item": "primary_mode_is_observable_only", "status": True, "evidence": "60 PAPER-EXP01 primary groups", "action_required": ""},
        {"audit_item": "position_error_used_for_selection", "status": False, "evidence": "forbidden and absent from strategy input", "action_required": "must remain false"},
        {"audit_item": "oracle_used_for_selection", "status": False, "evidence": "oracle computed only after PRE_TRUTH_SELECTION_KEYS", "action_required": "must remain false"},
        {"audit_item": "scenario_name_hardcode", "status": False, "evidence": "selector receives opaque scenario identifiers", "action_required": "must remain false"},
    ]
    pd.DataFrame(isolation_rows).to_csv(DIRS["protocol"] / "PAPER_EXP03A_TRUTH_ISOLATION_AUDIT.csv", index=False)

    audit = input_audit_rows(paths, details)
    audit.to_csv(AUDIT_PATH, index=False)
    protocol = build_protocol(paths, details)
    write_json(PROTOCOL_PATH, protocol)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    PROTOCOL_HASH_PATH.write_text(f"{protocol_hash}  PAPER_EXP03A_PROTOCOL.json\n", encoding="ascii")
    logging.info("Protocol frozen: %s", protocol_hash)
    logging.info("Prepared %d candidates across %d groups", len(strategy_input), strategy_input["group_id"].nunique())


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_HASH_PATH.is_file():
        raise RuntimeError("frozen protocol is missing; run prepare first")
    expected = PROTOCOL_HASH_PATH.read_text(encoding="ascii").strip().split()[0]
    actual = sha256_file(PROTOCOL_PATH)
    if expected != actual:
        raise RuntimeError(f"protocol hash mismatch: expected {expected}, got {actual}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not protocol.get("created_before_strategy_selection") or not protocol.get("solver_execution_prohibited"):
        raise RuntimeError("protocol does not preserve required ordering/boundary")
    return protocol


def sort_first(frame: pd.DataFrame, columns: list[str], ascending: list[bool] | None = None) -> pd.Series | None:
    if frame.empty:
        return None
    ascending = ascending or [True] * len(columns)
    return frame.sort_values(columns, ascending=ascending, kind="mergesort").iloc[0]


def information_criterion(frame: pd.DataFrame, criterion: str) -> pd.DataFrame:
    out = frame.copy()
    n = numeric(out["observation_count"])
    k = numeric(out["model_dof"])
    rmse = numeric(out["full_residual_rmse_mps"])
    variance = np.maximum(np.square(rmse.to_numpy(dtype=float)), np.finfo(float).tiny)
    base = n.to_numpy(dtype=float) * np.log(variance)
    if criterion == "aicc":
        denom = n.to_numpy(dtype=float) - k.to_numpy(dtype=float) - 1.0
        correction = np.where(denom > 0.0, 2.0 * k.to_numpy(dtype=float) * (k.to_numpy(dtype=float) + 1.0) / denom, np.inf)
        score = base + 2.0 * k.to_numpy(dtype=float) + correction
    elif criterion == "bic":
        score = base + k.to_numpy(dtype=float) * np.log(np.maximum(n.to_numpy(dtype=float), 1.0))
    else:
        raise ValueError(criterion)
    out["criterion_score"] = score
    out.loc[~np.isfinite(out["criterion_score"]), "criterion_score"] = np.inf
    return out


def percentile_rank(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.rank(method="average", pct=True, ascending=True)


def select_strategy(group: pd.DataFrame, strategy_id: str, selector_helpers: tuple[Any, Any]) -> dict[str, Any]:
    select_group_v71, select_v72_results_from_v71 = selector_helpers
    eligible = group.loc[bool_series(group["selectable"])].copy()
    result = {
        "evaluation_dataset": str(group["evaluation_dataset"].iloc[0]),
        "group_id": str(group["group_id"].iloc[0]),
        "strategy_id": strategy_id,
        "strategy_name": STRATEGIES[strategy_id],
        "selected_model": "",
        "selection_generated_without_truth": True,
        "selection_key_valid": False,
        "selection_reason": "",
        "eligible_candidate_count": int(len(eligible)),
        "missing_condition_handling_used": False,
        "criterion_score": np.nan,
    }

    selected: pd.Series | None = None
    if strategy_id == "STR-01":
        selected, _family, _bias, _robust, reason, low_quality = select_group_v71(group.copy())
        model = str(selected["candidate_model"])
        frozen_input = pd.DataFrame(
            [
                {
                    "dataset_type": str(group["dataset_type"].iloc[0]),
                    "scenario": str(group["scenario"].iloc[0]),
                    "position_init_label": str(group["position_init_label"].iloc[0]),
                    "velocity_init_label": str(group["velocity_init_label"].iloc[0]),
                    "beta_prior_profile": str(group["beta_prior_profile"].iloc[0]),
                    "selected_model_v71": model,
                    "selection_reason_v71": reason,
                }
            ]
        )
        frozen = select_v72_results_from_v71(frozen_input)
        if str(frozen.iloc[0]["selected_model_v72"]) != model:
            raise RuntimeError("v7.2 freeze policy unexpectedly changed v7.1 model")
        result.update(
            {
                "selected_model": model,
                "selection_key_valid": True,
                "selection_reason": f"{reason};v7.2_model_preserved;low_quality={bool(low_quality)}",
                "eligible_candidate_count": int(bool_series(group["base_filter_pass"]).sum()),
            }
        )
        return result

    if strategy_id in {"STR-02", "STR-03"}:
        target = "M2_ctd_full" if strategy_id == "STR-02" else "M13_robust_ctd_full_plus_gir_refine"
        subset = group.loc[(group["candidate_model"].astype(str) == target) & bool_series(group["fixed_numeric_valid"])].copy()
        if subset.empty:
            result["selection_reason"] = f"{target}_numeric_invalid_no_fallback"
            result["eligible_candidate_count"] = 0
            return result
        selected = subset.iloc[0]
        result.update(
            {
                "selected_model": target,
                "selection_key_valid": True,
                "selection_reason": f"fixed_{target}_numeric_valid",
                "eligible_candidate_count": 1,
            }
        )
        return result

    if eligible.empty:
        result["selection_reason"] = "no_candidate_after_common_deployable_filter"
        return result

    if strategy_id == "STR-04":
        selected = sort_first(eligible, ["raw_validation_rmse_mps", "trimmed_validation_rmse_mps", "model_dof", "candidate_model"])
        reason = "minimum_raw_then_trimmed_dof_model"
    elif strategy_id == "STR-05":
        selected = sort_first(eligible, ["trimmed_validation_rmse_mps", "raw_validation_rmse_mps", "inlier_validation_rmse_mps", "model_dof", "candidate_model"])
        reason = "minimum_trimmed_then_raw_inlier_dof_model"
    elif strategy_id in {"STR-06", "STR-07"}:
        subset = eligible.loc[eligible["candidate_model"].isin(CANDIDATES[:5])].copy()
        criterion = "aicc" if strategy_id == "STR-06" else "bic"
        scored = information_criterion(subset, criterion)
        scored = scored.loc[np.isfinite(scored["criterion_score"])].copy()
        selected = sort_first(scored, ["criterion_score", "candidate_model"])
        reason = f"minimum_{criterion}_nested_gaussian_M0_M4"
    elif strategy_id == "STR-08":
        scored = eligible.copy()
        rank_columns = {
            "raw": percentile_rank(scored["raw_validation_rmse_mps"]),
            "trimmed": percentile_rank(scored["trimmed_validation_rmse_mps"]),
            "inlier": percentile_rank(scored["inlier_validation_rmse_mps"]),
            "condition": percentile_rank(scored["condition_number"]),
            "dof": percentile_rank(scored["model_dof"]),
        }
        for name, values in rank_columns.items():
            scored[f"rank_{name}"] = values
        weights = {"raw": 0.25, "trimmed": 0.40, "inlier": 0.25, "condition": 0.05, "dof": 0.05}
        scores = []
        missing_condition = []
        for _, row in scored.iterrows():
            available = {name: weight for name, weight in weights.items() if np.isfinite(safe_float(row[f"rank_{name}"]))}
            denom = sum(available.values())
            scores.append(sum(weight * float(row[f"rank_{name}"]) for name, weight in available.items()) / denom if denom else np.inf)
            missing_condition.append("condition" not in available)
        scored["simple_score"] = scores
        scored["missing_condition"] = missing_condition
        selected = sort_first(scored, ["simple_score", "trimmed_validation_rmse_mps", "raw_validation_rmse_mps", "model_dof", "candidate_model"])
        reason = "minimum_pre_registered_simple_percentile_rank_score"
        result["missing_condition_handling_used"] = bool(scored["missing_condition"].any())
        if selected is not None:
            result["criterion_score"] = safe_float(selected["simple_score"])
    elif strategy_id == "STR-09":
        scored = eligible.copy()
        objectives = ["raw_validation_rmse_mps", "trimmed_validation_rmse_mps", "inlier_validation_rmse_mps", "condition_number", "model_dof"]
        matrix = scored[objectives].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        matrix[~np.isfinite(matrix)] = np.inf
        dominated = np.zeros(len(scored), dtype=bool)
        for i in range(len(scored)):
            for j in range(len(scored)):
                if i == j:
                    continue
                if np.all(matrix[j] <= matrix[i]) and np.any(matrix[j] < matrix[i]):
                    dominated[i] = True
                    break
        frontier = scored.loc[~dominated].copy()
        frontier["raw_rank"] = percentile_rank(scored["raw_validation_rmse_mps"]).loc[frontier.index]
        frontier["trimmed_rank"] = percentile_rank(scored["trimmed_validation_rmse_mps"]).loc[frontier.index]
        frontier["inlier_rank"] = percentile_rank(scored["inlier_validation_rmse_mps"]).loc[frontier.index]
        frontier["median_residual_rank"] = frontier[["raw_rank", "trimmed_rank", "inlier_rank"]].median(axis=1)
        frontier["condition_rank"] = percentile_rank(scored["condition_number"]).loc[frontier.index].fillna(1.0)
        selected = sort_first(frontier, ["median_residual_rank", "condition_rank", "model_dof", "candidate_model"])
        reason = f"generic_pareto_frontier_size_{len(frontier)}"
        if selected is not None:
            result["criterion_score"] = safe_float(selected["median_residual_rank"])
    elif strategy_id in {"STR-10", "STR-11"}:
        subset = eligible.loc[numeric(eligible["model_dof"]).notna()].copy()
        criterion = "aicc" if strategy_id == "STR-10" else "bic"
        scored = information_criterion(subset, criterion)
        scored = scored.loc[np.isfinite(scored["criterion_score"])].copy()
        selected = sort_first(scored, ["criterion_score", "candidate_model"])
        reason = f"minimum_gaussian_pseudo_{criterion}_full_pool"
    else:
        raise KeyError(strategy_id)

    if selected is None:
        result["selection_reason"] = f"{strategy_id}_no_valid_scored_candidate"
        return result
    result.update(
        {
            "selected_model": str(selected["candidate_model"]),
            "selection_key_valid": True,
            "selection_reason": reason,
        }
    )
    if "criterion_score" in selected.index:
        result["criterion_score"] = safe_float(selected["criterion_score"])
    return result


def run_deployable_selections(strategy_input: pd.DataFrame) -> pd.DataFrame:
    select_group_v71, select_v72, _, _ = load_selector_modules()
    helpers = (select_group_v71, select_v72)
    rows: list[dict[str, Any]] = []
    grouped = list(strategy_input.groupby(["evaluation_dataset", "group_id"], sort=True))
    for group_number, (_, group) in enumerate(grouped, start=1):
        for strategy_id in STRATEGIES:
            rows.append(select_strategy(group.copy(), strategy_id, helpers))
        if group_number % 25 == 0:
            logging.info("Selected %d/%d frozen candidate groups", group_number, len(grouped))
    raw = pd.DataFrame(rows)
    if len(raw) != len(grouped) * len(STRATEGIES):
        raise RuntimeError("deployable strategy selection row count mismatch")
    if not raw["selection_generated_without_truth"].all():
        raise RuntimeError("truth-free selection marker failed")
    return raw


def evaluation_metric(frame: pd.DataFrame) -> pd.Series:
    final = numeric(frame["final_position_error_m"])
    trajectory = numeric(frame["mean_trajectory_position_error_m"])
    return pd.Series(np.where(frame["evaluation_dataset"].astype(str).eq("URBANNAV_EXP02A"), trajectory, final), index=frame.index)


def evaluate_selection_keys(raw: pd.DataFrame, strategy_input: pd.DataFrame) -> pd.DataFrame:
    # This is intentionally the first read of evaluation truth in the run phase.
    truth = pd.read_csv(TRUTH_INPUT_PATH)
    references = pd.read_csv(REFERENCE_SELECTION_PATH)
    truth["evaluation_error_m"] = evaluation_metric(truth)
    finite_global = truth.loc[np.isfinite(truth["evaluation_error_m"]), "evaluation_error_m"]
    if finite_global.empty:
        raise RuntimeError("no finite post-selection evaluation error exists")
    global_penalty = float(finite_global.max())

    truth_keyed = truth.set_index(["evaluation_dataset", "group_id", "candidate_model"], drop=False)
    reference_keyed = references.set_index(["evaluation_dataset", "group_id"])
    input_keyed = strategy_input.set_index(["evaluation_dataset", "group_id", "candidate_model"], drop=False)
    evaluated_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for (dataset, group_id), group_truth in truth.groupby(["evaluation_dataset", "group_id"], sort=True):
        errors = numeric(group_truth["evaluation_error_m"])
        finite = group_truth.loc[np.isfinite(errors)].copy()
        run_penalty = float(errors[np.isfinite(errors)].max()) if np.isfinite(errors).any() else global_penalty
        if finite.empty:
            oracle_model = ""
            oracle_error = np.nan
        else:
            oracle_row = finite.sort_values(["evaluation_error_m", "candidate_model"], kind="mergesort").iloc[0]
            oracle_model = str(oracle_row["candidate_model"])
            oracle_error = float(oracle_row["evaluation_error_m"])
        metadata = group_truth.iloc[0]
        oracle_rows.append(
            {
                "evaluation_dataset": dataset,
                "group_id": group_id,
                "strategy_id": "STR-ORACLE",
                "strategy_name": "Post-selection oracle",
                "selected_model": oracle_model,
                "selected_family": MODEL_FAMILY.get(oracle_model, "selection_failure"),
                "selection_success": bool(oracle_model),
                "selection_reason": "post_selection_minimum_evaluation_error_non_deployable",
                "eligible_candidate_count": int(len(finite)),
                "criterion_score": np.nan,
                "selection_generated_without_truth": False,
                "family_or_scenario": metadata["family_or_scenario"],
                "realization_or_run_id": metadata["realization_or_run_id"],
                "route": metadata["route"],
                "motion_type": metadata["motion_type"],
                "perturbation_profile": metadata["perturbation_profile"],
                "selected_error_m": oracle_error,
                "error_penalized_m": oracle_error if np.isfinite(oracle_error) else run_penalty,
                "failure_penalty_m": run_penalty,
                "oracle_model": oracle_model,
                "oracle_family": MODEL_FAMILY.get(oracle_model, "unknown"),
                "oracle_error_m": oracle_error,
                "selected_minus_oracle_regret_m": 0.0 if np.isfinite(oracle_error) else np.nan,
                "exact_oracle": True,
                "oracle_family_agreement": True,
                "quality_pass": np.nan,
                "numerical_success": np.nan,
                "physical_plausible": np.nan,
                "reference_selected_model": reference_keyed.loc[(dataset, group_id), "reference_selected_model"],
                "matches_reference_selection": np.nan,
                "selected_beats_m0": np.nan,
                "selected_beats_m2": np.nan,
                "selected_beats_m13": np.nan,
                "failure_reason": "",
            }
        )

        group_raw = raw.loc[(raw["evaluation_dataset"] == dataset) & (raw["group_id"] == group_id)]
        if len(group_raw) != len(STRATEGIES):
            raise RuntimeError(f"missing deployable strategies for {dataset}:{group_id}")
        baseline_errors: dict[str, float] = {}
        for short, model in {"m0": "M0_static_position", "m2": "M2_ctd_full", "m13": "M13_robust_ctd_full_plus_gir_refine"}.items():
            key = (dataset, group_id, model)
            baseline_errors[short] = safe_float(truth_keyed.loc[key, "evaluation_error_m"]) if key in truth_keyed.index else np.nan
        expected_model = str(reference_keyed.loc[(dataset, group_id), "reference_selected_model"])

        for _, selection in group_raw.iterrows():
            model = str(selection["selected_model"])
            key = (dataset, group_id, model)
            selected_error = np.nan
            selected_input: pd.Series | None = None
            if model and key in truth_keyed.index:
                selected_error = safe_float(truth_keyed.loc[key, "evaluation_error_m"])
                selected_input = input_keyed.loc[key]
                if isinstance(selected_input, pd.DataFrame):
                    selected_input = selected_input.iloc[0]
            selection_success = bool(safe_bool(selection["selection_key_valid"]) and np.isfinite(selected_error))
            penalized = float(selected_error) if selection_success else run_penalty
            regret = penalized - oracle_error if np.isfinite(oracle_error) else np.nan
            failure_reason = "" if selection_success else (str(selection["selection_reason"]) or "nonfinite_selected_evaluation_error")
            evaluated_rows.append(
                {
                    **selection.to_dict(),
                    "selected_family": MODEL_FAMILY.get(model, "selection_failure") if selection_success else "selection_failure",
                    "selection_success": selection_success,
                    "family_or_scenario": metadata["family_or_scenario"],
                    "realization_or_run_id": metadata["realization_or_run_id"],
                    "route": metadata["route"],
                    "motion_type": metadata["motion_type"],
                    "perturbation_profile": metadata["perturbation_profile"],
                    "selected_error_m": selected_error,
                    "error_penalized_m": penalized,
                    "failure_penalty_m": run_penalty,
                    "oracle_model": oracle_model,
                    "oracle_family": MODEL_FAMILY.get(oracle_model, "unknown"),
                    "oracle_error_m": oracle_error,
                    "selected_minus_oracle_regret_m": regret,
                    "exact_oracle": bool(selection_success and model == oracle_model),
                    "oracle_family_agreement": bool(selection_success and MODEL_FAMILY.get(model) == MODEL_FAMILY.get(oracle_model)),
                    "quality_pass": safe_bool(selected_input["quality_pass"]) if selected_input is not None else False,
                    "numerical_success": safe_bool(selected_input["numerical_success"]) if selected_input is not None else False,
                    "physical_plausible": safe_bool(selected_input["physical_plausible"]) if selected_input is not None else False,
                    "reference_selected_model": expected_model,
                    "matches_reference_selection": bool(selection_success and model == expected_model),
                    "selected_beats_m0": bool(np.isfinite(baseline_errors["m0"]) and penalized <= baseline_errors["m0"]),
                    "selected_beats_m2": bool(np.isfinite(baseline_errors["m2"]) and penalized <= baseline_errors["m2"]),
                    "selected_beats_m13": bool(np.isfinite(baseline_errors["m13"]) and penalized <= baseline_errors["m13"]),
                    "failure_reason": failure_reason,
                }
            )
    return pd.concat([pd.DataFrame(evaluated_rows), pd.DataFrame(oracle_rows)], ignore_index=True)


def bootstrap_median_ci(values: Iterable[float], seed: int, resamples: int = 5000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def distribution_json(values: Iterable[str]) -> str:
    series = pd.Series(list(values), dtype="object")
    counts = series.value_counts(dropna=False).sort_index()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, ensure_ascii=False, sort_keys=True)


def entropy(values: Iterable[str]) -> float:
    counts = pd.Series(list(values), dtype="object").value_counts(dropna=False).to_numpy(dtype=float)
    if counts.sum() <= 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def aggregate_holdout(selections: pd.DataFrame) -> pd.DataFrame:
    primary = selections.loc[
        (selections["evaluation_dataset"] == "PAPER_EXP01_HOLDOUT")
        & (selections["strategy_id"].isin(STRATEGIES))
    ].copy()
    rows = []
    for strategy_id, group in primary.groupby("strategy_id", sort=True):
        errors = numeric(group["error_penalized_m"]).to_numpy(dtype=float)
        conditional = numeric(group.loc[bool_series(group["selection_success"]), "selected_error_m"]).to_numpy(dtype=float)
        regrets = numeric(group["selected_minus_oracle_regret_m"]).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_median_ci(errors, 20260726 + int(strategy_id.split("-")[1]))
        family_values = group["selected_family"].fillna("selection_failure").astype(str)
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": STRATEGIES[strategy_id],
                "total_runs": len(group),
                "valid_selections": int(bool_series(group["selection_success"]).sum()),
                "selection_coverage": float(bool_series(group["selection_success"]).mean()),
                "numerical_failure_rate": float((~bool_series(group["selection_success"])).mean()),
                "quality_pass_rate": float(bool_series(group["quality_pass"]).mean()),
                "conditional_median_position_error_m": float(np.median(conditional)) if len(conditional) else np.nan,
                "conditional_p95_position_error_m": float(np.quantile(conditional, 0.95)) if len(conditional) else np.nan,
                "median_position_error_m": float(np.median(errors)),
                "p75_position_error_m": float(np.quantile(errors, 0.75)),
                "p95_position_error_m": float(np.quantile(errors, 0.95)),
                "maximum_position_error_m": float(np.max(errors)),
                "bootstrap_median_ci_low_m": ci_low,
                "bootstrap_median_ci_high_m": ci_high,
                "median_selected_minus_oracle_regret_m": float(np.median(regrets)),
                "p95_regret_m": float(np.quantile(regrets, 0.95)),
                "maximum_regret_m": float(np.max(regrets)),
                "exact_oracle_rate": float(bool_series(group["exact_oracle"]).mean()),
                "oracle_family_agreement_rate": float(bool_series(group["oracle_family_agreement"]).mean()),
                "selected_beats_m0_rate": float(bool_series(group["selected_beats_m0"]).mean()),
                "selected_beats_m2_rate": float(bool_series(group["selected_beats_m2"]).mean()),
                "selected_beats_m13_rate": float(bool_series(group["selected_beats_m13"]).mean()),
                "error_gt_500m_rate": float((errors > 500.0).mean()),
                "error_gt_1000m_rate": float((errors > 1000.0).mean()),
                "error_gt_2000m_rate": float((errors > 2000.0).mean()),
                "family_selection_entropy": entropy(family_values),
                "model_selection_distribution": distribution_json(group["selected_model"].replace("", "SELECTION_FAILURE")),
                "family_selection_distribution": distribution_json(family_values),
            }
        )
    aggregate = pd.DataFrame(rows).sort_values("strategy_id").reset_index(drop=True)
    aggregate["rank_p95"] = aggregate["p95_position_error_m"].rank(method="min", ascending=True).astype(int)
    aggregate["rank_gt_1km_rate"] = aggregate["error_gt_1000m_rate"].rank(method="min", ascending=True).astype(int)
    return aggregate


def family_breakdown(selections: pd.DataFrame) -> pd.DataFrame:
    primary = selections.loc[
        (selections["evaluation_dataset"] == "PAPER_EXP01_HOLDOUT")
        & (selections["strategy_id"].isin(STRATEGIES))
    ].copy()
    rows = []
    for (strategy_id, family), group in primary.groupby(["strategy_id", "family_or_scenario"], sort=True):
        errors = numeric(group["error_penalized_m"]).to_numpy(dtype=float)
        regrets = numeric(group["selected_minus_oracle_regret_m"]).to_numpy(dtype=float)
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": STRATEGIES[strategy_id],
                "holdout_family": family,
                "runs": len(group),
                "coverage": float(bool_series(group["selection_success"]).mean()),
                "median_position_error_m": float(np.median(errors)),
                "p95_position_error_m": float(np.quantile(errors, 0.95)),
                "max_position_error_m": float(np.max(errors)),
                "median_regret_m": float(np.median(regrets)),
                "p95_regret_m": float(np.quantile(regrets, 0.95)),
                "exact_oracle_rate": float(bool_series(group["exact_oracle"]).mean()),
                "oracle_family_agreement_rate": float(bool_series(group["oracle_family_agreement"]).mean()),
                "error_gt_500m_rate": float((errors > 500.0).mean()),
                "error_gt_1000m_rate": float((errors > 1000.0).mean()),
                "selected_model_distribution": distribution_json(group["selected_model"].replace("", "SELECTION_FAILURE")),
            }
        )
    return pd.DataFrame(rows)


def core_scenario_comparison(selections: pd.DataFrame) -> pd.DataFrame:
    core = selections.loc[selections["evaluation_dataset"] == "TECH18_CORE"].copy()
    ma = core.loc[core["strategy_id"] == "STR-01"].set_index("group_id")
    rows = []
    for _, row in core.iterrows():
        ma_model = str(ma.loc[row["group_id"], "selected_model"])
        rows.append(
            {
                "scenario": row["family_or_scenario"],
                "strategy_id": row["strategy_id"],
                "strategy_name": row["strategy_name"],
                "selected_model": row["selected_model"],
                "position_error_m": row["selected_error_m"],
                "failure_penalized_error_m": row["error_penalized_m"],
                "oracle_model": row["oracle_model"],
                "oracle_error_m": row["oracle_error_m"],
                "oracle_gap_m": row["selected_minus_oracle_regret_m"],
                "matches_ma_bgtr": bool(row["selected_model"] == ma_model),
                "matches_oracle": bool(row["exact_oracle"]),
                "selection_success": bool(row["selection_success"]),
                "s7_behavior": "declared limitation retained in frozen policy" if row["family_or_scenario"] == "S7_fast_north_no_bias" else "not_applicable",
                "q3_behavior": "full-pool robust-refine reference with oracle gap" if row["family_or_scenario"] == "Q3_qatar_burst_outlier" else "not_applicable",
                "q4_behavior": "robust_selected" if row["family_or_scenario"] == "Q4_qatar_hard_geometry_noise" and str(row["selected_model"]).startswith(("M7", "M8", "M9", "M13")) else ("robust_rejected" if row["family_or_scenario"] == "Q4_qatar_hard_geometry_noise" else "not_applicable"),
                "real_static_behavior": "static_selected" if row["family_or_scenario"] == "real_iridium" and MODEL_FAMILY.get(str(row["selected_model"])) == "static" else ("dynamic_selected" if row["family_or_scenario"] == "real_iridium" else "not_applicable"),
            }
        )
    return pd.DataFrame(rows)


def urban_stress_comparison(selections: pd.DataFrame) -> pd.DataFrame:
    urban = selections.loc[
        (selections["evaluation_dataset"] == "URBANNAV_EXP02A")
        & (selections["strategy_id"].isin(STRATEGIES))
    ].copy()
    rows = []
    for strategy_id, group in urban.groupby("strategy_id", sort=True):
        errors = numeric(group["error_penalized_m"]).to_numpy(dtype=float)
        regrets = numeric(group["selected_minus_oracle_regret_m"]).to_numpy(dtype=float)
        families = group["selected_family"].astype(str)
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": STRATEGIES[strategy_id],
                "runs": len(group),
                "coverage": float(bool_series(group["selection_success"]).mean()),
                "median_mean_trajectory_error_m": float(np.median(errors)),
                "p95_mean_trajectory_error_m": float(np.quantile(errors, 0.95)),
                "max_mean_trajectory_error_m": float(np.max(errors)),
                "median_regret_m": float(np.median(regrets)),
                "error_gt_1000m_rate": float((errors > 1000.0).mean()),
                "static_family_selection_rate": float(families.eq("static").mean()),
                "dynamic_family_selection_rate": float((~families.isin(["static", "selection_failure"])).mean()),
                "model_selection_distribution": distribution_json(group["selected_model"].replace("", "SELECTION_FAILURE")),
                "claim_role": "supplementary model-out-of-family stress only",
            }
        )
    return pd.DataFrame(rows)


def catastrophic_risk(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    deployable = selections.loc[selections["strategy_id"].isin(STRATEGIES)].copy()
    for (dataset, strategy_id), group in deployable.groupby(["evaluation_dataset", "strategy_id"], sort=True):
        errors = numeric(group["error_penalized_m"]).to_numpy(dtype=float)
        for threshold in [500.0, 1000.0, 2000.0]:
            rows.append(
                {
                    "evaluation_dataset": dataset,
                    "strategy_id": strategy_id,
                    "strategy_name": STRATEGIES[strategy_id],
                    "threshold_m": threshold,
                    "catastrophic_count": int((errors > threshold).sum()),
                    "total_runs": len(errors),
                    "catastrophic_rate": float((errors > threshold).mean()),
                    "coverage": float(bool_series(group["selection_success"]).mean()),
                    "failure_penalized": True,
                }
            )
    return pd.DataFrame(rows)


def rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (np.abs(differences) > 0.0)]
    if len(differences) == 0:
        return 0.0
    if stats is not None:
        ranks = stats.rankdata(np.abs(differences))
    else:
        ranks = pd.Series(np.abs(differences)).rank(method="average").to_numpy(dtype=float)
    total = float(ranks.sum())
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / total) if total else 0.0


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    if stats is not None and hasattr(stats, "binomtest"):
        return float(stats.binomtest(min(discordant_a, discordant_b), total, 0.5, alternative="two-sided").pvalue)
    # Exact two-sided binomial tail without installing another package.
    tail = sum(math.comb(total, k) for k in range(0, min(discordant_a, discordant_b) + 1)) / (2**total)
    return float(min(1.0, 2.0 * tail))


def holm_adjust(p_values: list[float]) -> list[float]:
    array = np.asarray(p_values, dtype=float)
    order = np.argsort(np.where(np.isfinite(array), array, np.inf))
    adjusted = np.full(len(array), np.nan)
    running = 0.0
    m = len(array)
    for rank, index in enumerate(order):
        if not np.isfinite(array[index]):
            continue
        current = min(1.0, (m - rank) * array[index])
        running = max(running, current)
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(selections: pd.DataFrame) -> pd.DataFrame:
    primary = selections.loc[
        (selections["evaluation_dataset"] == "PAPER_EXP01_HOLDOUT")
        & (selections["strategy_id"].isin(STRATEGIES))
    ].copy()
    pivot = primary.pivot(index="group_id", columns="strategy_id", values="error_penalized_m").sort_index()
    ma = pivot["STR-01"].to_numpy(dtype=float)
    rows = []
    for strategy_id in [item for item in STRATEGIES if item != "STR-01"]:
        baseline = pivot[strategy_id].to_numpy(dtype=float)
        difference = baseline - ma
        if stats is not None:
            try:
                p_value = float(stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue) if np.any(np.abs(difference) > 0.0) else 1.0
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan
        rng = np.random.default_rng(20260726 + int(strategy_id.split("-")[1]))
        bootstrap = np.median(rng.choice(difference, size=(5000, len(difference)), replace=True), axis=1)
        row = {
            "comparison": f"STR-01_vs_{strategy_id}",
            "baseline_strategy_id": strategy_id,
            "baseline_strategy_name": STRATEGIES[strategy_id],
            "paired_sample_count": len(difference),
            "median_baseline_minus_ma_error_m": float(np.median(difference)),
            "mean_baseline_minus_ma_error_m": float(np.mean(difference)),
            "wilcoxon_p_value": p_value,
            "holm_adjusted_p_value": np.nan,
            "rank_biserial_effect_size": rank_biserial(difference),
            "bootstrap_median_difference_ci_low_m": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_median_difference_ci_high_m": float(np.quantile(bootstrap, 0.975)),
            "ma_bgtr_better_count": int((ma < baseline).sum()),
            "baseline_better_count": int((baseline < ma).sum()),
            "tie_count": int(np.isclose(ma, baseline, rtol=0.0, atol=1e-12).sum()),
            "difference_sign": "positive favors MA-BGTR",
            "exploratory": True,
        }
        for threshold, label in [(500.0, "500m"), (1000.0, "1000m")]:
            ma_cat = ma > threshold
            base_cat = baseline > threshold
            ma_only = int((ma_cat & ~base_cat).sum())
            baseline_only = int((~ma_cat & base_cat).sum())
            row[f"mcnemar_{label}_ma_only_catastrophe"] = ma_only
            row[f"mcnemar_{label}_baseline_only_catastrophe"] = baseline_only
            row[f"mcnemar_{label}_exact_p_value"] = exact_mcnemar(ma_only, baseline_only)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p_value"] = holm_adjust(frame["wilcoxon_p_value"].tolist())
    return frame


def disagreements(selections: pd.DataFrame) -> pd.DataFrame:
    deployable = selections.loc[selections["strategy_id"].isin(STRATEGIES)].copy()
    ma = deployable.loc[deployable["strategy_id"] == "STR-01"].set_index(["evaluation_dataset", "group_id"])
    rows = []
    for _, row in deployable.loc[deployable["strategy_id"] != "STR-01"].iterrows():
        key = (row["evaluation_dataset"], row["group_id"])
        ma_row = ma.loc[key]
        if str(row["selected_model"]) == str(ma_row["selected_model"]):
            continue
        rows.append(
            {
                "evaluation_dataset": row["evaluation_dataset"],
                "group_id": row["group_id"],
                "family_or_scenario": row["family_or_scenario"],
                "strategy_id": row["strategy_id"],
                "ma_bgtr_model": ma_row["selected_model"],
                "baseline_model": row["selected_model"],
                "ma_bgtr_error_m": ma_row["error_penalized_m"],
                "baseline_error_m": row["error_penalized_m"],
                "baseline_minus_ma_error_m": row["error_penalized_m"] - ma_row["error_penalized_m"],
                "oracle_model": row["oracle_model"],
                "oracle_error_m": row["oracle_error_m"],
                "baseline_selection_success": row["selection_success"],
            }
        )
    return pd.DataFrame(rows)


def failure_cases(selections: pd.DataFrame) -> pd.DataFrame:
    deployable = selections.loc[selections["strategy_id"].isin(STRATEGIES)].copy()
    mask = (
        ~bool_series(deployable["selection_success"])
        | (numeric(deployable["error_penalized_m"]) > 1000.0)
        | (numeric(deployable["selected_minus_oracle_regret_m"]) > 500.0)
    )
    out = deployable.loc[mask].copy()
    out["failure_case_reason"] = out.apply(
        lambda row: ";".join(
            item
            for item, flag in [
                ("selection_failure", not safe_bool(row["selection_success"])),
                ("error_gt_1km", safe_float(row["error_penalized_m"]) > 1000.0),
                ("regret_gt_500m", safe_float(row["selected_minus_oracle_regret_m"]) > 500.0),
            ]
            if flag
        ),
        axis=1,
    )
    columns = [
        "evaluation_dataset",
        "group_id",
        "family_or_scenario",
        "realization_or_run_id",
        "strategy_id",
        "strategy_name",
        "selected_model",
        "selection_success",
        "selected_error_m",
        "error_penalized_m",
        "oracle_model",
        "oracle_error_m",
        "selected_minus_oracle_regret_m",
        "failure_case_reason",
        "failure_reason",
    ]
    return out.loc[:, columns].sort_values(["evaluation_dataset", "group_id", "strategy_id"])


def decide_support(aggregate: pd.DataFrame, families: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    metrics = aggregate.set_index("strategy_id")
    ma = metrics.loc["STR-01"]
    simple_ids = [item for item in STRATEGIES if item != "STR-01"]

    weak_dominators: list[str] = []
    strong_dominators: list[str] = []
    critical_protection: dict[str, bool] = {}
    family_keyed = families.set_index(["strategy_id", "holdout_family"])
    dynamic_families = [
        "H1_dynamic_clean_holdout",
        "H2_bias_drift_holdout",
        "H3_outlier_holdout",
        "H4_geometry_ephemeris_holdout",
        "H5_no_bias_dynamic_holdout",
    ]
    for strategy_id in simple_ids:
        base = metrics.loc[strategy_id]
        comparisons = [
            base["selection_coverage"] >= ma["selection_coverage"] - 1e-12,
            base["median_position_error_m"] <= ma["median_position_error_m"] + 1e-12,
            base["p95_position_error_m"] <= ma["p95_position_error_m"] + 1e-12,
            base["error_gt_1000m_rate"] <= ma["error_gt_1000m_rate"] + 1e-12,
        ]
        strict = sum(
            [
                base["selection_coverage"] > ma["selection_coverage"] + 1e-12,
                base["median_position_error_m"] < ma["median_position_error_m"] - 1e-12,
                base["p95_position_error_m"] < ma["p95_position_error_m"] - 1e-12,
                base["error_gt_1000m_rate"] < ma["error_gt_1000m_rate"] - 1e-12,
            ]
        )
        if all(comparisons) and strict >= 1:
            weak_dominators.append(strategy_id)
        if all(comparisons) and strict >= 2:
            strong_dominators.append(strategy_id)
        protected_families = []
        for family in ["H0_static_holdout", *dynamic_families]:
            ma_family = family_keyed.loc[("STR-01", family)]
            base_family = family_keyed.loc[(strategy_id, family)]
            protected = bool(
                ma_family["p95_position_error_m"] < base_family["p95_position_error_m"] - 1e-12
                or ma_family["error_gt_1000m_rate"] < base_family["error_gt_1000m_rate"] - 1e-12
            )
            if protected:
                protected_families.append(family)
        critical_protection[strategy_id] = bool(
            "H0_static_holdout" in protected_families
            and sum(family in protected_families for family in dynamic_families) >= 2
        )

    top_three = bool(ma["rank_p95"] <= 3 and ma["rank_gt_1km_rate"] <= 3)
    fixed_advantages = []
    for strategy_id in ["STR-02", "STR-03"]:
        base = metrics.loc[strategy_id]
        fixed_advantages.append(
            bool(
                (ma["p95_position_error_m"] < base["p95_position_error_m"] - 1e-12 or ma["error_gt_1000m_rate"] < base["error_gt_1000m_rate"] - 1e-12)
                and ma["median_position_error_m"] <= 1.10 * base["median_position_error_m"]
            )
        )
    ma_family = families.loc[families["strategy_id"] == "STR-01"].set_index("holdout_family")
    family_nonfailure = {
        family: bool(ma_family.loc[family, "coverage"] > 0.0 and ma_family.loc[family, "error_gt_1000m_rate"] < 1.0)
        for family in ma_family.index
    }
    family_requirement = bool(
        family_nonfailure.get("H0_static_holdout", False)
        and sum(family_nonfailure.get(family, False) for family in dynamic_families) >= 2
    )
    support_conditions = {
        "no_weak_dominator": len(weak_dominators) == 0,
        "p95_and_gt1km_top_three": top_three,
        "fixed_baseline_tail_advantage_without_gt10pct_median_loss": any(fixed_advantages),
        "h0_and_two_dynamic_families_not_total_failure": family_requirement,
    }
    does_not_candidates = [strategy_id for strategy_id in strong_dominators if not critical_protection[strategy_id]]
    if all(support_conditions.values()):
        decision = "strategy_baseline_supports_ma_bgtr"
    elif does_not_candidates:
        decision = "strategy_baseline_does_not_support_ma_bgtr"
    else:
        decision = "strategy_baseline_partial_support"
    audit = {
        "decision": decision,
        "support_conditions": support_conditions,
        "weak_dominators": weak_dominators,
        "strong_dominators": strong_dominators,
        "critical_family_protection_by_strategy": critical_protection,
        "does_not_support_candidates": does_not_candidates,
        "ma_bgtr_rank_p95": int(ma["rank_p95"]),
        "ma_bgtr_rank_gt_1km_rate": int(ma["rank_gt_1km_rate"]),
        "fixed_advantage_checks": {"STR-02": fixed_advantages[0], "STR-03": fixed_advantages[1]},
        "ma_family_nonfailure": family_nonfailure,
    }
    return decision, audit


def figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(DIRS["figures"] / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(DIRS["figures"] / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def short_strategy(strategy_id: str) -> str:
    return "MA" if strategy_id == "STR-01" else strategy_id.replace("STR-", "S")


def generate_figures(
    selections: pd.DataFrame,
    aggregate: pd.DataFrame,
    families: pd.DataFrame,
    core: pd.DataFrame,
    urban: pd.DataFrame,
    best_simple_id: str,
) -> list[str]:
    figure_style()
    order = list(STRATEGIES)
    labels = [short_strategy(item) for item in order]
    colors = ["#0072B2" if item == "STR-01" else "#999999" for item in order]
    generated: list[str] = []

    data = aggregate.set_index("strategy_id").loc[order]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(order))
    width = 0.24
    for offset, column, label, hatch in [(-width, "median_position_error_m", "Median", ""), (0.0, "p95_position_error_m", "P95", "//"), (width, "maximum_position_error_m", "Maximum", "xx")]:
        bars = ax.bar(x + offset, data[column], width, label=label, color=colors, edgecolor="black", linewidth=0.45, hatch=hatch, alpha=0.9)
        for index, bar in enumerate(bars):
            if order[index] == "STR-01":
                bar.set_edgecolor("#003B5C")
                bar.set_linewidth(1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Position error (m, log scale)")
    ax.set_xticks(x, labels)
    ax.set_title("Hold-out error risk across frozen-candidate selection strategies")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    stem = "fig01_strategy_median_p95_max_comparison"
    save_figure(fig, stem)
    generated.append(stem)

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    width = 0.24
    for offset, column, label, hatch in [(-width, "error_gt_500m_rate", ">500 m", ""), (0.0, "error_gt_1000m_rate", ">1 km", "//"), (width, "error_gt_2000m_rate", ">2 km", "xx")]:
        ax.bar(x + offset, 100.0 * data[column], width, label=label, color=colors, edgecolor="black", linewidth=0.45, hatch=hatch)
    ax.set_ylim(0, max(5.0, 1.12 * 100.0 * data[["error_gt_500m_rate", "error_gt_1000m_rate", "error_gt_2000m_rate"]].to_numpy().max()))
    ax.set_ylabel("Failure rate (%)")
    ax.set_xticks(x, labels)
    ax.set_title("Failure-penalized catastrophic risk on 60 observable-only hold-outs")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    stem = "fig02_strategy_catastrophic_failure_rates"
    save_figure(fig, stem)
    generated.append(stem)

    family_order = [
        "H0_static_holdout",
        "H1_dynamic_clean_holdout",
        "H2_bias_drift_holdout",
        "H3_outlier_holdout",
        "H4_geometry_ephemeris_holdout",
        "H5_no_bias_dynamic_holdout",
    ]
    heat = families.pivot(index="strategy_id", columns="holdout_family", values="p95_position_error_m").reindex(index=order, columns=family_order)
    plot_values = np.log10(np.maximum(heat.to_numpy(dtype=float), 1e-3))
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    image = ax.imshow(plot_values, aspect="auto", cmap="cividis")
    for i in range(plot_values.shape[0]):
        for j in range(plot_values.shape[1]):
            value = heat.iloc[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.6, color="white" if plot_values[i, j] > np.nanmedian(plot_values) else "black")
    ax.set_xticks(range(len(family_order)), [item.split("_")[0] for item in family_order])
    ax.set_yticks(range(len(order)), labels)
    ax.set_xlabel("Hold-out family")
    ax.set_ylabel("Strategy")
    ax.set_title("P95 position error by hold-out family (annotations in metres)")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("log10(error in m)")
    fig.tight_layout()
    stem = "fig03_holdout_family_error_heatmap"
    save_figure(fig, stem)
    generated.append(stem)

    primary = selections.loc[(selections["evaluation_dataset"] == "PAPER_EXP01_HOLDOUT") & selections["strategy_id"].isin(order)].copy()
    regret_data = [numeric(primary.loc[primary["strategy_id"] == item, "selected_minus_oracle_regret_m"]).to_numpy(dtype=float) for item in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    box = ax.boxplot(regret_data, tick_labels=labels, showfliers=True, patch_artist=True, whis=1.5)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_ylabel("Selected-minus-oracle regret (m, symlog)")
    ax.set_title("Regret distribution with all long-tail runs retained")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    stem = "fig04_selected_minus_oracle_regret_distribution"
    save_figure(fig, stem)
    generated.append(stem)

    family_names = ["static", "core_ctd", "robust", "be_gtr", "direct_gir", "cascaded_refinement", "selection_failure"]
    counts = np.zeros((len(order), len(family_names)), dtype=float)
    for i, strategy_id in enumerate(order):
        values = primary.loc[primary["strategy_id"] == strategy_id, "selected_family"].astype(str).value_counts()
        for j, family in enumerate(family_names):
            counts[i, j] = values.get(family, 0) / 60.0
    family_colors = ["#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#0072B2", "#000000"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bottom = np.zeros(len(order))
    for j, family in enumerate(family_names):
        ax.bar(x, counts[:, j], bottom=bottom, color=family_colors[j], edgecolor="white", linewidth=0.4, label=family.replace("_", " "))
        bottom += counts[:, j]
    ax.set_ylim(0, 1)
    ax.set_ylabel("Selection proportion")
    ax.set_xticks(x, labels)
    ax.set_title("Model-family selection distribution on primary hold-outs")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    stem = "fig05_strategy_model_family_distribution"
    save_figure(fig, stem)
    generated.append(stem)

    core_deploy = core.loc[core["strategy_id"].isin(order)].copy()
    scenarios = list(dict.fromkeys(core_deploy["scenario"].astype(str)))
    matrix = np.full((len(scenarios), len(order)), np.nan)
    annotations = np.full((len(scenarios), len(order)), "", dtype=object)
    for i, scenario in enumerate(scenarios):
        for j, strategy_id in enumerate(order):
            subset = core_deploy.loc[(core_deploy["scenario"] == scenario) & (core_deploy["strategy_id"] == strategy_id)]
            if subset.empty or not safe_bool(subset.iloc[0]["selection_success"]):
                continue
            model = str(subset.iloc[0]["selected_model"])
            matrix[i, j] = int(model.split("_")[0][1:])
            annotations[i, j] = model.split("_")[0]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    image = ax.imshow(matrix, aspect="auto", cmap="tab20", vmin=0, vmax=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, annotations[i, j] or "FAIL", ha="center", va="center", fontsize=6.4, color="black")
    ax.set_xticks(range(len(order)), labels)
    ax.set_yticks(range(len(scenarios)), [item.replace("_", " ") for item in scenarios])
    ax.set_title("TECH18 core-scenario selected-model matrix")
    ax.set_xlabel("Strategy")
    fig.colorbar(image, ax=ax, pad=0.02, label="Model index")
    fig.tight_layout()
    stem = "fig06_core_scenario_strategy_matrix"
    save_figure(fig, stem)
    generated.append(stem)

    paired = primary.pivot(index="group_id", columns="strategy_id", values="error_penalized_m")
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(paired[best_simple_id], paired["STR-01"], c="#0072B2", edgecolors="black", linewidths=0.35, alpha=0.8)
    limits = [max(1e-2, float(np.nanmin(paired[[best_simple_id, "STR-01"]].to_numpy())) * 0.7), float(np.nanmax(paired[[best_simple_id, "STR-01"]].to_numpy())) * 1.3]
    ax.plot(limits, limits, linestyle="--", color="black", linewidth=1.0, label="Equal error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(f"{short_strategy(best_simple_id)} error (m)")
    ax.set_ylabel("MA-BGTR error (m)")
    ax.set_title("Paired hold-out errors: MA-BGTR vs best simple baseline")
    ax.grid(which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = "fig07_ma_bgtr_vs_best_simple_paired_errors"
    save_figure(fig, stem)
    generated.append(stem)

    urban_data = urban.set_index("strategy_id").loc[order]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for offset, column, label, hatch in [(-width, "median_mean_trajectory_error_m", "Median", ""), (0.0, "p95_mean_trajectory_error_m", "P95", "//"), (width, "max_mean_trajectory_error_m", "Maximum", "xx")]:
        ax.bar(x + offset, urban_data[column], width, label=label, color=colors, edgecolor="black", linewidth=0.45, hatch=hatch)
    ax.set_yscale("log")
    ax.set_ylabel("Mean trajectory error (m, log scale)")
    ax.set_xticks(x, labels)
    ax.set_title("UrbanNav model-out-of-family stress comparison (supplementary)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    stem = "fig08_urbannav_stress_strategy_comparison"
    save_figure(fig, stem)
    generated.append(stem)
    return generated


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, float_digits: int = 3) -> str:
    data = frame.loc[:, columns].copy() if columns else frame.copy()
    if data.empty:
        return "_No rows._"
    def render(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(column) for column in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in data.columns) + " |")
    return "\n".join(lines)


def finalize_protection(paths: dict[str, Path]) -> dict[str, Any]:
    baseline = json.loads(PROTECTION_BASELINE_PATH.read_text(encoding="utf-8"))
    critical_after = {name: sha256_file(path) for name, path in paths.items()}
    critical_unchanged = {name: critical_after[name] == baseline["critical_file_hashes"][name] for name in paths}
    trees_after = {name: tree_snapshot(path) for name, path in protected_tree_paths().items()}
    tree_unchanged = {
        name: trees_after[name].get("metadata_digest") == baseline["protected_trees"][name].get("metadata_digest")
        for name in trees_after
    }
    audit = pd.read_csv(AUDIT_PATH)
    after_map = {str(path): digest for name, path in paths.items() for digest in [critical_after[name]]}
    audit["sha256_after"] = audit["source_path"].map(after_map)
    audit["unchanged"] = audit["sha256_before"].astype(str) == audit["sha256_after"].astype(str)
    audit.to_csv(AUDIT_PATH, index=False)
    result = {
        "checked_local": datetime.now().astimezone().isoformat(),
        "critical_file_hashes_after": critical_after,
        "critical_file_unchanged": critical_unchanged,
        "protected_trees_after": trees_after,
        "protected_tree_unchanged": tree_unchanged,
        "selector_hash_unchanged": critical_unchanged["selector"],
        "freeze_policy_hash_unchanged": critical_unchanged["freeze_policy"],
        "paper_exp01_unchanged": tree_unchanged["paper_exp01"],
        "tech18_inputs_unchanged": all(critical_unchanged[name] for name in ["tech18_candidates", "tech18_selected", "tech18_reference", "tech18_decision"]),
        "urban_exp02a_unchanged": tree_unchanged["urban_exp02a"],
        "urban_exp02b_unchanged": tree_unchanged["urban_exp02b"],
        "urban_exp02c_unchanged": tree_unchanged["urban_exp02c"],
        "urban_exp02d_unchanged": tree_unchanged["urban_exp02d"],
        "manuscript_v0_8_unchanged": tree_unchanged["manuscript_v0_8"],
        "solver_rerun_count": 0,
        "new_candidate_estimation_count": 0,
        "frozen_candidate_rows_only": True,
        "pass": bool(all(critical_unchanged.values()) and all(tree_unchanged.values())),
    }
    write_json(DIRS["protocol"] / "PAPER_EXP03A_PROTECTION_AUDIT.json", result)
    return result


def local_file_rows(path: Path) -> int | str:
    if path.suffix.lower() == ".csv":
        return row_count(path)
    if path.suffix.lower() == ".json":
        return 1
    return ""


def local_file_index() -> pd.DataFrame:
    files = sorted(
        (
            path
            for path in OUT.rglob("*")
            if path.is_file()
            and path != OUT / "PAPER_EXP03A_TOTAL_REPORT.md"
            and path.name != "PAPER_EXP03A_LOCAL_FILE_INDEX.csv"
        ),
        key=lambda path: path.relative_to(OUT).as_posix(),
    )
    return pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(OUT).as_posix(),
                "absolute_path": str(path),
                "row_count": local_file_rows(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    )


def report_text(
    protocol: dict[str, Any],
    selections: pd.DataFrame,
    aggregate: pd.DataFrame,
    families: pd.DataFrame,
    core: pd.DataFrame,
    urban: pd.DataFrame,
    catastrophe: pd.DataFrame,
    statistical: pd.DataFrame,
    decision: str,
    decision_audit: dict[str, Any],
    protection: dict[str, Any],
    best_simple_id: str,
    index: pd.DataFrame,
    figure_stems: list[str],
) -> str:
    ma_recovery = selections.loc[selections["strategy_id"] == "STR-01"].groupby("evaluation_dataset")["matches_reference_selection"].agg(["sum", "count"])
    ma = aggregate.set_index("strategy_id").loc["STR-01"]
    best = aggregate.set_index("strategy_id").loc[best_simple_id]
    protocol_hash = PROTOCOL_HASH_PATH.read_text(encoding="ascii").split()[0]

    aggregate_view = aggregate[
        [
            "strategy_id",
            "strategy_name",
            "selection_coverage",
            "median_position_error_m",
            "p95_position_error_m",
            "maximum_position_error_m",
            "median_selected_minus_oracle_regret_m",
            "p95_regret_m",
            "exact_oracle_rate",
            "oracle_family_agreement_rate",
            "error_gt_500m_rate",
            "error_gt_1000m_rate",
            "rank_p95",
            "rank_gt_1km_rate",
        ]
    ].copy()
    for column in ["selection_coverage", "exact_oracle_rate", "oracle_family_agreement_rate", "error_gt_500m_rate", "error_gt_1000m_rate"]:
        aggregate_view[column] = 100.0 * numeric(aggregate_view[column])
    aggregate_view = aggregate_view.rename(columns={column: column.replace("_rate", "_pct") for column in aggregate_view.columns if column.endswith("_rate")})

    family_order = [
        "H0_static_holdout",
        "H1_dynamic_clean_holdout",
        "H2_bias_drift_holdout",
        "H3_outlier_holdout",
        "H4_geometry_ephemeris_holdout",
        "H5_no_bias_dynamic_holdout",
    ]
    compact_family_rows = []
    for strategy_id in STRATEGIES:
        row = {"strategy": strategy_id}
        subset = families.loc[families["strategy_id"] == strategy_id].set_index("holdout_family")
        for family in family_order:
            row[family.split("_")[0]] = f"{subset.loc[family, 'median_position_error_m']:.1f} / {subset.loc[family, 'p95_position_error_m']:.1f}"
        compact_family_rows.append(row)
    compact_family = pd.DataFrame(compact_family_rows)

    cat_primary = catastrophe.loc[
        (catastrophe["evaluation_dataset"] == "PAPER_EXP01_HOLDOUT")
        & catastrophe["strategy_id"].isin(STRATEGIES)
    ].pivot(index="strategy_id", columns="threshold_m", values="catastrophic_rate").reindex(list(STRATEGIES))
    cat_primary = (100.0 * cat_primary).reset_index().rename(columns={500.0: ">500m_pct", 1000.0: ">1km_pct", 2000.0: ">2km_pct"})

    core_models = core.loc[core["strategy_id"].isin([*STRATEGIES, "STR-ORACLE"])].pivot(index="scenario", columns="strategy_id", values="selected_model")
    core_models = core_models.reindex(columns=[*STRATEGIES, "STR-ORACLE"]).reset_index()

    family_winners = (
        families.sort_values(["holdout_family", "p95_position_error_m", "median_position_error_m", "strategy_id"])
        .groupby("holdout_family", as_index=False)
        .first()[["holdout_family", "strategy_id", "p95_position_error_m", "median_position_error_m"]]
    )
    ma_family = families.loc[families["strategy_id"] == "STR-01"].set_index("holdout_family")
    ma_protected = []
    ma_lost = []
    for _, winner in family_winners.iterrows():
        family = str(winner["holdout_family"])
        if winner["strategy_id"] == "STR-01":
            ma_protected.append(family)
        else:
            ratio = ma_family.loc[family, "p95_position_error_m"] / max(float(winner["p95_position_error_m"]), 1e-12)
            ma_lost.append(f"{family}: winner={winner['strategy_id']}, MA/winner p95={ratio:.2f}x")

    recovery_table = ma_recovery.reset_index().rename(columns={"sum": "exact_recovery_count", "count": "reference_runs"})
    definitions = strategy_definitions()[["strategy_id", "strategy_name", "candidate_scope", "selection_rule", "limitation"]]
    stats_view = statistical[
        [
            "baseline_strategy_id",
            "paired_sample_count",
            "median_baseline_minus_ma_error_m",
            "wilcoxon_p_value",
            "holm_adjusted_p_value",
            "rank_biserial_effect_size",
            "bootstrap_median_difference_ci_low_m",
            "bootstrap_median_difference_ci_high_m",
            "ma_bgtr_better_count",
            "baseline_better_count",
            "tie_count",
        ]
    ]
    urban_view = urban[
        [
            "strategy_id",
            "coverage",
            "median_mean_trajectory_error_m",
            "p95_mean_trajectory_error_m",
            "max_mean_trajectory_error_m",
            "median_regret_m",
            "error_gt_1000m_rate",
            "static_family_selection_rate",
            "dynamic_family_selection_rate",
        ]
    ].copy()
    for column in ["coverage", "error_gt_1000m_rate", "static_family_selection_rate", "dynamic_family_selection_rate"]:
        urban_view[column] = 100.0 * numeric(urban_view[column])

    decision_explanation = {
        "strategy_baseline_supports_ma_bgtr": "预注册的完整支持条件全部满足：没有简单策略在覆盖率、中位数、P95 与 >1 km 风险上共同支配 MA-BGTR，且门控在长尾或关键 family 上提供了可辨识保护。",
        "strategy_baseline_partial_support": "结果呈混合证据：MA-BGTR 在部分 family 或风险指标上有价值，但一个或多个简单策略在其他主指标上持平或更优；不存在足够证据支持全面必要性。",
        "strategy_baseline_does_not_support_ma_bgtr": "至少一个更简单策略按预注册标准支配 MA-BGTR，且未观察到足以抵消该结果的关键 family 保护优势。",
    }[decision]

    lines = [
        "# PAPER-EXP03A 总报告：冻结候选输出上的标准选择策略对比",
        "",
        "## 1. 任务目的",
        "",
        "本任务在完全相同、已经冻结的 M0--M14 候选输出上比较 MA-BGTR-v7.1 与十种标准选择策略，回答层级证据门控相对固定模型、最小验证残差、信息准则、固定秩评分和通用 Pareto 是否具有必要性。所有 deployable 选择先在无位置真值表上完成，随后才回连误差与 oracle；本任务没有重新运行 Doppler solver。",
        "",
        "## 2. 冻结边界",
        "",
        f"- 冻结 release：`{RELEASE}`",
        f"- selector SHA256：`{protocol['source_files']['selector']['sha256']}`",
        f"- freeze policy SHA256：`{protocol['source_files']['freeze_policy']['sha256']}`",
        "- 候选池：`FULL_POOL_M0_M14_V1`。",
        "- MA-BGTR-v7.1、v7.2 freeze policy、候选模型和 gate 阈值均未修改。",
        "- Solver rerun：0；new candidate estimation：0；只读取冻结候选行。",
        "- 论文 v0.8、PAPER-EXP01、TECH18 和 UrbanNav EXP02A--D 均为只读保护对象。",
        "",
        "## 3. 协议 SHA256",
        "",
        f"`{protocol_hash}`",
        "",
        "协议在任何策略选择前生成并冻结，固定了过滤规则、AICc/BIC 公式、DOF、秩评分权重、Pareto 规则、失败惩罚、统计方法和结论判定。",
        "",
        "## 4. 输入文件与规模",
        "",
        "用户给定的独立逻辑路径仅包含新论文/UrbanNav 分支；PAPER-EXP01 与 TECH18 从历史冻结实体 `C:\\Users\\32377\\Desktop\\leo-c\\_new_solver_lab` 解析。内容哈希纳入协议。",
        "",
        f"- PAPER-EXP01：900 条 primary candidate rows，60 个 observable-only run，每个 run 15 个候选，H0--H5 各 10 个。",
        f"- TECH18：135 条 candidate rows，9 个核心场景，每场景 15 个候选。",
        f"- UrbanNav EXP02A：1080 条 candidate rows，72 个 observable-only solver batch，每 batch 15 个候选。",
        f"- 标准化候选：{len(pd.read_csv(STANDARDIZED_PATH))} 行；无真值策略输入：{len(pd.read_csv(NO_TRUTH_PATH))} 行。",
        "",
        "## 5. Truth-isolation 检查",
        "",
        "策略输入不含 position/trajectory/velocity error、oracle、expected model、hold-out family、route、motion type、扰动 profile 或真值 bias/drift。group ID 被替换为不携带 H/S/Q/route 标签的 opaque ID。`PAPER_EXP03A_PRE_TRUTH_SELECTION_KEYS.csv` 在首次读取 post-selection truth lookup 前写出。",
        "",
        "- `truth_used_for_deployable_selection = false`。",
        "- `oracle_used_for_deployable_selection = false`。",
        "- `scenario_name_hardcode = false`。",
        "- TECH18 的 `dataset_type=real` 保留，因为冻结 static sanity guard 明确依赖这一可部署数据角色；具体场景名不进入策略。",
        "",
        "## 6. MA-BGTR 选择恢复率",
        "",
        markdown_table(recovery_table, float_digits=0),
        "",
        "PAPER-EXP01 必须 60/60、TECH18 必须 9/9；两项均作为继续分析的硬门槛。UrbanNav 的归档 candidate CSV 缺少 covariance/bootstrap/information-retention、BE retention/rank-loss 等 selector 曾读取的列，字段受限源码回放因此不是完整历史重放；缺失值保持缺失，没有用旧 selected model 反向补字段。",
        "",
        "## 7. 策略定义",
        "",
        markdown_table(definitions),
        "",
        "## 8. AICc/BIC 的范围与限制",
        "",
        "严格 AICc/BIC (`STR-06/07`) 仅比较普通高斯嵌套 M0--M4，参数维数固定为 3/4/8/7/6。`STR-10/11` 把同一 Gaussian RSS 近似扩展到完整候选池，只是 sensitivity baseline；Robust、BE 与 GIR 的有效自由度和似然未严格推导，因此不能称为严格信息准则比较。所有信息准则使用实际 observation count，`n <= k+1` 时 AICc 无效。",
        "",
        "## 9. 主 hold-out 汇总",
        "",
        "下表的误差风险采用预注册 failure penalty；coverage 之外没有静默删除失败选择。百分比列单位为 %。",
        "",
        markdown_table(aggregate_view),
        "",
        "## 10. H0--H5 分 family 结果",
        "",
        "单元格为 `median / p95`（m）。",
        "",
        markdown_table(compact_family),
        "",
        "各 family 的 P95 最优策略：",
        "",
        markdown_table(family_winners),
        "",
        "## 11. 灾难性失败率",
        "",
        "以下均为 60 个 primary run 的 failure-penalized 比例（%）；长尾没有截断。",
        "",
        markdown_table(cat_primary),
        "",
        "## 12. 配对统计与 Holm 校正",
        "",
        "差值定义为 `baseline - MA-BGTR`，正值表示 MA-BGTR 误差更低。Wilcoxon、rank-biserial、bootstrap CI 和 McNemar 均为 exploratory analysis，不参与权重或阈值选择。",
        "",
        markdown_table(stats_view),
        "",
        "## 13. TECH18 核心场景",
        "",
        "下表给出每个策略实际选择的模型；`STR-ORACLE` 仅为后验参照。",
        "",
        markdown_table(core_models),
        "",
        "S7 的 declared limitation 与 Q3 full-pool oracle gap 均保留。real Iridium 只用于 static sanity；任何简单 residual 策略若选到动态模型，不能被解释为真实动态能力。Q4 中 robust 选择被单独登记，避免把 outlier evidence 自动等同于 robust 胜出。",
        "",
        "## 14. UrbanNav 外部分布压力集",
        "",
        "UrbanNav 的时变车辆运动超出当前 static/CV 模型族。该表只描述选择策略面对模型外数据的行为，不构成正面动态 LEO 定位基准。百分比列单位为 %。",
        "",
        markdown_table(urban_view),
        "",
        "## 15. 表现最好的简单策略",
        "",
        f"按预注册的报告排序（最低 P95、再最低 >1 km rate、再中位数、coverage、ID），最佳简单策略为 **{best_simple_id} ({STRATEGIES[best_simple_id]})**。其 hold-out median/P95/>1 km rate 分别为 {best['median_position_error_m']:.3f} m、{best['p95_position_error_m']:.3f} m、{100*best['error_gt_1000m_rate']:.2f}%；MA-BGTR 对应为 {ma['median_position_error_m']:.3f} m、{ma['p95_position_error_m']:.3f} m、{100*ma['error_gt_1000m_rate']:.2f}%。",
        "",
        "## 16. MA-BGTR 的独特优势",
        "",
        ("MA-BGTR 在以下 family 获得预注册 P95 最优：" + ", ".join(ma_protected) + "。" if ma_protected else "MA-BGTR 没有在任何单独 family 获得最低 P95；其价值若存在，应来自跨 family 风险折中，而不是单 family 最优。"),
        "层级 gate 的独特作用应只在数据支持的 family 保护、static sanity 或 robust rejection 上表述，不能由架构复杂度本身推导。",
        "",
        "## 17. MA-BGTR 不如简单策略的条件",
        "",
        ("；".join(ma_lost) if ma_lost else "没有 family 的 P95 被简单策略严格改善。"),
        "这说明 validation residual、模型复杂度和导航误差之间仍存在错位；门控不能保证逐 run 或逐 family 最优。",
        "",
        "## 18. 是否回答“为什么需要层级门控”",
        "",
        decision_explanation,
        "若结论为 partial support，正文应把层级门控描述为风险管理机制而非普遍精度优势；若结论为 does-not-support，则必须降低模型选择必要性的贡献强度。",
        "",
        "## 19. 预注册结论",
        "",
        f"**{decision}**",
        "",
        "判定明细：",
        "",
        "```json",
        json.dumps(clean_json(decision_audit), ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
        "## 20. 对论文主文、补充材料与贡献表述的建议",
        "",
        "- 主文：报告 60-run observable-only 主汇总、H0--H5 风险差异、MA-BGTR 与最佳简单策略的配对图，以及预注册结论。",
        "- 补充材料：放完整模型分布、TECH18 9 场景矩阵、全部信息准则敏感性、失败清单和 UrbanNav stress。",
        "- 贡献措辞：只声称 evidence-gated model adaptation 在特定 family/risk 条件下提供保护；不得声称全面优于简单评分、信息准则或固定模型。",
        "- AICc/BIC：严格结果仅限 M0--M4；full-pool pseudo 结果必须保留 `pseudo` 标签。",
        "- UrbanNav：继续作为模型外限制分析，不作为真实 LEO 动态外场性能。",
        "",
        "## 21. 是否继续失败检测/拒绝机制",
        "",
        f"建议继续研究显式 failure detection / abstention，但不得在本任务回调 gate。MA-BGTR 的 >1 km rate 为 {100*ma['error_gt_1000m_rate']:.2f}%，P95 oracle regret 为 {ma['p95_regret_m']:.3f} m；这些量说明只输出一个模型标签仍不足以覆盖 residual--position mismatch。后续机制应在独立协议中预注册，并保持 evaluation truth 不进入在线决策。",
        "",
        "## 22. 保护审计与本地文件索引",
        "",
        f"保护审计总状态：`{str(protection['pass']).lower()}`；selector/freeze policy 未变化：`{str(protection['selector_hash_unchanged'] and protection['freeze_policy_hash_unchanged']).lower()}`；solver rerun：0。",
        "",
        f"已生成 {len(figure_stems)} 张图，每张同时有 PDF 与 300 dpi PNG：" + ", ".join(figure_stems) + "。",
        "",
        "下表列出报告生成前的全部本地详细文件（总报告自身因自引用哈希问题不列入）。",
        "",
        markdown_table(index),
        "",
        "## 审查结论边界",
        "",
        "本报告是冻结候选输出上的选择策略审计，不是新的 solver 实验。Post-selection oracle 仅用于误差上界；任何策略的好坏都不能反向修改 MA-BGTR gate，也不能用来筛除失败场景。未生成 review ZIP。",
        "",
    ]
    return "\n".join(lines)


def run_analysis() -> tuple[str, int]:
    configure_logging()
    protocol = verify_protocol()
    for name, artifact in protocol["prepared_artifacts"].items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"prepared artifact changed after protocol freeze: {name}")
    strategy_input = pd.read_csv(NO_TRUTH_PATH)
    if TRUTH_FORBIDDEN_COLUMNS.intersection(strategy_input.columns):
        raise RuntimeError("truth isolation failed at run entry")
    if strategy_input["group_id"].nunique() != 141:
        raise RuntimeError("expected 141 frozen candidate groups")

    logging.info("Executing eleven deployable strategies on frozen candidate rows only")
    raw = run_deployable_selections(strategy_input)
    raw.to_csv(PRE_TRUTH_SELECTION_PATH, index=False)
    logging.info("Truth-free model keys frozen before evaluation join: %s", PRE_TRUTH_SELECTION_PATH)
    selections = evaluate_selection_keys(raw, strategy_input)

    ma = selections.loc[selections["strategy_id"] == "STR-01"].copy()
    recovery = ma.groupby("evaluation_dataset")["matches_reference_selection"].agg(["sum", "count"])
    if int(recovery.loc["PAPER_EXP01_HOLDOUT", "sum"]) != 60 or int(recovery.loc["PAPER_EXP01_HOLDOUT", "count"]) != 60:
        raise RuntimeError("frozen MA-BGTR failed to recover PAPER-EXP01 Mode-A 60/60")
    if int(recovery.loc["TECH18_CORE", "sum"]) != 9 or int(recovery.loc["TECH18_CORE", "count"]) != 9:
        raise RuntimeError("frozen MA-BGTR failed to recover TECH18 9/9")

    selections.to_csv(DIRS["outputs"] / "PAPER_EXP03A_SELECTIONS_BY_STRATEGY.csv", index=False)
    aggregate = aggregate_holdout(selections)
    families = family_breakdown(selections)
    core = core_scenario_comparison(selections)
    urban = urban_stress_comparison(selections)
    catastrophe = catastrophic_risk(selections)
    statistical = paired_statistics(selections)
    disagree = disagreements(selections)
    failures = failure_cases(selections)

    aggregate.to_csv(DIRS["outputs"] / "PAPER_EXP03A_HOLDOUT_AGGREGATE.csv", index=False)
    families.to_csv(DIRS["outputs"] / "PAPER_EXP03A_HOLDOUT_FAMILY_BREAKDOWN.csv", index=False)
    core.to_csv(DIRS["outputs"] / "PAPER_EXP03A_CORE_SCENARIO_COMPARISON.csv", index=False)
    urban.to_csv(DIRS["outputs"] / "PAPER_EXP03A_URBANNAV_STRESS_COMPARISON.csv", index=False)
    catastrophe.to_csv(DIRS["outputs"] / "PAPER_EXP03A_CATASTROPHIC_RISK.csv", index=False)
    statistical.to_csv(DIRS["outputs"] / "PAPER_EXP03A_STATISTICAL_TESTS.csv", index=False)
    disagree.to_csv(DIRS["outputs"] / "PAPER_EXP03A_STRATEGY_DISAGREEMENTS.csv", index=False)
    failures.to_csv(DIRS["outputs"] / "PAPER_EXP03A_FAILURE_CASES.csv", index=False)

    decision, decision_audit = decide_support(aggregate, families)
    simple = aggregate.loc[aggregate["strategy_id"] != "STR-01"].sort_values(
        ["p95_position_error_m", "error_gt_1000m_rate", "median_position_error_m", "selection_coverage", "strategy_id"],
        ascending=[True, True, True, False, True],
    )
    best_simple_id = str(simple.iloc[0]["strategy_id"])
    figure_stems = generate_figures(selections, aggregate, families, core, urban, best_simple_id)

    paths = source_paths()
    protection = finalize_protection(paths)
    if not protection["pass"]:
        changed = [name for name, value in protection["critical_file_unchanged"].items() if not value]
        changed += [name for name, value in protection["protected_tree_unchanged"].items() if not value]
        raise RuntimeError(f"protected input/source changed: {changed}")

    recovery_dict = {
        dataset: {"matched": int(row["sum"]), "runs": int(row["count"]), "rate": float(row["sum"] / row["count"])}
        for dataset, row in recovery.iterrows()
    }
    metrics = {
        "task_name": "PAPER-EXP03A",
        "decision": decision,
        "protocol_sha256": PROTOCOL_HASH_PATH.read_text(encoding="ascii").split()[0],
        "primary_holdout_runs": 60,
        "solver_rerun": 0,
        "new_candidate_estimation": 0,
        "frozen_candidate_rows_only": True,
        "candidate_rows": {"PAPER_EXP01_HOLDOUT": 900, "TECH18_CORE": 135, "URBANNAV_EXP02A": 1080},
        "strategy_count_deployable": len(STRATEGIES),
        "selector_recovery": recovery_dict,
        "urban_recovery_limitation": {
            "missing_archival_selector_fields": ["position_cov_sqrt_trace_m", "position_bootstrap_spread_m", "information_retention_ratio", "geometry_preserved_info_retention", "retention_trace_ratio", "rank_loss", "tail_ratio", "mad_ratio", "base_model_for_refine"],
            "source_replay_matches": int(recovery.loc["URBANNAV_EXP02A", "sum"]),
            "reference_runs": int(recovery.loc["URBANNAV_EXP02A", "count"]),
            "old_selected_models_not_used_to_impute_missing_evidence": True,
        },
        "best_simple_baseline_by_preregistered_reporting_rule": best_simple_id,
        "holdout_aggregate": aggregate.to_dict(orient="records"),
        "decision_audit": decision_audit,
        "truth_used_for_deployable_selection": False,
        "oracle_used_for_deployable_selection": False,
        "protection_audit_pass": protection["pass"],
        "figures": figure_stems,
        "review_packet_generated": False,
    }
    write_json(DIRS["outputs"] / "PAPER_EXP03A_METRICS.json", metrics)

    logging.info("Analysis complete; decision=%s; preparing self-contained report", decision)
    logging.shutdown()
    index = local_file_index()
    index.to_csv(DIRS["outputs"] / "PAPER_EXP03A_LOCAL_FILE_INDEX.csv", index=False)
    report = report_text(
        protocol,
        selections,
        aggregate,
        families,
        core,
        urban,
        catastrophe,
        statistical,
        decision,
        decision_audit,
        protection,
        best_simple_id,
        index,
        figure_stems,
    )
    (OUT / "PAPER_EXP03A_TOTAL_REPORT.md").write_text(report, encoding="utf-8")
    return decision, len(selections.loc[(selections["evaluation_dataset"] == "PAPER_EXP01_HOLDOUT") & (selections["strategy_id"] == "STR-01")])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "all"])
    args = parser.parse_args()
    if args.phase in {"prepare", "all"}:
        prepare()
    if args.phase in {"run", "all"}:
        decision, runs = run_analysis()
        print(json.dumps({"status": "completed", "decision": decision, "primary_holdout_runs": runs, "solver_rerun": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
