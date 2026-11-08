from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

from baseline_common import *


FORBIDDEN = {
    "final_position_error_m", "mean_trajectory_position_error_m", "truth", "truth_position",
    "oracle", "best_candidate_by_truth", "capped_loss", "method_performance", "risk_score",
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def choose(group: pd.DataFrame, columns: list[str], ascending: list[bool] | None = None) -> tuple[pd.Series | None, int]:
    if group.empty:
        return None, 0
    ascending = ascending or [True] * len(columns)
    ordered = group.sort_values(columns, ascending=ascending, kind="mergesort")
    first = ordered.iloc[0]
    primary = columns[0]
    tie_count = int(np.sum(ordered[primary].to_numpy() == first[primary]))
    return first, tie_count


def selection_row(meta: pd.Series, method: str, subset: str, selected: pd.Series | None, metric: str, ties: int, source_hash: str, reason: str = "") -> dict:
    return {
        "batch_id": str(meta["batch_id"]),
        "group_id": str(meta["group_id"]),
        "scenario_id": str(meta["scenario_id"]),
        "scenario_family": str(meta["scenario_family"]),
        "seed_block_id": str(meta["seed_block_id"]),
        "method_id": method,
        "candidate_subset": subset,
        "selected_candidate": "" if selected is None else str(selected["candidate_model"]),
        "status": "explicit_failure" if selected is None else "selected",
        "metric_value": "" if selected is None or metric not in selected else float(selected[metric]),
        "tie_count": ties,
        "failure_reason": reason if selected is None else "",
        "source_hash": source_hash,
    }


def main() -> None:
    protocol_lock = BASELINE_ROOT / "protocol" / "PAPER_FLAGSHIP_A_ROUTE_BASELINE_PROTOCOL_LOCK.json"
    protocol_path = BASELINE_ROOT / "protocol" / "PAPER_FLAGSHIP_A_ROUTE_BASELINE_EXECUTION_PROTOCOL.json"
    registry_path = BASELINE_ROOT / "partition" / "PAPER_FLAGSHIP_A_ROUTE_HOLDOUT_GROUP_REGISTRY.csv"
    lock = json.loads(protocol_lock.read_text(encoding="utf-8"))
    if not lock.get("locked") or sha256_file(protocol_path) != lock.get("protocol_sha256"):
        raise RuntimeError("protocol lock failed")
    if sha256_file(registry_path) != lock.get("group_registry_sha256"):
        raise RuntimeError("group registry changed")

    candidates = pd.read_csv(TRUTH_FREE, keep_default_na=False)
    overlap = sorted(FORBIDDEN.intersection(candidates.columns))
    if overlap:
        raise RuntimeError(f"truth/evaluation fields in selection worker: {overlap}")
    candidates = candidates.loc[candidates["evaluation_dataset"].astype(str) == "PAPER_EXP01_HOLDOUT"].copy()
    registry = pd.read_csv(registry_path, keep_default_na=False)
    refs = pd.read_csv(REFERENCES, keep_default_na=False)
    refs = refs.loc[refs["evaluation_dataset"].astype(str) == "PAPER_EXP01_HOLDOUT"].copy()
    if len(candidates) != 900 or candidates["group_id"].nunique() != 60 or len(registry) != 60 or len(refs) != 60:
        raise RuntimeError("candidate/reference cardinality failed")

    numeric_cols = ["full_residual_rmse_mps", "raw_validation_rmse_mps", "trimmed_validation_rmse_mps", "inlier_validation_rmse_mps", "observation_count", "model_dof"]
    for col in numeric_cols:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates["eligible"] = as_bool(candidates["selectable"])
    candidates["candidate_order"] = candidates["candidate_model"].map(model_order)
    meta_map = registry.set_index("group_id")
    candidates["sigma0_mps"] = candidates["group_id"].map(meta_map["sigma0_mps"].astype(float))
    candidates["nu"] = candidates["observation_count"] - candidates["model_dof"]
    candidates["chi2"] = candidates["observation_count"] * np.square(candidates["full_residual_rmse_mps"]) / np.square(candidates["sigma0_mps"])
    candidates["reduced_chi2"] = candidates["chi2"] / candidates["nu"]
    candidates["consistency_deviation"] = np.abs(candidates["reduced_chi2"] - 1.0)
    candidates["AIC"] = candidates["chi2"] + 2.0 * candidates["model_dof"]
    denom = candidates["observation_count"] - candidates["model_dof"] - 1.0
    candidates["AICc"] = candidates["AIC"] + 2.0 * candidates["model_dof"] * (candidates["model_dof"] + 1.0) / denom
    candidates.loc[denom <= 0, "AICc"] = np.nan
    candidates["BIC"] = candidates["chi2"] + candidates["model_dof"] * np.log(candidates["observation_count"])
    if (candidates["nu"] <= 0).any() or not np.isfinite(candidates["sigma0_mps"]).all() or (candidates["sigma0_mps"] <= 0).any():
        raise RuntimeError("whitened residual identity failed")

    sys.path.insert(0, str(SELECTOR_SOURCE.parent))
    from leo_positioning.ma_bgtr_v71_solver import select_group_v71

    source_hash = sha256_text(sha256_file(TRUTH_FREE) + sha256_file(SCENARIOS) + sha256_file(SELECTOR_SOURCE / "ma_bgtr_v71_solver.py"))
    rows: list[dict] = []
    chi_rows: list[dict] = []
    reference_map = refs.set_index("group_id")["reference_selected_model"].astype(str).to_dict()
    recoveries = 0
    for group_id in sorted(candidates["group_id"].astype(str).unique()):
        group = candidates.loc[candidates["group_id"].astype(str) == group_id].copy()
        meta = meta_map.loc[group_id]
        eligible = group.loc[group["eligible"]].copy()
        finite_raw = eligible.loc[np.isfinite(eligible["full_residual_rmse_mps"])].copy()
        selected, ties = choose(finite_raw, ["full_residual_rmse_mps", "candidate_order"])
        rows.append(selection_row(meta, "B0", "M0-M14", selected, "full_residual_rmse_mps", ties, source_hash, "no eligible finite raw residual"))

        finite_b1 = eligible.loc[np.isfinite(eligible["reduced_chi2"]) & (eligible["nu"] > 0)].copy()
        selected, ties = choose(finite_b1, ["reduced_chi2", "candidate_order"])
        rows.append(selection_row(meta, "B1", "M0-M14", selected, "reduced_chi2", ties, source_hash, "no eligible finite reduced chi-square"))

        finite_b2 = eligible.loc[np.isfinite(eligible["raw_validation_rmse_mps"])].copy()
        selected, ties = choose(finite_b2, ["raw_validation_rmse_mps", "candidate_order"])
        rows.append(selection_row(meta, "B2", "M0-M14", selected, "raw_validation_rmse_mps", ties, source_hash, "no comparable held-out RMSE"))

        finite_b3 = eligible.loc[np.isfinite(eligible["reduced_chi2"])].copy()
        selected, ties = choose(finite_b3, ["consistency_deviation", "reduced_chi2", "model_dof", "candidate_order"])
        rows.append(selection_row(meta, "B3", "M0-M14", selected, "consistency_deviation", ties, source_hash, "no eligible finite residual consistency statistic"))

        compatible = eligible.loc[eligible["candidate_order"].between(0, 4)].copy()
        for method in ["AIC", "AICc", "BIC"]:
            valid = compatible.loc[np.isfinite(compatible[method])].copy()
            selected, ties = choose(valid, [method, "model_dof", "candidate_order"])
            rows.append(selection_row(meta, method, "M0-M4", selected, method, ties, source_hash, f"no eligible finite {method}"))

        ma_selected, _family, _bias, _robust, ma_reason, low_quality = select_group_v71(group.copy())
        if ma_selected is None:
            rows.append(selection_row(meta, "B8_MA_BGTR_FULL", "M0-M14", None, "", 0, source_hash, "frozen selector returned no selection"))
        else:
            rows.append(selection_row(meta, "B8_MA_BGTR_FULL", "M0-M14", ma_selected, "", 1, source_hash))
            if str(ma_selected["candidate_model"]) == reference_map[group_id]:
                recoveries += 1

        restricted_group = group.loc[group["candidate_order"].between(0, 4)].copy()
        restricted_selected, _family, _bias, _robust, restricted_reason, restricted_low_quality = select_group_v71(restricted_group)
        if restricted_selected is None:
            rows.append(selection_row(meta, "B9_MA_BGTR_M0_M4", "M0-M4", None, "", 0, source_hash, "restricted selector returned no selection"))
        else:
            rows.append(selection_row(meta, "B9_MA_BGTR_M0_M4", "M0-M4", restricted_selected, "", 1, source_hash))

        if str(meta["scenario_family"]) in {"H0_static_holdout", "H1_dynamic_holdout", "H2_bias_drift_holdout", "H5_no_bias_dynamic_holdout"}:
            for _, row in compatible.iterrows():
                nu = int(row["nu"])
                lower = float(chi2_dist.ppf(0.025, nu))
                upper = float(chi2_dist.ppf(0.975, nu))
                statistic = float(row["chi2"])
                chi_rows.append({
                    "batch_id": str(meta["batch_id"]), "group_id": group_id,
                    "scenario_family": str(meta["scenario_family"]), "candidate": str(row["candidate_model"]),
                    "nu": nu, "chi2_statistic": statistic, "alpha": 0.05,
                    "acceptance_lower": lower, "acceptance_upper": upper,
                    "accepted": lower <= statistic <= upper,
                    "diagnostic_only": True,
                })

    if recoveries != 60:
        raise RuntimeError(f"MA-BGTR reproduction failed: {recoveries}/60")
    selections = pd.DataFrame(rows).sort_values(["group_id", "method_id"], kind="stable")
    if len(selections) != 540 or selections.groupby("method_id").size().to_dict() != {m: 60 for m in METHODS}:
        raise RuntimeError("selection store cardinality failed")
    selection_path = BASELINE_ROOT / "results" / "PAPER_FLAGSHIP_A_ROUTE_BASELINE_SELECTION_STORE.csv"
    selections.to_csv(selection_path, index=False, lineterminator="\n")
    chi_path = BASELINE_ROOT / "results" / "PAPER_FLAGSHIP_A_ROUTE_FORMAL_CHI_SQUARE_DIAGNOSTIC.csv"
    pd.DataFrame(chi_rows).to_csv(chi_path, index=False, lineterminator="\n")
    selection_lock = {
        "created_at": now_z(),
        "protocol_sha256": sha256_file(protocol_path),
        "selection_worker_sha256": sha256_file(WORK / "selection_worker.py"),
        "truth_free_input_sha256": sha256_file(TRUTH_FREE),
        "selection_store_sha256": sha256_file(selection_path),
        "selection_rows": len(selections),
        "deployable_methods": METHODS,
        "ma_bgtr_reference_matches": recoveries,
        "truth_value_reads": 0,
        "truth_fields_loaded": 0,
        "candidate_solver_calls": 0,
        "locked_before_evaluation": True,
    }
    lock_path = BASELINE_ROOT / "results" / "PAPER_FLAGSHIP_A_ROUTE_BASELINE_SELECTION_LOCK.json"
    write_json(lock_path, selection_lock)
    access = {
        "worker": "selection_worker", "created_at": now_z(),
        "files_read": [str(TRUTH_FREE), str(REFERENCES), str(registry_path), str(protocol_path), str(SELECTOR_SOURCE / "ma_bgtr_v71_solver.py")],
        "truth_lookup_read": False, "standardized_truth_columns_read": False,
        "risk_files_read": False, "truth_fields_loaded": 0, "pass": True,
    }
    write_json(BASELINE_ROOT / "protection" / "PAPER_FLAGSHIP_A_ROUTE_SELECTION_WORKER_ACCESS_AUDIT.json", access)
    print(json.dumps({"status": "selection_locked", "rows": len(selections), "ma_recovery": recoveries, "selection_sha256": selection_lock["selection_store_sha256"]}))


if __name__ == "__main__":
    main()
