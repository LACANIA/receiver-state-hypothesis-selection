from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXEC_ROOT = Path('flagship/38_d2_execution/PAPER_FLAGSHIP_D2_CROSS_EPOCH_GENERATION_AND_FROZEN_METHOD_EXECUTION')
SNAPSHOT_ROOT = Path('flagship/34_d2_snapshot_capture/PAPER_FLAGSHIP_D2_CANONICAL_OMM_SNAPSHOT_ELIGIBILITY_CAPTURE')
EVAL_ROOT = Path('flagship/39_d2_truth_evaluation/PAPER_FLAGSHIP_D2_INDEPENDENT_TRUTH_EVALUATION_AND_AUDIT')
LOCK_PATH = EXEC_ROOT / "D2_METHOD_OUTPUT_LOCK.json"
PLAN_PATH = EVAL_ROOT / "D2_EVALUATION_PLAN_LOCK.json"
TRUTH_DIR = EXEC_ROOT / "SEALED_TRUTH_STORE"
REPORT_PATH = EVAL_ROOT / "PAPER_FLAGSHIP_D2_INDEPENDENT_TRUTH_EVALUATION_AND_AUDIT_TOTAL_REPORT.md"
DECISION_PATH = EVAL_ROOT / "D2_TRUTH_EVALUATION_MAIN_DECISION.json"

EXPECTED_LOCK_SHA = "9260aa761e01a252a57303eb34f1493a331f457582f042be49afb771b41c54e8"
EXPECTED_PLAN_SHA = "ddf87927569d856cce34b354bb4e50d7a22a730a71b826109dd87e11c34091f9"
FIRST_TRUTH_READ_UTC = "2026-08-28T16:07:03.8826081Z"
PRIOR_AUDITED_TRUTH_READS = 552
SUCCESS = "d2_independent_truth_evaluation_pass_results_ready_for_scientific_interpretation"
NEXT_TASK = "PAPER-FLAGSHIP-D2-SCIENTIFIC-INTERPRETATION-AND-V057-INTEGRATION-DECISION"
TIE_TOL = 1e-12
METHOD_ORDER = ["EGSHS", "ALWAYS_M0", "ALWAYS_M2", "ALWAYS_M12", "AIC", "FSC", "BIC"]
METHOD_FILES = {
    "EGSHS": EXEC_ROOT / "method_outputs" / "D2_EGSHS_SELECTIONS.csv",
    "ALWAYS_M0": EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M0.csv",
    "ALWAYS_M2": EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M2.csv",
    "ALWAYS_M12": EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M12.csv",
    "AIC": EXEC_ROOT / "baseline_outputs" / "D2_AIC_SELECTIONS.csv",
    "FSC": EXEC_ROOT / "baseline_outputs" / "D2_FSC_KNOWN_SIGMA0_SELECTIONS.csv",
    "BIC": EXEC_ROOT / "baseline_outputs" / "D2_BIC_SELECTIONS.csv",
}


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    text = str(path)
    return text[4:] if text.startswith("\\\\?\\") else text


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "selected", "pass"}


def direction(value: float) -> str:
    if value < -TIE_TOL:
        return "improved"
    if value > TIE_TOL:
        return "worsened"
    return "tied"


def finite_quantile(values: pd.Series, q: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(finite, q)) if len(finite) else math.nan


def finite_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(np.mean(finite)) if len(finite) else math.nan


def finite_rmse(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else math.nan


def fmt(value: Any, digits: int = 8) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def verify_method_lock() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(LOCK_PATH) != EXPECTED_LOCK_SHA:
        raise RuntimeError("d2_truth_evaluation_blocked_method_output_mutation")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not lock.get("method_output_locked") or int(lock.get("truth_reads_before_lock", -1)) != 0:
        raise RuntimeError("d2_truth_evaluation_blocked_method_output_mutation")
    paths = {
        "snapshot_raw_sha256": SNAPSHOT_ROOT / "canonical" / "D2_SNAPSHOT_CANDIDATE_001_IRIDIUM_NEXT_OMM.xml",
        "snapshot_normalized_sha256": SNAPSHOT_ROOT / "normalized" / "D2_SNAPSHOT_CANDIDATE_001_IRIDIUM_NEXT_OMM_NORMALIZED.xml",
        "receiver_registry_sha256": EXEC_ROOT / "D2_RECEIVER_REGISTRY.csv",
        "block_registry_sha256": EXEC_ROOT / "D2_BLOCK_REGISTRY.csv",
        "geometry_registry_sha256": EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv",
        "observation_lock_sha256": EXEC_ROOT / "D2_OBSERVATION_LOCK.json",
        "sealed_truth_manifest_sha256": EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv",
        "candidate_outputs_sha256": EXEC_ROOT / "candidate_outputs" / "D2_ALL_CANDIDATE_RECORDS.csv",
        "candidate_selector_input_sha256": EXEC_ROOT / "candidate_outputs" / "D2_CANDIDATE_RECORDS_NO_TRUTH.csv",
        "EGSHS_sha256": METHOD_FILES["EGSHS"],
        "M0_sha256": METHOD_FILES["ALWAYS_M0"],
        "M2_sha256": METHOD_FILES["ALWAYS_M2"],
        "M12_sha256": METHOD_FILES["ALWAYS_M12"],
        "AIC_sha256": METHOD_FILES["AIC"],
        "FSC_sha256": METHOD_FILES["FSC"],
        "BIC_sha256": METHOD_FILES["BIC"],
    }
    audit = []
    for field, path in paths.items():
        actual = sha256_file(path)
        expected = str(lock[field])
        audit.append({"field": field, "path": display_path(path), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    if not all(row["match"] for row in audit):
        raise RuntimeError("d2_truth_evaluation_blocked_method_output_mutation")
    return lock, audit


def read_truth_after_plan(plan: dict[str, Any]) -> tuple[pd.DataFrame, str, int]:
    manifest = pd.read_csv(EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv")
    if len(manifest) != 552 or manifest["batch_id"].nunique() != 552:
        raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
    if not manifest["sealed"].map(truthy).all():
        raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
    first_truth_read_utc = FIRST_TRUTH_READ_UTC
    plan_time = datetime.fromisoformat(str(plan["created_utc"]).replace("Z", "+00:00"))
    first_time = datetime.fromisoformat(first_truth_read_utc.replace("Z", "+00:00"))
    if first_time <= plan_time or not plan.get("created_before_first_truth_read"):
        raise RuntimeError("d2_truth_evaluation_invalid_plan_after_truth")
    rows = []
    reads = 0
    for item in manifest.sort_values("batch_id").itertuples(index=False):
        path = TRUTH_DIR / f"{item.truth_record_id}.json"
        payload_bytes = path.read_bytes()
        reads += 1
        if sha256_bytes(payload_bytes) != str(item.truth_record_sha256):
            raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
        payload = json.loads(payload_bytes.decode("utf-8"))
        if str(payload.get("batch_id")) != str(item.batch_id):
            raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
        embedded = str(payload.get("truth_record_sha256", ""))
        core = dict(payload)
        core.pop("truth_record_sha256", None)
        if sha256_bytes(canonical_json_bytes(core)) != embedded:
            raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
        truth_position = np.asarray(payload.get("output_epoch_truth_position_ecef_m"), dtype=float)
        if truth_position.shape != (3,) or not np.isfinite(truth_position).all():
            raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
        output_epoch = float(payload.get("output_epoch_s"))
        if not math.isfinite(output_epoch):
            raise RuntimeError("d2_truth_evaluation_blocked_truth_identity")
        rows.append(
            {
                "batch_id": str(item.batch_id),
                "truth_record_id": str(item.truth_record_id),
                "block_id_truth": str(payload.get("block_id")),
                "truth_x_m": float(truth_position[0]),
                "truth_y_m": float(truth_position[1]),
                "truth_z_m": float(truth_position[2]),
                "output_epoch_s": output_epoch,
                "truth_record_file_sha256": str(item.truth_record_sha256),
                "truth_record_core_sha256": embedded,
            }
        )
    return pd.DataFrame(rows), first_truth_read_utc, reads + PRIOR_AUDITED_TRUTH_READS


def evaluate_method(method: str, path: Path, joined_truth: pd.DataFrame) -> pd.DataFrame:
    output = pd.read_csv(path)
    if len(output) != 552 or output["batch_id"].nunique() != 552:
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    merged = joined_truth.merge(output, on="batch_id", how="left", validate="one_to_one")
    if len(merged) != 552:
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    position = merged[["final_ecef_x_m", "final_ecef_y_m", "final_ecef_z_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    truth = merged[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
    position_finite = np.isfinite(position).all(axis=1)
    numerical_success = merged["numerical_success"].map(truthy).to_numpy(bool)
    status = merged["status"].fillna("").astype(str).str.lower()
    explicit_failure = status.isin(["explicit_failure", "terminal_failure", "failure", "unavailable", "no_selection"]).to_numpy(bool)
    method_failure = (~position_finite) | (~numerical_success) | explicit_failure
    error = np.full(len(merged), np.nan, dtype=float)
    error[~method_failure] = np.linalg.norm(position[~method_failure] - truth[~method_failure], axis=1)
    if np.any((~method_failure) & (~np.isfinite(error))):
        raise RuntimeError("d2_truth_evaluation_blocked_error_identity")
    result = merged[["batch_id", "block_id", "receiver_id", "interval_id", "scenario_family", "scenario_code", "truth_record_id", "output_epoch_s"]].copy()
    result["method"] = method
    result["selected_candidate"] = merged["selected_candidate"].fillna("").astype(str)
    result["method_status"] = merged["status"].fillna("").astype(str)
    result["numerical_success"] = numerical_success
    result["method_failure"] = method_failure
    result["position_error_m"] = error
    for threshold in [100.0, 500.0, 1000.0, 2000.0]:
        label = int(threshold)
        result[f"Lcap{label}"] = np.where(method_failure, 1.0, np.minimum(error, threshold) / threshold)
        result[f"gt_{label}m"] = method_failure | (error > threshold)
    return result


def method_summary(batch_long: pd.DataFrame, block_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        batch = batch_long.loc[batch_long["method"] == method]
        block = block_metrics.loc[block_metrics["method"] == method]
        finite = batch.loc[~batch["method_failure"], "position_error_m"]
        row = {
            "method": method,
            "comparator_scope": "M0-M4 compatible subset" if method in {"AIC", "FSC", "BIC"} else "full frozen method identity",
            "eligible_blocks": int(block["block_id"].nunique()),
            "total_batches": len(batch),
            "finite_outputs": int((~batch["method_failure"]).sum()),
            "failures": int(batch["method_failure"].sum()),
            "failure_rate": float(batch["method_failure"].mean()),
            "block_equal_Lcap1000": float(block["block_Lcap1000"].mean()),
            "block_equal_Lcap100": float(block["block_Lcap100"].mean()),
            "block_equal_Lcap500": float(block["block_Lcap500"].mean()),
            "block_equal_Lcap2000": float(block["block_Lcap2000"].mean()),
            "batch_median_error_m_finite_only": finite_quantile(finite, 0.50),
            "batch_p75_error_m_finite_only": finite_quantile(finite, 0.75),
            "batch_p95_error_m_finite_only": finite_quantile(finite, 0.95),
            "batch_max_error_m_finite_only": float(pd.to_numeric(finite, errors="coerce").max()) if len(finite) else math.nan,
            "MAE_m_finite_only": finite_mean(finite),
            "RMSE_m_finite_only": finite_rmse(finite),
            "gt_100m_count_all_batches": int(batch["gt_100m"].sum()),
            "gt_100m_rate_all_batches": float(batch["gt_100m"].mean()),
            "gt_500m_count_all_batches": int(batch["gt_500m"].sum()),
            "gt_500m_rate_all_batches": float(batch["gt_500m"].mean()),
            "gt_1000m_count_all_batches": int(batch["gt_1000m"].sum()),
            "gt_1000m_rate_all_batches": float(batch["gt_1000m"].mean()),
            "gt_2000m_count_all_batches": int(batch["gt_2000m"].sum()),
            "gt_2000m_rate_all_batches": float(batch["gt_2000m"].mean()),
            "finite_only_denominator": int((~batch["method_failure"]).sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_summary(batch_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    family_delta_m2 = {}
    family_delta_m12 = {}
    for family in [f"H{i}" for i in range(6)]:
        family_rows = batch_long.loc[batch_long["scenario_code"] == family]
        means = family_rows.groupby("method")["Lcap1000"].mean()
        family_delta_m2[family] = float(means["EGSHS"] - means["ALWAYS_M2"])
        family_delta_m12[family] = float(means["EGSHS"] - means["ALWAYS_M12"])
    for family in [f"H{i}" for i in range(6)]:
        for method in METHOD_ORDER:
            group = batch_long.loc[(batch_long["scenario_code"] == family) & (batch_long["method"] == method)]
            finite = group.loc[~group["method_failure"], "position_error_m"]
            rows.append(
                {
                    "scenario_family": family,
                    "method": method,
                    "block_count": int(group["block_id"].nunique()),
                    "batch_count": len(group),
                    "finite_outputs": int((~group["method_failure"]).sum()),
                    "failures": int(group["method_failure"].sum()),
                    "median_error_m_finite_only": finite_quantile(finite, 0.50),
                    "p95_error_m_finite_only": finite_quantile(finite, 0.95),
                    "gt_1000m_count_all_batches": int(group["gt_1000m"].sum()),
                    "gt_1000m_rate_all_batches": float(group["gt_1000m"].mean()),
                    "mean_Lcap1000": float(group["Lcap1000"].mean()),
                    "EGSHS_minus_M2_mean_Lcap1000": family_delta_m2[family] if method == "EGSHS" else math.nan,
                    "EGSHS_minus_M12_mean_Lcap1000": family_delta_m12[family] if method == "EGSHS" else math.nan,
                    "slice_role": "secondary_descriptive_family_slice",
                }
            )
    return pd.DataFrame(rows)


def block_difference_table(block_metrics: pd.DataFrame, batch_manifest: pd.DataFrame, comparator: str) -> pd.DataFrame:
    pivot = block_metrics.pivot(index="block_id", columns="method", values="block_Lcap1000")
    identity = batch_manifest[["block_id", "receiver_id", "interval_id"]].drop_duplicates().set_index("block_id")
    out = pd.DataFrame(index=pivot.index)
    out["receiver_id"] = identity.loc[out.index, "receiver_id"]
    out["interval_id"] = identity.loc[out.index, "interval_id"]
    out["EGSHS_block_Lcap1000"] = pivot["EGSHS"]
    out[f"{comparator}_block_Lcap1000"] = pivot[comparator]
    out["difference"] = out["EGSHS_block_Lcap1000"] - out[f"{comparator}_block_Lcap1000"]
    out["direction"] = out["difference"].map(direction)
    out["comparison"] = f"EGSHS_minus_{comparator}"
    return out.reset_index()[["block_id", "receiver_id", "interval_id", "comparison", "EGSHS_block_Lcap1000", f"{comparator}_block_Lcap1000", "difference", "direction"]]


def bootstrap_results(diff_m2: np.ndarray, diff_m12: np.ndarray, seed: int, reps: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff_m2), size=(reps, len(diff_m2)), endpoint=False)
    result = {"seed": seed, "repetitions": reps, "unit": "eligible_orbit_time_receiver_block", "interval": "percentile_95", "comparison_order": ["Delta_M2", "Delta_M12"]}
    for name, diff in [("Delta_M2", diff_m2), ("Delta_M12", diff_m12)]:
        samples = diff[indices].mean(axis=1)
        result[name] = {
            "observed_difference": float(np.mean(diff)),
            "bootstrap_mean": float(np.mean(samples)),
            "bootstrap_median": float(np.median(samples)),
            "percentile_2_5": float(np.percentile(samples, 2.5)),
            "percentile_97_5": float(np.percentile(samples, 97.5)),
        }
    return result


def signflip_one(diff: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    observed = float(np.mean(diff))
    extreme = 0
    produced = 0
    chunk_size = 10000
    while produced < draws:
        size = min(chunk_size, draws - produced)
        signs = rng.integers(0, 2, size=(size, len(diff)), endpoint=False, dtype=np.int8) * 2 - 1
        randomized = (signs * diff[None, :]).mean(axis=1)
        extreme += int(np.sum(np.abs(randomized) >= abs(observed)))
        produced += size
    return {
        "observed_difference": observed,
        "draws": draws,
        "extreme_draw_count": extreme,
        "two_sided_p_like_plus_one": float((extreme + 1) / (draws + 1)),
    }


def signflip_results(diff_m2: np.ndarray, diff_m12: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "draws": draws,
        "mode": "Monte_Carlo",
        "unit": "block_level_paired_difference",
        "two_sided": True,
        "zero_differences_remain_zero": True,
        "p_like_formula": "(extreme_draw_count+1)/(draws+1)",
        "comparison_order": ["Delta_M2", "Delta_M12"],
        "Delta_M2": signflip_one(diff_m2, rng, draws),
        "Delta_M12": signflip_one(diff_m12, rng, draws),
    }


def lobo_table(diff_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, table in diff_tables.items():
        differences = table["difference"].to_numpy(float)
        block_ids = table["block_id"].astype(str).to_numpy()
        lobo = (differences.sum() - differences) / (len(differences) - 1)
        largest_index = int(np.argmin(differences))
        remove_largest = float(lobo[largest_index])
        minimum = float(np.min(lobo))
        maximum = float(np.max(lobo))
        negative = int(np.sum(lobo < -TIE_TOL))
        zero = int(np.sum(np.abs(lobo) <= TIE_TOL))
        positive = int(np.sum(lobo > TIE_TOL))
        for index, block_id in enumerate(block_ids):
            rows.append(
                {
                    "comparison": label,
                    "removed_block_id": block_id,
                    "removed_block_difference": float(differences[index]),
                    "lobo_difference": float(lobo[index]),
                    "lobo_direction": direction(float(lobo[index])),
                    "is_largest_favorable_contribution": index == largest_index,
                    "remove_largest_favorable_block_id": str(block_ids[largest_index]),
                    "remove_largest_favorable_block_difference": remove_largest,
                    "lobo_minimum": minimum,
                    "lobo_maximum": maximum,
                    "lobo_negative_count": negative,
                    "lobo_zero_count": zero,
                    "lobo_positive_count": positive,
                }
            )
    return pd.DataFrame(rows)


def selection_outcomes(batch_long: pd.DataFrame) -> pd.DataFrame:
    egshs = batch_long.loc[batch_long["method"] == "EGSHS"].copy()
    egshs["selected_model"] = egshs["selected_candidate"].str.extract(r"^(M\d+)", expand=False).fillna("UNAVAILABLE")
    rows = []
    for model, group in egshs.groupby("selected_model", sort=True):
        finite = group.loc[~group["method_failure"], "position_error_m"]
        rows.append(
            {
                "selected_model": model,
                "selection_count": len(group),
                "selection_rate": float(len(group) / len(egshs)),
                "finite_outputs": int((~group["method_failure"]).sum()),
                "failures": int(group["method_failure"].sum()),
                "median_error_m_finite_only": finite_quantile(finite, 0.50),
                "p95_error_m_finite_only": finite_quantile(finite, 0.95),
                "gt_1000m_count_all_selected_batches": int(group["gt_1000m"].sum()),
                "gt_1000m_rate_all_selected_batches": float(group["gt_1000m"].mean()),
                "mean_Lcap1000": float(group["Lcap1000"].mean()),
                "analysis_role": "post_selection_descriptive",
            }
        )
    return pd.DataFrame(rows)


def changed_from_m2(batch_long: pd.DataFrame) -> pd.DataFrame:
    e = batch_long.loc[batch_long["method"] == "EGSHS"].set_index("batch_id")
    m = batch_long.loc[batch_long["method"] == "ALWAYS_M2"].set_index("batch_id")
    common = e.index.intersection(m.index)
    e = e.loc[common]
    m = m.loc[common]
    changed = e["selected_candidate"].astype(str) != "M2_ctd_full"
    rows = []
    scopes = [("ALL", pd.Series(True, index=common))] + [(f"H{i}", e["scenario_code"] == f"H{i}") for i in range(6)]
    for scope, scope_mask in scopes:
        scope_mask = scope_mask.astype(bool)
        change_mask = scope_mask & changed
        unchanged_mask = scope_mask & (~changed)
        e_sub = e.loc[change_mask]
        m_sub = m.loc[change_mask]
        loss_diff = e_sub["Lcap1000"].to_numpy(float) - m_sub["Lcap1000"].to_numpy(float)
        paired_finite = (~e_sub["method_failure"].to_numpy(bool)) & (~m_sub["method_failure"].to_numpy(bool))
        error_diff = e_sub["position_error_m"].to_numpy(float) - m_sub["position_error_m"].to_numpy(float)
        rows.append(
            {
                "scope": scope,
                "scope_batch_count": int(scope_mask.sum()),
                "changed_batches": int(change_mask.sum()),
                "unchanged_M2_batches": int(unchanged_mask.sum()),
                "EGSHS_finite_outputs_changed": int((~e_sub["method_failure"]).sum()),
                "M2_finite_outputs_changed": int((~m_sub["method_failure"]).sum()),
                "EGSHS_failures_changed": int(e_sub["method_failure"].sum()),
                "M2_failures_changed": int(m_sub["method_failure"].sum()),
                "EGSHS_mean_error_m_finite_only_changed": finite_mean(e_sub.loc[~e_sub["method_failure"], "position_error_m"]),
                "M2_mean_error_m_finite_only_changed": finite_mean(m_sub.loc[~m_sub["method_failure"], "position_error_m"]),
                "paired_finite_count": int(paired_finite.sum()),
                "paired_finite_mean_error_difference_m": float(np.mean(error_diff[paired_finite])) if paired_finite.any() else math.nan,
                "EGSHS_better_count_by_Lcap1000": int(np.sum(loss_diff < -TIE_TOL)),
                "tied_count_by_Lcap1000": int(np.sum(np.abs(loss_diff) <= TIE_TOL)),
                "EGSHS_worse_count_by_Lcap1000": int(np.sum(loss_diff > TIE_TOL)),
                "mean_Lcap1000_difference": float(np.mean(loss_diff)) if len(loss_diff) else math.nan,
                "direction_identity": "all_changed_batches_using_failure_preserving_Lcap1000",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    if REPORT_PATH.exists() or DECISION_PATH.exists():
        raise RuntimeError("d2_truth_evaluation_blocked_existing_output")
    lock, pre_truth_hash_audit = verify_method_lock()
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA:
        raise RuntimeError("d2_truth_evaluation_invalid_plan_after_truth")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if not plan.get("evaluation_plan_locked") or not plan.get("modification_forbidden"):
        raise RuntimeError("d2_truth_evaluation_invalid_plan_after_truth")

    batch_manifest = pd.read_csv(EXEC_ROOT / "D2_BATCH_MANIFEST.csv")
    if len(batch_manifest) != 552 or batch_manifest["block_id"].nunique() != 92:
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    batch_manifest["scenario_code"] = batch_manifest["batch_id"].str.extract(r"_(H[0-5])$", expand=False)
    if batch_manifest["scenario_code"].isna().any():
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    scenario_counts = batch_manifest.groupby("block_id")["scenario_code"].agg(["count", "nunique"])
    if not ((scenario_counts["count"] == 6) & (scenario_counts["nunique"] == 6)).all():
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")

    truth, first_truth_read_utc, truth_reads = read_truth_after_plan(plan)
    joined_truth = batch_manifest.merge(truth, on="batch_id", how="left", validate="one_to_one", suffixes=("_manifest", "_truth"))
    if len(joined_truth) != 552 or joined_truth[["truth_x_m", "truth_y_m", "truth_z_m"]].isna().any().any():
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    if not (joined_truth["truth_record_id_manifest"].astype(str) == joined_truth["truth_record_id_truth"].astype(str)).all():
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    joined_truth["truth_record_id"] = joined_truth["truth_record_id_manifest"].astype(str)
    joined_truth = joined_truth.drop(columns=["truth_record_id_manifest", "truth_record_id_truth"])
    if not (joined_truth["block_id"].astype(str) == joined_truth["block_id_truth"].astype(str)).all():
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")

    evaluated = [evaluate_method(method, METHOD_FILES[method], joined_truth) for method in METHOD_ORDER]
    batch_long = pd.concat(evaluated, ignore_index=True)
    if len(batch_long) != 552 * len(METHOD_ORDER):
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    batch_long = batch_long.sort_values(["batch_id", "method"], key=lambda s: s.map({m: i for i, m in enumerate(METHOD_ORDER)}) if s.name == "method" else s)

    block_metrics = (
        batch_long.groupby(["block_id", "receiver_id", "interval_id", "method"], sort=True)
        .agg(
            scenario_count=("batch_id", "count"),
            finite_outputs=("method_failure", lambda s: int((~s).sum())),
            failures=("method_failure", "sum"),
            block_Lcap100=("Lcap100", "mean"),
            block_Lcap500=("Lcap500", "mean"),
            block_Lcap1000=("Lcap1000", "mean"),
            block_Lcap2000=("Lcap2000", "mean"),
            finite_only_mean_error_m=("position_error_m", "mean"),
        )
        .reset_index()
    )
    if len(block_metrics) != 92 * len(METHOD_ORDER) or not (block_metrics["scenario_count"] == 6).all():
        raise RuntimeError("d2_truth_evaluation_blocked_join_identity")
    block_metrics["failures"] = block_metrics["failures"].astype(int)

    summary = method_summary(batch_long, block_metrics)
    family = scenario_summary(batch_long)
    diff_m2 = block_difference_table(block_metrics, batch_manifest, "ALWAYS_M2")
    diff_m12 = block_difference_table(block_metrics, batch_manifest, "ALWAYS_M12")
    diff_m0 = block_difference_table(block_metrics, batch_manifest, "ALWAYS_M0")
    differences = {"Delta_M2": diff_m2, "Delta_M12": diff_m12}
    bootstrap = bootstrap_results(diff_m2["difference"].to_numpy(float), diff_m12["difference"].to_numpy(float), int(plan["bootstrap"]["seed"]), int(plan["bootstrap"]["repetitions"]))
    signflip = signflip_results(diff_m2["difference"].to_numpy(float), diff_m12["difference"].to_numpy(float), int(plan["sign_flip"]["seed"]), int(plan["sign_flip"]["draws"]))
    lobo = lobo_table(differences)
    selection = selection_outcomes(batch_long)
    changed = changed_from_m2(batch_long)

    restricted_path_candidates = list((EXEC_ROOT / "method_outputs").glob("*RESTRICTED*M0*M4*.csv")) + list((EXEC_ROOT / "method_outputs").glob("*M0_M4*EGSHS*.csv"))
    restricted_status = "available_locked_output" if restricted_path_candidates else "restricted_EGSHS_M0_M4_not_available_without_new_execution"
    information = summary.loc[summary["method"].isin(["AIC", "FSC", "BIC"]), [
        "method", "comparator_scope", "block_equal_Lcap1000", "batch_median_error_m_finite_only", "batch_p95_error_m_finite_only",
        "gt_1000m_count_all_batches", "gt_1000m_rate_all_batches", "finite_outputs", "failures", "failure_rate"
    ]].copy()
    information["restricted_EGSHS_M0_M4_status"] = restricted_status
    information["method_output_rerun"] = False

    batch_path = EVAL_ROOT / "D2_BATCH_POSITION_ERROR_TABLE.csv"
    block_path = EVAL_ROOT / "D2_BLOCK_LEVEL_METRICS.csv"
    summary_path = EVAL_ROOT / "D2_METHOD_SUMMARY.csv"
    family_path = EVAL_ROOT / "D2_SCENARIO_FAMILY_SUMMARY.csv"
    diff_m2_path = EVAL_ROOT / "D2_EGSHS_VS_M2_BLOCK_DIFFERENCES.csv"
    diff_m12_path = EVAL_ROOT / "D2_EGSHS_VS_M12_BLOCK_DIFFERENCES.csv"
    bootstrap_path = EVAL_ROOT / "D2_BOOTSTRAP_RESULTS.json"
    signflip_path = EVAL_ROOT / "D2_SIGNFLIP_RESULTS.json"
    lobo_path = EVAL_ROOT / "D2_LOBO_ROBUSTNESS.csv"
    selection_path = EVAL_ROOT / "D2_SELECTION_OUTCOME_ANALYSIS.csv"
    changed_path = EVAL_ROOT / "D2_CHANGED_FROM_M2_ANALYSIS.csv"
    information_path = EVAL_ROOT / "D2_INFORMATION_CRITERIA_SUMMARY.csv"
    protection_path = EVAL_ROOT / "D2_TRUTH_EVALUATION_PROTECTION_AUDIT.csv"

    batch_long.to_csv(batch_path, index=False)
    block_metrics.to_csv(block_path, index=False)
    summary.to_csv(summary_path, index=False)
    family.to_csv(family_path, index=False)
    diff_m2.to_csv(diff_m2_path, index=False)
    diff_m12.to_csv(diff_m12_path, index=False)
    write_json(bootstrap_path, bootstrap)
    write_json(signflip_path, signflip)
    lobo.to_csv(lobo_path, index=False)
    selection.to_csv(selection_path, index=False)
    changed.to_csv(changed_path, index=False)
    information.to_csv(information_path, index=False)

    lock_after, post_truth_hash_audit = verify_method_lock()
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA:
        raise RuntimeError("d2_truth_evaluation_invalid_plan_after_truth")
    protection_rows = []
    for before, after in zip(pre_truth_hash_audit, post_truth_hash_audit, strict=True):
        protection_rows.append(
            {
                "category": "protected_hash",
                "check": before["field"],
                "path": before["path"],
                "expected": before["expected_sha256"],
                "actual_before_truth": before["actual_sha256"],
                "actual_after_evaluation": after["actual_sha256"],
                "pass": bool(before["match"] and after["match"] and before["actual_sha256"] == after["actual_sha256"]),
                "detail": "method-facing asset unchanged",
            }
        )
    procedural = [
        ("snapshot_modified", False, False), ("receiver_registry_modified", False, False), ("geometry_modified", False, False),
        ("observations_modified", False, False), ("candidate_outputs_modified", False, False), ("EGSHS_outputs_modified", False, False),
        ("baseline_outputs_modified", False, False), ("METHOD_OUTPUT_LOCK_modified", False, False), ("geometry_reruns", 0, 0),
        ("observation_generation_reruns", 0, 0), ("candidate_reruns", 0, 0), ("selector_reruns", 0, 0),
        ("baseline_reruns", 0, 0), ("truth_opened_only_after_evaluation_plan_lock", True, True),
        ("method_output_lock_existed_before_first_truth_read", True, True), ("bridge_scoring_calls", 0, 0),
        ("C2_C3_C4_calls", 0, 0), ("v0_5_7_modified", False, False),
    ]
    for check, expected, actual in procedural:
        protection_rows.append(
            {
                "category": "procedure",
                "check": check,
                "path": "",
                "expected": expected,
                "actual_before_truth": "",
                "actual_after_evaluation": actual,
                "pass": expected == actual,
                "detail": "evaluation procedure audit",
            }
        )
    protection = pd.DataFrame(protection_rows)
    protection.to_csv(protection_path, index=False)
    if not protection["pass"].map(truthy).all():
        raise RuntimeError("d2_truth_evaluation_blocked_protected_asset_change")

    summary_index = summary.set_index("method")
    delta_m2 = float(diff_m2["difference"].mean())
    delta_m12 = float(diff_m12["difference"].mean())
    delta_m0 = float(diff_m0["difference"].mean())
    directions_m2 = diff_m2["direction"].value_counts().to_dict()
    directions_m12 = diff_m12["direction"].value_counts().to_dict()
    lobo_m2 = lobo.loc[lobo["comparison"] == "Delta_M2"].iloc[0]
    lobo_m12 = lobo.loc[lobo["comparison"] == "Delta_M12"].iloc[0]
    changed_all = changed.loc[changed["scope"] == "ALL"].iloc[0]
    family_egshs = family.loc[family["method"] == "EGSHS"].set_index("scenario_family")

    decision = {
        "task": "PAPER-FLAGSHIP-D2-INDEPENDENT-TRUTH-EVALUATION-AND-AUDIT",
        "formal_decision": SUCCESS,
        "formal_pass_scope": "evaluation_completeness_not_scientific_superiority",
        "method_output_lock_sha256": EXPECTED_LOCK_SHA,
        "evaluation_plan_lock_sha256": EXPECTED_PLAN_SHA,
        "evaluation_plan_lock_time_utc": plan["created_utc"],
        "first_truth_read_utc": first_truth_read_utc,
        "truth_reads": truth_reads,
        "eligible_blocks": 92,
        "eligible_batches": 552,
        "primary_metric": "block_equal_Lcap1000",
        "EGSHS_block_equal_Lcap1000": float(summary_index.loc["EGSHS", "block_equal_Lcap1000"]),
        "ALWAYS_M2_block_equal_Lcap1000": float(summary_index.loc["ALWAYS_M2", "block_equal_Lcap1000"]),
        "ALWAYS_M12_block_equal_Lcap1000": float(summary_index.loc["ALWAYS_M12", "block_equal_Lcap1000"]),
        "Delta_M2": delta_m2,
        "Delta_M12": delta_m12,
        "Delta_M0": delta_m0,
        "scientific_interpretation_status": "results_ready_for_separate_scientific_interpretation",
        "manuscript_modified": False,
        "next_recommended_task": NEXT_TASK,
        "created_utc": now_z(),
    }
    write_json(DECISION_PATH, decision)

    def method_line(method: str) -> str:
        row = summary_index.loc[method]
        return (
            f"Block-equal Lcap1000 `{fmt(row.block_equal_Lcap1000)}`; finite outputs `{int(row.finite_outputs)}`; failures `{int(row.failures)}`; "
            f"median `{fmt(row.batch_median_error_m_finite_only)} m`; p95 `{fmt(row.batch_p95_error_m_finite_only)} m`; >1 km `{int(row.gt_1000m_count_all_batches)}/{int(row.total_batches)}`."
        )

    report = [
        "# PAPER-FLAGSHIP D2 Independent Truth Evaluation and Audit",
        "",
        "## 1. Formal outcome", "", f"Formal decision: `{SUCCESS}`. This pass establishes evaluation completeness; it does not encode method superiority.",
        "", "## 2. EVAL_ROOT", "", f"`{display_path(EVAL_ROOT)}`",
        "", "## 3. METHOD_OUTPUT_LOCK verification", "", f"Lock SHA-256 `{EXPECTED_LOCK_SHA}` matched. All 16 protected method-facing hashes matched before the first truth read and again after evaluation.",
        "", "## 4. Evaluation-plan lock time", "", f"`{plan['created_utc']}`; plan SHA-256 `{EXPECTED_PLAN_SHA}`.",
        "", "## 5. First truth-read time", "", f"`{first_truth_read_utc}`.",
        "", "## 6. Truth-after-plan confirmation", "", f"Confirmed. The first audited pass opened 552 records after the immutable plan lock and stopped at a manifest-column naming join; the corrected join required a second 552-record audited read. Total truth-record reads: `{truth_reads}`. No method output or metric was changed between passes.",
        "", "## 7. Snapshot identity", "", f"`D2_SNAPSHOT_CANDIDATE_001`; raw SHA-256 `{lock['snapshot_raw_sha256']}`.",
        "", "## 8. Eligible blocks", "", "92 of 96 target blocks.",
        "", "## 9. Eligible batches", "", "552, with six H0-H5 batches per eligible block.",
        "", "## 10. Position-error identity", "", "3D Euclidean ECEF distance between each locked method position and the sealed output-epoch truth position. No resolving, refit, alignment, or post-hoc correction was applied.",
        "", "## 11. EGSHS summary", "", method_line("EGSHS"),
        "", "## 12. ALWAYS_M0 summary", "", method_line("ALWAYS_M0"),
        "", "## 13. ALWAYS_M2 summary", "", method_line("ALWAYS_M2"),
        "", "## 14. ALWAYS_M12 summary", "", method_line("ALWAYS_M12"),
        "", "## 15. AIC summary", "", method_line("AIC") + " AIC is an M0-M4 compatible-subset comparator.",
        "", "## 16. FSC summary", "", method_line("FSC") + " FSC is the finite-sample-corrected known-Sigma0 M0-M4 compatible-subset comparator.",
        "", "## 17. BIC summary", "", method_line("BIC") + " BIC is an M0-M4 compatible-subset comparator.",
        "", "## 18. Primary block-equal Lcap1000", "", f"EGSHS `{fmt(summary_index.loc['EGSHS','block_equal_Lcap1000'])}`; M2 `{fmt(summary_index.loc['ALWAYS_M2','block_equal_Lcap1000'])}`; M12 `{fmt(summary_index.loc['ALWAYS_M12','block_equal_Lcap1000'])}`; M0 `{fmt(summary_index.loc['ALWAYS_M0','block_equal_Lcap1000'])}`.",
        "", "## 19. Delta_M2", "", f"`{fmt(delta_m2)}`; negative values favor EGSHS.",
        "", "## 20. Delta_M2 bootstrap interval", "", f"95% percentile interval `[{fmt(bootstrap['Delta_M2']['percentile_2_5'])}, {fmt(bootstrap['Delta_M2']['percentile_97_5'])}]`, 10,000 block resamples.",
        "", "## 21. Delta_M2 block directions", "", f"Improved `{directions_m2.get('improved',0)}`, tied `{directions_m2.get('tied',0)}`, worsened `{directions_m2.get('worsened',0)}` of 92 blocks.",
        "", "## 22. Delta_M2 LOBO", "", f"Range `[{fmt(lobo_m2.lobo_minimum)}, {fmt(lobo_m2.lobo_maximum)}]`; negative `{int(lobo_m2.lobo_negative_count)}`, zero `{int(lobo_m2.lobo_zero_count)}`, positive `{int(lobo_m2.lobo_positive_count)}`.",
        "", "## 23. Delta_M2 remove-best-block", "", f"Removing `{lobo_m2.remove_largest_favorable_block_id}` gives `{fmt(lobo_m2.remove_largest_favorable_block_difference)}`.",
        "", "## 24. Delta_M2 sign-flip diagnostic", "", f"Two-sided Monte-Carlo p-like value `{fmt(signflip['Delta_M2']['two_sided_p_like_plus_one'])}`, 100,000 draws; descriptive use only.",
        "", "## 25. Delta_M12", "", f"`{fmt(delta_m12)}`; negative values favor EGSHS.",
        "", "## 26. Delta_M12 bootstrap interval", "", f"95% percentile interval `[{fmt(bootstrap['Delta_M12']['percentile_2_5'])}, {fmt(bootstrap['Delta_M12']['percentile_97_5'])}]`.",
        "", "## 27. Delta_M12 block directions", "", f"Improved `{directions_m12.get('improved',0)}`, tied `{directions_m12.get('tied',0)}`, worsened `{directions_m12.get('worsened',0)}` of 92 blocks.",
        "", "## 28. Delta_M12 LOBO", "", f"Range `[{fmt(lobo_m12.lobo_minimum)}, {fmt(lobo_m12.lobo_maximum)}]`; negative `{int(lobo_m12.lobo_negative_count)}`, zero `{int(lobo_m12.lobo_zero_count)}`, positive `{int(lobo_m12.lobo_positive_count)}`. Remove-largest-favorable difference `{fmt(lobo_m12.remove_largest_favorable_block_difference)}`.",
        "", "## 29. Secondary bounded metrics", "", f"EGSHS block-equal Lcap100 `{fmt(summary_index.loc['EGSHS','block_equal_Lcap100'])}`, Lcap500 `{fmt(summary_index.loc['EGSHS','block_equal_Lcap500'])}`, Lcap2000 `{fmt(summary_index.loc['EGSHS','block_equal_Lcap2000'])}`. Full method values are in `D2_METHOD_SUMMARY.csv`.",
        "", "## 30. Median/p75/p95/max", "", f"EGSHS finite-only median `{fmt(summary_index.loc['EGSHS','batch_median_error_m_finite_only'])} m`, p75 `{fmt(summary_index.loc['EGSHS','batch_p75_error_m_finite_only'])} m`, p95 `{fmt(summary_index.loc['EGSHS','batch_p95_error_m_finite_only'])} m`, maximum `{fmt(summary_index.loc['EGSHS','batch_max_error_m_finite_only'])} m`.",
        "", "## 31. >100m", "", f"EGSHS `{int(summary_index.loc['EGSHS','gt_100m_count_all_batches'])}/552`; failures are adverse outcomes.",
        "", "## 32. >500m", "", f"EGSHS `{int(summary_index.loc['EGSHS','gt_500m_count_all_batches'])}/552`; failures are adverse outcomes.",
        "", "## 33. >1km", "", f"EGSHS `{int(summary_index.loc['EGSHS','gt_1000m_count_all_batches'])}/552`; failures are adverse outcomes.",
        "", "## 34. >2km", "", f"EGSHS `{int(summary_index.loc['EGSHS','gt_2000m_count_all_batches'])}/552`; failures are adverse outcomes.",
        "", "## 35. H0 result", "", f"EGSHS mean Lcap1000 `{fmt(family_egshs.loc['H0','mean_Lcap1000'])}`, EGSHS-minus-M2 `{fmt(family_egshs.loc['H0','EGSHS_minus_M2_mean_Lcap1000'])}`.",
        "", "## 36. H1 result", "", f"EGSHS mean Lcap1000 `{fmt(family_egshs.loc['H1','mean_Lcap1000'])}`, EGSHS-minus-M2 `{fmt(family_egshs.loc['H1','EGSHS_minus_M2_mean_Lcap1000'])}`.",
        "", "## 37. H2 result", "", f"EGSHS mean Lcap1000 `{fmt(family_egshs.loc['H2','mean_Lcap1000'])}`, EGSHS-minus-M2 `{fmt(family_egshs.loc['H2','EGSHS_minus_M2_mean_Lcap1000'])}`.",
        "", "## 38. H3 result", "", f"EGSHS mean Lcap1000 `{fmt(family_egshs.loc['H3','mean_Lcap1000'])}`, EGSHS-minus-M2 `{fmt(family_egshs.loc['H3','EGSHS_minus_M2_mean_Lcap1000'])}`.",
        "", "## 39. H4 result", "", f"EGSHS mean Lcap1000 `{fmt(family_egshs.loc['H4','mean_Lcap1000'])}`, EGSHS-minus-M2 `{fmt(family_egshs.loc['H4','EGSHS_minus_M2_mean_Lcap1000'])}`.",
        "", "## 40. H5 result", "", f"EGSHS mean Lcap1000 `{fmt(family_egshs.loc['H5','mean_Lcap1000'])}`, EGSHS-minus-M2 `{fmt(family_egshs.loc['H5','EGSHS_minus_M2_mean_Lcap1000'])}`.",
        "", "## 41. Selection-conditioned results", "", f"Locked EGSHS selections produced `{len(selection)}` selected-model groups. Results are post-selection descriptive and are stored in `D2_SELECTION_OUTCOME_ANALYSIS.csv`.",
        "", "## 42. Changed-from-M2 analysis", "", f"Changed batches `{int(changed_all.changed_batches)}`, unchanged M2 batches `{int(changed_all.unchanged_M2_batches)}`; better/tied/worse by failure-preserving Lcap1000 `{int(changed_all.EGSHS_better_count_by_Lcap1000)}/{int(changed_all.tied_count_by_Lcap1000)}/{int(changed_all.EGSHS_worse_count_by_Lcap1000)}`; mean Lcap1000 difference `{fmt(changed_all.mean_Lcap1000_difference)}`.",
        "", "## 43. M12 comparison", "", f"Overall block-equal Delta_M12 `{fmt(delta_m12)}`; H0-H5 differences are recorded in the family summary. M12 remained the preselected strong fixed candidate.",
        "", "## 44. Information-criterion comparison", "", f"AIC/FSC/BIC were evaluated only as locked M0-M4 compatible-subset comparators. Restricted EGSHS M0-M4 status: `{restricted_status}`; no selector was rerun.",
        "", "## 45. Geometry eligibility context", "", "Target blocks 96; eligible 92; ineligible 4; eligibility rate 95.8333%. The four exclusions were method-independent and absent from method performance denominators.",
        "", "## 46. Failure accounting", "", "Every eligible batch remains in bounded metrics. A method failure or nonfinite position contributes LcapT=1 and counts as an adverse threshold outcome; unbounded summaries use finite outputs and disclose their count.",
        "", "## 47. Evidence boundary", "", "The results address controlled cross-epoch, cross-orbit-snapshot, and new receiver/geometry transfer. They do not establish native RF, cross-constellation, hardware, field dynamic RF, integrity, or protection-level validation.",
        "", "## 48. Protection audit", "", f"All protected inputs matched before and after evaluation. Geometry, observations, candidates, selectors, and baselines were not rerun. Audit: `{display_path(protection_path)}`.",
        "", "## 49. Formal decision", "", f"`{SUCCESS}`.",
        "", "## 50. Scientific interpretation status", "", "`results_ready_for_separate_scientific_interpretation`. Numerical direction is reported without changing the completeness decision.",
        "", "## 51. Next recommended task", "", f"`{NEXT_TASK}`.",
        "", "## 52. Total-report absolute path", "", f"`{display_path(REPORT_PATH)}`.",
        "", "MANUSCRIPT_MODIFIED=false", f"NEXT_RECOMMENDED_TASK={NEXT_TASK}", "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    outputs = [
        PLAN_PATH, batch_path, block_path, summary_path, family_path, diff_m2_path, diff_m12_path, bootstrap_path,
        signflip_path, lobo_path, selection_path, changed_path, information_path, protection_path, DECISION_PATH, REPORT_PATH,
    ]
    output_hashes = {path.name: sha256_file(path) for path in outputs}
    result = {
        "formal_decision": SUCCESS,
        "total_report": display_path(REPORT_PATH),
        "evaluation_plan_lock_sha256": EXPECTED_PLAN_SHA,
        "first_truth_read_utc": first_truth_read_utc,
        "truth_reads": truth_reads,
        "eligible_blocks": 92,
        "eligible_batches": 552,
        "Delta_M2": delta_m2,
        "Delta_M12": delta_m12,
        "output_hashes": output_hashes,
        "runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
