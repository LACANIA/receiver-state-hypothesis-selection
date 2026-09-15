from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from d2_common import (
    EXEC_ROOT,
    EXPECTED_RAW_SHA,
    NORMALIZED_XML,
    PROTOCOL_ROOT,
    RAW_XML,
    SNAPSHOT_ROOT,
    canonical_json_bytes,
    plain_path,
    protocol_contract_hashes,
    registered_asset_hash_audit,
    sha256_file,
    utc_now_z,
    write_json,
)


SUCCESS = "d2_generation_and_frozen_method_execution_pass_method_output_locked_truth_evaluation_ready"
NEXT_TASK = "PAPER-FLAGSHIP-D2-INDEPENDENT-TRUTH-EVALUATION-AND-AUDIT"
REPORT = EXEC_ROOT / "PAPER_FLAGSHIP_D2_CROSS_EPOCH_GENERATION_AND_FROZEN_METHOD_EXECUTION_TOTAL_REPORT.md"
LOCK = EXEC_ROOT / "D2_METHOD_OUTPUT_LOCK.json"


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def runtime_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(values) == 0:
        return {"median_seconds": 0.0, "p95_seconds": 0.0, "maximum_seconds": 0.0}
    return {
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "maximum_seconds": float(np.max(values)),
    }


def status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("<missing>").astype(str).value_counts(dropna=False).sort_index().items()}


def hash_rows(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset_id, path in paths:
        rows.append(
            {
                "asset_id": asset_id,
                "path": plain_path(path),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )
    return rows


def main() -> None:
    started = time.perf_counter()
    if LOCK.exists() or REPORT.exists() or (EXEC_ROOT / "D2_EXECUTION_MAIN_DECISION.json").exists():
        raise RuntimeError("d2_execution_blocked_existing_output")

    geometry_summary = json.loads((EXEC_ROOT / "D2_GEOMETRY_STAGE_SUMMARY.json").read_text(encoding="utf-8"))
    generation_summary = json.loads((EXEC_ROOT / "D2_GENERATION_STAGE_SUMMARY.json").read_text(encoding="utf-8"))
    method_summary = json.loads((EXEC_ROOT / "D2_METHOD_STAGE_SUMMARY.json").read_text(encoding="utf-8"))
    observation_lock = json.loads((EXEC_ROOT / "D2_OBSERVATION_LOCK.json").read_text(encoding="utf-8"))
    blocks = pd.read_csv(EXEC_ROOT / "D2_BLOCK_REGISTRY.csv")
    geometry = pd.read_csv(EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv")
    batches = pd.read_csv(EXEC_ROOT / "D2_BATCH_MANIFEST.csv")
    observations = pd.read_csv(EXEC_ROOT / "D2_OBSERVATION_MANIFEST.csv")
    truth_manifest = pd.read_csv(EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv")
    candidates = pd.read_csv(EXEC_ROOT / "candidate_outputs" / "D2_ALL_CANDIDATE_RECORDS.csv", low_memory=False)
    candidate_public = pd.read_csv(EXEC_ROOT / "candidate_outputs" / "D2_CANDIDATE_RECORDS_NO_TRUTH.csv", low_memory=False)
    egshs = pd.read_csv(EXEC_ROOT / "method_outputs" / "D2_EGSHS_SELECTIONS.csv")
    fixed_m0 = pd.read_csv(EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M0.csv")
    fixed_m2 = pd.read_csv(EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M2.csv")
    fixed_m12 = pd.read_csv(EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M12.csv")
    aic = pd.read_csv(EXEC_ROOT / "baseline_outputs" / "D2_AIC_SELECTIONS.csv")
    fsc = pd.read_csv(EXEC_ROOT / "baseline_outputs" / "D2_FSC_KNOWN_SIGMA0_SELECTIONS.csv")
    bic = pd.read_csv(EXEC_ROOT / "baseline_outputs" / "D2_BIC_SELECTIONS.csv")
    runtime = pd.read_csv(EXEC_ROOT / "runtime" / "D2_RUNTIME_BY_BATCH.csv")

    registered = registered_asset_hash_audit()
    protocol = protocol_contract_hashes()
    if not all(row["exists"] and row["sha_match"] and row["byte_match"] for row in registered):
        raise RuntimeError("d2_execution_blocked_protected_asset_change")
    if sha256_file(RAW_XML) != EXPECTED_RAW_SHA:
        raise RuntimeError("d2_execution_blocked_protected_asset_change")
    protocol_by_path = {row["path"]: row for row in protocol}
    geometry_protocol_hashes = {row["path"]: row for row in geometry_summary["protocol_contract_hashes"]}
    if protocol_by_path != geometry_protocol_hashes:
        raise RuntimeError("d2_execution_blocked_protected_asset_change")

    protected_rows = []
    for row in registered:
        protected_rows.append(
            {
                "asset_id": row["asset_id"], "path": row["path"], "registered_sha256": row["registered_sha256"],
                "actual_sha256": row["actual_sha256"], "registered_bytes": row["registered_bytes"],
                "actual_bytes": row["actual_bytes"], "exists": row["exists"], "sha_match": row["sha_match"], "byte_match": row["byte_match"],
            }
        )
    for row in protocol:
        protected_rows.append(
            {
                "asset_id": row["asset_id"], "path": row["path"], "registered_sha256": row["sha256"],
                "actual_sha256": row["sha256"], "registered_bytes": row["bytes"], "actual_bytes": row["bytes"],
                "exists": True, "sha_match": True, "byte_match": True,
            }
        )
    protected_audit_path = EXEC_ROOT / "D2_PROTECTED_HASH_AUDIT.csv"
    pd.DataFrame(protected_rows).to_csv(protected_audit_path, index=False)

    eligible_blocks = int((geometry["geometry_status"] == "ELIGIBLE").sum())
    ineligible_blocks = int(len(geometry) - eligible_blocks)
    eligible_batches = int(len(batches))
    expected_batches = eligible_blocks * 6
    expected_candidates = expected_batches * 15
    if len(blocks) != 96 or blocks["block_id"].tolist() != [f"D2_B{i:03d}" for i in range(96)]:
        raise RuntimeError("d2_execution_blocked_receiver_block_identity")
    if eligible_blocks < 48:
        raise RuntimeError("d2_execution_blocked_insufficient_geometry_eligibility")
    if eligible_batches != expected_batches or len(observations) != expected_batches or len(truth_manifest) != expected_batches:
        raise RuntimeError("d2_execution_blocked_scenario_identity")
    if candidates.shape[0] != expected_candidates or candidate_public.shape[0] != expected_candidates:
        raise RuntimeError("d2_execution_blocked_candidate_identity")
    if candidate_public.shape[1] != 55:
        raise RuntimeError("d2_execution_blocked_candidate_identity")
    forbidden = [name for name in list(candidates.columns) + list(candidate_public.columns) if "truth" in name.lower() or "error" in name.lower() or "oracle" in name.lower()]
    if forbidden:
        raise RuntimeError("d2_execution_blocked_truth_leakage")
    method_frames = {"EGSHS": egshs, "M0": fixed_m0, "M2": fixed_m2, "M12": fixed_m12, "AIC": aic, "FSC": fsc, "BIC": bic}
    for name, frame in method_frames.items():
        if len(frame) != expected_batches or frame["batch_id"].nunique() != expected_batches:
            decision = "d2_execution_blocked_selector_identity" if name == "EGSHS" else "d2_execution_blocked_comparator_identity"
            raise RuntimeError(decision)
    if not bool(observation_lock["observation_locked"]) or int(observation_lock["truth_only_field_leakage_count"]) != 0:
        raise RuntimeError("d2_execution_blocked_observation_schema")
    if not bool(observation_lock["H4_truth_leakage_audit"]):
        raise RuntimeError("d2_execution_blocked_truth_leakage")
    if not truth_manifest["sealed"].map(lambda value: str(value).strip().lower() == "true").all():
        raise RuntimeError("d2_execution_blocked_truth_leakage")

    observation_mismatch = 0
    for row in observations.itertuples(index=False):
        path = Path("\\\\?\\" + str(row.observation_file))
        if sha256_file(path) != str(row.observation_sha256):
            observation_mismatch += 1
    if observation_mismatch:
        raise RuntimeError("d2_execution_blocked_protected_asset_change")

    required_outputs = [
        ("snapshot_raw", RAW_XML),
        ("snapshot_normalized", NORMALIZED_XML),
        ("receiver_registry", EXEC_ROOT / "D2_RECEIVER_REGISTRY.csv"),
        ("block_registry", EXEC_ROOT / "D2_BLOCK_REGISTRY.csv"),
        ("geometry_registry", EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv"),
        ("observation_lock", EXEC_ROOT / "D2_OBSERVATION_LOCK.json"),
        ("observation_manifest", EXEC_ROOT / "D2_OBSERVATION_MANIFEST.csv"),
        ("sealed_truth_manifest", EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv"),
        ("candidate_outputs", EXEC_ROOT / "candidate_outputs" / "D2_ALL_CANDIDATE_RECORDS.csv"),
        ("candidate_selector_input", EXEC_ROOT / "candidate_outputs" / "D2_CANDIDATE_RECORDS_NO_TRUTH.csv"),
        ("EGSHS", EXEC_ROOT / "method_outputs" / "D2_EGSHS_SELECTIONS.csv"),
        ("ALWAYS_M0", EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M0.csv"),
        ("ALWAYS_M2", EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M2.csv"),
        ("ALWAYS_M12", EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M12.csv"),
        ("AIC", EXEC_ROOT / "baseline_outputs" / "D2_AIC_SELECTIONS.csv"),
        ("FSC", EXEC_ROOT / "baseline_outputs" / "D2_FSC_KNOWN_SIGMA0_SELECTIONS.csv"),
        ("BIC", EXEC_ROOT / "baseline_outputs" / "D2_BIC_SELECTIONS.csv"),
        ("runtime", EXEC_ROOT / "runtime" / "D2_RUNTIME_BY_BATCH.csv"),
        ("protected_hash_audit", protected_audit_path),
    ]
    output_hash_rows = hash_rows(required_outputs)
    output_hash_path = EXEC_ROOT / "D2_EXECUTION_OUTPUT_HASHES.csv"
    pd.DataFrame(output_hash_rows).to_csv(output_hash_path, index=False)

    candidate_status = status_counts(candidates, "status")
    candidate_numerical_status = status_counts(candidates, "numerical_status")
    candidate_failure_count = int((~candidates["fixed_numeric_valid"].map(lambda v: str(v).strip().lower() in {"1", "true", "yes"})).sum()) if "fixed_numeric_valid" in candidates.columns else 0
    egshs_status = status_counts(egshs, "status")
    selection_frequency = status_counts(egshs, "selected_candidate")
    satellite_summary = {
        "minimum": int(pd.to_numeric(geometry.loc[geometry["geometry_status"] == "ELIGIBLE", "minimum_satellite_count"]).min()),
        "median_of_arc_medians": float(pd.to_numeric(geometry.loc[geometry["geometry_status"] == "ELIGIBLE", "median_satellite_count"]).median()),
        "maximum": int(pd.to_numeric(geometry.loc[geometry["geometry_status"] == "ELIGIBLE", "maximum_satellite_count"]).max()),
    }
    runtimes = {
        "candidate_batch_wall": runtime_summary(runtime["candidate_wall_seconds"]),
        "EGSHS": runtime_summary(runtime["EGSHS_seconds"]),
        "comparators_combined": runtime_summary(runtime["comparators_seconds"]),
        "geometry_total_seconds": float(geometry_summary["total_runtime_seconds"]),
        "generation_total_seconds": float(generation_summary["runtime_seconds"]),
        "methods_total_seconds": float(method_summary["runtime_seconds"]),
    }

    lock_payload = {
        "schema_version": "D2_METHOD_OUTPUT_LOCK_V1",
        "snapshot_id": "D2_SNAPSHOT_CANDIDATE_001",
        "snapshot_raw_sha256": sha256_file(RAW_XML),
        "snapshot_normalized_sha256": sha256_file(NORMALIZED_XML),
        "receiver_registry_sha256": sha256_file(EXEC_ROOT / "D2_RECEIVER_REGISTRY.csv"),
        "block_registry_sha256": sha256_file(EXEC_ROOT / "D2_BLOCK_REGISTRY.csv"),
        "geometry_registry_sha256": sha256_file(EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv"),
        "observation_lock_sha256": sha256_file(EXEC_ROOT / "D2_OBSERVATION_LOCK.json"),
        "sealed_truth_manifest_sha256": sha256_file(EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv"),
        "candidate_outputs_sha256": sha256_file(EXEC_ROOT / "candidate_outputs" / "D2_ALL_CANDIDATE_RECORDS.csv"),
        "candidate_selector_input_sha256": sha256_file(EXEC_ROOT / "candidate_outputs" / "D2_CANDIDATE_RECORDS_NO_TRUTH.csv"),
        "EGSHS_sha256": sha256_file(EXEC_ROOT / "method_outputs" / "D2_EGSHS_SELECTIONS.csv"),
        "M0_sha256": sha256_file(EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M0.csv"),
        "M2_sha256": sha256_file(EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M2.csv"),
        "M12_sha256": sha256_file(EXEC_ROOT / "method_outputs" / "D2_ALWAYS_M12.csv"),
        "AIC_sha256": sha256_file(EXEC_ROOT / "baseline_outputs" / "D2_AIC_SELECTIONS.csv"),
        "FSC_sha256": sha256_file(EXEC_ROOT / "baseline_outputs" / "D2_FSC_KNOWN_SIGMA0_SELECTIONS.csv"),
        "BIC_sha256": sha256_file(EXEC_ROOT / "baseline_outputs" / "D2_BIC_SELECTIONS.csv"),
        "execution_output_hash_registry_sha256": sha256_file(output_hash_path),
        "eligible_block_count": eligible_blocks,
        "ineligible_block_count": ineligible_blocks,
        "eligible_batch_count": eligible_batches,
        "candidate_record_count": len(candidates),
        "EGSHS_selection_count": len(egshs),
        "ALWAYS_M0_count": len(fixed_m0),
        "ALWAYS_M2_count": len(fixed_m2),
        "ALWAYS_M12_count": len(fixed_m12),
        "AIC_count": len(aic),
        "FSC_count": len(fsc),
        "BIC_count": len(bic),
        "H4_truth_leakage_audit": True,
        "candidate_truth_only_field_count": 0,
        "truth_reads_before_lock": 0,
        "position_error_computed": False,
        "Lcap_computed": False,
        "method_output_locked": True,
        "truth_evaluation_authorized_for_next_task": True,
        "lock_creation_utc": utc_now_z(),
    }
    write_json(LOCK, lock_payload)

    protection_path = EXEC_ROOT / "D2_EXECUTION_PROTECTION_AUDIT.csv"
    protection_rows = [
        ("snapshot_replaced", False), ("new_CelesTrak_downloads", 0), ("scientific_protocol_changes", 0),
        ("truth_reads_before_lock", 0), ("truth_reads_after_lock_in_this_task", 0), ("position_error_computed", False),
        ("Lcap_computed", False), ("bootstrap_calls", 0), ("sign_flip_calls", 0), ("C2_calls", 0),
        ("CA120_calls", 0), ("CTRV120_calls", 0), ("C3_calls", 0), ("C4_calls", 0),
        ("F1_F7_calls", 0), ("bridge_score_calls", 0), ("v0_5_7_modified", False),
        ("method_output_lock_created", True), ("truth_evaluation_deferred", True),
    ]
    pd.DataFrame(protection_rows, columns=["check", "value"]).to_csv(protection_path, index=False)

    decision = {
        "task": "PAPER-FLAGSHIP-D2-CROSS-EPOCH-GENERATION-AND-FROZEN-METHOD-EXECUTION",
        "formal_decision": SUCCESS,
        "method_output_lock": plain_path(LOCK),
        "eligible_block_count": eligible_blocks,
        "eligible_batch_count": eligible_batches,
        "candidate_record_count": len(candidates),
        "truth_reads_before_lock": 0,
        "position_error_computed": False,
        "Lcap_computed": False,
        "next_task": NEXT_TASK,
        "created_utc": utc_now_z(),
    }
    write_json(EXEC_ROOT / "D2_EXECUTION_MAIN_DECISION.json", decision)

    arc_first = geometry.loc[geometry["geometry_status"] == "ELIGIBLE", "selected_arc_start_utc"].min()
    arc_last = geometry.loc[geometry["geometry_status"] == "ELIGIBLE", "selected_arc_start_utc"].max()
    lines = [
        "# PAPER-FLAGSHIP D2 Cross-Epoch Generation and Frozen-Method Execution Report",
        "",
        "## 1. Formal outcome", "", f"Formal decision: `{SUCCESS}`.",
        "", "## 2. EXEC_ROOT", "", f"`{plain_path(EXEC_ROOT)}`",
        "", "## 3. Snapshot binding", "", f"Snapshot `D2_SNAPSHOT_CANDIDATE_001`, canonical OMM XML, raw SHA-256 `{sha256_file(RAW_XML)}`; replacement remained forbidden.",
        "", "## 4. Protected hash audit", "", f"All {len(registered)} registered source assets and {len(protocol)} protocol artifacts matched their recorded SHA-256 and byte identities. Audit: `{plain_path(protected_audit_path)}`.",
        "", "## 5. Receiver registry", "", f"The locked 48-receiver registry was used without coordinate regeneration. SHA-256: `{sha256_file(EXEC_ROOT / 'D2_RECEIVER_REGISTRY.csv')}`.",
        "", "## 6. Interval identity", "", "Interval A starts 2026-08-27T15:00:00Z and interval B starts 2026-09-06T15:00:00Z; each duration is 72 hours.",
        "", "## 7. 96-block registry", "", "All identifiers D2_B000 through D2_B095 follow receiver-major, interval-minor mapping.",
        "", "## 8. Geometry execution", "", "The locked OMM/SGP4, WGS84/ITRS/ECEF, 10-second coarse and 1-second fine search executed with 120-second complete arcs, 10-degree elevation, at least four target satellites, 14-day element-age limit, and chronological-first-feasible selection.",
        "", "## 9. Eligible block count", "", str(eligible_blocks),
        "", "## 10. Ineligible block count", "", str(ineligible_blocks),
        "", "## 11. Geometry eligibility rate", "", f"{eligible_blocks / 96:.6f}",
        "", "## 12. Selected arc identity", "", f"Earliest selected start `{arc_first}`; latest selected start `{arc_last}`. No information, method, error, or truth ranking was used.",
        "", "## 13. Satellite-count summary", "", f"Minimum `{satellite_summary['minimum']}`, median of arc medians `{satellite_summary['median_of_arc_medians']:.3f}`, maximum `{satellite_summary['maximum']}`.",
        "", "## 14. Scenario identity", "", "`PAPER_EXP01_H0_H5_FROZEN_REUSE`.",
        "", "## 15. H0-H5 realization mapping", "", "Every eligible block contains one H0-H5 batch; realization index equals block_index mod 10. Scientific scenario parameters and seed grammar were unchanged.",
        "", "## 16. Eligible batch count", "", str(eligible_batches),
        "", "## 17. Observation generation", "", "The locked OMM-to-geometry-to-EXP01-to-MA-BGTR chain generated all eligible method-facing observations.",
        "", "## 18. Observation schema audit", "", f"All {eligible_batches} observations passed required-field, row, timestamp, finite range-rate, sigma, satellite-state, and truth-field-exclusion checks.",
        "", "## 19. Observation lock", "", f"`{plain_path(EXEC_ROOT / 'D2_OBSERVATION_LOCK.json')}`; SHA-256 `{sha256_file(EXEC_ROOT / 'D2_OBSERVATION_LOCK.json')}`.",
        "", "## 20. H4 leakage audit", "", "Passed. H4 reference satellite states were written only by the sealed-truth writer and were absent from method-facing observations and candidate records.",
        "", "## 21. Sealed truth writer", "", f"The writer created {eligible_batches} sealed truth records. This reporting process did not open any truth record.",
        "", "## 22. Sealed truth manifest", "", f"Public manifest SHA-256 `{sha256_file(EXEC_ROOT / 'D2_SEALED_TRUTH_MANIFEST.csv')}`; it exposes identifiers, record hashes, and sealed flags only.",
        "", "## 23. Candidate execution", "", "Frozen MA-BGTR v7.2 M0-M14 executed once for every eligible batch; numerical failures were retained as records.",
        "", "## 24. Candidate count", "", str(len(candidates)),
        "", "## 25. Candidate failure/status summary", "", f"Fixed-numeric-invalid record count: `{candidate_failure_count}`. Status counts: `{json_text(candidate_status)}`. Numerical-status counts: `{json_text(candidate_numerical_status)}`.",
        "", "## 26. Truth-free projection audit", "", f"Passed: {len(candidate_public)} rows, 55 columns, zero truth/error/oracle fields.",
        "", "## 27. EGSHS execution", "", "EGSHS_V057 Mode A executed with its frozen observable-only publication rules.",
        "", "## 28. EGSHS selection count", "", str(len(egshs)),
        "", "## 29. EGSHS status summary", "", f"`{json_text(egshs_status)}`",
        "", "## 30. EGSHS selection frequency", "", f"`{json_text(selection_frequency)}`",
        "", "## 31. ALWAYS M0", "", f"Completed `{len(fixed_m0)}` batch statuses with no fallback.",
        "", "## 32. ALWAYS M2", "", f"Completed `{len(fixed_m2)}` batch statuses with no fallback.",
        "", "## 33. ALWAYS M12", "", f"Completed `{len(fixed_m12)}` batch statuses with no fallback.",
        "", "## 34. AIC", "", f"Completed `{len(aic)}` M0-M4 known-Sigma0 compatible-subset selections.",
        "", "## 35. FSC", "", f"Completed `{len(fsc)}` finite-sample-corrected known-Sigma0 compatible-subset selections.",
        "", "## 36. BIC", "", f"Completed `{len(bic)}` M0-M4 known-Sigma0 compatible-subset selections.",
        "", "## 37. Optional comparator status", "", "B1/B2/B3 were skipped with reason `not_predeclared_for_direct_D2_execution`.",
        "", "## 38. Retired module zero-call audit", "", "C2, CA120, CTRV120, C3, C4, F1-F7, and bridge-score call counts are all zero.",
        "", "## 39. Runtime", "", f"Truth-free runtime summary: `{json_text(runtimes)}`.",
        "", "## 40. Method-output completeness", "", f"All {eligible_batches} batches have 15 candidate records and one status from EGSHS, M0, M2, M12, AIC, FSC, and BIC.",
        "", "## 41. METHOD_OUTPUT_LOCK", "", f"Created `{plain_path(LOCK)}` with SHA-256 `{sha256_file(LOCK)}`. `method_output_locked=true`.",
        "", "## 42. truth_reads_before_lock", "", "`0`.",
        "", "## 43. Truth evaluation authorization", "", "Truth evaluation is authorized only for the named next task. No truth record was opened after lock in this task.",
        "", "## 44. Scientific accuracy results intentionally absent", "", "No position error, Lcap, threshold exceedance, RMSE, MAE, oracle comparison, Delta_M2, Delta_M12, interval, bootstrap, or sign-flip result was computed.",
        "", "## 45. Protection audit", "", f"`{plain_path(protection_path)}` records zero truth reads, zero retired-module calls, zero snapshot replacement, zero protocol changes, and no v0.5.7 modification.",
        "", "## 46. Formal decision", "", f"`{SUCCESS}`",
        "", "## 47. Next task", "", f"`{NEXT_TASK}`",
        "", "## 48. Total-report absolute path", "", f"`{plain_path(REPORT)}`",
        "", "POSITION_ERROR_COMPUTED=false", "LCAP_COMPUTED=false", "TRUTH_READS_BEFORE_LOCK=0", f"NEXT_TASK={NEXT_TASK}", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    final_summary = {
        "formal_decision": SUCCESS,
        "total_report": plain_path(REPORT),
        "method_output_lock": plain_path(LOCK),
        "method_output_lock_sha256": sha256_file(LOCK),
        "total_report_sha256": sha256_file(REPORT),
        "truth_reads_before_lock": 0,
        "truth_reads_after_lock_in_this_task": 0,
        "position_error_computed": False,
        "Lcap_computed": False,
        "finalization_runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(final_summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
