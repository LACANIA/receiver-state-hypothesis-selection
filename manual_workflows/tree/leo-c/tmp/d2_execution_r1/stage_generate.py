from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from d2_common import (
    EXEC_ROOT,
    EXP01_RUN,
    EXP01_SCENARIOS,
    PROTOCOL_ROOT,
    RELEASE_ROOT,
    canonical_json_bytes,
    import_module,
    load_json,
    plain_path,
    sha256_bytes,
    sha256_file,
    utc_now_z,
    write_json,
)


def scalar(value: np.ndarray | Any) -> Any:
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def load_geometry(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def scenario_records() -> dict[tuple[str, int], dict[str, Any]]:
    frame = pd.read_csv(EXP01_SCENARIOS)
    families = [
        "H0_static_holdout",
        "H1_dynamic_clean_holdout",
        "H2_bias_drift_holdout",
        "H3_outlier_holdout",
        "H4_geometry_ephemeris_holdout",
        "H5_no_bias_dynamic_holdout",
    ]
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for family in families:
        subset = frame.loc[frame["holdout_family"].astype(str).eq(family)].copy()
        if len(subset) != 10 or set(subset["realization_index"].astype(int)) != set(range(10)):
            raise RuntimeError("d2_execution_blocked_scenario_identity")
        for row in subset.to_dict(orient="records"):
            result[(family, int(row["realization_index"]))] = row
    return result


def unique_truth_trajectory(time_s: np.ndarray, truth_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_times, first_indices = np.unique(time_s, return_index=True)
    order = np.argsort(unique_times)
    return unique_times[order], truth_positions[first_indices[order]]


def main() -> None:
    started = time.perf_counter()
    geometry_summary = load_json(EXEC_ROOT / "D2_GEOMETRY_STAGE_SUMMARY.json")
    if not geometry_summary["headline_gate_pass"]:
        raise RuntimeError("d2_execution_blocked_insufficient_geometry_eligibility")
    binding = load_json(PROTOCOL_ROOT / "D2_SCENARIO_REUSE_BINDING.json")
    schema = load_json(PROTOCOL_ROOT / "D2_OBSERVATION_SCHEMA.json")
    if binding["D2_scenario_identity"] != "PAPER_EXP01_H0_H5_FROZEN_REUSE" or binding["scenarios_per_eligible_block"] != 6:
        raise RuntimeError("d2_execution_blocked_scenario_identity")

    exp01 = import_module(EXP01_RUN, "d2_exp01_observation_builder")
    release_src = RELEASE_ROOT / "src"
    import sys
    if str(release_src) not in sys.path:
        sys.path.insert(0, str(release_src))
    from leo_positioning.coordinates import enu_axes
    from leo_positioning.qatar_error_model import sample_heavy_tail_outliers
    from leo_positioning.trajectory_models import FULL_CTD_CONFIG, pack_state, predict_ctd

    observation_dir = EXEC_ROOT / "observations"
    truth_dir = EXEC_ROOT / "SEALED_TRUTH_STORE"
    observation_dir.mkdir(exist_ok=False)
    truth_dir.mkdir(exist_ok=False)

    geometry = pd.read_csv(EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv")
    eligible = geometry.loc[geometry["geometry_status"].astype(str).eq("ELIGIBLE")].copy()
    records = scenario_records()
    family_order = list(binding["family_order"])
    observation_manifest: list[dict[str, Any]] = []
    truth_manifest: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    h4_leakage_pass = True

    for block_number, block in enumerate(eligible.itertuples(index=False), start=1):
        geometry_path = Path("\\\\?\\" + str(block.geometry_file))
        geom = load_geometry(geometry_path)
        block_id = str(block.block_id)
        block_index = int(block.block_index)
        realization = block_index % 10
        base_time = np.asarray(geom["time_s"], dtype=float)
        base_sats = np.asarray(geom["satellite_number"], dtype=np.int64)
        base_obs = {
            "time_s": base_time,
            "satellite_number": base_sats,
            "sat_pos_m": np.asarray(geom["sat_pos_m"], dtype=float),
            "sat_vel_mps": np.asarray(geom["sat_vel_mps"], dtype=float),
            "p_gt_ecef_m": np.asarray(geom["receiver_anchor_ecef_m"], dtype=float),
            "lat_deg": float(scalar(geom["lat_deg"])),
            "lon_deg": float(scalar(geom["lon_deg"])),
            "height_m": float(scalar(geom["height_m"])),
        }
        unique_satellite_count = int(np.unique(base_sats).size)
        for family_index, family in enumerate(family_order):
            source_record = dict(records[(family, realization)])
            source_holdout_id = str(source_record["holdout_id"])
            adapted = dict(source_record)
            adapted.update(
                {
                    "satellite_subset": ";".join(str(item) for item in sorted(np.unique(base_sats).astype(int))),
                    "window_start_s": 0.0,
                    "window_end_s": 120.0,
                    "observation_count": int(len(base_time)),
                    "duration_s": 120.0,
                    "sampling_interval_median_s": 1.0,
                    "unique_satellite_count": unique_satellite_count,
                }
            )
            obs_full, audit = exp01.build_holdout_observation(
                adapted,
                base_obs,
                enu_axes=enu_axes,
                pack_state=pack_state,
                predict_ctd=predict_ctd,
                full_ctd_config=FULL_CTD_CONFIG,
                sample_heavy_tail_outliers=sample_heavy_tail_outliers,
            )
            if int(audit["actual_observation_count"]) != int(len(base_time)) or float(audit["actual_duration_s"]) != 120.0:
                raise RuntimeError("d2_execution_blocked_observation_schema")
            batch_id = f"{block_id}_H{family_index}"
            observation_path = observation_dir / f"{batch_id}_OBS.npz"
            scenario_config = dict(obs_full.get("scenario_config", {}) or {})
            metadata = {
                "dataset_type": "synthetic",
                "outlier_ratio": 0.0,
                "dropout_ratio": 0.0,
                "burst_outlier": False,
            }
            method_payload = {
                "batch_id": np.asarray(batch_id),
                "block_id": np.asarray(block_id),
                "scenario": np.asarray(str(source_record["selector_scenario_key"])),
                "holdout_family": np.asarray(family),
                "source_holdout_id": np.asarray(source_holdout_id),
                "realization_index": np.asarray(realization),
                "time_s": np.asarray(obs_full["time_s"], dtype=float),
                "sat_pos_m": np.asarray(obs_full["sat_pos_m"], dtype=float),
                "sat_vel_mps": np.asarray(obs_full["sat_vel_mps"], dtype=float),
                "meas_mps": np.asarray(obs_full["meas_mps"], dtype=float),
                "sigma_mps": np.asarray(float(source_record["noise_sigma_mps"])),
                "satellite_number": np.asarray(obs_full["satellite_number"], dtype=np.int64),
                "row_index": np.asarray(obs_full["row_index"], dtype=np.int64),
                "lat_deg": np.asarray(float(obs_full["lat_deg"])),
                "lon_deg": np.asarray(float(obs_full["lon_deg"])),
                "height_m": np.asarray(float(obs_full["height_m"])),
                "t0_s": np.asarray(float(obs_full["t0_s"])),
                "public_receiver_anchor_ecef_m": np.asarray(base_obs["p_gt_ecef_m"], dtype=float),
                "confidence_values": np.asarray(obs_full["confidence_values"], dtype=float),
                "scenario_config_json": np.asarray(json.dumps(scenario_config, sort_keys=True, separators=(",", ":"))),
                "observable_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            }
            method_keys = sorted(method_payload)
            forbidden_key_hits = [
                key for key in method_keys
                if "truth" in key.lower() or key.lower() in {"p_gt_ecef_m", "p0_true_m", "v_true_mps", "b0_true_mps", "bdot_true_mps2"}
            ]
            if forbidden_key_hits:
                raise RuntimeError(f"d2_execution_blocked_truth_leakage:{forbidden_key_hits}")
            np.savez_compressed(observation_path, **method_payload)
            with np.load(observation_path, allow_pickle=False) as reopened:
                reopened_keys = sorted(reopened.files)
                schema_checks = {
                    "required_method_fields": all(
                        name in reopened_keys
                        for name in ["batch_id", "block_id", "scenario", "time_s", "sat_pos_m", "sat_vel_mps", "meas_mps", "sigma_mps", "satellite_number", "row_index", "lat_deg", "lon_deg", "t0_s", "public_receiver_anchor_ecef_m"]
                    ),
                    "finite_range_rate": bool(np.isfinite(reopened["meas_mps"]).all()),
                    "finite_satellite_state": bool(np.isfinite(reopened["sat_pos_m"]).all() and np.isfinite(reopened["sat_vel_mps"]).all()),
                    "row_identity": bool(len(reopened["row_index"]) == len(reopened["time_s"])),
                    "no_truth_key": not any("truth" in name.lower() for name in reopened_keys),
                }
            if not all(schema_checks.values()):
                raise RuntimeError(f"d2_execution_blocked_observation_schema:{batch_id}:{schema_checks}")

            truth_times, truth_trajectory = unique_truth_trajectory(
                np.asarray(obs_full["time_s"], dtype=float),
                np.asarray(obs_full["truth_positions_m"], dtype=float),
            )
            output_matches = np.flatnonzero(np.isclose(truth_times, 110.0, rtol=0.0, atol=1e-12))
            if len(output_matches) != 1:
                raise RuntimeError("d2_execution_blocked_scenario_identity")
            truth_core: dict[str, Any] = {
                "batch_id": batch_id,
                "block_id": block_id,
                "receiver_id": str(block.receiver_id),
                "receiver_truth_time_s": truth_times.tolist(),
                "receiver_truth_trajectory_ecef_m": truth_trajectory.tolist(),
                "output_epoch_s": 110.0,
                "output_epoch_truth_position_ecef_m": truth_trajectory[int(output_matches[0])].tolist(),
                "scenario_identity": {
                    "D2_identity": "PAPER_EXP01_H0_H5_FROZEN_REUSE",
                    "holdout_family": family,
                    "source_holdout_id": source_holdout_id,
                    "realization_index": realization,
                },
                "generation_identity": {
                    "snapshot_id": "D2_SNAPSHOT_CANDIDATE_001",
                    "block_geometry_sha256": str(block.geometry_sha256),
                    "EXP01_scenario_registry_sha256": sha256_file(EXP01_SCENARIOS),
                },
            }
            if family == "H4_geometry_ephemeris_holdout":
                truth_core["H4_reference_satellite_state"] = {
                    "row_index": np.asarray(obs_full["row_index"], dtype=np.int64).tolist(),
                    "satellite_number": np.asarray(obs_full["satellite_number"], dtype=np.int64).tolist(),
                    "sat_pos_reference_ecef_m": np.asarray(obs_full["sat_pos_truth_m"], dtype=float).tolist(),
                    "sat_vel_reference_ecef_mps": np.asarray(obs_full["sat_vel_truth_mps"], dtype=float).tolist(),
                }
                h4_leakage_pass = h4_leakage_pass and not any("reference" in name.lower() for name in method_keys)
            core_hash = sha256_bytes(canonical_json_bytes(truth_core))
            truth_payload = dict(truth_core)
            truth_payload["truth_record_sha256"] = core_hash
            truth_bytes = canonical_json_bytes(truth_payload)
            truth_record_id = f"{batch_id}_TRUTH"
            truth_path = truth_dir / f"{truth_record_id}.json"
            truth_path.write_bytes(truth_bytes)
            truth_file_hash = sha256_bytes(truth_bytes)

            observation_hash = sha256_file(observation_path)
            observation_manifest.append(
                {
                    "batch_id": batch_id,
                    "block_id": block_id,
                    "observation_file": plain_path(observation_path),
                    "observation_sha256": observation_hash,
                    "row_count": len(base_time),
                    "unique_satellite_count": unique_satellite_count,
                    "sigma_mps": float(source_record["noise_sigma_mps"]),
                    "schema_pass": True,
                    "truth_only_field_count": 0,
                    "locked": True,
                }
            )
            truth_manifest.append(
                {
                    "batch_id": batch_id,
                    "truth_record_id": truth_record_id,
                    "truth_record_sha256": truth_file_hash,
                    "sealed": True,
                }
            )
            batch_rows.append(
                {
                    "batch_id": batch_id,
                    "block_id": block_id,
                    "receiver_id": str(block.receiver_id),
                    "interval_id": str(block.interval_id),
                    "arc_start": str(block.selected_arc_start_utc),
                    "scenario_family": family,
                    "source_holdout_id": source_holdout_id,
                    "realization_index": realization,
                    "observation_status": "GENERATED_VALIDATED_LOCKED",
                    "truth_record_id": truth_record_id,
                    "truth_sealed": True,
                    "candidate_status": "PENDING",
                    "selector_status": "PENDING",
                }
            )
            scenario_rows.append(
                {
                    "batch_id": batch_id,
                    "block_id": block_id,
                    "family_index": family_index,
                    "holdout_family": family,
                    "source_holdout_id": source_holdout_id,
                    "realization_index": realization,
                    "parameter_seed": int(source_record["parameter_seed"]),
                    "noise_seed": int(source_record["noise_seed"]),
                    "ephemeris_seed": int(source_record["ephemeris_seed"]),
                    "label_seed": int(source_record["label_seed"]),
                    "scenario_parameters_unchanged": True,
                    "geometry_fields_adapted_only": True,
                }
            )
        if block_number % 8 == 0 or block_number == len(eligible):
            print(json.dumps({"stage": "generation", "eligible_blocks_completed": block_number, "batches_written": len(batch_rows)}), flush=True)

    if not h4_leakage_pass:
        raise RuntimeError("d2_execution_blocked_truth_leakage")
    batch_frame = pd.DataFrame(batch_rows).sort_values("batch_id")
    observation_frame = pd.DataFrame(observation_manifest).sort_values("batch_id")
    truth_frame = pd.DataFrame(truth_manifest).sort_values("batch_id")
    scenario_frame = pd.DataFrame(scenario_rows).sort_values("batch_id")
    expected_batches = len(eligible) * 6
    if len(batch_frame) != expected_batches or len(observation_frame) != expected_batches or len(truth_frame) != expected_batches:
        raise RuntimeError("d2_execution_blocked_scenario_identity")
    batch_frame.to_csv(EXEC_ROOT / "D2_BATCH_MANIFEST.csv", index=False)
    observation_frame.to_csv(EXEC_ROOT / "D2_OBSERVATION_MANIFEST.csv", index=False)
    truth_frame.to_csv(EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv", index=False)
    scenario_frame.to_csv(EXEC_ROOT / "D2_SCENARIO_REGISTRY.csv", index=False)
    observation_lock = {
        "schema_version": "D2_OBSERVATION_LOCK_V1",
        "created_utc": utc_now_z(),
        "eligible_block_count": len(eligible),
        "eligible_batch_count": expected_batches,
        "observation_manifest_sha256": sha256_file(EXEC_ROOT / "D2_OBSERVATION_MANIFEST.csv"),
        "batch_manifest_sha256": sha256_file(EXEC_ROOT / "D2_BATCH_MANIFEST.csv"),
        "scenario_registry_sha256": sha256_file(EXEC_ROOT / "D2_SCENARIO_REGISTRY.csv"),
        "sealed_truth_manifest_sha256": sha256_file(EXEC_ROOT / "D2_SEALED_TRUTH_MANIFEST.csv"),
        "observation_file_hashes": {row["batch_id"]: row["observation_sha256"] for row in observation_manifest},
        "required_schema_audit_pass": True,
        "truth_only_field_leakage_count": 0,
        "H4_truth_leakage_audit": True,
        "observation_locked": True,
        "regeneration_forbidden": True,
    }
    write_json(EXEC_ROOT / "D2_OBSERVATION_LOCK.json", observation_lock)
    write_json(
        EXEC_ROOT / "D2_GENERATION_STAGE_SUMMARY.json",
        {
            "completed_utc": utc_now_z(),
            "eligible_block_count": len(eligible),
            "eligible_batch_count": expected_batches,
            "scenario_identity": binding["D2_scenario_identity"],
            "observation_schema_version": schema["schema_version"],
            "H4_truth_leakage_audit": True,
            "sealed_truth_records_written": expected_batches,
            "truth_records_read": 0,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps({"stage": "generation", "eligible_blocks": len(eligible), "eligible_batches": expected_batches, "truth_reads": 0, "runtime_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
