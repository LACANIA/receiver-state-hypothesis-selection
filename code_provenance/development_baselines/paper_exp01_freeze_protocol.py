"""Freeze the PAPER-EXP01 hold-out protocol before any solver execution.

This script only reads the immutable r2 release and writes protocol/audit files
to the PAPER-EXP01 output directory.  It never invokes a positioning solver.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
LAB_ROOT = OUTPUT_DIR.parents[1]
RELEASE_ROOT = LAB_ROOT / "final_release" / "MA_BGTR_v7_2_freeze_r2"
PROTOCOL_PATH = OUTPUT_DIR / "PAPER_EXP01_HOLDOUT_PROTOCOL.json"
PROTOCOL_HASH_PATH = OUTPUT_DIR / "PAPER_EXP01_HOLDOUT_PROTOCOL_SHA256.txt"
SCENARIO_PATH = OUTPUT_DIR / "PAPER_EXP01_HOLDOUT_SCENARIOS.csv"
HASH_BEFORE_PATH = OUTPUT_DIR / "PAPER_EXP01_FROZEN_HASHES_BEFORE.csv"

MASTER_SEED = 20260723
PROTOCOL_ID = "PAPER_EXP01_HOLDOUT_MC_V1"
CANDIDATE_POOL_PROTOCOL = "FULL_POOL_M0_M14_V1"
CANDIDATE_MODELS = [
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_immutable_manifest() -> dict[str, str]:
    manifest_path = RELEASE_ROOT / "IMMUTABLE_MANIFEST_SHA256.txt"
    expected: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            digest, relative = line.split("  ", 1)
            expected[relative.replace("\\", "/")] = digest
    mismatches: list[str] = []
    for relative, digest in expected.items():
        path = RELEASE_ROOT / Path(relative)
        if not path.is_file() or sha256_file(path) != digest:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"immutable release verification failed for {len(mismatches)} files: {mismatches[:5]}")
    if len(expected) != 134:
        raise RuntimeError(f"expected 134 immutable files, found {len(expected)}")
    return expected


def source_hash_scope(manifest: dict[str, str]) -> list[dict[str, Any]]:
    paths = sorted((RELEASE_ROOT / "src" / "leo_positioning").glob("*.py"))
    paths.extend(sorted((RELEASE_ROOT / "configs").glob("*.json")))
    paths.append(RELEASE_ROOT / "scripts" / "release_runtime.py")
    records: list[dict[str, Any]] = []
    selector_names = {"src/leo_positioning/ma_bgtr_v71_solver.py"}
    freeze_names = {
        "src/leo_positioning/ma_bgtr_v72_solver.py",
        "src/leo_positioning/selector_freeze_policy.py",
    }
    candidate_core = {
        "src/leo_positioning/model_candidates.py",
        "src/leo_positioning/ma_bgtr_v4_solver.py",
        "src/leo_positioning/trajectory_solvers.py",
        "src/leo_positioning/solvers.py",
        "src/leo_positioning/be_gtr_solver.py",
        "src/leo_positioning/gir_tr_solver.py",
        "src/leo_positioning/cascade_refinement.py",
    }
    for path in sorted(set(paths)):
        relative = path.relative_to(RELEASE_ROOT).as_posix()
        digest = sha256_file(path)
        manifest_digest = manifest.get(relative, "")
        if manifest_digest and digest != manifest_digest:
            raise RuntimeError(f"pre-execution hash mismatch: {relative}")
        if relative in selector_names:
            category = "selector"
        elif relative in freeze_names:
            category = "freeze_policy"
        elif relative in candidate_core:
            category = "candidate_pool_core"
        elif relative.startswith("configs/"):
            category = "frozen_config"
        elif relative == "scripts/release_runtime.py":
            category = "release_execution_adapter"
        else:
            category = "candidate_pool_dependency"
        records.append(
            {
                "category": category,
                "relative_path": relative,
                "sha256_before": digest,
                "immutable_manifest_sha256": manifest_digest,
                "matches_immutable_manifest": bool(manifest_digest and digest == manifest_digest),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def load_base_geometry() -> dict[str, Any]:
    src = RELEASE_ROOT / "src"
    sys.path.insert(0, str(src))
    from leo_positioning.data import load_iridium_csv, normalize_observations

    data_path = RELEASE_ROOT / "data" / "real_iridium" / "Iridium_Doppler_measurements.csv"
    frame = load_iridium_csv(data_path)
    obs = normalize_observations(frame)
    time_s = np.asarray(obs["time_s"], dtype=float)
    satellites = np.asarray(obs["satellite_number"], dtype=int)
    return {
        "path": data_path,
        "sha256": sha256_file(data_path),
        "time_s": time_s,
        "satellites": satellites,
        "row_count": len(time_s),
        "duration_s": float(np.max(time_s) - np.min(time_s)),
        "time_min_s": float(np.min(time_s)),
        "time_max_s": float(np.max(time_s)),
        "unique_satellites": sorted(int(v) for v in np.unique(satellites)),
    }


def load_qatar_constants() -> dict[str, float]:
    path = RELEASE_ROOT / "configs" / "qatar_error_model_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    params = raw["parameters"]
    col4 = params["col4"]
    col5 = params["col5"]
    col4_z = abs(float(col4["p99"]) - float(col4["median"])) / float(col4["mad"])
    col5_z = abs(float(col5["p99"]) - float(col5["median"])) / float(col5["mad"])
    return {
        "col4_p99_mad_z": col4_z,
        "col5_p99_mad_z": col5_z,
        "h3_outlier_amplitude_mps": max(0.25, col4_z * 0.30),
        "h4_outlier_amplitude_mps": max(0.25, col5_z * 0.30),
        "student_t_df": 3.0,
        "lognormal_sigma": 0.25,
        "confidence_low_ratio": float(params["confidence"]["ratio_below_90"]),
    }


def uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(float(low), float(high)))


def build_records(base: dict[str, Any], qatar: dict[str, float]) -> list[dict[str, Any]]:
    family_names = [
        "H0_static_holdout",
        "H1_dynamic_clean_holdout",
        "H2_bias_drift_holdout",
        "H3_outlier_holdout",
        "H4_geometry_ephemeris_holdout",
        "H5_no_bias_dynamic_holdout",
    ]
    frequent_satellites = np.array([19, 35, 38, 55, 57, 59], dtype=int)
    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    children = seed_sequence.spawn(len(family_names) * 10)
    records: list[dict[str, Any]] = []
    global_index = 0
    for family_index, family in enumerate(family_names):
        for realization_index in range(10):
            state = children[global_index].generate_state(4, dtype=np.uint32)
            parameter_seed, noise_seed, ephemeris_seed, label_seed = (int(v) for v in state)
            rng = np.random.default_rng(parameter_seed)
            subset_size = 4 if family == "H4_geometry_ephemeris_holdout" else 5
            subset = sorted(int(v) for v in rng.choice(frequent_satellites, size=subset_size, replace=False))
            start_fraction = uniform(rng, 0.010, 0.080)
            end_fraction = uniform(rng, 0.920, 0.990)
            start_s = base["time_min_s"] + start_fraction * base["duration_s"]
            end_s = base["time_min_s"] + end_fraction * base["duration_s"]
            mask = (
                (base["time_s"] >= start_s)
                & (base["time_s"] <= end_s)
                & np.isin(base["satellites"], np.asarray(subset, dtype=int))
            )
            observation_count = int(np.sum(mask))
            unique_satellite_count = int(len(np.unique(base["satellites"][mask])))
            if observation_count < 120 or unique_satellite_count != subset_size:
                raise RuntimeError(
                    f"pre-frozen geometry is insufficient for {family}/{realization_index}: "
                    f"observations={observation_count}, satellites={unique_satellite_count}"
                )

            speed = 0.0
            direction_deg: float | None = None
            b0 = 0.0
            bdot = 0.0
            noise_sigma = 0.0
            outlier_ratio = 0.0
            outlier_amplitude = 0.0
            burst_start_fraction: float | None = None
            heavy_tail_enabled = False
            heavy_tail_scale_mps = 0.0
            sat_pos_sigma = 0.0
            sat_vel_sigma = 0.0
            confidence_simulated = False

            if family == "H0_static_holdout":
                b0 = uniform(rng, 0.0, 1.5)
                noise_sigma = uniform(rng, 0.15, 0.40)
            elif family == "H1_dynamic_clean_holdout":
                speed = uniform(rng, 15.0, 80.0)
                direction_deg = uniform(rng, 0.0, 360.0)
                b0 = uniform(rng, 0.0, 3.0)
                bdot = uniform(rng, 0.0, 0.08)
                noise_sigma = uniform(rng, 0.15, 0.40)
            elif family == "H2_bias_drift_holdout":
                speed = uniform(rng, 10.0, 60.0)
                direction_deg = uniform(rng, 0.0, 360.0)
                b0 = uniform(rng, 2.0, 8.0)
                bdot = uniform(rng, 0.05, 0.20)
                noise_sigma = uniform(rng, 0.20, 0.50)
            elif family == "H3_outlier_holdout":
                speed = uniform(rng, 20.0, 70.0)
                direction_deg = uniform(rng, 0.0, 360.0)
                b0 = uniform(rng, 0.0, 3.0)
                bdot = uniform(rng, 0.0, 0.08)
                noise_sigma = uniform(rng, 0.15, 0.40)
                outlier_ratio = uniform(rng, 0.03, 0.12)
                outlier_amplitude = qatar["h3_outlier_amplitude_mps"]
                burst_start_fraction = uniform(rng, 0.0, max(0.0, 1.0 - outlier_ratio))
                heavy_tail_enabled = True
                heavy_tail_scale_mps = 0.30
                confidence_simulated = True
            elif family == "H4_geometry_ephemeris_holdout":
                speed = uniform(rng, 30.0, 90.0)
                direction_deg = uniform(rng, 0.0, 360.0)
                # H4 isolates geometry/ephemeris while retaining the same bounded
                # nuisance-state range as the clean dynamic/outlier families.
                b0 = uniform(rng, 0.0, 3.0)
                bdot = uniform(rng, 0.0, 0.08)
                noise_sigma = uniform(rng, 0.30, 0.70)
                sat_pos_sigma = uniform(rng, 100.0, 800.0)
                sat_vel_sigma = uniform(rng, 0.05, 0.30)
                heavy_tail_enabled = bool(rng.random() < 0.5)
                outlier_ratio = uniform(rng, 0.01, 0.03) if heavy_tail_enabled else 0.0
                outlier_amplitude = qatar["h4_outlier_amplitude_mps"] if heavy_tail_enabled else 0.0
                heavy_tail_scale_mps = 0.30 if heavy_tail_enabled else 0.0
                confidence_simulated = True
            elif family == "H5_no_bias_dynamic_holdout":
                speed = uniform(rng, 40.0, 100.0)
                direction_deg = uniform(rng, 0.0, 360.0)
                noise_sigma = uniform(rng, 0.20, 0.50)
            else:
                raise AssertionError(family)

            duration = float(np.max(base["time_s"][mask]) - np.min(base["time_s"][mask]))
            sorted_times = np.sort(np.unique(base["time_s"][mask]))
            sampling_interval = float(np.median(np.diff(sorted_times))) if len(sorted_times) > 1 else float("nan")
            records.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "holdout_id": f"{family.split('_', 1)[0]}_R{realization_index:02d}",
                    "selector_scenario_key": f"holdout_{global_index:03d}",
                    "holdout_family": family,
                    "realization_index": realization_index,
                    "global_index": global_index,
                    "parameter_seed": parameter_seed,
                    "noise_seed": noise_seed,
                    "ephemeris_seed": ephemeris_seed,
                    "label_seed": label_seed,
                    "speed_mps": speed,
                    "direction_deg_enu": direction_deg,
                    "b0_mps": b0,
                    "bdot_mps2": bdot,
                    "noise_sigma_mps": noise_sigma,
                    "outlier_ratio": outlier_ratio,
                    "outlier_amplitude_mps": outlier_amplitude,
                    "outlier_shape": "qatar_p99_mad_lognormal_sigma_0.25" if outlier_ratio > 0.0 else "none",
                    "heavy_tail_enabled": heavy_tail_enabled,
                    "heavy_tail_distribution": "student_t_df_3" if heavy_tail_enabled else "none",
                    "heavy_tail_scale_mps": heavy_tail_scale_mps,
                    "burst_enabled": family == "H3_outlier_holdout",
                    "burst_start_fraction": burst_start_fraction,
                    "burst_length_fraction": outlier_ratio if family == "H3_outlier_holdout" else 0.0,
                    "confidence_simulated": confidence_simulated,
                    "confidence_low_ratio_target": qatar["confidence_low_ratio"] if confidence_simulated else 0.0,
                    "sat_pos_sigma_m": sat_pos_sigma,
                    "sat_vel_sigma_mps": sat_vel_sigma,
                    "satellite_subset": ";".join(str(v) for v in subset),
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "observation_count": observation_count,
                    "duration_s": duration,
                    "sampling_interval_median_s": sampling_interval,
                    "unique_satellite_count": unique_satellite_count,
                    "primary_position_init": "east_10km",
                    "primary_velocity_init": "zero_velocity",
                    "beta_prior_profile": "B0_none",
                    "candidate_pool_protocol": CANDIDATE_POOL_PROTOCOL,
                    "execution_mode": "solver_rerun",
                }
            )
            global_index += 1
    if len(records) != 60:
        raise RuntimeError(f"expected 60 records, generated {len(records)}")
    if len({record["parameter_seed"] for record in records}) != 60:
        raise RuntimeError("parameter seeds are not unique")
    return records


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_compatible(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    protected = [PROTOCOL_PATH, PROTOCOL_HASH_PATH, SCENARIO_PATH, HASH_BEFORE_PATH]
    existing = [path.name for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite frozen protocol artifacts: {existing}")
    if not RELEASE_ROOT.is_dir():
        raise RuntimeError(f"release not found: {RELEASE_ROOT}")

    manifest = parse_immutable_manifest()
    hash_records = source_hash_scope(manifest)
    base = load_base_geometry()
    qatar = load_qatar_constants()
    records = build_records(base, qatar)

    protocol = {
        "protocol_id": PROTOCOL_ID,
        "protocol_status": "frozen_before_solver_execution",
        "protocol_generation_date": "2026-07-23",
        "release": {
            "release_name": "MA-BGTR-v7.2-freeze-r2",
            "algorithm_version": "MA-BGTR-v7.1",
            "freeze_policy_version": "v7.2",
            "engineering_revision": "r2",
            "candidate_pool_protocol": CANDIDATE_POOL_PROTOCOL,
            "release_root": str(RELEASE_ROOT),
        },
        "master_seed": MASTER_SEED,
        "primary_evaluation": {
            "families": 6,
            "realizations_per_family": 10,
            "primary_runs": 60,
            "position_init": "east_10km",
            "velocity_init": "zero_velocity",
            "beta_prior_profile": "B0_none",
            "headline_half_truth_velocity_forbidden": True,
        },
        "initialization_sensitivity": {
            "realization_indices_per_family": [0, 1],
            "profiles": [
                ["east_10km", "zero_velocity", "B0_none"],
                ["east_100km", "zero_velocity", "B0_none"],
                ["east_10km", "half_truth_velocity", "B0_none"],
            ],
            "additional_solver_batches": 24,
            "reported_rows_including_primary_profile": 36,
            "headline_use": False,
        },
        "selector_modes": {
            "observable_only": {
                "primary": True,
                "allowed": [
                    "residual_statistics",
                    "blocked_validation",
                    "confidence_like_values",
                    "geometry_and_condition_diagnostics",
                    "information_retention",
                    "state_plausibility",
                    "cross_model_consistency",
                ],
                "forbidden": [
                    "injected_outlier_flag",
                    "qatar_outlier_mps_truth",
                    "dropout_truth_flag",
                    "holdout_family_expected_model",
                    "position_velocity_bias_drift_truth",
                    "oracle_model",
                ],
                "outlier_evidence_rule": "frozen_adapter residual tail_ratio>=0.05 or mad_ratio>=2.0 only",
            },
            "label_assisted_ablation": {
                "primary": False,
                "allowed_addition": "synthetic injected-outlier and burst labels mapped to frozen outlier-evidence fields",
                "claim_restriction": "ablation only; not an online receiver capability",
            },
        },
        "candidate_models": CANDIDATE_MODELS,
        "execution_mode": "solver_rerun",
        "cached_replay_allowed": False,
        "bootstrap": {
            "resamples": 1000,
            "interval": "percentile_95pct",
            "seed_basis": MASTER_SEED,
            "selector_use": False,
        },
        "paired_baselines": [
            "M0_static_position",
            "M2_ctd_full",
            "M3_ctd_no_drift",
            "M4_ctd_no_bias",
            "M7_robust_ctd_full",
            "M13_robust_ctd_full_plus_gir_refine",
        ],
        "geometry_source": {
            "relative_path": base["path"].relative_to(RELEASE_ROOT).as_posix(),
            "sha256": base["sha256"],
            "base_observation_count": base["row_count"],
            "base_duration_s": base["duration_s"],
            "base_unique_satellites": base["unique_satellites"],
            "holdout_rule": "pre-frozen trimmed time window plus custom 4- or 5-satellite subset; never all-satellite or TECH18 top-3 subset",
        },
        "qatar_shape_constants": qatar,
        "family_definitions": {
            "H0_static_holdout": {"speed_mps": [0.0, 0.0], "b0_mps": [0.0, 1.5], "bdot_mps2": [0.0, 0.0], "noise_sigma_mps": [0.15, 0.40]},
            "H1_dynamic_clean_holdout": {"speed_mps": [15.0, 80.0], "direction_deg_enu": [0.0, 360.0], "b0_mps": [0.0, 3.0], "bdot_mps2": [0.0, 0.08], "noise_sigma_mps": [0.15, 0.40], "outliers": False, "ephemeris_error": False},
            "H2_bias_drift_holdout": {"speed_mps": [10.0, 60.0], "direction_deg_enu": [0.0, 360.0], "b0_mps": [2.0, 8.0], "bdot_mps2": [0.05, 0.20], "noise_sigma_mps": [0.20, 0.50]},
            "H3_outlier_holdout": {"speed_mps": [20.0, 70.0], "direction_deg_enu": [0.0, 360.0], "b0_mps": [0.0, 3.0], "bdot_mps2": [0.0, 0.08], "noise_sigma_mps": [0.15, 0.40], "outlier_ratio": [0.03, 0.12], "shape": "Qatar DATA02 p99/MAD plus frozen Student-t/lognormal shape", "burst_start_and_length_random": True},
            "H4_geometry_ephemeris_holdout": {"speed_mps": [30.0, 90.0], "direction_deg_enu": [0.0, 360.0], "b0_mps_protocol_extension": [0.0, 3.0], "bdot_mps2_protocol_extension": [0.0, 0.08], "sat_pos_sigma_m": [100.0, 800.0], "sat_vel_sigma_mps": [0.05, 0.30], "noise_sigma_mps": [0.30, 0.70], "optional_heavy_tail_ratio": [0.01, 0.03]},
            "H5_no_bias_dynamic_holdout": {"speed_mps": [40.0, 100.0], "direction_deg_enu": [0.0, 360.0], "b0_mps": [0.0, 0.0], "bdot_mps2": [0.0, 0.0], "noise_sigma_mps": [0.20, 0.50]},
        },
        "immutable_release_manifest": {"verified_files": len(manifest), "status": "pass_before_protocol_freeze"},
        "frozen_hash_scope": [record["relative_path"] for record in hash_records],
        "scenario_records": records,
    }
    protocol = json_compatible(protocol)

    with HASH_BEFORE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hash_records[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(hash_records)
    with SCENARIO_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    protocol_bytes = (json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    PROTOCOL_PATH.write_bytes(protocol_bytes)
    protocol_digest = hashlib.sha256(protocol_bytes).hexdigest()
    PROTOCOL_HASH_PATH.write_text(f"{protocol_digest}  {PROTOCOL_PATH.name}\n", encoding="utf-8")

    # Mark both protocol artifacts read-only after hashing.  Downstream scripts
    # verify the digest and never write either file.
    os.chmod(PROTOCOL_PATH, stat.S_IREAD)
    os.chmod(PROTOCOL_HASH_PATH, stat.S_IREAD)
    print(
        json.dumps(
            {
                "status": "protocol_frozen",
                "protocol_sha256": protocol_digest,
                "primary_runs": len(records),
                "source_hash_records": len(hash_records),
                "immutable_manifest_files_verified": len(manifest),
                "min_observation_count": min(record["observation_count"] for record in records),
                "max_observation_count": max(record["observation_count"] for record in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
