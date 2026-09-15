from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


ROOT = Path('leo-c')
LAB = Path('leo-c_new_solver_lab')
EXP02A = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02a"
EXP02B = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02b_window_diagnostic"
DATA02A = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_audit"
OUT = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02c_diagnostic"
DIRS = {name: OUT / name for name in ["protocol", "outputs", "reports", "scripts", "figures", "source_snapshot", "logs"]}

PROTOCOL_VERSION = "PAPER_EXP02C_PROTOCOL_V1"
PRIMARY_PROFILE = "P0_clean_motion"
PRIMARY_SEED = 20260724
DURATIONS = [8, 12]
GEOMETRIES = ["G0", "G1"]
INIT_MODELS = [
    "M2_ctd_full",
    "M3_ctd_no_drift",
    "M4_ctd_no_bias",
    "M7_robust_ctd_full",
    "M13_robust_ctd_full_plus_gir_refine",
]
ZERO_NOISE_MODELS = ["M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias"]
OBSERVABILITY_MODELS = ["M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias"]
INIT_CATEGORIES = ["I0_cv_fit", "I1_near_100m", "I2_moderate_1km", "I3_frozen_10km"]

CAUSE_RULES = {
    "observation_consistency": {
        "max_abs_clean_prediction_difference_mps": 1e-8,
        "max_abs_noise_reconstruction_difference_mps": 1e-8,
        "minimum_sign_correlation": 0.999999,
        "max_abs_relative_time_difference_s": 1e-9,
    },
    "geometry_observability": {
        "m2_full_rank_required": 8,
        "near_rank_deficient_condition_threshold": 1e12,
        "window_fraction_for_strong_flag": 0.50,
    },
    "initialization_basin": {
        "i0_near_floor_threshold_m": "max(25 m, 5 * CV floor)",
        "i3_failure_threshold_m": "max(100 m, 3 * I0 error)",
        "window_fraction_for_strong_flag": 0.50,
    },
    "candidate_solver": {
        "i0_above_floor_threshold_m": "max(25 m, 5 * CV floor)",
        "window_fraction_for_strong_flag": 0.50,
        "requires_observation_consistency": True,
    },
    "selector_alignment": {
        "large_gap_threshold_m": "max(100 m, 0.5 * selected error)",
        "run_fraction_for_strong_flag": 0.50,
    },
    "primary_cause_policy": [
        "observation inconsistency alone -> observation_model_inconsistency_dominant",
        "two or more strong cause flags -> mixed_causes",
        "initialization only -> initialization_basin_dominant",
        "geometry only -> geometry_observability_dominant",
        "candidate solver only -> candidate_solver_dominant",
        "selector alignment only -> selector_alignment_dominant",
        "otherwise -> inconclusive",
    ],
}


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (DIRS["logs"] / "PAPER_EXP02C_EXECUTION.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    output = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_tree(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "total_size_bytes": sum(item.stat().st_size for item in files),
        "latest_mtime_ns": max((item.stat().st_mtime_ns for item in files), default=None),
    }


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if len(values) else np.nan


def corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if np.sum(mask) < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return np.nan
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    return np.column_stack((east, north, up))


def fit_motion(truth_position: np.ndarray, times: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tau = np.asarray(times, dtype=float) - float(np.min(times))
    if model == "static":
        design = np.ones((len(tau), 1))
    elif model == "cv":
        design = np.column_stack((np.ones(len(tau)), tau))
    elif model == "ca":
        design = np.column_stack((np.ones(len(tau)), tau, 0.5 * tau * tau))
    else:
        raise ValueError(model)
    coefficients, _, _, _ = np.linalg.lstsq(design, np.asarray(truth_position, dtype=float), rcond=None)
    fitted = design @ coefficients
    if model == "static":
        velocity = np.zeros_like(fitted)
    elif model == "cv":
        velocity = np.repeat(coefficients[1][None, :], len(tau), axis=0)
    else:
        velocity = coefficients[1][None, :] + tau[:, None] * coefficients[2][None, :]
    errors = np.linalg.norm(fitted - truth_position, axis=1)
    return fitted, velocity, {
        "coefficients": coefficients,
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
    }


def range_rate(receiver_position: np.ndarray, receiver_velocity: np.ndarray, sat_position: np.ndarray, sat_velocity: np.ndarray) -> np.ndarray:
    delta_position = np.asarray(receiver_position, dtype=float) - np.asarray(sat_position, dtype=float)
    delta_velocity = np.asarray(receiver_velocity, dtype=float) - np.asarray(sat_velocity, dtype=float)
    return np.sum(delta_position * delta_velocity, axis=1) / np.maximum(np.linalg.norm(delta_position, axis=1), 1e-12)


def load_observation(path: Path, anchor: dict[str, float]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(path)
    frame = frame[frame["used_by_solver"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    truth_position = frame[["receiver_true_x_m", "receiver_true_y_m", "receiver_true_z_m"]].to_numpy(float)
    truth_velocity = frame[["receiver_true_vx_mps", "receiver_true_vy_mps", "receiver_true_vz_mps"]].to_numpy(float)
    common_bias = pd.to_numeric(frame["common_bias_mps"], errors="coerce").to_numpy(float)
    obs = {
        "lat_deg": anchor["lat_deg"],
        "lon_deg": anchor["lon_deg"],
        "height_m": anchor["height_m"],
        "p_gt_ecef_m": truth_position[0],
        "p0_true_m": truth_position[0],
        "v_true_mps": np.mean(truth_velocity, axis=0),
        "truth_positions_m": truth_position,
        "truth_velocity_samples_mps": truth_velocity,
        "truth_bias_mps": common_bias,
        "b0_true_mps": float(common_bias[0]),
        "bdot_true_mps2": float(pd.to_numeric(frame["drift_mps2"], errors="coerce").iloc[0]),
        "sat_pos_m": frame[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(float),
        "sat_vel_mps": frame[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]].to_numpy(float),
        "sat_pos_truth_m": frame[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(float),
        "sat_vel_truth_mps": frame[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]].to_numpy(float),
        "meas_mps": pd.to_numeric(frame["measured_range_rate_mps"], errors="coerce").to_numpy(float),
        "time_s": pd.to_numeric(frame["observation_time_s"], errors="coerce").to_numpy(float),
        "satellite_number": frame["satellite_id"].to_numpy(),
        "row_index": np.arange(len(frame)),
        "t0_s": float(pd.to_numeric(frame["observation_time_s"], errors="coerce").min()),
    }
    return obs, frame


def required_exp02b_paths() -> list[Path]:
    relative = [
        "protocol/PAPER_EXP02B_PROTOCOL.json",
        "protocol/PAPER_EXP02B_PROTOCOL_SHA256.txt",
        "protocol/PAPER_EXP02B_GEOMETRY_WINDOW_PROTOCOL.csv",
        "protocol/PAPER_EXP02B_FROZEN_SOURCE_AUDIT.csv",
        "outputs/PAPER_EXP02B_SELECTED_RESULTS.csv",
        "outputs/PAPER_EXP02B_CANDIDATE_RESULTS.csv",
        "outputs/PAPER_EXP02B_ERROR_ATTRIBUTION.csv",
        "outputs/PAPER_EXP02B_MOTION_MODEL_REPRESENTATION_FLOORS.csv",
        "outputs/PAPER_EXP02B_METRICS.json",
    ]
    paths = [EXP02B / item for item in relative]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing required EXP02B inputs: {missing}")
    return paths


def audit_inputs() -> dict[str, Any]:
    required = required_exp02b_paths()
    protocol_path = EXP02B / "protocol" / "PAPER_EXP02B_PROTOCOL.json"
    protocol_digest = sha256_file(protocol_path)
    expected_digest = (EXP02B / "protocol" / "PAPER_EXP02B_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").split()[0]
    if protocol_digest != expected_digest:
        raise RuntimeError("EXP02B protocol SHA256 mismatch")
    exp02b_protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    release = Path(exp02b_protocol["frozen_release_path"])
    if not release.is_dir():
        raise FileNotFoundError(f"Frozen release missing: {release}")
    runtime = import_module("paper_exp02c_release_runtime", release / "scripts" / "release_runtime.py")
    exp02b_module = import_module(
        "paper_exp02c_exp02b_helpers", EXP02B / "scripts" / "paper_exp02b_run_window_diagnostic.py"
    )

    source_audit_b = pd.read_csv(EXP02B / "protocol" / "PAPER_EXP02B_FROZEN_SOURCE_AUDIT.csv")
    source_rows: list[dict[str, Any]] = []
    for _, row in source_audit_b.iterrows():
        source = Path(str(row["source_path"]))
        expected = str(row["expected_sha256"])
        current = sha256_file(source)
        if current != expected or current != str(row["sha256_after"]):
            raise RuntimeError(f"Frozen source hash mismatch: {source}")
        relative = Path("scripts") / source.name if source.name == "release_runtime.py" else Path("leo_positioning") / source.name
        destination = DIRS["source_snapshot"] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_rows.append(
            {
                "component": row["component"],
                "source_path": str(source),
                "expected_sha256": expected,
                "sha256_before": current,
                "sha256_after": "pending",
                "unchanged_during_exp02c": "pending",
            }
        )
    source_audit = pd.DataFrame(source_rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02C_FROZEN_SOURCE_AUDIT.csv", source_audit)

    geometry_hash = exp02b_protocol["source_hashes"]["canonical_iridium_geometry"]
    geometry_candidates = [
        release / "data" / "real_iridium" / "Iridium.csv",
        release / "data" / "real_iridium" / "Iridium_Doppler_measurements.csv",
    ]
    geometry_path = next((path for path in geometry_candidates if path.is_file() and sha256_file(path) == geometry_hash), None)
    if geometry_path is None:
        raise RuntimeError("Canonical Iridium geometry SHA256 could not be matched")
    geometry_frame = runtime.load_iridium_csv(geometry_path)
    base_obs = runtime.normalize_observations(geometry_frame)
    anchor = {
        "lat_deg": float(base_obs["lat_deg"]),
        "lon_deg": float(base_obs["lon_deg"]),
        "height_m": float(base_obs["height_m"]),
        "ecef_m": np.asarray(base_obs["p_gt_ecef_m"], dtype=float),
    }

    urban_hashes = exp02b_protocol["source_hashes"]["urban_nav"]
    urban_paths = {
        "Odaiba": DATA02A / "processed" / "UrbanNav_TK_Odaiba_reference_standardized.csv",
        "Shinjuku": DATA02A / "processed" / "UrbanNav_TK_Shinjuku_reference_standardized.csv",
    }
    for route, path in urban_paths.items():
        if sha256_file(path) != urban_hashes[route]:
            raise RuntimeError(f"UrbanNav standardized input changed: {route}")

    p0_paths = sorted((EXP02B / "observations").glob(f"*_P0_clean_motion_{PRIMARY_SEED}.csv"))
    p1_paths = sorted((EXP02B / "observations").glob(f"*_P1_bias_drift_{PRIMARY_SEED}.csv"))
    if len(p0_paths) != 32:
        raise RuntimeError(f"Expected 32 P0 observation windows, found {len(p0_paths)}")
    window_rows: list[dict[str, Any]] = []
    for path in p0_paths:
        frame = pd.read_csv(path, nrows=1)
        window_rows.append(
            {
                "run_id": frame.iloc[0]["run_id"],
                "route": frame.iloc[0]["route"],
                "source_segment_id": frame.iloc[0]["source_segment_id"],
                "motion_type": frame.iloc[0]["motion_type"],
                "duration_s": int(frame.iloc[0]["duration_s"]),
                "geometry_window_id": frame.iloc[0]["geometry_window_id"],
                "observation_path": str(path),
                "observation_sha256": sha256_file(path),
                "row_count": sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1,
            }
        )
    windows = pd.DataFrame(window_rows).sort_values(["source_segment_id", "duration_s", "geometry_window_id"])
    if sorted(windows["duration_s"].unique().tolist()) != DURATIONS:
        raise RuntimeError("P0 windows are not exactly 8 s and 12 s")
    if sorted(windows["geometry_window_id"].unique().tolist()) != GEOMETRIES:
        raise RuntimeError("P0 windows are not exactly G0 and G1")
    if windows["source_segment_id"].nunique() != 8:
        raise RuntimeError("P0 windows do not cover all eight frozen segments")
    write_csv(DIRS["protocol"] / "PAPER_EXP02C_WINDOW_INVENTORY.csv", windows)

    selected_b = pd.read_csv(EXP02B / "outputs" / "PAPER_EXP02B_SELECTED_RESULTS.csv")
    candidates_b = pd.read_csv(EXP02B / "outputs" / "PAPER_EXP02B_CANDIDATE_RESULTS.csv")
    p0_selected = selected_b[
        (selected_b["perturbation_profile"] == PRIMARY_PROFILE)
        & (pd.to_numeric(selected_b["seed"], errors="coerce") == PRIMARY_SEED)
    ]
    p0_candidates = candidates_b[
        (candidates_b["perturbation_profile"] == PRIMARY_PROFILE)
        & (pd.to_numeric(candidates_b["seed"], errors="coerce") == PRIMARY_SEED)
    ]
    if len(p0_selected) != 32 or len(p0_candidates) != 32 * 15:
        raise RuntimeError("EXP02B P0 selected/candidate rows are incomplete")
    if not (p0_candidates.groupby("run_id")["candidate_model"].nunique() == 15).all():
        raise RuntimeError("At least one EXP02B P0 run lacks the full candidate pool")

    protected_paths = {
        "frozen_release": release,
        "paper_exp02a": EXP02A,
        "paper_exp02b": EXP02B,
        "manuscript_v0_7": LAB / "paper_draft" / "Acta_Astronautica_MA_BGTR" / "manuscript_v0_7_codex",
    }
    protected_before = {name: snapshot_tree(path) for name, path in protected_paths.items()}
    exp02b_hashes = {path.relative_to(EXP02B).as_posix(): sha256_file(path) for path in required}
    write_json(DIRS["protocol"] / "PAPER_EXP02C_PROTECTED_BEFORE.json", protected_before)
    input_audit = {
        "exp02b_protocol_sha256": protocol_digest,
        "selector_sha256": exp02b_protocol["selector_sha256"],
        "freeze_policy_sha256": exp02b_protocol["freeze_policy_sha256"],
        "candidate_solver_hashes_match": True,
        "geometry_source": str(geometry_path),
        "geometry_source_sha256": geometry_hash,
        "urban_nav_hashes": urban_hashes,
        "exp02b_required_file_hashes": exp02b_hashes,
        "p0_observation_windows": len(p0_paths),
        "p1_observation_windows_available": len(p1_paths),
        "input_audit_pass": True,
    }
    write_json(DIRS["protocol"] / "PAPER_EXP02C_INPUT_AUDIT.json", input_audit)
    return {
        "release": release,
        "runtime": runtime,
        "exp02b_module": exp02b_module,
        "exp02b_protocol": exp02b_protocol,
        "windows": windows,
        "p0_paths": p0_paths,
        "p1_paths": p1_paths,
        "anchor": anchor,
        "source_audit": source_audit,
        "protected_paths": protected_paths,
        "protected_before": protected_before,
        "exp02b_required_hashes": exp02b_hashes,
        "p0_selected": p0_selected.copy(),
        "p0_candidates": p0_candidates.copy(),
    }


def freeze_protocol(audit: dict[str, Any]) -> tuple[dict[str, Any], str]:
    windows = audit["windows"]
    protocol = {
        "task_name": "PAPER-EXP02C",
        "protocol_version": PROTOCOL_VERSION,
        "created_before_any_diagnostic_solver_execution": True,
        "frozen_release_path": str(audit["release"]),
        "algorithm_version": audit["exp02b_protocol"]["algorithm_version"],
        "candidate_pool_protocol": audit["exp02b_protocol"]["candidate_pool_protocol"],
        "selector_sha256": audit["exp02b_protocol"]["selector_sha256"],
        "freeze_policy_sha256": audit["exp02b_protocol"]["freeze_policy_sha256"],
        "critical_source_hashes": audit["exp02b_protocol"]["critical_source_hashes"],
        "urban_nav_input_hashes": audit["exp02b_protocol"]["source_hashes"]["urban_nav"],
        "geometry_source_sha256": audit["exp02b_protocol"]["source_hashes"]["canonical_iridium_geometry"],
        "exp02b_required_file_hashes": audit["exp02b_required_hashes"],
        "window_scope": {
            "durations_s": DURATIONS,
            "geometry_windows": GEOMETRIES,
            "segment_ids": sorted(windows["source_segment_id"].unique().tolist()),
            "window_count": len(windows),
            "excluded_unavailable_durations_s": [4, 6],
            "exclusion_reason": "frozen EXP02B four-satellite and 20-observation rule was not met",
        },
        "primary_profile": PRIMARY_PROFILE,
        "primary_seed": PRIMARY_SEED,
        "p1_auxiliary_rule": "P1 observation-only consistency check is run only if P0 observation consistency is established; no P1 tuning or primary conclusion replacement.",
        "observability_models": OBSERVABILITY_MODELS,
        "observability_subsets": ["full", "train_first_70pct", "validation_last_30pct"],
        "jacobian_rank_tolerance": "max(J.shape) * machine_epsilon * largest_singular_value",
        "initialization_categories": {
            "I0_cv_fit": "post-evaluation CV-fit p0 and v; truth-assisted diagnostic",
            "I1_near_100m": "CV-fit p0 + 100 m local east, CV-fit v; truth-assisted diagnostic",
            "I2_moderate_1km": "CV-fit p0 + 1 km local east, zero v; truth-derived diagnostic",
            "I3_frozen_10km": "frozen cold start: first truth/anchor + 10 km local east, zero v",
        },
        "initialization_models": INIT_MODELS,
        "zero_noise_models": ZERO_NOISE_MODELS,
        "zero_noise_initializations": ["I0_cv_fit", "I3_frozen_10km"],
        "planned_initialization_batches": len(windows) * len(INIT_CATEGORIES),
        "planned_zero_noise_batches": len(windows) * 2,
        "cause_decision_rules": CAUSE_RULES,
        "selector_alignment": "re-run frozen selector on truth-scrubbed existing I3 full-pool candidates; post-selection truth ranks only",
        "theoretical_floors": "static/CV/CA are post-evaluation diagnostics only and never enter candidates or selector",
        "paper_placement_policy": {
            "observation_model_inconsistency_dominant": "do_not_include_until_generation_is_repaired",
            "mixed_causes": "supplementary_limitation_analysis_only",
            "initialization_basin_dominant": "supplementary_limitation_analysis_only",
            "geometry_observability_dominant": "supplementary_limitation_analysis_only",
            "candidate_solver_dominant": "supplementary_limitation_analysis_only",
            "selector_alignment_dominant": "supplementary_limitation_analysis_only",
            "inconclusive": "do_not_use_as_manuscript_evidence",
        },
        "prohibited_adaptations": [
            "solver, selector, freeze-policy, candidate-pool, gate, or threshold changes",
            "result-dependent tuning",
            "truth, position error, oracle, route, segment, or motion type in selection",
            "new positioning state model",
            "fabricating 4 s or 6 s geometry windows",
        ],
        "prohibited_claims": [
            "real dynamic LEO field validation",
            "native UrbanNav LEO Doppler",
            "CA floor as candidate solver",
            "deployment performance from I0/I1",
        ],
    }
    path = DIRS["protocol"] / "PAPER_EXP02C_PROTOCOL.json"
    payload = json.dumps(json_ready(protocol), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError("Existing immutable EXP02C protocol differs; a new protocol version is required")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    digest = sha256_file(path)
    hash_path = DIRS["protocol"] / "PAPER_EXP02C_PROTOCOL_SHA256.txt"
    expected = f"{digest}  PAPER_EXP02C_PROTOCOL.json\n"
    if hash_path.exists() and hash_path.read_text(encoding="utf-8") != expected:
        raise RuntimeError("Existing EXP02C protocol hash record differs")
    if not hash_path.exists():
        hash_path.write_text(expected, encoding="utf-8")
    return protocol, digest


def observation_consistency_one(path: Path, anchor: dict[str, float]) -> dict[str, Any]:
    obs, frame = load_observation(path, anchor)
    truth_position = np.asarray(obs["truth_positions_m"], dtype=float)
    truth_velocity = np.asarray(obs["truth_velocity_samples_mps"], dtype=float)
    sat_position = np.asarray(obs["sat_pos_m"], dtype=float)
    sat_velocity = np.asarray(obs["sat_vel_mps"], dtype=float)
    recomputed_clean = range_rate(truth_position, truth_velocity, sat_position, sat_velocity)
    saved_clean = pd.to_numeric(frame["clean_range_rate_mps"], errors="coerce").to_numpy(float)
    measured = np.asarray(obs["meas_mps"], dtype=float)
    noise = pd.to_numeric(frame["gaussian_noise_mps"], errors="coerce").to_numpy(float)
    common_bias = pd.to_numeric(frame["common_bias_mps"], errors="coerce").to_numpy(float)
    relative = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(float)
    time_values = np.asarray(obs["time_s"], dtype=float)
    relative_recomputed = time_values - float(np.min(time_values))

    static_position, static_velocity, static_fit = fit_motion(truth_position, time_values, "static")
    cv_position, cv_velocity, cv_fit = fit_motion(truth_position, time_values, "cv")
    ca_position, ca_velocity, ca_fit = fit_motion(truth_position, time_values, "ca")
    static_clean = range_rate(static_position, static_velocity, sat_position, sat_velocity)
    cv_clean = range_rate(cv_position, cv_velocity, sat_position, sat_velocity)
    ca_clean = range_rate(ca_position, ca_velocity, sat_position, sat_velocity)
    adjusted_measurement = measured - common_bias
    order = np.argsort(time_values)
    split = min(max(int(np.floor(0.70 * len(order))), 1), len(order) - 1)
    validation_mask = np.zeros(len(order), dtype=bool)
    validation_mask[order[split:]] = True
    clean_difference = recomputed_clean - saved_clean
    noise_reconstruction = measured - recomputed_clean - common_bias - noise
    raw_truth_residual = measured - recomputed_clean
    adjusted_truth_residual = adjusted_measurement - recomputed_clean
    profile = str(frame.iloc[0]["perturbation_profile"])
    thresholds = CAUSE_RULES["observation_consistency"]
    sign_correlation = corr(recomputed_clean, saved_clean)
    max_clean_difference = float(np.max(np.abs(clean_difference)))
    max_noise_difference = float(np.max(np.abs(noise_reconstruction)))
    max_time_difference = float(np.max(np.abs(relative - relative_recomputed)))
    inconsistent = bool(
        max_clean_difference > thresholds["max_abs_clean_prediction_difference_mps"]
        or max_noise_difference > thresholds["max_abs_noise_reconstruction_difference_mps"]
        or (np.isfinite(sign_correlation) and sign_correlation < thresholds["minimum_sign_correlation"])
        or max_time_difference > thresholds["max_abs_relative_time_difference_s"]
    )
    return {
        "run_id": frame.iloc[0]["run_id"],
        "route": frame.iloc[0]["route"],
        "source_segment_id": frame.iloc[0]["source_segment_id"],
        "motion_type": frame.iloc[0]["motion_type"],
        "duration_s": int(frame.iloc[0]["duration_s"]),
        "geometry_window_id": frame.iloc[0]["geometry_window_id"],
        "perturbation_profile": profile,
        "seed": int(frame.iloc[0]["seed"]),
        "observation_count": len(frame),
        "unique_satellites": int(frame["satellite_id"].nunique()),
        "max_abs_clean_prediction_difference_mps": max_clean_difference,
        "mean_abs_clean_prediction_difference_mps": float(np.mean(np.abs(clean_difference))),
        "clean_truth_residual_rmse_mps": rmse(clean_difference),
        "measured_truth_residual_rmse_mps": rmse(raw_truth_residual),
        "bias_adjusted_measured_truth_residual_rmse_mps": rmse(adjusted_truth_residual),
        "injected_noise_rmse_mps": rmse(noise),
        "max_abs_measured_noise_reconstruction_difference_mps": max_noise_difference,
        "measured_noise_reconstruction_difference_rmse_mps": rmse(noise_reconstruction),
        "clean_prediction_sign_correlation": sign_correlation,
        "negative_sign_correlation": corr(-recomputed_clean, saved_clean),
        "max_abs_relative_time_difference_s": max_time_difference,
        "time_monotonic": bool(np.all(np.diff(time_values) >= 0.0)),
        "static_clean_residual_rmse_mps": rmse(saved_clean - static_clean),
        "cv_clean_residual_rmse_mps": rmse(saved_clean - cv_clean),
        "cv_measured_residual_rmse_mps": rmse(adjusted_measurement - cv_clean),
        "cv_blocked_validation_rmse_mps": rmse((adjusted_measurement - cv_clean)[validation_mask]),
        "ca_clean_residual_rmse_mps": rmse(saved_clean - ca_clean),
        "static_floor_mean_error_m": static_fit["mean_error_m"],
        "static_floor_p95_error_m": static_fit["p95_error_m"],
        "cv_floor_mean_error_m": cv_fit["mean_error_m"],
        "cv_floor_p95_error_m": cv_fit["p95_error_m"],
        "ca_floor_mean_error_m": ca_fit["mean_error_m"],
        "ca_floor_p95_error_m": ca_fit["p95_error_m"],
        "cv_fit_p0_ecef_json": json.dumps(cv_fit["coefficients"][0].tolist()),
        "cv_fit_velocity_ecef_json": json.dumps(cv_fit["coefficients"][1].tolist()),
        "observation_model_inconsistency": inconsistent,
        "observation_consistency_status": "observation_model_inconsistency" if inconsistent else "consistent_with_saved_generation",
        "source_observation_file": str(path),
        "source_observation_sha256": sha256_file(path),
    }


def run_observation_consistency(audit: dict[str, Any]) -> pd.DataFrame:
    p0_rows = [observation_consistency_one(path, audit["anchor"]) for path in audit["p0_paths"]]
    p0 = pd.DataFrame(p0_rows)
    p0_pass = not p0["observation_model_inconsistency"].astype(bool).any()
    rows = list(p0_rows)
    if p0_pass and len(audit["p1_paths"]) == 32:
        log("P0 observation generation is consistent; running the pre-registered P1 observation-only auxiliary check")
        rows.extend(observation_consistency_one(path, audit["anchor"]) for path in audit["p1_paths"])
    frame = pd.DataFrame(rows)
    frame["diagnostic_role"] = np.where(
        frame["perturbation_profile"] == PRIMARY_PROFILE,
        "primary_clean_motion",
        "auxiliary_bias_drift_observation_only",
    )
    write_csv(DIRS["outputs"] / "PAPER_EXP02C_OBSERVATION_CONSISTENCY.csv", frame)
    return frame


def _column_coherence(jacobian: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = np.divide(jacobian, norms[None, :], out=np.zeros_like(jacobian), where=norms[None, :] > 0)
    return normalized.T @ normalized


def _max_cross_correlation(matrix: np.ndarray, left: list[int], right: list[int]) -> float:
    if not left or not right:
        return np.nan
    block = np.abs(matrix[np.ix_(left, right)])
    return float(np.max(block)) if block.size else np.nan


def observability_for_window(
    path: Path,
    anchor: dict[str, float],
) -> list[dict[str, Any]]:
    from leo_positioning.trajectory_models import (
        FULL_CTD_CONFIG,
        NO_BDOT_CONFIG,
        NO_BIAS_CONFIG,
        pack_state,
        residuals_and_jacobian_ctd,
    )

    obs, frame = load_observation(path, anchor)
    truth_position = np.asarray(obs["truth_positions_m"], dtype=float)
    _cv_positions, _cv_velocities, cv_fit = fit_motion(truth_position, obs["time_s"], "cv")
    p0 = np.asarray(cv_fit["coefficients"][0], dtype=float)
    velocity = np.asarray(cv_fit["coefficients"][1], dtype=float)
    configs = {
        "M2_ctd_full": (FULL_CTD_CONFIG, ["p0_x", "p0_y", "p0_z", "v_x", "v_y", "v_z", "b0", "bdot"]),
        "M3_ctd_no_drift": (NO_BDOT_CONFIG, ["p0_x", "p0_y", "p0_z", "v_x", "v_y", "v_z", "b0"]),
        "M4_ctd_no_bias": (NO_BIAS_CONFIG, ["p0_x", "p0_y", "p0_z", "v_x", "v_y", "v_z"]),
    }
    time_values = np.asarray(obs["time_s"], dtype=float)
    order = np.argsort(time_values)
    split = min(max(int(np.floor(0.70 * len(order))), 1), len(order) - 1)
    masks = {
        "full": np.ones(len(time_values), dtype=bool),
        "train_first_70pct": np.isin(np.arange(len(time_values)), order[:split]),
        "validation_last_30pct": np.isin(np.arange(len(time_values)), order[split:]),
    }
    rows: list[dict[str, Any]] = []
    for model, (config, labels) in configs.items():
        theta = pack_state(
            p0,
            velocity,
            float(obs["b0_true_mps"]) if config.estimate_b0 else 0.0,
            float(obs["bdot_true_mps2"]) if config.estimate_bdot else 0.0,
            config,
        )
        for subset_name, mask in masks.items():
            residual, jacobian, _prediction = residuals_and_jacobian_ctd(
                theta,
                time_values[mask],
                np.asarray(obs["sat_pos_m"])[mask],
                np.asarray(obs["sat_vel_mps"])[mask],
                np.asarray(obs["meas_mps"])[mask],
                float(obs["t0_s"]),
                config,
            )
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            largest = float(singular_values[0]) if len(singular_values) else np.nan
            smallest = float(singular_values[-1]) if len(singular_values) else np.nan
            tolerance = max(jacobian.shape) * np.finfo(float).eps * largest if np.isfinite(largest) else np.nan
            rank = int(np.sum(singular_values > tolerance)) if np.isfinite(tolerance) else 0
            condition = float(largest / smallest) if np.isfinite(smallest) and smallest > 0 else np.inf
            normal_condition = condition * condition if np.isfinite(condition) else np.inf
            coherence = _column_coherence(jacobian)
            normal = jacobian.T @ jacobian
            dof = max(len(residual) - jacobian.shape[1], 1)
            sigma2 = float(np.sum(residual * residual) / dof)
            covariance = np.linalg.pinv(normal, rcond=1e-12) * max(sigma2, 1e-12)
            position_covariance_trace = float(max(np.trace(covariance[:3, :3]), 0.0))
            b0_indices = [labels.index("b0")] if "b0" in labels else []
            bdot_indices = [labels.index("bdot")] if "bdot" in labels else []
            position_indices = [0, 1, 2]
            velocity_indices = [3, 4, 5]
            rows.append(
                {
                    "run_id": frame.iloc[0]["run_id"],
                    "route": frame.iloc[0]["route"],
                    "source_segment_id": frame.iloc[0]["source_segment_id"],
                    "motion_type": frame.iloc[0]["motion_type"],
                    "duration_s": int(frame.iloc[0]["duration_s"]),
                    "geometry_window_id": frame.iloc[0]["geometry_window_id"],
                    "candidate_model": model,
                    "observation_subset": subset_name,
                    "jacobian_rows": jacobian.shape[0],
                    "jacobian_columns": jacobian.shape[1],
                    "effective_numerical_rank": rank,
                    "full_column_rank": rank == jacobian.shape[1],
                    "rank_tolerance": tolerance,
                    "singular_values_json": json.dumps(singular_values.tolist()),
                    "smallest_singular_value": smallest,
                    "largest_singular_value": largest,
                    "condition_number": condition,
                    "normal_matrix_condition_number": normal_condition,
                    "position_subspace_information_trace": float(np.trace(normal[:3, :3])),
                    "velocity_subspace_information_trace": float(np.trace(normal[3:6, 3:6])),
                    "bias_drift_coherence": _max_cross_correlation(coherence, b0_indices, bdot_indices),
                    "velocity_drift_coherence": _max_cross_correlation(coherence, velocity_indices, bdot_indices),
                    "position_bias_coherence": _max_cross_correlation(coherence, position_indices, b0_indices),
                    "position_velocity_coherence": _max_cross_correlation(coherence, position_indices, velocity_indices),
                    "approx_position_cov_sqrt_trace_m": float(np.sqrt(position_covariance_trace)),
                    "column_labels_json": json.dumps(labels),
                    "column_correlation_matrix_json": json.dumps(coherence.tolist()),
                    "cv_fit_residual_rmse_mps": rmse(residual),
                    "cv_floor_mean_error_m": cv_fit["mean_error_m"],
                    "residual_jacobian_sign_convention": "residual=measurement-prediction; frozen analytic CTD Jacobian",
                    "truth_state_used_for_observability_only": True,
                }
            )
    return rows


def run_local_observability(audit: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(audit["p0_paths"], start=1):
        if index == 1 or index % 8 == 0 or index == len(audit["p0_paths"]):
            log(f"Local observability {index}/{len(audit['p0_paths'])}: {path.stem}")
        rows.extend(observability_for_window(path, audit["anchor"]))
    frame = pd.DataFrame(rows)
    write_csv(DIRS["outputs"] / "PAPER_EXP02C_LOCAL_OBSERVABILITY.csv", frame)
    return frame


def initialization_states(obs: dict[str, Any], anchor: dict[str, float]) -> dict[str, dict[str, Any]]:
    _cv_position, _cv_velocity, cv_fit = fit_motion(obs["truth_positions_m"], obs["time_s"], "cv")
    cv_p0 = np.asarray(cv_fit["coefficients"][0], dtype=float)
    cv_v = np.asarray(cv_fit["coefficients"][1], dtype=float)
    east = enu_to_ecef_rotation(anchor["lat_deg"], anchor["lon_deg"])[:, 0]
    return {
        "I0_cv_fit": {
            "p0": cv_p0,
            "v0": cv_v,
            "truth_assisted": True,
            "reference": "post-evaluation CV-fit p0 and velocity",
        },
        "I1_near_100m": {
            "p0": cv_p0 + 100.0 * east,
            "v0": cv_v,
            "truth_assisted": True,
            "reference": "post-evaluation CV-fit plus 100 m local east",
        },
        "I2_moderate_1km": {
            "p0": cv_p0 + 1000.0 * east,
            "v0": np.zeros(3),
            "truth_assisted": True,
            "reference": "post-evaluation CV-fit p0 plus 1 km local east; zero velocity",
        },
        "I3_frozen_10km": {
            "p0": np.asarray(obs["p0_true_m"], dtype=float) + 10_000.0 * east,
            "v0": np.zeros(3),
            "truth_assisted": False,
            "reference": "frozen controlled cold start: first trajectory point plus 10 km local east; zero velocity",
        },
    }


def custom_candidate_batch(
    runtime: Any,
    exp02b_module: Any,
    obs: dict[str, Any],
    run_id: str,
    init_name: str,
    init: dict[str, Any],
    models: list[str],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    initial_guess = {
        "dataset_type": "urban_real_trajectory_diagnostic",
        "scenario": run_id,
        "position_init_label": init_name,
        "velocity_init_label": init_name,
        "beta_prior_profile": "B0_none",
        "p0_init": np.asarray(init["p0"], dtype=float),
        "v_init": np.asarray(init["v0"], dtype=float),
    }
    started = time.perf_counter()
    rows, _robust_diag, _refine_diag = runtime.build_v4_candidate_rows_for_job(obs, initial_guess, models)
    frame = runtime._enrich_risk(pd.DataFrame(rows), obs)
    frame = runtime._add_observable_outlier_evidence(frame, metadata)
    frame["execution_mode"] = "solver_rerun"
    frame["candidate_pool_protocol"] = "DIAGNOSTIC_SUBSET_FROM_FULL_POOL_M0_M14_V1"
    frame["runtime_seconds"] = pd.to_numeric(frame.get("runtime_ms", np.nan), errors="coerce") / 1000.0
    frame["iterations"] = pd.to_numeric(frame.get("iterations", np.nan), errors="coerce")
    if "residual_rmse_mps" not in frame.columns:
        frame["residual_rmse_mps"] = pd.to_numeric(frame.get("full_residual_rmse_mps", np.nan), errors="coerce")
    frame = exp02b_module.add_post_selection_candidate_metrics(frame, obs)
    residual_rmse = pd.to_numeric(frame["residual_rmse_mps"], errors="coerce")
    frame["final_objective"] = 0.5 * len(obs["time_s"]) * residual_rmse * residual_rmse
    frame["final_objective_definition"] = "0.5 * n * full residual RMSE^2; post-run unweighted observation-domain diagnostic"
    frame["initialization_category"] = init_name
    frame["truth_assisted_initialization_flag"] = bool(init["truth_assisted"])
    frame["initialization_reference"] = init["reference"]
    frame["initial_p0_ecef_json"] = json.dumps(np.asarray(init["p0"], dtype=float).tolist())
    frame["initial_v_ecef_json"] = json.dumps(np.asarray(init["v0"], dtype=float).tolist())
    frame["candidate_batch_runtime_seconds"] = time.perf_counter() - started
    frame["truth_used_for_selection"] = False
    return frame


def run_initialization_basin(audit: dict[str, Any], consistency: pd.DataFrame) -> pd.DataFrame:
    runtime = audit["runtime"]
    helper = audit["exp02b_module"]
    floor_map = consistency[consistency["perturbation_profile"] == PRIMARY_PROFILE].set_index("run_id")["cv_floor_mean_error_m"].to_dict()
    output: list[pd.DataFrame] = []
    for index, path in enumerate(audit["p0_paths"], start=1):
        obs, observation_frame = load_observation(path, audit["anchor"])
        run_id = str(observation_frame.iloc[0]["run_id"])
        states = initialization_states(obs, audit["anchor"])
        for init_index, init_name in enumerate(INIT_CATEGORIES, start=1):
            log(f"Initialization basin {index}/{len(audit['p0_paths'])} {init_index}/4: {run_id} {init_name}")
            metadata = {
                "dataset_type": "urban_real_trajectory_diagnostic",
                "outlier_ratio": 0.0,
                "dropout_ratio": 0.0,
                "burst_outlier": False,
            }
            frame = custom_candidate_batch(runtime, helper, obs, run_id, init_name, states[init_name], INIT_MODELS, metadata)
            frame["run_id"] = run_id
            frame["route"] = observation_frame.iloc[0]["route"]
            frame["source_segment_id"] = observation_frame.iloc[0]["source_segment_id"]
            frame["motion_type"] = observation_frame.iloc[0]["motion_type"]
            frame["duration_s"] = int(observation_frame.iloc[0]["duration_s"])
            frame["geometry_window_id"] = observation_frame.iloc[0]["geometry_window_id"]
            frame["perturbation_profile"] = PRIMARY_PROFILE
            frame["seed"] = PRIMARY_SEED
            frame["cv_floor_mean_error_m"] = float(floor_map[run_id])
            frame["distance_to_cv_floor_m"] = frame["mean_trajectory_position_error_m"] - frame["cv_floor_mean_error_m"]
            output.append(frame)
    result = pd.concat(output, ignore_index=True)
    write_csv(DIRS["outputs"] / "PAPER_EXP02C_INITIALIZATION_BASIN.csv", result)
    return result


def zero_noise_observation(obs: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in obs.items():
        output[key] = value.copy() if isinstance(value, np.ndarray) else value
    clean = range_rate(output["truth_positions_m"], output["truth_velocity_samples_mps"], output["sat_pos_m"], output["sat_vel_mps"])
    output["meas_mps"] = clean
    output["truth_bias_mps"] = np.zeros(len(clean))
    output["b0_true_mps"] = 0.0
    output["bdot_true_mps2"] = 0.0
    return output


def run_zero_noise(audit: dict[str, Any], consistency: pd.DataFrame) -> pd.DataFrame:
    runtime = audit["runtime"]
    helper = audit["exp02b_module"]
    floor_map = consistency[consistency["perturbation_profile"] == PRIMARY_PROFILE].set_index("run_id")["cv_floor_mean_error_m"].to_dict()
    output: list[pd.DataFrame] = []
    for index, path in enumerate(audit["p0_paths"], start=1):
        base_obs, observation_frame = load_observation(path, audit["anchor"])
        obs = zero_noise_observation(base_obs)
        run_id = str(observation_frame.iloc[0]["run_id"])
        states = initialization_states(obs, audit["anchor"])
        for init_name in ["I0_cv_fit", "I3_frozen_10km"]:
            log(f"Zero-noise diagnostic {index}/{len(audit['p0_paths'])}: {run_id} {init_name}")
            metadata = {
                "dataset_type": "urban_real_trajectory_diagnostic",
                "outlier_ratio": 0.0,
                "dropout_ratio": 0.0,
                "burst_outlier": False,
            }
            frame = custom_candidate_batch(runtime, helper, obs, run_id, init_name, states[init_name], ZERO_NOISE_MODELS, metadata)
            frame["run_id"] = run_id
            frame["route"] = observation_frame.iloc[0]["route"]
            frame["source_segment_id"] = observation_frame.iloc[0]["source_segment_id"]
            frame["motion_type"] = observation_frame.iloc[0]["motion_type"]
            frame["duration_s"] = int(observation_frame.iloc[0]["duration_s"])
            frame["geometry_window_id"] = observation_frame.iloc[0]["geometry_window_id"]
            frame["noise_sigma_mps"] = 0.0
            frame["b0_mps"] = 0.0
            frame["bdot_mps2"] = 0.0
            frame["cv_floor_mean_error_m"] = float(floor_map[run_id])
            frame["distance_to_cv_floor_m"] = frame["mean_trajectory_position_error_m"] - frame["cv_floor_mean_error_m"]
            frame["near_cv_floor"] = frame["mean_trajectory_position_error_m"] <= np.maximum(25.0, 5.0 * frame["cv_floor_mean_error_m"])
            output.append(frame)
    result = pd.concat(output, ignore_index=True)
    write_csv(DIRS["outputs"] / "PAPER_EXP02C_ZERO_NOISE_DIAGNOSTIC.csv", result)
    return result


def _rank_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    subset = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 3:
        return np.nan
    return float(subset[left].rank(method="average").corr(subset[right].rank(method="average")))


def selector_alignment(audit: dict[str, Any], initialization: pd.DataFrame) -> pd.DataFrame:
    runtime = audit["runtime"]
    rows: list[dict[str, Any]] = []
    truth_columns = [
        "final_position_error_m",
        "mean_position_error_m",
        "max_position_error_m",
        "velocity_error_mps",
        "beta0_error_mps",
        "beta_dot_error_mps2",
        "initial_position_error_m",
        "mean_trajectory_position_error_m",
        "median_trajectory_position_error_m",
        "p95_trajectory_position_error_m",
        "max_trajectory_position_error_m",
        "mean_velocity_error_mps",
        "p95_velocity_error_mps",
    ]
    selected_reference = audit["p0_selected"].set_index("run_id")
    for run_id, group in audit["p0_candidates"].groupby("run_id", sort=True):
        selector_rows = group.copy()
        for column in truth_columns:
            if column in selector_rows.columns:
                selector_rows[column] = np.nan
        selected_model, reason, low_quality = runtime.select_actual_model(selector_rows)
        frozen_model = str(selected_reference.loc[run_id]["selected_model"])
        if selected_model != frozen_model:
            raise RuntimeError(f"Frozen selector replay mismatch for {run_id}: {selected_model} != {frozen_model}")
        evaluable = group[pd.to_numeric(group["mean_trajectory_position_error_m"], errors="coerce").notna()].copy()
        oracle = evaluable.loc[pd.to_numeric(evaluable["mean_trajectory_position_error_m"], errors="coerce").idxmin()]
        selected = evaluable[evaluable["candidate_model"] == selected_model].iloc[0]
        raw_best = evaluable.loc[pd.to_numeric(evaluable["raw_validation_rmse_mps"], errors="coerce").idxmin()]
        full_best = evaluable.loc[pd.to_numeric(evaluable["full_residual_rmse_mps"], errors="coerce").idxmin()]
        finite_condition = evaluable[np.isfinite(pd.to_numeric(evaluable["condition_number"], errors="coerce"))]
        condition_best = finite_condition.loc[pd.to_numeric(finite_condition["condition_number"], errors="coerce").idxmin()]
        selected_error = float(selected["mean_trajectory_position_error_m"])
        oracle_error = float(oracle["mean_trajectory_position_error_m"])
        init_group = initialization[initialization["run_id"] == run_id]
        i0 = init_group[init_group["initialization_category"] == "I0_cv_fit"]
        i3 = init_group[init_group["initialization_category"] == "I3_frozen_10km"]
        rows.append(
            {
                "run_id": run_id,
                "route": selected["route"],
                "source_segment_id": selected["source_segment_id"],
                "motion_type": selected["motion_type"],
                "duration_s": int(selected["duration_s"]),
                "geometry_window_id": selected["geometry_window_id"],
                "selected_model": selected_model,
                "selection_reason": reason,
                "selected_low_quality": low_quality,
                "candidate_oracle_model": oracle["candidate_model"],
                "selected_mean_trajectory_error_m": selected_error,
                "candidate_oracle_mean_trajectory_error_m": oracle_error,
                "selected_minus_oracle_m": selected_error - oracle_error,
                "best_raw_validation_model": raw_best["candidate_model"],
                "best_raw_validation_rmse_mps": raw_best["raw_validation_rmse_mps"],
                "best_full_residual_model": full_best["candidate_model"],
                "best_full_residual_rmse_mps": full_best["full_residual_rmse_mps"],
                "best_condition_model": condition_best["candidate_model"],
                "best_condition_number": condition_best["condition_number"],
                "selected_position_error_rank": int(pd.to_numeric(evaluable["mean_trajectory_position_error_m"], errors="coerce").rank(method="min").loc[selected.name]),
                "selected_raw_residual_rank": int(pd.to_numeric(evaluable["raw_validation_rmse_mps"], errors="coerce").rank(method="min").loc[selected.name]),
                "residual_position_spearman": _rank_correlation(evaluable, "raw_validation_rmse_mps", "mean_trajectory_position_error_m"),
                "condition_position_spearman": _rank_correlation(evaluable, "condition_number", "mean_trajectory_position_error_m"),
                "best_i0_subset_error_m": float(pd.to_numeric(i0["mean_trajectory_position_error_m"], errors="coerce").min()),
                "best_i3_subset_error_m": float(pd.to_numeric(i3["mean_trajectory_position_error_m"], errors="coerce").min()),
                "candidate_improvement_i3_to_i0_m": float(pd.to_numeric(i3["mean_trajectory_position_error_m"], errors="coerce").min() - pd.to_numeric(i0["mean_trajectory_position_error_m"], errors="coerce").min()),
                "truth_columns_scrubbed_before_selector": True,
                "truth_used_for_selection": False,
                "model_exact_match_exp02b": selected_model == frozen_model,
            }
        )
    result = pd.DataFrame(rows)
    write_csv(DIRS["outputs"] / "PAPER_EXP02C_SELECTOR_ALIGNMENT.csv", result)
    return result


def diagnose_causes(
    consistency: pd.DataFrame,
    observability: pd.DataFrame,
    initialization: pd.DataFrame,
    zero_noise: pd.DataFrame,
    alignment: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    primary_consistency = consistency[consistency["perturbation_profile"] == PRIMARY_PROFILE]
    observation_fraction = float(primary_consistency["observation_model_inconsistency"].astype(bool).mean())
    observation_flag = observation_fraction > 0.0

    m2_full = observability[
        (observability["candidate_model"] == "M2_ctd_full")
        & (observability["observation_subset"] == "full")
    ].copy()
    geometry_bad = (
        (pd.to_numeric(m2_full["effective_numerical_rank"], errors="coerce") < 8)
        | (pd.to_numeric(m2_full["condition_number"], errors="coerce") > CAUSE_RULES["geometry_observability"]["near_rank_deficient_condition_threshold"])
    )
    geometry_fraction = float(geometry_bad.mean())
    geometry_flag = geometry_fraction >= CAUSE_RULES["geometry_observability"]["window_fraction_for_strong_flag"]

    init_best = (
        initialization.groupby(["run_id", "initialization_category"], as_index=False)
        .agg(
            best_error_m=("mean_trajectory_position_error_m", "min"),
            cv_floor_m=("cv_floor_mean_error_m", "first"),
        )
        .pivot(index="run_id", columns="initialization_category", values="best_error_m")
    )
    floor_by_run = initialization.groupby("run_id")["cv_floor_mean_error_m"].first()
    init_eval = init_best.join(floor_by_run.rename("cv_floor_m"))
    i0_threshold = np.maximum(25.0, 5.0 * init_eval["cv_floor_m"])
    init_pattern = (
        (init_eval["I0_cv_fit"] <= i0_threshold)
        & (init_eval["I3_frozen_10km"] > np.maximum(100.0, 3.0 * init_eval["I0_cv_fit"]))
    )
    initialization_fraction = float(init_pattern.mean())
    initialization_flag = initialization_fraction >= CAUSE_RULES["initialization_basin"]["window_fraction_for_strong_flag"]

    candidate_bad = init_eval["I0_cv_fit"] > i0_threshold
    candidate_fraction = float(candidate_bad.mean())
    candidate_flag = (
        not observation_flag
        and candidate_fraction >= CAUSE_RULES["candidate_solver"]["window_fraction_for_strong_flag"]
    )

    large_selector_gap = pd.to_numeric(alignment["selected_minus_oracle_m"], errors="coerce") > np.maximum(
        100.0,
        0.5 * pd.to_numeric(alignment["selected_mean_trajectory_error_m"], errors="coerce"),
    )
    selector_fraction = float(large_selector_gap.mean())
    selector_flag = selector_fraction >= CAUSE_RULES["selector_alignment"]["run_fraction_for_strong_flag"]

    zero_best = (
        zero_noise.groupby(["run_id", "initialization_category"], as_index=False)
        .agg(best_error_m=("mean_trajectory_position_error_m", "min"), cv_floor_m=("cv_floor_mean_error_m", "first"))
        .pivot(index="run_id", columns="initialization_category", values="best_error_m")
    )
    zero_floor = zero_noise.groupby("run_id")["cv_floor_mean_error_m"].first()
    zero_eval = zero_best.join(zero_floor.rename("cv_floor_m"))
    zero_threshold = np.maximum(25.0, 5.0 * zero_eval["cv_floor_m"])
    zero_i0_near_floor_rate = float((zero_eval["I0_cv_fit"] <= zero_threshold).mean())
    zero_i3_near_floor_rate = float((zero_eval["I3_frozen_10km"] <= zero_threshold).mean())

    flags = {
        "observation_model_inconsistency": observation_flag,
        "initialization_basin": initialization_flag,
        "geometry_observability": geometry_flag,
        "candidate_solver": candidate_flag,
        "selector_alignment": selector_flag,
    }
    strong_count = sum(bool(value) for value in flags.values())
    mapping = {
        "observation_model_inconsistency": "observation_model_inconsistency_dominant",
        "initialization_basin": "initialization_basin_dominant",
        "geometry_observability": "geometry_observability_dominant",
        "candidate_solver": "candidate_solver_dominant",
        "selector_alignment": "selector_alignment_dominant",
    }
    if strong_count >= 2:
        diagnosis = "mixed_causes"
    elif strong_count == 1:
        diagnosis = mapping[next(key for key, value in flags.items() if value)]
    else:
        diagnosis = "inconclusive"
    details = {
        "primary_diagnosis": diagnosis,
        "strong_flags": flags,
        "strong_flag_count": strong_count,
        "observation_inconsistency_window_fraction": observation_fraction,
        "m2_near_rank_deficient_window_fraction": geometry_fraction,
        "initialization_basin_pattern_window_fraction": initialization_fraction,
        "best_i0_above_cv_floor_threshold_window_fraction": candidate_fraction,
        "large_selector_gap_run_fraction": selector_fraction,
        "zero_noise_i0_near_cv_floor_rate": zero_i0_near_floor_rate,
        "zero_noise_i3_near_cv_floor_rate": zero_i3_near_floor_rate,
        "median_selected_error_m": float(pd.to_numeric(alignment["selected_mean_trajectory_error_m"], errors="coerce").median()),
        "median_cold_oracle_error_m": float(pd.to_numeric(alignment["candidate_oracle_mean_trajectory_error_m"], errors="coerce").median()),
        "median_selector_gap_m": float(pd.to_numeric(alignment["selected_minus_oracle_m"], errors="coerce").median()),
        "median_best_i0_error_m": float(init_eval["I0_cv_fit"].median()),
        "median_best_i3_error_m": float(init_eval["I3_frozen_10km"].median()),
        "median_cv_floor_m": float(init_eval["cv_floor_m"].median()),
        "decision_rules_frozen_before_solver_execution": True,
    }
    write_json(DIRS["outputs"] / "PAPER_EXP02C_CAUSE_DECISION.json", details)
    write_csv(
        DIRS["outputs"] / "PAPER_EXP02C_CAUSE_INDICATORS.csv",
        [
            {"cause": key, "strong_flag": value, "evidence_fraction": details.get({
                "observation_model_inconsistency": "observation_inconsistency_window_fraction",
                "initialization_basin": "initialization_basin_pattern_window_fraction",
                "geometry_observability": "m2_near_rank_deficient_window_fraction",
                "candidate_solver": "best_i0_above_cv_floor_threshold_window_fraction",
                "selector_alignment": "large_selector_gap_run_fraction",
            }[key])}
            for key, value in flags.items()
        ],
    )
    return diagnosis, details


def generate_figures(
    consistency: pd.DataFrame,
    observability: pd.DataFrame,
    initialization: pd.DataFrame,
    zero_noise: pd.DataFrame,
    alignment: pd.DataFrame,
    cold_candidates: pd.DataFrame,
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.3,
            "axes.linewidth": 0.8,
        }
    )
    figure_rows: list[dict[str, Any]] = []

    def save(fig: Any, stem: str, source: str, purpose: str) -> None:
        pdf = DIRS["figures"] / f"{stem}.pdf"
        png = DIRS["figures"] / f"{stem}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append({"figure_stem": stem, "source_file": source, "purpose": purpose, "pdf_path": str(pdf), "png_path": str(png)})

    p0 = consistency[consistency["perturbation_profile"] == PRIMARY_PROFILE].copy()
    p0["window"] = p0["source_segment_id"] + " D" + p0["duration_s"].astype(str) + " " + p0["geometry_window_id"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), constrained_layout=True)
    axes[0].scatter(np.arange(len(p0)), p0["max_abs_clean_prediction_difference_mps"], s=18)
    axes[0].axhline(CAUSE_RULES["observation_consistency"]["max_abs_clean_prediction_difference_mps"], color="#D55E00", ls="--", label="protocol threshold")
    axes[0].set_yscale("symlog", linthresh=1e-14)
    axes[0].set_title("Clean prediction reproduction")
    axes[0].set_ylabel("Maximum absolute difference (m/s)")
    axes[0].legend(frameon=False)
    axes[1].scatter(p0["cv_floor_mean_error_m"], p0["cv_clean_residual_rmse_mps"], c=p0["duration_s"], cmap="viridis", s=28)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("CV position representation floor (m)")
    axes[1].set_ylabel("CV clean range-rate residual RMSE (m/s)")
    axes[1].set_title("Position floor versus Doppler-domain fit")
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, "summary_01_observation_consistency", "outputs/PAPER_EXP02C_OBSERVATION_CONSISTENCY.csv", "Observation-model reproduction and CV domain consistency")

    m2 = observability[(observability["candidate_model"] == "M2_ctd_full") & (observability["observation_subset"] == "full")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for geometry, marker in [("G0", "o"), ("G1", "s")]:
        subset = m2[m2["geometry_window_id"] == geometry]
        axes[0].scatter(subset["duration_s"], subset["condition_number"], marker=marker, label=geometry, alpha=0.75)
        axes[1].scatter(subset["duration_s"], subset["approx_position_cov_sqrt_trace_m"], marker=marker, label=geometry, alpha=0.75)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Jacobian condition number")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Approx. position covariance sqrt trace (m)")
    for ax in axes:
        ax.set_xlabel("Duration (s)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_title("M2 local conditioning")
    axes[1].set_title("M2 covariance proxy")
    save(fig, "summary_02_observability", "outputs/PAPER_EXP02C_LOCAL_OBSERVABILITY.csv", "G0/G1 local observability")

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    positions = np.arange(len(INIT_CATEGORIES))
    for model_index, model in enumerate(INIT_MODELS):
        medians = [pd.to_numeric(initialization[(initialization["candidate_model"] == model) & (initialization["initialization_category"] == init)]["mean_trajectory_position_error_m"], errors="coerce").median() for init in INIT_CATEGORIES]
        ax.plot(positions, medians, marker=["o", "s", "^", "D", "P"][model_index], label=model)
    ax.set_xticks(positions, ["I0 CV-fit", "I1 +100 m", "I2 +1 km/zero-v", "I3 +10 km/zero-v"], rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("Median mean trajectory error (m)")
    ax.set_title("Initialization convergence basin")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "summary_03_initialization_basin", "outputs/PAPER_EXP02C_INITIALIZATION_BASIN.csv", "Frozen candidate sensitivity to diagnostic initializations")

    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    labels: list[str] = []
    values: list[np.ndarray] = []
    for init_name in ["I0_cv_fit", "I3_frozen_10km"]:
        for model in ZERO_NOISE_MODELS:
            labels.append(f"{init_name[:2]} {model.split('_')[0]}")
            values.append(pd.to_numeric(zero_noise[(zero_noise["initialization_category"] == init_name) & (zero_noise["candidate_model"] == model)]["mean_trajectory_position_error_m"], errors="coerce").dropna().to_numpy(float))
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("Mean trajectory error (m)")
    ax.set_title("Zero-noise solver diagnostic")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "summary_04_zero_noise", "outputs/PAPER_EXP02C_ZERO_NOISE_DIAGNOSTIC.csv", "I0 versus I3 under exact clean observations")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    axes[0].scatter(alignment["candidate_oracle_mean_trajectory_error_m"], alignment["selected_mean_trajectory_error_m"], c=alignment["duration_s"], cmap="viridis", s=28)
    maximum = float(max(alignment["candidate_oracle_mean_trajectory_error_m"].max(), alignment["selected_mean_trajectory_error_m"].max()))
    axes[0].plot([1, maximum], [1, maximum], color="black", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Cold-start candidate oracle error (m)")
    axes[0].set_ylabel("Selected error (m)")
    axes[0].set_title("Selector gap")
    p0_candidates = cold_candidates[(cold_candidates["perturbation_profile"] == PRIMARY_PROFILE) & (pd.to_numeric(cold_candidates["seed"], errors="coerce") == PRIMARY_SEED)]
    axes[1].scatter(p0_candidates["raw_validation_rmse_mps"], p0_candidates["mean_trajectory_position_error_m"], alpha=0.35, s=15)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Raw validation RMSE (m/s)")
    axes[1].set_ylabel("Mean trajectory error (m)")
    axes[1].set_title("Residual-position alignment")
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, "summary_05_selector_alignment", "outputs/PAPER_EXP02C_SELECTOR_ALIGNMENT.csv;EXP02B candidate results", "Residual-position rank alignment and selector gap")

    for (run_id, subset_name), group in observability.groupby(["run_id", "observation_subset"], sort=True):
        if subset_name != "full":
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
        for ax, current_subset in zip(axes, ["full", "train_first_70pct", "validation_last_30pct"], strict=True):
            current = observability[(observability["run_id"] == run_id) & (observability["observation_subset"] == current_subset)]
            for _, row in current.iterrows():
                singular = np.asarray(json.loads(row["singular_values_json"]), dtype=float)
                ax.semilogy(np.arange(1, len(singular) + 1), singular, marker="o", label=row["candidate_model"])
            ax.set_title(current_subset.replace("_", " "))
            ax.set_xlabel("Singular-value index")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Singular value")
        axes[-1].legend(frameon=False)
        save(fig, f"svd_{run_id}", "outputs/PAPER_EXP02C_LOCAL_OBSERVABILITY.csv", f"Jacobian singular-value spectrum for {run_id}")

    inventory = pd.DataFrame(figure_rows)
    write_csv(DIRS["figures"] / "PAPER_EXP02C_FIGURE_INVENTORY.csv", inventory)
    return inventory


def finalize_protection(audit: dict[str, Any], protocol_hash: str) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    source = audit["source_audit"].copy()
    source["sha256_after"] = [sha256_file(Path(path)) for path in source["source_path"]]
    source["unchanged_during_exp02c"] = source["sha256_before"] == source["sha256_after"]
    write_csv(DIRS["protocol"] / "PAPER_EXP02C_FROZEN_SOURCE_AUDIT.csv", source)
    protected_after = {name: snapshot_tree(path) for name, path in audit["protected_paths"].items()}
    protected_unchanged = audit["protected_before"] == protected_after
    exp02b_after = {
        relative: sha256_file(EXP02B / relative)
        for relative in audit["exp02b_required_hashes"]
    }
    exact_exp02b_unchanged = exp02b_after == audit["exp02b_required_hashes"]
    protocol_unchanged = sha256_file(DIRS["protocol"] / "PAPER_EXP02C_PROTOCOL.json") == protocol_hash
    result = {
        "frozen_source_hashes_unchanged": bool(source["unchanged_during_exp02c"].astype(bool).all()),
        "protected_tree_snapshots_unchanged": protected_unchanged,
        "exp02b_required_file_hashes_unchanged": exact_exp02b_unchanged,
        "exp02c_protocol_hash_unchanged": protocol_unchanged,
        "protected_before": audit["protected_before"],
        "protected_after": protected_after,
        "exp02b_required_hashes_before": audit["exp02b_required_hashes"],
        "exp02b_required_hashes_after": exp02b_after,
    }
    write_json(DIRS["protocol"] / "PAPER_EXP02C_PROTECTION_AUDIT.json", result)
    passed = bool(
        result["frozen_source_hashes_unchanged"]
        and protected_unchanged
        and exact_exp02b_unchanged
        and protocol_unchanged
    )
    return source, passed, result


def count_rows(path: Path) -> int | None:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    if path.suffix.lower() in {".json", ".md", ".txt", ".log", ".py"}:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    return None


def build_file_index() -> pd.DataFrame:
    files: list[Path] = []
    for directory_name in ["protocol", "outputs", "scripts", "figures", "source_snapshot", "logs"]:
        files.extend(path for path in DIRS[directory_name].rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(OUT).as_posix(),
            "absolute_path": str(path),
            "row_or_line_count": count_rows(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files, key=lambda item: item.relative_to(OUT).as_posix().lower())
    ]
    frame = pd.DataFrame(rows)
    write_csv(DIRS["outputs"] / "PAPER_EXP02C_LOCAL_FILE_INDEX.csv", frame)
    return frame


def _format_value(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude != 0 and (magnitude >= 1e5 or magnitude < 1e-3):
            return f"{float(value):.3e}"
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str] | None = None, digits: int = 3) -> list[str]:
    labels = labels or columns
    labels = [str(label).replace("|", "\\|") for label in labels]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_value(row.get(column), digits) for column in columns) + " |")
    return lines


def build_total_report(
    audit: dict[str, Any],
    protocol_hash: str,
    consistency: pd.DataFrame,
    observability: pd.DataFrame,
    initialization: pd.DataFrame,
    zero_noise: pd.DataFrame,
    alignment: pd.DataFrame,
    diagnosis: str,
    cause_details: dict[str, Any],
    source_audit: pd.DataFrame,
    protection: dict[str, Any],
    file_index: pd.DataFrame,
    elapsed_seconds: float,
) -> str:
    exp02b_metrics = json.loads((EXP02B / "outputs" / "PAPER_EXP02B_METRICS.json").read_text(encoding="utf-8-sig"))
    p0 = consistency[consistency["perturbation_profile"] == PRIMARY_PROFILE].copy()
    consistency_summary = (
        p0.groupby(["duration_s", "geometry_window_id"], as_index=False)
        .agg(
            windows=("run_id", "count"),
            max_clean_difference_mps=("max_abs_clean_prediction_difference_mps", "max"),
            max_noise_reconstruction_difference_mps=("max_abs_measured_noise_reconstruction_difference_mps", "max"),
            measured_truth_rmse_mps=("measured_truth_residual_rmse_mps", "median"),
            cv_clean_rmse_mps=("cv_clean_residual_rmse_mps", "median"),
            cv_validation_rmse_mps=("cv_blocked_validation_rmse_mps", "median"),
            cv_floor_m=("cv_floor_mean_error_m", "median"),
            ca_floor_m=("ca_floor_mean_error_m", "median"),
        )
    )
    motion_consistency = (
        p0.groupby(["motion_type", "duration_s"], as_index=False)
        .agg(
            windows=("run_id", "count"),
            cv_floor_m=("cv_floor_mean_error_m", "median"),
            cv_clean_residual_rmse_mps=("cv_clean_residual_rmse_mps", "median"),
            ca_clean_residual_rmse_mps=("ca_clean_residual_rmse_mps", "median"),
        )
    )
    m2_full = observability[(observability["candidate_model"] == "M2_ctd_full") & (observability["observation_subset"] == "full")]
    obs_summary = (
        m2_full.groupby(["duration_s", "geometry_window_id"], as_index=False)
        .agg(
            windows=("run_id", "count"),
            median_rank=("effective_numerical_rank", "median"),
            full_rank_rate=("full_column_rank", "mean"),
            median_smallest_singular=("smallest_singular_value", "median"),
            median_condition=("condition_number", "median"),
            median_position_cov_sqrt_trace_m=("approx_position_cov_sqrt_trace_m", "median"),
            median_velocity_drift_coherence=("velocity_drift_coherence", "median"),
            median_position_bias_coherence=("position_bias_coherence", "median"),
        )
    )
    geometry_compare = (
        m2_full.groupby("geometry_window_id", as_index=False)
        .agg(
            windows=("run_id", "count"),
            median_rank=("effective_numerical_rank", "median"),
            median_condition=("condition_number", "median"),
            median_covariance_proxy_m=("approx_position_cov_sqrt_trace_m", "median"),
        )
    )
    init_summary = (
        initialization.groupby(["initialization_category", "candidate_model"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            numerical_success_rate=("numerical_success", "mean"),
            median_mean_error_m=("mean_trajectory_position_error_m", "median"),
            median_final_error_m=("final_position_error_m", "median"),
            median_velocity_error_mps=("mean_velocity_error_mps", "median"),
            median_residual_rmse_mps=("residual_rmse_mps", "median"),
            median_condition=("condition_number", "median"),
            median_distance_to_cv_floor_m=("distance_to_cv_floor_m", "median"),
        )
    )
    zero_summary = (
        zero_noise.groupby(["initialization_category", "candidate_model"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            numerical_success_rate=("numerical_success", "mean"),
            median_mean_error_m=("mean_trajectory_position_error_m", "median"),
            median_final_error_m=("final_position_error_m", "median"),
            median_residual_rmse_mps=("residual_rmse_mps", "median"),
            median_cv_floor_m=("cv_floor_mean_error_m", "median"),
            near_floor_rate=("near_cv_floor", "mean"),
        )
    )
    alignment_summary = (
        alignment.groupby(["duration_s", "geometry_window_id"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            median_selected_error_m=("selected_mean_trajectory_error_m", "median"),
            median_oracle_error_m=("candidate_oracle_mean_trajectory_error_m", "median"),
            median_selector_gap_m=("selected_minus_oracle_m", "median"),
            median_residual_position_spearman=("residual_position_spearman", "median"),
            median_condition_position_spearman=("condition_position_spearman", "median"),
            median_i3_to_i0_candidate_improvement_m=("candidate_improvement_i3_to_i0_m", "median"),
        )
    )
    selected_worse_residual_good = float(
        (
            (alignment["selected_position_error_rank"] > 1)
            & (alignment["selected_raw_residual_rank"] <= 3)
        ).mean()
    )
    i0_rows = initialization[initialization["initialization_category"] == "I0_cv_fit"].copy()
    best_i0_indices = i0_rows.groupby("run_id")["mean_trajectory_position_error_m"].idxmin()
    best_i0_rows = i0_rows.loc[best_i0_indices]
    median_best_i0_residual = float(pd.to_numeric(best_i0_rows["residual_rmse_mps"], errors="coerce").median())
    median_cv_clean_residual = float(pd.to_numeric(p0["cv_clean_residual_rmse_mps"], errors="coerce").median())
    median_m2_covariance_proxy = float(pd.to_numeric(m2_full["approx_position_cov_sqrt_trace_m"], errors="coerce").median())
    median_m2_condition = float(pd.to_numeric(m2_full["condition_number"], errors="coerce").median())
    p1 = consistency[consistency["perturbation_profile"] == "P1_bias_drift"]
    p1_note = (
        f"P0 一致性通过后执行了 32 个 P1 observation-only 辅助检查；最大 clean 重算差为 {p1['max_abs_clean_prediction_difference_mps'].max():.3e} m/s。"
        if len(p1)
        else "P1 辅助检查未触发，因为 P0 一致性前提未满足或 P1 文件不完整。"
    )
    flags = cause_details["strong_flags"]
    strong_names = [name for name, value in flags.items() if value]
    secondary = [name for name in strong_names if diagnosis != {
        "observation_model_inconsistency": "observation_model_inconsistency_dominant",
        "initialization_basin": "initialization_basin_dominant",
        "geometry_observability": "geometry_observability_dominant",
        "candidate_solver": "candidate_solver_dominant",
        "selector_alignment": "selector_alignment_dominant",
    }.get(name)]
    if diagnosis == "mixed_causes":
        secondary = strong_names
    placement = {
        "observation_model_inconsistency_dominant": "在观测生成修复前不纳入论文证据。",
        "mixed_causes": "仅作为补充材料中的失败归因与适用边界分析，不作为主文性能证据。",
        "initialization_basin_dominant": "仅作为补充材料中的初始化敏感性分析。",
        "geometry_observability_dominant": "仅作为补充材料中的几何限制分析。",
        "candidate_solver_dominant": "仅作为补充材料中的候选求解限制分析。",
        "selector_alignment_dominant": "仅作为补充材料中的 residual-position mismatch 分析。",
        "inconclusive": "不建议纳入本论文实证结果。",
    }[diagnosis]
    continue_plan = (
        "可以继续使用方案 A，但仅作为 real-trajectory-driven controlled stress diagnostic；不得将其升级为真实 LEO RF 或动态外场精度验证。"
        if not flags["observation_model_inconsistency"]
        else "应暂停方案 A 的论文使用，先修复并重新冻结观测生成链。"
    )

    lines = [
        "# PAPER-EXP02C UrbanNav-TK 外部轨迹误差来源总诊断报告",
        "",
        "## 1. 任务边界",
        "",
        "本任务只诊断 PAPER-EXP02A/02B 中候选 oracle 仍为数百米至公里级的原因。冻结 MA-BGTR-v7.1 selector、v7.2 freeze policy、M0-M14 候选池、gate 阈值、TECH18/PAPER-EXP01 结果和论文 v0.7 均未修改。Static/CV/CA 拟合、I0/I1 真值辅助初值与 oracle 只在选择结束后用于归因，不构成部署结果或新候选模型。未生成审核 ZIP。",
        "",
        "## 2. 文件与源码哈希核验",
        "",
        f"冻结 release：`{audit['release']}`。关键源码共 {len(source_audit)} 个，运行前后 SHA256 全部一致：{bool(source_audit['unchanged_during_exp02c'].astype(bool).all())}。EXP02A、EXP02B、release 与论文 v0.7 的文件数、总大小和最新修改时间快照一致：{protection['protected_tree_snapshots_unchanged']}；EXP02B 九个必需输入逐文件 SHA256 一致：{protection['exp02b_required_file_hashes_unchanged']}。",
        "",
        "| Item | SHA256 |",
        "|---|---|",
        f"| MA-BGTR-v7.1 selector | `{audit['exp02b_protocol']['selector_sha256']}` |",
        f"| v7.2 freeze policy | `{audit['exp02b_protocol']['freeze_policy_sha256']}` |",
        f"| Canonical Iridium geometry | `{audit['exp02b_protocol']['source_hashes']['canonical_iridium_geometry']}` |",
        f"| UrbanNav Odaiba standardized | `{audit['exp02b_protocol']['source_hashes']['urban_nav']['Odaiba']}` |",
        f"| UrbanNav Shinjuku standardized | `{audit['exp02b_protocol']['source_hashes']['urban_nav']['Shinjuku']}` |",
        "",
        "## 3. 协议 SHA256",
        "",
        f"EXP02C 协议在任何新诊断 solver 执行前冻结，SHA256=`{protocol_hash}`。协议固定了原因判定门槛、32 个窗口、I0-I3、零噪声配置和禁止适配项；运行后协议哈希保持一致：{protection['exp02c_protocol_hash_unchanged']}。",
        "",
        "## 4. 使用窗口和运行规模",
        "",
        f"使用 8 个固定 segment、8/12 s、G0/G1，共 32 个 P0 clean-motion 窗口。4/6 s 按 EXP02B 四星且至少 20 条观测规则维持 unavailable。本任务计算 {len(observability)} 条局部可观性记录；实际运行 {initialization.groupby(['run_id','initialization_category']).ngroups} 个初始化诊断批次、{len(initialization)} 次冻结候选执行，以及 {zero_noise.groupby(['run_id','initialization_category']).ngroups} 个零噪声批次、{len(zero_noise)} 次冻结候选执行。",
        "",
        "## 5. EXP02B 关键结果复盘",
        "",
        f"EXP02B 的 selected mean-trajectory error 中位数为 {exp02b_metrics['selected_error_summary_m']['median']:.2f} m，candidate-oracle 中位数为 {exp02b_metrics['oracle_error_summary_m']['median']:.2f} m，CV representation floor 中位数为 {exp02b_metrics['representation_floor_summary_m']['constant_velocity_median']:.3f} m。8 s 与 12 s 的 selected error 均保持公里量级，且 4/6 s 无合法几何，因此 EXP02B 结论为 `paper_exp02b_inconclusive`。",
        "",
        "## 6. 观测生成一致性",
        "",
        *markdown_table(
            consistency_summary,
            ["duration_s", "geometry_window_id", "windows", "max_clean_difference_mps", "max_noise_reconstruction_difference_mps", "measured_truth_rmse_mps", "cv_clean_rmse_mps", "cv_validation_rmse_mps", "cv_floor_m", "ca_floor_m"],
            ["Duration", "Geometry", "N", "max |clean diff| (m/s)", "max |noise diff| (m/s)", "measured-truth RMSE", "CV clean RMSE", "CV val RMSE", "CV floor (m)", "CA floor (m)"],
        ),
        "",
        f"P0 中 observation-model inconsistency 窗口比例为 {cause_details['observation_inconsistency_window_fraction']:.1%}。保存的 clean range rate、重新计算值、注入噪声和相对时间映射均按协议阈值核验。{p1_note}",
        "",
        "## 7. Doppler 域 CV-fit 残差",
        "",
        *markdown_table(
            motion_consistency,
            ["motion_type", "duration_s", "windows", "cv_floor_m", "cv_clean_residual_rmse_mps", "ca_clean_residual_rmse_mps"],
            ["Motion", "Duration", "N", "CV position floor (m)", "CV Doppler RMSE (m/s)", "CA Doppler RMSE (m/s)"],
        ),
        "",
        "几米级 CV 位置表示下限不必然对应同量级的位置解：Doppler 目标对速度、偏置和视线几何敏感，位置域最小二乘 CV 状态并不是 Doppler 域的定位误差最优状态。该表用于检查时间/速度映射是否自洽，不把 CA 拟合当作新算法。",
        "",
        "## 8. Jacobian rank / SVD",
        "",
        *markdown_table(
            obs_summary,
            ["duration_s", "geometry_window_id", "windows", "median_rank", "full_rank_rate", "median_smallest_singular", "median_condition", "median_position_cov_sqrt_trace_m", "median_velocity_drift_coherence", "median_position_bias_coherence"],
            ["Duration", "Geometry", "N", "M2 rank", "full-rank rate", "median s_min", "median cond(J)", "position cov proxy (m)", "v-drift coherence", "p-bias coherence"],
        ),
        "",
        f"按预注册规则，M2 full-set 近秩亏/高条件窗口比例为 {cause_details['m2_near_rank_deficient_window_fraction']:.1%}。每个窗口的 full/train/validation 奇异值谱均保存在 `figures/svd_*.pdf|png`。",
        "",
        "## 9. G0/G1 可观性对比",
        "",
        *markdown_table(
            geometry_compare,
            ["geometry_window_id", "windows", "median_rank", "median_condition", "median_covariance_proxy_m"],
            ["Geometry", "N", "median M2 rank", "median cond(J)", "median covariance proxy (m)"],
        ),
        "",
        "G0/G1 来自同一约 35.4 s 历史 Iridium span 的早端与晚端，不是独立外场几何。差异只能说明局部几何敏感性。",
        "",
        "## 10. 初始化收敛域",
        "",
        *markdown_table(
            init_summary,
            ["initialization_category", "candidate_model", "runs", "numerical_success_rate", "median_mean_error_m", "median_final_error_m", "median_velocity_error_mps", "median_residual_rmse_mps", "median_condition", "median_distance_to_cv_floor_m"],
            ["Init", "Model", "N", "success", "median mean err (m)", "median final err (m)", "median velocity err", "median residual", "median condition", "median above CV floor (m)"],
        ),
        "",
        f"满足“I0 接近 floor 而 I3 显著失败”的窗口比例为 {cause_details['initialization_basin_pattern_window_fraction']:.1%}。I0/I1/I2 都含不同程度的后验真值辅助，只能诊断收敛域；I3 才对应冻结的 10 km/zero-velocity controlled cold start。",
        "",
        "## 11. 零噪声诊断",
        "",
        *markdown_table(
            zero_summary,
            ["initialization_category", "candidate_model", "runs", "numerical_success_rate", "median_mean_error_m", "median_final_error_m", "median_residual_rmse_mps", "median_cv_floor_m", "near_floor_rate"],
            ["Init", "Model", "N", "success", "median mean err (m)", "median final err (m)", "median residual", "median CV floor (m)", "near-floor rate"],
        ),
        "",
        f"零噪声下 I0 最佳候选接近 CV-floor 阈值的窗口比例为 {cause_details['zero_noise_i0_near_cv_floor_rate']:.1%}，I3 为 {cause_details['zero_noise_i3_near_cv_floor_rate']:.1%}。由于精确真实轨迹并非严格 CV，零噪声仍保留运动模型失配；本项主要区分观测噪声与收敛/几何问题。",
        "",
        "## 12. Selector alignment",
        "",
        *markdown_table(
            alignment_summary,
            ["duration_s", "geometry_window_id", "runs", "median_selected_error_m", "median_oracle_error_m", "median_selector_gap_m", "median_residual_position_spearman", "median_condition_position_spearman", "median_i3_to_i0_candidate_improvement_m"],
            ["Duration", "Geometry", "N", "selected (m)", "cold oracle (m)", "selector gap (m)", "rho(residual,error)", "rho(condition,error)", "I3-to-I0 best improvement (m)"],
        ),
        "",
        f"大 selector-gap run 比例为 {cause_details['large_selector_gap_run_fraction']:.1%}；selector 选择 residual rank 前三但 position-error rank 非第一的比例为 {selected_worse_residual_good:.1%}。所有 selection 调用前均清空后验 truth/error 列，冻结模型与 EXP02B 逐 run 完全一致。",
        "",
        "## 13. 主原因判定",
        "",
        f"**Primary diagnosis: `{diagnosis}`**",
        "",
        f"预注册 strong flags：`{json.dumps(flags, ensure_ascii=False)}`。中位 best-I0 error={cause_details['median_best_i0_error_m']:.2f} m，best-I3 error={cause_details['median_best_i3_error_m']:.2f} m，CV floor={cause_details['median_cv_floor_m']:.3f} m；cold candidate oracle={cause_details['median_cold_oracle_error_m']:.2f} m，selected={cause_details['median_selected_error_m']:.2f} m，selector gap={cause_details['median_selector_gap_m']:.2f} m。该判定完全按 solver 前冻结的规则产生。",
        "",
        f"这里的 `candidate_solver_dominant` 不等同于已证明存在代码 bug。它表示：观测生成一致且 M2 未触发预设秩/条件强否决时，冻结候选即使从 I0 CV-fit 状态启动，也会沿 Doppler residual 目标移动到离位置域 CV floor 很远的解。best-I0 候选最终 residual RMSE 中位数为 {median_best_i0_residual:.3f} m/s，而原 CV-fit clean residual 中位数为 {median_cv_clean_residual:.3f} m/s；残差下降与位置误差恶化并存，属于候选求解目标、局部信息尺度和 position-residual alignment 的联合表现。",
        "",
        "## 14. 次要因素",
        "",
        "强证据因素为：" + (", ".join(f"`{item}`" for item in secondary) if secondary else "无达到预注册 strong threshold 的次要因素") + f"。但连续诊断仍显示弱绝对位置尺度：M2 full Jacobian 的 condition 中位数为 {median_m2_condition:.3g}，位置协方差 sqrt-trace 代理中位数为 {median_m2_covariance_proxy:.1f} m；selector large-gap 比例为 {cause_details['large_selector_gap_run_fraction']:.1%}，接近但未达到预注册 50% strong threshold。这些作为次要限制登记，不回头修改主判定。",
        "",
        "## 15. 已排除或受限的假设",
        "",
        "- 已检查 clean range-rate 符号、receiver/satellite 状态、注入噪声重构和相对时间轴。",
        "- 4/6 s 不可用不是 solver 失败，而是冻结几何不满足四星/20 观测。",
        "- CA floor 未进入候选池，不能据此宣称 acceleration solver 已实现。",
        "- I0/I1/I2 使用后验轨迹信息，不是可部署初始化。",
        "- Oracle 与位置误差只在选择结束后计算。",
        "- 单一 35.4 s Iridium span 无法代表广泛 LEO 几何分布。",
        "",
        "## 16. 当前算法和实验适用边界",
        "",
        "该冻结框架仍适用于受控、局部、候选模型与 observable evidence 明确匹配的 Doppler 定位研究；本 UrbanNav transplant 只增加真实运动学，不增加真实 LEO RF、真实接收机时钟/CFO 或同场星历证据。公里级误差不能被包装成真实动态定位精度验证，也不能用位置域 floor 证明 Doppler solver 应达到同一数值。",
        "",
        "## 17. 是否继续使用 UrbanNav 方案 A",
        "",
        continue_plan,
        "",
        "## 18. EXP02A/02B 的论文位置",
        "",
        placement,
        "",
        "## 19. 下一步最小行动",
        "",
        "1. 保持冻结算法不变，先用更多预注册 LEO geometry spans 重复同一 Jacobian/初始化诊断，确认几何结论是否可外推。",
        "2. 若初始化 basin flag 强，单独启动新的研究任务评估全局/多起点初始化；不得回写当前门限。",
        "3. 若 candidate-solver 或 selector-alignment flag 强，分别开展 Doppler-objective 与 position-error alignment 研究；当前论文只登记限制。",
        "4. UrbanNav 方案 A 如继续，仅保留为 controlled stress diagnostic，并在正文/补充材料中明确 transplant 与非原生 LEO RF 边界。",
        "",
        "## 20. 本地详细结果文件索引",
        "",
        f"总运行时间约 {elapsed_seconds:.1f} s。以下索引列出生成总报告前的全部本地详细文件；本报告和索引 CSV 本身因自引用哈希不可能稳定，故不列入自身索引。用户默认只需发送本报告。",
        "",
        *markdown_table(file_index, ["relative_path", "row_or_line_count", "size_bytes", "sha256"], ["Relative path", "Rows/lines", "Bytes", "SHA256"], digits=0),
        "",
        "---",
        "",
        "本任务未生成 review packet 或任何 ZIP。",
    ]
    return "\n".join(lines) + "\n"


def run(protocol_only: bool = False) -> dict[str, Any]:
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    stale = OUT / "PAPER_EXP02C_FAILURE_REPORT.md"
    if stale.exists():
        stale.unlink()
    start = time.perf_counter()
    log("Starting immutable input and frozen-source audit")
    audit = audit_inputs()
    protocol, protocol_hash = freeze_protocol(audit)
    log(f"Protocol frozen before diagnostic solvers: {protocol_hash}")
    if protocol_only:
        return {
            "protocol_only": True,
            "protocol_sha256": protocol_hash,
            "window_count": len(audit["windows"]),
            "planned_initialization_batches": protocol["planned_initialization_batches"],
            "planned_zero_noise_batches": protocol["planned_zero_noise_batches"],
        }

    consistency = run_observation_consistency(audit)
    observability = run_local_observability(audit)
    initialization = run_initialization_basin(audit, consistency)
    zero_noise = run_zero_noise(audit, consistency)
    alignment = selector_alignment(audit, initialization)
    diagnosis, cause_details = diagnose_causes(consistency, observability, initialization, zero_noise, alignment)
    figure_inventory = generate_figures(
        consistency,
        observability,
        initialization,
        zero_noise,
        alignment,
        audit["p0_candidates"],
    )
    source_audit, protection_pass, protection = finalize_protection(audit, protocol_hash)
    if not protection_pass:
        raise RuntimeError("Frozen source, EXP02A/EXP02B, release, or manuscript protection audit failed")
    metrics = {
        "task_name": "PAPER-EXP02C",
        "protocol_sha256": protocol_hash,
        "primary_diagnosis": diagnosis,
        "cause_details": cause_details,
        "p0_windows": int((consistency["perturbation_profile"] == PRIMARY_PROFILE).sum()),
        "p1_auxiliary_windows": int((consistency["perturbation_profile"] == "P1_bias_drift").sum()),
        "observability_rows": len(observability),
        "initialization_solver_batches": int(initialization.groupby(["run_id", "initialization_category"]).ngroups),
        "initialization_candidate_executions": len(initialization),
        "zero_noise_solver_batches": int(zero_noise.groupby(["run_id", "initialization_category"]).ngroups),
        "zero_noise_candidate_executions": len(zero_noise),
        "selector_alignment_runs": len(alignment),
        "frozen_source_unchanged": True,
        "protected_inputs_unchanged": True,
        "review_packet_generated": False,
        "figure_count": len(figure_inventory),
        "elapsed_seconds_before_report": time.perf_counter() - start,
    }
    write_json(DIRS["outputs"] / "PAPER_EXP02C_METRICS.json", metrics)
    log(f"Diagnostics complete with primary diagnosis {diagnosis}; building self-contained report")
    file_index = build_file_index()
    elapsed = time.perf_counter() - start
    report = build_total_report(
        audit,
        protocol_hash,
        consistency,
        observability,
        initialization,
        zero_noise,
        alignment,
        diagnosis,
        cause_details,
        source_audit,
        protection,
        file_index,
        elapsed,
    )
    report_path = OUT / "PAPER_EXP02C_TOTAL_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    required = [
        DIRS["outputs"] / "PAPER_EXP02C_OBSERVATION_CONSISTENCY.csv",
        DIRS["outputs"] / "PAPER_EXP02C_LOCAL_OBSERVABILITY.csv",
        DIRS["outputs"] / "PAPER_EXP02C_INITIALIZATION_BASIN.csv",
        DIRS["outputs"] / "PAPER_EXP02C_ZERO_NOISE_DIAGNOSTIC.csv",
        DIRS["outputs"] / "PAPER_EXP02C_SELECTOR_ALIGNMENT.csv",
        report_path,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Required EXP02C outputs missing: {missing}")
    if list(OUT.rglob("*.zip")):
        raise RuntimeError("A ZIP was generated even though EXP02C forbids review packets")
    return {
        "protocol_only": False,
        "primary_diagnosis": diagnosis,
        "total_report": str(report_path),
        "detailed_outputs": str(DIRS["outputs"]),
        "review_packet_generated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PAPER-EXP02C frozen error-source diagnostic")
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.protocol_only)
        print(json.dumps(json_ready(result), ensure_ascii=False))
        return 0
    except Exception as exc:
        for directory in DIRS.values():
            directory.mkdir(parents=True, exist_ok=True)
        failure_path = OUT / "PAPER_EXP02C_FAILURE_REPORT.md"
        failure_path.write_text(
            "# PAPER-EXP02C failure\n\n"
            f"Reason: {type(exc).__name__}: {exc}\n\n"
            "```text\n"
            f"{traceback.format_exc()}"
            "```\n",
            encoding="utf-8",
        )
        print(f"PAPER-EXP02C failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
