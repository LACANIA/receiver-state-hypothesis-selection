from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


ROOT = Path('leo-c')
LAB = Path('leo-c_new_solver_lab')
AUDIT = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_audit"
OUT = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02a"
DATA_ROOT = Path('datasets/UrbanNav/UrbanNav-TK-20181219')

DIRS = {name: OUT / name for name in ["protocol", "inputs", "observations", "outputs", "reports", "scripts", "figures", "source_snapshot", "review_packet"]}
PROTOCOL_VERSION = "PAPER_EXP02A_PROTOCOL_V1"
CANDIDATE_PROTOCOL = "FULL_POOL_M0_M14_V1"
SEEDS = [20260724, 20260725, 20260726]
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
FIXED_SEGMENTS = [
    ("ODA_0012", "high_speed_straight", 30.0),
    ("ODA_0054", "turn", 12.0),
    ("ODA_0063", "deceleration", 15.0),
    ("ODA_0121", "stop_and_go", 30.0),
    ("SHI_0118", "turn", 12.0),
    ("SHI_0214", "deceleration", 15.0),
    ("SHI_0259", "high_speed_straight", 30.0),
    ("SHI_0274", "stop_and_go", 30.0),
]
PROFILES = {
    "P0_clean_motion": {
        "b0_mps": 0.0,
        "bdot_mps2": 0.0,
        "gaussian_sigma_mps": 0.20,
        "qatar_stress": False,
        "selector_modes": ["observable-only"],
    },
    "P1_bias_drift": {
        "b0_mps": 2.0,
        "bdot_mps2": 0.05,
        "gaussian_sigma_mps": 0.30,
        "qatar_stress": False,
        "selector_modes": ["observable-only"],
    },
    "P2_qatar_stress": {
        "b0_mps": 2.0,
        "bdot_mps2": 0.05,
        "gaussian_sigma_mps": 0.30,
        "qatar_stress": True,
        "qatar_column": "col5",
        "cfo_scale_to_mps": 0.45,
        "heavy_tail": True,
        "outlier_ratio_rule": "max(col4.outlier_ratio_3mad,col5.outlier_ratio_3mad)",
        "outlier_amplitude_rule": "col5 p99-to-median robust-z at scale 0.45 m/s",
        "burst_outlier": True,
        "dropout_ratio_rule": "0.5*confidence.ratio_below_90",
        "low_conf_noise_multiplier": 3.5,
        "selector_modes": ["observable-only", "configuration-assisted controlled selection"],
    },
}

CRITICAL_SOURCE_NAMES = [
    "ma_bgtr_v71_solver.py",
    "ma_bgtr_v72_solver.py",
    "selector_freeze_policy.py",
    "model_candidates.py",
    "ma_bgtr_v4_solver.py",
    "ma_bgtr_v3_solver.py",
    "ma_bgtr_v2_solver.py",
    "trajectory_solvers.py",
    "trajectory_models.py",
    "solvers.py",
    "models.py",
    "robust_gates.py",
    "hierarchical_gates.py",
    "family_evidence.py",
    "bias_gate_fix.py",
    "risk_veto.py",
    "qatar_error_model.py",
    "synthetic.py",
    "cascade_refinement.py",
    "gir_tr_solver.py",
    "be_gtr_solver.py",
    "uncertainty_diagnostics.py",
    "geometry_selection.py",
    "ephemeris_covariance.py",
    "validation_split.py",
    "influence_diagnostics.py",
    "coordinates.py",
    "quality.py",
]

MODEL_FAMILIES = {
    "M0_static_position": "static",
    "M1_static_position_bias": "static",
    "M2_ctd_full": "ctd_full",
    "M3_ctd_no_drift": "ctd_no_drift",
    "M4_ctd_no_bias": "ctd_no_bias",
    "M5_be_b0_only": "be_diagnostic",
    "M6_be_full": "be_diagnostic",
    "M7_robust_ctd_full": "robust_ctd",
    "M8_robust_be_b0_only": "robust_be",
    "M9_robust_be_full": "robust_be",
    "M10_gir_tr_fixed_scale": "gir_direct",
    "M11_gir_tr_mad_scale": "gir_direct",
    "M12_ctd_full_plus_gir_refine": "ctd_refine",
    "M13_robust_ctd_full_plus_gir_refine": "robust_refine",
    "M14_static_plus_gir_refine": "static_refine",
}


def log(message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    path = DIRS["reports"] / "PAPER_EXP02A_EXECUTION_LOG.txt"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str] | None = None) -> None:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[columns]
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def snapshot_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "total_size_bytes": 0, "latest_mtime_ns": None}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "total_size_bytes": sum(item.stat().st_size for item in files),
        "latest_mtime_ns": max((item.stat().st_mtime_ns for item in files), default=None),
    }


def _release_components(path: Path) -> dict[str, Path]:
    src = path / "src" / "leo_positioning"
    decision_options = [path / "TECH18_PROJECT_COMPLETION_DECISION.json", path.parent / "TECH18_PROJECT_COMPLETION_DECISION.json"]
    freeze_options = [path / "TECH18_FULL_POOL_FREEZE_REFERENCE.csv", path / "frozen_results" / "TECH18_FULL_POOL_FREEZE_REFERENCE.csv", path.parent / "TECH18_FULL_POOL_FREEZE_REFERENCE.csv"]
    decision = next((p for p in decision_options if p.is_file()), Path())
    freeze_ref = next((p for p in freeze_options if p.is_file()), Path())
    return {
        "selector": src / "ma_bgtr_v71_solver.py",
        "freeze_policy": src / "selector_freeze_policy.py",
        "v72_shell": src / "ma_bgtr_v72_solver.py",
        "model_candidates": src / "model_candidates.py",
        "runtime": path / "scripts" / "release_runtime.py",
        "solver_config": path / "configs" / "frozen_solver_config.json",
        "decision": decision,
        "freeze_reference": freeze_ref,
        "version": path / "VERSION.json",
        "iridium": path / "data" / "real_iridium" / "Iridium.csv",
        "qatar_config": path / "configs" / "qatar_error_model_config.json",
    }


def resolve_release() -> tuple[Path, dict[str, Any], dict[str, Path]]:
    requested = [
        LAB / "final_release" / "MA_BGTR_v7_2_freeze_r2",
        Path('leo-c_new_solver_lab/final_release/MA_BGTR_v7_2_freeze_r2'),
    ]
    historical = [
        ROOT / "_new_solver_lab" / "final_release" / "MA_BGTR_v7_2_freeze_r2",
        Path('leo-c/_new_solver_lab/final_release/MA_BGTR_v7_2_freeze_r2'),
    ]
    candidates: list[Path] = []
    for path in requested + historical:
        if path.is_dir() and str(path).lower() not in {str(p).lower() for p in candidates}:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No MA_BGTR_v7_2_freeze_r2 release directory was found")

    valid: list[tuple[Path, dict[str, Path], dict[str, str], dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in candidates:
        components = _release_components(path)
        missing = [name for name, item in components.items() if name not in {"decision", "freeze_reference"} and not item.is_file()]
        if not components["decision"].is_file():
            missing.append("TECH18 decision (release root or adjacent package root)")
        if not components["freeze_reference"].is_file():
            missing.append("TECH18 full-pool freeze reference")
        if missing:
            rejected.append({"path": str(path), "reason": ";".join(missing)})
            continue
        decision = json.loads(components["decision"].read_text(encoding="utf-8-sig"))
        version = json.loads(components["version"].read_text(encoding="utf-8-sig"))
        if decision.get("technical_phase_decision") != "project_technical_phase_complete":
            rejected.append({"path": str(path), "reason": "technical_phase_decision is not complete"})
            continue
        if decision.get("candidate_pool_protocol") != CANDIDATE_PROTOCOL or version.get("candidate_pool_protocol") != CANDIDATE_PROTOCOL:
            rejected.append({"path": str(path), "reason": "candidate pool protocol mismatch"})
            continue
        hashes = {name: sha256_file(item) for name, item in components.items() if item.is_file() and name in {"selector", "freeze_policy", "v72_shell", "model_candidates", "runtime", "solver_config", "freeze_reference"}}
        valid.append((path, components, hashes, {"decision": decision, "version": version}))
    if not valid:
        raise RuntimeError(f"No valid frozen release: {rejected}")
    reference_hashes = valid[0][2]
    consistent = all(item[2] == reference_hashes for item in valid)
    if not consistent:
        raise RuntimeError("Multiple frozen release copies have inconsistent critical hashes")
    selected_path, components, hashes, metadata = min(valid, key=lambda item: len(str(item[0])))
    resolution = {
        "requested_paths": [str(p) for p in requested],
        "discovered_path": [str(item[0]) for item in valid],
        "valid_release": True,
        "algorithm_version": metadata["decision"].get("algorithm_version"),
        "freeze_policy_version": metadata["decision"].get("freeze_policy_version"),
        "candidate_pool_protocol": metadata["decision"].get("candidate_pool_protocol"),
        "selector_sha256": hashes["selector"],
        "freeze_policy_sha256": hashes["freeze_policy"],
        "candidate_source_hashes": hashes,
        "duplicate_paths": [str(item[0]) for item in valid],
        "duplicate_hash_consistency": consistent,
        "selected_release_path": str(selected_path),
        "selection_reason": "All discovered aliases had identical critical hashes; selected the shortest complete path. TECH18 decision is an adjacent package-root artifact in the frozen r2 layout.",
        "rejected_paths": rejected,
        "decision_file": str(components["decision"]),
        "freeze_reference_file": str(components["freeze_reference"]),
    }
    write_json(DIRS["protocol"] / "PAPER_EXP02A_PATH_RESOLUTION.json", resolution)
    return selected_path, resolution, components


def parse_manifest(path: Path) -> dict[str, tuple[int, str]]:
    entries: dict[str, tuple[int, str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("#") or line.startswith("relative_path"):
            continue
        rel, size, digest = line.rsplit(",", 2)
        entries[rel.replace("\\", "/")] = (int(size), digest)
    return entries


def input_audit() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    manifest = parse_manifest(AUDIT / "MANIFEST.txt")
    route_files = {
        "Odaiba": AUDIT / "processed" / "UrbanNav_TK_Odaiba_reference_standardized.csv",
        "Shinjuku": AUDIT / "processed" / "UrbanNav_TK_Shinjuku_reference_standardized.csv",
    }
    expected_rows = {"Odaiba": 12410, "Shinjuku": 20950}
    frames: dict[str, pd.DataFrame] = {}
    route_results: dict[str, Any] = {}
    for route, path in route_files.items():
        rel = path.relative_to(AUDIT).as_posix()
        if rel not in manifest:
            raise RuntimeError(f"PAPER-DATA02A manifest does not contain {rel}")
        size, digest = manifest[rel]
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"PAPER-DATA02A standardized input hash mismatch: {route}")
        frame = pd.read_csv(path)
        required = ["time_s", "ecef_x_m", "ecef_y_m", "ecef_z_m", "enu_e_m", "enu_n_m", "enu_u_m", "derived_ve_mps", "derived_vn_mps", "derived_vu_mps", "gap_flag", "source_time"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise RuntimeError(f"{route} standardized input missing columns: {missing}")
        times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
        monotonic = bool(np.all(np.diff(times) > 0))
        gaps = int(frame["gap_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        finite = bool(np.all(np.isfinite(frame[["ecef_x_m", "ecef_y_m", "ecef_z_m", "derived_ve_mps", "derived_vn_mps", "derived_vu_mps"]].to_numpy(float))))
        if len(frame) != expected_rows[route] or not monotonic or gaps or not finite:
            raise RuntimeError(f"{route} input audit failed")
        frames[route] = frame
        route_results[route] = {
            "path": str(path),
            "relative_manifest_path": rel,
            "sha256": digest,
            "size_bytes": size,
            "row_count": len(frame),
            "expected_row_count": expected_rows[route],
            "time_monotonic_strict": monotonic,
            "gap_flag_count": gaps,
            "ecef_and_derived_enu_velocity_finite": finite,
            "source_velocity_direction_used_as_truth": False,
            "velocity_policy": "Receiver trajectory velocity is computed consistently from transplanted/interpolated ECEF position; PAPER-DATA02A derived ENU velocity is cross-check only.",
        }
        frame[["time_s", "enu_e_m", "enu_n_m", "enu_u_m", "speed_mps", "acceleration_mps2", "yaw_rate_degps"]].to_csv(
            DIRS["inputs"] / f"UrbanNav_TK_{route}_trajectory_for_plot.csv", index=False, encoding="utf-8-sig"
        )
    segments = pd.read_csv(AUDIT / "PAPER_DATA02A_MOTION_SEGMENT_CATALOG.csv")
    fixed = segments[segments["segment_id"].isin([item[0] for item in FIXED_SEGMENTS])].copy()
    if len(fixed) != len(FIXED_SEGMENTS):
        raise RuntimeError("Fixed eight-segment set is incomplete")
    metrics = json.loads((AUDIT / "PAPER_DATA02A_METRICS.json").read_text(encoding="utf-8-sig"))
    audit_result = {
        "task": "PAPER-EXP02A input audit",
        "paper_data02a_decision": metrics.get("decision"),
        "paper_data02a_manifest_sha256": sha256_file(AUDIT / "MANIFEST.txt"),
        "routes": route_results,
        "fixed_segment_ids": [item[0] for item in FIXED_SEGMENTS],
        "fixed_segments_present": True,
        "source_velocity_xyz_direction_used": False,
        "input_audit_pass": True,
    }
    write_json(DIRS["protocol"] / "PAPER_EXP02A_INPUT_AUDIT.json", audit_result)
    return frames, fixed, audit_result


def import_release_runtime(release: Path):
    runtime_path = release / "scripts" / "release_runtime.py"
    spec = importlib.util.spec_from_file_location("paper_exp02a_frozen_release_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen release runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry_row(path: Path, stage: str) -> dict[str, Any]:
    result = {
        "source_file": str(path),
        "sha256": sha256_file(path) if path.is_file() else "",
        "source_stage": stage,
        "row_count": 0,
        "time_start": np.nan,
        "time_end": np.nan,
        "duration_s": np.nan,
        "sampling_interval": np.nan,
        "unique_satellite_count": 0,
        "has_sat_position": False,
        "has_sat_velocity": False,
        "has_receiver_anchor": False,
        "coordinate_frame": "unknown",
        "time_definition": "unknown",
        "usable": False,
        "rejection_reason": "missing",
    }
    if not path.is_file():
        return result
    try:
        frame = pd.read_csv(path)
        result["row_count"] = len(frame)
        if "Time/s" in frame.columns:
            time_col = "Time/s"
            sat_col = "Satellite number"
            pos_cols = ["Sat_position_x/m", "Sat_position_y/m", "Sat_position_z/m"]
            vel_cols = ["Sat_velocity_x/(m/s)", "Sat_velocity_y/(m/s)", "Sat_velocity_z/(m/s)"]
            anchor_cols = ["User_latitude", "User_longitude", "User_height"]
        elif "time_s" in frame.columns and "sat_pos_x_m" in frame.columns:
            time_col = "time_s"
            sat_col = "satellite_number"
            pos_cols = ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]
            vel_cols = ["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]
            anchor_cols = ["true_px_m", "true_py_m", "true_pz_m"]
        else:
            result["rejection_reason"] = "no observation-level satellite-state schema"
            return result
        times = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
        finite_times = np.sort(times[np.isfinite(times)])
        result.update(
            {
                "time_start": float(finite_times[0]),
                "time_end": float(finite_times[-1]),
                "duration_s": float(finite_times[-1] - finite_times[0]),
                "sampling_interval": float(np.median(np.diff(finite_times))) if len(finite_times) > 1 else np.nan,
                "unique_satellite_count": int(frame[sat_col].nunique()) if sat_col in frame else 0,
                "has_sat_position": all(col in frame for col in pos_cols),
                "has_sat_velocity": all(col in frame for col in vel_cols),
                "has_receiver_anchor": all(col in frame for col in anchor_cols) and not frame[anchor_cols].dropna().empty,
                "coordinate_frame": "ECEF meters and ECEF m/s (explicit headers)",
                "time_definition": "file-relative observation seconds; preserved without UTC interpretation",
            }
        )
        result["usable"] = bool(result["has_sat_position"] and result["has_sat_velocity"] and result["has_receiver_anchor"] and result["unique_satellite_count"] >= 4)
        result["rejection_reason"] = "" if result["usable"] else "missing anchor/state fields or fewer than four satellites"
    except Exception as exc:
        result["rejection_reason"] = f"parse_error:{type(exc).__name__}:{exc}"
    return result


def geometry_audit(release: Path, components: dict[str, Path], runtime: Any) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    candidates = [
        (components["iridium"], "frozen_release_actual_iridium_observation_input"),
        (release / "runtime_outputs" / "generated_observations" / "S1_dynamic_clean.csv", "TECH18_actual_rerun_observation_geometry"),
        (release / "data" / "synthetic" / "step04_synthetic_truth.csv", "STEP04_synthetic_truth_without_satellite_states"),
        (ROOT / "Certifiable-Doppler-positioning-main" / "matlab" / "data" / "iridium" / "Iridium.csv", "old_open_source_iridium_csv_read_only"),
    ]
    rows = [geometry_row(path, stage) for path, stage in candidates]
    audit = pd.DataFrame(rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02A_LEO_GEOMETRY_AUDIT.csv", audit)
    canonical = audit[(audit["source_file"] == str(components["iridium"])) & audit["usable"].astype(bool)]
    if canonical.empty:
        raise RuntimeError("Frozen release Iridium source does not contain usable satellite positions and velocities")
    frame = runtime.load_iridium_csv(components["iridium"])
    base_obs = runtime.normalize_observations(frame)
    anchor = {
        "lat_deg": float(base_obs["lat_deg"]),
        "lon_deg": float(base_obs["lon_deg"]),
        "height_m": float(base_obs["height_m"]),
        "ecef_m": np.asarray(base_obs["p_gt_ecef_m"], float),
        "sign_convention": "meas_mps=-measured_hz*c/f; prediction=(p_r-p_s)^T(v_r-v_s)/||p_r-p_s||",
        "source_file": str(components["iridium"]),
        "source_sha256": sha256_file(components["iridium"]),
    }
    return audit, anchor, frame


def select_geometry_windows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    times = pd.to_numeric(frame["Time/s"], errors="coerce").to_numpy(float)
    sats = frame["Satellite number"].to_numpy()
    valid = np.isfinite(times) & np.all(np.isfinite(frame[["Sat_position_x/m", "Sat_position_y/m", "Sat_position_z/m", "Sat_velocity_x/(m/s)", "Sat_velocity_y/(m/s)", "Sat_velocity_z/(m/s)"]].to_numpy(float)), axis=1)
    times = times[valid]
    sats = sats[valid]
    starts = np.sort(np.unique(times))
    duration = 30.0
    eligible: list[dict[str, Any]] = []
    for start in starts:
        end = start + duration
        if end > np.max(times) + 1e-9:
            continue
        mask = (times >= start) & (times <= end)
        count = int(mask.sum())
        unique = int(np.unique(sats[mask]).size)
        if count >= 20 and unique >= 4:
            eligible.append({"start_time_s": float(start), "end_time_s": float(end), "duration_s": duration, "observation_count": count, "unique_satellite_count": unique})
    if not eligible:
        raise RuntimeError("No 30-second LEO geometry window satisfies four satellites and 20 observations")
    g0 = dict(eligible[0])
    g0["geometry_window_id"] = "G0"
    g0["selection_rule"] = "earliest 30-second valid window with >=4 satellites and >=20 observations"
    nonoverlap = [item for item in eligible if item["start_time_s"] >= g0["end_time_s"]]
    windows = [g0]
    if nonoverlap:
        g1 = dict(nonoverlap[-1])
        g1["geometry_window_id"] = "G1"
        g1["selection_rule"] = "latest qualifying 30-second window non-overlapping G0"
        windows.append(g1)
    return windows


SOURCE_TIME_RE = re.compile(r"GPSW(?P<week>\d+):TOW(?P<tow>[0-9.]+)")


def parse_source_time(text: str) -> tuple[int, float]:
    match = SOURCE_TIME_RE.fullmatch(str(text))
    if not match:
        raise ValueError(f"Unrecognized source_time: {text}")
    return int(match.group("week")), float(match.group("tow"))


def earliest_geometry_subwindow(frame: pd.DataFrame, geometry: dict[str, Any], duration_s: float) -> dict[str, Any]:
    g = frame[(frame["Time/s"] >= geometry["start_time_s"]) & (frame["Time/s"] <= geometry["end_time_s"])].copy()
    starts = np.sort(g["Time/s"].unique())
    best = {"unique_satellite_count": 0, "observation_count": 0, "start_time_s": np.nan, "end_time_s": np.nan}
    for start in starts:
        end = float(start) + float(duration_s)
        if end > geometry["end_time_s"] + 1e-9:
            continue
        sub = g[(g["Time/s"] >= start) & (g["Time/s"] <= end)]
        candidate = {
            "unique_satellite_count": int(sub["Satellite number"].nunique()),
            "observation_count": int(len(sub)),
            "start_time_s": float(start),
            "end_time_s": float(end),
        }
        if (candidate["unique_satellite_count"], candidate["observation_count"]) > (best["unique_satellite_count"], best["observation_count"]):
            best = candidate
        if candidate["unique_satellite_count"] >= 4 and candidate["observation_count"] >= 20:
            candidate["available"] = True
            candidate["reason"] = "earliest duration-matched subwindow meeting >=4 satellites and >=20 observations"
            return candidate
    best["available"] = False
    best["reason"] = "no duration-matched subwindow meets both >=4 satellites and >=20 observations"
    return best


def stop_start_center(frame: pd.DataFrame, segment_start: float, segment_end: float) -> tuple[float, str]:
    sub = frame[(frame["time_s"] >= segment_start) & (frame["time_s"] <= segment_end)].copy()
    times = sub["time_s"].to_numpy(float)
    speed = sub["speed_mps"].to_numpy(float)
    best_score = -np.inf
    best_time = float((segment_start + segment_end) / 2.0)
    for idx, current in enumerate(times):
        before = speed[(times >= current - 3.0) & (times <= current)]
        after = speed[(times >= current) & (times <= current + 5.0)]
        if before.size < 5 or after.size < 5:
            continue
        score = float(np.nanmax(after) - np.nanmedian(before))
        if np.nanmedian(before) < 0.5 and np.nanmax(after) >= 3.0 and score > best_score:
            best_score = score
            best_time = float(current)
    rule = "maximum trajectory-only 5-second speed rise following a 3-second stationary median"
    if not np.isfinite(best_score):
        rule = "segment midpoint fallback because no deterministic stop-start transition met the threshold"
    return best_time, rule


def build_window_protocol(
    route_frames: dict[str, pd.DataFrame],
    segment_catalog: pd.DataFrame,
    geometry_frame: pd.DataFrame,
    geometry_windows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    extracts: dict[str, pd.DataFrame] = {}
    g0 = geometry_windows[0]
    catalog = segment_catalog.set_index("segment_id")
    for segment_id, motion_type, target_duration in FIXED_SEGMENTS:
        source = catalog.loc[segment_id]
        route = str(source["route"])
        frame = route_frames[route]
        week_start, tow_start = parse_source_time(source["start_time"])
        week_end, tow_end = parse_source_time(source["end_time"])
        route_week, route_tow0 = parse_source_time(frame.iloc[0]["source_time"])
        if week_start != route_week or week_end != route_week:
            raise RuntimeError(f"GPS week mismatch for {segment_id}")
        segment_start = tow_start - route_tow0
        segment_end = tow_end - route_tow0
        if motion_type == "stop_and_go":
            center, center_rule = stop_start_center(frame, segment_start, segment_end)
        else:
            center = 0.5 * (segment_start + segment_end)
            center_rule = "source-segment midpoint"
        route_min = float(frame["time_s"].min())
        route_max = float(frame["time_s"].max())
        start = max(route_min, min(center - target_duration / 2.0, route_max - target_duration))
        end = start + target_duration
        window = frame[(frame["time_s"] >= start - 0.11) & (frame["time_s"] <= end + 0.11)].copy()
        inside = (window["time_s"] >= start) & (window["time_s"] <= end)
        window["inside_experiment_window"] = inside
        extracts[segment_id] = window
        window.to_csv(DIRS["inputs"] / f"{segment_id}_trajectory_window.csv", index=False, encoding="utf-8-sig")
        gap_count = int(window.loc[inside, "gap_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        geometry_sub = earliest_geometry_subwindow(geometry_frame, g0, target_duration)
        available = bool(gap_count == 0 and geometry_sub["available"])
        unavailable_reason = "" if available else ("trajectory gap in fixed window" if gap_count else geometry_sub["reason"])
        rows.append(
            {
                "route": route,
                "source_segment_id": segment_id,
                "experiment_window_id": f"W_{segment_id}",
                "motion_type": motion_type,
                "source_catalog_type": source["segment_type"],
                "target_duration_s": target_duration,
                "window_start_time_s": start,
                "window_end_time_s": end,
                "actual_duration_s": target_duration,
                "source_gps_week": route_week,
                "source_start_tow_s": route_tow0 + start,
                "source_end_tow_s": route_tow0 + end,
                "center_selection_rule": center_rule,
                "extension_rule": "centered target-duration window; clipped only to route boundaries; no time scaling",
                "gap_count": gap_count,
                "gap_free": gap_count == 0,
                "trajectory_sample_count": int(inside.sum()),
                "median_speed_mps": float(pd.to_numeric(window.loc[inside, "speed_mps"], errors="coerce").median()),
                "max_speed_mps": float(pd.to_numeric(window.loc[inside, "speed_mps"], errors="coerce").max()),
                "max_abs_acceleration_mps2": float(pd.to_numeric(window.loc[inside, "acceleration_mps2"], errors="coerce").abs().max()),
                "max_abs_yaw_rate_degps": float(pd.to_numeric(window.loc[inside, "yaw_rate_degps"], errors="coerce").abs().max()),
                "geometry_window_id": "G0",
                "geometry_subwindow_start_s": geometry_sub["start_time_s"],
                "geometry_subwindow_end_s": geometry_sub["end_time_s"],
                "geometry_observation_count": geometry_sub["observation_count"],
                "geometry_unique_satellite_count": geometry_sub["unique_satellite_count"],
                "available_for_solver_run": available,
                "unavailable_reason": unavailable_reason,
                "selected_without_solver_results": True,
            }
        )
    protocol = pd.DataFrame(rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02A_WINDOW_PROTOCOL.csv", protocol)
    return protocol, extracts


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    return np.column_stack((east, north, up))


def interpolate_columns(frame: pd.DataFrame, query: np.ndarray, columns: list[str]) -> np.ndarray:
    source_t = frame["time_s"].to_numpy(float)
    return np.column_stack([np.interp(query, source_t, frame[column].to_numpy(float)) for column in columns])


def finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if len(times) < 2:
        return out
    for idx in range(len(times)):
        if idx == 0:
            left, right = 0, 1
        elif idx == len(times) - 1:
            left, right = len(times) - 2, len(times) - 1
        else:
            left, right = idx - 1, idx + 1
        dt = times[right] - times[left]
        if dt > 0:
            out[idx] = (values[right] - values[left]) / dt
    return out


def build_transplant_map(window_protocol: pd.DataFrame, route_frames: dict[str, pd.DataFrame], anchor: dict[str, Any]) -> pd.DataFrame:
    rotation = enu_to_ecef_rotation(anchor["lat_deg"], anchor["lon_deg"])
    rows: list[dict[str, Any]] = []
    for _, window in window_protocol.iterrows():
        frame = route_frames[str(window["route"])]
        start = float(window["window_start_time_s"])
        urban_origin_ecef = interpolate_columns(frame, np.array([start]), ["ecef_x_m", "ecef_y_m", "ecef_z_m"])[0]
        rows.append(
            {
                "route": window["route"],
                "experiment_window_id": window["experiment_window_id"],
                "source_segment_id": window["source_segment_id"],
                "source_gps_week": int(window["source_gps_week"]),
                "source_start_tow_s": float(window["source_start_tow_s"]),
                "source_end_tow_s": float(window["source_end_tow_s"]),
                "urban_relative_time_s": f"0..{float(window['actual_duration_s']):.3f}",
                "leo_relative_time_s": f"0..{float(window['actual_duration_s']):.3f}" if bool(window["available_for_solver_run"]) else "unavailable_geometry_subwindow",
                "urban_origin_ecef": json.dumps(urban_origin_ecef.tolist()),
                "leo_anchor_ecef": json.dumps(np.asarray(anchor["ecef_m"]).tolist()),
                "enu_to_ecef_rotation": json.dumps(rotation.tolist()),
                "mapping_rule": "p_anchor + R_ENU_to_ECEF * (ENU(t)-ENU(window_start)); no absolute Tokyo position retained",
                "interpolation_rule": "piecewise-linear ENU position at original LEO relative epochs; ECEF velocity from finite differences of interpolated ECEF positions",
                "source_truth_used_for_selection": False,
            }
        )
    result = pd.DataFrame(rows)
    write_csv(DIRS["inputs"] / "PAPER_EXP02A_TRAJECTORY_TRANSPLANT_MAP.csv", result)
    return result


def source_audit_before(release: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    src = release / "src" / "leo_positioning"
    rows = []
    hashes: dict[str, str] = {}
    snapshot_src = DIRS["source_snapshot"] / "leo_positioning"
    snapshot_src.mkdir(parents=True, exist_ok=True)
    for name in CRITICAL_SOURCE_NAMES:
        path = src / name
        if not path.is_file():
            raise FileNotFoundError(f"Frozen critical source is missing: {path}")
        digest = sha256_file(path)
        hashes[name] = digest
        shutil.copy2(path, snapshot_src / name)
        rows.append({"component": name, "source_path": str(path), "sha256_before": digest, "sha256_after": "", "unchanged": "pending", "role": "frozen solver/selector/diagnostic dependency"})
    runtime = release / "scripts" / "release_runtime.py"
    digest = sha256_file(runtime)
    hashes["scripts/release_runtime.py"] = digest
    (DIRS["source_snapshot"] / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime, DIRS["source_snapshot"] / "scripts" / "release_runtime.py")
    rows.append({"component": "scripts/release_runtime.py", "source_path": str(runtime), "sha256_before": digest, "sha256_after": "", "unchanged": "pending", "role": "frozen engineering execution adapter"})
    frame = pd.DataFrame(rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02A_FROZEN_SOURCE_AUDIT.csv", frame)
    return frame, hashes


def finalize_source_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    out = frame.copy()
    after = []
    unchanged = []
    for _, row in out.iterrows():
        digest = sha256_file(Path(row["source_path"]))
        after.append(digest)
        unchanged.append(digest == row["sha256_before"])
    out["sha256_after"] = after
    out["unchanged"] = unchanged
    write_csv(DIRS["protocol"] / "PAPER_EXP02A_FROZEN_SOURCE_AUDIT.csv", out)
    return out, bool(all(unchanged))


def freeze_protocol(
    release: Path,
    resolution: dict[str, Any],
    input_info: dict[str, Any],
    geometry_hash: str,
    geometry_windows: list[dict[str, Any]],
    window_protocol: pd.DataFrame,
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], str]:
    protocol = {
        "task_name": "PAPER-EXP02A",
        "protocol_version": PROTOCOL_VERSION,
        "created_before_solver_execution": True,
        "frozen_release_path": str(release),
        "algorithm_version": resolution["algorithm_version"],
        "selector_sha256": resolution["selector_sha256"],
        "freeze_policy_sha256": resolution["freeze_policy_sha256"],
        "critical_source_hashes": source_hashes,
        "candidate_pool_protocol": CANDIDATE_PROTOCOL,
        "candidate_pool": CANDIDATES,
        "urban_nav_input_hashes": {route: info["sha256"] for route, info in input_info["routes"].items()},
        "geometry_source_hash": geometry_hash,
        "source_segment_ids": [item[0] for item in FIXED_SEGMENTS],
        "final_trajectory_windows": window_protocol.to_dict(orient="records"),
        "geometry_window_ids": [item["geometry_window_id"] for item in geometry_windows],
        "geometry_windows": geometry_windows,
        "transplant_rule": "Use source-window relative ENU displacement/velocity shape, rotate at frozen receiver anchor, preserve relative timing, and discard Tokyo absolute location.",
        "observation_sampling_rule": "Use original qualifying Iridium observation epochs without compression/stretching; window requires >=4 satellites and >=20 observations.",
        "perturbation_profiles": PROFILES,
        "seeds": SEEDS,
        "seed_derivation": "numpy SeedSequence(base_seed, stable SHA256-derived window integer, geometry index, profile index)",
        "initial_conditions": {"position": "frozen anchor + 10 km local east", "velocity": "zero velocity", "beta_prior": "B0_none"},
        "selector_evidence_modes": {
            "observable-only": "residual tail/MAD, blocked validation, condition/geometry/state plausibility; no injection truth",
            "configuration-assisted controlled selection": "same candidate fits; declared aggregate perturbation configuration may enter existing robust-evidence interface; no receiver truth/error/oracle",
        },
        "evaluation_metrics": ["mean trajectory position error (primary)", "initial/final/median/p95/max position error", "mean/p95 velocity error", "post-selection oracle gap", "residual and runtime metrics"],
        "exclusion_rules": ["pre-audited trajectory gap", "fewer than four independent satellites", "fewer than 20 valid observations", "non-finite satellite state", "dropout removal is recorded, never silently deleted"],
        "truth_isolation": "Truth fields are scrubbed before frozen selector; new trajectory metrics are calculated only after selection.",
        "prohibited_claims": ["real LEO dynamic field experiment", "native UrbanNav LEO Doppler", "real RF/CFO validation", "end-to-end LEO receiver validation", "all real trajectories solved accurately"],
    }
    path = DIRS["protocol"] / "PAPER_EXP02A_PROTOCOL.json"
    serialized = json.dumps(json_ready(protocol), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Existing immutable protocol differs from proposed protocol; a new protocol version is required")
    if not path.exists():
        path.write_text(serialized, encoding="utf-8")
    digest = sha256_file(path)
    hash_path = DIRS["protocol"] / "PAPER_EXP02A_PROTOCOL_SHA256.txt"
    expected_line = f"{digest}  PAPER_EXP02A_PROTOCOL.json\n"
    if hash_path.exists() and hash_path.read_text(encoding="utf-8") != expected_line:
        raise RuntimeError("Existing protocol hash file differs")
    if not hash_path.exists():
        hash_path.write_text(expected_line, encoding="utf-8")
    return protocol, digest


def qatar_profile_parameters(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    params = raw["parameters"]
    return {
        "outlier_ratio": max(float(params["col4"]["outlier_ratio_3mad"]), float(params["col5"]["outlier_ratio_3mad"])),
        "dropout_ratio": 0.5 * float(params["confidence"]["ratio_below_90"]),
        "confidence_low_ratio": float(params["confidence"]["ratio_below_90"]),
        "config_sha256": sha256_file(config_path),
    }


def generate_observation(
    runtime: Any,
    qatar_module: Any,
    qatar_model: Any,
    qatar_params: dict[str, Any],
    route_frame: pd.DataFrame,
    window: pd.Series,
    geometry_frame: pd.DataFrame,
    anchor: dict[str, Any],
    profile_name: str,
    seed: int,
    geometry_index: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    profile = PROFILES[profile_name]
    g_start = float(window["geometry_subwindow_start_s"])
    g_end = float(window["geometry_subwindow_end_s"])
    geometry = geometry_frame[(geometry_frame["Time/s"] >= g_start) & (geometry_frame["Time/s"] <= g_end)].sort_values("Time/s").copy()
    tau = geometry["Time/s"].to_numpy(float) - g_start
    urban_query = float(window["window_start_time_s"]) + tau
    enu = interpolate_columns(route_frame, urban_query, ["enu_e_m", "enu_n_m", "enu_u_m"])
    enu_origin = interpolate_columns(route_frame, np.array([float(window["window_start_time_s"])]), ["enu_e_m", "enu_n_m", "enu_u_m"])[0]
    delta_enu = enu - enu_origin
    rotation = enu_to_ecef_rotation(anchor["lat_deg"], anchor["lon_deg"])
    receiver_pos = np.asarray(anchor["ecef_m"], float)[None, :] + delta_enu @ rotation.T
    receiver_vel = finite_difference(receiver_pos, geometry["Time/s"].to_numpy(float))
    sat_pos = geometry[["Sat_position_x/m", "Sat_position_y/m", "Sat_position_z/m"]].to_numpy(float)
    sat_vel = geometry[["Sat_velocity_x/(m/s)", "Sat_velocity_y/(m/s)", "Sat_velocity_z/(m/s)"]].to_numpy(float)
    dp = receiver_pos - sat_pos
    dv = receiver_vel - sat_vel
    clean = np.sum(dp * dv, axis=1) / np.maximum(np.linalg.norm(dp, axis=1), 1e-9)
    run_seed = np.random.SeedSequence([seed, stable_hash_int(str(window["experiment_window_id"])), geometry_index, list(PROFILES).index(profile_name)])
    rng = np.random.default_rng(run_seed)
    gaussian = rng.normal(0.0, float(profile["gaussian_sigma_mps"]), size=len(geometry))
    common_bias = float(profile["b0_mps"]) + float(profile["bdot_mps2"]) * tau
    cfo = np.zeros(len(geometry))
    outlier = np.zeros(len(geometry))
    confidence = np.full(len(geometry), np.nan)
    dropout = np.zeros(len(geometry), dtype=bool)
    if profile["qatar_stress"]:
        cfo, drift_shape = qatar_module.sample_cfo_like_bias(rng, len(geometry), qatar_model, "col5", 0.45, heavy_tail=True)
        cfo = cfo + drift_shape * tau
        amplitude = qatar_module.qatar_outlier_amplitude_mps(qatar_model, "col5", 0.45, use_p99=True, multiplier=1.0)
        outlier = qatar_module.apply_burst_outliers(rng, geometry["Time/s"].to_numpy(float), qatar_params["outlier_ratio"], amplitude)
        confidence, dropout, _weights = qatar_module.sample_confidence_dropout(rng, len(geometry), qatar_model, qatar_params["dropout_ratio"])
        low_conf = confidence < 90.0
        gaussian = gaussian + low_conf.astype(float) * rng.normal(0.0, float(profile["gaussian_sigma_mps"]) * float(profile["low_conf_noise_multiplier"]), size=len(geometry))
    measured = clean + common_bias + gaussian + cfo + outlier
    used = ~dropout
    if int(used.sum()) < 20 or int(np.unique(geometry.loc[used, "Satellite number"]).size) < 4:
        raise RuntimeError(f"Post-dropout observation minimum failed for {window['experiment_window_id']} {profile_name} seed={seed}")
    run_id = f"{window['source_segment_id']}_G0_{profile_name}_{seed}"
    obs_frame = pd.DataFrame(
        {
            "run_id": run_id,
            "route": window["route"],
            "source_segment_id": window["source_segment_id"],
            "experiment_window_id": window["experiment_window_id"],
            "geometry_window_id": "G0",
            "seed": seed,
            "perturbation_profile": profile_name,
            "observation_time_s": geometry["Time/s"].to_numpy(float),
            "satellite_id": geometry["Satellite number"].to_numpy(),
            "satellite_ecef_position": [json.dumps(row.tolist()) for row in sat_pos],
            "satellite_ecef_velocity": [json.dumps(row.tolist()) for row in sat_vel],
            "sat_pos_x_m": sat_pos[:, 0], "sat_pos_y_m": sat_pos[:, 1], "sat_pos_z_m": sat_pos[:, 2],
            "sat_vel_x_mps": sat_vel[:, 0], "sat_vel_y_mps": sat_vel[:, 1], "sat_vel_z_mps": sat_vel[:, 2],
            "receiver_true_ecef_position": [json.dumps(row.tolist()) for row in receiver_pos],
            "receiver_true_ecef_velocity": [json.dumps(row.tolist()) for row in receiver_vel],
            "receiver_true_x_m": receiver_pos[:, 0], "receiver_true_y_m": receiver_pos[:, 1], "receiver_true_z_m": receiver_pos[:, 2],
            "receiver_true_vx_mps": receiver_vel[:, 0], "receiver_true_vy_mps": receiver_vel[:, 1], "receiver_true_vz_mps": receiver_vel[:, 2],
            "clean_range_rate_mps": clean,
            "common_bias_mps": common_bias,
            "drift_mps2": float(profile["bdot_mps2"]),
            "gaussian_noise_mps": gaussian,
            "qatar_cfo_like_mps": cfo,
            "outlier_mps": outlier,
            "confidence_simulated": confidence,
            "dropout_flag": dropout,
            "measured_range_rate_mps": measured,
            "used_by_solver": used,
        }
    )
    obs_path = DIRS["observations"] / f"{run_id}.csv"
    obs_frame.to_csv(obs_path, index=False, encoding="utf-8-sig")
    truth_pos = receiver_pos[used]
    truth_vel = receiver_vel[used]
    times_used = geometry.loc[used, "Time/s"].to_numpy(float)
    obs = {
        "lat_deg": anchor["lat_deg"],
        "lon_deg": anchor["lon_deg"],
        "height_m": anchor["height_m"],
        "p_gt_ecef_m": truth_pos[0],
        "p0_true_m": truth_pos[0],
        "v_true_mps": np.mean(truth_vel, axis=0),
        "truth_positions_m": truth_pos,
        "truth_velocity_samples_mps": truth_vel,
        "truth_bias_mps": common_bias[used],
        "b0_true_mps": float(profile["b0_mps"]),
        "bdot_true_mps2": float(profile["bdot_mps2"]),
        "sat_pos_m": sat_pos[used],
        "sat_vel_mps": sat_vel[used],
        "sat_pos_truth_m": sat_pos[used],
        "sat_vel_truth_mps": sat_vel[used],
        "meas_mps": measured[used],
        "time_s": times_used,
        "satellite_number": geometry.loc[used, "Satellite number"].to_numpy(),
        "row_index": np.flatnonzero(used),
        "t0_s": float(times_used.min()),
    }
    metadata = {
        "run_id": run_id,
        "observation_path": str(obs_path),
        "observation_sha256": sha256_file(obs_path),
        "observation_rows": len(obs_frame),
        "solver_rows": int(used.sum()),
        "unique_satellites_solver": int(np.unique(obs["satellite_number"]).size),
        "outlier_ratio_actual": float(np.mean(np.abs(outlier) > 0)),
        "dropout_ratio_actual": float(np.mean(dropout)),
        "confidence_low_ratio_actual": float(np.mean(confidence < 90)) if profile["qatar_stress"] else 0.0,
        "declared_outlier_ratio": qatar_params["outlier_ratio"] if profile["qatar_stress"] else 0.0,
        "declared_dropout_ratio": qatar_params["dropout_ratio"] if profile["qatar_stress"] else 0.0,
        "burst_outlier": bool(profile["qatar_stress"]),
        "profile": profile_name,
        "seed": seed,
    }
    return obs, obs_frame, metadata


def estimated_trajectory(row: pd.Series, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final = np.array([row.get("final_ecef_x_m", np.nan), row.get("final_ecef_y_m", np.nan), row.get("final_ecef_z_m", np.nan)], float)
    velocity = np.array([row.get("estimated_vx_mps", np.nan), row.get("estimated_vy_mps", np.nan), row.get("estimated_vz_mps", np.nan)], float)
    if not np.all(np.isfinite(final)):
        return np.full((len(times), 3), np.nan), np.full((len(times), 3), np.nan)
    if str(row["candidate_model"]) in {"M0_static_position", "M1_static_position_bias"}:
        velocity = np.zeros(3)
        positions = np.repeat(final[None, :], len(times), axis=0)
    elif np.all(np.isfinite(velocity)):
        tau = times - float(np.min(times))
        p0 = final - velocity * tau[-1]
        positions = p0[None, :] + tau[:, None] * velocity[None, :]
    else:
        positions = np.full((len(times), 3), np.nan)
    velocities = np.repeat(velocity[None, :], len(times), axis=0)
    return positions, velocities


def add_post_selection_metrics(frame: pd.DataFrame, obs: dict[str, Any]) -> pd.DataFrame:
    truth_pos = np.asarray(obs["truth_positions_m"], float)
    truth_vel = np.asarray(obs["truth_velocity_samples_mps"], float)
    times = np.asarray(obs["time_s"], float)
    rows = []
    for _, row in frame.iterrows():
        current = row.to_dict()
        positions, velocities = estimated_trajectory(row, times)
        pos_error = np.linalg.norm(positions - truth_pos, axis=1)
        vel_error = np.linalg.norm(velocities - truth_vel, axis=1)
        finite_pos = pos_error[np.isfinite(pos_error)]
        finite_vel = vel_error[np.isfinite(vel_error)]
        current.update(
            {
                "initial_position_error_m": float(finite_pos[0]) if len(finite_pos) else np.nan,
                "final_position_error_m": float(finite_pos[-1]) if len(finite_pos) else np.nan,
                "mean_trajectory_position_error_m": float(np.mean(finite_pos)) if len(finite_pos) else np.nan,
                "median_trajectory_position_error_m": float(np.median(finite_pos)) if len(finite_pos) else np.nan,
                "p95_trajectory_position_error_m": float(np.quantile(finite_pos, 0.95)) if len(finite_pos) else np.nan,
                "max_trajectory_position_error_m": float(np.max(finite_pos)) if len(finite_pos) else np.nan,
                "mean_velocity_error_mps": float(np.mean(finite_vel)) if len(finite_vel) else np.nan,
                "p95_velocity_error_mps": float(np.quantile(finite_vel, 0.95)) if len(finite_vel) else np.nan,
                "model_family": MODEL_FAMILIES.get(str(row["candidate_model"]), "unknown"),
            }
        )
        rows.append(current)
    return pd.DataFrame(rows)


def run_candidates_and_select(
    runtime: Any,
    qatar_module: Any,
    qatar_model: Any,
    qatar_params: dict[str, Any],
    route_frames: dict[str, pd.DataFrame],
    window_protocol: pd.DataFrame,
    geometry_frame: pd.DataFrame,
    anchor: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    observation_inventory: list[dict[str, Any]] = []
    truth_audit: list[dict[str, Any]] = []
    available = window_protocol[window_protocol["available_for_solver_run"].astype(bool)].copy()
    total_batches = len(available) * len(PROFILES) * len(SEEDS)
    batch = 0
    for _, window in available.iterrows():
        route_frame = route_frames[str(window["route"])]
        for profile_name in PROFILES:
            for seed in SEEDS:
                batch += 1
                log(f"Solver batch {batch}/{total_batches}: {window['source_segment_id']} G0 {profile_name} seed={seed}")
                obs, _obs_frame, obs_meta = generate_observation(runtime, qatar_module, qatar_model, qatar_params, route_frame, window, geometry_frame, anchor, profile_name, seed, 0)
                candidate = runtime.run_actual_candidates(
                    "urban_real_trajectory_controlled",
                    obs_meta["run_id"],
                    obs,
                    {"dataset_type": "urban_real_trajectory_controlled"},
                    candidate_models=CANDIDATES,
                    position_init_label="east_10km",
                    velocity_init_label="zero_velocity",
                    beta_prior_profile="B0_none",
                )
                if set(candidate["candidate_model"]) != set(CANDIDATES) or len(candidate) != 15:
                    raise RuntimeError(f"Incomplete candidate pool for {obs_meta['run_id']}")
                candidate["run_id"] = obs_meta["run_id"]
                candidate["route"] = window["route"]
                candidate["source_segment_id"] = window["source_segment_id"]
                candidate["experiment_window_id"] = window["experiment_window_id"]
                candidate["motion_type"] = window["motion_type"]
                candidate["geometry_window_id"] = "G0"
                candidate["perturbation_profile"] = profile_name
                candidate["seed"] = seed
                candidate["random_seed"] = seed
                candidate["source_observation_file"] = obs_meta["observation_path"]
                candidate["source_config_file"] = str(DIRS["protocol"] / "PAPER_EXP02A_PROTOCOL.json")
                selector_mode_frames: list[tuple[str, pd.DataFrame, bool]] = [("observable-only", candidate.copy(), False)]
                if profile_name == "P2_qatar_stress":
                    assisted = runtime._add_observable_outlier_evidence(
                        candidate.copy(),
                        {
                            "dataset_type": "qatar_synthetic",
                            "outlier_ratio": qatar_params["outlier_ratio"],
                            "dropout_ratio": qatar_params["dropout_ratio"],
                            "burst_outlier": True,
                        },
                    )
                    selector_mode_frames.append(("configuration-assisted controlled selection", assisted, True))
                selections: list[tuple[str, str, str, bool, bool]] = []
                for evidence_mode, selector_frame, metadata_used in selector_mode_frames:
                    scrubbed = runtime.selector_input_frame(selector_frame)
                    truth_columns = [col for col in runtime.TRUTH_ONLY_COLUMNS if col in scrubbed.columns]
                    truth_clean = all(pd.to_numeric(scrubbed[col], errors="coerce").isna().all() for col in truth_columns)
                    if not truth_clean:
                        raise RuntimeError(f"Truth isolation failed before selector for {obs_meta['run_id']} {evidence_mode}")
                    model, reason, low_quality = runtime.select_actual_model(selector_frame)
                    selections.append((evidence_mode, model, reason, low_quality, metadata_used))
                    truth_audit.append(
                        {
                            "run_id": obs_meta["run_id"],
                            "evidence_mode": evidence_mode,
                            "truth_columns_scrubbed": ";".join(sorted(truth_columns)),
                            "truth_columns_all_nan_before_selection": truth_clean,
                            "position_error_used_for_selection": False,
                            "oracle_used_for_selection": False,
                            "route_or_motion_model_hardcode": False,
                        }
                    )
                evaluated = add_post_selection_metrics(candidate, obs)
                candidate_frames.append(evaluated)
                successful = evaluated[pd.to_numeric(evaluated["mean_trajectory_position_error_m"], errors="coerce").notna()].copy()
                oracle_row = successful.loc[pd.to_numeric(successful["mean_trajectory_position_error_m"], errors="coerce").idxmin()] if not successful.empty else None
                for evidence_mode, model, reason, low_quality, metadata_used in selections:
                    selected = evaluated[evaluated["candidate_model"] == model].iloc[0]
                    oracle_error = float(oracle_row["mean_trajectory_position_error_m"]) if oracle_row is not None else np.nan
                    oracle_model = str(oracle_row["candidate_model"]) if oracle_row is not None else ""
                    baselines = {name: evaluated[evaluated["candidate_model"] == name].iloc[0] for name in ["M0_static_position", "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine"]}
                    selected_error = float(selected["mean_trajectory_position_error_m"])
                    selected_rows.append(
                        {
                            "run_id": obs_meta["run_id"],
                            "route": window["route"],
                            "source_segment_id": window["source_segment_id"],
                            "experiment_window_id": window["experiment_window_id"],
                            "motion_type": window["motion_type"],
                            "geometry_window_id": "G0",
                            "perturbation_profile": profile_name,
                            "seed": seed,
                            "run_available": True,
                            "evidence_mode": evidence_mode,
                            "selected_model": model,
                            "selected_family": MODEL_FAMILIES.get(model, "unknown"),
                            "selection_reason": reason,
                            "selected_low_quality": low_quality,
                            "mean_trajectory_position_error_m": selected_error,
                            "p95_trajectory_position_error_m": selected["p95_trajectory_position_error_m"],
                            "final_position_error_m": selected["final_position_error_m"],
                            "mean_velocity_error_mps": selected["mean_velocity_error_mps"],
                            "residual_rmse_mps": selected.get("residual_rmse_mps", selected.get("full_residual_rmse_mps", np.nan)),
                            "trimmed_validation_rmse_mps": selected.get("trimmed_validation_rmse_mps", np.nan),
                            "numerical_success": selected.get("numerical_success", False),
                            "quality_pass": selected.get("quality_pass", False),
                            "physical_plausible": selected.get("physical_plausible", False),
                            "oracle_model_post_selection": oracle_model,
                            "oracle_family_post_selection": MODEL_FAMILIES.get(oracle_model, "unknown"),
                            "oracle_mean_trajectory_error_m": oracle_error,
                            "selected_minus_oracle_mean_error_m": selected_error - oracle_error if np.isfinite(oracle_error) else np.nan,
                            **{f"beats_{short}": selected_error <= float(row["mean_trajectory_position_error_m"]) for short, row in [("M0", baselines["M0_static_position"]), ("M2", baselines["M2_ctd_full"]), ("M3", baselines["M3_ctd_no_drift"]), ("M4", baselines["M4_ctd_no_bias"]), ("M7", baselines["M7_robust_ctd_full"]), ("M13", baselines["M13_robust_ctd_full_plus_gir_refine"])]},
                            "truth_used_for_selection": False,
                            "configuration_metadata_used": metadata_used,
                        }
                    )
                observation_inventory.append(
                    {
                        "run_id": obs_meta["run_id"],
                        "relative_path": Path(obs_meta["observation_path"]).relative_to(OUT).as_posix(),
                        "absolute_path": obs_meta["observation_path"],
                        "size_bytes": Path(obs_meta["observation_path"]).stat().st_size,
                        "sha256": obs_meta["observation_sha256"],
                        "row_count": obs_meta["observation_rows"],
                        "used_by_solver_count": obs_meta["solver_rows"],
                        "unique_satellites_used": obs_meta["unique_satellites_solver"],
                        "execution_mode": "solver_rerun",
                        "included_in_review_packet": False,
                    }
                )
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    selected = pd.DataFrame(selected_rows)
    inventory = pd.DataFrame(observation_inventory)
    truth = pd.DataFrame(truth_audit)
    return candidates, selected, inventory, truth


def aggregate_selected(selected: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in selected.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        errors = pd.to_numeric(group["mean_trajectory_position_error_m"], errors="coerce")
        oracle_gaps = pd.to_numeric(group["selected_minus_oracle_mean_error_m"], errors="coerce")
        models = group["selected_model"].astype(str).value_counts(normalize=True).to_dict()
        row = dict(zip(keys, values, strict=True))
        row.update(
            {
                "runs": len(group),
                "success_rate": float(group["numerical_success"].astype(bool).mean()),
                "quality_pass_rate": float(group["quality_pass"].astype(bool).mean()),
                "median_mean_trajectory_error_m": float(errors.median()),
                "p75_mean_trajectory_error_m": float(errors.quantile(0.75)),
                "p95_mean_trajectory_error_m": float(errors.quantile(0.95)),
                "median_final_error_m": float(pd.to_numeric(group["final_position_error_m"], errors="coerce").median()),
                "median_velocity_error_mps": float(pd.to_numeric(group["mean_velocity_error_mps"], errors="coerce").median()),
                "median_oracle_gap_m": float(oracle_gaps.median()),
                "p95_oracle_gap_m": float(oracle_gaps.quantile(0.95)),
                "model_selection_distribution": json.dumps(models, sort_keys=True),
                "oracle_family_agreement_rate": float((group["selected_family"] == group["oracle_family_post_selection"]).mean()),
                "selected_beats_M0_rate": float(group["beats_M0"].astype(bool).mean()),
                "selected_beats_M2_rate": float(group["beats_M2"].astype(bool).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_comparison(selected: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    baseline_models = ["M0_static_position", "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine"]
    candidate_lookup = candidates.set_index(["run_id", "candidate_model"])
    rows = []
    for _, selected_row in selected.iterrows():
        for model in baseline_models:
            baseline = candidate_lookup.loc[(selected_row["run_id"], model)]
            baseline_error = float(baseline["mean_trajectory_position_error_m"])
            selected_error = float(selected_row["mean_trajectory_position_error_m"])
            rows.append(
                {
                    "run_id": selected_row["run_id"],
                    "evidence_mode": selected_row["evidence_mode"],
                    "route": selected_row["route"],
                    "motion_type": selected_row["motion_type"],
                    "perturbation_profile": selected_row["perturbation_profile"],
                    "seed": selected_row["seed"],
                    "selected_model": selected_row["selected_model"],
                    "baseline_model": model,
                    "selected_mean_trajectory_error_m": selected_error,
                    "baseline_mean_trajectory_error_m": baseline_error,
                    "baseline_minus_selected_mean_error_m": baseline_error - selected_error,
                    "selected_beats_baseline": selected_error <= baseline_error,
                }
            )
    return pd.DataFrame(rows)


def evidence_mode_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    p2 = selected[selected["perturbation_profile"] == "P2_qatar_stress"].copy()
    rows = []
    for run_id, group in p2.groupby("run_id"):
        a = group[group["evidence_mode"] == "observable-only"]
        b = group[group["evidence_mode"] == "configuration-assisted controlled selection"]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "route": a["route"],
                "motion_type": a["motion_type"],
                "seed": a["seed"],
                "observable_only_model": a["selected_model"],
                "configuration_assisted_model": b["selected_model"],
                "model_changed": a["selected_model"] != b["selected_model"],
                "observable_only_mean_error_m": a["mean_trajectory_position_error_m"],
                "configuration_assisted_mean_error_m": b["mean_trajectory_position_error_m"],
                "configuration_assisted_minus_observable_error_m": b["mean_trajectory_position_error_m"] - a["mean_trajectory_position_error_m"],
                "truth_used_for_either_selection": False,
            }
        )
    return pd.DataFrame(rows)


def statistical_tests(baselines: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    rows = []
    for model, group in baselines.groupby("baseline_model", sort=False):
        diff = pd.to_numeric(group["baseline_minus_selected_mean_error_m"], errors="coerce").dropna().to_numpy(float)
        nonzero = diff[np.abs(diff) > 1e-12]
        try:
            pvalue = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue) if len(nonzero) else 1.0
        except ValueError:
            pvalue = 1.0
        positive = int(np.sum(nonzero > 0))
        negative = int(np.sum(nonzero < 0))
        effect = (positive - negative) / len(nonzero) if len(nonzero) else 0.0
        rows.append(
            {
                "comparison": f"selected_vs_{model}",
                "baseline_model": model,
                "paired_sample_count": len(diff),
                "median_difference_baseline_minus_selected_m": float(np.median(diff)) if len(diff) else np.nan,
                "mean_difference_baseline_minus_selected_m": float(np.mean(diff)) if len(diff) else np.nan,
                "wilcoxon_two_sided_p_value": pvalue,
                "rank_biserial_effect_size": effect,
                "holm_adjusted_p_value": np.nan,
                "interpretation": "exploratory; positive difference/effect favors selected solution",
            }
        )
    order = sorted(range(len(rows)), key=lambda idx: rows[idx]["wilcoxon_two_sided_p_value"])
    adjusted = [np.nan] * len(rows)
    running = 0.0
    m = len(rows)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * rows[idx]["wilcoxon_two_sided_p_value"])
        running = max(running, value)
        adjusted[idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["holm_adjusted_p_value"] = value
    return pd.DataFrame(rows)


def failure_cases(window_protocol: pd.DataFrame, selected: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, window in window_protocol[~window_protocol["available_for_solver_run"].astype(bool)].iterrows():
        rows.append(
            {
                "case_id": window["experiment_window_id"],
                "case_type": "protocol_unavailable_window",
                "run_id": "",
                "source_segment_id": window["source_segment_id"],
                "motion_type": window["motion_type"],
                "selected_model": "",
                "mean_trajectory_position_error_m": np.nan,
                "oracle_gap_m": np.nan,
                "failure_reason": window["unavailable_reason"],
                "retained_without_filtering": True,
            }
        )
    for _, row in candidates[~candidates["numerical_success"].astype(bool)].iterrows():
        rows.append(
            {
                "case_id": f"{row['run_id']}::{row['candidate_model']}",
                "case_type": "candidate_numerical_failure",
                "run_id": row["run_id"],
                "source_segment_id": row["source_segment_id"],
                "motion_type": row["motion_type"],
                "selected_model": row["candidate_model"],
                "mean_trajectory_position_error_m": row.get("mean_trajectory_position_error_m", np.nan),
                "oracle_gap_m": np.nan,
                "failure_reason": row.get("failure_reason", ""),
                "retained_without_filtering": True,
            }
        )
    if not selected.empty:
        error_threshold = float(pd.to_numeric(selected["mean_trajectory_position_error_m"], errors="coerce").quantile(0.90))
        gap_threshold = float(pd.to_numeric(selected["selected_minus_oracle_mean_error_m"], errors="coerce").quantile(0.90))
        for _, row in selected.iterrows():
            types = []
            if not bool(row["quality_pass"]):
                types.append("selected_quality_failure")
            if float(row["mean_trajectory_position_error_m"]) >= error_threshold:
                types.append("large_selected_error")
            if float(row["selected_minus_oracle_mean_error_m"]) >= gap_threshold:
                types.append("large_oracle_gap")
            for case_type in types:
                rows.append(
                    {
                        "case_id": f"{row['run_id']}::{row['evidence_mode']}::{case_type}",
                        "case_type": case_type,
                        "run_id": row["run_id"],
                        "source_segment_id": row["source_segment_id"],
                        "motion_type": row["motion_type"],
                        "selected_model": row["selected_model"],
                        "mean_trajectory_position_error_m": row["mean_trajectory_position_error_m"],
                        "oracle_gap_m": row["selected_minus_oracle_mean_error_m"],
                        "failure_reason": row.get("selection_reason", ""),
                        "retained_without_filtering": True,
                    }
                )
    return pd.DataFrame(rows)


def build_report(
    decision: str,
    release: Path,
    geometry_windows: list[dict[str, Any]],
    window_protocol: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    motion_aggregate: pd.DataFrame,
    profile_aggregate: pd.DataFrame,
    mode_comparison: pd.DataFrame,
    stats: pd.DataFrame,
    source_unchanged: bool,
    protocol_hash: str,
    elapsed_s: float,
) -> str:
    available = window_protocol[window_protocol["available_for_solver_run"].astype(bool)]
    unavailable = window_protocol[~window_protocol["available_for_solver_run"].astype(bool)]
    primary = selected[selected["evidence_mode"] == "observable-only"].copy()
    model_counts = primary["selected_model"].value_counts().to_dict()
    numerical_success_rate = float(candidates["numerical_success"].astype(bool).mean())
    quality_rate = float(candidates["quality_pass"].astype(bool).mean())
    selected_median = float(pd.to_numeric(primary["mean_trajectory_position_error_m"], errors="coerce").median())
    selected_p95 = float(pd.to_numeric(primary["mean_trajectory_position_error_m"], errors="coerce").quantile(0.95))
    oracle_gap_median = float(pd.to_numeric(primary["selected_minus_oracle_mean_error_m"], errors="coerce").median())
    oracle_gap_p95 = float(pd.to_numeric(primary["selected_minus_oracle_mean_error_m"], errors="coerce").quantile(0.95))
    mode_changes = int(mode_comparison["model_changed"].astype(bool).sum()) if not mode_comparison.empty else 0
    candidate_failures = int((~candidates["numerical_success"].astype(bool)).sum())

    lines = [
        "# PAPER-EXP02A UrbanNav-TK 真实轨迹驱动 LEO Doppler 受控外部验证报告",
        "",
        "## 1. 研究目的",
        "",
        "本实验使用独立记录的 UrbanNav-TK 车辆轨迹驱动受控 Iridium/LEO range-rate 观测，并实际运行冻结的 FULL_POOL_M0_M14_V1、MA-BGTR-v7.1 selector 与 v7.2 freeze policy。它检验的是运动真实性扩展，不是 LEO RF 外场验证。",
        "",
        "## 2. 与 PAPER-EXP01 的区别",
        "",
        "PAPER-EXP01 使用项目定义的独立 hold-out 动态协议；本实验的转弯、速度变化、停车和再启动来自 UrbanNav 实测车辆轨迹。卫星状态、LEO range-rate 与扰动仍为受控构造，因此两组证据互补，但均不能替代真实动态 LEO RF 真值数据。",
        "",
        "## 3. UrbanNav 数据角色",
        "",
        "UrbanNav 仅提供 receiver-motion trajectory。实验没有把 UrbanNav RINEX Doppler 当作 LEO Doppler，也没有采用语义未确认的 source Velocity X/Y/Z 方向；接收机速度由移植并插值后的 ECEF 位置一致差分得到，审计派生 ENU 速度只用于交叉检查。",
        "",
        "## 4. 真实轨迹移植方法",
        "",
        "每个窗口以首时刻 ENU 为局部零点，将相对 ENU 位移旋转到冻结 Iridium receiver anchor 的 ECEF 局部坐标架。速度变化、轨迹曲率、转弯、加减速和停启得到保留；东京绝对位置及原始 GPS 周时间不与历史 Iridium 文件强行同步。",
        "",
        "## 5. LEO 几何来源",
        "",
        f"Canonical geometry 为 `{release / 'data' / 'real_iridium' / 'Iridium.csv'}`。该文件含 436 条 observation-level 记录、9 颗卫星及 ECEF position/velocity，时间跨度约 35.40 s。按预注册规则可形成 {len(geometry_windows)} 个非重叠 canonical geometry window。",
        "",
        "## 6. 协议冻结过程",
        "",
        f"在任何 solver 调用前冻结 `PAPER_EXP02A_PROTOCOL.json`，SHA256=`{protocol_hash}`。固定内容包括 8 个 source segment、轨迹窗口规则、几何窗口规则、3 个 profile、3 个 seeds、east-10 km/zero-velocity/B0_none 初始化、完整候选池与 P2 两种 evidence mode；运行后协议哈希保持不变。",
        "",
        "## 7. 轨迹窗口",
        "",
        f"预注册 8 个窗口，其中 {len(available)} 个满足对应时长内至少 4 颗卫星和 20 条有效观测，{len(unavailable)} 个按协议标记 unavailable。没有依据 solver 结果替换区段。",
        "",
        "| Segment | Motion | Duration (s) | Satellites | Observations | Available | Reason |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for _, row in window_protocol.iterrows():
        lines.append(
            f"| {row['source_segment_id']} | {row['motion_type']} | {row['actual_duration_s']:.1f} | "
            f"{int(row['geometry_unique_satellite_count'])} | {int(row['geometry_observation_count'])} | "
            f"{bool(row['available_for_solver_run'])} | {row['unavailable_reason']} |"
        )

    lines += [
        "",
        "## 8. 扰动 profile",
        "",
        "- P0_clean_motion: b0=0 m/s, bdot=0 m/s/s, Gaussian sigma=0.20 m/s。",
        "- P1_bias_drift: b0=2.0 m/s, bdot=0.05 m/s/s, Gaussian sigma=0.30 m/s。",
        "- P2_qatar_stress: 在 P1 基础上使用冻结 Qatar distribution shape 生成 CFO-like、confidence/dropout 与 heavy-tail burst。Qatar 原始统计单位不被解释为真实 LEO m/s。",
        "",
        "## 9. Frozen candidate/selector",
        "",
        f"每个主运行实际执行 M0-M14，共 {len(candidates)} 条 candidate rows（{int(len(candidates) / 15)} 个批次），execution_mode 全部为 solver_rerun，cached replay=0。候选数值成功率为 {numerical_success_rate:.1%}，quality-pass 率为 {quality_rate:.1%}；失败候选未被静默删除。冻结源码前后 SHA256 一致：{source_unchanged}。",
        "",
        "## 10. 主结果",
        "",
        f"Observable-only 共 {len(primary)} 次选择；中位 mean trajectory position error={selected_median:.2f} m，P95={selected_p95:.2f} m。模型选择分布为 `{json.dumps(model_counts, ensure_ascii=False)}`。这些结果显示实验执行完整，但定位误差存在明显长尾，不能概括为所有真实运动均被准确求解。",
        "",
        "| Profile | Evidence mode | Runs | Median mean error (m) | P95 mean error (m) | Median oracle gap (m) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in profile_aggregate.iterrows():
        lines.append(
            f"| {row['perturbation_profile']} | {row['evidence_mode']} | {int(row['runs'])} | "
            f"{row['median_mean_trajectory_error_m']:.2f} | {row['p95_mean_trajectory_error_m']:.2f} | {row['median_oracle_gap_m']:.2f} |"
        )

    lines += [
        "",
        "## 11. Motion-type 结果",
        "",
        "直行、转弯、减速和 stop-and-go 四类预注册运动均获得 geometry-supported 运行。下表保留 observable-only 的分组结果；窗口数量和单一历史 Iridium geometry span 限制了跨几何外推。",
        "",
        "| Route | Motion | Profile | Runs | Median mean error (m) | P95 mean error (m) | Median oracle gap (m) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in motion_aggregate[motion_aggregate["evidence_mode"] == "observable-only"].iterrows():
        lines.append(
            f"| {row['route']} | {row['motion_type']} | {row['perturbation_profile']} | {int(row['runs'])} | "
            f"{row['median_mean_trajectory_error_m']:.2f} | {row['p95_mean_trajectory_error_m']:.2f} | {row['median_oracle_gap_m']:.2f} |"
        )

    lines += [
        "",
        "## 12. P0/P1/P2 结果",
        "",
        "P0、P1 与 P2 的完整分组统计见 `PAPER_EXP02A_PROFILE_AGGREGATE.csv` 和 `PAPER_EXP02A_MOTION_TYPE_AGGREGATE.csv`。所有 seeds 均保留；图中使用完整坐标或 symlog 表示长尾，没有删除大误差运行。",
        "",
        "## 13. Observable-only 与 configuration-assisted",
        "",
        f"P2 产生 {len(mode_comparison)} 对共享同一 candidate fits 的 evidence-mode 比较。Mode B 仅把声明的聚合扰动配置送入现有 robust-evidence interface，不重复求解，也不使用 receiver truth、position error 或 oracle。本次共有 {mode_changes} 对选择发生变化；没有变化同样是实验结果，不能解读为 configuration metadata 普遍无用。",
        "",
        "## 14. Baseline 对比与统计",
        "",
        "对 M0/M2/M3/M4/M7/M13 的全部预声明 paired comparisons 均已输出，并报告 Wilcoxon、rank-biserial 与 Holm adjustment。P2 的两种 evidence mode 共享候选解，因此统计检验仅作 exploratory analysis，不能把重复模式当作独立样本。",
        "",
        "| Comparison | Pairs | Median baseline-selected (m) | Holm p | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['comparison']} | {int(row['paired_sample_count'])} | {row['median_difference_baseline_minus_selected_m']:.2f} | "
            f"{row['holm_adjusted_p_value']:.4g} | {row['interpretation']} |"
        )

    lines += [
        "",
        "## 15. Oracle gap",
        "",
        f"Post-selection oracle 只在选择结束后按最小 mean trajectory error 定义。Observable-only 的 median oracle gap={oracle_gap_median:.2f} m，P95={oracle_gap_p95:.2f} m。该差距说明 residual evidence 与轨迹误差并不总对齐，不能把所选模型解释为位置误差 oracle。",
        "",
        "## 16. Residual-trajectory-error alignment",
        "",
        "Validation residual 与非恒速轨迹位置误差不是同一量。散点图和 failure table 保留两者不一致案例；冻结 selector 只读取 residual/risk/physical evidence，后验轨迹误差未进入选择。",
        "",
        "## 17. 失败案例",
        "",
        f"共有 {candidate_failures} 条 candidate rows numerical_success=False。候选失败、selected quality failure、最高 10% 误差与最高 10% oracle gap 均写入 `PAPER_EXP02A_FAILURE_CASES.csv`，未因表现不佳而过滤。",
        "",
        "## 18. 计算成本",
        "",
        f"主流程总执行时间约 {elapsed_s:.1f} s。按 candidate model/profile 的 runtime 分布见 `PAPER_EXP02A_RUNTIME_SUMMARY.csv`；该时间包含输入审计、观测生成、求解、选择、统计和打包。",
        "",
        "## 19. 可支持的论文声明",
        "",
        "- Independently recorded real vehicle trajectories drive controlled LEO Doppler observations.",
        "- The frozen framework is evaluated under real stop-turn-acceleration kinematics.",
        "- The experiment extends motion realism beyond hand-designed constant-velocity trajectories.",
        "- Selection remains evaluation-truth-free.",
        "",
        "## 20. 禁止的论文声明",
        "",
        "不能称为 real LEO dynamic field experiment、native UrbanNav LEO Doppler、real RF/CFO validation 或 end-to-end LEO receiver validation，也不能宣称 all real trajectories are solved accurately。",
        "",
        "## 21. 对 Acta Astronautica 投稿证据的影响",
        "",
        "本实验增加了独立真实车辆运动形态这一证据维度，覆盖直行、转弯、减速和停启；同时，公里级中位误差、长尾和数百米 oracle gap 表明证据提升是有限且带明确负面结果的。它可补强 controlled validation 与 failure analysis，不能填补真实动态 LEO RF 真值缺口。",
        "",
        "## 22. 是否建议修订论文 v0.7",
        "",
        "建议在不改动既有冻结结果的前提下，将本实验作为 real-trajectory-driven controlled validation 与 limitation/failure evidence 纳入后续论文版本；必须同时报告完整窗口、公里级长尾、oracle gap、P2 evidence-mode 无变化以及非外场验证边界。本任务未修改论文 v0.7。",
        "",
        "## 23. 最终决策",
        "",
        f"**Decision: `{decision}`**",
        "",
        "该决策表示协议、真实 solver rerun、真值隔离和审计链条均成功完成，但定位精度与 residual-position alignment 存在重大限制。没有因结果好坏修改协议、候选、selector 或 gate。",
    ]
    return "\n".join(lines) + "\n"


def package_review(required_files: list[Path]) -> Path:
    packet = DIRS["review_packet"] / "PAPER_EXP02A_REVIEW_PACKET.zip"
    manifest = DIRS["review_packet"] / "MANIFEST.txt"
    unique = sorted(set(required_files), key=lambda path: path.relative_to(OUT).as_posix().lower())
    lines = ["relative_path,size_bytes,sha256", "# MANIFEST.txt is excluded from its own hash by definition."]
    for path in unique:
        rel = path.relative_to(OUT).as_posix()
        lines.append(f"{rel},{path.stat().st_size},{sha256_file(path)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    members = unique + [manifest]
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(packet, "r") as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}
        for line in lines[2:]:
            rel, size, digest = line.rsplit(",", 2)
            payload = archive.read(rel)
            if len(payload) != int(size) or hashlib.sha256(payload).hexdigest() != digest:
                raise RuntimeError(f"Review packet manifest mismatch: {rel}")
        if set(entries) != {path.relative_to(OUT).as_posix() for path in members}:
            raise RuntimeError("Review packet file-set mismatch")
    return packet


def main() -> int:
    started = time.perf_counter()
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    if OUT != Path('leo-c_new_solver_lab/paper_draft/external_dynamic_data/UrbanNav_TK_exp02a'):
        raise RuntimeError("Output root mismatch")
    protected_paths = {
        "paper_v07": LAB / "paper_draft" / "Acta_Astronautica_MA_BGTR" / "manuscript_v0_7_codex",
        "paper_exp01": LAB / "paper_draft" / "holdout_experiments",
        "old_open_source": ROOT / "Certifiable-Doppler-positioning-main",
        "old_patent": ROOT / "patent_standalone",
        "urban_raw": DATA_ROOT,
    }
    protected_before = {name: snapshot_tree(path) for name, path in protected_paths.items()}

    log("Phase 0: resolving frozen release")
    release, resolution, components = resolve_release()
    release_before = snapshot_tree(release)
    source_audit, source_hashes = source_audit_before(release)
    route_frames, fixed_segments, input_info = input_audit()
    runtime = import_release_runtime(release)
    if runtime.CANDIDATE_POOL_PROTOCOL != CANDIDATE_PROTOCOL or list(runtime.V4_CANDIDATE_MODEL_LIST) != CANDIDATES:
        raise RuntimeError("Frozen release candidate list differs from FULL_POOL_M0_M14_V1")
    geometry_table, anchor, geometry_frame = geometry_audit(release, components, runtime)
    geometry_windows = select_geometry_windows(geometry_frame)
    write_csv(DIRS["protocol"] / "PAPER_EXP02A_GEOMETRY_WINDOWS.csv", geometry_windows)
    window_protocol, _extracts = build_window_protocol(route_frames, fixed_segments, geometry_frame, geometry_windows)
    transplant = build_transplant_map(window_protocol, route_frames, anchor)
    protocol, protocol_hash = freeze_protocol(release, resolution, input_info, anchor["source_sha256"], geometry_windows, window_protocol, source_hashes)
    log(f"Protocol frozen before solver execution: {protocol_hash}")

    qatar_module = sys.modules[runtime.load_qatar_error_model.__module__]
    qatar_model = runtime.load_qatar_error_model(components["qatar_config"])
    qatar_params = qatar_profile_parameters(components["qatar_config"])
    candidates, selected, observation_inventory, truth_audit = run_candidates_and_select(runtime, qatar_module, qatar_model, qatar_params, route_frames, window_protocol, geometry_frame, anchor)
    if candidates.empty or selected.empty:
        raise RuntimeError("No solver batches completed")
    if not (candidates["execution_mode"] == "solver_rerun").all():
        raise RuntimeError("A candidate row is not a solver_rerun")
    if not (~selected["truth_used_for_selection"].astype(bool)).all():
        raise RuntimeError("Truth was marked as used for selection")

    candidate_columns = [
        "run_id", "route", "source_segment_id", "experiment_window_id", "motion_type", "geometry_window_id", "perturbation_profile", "seed",
        "candidate_model", "model_family", "execution_mode", "numerical_success", "converged", "quality_pass", "physical_plausible", "iterations", "runtime_seconds",
        "initial_position_error_m", "final_position_error_m", "mean_trajectory_position_error_m", "median_trajectory_position_error_m", "p95_trajectory_position_error_m",
        "max_trajectory_position_error_m", "mean_velocity_error_mps", "p95_velocity_error_mps", "estimated_speed_mps", "beta0_estimated_mps", "beta_dot_estimated_mps2",
        "residual_rmse_mps", "raw_validation_rmse_mps", "trimmed_validation_rmse_mps", "inlier_validation_rmse_mps", "robust_validation_cost", "condition_number",
        "outlier_evidence", "burst_outlier_evidence", "severe_risk_veto", "refine_gate_pass", "robust_gate_pass", "failure_reason", "source_observation_file", "source_config_file",
    ]
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_CANDIDATE_RESULTS.csv", candidates, candidate_columns)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_SELECTED_RESULTS.csv", selected)
    write_csv(DIRS["observations"] / "PAPER_EXP02A_OBSERVATION_INVENTORY.csv", observation_inventory)
    write_csv(DIRS["protocol"] / "PAPER_EXP02A_TRUTH_ISOLATION_AUDIT.csv", truth_audit)

    motion_aggregate = aggregate_selected(selected, ["route", "motion_type", "perturbation_profile", "evidence_mode"])
    profile_aggregate = aggregate_selected(selected, ["perturbation_profile", "evidence_mode"])
    baselines = baseline_comparison(selected, candidates)
    mode_comparison = evidence_mode_comparison(selected)
    failures = failure_cases(window_protocol, selected, candidates)
    runtime_summary = candidates.groupby(["perturbation_profile", "candidate_model"], dropna=False).agg(
        candidate_runs=("run_id", "size"),
        numerical_success_rate=("numerical_success", "mean"),
        median_runtime_seconds=("runtime_seconds", "median"),
        p95_runtime_seconds=("runtime_seconds", lambda values: pd.to_numeric(values, errors="coerce").quantile(0.95)),
        max_runtime_seconds=("runtime_seconds", "max"),
    ).reset_index()
    stats = statistical_tests(baselines)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_MOTION_TYPE_AGGREGATE.csv", motion_aggregate)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_PROFILE_AGGREGATE.csv", profile_aggregate)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_BASELINE_COMPARISON.csv", baselines)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_EVIDENCE_MODE_COMPARISON.csv", mode_comparison)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_FAILURE_CASES.csv", failures)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_RUNTIME_SUMMARY.csv", runtime_summary)
    write_csv(DIRS["outputs"] / "PAPER_EXP02A_STATISTICAL_TESTS.csv", stats)

    source_audit_final, source_unchanged = finalize_source_audit(source_audit)
    release_after = snapshot_tree(release)
    protected_after = {name: snapshot_tree(path) for name, path in protected_paths.items()}
    protected_unchanged = protected_before == protected_after
    release_unchanged = release_before == release_after
    if not source_unchanged or not release_unchanged or not protected_unchanged:
        raise RuntimeError("Frozen source/release/protected path changed during PAPER-EXP02A")
    if sha256_file(DIRS["protocol"] / "PAPER_EXP02A_PROTOCOL.json") != protocol_hash:
        raise RuntimeError("Immutable protocol changed after solver execution")

    unavailable_count = int((~window_protocol["available_for_solver_run"].astype(bool)).sum())
    primary_selected = selected[selected["evidence_mode"] == "observable-only"]
    candidate_failure_count = int((~candidates["numerical_success"].astype(bool)).sum())
    primary_median_error_m = float(pd.to_numeric(primary_selected["mean_trajectory_position_error_m"], errors="coerce").median())
    primary_median_oracle_gap_m = float(pd.to_numeric(primary_selected["selected_minus_oracle_mean_error_m"], errors="coerce").median())
    major_result_limitation = candidate_failure_count > 0 or primary_median_error_m > 1000.0 or primary_median_oracle_gap_m > 100.0
    decision = "paper_exp02a_pass_with_major_limitations" if unavailable_count or not selected["quality_pass"].astype(bool).all() or major_result_limitation else "paper_exp02a_pass"
    elapsed = time.perf_counter() - started
    metrics = {
        "task": "PAPER-EXP02A",
        "decision": decision,
        "release": str(release),
        "algorithm_version": resolution["algorithm_version"],
        "candidate_pool_protocol": CANDIDATE_PROTOCOL,
        "protocol_sha256": protocol_hash,
        "trajectory_windows_registered": len(window_protocol),
        "trajectory_windows_available": int(window_protocol["available_for_solver_run"].astype(bool).sum()),
        "trajectory_windows_unavailable": unavailable_count,
        "geometry_windows": len(geometry_windows),
        "primary_solver_batches": int(len(candidates) / 15),
        "candidate_solver_executions": len(candidates),
        "cached_replay": 0,
        "selected_rows_including_p2_mode_ablation": len(selected),
        "frozen_source_unchanged": source_unchanged,
        "release_tree_unchanged": release_unchanged,
        "protected_paths_unchanged": protected_unchanged,
        "truth_used_for_selection": False,
        "source_velocity_xyz_direction_used": False,
        "urban_gnss_doppler_used": False,
        "qatar_parameters": qatar_params,
        "model_selection_counts_observable_only": selected[selected["evidence_mode"] == "observable-only"]["selected_model"].value_counts().to_dict(),
        "profile_aggregate": profile_aggregate.to_dict(orient="records"),
        "major_limitations": [
            "Only one non-overlapping canonical Iridium geometry window is available.",
            f"The observable-only median mean trajectory error is {primary_median_error_m:.2f} m.",
            f"The observable-only median post-selection oracle gap is {primary_median_oracle_gap_m:.2f} m.",
            f"{candidate_failure_count} candidate rows reported numerical_success=False and remain in the outputs.",
            "Observations are controlled LEO range-rate, not native LEO RF data.",
            "UrbanNav absolute Tokyo coordinates/time are not retained as contemporaneous geometry.",
        ],
        "elapsed_seconds": elapsed,
    }
    write_json(DIRS["outputs"] / "PAPER_EXP02A_METRICS.json", metrics)
    report = build_report(decision, release, geometry_windows, window_protocol, candidates, selected, motion_aggregate, profile_aggregate, mode_comparison, stats, source_unchanged, protocol_hash, elapsed)
    report_path = DIRS["reports"] / "PAPER_EXP02A_URBANNAV_REAL_TRAJECTORY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    figure_script = DIRS["scripts"] / "paper_exp02a_generate_figures.py"
    spec = importlib.util.spec_from_file_location("paper_exp02a_figures", figure_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load figure script")
    figure_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figure_module)
    figure_maps = figure_module.generate_all(OUT)
    if len(figure_maps) < 9:
        raise RuntimeError("Fewer than nine figures generated")

    required = [
        report_path,
        DIRS["protocol"] / "PAPER_EXP02A_PROTOCOL.json",
        DIRS["protocol"] / "PAPER_EXP02A_PROTOCOL_SHA256.txt",
        DIRS["protocol"] / "PAPER_EXP02A_PATH_RESOLUTION.json",
        DIRS["protocol"] / "PAPER_EXP02A_INPUT_AUDIT.json",
        DIRS["protocol"] / "PAPER_EXP02A_LEO_GEOMETRY_AUDIT.csv",
        DIRS["inputs"] / "PAPER_EXP02A_TRAJECTORY_TRANSPLANT_MAP.csv",
        DIRS["protocol"] / "PAPER_EXP02A_WINDOW_PROTOCOL.csv",
        DIRS["protocol"] / "PAPER_EXP02A_FROZEN_SOURCE_AUDIT.csv",
        DIRS["protocol"] / "PAPER_EXP02A_TRUTH_ISOLATION_AUDIT.csv",
        DIRS["outputs"] / "PAPER_EXP02A_CANDIDATE_RESULTS.csv",
        DIRS["outputs"] / "PAPER_EXP02A_SELECTED_RESULTS.csv",
        DIRS["outputs"] / "PAPER_EXP02A_MOTION_TYPE_AGGREGATE.csv",
        DIRS["outputs"] / "PAPER_EXP02A_PROFILE_AGGREGATE.csv",
        DIRS["outputs"] / "PAPER_EXP02A_BASELINE_COMPARISON.csv",
        DIRS["outputs"] / "PAPER_EXP02A_EVIDENCE_MODE_COMPARISON.csv",
        DIRS["outputs"] / "PAPER_EXP02A_FAILURE_CASES.csv",
        DIRS["outputs"] / "PAPER_EXP02A_STATISTICAL_TESTS.csv",
        DIRS["outputs"] / "PAPER_EXP02A_RUNTIME_SUMMARY.csv",
        DIRS["outputs"] / "PAPER_EXP02A_METRICS.json",
        DIRS["observations"] / "PAPER_EXP02A_OBSERVATION_INVENTORY.csv",
        DIRS["figures"] / "PAPER_EXP02A_FIGURE_DATA_MAP.csv",
        DIRS["scripts"] / "paper_exp02a_run_urbannav.py",
        figure_script,
    ]
    required += sorted(DIRS["figures"].glob("figure_*.pdf")) + sorted(DIRS["figures"].glob("figure_*.png"))
    required += sorted((DIRS["source_snapshot"] / "leo_positioning").glob("*.py")) + [DIRS["source_snapshot"] / "scripts" / "release_runtime.py"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required review artifact missing/empty: {path}")
        if OUT not in path.parents:
            raise RuntimeError(f"Artifact escaped output root: {path}")
    packet = package_review(required)
    log(f"Completed decision={decision}; packet={packet}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        for directory in DIRS.values():
            directory.mkdir(parents=True, exist_ok=True)
        failure = DIRS["reports"] / "PAPER_EXP02A_FAILURE_REPORT.md"
        failure.write_text(f"# PAPER-EXP02A failure\n\nReason: `{type(exc).__name__}: {exc}`\n\n```text\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        print(f"PAPER-EXP02A_SCRIPT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
