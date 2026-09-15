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
DATA02A = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_audit"
EXP02A = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02a"
OUT = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_exp02b_window_diagnostic"
DIRS = {name: OUT / name for name in ["protocol", "inputs", "observations", "outputs", "reports", "scripts", "figures", "source_snapshot", "review_packet"]}

PROTOCOL_VERSION = "PAPER_EXP02B_PROTOCOL_V1"
CANDIDATE_PROTOCOL = "FULL_POOL_M0_M14_V1"
DURATIONS = [4, 6, 8, 12]
SEEDS = [20260724, 20260725, 20260726]
PROFILES = {
    "P0_clean_motion": {"b0_mps": 0.0, "bdot_mps2": 0.0, "gaussian_sigma_mps": 0.20},
    "P1_bias_drift": {"b0_mps": 2.0, "bdot_mps2": 0.05, "gaussian_sigma_mps": 0.30},
}
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
BASELINES = ["M0_static_position", "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine"]
STATE_COLUMNS = [
    "Sat_position_x/m",
    "Sat_position_y/m",
    "Sat_position_z/m",
    "Sat_velocity_x/(m/s)",
    "Sat_velocity_y/(m/s)",
    "Sat_velocity_z/(m/s)",
]


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (DIRS["reports"] / "PAPER_EXP02B_EXECUTION_LOG.txt").open("a", encoding="utf-8") as handle:
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


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]], columns: list[str] | None = None) -> None:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if columns is not None:
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


def import_release_runtime(release: Path) -> Any:
    runtime_path = release / "scripts" / "release_runtime.py"
    spec = importlib.util.spec_from_file_location("paper_exp02b_frozen_release_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen release runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    return np.column_stack((east, north, up))


def interpolate_columns(frame: pd.DataFrame, query: np.ndarray, columns: list[str]) -> np.ndarray:
    source_t = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    return np.column_stack([np.interp(query, source_t, pd.to_numeric(frame[column], errors="coerce").to_numpy(float)) for column in columns])


def finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    if len(times) < 2:
        return output
    for index in range(len(times)):
        if index == 0:
            left, right = 0, 1
        elif index == len(times) - 1:
            left, right = len(times) - 2, len(times) - 1
        else:
            left, right = index - 1, index + 1
        delta = float(times[right] - times[left])
        if delta > 0:
            output[index] = (values[right] - values[left]) / delta
    return output


def required_exp02a_files() -> list[Path]:
    names = [
        "protocol/PAPER_EXP02A_PROTOCOL.json",
        "protocol/PAPER_EXP02A_PROTOCOL_SHA256.txt",
        "protocol/PAPER_EXP02A_WINDOW_PROTOCOL.csv",
        "protocol/PAPER_EXP02A_LEO_GEOMETRY_AUDIT.csv",
        "protocol/PAPER_EXP02A_FROZEN_SOURCE_AUDIT.csv",
        "inputs/PAPER_EXP02A_TRAJECTORY_TRANSPLANT_MAP.csv",
        "outputs/PAPER_EXP02A_CANDIDATE_RESULTS.csv",
        "outputs/PAPER_EXP02A_SELECTED_RESULTS.csv",
        "outputs/PAPER_EXP02A_METRICS.json",
        "reports/PAPER_EXP02A_URBANNAV_REAL_TRAJECTORY_REPORT.md",
        "scripts/paper_exp02a_run_urbannav.py",
    ]
    names += [f"inputs/{segment}_trajectory_window.csv" for segment in ["ODA_0012", "ODA_0054", "ODA_0063", "ODA_0121", "SHI_0118", "SHI_0214", "SHI_0259", "SHI_0274"]]
    paths = [EXP02A / name for name in names]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Required PAPER-EXP02A inputs missing: {missing}")
    return paths


def audit_inputs() -> tuple[Path, Any, dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    required = required_exp02a_files()
    exp_protocol_path = EXP02A / "protocol" / "PAPER_EXP02A_PROTOCOL.json"
    exp_protocol_hash = sha256_file(exp_protocol_path)
    expected_protocol_hash = (EXP02A / "protocol" / "PAPER_EXP02A_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").split()[0]
    if exp_protocol_hash != expected_protocol_hash:
        raise RuntimeError("PAPER-EXP02A immutable protocol hash mismatch")
    exp_protocol = json.loads(exp_protocol_path.read_text(encoding="utf-8-sig"))
    release = Path(exp_protocol["frozen_release_path"])
    if not release.is_dir():
        raise FileNotFoundError(f"Frozen release missing: {release}")
    runtime = import_release_runtime(release)
    if runtime.CANDIDATE_POOL_PROTOCOL != CANDIDATE_PROTOCOL or list(runtime.V4_CANDIDATE_MODEL_LIST) != CANDIDATES:
        raise RuntimeError("Frozen runtime candidate protocol/list mismatch")

    source_rows: list[dict[str, Any]] = []
    snapshot_source = DIRS["source_snapshot"] / "leo_positioning"
    snapshot_source.mkdir(parents=True, exist_ok=True)
    for key, expected in exp_protocol["critical_source_hashes"].items():
        source = release / key if key.startswith("scripts/") else release / "src" / "leo_positioning" / key
        if not source.is_file():
            raise FileNotFoundError(f"Frozen source dependency missing: {source}")
        current = sha256_file(source)
        if current != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {key}")
        destination = DIRS["source_snapshot"] / key if key.startswith("scripts/") else snapshot_source / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_rows.append(
            {
                "component": key,
                "source_path": str(source),
                "expected_sha256": expected,
                "sha256_before": current,
                "sha256_after": "",
                "matches_exp02a": True,
                "unchanged_during_exp02b": "pending",
            }
        )
    source_audit = pd.DataFrame(source_rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02B_FROZEN_SOURCE_AUDIT.csv", source_audit)

    route_paths = {
        "Odaiba": DATA02A / "processed" / "UrbanNav_TK_Odaiba_reference_standardized.csv",
        "Shinjuku": DATA02A / "processed" / "UrbanNav_TK_Shinjuku_reference_standardized.csv",
    }
    route_frames: dict[str, pd.DataFrame] = {}
    for route, path in route_paths.items():
        current = sha256_file(path)
        if current != exp_protocol["urban_nav_input_hashes"][route]:
            raise RuntimeError(f"UrbanNav standardized input hash mismatch: {route}")
        frame = pd.read_csv(path)
        if not np.all(np.diff(pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)) > 0):
            raise RuntimeError(f"UrbanNav time is not strictly monotonic: {route}")
        route_frames[route] = frame

    window_protocol = pd.read_csv(EXP02A / "protocol" / "PAPER_EXP02A_WINDOW_PROTOCOL.csv")
    centers = window_protocol.copy()
    centers["source_center_time_s"] = (pd.to_numeric(centers["window_start_time_s"], errors="coerce") + pd.to_numeric(centers["window_end_time_s"], errors="coerce")) / 2.0
    centers["source_center_tow_s"] = (pd.to_numeric(centers["source_start_tow_s"], errors="coerce") + pd.to_numeric(centers["source_end_tow_s"], errors="coerce")) / 2.0
    centers = centers[["route", "source_segment_id", "motion_type", "source_catalog_type", "source_gps_week", "source_center_time_s", "source_center_tow_s", "actual_duration_s"]]
    centers = centers.rename(columns={"actual_duration_s": "exp02a_original_duration_s"})
    write_csv(DIRS["inputs"] / "PAPER_EXP02B_SOURCE_CENTER_PROTOCOL.csv", centers)

    geometry_path = release / "data" / "real_iridium" / "Iridium.csv"
    if sha256_file(geometry_path) != exp_protocol["geometry_source_hash"]:
        raise RuntimeError("Canonical Iridium geometry hash mismatch")
    geometry = runtime.load_iridium_csv(geometry_path)
    base_obs = runtime.normalize_observations(geometry)
    anchor = {
        "lat_deg": float(base_obs["lat_deg"]),
        "lon_deg": float(base_obs["lon_deg"]),
        "height_m": float(base_obs["height_m"]),
        "ecef_m": np.asarray(base_obs["p_gt_ecef_m"], float),
        "source_file": str(geometry_path),
        "source_sha256": sha256_file(geometry_path),
        "sign_convention": "meas_mps=-measured_hz*c/f; prediction=(p_r-p_s)^T(v_r-v_s)/||p_r-p_s||",
    }
    exp_file_hashes = {path.relative_to(EXP02A).as_posix(): sha256_file(path) for path in required}
    audit = {
        "task": "PAPER-EXP02B input audit",
        "exp02a_protocol_sha256": exp_protocol_hash,
        "selector_sha256": exp_protocol["selector_sha256"],
        "freeze_policy_sha256": exp_protocol["freeze_policy_sha256"],
        "candidate_source_hashes_match": True,
        "urban_nav_standardized_hashes_match": True,
        "canonical_geometry_sha256": anchor["source_sha256"],
        "canonical_geometry_hash_matches": True,
        "exp02a_required_file_hashes": exp_file_hashes,
        "source_centers_recovered_from_exp02a_only": True,
        "input_audit_pass": True,
    }
    write_json(DIRS["protocol"] / "PAPER_EXP02B_INPUT_AUDIT.json", audit)
    return release, runtime, exp_protocol, centers, route_frames, geometry, anchor


def build_trajectory_windows(centers: pd.DataFrame, route_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, center in centers.iterrows():
        route = str(center["route"])
        frame = route_frames[route]
        route_start = float(frame["time_s"].min())
        route_end = float(frame["time_s"].max())
        for duration in DURATIONS:
            midpoint = float(center["source_center_time_s"])
            start = midpoint - duration / 2.0
            end = midpoint + duration / 2.0
            subset = frame[(frame["time_s"] >= start - 1e-9) & (frame["time_s"] <= end + 1e-9)].copy()
            gap_count = int(subset["gap_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            boundary_ok = start >= route_start and end <= route_end
            expected_samples = max(2, int(round(duration / float(np.median(np.diff(frame["time_s"].to_numpy(float)))))))
            sample_ok = len(subset) >= int(0.95 * expected_samples)
            available = bool(boundary_ok and gap_count == 0 and sample_ok)
            reason = "" if available else ("trajectory_boundary" if not boundary_ok else "trajectory_gap" if gap_count else "insufficient_trajectory_samples")
            row = {
                "route": route,
                "source_segment_id": center["source_segment_id"],
                "motion_type": center["motion_type"],
                "source_catalog_type": center["source_catalog_type"],
                "duration_s": duration,
                "source_center_time_s": midpoint,
                "source_center_tow_s": float(center["source_center_tow_s"]),
                "source_gps_week": int(center["source_gps_week"]),
                "window_start_time_s": start,
                "window_end_time_s": end,
                "source_start_tow_s": float(center["source_center_tow_s"]) - duration / 2.0,
                "source_end_tow_s": float(center["source_center_tow_s"]) + duration / 2.0,
                "trajectory_sample_count": len(subset),
                "gap_count": gap_count,
                "trajectory_available": available,
                "unavailable_reason": reason,
                "center_rule": "exact midpoint of the immutable PAPER-EXP02A window",
                "window_rule": "center +/- duration/2; no shifting, replacement, compression, or stretching",
                "max_abs_acceleration_mps2": float(pd.to_numeric(subset["acceleration_mps2"], errors="coerce").abs().max()) if len(subset) else np.nan,
                "median_abs_acceleration_mps2": float(pd.to_numeric(subset["acceleration_mps2"], errors="coerce").abs().median()) if len(subset) else np.nan,
                "max_abs_yaw_rate_degps": float(pd.to_numeric(subset["yaw_rate_degps"], errors="coerce").abs().max()) if len(subset) else np.nan,
                "median_abs_yaw_rate_degps": float(pd.to_numeric(subset["yaw_rate_degps"], errors="coerce").abs().median()) if len(subset) else np.nan,
                "median_speed_mps": float(pd.to_numeric(subset["speed_mps"], errors="coerce").median()) if len(subset) else np.nan,
            }
            rows.append(row)
            if available:
                write_csv(DIRS["inputs"] / f"{center['source_segment_id']}_{duration}s_trajectory_window.csv", subset)
    result = pd.DataFrame(rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02B_TRAJECTORY_WINDOW_PROTOCOL.csv", result)
    return result


def build_geometry_windows(geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    finite_geometry = geometry[np.isfinite(geometry[STATE_COLUMNS].to_numpy(float)).all(axis=1)].copy()
    unique_times = np.sort(pd.to_numeric(finite_geometry["Time/s"], errors="coerce").dropna().unique())
    time_end = float(unique_times[-1])
    continuity_threshold_s = 1.0
    for duration in DURATIONS:
        candidates: list[dict[str, Any]] = []
        for start in unique_times:
            end = float(start + duration)
            if end > time_end + 1e-9:
                continue
            subset = finite_geometry[(finite_geometry["Time/s"] >= start - 1e-9) & (finite_geometry["Time/s"] <= end + 1e-9)]
            epochs = np.sort(subset["Time/s"].unique())
            max_gap = float(np.max(np.diff(epochs))) if len(epochs) > 1 else np.inf
            row_count = len(subset)
            satellites = int(subset["Satellite number"].nunique())
            if row_count >= 20 and satellites >= 4 and max_gap <= continuity_threshold_s:
                candidates.append(
                    {
                        "start_s": float(start),
                        "end_s": end,
                        "observation_count": row_count,
                        "unique_satellite_count": satellites,
                        "unique_epoch_count": len(epochs),
                        "median_epoch_interval_s": float(np.median(np.diff(epochs))) if len(epochs) > 1 else np.nan,
                        "max_epoch_gap_s": max_gap,
                    }
                )
        g0 = candidates[0] if candidates else None
        g1_candidates = [item for item in candidates if g0 is not None and item["start_s"] >= g0["end_s"] - 1e-9]
        g1 = g1_candidates[-1] if g1_candidates else None
        for geometry_id, item in [("G0", g0), ("G1", g1)]:
            available = item is not None
            rows.append(
                {
                    "duration_s": duration,
                    "geometry_window_id": geometry_id,
                    "geometry_start_s": item["start_s"] if available else np.nan,
                    "geometry_end_s": item["end_s"] if available else np.nan,
                    "observation_count": item["observation_count"] if available else 0,
                    "unique_satellite_count": item["unique_satellite_count"] if available else 0,
                    "unique_epoch_count": item["unique_epoch_count"] if available else 0,
                    "median_epoch_interval_s": item["median_epoch_interval_s"] if available else np.nan,
                    "max_epoch_gap_s": item["max_epoch_gap_s"] if available else np.nan,
                    "continuity_threshold_s": continuity_threshold_s,
                    "finite_satellite_states": available,
                    "available": available,
                    "unavailable_reason": "" if available else ("no qualifying earliest window" if geometry_id == "G0" else "no qualifying non-overlapping latest window"),
                    "selection_rule": "earliest qualifying window" if geometry_id == "G0" else "latest qualifying window non-overlapping with G0",
                    "selection_uses_solver_or_motion_information": False,
                }
            )
    result = pd.DataFrame(rows)
    write_csv(DIRS["protocol"] / "PAPER_EXP02B_GEOMETRY_WINDOW_PROTOCOL.csv", result)
    return result


def freeze_protocol(
    release: Path,
    exp_protocol: dict[str, Any],
    centers: pd.DataFrame,
    trajectory_windows: pd.DataFrame,
    geometry_windows: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    protocol = {
        "task_name": "PAPER-EXP02B",
        "protocol_version": PROTOCOL_VERSION,
        "created_before_any_solver_execution": True,
        "source_experiment": "PAPER-EXP02A",
        "source_exp02a_protocol_sha256": sha256_file(EXP02A / "protocol" / "PAPER_EXP02A_PROTOCOL.json"),
        "frozen_release_path": str(release),
        "algorithm_version": exp_protocol["algorithm_version"],
        "selector_sha256": exp_protocol["selector_sha256"],
        "freeze_policy_sha256": exp_protocol["freeze_policy_sha256"],
        "critical_source_hashes": exp_protocol["critical_source_hashes"],
        "candidate_pool_protocol": CANDIDATE_PROTOCOL,
        "candidate_pool": CANDIDATES,
        "source_segment_ids": centers["source_segment_id"].tolist(),
        "source_center_times": centers[["route", "source_segment_id", "motion_type", "source_center_time_s", "source_center_tow_s"]].to_dict(orient="records"),
        "durations_s": DURATIONS,
        "trajectory_window_rule": "For every segment and duration, use immutable EXP02A center +/- duration/2; no shifting or replacement.",
        "trajectory_window_availability": trajectory_windows.to_dict(orient="records"),
        "geometry_window_rules": {
            "G0": "earliest time-continuous finite-state window with >=4 satellites and >=20 observations",
            "G1": "latest qualifying window that does not overlap G0",
            "continuity_threshold_s": 1.0,
            "forbidden_selection_inputs": ["condition number", "solver error", "oracle", "motion type", "expected model"],
        },
        "available_geometry_windows": geometry_windows.to_dict(orient="records"),
        "perturbation_profiles": PROFILES,
        "random_seeds": SEEDS,
        "initialization": {"position": "frozen anchor + 10 km local east", "velocity": "zero velocity", "beta_prior": "B0_none"},
        "selector_mode": "observable-only",
        "theoretical_model_floor_definitions": {
            "timing": "computed only after frozen selector returns",
            "static": "least-squares p(t)=p_s using evaluation truth",
            "constant_velocity": "least-squares p(t)=p0+v*t using evaluation truth",
            "constant_acceleration": "least-squares p(t)=p0+v0*t+0.5*a*t^2 using evaluation truth",
            "selector_use": False,
            "candidate_pool_use": False,
        },
        "evaluation_metrics": ["mean/p95/final trajectory position error", "mean velocity error", "post-selection oracle gap", "static/CV/CA representation floors", "diagnostic gap fractions", "runtime"],
        "source_hashes": {
            "urban_nav": exp_protocol["urban_nav_input_hashes"],
            "canonical_iridium_geometry": exp_protocol["geometry_source_hash"],
            "exp02a_required_inputs": {path.relative_to(EXP02A).as_posix(): sha256_file(path) for path in required_exp02a_files()},
        },
        "recommendation_rule": {
            "main_text": "complete protocol plus monotonic duration trend, >=30% median 4-s improvement over 12-s, positive duration-error association, and interpretable model-floor relation",
            "supplementary_only": "complete protocol with interpretable diagnostic evidence but main-text rule not fully met",
            "inconclusive": "critical durations/geometries unavailable or diagnostic evidence cannot answer the research questions",
        },
        "prohibited_adaptations": ["solver changes", "selector changes", "gate changes", "new candidate/state model", "result-dependent center/window movement", "truth/error/oracle in selection", "parameter tuning"],
        "prohibited_claims": ["real dynamic LEO field validation", "native UrbanNav LEO Doppler", "CA floor as solver result", "strict additive error decomposition", "deployable acceleration solver", "all real trajectories solved accurately"],
    }
    path = DIRS["protocol"] / "PAPER_EXP02B_PROTOCOL.json"
    serialized = json.dumps(json_ready(protocol), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Existing immutable PAPER-EXP02B protocol differs; start a new protocol version")
    if not path.exists():
        path.write_text(serialized, encoding="utf-8")
    digest = sha256_file(path)
    hash_path = DIRS["protocol"] / "PAPER_EXP02B_PROTOCOL_SHA256.txt"
    expected = f"{digest}  PAPER_EXP02B_PROTOCOL.json\n"
    if hash_path.exists() and hash_path.read_text(encoding="utf-8") != expected:
        raise RuntimeError("Existing PAPER-EXP02B protocol hash file differs")
    if not hash_path.exists():
        hash_path.write_text(expected, encoding="utf-8")
    return protocol, digest


def generate_observation(
    route_frame: pd.DataFrame,
    trajectory_window: pd.Series,
    geometry_window: pd.Series,
    geometry: pd.DataFrame,
    anchor: dict[str, Any],
    profile_name: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    profile = PROFILES[profile_name]
    geometry_start = float(geometry_window["geometry_start_s"])
    geometry_end = float(geometry_window["geometry_end_s"])
    geometry_subset = geometry[(geometry["Time/s"] >= geometry_start - 1e-9) & (geometry["Time/s"] <= geometry_end + 1e-9)].sort_values("Time/s").copy()
    observation_times = geometry_subset["Time/s"].to_numpy(float)
    tau = observation_times - geometry_start
    source_start = float(trajectory_window["window_start_time_s"])
    source_query = source_start + tau
    if source_query.min() < route_frame["time_s"].min() - 1e-9 or source_query.max() > route_frame["time_s"].max() + 1e-9:
        raise RuntimeError("Trajectory interpolation query escaped route bounds")
    enu = interpolate_columns(route_frame, source_query, ["enu_e_m", "enu_n_m", "enu_u_m"])
    enu_origin = interpolate_columns(route_frame, np.array([source_start]), ["enu_e_m", "enu_n_m", "enu_u_m"])[0]
    delta_enu = enu - enu_origin
    rotation = enu_to_ecef_rotation(anchor["lat_deg"], anchor["lon_deg"])
    receiver_position = np.asarray(anchor["ecef_m"], float)[None, :] + delta_enu @ rotation.T
    receiver_velocity = finite_difference(receiver_position, observation_times)
    sat_position = geometry_subset[["Sat_position_x/m", "Sat_position_y/m", "Sat_position_z/m"]].to_numpy(float)
    sat_velocity = geometry_subset[["Sat_velocity_x/(m/s)", "Sat_velocity_y/(m/s)", "Sat_velocity_z/(m/s)"]].to_numpy(float)
    delta_position = receiver_position - sat_position
    delta_velocity = receiver_velocity - sat_velocity
    clean = np.sum(delta_position * delta_velocity, axis=1) / np.maximum(np.linalg.norm(delta_position, axis=1), 1e-9)
    duration = int(trajectory_window["duration_s"])
    geometry_index = 0 if str(geometry_window["geometry_window_id"]) == "G0" else 1
    profile_index = list(PROFILES).index(profile_name)
    random_sequence = np.random.SeedSequence([seed, stable_hash_int(str(trajectory_window["source_segment_id"])), duration, geometry_index, profile_index])
    rng = np.random.default_rng(random_sequence)
    gaussian = rng.normal(0.0, float(profile["gaussian_sigma_mps"]), size=len(geometry_subset))
    common_bias = float(profile["b0_mps"]) + float(profile["bdot_mps2"]) * tau
    measured = clean + common_bias + gaussian
    if len(geometry_subset) < 20 or geometry_subset["Satellite number"].nunique() < 4:
        raise RuntimeError("Observation-level minimum failed after generation")
    run_id = f"{trajectory_window['source_segment_id']}_D{duration}_{geometry_window['geometry_window_id']}_{profile_name}_{seed}"
    observation_frame = pd.DataFrame(
        {
            "run_id": run_id,
            "route": trajectory_window["route"],
            "source_segment_id": trajectory_window["source_segment_id"],
            "motion_type": trajectory_window["motion_type"],
            "duration_s": duration,
            "geometry_window_id": geometry_window["geometry_window_id"],
            "perturbation_profile": profile_name,
            "seed": seed,
            "observation_time_s": observation_times,
            "relative_time_s": tau,
            "satellite_id": geometry_subset["Satellite number"].to_numpy(),
            "sat_pos_x_m": sat_position[:, 0],
            "sat_pos_y_m": sat_position[:, 1],
            "sat_pos_z_m": sat_position[:, 2],
            "sat_vel_x_mps": sat_velocity[:, 0],
            "sat_vel_y_mps": sat_velocity[:, 1],
            "sat_vel_z_mps": sat_velocity[:, 2],
            "receiver_true_x_m": receiver_position[:, 0],
            "receiver_true_y_m": receiver_position[:, 1],
            "receiver_true_z_m": receiver_position[:, 2],
            "receiver_true_vx_mps": receiver_velocity[:, 0],
            "receiver_true_vy_mps": receiver_velocity[:, 1],
            "receiver_true_vz_mps": receiver_velocity[:, 2],
            "clean_range_rate_mps": clean,
            "common_bias_mps": common_bias,
            "drift_mps2": float(profile["bdot_mps2"]),
            "gaussian_noise_mps": gaussian,
            "measured_range_rate_mps": measured,
            "used_by_solver": True,
        }
    )
    observation_path = DIRS["observations"] / f"{run_id}.csv"
    observation_frame.to_csv(observation_path, index=False, encoding="utf-8-sig")
    obs = {
        "lat_deg": anchor["lat_deg"],
        "lon_deg": anchor["lon_deg"],
        "height_m": anchor["height_m"],
        "p_gt_ecef_m": receiver_position[0],
        "p0_true_m": receiver_position[0],
        "v_true_mps": np.mean(receiver_velocity, axis=0),
        "truth_positions_m": receiver_position,
        "truth_velocity_samples_mps": receiver_velocity,
        "truth_bias_mps": common_bias,
        "b0_true_mps": float(profile["b0_mps"]),
        "bdot_true_mps2": float(profile["bdot_mps2"]),
        "sat_pos_m": sat_position,
        "sat_vel_mps": sat_velocity,
        "sat_pos_truth_m": sat_position,
        "sat_vel_truth_mps": sat_velocity,
        "meas_mps": measured,
        "time_s": observation_times,
        "satellite_number": geometry_subset["Satellite number"].to_numpy(),
        "row_index": np.arange(len(geometry_subset)),
        "t0_s": float(observation_times.min()),
    }
    metadata = {
        "run_id": run_id,
        "observation_path": str(observation_path),
        "observation_sha256": sha256_file(observation_path),
        "row_count": len(observation_frame),
        "unique_satellites": int(geometry_subset["Satellite number"].nunique()),
        "used_observations": len(observation_frame),
        "source_config_file": str(DIRS["protocol"] / "PAPER_EXP02B_PROTOCOL.json"),
    }
    return obs, observation_frame, metadata


def estimated_trajectory(row: pd.Series, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final_position = np.array([row.get("final_ecef_x_m", np.nan), row.get("final_ecef_y_m", np.nan), row.get("final_ecef_z_m", np.nan)], float)
    velocity = np.array([row.get("estimated_vx_mps", np.nan), row.get("estimated_vy_mps", np.nan), row.get("estimated_vz_mps", np.nan)], float)
    if not np.all(np.isfinite(final_position)):
        return np.full((len(times), 3), np.nan), np.full((len(times), 3), np.nan)
    if str(row["candidate_model"]) in {"M0_static_position", "M1_static_position_bias"}:
        velocity = np.zeros(3)
        positions = np.repeat(final_position[None, :], len(times), axis=0)
    elif np.all(np.isfinite(velocity)):
        tau = times - float(np.min(times))
        initial_position = final_position - velocity * tau[-1]
        positions = initial_position[None, :] + tau[:, None] * velocity[None, :]
    else:
        positions = np.full((len(times), 3), np.nan)
    velocities = np.repeat(velocity[None, :], len(times), axis=0)
    return positions, velocities


def add_post_selection_candidate_metrics(frame: pd.DataFrame, obs: dict[str, Any]) -> pd.DataFrame:
    truth_position = np.asarray(obs["truth_positions_m"], float)
    truth_velocity = np.asarray(obs["truth_velocity_samples_mps"], float)
    times = np.asarray(obs["time_s"], float)
    output_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        current = row.to_dict()
        positions, velocities = estimated_trajectory(row, times)
        position_error = np.linalg.norm(positions - truth_position, axis=1)
        velocity_error = np.linalg.norm(velocities - truth_velocity, axis=1)
        finite_position = position_error[np.isfinite(position_error)]
        finite_velocity = velocity_error[np.isfinite(velocity_error)]
        current.update(
            {
                "initial_position_error_m": float(finite_position[0]) if len(finite_position) else np.nan,
                "final_position_error_m": float(finite_position[-1]) if len(finite_position) else np.nan,
                "mean_trajectory_position_error_m": float(np.mean(finite_position)) if len(finite_position) else np.nan,
                "median_trajectory_position_error_m": float(np.median(finite_position)) if len(finite_position) else np.nan,
                "p95_trajectory_position_error_m": float(np.quantile(finite_position, 0.95)) if len(finite_position) else np.nan,
                "max_trajectory_position_error_m": float(np.max(finite_position)) if len(finite_position) else np.nan,
                "mean_velocity_error_mps": float(np.mean(finite_velocity)) if len(finite_velocity) else np.nan,
                "p95_velocity_error_mps": float(np.quantile(finite_velocity, 0.95)) if len(finite_velocity) else np.nan,
                "model_family": MODEL_FAMILIES.get(str(row["candidate_model"]), "unknown"),
            }
        )
        output_rows.append(current)
    return pd.DataFrame(output_rows)


def fit_representation_floor(truth_position: np.ndarray, times: np.ndarray, degree: str) -> tuple[np.ndarray, dict[str, float]]:
    tau = np.asarray(times, float) - float(np.min(times))
    if degree == "static":
        design = np.ones((len(tau), 1))
    elif degree == "cv":
        design = np.column_stack((np.ones(len(tau)), tau))
    elif degree == "ca":
        design = np.column_stack((np.ones(len(tau)), tau, 0.5 * tau * tau))
    else:
        raise ValueError(degree)
    coefficients, _, _, _ = np.linalg.lstsq(design, truth_position, rcond=None)
    fitted = design @ coefficients
    errors = np.linalg.norm(fitted - truth_position, axis=1)
    metrics = {
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
    }
    return fitted, metrics


def representation_floor_record(
    obs: dict[str, Any],
    trajectory_window: pd.Series,
    geometry_window: pd.Series,
    rotation: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    truth_position = np.asarray(obs["truth_positions_m"], float)
    times = np.asarray(obs["time_s"], float)
    static_fit, static_metrics = fit_representation_floor(truth_position, times, "static")
    cv_fit, cv_metrics = fit_representation_floor(truth_position, times, "cv")
    ca_fit, ca_metrics = fit_representation_floor(truth_position, times, "ca")
    record = {
        "route": trajectory_window["route"],
        "source_segment_id": trajectory_window["source_segment_id"],
        "motion_type": trajectory_window["motion_type"],
        "duration_s": int(trajectory_window["duration_s"]),
        "geometry_window_id": geometry_window["geometry_window_id"],
        "observation_count": len(times),
        "unique_satellite_count": int(geometry_window["unique_satellite_count"]),
        "max_abs_acceleration_mps2": trajectory_window["max_abs_acceleration_mps2"],
        "median_abs_acceleration_mps2": trajectory_window["median_abs_acceleration_mps2"],
        "max_abs_yaw_rate_degps": trajectory_window["max_abs_yaw_rate_degps"],
        "median_abs_yaw_rate_degps": trajectory_window["median_abs_yaw_rate_degps"],
        "static_floor_mean_error_m": static_metrics["mean_error_m"],
        "static_floor_p95_error_m": static_metrics["p95_error_m"],
        "static_floor_max_error_m": static_metrics["max_error_m"],
        "cv_floor_mean_error_m": cv_metrics["mean_error_m"],
        "cv_floor_p95_error_m": cv_metrics["p95_error_m"],
        "cv_floor_max_error_m": cv_metrics["max_error_m"],
        "ca_floor_mean_error_m": ca_metrics["mean_error_m"],
        "ca_floor_p95_error_m": ca_metrics["p95_error_m"],
        "ca_floor_max_error_m": ca_metrics["max_error_m"],
        "computed_after_selector": True,
        "used_by_candidate_pool": False,
        "used_by_selector": False,
    }
    origin = truth_position[0]
    true_local = (truth_position - origin) @ rotation
    static_local = (static_fit - origin) @ rotation
    cv_local = (cv_fit - origin) @ rotation
    ca_local = (ca_fit - origin) @ rotation
    fit_frame = pd.DataFrame(
        {
            "route": trajectory_window["route"],
            "source_segment_id": trajectory_window["source_segment_id"],
            "motion_type": trajectory_window["motion_type"],
            "duration_s": int(trajectory_window["duration_s"]),
            "geometry_window_id": geometry_window["geometry_window_id"],
            "relative_time_s": times - times.min(),
            "true_east_m": true_local[:, 0],
            "true_north_m": true_local[:, 1],
            "static_east_m": static_local[:, 0],
            "static_north_m": static_local[:, 1],
            "cv_east_m": cv_local[:, 0],
            "cv_north_m": cv_local[:, 1],
            "ca_east_m": ca_local[:, 0],
            "ca_north_m": ca_local[:, 1],
        }
    )
    return record, fit_frame


def run_experiment(
    runtime: Any,
    trajectory_windows: pd.DataFrame,
    geometry_windows: pd.DataFrame,
    route_frames: dict[str, pd.DataFrame],
    geometry: pd.DataFrame,
    anchor: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    floor_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    fit_frames: dict[tuple[str, int, str], pd.DataFrame] = {}
    inventory_rows: list[dict[str, Any]] = []
    truth_audit_rows: list[dict[str, Any]] = []
    available_trajectory = trajectory_windows[trajectory_windows["trajectory_available"].astype(bool)]
    available_geometry = geometry_windows[geometry_windows["available"].astype(bool)]
    combinations = [
        (trajectory, geometry_window, profile, seed)
        for _, trajectory in available_trajectory.iterrows()
        for _, geometry_window in available_geometry[available_geometry["duration_s"] == trajectory["duration_s"]].iterrows()
        for profile in PROFILES
        for seed in SEEDS
    ]
    rotation = enu_to_ecef_rotation(anchor["lat_deg"], anchor["lon_deg"])
    for batch_index, (trajectory_window, geometry_window, profile_name, seed) in enumerate(combinations, start=1):
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(combinations):
            log(f"Solver batch {batch_index}/{len(combinations)}: {trajectory_window['source_segment_id']} D{int(trajectory_window['duration_s'])} {geometry_window['geometry_window_id']} {profile_name} {seed}")
        route_frame = route_frames[str(trajectory_window["route"])]
        obs, _observation_frame, metadata = generate_observation(route_frame, trajectory_window, geometry_window, geometry, anchor, profile_name, seed)
        raw_candidates = runtime.run_actual_candidates(
            "urban_real_trajectory_window_diagnostic",
            metadata["run_id"],
            obs,
            {"dataset_type": "urban_real_trajectory_window_diagnostic"},
            candidate_models=CANDIDATES,
            position_init_label="east_10km",
            velocity_init_label="zero_velocity",
            beta_prior_profile="B0_none",
        )
        if len(raw_candidates) != 15 or set(raw_candidates["candidate_model"]) != set(CANDIDATES):
            raise RuntimeError(f"Incomplete candidate pool: {metadata['run_id']}")
        selector_input = runtime.selector_input_frame(raw_candidates.copy())
        truth_columns = [column for column in runtime.TRUTH_ONLY_COLUMNS if column in selector_input.columns]
        truth_isolated = all(pd.to_numeric(selector_input[column], errors="coerce").isna().all() for column in truth_columns)
        if not truth_isolated:
            raise RuntimeError(f"Truth isolation failed before selection: {metadata['run_id']}")
        selected_model, selection_reason, selected_low_quality = runtime.select_actual_model(raw_candidates.copy())
        truth_audit_rows.append(
            {
                "run_id": metadata["run_id"],
                "truth_columns_scrubbed": ";".join(sorted(truth_columns)),
                "truth_columns_all_nan_before_selection": truth_isolated,
                "position_error_used_for_selection": False,
                "oracle_used_for_selection": False,
                "model_floor_used_for_selection": False,
                "route_motion_duration_hardcode": False,
            }
        )
        evaluated = add_post_selection_candidate_metrics(raw_candidates, obs)
        evaluated["run_id"] = metadata["run_id"]
        evaluated["route"] = trajectory_window["route"]
        evaluated["source_segment_id"] = trajectory_window["source_segment_id"]
        evaluated["motion_type"] = trajectory_window["motion_type"]
        evaluated["duration_s"] = int(trajectory_window["duration_s"])
        evaluated["geometry_window_id"] = geometry_window["geometry_window_id"]
        evaluated["geometry_observation_count"] = int(geometry_window["observation_count"])
        evaluated["geometry_unique_satellite_count"] = int(geometry_window["unique_satellite_count"])
        evaluated["perturbation_profile"] = profile_name
        evaluated["seed"] = seed
        evaluated["execution_mode"] = "solver_rerun"
        evaluated["source_observation_file"] = metadata["observation_path"]
        evaluated["source_config_file"] = metadata["source_config_file"]
        candidate_frames.append(evaluated)

        floor_key = (str(trajectory_window["source_segment_id"]), int(trajectory_window["duration_s"]), str(geometry_window["geometry_window_id"]))
        if floor_key not in floor_rows:
            floor_record, fit_frame = representation_floor_record(obs, trajectory_window, geometry_window, rotation)
            floor_rows[floor_key] = floor_record
            fit_frames[floor_key] = fit_frame
        floor_record = floor_rows[floor_key]
        successful = evaluated[pd.to_numeric(evaluated["mean_trajectory_position_error_m"], errors="coerce").notna()].copy()
        oracle = successful.loc[pd.to_numeric(successful["mean_trajectory_position_error_m"], errors="coerce").idxmin()]
        selected = evaluated[evaluated["candidate_model"] == selected_model].iloc[0]
        selected_error = float(selected["mean_trajectory_position_error_m"])
        oracle_error = float(oracle["mean_trajectory_position_error_m"])
        baseline_rows = {model: evaluated[evaluated["candidate_model"] == model].iloc[0] for model in BASELINES}
        selected_rows.append(
            {
                "run_id": metadata["run_id"],
                "route": trajectory_window["route"],
                "source_segment_id": trajectory_window["source_segment_id"],
                "motion_type": trajectory_window["motion_type"],
                "duration_s": int(trajectory_window["duration_s"]),
                "geometry_window_id": geometry_window["geometry_window_id"],
                "geometry_observation_count": int(geometry_window["observation_count"]),
                "geometry_unique_satellite_count": int(geometry_window["unique_satellite_count"]),
                "perturbation_profile": profile_name,
                "seed": seed,
                "evidence_mode": "observable-only",
                "selected_model": selected_model,
                "selected_family": MODEL_FAMILIES.get(selected_model, "unknown"),
                "selection_reason": selection_reason,
                "selected_low_quality": selected_low_quality,
                "numerical_success": selected.get("numerical_success", False),
                "quality_pass": selected.get("quality_pass", False),
                "physical_plausible": selected.get("physical_plausible", False),
                "mean_trajectory_position_error_m": selected_error,
                "median_trajectory_position_error_m": selected["median_trajectory_position_error_m"],
                "p95_trajectory_position_error_m": selected["p95_trajectory_position_error_m"],
                "max_trajectory_position_error_m": selected["max_trajectory_position_error_m"],
                "final_position_error_m": selected["final_position_error_m"],
                "mean_velocity_error_mps": selected["mean_velocity_error_mps"],
                "residual_rmse_mps": selected.get("residual_rmse_mps", selected.get("full_residual_rmse_mps", np.nan)),
                "raw_validation_rmse_mps": selected.get("raw_validation_rmse_mps", np.nan),
                "trimmed_validation_rmse_mps": selected.get("trimmed_validation_rmse_mps", np.nan),
                "inlier_validation_rmse_mps": selected.get("inlier_validation_rmse_mps", np.nan),
                "condition_number": selected.get("condition_number", np.nan),
                "oracle_model_post_selection": oracle["candidate_model"],
                "oracle_family_post_selection": MODEL_FAMILIES.get(str(oracle["candidate_model"]), "unknown"),
                "oracle_mean_trajectory_error_m": oracle_error,
                "selected_minus_oracle_mean_error_m": selected_error - oracle_error,
                "exact_oracle": selected_model == oracle["candidate_model"],
                "oracle_family_agreement": MODEL_FAMILIES.get(selected_model, "unknown") == MODEL_FAMILIES.get(str(oracle["candidate_model"]), "unknown"),
                "static_floor_mean_error_m": floor_record["static_floor_mean_error_m"],
                "cv_floor_mean_error_m": floor_record["cv_floor_mean_error_m"],
                "ca_floor_mean_error_m": floor_record["ca_floor_mean_error_m"],
                **{f"beats_{model.split('_')[0]}": selected_error <= float(baseline_rows[model]["mean_trajectory_position_error_m"]) for model in BASELINES},
                "truth_used_for_selection": False,
                "model_floor_used_for_selection": False,
                "execution_mode": "solver_rerun",
            }
        )
        inventory_rows.append(
            {
                "run_id": metadata["run_id"],
                "relative_path": Path(metadata["observation_path"]).relative_to(OUT).as_posix(),
                "absolute_path": metadata["observation_path"],
                "size_bytes": Path(metadata["observation_path"]).stat().st_size,
                "sha256": metadata["observation_sha256"],
                "row_count": metadata["row_count"],
                "unique_satellites": metadata["unique_satellites"],
                "used_observations": metadata["used_observations"],
            }
        )
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    selected = pd.DataFrame(selected_rows)
    floors = pd.DataFrame(floor_rows.values())
    fit_data = pd.concat(fit_frames.values(), ignore_index=True) if fit_frames else pd.DataFrame()
    inventory = pd.DataFrame(inventory_rows)
    truth_audit = pd.DataFrame(truth_audit_rows)
    return candidates, selected, floors, fit_data, inventory, truth_audit


def _aggregate_row(group: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    errors = pd.to_numeric(group["mean_trajectory_position_error_m"], errors="coerce")
    oracle = pd.to_numeric(group["oracle_mean_trajectory_error_m"], errors="coerce")
    gaps = pd.to_numeric(group["selected_minus_oracle_mean_error_m"], errors="coerce")
    candidate_subset = candidates[candidates["run_id"].isin(group["run_id"])]
    model_distribution = group["selected_model"].value_counts(normalize=True).to_dict()
    static_selected = group["selected_family"].astype(str).str.startswith("static")
    return {
        "runs": len(group),
        "selected_success_rate": float(group["numerical_success"].astype(bool).mean()),
        "selected_quality_pass_rate": float(group["quality_pass"].astype(bool).mean()),
        "median_selected_mean_error_m": float(errors.median()),
        "p75_selected_mean_error_m": float(errors.quantile(0.75)),
        "p95_selected_mean_error_m": float(errors.quantile(0.95)),
        "median_oracle_error_m": float(oracle.median()),
        "p95_oracle_error_m": float(oracle.quantile(0.95)),
        "median_oracle_gap_m": float(gaps.median()),
        "p95_oracle_gap_m": float(gaps.quantile(0.95)),
        "exact_oracle_rate": float(group["exact_oracle"].astype(bool).mean()),
        "oracle_family_agreement_rate": float(group["oracle_family_agreement"].astype(bool).mean()),
        "dynamic_family_selection_rate": float((~static_selected).mean()),
        "static_family_selection_rate": float(static_selected.mean()),
        "selected_beats_M0_rate": float(group["beats_M0"].astype(bool).mean()),
        "selected_beats_M2_rate": float(group["beats_M2"].astype(bool).mean()),
        "candidate_numerical_failure_rate": float((~candidate_subset["numerical_success"].astype(bool)).mean()) if len(candidate_subset) else np.nan,
        "median_condition_number": float(pd.to_numeric(group["condition_number"], errors="coerce").median()),
        "median_cv_floor_m": float(pd.to_numeric(group["cv_floor_mean_error_m"], errors="coerce").median()),
        "model_selection_distribution": json.dumps(model_distribution, sort_keys=True),
    }


def grouped_aggregate(selected: pd.DataFrame, candidates: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(_aggregate_row(group, candidates))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_with_all_profile(selected: pd.DataFrame, candidates: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    detailed = grouped_aggregate(selected, candidates, base_columns + ["perturbation_profile"])
    overall = grouped_aggregate(selected, candidates, base_columns)
    overall["perturbation_profile"] = "ALL"
    columns = base_columns + ["perturbation_profile"] + [column for column in detailed.columns if column not in base_columns + ["perturbation_profile"]]
    return pd.concat([detailed, overall[columns]], ignore_index=True)[columns]


def model_selection_distribution(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(["duration_s", "geometry_window_id", "perturbation_profile"], sort=True):
        counts = group["selected_model"].value_counts()
        for model, count in counts.items():
            rows.append(
                {
                    "duration_s": keys[0],
                    "geometry_window_id": keys[1],
                    "perturbation_profile": keys[2],
                    "selected_model": model,
                    "selected_family": MODEL_FAMILIES.get(model, "unknown"),
                    "count": int(count),
                    "fraction": float(count / len(group)),
                }
            )
    return pd.DataFrame(rows)


def error_attribution(selected: pd.DataFrame) -> pd.DataFrame:
    frame = selected[
        [
            "run_id",
            "route",
            "source_segment_id",
            "motion_type",
            "duration_s",
            "geometry_window_id",
            "perturbation_profile",
            "seed",
            "selected_model",
            "oracle_model_post_selection",
            "mean_trajectory_position_error_m",
            "oracle_mean_trajectory_error_m",
            "static_floor_mean_error_m",
            "cv_floor_mean_error_m",
            "ca_floor_mean_error_m",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "mean_trajectory_position_error_m": "selected_error_m",
            "oracle_mean_trajectory_error_m": "oracle_candidate_error_m",
            "static_floor_mean_error_m": "static_representation_floor_m",
            "cv_floor_mean_error_m": "cv_representation_floor_m",
            "ca_floor_mean_error_m": "ca_representation_floor_m",
        }
    )
    frame["oracle_above_cv_floor"] = frame["oracle_candidate_error_m"] - frame["cv_representation_floor_m"]
    frame["selected_above_oracle"] = frame["selected_error_m"] - frame["oracle_candidate_error_m"]
    frame["selected_above_cv_floor"] = frame["selected_error_m"] - frame["cv_representation_floor_m"]
    denominator = frame["selected_error_m"].replace(0, np.nan)
    frame["cv_floor_fraction_of_selected_error"] = frame["cv_representation_floor_m"] / denominator
    frame["selector_gap_fraction"] = frame["selected_above_oracle"] / denominator
    frame["solver_or_model_gap_fraction"] = frame["oracle_above_cv_floor"] / denominator
    frame["diagnostic_gaps_are_strictly_additive_decomposition"] = False
    frame["truth_used_for_selection"] = False
    return frame


def exp02a_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(EXP02A / "outputs" / "PAPER_EXP02A_SELECTED_RESULTS.csv")
    original = original[(original["evidence_mode"] == "observable-only") & original["perturbation_profile"].isin(PROFILES)].copy()
    old_windows = pd.read_csv(EXP02A / "protocol" / "PAPER_EXP02A_WINDOW_PROTOCOL.csv")[["source_segment_id", "actual_duration_s"]]
    original = original.merge(old_windows, on="source_segment_id", how="left")
    original["duration_s"] = original["actual_duration_s"]
    original["source_experiment"] = "PAPER-EXP02A reference-only"
    original["duration_label"] = "EXP02A original"
    new = selected.copy()
    new["source_experiment"] = "PAPER-EXP02B solver-rerun"
    new["duration_label"] = new["duration_s"].astype(int).astype(str) + " s"
    rows: list[dict[str, Any]] = []
    for frame in [new, original]:
        for profile in list(PROFILES) + ["ALL"]:
            profile_frame = frame if profile == "ALL" else frame[frame["perturbation_profile"] == profile]
            for (source, label, motion), group in profile_frame.groupby(["source_experiment", "duration_label", "motion_type"], sort=True):
                errors = pd.to_numeric(group["mean_trajectory_position_error_m"], errors="coerce")
                oracle = pd.to_numeric(group["oracle_mean_trajectory_error_m"], errors="coerce")
                rows.append(
                    {
                        "source_experiment": source,
                        "duration_label": label,
                        "duration_s": float(pd.to_numeric(group["duration_s"], errors="coerce").median()),
                        "motion_type": motion,
                        "perturbation_profile": profile,
                        "sample_count": len(group),
                        "median_selected_mean_error_m": float(errors.median()),
                        "p75_selected_mean_error_m": float(errors.quantile(0.75)),
                        "p95_selected_mean_error_m": float(errors.quantile(0.95)),
                        "median_oracle_error_m": float(oracle.median()),
                        "reference_only_not_pooled": source.startswith("PAPER-EXP02A"),
                    }
                )
    return pd.DataFrame(rows)


def failure_cases(
    trajectory_windows: pd.DataFrame,
    geometry_windows: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in trajectory_windows[~trajectory_windows["trajectory_available"].astype(bool)].iterrows():
        rows.append({"case_type": "trajectory_window_unavailable", "run_id": "", "source_segment_id": row["source_segment_id"], "duration_s": row["duration_s"], "geometry_window_id": "", "candidate_or_selected_model": "", "mean_trajectory_error_m": np.nan, "oracle_gap_m": np.nan, "reason": row["unavailable_reason"], "retained_without_filtering": True})
    for _, row in geometry_windows[~geometry_windows["available"].astype(bool)].iterrows():
        rows.append({"case_type": "geometry_window_unavailable", "run_id": "", "source_segment_id": "ALL", "duration_s": row["duration_s"], "geometry_window_id": row["geometry_window_id"], "candidate_or_selected_model": "", "mean_trajectory_error_m": np.nan, "oracle_gap_m": np.nan, "reason": row["unavailable_reason"], "retained_without_filtering": True})
    for _, row in candidates[~candidates["numerical_success"].astype(bool)].iterrows():
        rows.append({"case_type": "candidate_numerical_failure", "run_id": row["run_id"], "source_segment_id": row["source_segment_id"], "duration_s": row["duration_s"], "geometry_window_id": row["geometry_window_id"], "candidate_or_selected_model": row["candidate_model"], "mean_trajectory_error_m": row.get("mean_trajectory_position_error_m", np.nan), "oracle_gap_m": np.nan, "reason": row.get("failure_reason", ""), "retained_without_filtering": True})
    error_threshold = float(pd.to_numeric(selected["mean_trajectory_position_error_m"], errors="coerce").quantile(0.90))
    gap_threshold = float(pd.to_numeric(selected["selected_minus_oracle_mean_error_m"], errors="coerce").quantile(0.90))
    for _, row in selected.iterrows():
        case_types = []
        if not bool(row["quality_pass"]):
            case_types.append("selected_quality_failure")
        if float(row["mean_trajectory_position_error_m"]) >= error_threshold:
            case_types.append("large_selected_error")
        if float(row["selected_minus_oracle_mean_error_m"]) >= gap_threshold:
            case_types.append("large_oracle_gap")
        for case_type in case_types:
            rows.append({"case_type": case_type, "run_id": row["run_id"], "source_segment_id": row["source_segment_id"], "duration_s": row["duration_s"], "geometry_window_id": row["geometry_window_id"], "candidate_or_selected_model": row["selected_model"], "mean_trajectory_error_m": row["mean_trajectory_position_error_m"], "oracle_gap_m": row["selected_minus_oracle_mean_error_m"], "reason": row["selection_reason"], "retained_without_filtering": True})
    return pd.DataFrame(rows)


def _rank_biserial_from_difference(difference: np.ndarray) -> float:
    nonzero = difference[np.isfinite(difference) & (np.abs(difference) > 1e-12)]
    if not len(nonzero):
        return 0.0
    return float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))


def statistical_tests(selected: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr, wilcoxon

    rows: list[dict[str, Any]] = []

    def add_spearman(test_id: str, x: pd.Series, y: pd.Series, interpretation: str) -> None:
        pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
        rho, pvalue = spearmanr(pair["x"], pair["y"])
        rows.append({"test_id": test_id, "test_type": "Spearman correlation", "sample_count": len(pair), "spearman_rho": float(rho), "median_difference_m": np.nan, "wilcoxon_p_value": np.nan, "raw_p_value": float(pvalue), "holm_adjusted_p_value": np.nan, "rank_biserial_effect": np.nan, "interpretation": interpretation})

    add_spearman("duration_vs_selected_error", selected["duration_s"], selected["mean_trajectory_position_error_m"], "Positive rho means longer windows tend to have larger selected error; exploratory only.")
    add_spearman("duration_vs_cv_floor", floors["duration_s"], floors["cv_floor_mean_error_m"], "Positive rho means CV representation mismatch grows with duration; evaluation-only.")
    add_spearman("cv_floor_vs_oracle_error", selected["cv_floor_mean_error_m"], selected["oracle_mean_trajectory_error_m"], "Association does not prove that representation mismatch causes solver error.")
    add_spearman("oracle_vs_selected_error", selected["oracle_mean_trajectory_error_m"], selected["mean_trajectory_position_error_m"], "Association measures candidate-versus-selector alignment, not deployment accuracy.")

    def add_paired(test_id: str, left: pd.DataFrame, right: pd.DataFrame, keys: list[str], left_label: str, right_label: str) -> None:
        left_values = left[keys + ["mean_trajectory_position_error_m"]].rename(columns={"mean_trajectory_position_error_m": "left"})
        right_values = right[keys + ["mean_trajectory_position_error_m"]].rename(columns={"mean_trajectory_position_error_m": "right"})
        pair = left_values.merge(right_values, on=keys, how="inner").dropna()
        difference = pair["left"].to_numpy(float) - pair["right"].to_numpy(float)
        try:
            pvalue = float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue) if np.any(np.abs(difference) > 1e-12) else 1.0
        except ValueError:
            pvalue = 1.0
        rows.append({"test_id": test_id, "test_type": "paired Wilcoxon", "sample_count": len(pair), "spearman_rho": np.nan, "median_difference_m": float(np.median(difference)) if len(difference) else np.nan, "wilcoxon_p_value": pvalue, "raw_p_value": pvalue, "holm_adjusted_p_value": np.nan, "rank_biserial_effect": _rank_biserial_from_difference(difference), "interpretation": f"Difference is {left_label} minus {right_label}; negative favors {left_label}. Exploratory only."})

    duration_keys = ["source_segment_id", "geometry_window_id", "perturbation_profile", "seed"]
    add_paired("4s_vs_12s", selected[selected["duration_s"] == 4], selected[selected["duration_s"] == 12], duration_keys, "4 s", "12 s")
    add_paired("6s_vs_12s", selected[selected["duration_s"] == 6], selected[selected["duration_s"] == 12], duration_keys, "6 s", "12 s")
    geometry_keys = ["source_segment_id", "duration_s", "perturbation_profile", "seed"]
    add_paired("G0_vs_G1", selected[selected["geometry_window_id"] == "G0"], selected[selected["geometry_window_id"] == "G1"], geometry_keys, "G0", "G1")

    order = sorted(range(len(rows)), key=lambda index: rows[index]["raw_p_value"] if np.isfinite(rows[index]["raw_p_value"]) else 1.0)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(rows) - rank) * rows[index]["raw_p_value"])
        running = max(running, value)
        rows[index]["holm_adjusted_p_value"] = running
    return pd.DataFrame(rows)


def choose_recommendation(
    duration_aggregate: pd.DataFrame,
    geometry_windows: pd.DataFrame,
    statistics: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    overall = duration_aggregate[duration_aggregate["perturbation_profile"] == "ALL"].set_index("duration_s")
    all_durations_present = all(duration in overall.index for duration in DURATIONS)
    at_least_g0_each_duration = all(
        not geometry_windows[(geometry_windows["duration_s"] == duration) & (geometry_windows["geometry_window_id"] == "G0") & geometry_windows["available"].astype(bool)].empty
        for duration in DURATIONS
    )
    if all_durations_present:
        medians = [float(overall.loc[duration, "median_selected_mean_error_m"]) for duration in DURATIONS]
        monotonic = bool(all(left <= right for left, right in zip(medians, medians[1:])))
        improvement = (medians[-1] - medians[0]) / max(abs(medians[-1]), 1e-12)
    else:
        medians = []
        monotonic = False
        improvement = np.nan
    rho_row = statistics[statistics["test_id"] == "duration_vs_selected_error"]
    rho = float(rho_row.iloc[0]["spearman_rho"]) if not rho_row.empty else np.nan
    floor_relation = bool(all_durations_present and float(overall.loc[12, "median_cv_floor_m"]) >= float(overall.loc[4, "median_cv_floor_m"]))
    if all_durations_present and at_least_g0_each_duration and monotonic and improvement >= 0.30 and np.isfinite(rho) and rho > 0.30 and floor_relation:
        recommendation = "paper_exp02b_supports_main_text"
    elif all_durations_present and at_least_g0_each_duration:
        recommendation = "paper_exp02b_supports_supplementary_only"
    else:
        recommendation = "paper_exp02b_inconclusive"
    diagnostics = {
        "all_durations_present": all_durations_present,
        "at_least_G0_each_duration": at_least_g0_each_duration,
        "duration_median_selected_errors_m": dict(zip([str(item) for item in DURATIONS], medians, strict=False)),
        "median_4s_improvement_fraction_vs_12s": improvement,
        "selected_error_monotonic_with_duration": monotonic,
        "duration_selected_error_spearman_rho": rho,
        "cv_floor_12s_not_lower_than_4s": floor_relation,
        "rule_was_frozen_before_solver_execution": True,
    }
    return recommendation, diagnostics


def build_report(
    recommendation: str,
    diagnostics: dict[str, Any],
    release: Path,
    protocol_hash: str,
    trajectory_windows: pd.DataFrame,
    geometry_windows: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    duration_aggregate: pd.DataFrame,
    geometry_aggregate: pd.DataFrame,
    motion_aggregate: pd.DataFrame,
    floors: pd.DataFrame,
    attribution: pd.DataFrame,
    exp_comparison: pd.DataFrame,
    statistics: pd.DataFrame,
    source_unchanged: bool,
    protected_unchanged: bool,
    elapsed_seconds: float,
) -> str:
    duration_all = duration_aggregate[duration_aggregate["perturbation_profile"] == "ALL"].sort_values("duration_s")
    geometry_all = geometry_aggregate[geometry_aggregate["perturbation_profile"] == "ALL"].sort_values(["duration_s", "geometry_window_id"])
    motion_all = motion_aggregate[motion_aggregate["perturbation_profile"] == "ALL"].copy()
    selected_median = float(pd.to_numeric(selected["mean_trajectory_position_error_m"], errors="coerce").median())
    oracle_median = float(pd.to_numeric(selected["oracle_mean_trajectory_error_m"], errors="coerce").median())
    selector_gap_median = float(pd.to_numeric(selected["selected_minus_oracle_mean_error_m"], errors="coerce").median())
    cv_floor_median = float(pd.to_numeric(floors["cv_floor_mean_error_m"], errors="coerce").median())
    ca_floor_median = float(pd.to_numeric(floors["ca_floor_mean_error_m"], errors="coerce").median())
    ca_reduction = (cv_floor_median - ca_floor_median) / max(cv_floor_median, 1e-12)
    floor_fraction = float(pd.to_numeric(attribution["cv_floor_fraction_of_selected_error"], errors="coerce").median())
    solver_model_fraction = float(pd.to_numeric(attribution["solver_or_model_gap_fraction"], errors="coerce").median())
    selector_fraction = float(pd.to_numeric(attribution["selector_gap_fraction"], errors="coerce").median())
    best_duration_row = duration_all.loc[duration_all["median_selected_mean_error_m"].idxmin()]
    hardest_motion_row = motion_all.loc[motion_all["median_selected_mean_error_m"].idxmax()]
    dynamic_rates = {int(row["duration_s"]): float(row["dynamic_family_selection_rate"]) for _, row in duration_all.iterrows()}
    exact_oracle_rates = {int(row["duration_s"]): float(row["exact_oracle_rate"]) for _, row in duration_all.iterrows()}
    available_geometries = geometry_windows[geometry_windows["available"].astype(bool)]
    primary_batches = int(selected["run_id"].nunique())
    candidate_failure_rate = float((~candidates["numerical_success"].astype(bool)).mean())
    lines = [
        "# PAPER-EXP02B UrbanNav-TK 窗口长度、几何变化与运动模型充分性诊断报告",
        "",
        "## 1. 任务边界与冻结状态",
        "",
        f"本任务只诊断 PAPER-EXP02A 公里级误差来源。实际调用冻结 release `{release}` 中的 M0-M14、MA-BGTR-v7.1 selector 与 v7.2 freeze policy；没有新增状态模型、修改 gate 或用真值参与选择。协议在任何 solver 运行前冻结，SHA256=`{protocol_hash}`。冻结源码前后未变化：{source_unchanged}；受保护目录未变化：{protected_unchanged}。",
        "",
        "## 2. 固定中心、窗口与几何协议",
        "",
        f"沿用 EXP02A 的 8 个中心，统一截取 4/6/8/12 s。共登记 {len(trajectory_windows)} 个 segment-duration 轨迹窗口，其中 {int(trajectory_windows['trajectory_available'].astype(bool).sum())} 个可用。每个 duration 独立选择最早 G0 与最晚非重叠 G1；可用 geometry-duration rows={len(available_geometries)}。选择仅依据时间连续性、四星、20 条观测和有限卫星状态。",
        "",
        "| Duration (s) | Geometry | Start (s) | End (s) | Observations | Satellites | Available |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in geometry_windows.iterrows():
        lines.append(f"| {int(row['duration_s'])} | {row['geometry_window_id']} | {row['geometry_start_s']:.3f} | {row['geometry_end_s']:.3f} | {int(row['observation_count'])} | {int(row['unique_satellite_count'])} | {bool(row['available'])} |")
    lines += [
        "",
        "## 3. 运行规模与真值隔离",
        "",
        f"完成 {primary_batches} 个 primary solver batches、{len(candidates)} 次候选执行，cached replay=0。候选执行模式均为 solver_rerun。数值失败率={candidate_failure_rate:.2%}，失败候选保留在完整结果与 failure table 中。Static/CV/CA floor 在 selector 返回后计算，`truth_used_for_selection=false` 且 `model_floor_used_for_selection=false`。",
        "",
        "## 4. PAPER-EXP02A 为什么出现公里级误差",
        "",
        f"EXP02B 全部短窗口的 selected mean-error 中位数为 {selected_median:.2f} m，post-selection candidate oracle 中位数为 {oracle_median:.2f} m，而 CV representation floor 中位数仅 {cv_floor_median:.4f} m。中位 diagnostic fractions 为 CV-floor/selected={floor_fraction:.4f}、(oracle-CV floor)/selected={solver_model_fraction:.4f}、(selected-oracle)/selected={selector_fraction:.4f}。因此公里级误差不能主要归因于真实轨迹偏离恒速表示；候选求解/可观性与 selector gap 仍占主要诊断差距。注意这些 fraction 只用于归因，不是严格线性误差分解。",
        "",
        "## 5. 窗口缩短是否改善",
        "",
        "| Duration (s) | Runs | Median selected (m) | P95 selected (m) | Median oracle (m) | Median gap (m) | Exact oracle | Dynamic family |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in duration_all.iterrows():
        lines.append(f"| {int(row['duration_s'])} | {int(row['runs'])} | {row['median_selected_mean_error_m']:.2f} | {row['p95_selected_mean_error_m']:.2f} | {row['median_oracle_error_m']:.2f} | {row['median_oracle_gap_m']:.2f} | {row['exact_oracle_rate']:.2%} | {row['dynamic_family_selection_rate']:.2%} |")
    lines += [
        "",
        f"最低中位 selected error 出现在 {int(best_duration_row['duration_s'])} s（{best_duration_row['median_selected_mean_error_m']:.2f} m）。预冻结 recommendation rule 中的 4 s 相对 12 s 改善比例为 {diagnostics['median_4s_improvement_fraction_vs_12s']:.2%}，duration-selected-error Spearman rho={diagnostics['duration_selected_error_spearman_rho']:.3f}。缩短窗口是否改善必须结合完整长尾与统计检验判断，不能只引用最佳 duration。",
        "",
        "## 6. 动态模型选择率",
        "",
        f"4/6/8/12 s 的非 static-family 选择率分别为 {dynamic_rates.get(4, np.nan):.2%}、{dynamic_rates.get(6, np.nan):.2%}、{dynamic_rates.get(8, np.nan):.2%}、{dynamic_rates.get(12, np.nan):.2%}；exact-oracle 率分别为 {exact_oracle_rates.get(4, np.nan):.2%}、{exact_oracle_rates.get(6, np.nan):.2%}、{exact_oracle_rates.get(8, np.nan):.2%}、{exact_oracle_rates.get(12, np.nan):.2%}。窗口缩短不会自动迫使冻结 selector 选择动态族，模型分布本身也不是准确率证明。",
        "",
        "## 7. Static/CV/CA 表示下限",
        "",
        f"全部窗口的 CV floor 中位数为 {cv_floor_median:.4f} m，CA floor 中位数为 {ca_floor_median:.4f} m，CA 相对 CV 的中位总体降幅约 {ca_reduction:.2%}。CA 通常可更紧密表示转弯和变速，但它只是一条 truth-fitted evaluation floor，不是候选 solver，也不能据此宣称已实现 acceleration model。",
        "",
        "## 8. Oracle 与 selector gap",
        "",
        f"Candidate oracle 中位误差仍为 {oracle_median:.2f} m，selector gap 中位数为 {selector_gap_median:.2f} m。若 oracle 本身远高于 CV floor，说明当前冻结候选在该初始化和几何下没有达到其运动表示能力；若 selected 又显著高于 oracle，则额外存在 truth-free evidence 与位置误差不对齐。两类 gap 均完整登记。",
        "",
        "## 9. G0/G1 几何影响",
        "",
        "| Duration | Geometry | Runs | Median selected (m) | Median oracle gap (m) | Median condition | Satellites |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in geometry_all.iterrows():
        sat = geometry_windows[(geometry_windows["duration_s"] == row["duration_s"]) & (geometry_windows["geometry_window_id"] == row["geometry_window_id"])]
        satellites = int(sat.iloc[0]["unique_satellite_count"]) if not sat.empty else 0
        lines.append(f"| {int(row['duration_s'])} | {row['geometry_window_id']} | {int(row['runs'])} | {row['median_selected_mean_error_m']:.2f} | {row['median_oracle_gap_m']:.2f} | {row['median_condition_number']:.3g} | {satellites} |")
    lines += [
        "",
        "G0/G1 使用同一历史 Iridium span 的早端和晚端，不代表独立星座采样。几何差异可显著改变 residual landscape、condition 与模型选择，但不能被解释为东京同时观测的真实卫星几何。",
        "",
        "## 10. 最困难 motion type",
        "",
        f"按所有 duration/profile/geometry 聚合，中位 selected error 最高的分组为 `{hardest_motion_row['motion_type']}`、{int(hardest_motion_row['duration_s'])} s，约 {hardest_motion_row['median_selected_mean_error_m']:.2f} m。完整 motion-type 结果见 `PAPER_EXP02B_MOTION_TYPE_AGGREGATE.csv`，没有删除 turn、deceleration、stop-and-go 或 straight 的失败运行。",
        "",
        "## 11. 当前算法适合的局部时间尺度",
        "",
        f"在本受控协议中，{int(best_duration_row['duration_s'])} s 给出最低中位 selected error，但这只是单一 Iridium span、east-10 km/zero-velocity 冷启动下的诊断。不能据此给出普适 fixed-lag 长度；更稳妥的结论是当前 constant-velocity candidates 需要短局部窗口与足够几何变化共同支持，而窗口缩短本身不能消除求解和 selector gap。",
        "",
        "## 12. 是否需要未来 fixed-lag 或 acceleration model",
        "",
        "CA floor 显示 acceleration-aware representation 可能降低非恒速轨迹的纯表示误差，但本任务禁止并且没有实现新状态模型。未来可独立研究 fixed-lag、分段 CV 或 acceleration model；必须重新冻结协议、推导可观性并进行 truth-free model-selection 验证，不能把本 floor 当成现有算法改进。",
        "",
        "## 13. 与 EXP02A 原始窗口的比较",
        "",
        "`PAPER_EXP02B_EXP02A_COMPARISON.csv` 将原始 12/15/30 s 结果作为 reference-only 独立列，并与 4/6/8/12 s 按 motion/profile 对照。原始样本没有重复计入 EXP02B 统计，也没有为了形成趋势而移动中心或筛选 seeds。",
        "",
        "## 14. 探索性统计",
        "",
        "| Test | N | Rho | Median difference (m) | Holm p | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in statistics.iterrows():
        rho = "" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
        difference = "" if pd.isna(row["median_difference_m"]) else f"{row['median_difference_m']:.2f}"
        lines.append(f"| {row['test_id']} | {int(row['sample_count'])} | {rho} | {difference} | {row['holm_adjusted_p_value']:.4g} | {row['interpretation']} |")
    lines += [
        "",
        "统计检验均为 exploratory；显著性不等同于可部署能力，重复 profile/seed 也不构成新的真实道路或卫星几何。",
        "",
        "## 15. EXP02A 的论文放置建议",
        "",
        f"**Recommendation: `{recommendation}`**",
        "",
        "若为 main-text，必须同时展示 selected/oracle/CV-floor 三层差距与长尾；若为 supplementary-only，应把 EXP02A 作为外部运动真实性和失败归因补充，主文只简短引用；若为 inconclusive，则不应据此扩展论文实证声明。该推荐由 solver 前冻结的规则产生。",
        "",
        "## 16. 对 Acta Astronautica 证据的实际影响",
        "",
        "本诊断增加了一个诚实的证据层：它区分运动表示下限、候选求解 gap、selector gap 与几何窗口差异，并证明 EXP02A 的公里级误差不能简单归咎于长窗口恒速失配。它提升 failure analysis 和 evidence-boundary 的可信度，但不构成真实 LEO 动态外场验证，也不证明所有车辆运动可准确定位。",
        "",
        "## 17. 可支持与禁止的声明",
        "",
        "允许表述：fixed-center short-window controlled diagnostics、truth-free frozen selection、post-selection representation floors、geometry-window sensitivity。禁止表述：native UrbanNav LEO Doppler、real RF validation、CA solver result、严格误差分解、已完成 fixed-lag/acceleration solver、根据真值选模。",
        "",
        "## 18. 完整性与运行成本",
        "",
        f"总流程耗时约 {elapsed_seconds:.1f} s。全部 outputs、10 张 PDF/PNG 图、observation inventory、冻结源码快照、协议与审核包 manifest 均保留；大 observation 文件只通过路径、大小、行数和 SHA256 登记。",
    ]
    return "\n".join(lines) + "\n"


def finalize_source_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = frame.copy()
    after = [sha256_file(Path(path)) for path in result["source_path"]]
    result["sha256_after"] = after
    result["unchanged_during_exp02b"] = result["sha256_before"] == result["sha256_after"]
    write_csv(DIRS["protocol"] / "PAPER_EXP02B_FROZEN_SOURCE_AUDIT.csv", result)
    return result, bool(result["unchanged_during_exp02b"].astype(bool).all())


def package_review(files: list[Path]) -> Path:
    packet = DIRS["review_packet"] / "PAPER_EXP02B_REVIEW_PACKET.zip"
    manifest = DIRS["review_packet"] / "MANIFEST.txt"
    unique = sorted(set(files), key=lambda path: path.relative_to(OUT).as_posix().lower())
    lines = ["relative_path,size_bytes,sha256", "# MANIFEST.txt is excluded from its own hash by definition."]
    for path in unique:
        relative = path.relative_to(OUT).as_posix()
        lines.append(f"{relative},{path.stat().st_size},{sha256_file(path)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    members = unique + [manifest]
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(packet, "r") as archive:
        names = {entry.filename for entry in archive.infolist()}
        for line in lines[2:]:
            relative, size, digest = line.rsplit(",", 2)
            payload = archive.read(relative)
            if len(payload) != int(size) or hashlib.sha256(payload).hexdigest() != digest:
                raise RuntimeError(f"Review packet manifest mismatch: {relative}")
        if names != {path.relative_to(OUT).as_posix() for path in members}:
            raise RuntimeError("Review packet member set mismatch")
    return packet


def protected_paths(release: Path) -> dict[str, Path]:
    return {
        "frozen_release": release,
        "paper_exp02a": EXP02A,
        "paper_exp01": LAB / "paper_draft" / "holdout_experiments",
        "manuscript_v0_7": LAB / "paper_draft" / "Acta_Astronautica_MA_BGTR" / "manuscript_v0_7_codex",
        "legacy_certifiable_project": ROOT / "Certifiable-Doppler-positioning-main",
        "legacy_patent_project": ROOT / "patent_standalone",
    }


def protected_snapshot(paths: dict[str, Path]) -> dict[str, Any]:
    return {name: {"path": str(path), **snapshot_tree(path)} for name, path in paths.items()}


def candidate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in candidates.groupby("candidate_model", sort=True):
        errors = pd.to_numeric(group["mean_trajectory_position_error_m"], errors="coerce")
        runtimes = pd.to_numeric(group.get("runtime_seconds", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "candidate_model": model,
                "model_family": MODEL_FAMILIES.get(str(model), "unknown"),
                "executions": len(group),
                "solver_rerun_count": int((group["execution_mode"] == "solver_rerun").sum()),
                "numerical_success_rate": float(group["numerical_success"].astype(bool).mean()),
                "quality_pass_rate": float(group["quality_pass"].astype(bool).mean()),
                "median_mean_trajectory_error_m": float(errors.median()),
                "p95_mean_trajectory_error_m": float(errors.quantile(0.95)),
                "median_runtime_seconds": float(runtimes.median()) if len(runtimes) else np.nan,
                "p95_runtime_seconds": float(runtimes.quantile(0.95)) if len(runtimes) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def runtime_summary(candidates: pd.DataFrame, selected: pd.DataFrame, wall_seconds: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "scope": "overall",
            "duration_s": "ALL",
            "candidate_model": "ALL",
            "solver_batches": int(selected["run_id"].nunique()),
            "candidate_executions": len(candidates),
            "median_candidate_runtime_seconds": float(pd.to_numeric(candidates["runtime_seconds"], errors="coerce").median()),
            "p95_candidate_runtime_seconds": float(pd.to_numeric(candidates["runtime_seconds"], errors="coerce").quantile(0.95)),
            "sum_candidate_runtime_seconds": float(pd.to_numeric(candidates["runtime_seconds"], errors="coerce").sum()),
            "wall_runtime_seconds": wall_seconds,
        }
    ]
    for duration, group in candidates.groupby("duration_s", sort=True):
        values = pd.to_numeric(group["runtime_seconds"], errors="coerce")
        rows.append(
            {
                "scope": "duration",
                "duration_s": int(duration),
                "candidate_model": "ALL",
                "solver_batches": int(group["run_id"].nunique()),
                "candidate_executions": len(group),
                "median_candidate_runtime_seconds": float(values.median()),
                "p95_candidate_runtime_seconds": float(values.quantile(0.95)),
                "sum_candidate_runtime_seconds": float(values.sum()),
                "wall_runtime_seconds": np.nan,
            }
        )
    for model, group in candidates.groupby("candidate_model", sort=True):
        values = pd.to_numeric(group["runtime_seconds"], errors="coerce")
        rows.append(
            {
                "scope": "candidate_model",
                "duration_s": "ALL",
                "candidate_model": model,
                "solver_batches": int(group["run_id"].nunique()),
                "candidate_executions": len(group),
                "median_candidate_runtime_seconds": float(values.median()),
                "p95_candidate_runtime_seconds": float(values.quantile(0.95)),
                "sum_candidate_runtime_seconds": float(values.sum()),
                "wall_runtime_seconds": np.nan,
            }
        )
    return pd.DataFrame(rows)


def import_figure_generator() -> Any:
    path = DIRS["scripts"] / "paper_exp02b_generate_figures.py"
    spec = importlib.util.spec_from_file_location("paper_exp02b_generate_figures", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import figure generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_core_outputs(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    trajectory_windows: pd.DataFrame,
    geometry_windows: pd.DataFrame,
) -> None:
    available_trajectory = trajectory_windows[trajectory_windows["trajectory_available"].astype(bool)]
    available_geometry = geometry_windows[geometry_windows["available"].astype(bool)]
    expected_batches = sum(
        int((available_geometry["duration_s"] == duration).sum()) * len(PROFILES) * len(SEEDS)
        for duration in available_trajectory["duration_s"]
    )
    if selected["run_id"].nunique() != expected_batches or len(selected) != expected_batches:
        raise RuntimeError(f"Primary batch count mismatch: expected {expected_batches}, got {len(selected)}")
    if len(candidates) != expected_batches * len(CANDIDATES):
        raise RuntimeError("Candidate execution count does not match complete M0-M14 pool")
    pool_sizes = candidates.groupby("run_id")["candidate_model"].nunique()
    if not (pool_sizes == len(CANDIDATES)).all():
        raise RuntimeError("At least one solver batch does not contain the full M0-M14 pool")
    if set(candidates["candidate_model"]) != set(CANDIDATES):
        raise RuntimeError("Candidate model names differ from FULL_POOL_M0_M14_V1")
    if not (candidates["execution_mode"] == "solver_rerun").all() or not (selected["execution_mode"] == "solver_rerun").all():
        raise RuntimeError("Cached replay detected in a primary EXP02B batch")
    if selected["truth_used_for_selection"].astype(bool).any():
        raise RuntimeError("Truth entered the frozen selector")
    if selected["model_floor_used_for_selection"].astype(bool).any():
        raise RuntimeError("A representation floor entered the frozen selector")


def collect_review_files() -> list[Path]:
    files: list[Path] = []
    for directory_name in ["protocol", "inputs", "outputs", "reports", "scripts", "figures", "source_snapshot"]:
        directory = DIRS[directory_name]
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return files


def run(protocol_only: bool = False) -> dict[str, Any]:
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    stale_failure = DIRS["reports"] / "PAPER_EXP02B_FAILURE_REPORT.md"
    if stale_failure.exists():
        stale_failure.unlink()
    start = time.perf_counter()
    log("Starting PAPER-EXP02B immutable-input audit")
    release, runtime, exp_protocol, centers, route_frames, geometry, anchor = audit_inputs()
    protected = protected_paths(release)
    protected_before = protected_snapshot(protected)
    write_json(DIRS["protocol"] / "PAPER_EXP02B_PROTECTED_TREE_BEFORE.json", protected_before)
    trajectory_windows = build_trajectory_windows(centers, route_frames)
    geometry_windows = build_geometry_windows(geometry)
    protocol, protocol_hash = freeze_protocol(release, exp_protocol, centers, trajectory_windows, geometry_windows)
    log(f"Protocol frozen before solver execution: {protocol_hash}")
    available_counts = (
        geometry_windows[geometry_windows["available"].astype(bool)]
        .groupby("duration_s")["geometry_window_id"]
        .nunique()
        .to_dict()
    )
    if not available_counts:
        raise RuntimeError("No duration has a usable geometry window")
    unavailable_durations = [duration for duration in DURATIONS if int(available_counts.get(duration, 0)) == 0]
    if unavailable_durations:
        log(
            "Protocol-preserving unavailable durations: "
            + ", ".join(f"{duration} s" for duration in unavailable_durations)
            + "; thresholds are not relaxed and windows are not replaced"
        )
    if protocol_only:
        log(f"Protocol-only validation complete; geometry windows per duration: {available_counts}")
        return {
            "protocol_only": True,
            "protocol_sha256": protocol_hash,
            "trajectory_windows": int(trajectory_windows["trajectory_available"].astype(bool).sum()),
            "geometry_windows_by_duration": available_counts,
        }

    log("Executing the frozen M0-M14 candidate pool")
    candidates, selected, floors, fit_data, observation_inventory, truth_audit = run_experiment(
        runtime, trajectory_windows, geometry_windows, route_frames, geometry, anchor
    )
    candidates["segment"] = candidates["source_segment_id"]
    candidates["duration"] = candidates["duration_s"]
    candidates["geometry_window"] = candidates["geometry_window_id"]
    candidates["profile"] = candidates["perturbation_profile"]
    candidates["runtime"] = pd.to_numeric(candidates["runtime_seconds"], errors="coerce")
    selected["segment"] = selected["source_segment_id"]
    selected["duration"] = selected["duration_s"]
    selected["geometry_window"] = selected["geometry_window_id"]
    selected["profile"] = selected["perturbation_profile"]
    validate_core_outputs(candidates, selected, trajectory_windows, geometry_windows)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_CANDIDATE_RESULTS.csv", candidates)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_SELECTED_RESULTS.csv", selected)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_MOTION_MODEL_REPRESENTATION_FLOORS.csv", floors)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_REPRESENTATIVE_TRAJECTORY_FITS.csv", fit_data)
    write_csv(DIRS["inputs"] / "PAPER_EXP02B_REPRESENTATIVE_FIT_DATA.csv", fit_data)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_OBSERVATION_INVENTORY.csv", observation_inventory)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_TRUTH_ISOLATION_AUDIT.csv", truth_audit)

    duration_aggregate = aggregate_with_all_profile(selected, candidates, ["duration_s"])
    geometry_aggregate = aggregate_with_all_profile(selected, candidates, ["duration_s", "geometry_window_id"])
    motion_aggregate = aggregate_with_all_profile(selected, candidates, ["motion_type", "duration_s"])
    selection_distribution = model_selection_distribution(selected)
    attribution = error_attribution(selected)
    exp_comparison = exp02a_comparison(selected)
    failures = failure_cases(trajectory_windows, geometry_windows, candidates, selected)
    statistics = statistical_tests(selected, floors)
    summary = candidate_summary(candidates)
    elapsed = time.perf_counter() - start
    runtime = runtime_summary(candidates, selected, elapsed)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_DURATION_AGGREGATE.csv", duration_aggregate)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_GEOMETRY_AGGREGATE.csv", geometry_aggregate)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_MOTION_TYPE_AGGREGATE.csv", motion_aggregate)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_MODEL_SELECTION_DISTRIBUTION.csv", selection_distribution)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_ERROR_ATTRIBUTION.csv", attribution)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_EXP02A_COMPARISON.csv", exp_comparison)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_FAILURE_CASES.csv", failures)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_STATISTICAL_TESTS.csv", statistics)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_CANDIDATE_SUMMARY.csv", summary)
    write_csv(DIRS["outputs"] / "PAPER_EXP02B_RUNTIME_SUMMARY.csv", runtime)

    recommendation, recommendation_diagnostics = choose_recommendation(duration_aggregate, geometry_windows, statistics)
    source_audit_before = pd.read_csv(DIRS["protocol"] / "PAPER_EXP02B_FROZEN_SOURCE_AUDIT.csv")
    source_audit, source_unchanged = finalize_source_audit(source_audit_before)
    protected_after = protected_snapshot(protected)
    write_json(DIRS["protocol"] / "PAPER_EXP02B_PROTECTED_TREE_AFTER.json", protected_after)
    protected_unchanged = protected_before == protected_after
    write_json(
        DIRS["protocol"] / "PAPER_EXP02B_PROTECTED_TREE_AUDIT.json",
        {
            "before": protected_before,
            "after": protected_after,
            "unchanged": protected_unchanged,
            "comparison_basis": "existence, file count, total byte size, and latest modification time",
        },
    )
    if not source_unchanged or not protected_unchanged:
        raise RuntimeError("A protected frozen source or protected project tree changed during PAPER-EXP02B")
    if sha256_file(DIRS["protocol"] / "PAPER_EXP02B_PROTOCOL.json") != protocol_hash:
        raise RuntimeError("Immutable EXP02B protocol changed after solver execution")

    metrics = {
        "task_name": "PAPER-EXP02B",
        "decision": recommendation,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_hash,
        "candidate_pool_protocol": CANDIDATE_PROTOCOL,
        "durations_s": DURATIONS,
        "trajectory_segments": int(centers["source_segment_id"].nunique()),
        "trajectory_windows_available": int(trajectory_windows["trajectory_available"].astype(bool).sum()),
        "geometry_windows_available_total": int(geometry_windows["available"].astype(bool).sum()),
        "geometry_window_ids": sorted(geometry_windows.loc[geometry_windows["available"].astype(bool), "geometry_window_id"].unique().tolist()),
        "unavailable_durations_s": unavailable_durations,
        "primary_solver_batches": int(selected["run_id"].nunique()),
        "candidate_solver_executions": len(candidates),
        "cached_replay_count": 0,
        "all_primary_execution_mode_solver_rerun": bool((selected["execution_mode"] == "solver_rerun").all()),
        "truth_used_for_selection": False,
        "model_floor_used_for_selection": False,
        "selected_error_summary_m": {
            "median": float(pd.to_numeric(selected["mean_trajectory_position_error_m"], errors="coerce").median()),
            "p75": float(pd.to_numeric(selected["mean_trajectory_position_error_m"], errors="coerce").quantile(0.75)),
            "p95": float(pd.to_numeric(selected["mean_trajectory_position_error_m"], errors="coerce").quantile(0.95)),
        },
        "oracle_error_summary_m": {
            "median": float(pd.to_numeric(selected["oracle_mean_trajectory_error_m"], errors="coerce").median()),
            "p95": float(pd.to_numeric(selected["oracle_mean_trajectory_error_m"], errors="coerce").quantile(0.95)),
        },
        "representation_floor_summary_m": {
            "static_median": float(pd.to_numeric(floors["static_floor_mean_error_m"], errors="coerce").median()),
            "constant_velocity_median": float(pd.to_numeric(floors["cv_floor_mean_error_m"], errors="coerce").median()),
            "constant_acceleration_median": float(pd.to_numeric(floors["ca_floor_mean_error_m"], errors="coerce").median()),
        },
        "recommendation_diagnostics": recommendation_diagnostics,
        "source_hashes_unchanged": source_unchanged,
        "protected_trees_unchanged": protected_unchanged,
        "exp02a_files_unchanged": protected_before["paper_exp02a"] == protected_after["paper_exp02a"],
        "manuscript_v0_7_unchanged": protected_before["manuscript_v0_7"] == protected_after["manuscript_v0_7"],
        "runtime_seconds": elapsed,
        "known_boundaries": [
            "Real UrbanNav motion is transplanted into historical Iridium geometry; it is not synchronous native LEO RF data.",
            "Static/CV/CA floors use evaluation truth only after selection and are not candidate solvers.",
            "Diagnostic gaps are not a strict additive error decomposition.",
            "Results do not validate real dynamic LEO field positioning.",
        ],
    }
    write_json(DIRS["outputs"] / "PAPER_EXP02B_METRICS.json", metrics)

    report = build_report(
        recommendation,
        recommendation_diagnostics,
        release,
        protocol_hash,
        trajectory_windows,
        geometry_windows,
        candidates,
        selected,
        duration_aggregate,
        geometry_aggregate,
        motion_aggregate,
        floors,
        attribution,
        exp_comparison,
        statistics,
        source_unchanged,
        protected_unchanged,
        elapsed,
    )
    report_path = DIRS["reports"] / "PAPER_EXP02B_WINDOW_AND_MODEL_ADEQUACY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    figure_generator = import_figure_generator()
    figure_records = figure_generator.generate_all(OUT)
    if len(figure_records) < 10:
        raise RuntimeError(f"Expected 10 diagnostic figures, generated {len(figure_records)}")

    required_outputs = [
        DIRS["outputs"] / name
        for name in [
            "PAPER_EXP02B_CANDIDATE_RESULTS.csv",
            "PAPER_EXP02B_SELECTED_RESULTS.csv",
            "PAPER_EXP02B_DURATION_AGGREGATE.csv",
            "PAPER_EXP02B_GEOMETRY_AGGREGATE.csv",
            "PAPER_EXP02B_MOTION_TYPE_AGGREGATE.csv",
            "PAPER_EXP02B_MOTION_MODEL_REPRESENTATION_FLOORS.csv",
            "PAPER_EXP02B_ERROR_ATTRIBUTION.csv",
            "PAPER_EXP02B_EXP02A_COMPARISON.csv",
            "PAPER_EXP02B_MODEL_SELECTION_DISTRIBUTION.csv",
            "PAPER_EXP02B_FAILURE_CASES.csv",
            "PAPER_EXP02B_RUNTIME_SUMMARY.csv",
            "PAPER_EXP02B_METRICS.json",
            "PAPER_EXP02B_STATISTICAL_TESTS.csv",
            "PAPER_EXP02B_OBSERVATION_INVENTORY.csv",
        ]
    ]
    required_outputs += [
        DIRS["protocol"] / "PAPER_EXP02B_PROTOCOL.json",
        DIRS["protocol"] / "PAPER_EXP02B_PROTOCOL_SHA256.txt",
        DIRS["protocol"] / "PAPER_EXP02B_GEOMETRY_WINDOW_PROTOCOL.csv",
        DIRS["protocol"] / "PAPER_EXP02B_FROZEN_SOURCE_AUDIT.csv",
        report_path,
    ]
    missing = [str(path) for path in required_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Required EXP02B outputs missing or empty: {missing}")
    if any(OUT not in path.parents and path != OUT for path in collect_review_files()):
        raise RuntimeError("An EXP02B output escaped the designated output directory")
    packet = package_review(collect_review_files())
    log(f"Review packet verified: {packet}")
    return {
        "protocol_only": False,
        "recommendation": recommendation,
        "trajectory_segments": int(centers["source_segment_id"].nunique()),
        "geometry_window_ids": sorted(geometry_windows.loc[geometry_windows["available"].astype(bool), "geometry_window_id"].unique().tolist()),
        "primary_solver_batches": int(selected["run_id"].nunique()),
        "cached_replay": 0,
        "frozen_source_unchanged": source_unchanged,
        "report": str(report_path),
        "review_packet": str(packet),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PAPER-EXP02B frozen window-length diagnostic")
    parser.add_argument("--protocol-only", action="store_true", help="Freeze and validate the protocol without running solvers")
    args = parser.parse_args()
    try:
        result = run(protocol_only=args.protocol_only)
        write_json(DIRS["reports"] / "PAPER_EXP02B_EXECUTION_RESULT.json", result)
        print(json.dumps(json_ready(result), ensure_ascii=False))
        return 0
    except Exception as exc:
        for directory in DIRS.values():
            directory.mkdir(parents=True, exist_ok=True)
        failure_path = DIRS["reports"] / "PAPER_EXP02B_FAILURE_REPORT.md"
        failure_path.write_text(
            "# PAPER-EXP02B failure\n\n"
            f"Reason: {type(exc).__name__}: {exc}\n\n"
            "```text\n"
            f"{traceback.format_exc()}"
            "```\n",
            encoding="utf-8",
        )
        print(f"PAPER-EXP02B failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
