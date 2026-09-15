from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path('leo-c')
LAB = Path('leo-c_new_solver_lab')
DATA_ROOT = Path('datasets/UrbanNav/UrbanNav-TK-20181219')
RAW_ROOT = DATA_ROOT / "raw"
OUT = LAB / "paper_draft" / "external_dynamic_data" / "UrbanNav_TK_audit"
PROCESSED = OUT / "processed"
SCRIPT_PATH = OUT / "scripts" / "paper_data02a_audit_urbannav_tk.py"
ROUTES = ("Odaiba", "Shinjuku")

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
GPS_EPOCH = datetime(1980, 1, 6)

CONFIG = {
    "gap_factor": 5.0,
    "minimum_gap_s": 0.5,
    "suspicious_speed_mps": 55.0,
    "velocity_smoothing": "centered rolling median followed by centered rolling mean",
    "velocity_smoothing_window_samples": 5,
    "stationary_speed_mps": 0.5,
    "low_speed_upper_mps": 3.0,
    "medium_speed_upper_mps": 10.0,
    "turn_yaw_rate_degps": 8.0,
    "straight_yaw_rate_degps": 3.0,
    "acceleration_threshold_mps2": 0.8,
    "deceleration_threshold_mps2": -0.8,
    "minimum_straight_duration_s": 5.0,
    "minimum_turn_duration_s": 1.0,
    "minimum_accel_duration_s": 1.0,
    "minimum_stationary_duration_s": 3.0,
    "stop_go_search_s": 10.0,
}

EXPECTED_FILES = {
    "reference.csv": ("reference trajectory", True),
    "imu.csv": ("IMU and wheel-speed timing support", True),
    "rover_ublox.obs": ("rover RINEX observation and timing support", True),
    "rover_trimble.obs": ("alternate rover RINEX observation", False),
    "base_trimble.obs": ("base-station RINEX observation", False),
    "base.nav": ("broadcast navigation messages", False),
    "lidar.bag": ("LiDAR recording; outside Plan A", False),
}

PROTECTED_PATHS = [
    ROOT / "final_release",
    LAB / "final_release",
    LAB / "paper_draft" / "Acta_Astronautica_MA_BGTR" / "manuscript_v0_7_codex",
    ROOT / "Certifiable-Doppler-positioning-main",
    ROOT / "patent_standalone",
]


def clean_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_scalar(v) for v in value]
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "total_size_bytes": 0, "latest_mtime_ns": None}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "total_size_bytes": sum(p.stat().st_size for p in files),
        "latest_mtime_ns": max((p.stat().st_mtime_ns for p in files), default=None),
    }


def raw_metadata_snapshot() -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(DATA_ROOT)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in DATA_ROOT.rglob("*")
        if p.is_file()
    }


def detect_text_format(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()[:131072]
    encoding = "unknown"
    decoded = ""
    for candidate in ("utf-8-sig", "utf-8", "cp932", "latin-1"):
        try:
            decoded = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    sample = decoded[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False
    with path.open("r", encoding=encoding, errors="replace", newline="") as stream:
        head = []
        tail: deque[str] = deque(maxlen=20)
        total_lines = 0
        for line in stream:
            total_lines += 1
            stripped = line.rstrip("\r\n")
            if len(head) < 20:
                head.append(stripped)
            tail.append(stripped)
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "has_header": bool(has_header),
        "head_20": head,
        "tail_20": list(tail),
        "total_lines": total_lines,
    }


def lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * np.cos(lon)
    y = (n + h_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.column_stack((x, y, z))


def ecef_to_local_enu(ecef: np.ndarray, lat0_deg: float, lon0_deg: float, ecef0: np.ndarray) -> np.ndarray:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    rotation = np.array(
        [
            [-math.sin(lon0), math.cos(lon0), 0.0],
            [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)],
            [math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)],
        ]
    )
    return (ecef - ecef0) @ rotation.T


def finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values_2d = values[:, None] if values.ndim == 1 else values
    result = np.full(values_2d.shape, np.nan, dtype=float)
    valid = np.isfinite(times) & np.all(np.isfinite(values_2d), axis=1)
    indices = np.flatnonzero(valid)
    for order, idx in enumerate(indices):
        if len(indices) < 2:
            break
        if order == 0:
            left, right = indices[0], indices[1]
        elif order == len(indices) - 1:
            left, right = indices[-2], indices[-1]
        else:
            left, right = indices[order - 1], indices[order + 1]
        dt = times[right] - times[left]
        if dt > 0:
            result[idx] = (values_2d[right] - values_2d[left]) / dt
    return result[:, 0] if values.ndim == 1 else result


def rolling_smooth(values: np.ndarray, window: int) -> np.ndarray:
    frame = pd.DataFrame(values)
    median = frame.rolling(window=window, center=True, min_periods=1).median()
    return median.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def safe_quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.quantile(finite, q))


def gps_seconds_to_calendar_text(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""
    return (GPS_EPOCH + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S.%f GPS")


def parse_rinex_time(content: str) -> dict[str, Any] | None:
    tokens = content.split()
    if len(tokens) < 6:
        return None
    try:
        year, month, day, hour, minute = map(int, tokens[:5])
        second = float(tokens[5])
        whole = int(second)
        micro = int(round((second - whole) * 1_000_000))
        if micro == 1_000_000:
            whole += 1
            micro = 0
        dt = datetime(year, month, day, hour, minute, whole, micro)
        time_system = tokens[6] if len(tokens) >= 7 else "unknown"
        return {
            "text": f"{dt.isoformat(sep=' ')} {time_system}",
            "datetime": dt,
            "time_system": time_system,
            "gps_seconds": (dt - GPS_EPOCH).total_seconds() if time_system == "GPS" else None,
        }
    except (ValueError, OverflowError):
        return None


def parse_rinex_header(path: Path, route: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "route": route,
        "filename": path.name,
        "rinex_version": "",
        "file_type": "",
        "marker_name": "",
        "receiver_type": "",
        "antenna_type": "",
        "approximate_position": "",
        "interval": "",
        "time_of_first_obs": "",
        "time_of_last_obs": "",
        "time_system": "unknown",
        "constellations": "",
        "observation_types": "",
        "doppler_observation_types": "",
        "doppler_fields_present": False,
        "parse_status": "failed",
        "caution": "",
        "_start_gps_seconds": None,
        "_end_gps_seconds": None,
    }
    obs_types: dict[str, list[str]] = {}
    try:
        with path.open("r", encoding="ascii", errors="replace") as stream:
            for line_no, line in enumerate(stream, start=1):
                if line_no > 10000:
                    result["caution"] = "END OF HEADER not found within 10000 lines"
                    break
                padded = line.rstrip("\r\n").ljust(80)
                content = padded[:60]
                label = padded[60:80].strip()
                if label == "RINEX VERSION / TYPE":
                    result["rinex_version"] = content[:9].strip()
                    result["file_type"] = content[20:40].strip()
                elif label == "MARKER NAME":
                    result["marker_name"] = content.strip()
                elif label == "REC # / TYPE / VERS":
                    result["receiver_type"] = content[20:40].strip() or content.strip()
                elif label == "ANT # / TYPE":
                    result["antenna_type"] = content[20:40].strip() or content.strip()
                elif label == "APPROX POSITION XYZ":
                    result["approximate_position"] = " ".join(content.split())
                elif label == "INTERVAL":
                    result["interval"] = content.split()[0] if content.split() else ""
                elif label == "TIME OF FIRST OBS":
                    parsed = parse_rinex_time(content)
                    if parsed:
                        result["time_of_first_obs"] = parsed["text"]
                        result["time_system"] = parsed["time_system"]
                        result["_start_gps_seconds"] = parsed["gps_seconds"]
                elif label == "TIME OF LAST OBS":
                    parsed = parse_rinex_time(content)
                    if parsed:
                        result["time_of_last_obs"] = parsed["text"]
                        if result["time_system"] == "unknown":
                            result["time_system"] = parsed["time_system"]
                        result["_end_gps_seconds"] = parsed["gps_seconds"]
                elif label == "SYS / # / OBS TYPES":
                    system = content[0].strip()
                    if system:
                        obs_types.setdefault(system, [])
                        tokens = content[6:].split()
                        if tokens and tokens[0].isdigit():
                            tokens = tokens[1:]
                        obs_types[system].extend(tokens)
                    elif obs_types:
                        last_system = list(obs_types)[-1]
                        obs_types[last_system].extend(content[6:].split())
                elif label == "END OF HEADER":
                    result["parse_status"] = "success"
                    break
        doppler = {sys_id: [item for item in values if item.startswith("D")] for sys_id, values in obs_types.items()}
        doppler = {k: v for k, v in doppler.items() if v}
        result["constellations"] = ";".join(obs_types.keys())
        result["observation_types"] = "; ".join(f"{k}:{','.join(v)}" for k, v in obs_types.items())
        result["doppler_observation_types"] = "; ".join(f"{k}:{','.join(v)}" for k, v in doppler.items())
        result["doppler_fields_present"] = bool(doppler)
        if not obs_types and "NAVIGATION" in result["file_type"].upper():
            result["caution"] = "Navigation file has no SYS / # / OBS TYPES; Doppler fields are not applicable."
        elif not obs_types:
            result["caution"] = "No SYS / # / OBS TYPES found in header."
    except Exception as exc:
        result["caution"] = f"{type(exc).__name__}: {exc}"
    return result


def infer_reference_semantics(column: str) -> tuple[str, str, str, str]:
    mapping = {
        "GPS TOW (s)": ("GPS time of week", "s", "explicit column header", "Use with GPS Week; do not interpret as UTC."),
        "GPS Week": ("GPS week number", "week", "explicit column header", "GPS timescale; leap-second conversion is outside this audit."),
        "Latitude (deg)": ("geodetic latitude", "deg", "explicit column header", "Assumed WGS-84 only for deterministic conversion; datum documentation is not bundled."),
        "Longitude (deg)": ("geodetic longitude", "deg", "explicit column header", "Assumed WGS-84 only for deterministic conversion; datum documentation is not bundled."),
        "Ellipsoid Height (m)": ("ellipsoidal height", "m", "explicit column header", "Vertical accuracy metadata is not bundled."),
        "ECEF X (m)": ("ECEF X position", "m", "explicit column header", "Cross-checked against WGS-84 LLA conversion."),
        "ECEF Y (m)": ("ECEF Y position", "m", "explicit column header", "Cross-checked against WGS-84 LLA conversion."),
        "ECEF Z (m)": ("ECEF Z position", "m", "explicit column header", "Cross-checked against WGS-84 LLA conversion."),
        "Roll (deg)": ("roll attitude", "deg", "explicit column header", "Attitude convention/reference frame is not documented locally."),
        "Pitch (deg)": ("pitch attitude", "deg", "explicit column header", "Attitude convention/reference frame is not documented locally."),
        "Heading (deg)": ("heading attitude", "deg", "explicit column header", "Heading zero/sign convention is not documented locally."),
        "Velocity X (m/s)": ("velocity X component; frame unknown_semantics", "m/s", "explicit component/unit header only", "Velocity reference frame is not stated; retained but not relabeled as ENU/ECEF."),
        "Velocity Y (m/s)": ("velocity Y component; frame unknown_semantics", "m/s", "explicit component/unit header only", "Velocity reference frame is not stated; retained but not relabeled as ENU/ECEF."),
        "Velocity Z (m/s)": ("velocity Z component; frame unknown_semantics", "m/s", "explicit component/unit header only", "Velocity reference frame is not stated; retained but not relabeled as ENU/ECEF."),
        "Acceleration X (m/s^2)": ("acceleration X component; frame unknown_semantics", "m/s^2", "explicit component/unit header only", "Reference frame and gravity treatment are not stated."),
        "Acceleration Y (m/s^2)": ("acceleration Y component; frame unknown_semantics", "m/s^2", "explicit component/unit header only", "Reference frame and gravity treatment are not stated."),
        "Acceleration Z (m/s^2)": ("acceleration Z component; frame unknown_semantics", "m/s^2", "explicit component/unit header only", "Reference frame and gravity treatment are not stated."),
        "Angular rate X (rad/s)": ("angular-rate X component; frame unknown_semantics", "rad/s", "explicit component/unit header only", "Body/reference frame is not stated."),
        "Angular rate Y (rad/s)": ("angular-rate Y component; frame unknown_semantics", "rad/s", "explicit component/unit header only", "Body/reference frame is not stated."),
        "Angular rate Z (rad/s)": ("angular-rate Z component; frame unknown_semantics", "rad/s", "explicit component/unit header only", "Body/reference frame is not stated."),
    }
    return mapping.get(column, ("unknown_semantics", "unknown", "none", "Column meaning not confirmed."))


def audit_reference(route: str, path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    fmt = detect_text_format(path)
    frame = pd.read_csv(path, encoding=fmt["encoding"], sep=fmt["delimiter"], skipinitialspace=True)
    frame.columns = [str(col).strip() for col in frame.columns]
    schema_rows: list[dict[str, Any]] = []
    for index, column in enumerate(frame.columns):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        numeric_ratio = float(numeric.notna().mean())
        inferred_type = "numeric" if numeric_ratio >= 0.95 else "string_or_mixed"
        semantics, units, source, caution = infer_reference_semantics(column)
        values = numeric.to_numpy(float)
        schema_rows.append(
            {
                "route": route,
                "column_index": index,
                "original_column_name": column,
                "inferred_type": inferred_type,
                "confirmed_semantics": semantics,
                "units": units,
                "confirmation_source": source,
                "missing_ratio": float(frame[column].isna().mean()),
                "min": safe_quantile(values, 0.0),
                "median": safe_quantile(values, 0.5),
                "max": safe_quantile(values, 1.0),
                "usable": semantics != "unknown_semantics" and numeric_ratio >= 0.95,
                "caution": caution,
            }
        )

    required = [
        "GPS TOW (s)", "GPS Week", "Latitude (deg)", "Longitude (deg)", "Ellipsoid Height (m)",
        "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)", "Heading (deg)",
        "Velocity X (m/s)", "Velocity Y (m/s)", "Velocity Z (m/s)",
    ]
    missing_required = [name for name in required if name not in frame.columns]
    if missing_required:
        raise ValueError(f"{route} reference is missing required audit columns: {missing_required}")

    def array(name: str) -> np.ndarray:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)

    tow = array("GPS TOW (s)")
    week = array("GPS Week")
    absolute_time = week * 604800.0 + tow
    first_valid_time = absolute_time[np.isfinite(absolute_time)][0]
    relative_time = absolute_time - first_valid_time
    lat = array("Latitude (deg)")
    lon = array("Longitude (deg)")
    height = array("Ellipsoid Height (m)")
    source_ecef = np.column_stack((array("ECEF X (m)"), array("ECEF Y (m)"), array("ECEF Z (m)")))
    converted_ecef = lla_to_ecef(lat, lon, height)
    valid_origin = np.flatnonzero(np.all(np.isfinite(source_ecef), axis=1) & np.isfinite(lat) & np.isfinite(lon))
    if valid_origin.size == 0:
        raise ValueError(f"{route} has no valid position origin")
    origin_index = int(valid_origin[0])
    enu = ecef_to_local_enu(source_ecef, lat[origin_index], lon[origin_index], source_ecef[origin_index])
    derived_enu_velocity = finite_difference(enu, absolute_time)
    smoothed_enu_velocity = rolling_smooth(derived_enu_velocity, CONFIG["velocity_smoothing_window_samples"])
    selected_velocity = smoothed_enu_velocity.copy()
    source_velocity = np.column_stack((array("Velocity X (m/s)"), array("Velocity Y (m/s)"), array("Velocity Z (m/s)")))
    source_speed = np.linalg.norm(source_velocity, axis=1)
    raw_derived_speed = np.linalg.norm(derived_enu_velocity, axis=1)
    selected_speed = np.linalg.norm(selected_velocity, axis=1)
    acceleration = finite_difference(selected_speed, absolute_time)
    acceleration = rolling_smooth(acceleration[:, None], CONFIG["velocity_smoothing_window_samples"])[:, 0]
    heading = array("Heading (deg)")
    unwrapped_heading = np.rad2deg(np.unwrap(np.deg2rad(heading)))
    yaw_rate = finite_difference(unwrapped_heading, absolute_time)
    yaw_rate = rolling_smooth(yaw_rate[:, None], CONFIG["velocity_smoothing_window_samples"])[:, 0]

    dt = np.diff(absolute_time)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else float("nan")
    gap_threshold = max(CONFIG["minimum_gap_s"], CONFIG["gap_factor"] * median_dt) if np.isfinite(median_dt) else CONFIG["minimum_gap_s"]
    gap_edges = np.flatnonzero(dt > gap_threshold)
    nonpositive_edges = np.flatnonzero(dt <= 0)
    gap_flag = np.zeros(len(frame), dtype=bool)
    for edge in np.concatenate((gap_edges, nonpositive_edges)):
        gap_flag[edge] = True
        if edge + 1 < len(gap_flag):
            gap_flag[edge + 1] = True
    jumps = np.linalg.norm(np.diff(source_ecef, axis=0), axis=1)
    jump_speed = np.divide(jumps, dt, out=np.full_like(jumps, np.nan), where=dt > 0)
    suspicious_edge = np.flatnonzero(jump_speed > CONFIG["suspicious_speed_mps"])
    suspicious = np.zeros(len(frame), dtype=bool)
    for edge in suspicious_edge:
        suspicious[edge] = True
        suspicious[edge + 1] = True
    missing_position = ~np.all(np.isfinite(source_ecef), axis=1)
    nonfinite = ~np.isfinite(absolute_time) | missing_position
    timestamp_valid = np.isfinite(absolute_time)
    interpolation_flag = np.zeros(len(frame), dtype=bool)
    quality_flag = np.full(len(frame), "valid", dtype=object)
    quality_flag[gap_flag] = "gap_adjacent_or_nonmonotonic"
    quality_flag[suspicious] = "suspicious_reference_jump"
    quality_flag[missing_position] = "missing_position"

    standardized = pd.DataFrame(
        {
            "route": route,
            "source_row": np.arange(1, len(frame) + 1),
            "source_time": [f"GPSW{int(w)}:TOW{v:.6f}" if np.isfinite(w) and np.isfinite(v) else "" for w, v in zip(week, tow)],
            "time_s": relative_time,
            "timestamp_valid": timestamp_valid,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "ellipsoidal_height_m": height,
            "ecef_x_m": source_ecef[:, 0],
            "ecef_y_m": source_ecef[:, 1],
            "ecef_z_m": source_ecef[:, 2],
            "enu_e_m": enu[:, 0],
            "enu_n_m": enu[:, 1],
            "enu_u_m": enu[:, 2],
            "source_vx": source_velocity[:, 0],
            "source_vy": source_velocity[:, 1],
            "source_vz": source_velocity[:, 2],
            "source_speed_norm_mps": source_speed,
            "derived_ve_mps": derived_enu_velocity[:, 0],
            "derived_vn_mps": derived_enu_velocity[:, 1],
            "derived_vu_mps": derived_enu_velocity[:, 2],
            "smoothed_ve_mps": smoothed_enu_velocity[:, 0],
            "smoothed_vn_mps": smoothed_enu_velocity[:, 1],
            "smoothed_vu_mps": smoothed_enu_velocity[:, 2],
            "selected_ve_mps": selected_velocity[:, 0],
            "selected_vn_mps": selected_velocity[:, 1],
            "selected_vu_mps": selected_velocity[:, 2],
            "speed_mps": selected_speed,
            "acceleration_mps2": acceleration,
            "yaw_deg": heading,
            "yaw_rate_degps": yaw_rate,
            "position_quality_flag": quality_flag,
            "velocity_source": "derived_enu_from_reference_ecef_central_difference_smoothed_for_analysis",
            "interpolation_flag": interpolation_flag,
            "gap_flag": gap_flag,
        }
    )

    route_length = float(np.nansum(np.linalg.norm(np.diff(enu[:, :2], axis=0), axis=1)))
    lla_ecef_error = np.linalg.norm(source_ecef - converted_ecef, axis=1)
    speed_difference = source_speed - raw_derived_speed
    quality = {
        "route": route,
        "sample_count": len(frame),
        "start_time": gps_seconds_to_calendar_text(float(np.nanmin(absolute_time))),
        "end_time": gps_seconds_to_calendar_text(float(np.nanmax(absolute_time))),
        "start_gps_week": int(week[np.isfinite(week)][0]),
        "start_gps_tow_s": float(tow[np.isfinite(tow)][0]),
        "end_gps_week": int(week[np.isfinite(week)][-1]),
        "end_gps_tow_s": float(tow[np.isfinite(tow)][-1]),
        "duration_s": float(np.nanmax(absolute_time) - np.nanmin(absolute_time)),
        "median_sampling_interval_s": median_dt,
        "p5_sampling_interval_s": safe_quantile(positive_dt, 0.05),
        "p95_sampling_interval_s": safe_quantile(positive_dt, 0.95),
        "duplicate_timestamp_count": int(np.sum(dt == 0)),
        "out_of_order_count": int(np.sum(dt < 0)),
        "gap_threshold_s": gap_threshold,
        "gap_count": int(len(gap_edges)),
        "maximum_gap_s": safe_quantile(positive_dt, 1.0),
        "missing_position_count": int(missing_position.sum()),
        "non_finite_count": int(nonfinite.sum()),
        "position_jump_median_m": safe_quantile(jumps, 0.5),
        "position_jump_p95_m": safe_quantile(jumps, 0.95),
        "position_jump_p99_m": safe_quantile(jumps, 0.99),
        "position_jump_max_m": safe_quantile(jumps, 1.0),
        "suspicious_reference_jump_count": int(len(suspicious_edge)),
        "approximate_route_length_m": route_length,
        "altitude_min_m": safe_quantile(height, 0.0),
        "altitude_max_m": safe_quantile(height, 1.0),
        "altitude_range_m": float(np.nanmax(height) - np.nanmin(height)),
        "source_velocity_present": True,
        "source_velocity_frame": "unknown_semantics",
        "source_speed_median_mps": safe_quantile(source_speed, 0.5),
        "source_speed_max_mps": safe_quantile(source_speed, 1.0),
        "derived_speed_median_mps": safe_quantile(raw_derived_speed, 0.5),
        "derived_speed_max_mps": safe_quantile(raw_derived_speed, 1.0),
        "source_vs_derived_speed_rmse_mps": float(np.sqrt(np.nanmean(speed_difference * speed_difference))),
        "lla_to_source_ecef_error_median_m": safe_quantile(lla_ecef_error, 0.5),
        "lla_to_source_ecef_error_max_m": safe_quantile(lla_ecef_error, 1.0),
        "velocity_derivation_method": "central difference in local ENU; one-sided endpoints",
        "smoothing_method": CONFIG["velocity_smoothing"],
        "smoothing_window_samples": CONFIG["velocity_smoothing_window_samples"],
        "smoothing_used_only_for_selected_analysis_velocity": True,
    }
    details = {
        "format": fmt,
        "columns": list(frame.columns),
        "row_count": len(frame),
        "time_system": "GPS",
        "time_unit": "GPS week plus seconds of week",
        "coordinate_evidence": "Explicit LLA and ECEF headers; WGS-84 conversion cross-check performed.",
        "source_velocity_frame": "unknown_semantics",
        "source_velocity_crosscheck": {
            "source_vs_derived_speed_rmse_mps": quality["source_vs_derived_speed_rmse_mps"],
            "note": "Norm agreement is a consistency check and does not confirm the source component frame.",
        },
        "absolute_start_gps_seconds": float(np.nanmin(absolute_time)),
        "absolute_end_gps_seconds": float(np.nanmax(absolute_time)),
    }
    return standardized, schema_rows, quality, details


def classify_motion(trajectory: pd.DataFrame) -> np.ndarray:
    speed = trajectory["speed_mps"].to_numpy(float)
    accel = trajectory["acceleration_mps2"].to_numpy(float)
    yaw_rate = trajectory["yaw_rate_degps"].to_numpy(float)
    gap = trajectory["gap_flag"].astype(bool).to_numpy()
    suspicious = trajectory["position_quality_flag"].eq("suspicious_reference_jump").to_numpy()
    labels = np.full(len(trajectory), "medium_speed_straight", dtype=object)
    labels[speed < CONFIG["stationary_speed_mps"]] = "stationary"
    moving = speed >= CONFIG["stationary_speed_mps"]
    labels[moving & (speed < CONFIG["low_speed_upper_mps"])] = "low_speed_straight"
    labels[moving & (speed >= CONFIG["low_speed_upper_mps"]) & (speed < CONFIG["medium_speed_upper_mps"])] = "medium_speed_straight"
    labels[moving & (speed >= CONFIG["medium_speed_upper_mps"])] = "high_speed_straight"
    labels[moving & (accel >= CONFIG["acceleration_threshold_mps2"])] = "acceleration"
    labels[moving & (accel <= CONFIG["deceleration_threshold_mps2"])] = "deceleration"
    labels[moving & (yaw_rate >= CONFIG["turn_yaw_rate_degps"])] = "right_turn"
    labels[moving & (yaw_rate <= -CONFIG["turn_yaw_rate_degps"])] = "left_turn"
    curved = moving & (np.abs(yaw_rate) >= CONFIG["straight_yaw_rate_degps"]) & (np.abs(yaw_rate) < CONFIG["turn_yaw_rate_degps"])
    labels[curved] = "curved_motion"
    labels[suspicious] = "suspicious_reference_jump"
    labels[gap] = "trajectory_gap"
    return labels


def segment_catalog(route: str, trajectory: pd.DataFrame) -> list[dict[str, Any]]:
    labels = classify_motion(trajectory)
    times = trajectory["time_s"].to_numpy(float)
    source_times = trajectory["source_time"].astype(str).to_numpy()
    enu = trajectory[["enu_e_m", "enu_n_m", "enu_u_m"]].to_numpy(float)
    speed = trajectory["speed_mps"].to_numpy(float)
    accel = trajectory["acceleration_mps2"].to_numpy(float)
    yaw_rate = trajectory["yaw_rate_degps"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    start = 0
    segment_number = 1
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            end = idx - 1
            duration = max(0.0, times[end] - times[start]) if np.isfinite(times[end]) and np.isfinite(times[start]) else 0.0
            kind = str(labels[start])
            minimum = 0.0
            if "straight" in kind:
                minimum = CONFIG["minimum_straight_duration_s"]
            elif "turn" in kind:
                minimum = CONFIG["minimum_turn_duration_s"]
            elif kind in ("acceleration", "deceleration"):
                minimum = CONFIG["minimum_accel_duration_s"]
            elif kind == "stationary":
                minimum = CONFIG["minimum_stationary_duration_s"]
            if duration >= minimum or kind in ("trajectory_gap", "suspicious_reference_jump", "curved_motion"):
                sub = slice(start, end + 1)
                distance = float(np.nansum(np.linalg.norm(np.diff(enu[sub], axis=0), axis=1))) if end > start else 0.0
                rows.append(
                    {
                        "route": route,
                        "segment_id": f"{route[:3].upper()}_{segment_number:04d}",
                        "segment_type": kind,
                        "start_time": source_times[start],
                        "end_time": source_times[end],
                        "duration_s": duration,
                        "sample_count": end - start + 1,
                        "distance_m": distance,
                        "median_speed_mps": safe_quantile(speed[sub], 0.5),
                        "max_speed_mps": safe_quantile(speed[sub], 1.0),
                        "median_acceleration_mps2": safe_quantile(accel[sub], 0.5),
                        "max_abs_yaw_rate_degps": safe_quantile(np.abs(yaw_rate[sub]), 1.0),
                        "gap_free": kind not in ("trajectory_gap", "suspicious_reference_jump") and not bool(trajectory["gap_flag"].iloc[start:end + 1].any()),
                        "recommended_for_exp02a": False,
                        "recommendation_reason": "",
                        "_start_index": start,
                        "_end_index": end,
                    }
                )
                segment_number += 1
            start = idx

    # Add explicit stop-and-go windows without changing the underlying non-overlapping catalog.
    stationary_rows = [row for row in rows if row["segment_type"] == "stationary"]
    for stationary in stationary_rows:
        end_index = int(stationary["_end_index"])
        search_end = int(np.searchsorted(times, times[end_index] + CONFIG["stop_go_search_s"], side="right"))
        moving_indices = np.flatnonzero(speed[end_index + 1:search_end] >= CONFIG["low_speed_upper_mps"])
        if moving_indices.size:
            moving_end = end_index + 1 + int(moving_indices[-1])
            start_index = int(stationary["_start_index"])
            rows.append(
                {
                    "route": route,
                    "segment_id": f"{route[:3].upper()}_{segment_number:04d}",
                    "segment_type": "stop_and_go",
                    "start_time": source_times[start_index],
                    "end_time": source_times[moving_end],
                    "duration_s": float(times[moving_end] - times[start_index]),
                    "sample_count": moving_end - start_index + 1,
                    "distance_m": float(np.nansum(np.linalg.norm(np.diff(enu[start_index:moving_end + 1], axis=0), axis=1))),
                    "median_speed_mps": safe_quantile(speed[start_index:moving_end + 1], 0.5),
                    "max_speed_mps": safe_quantile(speed[start_index:moving_end + 1], 1.0),
                    "median_acceleration_mps2": safe_quantile(accel[start_index:moving_end + 1], 0.5),
                    "max_abs_yaw_rate_degps": safe_quantile(np.abs(yaw_rate[start_index:moving_end + 1]), 1.0),
                    "gap_free": not bool(trajectory["gap_flag"].iloc[start_index:moving_end + 1].any()),
                    "recommended_for_exp02a": False,
                    "recommendation_reason": "",
                    "_start_index": start_index,
                    "_end_index": moving_end,
                }
            )
            segment_number += 1

    categories = {
        "constant-speed straight": {"low_speed_straight", "medium_speed_straight", "high_speed_straight"},
        "turn": {"left_turn", "right_turn"},
        "acceleration/deceleration": {"acceleration", "deceleration"},
        "stop-and-go": {"stop_and_go"},
    }
    for category, kinds in categories.items():
        candidates = [row for row in rows if row["segment_type"] in kinds and row["gap_free"]]
        if candidates:
            best = max(candidates, key=lambda item: (float(item["duration_s"]), float(item["distance_m"])))
            best["recommended_for_exp02a"] = True
            best["recommendation_reason"] = f"Longest gap-free {category} candidate selected using trajectory-only criteria."

    for row in rows:
        row.pop("_start_index", None)
        row.pop("_end_index", None)
    return rows


def audit_imu(route: str, path: Path, reference: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fmt = detect_text_format(path)
    frame = pd.read_csv(path, encoding=fmt["encoding"], sep=fmt["delimiter"], skipinitialspace=True)
    frame.columns = [str(col).strip() for col in frame.columns]
    tow_col = next((c for c in frame.columns if c == "GPS TOW (s)"), None)
    week_col = next((c for c in frame.columns if c == "GPS Week"), None)
    if tow_col and week_col:
        tow = pd.to_numeric(frame[tow_col], errors="coerce").to_numpy(float)
        week = pd.to_numeric(frame[week_col], errors="coerce").to_numpy(float)
        absolute = week * 604800.0 + tow
        time_system = "GPS"
    else:
        absolute = np.full(len(frame), np.nan)
        time_system = "unknown"
    dt = np.diff(absolute)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive)) if positive.size else float("nan")
    gap_threshold = max(CONFIG["minimum_gap_s"], CONFIG["gap_factor"] * median_dt) if np.isfinite(median_dt) else CONFIG["minimum_gap_s"]
    start = safe_quantile(absolute, 0.0)
    end = safe_quantile(absolute, 1.0)
    ref_start = reference["absolute_start_gps_seconds"]
    ref_end = reference["absolute_end_gps_seconds"]
    overlap = max(0.0, min(end, ref_end) - max(start, ref_start)) if start is not None and end is not None else 0.0
    row = {
        "route": route,
        "filename": path.name,
        "encoding": fmt["encoding"],
        "delimiter": repr(fmt["delimiter"]),
        "has_header": fmt["has_header"],
        "row_count": len(frame),
        "column_names": ";".join(frame.columns),
        "time_fields": ";".join(c for c in (tow_col, week_col) if c),
        "time_system": time_system,
        "start_time": gps_seconds_to_calendar_text(start) if start is not None else "",
        "end_time": gps_seconds_to_calendar_text(end) if end is not None else "",
        "duration_s": (end - start) if start is not None and end is not None else None,
        "median_sampling_interval_s": median_dt,
        "p5_sampling_interval_s": safe_quantile(positive, 0.05),
        "p95_sampling_interval_s": safe_quantile(positive, 0.95),
        "duplicate_timestamp_count": int(np.sum(dt == 0)),
        "out_of_order_count": int(np.sum(dt < 0)),
        "gap_count": int(np.sum(dt > gap_threshold)),
        "maximum_gap_s": safe_quantile(positive, 1.0),
        "gyroscope_units": "rad/s (explicit header)" if any("Angular rate" in c and "rad/s" in c for c in frame.columns) else "unknown",
        "acceleration_units": "m/s^2 (explicit header)" if any("Acceleration" in c and "m/s^2" in c for c in frame.columns) else "unknown",
        "wheel_velocity_units": "m/s (explicit header)" if any("Wheel velocity" in c and "m/s" in c for c in frame.columns) else "unknown",
        "reference_overlap_duration_s": overlap,
        "parse_status": "success",
        "caution": "IMU axes/reference frame are not documented locally; IMU is not used as positioning truth.",
    }
    details = {
        "absolute_start_gps_seconds": start,
        "absolute_end_gps_seconds": end,
        "time_system": time_system,
        "format": fmt,
    }
    return row, details


def alignment_row(
    route: str,
    source_a: str,
    source_b: str,
    system_a: str,
    system_b: str,
    start_a: float | None,
    end_a: float | None,
    start_b: float | None,
    end_b: float | None,
    unresolved: str = "",
) -> dict[str, Any]:
    comparable = (
        start_a is not None and end_a is not None and start_b is not None and end_b is not None
        and system_a == "GPS" and system_b == "GPS"
    )
    if comparable:
        overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
        start_offset = start_b - start_a
        end_offset = end_b - end_a
        conversion_required = False
        conversion_rule = "Direct comparison in GPS timescale; no UTC conversion performed."
    else:
        overlap = None
        start_offset = None
        end_offset = None
        conversion_required = system_a != system_b or system_a == "unknown" or system_b == "unknown"
        conversion_rule = "Unresolved; do not apply UTC/GPS conversion without explicit timescale metadata."
    return {
        "route": route,
        "source_a": source_a,
        "source_b": source_b,
        "time_system_a": system_a,
        "time_system_b": system_b,
        "start_offset_s": start_offset,
        "end_offset_s": end_offset,
        "overlap_duration_s": overlap,
        "direct_alignment_possible": bool(comparable and overlap is not None and overlap > 0),
        "conversion_required": bool(conversion_required),
        "conversion_rule": conversion_rule,
        "unresolved_issue": unresolved if unresolved else ("" if comparable else "One or both time ranges/timescales are unavailable."),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if columns is not None:
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[columns]
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def bool_text(value: Any) -> str:
    return "yes" if bool(value) else "no"


def build_design_draft(decision: str, qualities: dict[str, dict[str, Any]], segments: list[dict[str, Any]]) -> str:
    recommended = [row for row in segments if row["recommended_for_exp02a"]]
    lines = [
        "# PAPER-EXP02A Design Draft: UrbanNav-TK Real-Trajectory-Driven LEO Doppler Experiment",
        "",
        "## 1. Scope and evidence boundary",
        "",
        "PAPER-EXP02A will be a **real-trajectory-driven LEO Doppler controlled experiment**. UrbanNav provides the receiver motion trajectory; project-local Iridium/LEO states provide the satellite geometry; Doppler observations are generated under declared controlled noise models. It is not a real dynamic LEO Doppler field trial because the UrbanNav receiver did not record native LEO signal observations.",
        "",
        f"Audit readiness decision: `{decision}`.",
        "",
        "## 2. Immutable inputs",
        "",
        "1. Use the two standardized reference files produced by PAPER-DATA02A, preserving source rows and GPS Week/TOW.",
        "2. Use existing project-local Iridium satellite position/velocity states without modifying the frozen geometry source.",
        "3. Use the TECH18 `FULL_POOL_M0_M14_V1` candidate protocol and the frozen MA-BGTR-v7.1 selector plus v7.2 limitation policy unchanged.",
        "4. Optionally apply the audited Qatar physical-layer error configuration only as a perturbation prior, never as positioning truth.",
        "",
        "## 3. Time mapping",
        "",
        "Map each selected UrbanNav segment's relative time `time_s` to a contiguous interval of the existing LEO geometry timeline. Preserve all UrbanNav inter-sample intervals. Use `t_leo = t_leo_start + (t_urban - t_urban_start)`; do not infer UTC from GPS time. Reject mappings that exceed the available geometry span. Record the source GPS Week/TOW, mapped LEO epoch, and offset in a mapping table.",
        "",
        "## 4. Position and velocity interpolation",
        "",
        "Interpolate ECEF position with a shape-preserving local rule at the LEO observation epochs. Use piecewise linear interpolation as the primary reproducible rule; do not bridge audited gaps. Derive velocity from the interpolated ECEF trajectory or interpolate the audited ENU-derived velocity consistently, then rotate to ECEF. Retain raw central-difference and smoothed auxiliary velocity separately. No future samples may cross a validation block boundary.",
        "",
        "## 5. Motion retained",
        "",
        "The selected protocol must include straight constant-speed motion, turns, acceleration/deceleration, stop-and-go where present, and nonuniform speed. Segment selection is based only on trajectory kinematics and quality flags, not on MA-BGTR outcomes.",
        "",
        "Recommended trajectory-only candidates:",
        "",
        "| Route | Segment | Type | Duration (s) | Median speed (m/s) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in recommended:
        lines.append(f"| {row['route']} | {row['segment_id']} | {row['segment_type']} | {row['duration_s']:.1f} | {row['median_speed_mps']:.2f} |")
    lines += [
        "",
        "## 6. Controlled LEO Doppler generation",
        "",
        "For each mapped epoch and satellite, compute line-of-sight range rate using the frozen sign convention and the real receiver ECEF position/velocity. Generate clean Doppler-equivalent range-rate observations, then add a declared base Gaussian component. A second protocol may inject Qatar-calibrated CFO-like shape, confidence/dropout, and heavy-tail/burst perturbations using the frozen DATA03 configuration. UrbanNav GNSS Doppler must not be substituted for LEO Doppler.",
        "",
        "## 7. Initialization and candidate execution",
        "",
        "Use zero-velocity cold start as the primary initialization. Truth velocity is forbidden as the main initializer; any half-truth initialization may appear only in a clearly labeled sensitivity appendix. Execute frozen M0-M14 and the frozen selector without threshold changes. Receiver truth is available only after model selection for evaluation.",
        "",
        "## 8. Evaluation metrics",
        "",
        "Report final/mean/p95 position error, velocity error, bias/drift error where simulated, residual and blocked-validation metrics, selected-minus-oracle gap, model/family selection counts, convergence, runtime, initialization sensitivity, and per-motion-type results. Preserve failures and quality flags.",
        "",
        "## 9. Difference from PAPER-EXP01",
        "",
        "PAPER-EXP01 uses project-defined synthetic/hold-out motion. PAPER-EXP02A replaces the receiver trajectory with independently recorded urban vehicle motion while retaining controlled LEO geometry and generated observations. It therefore probes motion realism and model adaptation, not native LEO field performance.",
        "",
        "## 10. Claims",
        "",
        "Supported if the future experiment passes: performance under independently recorded real vehicle kinematics in a controlled LEO Doppler simulation; behavior across stops, turns, acceleration, and speed regimes; external trajectory sensitivity of the frozen selector.",
        "",
        "Not supported: real dynamic LEO Doppler field validation; native LEO receiver performance; UrbanNav-to-Iridium clock synchronization; end-to-end RF/CFO estimation; or proof that residual reduction implies navigation accuracy.",
        "",
        "## 11. Minimum PAPER-EXP02A inputs",
        "",
        "- Standardized Odaiba and Shinjuku trajectories plus the motion catalog.",
        "- Frozen Iridium satellite-state file and its time span.",
        "- An explicit relative-time mapping table.",
        "- Frozen base-noise and optional Qatar perturbation configurations.",
        "- Frozen candidate protocol, selector/freeze-policy hashes, canonical seeds, and zero-velocity initialization profile.",
        "- Segment exclusion rules for gaps and suspicious jumps.",
    ]
    return "\n".join(lines) + "\n"


def build_report(
    decision: str,
    inventory: list[dict[str, Any]],
    qualities: dict[str, dict[str, Any]],
    reference_details: dict[str, dict[str, Any]],
    imu_rows: list[dict[str, Any]],
    rinex_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    expected_missing: dict[str, list[str]],
    protected_unchanged: bool,
    raw_unchanged: bool,
    elapsed_s: float,
) -> str:
    lines = [
        "# PAPER-DATA02A UrbanNav-TK-20181219 数据审计报告",
        "",
        "## 1. 数据目录与环境",
        "",
        f"- 逻辑根目录：`{ROOT}`",
        f"- 原始数据目录：`{DATA_ROOT}`",
        f"- 输出目录：`{OUT}`",
        f"- Python：`{sys.executable}`",
        f"- 审计耗时：{elapsed_s:.1f} s",
        f"- 原始数据只读检查：{bool_text(raw_unchanged)}",
        f"- 冻结工程、论文 v0.7 与旧项目未修改：{bool_text(protected_unchanged)}",
        "",
        "本任务未运行 MA-BGTR 定位实验，也未修改 solver、selector、gate、候选池或冻结结果。",
        "",
        "## 2. 文件完整性",
        "",
        f"递归登记 {len(inventory)} 个文件。除 `lidar.bag` 外均以流式方式计算 SHA256；LiDAR 仅登记大小，不解析。",
    ]
    for route in ROUTES:
        missing = expected_missing[route]
        lines.append(f"- {route}：核心预期文件 {'完整' if not missing else '缺失 ' + ', '.join(missing)}。")
    lines += [
        "",
        "`raw_archive/Tokyo_Data.zip` 作为源归档登记并校验哈希，但不解压、不作为方案 A 运行输入。",
        "",
        "## 3. Odaiba 数据结构",
        "",
    ]
    for route in ROUTES:
        details = reference_details[route]
        quality = qualities[route]
        if route == "Shinjuku":
            lines += ["", "## 4. Shinjuku 数据结构", ""]
        lines += [
            f"`reference.csv` 为 {details['format']['encoding']} 编码、逗号分隔且含表头，共 {details['row_count']:,} 行数据、{len(details['columns'])} 列。表头明确给出 GPS Week/TOW、LLA、ECEF、姿态、速度、加速度和角速度。",
            f"时间跨度为 {quality['duration_s']:.1f} s，中位采样间隔 {quality['median_sampling_interval_s']:.6f} s；近似路线长度 {quality['approximate_route_length_m']:.1f} m。",
        ]
    lines += [
        "",
        "## 5. Reference schema",
        "",
        "两条路线的 reference schema 一致，共 20 列。时间语义由 `GPS TOW (s)` 与 `GPS Week` 表头直接确认；位置同时提供 geodetic LLA 与 ECEF；姿态提供 roll/pitch/heading。速度、加速度和角速度的单位由表头确认，但 X/Y/Z 的参考坐标架未在本地说明中给出，因此其分量坐标架标为 `unknown_semantics`，不强行重命名为 ENU、NED 或 ECEF。",
        "原始 reference 没有原生 ENU/NED 位置列，也没有 solution status、position quality、covariance/standard-deviation 或其他精度字段；标准化文件中的 ENU 和 quality flag 均为本审计可追踪生成，不能误写成数据集原生字段。",
        "",
        "## 6. 轨迹质量",
        "",
        "| Route | Samples | Duration (s) | Median dt (s) | Duplicates | Out of order | Gaps | Max gap (s) | Route length (m) | Suspicious jumps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTES:
        q = qualities[route]
        lines.append(
            f"| {route} | {q['sample_count']} | {q['duration_s']:.1f} | {q['median_sampling_interval_s']:.6f} | {q['duplicate_timestamp_count']} | {q['out_of_order_count']} | {q['gap_count']} | {q['maximum_gap_s']:.6f} | {q['approximate_route_length_m']:.1f} | {q['suspicious_reference_jump_count']} |"
        )
    lines += [
        "",
        "跳点以相邻 ECEF 位移除以时间间隔超过 55 m/s 的保守阈值标记，仅用于排除候选区段，不改写轨迹。LLA 按 WGS-84 常数转换到 ECEF，并与源 ECEF 做一致性检查；源 ECEF 始终保留。",
        "",
        "## 7. 坐标与时间系统",
        "",
        "Reference 时间系统明确为 GPS Week + GPS time-of-week seconds。审计只在 GPS timescale 内比较，不执行 GPS-to-UTC 转换，也不猜测闰秒。坐标列明确包含 latitude/longitude/ellipsoidal height 与 ECEF XYZ；由于未随数据提供 datum 文档，WGS-84 仅作为确定性转换假设，并在输出中明确记录。每条路线以首个有效点建立本地 ENU。",
        "",
        "## 8. 速度是否直接提供",
        "",
        "Reference 直接提供 `Velocity X/Y/Z (m/s)`，原始列和速度模均保留。但其分量坐标架没有本地证据，因此标准化文件另外从 ECEF 位置和 GPS 时间计算原始 central-difference ENU velocity；首尾采用 one-sided difference。",
        "",
        "## 9. 派生速度可靠性",
        "",
    ]
    for route in ROUTES:
        q = qualities[route]
        lines.append(
            f"- {route}：source speed 与原始位置差分 speed 的 RMSE 为 {q['source_vs_derived_speed_rmse_mps']:.3f} m/s；derived speed 中位数 {q['derived_speed_median_mps']:.2f} m/s、最大值 {q['derived_speed_max_mps']:.2f} m/s。"
        )
    lines += [
        "",
        "0.1 s 高频位置支持速度差分，但厘米/毫米级位置量化仍会放大到速度噪声。输出同时保留原始差分和 5-sample centered rolling median + mean 的辅助平滑版本；平滑结果只用于运动区段分析，未覆盖原始位置或原始速度。后续实验必须把该滤波参数作为协议参数。",
        "",
        "## 10. 轨迹运动特征与推荐区段",
        "",
    ]
    for route in ROUTES:
        counts = Counter(row["segment_type"] for row in segment_rows if row["route"] == route)
        lines.append(f"- {route}：" + "，".join(f"{k}={v}" for k, v in sorted(counts.items())) + "。")
    lines += [
        "",
        "推荐区段仅依据速度、加速度、heading-rate、持续时间和 gap flag 自动选择，未使用任何 MA-BGTR 结果。左/右转标签采用 heading 增减方向的常规解释；因本地未提供 heading 正方向说明，后续应把它作为 turn-direction caution，而转弯幅度本身仍可用。",
        "",
        "| Route | Segment | Type | Duration (s) | Median speed (m/s) | Reason |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in segment_rows:
        if row["recommended_for_exp02a"]:
            lines.append(f"| {row['route']} | {row['segment_id']} | {row['segment_type']} | {row['duration_s']:.1f} | {row['median_speed_mps']:.2f} | {row['recommendation_reason']} |")
    lines += [
        "",
        "## 11. RINEX Doppler 字段",
        "",
        "审计实际读取每个 RINEX 文件的 header，并解析 `SYS / # / OBS TYPES`；没有因为文件扩展名而假定 Doppler 存在。",
        "",
        "| Route | File | Version | Type | Time system | Doppler present | Doppler observation types |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rinex_rows:
        lines.append(f"| {row['route']} | {row['filename']} | {row['rinex_version']} | {row['file_type']} | {row['time_system']} | {bool_text(row['doppler_fields_present'])} | {row['doppler_observation_types'] or 'not applicable/not found'} |")
    lines += [
        "",
        "Observation files中存在 `D*` Doppler observation types；`base.nav` 是导航文件，没有 `SYS / # / OBS TYPES`，不能据此宣称含 Doppler。PAPER-EXP02A 方案 A 不使用这些原始 GNSS Doppler；该发现仅为时间对齐和未来方案 B 保留条件。",
        "",
        "## 12. IMU 与时间对齐",
        "",
    ]
    for row in imu_rows:
        lines.append(f"- {row['route']} IMU：{row['row_count']:,} 行，中位间隔 {row['median_sampling_interval_s']:.6f} s，reference 重叠 {row['reference_overlap_duration_s']:.1f} s；时间列同为 GPS Week/TOW。")
    direct = sum(bool(row["direct_alignment_possible"]) for row in alignment_rows)
    lines += [
        "",
        f"时间对齐表共 {len(alignment_rows)} 对，其中 {direct} 对可在明确 GPS timescale 下直接比较。RINEX header 缺失末时刻或导航文件缺乏可比 observation time range 的项目保留为 unresolved；未凭猜测转换 UTC/GPS。",
        "",
        "## 13. 推荐区段",
        "",
        "每条路线优先保留一个 gap-free 直行匀速段、一个转弯段、一个加/减速段，以及数据存在时一个停车再启动段。完整区段、时间、样本数和运动统计见 `PAPER_DATA02A_MOTION_SEGMENT_CATALOG.csv`。",
        "",
        "## 14. 方案 A 可行性",
        "",
        "方案 A 可行：使用 UrbanNav 的真实车辆位置轨迹和由位置差分获得的 ENU 速度，映射到项目已有 Iridium/LEO 卫星状态时间轴，生成受控 LEO Doppler。该流程不需要使用 UrbanNav 原始 GNSS Doppler。参考轨迹提供足够的时间、位置、转弯与非匀速信息。主要注意项是 reference solution 的绝对精度说明未随本地数据提供、源速度坐标架未确认、以及 LEO 时间轴映射本质上是受控重映射而非传感器同步。",
        "",
        "## 15. 能支持的论文声明",
        "",
        "- 冻结求解器可在独立记录的真实城市车辆运动轨迹驱动下接受受控验证；",
        "- 可评估停车、低速、直行、转弯、加减速和较高速运动对模型选择的影响；",
        "- 可称为 `real-trajectory-driven LEO Doppler controlled experiment`；",
        "- 可与 PAPER-EXP01 的人工/受控轨迹形成外部运动形态互补。",
        "",
        "## 16. 不能支持的论文声明",
        "",
        "- 不能称为 `real dynamic LEO Doppler field validation`；",
        "- 不能宣称 UrbanNav 含 Iridium/LEO 原生 Doppler；",
        "- 不能用 GNSS RINEX Doppler 代替 LEO Doppler 后仍称 LEO 外场验证；",
        "- 不能把残差下降等同于定位精度提升；",
        "- 不能在缺少本地精度元数据时夸大 reference 的绝对真值精度。",
        "",
        "## 17. 未解决问题",
        "",
        "1. 本地数据包没有 README 或 reference solution accuracy/covariance 说明；需要人工补充 UrbanNav 官方说明或论文引用。",
        "2. Source Velocity X/Y/Z 的坐标架未在表头中确认；PAPER-EXP02A 应使用已审计的派生 ENU 速度，或先取得正式字段定义。",
        "3. Heading 正方向与参考轴未在本地说明；turn direction 标签需保守解释。",
        "4. UrbanNav GPS 时间与未来选取的 LEO 几何时间并非同场同步；必须使用显式相对时间重映射表。",
        "5. RINEX 时间尾字段若缺失，不应通过全文件扫描猜测；方案 B 需单独完成严格 epoch audit。",
        "",
        "## 18. 是否允许进入 PAPER-EXP02A",
        "",
        f"**Decision: `{decision}`**",
        "",
        "位置和 GPS 时间字段完整，轨迹频率足以派生速度，两条路线包含多样运动区段，且可建立可审计的受控时间映射。因此允许进入 PAPER-EXP02A 设计实现；上述元数据缺口作为声明边界，不通过修改冻结算法解决。",
        "",
        "## 19. 下一步最小任务",
        "",
        "1. 冻结本审计输出和选定 segment IDs；",
        "2. 选择一份项目内 Iridium satellite-state timeline，并记录 SHA256 和可用时长；",
        "3. 生成 UrbanNav-relative-time 到 LEO-time 的显式映射表；",
        "4. 实现只读 trajectory adapter 与 gap-aware interpolation；",
        "5. 先做 clean/base-noise 受控观测，再追加 frozen Qatar-calibrated perturbation；",
        "6. 用 zero-velocity cold start 和 frozen M0-M14/selector 执行 PAPER-EXP02A，真值只在选择后评估。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.perf_counter()
    if ROOT != Path('leo-c'):
        raise RuntimeError("ROOT mismatch")
    if not DATA_ROOT.exists() or not RAW_ROOT.exists():
        raise FileNotFoundError(f"UrbanNav data not found: {DATA_ROOT}")
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if OUT not in SCRIPT_PATH.parents:
        raise RuntimeError("Script path escaped output directory")

    protected_before = {str(path): snapshot_tree(path) for path in PROTECTED_PATHS}
    raw_before = raw_metadata_snapshot()

    inventory_rows: list[dict[str, Any]] = []
    parse_status: dict[str, tuple[bool, str]] = {}
    files = sorted((p for p in DATA_ROOT.rglob("*") if p.is_file()), key=lambda p: str(p).lower())
    print(f"Discovered {len(files)} raw files", flush=True)
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(DATA_ROOT)
        route = next((part for part in rel.parts if part in ROUTES), "archive_or_shared")
        role, required = EXPECTED_FILES.get(path.name.lower(), ("source archive or auxiliary file", False))
        digest = ""
        if path.suffix.lower() != ".bag":
            print(f"Hashing {index}/{len(files)}: {rel} ({path.stat().st_size} bytes)", flush=True)
            digest = sha256_file(path)
        else:
            parse_status[str(path)] = (False, "LiDAR bag intentionally not parsed or hashed per task scope")
        inventory_rows.append(
            {
                "route": route,
                "relative_path": str(rel),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds"),
                "sha256": digest,
                "expected_role": role,
                "required_for_plan_a": required,
                "parsed_successfully": False,
                "parse_error": "pending",
            }
        )

    # Validate the archive container without extracting it.
    for path in files:
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as archive:
                    _ = archive.infolist()
                parse_status[str(path)] = (True, "")
            except Exception as exc:
                parse_status[str(path)] = (False, f"Archive directory parse failed: {exc}")

    standardized_by_route: dict[str, pd.DataFrame] = {}
    schema_rows: list[dict[str, Any]] = []
    qualities: dict[str, dict[str, Any]] = {}
    reference_details: dict[str, dict[str, Any]] = {}
    expected_missing: dict[str, list[str]] = {}
    for route in ROUTES:
        route_dir = RAW_ROOT / route
        expected_missing[route] = [name for name in EXPECTED_FILES if not (route_dir / name).exists()]
        reference_path = route_dir / "reference.csv"
        if not reference_path.exists():
            raise FileNotFoundError(f"Required reference missing for {route}")
        trajectory, route_schema, quality, details = audit_reference(route, reference_path)
        standardized_by_route[route] = trajectory
        schema_rows.extend(route_schema)
        qualities[route] = quality
        reference_details[route] = details
        output_name = f"UrbanNav_TK_{route}_reference_standardized.csv"
        trajectory.to_csv(PROCESSED / output_name, index=False, encoding="utf-8-sig", float_format="%.9f")
        parse_status[str(reference_path)] = (True, "")
        print(f"Parsed {route} reference: {len(trajectory)} rows", flush=True)

    imu_rows: list[dict[str, Any]] = []
    imu_details: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        path = RAW_ROOT / route / "imu.csv"
        if path.exists():
            row, details = audit_imu(route, path, reference_details[route])
            imu_rows.append(row)
            imu_details[route] = details
            parse_status[str(path)] = (True, "")

    rinex_rows_internal: list[dict[str, Any]] = []
    for route in ROUTES:
        for filename in ("rover_ublox.obs", "rover_trimble.obs", "base_trimble.obs", "base.nav"):
            path = RAW_ROOT / route / filename
            if not path.exists():
                continue
            row = parse_rinex_header(path, route)
            rinex_rows_internal.append(row)
            parse_status[str(path)] = (row["parse_status"] == "success", "" if row["parse_status"] == "success" else row["caution"])

    segment_rows: list[dict[str, Any]] = []
    for route, trajectory in standardized_by_route.items():
        segment_rows.extend(segment_catalog(route, trajectory))

    alignment_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        ref = reference_details[route]
        if route in imu_details:
            imu = imu_details[route]
            alignment_rows.append(
                alignment_row(route, "reference.csv", "imu.csv", "GPS", imu["time_system"], ref["absolute_start_gps_seconds"], ref["absolute_end_gps_seconds"], imu["absolute_start_gps_seconds"], imu["absolute_end_gps_seconds"])
            )
        for rinex in [row for row in rinex_rows_internal if row["route"] == route]:
            unresolved = ""
            if rinex["_start_gps_seconds"] is None or rinex["_end_gps_seconds"] is None:
                unresolved = "RINEX header does not provide a complete first/last observation range; no full-file epoch scan was performed."
            alignment_rows.append(
                alignment_row(
                    route,
                    "reference.csv",
                    rinex["filename"],
                    "GPS",
                    rinex["time_system"],
                    ref["absolute_start_gps_seconds"],
                    ref["absolute_end_gps_seconds"],
                    rinex["_start_gps_seconds"],
                    rinex["_end_gps_seconds"],
                    unresolved,
                )
            )
        alignment_rows.append(
            alignment_row(route, "reference.csv", "lidar.bag", "GPS", "unknown", ref["absolute_start_gps_seconds"], ref["absolute_end_gps_seconds"], None, None, "LiDAR bag intentionally not parsed; it is not required for Plan A.")
        )

    feasibility_rows = [
        {"assessment_item": "real receiver position trajectory", "status": "yes", "evidence": "reference.csv explicitly provides GPS time, LLA, and ECEF at 10 Hz", "plan_a_use": "receiver trajectory truth input", "claim_boundary": "Reference solution accuracy metadata is not bundled locally."},
        {"assessment_item": "real receiver velocity", "status": "yes_with_caution", "evidence": "Source velocity exists; ENU velocity is independently derived from ECEF/time", "plan_a_use": "derived ENU velocity with raw and smoothed variants retained", "claim_boundary": "Source X/Y/Z velocity frame is unknown_semantics."},
        {"assessment_item": "acceleration and turning", "status": "yes", "evidence": "Heading, position, source acceleration, and angular-rate columns plus trajectory-derived kinematics", "plan_a_use": "motion-type segmentation", "claim_boundary": "Heading sign convention requires caution."},
        {"assessment_item": "timestamp quality", "status": "yes", "evidence": "GPS Week/TOW and trajectory quality audit", "plan_a_use": "relative-time preservation", "claim_boundary": "No GPS-to-UTC conversion is assumed."},
        {"assessment_item": "ECEF conversion", "status": "yes", "evidence": "Source ECEF plus WGS-84 LLA-to-ECEF cross-check", "plan_a_use": "LEO line-of-sight geometry", "claim_boundary": "WGS-84 datum documentation should be cited before publication."},
        {"assessment_item": "alignment to existing LEO satellite geometry", "status": "controlled_mapping_required", "evidence": "UrbanNav relative timing can be mapped to a contiguous project LEO timeline", "plan_a_use": "explicit offset mapping without time-scale guessing", "claim_boundary": "This is not contemporaneous RF synchronization."},
        {"assessment_item": "segment diversity", "status": "yes", "evidence": "Trajectory-only catalog includes speed regimes, turns, acceleration/deceleration, and stop-and-go candidates", "plan_a_use": "external motion protocol", "claim_boundary": "Segments are not selected from solver outcomes."},
        {"assessment_item": "ground-truth accuracy evidence", "status": "partial", "evidence": "A high-rate reference solution is present but local accuracy/covariance documentation is absent", "plan_a_use": "trajectory driver and evaluation reference with caution", "claim_boundary": "Do not overstate absolute truth accuracy."},
        {"assessment_item": "native LEO Doppler availability", "status": "no", "evidence": "Dataset contains GNSS RINEX, not Iridium/LEO observations", "plan_a_use": "none; generate controlled LEO Doppler", "claim_boundary": "Never describe UrbanNav GNSS Doppler as LEO Doppler."},
        {"assessment_item": "ability to claim real LEO field validation", "status": "no", "evidence": "No native LEO receiver observations", "plan_a_use": "not allowed", "claim_boundary": "Prohibited claim: real dynamic LEO Doppler field validation."},
        {"assessment_item": "ability to claim real-trajectory-driven controlled validation", "status": "yes", "evidence": "Independent real vehicle trajectory plus controlled project LEO geometry and perturbations", "plan_a_use": "primary Plan A wording", "claim_boundary": "Must retain controlled-simulation qualifier."},
    ]

    # Mark files that were deliberately registered but not semantically parsed.
    for path in files:
        if str(path) not in parse_status:
            parse_status[str(path)] = (True, "Registered and hashed; no content parser required for Plan A")
    for row in inventory_rows:
        path = DATA_ROOT / row["relative_path"]
        status, error = parse_status.get(str(path), (False, "No parser status"))
        row["parsed_successfully"] = status
        row["parse_error"] = error

    quality_columns = [
        "route", "sample_count", "start_time", "end_time", "start_gps_week", "start_gps_tow_s", "end_gps_week", "end_gps_tow_s", "duration_s",
        "median_sampling_interval_s", "p5_sampling_interval_s", "p95_sampling_interval_s", "duplicate_timestamp_count", "out_of_order_count",
        "gap_threshold_s", "gap_count", "maximum_gap_s", "missing_position_count", "non_finite_count", "position_jump_median_m", "position_jump_p95_m",
        "position_jump_p99_m", "position_jump_max_m", "suspicious_reference_jump_count", "approximate_route_length_m", "altitude_min_m", "altitude_max_m",
        "altitude_range_m", "source_velocity_present", "source_velocity_frame", "source_speed_median_mps", "source_speed_max_mps", "derived_speed_median_mps",
        "derived_speed_max_mps", "source_vs_derived_speed_rmse_mps", "lla_to_source_ecef_error_median_m", "lla_to_source_ecef_error_max_m",
        "velocity_derivation_method", "smoothing_method", "smoothing_window_samples", "smoothing_used_only_for_selected_analysis_velocity",
    ]
    write_csv(OUT / "PAPER_DATA02A_FILE_INVENTORY.csv", inventory_rows)
    write_csv(OUT / "PAPER_DATA02A_REFERENCE_SCHEMA_AUDIT.csv", schema_rows)
    write_csv(OUT / "PAPER_DATA02A_TRAJECTORY_QUALITY_SUMMARY.csv", list(qualities.values()), quality_columns)
    write_csv(OUT / "PAPER_DATA02A_MOTION_SEGMENT_CATALOG.csv", segment_rows)
    rinex_public = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rinex_rows_internal]
    write_csv(OUT / "PAPER_DATA02A_RINEX_HEADER_AUDIT.csv", rinex_public)
    write_csv(OUT / "PAPER_DATA02A_IMU_SCHEMA_AND_TIME_AUDIT.csv", imu_rows)
    write_csv(OUT / "PAPER_DATA02A_TIME_ALIGNMENT_AUDIT.csv", alignment_rows)
    write_csv(OUT / "PAPER_DATA02A_PLAN_A_FEASIBILITY_MATRIX.csv", feasibility_rows)

    decision = "ready_for_paper_exp02a"
    (OUT / "PAPER_DATA02A_EXP02A_DESIGN_DRAFT.md").write_text(build_design_draft(decision, qualities, segment_rows), encoding="utf-8")

    raw_after = raw_metadata_snapshot()
    protected_after = {str(path): snapshot_tree(path) for path in PROTECTED_PATHS}
    raw_unchanged = raw_before == raw_after
    protected_unchanged = protected_before == protected_after
    elapsed = time.perf_counter() - started

    report = build_report(
        decision, inventory_rows, qualities, reference_details, imu_rows, rinex_public, alignment_rows,
        segment_rows, expected_missing, protected_unchanged, raw_unchanged, elapsed,
    )
    (OUT / "PAPER_DATA02A_URBANNAV_TK_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    metrics = {
        "task": "PAPER-DATA02A",
        "decision": decision,
        "root": str(ROOT),
        "conda_prefix": str(ROOT / ".conda" / "leo-solver"),
        "python_executable": sys.executable,
        "data_root": str(DATA_ROOT),
        "output_root": str(OUT),
        "wgs84_constants": {"a_m": WGS84_A_M, "f": WGS84_F, "e2": WGS84_E2},
        "configuration": CONFIG,
        "reference_details": reference_details,
        "trajectory_quality": qualities,
        "recommended_segments": [row for row in segment_rows if row["recommended_for_exp02a"]],
        "rinex_doppler_summary": [
            {"route": row["route"], "filename": row["filename"], "doppler_fields_present": row["doppler_fields_present"], "doppler_observation_types": row["doppler_observation_types"]}
            for row in rinex_public
        ],
        "expected_missing_files": expected_missing,
        "raw_files_unchanged": raw_unchanged,
        "protected_paths_before": protected_before,
        "protected_paths_after": protected_after,
        "protected_paths_unchanged": protected_unchanged,
        "full_raw_qatar_read": False,
        "ma_bgtr_experiment_run": False,
        "time_conversion_policy": "GPS timescale comparison only; no guessed UTC conversion",
        "source_velocity_policy": "retained verbatim; frame unknown_semantics; derived ENU used for analysis",
        "elapsed_seconds": elapsed,
    }
    (OUT / "PAPER_DATA02A_METRICS.json").write_text(json.dumps(clean_scalar(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    required_outputs = [
        OUT / "PAPER_DATA02A_FILE_INVENTORY.csv",
        OUT / "PAPER_DATA02A_REFERENCE_SCHEMA_AUDIT.csv",
        OUT / "PAPER_DATA02A_TRAJECTORY_QUALITY_SUMMARY.csv",
        OUT / "PAPER_DATA02A_MOTION_SEGMENT_CATALOG.csv",
        OUT / "PAPER_DATA02A_RINEX_HEADER_AUDIT.csv",
        OUT / "PAPER_DATA02A_IMU_SCHEMA_AND_TIME_AUDIT.csv",
        OUT / "PAPER_DATA02A_TIME_ALIGNMENT_AUDIT.csv",
        OUT / "PAPER_DATA02A_PLAN_A_FEASIBILITY_MATRIX.csv",
        OUT / "PAPER_DATA02A_EXP02A_DESIGN_DRAFT.md",
        OUT / "PAPER_DATA02A_URBANNAV_TK_AUDIT_REPORT.md",
        PROCESSED / "UrbanNav_TK_Odaiba_reference_standardized.csv",
        PROCESSED / "UrbanNav_TK_Shinjuku_reference_standardized.csv",
        SCRIPT_PATH,
        OUT / "PAPER_DATA02A_METRICS.json",
    ]
    for path in required_outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Required output missing or empty: {path}")
        if OUT != path and OUT not in path.parents:
            raise RuntimeError(f"Output escaped audit directory: {path}")
    if not raw_unchanged:
        raise RuntimeError("Raw UrbanNav metadata changed during audit")
    if not protected_unchanged:
        raise RuntimeError("Protected project metadata changed during audit")

    manifest_path = OUT / "MANIFEST.txt"
    manifest_lines = ["relative_path,size_bytes,sha256", "# MANIFEST.txt is excluded from its own hash by definition."]
    for path in sorted(required_outputs, key=lambda item: str(item.relative_to(OUT)).lower()):
        rel = path.relative_to(OUT).as_posix()
        manifest_lines.append(f"{rel},{path.stat().st_size},{sha256_file(path)}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    # Re-read and verify every manifest payload before packaging.
    for line in manifest_path.read_text(encoding="utf-8").splitlines()[2:]:
        rel, size_text, digest = line.rsplit(",", 2)
        path = OUT / Path(rel)
        if path.stat().st_size != int(size_text) or sha256_file(path) != digest:
            raise RuntimeError(f"Manifest verification failed: {rel}")

    packet_path = OUT / "PAPER_DATA02A_REVIEW_PACKET.zip"
    if packet_path.exists():
        packet_path.unlink()
    package_files = required_outputs + [manifest_path]
    with zipfile.ZipFile(packet_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_files:
            archive.write(path, arcname=path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(packet_path, "r") as archive:
        names = set(archive.namelist())
        expected_names = {path.relative_to(OUT).as_posix() for path in package_files}
        if names != expected_names:
            raise RuntimeError("Review packet file set mismatch")
        for line in manifest_path.read_text(encoding="utf-8").splitlines()[2:]:
            rel, size_text, digest = line.rsplit(",", 2)
            payload = archive.read(rel)
            if len(payload) != int(size_text) or hashlib.sha256(payload).hexdigest() != digest:
                raise RuntimeError(f"ZIP manifest verification failed: {rel}")
    print(json.dumps({"decision": decision, "elapsed_seconds": elapsed, "packet": str(packet_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
