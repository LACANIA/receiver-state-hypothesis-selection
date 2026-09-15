"""Data loading and normalization for Iridium Doppler baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .coordinates import llh_to_ecef

C_MPS = 299_792_458.0
IRIDIUM_FREQ_HZ = 1_626_270_833.0

REQUIRED_COLUMNS = [
    "Time/s",
    "Satellite number",
    "Measured Doppler/Hz",
    "Predicted Doppler/Hz",
    "Sat_position_x/m",
    "Sat_position_y/m",
    "Sat_position_z/m",
    "Sat_velocity_x/(m/s)",
    "Sat_velocity_y/(m/s)",
    "Sat_velocity_z/(m/s)",
    "User_latitude",
    "User_longitude",
    "User_height",
]


def candidate_csv_paths(root: Path) -> list[Path]:
    return [
        root / "Iridium_Doppler_measurements.csv",
        root / "Certifiable-Doppler-positioning-main" / "matlab" / "data" / "iridium" / "Iridium_Doppler_measurements.csv",
        root / "Certifiable-Doppler-positioning-main" / "matlab" / "data" / "iridium" / "Iridium.csv",
        root / "patent_standalone" / "data" / "hk" / "Iridium_Doppler_measurements.csv",
    ]


def find_iridium_csv(root: Path) -> Path:
    for path in candidate_csv_paths(root):
        if path.is_file():
            return path
    raise FileNotFoundError("No Iridium Doppler CSV found in configured candidate paths.")


def load_iridium_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; actual columns: {list(df.columns)}")
    return df


def normalize_observations(df: pd.DataFrame) -> dict[str, Any]:
    """Return arrays for the unified range-rate convention.

    The STEP02 convention is m/s range-rate Doppler:
        meas_mps = -measured_hz * c / f
    """
    truth_rows = df[["User_latitude", "User_longitude", "User_height"]].dropna()
    if truth_rows.empty:
        raise ValueError("Ground truth LLH fields are missing.")
    lat, lon, h = [float(v) for v in truth_rows.iloc[0].to_numpy()]
    p_gt = llh_to_ecef(lat, lon, h)

    sat_pos = df[["Sat_position_x/m", "Sat_position_y/m", "Sat_position_z/m"]].to_numpy(dtype=float)
    sat_vel = df[["Sat_velocity_x/(m/s)", "Sat_velocity_y/(m/s)", "Sat_velocity_z/(m/s)"]].to_numpy(dtype=float)
    measured_hz = df["Measured Doppler/Hz"].to_numpy(dtype=float)
    predicted_hz = df["Predicted Doppler/Hz"].to_numpy(dtype=float)
    meas_mps = -measured_hz * C_MPS / IRIDIUM_FREQ_HZ

    return {
        "lat_deg": lat,
        "lon_deg": lon,
        "height_m": h,
        "p_gt_ecef_m": p_gt,
        "sat_pos_m": sat_pos,
        "sat_vel_mps": sat_vel,
        "measured_hz": measured_hz,
        "predicted_hz": predicted_hz,
        "meas_mps": meas_mps,
        "time_s": df["Time/s"].to_numpy(dtype=float),
        "satellite_number": df["Satellite number"].to_numpy(),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median_abs": float(np.median(np.abs(arr))),
    }


def summarize_dataframe(df: pd.DataFrame, obs: dict[str, Any]) -> dict[str, Any]:
    counts = df["Satellite number"].value_counts().sort_index()
    residual_hz = df["Measured Doppler/Hz"].to_numpy(dtype=float) - df["Predicted Doppler/Hz"].to_numpy(dtype=float)
    residual_mps = residual_hz * C_MPS / IRIDIUM_FREQ_HZ
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "time_min_s": float(np.min(obs["time_s"])),
        "time_max_s": float(np.max(obs["time_s"])),
        "time_span_s": float(np.max(obs["time_s"]) - np.min(obs["time_s"])),
        "satellite_count": int(counts.size),
        "satellite_observation_counts": {str(k): int(v) for k, v in counts.items()},
        "measured_minus_predicted_hz": _stats(residual_hz),
        "measured_minus_predicted_mps": _stats(residual_mps),
    }
