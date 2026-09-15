"""Temporal validation splits for MA-BGTR-v2."""

from __future__ import annotations

from typing import Any

import numpy as np


OBS_ARRAY_KEYS = [
    "sat_pos_m",
    "sat_vel_mps",
    "sat_pos_truth_m",
    "sat_vel_truth_mps",
    "meas_mps",
    "measured_hz",
    "predicted_hz",
    "time_s",
    "satellite_number",
    "row_index",
    "truth_positions_m",
    "truth_bias_mps",
]


def temporal_train_validation_masks(time_s: np.ndarray, train_fraction: float = 0.70) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float)
    order = np.argsort(time_s)
    split = int(np.floor(len(time_s) * float(train_fraction)))
    split = min(max(split, 1), len(time_s) - 1)
    train_mask = np.zeros(len(time_s), dtype=bool)
    val_mask = np.zeros(len(time_s), dtype=bool)
    train_mask[order[:split]] = True
    val_mask[order[split:]] = True
    return train_mask, val_mask


def subset_observations(obs: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    out: dict[str, Any] = {}
    for key, value in obs.items():
        if key in OBS_ARRAY_KEYS and isinstance(value, np.ndarray) and len(value) == len(mask):
            out[key] = value[mask]
        else:
            out[key] = value.copy() if isinstance(value, np.ndarray) else value
    return out


def blocked_train_validation(obs: dict[str, Any], train_fraction: float = 0.70) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_mask, val_mask = temporal_train_validation_masks(obs["time_s"], train_fraction)
    return subset_observations(obs, train_mask), subset_observations(obs, val_mask), {
        "split_policy": "temporal_first_70_percent_train_last_30_percent_validation",
        "train_count": int(np.sum(train_mask)),
        "validation_count": int(np.sum(val_mask)),
        "train_time_min": float(np.min(np.asarray(obs["time_s"])[train_mask])),
        "train_time_max": float(np.max(np.asarray(obs["time_s"])[train_mask])),
        "validation_time_min": float(np.min(np.asarray(obs["time_s"])[val_mask])),
        "validation_time_max": float(np.max(np.asarray(obs["time_s"])[val_mask])),
    }
