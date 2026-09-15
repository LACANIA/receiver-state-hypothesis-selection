"""Stress scenario construction for STEP03 baseline experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .coordinates import enu_axes


def copy_observations(obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obs.items():
        out[key] = value.copy() if isinstance(value, np.ndarray) else value
    return out


def named_directions(lat_deg: float, lon_deg: float, seed: int = 20260628) -> list[tuple[str, np.ndarray]]:
    east, north, up = enu_axes(lat_deg, lon_deg)
    rng = np.random.default_rng(seed)
    random_dirs = rng.normal(size=(10, 3))
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1)[:, None]
    directions = [
        ("east", east),
        ("north", north),
        ("up", up),
        ("-east", -east),
        ("-north", -north),
        ("-up", -up),
    ]
    directions.extend((f"random_{i:02d}", random_dirs[i]) for i in range(10))
    return directions


def init_position(p_gt_ecef_m: np.ndarray, direction: np.ndarray, distance_m: float) -> np.ndarray:
    return np.asarray(p_gt_ecef_m, dtype=float).reshape(3) + float(distance_m) * np.asarray(direction, dtype=float)


def add_common_bias(obs: dict[str, Any], bias_mps: float) -> dict[str, Any]:
    out = copy_observations(obs)
    out["meas_mps"] = out["meas_mps"] + float(bias_mps)
    return out


def add_outliers(
    obs: dict[str, Any],
    outlier_fraction: float,
    outlier_offset_mps: float,
    seed: int = 20260628,
) -> dict[str, Any]:
    out = copy_observations(obs)
    fraction = float(outlier_fraction)
    n_obs = len(out["meas_mps"])
    n_outliers = int(round(fraction * n_obs))
    out["outlier_indices"] = np.array([], dtype=int)
    if n_outliers <= 0:
        return out
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_obs, size=n_outliers, replace=False))
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_outliers)
    out["meas_mps"] = out["meas_mps"].copy()
    out["meas_mps"][indices] += signs * float(outlier_offset_mps)
    out["outlier_indices"] = indices
    return out


def add_ephemeris_noise(
    obs: dict[str, Any],
    sat_pos_sigma_m: float,
    sat_vel_sigma_mps: float,
    seed: int = 20260628,
) -> dict[str, Any]:
    out = copy_observations(obs)
    rng = np.random.default_rng(seed)
    pos_sigma = float(sat_pos_sigma_m)
    vel_sigma = float(sat_vel_sigma_mps)
    if pos_sigma > 0.0:
        out["sat_pos_m"] = out["sat_pos_m"] + rng.normal(scale=pos_sigma, size=out["sat_pos_m"].shape)
    if vel_sigma > 0.0:
        out["sat_vel_mps"] = out["sat_vel_mps"] + rng.normal(scale=vel_sigma, size=out["sat_vel_mps"].shape)
    return out


def subset_observations(obs: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    out = copy_observations(obs)
    for key in [
        "sat_pos_m",
        "sat_vel_mps",
        "measured_hz",
        "predicted_hz",
        "meas_mps",
        "time_s",
        "satellite_number",
    ]:
        out[key] = out[key][mask]
    return out


def geometry_subsets(obs: dict[str, Any], seed: int = 20260628) -> dict[str, dict[str, Any]]:
    sats = np.asarray(obs["satellite_number"])
    unique, counts = np.unique(sats, return_counts=True)
    count_map = dict(zip(unique, counts, strict=False))
    top_order = unique[np.argsort(-counts)]
    rng = np.random.default_rng(seed)
    random_half = rng.random(len(sats)) < 0.5
    if not np.any(random_half):
        random_half[0] = True

    masks = {
        "all": np.ones(len(sats), dtype=bool),
        "frequent_only": np.array([count_map[s] >= 10 for s in sats], dtype=bool),
        "top3_by_count": np.isin(sats, top_order[:3]),
        "remove_top1": ~np.isin(sats, top_order[:1]),
        "singletons_removed": np.array([count_map[s] > 1 for s in sats], dtype=bool),
        "sparse_random_half": random_half,
    }
    return {label: subset_observations(obs, mask) for label, mask in masks.items()}


def observation_counts(obs: dict[str, Any]) -> tuple[int, int]:
    return int(len(obs["meas_mps"])), int(np.unique(obs["satellite_number"]).size)
