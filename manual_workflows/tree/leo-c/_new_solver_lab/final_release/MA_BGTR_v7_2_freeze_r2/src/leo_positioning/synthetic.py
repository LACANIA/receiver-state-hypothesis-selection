"""Synthetic dynamic Doppler data generation for STEP04."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .coordinates import enu_axes
from .stress import observation_counts
from .trajectory_models import FULL_CTD_CONFIG, pack_state, predict_ctd


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    direction_label: str
    speed_mps: float
    b0_mps: float
    bdot_mps2: float
    noise_sigma_mps: float
    outlier_fraction: float = 0.0
    outlier_offset_mps: float = 0.0
    sat_pos_sigma_m: float = 0.0
    sat_vel_sigma_mps: float = 0.0
    subset_label: str = "all"
    keep_fraction: float = 1.0
    seed: int = 20260628


def scenario_definitions() -> list[SyntheticScenario]:
    return [
        SyntheticScenario("S0_static_clean", "static", 0.0, 0.0, 0.0, 0.2),
        SyntheticScenario("S1_dynamic_clean", "east", 20.0, 0.5, 0.02, 0.2),
        SyntheticScenario("S2_dynamic_bias_strong", "east+north", 50.0, 2.0, 0.05, 0.3),
        SyntheticScenario("S3_dynamic_outlier", "east", 20.0, 0.5, 0.02, 0.2, 0.10, 5.0),
        SyntheticScenario("S4_dynamic_ephemeris_error", "east", 20.0, 0.5, 0.02, 0.2, sat_pos_sigma_m=100.0, sat_vel_sigma_mps=0.1),
        SyntheticScenario(
            "S5_dynamic_hard",
            "east+north",
            80.0,
            3.0,
            0.08,
            0.5,
            0.15,
            10.0,
            sat_pos_sigma_m=500.0,
            sat_vel_sigma_mps=0.2,
        ),
        SyntheticScenario("S6_slow_west_bias", "west", 5.0, -1.0, 0.01, 0.2),
        SyntheticScenario("S7_fast_north_no_bias", "north", 120.0, 0.0, 0.0, 0.3),
        SyntheticScenario("S8_dynamic_dropout_like", "east", 30.0, 1.0, 0.03, 0.3, subset_label="frequent_only", keep_fraction=0.70),
        SyntheticScenario("S9_dynamic_geometry_hard", "east+north", 30.0, 1.0, 0.03, 0.3, subset_label="top3_by_count"),
    ]


def direction_vector(label: str, lat_deg: float, lon_deg: float) -> np.ndarray:
    east, north, _up = enu_axes(lat_deg, lon_deg)
    if label == "static":
        return np.zeros(3, dtype=float)
    if label == "east":
        return east
    if label == "north":
        return north
    if label == "west":
        return -east
    if label == "east+north":
        vec = east + north
        return vec / np.linalg.norm(vec)
    raise ValueError(f"Unknown direction label: {label}")


def scenario_velocity(scenario: SyntheticScenario, lat_deg: float, lon_deg: float) -> np.ndarray:
    direction = direction_vector(scenario.direction_label, lat_deg, lon_deg)
    if np.linalg.norm(direction) == 0.0:
        return np.zeros(3, dtype=float)
    return float(scenario.speed_mps) * direction


def _subset_mask(scenario: SyntheticScenario, base_obs: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    sats = np.asarray(base_obs["satellite_number"])
    mask = np.ones(len(sats), dtype=bool)
    if scenario.subset_label == "frequent_only":
        unique, counts = np.unique(sats, return_counts=True)
        count_map = dict(zip(unique, counts, strict=False))
        mask &= np.array([count_map[s] >= 10 for s in sats], dtype=bool)
    elif scenario.subset_label == "top3_by_count":
        unique, counts = np.unique(sats, return_counts=True)
        top3 = unique[np.argsort(-counts)[:3]]
        mask &= np.isin(sats, top3)
    elif scenario.subset_label != "all":
        raise ValueError(f"Unknown subset label: {scenario.subset_label}")

    if scenario.keep_fraction < 1.0:
        keep = rng.random(len(sats)) < float(scenario.keep_fraction)
        mask &= keep
    if not np.any(mask):
        raise ValueError(f"Scenario {scenario.name} selected no observations.")
    return mask


def generate_scenario(
    scenario: SyntheticScenario,
    base_obs: dict[str, Any],
    base_df: pd.DataFrame,
    scenario_index: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(scenario.seed) + 1009 * int(scenario_index))
    n = len(base_obs["meas_mps"])
    mask = _subset_mask(scenario, base_obs, rng)
    p0_true = np.asarray(base_obs["p_gt_ecef_m"], dtype=float)
    v_true = scenario_velocity(scenario, base_obs["lat_deg"], base_obs["lon_deg"])
    time_s = np.asarray(base_obs["time_s"], dtype=float)
    t0_s = float(np.min(time_s))
    tau = time_s - t0_s
    theta_true = pack_state(p0_true, v_true, scenario.b0_mps, scenario.bdot_mps2, FULL_CTD_CONFIG)

    clean_pred = predict_ctd(
        theta_true,
        time_s,
        base_obs["sat_pos_m"],
        base_obs["sat_vel_mps"],
        t0_s=t0_s,
        config=FULL_CTD_CONFIG,
    )
    noise = rng.normal(scale=float(scenario.noise_sigma_mps), size=n)
    outlier = np.zeros(n, dtype=float)
    used_indices = np.flatnonzero(mask)
    n_outliers = int(round(float(scenario.outlier_fraction) * len(used_indices)))
    if n_outliers > 0:
        chosen = rng.choice(used_indices, size=n_outliers, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_outliers)
        outlier[chosen] = signs * float(scenario.outlier_offset_mps)
    measured = clean_pred + noise + outlier

    sat_pos_solver = np.asarray(base_obs["sat_pos_m"], dtype=float).copy()
    sat_vel_solver = np.asarray(base_obs["sat_vel_mps"], dtype=float).copy()
    if scenario.sat_pos_sigma_m > 0.0:
        sat_pos_solver += rng.normal(scale=float(scenario.sat_pos_sigma_m), size=sat_pos_solver.shape)
    if scenario.sat_vel_sigma_mps > 0.0:
        sat_vel_solver += rng.normal(scale=float(scenario.sat_vel_sigma_mps), size=sat_vel_solver.shape)

    p_true = p0_true[None, :] + tau[:, None] * v_true[None, :]
    b_true = float(scenario.b0_mps) + float(scenario.bdot_mps2) * tau
    truth_df = pd.DataFrame(
        {
            "scenario": scenario.name,
            "row_index": np.arange(n, dtype=int),
            "time_s": time_s,
            "tau_s": tau,
            "satellite_number": base_obs["satellite_number"],
            "used_in_scenario": mask,
            "true_px_m": p_true[:, 0],
            "true_py_m": p_true[:, 1],
            "true_pz_m": p_true[:, 2],
            "true_vx_mps": v_true[0],
            "true_vy_mps": v_true[1],
            "true_vz_mps": v_true[2],
            "true_b_mps": b_true,
            "true_bdot_mps2": float(scenario.bdot_mps2),
            "clean_pred_mps": clean_pred,
            "measured_synthetic_mps": measured,
            "noise_mps": noise,
            "outlier_mps": outlier,
            "sat_pos_sigma_m": float(scenario.sat_pos_sigma_m),
            "sat_vel_sigma_mps": float(scenario.sat_vel_sigma_mps),
        }
    )

    obs = {
        "lat_deg": base_obs["lat_deg"],
        "lon_deg": base_obs["lon_deg"],
        "height_m": base_obs["height_m"],
        "p_gt_ecef_m": p0_true,
        "sat_pos_m": sat_pos_solver[mask],
        "sat_vel_mps": sat_vel_solver[mask],
        "sat_pos_truth_m": np.asarray(base_obs["sat_pos_m"], dtype=float)[mask],
        "sat_vel_truth_mps": np.asarray(base_obs["sat_vel_mps"], dtype=float)[mask],
        "meas_mps": measured[mask],
        "time_s": time_s[mask],
        "satellite_number": np.asarray(base_obs["satellite_number"])[mask],
        "row_index": np.arange(n, dtype=int)[mask],
        "t0_s": t0_s,
        "p0_true_m": p0_true,
        "v_true_mps": v_true,
        "b0_true_mps": float(scenario.b0_mps),
        "bdot_true_mps2": float(scenario.bdot_mps2),
        "scenario": scenario.name,
        "scenario_config": asdict(scenario),
        "truth_positions_m": p_true[mask],
        "truth_bias_mps": b_true[mask],
        "base_row_count": n,
    }
    obs_count, sat_count = observation_counts(obs)
    metadata = {
        "scenario": asdict(scenario),
        "observation_count": obs_count,
        "satellite_count": sat_count,
        "used_fraction": float(np.mean(mask)),
    }
    return obs, truth_df, metadata


def generate_all_scenarios(base_obs: dict[str, Any], base_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    scenario_obs: dict[str, dict[str, Any]] = {}
    truth_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenario_definitions()):
        obs, truth_df, meta = generate_scenario(scenario, base_obs, base_df, idx)
        scenario_obs[scenario.name] = obs
        truth_frames.append(truth_df)
        metadata.append(meta)
    return scenario_obs, pd.concat(truth_frames, ignore_index=True), metadata
