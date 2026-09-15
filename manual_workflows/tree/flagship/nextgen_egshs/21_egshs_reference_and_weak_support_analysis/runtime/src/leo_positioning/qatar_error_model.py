"""Qatar/Mendeley physical-layer error model for synthetic Doppler tests.

The Qatar audit data is B-class physical-layer evidence. Its frequency-like
columns have unknown units, so this module only uses distribution shape and
maps robust spreads into configured m/s perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

from .synthetic import direction_vector
from .trajectory_models import FULL_CTD_CONFIG, pack_state, predict_ctd


@dataclass(frozen=True)
class QatarColumnStats:
    median: float
    mad: float
    p95: float
    p99: float
    outlier_ratio_3mad: float
    drift_slope_per_s: float


@dataclass(frozen=True)
class QatarErrorModel:
    source_file: str
    col4: QatarColumnStats
    col5: QatarColumnStats
    confidence: dict[str, float]
    iq_phase: dict[str, float]
    warnings: list[str]


@dataclass(frozen=True)
class QatarScenario:
    name: str
    recipe_name: str
    direction_label: str
    speed_mps: float
    b0_mps: float
    bdot_mps2: float
    base_noise_sigma_mps: float
    cfo_column: str = "col4"
    cfo_scale_to_mps: float = 0.2
    cfo_enabled: bool = True
    heavy_tail: bool = False
    outlier_ratio: float | None = None
    outlier_scale_multiplier: float = 1.0
    confidence_dropout: bool = False
    dropout_ratio: float | None = None
    low_conf_noise_multiplier: float = 3.0
    burst_outlier: bool = False
    subset_label: str = "all"
    sat_pos_sigma_m: float = 0.0
    sat_vel_sigma_mps: float = 0.0
    seed: int = 20260628


def _column_stats(raw: dict[str, Any]) -> QatarColumnStats:
    return QatarColumnStats(
        median=float(raw.get("median", np.nan)),
        mad=float(raw.get("mad", np.nan)),
        p95=float(raw.get("p95", np.nan)),
        p99=float(raw.get("p99", np.nan)),
        outlier_ratio_3mad=float(raw.get("outlier_ratio_3mad", 0.0)),
        drift_slope_per_s=float(raw.get("drift_slope_per_s", 0.0)),
    )


def load_qatar_error_model(config_path: str | Path) -> QatarErrorModel:
    """Load DATA02's compact Qatar error-model JSON."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    params = raw.get("parameters", {})
    return QatarErrorModel(
        source_file=str(raw.get("source_file", "")),
        col4=_column_stats(params.get("col4", {})),
        col5=_column_stats(params.get("col5", {})),
        confidence={k: float(v) for k, v in params.get("confidence", {}).items() if isinstance(v, (int, float))},
        iq_phase={k: float(v) for k, v in params.get("iq_phase", {}).items() if isinstance(v, (int, float, bool))},
        warnings=[str(v) for v in raw.get("warnings", [])],
    )


def qatar_scenario_definitions(model: QatarErrorModel) -> list[QatarScenario]:
    """Return the DATA03 Q0-Q4 Qatar-calibrated scenarios."""
    col4_ratio = float(model.col4.outlier_ratio_3mad)
    col5_ratio = float(model.col5.outlier_ratio_3mad)
    conf_low = float(model.confidence.get("ratio_below_90", 0.095))
    return [
        QatarScenario(
            "Q0_qatar_cfo_mild",
            "qatar_col4_mild_mad_scaled",
            "east",
            20.0,
            0.5,
            0.02,
            0.2,
            cfo_column="col4",
            cfo_scale_to_mps=0.20,
            outlier_ratio=0.5 * col4_ratio,
            outlier_scale_multiplier=0.65,
        ),
        QatarScenario(
            "Q1_qatar_cfo_heavy_tail",
            "qatar_col4_col5_heavy_tail",
            "east+north",
            50.0,
            2.0,
            0.05,
            0.3,
            cfo_column="col5",
            cfo_scale_to_mps=0.45,
            heavy_tail=True,
            outlier_ratio=max(col4_ratio, col5_ratio),
            outlier_scale_multiplier=1.0,
        ),
        QatarScenario(
            "Q2_qatar_confidence_dropout",
            "qatar_confidence_dropout_below90",
            "east",
            30.0,
            1.0,
            0.03,
            0.3,
            cfo_enabled=False,
            confidence_dropout=True,
            dropout_ratio=0.5 * conf_low,
            low_conf_noise_multiplier=3.5,
        ),
        QatarScenario(
            "Q3_qatar_burst_outlier",
            "qatar_burst_outlier_p99_scaled",
            "east",
            20.0,
            0.5,
            0.02,
            0.2,
            cfo_column="col4",
            cfo_scale_to_mps=0.30,
            outlier_ratio=col4_ratio,
            outlier_scale_multiplier=1.25,
            burst_outlier=True,
        ),
        QatarScenario(
            "Q4_qatar_hard_geometry_noise",
            "qatar_hard_geometry_heavy_tail_dropout",
            "east+north",
            80.0,
            3.0,
            0.08,
            0.5,
            cfo_column="col5",
            cfo_scale_to_mps=0.55,
            heavy_tail=True,
            outlier_ratio=max(col4_ratio, col5_ratio),
            outlier_scale_multiplier=1.35,
            confidence_dropout=True,
            dropout_ratio=0.5 * conf_low,
            low_conf_noise_multiplier=4.0,
            subset_label="top3_by_count",
            sat_pos_sigma_m=500.0,
            sat_vel_sigma_mps=0.2,
        ),
    ]


def _stats_for_column(model: QatarErrorModel, column: str) -> QatarColumnStats:
    if column == "col4":
        return model.col4
    if column == "col5":
        return model.col5
    raise ValueError(f"Unknown Qatar frequency-like column: {column}")


def make_qatar_noise_sampler(seed: int = 20260628) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sample_cfo_like_bias(
    rng: np.random.Generator,
    n: int,
    model: QatarErrorModel,
    column: str,
    scale_to_mps: float,
    heavy_tail: bool = False,
) -> np.ndarray:
    """Sample zero-centered CFO-like perturbations in synthetic m/s units."""
    stats = _stats_for_column(model, column)
    scale = max(float(scale_to_mps), 1e-9)
    common = rng.normal(0.0, 0.20 * scale)
    if heavy_tail:
        # Student-t retains Qatar-like heavy tails without assuming units.
        perturb = rng.standard_t(df=3.0, size=int(n)) * scale / np.sqrt(3.0)
    else:
        perturb = rng.normal(0.0, scale, size=int(n))
    drift = stats.drift_slope_per_s
    if np.isfinite(drift) and abs(drift) > 0.0:
        # Keep drift shape only; scale by the robust spread to synthetic units.
        drift = np.clip(drift / max(stats.mad, 1e-9), -1e-3, 1e-3) * scale
    else:
        drift = 0.0
    return np.asarray(perturb + common, dtype=float), float(drift)


def qatar_outlier_amplitude_mps(
    model: QatarErrorModel,
    column: str,
    scale_to_mps: float,
    use_p99: bool = True,
    multiplier: float = 1.0,
) -> float:
    stats = _stats_for_column(model, column)
    high = stats.p99 if use_p99 else stats.p95
    robust_z = abs(float(high) - float(stats.median)) / max(float(stats.mad), 1e-9)
    return float(max(0.25, robust_z * float(scale_to_mps) * float(multiplier)))


def sample_heavy_tail_outliers(
    rng: np.random.Generator,
    n: int,
    outlier_ratio: float,
    amplitude_mps: float,
) -> np.ndarray:
    out = np.zeros(int(n), dtype=float)
    count = int(round(np.clip(float(outlier_ratio), 0.0, 1.0) * int(n)))
    if count <= 0:
        return out
    idx = rng.choice(np.arange(int(n)), size=count, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    magnitudes = rng.lognormal(mean=np.log(max(float(amplitude_mps), 1e-9)), sigma=0.25, size=count)
    out[idx] = signs * magnitudes
    return out


def sample_confidence_dropout(
    rng: np.random.Generator,
    n: int,
    model: QatarErrorModel,
    dropout_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_ratio = float(model.confidence.get("ratio_below_90", 0.095))
    drop_ratio = low_ratio if dropout_ratio is None else float(dropout_ratio)
    low_conf = rng.random(int(n)) < np.clip(low_ratio, 0.0, 1.0)
    conf = np.where(low_conf, rng.integers(65, 90, size=int(n)), rng.integers(95, 102, size=int(n))).astype(float)
    dropout = low_conf & (rng.random(int(n)) < np.clip(drop_ratio / max(low_ratio, 1e-9), 0.0, 1.0))
    weights = np.clip((conf - 70.0) / 30.0, 0.05, 1.0)
    return conf, dropout.astype(bool), weights.astype(float)


def apply_burst_outliers(
    rng: np.random.Generator,
    time_s: np.ndarray,
    target_fraction: float,
    amplitude_mps: float,
) -> np.ndarray:
    n = len(time_s)
    out = np.zeros(n, dtype=float)
    target = int(round(np.clip(float(target_fraction), 0.0, 1.0) * n))
    if target <= 0:
        return out
    order = np.argsort(np.asarray(time_s, dtype=float))
    block_len = max(2, min(max(target, 2), max(2, n // 6)))
    remaining = target
    used = set()
    while remaining > 0 and len(used) < n:
        start = int(rng.integers(0, max(n - block_len + 1, 1)))
        length = min(block_len, remaining, n - start)
        chosen = order[start : start + length]
        sign = float(rng.choice(np.array([-1.0, 1.0])))
        ramp = rng.normal(1.0, 0.15, size=len(chosen))
        for idx, value in zip(chosen, sign * float(amplitude_mps) * ramp, strict=False):
            if int(idx) not in used:
                out[int(idx)] = float(value)
                used.add(int(idx))
                remaining -= 1
                if remaining <= 0:
                    break
        block_len = max(2, block_len // 2)
    return out


def _subset_mask(scenario: QatarScenario, base_obs: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    sats = np.asarray(base_obs["satellite_number"])
    mask = np.ones(len(sats), dtype=bool)
    if scenario.subset_label == "top3_by_count":
        unique, counts = np.unique(sats, return_counts=True)
        top = unique[np.argsort(-counts)[:3]]
        mask &= np.isin(sats, top)
    elif scenario.subset_label != "all":
        raise ValueError(f"Unknown subset label: {scenario.subset_label}")
    if not np.any(mask):
        raise ValueError(f"Scenario {scenario.name} selected no observations.")
    return mask


def generate_qatar_calibrated_scenario(
    scenario: QatarScenario,
    model: QatarErrorModel,
    base_obs: dict[str, Any],
    scenario_index: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    rng = make_qatar_noise_sampler(int(scenario.seed) + 7919 * int(scenario_index))
    n = len(base_obs["time_s"])
    time_s = np.asarray(base_obs["time_s"], dtype=float)
    t0_s = float(np.min(time_s))
    tau = time_s - t0_s
    p0_true = np.asarray(base_obs["p_gt_ecef_m"], dtype=float)
    direction = direction_vector(scenario.direction_label, float(base_obs["lat_deg"]), float(base_obs["lon_deg"]))
    v_true = float(scenario.speed_mps) * direction
    theta_true = pack_state(p0_true, v_true, scenario.b0_mps, scenario.bdot_mps2, FULL_CTD_CONFIG)
    clean_pred = predict_ctd(theta_true, time_s, base_obs["sat_pos_m"], base_obs["sat_vel_mps"], t0_s, FULL_CTD_CONFIG)

    base_noise = rng.normal(0.0, float(scenario.base_noise_sigma_mps), size=n)
    qatar_cfo = np.zeros(n, dtype=float)
    drift_shape = 0.0
    if scenario.cfo_enabled:
        qatar_cfo, drift_shape = sample_cfo_like_bias(
            rng,
            n,
            model,
            scenario.cfo_column,
            scenario.cfo_scale_to_mps,
            heavy_tail=bool(scenario.heavy_tail),
        )
        qatar_cfo = qatar_cfo + drift_shape * tau

    outlier_amp = qatar_outlier_amplitude_mps(
        model,
        scenario.cfo_column,
        scenario.cfo_scale_to_mps,
        use_p99=True,
        multiplier=scenario.outlier_scale_multiplier,
    )
    ratio = float(scenario.outlier_ratio or 0.0)
    if scenario.burst_outlier:
        qatar_outlier = apply_burst_outliers(rng, time_s, ratio, outlier_amp)
    else:
        qatar_outlier = sample_heavy_tail_outliers(rng, n, ratio, outlier_amp)

    if scenario.confidence_dropout:
        conf, dropout_flag, conf_weight = sample_confidence_dropout(rng, n, model, scenario.dropout_ratio)
        low_conf = conf < 90.0
        base_noise = base_noise + low_conf.astype(float) * rng.normal(
            0.0, scenario.base_noise_sigma_mps * scenario.low_conf_noise_multiplier, size=n
        )
    else:
        conf = np.full(n, np.nan, dtype=float)
        dropout_flag = np.zeros(n, dtype=bool)
        conf_weight = np.ones(n, dtype=float)

    measured = clean_pred + base_noise + qatar_cfo + qatar_outlier
    mask = _subset_mask(scenario, base_obs, rng)
    mask &= ~dropout_flag

    sat_pos_solver = np.asarray(base_obs["sat_pos_m"], dtype=float).copy()
    sat_vel_solver = np.asarray(base_obs["sat_vel_mps"], dtype=float).copy()
    if scenario.sat_pos_sigma_m > 0.0:
        sat_pos_solver += rng.normal(0.0, scenario.sat_pos_sigma_m, size=sat_pos_solver.shape)
    if scenario.sat_vel_sigma_mps > 0.0:
        sat_vel_solver += rng.normal(0.0, scenario.sat_vel_sigma_mps, size=sat_vel_solver.shape)

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
            "measured_qatar_synthetic_mps": measured,
            "base_noise_mps": base_noise,
            "qatar_cfo_like_mps": qatar_cfo,
            "qatar_outlier_mps": qatar_outlier,
            "confidence_value_simulated": conf,
            "dropout_flag": dropout_flag,
            "sat_pos_sigma_m": float(scenario.sat_pos_sigma_m),
            "sat_vel_sigma_mps": float(scenario.sat_vel_sigma_mps),
            "qatar_recipe_name": scenario.recipe_name,
        }
    )
    obs = {
        "lat_deg": base_obs["lat_deg"],
        "lon_deg": base_obs["lon_deg"],
        "height_m": base_obs["height_m"],
        "p_gt_ecef_m": p0_true,
        "p0_true_m": p0_true,
        "v_true_mps": v_true,
        "b0_true_mps": float(scenario.b0_mps),
        "bdot_true_mps2": float(scenario.bdot_mps2),
        "sat_pos_m": sat_pos_solver[mask],
        "sat_vel_mps": sat_vel_solver[mask],
        "sat_pos_truth_m": np.asarray(base_obs["sat_pos_m"], dtype=float)[mask],
        "sat_vel_truth_mps": np.asarray(base_obs["sat_vel_mps"], dtype=float)[mask],
        "meas_mps": measured[mask],
        "time_s": time_s[mask],
        "satellite_number": np.asarray(base_obs["satellite_number"])[mask],
        "row_index": np.arange(n, dtype=int)[mask],
        "t0_s": t0_s,
        "scenario": scenario.name,
        "scenario_config": scenario.__dict__,
        "truth_positions_m": p_true[mask],
        "truth_bias_mps": b_true[mask],
        "confidence_weight": conf_weight[mask],
        "base_row_count": n,
    }
    meta = {
        "scenario": scenario.__dict__,
        "observation_count": int(np.sum(mask)),
        "satellite_count": int(len(np.unique(np.asarray(base_obs["satellite_number"])[mask]))),
        "dropout_count": int(np.sum(dropout_flag)),
        "qatar_outlier_count": int(np.sum(np.abs(qatar_outlier) > 0.0)),
        "qatar_outlier_amplitude_mps": float(outlier_amp),
        "cfo_scale_to_mps": float(scenario.cfo_scale_to_mps),
        "unit_warning": "Qatar col4/col5 units are unknown; only distribution shape is mapped into m/s.",
    }
    return obs, truth_df, meta
