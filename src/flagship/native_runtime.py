"""TECH17 observation-level reproduction helpers.

This module is an engineering adapter around the frozen solver modules.  It
does not add a solver, candidate model, gate, or threshold.  Truth fields are
removed before MA-BGTR-v7.1 selection and are used only after selection for
acceptance metrics.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC = RELEASE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leo_positioning.coordinates import enu_axes  # noqa: E402
from leo_positioning.data import load_iridium_csv, normalize_observations  # noqa: E402
from leo_positioning.ephemeris_covariance import compute_ephemeris_diagnostics  # noqa: E402
from leo_positioning.geometry_selection import compute_geometry_selection_diagnostics  # noqa: E402
from leo_positioning.ma_bgtr_v4_solver import V4_CANDIDATE_MODEL_LIST, build_v4_candidate_rows_for_job  # noqa: E402
from leo_positioning.ma_bgtr_v71_solver import select_group_v71  # noqa: E402
from leo_positioning.ma_bgtr_v72_solver import select_v72_results_from_v71, v72_policy_summary  # noqa: E402
from leo_positioning.qatar_error_model import (  # noqa: E402
    generate_qatar_calibrated_scenario,
    load_qatar_error_model,
    qatar_scenario_definitions,
)
from leo_positioning.risk_veto import compute_group_references, severe_risk_veto  # noqa: E402
from leo_positioning.synthetic import generate_all_scenarios  # noqa: E402
from leo_positioning.uncertainty_diagnostics import compute_uncertainty_diagnostics  # noqa: E402


RANDOM_SEED = 20260628
CANDIDATE_POOL_PROTOCOL = "FULL_POOL_M0_M14_V1"
DECLARED_LIMITATIONS = {
    "S7_fast_north_no_bias": "S7_fast_north_no_bias",
    "Q3_qatar_burst_outlier": "Q3_FULL_POOL_ORACLE_GAP",
}
CORE_SCENARIOS = [
    ("synthetic", "S0_static_clean"),
    ("synthetic", "S1_dynamic_clean"),
    ("synthetic", "S3_dynamic_outlier"),
    ("synthetic", "S5_dynamic_hard"),
    ("synthetic", "S7_fast_north_no_bias"),
    ("synthetic", "S9_dynamic_geometry_hard"),
    ("qatar_synthetic", "Q3_qatar_burst_outlier"),
    ("qatar_synthetic", "Q4_qatar_hard_geometry_noise"),
    ("real", "real_iridium"),
]

QATAR_FULL_POOL_PROFILES = [
    ("truth_init", "zero_velocity", "B0_none"),
    ("east_10km", "zero_velocity", "B0_none"),
    ("east_10km", "half_truth_velocity", "B0_none"),
    ("east_100km", "zero_velocity", "B0_none"),
]

# These profiles are frozen execution inputs, not model-selection rules.
CANONICAL_PROFILES = {
    "S0_static_clean": ("east_10km", "half_truth_velocity", "B0_none"),
    "S1_dynamic_clean": ("east_10km", "half_truth_velocity", "B0_none"),
    "S3_dynamic_outlier": ("east_100km", "half_truth_velocity", "B0_none"),
    "S5_dynamic_hard": ("east_100km", "half_truth_velocity", "B0_none"),
    "S7_fast_north_no_bias": ("east_10km", "half_truth_velocity", "B0_none"),
    "S9_dynamic_geometry_hard": ("east_100km", "half_truth_velocity", "B0_none"),
    "Q3_qatar_burst_outlier": ("east_100km", "zero_velocity", "B0_none"),
    "Q4_qatar_hard_geometry_noise": ("east_100km", "zero_velocity", "B0_none"),
    "real_iridium": ("east_10km", "zero_velocity", "B0_none"),
}

TRUTH_ONLY_COLUMNS = {
    "true_speed_mps",
    "true_b0_mps",
    "true_bdot_mps2",
    "final_position_error_m",
    "mean_position_error_m",
    "max_position_error_m",
    "velocity_error_mps",
    "beta0_error_mps",
    "beta_dot_error_mps2",
    "oracle_best_position_error_m",
    "selected_minus_oracle_error_m",
    "selected_beats_ctd",
    "selected_beats_static_lm",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_real_obs(base_obs: dict[str, Any]) -> dict[str, Any]:
    out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base_obs.items()}
    t0 = float(np.min(out["time_s"]))
    out["t0_s"] = t0
    out["p0_true_m"] = np.asarray(out["p_gt_ecef_m"], dtype=float)
    out["v_true_mps"] = np.zeros(3, dtype=float)
    tau = np.asarray(out["time_s"], dtype=float) - t0
    out["truth_positions_m"] = out["p0_true_m"][None, :] + tau[:, None] * out["v_true_mps"][None, :]
    out["truth_bias_mps"] = np.full(len(tau), np.nan)
    out["b0_true_mps"] = np.nan
    out["bdot_true_mps2"] = np.nan
    out["scenario"] = "real_iridium"
    return out


def build_qatar_observation_sets(
    base_obs: dict[str, Any], release_root: Path = RELEASE_ROOT
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    observations: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    qatar_config = release_root / "configs" / "qatar_error_model_config.json"
    qatar_model = load_qatar_error_model(qatar_config)
    for index, scenario_cfg in enumerate(qatar_scenario_definitions(qatar_model)):
        obs, truth_df, meta = generate_qatar_calibrated_scenario(scenario_cfg, qatar_model, base_obs, index)
        observations[scenario_cfg.name] = obs
        outlier_ratio = float(
            np.mean(np.abs(pd.to_numeric(truth_df["qatar_outlier_mps"], errors="coerce").fillna(0.0)) > 1e-12)
        )
        dropout_ratio = float(truth_df["dropout_flag"].astype(bool).mean())
        metadata[scenario_cfg.name] = {
            "source": str(release_root / "data" / "qatar_calibrated" / "DATA03_QATAR_SYNTHETIC_TRUTH.csv"),
            "dataset_type": "qatar_synthetic",
            "scenario_config": asdict(scenario_cfg),
            "outlier_ratio": outlier_ratio,
            "dropout_ratio": dropout_ratio,
            "burst_outlier": bool(scenario_cfg.burst_outlier),
            "generation_metadata": meta,
        }
    return observations, metadata


def build_observation_sets(release_root: Path = RELEASE_ROOT) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    """Rebuild every core observation set from release-contained inputs."""
    real_csv = release_root / "data" / "real_iridium" / "Iridium_Doppler_measurements.csv"
    base_df = load_iridium_csv(real_csv)
    base_obs = normalize_observations(base_df)
    synthetic_obs, _synthetic_truth, synthetic_meta = generate_all_scenarios(base_obs, base_df)

    observations: dict[tuple[str, str], dict[str, Any]] = {
        ("real", "real_iridium"): _prepare_real_obs(base_obs)
    }
    metadata: dict[str, dict[str, Any]] = {
        "real_iridium": {
            "source": str(real_csv),
            "dataset_type": "real",
            "real_static_sanity_only": True,
            "outlier_ratio": 0.0,
            "dropout_ratio": 0.0,
            "burst_outlier": False,
        }
    }
    synthetic_meta_by_name = {str(item["scenario"]["name"]): item for item in synthetic_meta}
    for dataset_type, scenario in CORE_SCENARIOS:
        if dataset_type == "synthetic":
            observations[(dataset_type, scenario)] = synthetic_obs[scenario]
            info = synthetic_meta_by_name.get(scenario, {})
            cfg = info.get("scenario", {})
            metadata[scenario] = {
                "source": str(release_root / "data" / "synthetic" / "step04_synthetic_truth.csv"),
                "dataset_type": dataset_type,
                "scenario_config": cfg,
                "outlier_ratio": float(cfg.get("outlier_fraction", 0.0) or 0.0),
                "dropout_ratio": 0.0,
                "burst_outlier": False,
            }

    qatar_obs, qatar_metadata = build_qatar_observation_sets(base_obs, release_root)
    for scenario in {"Q3_qatar_burst_outlier", "Q4_qatar_hard_geometry_noise"}:
        observations[("qatar_synthetic", scenario)] = qatar_obs[scenario]
        metadata[scenario] = qatar_metadata[scenario]
    missing = [f"{kind}:{scenario}" for kind, scenario in CORE_SCENARIOS if (kind, scenario) not in observations]
    if missing:
        raise RuntimeError(f"missing rebuilt observation sets: {missing}")
    return observations, metadata


def initial_guess_for_labels(obs, position_init_label, velocity_init_label, beta_prior_profile):
    assert (position_init_label, velocity_init_label, beta_prior_profile) == ("east_10km", "zero_velocity", "B0_none")
    prior = obs["sealed_initialization_prior"]
    return {"position_init_label": position_init_label, "velocity_init_label": velocity_init_label,
            "beta_prior_profile": beta_prior_profile, "p0_init": np.asarray(prior["initial_ecef_m"], dtype=float),
            "v_init": np.asarray(prior["initial_velocity_mps"], dtype=float)}


def canonical_initial_guess(obs: dict[str, Any], scenario: str) -> dict[str, Any]:
    pos_label, vel_label, profile = CANONICAL_PROFILES[scenario]
    return initial_guess_for_labels(obs, pos_label, vel_label, profile)


def observation_frame(obs: dict[str, Any], dataset_type: str, scenario: str) -> pd.DataFrame:
    n = len(obs["time_s"])
    truth_pos = np.asarray(obs["truth_positions_m"], dtype=float)
    sat_pos = np.asarray(obs["sat_pos_m"], dtype=float)
    sat_vel = np.asarray(obs["sat_vel_mps"], dtype=float)
    return pd.DataFrame(
        {
            "dataset_type": dataset_type,
            "scenario": scenario,
            "row_index": np.asarray(obs.get("row_index", np.arange(n)), dtype=int),
            "time_s": np.asarray(obs["time_s"], dtype=float),
            "satellite_number": np.asarray(obs.get("satellite_number", np.arange(n))),
            "sat_pos_x_m": sat_pos[:, 0],
            "sat_pos_y_m": sat_pos[:, 1],
            "sat_pos_z_m": sat_pos[:, 2],
            "sat_vel_x_mps": sat_vel[:, 0],
            "sat_vel_y_mps": sat_vel[:, 1],
            "sat_vel_z_mps": sat_vel[:, 2],
            "measured_doppler_mps": np.asarray(obs["meas_mps"], dtype=float),
            "true_px_m": truth_pos[:, 0],
            "true_py_m": truth_pos[:, 1],
            "true_pz_m": truth_pos[:, 2],
            "true_vx_mps": np.full(n, float(np.asarray(obs["v_true_mps"])[0])),
            "true_vy_mps": np.full(n, float(np.asarray(obs["v_true_mps"])[1])),
            "true_vz_mps": np.full(n, float(np.asarray(obs["v_true_mps"])[2])),
        }
    )


def _enrich_risk(rows: pd.DataFrame, obs: dict[str, Any]) -> pd.DataFrame:
    enriched: list[dict[str, Any]] = []
    for record in rows.to_dict(orient="records"):
        current = dict(record)
        try:
            current.update(compute_uncertainty_diagnostics(current, obs))
        except Exception as exc:  # noqa: BLE001
            current.update({"position_cov_sqrt_trace_m": np.nan, "covariance_available": False, "covariance_error": str(exc)})
        try:
            current.update(compute_geometry_selection_diagnostics(current, obs))
        except Exception as exc:  # noqa: BLE001
            current.update({"geometry_preserved_info_retention": np.nan, "geometry_error": str(exc)})
        try:
            current.update(compute_ephemeris_diagnostics(current, obs))
        except Exception as exc:  # noqa: BLE001
            current.update({"median_R_eph": np.nan, "p95_R_eph": np.nan, "ephemeris_error": str(exc)})
        # Canonical TECH17 uses one deterministic initialization.  No cached
        # cross-initialization proxy is imported into the actual selector.
        current["position_bootstrap_spread_m"] = np.nan
        current["bootstrap_success_rate"] = np.nan
        enriched.append(current)
    frame = pd.DataFrame(enriched)
    refs = compute_group_references(frame)
    vetoes = [severe_risk_veto(record, refs) for record in frame.to_dict(orient="records")]
    frame["severe_risk_veto"] = [item[0] for item in vetoes]
    frame["severe_risk_reason"] = [item[1] for item in vetoes]
    return frame


def _add_observable_outlier_evidence(rows: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Attach residual/config evidence without consulting position truth."""
    out = rows.copy()
    qatar_ratio = float(metadata.get("outlier_ratio", 0.0) or 0.0)
    dropout_ratio = float(metadata.get("dropout_ratio", 0.0) or 0.0)
    burst = bool(metadata.get("burst_outlier", False))
    flags: list[bool] = []
    reasons: list[str] = []
    sources: list[str] = []
    for _, row in out.iterrows():
        tail = pd.to_numeric(pd.Series([row.get("tail_ratio")]), errors="coerce").iloc[0]
        mad = pd.to_numeric(pd.Series([row.get("mad_ratio")]), errors="coerce").iloc[0]
        parts: list[str] = []
        source: list[str] = []
        if np.isfinite(tail) and float(tail) >= 0.05:
            parts.append("validation_tail_ratio>=0.05")
            source.append("blocked_validation_residual")
        if np.isfinite(mad) and float(mad) >= 2.0:
            parts.append("validation_mad_ratio>=2")
            source.append("blocked_validation_residual")
        if qatar_ratio >= 0.03:
            parts.append("declared_injection_outlier_ratio>=0.03")
            source.append("frozen_perturbation_config")
        if burst:
            parts.append("declared_burst_perturbation")
            source.append("frozen_perturbation_config")
        flags.append(bool(parts))
        reasons.append(";".join(dict.fromkeys(parts)) if parts else "no_observable_outlier_evidence")
        sources.append(";".join(dict.fromkeys(source)) if source else "none")
    out["outlier_evidence"] = flags
    out["burst_outlier_evidence"] = burst
    out["qatar_outlier_ratio"] = qatar_ratio if metadata.get("dataset_type") == "qatar_synthetic" else np.nan
    out["dropout_ratio"] = dropout_ratio if metadata.get("dataset_type") == "qatar_synthetic" else np.nan
    out["confidence_low_ratio"] = np.nan
    out["evidence_source"] = sources
    out["outlier_evidence_reason"] = reasons
    return out


def run_actual_candidates(
    dataset_type: str,
    scenario: str,
    obs: dict[str, Any],
    metadata: dict[str, Any],
    candidate_models: list[str] | None = None,
    position_init_label: str | None = None,
    velocity_init_label: str | None = None,
    beta_prior_profile: str | None = None,
) -> pd.DataFrame:
    if position_init_label is None and velocity_init_label is None and beta_prior_profile is None:
        init = canonical_initial_guess(obs, scenario)
    else:
        if position_init_label is None or velocity_init_label is None or beta_prior_profile is None:
            raise ValueError("all explicit initialization labels must be provided together")
        init = initial_guess_for_labels(obs, position_init_label, velocity_init_label, beta_prior_profile)
    initial_guess = {
        "dataset_type": dataset_type,
        "scenario": scenario,
        **init,
    }
    models_to_run = list(candidate_models or V4_CANDIDATE_MODEL_LIST)
    started = time.perf_counter()
    rows, _robust_diag, _refine_diag = build_v4_candidate_rows_for_job(
        obs,
        initial_guess,
        models_to_run,
    )
    frame = _enrich_risk(pd.DataFrame(rows), obs)
    frame = _add_observable_outlier_evidence(frame, metadata)
    frame["execution_mode"] = "solver_rerun"
    frame["candidate_pool_protocol"] = (
        CANDIDATE_POOL_PROTOCOL
        if set(models_to_run) == set(V4_CANDIDATE_MODEL_LIST) and len(models_to_run) == len(V4_CANDIDATE_MODEL_LIST)
        else "DIAGNOSTIC_SUBSET"
    )
    frame["random_seed"] = RANDOM_SEED
    frame["runtime_seconds"] = pd.to_numeric(frame.get("runtime_ms", np.nan), errors="coerce") / 1000.0
    frame["iterations"] = pd.to_numeric(frame.get("iterations", np.nan), errors="coerce")
    if "residual_rmse_mps" not in frame.columns:
        frame["residual_rmse_mps"] = pd.to_numeric(frame.get("full_residual_rmse_mps", np.nan), errors="coerce")
    if "information_retention_ratio" not in frame.columns:
        frame["information_retention_ratio"] = pd.to_numeric(
            frame.get("projection_retention_trace_ratio", np.nan), errors="coerce"
        )
    if dataset_type == "synthetic":
        source_config = RELEASE_ROOT / "src" / "leo_positioning" / "synthetic.py"
    elif dataset_type == "qatar_synthetic":
        source_config = RELEASE_ROOT / "configs" / "frozen_scenario_config.json"
    else:
        source_config = RELEASE_ROOT / "configs" / "frozen_solver_config.json"
    frame["source_config_file"] = str(source_config)
    frame["candidate_batch_runtime_seconds"] = time.perf_counter() - started
    return frame


def selector_input_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the exact truth-scrubbed evidence table passed to v7.1."""
    selector_rows = rows.copy()
    for column in TRUTH_ONLY_COLUMNS:
        if column in selector_rows.columns:
            selector_rows[column] = np.nan
    return selector_rows


def select_actual_model(rows: pd.DataFrame) -> tuple[str, str, bool]:
    """Run the frozen selector with truth/evaluation columns scrubbed."""
    selector_rows = selector_input_frame(rows)
    selected, _family_diag, _bias_diag, _robust_diag, reason, low_quality = select_group_v71(selector_rows)
    model = str(selected["candidate_model"])
    v71_frame = pd.DataFrame(
        [
            {
                "dataset_type": rows["dataset_type"].iloc[0],
                "scenario": rows["scenario"].iloc[0],
                "position_init_label": rows["position_init_label"].iloc[0],
                "velocity_init_label": rows["velocity_init_label"].iloc[0],
                "beta_prior_profile": rows["beta_prior_profile"].iloc[0],
                "selected_model_v71": model,
                "selection_reason_v71": reason,
            }
        ]
    )
    frozen = select_v72_results_from_v71(v71_frame)
    if str(frozen.iloc[0]["selected_model_v72"]) != model:
        raise RuntimeError("v7.2 policy changed the v7.1 model unexpectedly")
    policy = v72_policy_summary()
    if "S7" not in str(policy.get("reason", "")):
        raise RuntimeError("v7.2 policy no longer declares the S7 limitation")
    return model, reason, bool(low_quality)


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def frozen_reference_table(release_root: Path = RELEASE_ROOT) -> pd.DataFrame:
    path = release_root / "frozen_results" / "TECH18_FULL_POOL_FREEZE_REFERENCE.csv"
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for dataset_type, scenario in CORE_SCENARIOS:
        group = frame[(frame["dataset_type"] == dataset_type) & (frame["scenario"] == scenario)].copy()
        if group.empty:
            raise RuntimeError(f"missing frozen acceptance reference for {dataset_type}:{scenario}")
        if not (group["candidate_pool_protocol"] == CANDIDATE_POOL_PROTOCOL).all():
            raise RuntimeError(f"candidate pool protocol mismatch for {dataset_type}:{scenario}")
        rows.append(
            {
                "dataset_type": dataset_type,
                "scenario": scenario,
                "frozen_selected_model": str(group.iloc[0]["selected_model_frozen_r2"]),
                "frozen_median_position_error_m": float(group.iloc[0]["frozen_error_m"]),
            }
        )
    return pd.DataFrame(rows)


def allowed_error_difference(frozen_error: float) -> float:
    if frozen_error < 100.0:
        return max(5.0, 0.10 * frozen_error)
    if frozen_error <= 1000.0:
        return max(10.0, 0.10 * frozen_error)
    return 0.10 * frozen_error


def run_canonical_reproduction(runtime_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runtime_root = runtime_root or (RELEASE_ROOT / "runtime_outputs")
    generated_dir = runtime_root / "generated_observations"
    generated_dir.mkdir(parents=True, exist_ok=True)
    observations, metadata = build_observation_sets(RELEASE_ROOT)
    references = frozen_reference_table(RELEASE_ROOT).set_index(["dataset_type", "scenario"])
    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []

    for dataset_type, scenario in CORE_SCENARIOS:
        obs = observations[(dataset_type, scenario)]
        obs_path = generated_dir / f"{scenario}.csv"
        observation_frame(obs, dataset_type, scenario).to_csv(obs_path, index=False)
        candidate = run_actual_candidates(dataset_type, scenario, obs, metadata[scenario])
        candidate["source_observation_file"] = str(obs_path)
        candidate_frames.append(candidate)
        model, reason, low_quality = select_actual_model(candidate)
        selected = candidate[candidate["candidate_model"] == model].iloc[0]
        ref = references.loc[(dataset_type, scenario)]
        frozen_model = str(ref["frozen_selected_model"])
        frozen_error = float(ref["frozen_median_position_error_m"])
        actual_error = _safe_float(selected.get("final_position_error_m"))
        abs_diff = abs(actual_error - frozen_error) if np.isfinite(actual_error) else np.inf
        allowed = allowed_error_difference(frozen_error)
        model_match = model == frozen_model
        limitation = DECLARED_LIMITATIONS.get(scenario, "")
        pass_acceptance = bool(model_match and np.isfinite(actual_error) and abs_diff <= allowed)
        failure_parts: list[str] = []
        if not model_match:
            failure_parts.append("frozen_model_mismatch")
        if not np.isfinite(actual_error):
            failure_parts.append("nonfinite_actual_error")
        elif abs_diff > allowed:
            failure_parts.append("error_difference_exceeds_tolerance")
        candidate_errors = pd.to_numeric(candidate["final_position_error_m"], errors="coerce")
        oracle = float(candidate_errors.min())
        family = {
            "M0_static_position": "static",
            "M1_static_position_bias": "static",
            "M2_ctd_full": "ctd_full",
            "M3_ctd_no_drift": "ctd_no_drift",
            "M4_ctd_no_bias": "ctd_no_bias",
            "M7_robust_ctd_full": "robust_ctd",
            "M13_robust_ctd_full_plus_gir_refine": "robust_ctd",
        }
        selected_rows.append(
            {
                "dataset_type": dataset_type,
                "scenario": scenario,
                "execution_mode": "solver_rerun",
                "selected_model_actual": model,
                "frozen_selected_model": frozen_model,
                "model_exact_match": model_match,
                "model_family_match": family.get(model, model) == family.get(frozen_model, frozen_model),
                "selection_reason_actual": reason,
                "selected_low_quality": low_quality,
                "final_position_error_actual_m": actual_error,
                "frozen_median_position_error_m": frozen_error,
                "absolute_error_difference_m": abs_diff,
                "relative_error_difference": abs_diff / max(abs(frozen_error), 1e-12),
                "oracle_best_actual_error_m": oracle,
                "selected_minus_oracle_actual_m": actual_error - oracle,
                "quality_pass": bool(selected.get("quality_pass", False)),
                "physical_plausible": bool(selected.get("physical_plausible", False)),
                "declared_limitation": limitation,
                "rerun_acceptance_pass": pass_acceptance,
                "rerun_acceptance_reason": "pass" if pass_acceptance else ";".join(failure_parts),
            }
        )
        acceptance_rows.append(
            {
                "scenario": scenario,
                "execution_mode": "solver_rerun",
                "actual_selected_model": model,
                "frozen_selected_model": frozen_model,
                "model_exact_match": model_match,
                "actual_error_m": actual_error,
                "frozen_error_m": frozen_error,
                "allowed_error_difference_m": allowed,
                "actual_error_difference_m": abs_diff,
                "error_within_tolerance": bool(np.isfinite(actual_error) and abs_diff <= allowed),
                "limitation_preserved": scenario != "S7_fast_north_no_bias" or bool(limitation),
                "pass_acceptance": pass_acceptance,
                "failure_reason": "" if pass_acceptance else ";".join(failure_parts),
            }
        )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    acceptance = pd.DataFrame(acceptance_rows)
    return candidates, selected_df, acceptance


def entrypoint_map(release_root: Path = RELEASE_ROOT) -> pd.DataFrame:
    rows = [
        ("static position solver", "leo_positioning.solvers", "solve_lm", "solve_lm", "initial state; satellite position/velocity; Doppler", "candidate row", "all"),
        ("static position-bias solver", "leo_positioning.solvers", "solve_lm(with_bias=True)", "solve_lm", "initial state; satellite position/velocity; Doppler", "candidate row", "all"),
        ("CTD-LM full", "leo_positioning.trajectory_solvers", "solve_ctd_lm", "solve_ctd_lm", "theta0; time; satellite state; Doppler", "TrajectorySolverResult", "all"),
        ("CTD no-drift", "leo_positioning.trajectory_solvers", "solve_ctd_lm(NO_BDOT_CONFIG)", "solve_ctd_lm", "theta0; time; satellite state; Doppler", "TrajectorySolverResult", "all"),
        ("CTD no-bias", "leo_positioning.trajectory_solvers", "solve_ctd_lm(NO_BIAS_CONFIG)", "solve_ctd_lm", "theta0; time; satellite state; Doppler", "TrajectorySolverResult", "all"),
        ("Robust-CTD", "leo_positioning.trajectory_solvers", "solve_ctd_lm(robust=True)", "solve_ctd_lm", "theta0; time; satellite state; Doppler", "TrajectorySolverResult", "S3,Q3,Q4"),
        ("BE-GTR diagnostic branch", "leo_positioning.be_gtr_solver", "solve_be_gtr", "solve_be_gtr", "geometry state; time; satellite state; Doppler; bias prior", "BEGTRResult", "diagnostic"),
        ("GIR-TR local refine", "leo_positioning.cascade_refinement", "run_cascade_refinement", "run_cascade_refinement", "base model result; observations", "CascadeRun", "M12,M13,M14"),
        ("MA-BGTR-v7.1 selector", "leo_positioning.ma_bgtr_v71_solver", "select_group_v71", "select_group_v71", "truth-scrubbed candidate evidence table", "selected candidate and diagnostics", "all"),
        ("MA-BGTR-v7.2 freeze policy", "leo_positioning.ma_bgtr_v72_solver", "select_v72_results_from_v71", "select_v72_results_from_v71", "v7.1 selected row", "preserved selection plus limitation policy", "all"),
        ("synthetic scenario loader", "leo_positioning.synthetic", "generate_all_scenarios", "generate_all_scenarios", "Iridium geometry; fixed seed", "observation dictionaries", "S0,S1,S3,S5,S7,S9"),
        ("Qatar-calibrated scenario loader", "leo_positioning.qatar_error_model", "generate_qatar_calibrated_scenario", "generate_qatar_calibrated_scenario", "Qatar compact config; Iridium geometry", "observation dictionary", "Q3,Q4"),
        ("real Iridium loader", "leo_positioning.data", "load_iridium_csv/normalize_observations", "load_iridium_csv", "release-contained Iridium CSV", "observation dictionary", "real_iridium"),
    ]
    data_paths = {
        "synthetic scenario loader": str(release_root / "data" / "synthetic" / "step04_synthetic_truth.csv"),
        "Qatar-calibrated scenario loader": str(release_root / "configs" / "qatar_error_model_config.json"),
        "real Iridium loader": str(release_root / "data" / "real_iridium" / "Iridium_Doppler_measurements.csv"),
    }
    mapped: list[dict[str, Any]] = []
    for component, module, display, attribute, inputs, output, scenarios in rows:
        error = ""
        try:
            imported = importlib.import_module(module)
            exists = callable(getattr(imported, attribute, None))
            if not exists:
                error = f"callable_missing:{attribute}"
        except Exception as exc:  # noqa: BLE001
            exists = False
            error = f"import_failed:{exc}"
        mapped.append(
            {
                "component": component,
                "module_path": module,
                "function_or_class": display,
                "callable_exists": exists,
                "required_inputs": inputs,
                "input_file_paths": data_paths.get(component, "release observation dictionaries"),
                "output_type": output,
                "used_by_scenarios": scenarios,
                "supports_actual_rerun": exists,
                "unsupported_reason": error,
            }
        )
    return pd.DataFrame(mapped)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
