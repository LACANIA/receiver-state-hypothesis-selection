"""Execute the pre-frozen PAPER-EXP01 hold-out protocol.

This script is an experiment harness only.  It imports the immutable r2
release, executes its M0--M14 candidate builders and frozen v7.1/v7.2
selection path, and writes evaluation outputs outside the release tree.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon


OUT = Path(__file__).resolve().parent
LAB = OUT.parents[1]
RELEASE = LAB / "final_release" / "MA_BGTR_v7_2_freeze_r2"
PROTOCOL_PATH = OUT / "PAPER_EXP01_HOLDOUT_PROTOCOL.json"
PROTOCOL_HASH_PATH = OUT / "PAPER_EXP01_HOLDOUT_PROTOCOL_SHA256.txt"
BEFORE_HASH_PATH = OUT / "PAPER_EXP01_FROZEN_HASHES_BEFORE.csv"
BASE_DATA = RELEASE / "data" / "real_iridium" / "Iridium_Doppler_measurements.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_protocol() -> dict[str, Any]:
    expected = PROTOCOL_HASH_PATH.read_text(encoding="utf-8").split()[0].strip()
    actual = sha256_file(PROTOCOL_PATH)
    if actual != expected:
        raise RuntimeError(f"hold-out protocol hash mismatch: expected {expected}, got {actual}")
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def verify_frozen_hashes(output_name: str) -> pd.DataFrame:
    before = pd.read_csv(BEFORE_HASH_PATH)
    rows: list[dict[str, Any]] = []
    for record in before.to_dict(orient="records"):
        rel = str(record["relative_path"])
        path = RELEASE / Path(rel)
        after = sha256_file(path) if path.is_file() else "missing"
        prior = str(record["sha256_before"])
        rows.append(
            {
                "category": record["category"],
                "relative_path": rel.replace("\\", "/"),
                "sha256_before": prior,
                "sha256_after": after,
                "unchanged": bool(after == prior),
                "size_bytes_after": path.stat().st_size if path.is_file() else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / output_name, index=False)
    if not bool(result["unchanged"].all()):
        changed = result.loc[~result["unchanged"], "relative_path"].tolist()
        raise RuntimeError(f"frozen source/config changed: {changed}")
    return result


def load_release_runtime():
    path = RELEASE / "scripts" / "release_runtime.py"
    spec = importlib.util.spec_from_file_location("paper_exp01_release_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import release runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value) if not pd.isna(value) else False


def model_family(model: str) -> str:
    """Return the predeclared experimental branch family."""
    mapping = {
        "M0_static_position": "static",
        "M1_static_position_bias": "static",
        "M2_ctd_full": "core_ctd",
        "M3_ctd_no_drift": "core_ctd",
        "M4_ctd_no_bias": "core_ctd",
        "M5_be_b0_only": "be_gtr",
        "M6_be_full": "be_gtr",
        "M7_robust_ctd_full": "robust",
        "M8_robust_be_b0_only": "robust",
        "M9_robust_be_full": "robust",
        "M10_gir_tr_fixed_scale": "direct_gir",
        "M11_gir_tr_mad_scale": "direct_gir",
        "M12_ctd_full_plus_gir_refine": "cascaded_refinement",
        "M13_robust_ctd_full_plus_gir_refine": "cascaded_refinement",
        "M14_static_plus_gir_refine": "cascaded_refinement",
    }
    if model not in mapping:
        raise KeyError(f"unknown frozen model: {model}")
    return mapping[model]


def model_subfamily(model: str) -> str:
    """Return a state/robustness-specific subfamily for secondary analysis."""
    mapping = {
        "M0_static_position": "static_position",
        "M1_static_position_bias": "static_bias",
        "M2_ctd_full": "dynamic_full",
        "M3_ctd_no_drift": "dynamic_no_drift",
        "M4_ctd_no_bias": "dynamic_no_bias",
        "M5_be_b0_only": "bias_elimination_diagnostic",
        "M6_be_full": "bias_elimination_diagnostic",
        "M7_robust_ctd_full": "robust_dynamic_full",
        "M8_robust_be_b0_only": "robust_bias_elimination_diagnostic",
        "M9_robust_be_full": "robust_bias_elimination_diagnostic",
        "M10_gir_tr_fixed_scale": "direct_gir_diagnostic",
        "M11_gir_tr_mad_scale": "direct_gir_diagnostic",
        "M12_ctd_full_plus_gir_refine": "cascaded_dynamic_full",
        "M13_robust_ctd_full_plus_gir_refine": "cascaded_robust_dynamic_full",
        "M14_static_plus_gir_refine": "cascaded_static",
    }
    if model not in mapping:
        raise KeyError(f"unknown frozen model: {model}")
    return mapping[model]


def _copy_array(value: Any) -> Any:
    return value.copy() if isinstance(value, np.ndarray) else value


def build_holdout_observation(
    record: dict[str, Any],
    base_obs: dict[str, Any],
    *,
    enu_axes,
    pack_state,
    predict_ctd,
    full_ctd_config,
    sample_heavy_tail_outliers,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one deterministic observation set from a protocol record."""
    base_time = np.asarray(base_obs["time_s"], dtype=float)
    base_sats = np.asarray(base_obs["satellite_number"])
    subset = {int(item) for item in str(record["satellite_subset"]).split(";")}
    mask = (
        np.isin(base_sats.astype(int), sorted(subset))
        & (base_time >= float(record["window_start_s"]))
        & (base_time <= float(record["window_end_s"]))
    )
    if int(mask.sum()) != int(record["observation_count"]):
        raise RuntimeError(
            f"{record['holdout_id']} observation-count mismatch: protocol={record['observation_count']} built={mask.sum()}"
        )

    time_s = base_time[mask]
    sat_pos_truth = np.asarray(base_obs["sat_pos_m"], dtype=float)[mask]
    sat_vel_truth = np.asarray(base_obs["sat_vel_mps"], dtype=float)[mask]
    sats = base_sats[mask]
    row_index = np.arange(len(base_time), dtype=int)[mask]
    t0_s = float(np.min(time_s))
    tau = time_s - t0_s
    p0_true = np.asarray(base_obs["p_gt_ecef_m"], dtype=float).copy()
    east, north, _up = enu_axes(float(base_obs["lat_deg"]), float(base_obs["lon_deg"]))
    speed = float(record["speed_mps"])
    direction_deg = safe_float(record.get("direction_deg_enu"))
    if speed == 0.0:
        v_true = np.zeros(3, dtype=float)
    else:
        angle = np.deg2rad(direction_deg)
        direction = np.cos(angle) * east + np.sin(angle) * north
        v_true = speed * direction / np.linalg.norm(direction)

    b0 = float(record["b0_mps"])
    bdot = float(record["bdot_mps2"])
    theta_true = pack_state(p0_true, v_true, b0, bdot, full_ctd_config)
    clean = predict_ctd(
        theta_true,
        time_s,
        sat_pos_truth,
        sat_vel_truth,
        t0_s=t0_s,
        config=full_ctd_config,
    )

    noise_rng = np.random.default_rng(int(record["noise_seed"]))
    sigma = float(record["noise_sigma_mps"])
    if safe_bool(record["heavy_tail_enabled"]):
        # Student-t(df=3)/sqrt(3) has unit variance.  The scale and shape were
        # fixed in the protocol before any solver run.
        noise = noise_rng.standard_t(df=3.0, size=len(time_s)) * sigma / np.sqrt(3.0)
    else:
        noise = noise_rng.normal(0.0, sigma, size=len(time_s))

    label_rng = np.random.default_rng(int(record["label_seed"]))
    injected = np.zeros(len(time_s), dtype=float)
    burst_mask = np.zeros(len(time_s), dtype=bool)
    ratio = float(record["outlier_ratio"])
    amplitude = float(record["outlier_amplitude_mps"])
    if safe_bool(record["burst_enabled"]) and ratio > 0.0:
        count = max(1, int(round(ratio * len(time_s))))
        start_fraction = safe_float(record["burst_start_fraction"])
        max_start = max(0, len(time_s) - count)
        start = int(round(np.clip(start_fraction, 0.0, 1.0) * max_start))
        indices = np.arange(start, min(start + count, len(time_s)))
        sign = float(label_rng.choice(np.array([-1.0, 1.0])))
        magnitudes = label_rng.lognormal(
            mean=np.log(max(amplitude, 1e-9)), sigma=0.25, size=len(indices)
        )
        injected[indices] = sign * magnitudes
        burst_mask[indices] = True
    elif ratio > 0.0:
        injected = sample_heavy_tail_outliers(label_rng, len(time_s), ratio, amplitude)

    confidence_target = float(record["confidence_low_ratio_target"])
    if safe_bool(record["confidence_simulated"]) and confidence_target > 0.0:
        confidence_low = label_rng.random(len(time_s)) < confidence_target
        confidence = np.where(
            confidence_low,
            label_rng.integers(65, 90, size=len(time_s)),
            label_rng.integers(95, 102, size=len(time_s)),
        ).astype(float)
    else:
        confidence_low = np.zeros(len(time_s), dtype=bool)
        confidence = np.full(len(time_s), 100.0, dtype=float)

    sat_pos_solver = sat_pos_truth.copy()
    sat_vel_solver = sat_vel_truth.copy()
    ephemeris_rng = np.random.default_rng(int(record["ephemeris_seed"]))
    sat_pos_sigma = float(record["sat_pos_sigma_m"])
    sat_vel_sigma = float(record["sat_vel_sigma_mps"])
    if sat_pos_sigma > 0.0:
        sat_pos_solver += ephemeris_rng.normal(0.0, sat_pos_sigma, size=sat_pos_solver.shape)
    if sat_vel_sigma > 0.0:
        sat_vel_solver += ephemeris_rng.normal(0.0, sat_vel_sigma, size=sat_vel_solver.shape)

    measured = clean + noise + injected
    truth_positions = p0_true[None, :] + tau[:, None] * v_true[None, :]
    truth_bias = b0 + bdot * tau
    obs = {
        "lat_deg": float(base_obs["lat_deg"]),
        "lon_deg": float(base_obs["lon_deg"]),
        "height_m": float(base_obs["height_m"]),
        "p_gt_ecef_m": p0_true.copy(),
        "sat_pos_m": sat_pos_solver,
        "sat_vel_mps": sat_vel_solver,
        "sat_pos_truth_m": sat_pos_truth,
        "sat_vel_truth_mps": sat_vel_truth,
        "meas_mps": measured,
        "time_s": time_s,
        "satellite_number": sats,
        "row_index": row_index,
        "t0_s": t0_s,
        "p0_true_m": p0_true,
        "v_true_mps": v_true,
        "b0_true_mps": b0,
        "bdot_true_mps2": bdot,
        "scenario": str(record["selector_scenario_key"]),
        "scenario_config": {"protocol_id": "PAPER_EXP01_HOLDOUT_MC_V1"},
        "truth_positions_m": truth_positions,
        "truth_bias_mps": truth_bias,
        "base_row_count": len(base_time),
        "confidence_values": confidence,
    }
    audit = {
        "actual_observation_count": len(time_s),
        "actual_duration_s": float(np.max(time_s) - np.min(time_s)),
        "actual_sampling_interval_median_s": float(np.median(np.diff(np.unique(time_s)))),
        "actual_unique_satellite_count": int(np.unique(sats).size),
        "injected_outlier_count": int(np.count_nonzero(injected)),
        "injected_outlier_fraction_actual": float(np.mean(np.abs(injected) > 0.0)),
        "burst_outlier_count": int(burst_mask.sum()),
        "confidence_low_ratio_actual": float(confidence_low.mean()),
        "noise_realization_rmse_mps": float(np.sqrt(np.mean(noise**2))),
        "injected_outlier_abs_median_mps": (
            float(np.median(np.abs(injected[np.abs(injected) > 0.0])))
            if np.any(np.abs(injected) > 0.0)
            else 0.0
        ),
    }
    return obs, audit


def selection_with_diagnostics(rr, frame: pd.DataFrame) -> tuple[str, str, bool, dict[str, Any]]:
    selector_frame = rr.selector_input_frame(frame)
    for column in rr.TRUTH_ONLY_COLUMNS:
        if column in selector_frame and selector_frame[column].notna().any():
            raise RuntimeError(f"truth isolation failed for column {column}")
    forbidden_names = {
        "holdout_family",
        "expected_model",
        "oracle_model",
        "injected_outlier_flag",
        "qatar_outlier_mps",
        "dropout_truth_flag",
    }
    leaked = sorted(forbidden_names.intersection(selector_frame.columns))
    if leaked:
        raise RuntimeError(f"forbidden fields present in selector interface: {leaked}")

    selected, family_diag, bias_diag, robust_diag, reason, low_quality = rr.select_group_v71(selector_frame)
    direct_model = str(selected["candidate_model"])
    frozen_model, frozen_reason, frozen_low_quality = rr.select_actual_model(frame)
    if (direct_model, reason, bool(low_quality)) != (
        frozen_model,
        frozen_reason,
        bool(frozen_low_quality),
    ):
        raise RuntimeError("direct v7.1 diagnostics and frozen v7.2 adapter disagree")
    diagnostics = {
        "family_diagnostics": family_diag,
        "bias_diagnostics": bias_diag,
        "robust_diagnostics": robust_diag,
    }
    return frozen_model, frozen_reason, bool(frozen_low_quality), diagnostics


def select_one_mode(
    rr,
    candidate_frame: pd.DataFrame,
    *,
    mode: str,
    record: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    frame = candidate_frame.copy()
    if mode == "observable_only":
        # run_actual_candidates received zero label metadata, so these fields
        # are derived only from residual tails/MAD.  Confidence is a simulated
        # runtime-observable quantity, not an injection flag.
        frame["confidence_low_ratio"] = float(audit["confidence_low_ratio_actual"])
    elif mode == "label_assisted_ablation":
        has_injection = bool(audit["injected_outlier_count"] > 0)
        has_burst = bool(audit["burst_outlier_count"] > 0)
        residual_flag = frame["outlier_evidence"].map(safe_bool)
        frame["outlier_evidence"] = residual_flag | has_injection
        frame["burst_outlier_evidence"] = has_burst
        frame["confidence_low_ratio"] = float(audit["confidence_low_ratio_actual"])
        if has_injection:
            frame["evidence_source"] = frame["evidence_source"].astype(str).map(
                lambda value: ";".join(
                    item for item in dict.fromkeys([value, "synthetic_injection_label"]) if item not in {"", "none", "nan"}
                )
            )
            frame["outlier_evidence_reason"] = frame["outlier_evidence_reason"].astype(str).map(
                lambda value: ";".join(
                    item
                    for item in dict.fromkeys([value, "label_assisted_injected_outlier_present"])
                    if item not in {"", "no_observable_outlier_evidence", "nan"}
                )
            )
    else:
        raise ValueError(mode)

    selected_model, reason, low_quality, diagnostics = selection_with_diagnostics(rr, frame)
    selected = frame.loc[frame["candidate_model"].astype(str) == selected_model].iloc[0]
    finite_errors = pd.to_numeric(frame["final_position_error_m"], errors="coerce")
    finite_mask = np.isfinite(finite_errors.to_numpy(dtype=float))
    if not finite_mask.any():
        oracle_model = "none_no_finite_candidate"
        oracle_error = float("nan")
    else:
        oracle_idx = finite_errors[finite_mask].idxmin()
        oracle_model = str(frame.loc[oracle_idx, "candidate_model"])
        oracle_error = float(frame.loc[oracle_idx, "final_position_error_m"])
    selected_error = safe_float(selected.get("final_position_error_m"))
    gap = selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else np.nan

    def baseline_error(model: str) -> float:
        values = frame.loc[frame["candidate_model"].astype(str) == model, "final_position_error_m"]
        return safe_float(values.iloc[0]) if len(values) else np.nan

    numerical = safe_bool(selected.get("numerical_success")) and np.isfinite(selected_error)
    output = {
        "protocol_id": "PAPER_EXP01_HOLDOUT_MC_V1",
        "holdout_id": record["holdout_id"],
        "selector_scenario_key": record["selector_scenario_key"],
        "holdout_family": record["holdout_family"],
        "realization_index": int(record["realization_index"]),
        "evaluation_scope": "primary" if str(frame["position_init_label"].iloc[0]) == "east_10km" and str(frame["velocity_init_label"].iloc[0]) == "zero_velocity" else "initialization_sensitivity",
        "selector_mode": mode,
        "position_init_label": frame["position_init_label"].iloc[0],
        "velocity_init_label": frame["velocity_init_label"].iloc[0],
        "beta_prior_profile": frame["beta_prior_profile"].iloc[0],
        "selected_model": selected_model,
        "selected_model_family": model_family(selected_model),
        "selected_model_subfamily": model_subfamily(selected_model),
        "selection_reason": reason,
        "selected_low_quality": low_quality,
        "numerical_success": numerical,
        "converged": safe_bool(selected.get("converged")),
        "quality_pass": safe_bool(selected.get("quality_pass")),
        "physical_plausible": safe_bool(selected.get("physical_plausible")),
        "final_position_error_m": selected_error,
        "mean_position_error_m": safe_float(selected.get("mean_position_error_m")),
        "velocity_error_mps": safe_float(selected.get("velocity_error_mps")),
        "residual_rmse_mps": safe_float(selected.get("residual_rmse_mps", selected.get("full_residual_rmse_mps"))),
        "raw_validation_rmse_mps": safe_float(selected.get("raw_validation_rmse_mps")),
        "trimmed_validation_rmse_mps": safe_float(selected.get("trimmed_validation_rmse_mps")),
        "inlier_validation_rmse_mps": safe_float(selected.get("inlier_validation_rmse_mps")),
        "condition_number": safe_float(selected.get("condition_number")),
        "estimated_speed_mps": safe_float(selected.get("estimated_speed_mps")),
        "beta0_estimated_mps": safe_float(selected.get("beta0_estimated_mps")),
        "beta_dot_estimated_mps2": safe_float(selected.get("beta_dot_estimated_mps2")),
        "oracle_model": oracle_model,
        "oracle_model_family": model_family(oracle_model) if oracle_model.startswith("M") else "none",
        "oracle_model_subfamily": model_subfamily(oracle_model) if oracle_model.startswith("M") else "none",
        "oracle_error_m": oracle_error,
        "selected_minus_oracle_error_m": gap,
        "exact_oracle_selected": bool(selected_model == oracle_model),
        "selected_family_equals_oracle_family": bool(
            oracle_model.startswith("M") and model_family(selected_model) == model_family(oracle_model)
        ),
        "m0_error_m": baseline_error("M0_static_position"),
        "m2_error_m": baseline_error("M2_ctd_full"),
        "selected_beats_m0": bool(np.isfinite(selected_error) and selected_error <= baseline_error("M0_static_position")),
        "selected_beats_m2": bool(np.isfinite(selected_error) and selected_error <= baseline_error("M2_ctd_full")),
        "failure": bool(not numerical),
        "candidate_count": int(len(frame)),
        "candidate_pool_protocol": str(frame["candidate_pool_protocol"].iloc[0]),
        "execution_mode": str(frame["execution_mode"].iloc[0]),
        "candidate_batch_runtime_seconds": safe_float(frame["candidate_batch_runtime_seconds"].iloc[0]),
        "sum_candidate_runtime_seconds": float(pd.to_numeric(frame["runtime_seconds"], errors="coerce").sum()),
        "mode_a_truth_columns_scrubbed": True,
        "synthetic_labels_supplied_to_selector": bool(mode == "label_assisted_ablation"),
        "confidence_low_ratio_observed": float(audit["confidence_low_ratio_actual"]),
        "post_selection_true_speed_mps": float(record["speed_mps"]),
        "post_selection_true_b0_mps": float(record["b0_mps"]),
        "post_selection_true_bdot_mps2": float(record["bdot_mps2"]),
        "post_selection_injected_outlier_fraction": float(audit["injected_outlier_fraction_actual"]),
    }
    diag_row = {
        "holdout_id": record["holdout_id"],
        "holdout_family": record["holdout_family"],
        "selector_mode": mode,
        "position_init_label": output["position_init_label"],
        "velocity_init_label": output["velocity_init_label"],
        "selected_model": selected_model,
        "family_diagnostics_json": json.dumps(diagnostics["family_diagnostics"], sort_keys=True, default=str),
        "bias_diagnostics_json": json.dumps(diagnostics["bias_diagnostics"], sort_keys=True, default=str),
        "robust_diagnostics_json": json.dumps(diagnostics["robust_diagnostics"], sort_keys=True, default=str),
    }
    return output, diag_row, frame


def bootstrap_median_ci(values: Iterable[float], rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan
    draws = rng.choice(array, size=(n_boot, array.size), replace=True)
    medians = np.median(draws, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def distribution_json(values: Iterable[str]) -> str:
    counter = Counter(str(value) for value in values)
    return json.dumps(dict(sorted(counter.items())), separators=(",", ":"), ensure_ascii=True)


def aggregate_selected(primary_selected: pd.DataFrame, master_seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (family, mode), group in primary_selected.groupby(["holdout_family", "selector_mode"], sort=True):
        groups.append((str(family), str(mode), group.copy()))
    for mode, group in primary_selected.groupby("selector_mode", sort=True):
        groups.append(("ALL_FAMILIES", str(mode), group.copy()))
    for idx, (family, mode, group) in enumerate(groups):
        pos = pd.to_numeric(group["final_position_error_m"], errors="coerce")
        gap = pd.to_numeric(group["selected_minus_oracle_error_m"], errors="coerce")
        rng_pos = np.random.default_rng(master_seed + 10000 + idx * 2)
        rng_gap = np.random.default_rng(master_seed + 10001 + idx * 2)
        pos_ci = bootstrap_median_ci(pos, rng_pos)
        gap_ci = bootstrap_median_ci(gap, rng_gap)
        finite = np.isfinite(pos.to_numpy(dtype=float))
        rows.append(
            {
                "holdout_family": family,
                "selector_mode": mode,
                "runs": int(len(group)),
                "finite_position_error_runs": int(finite.sum()),
                "numerical_success_rate": float(group["numerical_success"].map(safe_bool).mean()),
                "quality_pass_rate": float(group["quality_pass"].map(safe_bool).mean()),
                "selected_model_distribution": distribution_json(group["selected_model"]),
                "selected_model_family_distribution": distribution_json(group["selected_model_family"]),
                "median_final_position_error_m": float(pos.median()),
                "p75_final_position_error_m": float(pos.quantile(0.75)),
                "p95_final_position_error_m": float(pos.quantile(0.95)),
                "maximum_final_position_error_m": float(pos.max()),
                "median_velocity_error_mps": float(pd.to_numeric(group["velocity_error_mps"], errors="coerce").median()),
                "median_residual_rmse_mps": float(pd.to_numeric(group["residual_rmse_mps"], errors="coerce").median()),
                "median_trimmed_validation_rmse_mps": float(pd.to_numeric(group["trimmed_validation_rmse_mps"], errors="coerce").median()),
                "median_selected_minus_oracle_error_m": float(gap.median()),
                "p95_selected_minus_oracle_error_m": float(gap.quantile(0.95)),
                "exact_oracle_selection_rate": float(group["exact_oracle_selected"].map(safe_bool).mean()),
                "selected_family_equals_oracle_family_rate": float(group["selected_family_equals_oracle_family"].map(safe_bool).mean()),
                "selected_beats_m0_rate": float(group["selected_beats_m0"].map(safe_bool).mean()),
                "selected_beats_m2_rate": float(group["selected_beats_m2"].map(safe_bool).mean()),
                "failure_rate": float(group["failure"].map(safe_bool).mean()),
                "median_position_error_ci95_low_m": pos_ci[0],
                "median_position_error_ci95_high_m": pos_ci[1],
                "median_oracle_gap_ci95_low_m": gap_ci[0],
                "median_oracle_gap_ci95_high_m": gap_ci[1],
                "bootstrap_replicates": 1000,
                "bootstrap_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def make_baseline_comparisons(primary_a: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    baselines = [
        "M0_static_position",
        "M2_ctd_full",
        "M3_ctd_no_drift",
        "M4_ctd_no_bias",
        "M7_robust_ctd_full",
        "M13_robust_ctd_full_plus_gir_refine",
    ]
    primary_candidates = candidates.loc[candidates["evaluation_scope"] == "primary"].copy()
    rows: list[dict[str, Any]] = []
    for selected in primary_a.to_dict(orient="records"):
        pool = primary_candidates.loc[primary_candidates["batch_id"] == selected["batch_id"]]
        for baseline in baselines:
            base = pool.loc[pool["candidate_model"].astype(str) == baseline]
            base_error = safe_float(base["final_position_error_m"].iloc[0]) if len(base) else np.nan
            selected_error = safe_float(selected["final_position_error_m"])
            rows.append(
                {
                    "holdout_id": selected["holdout_id"],
                    "holdout_family": selected["holdout_family"],
                    "selector_mode": "observable_only",
                    "selected_model": selected["selected_model"],
                    "baseline_model": baseline,
                    "selected_error_m": selected_error,
                    "baseline_error_m": base_error,
                    "paired_difference_selected_minus_baseline_m": (
                        selected_error - base_error if np.isfinite(selected_error) and np.isfinite(base_error) else np.nan
                    ),
                    "selected_beats_or_ties_baseline": bool(
                        np.isfinite(selected_error) and np.isfinite(base_error) and selected_error <= base_error
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_rank_biserial(differences: np.ndarray) -> float:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs) & (diffs != 0.0)]
    if diffs.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs), method="average")
    positive = float(ranks[diffs > 0].sum())
    negative = float(ranks[diffs < 0].sum())
    return (positive - negative) / (positive + negative)


def statistical_tests(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(family), group.copy()) for family, group in comparisons.groupby("holdout_family", sort=True)
    ]
    groups.append(("ALL_FAMILIES", comparisons.copy()))
    for family, family_frame in groups:
        for baseline, group in family_frame.groupby("baseline_model", sort=True):
            diff = pd.to_numeric(group["paired_difference_selected_minus_baseline_m"], errors="coerce").to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            if diff.size == 0:
                stat = p_value = np.nan
            elif np.all(diff == 0.0):
                stat, p_value = 0.0, 1.0
            else:
                try:
                    test = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
                    stat, p_value = float(test.statistic), float(test.pvalue)
                except ValueError:
                    stat = p_value = np.nan
            rows.append(
                {
                    "holdout_family": family,
                    "selector_mode": "observable_only",
                    "baseline_model": baseline,
                    "effective_sample_size": int(diff.size),
                    "paired_median_difference_selected_minus_baseline_m": float(np.median(diff)) if diff.size else np.nan,
                    "wilcoxon_signed_rank_statistic": stat,
                    "wilcoxon_two_sided_p_value": p_value,
                    "paired_rank_biserial_effect_size": paired_rank_biserial(diff),
                    "effect_size_sign_convention": "positive means selected error tends to exceed baseline; negative favors selected",
                    "engineering_value_inferred_from_p_value": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    overall_started = time.perf_counter()
    protocol = read_protocol()
    verify_frozen_hashes("PAPER_EXP01_FROZEN_HASHES_PRE_RUN.csv")
    rr = load_release_runtime()
    from leo_positioning.coordinates import enu_axes
    from leo_positioning.data import load_iridium_csv, normalize_observations
    from leo_positioning.qatar_error_model import sample_heavy_tail_outliers
    from leo_positioning.trajectory_models import FULL_CTD_CONFIG, pack_state, predict_ctd

    expected_models = list(protocol["candidate_models"])
    if expected_models != list(rr.V4_CANDIDATE_MODEL_LIST):
        raise RuntimeError("protocol candidate list differs from immutable release list")
    if protocol["release"]["candidate_pool_protocol"] != "FULL_POOL_M0_M14_V1":
        raise RuntimeError("protocol candidate-pool identifier is not FULL_POOL_M0_M14_V1")

    base_df = load_iridium_csv(BASE_DATA)
    base_obs = normalize_observations(base_df)
    records = list(protocol["scenario_records"])
    if len(records) != 60:
        raise RuntimeError(f"protocol contains {len(records)} primary records, expected 60")

    batches: list[tuple[dict[str, Any], str, str, str]] = []
    for record in records:
        batches.append((record, "east_10km", "zero_velocity", "B0_none"))
        if int(record["realization_index"]) in {0, 1}:
            batches.append((record, "east_100km", "zero_velocity", "B0_none"))
            batches.append((record, "east_10km", "half_truth_velocity", "B0_none"))
    if len(batches) != 84:
        raise RuntimeError(f"execution plan contains {len(batches)} batches, expected 84")

    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    scenario_audits: list[dict[str, Any]] = []
    for batch_index, (record, position_init, velocity_init, beta_profile) in enumerate(batches, start=1):
        obs, observation_audit = build_holdout_observation(
            record,
            base_obs,
            enu_axes=enu_axes,
            pack_state=pack_state,
            predict_ctd=predict_ctd,
            full_ctd_config=FULL_CTD_CONFIG,
            sample_heavy_tail_outliers=sample_heavy_tail_outliers,
        )
        metadata_mode_a = {
            "dataset_type": "synthetic",
            "outlier_ratio": 0.0,
            "dropout_ratio": 0.0,
            "burst_outlier": False,
        }
        batch_started = time.perf_counter()
        frame = rr.run_actual_candidates(
            "synthetic",
            str(record["selector_scenario_key"]),
            obs,
            metadata_mode_a,
            candidate_models=expected_models,
            position_init_label=position_init,
            velocity_init_label=velocity_init,
            beta_prior_profile=beta_profile,
        )
        batch_elapsed = time.perf_counter() - batch_started
        actual_models = frame["candidate_model"].astype(str).tolist()
        if len(frame) != 15 or set(actual_models) != set(expected_models) or len(set(actual_models)) != 15:
            raise RuntimeError(f"{record['holdout_id']} did not execute the complete unique M0--M14 pool")
        if set(frame["execution_mode"].astype(str)) != {"solver_rerun"}:
            raise RuntimeError(f"{record['holdout_id']} used a non-rerun execution mode")
        if set(frame["candidate_pool_protocol"].astype(str)) != {"FULL_POOL_M0_M14_V1"}:
            raise RuntimeError(f"{record['holdout_id']} used a non-full candidate pool")

        batch_id = f"{record['holdout_id']}__{position_init}__{velocity_init}__{beta_profile}"
        evaluation_scope = (
            "primary"
            if (position_init, velocity_init, beta_profile) == ("east_10km", "zero_velocity", "B0_none")
            else "initialization_sensitivity"
        )

        # Select from the untouched release-adapter table.  Hold-out family,
        # injection labels, oracle quantities, and batch bookkeeping are added
        # only after both selector calls have returned.
        for mode in ("observable_only", "label_assisted_ablation"):
            selected, diagnostics, _selection_frame = select_one_mode(
                rr,
                frame,
                mode=mode,
                record=record,
                audit=observation_audit,
            )
            selected["batch_id"] = batch_id
            selected["actual_batch_wall_seconds"] = batch_elapsed
            selected_rows.append(selected)
            diagnostic_rows.append(diagnostics)

        frame["batch_id"] = batch_id
        frame["holdout_id"] = record["holdout_id"]
        frame["holdout_family"] = record["holdout_family"]
        frame["realization_index"] = int(record["realization_index"])
        frame["evaluation_scope"] = evaluation_scope
        frame["actual_batch_wall_seconds"] = batch_elapsed
        frame["observable_only_outlier_evidence"] = frame["outlier_evidence"].map(safe_bool)
        frame["observable_only_outlier_evidence_reason"] = frame["outlier_evidence_reason"]
        frame["label_assisted_outlier_evidence"] = frame["outlier_evidence"].map(safe_bool) | bool(
            observation_audit["injected_outlier_count"] > 0
        )
        frame["label_assisted_burst_evidence"] = bool(observation_audit["burst_outlier_count"] > 0)
        frame["post_selection_injected_outlier_count"] = observation_audit["injected_outlier_count"]
        frame["post_selection_injected_outlier_fraction"] = observation_audit["injected_outlier_fraction_actual"]
        frame["post_selection_confidence_low_ratio"] = observation_audit["confidence_low_ratio_actual"]
        frame["post_selection_holdout_speed_mps"] = float(record["speed_mps"])
        frame["post_selection_holdout_b0_mps"] = float(record["b0_mps"])
        frame["post_selection_holdout_bdot_mps2"] = float(record["bdot_mps2"])
        candidate_frames.append(frame)

        scenario_audits.append(
            {
                "batch_id": batch_id,
                "holdout_id": record["holdout_id"],
                "holdout_family": record["holdout_family"],
                "realization_index": int(record["realization_index"]),
                "evaluation_scope": evaluation_scope,
                "position_init_label": position_init,
                "velocity_init_label": velocity_init,
                "beta_prior_profile": beta_profile,
                **observation_audit,
                "actual_batch_wall_seconds": batch_elapsed,
            }
        )
        if batch_index == 1 or batch_index % 5 == 0 or batch_index == len(batches):
            print(
                f"completed solver batch {batch_index}/{len(batches)}: {batch_id} ({batch_elapsed:.2f}s)",
                flush=True,
            )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    scenario_audit = pd.DataFrame(scenario_audits)
    if len(candidates) != 84 * 15:
        raise RuntimeError(f"candidate row count {len(candidates)} != 1260")
    if len(selected) != 84 * 2:
        raise RuntimeError(f"selected row count {len(selected)} != 168")

    candidates.to_csv(OUT / "PAPER_EXP01_ACTUAL_CANDIDATES.csv", index=False)
    selected.to_csv(OUT / "PAPER_EXP01_ACTUAL_SELECTED.csv", index=False)
    diagnostics.to_csv(OUT / "PAPER_EXP01_SELECTOR_DIAGNOSTICS.csv", index=False)
    scenario_audit.to_csv(OUT / "PAPER_EXP01_EXECUTION_AUDIT.csv", index=False)

    primary = selected.loc[selected["evaluation_scope"] == "primary"].copy()
    if len(primary) != 120:
        raise RuntimeError(f"primary selected-mode rows {len(primary)} != 120")
    primary_a = primary.loc[primary["selector_mode"] == "observable_only"].copy()
    primary_b = primary.loc[primary["selector_mode"] == "label_assisted_ablation"].copy()
    if len(primary_a) != 60 or len(primary_b) != 60:
        raise RuntimeError("primary Mode A/Mode B row count mismatch")

    aggregate = aggregate_selected(primary, int(protocol["master_seed"]))
    aggregate.to_csv(OUT / "PAPER_EXP01_FAMILY_AGGREGATE.csv", index=False)

    sensitivity = selected.loc[
        (selected["selector_mode"] == "observable_only")
        & (selected["realization_index"].astype(int).isin([0, 1]))
    ].copy()
    if len(sensitivity) != 36:
        raise RuntimeError(f"initialization sensitivity row count {len(sensitivity)} != 36")
    primary_lookup = primary_a.set_index("holdout_id")
    sensitivity["primary_selected_model"] = sensitivity["holdout_id"].map(primary_lookup["selected_model"])
    sensitivity["primary_position_error_m"] = sensitivity["holdout_id"].map(primary_lookup["final_position_error_m"])
    sensitivity["selected_model_changed_vs_primary"] = sensitivity["selected_model"] != sensitivity["primary_selected_model"]
    sensitivity["position_error_difference_vs_primary_m"] = (
        pd.to_numeric(sensitivity["final_position_error_m"], errors="coerce")
        - pd.to_numeric(sensitivity["primary_position_error_m"], errors="coerce")
    )
    sensitivity.to_csv(OUT / "PAPER_EXP01_INITIALIZATION_SENSITIVITY.csv", index=False)

    compare_cols = [
        "holdout_id",
        "holdout_family",
        "realization_index",
        "selected_model",
        "selected_model_family",
        "final_position_error_m",
        "selected_minus_oracle_error_m",
        "quality_pass",
    ]
    mode_compare = primary_a[compare_cols].merge(
        primary_b[compare_cols],
        on=["holdout_id", "holdout_family", "realization_index"],
        suffixes=("_mode_a", "_mode_b"),
        validate="one_to_one",
    )
    mode_compare["selected_model_changed"] = mode_compare["selected_model_mode_a"] != mode_compare["selected_model_mode_b"]
    mode_compare["position_error_difference_b_minus_a_m"] = (
        pd.to_numeric(mode_compare["final_position_error_m_mode_b"], errors="coerce")
        - pd.to_numeric(mode_compare["final_position_error_m_mode_a"], errors="coerce")
    )
    mode_compare["mode_b_claim_restriction"] = "synthetic-label ablation only; not an online receiver capability"
    mode_compare.to_csv(OUT / "PAPER_EXP01_OBSERVABLE_ONLY_VS_LABEL_ASSISTED.csv", index=False)

    comparisons = make_baseline_comparisons(primary_a, candidates)
    if len(comparisons) != 360:
        raise RuntimeError(f"baseline comparison row count {len(comparisons)} != 360")
    comparisons.to_csv(OUT / "PAPER_EXP01_BASELINE_COMPARISON.csv", index=False)
    tests = statistical_tests(comparisons)
    tests.to_csv(OUT / "PAPER_EXP01_STATISTICAL_TESTS.csv", index=False)

    after = verify_frozen_hashes("PAPER_EXP01_FROZEN_HASHES_AFTER.csv")
    unchanged = bool(after["unchanged"].all())
    run_log = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "master_seed": int(protocol["master_seed"]),
        "primary_runs": int(len(primary_a)),
        "additional_sensitivity_solver_batches": 24,
        "total_solver_batches": 84,
        "candidate_rows": int(len(candidates)),
        "candidates_per_batch": 15,
        "selection_mode_rows": int(len(selected)),
        "execution_mode": "solver_rerun",
        "cached_replay_count": 0,
        "candidate_pool_protocol": "FULL_POOL_M0_M14_V1",
        "primary_initialization": "east_10km + zero_velocity + B0_none",
        "frozen_source_unchanged": unchanged,
        "frozen_hash_record_count": int(len(after)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "total_wall_seconds": time.perf_counter() - overall_started,
        "outputs_written_before_reporting": [
            "PAPER_EXP01_ACTUAL_CANDIDATES.csv",
            "PAPER_EXP01_ACTUAL_SELECTED.csv",
            "PAPER_EXP01_FAMILY_AGGREGATE.csv",
            "PAPER_EXP01_INITIALIZATION_SENSITIVITY.csv",
            "PAPER_EXP01_OBSERVABLE_ONLY_VS_LABEL_ASSISTED.csv",
            "PAPER_EXP01_BASELINE_COMPARISON.csv",
            "PAPER_EXP01_STATISTICAL_TESTS.csv",
        ],
    }
    (OUT / "PAPER_EXP01_RUN_LOG.json").write_text(
        json.dumps(run_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_log, indent=2), flush=True)


if __name__ == "__main__":
    main()
