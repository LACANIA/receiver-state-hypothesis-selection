"""MA-BGTR-v3 with GIR-TR branch and influence-aware validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .gir_tr_solver import initial_full_state, solve_gir_tr
from .influence_diagnostics import diagnostics_dict, summarize_downweighted_satellites, validation_residual_metrics
from .ma_bgtr_v2_solver import TrainFit, evaluate_fit_residual, fit_candidate_on_block
from .model_candidates import CANDIDATE_MODEL_LIST, run_candidate_model
from .model_evidence import EvidenceConfig, apply_evidence_gates, group_evidence
from .model_scoring import V3ScoringConfig, compute_score_v3
from .quality import physical_plausibility, quality_gate
from .trajectory_models import FULL_CTD_CONFIG, residuals_and_jacobian_ctd, unpack_state, weighted_normal_condition
from .validation_split import blocked_train_validation


V3_CANDIDATE_MODEL_LIST = CANDIDATE_MODEL_LIST


@dataclass
class MaBgtrV3Result:
    selected_model_v3: str
    selected_score_v3: float
    candidate_results: list[dict[str, Any]]
    influence_rows: list[dict[str, Any]]
    selected_row: dict[str, Any]
    selection_reason_v3: str
    selected_low_quality: bool
    split_summary: dict[str, Any]


def _gir_train_fit(model: str, obs: dict[str, Any], p0_init: np.ndarray, v_init: np.ndarray, dataset_type: str) -> TrainFit:
    try:
        mode = "mad" if model == "M11_gir_tr_mad_scale" else "fixed"
        t0_s = float(obs.get("t0_s", np.min(obs["time_s"])))
        result = solve_gir_tr(
            initial_full_state(p0_init, v_init),
            obs["time_s"],
            obs["sat_pos_m"],
            obs["sat_vel_mps"],
            obs["meas_mps"],
            t0_s,
            scale_mode=mode,
            dataset_type=dataset_type,
        )
        _p, v, beta0, bdot = unpack_state(result.state, FULL_CTD_CONFIG)
        return TrainFit(
            success=bool(result.numerical_success),
            converged=bool(result.converged),
            model=model,
            family="ctd",
            residual_train=result.residual,
            state=result.state.copy(),
            config=FULL_CTD_CONFIG,
            with_bias=False,
            robust=True,
            projection_mode="none",
            beta=np.array([beta0, bdot], dtype=float),
            condition_number=float(result.condition_number),
            estimated_speed_mps=float(np.linalg.norm(v)),
            beta0_estimated_mps=float(beta0),
            beta_dot_estimated_mps2=float(bdot),
            failure_reason=result.failure_reason,
        )
    except Exception as exc:  # noqa: BLE001
        return TrainFit(
            success=False,
            converged=False,
            model=model,
            family="ctd",
            residual_train=np.array([], dtype=float),
            state=np.array([], dtype=float),
            config=FULL_CTD_CONFIG,
            with_bias=False,
            robust=True,
            projection_mode="none",
            beta=np.array([np.nan, np.nan], dtype=float),
            condition_number=np.nan,
            estimated_speed_mps=np.nan,
            beta0_estimated_mps=np.nan,
            beta_dot_estimated_mps2=np.nan,
            failure_reason=f"gir_train_fit_exception: {exc}",
        )


def fit_candidate_v3(model: str, obs: dict[str, Any], p0_init: np.ndarray, v_init: np.ndarray, beta_prior_profile: str, dataset_type: str) -> TrainFit:
    if model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}:
        return _gir_train_fit(model, obs, p0_init, v_init, dataset_type)
    return fit_candidate_on_block(model, obs, p0_init, v_init, beta_prior_profile)


def _add_validation_metrics(row: dict[str, Any], train_fit: TrainFit, train_obs: dict[str, Any], val_obs: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    train_residual = evaluate_fit_residual(train_fit, train_obs)
    val_residual = evaluate_fit_residual(train_fit, val_obs)
    metrics = validation_residual_metrics(val_residual, trim_fraction=0.10, c_scale_mps=2.0)
    out.update(metrics)
    out["train_residual_rmse_mps"] = validation_residual_metrics(train_residual)["raw_validation_rmse_mps"]
    out["full_residual_rmse_mps"] = float(out.get("residual_rmse_mps", np.nan))
    if not train_fit.success:
        out["numerical_success"] = False
        out["quality_pass"] = False
        out["failure_reason"] = ";".join(r for r in [str(out.get("failure_reason", "")), train_fit.failure_reason] if r)
    physical, physical_reason = physical_plausibility(
        str(out.get("dataset_type", "synthetic")),
        float(out.get("estimated_speed_mps", np.nan)),
        float(out.get("beta0_estimated_mps", np.nan)),
        float(out.get("beta_dot_estimated_mps2", np.nan)),
    )
    quality, quality_reason = quality_gate(str(out.get("dataset_type", "synthetic")), bool(out.get("numerical_success", False)), physical, None)
    out["physical_plausible"] = bool(physical)
    out["quality_pass"] = bool(quality)
    out["converged"] = bool(out.get("converged", False)) and bool(train_fit.converged)
    out["failure_reason"] = ";".join(
        dict.fromkeys([r for r in [str(out.get("failure_reason", "")), physical_reason, quality_reason] if r])
    )
    out.setdefault("information_retention_ratio", np.nan)
    out.setdefault("floor_active_fraction", np.nan)
    out.setdefault("hard_outlier_fraction", np.nan)
    out.setdefault("effective_weight_entropy", np.nan)
    return out


def _add_ctd_information_retention(row: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("candidate_model")) not in {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full"}:
        return row
    try:
        from .trajectory_models import FULL_CTD_CONFIG, NO_BDOT_CONFIG, NO_BIAS_CONFIG
        config = {
            "M2_ctd_full": FULL_CTD_CONFIG,
            "M3_ctd_no_drift": NO_BDOT_CONFIG,
            "M4_ctd_no_bias": NO_BIAS_CONFIG,
            "M7_robust_ctd_full": FULL_CTD_CONFIG,
        }[str(row.get("candidate_model"))]
        state = np.array([], dtype=float)
        # Information retention for non-GIR models is treated as full retention.
        row["information_retention_ratio"] = 1.0
        row["floor_active_fraction"] = np.nan
        row["hard_outlier_fraction"] = np.nan
        row["effective_weight_entropy"] = np.nan
    except Exception:
        row["information_retention_ratio"] = 1.0
    return row


def _robust_gain_fill(rows: list[dict[str, Any]]) -> None:
    nonrobust = [r for r in rows if str(r.get("candidate_model")) in {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias"} and bool(r.get("numerical_success", False))]
    best_trim = min((float(r.get("trimmed_validation_rmse_mps", np.inf)) for r in nonrobust), default=np.nan)
    best_inlier = min((float(r.get("inlier_validation_rmse_mps", np.inf)) for r in nonrobust), default=np.nan)
    for row in rows:
        trim = float(row.get("trimmed_validation_rmse_mps", np.nan))
        inlier = float(row.get("inlier_validation_rmse_mps", np.nan))
        row["robust_trimmed_gain"] = (best_trim - trim) / best_trim if np.isfinite(best_trim) and best_trim > 0 and np.isfinite(trim) else 0.0
        row["robust_inlier_gain"] = (best_inlier - inlier) / best_inlier if np.isfinite(best_inlier) and best_inlier > 0 and np.isfinite(inlier) else 0.0


def _influence_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "robust_weight_min",
        "robust_weight_median",
        "robust_weight_max",
        "geometry_floor_min",
        "geometry_floor_median",
        "geometry_floor_max",
        "floor_active_fraction",
        "hard_outlier_fraction",
        "information_retention_ratio",
        "leverage_max",
        "leverage_median",
        "weight_geometry_correlation",
    ]
    out = {
        "dataset_type": row.get("dataset_type"),
        "scenario": row.get("scenario"),
        "candidate_model": row.get("candidate_model"),
        "position_init_label": row.get("position_init_label"),
        "velocity_init_label": row.get("velocity_init_label"),
        "beta_prior_profile": row.get("beta_prior_profile"),
        "effective_weight_entropy": row.get("effective_weight_entropy", np.nan),
        "top_downweighted_satellites": row.get("top_downweighted_satellites", ""),
        "notes": "GIR-TR diagnostics" if str(row.get("candidate_model", "")).startswith("M1") is False else "",
    }
    for key in keys:
        out[key] = row.get(key, np.nan)
    return out


def run_ma_bgtr_v3(
    observations: dict[str, Any],
    initial_guess: dict[str, Any],
    scoring_config: V3ScoringConfig | None = None,
    evidence_config: EvidenceConfig | None = None,
    candidate_models: list[str] | None = None,
    precomputed_full_rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]] | None = None,
) -> MaBgtrV3Result:
    models = candidate_models or V3_CANDIDATE_MODEL_LIST
    train_obs, val_obs, split_summary = blocked_train_validation(observations, train_fraction=0.70)
    dataset_type = str(initial_guess["dataset_type"])
    scenario = str(initial_guess["scenario"])
    pos_label = str(initial_guess["position_init_label"])
    vel_label = str(initial_guess["velocity_init_label"])
    profile = str(initial_guess["beta_prior_profile"])
    p0 = np.asarray(initial_guess["p0_init"], dtype=float)
    v0 = np.asarray(initial_guess["v_init"], dtype=float)

    rows: list[dict[str, Any]] = []
    for model in models:
        key = (dataset_type, scenario, model, pos_label, vel_label, profile)
        if precomputed_full_rows is not None and key in precomputed_full_rows:
            full_row = dict(precomputed_full_rows[key])
        else:
            full_row = run_candidate_model(model, dataset_type, scenario, observations, pos_label, p0, vel_label, v0, profile)
        train_fit = fit_candidate_v3(model, train_obs, p0, v0, profile, dataset_type)
        row = _add_validation_metrics(full_row, train_fit, train_obs, val_obs)
        row = _add_ctd_information_retention(row, observations)
        rows.append(row)

    _robust_gain_fill(rows)
    evidence = group_evidence(
        [
            {
                **row,
                "validation_residual_rmse_mps": row.get("raw_validation_rmse_mps", np.nan),
            }
            for row in rows
        ],
        dataset_type,
        evidence_config,
    )
    gated_rows = []
    for row in rows:
        proxy = dict(row)
        proxy["validation_residual_rmse_mps"] = proxy.get("raw_validation_rmse_mps", np.nan)
        gated_rows.append(apply_evidence_gates(proxy, evidence, evidence_config))

    scored: list[dict[str, Any]] = []
    for row in gated_rows:
        merged = dict(row)
        # Robust evidence can unblock GIR-TR when trimmed/inlier validation improves.
        if str(merged.get("candidate_model")) in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}:
            if float(merged.get("tail_ratio", 0.0)) > 0.05 or float(merged.get("mad_ratio", 0.0)) > 2.0:
                if float(merged.get("robust_trimmed_gain", 0.0)) > 0.02 or float(merged.get("robust_inlier_gain", 0.0)) > 0.02:
                    removable = {"robust_evidence_absent", "bias_gain_below_threshold", "drift_gain_below_threshold"}
                    reasons = [r for r in str(merged.get("gate_block_reason", "")).split(";") if r and r not in removable]
                    merged["gate_block_reason"] = ";".join(reasons)
                    merged["robust_gate_pass"] = True
                    merged["bias_gate_pass"] = True
                    merged["drift_gate_pass"] = True
        merged.update(compute_score_v3(merged, int(split_summary["validation_count"]), scoring_config))
        scored.append(merged)

    selectable = [r for r in scored if bool(r.get("selectable_v3", False))]
    high_quality = [r for r in selectable if bool(r.get("quality_pass", False))]
    if high_quality:
        selected = min(high_quality, key=lambda r: float(r["total_score_v3"]))
        low_quality = False
        reason = "lowest_gir_v3_score_among_quality_pass_candidates"
    elif selectable:
        selected = min(selectable, key=lambda r: float(r["total_score_v3"]))
        low_quality = True
        reason = "no_quality_pass_candidate; lowest_gir_v3_score_among_selectable"
    else:
        finite = [r for r in scored if np.isfinite(float(r.get("total_score_v3", np.inf)))]
        selected = min(finite or scored, key=lambda r: float(r.get("total_score_v3", 1e9)))
        low_quality = True
        reason = "no_selectable_candidate; reported_lowest_gir_v3_score"

    influence_rows = [_influence_row(r) for r in scored if str(r.get("candidate_model")) in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}]
    return MaBgtrV3Result(
        selected_model_v3=str(selected["candidate_model"]),
        selected_score_v3=float(selected["total_score_v3"]),
        candidate_results=scored,
        influence_rows=influence_rows,
        selected_row=selected,
        selection_reason_v3=reason,
        selected_low_quality=bool(low_quality),
        split_summary=split_summary,
    )
