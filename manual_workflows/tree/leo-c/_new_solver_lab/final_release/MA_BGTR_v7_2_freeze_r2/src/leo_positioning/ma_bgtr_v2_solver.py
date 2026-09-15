"""MA-BGTR-v2: validation and evidence-gated model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .be_gtr_solver import h_and_jacobian_x, solve_be_gtr
from .metrics import residual_metrics
from .model_candidates import CANDIDATE_MODEL_LIST, MODEL_DOF, robust_cost_per_dof, run_candidate_model
from .model_evidence import EvidenceConfig, apply_evidence_gates, group_evidence
from .model_scoring import V2ScoringConfig, compute_score_v2
from .models import normal_matrix_condition, residuals_and_jacobian
from .projection_diagnostics import compute_projection_diagnostics, diagnostics_to_dict
from .quality import physical_plausibility, quality_gate
from .solvers import solve_lm
from .trajectory_models import (
    FULL_CTD_CONFIG,
    NO_BDOT_CONFIG,
    NO_BIAS_CONFIG,
    TrajectoryModelConfig,
    pack_state,
    residuals_and_jacobian_ctd,
    unpack_state,
    weighted_normal_condition,
)
from .trajectory_solvers import solve_ctd_lm
from .validation_split import blocked_train_validation
from .variable_projection import beta_prior_profiles


@dataclass
class TrainFit:
    success: bool
    converged: bool
    model: str
    family: str
    residual_train: np.ndarray
    state: np.ndarray
    config: TrajectoryModelConfig | None
    with_bias: bool
    robust: bool
    projection_mode: str
    beta: np.ndarray
    condition_number: float
    estimated_speed_mps: float
    beta0_estimated_mps: float
    beta_dot_estimated_mps2: float
    failure_reason: str


@dataclass
class MaBgtrV2Result:
    selected_model_v2: str
    selected_score_v2: float
    candidate_results: list[dict[str, Any]]
    evidence_gate_rows: list[dict[str, Any]]
    selected_row: dict[str, Any]
    selection_reason_v2: str
    selected_low_quality: bool
    split_summary: dict[str, Any]


def _safe_rmse(residual: np.ndarray) -> float:
    rmse, _median_abs = residual_metrics(residual)
    return float(rmse)


def _tail_stats(residual: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(residual, dtype=float)
    if residual.size == 0 or not np.all(np.isfinite(residual)):
        return np.nan, np.nan
    med = float(np.median(residual))
    robust_sigma = 1.4826 * float(np.median(np.abs(residual - med))) + 1e-9
    standardized = np.abs((residual - med) / robust_sigma)
    tail_ratio = float(np.mean(standardized > 3.0))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mad_ratio = float(rmse / robust_sigma)
    return tail_ratio, mad_ratio


def _ctd_config_for_model(model: str) -> tuple[TrajectoryModelConfig, bool]:
    if model == "M2_ctd_full":
        return FULL_CTD_CONFIG, False
    if model == "M3_ctd_no_drift":
        return NO_BDOT_CONFIG, False
    if model == "M4_ctd_no_bias":
        return NO_BIAS_CONFIG, False
    if model == "M7_robust_ctd_full":
        return FULL_CTD_CONFIG, True
    raise ValueError(model)


def _be_mode_for_model(model: str) -> tuple[str, bool]:
    if model in {"M6_be_full", "M9_robust_be_full"}:
        return "full", model == "M9_robust_be_full"
    if model in {"M5_be_b0_only", "M8_robust_be_b0_only"}:
        return "b0_only", model == "M8_robust_be_b0_only"
    raise ValueError(model)


def _failed_fit(model: str, family: str, reason: str) -> TrainFit:
    return TrainFit(
        success=False,
        converged=False,
        model=model,
        family=family,
        residual_train=np.array([], dtype=float),
        state=np.array([], dtype=float),
        config=None,
        with_bias=False,
        robust=False,
        projection_mode="none",
        beta=np.array([np.nan, np.nan], dtype=float),
        condition_number=np.nan,
        estimated_speed_mps=np.nan,
        beta0_estimated_mps=np.nan,
        beta_dot_estimated_mps2=np.nan,
        failure_reason=reason,
    )


def fit_candidate_on_block(
    model: str,
    obs: dict[str, Any],
    p0_init: np.ndarray,
    v_init: np.ndarray,
    beta_prior_profile: str,
) -> TrainFit:
    """Fit one candidate on a train block and keep enough state for validation."""
    try:
        if model in {"M0_static_position", "M1_static_position_bias"}:
            with_bias = model == "M1_static_position_bias"
            initial = np.r_[p0_init, 0.0] if with_bias else np.asarray(p0_init, dtype=float)
            result = solve_lm(initial, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=with_bias, robust=False)
            residual, jac, _pred = residuals_and_jacobian(result.state, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=with_bias)
            beta0 = float(result.state[3]) if with_bias else np.nan
            return TrainFit(
                success=bool(result.success),
                converged=bool(result.converged),
                model=model,
                family="static",
                residual_train=residual,
                state=result.state.copy(),
                config=None,
                with_bias=with_bias,
                robust=False,
                projection_mode="none",
                beta=np.array([beta0, np.nan], dtype=float),
                condition_number=normal_matrix_condition(jac),
                estimated_speed_mps=0.0,
                beta0_estimated_mps=beta0,
                beta_dot_estimated_mps2=np.nan,
                failure_reason=result.failure_reason,
            )

        if model in {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full"}:
            config, robust = _ctd_config_for_model(model)
            t0_s = float(obs.get("t0_s", np.min(obs["time_s"])))
            theta0 = pack_state(p0_init, v_init, 0.0, 0.0, config)
            result = solve_ctd_lm(
                theta0,
                obs["time_s"],
                obs["sat_pos_m"],
                obs["sat_vel_mps"],
                obs["meas_mps"],
                t0_s,
                config,
                robust=robust,
                max_iter=80,
            )
            residual, jac, _pred = residuals_and_jacobian_ctd(
                result.state, obs["time_s"], obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], t0_s, config
            )
            _p, v, beta0, bdot = unpack_state(result.state, config)
            return TrainFit(
                success=bool(result.success),
                converged=bool(result.converged),
                model=model,
                family="ctd",
                residual_train=residual,
                state=result.state.copy(),
                config=config,
                with_bias=False,
                robust=robust,
                projection_mode="none",
                beta=np.array([beta0, bdot], dtype=float),
                condition_number=weighted_normal_condition(jac, residual, robust),
                estimated_speed_mps=float(np.linalg.norm(v)),
                beta0_estimated_mps=float(beta0) if np.isfinite(beta0) else np.nan,
                beta_dot_estimated_mps2=float(bdot) if np.isfinite(bdot) else np.nan,
                failure_reason=result.failure_reason,
            )

        if model in {"M5_be_b0_only", "M6_be_full", "M8_robust_be_b0_only", "M9_robust_be_full"}:
            projection_mode, robust = _be_mode_for_model(model)
            t0_s = float(obs.get("t0_s", np.min(obs["time_s"])))
            x0 = np.r_[p0_init, v_init]
            profile = beta_prior_profiles()[beta_prior_profile]
            result = solve_be_gtr(
                x0,
                obs["time_s"],
                obs["sat_pos_m"],
                obs["sat_vel_mps"],
                obs["meas_mps"],
                t0_s,
                profile,
                robust=robust,
                max_iter_per_scale=12,
                projection_mode=projection_mode,
                include_prior_cost=True,
            )
            return TrainFit(
                success=bool(result.numerical_success),
                converged=bool(result.converged),
                model=model,
                family="be",
                residual_train=np.asarray(result.residual, dtype=float),
                state=result.state_x.copy(),
                config=None,
                with_bias=False,
                robust=robust,
                projection_mode=projection_mode,
                beta=result.beta.copy(),
                condition_number=float(result.condition_number),
                estimated_speed_mps=float(np.linalg.norm(result.state_x[3:6])),
                beta0_estimated_mps=float(result.beta[0]),
                beta_dot_estimated_mps2=float(result.beta[1]),
                failure_reason=result.failure_reason,
            )

    except Exception as exc:  # noqa: BLE001
        return _failed_fit(model, "unknown", f"train_fit_exception: {exc}")

    return _failed_fit(model, "unknown", f"unknown_model: {model}")


def evaluate_fit_residual(fit: TrainFit, obs: dict[str, Any]) -> np.ndarray:
    if not fit.success:
        return np.full(len(obs["meas_mps"]), np.nan)
    if fit.family == "static":
        residual, _jac, _pred = residuals_and_jacobian(
            fit.state, obs["sat_pos_m"], obs["sat_vel_mps"], obs["meas_mps"], with_bias=fit.with_bias
        )
        return residual
    if fit.family == "ctd" and fit.config is not None:
        residual, _jac, _pred = residuals_and_jacobian_ctd(
            fit.state,
            obs["time_s"],
            obs["sat_pos_m"],
            obs["sat_vel_mps"],
            obs["meas_mps"],
            float(obs.get("t0_s", np.min(obs["time_s"]))),
            fit.config,
        )
        return residual
    if fit.family == "be":
        h, _jac = h_and_jacobian_x(
            fit.state,
            obs["time_s"],
            obs["sat_pos_m"],
            obs["sat_vel_mps"],
            float(obs.get("t0_s", np.min(obs["time_s"]))),
        )
        tau = np.asarray(obs["time_s"], dtype=float) - float(obs.get("t0_s", np.min(obs["time_s"])))
        beta0 = float(fit.beta[0]) if len(fit.beta) > 0 else 0.0
        bdot = float(fit.beta[1]) if len(fit.beta) > 1 else 0.0
        return np.asarray(obs["meas_mps"], dtype=float) - (h + beta0 + bdot * tau)
    return np.full(len(obs["meas_mps"]), np.nan)


def _augment_with_validation(
    full_row: dict[str, Any],
    train_fit: TrainFit,
    train_obs: dict[str, Any],
    val_obs: dict[str, Any],
) -> dict[str, Any]:
    out = dict(full_row)
    train_residual = evaluate_fit_residual(train_fit, train_obs)
    val_residual = evaluate_fit_residual(train_fit, val_obs)
    train_rmse = _safe_rmse(train_residual)
    val_rmse = _safe_rmse(val_residual)
    tail_ratio, mad_ratio = _tail_stats(val_residual)
    full_rmse = float(out.get("residual_rmse_mps", np.nan))
    if not train_fit.success:
        out["numerical_success"] = False
        out["quality_pass"] = False
        out["failure_reason"] = ";".join(
            r for r in [str(out.get("failure_reason", "")), train_fit.failure_reason] if r
        )
    physical, physical_reason = physical_plausibility(
        str(out.get("dataset_type", "synthetic")),
        float(out.get("estimated_speed_mps", np.nan)),
        float(out.get("beta0_estimated_mps", np.nan)),
        float(out.get("beta_dot_estimated_mps2", np.nan)),
    )
    quality, quality_reason = quality_gate(str(out.get("dataset_type", "synthetic")), bool(out.get("numerical_success", False)), physical, None)
    reason = ";".join(
        dict.fromkeys(
            [
                r
                for r in [
                    str(out.get("failure_reason", "")),
                    physical_reason,
                    quality_reason,
                ]
                if r
            ]
        )
    )
    out.update(
        {
            "converged": bool(out.get("converged", False)) and bool(train_fit.converged),
            "physical_plausible": bool(physical),
            "quality_pass": bool(quality),
            "train_residual_rmse_mps": train_rmse,
            "validation_residual_rmse_mps": val_rmse,
            "full_residual_rmse_mps": full_rmse,
            "validation_robust_cost_per_dof": robust_cost_per_dof(val_residual, int(out.get("model_dof", MODEL_DOF.get(str(out.get("candidate_model")), 1)))),
            "train_validation_gap_mps": float(val_rmse - train_rmse) if np.isfinite(val_rmse) and np.isfinite(train_rmse) else np.nan,
            "tail_ratio": tail_ratio,
            "mad_ratio": mad_ratio,
            "speed_train_full_delta_mps": abs(float(out.get("estimated_speed_mps", np.nan)) - train_fit.estimated_speed_mps)
            if np.isfinite(train_fit.estimated_speed_mps)
            else np.nan,
            "beta0_train_full_delta_mps": abs(float(out.get("beta0_estimated_mps", np.nan)) - train_fit.beta0_estimated_mps)
            if np.isfinite(train_fit.beta0_estimated_mps) and np.isfinite(float(out.get("beta0_estimated_mps", np.nan)))
            else np.nan,
            "bdot_train_full_delta_mps2": abs(float(out.get("beta_dot_estimated_mps2", np.nan)) - train_fit.beta_dot_estimated_mps2)
            if np.isfinite(train_fit.beta_dot_estimated_mps2) and np.isfinite(float(out.get("beta_dot_estimated_mps2", np.nan)))
            else np.nan,
            "failure_reason": reason,
        }
    )
    return out


def _projection_fill(row: dict[str, Any], obs: dict[str, Any], p0_init: np.ndarray, v_init: np.ndarray) -> dict[str, Any]:
    model = str(row.get("candidate_model", ""))
    if model not in {"M5_be_b0_only", "M6_be_full", "M8_robust_be_b0_only", "M9_robust_be_full"}:
        row.setdefault("retention_trace_ratio", np.nan)
        row.setdefault("subspace_coherence_max", np.nan)
        row.setdefault("rank_loss", np.nan)
        return row
    basis = "full" if model in {"M6_be_full", "M9_robust_be_full"} else "b0_only"
    try:
        diag = diagnostics_to_dict(
            compute_projection_diagnostics(
                np.r_[p0_init, v_init],
                obs["time_s"],
                obs["sat_pos_m"],
                obs["sat_vel_mps"],
                obs["meas_mps"],
                float(obs.get("t0_s", np.min(obs["time_s"]))),
                np.ones(len(obs["meas_mps"]), dtype=float),
                basis,
                str(row.get("beta_prior_profile", "B0_none")),
            )
        )
        row.update(
            {
                "retention_trace_ratio": diag.get("retention_trace_ratio", np.nan),
                "subspace_coherence_max": diag.get("subspace_coherence_max", np.nan),
                "rank_loss": diag.get("rank_loss", np.nan),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["failure_reason"] = ";".join(r for r in [str(row.get("failure_reason", "")), f"projection_diag_exception: {exc}"] if r)
    return row


def run_ma_bgtr_v2(
    observations: dict[str, Any],
    initial_guess: dict[str, Any],
    scoring_config: V2ScoringConfig | None = None,
    evidence_config: EvidenceConfig | None = None,
    candidate_models: list[str] | None = None,
    precomputed_full_rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]] | None = None,
) -> MaBgtrV2Result:
    models = candidate_models or CANDIDATE_MODEL_LIST
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
        train_fit = fit_candidate_on_block(model, train_obs, p0, v0, profile)
        row = _augment_with_validation(full_row, train_fit, train_obs, val_obs)
        row = _projection_fill(row, observations, p0, v0)
        rows.append(row)

    evidence = group_evidence(rows, dataset_type, evidence_config)
    gated_rows = [apply_evidence_gates(row, evidence, evidence_config) for row in rows]
    scored: list[dict[str, Any]] = []
    for row in gated_rows:
        merged = dict(row)
        merged.update(compute_score_v2(merged, int(split_summary["validation_count"]), scoring_config))
        scored.append(merged)

    selectable = [r for r in scored if bool(r.get("selectable_v2", False))]
    high_quality = [r for r in selectable if bool(r.get("quality_pass", False))]
    if high_quality:
        selected = min(high_quality, key=lambda r: float(r["total_score_v2"]))
        low_quality = False
        reason = "lowest_validation_score_among_quality_pass_evidence_gated_candidates"
    elif selectable:
        selected = min(selectable, key=lambda r: float(r["total_score_v2"]))
        low_quality = True
        reason = "no_quality_pass_candidate; lowest_validation_score_among_selectable"
    else:
        finite = [r for r in scored if np.isfinite(float(r.get("total_score_v2", np.inf)))]
        selected = min(finite or scored, key=lambda r: float(r.get("total_score_v2", 1e9)))
        low_quality = True
        reason = "no_selectable_candidate; reported_lowest_score"

    gate_rows = [
        {
            "dataset_type": r.get("dataset_type"),
            "scenario": r.get("scenario"),
            "position_init_label": r.get("position_init_label"),
            "velocity_init_label": r.get("velocity_init_label"),
            "beta_prior_profile": r.get("beta_prior_profile"),
            "candidate_model": r.get("candidate_model"),
            "dynamic_gate_pass": r.get("dynamic_gate_pass", True),
            "bias_gate_pass": r.get("bias_gate_pass", True),
            "drift_gate_pass": r.get("drift_gate_pass", True),
            "robust_gate_pass": r.get("robust_gate_pass", True),
            "projection_gate_pass": r.get("projection_gate_pass", True),
            "real_static_guard_pass": r.get("real_static_guard_pass", True),
            "gate_block_reason": r.get("gate_block_reason", ""),
            "dynamic_gain": r.get("dynamic_gain", np.nan),
            "bias_gain": r.get("bias_gain", np.nan),
            "drift_gain": r.get("drift_gain", np.nan),
            "tail_ratio": r.get("tail_ratio", np.nan),
            "mad_ratio": r.get("mad_ratio", np.nan),
        }
        for r in scored
    ]

    return MaBgtrV2Result(
        selected_model_v2=str(selected["candidate_model"]),
        selected_score_v2=float(selected["total_score_v2"]),
        candidate_results=scored,
        evidence_gate_rows=gate_rows,
        selected_row=selected,
        selection_reason_v2=reason,
        selected_low_quality=bool(low_quality),
        split_summary=split_summary,
    )
