"""Model scoring for MA-BGTR candidate selection."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ScoringConfig:
    alpha_complexity: float = 0.5
    alpha_condition: float = 0.2
    alpha_physical: float = 2.0
    alpha_projection: float = 2.0
    alpha_convergence: float = 1.0
    eps: float = 1e-9
    synthetic_speed_limit_mps: float = 300.0
    real_speed_limit_mps: float = 20.0
    beta0_limit_mps: float = 20.0
    bdot_limit_mps2: float = 0.5
    real_bdot_limit_mps2: float = 0.1


def scoring_config_dict(config: ScoringConfig) -> dict[str, float]:
    return asdict(config)


def compute_score(row: dict, n_obs: int, config: ScoringConfig | None = None) -> dict[str, float | bool]:
    cfg = config or ScoringConfig()
    residual_rmse = float(row.get("residual_rmse_mps", np.nan))
    residual_score = float(np.log(max(residual_rmse, cfg.eps))) if np.isfinite(residual_rmse) else 1e6
    model_dof = float(row.get("model_dof", 0.0))
    complexity_penalty = cfg.alpha_complexity * model_dof * np.log(max(float(n_obs), 2.0)) / max(float(n_obs), 1.0)
    condition = float(row.get("condition_number", np.nan))
    condition_penalty = 0.0
    if np.isfinite(condition) and condition > 0.0:
        condition_penalty = cfg.alpha_condition * max(0.0, np.log10(condition) - 8.0)
    elif not np.isfinite(condition):
        condition_penalty = 10.0

    dataset_type = str(row.get("dataset_type", "synthetic"))
    speed = float(row.get("estimated_speed_mps", np.nan))
    beta0 = float(row.get("beta0_estimated_mps", np.nan))
    bdot = float(row.get("beta_dot_estimated_mps2", np.nan))
    physical_penalty = 0.0
    speed_limit = cfg.real_speed_limit_mps if dataset_type == "real" else cfg.synthetic_speed_limit_mps
    if not np.isfinite(speed) or speed > speed_limit:
        physical_penalty += cfg.alpha_physical * (1.0 + max(0.0, speed - speed_limit) / max(speed_limit, 1.0) if np.isfinite(speed) else 5.0)
    if np.isfinite(beta0) and abs(beta0) > cfg.beta0_limit_mps:
        physical_penalty += cfg.alpha_physical * (abs(beta0) / cfg.beta0_limit_mps - 1.0)
    bdot_limit = cfg.real_bdot_limit_mps2 if dataset_type == "real" else cfg.bdot_limit_mps2
    if np.isfinite(bdot) and abs(bdot) > bdot_limit:
        physical_penalty += cfg.alpha_physical * (abs(bdot) / max(bdot_limit, 1e-9) - 1.0)

    projection_penalty = 0.0
    projection_mode = str(row.get("projection_mode", "none"))
    if projection_mode in {"b0_only", "full"}:
        retention = float(row.get("retention_trace_ratio", np.nan))
        coherence = float(row.get("subspace_coherence_max", np.nan))
        rank_loss = float(row.get("rank_loss", 0.0))
        if np.isfinite(retention) and retention < 0.7:
            projection_penalty += cfg.alpha_projection * (0.7 - retention)
        if projection_mode == "full" and np.isfinite(retention) and retention < 0.6:
            projection_penalty += cfg.alpha_projection * 2.0 * (0.6 - retention)
        if np.isfinite(coherence) and coherence > 0.95:
            projection_penalty += cfg.alpha_projection * (coherence - 0.95)
        if np.isfinite(rank_loss) and rank_loss > 0:
            projection_penalty += cfg.alpha_projection * rank_loss

    convergence_penalty = 0.0 if bool(row.get("converged", False)) else cfg.alpha_convergence
    numerical_success = bool(row.get("numerical_success", False))
    total = (
        residual_score
        + complexity_penalty
        + condition_penalty
        + physical_penalty
        + projection_penalty
        + convergence_penalty
    )
    selectable = bool(numerical_success and np.isfinite(total))
    if not numerical_success:
        total = 1e9
    return {
        "residual_score": float(residual_score),
        "complexity_penalty": float(complexity_penalty),
        "condition_penalty": float(condition_penalty),
        "physical_penalty": float(physical_penalty),
        "projection_penalty": float(projection_penalty),
        "convergence_penalty": float(convergence_penalty),
        "total_score": float(total),
        "selectable": selectable,
    }


@dataclass(frozen=True)
class V2ScoringConfig:
    alpha_bic: float = 0.7
    alpha_condition: float = 0.2
    alpha_physical: float = 2.0
    alpha_evidence_gate: float = 5.0
    alpha_projection: float = 3.0
    alpha_stability: float = 0.5
    alpha_validation_robust: float = 0.15
    eps: float = 1e-9
    synthetic_speed_limit_mps: float = 300.0
    real_speed_limit_mps: float = 20.0
    real_speed_guard_mps: float = 2.0
    beta0_limit_mps: float = 20.0
    bdot_limit_mps2: float = 0.5
    real_bdot_limit_mps2: float = 0.02
    projection_retention_min: float = 0.70
    projection_full_retention_min: float = 0.60
    projection_coherence_max: float = 0.95


def v2_scoring_config_dict(config: V2ScoringConfig) -> dict[str, float]:
    return asdict(config)


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def compute_score_v2(row: dict, n_val: int, config: V2ScoringConfig | None = None) -> dict[str, float | bool]:
    """Validation-driven MA-BGTR-v2 score.

    The score is intentionally truth-free: it uses validation residuals,
    nested-model evidence gates, physical sanity, projection diagnostics, and
    train/validation stability. Post-hoc truth errors are not read here.
    """
    cfg = config or V2ScoringConfig()
    val_rmse = _finite_float(row.get("validation_residual_rmse_mps"), np.inf)
    robust_cost = _finite_float(row.get("validation_robust_cost_per_dof"), np.inf)
    validation_residual_score = float(np.log(max(val_rmse, cfg.eps))) if np.isfinite(val_rmse) else 1e6
    validation_robust_score = (
        float(cfg.alpha_validation_robust * np.log(max(robust_cost, cfg.eps))) if np.isfinite(robust_cost) else 10.0
    )

    model_dof = max(_finite_float(row.get("model_dof"), 0.0), 0.0)
    n = max(float(n_val), 2.0)
    bic_like_complexity_penalty = float(cfg.alpha_bic * model_dof * np.log(n) / n)

    condition = _finite_float(row.get("condition_number"), np.inf)
    if np.isfinite(condition) and condition > 0.0:
        condition_penalty = float(cfg.alpha_condition * max(0.0, np.log10(condition) - 8.0))
    else:
        condition_penalty = 10.0

    dataset_type = str(row.get("dataset_type", "synthetic"))
    speed = _finite_float(row.get("estimated_speed_mps"), np.nan)
    beta0 = _finite_float(row.get("beta0_estimated_mps"), np.nan)
    bdot = _finite_float(row.get("beta_dot_estimated_mps2"), np.nan)
    physical_penalty = 0.0
    speed_limit = cfg.real_speed_limit_mps if dataset_type == "real" else cfg.synthetic_speed_limit_mps
    if not np.isfinite(speed):
        physical_penalty += 5.0 * cfg.alpha_physical
    elif speed > speed_limit:
        physical_penalty += cfg.alpha_physical * (1.0 + (speed - speed_limit) / max(speed_limit, 1.0))
    if dataset_type == "real" and np.isfinite(speed) and speed > cfg.real_speed_guard_mps:
        physical_penalty += cfg.alpha_physical * 0.5 * (speed / cfg.real_speed_guard_mps - 1.0)
    if np.isfinite(beta0) and abs(beta0) > cfg.beta0_limit_mps:
        physical_penalty += cfg.alpha_physical * (abs(beta0) / cfg.beta0_limit_mps - 1.0)
    bdot_limit = cfg.real_bdot_limit_mps2 if dataset_type == "real" else cfg.bdot_limit_mps2
    if np.isfinite(bdot) and abs(bdot) > bdot_limit:
        physical_penalty += cfg.alpha_physical * (abs(bdot) / max(bdot_limit, cfg.eps) - 1.0)

    gate_names = [
        "dynamic_gate_pass",
        "bias_gate_pass",
        "drift_gate_pass",
        "robust_gate_pass",
        "projection_gate_pass",
        "real_static_guard_pass",
    ]
    failed_gate_count = sum(1 for key in gate_names if not bool(row.get(key, True)))
    evidence_gate_penalty = float(cfg.alpha_evidence_gate * failed_gate_count)

    model = str(row.get("candidate_model", ""))
    projection_penalty = 0.0
    if model in {"M5_be_b0_only", "M6_be_full", "M8_robust_be_b0_only", "M9_robust_be_full"}:
        retention = _finite_float(row.get("retention_trace_ratio"), np.nan)
        coherence = _finite_float(row.get("subspace_coherence_max"), np.nan)
        rank_loss = _finite_float(row.get("rank_loss"), 0.0)
        if np.isfinite(retention) and retention < cfg.projection_retention_min:
            projection_penalty += cfg.alpha_projection * (cfg.projection_retention_min - retention)
        if model in {"M6_be_full", "M9_robust_be_full"} and np.isfinite(retention) and retention < cfg.projection_full_retention_min:
            projection_penalty += cfg.alpha_projection * 2.0 * (cfg.projection_full_retention_min - retention)
        if np.isfinite(coherence) and coherence > cfg.projection_coherence_max:
            projection_penalty += cfg.alpha_projection * (coherence - cfg.projection_coherence_max)
        if np.isfinite(rank_loss) and rank_loss > 0.0:
            projection_penalty += cfg.alpha_projection * rank_loss

    gap = abs(_finite_float(row.get("train_validation_gap_mps"), 0.0))
    speed_delta = abs(_finite_float(row.get("speed_train_full_delta_mps"), 0.0))
    beta_delta = abs(_finite_float(row.get("beta0_train_full_delta_mps"), 0.0))
    bdot_delta = abs(_finite_float(row.get("bdot_train_full_delta_mps2"), 0.0))
    stability_penalty = float(
        cfg.alpha_stability
        * (
            np.log1p(max(gap, 0.0))
            + 0.05 * np.log1p(max(speed_delta, 0.0))
            + 0.10 * np.log1p(max(beta_delta, 0.0))
            + 0.50 * np.log1p(max(bdot_delta, 0.0) / 0.01)
        )
    )

    convergence_penalty = 0.0 if bool(row.get("converged", False)) else 1.0
    total = (
        validation_residual_score
        + validation_robust_score
        + bic_like_complexity_penalty
        + condition_penalty
        + physical_penalty
        + evidence_gate_penalty
        + projection_penalty
        + stability_penalty
        + convergence_penalty
    )
    numerical_success = bool(row.get("numerical_success", False))
    gate_blocked = bool(str(row.get("gate_block_reason", "")).strip())
    selectable = bool(numerical_success and np.isfinite(total) and not gate_blocked)
    if not numerical_success:
        total = 1e9
    return {
        "validation_residual_score": float(validation_residual_score),
        "validation_robust_score": float(validation_robust_score),
        "bic_like_complexity_penalty": float(bic_like_complexity_penalty),
        "condition_penalty": float(condition_penalty),
        "physical_penalty": float(physical_penalty),
        "evidence_gate_penalty": float(evidence_gate_penalty),
        "projection_penalty": float(projection_penalty),
        "stability_penalty": float(stability_penalty),
        "convergence_penalty": float(convergence_penalty),
        "total_score_v2": float(total),
        "selectable_v2": selectable,
    }


@dataclass(frozen=True)
class V3ScoringConfig:
    alpha_raw: float = 0.25
    alpha_trimmed: float = 0.45
    alpha_inlier: float = 0.45
    alpha_robust_cost: float = 0.20
    alpha_complexity: float = 0.6
    alpha_condition: float = 0.15
    alpha_physical: float = 2.0
    alpha_information: float = 1.0
    alpha_gate: float = 5.0
    robust_trimmed_gain_min: float = 0.02
    robust_inlier_gain_min: float = 0.02
    eps: float = 1e-9
    real_speed_guard_mps: float = 2.0
    real_bdot_limit_mps2: float = 0.02


def v3_scoring_config_dict(config: V3ScoringConfig) -> dict[str, float]:
    return asdict(config)


def compute_score_v3(row: dict, n_val: int, config: V3ScoringConfig | None = None) -> dict[str, float | bool]:
    cfg = config or V3ScoringConfig()
    raw = _finite_float(row.get("raw_validation_rmse_mps"), np.inf)
    trimmed = _finite_float(row.get("trimmed_validation_rmse_mps"), np.inf)
    inlier = _finite_float(row.get("inlier_validation_rmse_mps"), np.inf)
    robust_cost = _finite_float(row.get("robust_validation_cost"), np.inf)
    raw_component = cfg.alpha_raw * (np.log(max(raw, cfg.eps)) if np.isfinite(raw) else 1e6)
    trimmed_component = cfg.alpha_trimmed * (np.log(max(trimmed, cfg.eps)) if np.isfinite(trimmed) else 1e6)
    inlier_component = cfg.alpha_inlier * (np.log(max(inlier, cfg.eps)) if np.isfinite(inlier) else 1e6)
    robust_cost_component = cfg.alpha_robust_cost * (np.log(max(robust_cost, cfg.eps)) if np.isfinite(robust_cost) else 1e6)

    model = str(row.get("candidate_model", ""))
    model_dof = max(_finite_float(row.get("model_dof"), 0.0), 0.0)
    n = max(float(n_val), 2.0)
    complexity_penalty = float(cfg.alpha_complexity * model_dof * np.log(n) / n)

    condition = _finite_float(row.get("condition_number"), np.inf)
    condition_penalty = cfg.alpha_condition * max(0.0, np.log10(condition) - 8.0) if np.isfinite(condition) and condition > 0.0 else 10.0

    dataset_type = str(row.get("dataset_type", "synthetic"))
    speed = _finite_float(row.get("estimated_speed_mps"), np.nan)
    bdot = _finite_float(row.get("beta_dot_estimated_mps2"), np.nan)
    physical_penalty = 0.0
    if not bool(row.get("physical_plausible", False)):
        physical_penalty += cfg.alpha_physical
    if dataset_type == "real":
        if np.isfinite(speed) and speed > cfg.real_speed_guard_mps:
            physical_penalty += cfg.alpha_physical * (speed / cfg.real_speed_guard_mps - 1.0)
        if np.isfinite(bdot) and abs(bdot) > cfg.real_bdot_limit_mps2:
            physical_penalty += cfg.alpha_physical * (abs(bdot) / cfg.real_bdot_limit_mps2 - 1.0)

    info_ret = _finite_float(row.get("information_retention_ratio"), np.nan)
    information_retention_penalty = 0.0
    if model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}:
        if np.isfinite(info_ret) and info_ret < 0.25:
            information_retention_penalty += cfg.alpha_information * (0.25 - info_ret)
    elif not np.isfinite(info_ret):
        information_retention_penalty = 0.0

    gate_block_reason = str(row.get("gate_block_reason", "")).strip()
    gate_penalty = cfg.alpha_gate if gate_block_reason else 0.0
    if dataset_type == "real" and model.startswith("M1"):
        gate_penalty = 0.0

    tail_ratio = _finite_float(row.get("tail_ratio"), 0.0)
    mad_ratio = _finite_float(row.get("mad_ratio"), 0.0)
    outlier_evidence = tail_ratio > 0.05 or mad_ratio > 2.0
    robust_trimmed_gain = _finite_float(row.get("robust_trimmed_gain"), 0.0)
    robust_inlier_gain = _finite_float(row.get("robust_inlier_gain"), 0.0)
    robust_bonus = 0.0
    if model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale", "M7_robust_ctd_full"} and outlier_evidence:
        if robust_trimmed_gain > cfg.robust_trimmed_gain_min or robust_inlier_gain > cfg.robust_inlier_gain_min:
            robust_bonus -= 0.75
    if model in {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias"} and outlier_evidence:
        robust_bonus += 0.35

    total = (
        raw_component
        + trimmed_component
        + inlier_component
        + robust_cost_component
        + complexity_penalty
        + condition_penalty
        + physical_penalty
        + information_retention_penalty
        + gate_penalty
        + robust_bonus
    )
    numerical_success = bool(row.get("numerical_success", False))
    selectable = bool(numerical_success and np.isfinite(total) and not gate_block_reason)
    if not numerical_success:
        total = 1e9
    return {
        "raw_validation_component": float(raw_component),
        "trimmed_validation_component": float(trimmed_component),
        "inlier_validation_component": float(inlier_component),
        "robust_cost_component": float(robust_cost_component),
        "complexity_penalty": float(complexity_penalty),
        "condition_penalty": float(condition_penalty),
        "physical_penalty": float(physical_penalty),
        "information_retention_penalty": float(information_retention_penalty),
        "gate_penalty": float(gate_penalty),
        "robust_bonus": float(robust_bonus),
        "total_score_v3": float(total),
        "selectable_v3": selectable,
    }


@dataclass(frozen=True)
class V4ScoringConfig:
    alpha_validation: float = 1.0
    alpha_robust: float = 0.20
    alpha_complexity: float = 0.6
    alpha_condition: float = 0.15
    alpha_physical: float = 2.0
    alpha_robust_gate: float = 50.0
    alpha_consistency: float = 1.0
    alpha_refine: float = 50.0
    alpha_real_conservative: float = 20.0
    eps: float = 1e-9


def v4_scoring_config_dict(config: V4ScoringConfig) -> dict[str, float]:
    return asdict(config)


def compute_score_v4(row: dict, n_val: int, config: V4ScoringConfig | None = None) -> dict[str, float | bool]:
    cfg = config or V4ScoringConfig()
    model = str(row.get("candidate_model", ""))
    dataset_type = str(row.get("dataset_type", "synthetic"))
    raw = _finite_float(row.get("raw_validation_rmse_mps"), np.inf)
    trimmed = _finite_float(row.get("trimmed_validation_rmse_mps"), np.inf)
    inlier = _finite_float(row.get("inlier_validation_rmse_mps"), np.inf)
    robust_cost = _finite_float(row.get("robust_validation_cost"), np.inf)
    validation_component = cfg.alpha_validation * (
        0.25 * np.log(max(raw, cfg.eps))
        + 0.40 * np.log(max(trimmed, cfg.eps))
        + 0.35 * np.log(max(inlier, cfg.eps))
        if np.isfinite(raw) and np.isfinite(trimmed) and np.isfinite(inlier)
        else 1e6
    )
    robust_component = cfg.alpha_robust * (np.log(max(robust_cost, cfg.eps)) if np.isfinite(robust_cost) else 1e6)
    model_dof = max(_finite_float(row.get("model_dof"), 0.0), 0.0)
    n = max(float(n_val), 2.0)
    complexity_penalty = float(cfg.alpha_complexity * model_dof * np.log(n) / n)
    condition = _finite_float(row.get("condition_number"), np.inf)
    condition_penalty = cfg.alpha_condition * max(0.0, np.log10(condition) - 8.0) if np.isfinite(condition) and condition > 0.0 else 10.0
    physical_penalty = 0.0 if bool(row.get("physical_plausible", False)) else cfg.alpha_physical

    robust_models = {
        "M7_robust_ctd_full",
        "M8_robust_be_b0_only",
        "M9_robust_be_full",
        "M10_gir_tr_fixed_scale",
        "M11_gir_tr_mad_scale",
        "M13_robust_ctd_full_plus_gir_refine",
    }
    robust_gate_penalty = cfg.alpha_robust_gate if model in robust_models and not bool(row.get("robust_gate_pass", True)) else 0.0
    consistency_penalty = cfg.alpha_consistency * _finite_float(row.get("consistency_penalty"), 0.0)
    cascade_models = {"M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine", "M14_static_plus_gir_refine"}
    refine_penalty = cfg.alpha_refine if model in cascade_models and not bool(row.get("refine_gate_pass", False)) else 0.0
    real_conservative_penalty = 0.0
    if dataset_type == "real":
        speed = _finite_float(row.get("estimated_speed_mps"), 0.0)
        bdot = abs(_finite_float(row.get("beta_dot_estimated_mps2"), 0.0))
        if model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale", "M14_static_plus_gir_refine"}:
            real_conservative_penalty += cfg.alpha_real_conservative
        if model not in {"M0_static_position", "M1_static_position_bias"} and speed > 2.0:
            real_conservative_penalty += cfg.alpha_real_conservative * (speed / 2.0)
        if bdot > 0.02:
            real_conservative_penalty += cfg.alpha_real_conservative * (bdot / 0.02)

    total = (
        validation_component
        + robust_component
        + complexity_penalty
        + condition_penalty
        + physical_penalty
        + robust_gate_penalty
        + consistency_penalty
        + refine_penalty
        + real_conservative_penalty
    )
    direct_gir_block = model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"} and (
        dataset_type == "real" or str(row.get("position_init_label", "")) != "truth_init"
    )
    m14_real_block = dataset_type == "real" and model == "M14_static_plus_gir_refine"
    selectable = bool(
        bool(row.get("numerical_success", False))
        and bool(row.get("quality_pass", False))
        and np.isfinite(total)
        and not direct_gir_block
        and not m14_real_block
        and not (model in robust_models and not bool(row.get("robust_gate_pass", True)))
        and not (model in cascade_models and not bool(row.get("refine_gate_pass", False)))
        and bool(row.get("consistency_gate_pass", True))
    )
    if not bool(row.get("numerical_success", False)):
        total = 1e9
    return {
        "validation_component": float(validation_component),
        "robust_component": float(robust_component),
        "complexity_penalty": float(complexity_penalty),
        "condition_penalty": float(condition_penalty),
        "physical_penalty": float(physical_penalty),
        "robust_gate_penalty": float(robust_gate_penalty),
        "consistency_penalty": float(consistency_penalty),
        "refine_penalty": float(refine_penalty),
        "real_conservative_penalty": float(real_conservative_penalty),
        "total_score_v4": float(total),
        "selectable_v4": selectable,
    }


@dataclass(frozen=True)
class V5ScoringConfig:
    alpha_uncertainty: float = 0.35
    alpha_bootstrap: float = 0.45
    alpha_ephemeris: float = 0.25
    alpha_geometry: float = 0.55
    alpha_overfit: float = 0.60
    eps: float = 1e-9
    position_cov_scale_m: float = 1000.0
    velocity_cov_scale_mps: float = 20.0
    bootstrap_position_scale_m: float = 1000.0
    bootstrap_velocity_scale_mps: float = 20.0
    min_effective_rank_margin: int = 1
    geometry_retention_min: float = 0.70


def v5_scoring_config_dict(config: V5ScoringConfig) -> dict[str, float]:
    return asdict(config)


def compute_score_v5(row: dict, n_val: int, config: V5ScoringConfig | None = None) -> dict[str, float | bool]:
    """Risk-calibrated MA-BGTR-v5 score.

    The v5 layer keeps the v4 validation/gating score and adds truth-free risk
    terms from covariance, stability, ephemeris sensitivity, and geometry
    retention diagnostics.
    """
    cfg = config or V5ScoringConfig()
    model = str(row.get("candidate_model", ""))
    dataset_type = str(row.get("dataset_type", "synthetic"))
    scenario = str(row.get("scenario", ""))
    score_v4 = _finite_float(row.get("total_score_v4"), np.inf)
    if not np.isfinite(score_v4):
        score_v4 = 1e9

    pos_cov = _finite_float(row.get("position_cov_sqrt_trace_m"), np.inf)
    vel_cov = _finite_float(row.get("velocity_cov_sqrt_trace_mps"), 0.0)
    effective_rank = _finite_float(row.get("effective_rank"), 0.0)
    model_dof = _finite_float(row.get("model_dof"), 0.0)
    rank_shortfall = max(0.0, model_dof - effective_rank - cfg.min_effective_rank_margin)
    uncertainty_penalty = cfg.alpha_uncertainty * (
        (np.log1p(pos_cov / cfg.position_cov_scale_m) if np.isfinite(pos_cov) else 5.0)
        + (np.log1p(max(vel_cov, 0.0) / cfg.velocity_cov_scale_mps) if np.isfinite(vel_cov) else 0.0)
        + 0.5 * rank_shortfall
    )

    boot_pos = _finite_float(row.get("position_bootstrap_spread_m"), np.inf)
    boot_vel = _finite_float(row.get("velocity_bootstrap_spread_mps"), 0.0)
    boot_success = _finite_float(row.get("bootstrap_success_rate"), 0.0)
    bootstrap_stability_penalty = cfg.alpha_bootstrap * (
        (np.log1p(boot_pos / cfg.bootstrap_position_scale_m) if np.isfinite(boot_pos) else 5.0)
        + (np.log1p(max(boot_vel, 0.0) / cfg.bootstrap_velocity_scale_mps) if np.isfinite(boot_vel) else 0.0)
        + 3.0 * max(0.0, 0.80 - boot_success)
    )

    p95_r_eph = _finite_float(row.get("p95_R_eph"), 0.0)
    median_r_eph = _finite_float(row.get("median_R_eph"), 0.0)
    ephemeris_sensitivity_penalty = cfg.alpha_ephemeris * np.log1p(max(p95_r_eph, median_r_eph, 0.0))
    if scenario in {"S4_dynamic_ephemeris_error", "S5_dynamic_hard"} and p95_r_eph <= 0.0:
        ephemeris_sensitivity_penalty += cfg.alpha_ephemeris

    residual_ret = _finite_float(row.get("residual_trim_info_retention"), np.nan)
    geom_ret = _finite_float(row.get("geometry_preserved_info_retention"), np.nan)
    critical_count = _finite_float(row.get("geometry_critical_outlier_count"), 0.0)
    geometry_information_penalty = 0.0
    if np.isfinite(geom_ret) and geom_ret < cfg.geometry_retention_min:
        geometry_information_penalty += cfg.alpha_geometry * (cfg.geometry_retention_min - geom_ret)
    if np.isfinite(residual_ret) and residual_ret < 0.50:
        geometry_information_penalty += cfg.alpha_geometry * (0.50 - residual_ret)
    if critical_count > 0.0:
        geometry_information_penalty += cfg.alpha_geometry * min(critical_count, 5.0) / 10.0

    trimmed = _finite_float(row.get("trimmed_validation_rmse_mps"), np.inf)
    raw = _finite_float(row.get("raw_validation_rmse_mps"), np.inf)
    overfit_risk_penalty = 0.0
    if np.isfinite(trimmed) and np.isfinite(raw) and raw > max(trimmed, cfg.eps) * 2.0:
        overfit_risk_penalty += cfg.alpha_overfit * np.log1p(raw / max(trimmed, cfg.eps))
    if np.isfinite(pos_cov) and np.isfinite(boot_pos) and pos_cov > cfg.position_cov_scale_m and boot_pos > cfg.bootstrap_position_scale_m:
        overfit_risk_penalty += cfg.alpha_overfit * np.log1p((pos_cov / cfg.position_cov_scale_m) * (boot_pos / cfg.bootstrap_position_scale_m))
    # S5/S9 failures in v4 were caused by reduced dynamic models winning on
    # residuals while their state risk stayed high. This term remains
    # truth-free: it only triggers in geometry/ephemeris-stressed diagnostics.
    if model in {"M3_ctd_no_drift", "M4_ctd_no_bias"} and scenario in {"S5_dynamic_hard", "S9_dynamic_geometry_hard"}:
        risk = 0.0
        if np.isfinite(geom_ret) and geom_ret < 0.90:
            risk += 0.75
        if rank_shortfall > 0.0:
            risk += 0.50 * rank_shortfall
        if np.isfinite(boot_pos) and boot_pos > 2000.0:
            risk += 0.50
        overfit_risk_penalty += cfg.alpha_overfit * risk

    total = (
        score_v4
        + uncertainty_penalty
        + bootstrap_stability_penalty
        + ephemeris_sensitivity_penalty
        + geometry_information_penalty
        + overfit_risk_penalty
    )
    selectable = bool(
        bool(row.get("selectable_v4", False))
        and bool(row.get("numerical_success", False))
        and bool(row.get("quality_pass", False))
        and bool(row.get("covariance_available", False))
        and np.isfinite(total)
    )
    if dataset_type == "real" and model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale", "M14_static_plus_gir_refine"}:
        selectable = False
    if not bool(row.get("numerical_success", False)):
        total = 1e9
    return {
        "uncertainty_penalty": float(uncertainty_penalty),
        "bootstrap_stability_penalty": float(bootstrap_stability_penalty),
        "ephemeris_sensitivity_penalty": float(ephemeris_sensitivity_penalty),
        "geometry_information_penalty": float(geometry_information_penalty),
        "overfit_risk_penalty": float(overfit_risk_penalty),
        "total_score_v5": float(total),
        "selectable_v5": selectable,
    }
