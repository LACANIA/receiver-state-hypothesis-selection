"""Evidence gates for MA-BGTR-v2 model selection."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


STATIC_MODELS = {"M0_static_position", "M1_static_position_bias"}
DYNAMIC_MODELS = {
    "M2_ctd_full",
    "M3_ctd_no_drift",
    "M4_ctd_no_bias",
    "M5_be_b0_only",
    "M6_be_full",
    "M7_robust_ctd_full",
    "M8_robust_be_b0_only",
    "M9_robust_be_full",
    "M10_gir_tr_fixed_scale",
    "M11_gir_tr_mad_scale",
}
NO_BIAS_MODELS = {"M0_static_position", "M4_ctd_no_bias"}
BIAS_MODELS = set(STATIC_MODELS | DYNAMIC_MODELS) - NO_BIAS_MODELS
NO_DRIFT_MODELS = {"M0_static_position", "M1_static_position_bias", "M3_ctd_no_drift", "M4_ctd_no_bias", "M5_be_b0_only", "M8_robust_be_b0_only"}
DRIFT_MODELS = {
    "M2_ctd_full",
    "M6_be_full",
    "M7_robust_ctd_full",
    "M9_robust_be_full",
    "M10_gir_tr_fixed_scale",
    "M11_gir_tr_mad_scale",
}
ROBUST_MODELS = {"M7_robust_ctd_full", "M8_robust_be_b0_only", "M9_robust_be_full", "M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}
PROJECTION_MODELS = {"M5_be_b0_only", "M6_be_full", "M8_robust_be_b0_only", "M9_robust_be_full"}
FULL_PROJECTION_MODELS = {"M6_be_full", "M9_robust_be_full"}


@dataclass(frozen=True)
class EvidenceConfig:
    synthetic_dynamic_gain_min: float = 0.02
    real_dynamic_gain_min: float = 0.05
    real_dynamic_strong_gain_min: float = 0.10
    bias_gain_min: float = 0.005
    synthetic_drift_gain_min: float = 0.005
    real_drift_gain_min: float = 0.02
    robust_tail_ratio_min: float = 0.05
    robust_mad_ratio_min: float = 2.0
    projection_retention_min: float = 0.70
    projection_coherence_max: float = 0.95
    real_speed_soft_limit_mps: float = 2.0
    real_bdot_limit_mps2: float = 0.02
    real_static_rmse_tie_mps: float = 0.03


def evidence_config_dict(config: EvidenceConfig) -> dict[str, float]:
    return asdict(config)


def _best_rmse(rows: list[dict], models: set[str]) -> float:
    vals = [
        float(r.get("validation_residual_rmse_mps", np.nan))
        for r in rows
        if str(r.get("candidate_model")) in models and bool(r.get("numerical_success", False))
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(min(vals)) if vals else np.nan


def _gain(reference: float, improved: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(improved) or reference <= 0.0:
        return 0.0
    return float((reference - improved) / reference)


def group_evidence(rows: list[dict], dataset_type: str, config: EvidenceConfig | None = None) -> dict[str, float]:
    cfg = config or EvidenceConfig()
    best_static = _best_rmse(rows, STATIC_MODELS)
    best_dynamic = _best_rmse(rows, DYNAMIC_MODELS)
    best_no_bias = _best_rmse(rows, NO_BIAS_MODELS)
    best_bias = _best_rmse(rows, BIAS_MODELS)
    best_no_drift = _best_rmse(rows, NO_DRIFT_MODELS)
    best_drift = _best_rmse(rows, DRIFT_MODELS)
    return {
        "best_static_val_rmse": best_static,
        "best_dynamic_val_rmse": best_dynamic,
        "best_no_bias_val_rmse": best_no_bias,
        "best_bias_val_rmse": best_bias,
        "best_no_drift_val_rmse": best_no_drift,
        "best_drift_val_rmse": best_drift,
        "dynamic_gain": _gain(best_static, best_dynamic),
        "bias_gain": _gain(best_no_bias, best_bias),
        "drift_gain": _gain(best_no_drift, best_drift),
        "dynamic_gain_min": cfg.real_dynamic_gain_min if dataset_type == "real" else cfg.synthetic_dynamic_gain_min,
        "drift_gain_min": cfg.real_drift_gain_min if dataset_type == "real" else cfg.synthetic_drift_gain_min,
    }


def apply_evidence_gates(row: dict, evidence: dict[str, float], config: EvidenceConfig | None = None) -> dict:
    cfg = config or EvidenceConfig()
    model = str(row.get("candidate_model", ""))
    dataset_type = str(row.get("dataset_type", "synthetic"))
    dynamic_gain = float(evidence.get("dynamic_gain", 0.0))
    bias_gain = float(evidence.get("bias_gain", 0.0))
    drift_gain = float(evidence.get("drift_gain", 0.0))
    tail_ratio = float(row.get("tail_ratio", np.nan))
    mad_ratio = float(row.get("mad_ratio", np.nan))
    retention = float(row.get("retention_trace_ratio", np.nan))
    coherence = float(row.get("subspace_coherence_max", np.nan))
    rank_loss = float(row.get("rank_loss", np.nan))
    speed = float(row.get("estimated_speed_mps", np.nan))
    bdot = float(row.get("beta_dot_estimated_mps2", np.nan))
    val_rmse = float(row.get("validation_residual_rmse_mps", np.nan))
    best_static = float(evidence.get("best_static_val_rmse", np.nan))

    reasons: list[str] = []
    dynamic_pass = True
    bias_pass = True
    drift_pass = True
    robust_pass = True
    projection_pass = True
    real_guard_pass = True

    if model in DYNAMIC_MODELS:
        min_gain = cfg.real_dynamic_gain_min if dataset_type == "real" else cfg.synthetic_dynamic_gain_min
        dynamic_pass = dynamic_gain > min_gain
        if not dynamic_pass:
            reasons.append("dynamic_gain_below_threshold")

    if model in BIAS_MODELS and model not in {"M0_static_position"}:
        bias_pass = bias_gain > cfg.bias_gain_min
        if not bias_pass and model in {"M1_static_position_bias", "M2_ctd_full", "M3_ctd_no_drift", "M5_be_b0_only", "M6_be_full", "M7_robust_ctd_full", "M8_robust_be_b0_only", "M9_robust_be_full"}:
            reasons.append("bias_gain_below_threshold")

    if model in DRIFT_MODELS:
        min_gain = cfg.real_drift_gain_min if dataset_type == "real" else cfg.synthetic_drift_gain_min
        drift_pass = drift_gain > min_gain
        if not drift_pass:
            reasons.append("drift_gain_below_threshold")

    if model in ROBUST_MODELS:
        robust_pass = (np.isfinite(tail_ratio) and tail_ratio > cfg.robust_tail_ratio_min) or (
            np.isfinite(mad_ratio) and mad_ratio > cfg.robust_mad_ratio_min
        )
        if not robust_pass:
            reasons.append("robust_evidence_absent")

    if model in PROJECTION_MODELS:
        projection_pass = True
        if np.isfinite(retention) and retention < cfg.projection_retention_min:
            projection_pass = False
            reasons.append("projection_retention_low")
        if model in FULL_PROJECTION_MODELS and np.isfinite(rank_loss) and rank_loss > 0:
            projection_pass = False
            reasons.append("full_projection_rank_loss")
        if np.isfinite(coherence) and coherence > cfg.projection_coherence_max:
            projection_pass = False
            reasons.append("projection_coherence_high")

    if dataset_type == "real" and model in DYNAMIC_MODELS:
        if dynamic_gain < cfg.real_dynamic_gain_min:
            real_guard_pass = False
            reasons.append("real_guard_dynamic_gain_lt_5pct")
        if np.isfinite(speed) and speed > cfg.real_speed_soft_limit_mps and dynamic_gain <= cfg.real_dynamic_strong_gain_min:
            real_guard_pass = False
            reasons.append("real_guard_speed_without_10pct_gain")
        if model in DRIFT_MODELS and np.isfinite(bdot) and abs(bdot) > cfg.real_bdot_limit_mps2 and dynamic_gain <= cfg.real_dynamic_strong_gain_min:
            real_guard_pass = False
            reasons.append("real_guard_bdot_without_10pct_gain")
        if np.isfinite(best_static) and np.isfinite(val_rmse) and abs(best_static - val_rmse) < cfg.real_static_rmse_tie_mps:
            real_guard_pass = False
            reasons.append("real_guard_static_tie")

    out = dict(row)
    out.update(
        {
            "dynamic_gain": dynamic_gain,
            "bias_gain": bias_gain,
            "drift_gain": drift_gain,
            "dynamic_gate_pass": bool(dynamic_pass),
            "bias_gate_pass": bool(bias_pass),
            "drift_gate_pass": bool(drift_pass),
            "robust_gate_pass": bool(robust_pass),
            "projection_gate_pass": bool(projection_pass),
            "real_static_guard_pass": bool(real_guard_pass),
            "gate_block_reason": ";".join(dict.fromkeys(reasons)),
        }
    )
    return out
