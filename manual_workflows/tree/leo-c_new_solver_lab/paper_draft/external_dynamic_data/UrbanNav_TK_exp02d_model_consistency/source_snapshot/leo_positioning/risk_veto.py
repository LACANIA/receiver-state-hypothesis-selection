"""Risk veto rules for MA-BGTR-v6 Pareto selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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
    "M12_ctd_full_plus_gir_refine",
    "M13_robust_ctd_full_plus_gir_refine",
    "M14_static_plus_gir_refine",
}
FULL_DRIFT_MODELS = {"M2_ctd_full", "M7_robust_ctd_full", "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine"}
ROBUST_MODELS = {
    "M7_robust_ctd_full",
    "M8_robust_be_b0_only",
    "M9_robust_be_full",
    "M10_gir_tr_fixed_scale",
    "M11_gir_tr_mad_scale",
    "M13_robust_ctd_full_plus_gir_refine",
}
BE_FULL_MODELS = {"M6_be_full", "M9_robust_be_full"}
DIRECT_GIR_MODELS = {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}
REAL_FORBIDDEN_GIR_MODELS = {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale", "M14_static_plus_gir_refine"}

MODEL_DOF = {
    "M0_static_position": 3,
    "M1_static_position_bias": 4,
    "M2_ctd_full": 8,
    "M3_ctd_no_drift": 7,
    "M4_ctd_no_bias": 6,
    "M5_be_b0_only": 6,
    "M6_be_full": 6,
    "M7_robust_ctd_full": 8,
    "M8_robust_be_b0_only": 6,
    "M9_robust_be_full": 6,
    "M10_gir_tr_fixed_scale": 8,
    "M11_gir_tr_mad_scale": 8,
    "M12_ctd_full_plus_gir_refine": 8,
    "M13_robust_ctd_full_plus_gir_refine": 8,
    "M14_static_plus_gir_refine": 8,
}


@dataclass(frozen=True)
class RiskVetoConfig:
    position_cov_limit_m: float = 5000.0
    position_bootstrap_limit_m: float = 5000.0
    condition_limit: float = 1.0e14
    speed_limit_mps: float = 300.0
    beta0_limit_mps: float = 30.0
    bdot_limit_mps2: float = 1.0
    information_retention_min: float = 0.4
    geometry_preserved_retention_min: float = 0.5
    be_full_retention_min: float = 0.7
    real_dynamic_gain_min: float = 0.05
    real_dynamic_gain_strong: float = 0.10
    real_speed_limit_mps: float = 2.0
    real_bdot_limit_mps2: float = 0.02


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    if isinstance(value, float) and not np.isfinite(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def model_dof(model: str, row: dict[str, Any] | pd.Series | None = None) -> int:
    if row is not None:
        dof = finite_float(row.get("model_dof"), np.nan)
        if np.isfinite(dof) and dof > 0:
            return int(dof)
    return MODEL_DOF.get(str(model), 8)


def is_static_model(model: str) -> bool:
    return str(model) in STATIC_MODELS


def is_dynamic_model(model: str) -> bool:
    return str(model) in DYNAMIC_MODELS and str(model) not in STATIC_MODELS


def is_robust_model(model: str) -> bool:
    return str(model) in ROBUST_MODELS


def is_full_drift_model(model: str) -> bool:
    return str(model) in FULL_DRIFT_MODELS


def _best_metric(group: pd.DataFrame, models: set[str], metric: str) -> float:
    if metric not in group.columns:
        return np.nan
    subset = group[group["candidate_model"].isin(models)].copy()
    if subset.empty:
        return np.nan
    values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else np.nan


def compute_group_references(group: pd.DataFrame) -> dict[str, float]:
    best_static_raw = _best_metric(group, STATIC_MODELS, "raw_validation_rmse_mps")
    best_static_trim = _best_metric(group, STATIC_MODELS, "trimmed_validation_rmse_mps")
    best_dynamic_raw = _best_metric(group, DYNAMIC_MODELS, "raw_validation_rmse_mps")
    best_ctd_raw = _best_metric(group, {"M2_ctd_full"}, "raw_validation_rmse_mps")
    best_ctd_trim = _best_metric(group, {"M2_ctd_full"}, "trimmed_validation_rmse_mps")
    best_ctd_inlier = _best_metric(group, {"M2_ctd_full"}, "inlier_validation_rmse_mps")
    best_no_drift_raw = _best_metric(group, {"M3_ctd_no_drift"}, "raw_validation_rmse_mps")
    best_no_drift_trim = _best_metric(group, {"M3_ctd_no_drift"}, "trimmed_validation_rmse_mps")
    best_nonrobust_trim = _best_metric(group, {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias"}, "trimmed_validation_rmse_mps")
    best_nonrobust_inlier = _best_metric(group, {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias"}, "inlier_validation_rmse_mps")
    return {
        "best_static_raw": best_static_raw,
        "best_static_trimmed": best_static_trim,
        "best_dynamic_raw": best_dynamic_raw,
        "best_ctd_raw": best_ctd_raw,
        "best_ctd_trimmed": best_ctd_trim,
        "best_ctd_inlier": best_ctd_inlier,
        "best_no_drift_raw": best_no_drift_raw,
        "best_no_drift_trimmed": best_no_drift_trim,
        "best_nonrobust_trimmed": best_nonrobust_trim,
        "best_nonrobust_inlier": best_nonrobust_inlier,
    }


def relative_gain(reference: float, candidate: float) -> float:
    if not (np.isfinite(reference) and np.isfinite(candidate)) or abs(reference) < 1.0e-12:
        return 0.0
    return float((reference - candidate) / abs(reference))


def dynamic_gain(row: dict[str, Any] | pd.Series, refs: dict[str, float]) -> float:
    return relative_gain(refs.get("best_static_raw", np.nan), finite_float(row.get("raw_validation_rmse_mps"), np.nan))


def drift_gain(row: dict[str, Any] | pd.Series, refs: dict[str, float]) -> float:
    raw_gain = relative_gain(refs.get("best_no_drift_raw", np.nan), finite_float(row.get("raw_validation_rmse_mps"), np.nan))
    trim_gain = relative_gain(refs.get("best_no_drift_trimmed", np.nan), finite_float(row.get("trimmed_validation_rmse_mps"), np.nan))
    return float(max(raw_gain, trim_gain))


def robust_gain(row: dict[str, Any] | pd.Series, refs: dict[str, float]) -> float:
    trim_gain = relative_gain(refs.get("best_nonrobust_trimmed", np.nan), finite_float(row.get("trimmed_validation_rmse_mps"), np.nan))
    inlier_gain = relative_gain(refs.get("best_nonrobust_inlier", np.nan), finite_float(row.get("inlier_validation_rmse_mps"), np.nan))
    return float(max(trim_gain, inlier_gain))


def has_outlier_evidence(row: dict[str, Any] | pd.Series) -> bool:
    tail = finite_float(row.get("tail_ratio"), 0.0)
    mad = finite_float(row.get("mad_ratio"), 0.0)
    return bool(tail >= 0.05 or mad >= 2.0)


def severe_risk_veto(row: dict[str, Any] | pd.Series, refs: dict[str, float], config: RiskVetoConfig | None = None) -> tuple[bool, str]:
    cfg = config or RiskVetoConfig()
    reasons: list[str] = []
    model = str(row.get("candidate_model", ""))
    dataset_type = str(row.get("dataset_type", ""))

    checks = [
        ("position_cov_sqrt_trace_m", cfg.position_cov_limit_m, ">"),
        ("position_bootstrap_spread_m", cfg.position_bootstrap_limit_m, ">"),
        ("condition_number", cfg.condition_limit, ">"),
        ("estimated_speed_mps", cfg.speed_limit_mps, ">"),
        ("beta0_estimated_mps", cfg.beta0_limit_mps, "abs>"),
        ("beta_dot_estimated_mps2", cfg.bdot_limit_mps2, "abs>"),
    ]
    for col, limit, op in checks:
        value = finite_float(row.get(col), np.nan)
        if not np.isfinite(value):
            continue
        if op == ">" and value > limit:
            reasons.append(f"{col}>{limit:g}")
        elif op == "abs>" and abs(value) > limit:
            reasons.append(f"abs({col})>{limit:g}")

    info = finite_float(row.get("information_retention_ratio"), np.nan)
    if np.isfinite(info) and info < cfg.information_retention_min:
        reasons.append(f"information_retention_ratio<{cfg.information_retention_min:g}")
    geo = finite_float(row.get("geometry_preserved_info_retention"), np.nan)
    if np.isfinite(geo) and geo < cfg.geometry_preserved_retention_min:
        reasons.append(f"geometry_preserved_info_retention<{cfg.geometry_preserved_retention_min:g}")

    if dataset_type == "real" and is_dynamic_model(model):
        gain = dynamic_gain(row, refs)
        speed = finite_float(row.get("estimated_speed_mps"), 0.0)
        bdot = abs(finite_float(row.get("beta_dot_estimated_mps2"), 0.0))
        if gain < cfg.real_dynamic_gain_min:
            reasons.append(f"real_dynamic_gain<{cfg.real_dynamic_gain_min:g}")
        if speed > cfg.real_speed_limit_mps and gain <= cfg.real_dynamic_gain_strong:
            reasons.append("real_speed_without_strong_gain")
        if bdot > cfg.real_bdot_limit_mps2 and gain <= cfg.real_dynamic_gain_strong:
            reasons.append("real_bdot_without_strong_gain")

    return bool(reasons), ";".join(reasons)


def hard_filter(row: dict[str, Any] | pd.Series, refs: dict[str, float], config: RiskVetoConfig | None = None) -> tuple[bool, str]:
    cfg = config or RiskVetoConfig()
    reasons: list[str] = []
    model = str(row.get("candidate_model", ""))
    dataset_type = str(row.get("dataset_type", ""))

    if not bool_value(row.get("numerical_success"), False):
        reasons.append("numerical_success_false")
    if not bool_value(row.get("physical_plausible"), False):
        reasons.append("physical_plausible_false")
    if not bool_value(row.get("quality_pass"), False):
        reasons.append("quality_pass_false")
    selectable = row.get("selectable_v5", np.nan)
    if pd.notna(selectable) and not bool_value(selectable, True):
        reasons.append("selectable_false")

    if model in BE_FULL_MODELS:
        retention = finite_float(row.get("retention_trace_ratio"), np.nan)
        rank_loss = finite_float(row.get("rank_loss"), 0.0)
        if np.isfinite(retention) and retention < cfg.be_full_retention_min:
            reasons.append("be_full_projection_retention_low")
        if np.isfinite(rank_loss) and rank_loss > 0:
            reasons.append("be_full_projection_rank_loss")

    if model in DIRECT_GIR_MODELS:
        reasons.append("direct_gir_not_cascade_refine")
    if dataset_type == "real" and model in REAL_FORBIDDEN_GIR_MODELS:
        reasons.append("real_sanity_gir_or_static_refine_forbidden")

    veto, veto_reason = severe_risk_veto(row, refs, cfg)
    if veto:
        reasons.append(f"severe_risk:{veto_reason}")

    return not reasons, ";".join(reasons)
