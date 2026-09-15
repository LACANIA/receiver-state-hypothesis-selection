"""Freeze-audit checks for MA-BGTR-v7.1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FOCUS_SCENARIOS = [
    ("synthetic", "S0_static_clean", "select static family and stay near 7 m"),
    ("synthetic", "S1_dynamic_clean", "keep CTD-series dynamic advantage"),
    ("synthetic", "S2_dynamic_bias_strong", "keep bias-aware CTD-series behavior"),
    ("synthetic", "S3_dynamic_outlier", "allow robust CTD/refine without degradation"),
    ("synthetic", "S4_dynamic_ephemeris_error", "do not regress ephemeris-hard case"),
    ("synthetic", "S5_dynamic_hard", "keep CTD full or equivalent near 471 m"),
    ("synthetic", "S7_fast_north_no_bias", "improve no-bias case toward oracle"),
    ("synthetic", "S9_dynamic_geometry_hard", "keep no-drift/equivalent near 125 m"),
    ("qatar_synthetic", "Q0_qatar_cfo_mild", "keep Qatar mild CFO case stable"),
    ("qatar_synthetic", "Q1_qatar_cfo_heavy_tail", "keep Qatar heavy-tail case stable"),
    ("qatar_synthetic", "Q2_qatar_confidence_dropout", "keep Qatar dropout case stable"),
    ("qatar_synthetic", "Q3_qatar_burst_outlier", "improve burst outlier toward robust result"),
    ("qatar_synthetic", "Q4_qatar_hard_geometry_noise", "keep CTD full near 865 m"),
    ("real", "real_iridium", "continue static sanity near 132 m"),
]


def _row(df: pd.DataFrame, dataset_type: str, scenario: str) -> pd.Series | None:
    subset = df[(df["dataset_type"] == dataset_type) & (df["scenario"] == scenario)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _model(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return str(row.get("most_selected_model_v71", row.get("selected_model_v71", "")))


def _prev_error(v7_aggregate: pd.DataFrame, dataset_type: str, scenario: str) -> float:
    row = _row(v7_aggregate, dataset_type, scenario)
    return _finite(row.get("median_final_position_error_m"), np.nan) if row is not None else np.nan


def _check(dataset_type: str, scenario: str, model: str, error: float, previous: float, oracle: float) -> tuple[bool, str]:
    reasons: list[str] = []
    if scenario == "S0_static_clean":
        if model not in {"M0_static_position", "M1_static_position_bias"}:
            reasons.append("not_static_family")
        if error > 10.0:
            reasons.append("error_gt_10m")
    elif scenario in {"S1_dynamic_clean", "S2_dynamic_bias_strong", "S4_dynamic_ephemeris_error"}:
        if np.isfinite(previous) and error > previous * 1.10:
            reasons.append("regressed_gt_10pct_vs_v7")
    elif scenario == "S3_dynamic_outlier":
        if model not in {"M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine", "M2_ctd_full", "M3_ctd_no_drift"}:
            reasons.append("unexpected_outlier_model")
        if np.isfinite(previous) and error > previous * 1.10:
            reasons.append("regressed_gt_10pct_vs_v7")
    elif scenario == "S5_dynamic_hard":
        if model not in {"M2_ctd_full", "M12_ctd_full_plus_gir_refine"}:
            reasons.append("not_ctd_full_family")
        if error > 550.0:
            reasons.append("error_gt_550m")
    elif scenario == "S7_fast_north_no_bias":
        if model != "M4_ctd_no_bias":
            reasons.append("not_no_bias")
        if error > 90.0:
            reasons.append("error_gt_90m")
        if np.isfinite(previous) and error >= previous:
            reasons.append("not_improved_vs_v7")
    elif scenario == "S9_dynamic_geometry_hard":
        if model not in {"M3_ctd_no_drift", "M2_ctd_full", "M12_ctd_full_plus_gir_refine"}:
            reasons.append("not_no_drift_or_equivalent")
        if error > 160.0:
            reasons.append("error_gt_160m")
    elif scenario == "Q3_qatar_burst_outlier":
        if model not in {"M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine", "M2_ctd_full"}:
            reasons.append("unexpected_q3_model")
        if error > 205.0:
            reasons.append("error_gt_205m")
        if np.isfinite(previous) and error >= previous:
            reasons.append("not_improved_vs_v7")
    elif scenario == "Q4_qatar_hard_geometry_noise":
        if model != "M2_ctd_full":
            reasons.append("not_ctd_full")
        if error > 1000.0:
            reasons.append("error_gt_1000m")
    elif dataset_type == "qatar_synthetic":
        if np.isfinite(previous) and error > previous * 1.10:
            reasons.append("qatar_regressed_gt_10pct_vs_v7")
    elif dataset_type == "real":
        if model not in {"M0_static_position", "M1_static_position_bias"}:
            reasons.append("real_not_static")
        if error > 140.0:
            reasons.append("real_error_gt_140m")
    return not reasons, "pass" if not reasons else ";".join(reasons)


def build_freeze_audit(aggregate_v71: pd.DataFrame, aggregate_v7: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_type, scenario, requirement in FOCUS_SCENARIOS:
        current = _row(aggregate_v71, dataset_type, scenario)
        if current is None:
            rows.append(
                {
                    "dataset_type": dataset_type,
                    "scenario": scenario,
                    "freeze_requirement": requirement,
                    "target_reference": "",
                    "selected_model_v71": "",
                    "median_final_position_error_m": np.nan,
                    "oracle_best_position_error_m": np.nan,
                    "selected_minus_oracle_error_m": np.nan,
                    "previous_v7_error_m": _prev_error(aggregate_v7, dataset_type, scenario),
                    "improvement_vs_v7_m": np.nan,
                    "pass_freeze_audit": False,
                    "freeze_audit_reason": "missing_current_aggregate",
                }
            )
            continue
        model = _model(current)
        error = _finite(current.get("median_final_position_error_m"), np.nan)
        oracle = _finite(current.get("oracle_best_position_error_m"), np.nan)
        previous = _prev_error(aggregate_v7, dataset_type, scenario)
        passed, reason = _check(dataset_type, scenario, model, error, previous, oracle)
        rows.append(
            {
                "dataset_type": dataset_type,
                "scenario": scenario,
                "freeze_requirement": requirement,
                "target_reference": "TECH12 v7 median and scenario-specific threshold",
                "selected_model_v71": model,
                "median_final_position_error_m": error,
                "oracle_best_position_error_m": oracle,
                "selected_minus_oracle_error_m": _finite(current.get("selected_minus_oracle_error_m"), np.nan),
                "previous_v7_error_m": previous,
                "improvement_vs_v7_m": previous - error if np.isfinite(previous) and np.isfinite(error) else np.nan,
                "pass_freeze_audit": bool(passed),
                "freeze_audit_reason": reason,
            }
        )
    return pd.DataFrame(rows)
