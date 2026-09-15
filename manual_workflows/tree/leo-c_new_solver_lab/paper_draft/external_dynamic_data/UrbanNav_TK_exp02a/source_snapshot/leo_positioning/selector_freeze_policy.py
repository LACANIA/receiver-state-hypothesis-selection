"""TECH14 freeze policy for MA-BGTR-v7.2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FOCUS_SCENARIOS = [
    ("synthetic", "S0_static_clean", "select static family and stay near 7 m"),
    ("synthetic", "S1_dynamic_clean", "keep CTD-series dynamic advantage"),
    ("synthetic", "S2_dynamic_bias_strong", "keep bias-aware CTD-series behavior"),
    ("synthetic", "S3_dynamic_outlier", "allow robust CTD/refine without degradation"),
    ("synthetic", "S5_dynamic_hard", "keep CTD full or equivalent near 471 m"),
    ("synthetic", "S7_fast_north_no_bias", "resolve no-bias case or declare limitation"),
    ("synthetic", "S9_dynamic_geometry_hard", "keep no-drift/equivalent near 125 m"),
    ("qatar_synthetic", "Q3_qatar_burst_outlier", "keep burst-outlier robust improvement"),
    ("qatar_synthetic", "Q4_qatar_hard_geometry_noise", "keep CTD full near 865 m"),
    ("real", "real_iridium", "continue static sanity near 132 m"),
]


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _mode(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


def _scenario_rows(selected: pd.DataFrame, dataset_type: str, scenario: str) -> pd.DataFrame:
    return selected[(selected["dataset_type"] == dataset_type) & (selected["scenario"] == scenario)].copy()


def _aggregate_focus(selected_v72: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for (dataset_type, scenario), group in selected_v72.groupby(["dataset_type", "scenario"], sort=True):
        errors = pd.to_numeric(group["final_position_error_m"], errors="coerce")
        out.append(
            {
                "dataset_type": dataset_type,
                "scenario": scenario,
                "runs": int(len(group)),
                "selected_model_v72": _mode(group["selected_model_v72"]),
                "median_final_position_error_m": float(errors.median()),
                "oracle_best_position_error_m": float(pd.to_numeric(group["oracle_best_position_error_m"], errors="coerce").median()),
                "selected_minus_oracle_error_m": float(pd.to_numeric(group["selected_minus_oracle_error_m"], errors="coerce").median()),
                "median_trimmed_validation_rmse_mps": float(pd.to_numeric(group["trimmed_validation_rmse_mps"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(out)


def _previous_error(selected_v71: pd.DataFrame, dataset_type: str, scenario: str) -> float:
    rows = _scenario_rows(selected_v71, dataset_type, scenario)
    if rows.empty:
        return np.nan
    return float(pd.to_numeric(rows["final_position_error_m"], errors="coerce").median())


def _check(dataset_type: str, scenario: str, model: str, error: float, previous: float, s7_known_limitation: bool) -> tuple[bool, bool, str]:
    reasons: list[str] = []
    accepted_limitation = False
    if scenario == "S0_static_clean":
        if model not in {"M0_static_position", "M1_static_position_bias"}:
            reasons.append("not_static_family")
        if error > 10.0:
            reasons.append("error_gt_10m")
    elif scenario in {"S1_dynamic_clean", "S2_dynamic_bias_strong"}:
        if np.isfinite(previous) and error > previous * 1.10:
            reasons.append("regressed_gt_10pct_vs_v71")
    elif scenario == "S3_dynamic_outlier":
        if model not in {"M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine", "M2_ctd_full", "M3_ctd_no_drift"}:
            reasons.append("unexpected_outlier_model")
        if np.isfinite(previous) and error > previous * 1.10:
            reasons.append("regressed_gt_10pct_vs_v71")
    elif scenario == "S5_dynamic_hard":
        if model not in {"M2_ctd_full", "M12_ctd_full_plus_gir_refine"}:
            reasons.append("not_ctd_full_family")
        if error > 550.0:
            reasons.append("error_gt_550m")
    elif scenario == "S7_fast_north_no_bias":
        if s7_known_limitation:
            accepted_limitation = True
            reasons.append("declared_known_limitation_oracle_not_truth_free_selectable")
        else:
            if model != "M4_ctd_no_bias":
                reasons.append("not_no_bias")
            if error > 90.0:
                reasons.append("error_gt_90m")
            if np.isfinite(previous) and error >= previous:
                reasons.append("not_improved_vs_v71")
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
    elif scenario == "Q4_qatar_hard_geometry_noise":
        if model != "M2_ctd_full":
            reasons.append("not_ctd_full")
        if error > 1000.0:
            reasons.append("error_gt_1000m")
    elif dataset_type == "real":
        if model not in {"M0_static_position", "M1_static_position_bias"}:
            reasons.append("real_not_static")
        if error > 140.0:
            reasons.append("real_error_gt_140m")
    passed = not reasons
    if accepted_limitation:
        passed = False
    return passed, accepted_limitation, "pass" if passed else ";".join(reasons)


def build_freeze_audit_v72(selected_v72: pd.DataFrame, selected_v71: pd.DataFrame, s7_known_limitation: bool) -> tuple[pd.DataFrame, str]:
    aggregate = _aggregate_focus(selected_v72)
    rows: list[dict[str, Any]] = []
    for dataset_type, scenario, requirement in FOCUS_SCENARIOS:
        current = aggregate[(aggregate["dataset_type"] == dataset_type) & (aggregate["scenario"] == scenario)]
        previous = _previous_error(selected_v71, dataset_type, scenario)
        if current.empty:
            rows.append(
                {
                    "dataset_type": dataset_type,
                    "scenario": scenario,
                    "freeze_requirement": requirement,
                    "selected_model_v72": "",
                    "median_final_position_error_m": np.nan,
                    "oracle_best_position_error_m": np.nan,
                    "selected_minus_oracle_error_m": np.nan,
                    "previous_v71_error_m": previous,
                    "improvement_vs_v71_m": np.nan,
                    "pass_freeze_audit": False,
                    "accepted_as_declared_limitation": False,
                    "freeze_audit_reason": "missing_current_selection",
                }
            )
            continue
        row = current.iloc[0]
        model = str(row.get("selected_model_v72", ""))
        error = _finite(row.get("median_final_position_error_m"), np.nan)
        passed, accepted_limitation, reason = _check(dataset_type, scenario, model, error, previous, s7_known_limitation)
        rows.append(
            {
                "dataset_type": dataset_type,
                "scenario": scenario,
                "freeze_requirement": requirement,
                "selected_model_v72": model,
                "median_final_position_error_m": error,
                "oracle_best_position_error_m": row.get("oracle_best_position_error_m", np.nan),
                "selected_minus_oracle_error_m": row.get("selected_minus_oracle_error_m", np.nan),
                "previous_v71_error_m": previous,
                "improvement_vs_v71_m": previous - error if np.isfinite(previous) and np.isfinite(error) else np.nan,
                "pass_freeze_audit": bool(passed),
                "accepted_as_declared_limitation": bool(accepted_limitation),
                "freeze_audit_reason": reason,
            }
        )
    audit = pd.DataFrame(rows)
    failures = audit[~audit["pass_freeze_audit"].astype(bool)].copy()
    accepted = failures["accepted_as_declared_limitation"].astype(bool).all() if not failures.empty else False
    if failures.empty:
        decision = "freeze_ready"
    elif accepted and set(failures["scenario"].astype(str)) == {"S7_fast_north_no_bias"}:
        decision = "freeze_ready_with_declared_S7_limitation"
    else:
        decision = "do_not_freeze"
    return audit, decision
