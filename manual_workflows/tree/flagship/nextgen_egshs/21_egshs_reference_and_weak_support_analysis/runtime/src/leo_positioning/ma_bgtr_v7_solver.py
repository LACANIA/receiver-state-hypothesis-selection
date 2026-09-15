"""MA-BGTR-v7 hierarchical evidence-gated selector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .hierarchical_gates import HierarchicalGateConfig, select_group_v7
from .risk_veto import finite_float


GROUP_KEYS = ["dataset_type", "scenario", "position_init_label", "velocity_init_label", "beta_prior_profile"]
PENALTY_ERROR_M = 1.0e7


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_series(series: pd.Series, penalty: float = PENALTY_ERROR_M) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, penalty)


def _first_error(group: pd.DataFrame, model: str) -> float:
    subset = group[group["candidate_model"] == model]
    if subset.empty:
        return np.nan
    return finite_float(subset.iloc[0].get("final_position_error_m"), np.nan)


def _selected_result(keys: tuple[Any, ...], group: pd.DataFrame, selected: pd.Series, reason: str, low_quality: bool) -> dict[str, Any]:
    selected_error = finite_float(selected.get("final_position_error_m"), np.nan)
    oracle_errors = _finite_series(group["final_position_error_m"]) if "final_position_error_m" in group.columns else np.array([PENALTY_ERROR_M])
    oracle_error = float(np.min(oracle_errors)) if oracle_errors.size else np.nan
    ctd_error = _first_error(group, "M2_ctd_full")
    static_error = _first_error(group, "M0_static_position")
    return {
        "dataset_type": keys[0],
        "scenario": keys[1],
        "position_init_label": keys[2],
        "velocity_init_label": keys[3],
        "beta_prior_profile": keys[4],
        "selected_model_v7": selected.get("candidate_model"),
        "selection_reason_v7": reason,
        "selected_low_quality": bool(low_quality),
        "true_speed_mps": selected.get("true_speed_mps", np.nan),
        "true_b0_mps": selected.get("true_b0_mps", np.nan),
        "true_bdot_mps2": selected.get("true_bdot_mps2", np.nan),
        "final_position_error_m": selected.get("final_position_error_m", np.nan),
        "mean_position_error_m": selected.get("mean_position_error_m", np.nan),
        "max_position_error_m": selected.get("max_position_error_m", np.nan),
        "velocity_error_mps": selected.get("velocity_error_mps", np.nan),
        "estimated_speed_mps": selected.get("estimated_speed_mps", np.nan),
        "beta0_estimated_mps": selected.get("beta0_estimated_mps", np.nan),
        "beta_dot_estimated_mps2": selected.get("beta_dot_estimated_mps2", np.nan),
        "beta0_error_mps": selected.get("beta0_error_mps", np.nan),
        "beta_dot_error_mps2": selected.get("beta_dot_error_mps2", np.nan),
        "raw_validation_rmse_mps": selected.get("raw_validation_rmse_mps", np.nan),
        "trimmed_validation_rmse_mps": selected.get("trimmed_validation_rmse_mps", np.nan),
        "inlier_validation_rmse_mps": selected.get("inlier_validation_rmse_mps", np.nan),
        "full_residual_rmse_mps": selected.get("full_residual_rmse_mps", selected.get("residual_rmse_mps", np.nan)),
        "condition_number": selected.get("condition_number", np.nan),
        "quality_pass": selected.get("quality_pass", False),
        "physical_plausible": selected.get("physical_plausible", False),
        "oracle_best_position_error_m": oracle_error,
        "selected_minus_oracle_error_m": selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else np.nan,
        "selected_beats_ctd": bool(np.isfinite(selected_error) and np.isfinite(ctd_error) and selected_error <= ctd_error),
        "selected_beats_static_lm": bool(np.isfinite(selected_error) and np.isfinite(static_error) and selected_error <= static_error),
    }


def select_v7_results(candidate_df: pd.DataFrame, config: HierarchicalGateConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or HierarchicalGateConfig()
    selected_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    for keys, group in candidate_df.groupby(GROUP_KEYS, dropna=False, sort=True):
        selected, family_diag, robust_diag, reason, low_quality = select_group_v7(group, cfg)
        selected_rows.append(_selected_result(keys, group, selected, reason, low_quality))
        family_rows.append(family_diag)
        robust_rows.extend(robust_diag)
    return pd.DataFrame(selected_rows), pd.DataFrame(family_rows), pd.DataFrame(robust_rows)


def entropy(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def aggregate_v7_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in selected_df.groupby(["dataset_type", "scenario"], sort=True):
        counts = group["selected_model_v7"].value_counts()
        errors = _finite_series(group["final_position_error_m"])
        rows.append(
            {
                "dataset_type": keys[0],
                "scenario": keys[1],
                "runs": int(len(group)),
                "quality_pass_rate": float(group["quality_pass"].map(_safe_bool).mean()),
                "median_final_position_error_m": float(np.median(errors)),
                "p95_final_position_error_m": float(np.quantile(errors, 0.95)),
                "median_mean_position_error_m": float(pd.to_numeric(group["mean_position_error_m"], errors="coerce").median()),
                "median_velocity_error_mps": float(pd.to_numeric(group["velocity_error_mps"], errors="coerce").median()),
                "median_beta0_error_mps": float(pd.to_numeric(group["beta0_error_mps"], errors="coerce").median()),
                "median_beta_dot_error_mps2": float(pd.to_numeric(group["beta_dot_error_mps2"], errors="coerce").median()),
                "median_trimmed_validation_rmse_mps": float(pd.to_numeric(group["trimmed_validation_rmse_mps"], errors="coerce").median()),
                "most_selected_model_v7": str(counts.index[0]) if len(counts) else "",
                "selected_model_entropy_v7": entropy(counts),
                "oracle_best_position_error_m": float(pd.to_numeric(group["oracle_best_position_error_m"], errors="coerce").median()),
                "selected_minus_oracle_error_m": float(pd.to_numeric(group["selected_minus_oracle_error_m"], errors="coerce").median()),
                "selected_beats_ctd_rate": float(group["selected_beats_ctd"].map(_safe_bool).mean()),
                "selected_beats_static_lm_rate": float(group["selected_beats_static_lm"].map(_safe_bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_selection_counts_v7(selected_df: pd.DataFrame) -> pd.DataFrame:
    counts = selected_df.groupby(["dataset_type", "scenario", "selected_model_v7"]).size().reset_index(name="count")
    totals = counts.groupby(["dataset_type", "scenario"])["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts


def compare_v6_v7(v6_selected: pd.DataFrame, v7_selected: pd.DataFrame) -> pd.DataFrame:
    if v6_selected.empty or v7_selected.empty:
        return pd.DataFrame()
    merged = v6_selected.merge(v7_selected, on=GROUP_KEYS, how="inner", suffixes=("_v6", "_v7"))
    return pd.DataFrame(
        {
            "dataset_type": merged["dataset_type"],
            "scenario": merged["scenario"],
            "position_init_label": merged["position_init_label"],
            "velocity_init_label": merged["velocity_init_label"],
            "beta_prior_profile": merged["beta_prior_profile"],
            "selected_model_v6": merged["selected_model_v6"],
            "selected_model_v7": merged["selected_model_v7"],
            "v6_final_position_error_m": merged["final_position_error_m_v6"],
            "v7_final_position_error_m": merged["final_position_error_m_v7"],
            "improvement_m": merged["final_position_error_m_v6"] - merged["final_position_error_m_v7"],
            "v6_trimmed_validation_rmse_mps": merged["trimmed_validation_rmse_mps_v6"],
            "v7_trimmed_validation_rmse_mps": merged["trimmed_validation_rmse_mps_v7"],
            "v7_selection_reason": merged["selection_reason_v7"],
        }
    )
