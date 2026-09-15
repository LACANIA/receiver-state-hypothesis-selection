"""S7 evidence diagnostics for TECH14."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .candidate_alignment import (
    BE_MODELS,
    CORE_FINAL_MODELS,
    DIRECT_GIR_MODELS,
    GROUP_KEYS,
    RISK_METRICS,
    S7_SCENARIO,
    VALIDATION_METRICS,
    best_by_metric,
    best_risk_candidate,
    bool_value,
    finite_float,
    model_family,
    selected_model_map,
    spearman_rank_corr,
    validation_score,
)


NUMERIC_COLUMNS = [
    "final_position_error_m",
    "mean_position_error_m",
    "velocity_error_mps",
    "estimated_speed_mps",
    "beta0_estimated_mps",
    "beta_dot_estimated_mps2",
    "raw_validation_rmse_mps",
    "trimmed_validation_rmse_mps",
    "inlier_validation_rmse_mps",
    "robust_validation_cost",
    "condition_number",
    "position_cov_sqrt_trace_m",
    "position_bootstrap_spread_m",
    "information_retention_ratio",
    "geometry_preserved_info_retention",
]


def s7_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = candidate_df[(candidate_df["dataset_type"] == "synthetic") & (candidate_df["scenario"] == S7_SCENARIO)].copy()
    for col in NUMERIC_COLUMNS:
        if col not in rows.columns:
            rows[col] = np.nan
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    for col in ["numerical_success", "quality_pass", "physical_plausible"]:
        if col not in rows.columns:
            rows[col] = False
    rows["model_family"] = rows["candidate_model"].map(model_family)
    rows["_validation_score"] = rows.apply(validation_score, axis=1)
    return rows


def _rank_within_group(rows: pd.DataFrame, metric: str, rank_col: str) -> pd.Series:
    if metric not in rows.columns:
        return pd.Series(np.nan, index=rows.index)
    return rows.groupby(["position_init_label", "velocity_init_label", "beta_prior_profile"])[metric].rank(method="min", ascending=True)


def _oracle_truth_free_reason(row: pd.Series, group: pd.DataFrame) -> tuple[bool, str]:
    model = str(row.get("candidate_model", ""))
    if model in DIRECT_GIR_MODELS:
        return False, "direct_gir_local_only_not_allowed_as_global_selector"
    if model in BE_MODELS:
        return False, "be_projection_branch_unstable_across_initializations"
    if model not in CORE_FINAL_MODELS:
        return False, "not_core_final_candidate"
    if not bool_value(row.get("numerical_success"), True) or not bool_value(row.get("quality_pass"), True) or not bool_value(row.get("physical_plausible"), True):
        return False, "failed_basic_quality_or_physical_gate"
    ranks = []
    for metric in VALIDATION_METRICS:
        ranks.append(int(group[metric].rank(method="min", ascending=True).loc[row.name]))
    if max(ranks) <= 3:
        return True, "oracle_near_top_validation_evidence"
    best_validation = group.sort_values("_validation_score").iloc[0]
    oracle_score = finite_float(row.get("_validation_score"), np.nan)
    best_score = finite_float(best_validation.get("_validation_score"), np.nan)
    if np.isfinite(oracle_score) and np.isfinite(best_score) and oracle_score <= best_score * 1.01:
        return True, "oracle_near_tie_validation_score"
    return False, f"validation_ranks_not_near_top:{'/'.join(str(r) for r in ranks)}"


def build_s7_candidate_leaderboard(candidate_df: pd.DataFrame, v7_selected: pd.DataFrame, v71_selected: pd.DataFrame) -> pd.DataFrame:
    rows = s7_candidates(candidate_df)
    v7_map = selected_model_map(v7_selected, "selected_model_v7")
    v71_map = selected_model_map(v71_selected, "selected_model_v71")
    for rank_col, metric in [
        ("rank_by_true_error", "final_position_error_m"),
        ("rank_by_raw_validation", "raw_validation_rmse_mps"),
        ("rank_by_trimmed_validation", "trimmed_validation_rmse_mps"),
        ("rank_by_inlier_validation", "inlier_validation_rmse_mps"),
        ("rank_by_condition", "condition_number"),
    ]:
        rows[rank_col] = _rank_within_group(rows, metric, rank_col)
    rows["is_oracle_candidate"] = rows["rank_by_true_error"] == 1
    selected_v7 = []
    selected_v71 = []
    truth_free = []
    for idx, row in rows.iterrows():
        key = tuple(row.get(k) for k in GROUP_KEYS)
        selected_v7.append(str(row.get("candidate_model")) == v7_map.get(key, ""))
        selected_v71.append(str(row.get("candidate_model")) == v71_map.get(key, ""))
        group = rows[
            (rows["position_init_label"] == row.get("position_init_label"))
            & (rows["velocity_init_label"] == row.get("velocity_init_label"))
            & (rows["beta_prior_profile"] == row.get("beta_prior_profile"))
        ]
        ok, _reason = _oracle_truth_free_reason(row, group)
        truth_free.append(bool(ok))
    rows["selected_by_v7"] = selected_v7
    rows["selected_by_v71"] = selected_v71
    rows["selectable_by_truth_free_evidence"] = truth_free
    cols = [
        "scenario",
        "candidate_model",
        "model_family",
        "position_init_label",
        "velocity_init_label",
        "beta_prior_profile",
        "final_position_error_m",
        "mean_position_error_m",
        "velocity_error_mps",
        "estimated_speed_mps",
        "beta0_estimated_mps",
        "beta_dot_estimated_mps2",
        "raw_validation_rmse_mps",
        "trimmed_validation_rmse_mps",
        "inlier_validation_rmse_mps",
        "robust_validation_cost",
        "condition_number",
        "position_cov_sqrt_trace_m",
        "position_bootstrap_spread_m",
        "information_retention_ratio",
        "geometry_preserved_info_retention",
        "numerical_success",
        "quality_pass",
        "physical_plausible",
        "rank_by_true_error",
        "rank_by_raw_validation",
        "rank_by_trimmed_validation",
        "rank_by_inlier_validation",
        "rank_by_condition",
        "is_oracle_candidate",
        "selected_by_v7",
        "selected_by_v71",
        "selectable_by_truth_free_evidence",
    ]
    for col in cols:
        if col not in rows.columns:
            rows[col] = np.nan
    return rows[cols].sort_values(["rank_by_true_error", "position_init_label", "velocity_init_label", "beta_prior_profile", "candidate_model"])


def _selected_error(selected: pd.DataFrame, key_cols: dict[str, Any], model_col: str) -> tuple[str, float]:
    if selected.empty or model_col not in selected.columns:
        return "", np.nan
    mask = pd.Series(True, index=selected.index)
    for key, value in key_cols.items():
        mask &= selected[key] == value
    subset = selected[mask]
    if subset.empty:
        return "", np.nan
    row = subset.iloc[0]
    return str(row.get(model_col, "")), finite_float(row.get("final_position_error_m"), np.nan)


def build_s7_oracle_trace(candidate_df: pd.DataFrame, v7_selected: pd.DataFrame, v71_selected: pd.DataFrame) -> pd.DataFrame:
    rows = s7_candidates(candidate_df)
    out: list[dict[str, Any]] = []
    for keys, group in rows.groupby(["position_init_label", "velocity_init_label", "beta_prior_profile"], sort=True):
        group = group.copy()
        oracle = group.sort_values("final_position_error_m").iloc[0]
        best_raw = best_by_metric(group, "raw_validation_rmse_mps")
        best_trim = best_by_metric(group, "trimmed_validation_rmse_mps")
        best_inlier = best_by_metric(group, "inlier_validation_rmse_mps")
        best_risk = best_risk_candidate(group)
        ok, reason = _oracle_truth_free_reason(oracle, group)
        key_cols = {
            "dataset_type": "synthetic",
            "scenario": S7_SCENARIO,
            "position_init_label": keys[0],
            "velocity_init_label": keys[1],
            "beta_prior_profile": keys[2],
        }
        v7_model, v7_error = _selected_error(v7_selected, key_cols, "selected_model_v7")
        v71_model, v71_error = _selected_error(v71_selected, key_cols, "selected_model_v71")
        out.append(
            {
                "position_init_label": keys[0],
                "velocity_init_label": keys[1],
                "beta_prior_profile": keys[2],
                "oracle_model": oracle.get("candidate_model"),
                "oracle_error_m": oracle.get("final_position_error_m"),
                "v7_selected_model": v7_model,
                "v7_error_m": v7_error,
                "v71_selected_model": v71_model,
                "v71_error_m": v71_error,
                "best_raw_validation_model": best_raw.get("candidate_model"),
                "best_trimmed_validation_model": best_trim.get("candidate_model"),
                "best_inlier_validation_model": best_inlier.get("candidate_model"),
                "best_risk_model": best_risk.get("candidate_model"),
                "oracle_rank_by_raw": int(group["raw_validation_rmse_mps"].rank(method="min").loc[oracle.name]),
                "oracle_rank_by_trimmed": int(group["trimmed_validation_rmse_mps"].rank(method="min").loc[oracle.name]),
                "oracle_rank_by_inlier": int(group["inlier_validation_rmse_mps"].rank(method="min").loc[oracle.name]),
                "oracle_rank_by_condition": int(group["condition_number"].rank(method="min").loc[oracle.name]),
                "oracle_truth_free_selectable": bool(ok),
                "oracle_trace_reason": reason,
            }
        )
    return pd.DataFrame(out)


def build_validation_truth_alignment(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = s7_candidates(candidate_df)
    scopes = {
        "all_candidates": rows,
        "core_final_models": rows[rows["candidate_model"].isin(CORE_FINAL_MODELS)].copy(),
        "core_east_10km": rows[(rows["candidate_model"].isin(CORE_FINAL_MODELS)) & (rows["position_init_label"] == "east_10km")].copy(),
        "core_truth_init": rows[(rows["candidate_model"].isin(CORE_FINAL_MODELS)) & (rows["position_init_label"] == "truth_init")].copy(),
    }
    metrics = [
        "raw_validation_rmse_mps",
        "trimmed_validation_rmse_mps",
        "inlier_validation_rmse_mps",
        "robust_validation_cost",
        "condition_number",
        "position_cov_sqrt_trace_m",
        "position_bootstrap_spread_m",
    ]
    out: list[dict[str, Any]] = []
    for scope, subset in scopes.items():
        for metric in metrics:
            corr = spearman_rank_corr(subset["final_position_error_m"], subset[metric])
            if not np.isfinite(corr):
                interpretation = "insufficient_data"
                usable = False
            elif corr >= 0.5:
                interpretation = "positive_alignment"
                usable = scope == "core_final_models"
            elif corr <= -0.3:
                interpretation = "inverse_alignment_selection_risk"
                usable = False
            else:
                interpretation = "weak_alignment"
                usable = False
            out.append(
                {
                    "scenario": S7_SCENARIO,
                    "metric_name": f"{metric}__{scope}",
                    "spearman_corr_with_position_error": corr,
                    "interpretation": interpretation,
                    "usable_for_selection": bool(usable),
                }
            )
    return pd.DataFrame(out)


def summarize_s7_diagnostics(leaderboard: pd.DataFrame, oracle_trace: pd.DataFrame, alignment: pd.DataFrame) -> dict[str, Any]:
    all_oracles = leaderboard[leaderboard["is_oracle_candidate"]].copy()
    core_rows = leaderboard[leaderboard["candidate_model"].isin(CORE_FINAL_MODELS)].copy()
    core_oracles = core_rows.sort_values("final_position_error_m").head(12)
    oracle_truth_free_rate = float(oracle_trace["oracle_truth_free_selectable"].mean()) if not oracle_trace.empty else 0.0
    core_alignment = alignment[alignment["metric_name"].str.endswith("__core_final_models")]
    inverse_metrics = core_alignment[core_alignment["interpretation"] == "inverse_alignment_selection_risk"]["metric_name"].tolist()
    direct_gir_oracle_count = int(all_oracles["candidate_model"].isin(DIRECT_GIR_MODELS).sum()) if not all_oracles.empty else 0
    truth_free_rule_exists = bool(oracle_truth_free_rate >= 0.8 and not inverse_metrics)
    conclusion = (
        "truth_free_rule_available"
        if truth_free_rule_exists
        else "S7 oracle not selectable by current validation evidence; v7.2 should declare S7 as a known limitation"
    )
    return {
        "scenario": S7_SCENARIO,
        "candidate_rows": int(len(leaderboard)),
        "oracle_candidate_models": sorted(all_oracles["candidate_model"].dropna().astype(str).unique().tolist()),
        "direct_gir_oracle_count": direct_gir_oracle_count,
        "core_best_models_by_error": core_oracles[["candidate_model", "final_position_error_m", "position_init_label", "velocity_init_label", "beta_prior_profile"]].head(10).to_dict(orient="records"),
        "oracle_truth_free_selectable_rate": oracle_truth_free_rate,
        "core_inverse_alignment_metrics": inverse_metrics,
        "truth_free_rule_exists": truth_free_rule_exists,
        "conclusion": conclusion,
    }
