"""MA-BGTR-v5 risk-calibrated selection layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ephemeris_covariance import compute_ephemeris_diagnostics
from .geometry_selection import compute_geometry_selection_diagnostics
from .model_scoring import V5ScoringConfig, compute_score_v5
from .stability_diagnostics import compute_stability_diagnostics, stability_features_for_candidates
from .uncertainty_diagnostics import compute_uncertainty_diagnostics


GROUP_KEYS = ["dataset_type", "scenario", "position_init_label", "velocity_init_label", "beta_prior_profile"]
ROW_KEYS = [*GROUP_KEYS, "candidate_model"]
PENALTY_ERROR_M = 1e7


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def augment_candidates_with_risk(candidate_df: pd.DataFrame, observations: dict[tuple[str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach STEP10 risk diagnostics and v5 scores to STEP09 rows."""
    rows = candidate_df.copy()
    uncertainty_rows: list[dict[str, Any]] = []
    ephemeris_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for record in rows.to_dict(orient="records"):
        key = (str(record.get("dataset_type")), str(record.get("scenario")))
        obs = observations.get(key)
        base_key = {col: record.get(col) for col in ROW_KEYS if col in record}
        if obs is None:
            uncertainty = compute_uncertainty_diagnostics(record, {"time_s": np.array([]), "sat_pos_m": np.empty((0, 3)), "sat_vel_mps": np.empty((0, 3)), "meas_mps": np.array([])})
            ephemeris = {"median_R_eph": np.nan, "p95_R_eph": np.nan, "ephemeris_weight_effect": np.nan, "ephemeris_sensitivity_status": "missing_observations"}
            geometry = {
                "residual_trim_info_retention": np.nan,
                "geometry_preserved_info_retention": np.nan,
                "residual_trim_condition_change": np.nan,
                "geometry_preserved_condition_change": np.nan,
                "geometry_critical_outlier_count": 0,
                "geometry_selection_recommendation": "missing_observations",
            }
        else:
            uncertainty = compute_uncertainty_diagnostics(record, obs)
            ephemeris = compute_ephemeris_diagnostics(record, obs)
            geometry = compute_geometry_selection_diagnostics(record, obs)
        uncertainty_rows.append({**base_key, **uncertainty})
        ephemeris_rows.append({**base_key, **ephemeris})
        geometry_rows.append({**base_key, **geometry})

    uncertainty_df = pd.DataFrame(uncertainty_rows)
    ephemeris_df = pd.DataFrame(ephemeris_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    stability_df = compute_stability_diagnostics(rows)
    stability_features = stability_features_for_candidates(rows, stability_df)

    for diag_df in [uncertainty_df, ephemeris_df, geometry_df, stability_features]:
        rows = rows.merge(diag_df, on=[c for c in ROW_KEYS if c in diag_df.columns], how="left", suffixes=("", "_diag"))

    score_rows = [compute_score_v5(record, n_val=128, config=V5ScoringConfig()) for record in rows.to_dict(orient="records")]
    score_df = pd.DataFrame(score_rows)
    rows = pd.concat([rows.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
    if "gate_block_reason" not in rows.columns:
        rows["gate_block_reason"] = ""
    rows["gate_block_reason"] = rows["gate_block_reason"].fillna("")
    blocked = []
    for record in rows.to_dict(orient="records"):
        reasons: list[str] = []
        if not bool(record.get("selectable_v4", False)):
            reasons.append("v4_not_selectable")
        if not bool(record.get("covariance_available", False)):
            reasons.append("covariance_unavailable")
        if not bool(record.get("selectable_v5", False)):
            reasons.append("v5_risk_gate")
        blocked.append(";".join(dict.fromkeys([str(record.get("gate_block_reason", "")).strip(), *reasons])) .strip(";"))
    rows["gate_block_reason"] = blocked
    return rows, uncertainty_df, stability_df, ephemeris_df, geometry_df


def select_v5_results(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for keys, group in candidate_df.groupby(GROUP_KEYS, dropna=False, sort=True):
        group = group.copy()
        selectable = group[group["selectable_v5"].astype(bool)].sort_values("total_score_v5")
        if not selectable.empty:
            selected = selectable.iloc[0]
            low_quality = False
            reason = "lowest_v5_risk_score_after_uncertainty_stability_ephemeris_geometry"
        else:
            finite = group[np.isfinite(pd.to_numeric(group["total_score_v5"], errors="coerce"))].sort_values("total_score_v5")
            selected = finite.iloc[0] if not finite.empty else group.iloc[0]
            low_quality = True
            reason = "no_selectable_candidate; lowest_reported_v5_score"
        runner_pool = group[group.index != selected.name].sort_values("total_score_v5")
        runner = runner_pool.iloc[0] if not runner_pool.empty else selected
        selected_error = _finite_float(selected.get("final_position_error_m"), np.nan)
        candidate_errors = pd.to_numeric(group["final_position_error_m"], errors="coerce").to_numpy(dtype=float)
        candidate_errors = np.where(np.isfinite(candidate_errors), candidate_errors, PENALTY_ERROR_M)
        oracle_error = float(np.min(candidate_errors)) if candidate_errors.size else np.nan
        ctd = group[group["candidate_model"] == "M2_ctd_full"]
        static = group[group["candidate_model"] == "M0_static_position"]
        ctd_error = _finite_float(ctd.iloc[0].get("final_position_error_m"), np.nan) if not ctd.empty else np.nan
        static_error = _finite_float(static.iloc[0].get("final_position_error_m"), np.nan) if not static.empty else np.nan
        selected_rows.append(
            {
                "dataset_type": keys[0],
                "scenario": keys[1],
                "position_init_label": keys[2],
                "velocity_init_label": keys[3],
                "beta_prior_profile": keys[4],
                "selected_model_v5": selected.get("candidate_model"),
                "selected_score_v5": selected.get("total_score_v5"),
                "selected_low_quality": bool(low_quality),
                "selection_reason_v5": reason,
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
                "position_cov_sqrt_trace_m": selected.get("position_cov_sqrt_trace_m", np.nan),
                "position_bootstrap_spread_m": selected.get("position_bootstrap_spread_m", np.nan),
                "geometry_preserved_info_retention": selected.get("geometry_preserved_info_retention", np.nan),
                "quality_pass": selected.get("quality_pass", False),
                "physical_plausible": selected.get("physical_plausible", False),
                "runtime_ms_total": float(pd.to_numeric(group.get("runtime_ms", 0.0), errors="coerce").fillna(0.0).sum()),
                "oracle_best_position_error_m": oracle_error,
                "selected_minus_oracle_error_m": selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else np.nan,
                "selected_beats_ctd": bool(np.isfinite(selected_error) and np.isfinite(ctd_error) and selected_error <= ctd_error),
                "selected_beats_static_lm": bool(np.isfinite(selected_error) and np.isfinite(static_error) and selected_error <= static_error),
            }
        )
        score_rows.append(
            {
                "dataset_type": keys[0],
                "scenario": keys[1],
                "position_init_label": keys[2],
                "velocity_init_label": keys[3],
                "beta_prior_profile": keys[4],
                "selected_model_v5": selected.get("candidate_model"),
                "runner_up_model_v5": runner.get("candidate_model"),
                "selected_total_score_v5": selected.get("total_score_v5"),
                "runner_up_total_score_v5": runner.get("total_score_v5"),
                "score_margin": _finite_float(runner.get("total_score_v5"), 0.0) - _finite_float(selected.get("total_score_v5"), 0.0),
                "selected_validation_component": selected.get("validation_component", np.nan),
                "selected_uncertainty_penalty": selected.get("uncertainty_penalty", np.nan),
                "selected_bootstrap_stability_penalty": selected.get("bootstrap_stability_penalty", np.nan),
                "selected_ephemeris_sensitivity_penalty": selected.get("ephemeris_sensitivity_penalty", np.nan),
                "selected_geometry_information_penalty": selected.get("geometry_information_penalty", np.nan),
                "selected_overfit_risk_penalty": selected.get("overfit_risk_penalty", np.nan),
            }
        )
    return pd.DataFrame(selected_rows), pd.DataFrame(score_rows)


def entropy(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    probs = values / max(float(np.sum(values)), 1.0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def aggregate_v5_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in selected_df.groupby(["dataset_type", "scenario"], sort=True):
        counts = group["selected_model_v5"].value_counts()
        errors = pd.to_numeric(group["final_position_error_m"], errors="coerce").to_numpy(dtype=float)
        errors = np.where(np.isfinite(errors), errors, PENALTY_ERROR_M)
        rows.append(
            {
                "dataset_type": keys[0],
                "scenario": keys[1],
                "runs": int(len(group)),
                "quality_pass_rate": float(group["quality_pass"].astype(bool).mean()),
                "median_final_position_error_m": float(np.median(errors)),
                "p95_final_position_error_m": float(np.quantile(errors, 0.95)),
                "median_mean_position_error_m": float(pd.to_numeric(group["mean_position_error_m"], errors="coerce").median()),
                "median_velocity_error_mps": float(pd.to_numeric(group["velocity_error_mps"], errors="coerce").median()),
                "median_beta0_error_mps": float(pd.to_numeric(group["beta0_error_mps"], errors="coerce").median()),
                "median_beta_dot_error_mps2": float(pd.to_numeric(group["beta_dot_error_mps2"], errors="coerce").median()),
                "median_trimmed_validation_rmse_mps": float(pd.to_numeric(group["trimmed_validation_rmse_mps"], errors="coerce").median()),
                "median_position_cov_sqrt_trace_m": float(pd.to_numeric(group["position_cov_sqrt_trace_m"], errors="coerce").median()),
                "median_position_bootstrap_spread_m": float(pd.to_numeric(group["position_bootstrap_spread_m"], errors="coerce").median()),
                "median_geometry_preserved_info_retention": float(pd.to_numeric(group["geometry_preserved_info_retention"], errors="coerce").median()),
                "most_selected_model_v5": str(counts.index[0]) if len(counts) else "",
                "selected_model_entropy_v5": entropy(counts),
                "oracle_best_position_error_m": float(pd.to_numeric(group["oracle_best_position_error_m"], errors="coerce").median()),
                "selected_minus_oracle_error_m": float(pd.to_numeric(group["selected_minus_oracle_error_m"], errors="coerce").median()),
                "selected_beats_ctd_rate": float(group["selected_beats_ctd"].astype(bool).mean()),
                "selected_beats_static_lm_rate": float(group["selected_beats_static_lm"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_selection_counts(selected_df: pd.DataFrame) -> pd.DataFrame:
    counts = selected_df.groupby(["dataset_type", "scenario", "selected_model_v5"]).size().reset_index(name="count")
    totals = counts.groupby(["dataset_type", "scenario"])["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts
