"""Model-Adaptive Bias-aware Geometry Trust-Region Doppler Solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model_candidates import CANDIDATE_MODEL_LIST, run_all_candidates
from .model_scoring import ScoringConfig, compute_score


@dataclass
class MaBgtrResult:
    selected_model: str
    selected_state: np.ndarray
    selected_beta: np.ndarray
    selected_score: float
    candidate_results: list[dict[str, Any]]
    candidate_scores: list[dict[str, Any]]
    selection_reason: str
    selected_low_quality: bool
    quality_pass: bool
    physical_plausible: bool
    observability_summary: dict[str, Any]


def _projection_allowed(row: dict[str, Any]) -> bool:
    model = str(row.get("candidate_model", ""))
    if model not in {"M6_be_full", "M9_robust_be_full"}:
        return True
    retention = float(row.get("retention_trace_ratio", np.nan))
    rank_loss = float(row.get("rank_loss", np.nan))
    if np.isfinite(retention) and retention < 0.6:
        return False
    if np.isfinite(rank_loss) and rank_loss > 0:
        return False
    return True


def score_candidates(
    candidate_rows: list[dict[str, Any]],
    n_obs: int,
    config: ScoringConfig | None = None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in candidate_rows:
        score = compute_score(row, n_obs, config)
        merged = dict(row)
        merged.update(score)
        if not _projection_allowed(merged):
            merged["selectable"] = False
            merged["projection_penalty"] = float(merged["projection_penalty"]) + 100.0
            merged["total_score"] = float(merged["total_score"]) + 100.0
            reason = str(merged.get("failure_reason", ""))
            merged["failure_reason"] = ";".join([r for r in [reason, "full_projection_not_allowed"] if r])
        scored.append(merged)
    return scored


def select_candidate(scored_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, str]:
    selectable = [r for r in scored_rows if bool(r.get("selectable", False))]
    high_quality = [r for r in selectable if bool(r.get("quality_pass", False))]
    if high_quality:
        selected = min(high_quality, key=lambda r: float(r["total_score"]))
        return selected, False, "lowest_score_among_quality_pass_candidates"
    if selectable:
        selected = min(selectable, key=lambda r: float(r["total_score"]))
        return selected, True, "no_quality_pass_candidate; lowest_score_among_numerically_selectable"
    selected = min(scored_rows, key=lambda r: float(r.get("total_score", 1e9)))
    return selected, True, "no_selectable_candidate; lowest_score_reported"


def run_ma_bgtr(
    observations: dict[str, Any],
    initial_guess: dict[str, Any],
    config: ScoringConfig | None = None,
    candidate_models: list[str] | None = None,
) -> MaBgtrResult:
    models = candidate_models or CANDIDATE_MODEL_LIST
    candidates = run_all_candidates(
        dataset_type=str(initial_guess["dataset_type"]),
        scenario=str(initial_guess["scenario"]),
        obs=observations,
        position_init_label=str(initial_guess["position_init_label"]),
        p0_init=np.asarray(initial_guess["p0_init"], dtype=float),
        velocity_init_label=str(initial_guess["velocity_init_label"]),
        v_init=np.asarray(initial_guess["v_init"], dtype=float),
        beta_prior_profile=str(initial_guess["beta_prior_profile"]),
        candidate_models=models,
    )
    scored = score_candidates(candidates, len(observations["meas_mps"]), config)
    selected, low_quality, reason = select_candidate(scored)
    beta = np.array(
        [
            float(selected.get("beta0_estimated_mps", np.nan)),
            float(selected.get("beta_dot_estimated_mps2", np.nan)),
        ],
        dtype=float,
    )
    observability = {
        "condition_number": selected.get("condition_number"),
        "retention_trace_ratio": selected.get("retention_trace_ratio"),
        "subspace_coherence_max": selected.get("subspace_coherence_max"),
        "rank_loss": selected.get("rank_loss"),
    }
    return MaBgtrResult(
        selected_model=str(selected["candidate_model"]),
        selected_state=np.array([], dtype=float),
        selected_beta=beta,
        selected_score=float(selected["total_score"]),
        candidate_results=candidates,
        candidate_scores=scored,
        selection_reason=reason,
        selected_low_quality=bool(low_quality),
        quality_pass=bool(selected.get("quality_pass", False)),
        physical_plausible=bool(selected.get("physical_plausible", False)),
        observability_summary=observability,
    )
