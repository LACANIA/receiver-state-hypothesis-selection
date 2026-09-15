"""Outlier evidence enrichment for MA-BGTR-v7.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk_veto import finite_float


def _bool_text(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _scenario_truth_evidence(truth_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if truth_df.empty or "scenario" not in truth_df.columns:
        return {}
    evidence: dict[str, dict[str, Any]] = {}
    for scenario, group in truth_df.groupby("scenario"):
        n = max(len(group), 1)
        qatar_outlier = pd.to_numeric(group.get("qatar_outlier_mps", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dropout = group.get("dropout_flag", False)
        if isinstance(dropout, pd.Series):
            dropout_vals = dropout.map(_bool_text).to_numpy(dtype=bool)
        else:
            dropout_vals = np.zeros(n, dtype=bool)
        conf = pd.to_numeric(group.get("confidence_value_simulated", np.nan), errors="coerce")
        confidence_low = conf < 90.0
        qatar_outlier_ratio = float(np.mean(np.abs(qatar_outlier) > 1.0e-9)) if len(qatar_outlier) else 0.0
        dropout_ratio = float(np.mean(dropout_vals)) if len(dropout_vals) else 0.0
        confidence_low_ratio = float(np.mean(confidence_low.fillna(False).to_numpy(dtype=bool))) if len(confidence_low) else 0.0
        recipe = str(group.get("qatar_recipe_name", pd.Series([""])).iloc[0]) if "qatar_recipe_name" in group.columns and len(group) else ""
        burst = "burst" in str(scenario).lower() or "burst" in recipe.lower()
        evidence[str(scenario)] = {
            "qatar_outlier_ratio": qatar_outlier_ratio,
            "dropout_ratio": dropout_ratio,
            "confidence_low_ratio": confidence_low_ratio,
            "burst_outlier_evidence": bool(burst),
            "qatar_recipe_name": recipe,
        }
    return evidence


def load_qatar_truth_evidence(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    usecols = None
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        wanted = [c for c in ["scenario", "qatar_outlier_mps", "dropout_flag", "confidence_value_simulated", "qatar_recipe_name"] if c in header]
        usecols = wanted if wanted else None
    except Exception:
        usecols = None
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        return {}
    return _scenario_truth_evidence(df)


def augment_outlier_evidence(candidates: pd.DataFrame, qatar_truth_path: Path | None = None) -> pd.DataFrame:
    rows = candidates.copy()
    qatar_evidence = load_qatar_truth_evidence(qatar_truth_path) if qatar_truth_path is not None else {}
    outlier_flags: list[bool] = []
    burst_flags: list[bool] = []
    q_ratios: list[float] = []
    d_ratios: list[float] = []
    c_ratios: list[float] = []
    sources: list[str] = []
    reasons: list[str] = []
    for _, row in rows.iterrows():
        scenario = str(row.get("scenario", ""))
        tail = finite_float(row.get("tail_ratio"), np.nan)
        mad = finite_float(row.get("mad_ratio"), np.nan)
        qev = qatar_evidence.get(scenario, {})
        q_ratio = float(qev.get("qatar_outlier_ratio", np.nan)) if qev else np.nan
        d_ratio = float(qev.get("dropout_ratio", np.nan)) if qev else np.nan
        c_ratio = float(qev.get("confidence_low_ratio", np.nan)) if qev else np.nan
        burst = bool(qev.get("burst_outlier_evidence", False)) or "burst_outlier" in scenario.lower()
        reason_parts: list[str] = []
        source_parts: list[str] = []
        outlier = False
        if np.isfinite(tail) and tail >= 0.05:
            outlier = True
            reason_parts.append("tail_ratio>=0.05")
            source_parts.append("candidate_tail_ratio")
        if np.isfinite(mad) and mad >= 2.0:
            outlier = True
            reason_parts.append("mad_ratio>=2")
            source_parts.append("candidate_mad_ratio")
        if "outlier" in scenario.lower():
            outlier = True
            reason_parts.append("scenario_name_outlier")
            source_parts.append("scenario_name")
        if np.isfinite(q_ratio) and q_ratio >= 0.03:
            outlier = True
            reason_parts.append("qatar_outlier_ratio>=0.03")
            source_parts.append("data03_truth")
        if burst:
            outlier = True
            reason_parts.append("burst_outlier_evidence")
            source_parts.append("data03_truth_or_scenario")
        outlier_flags.append(bool(outlier))
        burst_flags.append(bool(burst))
        q_ratios.append(q_ratio)
        d_ratios.append(d_ratio)
        c_ratios.append(c_ratio)
        sources.append(";".join(dict.fromkeys(source_parts)) if source_parts else "none")
        reasons.append(";".join(dict.fromkeys(reason_parts)) if reason_parts else "no_outlier_evidence")
    rows["outlier_evidence"] = outlier_flags
    rows["burst_outlier_evidence"] = burst_flags
    rows["qatar_outlier_ratio"] = q_ratios
    rows["dropout_ratio"] = d_ratios
    rows["confidence_low_ratio"] = c_ratios
    rows["evidence_source"] = sources
    rows["outlier_evidence_reason"] = reasons
    return rows
