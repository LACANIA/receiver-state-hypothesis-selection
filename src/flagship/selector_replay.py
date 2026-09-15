"""Independent publication-spec interpreter for MA-BGTR Mode A.

This module intentionally does not import the frozen selector package.  It is a
literal executable transcription of the v0.5.2 JSON/CSV publication contract
and is used only to replay the 60 locked candidate-output groups.  It never
reads receiver truth, position error, or oracle fields.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATIC = {"M0_static_position", "M1_static_position_bias"}
NO_BIAS = {"M4_ctd_no_bias"}
NO_DRIFT = {"M3_ctd_no_drift"}
FULL_DRIFT = {
    "M2_ctd_full", "M7_robust_ctd_full",
    "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine",
}
BIAS_CAPABLE = {
    "M2_ctd_full", "M3_ctd_no_drift", "M7_robust_ctd_full",
    "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine",
}
ROBUST = {
    "M7_robust_ctd_full", "M8_robust_be_b0_only", "M9_robust_be_full",
    "M13_robust_ctd_full_plus_gir_refine",
}
REFINE = {
    "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine",
    "M14_static_plus_gir_refine",
}
CORE_DYNAMIC = {
    "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias",
    "M7_robust_ctd_full", "M12_ctd_full_plus_gir_refine",
    "M13_robust_ctd_full_plus_gir_refine",
}
CORE_FINAL = STATIC | CORE_DYNAMIC
BE_FULL = {"M6_be_full", "M9_robust_be_full"}
DIRECT_GIR = {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}
REAL_FORBIDDEN_GIR = DIRECT_GIR | {"M14_static_plus_gir_refine"}
DYNAMIC = {
    "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M5_be_b0_only",
    "M6_be_full", "M7_robust_ctd_full", "M8_robust_be_b0_only",
    "M9_robust_be_full", "M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale",
    "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine",
    "M14_static_plus_gir_refine",
}
ROBUST_V4 = {
    "M7_robust_ctd_full", "M8_robust_be_b0_only", "M9_robust_be_full",
    "M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale",
    "M13_robust_ctd_full_plus_gir_refine",
}
MODEL_DOF = {
    "M0_static_position": 3, "M1_static_position_bias": 4,
    "M2_ctd_full": 8, "M3_ctd_no_drift": 7, "M4_ctd_no_bias": 6,
    "M5_be_b0_only": 6, "M6_be_full": 6, "M7_robust_ctd_full": 8,
    "M8_robust_be_b0_only": 6, "M9_robust_be_full": 6,
    "M10_gir_tr_fixed_scale": 8, "M11_gir_tr_mad_scale": 8,
    "M12_ctd_full_plus_gir_refine": 8,
    "M13_robust_ctd_full_plus_gir_refine": 8,
    "M14_static_plus_gir_refine": 8,
}
METRICS = [
    "raw_validation_rmse_mps", "trimmed_validation_rmse_mps",
    "inlier_validation_rmse_mps",
]


def long_path(path: Path) -> Path:
    text = str(path)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\" + text)
    return path


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def dof(row: pd.Series) -> int:
    v = finite(row.get("model_dof"), np.nan)
    return int(v) if np.isfinite(v) and v > 0 else MODEL_DOF.get(str(row.get("candidate_model")), 8)


def values(row: pd.Series) -> dict[str, float]:
    return {name: finite(row.get(name), np.nan) for name in METRICS}


def validation_score(row: pd.Series) -> float:
    v = values(row)
    raw, trimmed, inlier = v[METRICS[0]], v[METRICS[1]], v[METRICS[2]]
    if not np.isfinite(trimmed):
        trimmed = raw
    if not np.isfinite(inlier):
        inlier = raw
    if not np.isfinite(raw):
        raw = max(trimmed, inlier)
    if not np.isfinite(raw):
        return 1.0e9
    return float(0.45 * trimmed + 0.35 * inlier + 0.20 * raw)


def relative_gain(reference: float, candidate: float) -> float:
    if not (np.isfinite(reference) and np.isfinite(candidate)) or abs(reference) < 1.0e-12:
        return 0.0
    return float((reference - candidate) / abs(reference))


def gains(reference: pd.Series, candidate: pd.Series) -> dict[str, float]:
    rv, cv = values(reference), values(candidate)
    return {m: relative_gain(rv[m], cv[m]) for m in METRICS}


def no_worse(candidate: pd.Series, reference: pd.Series, eps: float) -> bool:
    cv, rv = values(candidate), values(reference)
    return not any(
        np.isfinite(cv[m]) and np.isfinite(rv[m]) and cv[m] > rv[m] * (1.0 + eps)
        for m in METRICS
    )


def best(group: pd.DataFrame, models: set[str] | None = None) -> pd.Series | None:
    subset = group.copy() if models is None else group[group.candidate_model.isin(models)].copy()
    if subset.empty:
        return None
    subset["_validation_score"] = subset.apply(validation_score, axis=1)
    subset["_risk_condition"] = pd.to_numeric(subset.get("condition_number", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e15)
    subset["_dof"] = subset.apply(dof, axis=1)
    return subset.sort_values(["_validation_score", "_risk_condition", "_dof", "candidate_model"]).iloc[0]


def choose_near_tie(group: pd.DataFrame, eps: float) -> tuple[pd.Series, list[str]]:
    reference = best(group)
    if reference is None:
        raise ValueError("empty near-tie group")
    rv = values(reference)
    keep = []
    for idx, row in group.iterrows():
        cv = values(row)
        if all(np.isfinite(cv[m]) and np.isfinite(rv[m]) and cv[m] <= rv[m] * (1.0 + eps) for m in METRICS):
            keep.append(idx)
    near = group.loc[keep].copy() if keep else group.loc[[reference.name]].copy()
    members = list(near.candidate_model.astype(str))
    if len(near) == 1:
        return near.iloc[0], members
    near["_cov"] = pd.to_numeric(near.get("position_cov_sqrt_trace_m", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e9)
    near["_boot"] = pd.to_numeric(near.get("position_bootstrap_spread_m", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e9)
    near["_cond"] = pd.to_numeric(near.get("condition_number", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e15)
    near["_dof"] = near.apply(dof, axis=1)
    return near.sort_values(["_cov", "_boot", "_cond", "_dof", "candidate_model"]).iloc[0], members


def base_filter(group: pd.DataFrame, best_static: pd.Series | None, cfg: dict[str, Any]) -> pd.DataFrame:
    keep = []
    best_static_trim = np.nan if best_static is None else finite(best_static.get("trimmed_validation_rmse_mps"), np.nan)
    for idx, row in group.iterrows():
        model, reasons = str(row.get("candidate_model", "")), []
        if not truthy(row.get("numerical_success"), False): reasons.append("numerical_success_false")
        if not truthy(row.get("physical_plausible"), False): reasons.append("physical_plausible_false")
        if not truthy(row.get("quality_pass"), False): reasons.append("quality_pass_false")
        if truthy(row.get("severe_risk_veto"), False): reasons.append("severe_risk_veto")
        if model in BE_FULL:
            retention = finite(row.get("retention_trace_ratio"), np.nan)
            rank_loss = finite(row.get("rank_loss"), 0.0)
            if np.isfinite(retention) and retention < cfg["projection_retention_min"]: reasons.append("be_full_retention")
            if np.isfinite(rank_loss) and rank_loss > 0: reasons.append("be_full_rank_loss")
        if model in DIRECT_GIR: reasons.append("direct_gir_not_local_refine")
        if str(row.get("dataset_type", "")) == "real" and model in REAL_FORBIDDEN_GIR: reasons.append("real_gir_or_static_refine_forbidden")
        if str(row.get("dataset_type", "")) == "real" and model in DYNAMIC and model not in STATIC:
            dg = relative_gain(best_static_trim, finite(row.get("trimmed_validation_rmse_mps"), np.nan))
            speed = finite(row.get("estimated_speed_mps"), 0.0)
            bdot = abs(finite(row.get("beta_dot_estimated_mps2"), 0.0))
            if dg < cfg["real_dynamic_gain_min"]: reasons.append("real_dynamic_gain")
            if speed > 2.0 and dg <= cfg["real_strong_gain"]: reasons.append("real_speed_gain")
            if bdot > 0.02 and dg <= cfg["real_strong_gain"]: reasons.append("real_bdot_gain")
        if not reasons: keep.append(idx)
    return group.loc[keep].copy()


def dynamic_gate(best_static: pd.Series | None, best_dynamic: pd.Series | None, dataset_type: str, cfg: dict[str, Any]) -> tuple[bool, str]:
    if best_static is None: return True, "no_static_reference"
    if best_dynamic is None: return False, "no_dynamic_candidate"
    gs = gains(best_static, best_dynamic)
    threshold = cfg["real_dynamic_gain_min"] if dataset_type == "real" else cfg["dynamic_gain_min"]
    reasons = []
    if gs[METRICS[1]] < threshold: reasons.append("trimmed_gain")
    if gs[METRICS[0]] <= 0: reasons.append("raw_not_support_dynamic")
    if gs[METRICS[2]] <= 0: reasons.append("inlier_not_support_dynamic")
    if dataset_type == "real" and finite(best_dynamic.get("estimated_speed_mps"), 0.0) > 2.0 and gs[METRICS[1]] <= cfg["real_strong_gain"]:
        reasons.append("real_speed_without_strong_gain")
    return not reasons, "pass" if not reasons else ";".join(reasons)


def bias_gate(candidates: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    nb, bc = best(candidates, NO_BIAS), best(candidates, BIAS_CAPABLE)
    if nb is None or bc is None: return candidates.copy()
    ng = {}
    for metric in METRICS:
        nbv, bcv = finite(nb.get(metric), np.nan), finite(bc.get(metric), np.nan)
        ng[metric] = 0.0 if not (np.isfinite(nbv) and np.isfinite(bcv)) or abs(bcv) < 1e-12 else float((bcv - nbv) / abs(bcv))
    bg = gains(nb, bc)
    beta = abs(finite(bc.get("beta0_estimated_mps"), 0.0))
    cond = finite(bc.get("condition_number"), np.inf)
    nontrivial = beta >= cfg["bias_nontrivial_mps"] or bg[METRICS[1]] >= cfg["bias_gain_min"] or bg[METRICS[2]] >= cfg["bias_gain_min"]
    not_worse_count = sum(x >= -cfg["near_tie_eps"] for x in ng.values())
    no_bias_clear = max(ng.values()) >= cfg["bias_gain_min"] and not_worse_count == 3
    if no_bias_clear or (not_worse_count >= 2 and not nontrivial) or (not_worse_count >= 2 and not (nontrivial and np.isfinite(cond) and cond <= cfg["bias_condition_limit"])):
        selected_models = NO_BIAS
    elif nontrivial and np.isfinite(cond) and cond <= cfg["bias_condition_limit"]:
        selected_models = BIAS_CAPABLE
    else:
        selected_models = BIAS_CAPABLE
    return candidates[candidates.candidate_model.isin(selected_models)].copy()


def refine_ok(row: pd.Series, all_rows: pd.DataFrame, cfg: dict[str, Any]) -> bool:
    model = str(row.get("candidate_model", ""))
    if model not in REFINE: return True
    if "refine_gate_pass" in row.index and pd.notna(row.get("refine_gate_pass")) and not truthy(row.get("refine_gate_pass"), True): return False
    base_name = str(row.get("base_model_for_refine") or "")
    if not base_name or base_name.lower() == "nan":
        base_name = {"M12_ctd_full_plus_gir_refine":"M2_ctd_full", "M13_robust_ctd_full_plus_gir_refine":"M7_robust_ctd_full", "M14_static_plus_gir_refine":"M0_static_position"}.get(model, "")
    base = best(all_rows, {base_name}) if base_name else None
    if base is None: return False
    raw_ratio = finite(row.get(METRICS[0]), np.inf) / max(finite(base.get(METRICS[0]), np.nan), 1e-12)
    gs = gains(base, row)
    cond_ratio = finite(row.get("condition_number"), np.inf) / max(finite(base.get("condition_number"), np.nan), 1e-12)
    info = finite(row.get("information_retention_ratio"), 1.0)
    return bool(
        raw_ratio <= cfg["refine_raw_ratio_max"] and
        max(gs[METRICS[1]], gs[METRICS[2]]) >= cfg["refine_gain_min"] and
        (not np.isfinite(cond_ratio) or cond_ratio <= cfg["refine_condition_ratio_max"]) and
        finite(row.get("estimated_speed_mps"), 0.0) <= 300.0 and
        abs(finite(row.get("beta0_estimated_mps"), 0.0)) <= 30.0 and
        abs(finite(row.get("beta_dot_estimated_mps2"), 0.0)) <= 1.0 and
        (not np.isfinite(info) or info >= cfg["refine_information_min"])
    )


def robust_ok(row: pd.Series, reference: pd.Series | None, cfg: dict[str, Any]) -> bool:
    model = str(row.get("candidate_model", ""))
    if model not in ROBUST_V4: return True
    if reference is None: return False
    trim_gain = relative_gain(finite(reference.get(METRICS[1]), np.nan), finite(row.get(METRICS[1]), np.nan))
    inlier_gain = relative_gain(finite(reference.get(METRICS[2]), np.nan), finite(row.get(METRICS[2]), np.nan))
    rr = finite(row.get(METRICS[0]), np.nan) / finite(reference.get(METRICS[0]), np.nan)
    rc, bc = finite(row.get("robust_validation_cost"), np.nan), finite(reference.get("robust_validation_cost"), np.nan)
    cost_gain = relative_gain(bc, rc)
    outlier, burst = truthy(row.get("outlier_evidence"), False), truthy(row.get("burst_outlier_evidence"), False)
    speed_delta = finite(row.get("estimated_speed_mps"), 0.0) - finite(reference.get("estimated_speed_mps"), 0.0)
    bdot_delta = abs(finite(row.get("beta_dot_estimated_mps2"), 0.0)) - abs(finite(reference.get("beta_dot_estimated_mps2"), 0.0))
    info = finite(row.get("information_retention_ratio"), 1.0)
    return bool(
        outlier and (max(trim_gain, inlier_gain) >= cfg["robust_gain_min"] or (burst and cost_gain >= cfg["robust_gain_min"])) and
        rr <= cfg["robust_raw_ratio_max"] and speed_delta <= 20.0 and bdot_delta <= 0.10 and
        (not np.isfinite(info) or info >= cfg["robust_information_min"])
    )


def select_group(group: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    cfg = spec["thresholds"]
    rows = group.copy()
    best_static_all = best(rows, STATIC)
    eligible = base_filter(rows, best_static_all, cfg)
    if eligible.empty:
        fallback = best(rows)
        if fallback is None: raise ValueError("candidate group is empty")
        return {"selected_candidate": str(fallback.candidate_model), "status": "low_quality", "low_quality": True, "near_tie_members": [str(fallback.candidate_model)], "selection_reason": "no_candidate_after_base_filter;fallback_best_validation"}
    core = eligible[eligible.candidate_model.isin(CORE_FINAL)].copy()
    if core.empty: core = eligible.copy()
    best_static, dynamic_core = best(core, STATIC), core[core.candidate_model.isin(CORE_DYNAMIC)].copy()
    best_dynamic = best(dynamic_core)
    dpass, dreason = dynamic_gate(best_static, best_dynamic, str(rows.dataset_type.iloc[0]), cfg)
    if not dpass and best_static is not None:
        selected, near = choose_near_tie(core[core.candidate_model.isin(STATIC)].copy(), cfg["near_tie_eps"])
        return {"selected_candidate":str(selected.candidate_model), "status":"selected", "low_quality":False, "near_tie_members":near, "selection_reason":f"static_dynamic_gate:{dreason};static_family_selected"}
    candidates = bias_gate(dynamic_core, cfg)
    nd, fd = best(candidates, NO_DRIFT), best(candidates, FULL_DRIFT)
    if nd is not None and fd is not None:
        dg = max(gains(nd, fd).values())
        drift = abs(finite(fd.get("beta_dot_estimated_mps2"), 0.0))
        if dg >= cfg["drift_gain_min"] and drift <= 1.0:
            candidates = candidates[candidates.candidate_model.isin(FULL_DRIFT)].copy()
        elif no_worse(fd, nd, cfg["near_tie_eps"]) and drift >= cfg["drift_small_mps2"] and drift <= 1.0:
            candidates = candidates[candidates.candidate_model.isin(FULL_DRIFT)].copy()
        else:
            candidates = candidates[candidates.candidate_model.isin(NO_DRIFT)].copy()
    candidates = candidates.loc[[idx for idx,row in candidates.iterrows() if refine_ok(row, eligible, cfg)]].copy()
    reference = best(candidates[~candidates.candidate_model.isin(ROBUST)].copy())
    candidates = candidates.loc[[idx for idx,row in candidates.iterrows() if robust_ok(row, reference, cfg)]].copy()
    if candidates.empty:
        fallback = best(dynamic_core if not dynamic_core.empty else core)
        if fallback is None: fallback = best(rows)
        return {"selected_candidate":str(fallback.candidate_model), "status":"low_quality", "low_quality":True, "near_tie_members":[str(fallback.candidate_model)], "selection_reason":"empty_after_family_gates;fallback_best_validation"}
    selected, near = choose_near_tie(candidates, cfg["near_tie_eps"])
    return {"selected_candidate":str(selected.candidate_model), "status":"selected", "low_quality":False, "near_tie_members":near, "selection_reason":"admitted_dynamic_or_static_family"}


def replay(candidate_csv: Path, reference_csv: Path, spec_json: Path) -> pd.DataFrame:
    spec = json.loads(long_path(spec_json).read_text(encoding="utf-8"))
    candidates = pd.read_csv(long_path(candidate_csv))
    candidates = candidates[(candidates.evaluation_dataset == "PAPER_EXP01_HOLDOUT") & (candidates.evidence_mode == "observable_only")].copy()
    forbidden = [c for c in candidates.columns if any(x in c.lower() for x in ("truth", "position_error", "oracle"))]
    if forbidden:
        raise RuntimeError(f"Forbidden input fields present: {forbidden}")
    refs = pd.read_csv(long_path(reference_csv))
    refs = refs[(refs.evaluation_dataset == "PAPER_EXP01_HOLDOUT") & (refs.reference_mode == "observable_only")].copy()
    rows = []
    for gid, group in candidates.groupby("group_id", sort=True):
        result = select_group(group, spec)
        ref = refs.loc[refs.group_id == gid, "reference_selected_model"]
        if len(ref) != 1: raise RuntimeError(f"Reference identity error for {gid}")
        rows.append({"group_id":gid, **result, "reference_selected_candidate":str(ref.iloc[0]), "candidate_match":str(result["selected_candidate"]) == str(ref.iloc[0]), "reference_status":"selected", "status_match":result["status"] == "selected"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("spec_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    out = replay(args.candidate_csv, args.reference_csv, args.spec_json)
    output_csv = long_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(json.dumps({"rows":len(out), "candidate_matches":int(out.candidate_match.sum()), "status_matches":int(out.status_match.sum()), "low_quality":int(out.low_quality.sum())}))
