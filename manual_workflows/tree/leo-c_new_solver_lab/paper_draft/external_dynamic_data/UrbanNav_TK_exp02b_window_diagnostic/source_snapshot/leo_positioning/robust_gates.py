"""Hard evidence gates for robust branches in MA-BGTR-v4."""

from __future__ import annotations

import numpy as np


ROBUST_V4_MODELS = {
    "M7_robust_ctd_full",
    "M8_robust_be_b0_only",
    "M9_robust_be_full",
    "M10_gir_tr_fixed_scale",
    "M11_gir_tr_mad_scale",
    "M13_robust_ctd_full_plus_gir_refine",
}


def _finite(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def apply_robust_hard_gate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ctd = next((r for r in rows if r.get("candidate_model") == "M2_ctd_full" and bool(r.get("numerical_success", False))), None)
    ctd_trim = _finite(ctd.get("trimmed_validation_rmse_mps")) if ctd else np.nan
    ctd_inlier = _finite(ctd.get("inlier_validation_rmse_mps")) if ctd else np.nan
    ctd_full = _finite(ctd.get("full_residual_rmse_mps")) if ctd else np.nan
    out: list[dict] = []
    diag: list[dict] = []
    for row in rows:
        model = str(row.get("candidate_model", ""))
        trimmed = _finite(row.get("trimmed_validation_rmse_mps"))
        inlier = _finite(row.get("inlier_validation_rmse_mps"))
        full = _finite(row.get("full_residual_rmse_mps"))
        trimmed_gain = (ctd_trim - trimmed) / ctd_trim if np.isfinite(ctd_trim) and ctd_trim > 0 and np.isfinite(trimmed) else 0.0
        inlier_gain = (ctd_inlier - inlier) / ctd_inlier if np.isfinite(ctd_inlier) and ctd_inlier > 0 and np.isfinite(inlier) else 0.0
        full_ratio = full / ctd_full if np.isfinite(ctd_full) and ctd_full > 0 and np.isfinite(full) else np.inf
        tail = _finite(row.get("tail_ratio"), 0.0)
        mad = _finite(row.get("mad_ratio"), 0.0)
        info = _finite(row.get("information_retention_ratio"), 1.0)
        reasons: list[str] = []
        if model in ROBUST_V4_MODELS:
            condition_a = (tail >= 0.05 or mad >= 2.0) and trimmed_gain >= 0.05 and inlier_gain >= 0.05 and full_ratio <= 1.05
            condition_b = trimmed_gain >= 0.10 and inlier_gain >= 0.10 and info >= 0.6
            passed = bool(condition_a or condition_b)
            if not (tail >= 0.05 or mad >= 2.0):
                reasons.append("no_outlier_evidence")
            if trimmed_gain < 0.05:
                reasons.append("trimmed_gain_lt_5pct")
            if inlier_gain < 0.05:
                reasons.append("inlier_gain_lt_5pct")
            if full_ratio > 1.05:
                reasons.append("full_rmse_ratio_gt_1_05")
            if not condition_a and condition_b:
                reasons = ["localized_outlier_gate_pass"]
        else:
            passed = True
            reasons = []
        updated = dict(row)
        updated.update(
            {
                "robust_gate_pass": bool(passed),
                "robust_gate_reason": ";".join(dict.fromkeys(reasons)),
                "trimmed_gain_vs_ctd": float(trimmed_gain),
                "inlier_gain_vs_ctd": float(inlier_gain),
                "full_rmse_ratio_vs_ctd": float(full_ratio) if np.isfinite(full_ratio) else np.nan,
            }
        )
        out.append(updated)
        diag.append(
            {
                "dataset_type": row.get("dataset_type"),
                "scenario": row.get("scenario"),
                "candidate_model": model,
                "position_init_label": row.get("position_init_label"),
                "velocity_init_label": row.get("velocity_init_label"),
                "beta_prior_profile": row.get("beta_prior_profile"),
                "tail_ratio": tail,
                "mad_ratio": mad,
                "trimmed_gain_vs_ctd": trimmed_gain,
                "inlier_gain_vs_ctd": inlier_gain,
                "full_rmse_ratio_vs_ctd": full_ratio if np.isfinite(full_ratio) else np.nan,
                "information_retention_ratio": info,
                "robust_gate_pass": bool(passed),
                "robust_gate_reason": ";".join(dict.fromkeys(reasons)),
            }
        )
    return out, diag


def evaluate_robust_gate_v7(row: dict, reference: dict | None) -> tuple[bool, dict]:
    """Evaluate TECH12 robust hard gate against a non-robust reference.

    This stricter gate is used only for final v7 selection. It keeps robust
    branches useful in clear outlier cases while preventing tiny trimmed-RMSE
    gains from beating the CTD reference in hard geometry scenarios.
    """
    model = str(row.get("candidate_model", ""))
    if model not in ROBUST_V4_MODELS:
        diag = {
            "reference_model": reference.get("candidate_model") if reference else "",
            "tail_ratio": _finite(row.get("tail_ratio"), 0.0),
            "mad_ratio": _finite(row.get("mad_ratio"), 0.0),
            "trimmed_gain_vs_reference": 0.0,
            "inlier_gain_vs_reference": 0.0,
            "raw_rmse_ratio_vs_reference": 1.0,
            "speed_delta_vs_reference": 0.0,
            "bdot_delta_vs_reference": 0.0,
            "information_retention_ratio": _finite(row.get("information_retention_ratio"), 1.0),
            "robust_gate_pass": True,
            "robust_gate_reason": "non_robust_candidate",
        }
        return True, diag

    if reference is None:
        diag = {
            "reference_model": "",
            "tail_ratio": _finite(row.get("tail_ratio"), 0.0),
            "mad_ratio": _finite(row.get("mad_ratio"), 0.0),
            "trimmed_gain_vs_reference": 0.0,
            "inlier_gain_vs_reference": 0.0,
            "raw_rmse_ratio_vs_reference": np.nan,
            "speed_delta_vs_reference": np.nan,
            "bdot_delta_vs_reference": np.nan,
            "information_retention_ratio": _finite(row.get("information_retention_ratio"), 1.0),
            "robust_gate_pass": False,
            "robust_gate_reason": "missing_nonrobust_reference",
        }
        return False, diag

    ref_trim = _finite(reference.get("trimmed_validation_rmse_mps"))
    ref_inlier = _finite(reference.get("inlier_validation_rmse_mps"))
    ref_raw = _finite(reference.get("raw_validation_rmse_mps"))
    trim = _finite(row.get("trimmed_validation_rmse_mps"))
    inlier = _finite(row.get("inlier_validation_rmse_mps"))
    raw = _finite(row.get("raw_validation_rmse_mps"))
    trimmed_gain = (ref_trim - trim) / ref_trim if np.isfinite(ref_trim) and ref_trim > 0 and np.isfinite(trim) else 0.0
    inlier_gain = (ref_inlier - inlier) / ref_inlier if np.isfinite(ref_inlier) and ref_inlier > 0 and np.isfinite(inlier) else 0.0
    raw_ratio = raw / ref_raw if np.isfinite(ref_raw) and ref_raw > 0 and np.isfinite(raw) else np.inf
    tail = _finite(row.get("tail_ratio"), 0.0)
    mad = _finite(row.get("mad_ratio"), 0.0)
    if not np.isfinite(tail) or not np.isfinite(mad):
        robust_cost = _finite(row.get("robust_validation_cost"), np.nan)
        ref_cost = _finite(reference.get("robust_validation_cost"), np.nan)
        if np.isfinite(robust_cost) and np.isfinite(ref_cost) and robust_cost < 0.7 * ref_cost:
            mad = max(mad if np.isfinite(mad) else 0.0, 2.0)
    speed_delta = _finite(row.get("estimated_speed_mps"), 0.0) - _finite(reference.get("estimated_speed_mps"), 0.0)
    bdot_delta = abs(_finite(row.get("beta_dot_estimated_mps2"), 0.0)) - abs(_finite(reference.get("beta_dot_estimated_mps2"), 0.0))
    info = _finite(row.get("information_retention_ratio"), 1.0)

    reasons: list[str] = []
    outlier_evidence = tail >= 0.05 or mad >= 2.0
    if not outlier_evidence:
        reasons.append("no_outlier_evidence")
    if max(trimmed_gain, inlier_gain) < 0.05:
        reasons.append("robust_gain_lt_5pct")
    if raw_ratio > 1.02:
        reasons.append("raw_rmse_ratio_gt_1_02")
    if speed_delta > 20.0:
        reasons.append("speed_delta_gt_20mps")
    if bdot_delta > 0.10:
        reasons.append("bdot_delta_gt_0_10")
    if np.isfinite(info) and info < 0.6:
        reasons.append("information_retention_lt_0_6")

    passed = bool(outlier_evidence and max(trimmed_gain, inlier_gain) >= 0.05 and raw_ratio <= 1.02 and speed_delta <= 20.0 and bdot_delta <= 0.10 and (not np.isfinite(info) or info >= 0.6))
    diag = {
        "reference_model": reference.get("candidate_model"),
        "tail_ratio": tail,
        "mad_ratio": mad,
        "trimmed_gain_vs_reference": float(trimmed_gain),
        "inlier_gain_vs_reference": float(inlier_gain),
        "raw_rmse_ratio_vs_reference": float(raw_ratio) if np.isfinite(raw_ratio) else np.nan,
        "speed_delta_vs_reference": float(speed_delta),
        "bdot_delta_vs_reference": float(bdot_delta),
        "information_retention_ratio": info,
        "robust_gate_pass": bool(passed),
        "robust_gate_reason": "pass" if passed else ";".join(dict.fromkeys(reasons)),
    }
    return passed, diag


def evaluate_robust_gate_v71(row: dict, reference: dict | None) -> tuple[bool, dict]:
    """TECH13 robust gate with explicit outlier evidence input.

    Robust cost can support a burst-outlier decision, but only for burst
    evidence. For hard-geometry scenarios such as Q4, robust branches still
    need a trimmed or inlier validation gain.
    """
    model = str(row.get("candidate_model", ""))
    outlier_evidence = bool(row.get("outlier_evidence", False))
    burst_outlier = bool(row.get("burst_outlier_evidence", False))
    if model not in ROBUST_V4_MODELS:
        diag = {
            "reference_model": reference.get("candidate_model") if reference else "",
            "outlier_evidence": outlier_evidence,
            "burst_outlier_evidence": burst_outlier,
            "tail_ratio": _finite(row.get("tail_ratio"), np.nan),
            "mad_ratio": _finite(row.get("mad_ratio"), np.nan),
            "trimmed_gain_vs_reference": 0.0,
            "inlier_gain_vs_reference": 0.0,
            "raw_rmse_ratio_vs_reference": 1.0,
            "speed_delta_vs_reference": 0.0,
            "bdot_delta_vs_reference": 0.0,
            "information_retention_ratio": _finite(row.get("information_retention_ratio"), 1.0),
            "robust_gate_pass": True,
            "robust_gate_reason": "non_robust_candidate",
        }
        return True, diag

    if reference is None:
        diag = {
            "reference_model": "",
            "outlier_evidence": outlier_evidence,
            "burst_outlier_evidence": burst_outlier,
            "tail_ratio": _finite(row.get("tail_ratio"), np.nan),
            "mad_ratio": _finite(row.get("mad_ratio"), np.nan),
            "trimmed_gain_vs_reference": 0.0,
            "inlier_gain_vs_reference": 0.0,
            "raw_rmse_ratio_vs_reference": np.nan,
            "speed_delta_vs_reference": np.nan,
            "bdot_delta_vs_reference": np.nan,
            "information_retention_ratio": _finite(row.get("information_retention_ratio"), 1.0),
            "robust_gate_pass": False,
            "robust_gate_reason": "missing_nonrobust_reference",
        }
        return False, diag

    ref_trim = _finite(reference.get("trimmed_validation_rmse_mps"))
    ref_inlier = _finite(reference.get("inlier_validation_rmse_mps"))
    ref_raw = _finite(reference.get("raw_validation_rmse_mps"))
    trim = _finite(row.get("trimmed_validation_rmse_mps"))
    inlier = _finite(row.get("inlier_validation_rmse_mps"))
    raw = _finite(row.get("raw_validation_rmse_mps"))
    trimmed_gain = (ref_trim - trim) / ref_trim if np.isfinite(ref_trim) and ref_trim > 0 and np.isfinite(trim) else 0.0
    inlier_gain = (ref_inlier - inlier) / ref_inlier if np.isfinite(ref_inlier) and ref_inlier > 0 and np.isfinite(inlier) else 0.0
    raw_ratio = raw / ref_raw if np.isfinite(ref_raw) and ref_raw > 0 and np.isfinite(raw) else np.inf
    robust_cost = _finite(row.get("robust_validation_cost"), np.nan)
    ref_cost = _finite(reference.get("robust_validation_cost"), np.nan)
    robust_cost_gain = (ref_cost - robust_cost) / ref_cost if np.isfinite(ref_cost) and ref_cost > 0 and np.isfinite(robust_cost) else 0.0
    speed_delta = _finite(row.get("estimated_speed_mps"), 0.0) - _finite(reference.get("estimated_speed_mps"), 0.0)
    bdot_delta = abs(_finite(row.get("beta_dot_estimated_mps2"), 0.0)) - abs(_finite(reference.get("beta_dot_estimated_mps2"), 0.0))
    info = _finite(row.get("information_retention_ratio"), 1.0)
    reasons: list[str] = []
    gain_pass = max(trimmed_gain, inlier_gain) >= 0.03
    burst_cost_pass = burst_outlier and robust_cost_gain >= 0.03
    if not outlier_evidence:
        reasons.append("no_outlier_evidence")
    if not gain_pass and not burst_cost_pass:
        reasons.append("robust_gain_lt_3pct")
    if raw_ratio > 1.03:
        reasons.append("raw_rmse_ratio_gt_1_03")
    if speed_delta > 20.0:
        reasons.append("speed_delta_gt_20mps")
    if bdot_delta > 0.10:
        reasons.append("bdot_delta_gt_0_10")
    if np.isfinite(info) and info < 0.6:
        reasons.append("information_retention_lt_0_6")
    passed = bool(outlier_evidence and (gain_pass or burst_cost_pass) and raw_ratio <= 1.03 and speed_delta <= 20.0 and bdot_delta <= 0.10 and (not np.isfinite(info) or info >= 0.6))
    diag = {
        "reference_model": reference.get("candidate_model"),
        "outlier_evidence": bool(outlier_evidence),
        "burst_outlier_evidence": bool(burst_outlier),
        "tail_ratio": _finite(row.get("tail_ratio"), np.nan),
        "mad_ratio": _finite(row.get("mad_ratio"), np.nan),
        "trimmed_gain_vs_reference": float(trimmed_gain),
        "inlier_gain_vs_reference": float(inlier_gain),
        "raw_rmse_ratio_vs_reference": float(raw_ratio) if np.isfinite(raw_ratio) else np.nan,
        "speed_delta_vs_reference": float(speed_delta),
        "bdot_delta_vs_reference": float(bdot_delta),
        "information_retention_ratio": info,
        "robust_cost_gain_vs_reference": float(robust_cost_gain),
        "robust_gate_pass": bool(passed),
        "robust_gate_reason": "pass" if passed else ";".join(dict.fromkeys(reasons)),
    }
    return passed, diag
