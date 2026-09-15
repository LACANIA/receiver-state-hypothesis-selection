"""Apply the predeclared branch-family taxonomy to completed outputs.

No solver or selector is called here.  This keeps the primary branch-family
metric at the experimental-role level while retaining state-specific
subfamilies as a secondary field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_exp01_run_holdout import aggregate_selected, model_family, model_subfamily


OUT = Path(__file__).resolve().parent


def main() -> None:
    protocol = json.loads((OUT / "PAPER_EXP01_HOLDOUT_PROTOCOL.json").read_text(encoding="utf-8"))
    selected_path = OUT / "PAPER_EXP01_ACTUAL_SELECTED.csv"
    selected = pd.read_csv(selected_path)
    selected["selected_model_family"] = selected["selected_model"].astype(str).map(model_family)
    selected["selected_model_subfamily"] = selected["selected_model"].astype(str).map(model_subfamily)
    selected["oracle_model_family"] = selected["oracle_model"].astype(str).map(
        lambda model: model_family(model) if model.startswith("M") else "none"
    )
    selected["oracle_model_subfamily"] = selected["oracle_model"].astype(str).map(
        lambda model: model_subfamily(model) if model.startswith("M") else "none"
    )
    selected["selected_family_equals_oracle_family"] = (
        selected["selected_model_family"] == selected["oracle_model_family"]
    ) & (selected["oracle_model_family"] != "none")
    selected.to_csv(selected_path, index=False)

    primary = selected.loc[selected["evaluation_scope"] == "primary"].copy()
    aggregate_selected(primary, int(protocol["master_seed"])).to_csv(
        OUT / "PAPER_EXP01_FAMILY_AGGREGATE.csv", index=False
    )

    primary_a = primary.loc[primary["selector_mode"] == "observable_only"].copy()
    sensitivity = selected.loc[
        (selected["selector_mode"] == "observable_only")
        & (selected["realization_index"].astype(int).isin([0, 1]))
    ].copy()
    primary_lookup = primary_a.set_index("holdout_id")
    sensitivity["primary_selected_model"] = sensitivity["holdout_id"].map(primary_lookup["selected_model"])
    sensitivity["primary_position_error_m"] = sensitivity["holdout_id"].map(primary_lookup["final_position_error_m"])
    sensitivity["selected_model_changed_vs_primary"] = sensitivity["selected_model"] != sensitivity["primary_selected_model"]
    sensitivity["position_error_difference_vs_primary_m"] = (
        pd.to_numeric(sensitivity["final_position_error_m"], errors="coerce")
        - pd.to_numeric(sensitivity["primary_position_error_m"], errors="coerce")
    )
    sensitivity.to_csv(OUT / "PAPER_EXP01_INITIALIZATION_SENSITIVITY.csv", index=False)

    primary_b = primary.loc[primary["selector_mode"] == "label_assisted_ablation"].copy()
    compare_cols = [
        "holdout_id",
        "holdout_family",
        "realization_index",
        "selected_model",
        "selected_model_family",
        "selected_model_subfamily",
        "final_position_error_m",
        "selected_minus_oracle_error_m",
        "quality_pass",
    ]
    mode_compare = primary_a[compare_cols].merge(
        primary_b[compare_cols],
        on=["holdout_id", "holdout_family", "realization_index"],
        suffixes=("_mode_a", "_mode_b"),
        validate="one_to_one",
    )
    mode_compare["selected_model_changed"] = (
        mode_compare["selected_model_mode_a"] != mode_compare["selected_model_mode_b"]
    )
    mode_compare["position_error_difference_b_minus_a_m"] = (
        pd.to_numeric(mode_compare["final_position_error_m_mode_b"], errors="coerce")
        - pd.to_numeric(mode_compare["final_position_error_m_mode_a"], errors="coerce")
    )
    mode_compare["mode_b_claim_restriction"] = (
        "synthetic-label ablation only; not an online receiver capability"
    )
    mode_compare.to_csv(OUT / "PAPER_EXP01_OBSERVABLE_ONLY_VS_LABEL_ASSISTED.csv", index=False)


if __name__ == "__main__":
    main()
