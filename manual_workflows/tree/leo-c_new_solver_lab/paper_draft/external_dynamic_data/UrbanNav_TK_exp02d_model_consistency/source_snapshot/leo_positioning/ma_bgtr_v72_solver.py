"""MA-BGTR-v7.2 selector shell for TECH14.

TECH14 found no defensible truth-free rule that reliably selects the S7 oracle
without weakening already-fixed scenarios. Therefore v7.2 deliberately keeps
the v7.1 selector unchanged and exposes S7 as a declared limitation through the
freeze policy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .candidate_alignment import compare_v71_v72, model_selection_counts_v72, v72_selected_from_v71


def select_v72_results_from_v71(v71_selected: pd.DataFrame) -> pd.DataFrame:
    """Return v7.2 selections by preserving v7.1 choices.

    This function is intentionally small: v7.2 is a freeze-policy update, not a
    new solver or model-pool change.
    """

    return v72_selected_from_v71(v71_selected)


def compare_v71_to_v72(v71_selected: pd.DataFrame, v72_selected: pd.DataFrame) -> pd.DataFrame:
    return compare_v71_v72(v71_selected, v72_selected)


def selection_counts_v72(v72_selected: pd.DataFrame) -> pd.DataFrame:
    return model_selection_counts_v72(v72_selected)


def v72_policy_summary() -> dict[str, Any]:
    return {
        "name": "MA-BGTR-v7.2",
        "selector_change": "No candidate model or solver change; v7.2 preserves v7.1 selection.",
        "reason": "S7 oracle is not selectable by current truth-free validation/risk evidence without scenario-specific hard-coding.",
        "freeze_policy": "Allow freeze only as freeze_ready_with_declared_S7_limitation when all other focused scenarios pass.",
    }
