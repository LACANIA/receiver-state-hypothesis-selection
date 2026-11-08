from __future__ import annotations

import json

import selection_worker as worker
from baseline_common import BASELINE_ROOT, WORK, now_z, sha256_file, write_json


original = worker.selection_row


def corrected_selection_row(meta, method, subset, selected, metric, ties, source_hash, reason=""):
    if "group_id" not in meta.index:
        meta = meta.copy()
        meta["group_id"] = str(meta.name)
    return original(meta, method, subset, selected, metric, ties, source_hash, reason)


worker.selection_row = corrected_selection_row
worker.main()

lock_path = BASELINE_ROOT / "results" / "PAPER_FLAGSHIP_A_ROUTE_BASELINE_SELECTION_LOCK.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
lock["execution_wrapper"] = str(WORK / "selection_worker_interface_correction.py")
lock["execution_wrapper_sha256"] = sha256_file(WORK / "selection_worker_interface_correction.py")
lock["interface_correction"] = "restore group_id from the registry index when serializing selection rows"
lock["method_formula_changed"] = False
lock["candidate_subset_changed"] = False
lock["tie_rule_changed"] = False
lock["truth_value_reads_before_correction"] = 0
write_json(lock_path, lock)

write_json(
    BASELINE_ROOT / "protocol" / "PAPER_FLAGSHIP_A_ROUTE_PRE_EVALUATION_INTERFACE_CORRECTION_PROVENANCE.json",
    {
        "created_at": now_z(),
        "status": "result_independent_interface_correction_before_any_selection_output_or_truth_read",
        "first_attempt_output_rows": 0,
        "truth_value_reads": 0,
        "baseline_performance_reads": 0,
        "method_formula_changed": False,
        "candidate_subset_changed": False,
        "tie_rule_changed": False,
        "primary_metric_changed": False,
        "correction": "The group registry uses group_id as its index. The serializer now copies that index value into the output field.",
        "wrapper_sha256": sha256_file(WORK / "selection_worker_interface_correction.py"),
    },
)
