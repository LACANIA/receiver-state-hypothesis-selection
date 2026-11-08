from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(r"E:\LEO_Doppler_Project\leo-c_new_solver_lab")
PAPER_F_CANONICAL = PROJECT / "paper_draft" / "manuscript_portfolio" / "workspaces" / "PAPER_F_SATELLITE_NAVIGATION_FLAGSHIP"
A_ROUTE_PLAN_CANONICAL = PAPER_F_CANONICAL / "17_A_route_strengthening" / "PAPER_FLAGSHIP_A_ROUTE_RAIM_EXTERNAL_BASELINE_MANUSCRIPT_PLANNING"
BASELINE_CANONICAL = PAPER_F_CANONICAL / "17_A_route_strengthening" / "PAPER_FLAGSHIP_A_ROUTE_EXTERNAL_SELECTION_BASELINES_SUMMARY_EVIDENCE"
PAPER_F_ROOT = Path(r"E:\LEO_Doppler_Project\leo-c\tmp\PAPER_F_SHORT")
A_ROUTE_PLAN_ROOT = Path(r"E:\LEO_Doppler_Project\leo-c\tmp\A_ROUTE_OUTPUT")
BASELINE_ROOT = Path(r"E:\LEO_Doppler_Project\leo-c\tmp\A_ROUTE_BASELINE_OUTPUT")
EXP03A = PROJECT / "paper_draft" / "selection_strategy_baselines" / "PAPER_EXP03A"
EXP01 = PROJECT / "paper_draft" / "Acta_Astronautica_MA_BGTR" / "manuscript_v0_8_urbannav_limit" / "evidence" / "source_data" / "paper_exp01"
MONITOR = PROJECT / "paper_draft" / "monitor_development" / "PAPER_MONITOR01_HIERARCHICAL_RISK_DEVELOPMENT"
R3 = PROJECT / "paper_draft" / "monitor_development" / "PAPER_MONITOR01_R3_DEVELOPMENT_EXPANSION"
WORK = Path(r"E:\LEO_Doppler_Project\leo-c\tmp\a_route_baseline_work")

TRUTH_FREE = EXP03A / "outputs" / "PAPER_EXP03A_STRATEGY_INPUT_NO_TRUTH.csv"
STANDARDIZED = EXP03A / "outputs" / "PAPER_EXP03A_STANDARDIZED_CANDIDATES.csv"
REFERENCES = EXP03A / "inputs" / "PAPER_EXP03A_REFERENCE_SELECTIONS.csv"
TRUTH_LOOKUP = EXP03A / "inputs" / "PAPER_EXP03A_POST_SELECTION_TRUTH_LOOKUP.csv"
SCENARIOS = EXP01 / "PAPER_EXP01_HOLDOUT_SCENARIOS.csv"
SELECTOR_SOURCE = EXP03A / "source_snapshot" / "leo_positioning"

CANDIDATES = [f"M{i}" for i in range(15)]
MODEL_DOF = {"M0": 3, "M1": 4, "M2": 8, "M3": 7, "M4": 6,
             "M5": 7, "M6": 8, "M7": 8, "M8": 7, "M9": 8,
             "M10": 8, "M11": 8, "M12": 8, "M13": 8, "M14": 8}
METHODS = ["B0", "B1", "B2", "B3", "AIC", "AICc", "BIC", "B8_MA_BGTR_FULL", "B9_MA_BGTR_M0_M4"]


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in fieldnames})


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "bytes": 0, "sha256": ""}
    files = [path] if path.is_file() else sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: str(p).lower())
    total = 0
    h = hashlib.sha256()
    for item in files:
        rel = item.name if path.is_file() else item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest = sha256_file(item)
        total += size
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return {"path": str(path), "exists": True, "file_count": len(files), "bytes": total, "sha256": h.hexdigest()}


def model_short(name: str) -> str:
    return str(name).split("_", 1)[0]


def model_order(name: str) -> int:
    short = model_short(name)
    try:
        return int(short[1:])
    except Exception:
        return 999
