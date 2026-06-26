#!/usr/bin/env python3
"""Map surviving candidates after excluding the broken ID242 descendant chain.

This report-only diagnostic scans existing base prediction artifacts, parses
experiment metadata dependencies, excludes candidates that depend on the
ID239/ID242/ID249/ID250/ID251/ID252 calibration chain, and ranks the remaining
reviewable artifacts using conservative public-anchor geometry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    align,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.diagnose_public_anchor_rerank_post_id335 import (  # noqa: E402
    BASE_DIR,
    conservative_proxy,
    family_for,
    find_base_test_files,
    load_log,
    load_pred,
    log_row,
    projection,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
BROKEN_CHAIN_ROOTS = {239, 242, 243, 249, 250, 251, 252}
REFERENCE_IDS = {178, 287, 329, 334, 335}


def experiment_json_path(candidate_id: int) -> Path | None:
    matches = sorted(REPORTS_DIR.glob(f"experiment_{candidate_id}_*.json"))
    return matches[0] if len(matches) == 1 else None


def add_id_from_key(key: str, value: Any, found: set[int]) -> None:
    key_lower = key.lower()
    id_like_keys = {
        "id",
        "id_a",
        "id_b",
        "base_id",
        "anchor_id",
        "frontier_id",
        "source_id",
        "candidate_id",
        "original_id",
        "experiment_id",
    }
    if key_lower in id_like_keys and isinstance(value, int):
        found.add(value)
    if key_lower.endswith("_ids") and isinstance(value, list):
        found.update(int(item) for item in value if isinstance(item, int))


def ids_from_string(value: str) -> set[int]:
    found = {int(match) for match in re.findall(r"base_(?:oof|test)_id_(\d+)_", value)}
    found.update(int(match) for match in re.findall(r"\bID(\d+)\b", value, flags=re.IGNORECASE))
    return found


def collect_dependency_ids(value: Any, found: set[int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            add_id_from_key(str(key), child, found)
            collect_dependency_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            collect_dependency_ids(child, found)
    elif isinstance(value, str):
        found.update(ids_from_string(value))


def direct_dependencies(candidate_id: int) -> set[int]:
    path = experiment_json_path(candidate_id)
    if path is None:
        return set()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    found: set[int] = set()
    for key in ["selector_info", "validation", "method", "base_oof_file", "base_test_file", "submission_file"]:
        collect_dependency_ids(metadata.get(key), found)
    found.discard(candidate_id)
    return found


def dependency_graph(candidate_ids: list[int]) -> dict[int, set[int]]:
    all_ids = set(candidate_ids)
    graph: dict[int, set[int]] = {}
    frontier = set(candidate_ids)
    while frontier:
        candidate_id = frontier.pop()
        if candidate_id in graph:
            continue
        deps = direct_dependencies(candidate_id)
        graph[candidate_id] = deps
        new_ids = deps - all_ids
        all_ids.update(new_ids)
        frontier.update(new_ids)
    return graph


def dependency_closure(candidate_id: int, graph: dict[int, set[int]]) -> set[int]:
    seen: set[int] = set()
    stack = list(graph.get(candidate_id, set()))
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        stack.extend(graph.get(dep, set()) - seen)
    return seen


def base_oof_path(candidate_id: int) -> Path | None:
    matches = sorted(BASE_DIR.glob(f"base_oof_id_{candidate_id:03d}_*.csv"))
    return matches[0] if len(matches) == 1 else None


def local_robust_row(candidate_id: int, train: pd.DataFrame) -> dict[str, Any]:
    path = base_oof_path(candidate_id)
    if path is None:
        return {"local_oof": None, "robust_oof": None, "worst_target": None, "worst_target_loss": None}
    pred = align(pd.read_csv(path), train)
    overall, target_scores = local_scores(train, pred)
    slices = slice_summary(train, pred)
    return {
        "local_oof": overall,
        "robust_oof": robust_score(overall, slices),
        "target_scores": target_scores,
        "worst_target": slices["worst_target"],
        "worst_target_loss": slices["worst_target_loss"],
        "worst_subject": slices["worst_subject"],
        "worst_subject_loss": slices["worst_subject_loss"],
    }


def row_for_candidate(
    *,
    candidate_id: int,
    path: Path,
    log: pd.DataFrame,
    train: pd.DataFrame,
    arr178: np.ndarray,
    arr287: np.ndarray,
    arr329: np.ndarray,
    arr335: np.ndarray,
    graph: dict[int, set[int]],
) -> dict[str, Any]:
    info = log_row(log, candidate_id)
    arr = load_pred(path)
    failed_direction = (arr287 - arr178).reshape(-1)
    vec178 = (arr - arr178).reshape(-1)
    closure = dependency_closure(candidate_id, graph)
    broken_intersection = sorted(closure & BROKEN_CHAIN_ROOTS)
    local_cv = info.get("local_cv_score_num", info.get("local_cv_score"))
    public_lb = info.get("public_lb_score_num", info.get("public_lb_score"))
    public_lb = None if public_lb is None or pd.isna(public_lb) else float(public_lb)
    row = {
        "id": candidate_id,
        "submission_file": info.get("submission_file", ""),
        "base_test_file": str(path.relative_to(ROOT)),
        "family": family_for(candidate_id, str(info.get("submission_file", "")), path.name),
        "local_cv": None if local_cv is None or pd.isna(local_cv) else float(local_cv),
        "public_lb": public_lb,
        "status_clean": info.get("status_clean", ""),
        "dependency_closure": sorted(closure),
        "broken_chain_roots_in_closure": broken_intersection,
        "is_broken_chain_descendant": bool(candidate_id in BROKEN_CHAIN_ROOTS or broken_intersection),
        "mean_delta_vs_178": float(np.mean(np.abs(arr - arr178))),
        "max_delta_vs_178": float(np.max(np.abs(arr - arr178))),
        "mean_delta_vs_329": float(np.mean(np.abs(arr - arr329))),
        "max_delta_vs_329": float(np.max(np.abs(arr - arr329))),
        "mean_delta_vs_335": float(np.mean(np.abs(arr - arr335))),
        "max_delta_vs_335": float(np.max(np.abs(arr - arr335))),
        "projection_failed": projection(vec178, failed_direction),
        "prediction_mean": float(np.mean(arr)),
        "prediction_std": float(np.std(arr)),
    }
    row.update(local_robust_row(candidate_id, train))
    row["conservative_public_risk_proxy"] = conservative_proxy(row)
    return row


def is_reviewable_survivor(row: dict[str, Any]) -> bool:
    if row["is_broken_chain_descendant"]:
        return False
    if row["id"] in REFERENCE_IDS:
        return False
    if row["public_lb"] is not None:
        return False
    if row["status_clean"] not in {"generated_unsubmitted", "cv_lb_gap_risk", ""}:
        return False
    if row["projection_failed"] > 0.04:
        return False
    if row["max_delta_vs_178"] > 0.16:
        return False
    return True


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Surviving candidates excluding ID242 chain — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook, submission-log row, base artifact, or DACON submission.",
        "",
        "## Counts",
        "",
        f"- Total base-test candidates scanned: `{report['n_candidates']}`",
        f"- Broken-chain descendants: `{report['n_broken_chain_descendants']}`",
        f"- Reviewable survivors: `{len(report['reviewable_survivors'])}`",
        "",
        "## Reviewable survivors",
        "",
        "| rank | id | local | robust | proxy | family | max Δ178 | proj failed | worst target | file |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for rank, row in enumerate(report["reviewable_survivors"], start=1):
        local = "" if row["local_oof"] is None else f"{row['local_oof']:.6f}"
        robust = "" if row["robust_oof"] is None else f"{row['robust_oof']:.6f}"
        worst = "" if row["worst_target"] is None else f"{row['worst_target']} `{row['worst_target_loss']:.6f}`"
        lines.append(
            f"| {rank} | {row['id']} | {local} | {robust} | "
            f"{row['conservative_public_risk_proxy']:.6f} | `{row['family']}` | "
            f"{row['max_delta_vs_178']:.6f} | {row['projection_failed']:+.4f} | "
            f"{worst} | `{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Top excluded descendants by local score",
            "",
            "| id | local | family | broken roots | file |",
            "|---:|---:|---|---|---|",
        ]
    )
    for row in report["top_excluded_descendants_by_local"][:20]:
        local = "" if row["local_oof"] is None else f"{row['local_oof']:.6f}"
        lines.append(
            f"| {row['id']} | {local} | `{row['family']}` | "
            f"`{row['broken_chain_roots_in_closure']}` | `{row['submission_file']}` |"
        )
    lines.extend(["", "## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, _test = load_train_test()
    log = load_log()
    files = find_base_test_files()
    required = [178, 287, 329, 335]
    missing = [candidate_id for candidate_id in required if candidate_id not in files]
    if missing:
        raise FileNotFoundError(f"Missing required base_test ids: {missing}")
    graph = dependency_graph(sorted(files))
    arr178 = load_pred(files[178])
    arr287 = load_pred(files[287])
    arr329 = load_pred(files[329])
    arr335 = load_pred(files[335])
    rows = [
        row_for_candidate(
            candidate_id=candidate_id,
            path=path,
            log=log,
            train=train,
            arr178=arr178,
            arr287=arr287,
            arr329=arr329,
            arr335=arr335,
            graph=graph,
        )
        for candidate_id, path in sorted(files.items())
    ]
    descendants = [row for row in rows if row["is_broken_chain_descendant"]]
    survivors = [row for row in rows if is_reviewable_survivor(row)]
    survivors = sorted(
        survivors,
        key=lambda row: (
            row["conservative_public_risk_proxy"],
            row["robust_oof"] if row["robust_oof"] is not None else 9.0,
            row["max_delta_vs_178"],
        ),
    )[:30]
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "surviving_candidates_excluding_id242_post_id335",
        "broken_chain_roots": sorted(BROKEN_CHAIN_ROOTS),
        "n_candidates": len(rows),
        "n_broken_chain_descendants": len(descendants),
        "family_counts_all": dict(Counter(row["family"] for row in rows)),
        "family_counts_survivors": dict(Counter(row["family"] for row in survivors)),
        "reviewable_survivors": survivors,
        "top_excluded_descendants_by_local": sorted(
            descendants,
            key=lambda row: row["local_oof"] if row["local_oof"] is not None else 9.0,
        )[:50],
        "top_all_rows": sorted(
            rows,
            key=lambda row: row["local_oof"] if row["local_oof"] is not None else 9.0,
        )[:100],
        "decision": {
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "reason": (
                "After excluding the broken ID239/ID242/ID249/ID250/ID251/ID252 dependency chain, remaining "
                "candidates should be treated as review targets only. This dependency-aware map is not a public "
                "prediction and does not authorize a DACON submission."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "surviving_candidates_excluding_id242_post_id335_20260617.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(
        f"scanned={report['n_candidates']} descendants={report['n_broken_chain_descendants']} "
        f"survivors={len(report['reviewable_survivors'])}"
    )
    if report["reviewable_survivors"]:
        best = report["reviewable_survivors"][0]
        print(
            f"top survivor: id={best['id']} local={best['local_oof']} "
            f"robust={best['robust_oof']} proxy={best['conservative_public_risk_proxy']:.6f}"
        )
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
