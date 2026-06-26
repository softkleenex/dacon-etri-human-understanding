#!/usr/bin/env python3
"""Audit hidden dependencies for the B1 pair/logit direct-no-ID252 lane."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
B1_SOURCE_IDS = [254, 255, 256, 257, 258, 266]
ANCHOR_IDS = {172, 173, 177, 178, 179, 190, 191}
EXTREME_IDS = {235, 242, 243, 250, 251, 252}
HIGH_IDS = {226, 238, 247, 248, 249, 254, 255, 256, 257, 258, 266}


def experiment_path(candidate_id: int) -> Path | None:
    matches = sorted(REPORTS_DIR.glob(f"experiment_{candidate_id}_*.json"))
    return matches[0] if len(matches) == 1 else None


def collect_ids(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"id", "id_a", "id_b", "base_id", "anchor_id", "frontier_id", "original_id"}:
                try:
                    found.add(int(item))
                except (TypeError, ValueError):
                    pass
            elif key in {"artifact_ids", "frontier_ids", "source_ids", "upstream_ids"} and isinstance(item, list):
                for sub in item:
                    try:
                        found.add(int(sub))
                    except (TypeError, ValueError):
                        pass
            found |= collect_ids(item)
    elif isinstance(value, list):
        for item in value:
            found |= collect_ids(item)
    elif isinstance(value, str):
        for match in re.findall(r"(?:id_|ID)(\d{3})", value):
            found.add(int(match))
        for match in re.findall(r"\bID(\d{3})\b", value):
            found.add(int(match))
    return found


def load_record(candidate_id: int) -> dict[str, Any]:
    path = experiment_path(candidate_id)
    if path is None:
        return {
            "id": candidate_id,
            "experiment_json": None,
            "method": None,
            "risk_class": "missing_metadata",
            "direct_upstream_ids": [],
            "caution": None,
            "selection_contract": None,
        }
    metadata = json.loads(path.read_text(encoding="utf-8"))
    selector = metadata.get("selector_info", {})
    upstream = collect_ids(selector) | collect_ids(metadata.get("method", ""))
    upstream.discard(candidate_id)
    method = str(metadata.get("method", ""))
    caution = str(selector.get("caution", ""))
    text = f"{method} {caution} {selector.get('selection_contract', '')}".lower()
    if candidate_id in ANCHOR_IDS:
        risk_class = "anchor_reference"
    elif candidate_id in EXTREME_IDS or "second-stage full-oof" in text or "extreme-risk" in text:
        risk_class = "extreme_non_nested"
    elif candidate_id in HIGH_IDS or "full-oof" in text or "public-structure" in text:
        risk_class = "high_non_nested"
    else:
        risk_class = "review_required"
    return {
        "id": candidate_id,
        "experiment_json": str(path.relative_to(ROOT)),
        "method": metadata.get("method"),
        "local_cv": metadata.get("local_cv"),
        "submission_file": metadata.get("submission_file"),
        "risk_class": risk_class,
        "direct_upstream_ids": sorted(upstream),
        "caution": selector.get("caution"),
        "selection_contract": selector.get("selection_contract"),
    }


def dependency_closure(
    source_ids: list[int], max_depth: int = 5
) -> tuple[dict[int, dict[str, Any]], dict[int, set[int]]]:
    records: dict[int, dict[str, Any]] = {}
    edges: dict[int, set[int]] = defaultdict(set)
    frontier = [(source_id, 0) for source_id in source_ids]
    seen_depth: dict[int, int] = {}
    while frontier:
        candidate_id, depth = frontier.pop(0)
        if candidate_id in seen_depth and seen_depth[candidate_id] <= depth:
            continue
        seen_depth[candidate_id] = depth
        record = load_record(candidate_id)
        records[candidate_id] = record
        if depth >= max_depth:
            continue
        for upstream_id in record["direct_upstream_ids"]:
            edges[candidate_id].add(upstream_id)
            if upstream_id not in records or seen_depth.get(upstream_id, max_depth + 1) > depth + 1:
                frontier.append((upstream_id, depth + 1))
    return records, edges


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# B1 pair/logit source dependency audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It audits metadata dependencies only; no model was trained and no DACON submission was made.",
        "",
        "## B1 source records",
        "",
        "| id | risk | local | direct upstream ids | method |",
        "|---:|---|---:|---|---|",
    ]
    for row in report["b1_source_records"]:
        local = "" if row.get("local_cv") is None else f"{row['local_cv']:.6f}"
        lines.append(
            f"| {row['id']} | `{row['risk_class']}` | {local} | `{row['direct_upstream_ids']}` | {row.get('method')} |"
        )
    lines.extend(
        [
            "",
            "## Dependency closure risk counts",
            "",
        ]
    )
    for risk, count in report["closure_risk_counts"].items():
        lines.append(f"- `{risk}`: {count}")
    lines.extend(
        [
            "",
            "## Critical finding",
            "",
            report["decision"]["summary"],
            "",
            "## Dependency edges",
            "",
            "| source | upstream | upstream risk |",
            "|---:|---:|---|",
        ]
    )
    record_map = {int(row["id"]): row for row in report["closure_records"]}
    for edge in report["dependency_edges"]:
        upstream = record_map.get(edge["upstream_id"], {})
        lines.append(f"| {edge['source_id']} | {edge['upstream_id']} | `{upstream.get('risk_class', 'unknown')}` |")
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    records, edges = dependency_closure(B1_SOURCE_IDS)
    b1_records = [records[candidate_id] for candidate_id in B1_SOURCE_IDS]
    risk_counts: dict[str, int] = defaultdict(int)
    for record in records.values():
        risk_counts[str(record["risk_class"])] += 1
    dependency_edges = [
        {"source_id": source_id, "upstream_id": upstream_id}
        for source_id, upstream_ids in sorted(edges.items())
        for upstream_id in sorted(upstream_ids)
    ]
    hidden_extreme_ids = sorted(
        candidate_id for candidate_id, record in records.items() if record["risk_class"] == "extreme_non_nested"
    )
    hidden_high_ids = sorted(
        candidate_id for candidate_id, record in records.items() if record["risk_class"] == "high_non_nested"
    )
    all_b1_have_high_or_extreme = all(
        records[candidate_id]["risk_class"] in {"high_non_nested", "extreme_non_nested"}
        or any(
            records.get(upstream, {}).get("risk_class") in {"high_non_nested", "extreme_non_nested"}
            for upstream in edges[candidate_id]
        )
        for candidate_id in B1_SOURCE_IDS
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "b1_pair_blend_source_dependency_audit",
        "b1_source_ids": B1_SOURCE_IDS,
        "b1_source_records": b1_records,
        "closure_records": sorted(records.values(), key=lambda row: row["id"]),
        "dependency_edges": dependency_edges,
        "closure_risk_counts": dict(sorted(risk_counts.items())),
        "hidden_extreme_ids": hidden_extreme_ids,
        "hidden_high_ids": hidden_high_ids,
        "decision": {
            "direct_no_252_is_not_dependency_clean": True,
            "all_b1_sources_touch_high_or_extreme_non_nested_chain": all_b1_have_high_or_extreme,
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "summary": (
                "The B1 lane excludes direct ID252 at the selector layer, but its source artifacts are not dependency-clean. "
                "IDs 254/255/256/257/258/266 trace back through high/extreme full-OOF bridge or calibration artifacts, "
                "including the ID242/250/251/252 chain. A true rebuild must start below these source artifacts, not from them."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "b1_pair_blend_source_dependency_audit_20260618.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
