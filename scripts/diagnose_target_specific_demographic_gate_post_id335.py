#!/usr/bin/env python3
"""Derive a target-specific demographic gate from the strict audit outputs.

This is report-only and does not train models or create submission artifacts.
It translates the global demographic audit into a conservative per-target
allow/deny list for future experiments.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))


def uses_demo(candidate_name: str) -> bool:
    return candidate_name.startswith("demo_only") or "plus_demo" in candidate_name


def target_demo_delta(split: dict[str, Any], target: str) -> dict[str, Any]:
    rows = split["rows"]
    demo_rows = [row for row in rows if uses_demo(str(row["candidate"]))]
    no_demo_rows = [row for row in rows if not uses_demo(str(row["candidate"]))]
    best_demo = min(demo_rows, key=lambda row: float(row["target_losses"][target]))
    best_no_demo = min(no_demo_rows, key=lambda row: float(row["target_losses"][target]))
    demo_loss = float(best_demo["target_losses"][target])
    no_demo_loss = float(best_no_demo["target_losses"][target])
    return {
        "best_demo_candidate": best_demo["candidate"],
        "best_demo_loss": demo_loss,
        "best_no_demo_candidate": best_no_demo["candidate"],
        "best_no_demo_loss": no_demo_loss,
        "delta_demo_minus_no_demo": demo_loss - no_demo_loss,
        "demo_wins": demo_loss < no_demo_loss,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Target-specific demographic gate — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no model predictions, submission CSV, notebook, submission-log row, or DACON submission.",
        "",
        "## Gate summary",
        "",
        "| target | decision | subject-hole Δ demo | LOSO Δ demo | reason |",
        "|---|---|---:|---:|---|",
    ]
    for target, row in report["targets"].items():
        lines.append(
            f"| {target} | `{row['decision']}` | "
            f"{row['subject_hole']['delta_demo_minus_no_demo']:+.6f} | "
            f"{row['loso']['delta_demo_minus_no_demo']:+.6f} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Only `allow_research` targets may be used in a future target-specific demographic branch.",
            "- This gate is not promotion evidence by itself; any future demographic artifact still needs fold-safe OOF/test predictions and ID178/ID329/ID287 anchor geometry.",
            "- Global demographic overlays remain frozen because LOSO did not support them.",
            "",
        ]
    )
    return "\n".join(lines)


def run(input_json: Path, output_json: Path) -> dict[str, Any]:
    audit = json.loads(input_json.read_text())
    subject_hole = audit["splits"]["subject_hole_temporal"]
    loso = audit["splits"]["leave_one_subject_out_stress"]
    targets: dict[str, Any] = {}
    allow_targets = []
    watch_targets = []
    reject_targets = []

    for target in TARGETS:
        sh = target_demo_delta(subject_hole, target)
        lo = target_demo_delta(loso, target)
        if sh["demo_wins"] and lo["demo_wins"]:
            decision = "allow_research"
            reason = "demo wins under both subject-hole and LOSO stress"
            allow_targets.append(target)
        elif lo["demo_wins"]:
            decision = "watch_loso_only"
            reason = "demo wins under LOSO but not subject-hole"
            watch_targets.append(target)
        else:
            decision = "reject_for_now"
            reason = "demo does not win under LOSO stress"
            reject_targets.append(target)
        targets[target] = {
            "decision": decision,
            "reason": reason,
            "subject_hole": sh,
            "loso": lo,
        }

    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "target_specific_demographic_gate_post_id335",
        "source_audit": str(input_json.relative_to(ROOT)),
        "targets": targets,
        "allow_research_targets": allow_targets,
        "watch_loso_only_targets": watch_targets,
        "reject_targets": reject_targets,
        "decision": {
            "promote_submission_candidate": False,
            "public_058_plausible": False,
            "next_allowed_branch": "target-specific demographic research only for allow_research targets",
            "reason": "The gate narrows demographic/static features to targets with agreement across subject-hole and LOSO stress.",
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=REPORTS_DIR / "strict_demographic_feature_audit_kaggle_20260617.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "target_specific_demographic_gate_post_id335_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.input_json, args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"] | {"allow_research_targets": report["allow_research_targets"]}, indent=2))


if __name__ == "__main__":
    main()
