#!/usr/bin/env python3
"""Target-level salvage review for the A1 q240-free replay after ID341.

The q240-free A1 source replay collapsed at the overall level. This report asks
whether any individual target still improves ID235 under both source-replay
splits, and whether a conservative target-only salvage rule would be worth
turning into a future artifact.

Report-only: no CSV, notebook/history artifact, base prediction artifact,
submission-log row, or DACON submission is created.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
A1_REPLAY_PATH = REPORTS_DIR / "a1_q240_free_nested_source_replay_post_id338_20260618.json"
SELECTOR_REPLAY_PATH = REPORTS_DIR / "fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.json"
OUTPUT_PATH = REPORTS_DIR / "a1_q240_free_target_salvage_post_id341_20260618.json"
KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SPLIT_MAP = {
    "subject_hole_source_replay": "subject_hole_fold_internal",
    "loso_source_replay": "loso_fold_internal",
}


def best_q240_free_selector_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split_name, split_rows in report["splits"].items():
        qfree = [row for row in split_rows if row["scenario"]["name"] == "no_q240_extremes"]
        rows[split_name] = min(qfree, key=lambda row: (float(row["overall"]), float(row["robust_score"])))
    return rows


def mean_target_score(target_scores: dict[str, float]) -> float:
    return float(np.mean([float(target_scores[target]) for target in TARGETS]))


def classify_target(per_split: dict[str, dict[str, float]]) -> str:
    improves_both = all(row["replay_delta_vs_base"] < 0 for row in per_split.values())
    selector_improves_both = all(row["selector_delta_vs_base"] < 0 for row in per_split.values())
    max_regret = max(row["replay_regret_vs_selector"] for row in per_split.values())
    min_gain = min(-row["replay_delta_vs_base"] for row in per_split.values())
    if improves_both and max_regret <= 0.03 and min_gain >= 0.005:
        return "stable_salvage_candidate"
    if improves_both:
        return "weak_salvage_watch"
    if selector_improves_both:
        return "selector_only_trap"
    return "reject"


def build_report() -> dict[str, Any]:
    replay = json.loads(A1_REPLAY_PATH.read_text(encoding="utf-8"))
    selector = json.loads(SELECTOR_REPLAY_PATH.read_text(encoding="utf-8"))
    selector_rows = best_q240_free_selector_rows(selector)
    base_scores = replay["base_235_summary"]["target_scores"]
    replay_rows = {row["split"]: row for row in replay["splits"]}

    target_rows = []
    for target in TARGETS:
        per_split: dict[str, dict[str, float]] = {}
        for replay_split, selector_split in SPLIT_MAP.items():
            replay_score = float(replay_rows[replay_split]["target_scores"][target])
            base_score = float(base_scores[target])
            selector_score = float(selector_rows[selector_split]["target_scores"][target])
            per_split[replay_split] = {
                "base": base_score,
                "source_replay": replay_score,
                "selector_only_qfree": selector_score,
                "replay_delta_vs_base": replay_score - base_score,
                "selector_delta_vs_base": selector_score - base_score,
                "replay_regret_vs_selector": replay_score - selector_score,
            }
        classification = classify_target(per_split)
        target_rows.append(
            {
                "target": target,
                "classification": classification,
                "per_split": per_split,
                "mean_replay_delta_vs_base": float(
                    np.mean([row["replay_delta_vs_base"] for row in per_split.values()])
                ),
                "max_replay_regret_vs_selector": float(
                    max(row["replay_regret_vs_selector"] for row in per_split.values())
                ),
            }
        )
    stable_targets = [row["target"] for row in target_rows if row["classification"] == "stable_salvage_candidate"]
    weak_watch_targets = [row["target"] for row in target_rows if row["classification"] == "weak_salvage_watch"]
    selector_trap_targets = [row["target"] for row in target_rows if row["classification"] == "selector_only_trap"]

    hybrid_rows = {}
    for replay_split, selector_split in SPLIT_MAP.items():
        hybrid_scores = {}
        weak_hybrid_scores = {}
        for target in TARGETS:
            replay_score = float(replay_rows[replay_split]["target_scores"][target])
            base_score = float(base_scores[target])
            hybrid_scores[target] = replay_score if target in stable_targets else base_score
            weak_hybrid_scores[target] = (
                replay_score if target in {*stable_targets, *weak_watch_targets} else base_score
            )
        base_mean = mean_target_score(base_scores)
        replay_mean = mean_target_score(replay_rows[replay_split]["target_scores"])
        selector_mean = mean_target_score(selector_rows[selector_split]["target_scores"])
        stable_mean = mean_target_score(hybrid_scores)
        weak_mean = mean_target_score(weak_hybrid_scores)
        hybrid_rows[replay_split] = {
            "base_id235_mean": base_mean,
            "full_source_replay_mean": replay_mean,
            "selector_only_qfree_mean": selector_mean,
            "stable_salvage_hybrid_mean": stable_mean,
            "weak_watch_hybrid_mean": weak_mean,
            "stable_salvage_delta_vs_base": stable_mean - base_mean,
            "weak_watch_delta_vs_base": weak_mean - base_mean,
            "stable_salvage_target_scores": hybrid_scores,
            "weak_watch_target_scores": weak_hybrid_scores,
        }

    promote = bool(stable_targets) and all(row["stable_salvage_delta_vs_base"] <= -0.01 for row in hybrid_rows.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "a1_q240_free_target_salvage_post_id341",
        "input_reports": {
            "source_replay": str(A1_REPLAY_PATH.relative_to(ROOT)),
            "selector_replay": str(SELECTOR_REPLAY_PATH.relative_to(ROOT)),
        },
        "target_rows": sorted(
            target_rows,
            key=lambda row: (
                {
                    "stable_salvage_candidate": 0,
                    "weak_salvage_watch": 1,
                    "selector_only_trap": 2,
                    "reject": 3,
                }[row["classification"]],
                row["mean_replay_delta_vs_base"],
                row["target"],
            ),
        ),
        "stable_salvage_targets": stable_targets,
        "weak_watch_targets": weak_watch_targets,
        "selector_only_trap_targets": selector_trap_targets,
        "hybrid_rows": hybrid_rows,
        "decision": {
            "promote_submission_candidate_now": False,
            "create_artifact_now": False,
            "submit_now": False,
            "has_stable_salvage_target": bool(stable_targets),
            "stable_salvage_artifact_review_pass": promote,
            "public_058_plausible": False,
            "next_action": (
                "If no stable salvage target passes the hybrid threshold, stop local A1 q240-free work and pivot to "
                "new source families. If stable targets exist, require a full prediction-geometry gate before any artifact."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 q240-free target salvage review — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.",
        "",
        "## Target classification",
        "",
        "| target | class | mean replay Δ vs base | max regret vs selector | SH replay/base/selector | LOSO replay/base/selector |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in report["target_rows"]:
        sh = row["per_split"]["subject_hole_source_replay"]
        loso = row["per_split"]["loso_source_replay"]
        lines.append(
            f"| `{row['target']}` | `{row['classification']}` | {row['mean_replay_delta_vs_base']:+.6f} | "
            f"{row['max_replay_regret_vs_selector']:+.6f} | "
            f"{sh['source_replay']:.6f}/{sh['base']:.6f}/{sh['selector_only_qfree']:.6f} | "
            f"{loso['source_replay']:.6f}/{loso['base']:.6f}/{loso['selector_only_qfree']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid sanity check",
            "",
            "| split | base | full replay | selector-only qfree | stable-target hybrid | weak-watch hybrid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name, row in report["hybrid_rows"].items():
        lines.append(
            f"| `{split_name}` | {row['base_id235_mean']:.6f} | {row['full_source_replay_mean']:.6f} | "
            f"{row['selector_only_qfree_mean']:.6f} | {row['stable_salvage_hybrid_mean']:.6f} "
            f"({row['stable_salvage_delta_vs_base']:+.6f}) | {row['weak_watch_hybrid_mean']:.6f} "
            f"({row['weak_watch_delta_vs_base']:+.6f}) |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stable salvage targets: `{report['stable_salvage_targets']}`",
            f"- Weak watch targets: `{report['weak_watch_targets']}`",
            f"- Selector-only trap targets: `{report['selector_only_trap_targets']}`",
            f"- Stable salvage artifact review pass: `{decision['stable_salvage_artifact_review_pass']}`",
            "- Create artifact now: **no**",
            "- Submit now: **no**",
            "",
            decision["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
