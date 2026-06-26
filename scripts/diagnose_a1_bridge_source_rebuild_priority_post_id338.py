#!/usr/bin/env python3
"""Prioritize A1 bridge source rebuild lanes after the post-ID338 replay.

This report-only diagnostic reads the fold-internal A1 bridge selector replay
and turns its per-fold choices into a source-layer rebuild map. It creates no
submission artifact; its purpose is to decide which bridge sources should be
rebuilt in true outer folds before any downstream pair/logit selector is
replayed.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
INPUT_PATH = REPORTS_DIR / "fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.json"
OUTPUT_PATH = REPORTS_DIR / "a1_bridge_source_rebuild_priority_post_id338_20260618.json"
KST = timezone(timedelta(hours=9))
Q240_IDS = {242, 243}

SOURCE_CATALOG: dict[int, dict[str, str]] = {
    235: {
        "role": "bridge_base",
        "risk": "mid_non_nested",
        "read": "ID235 is the base of the A1 bridge ladder; useful as a reconstruction anchor, not proof.",
    },
    238: {
        "role": "lower_frontier_source",
        "risk": "high_non_nested",
        "read": "ID238 is the lower aggressive pair/logit source below later q-calibration branches.",
    },
    239: {
        "role": "q240_free_recalibration",
        "risk": "high_non_nested",
        "read": "ID239 is ID238 isotonic recalibration; still full-OOF, but not in the q240 branch.",
    },
    240: {
        "role": "q240_free_recalibration",
        "risk": "high_non_nested",
        "read": "ID240 is ID238 q80/S4-isotonic recalibration; q240-free but calibration-heavy.",
    },
    241: {
        "role": "q240_free_recalibration",
        "risk": "high_non_nested",
        "read": "ID241 is ID238 q160/S4-isotonic recalibration; main q240-free replacement for ID242.",
    },
    242: {
        "role": "blocked_q240_calibration",
        "risk": "extreme_non_nested",
        "read": "ID242 dominates the all-extreme A1 selector but belongs to the blocked q240 chain.",
    },
    243: {
        "role": "blocked_q240_calibration",
        "risk": "extreme_non_nested",
        "read": "ID243 is the q240 all-target sibling; keep blocked with ID242 until lower layers are rebuilt.",
    },
}


def load_input() -> dict[str, Any]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def choice_key(choice: dict[str, Any]) -> str:
    if choice["kind"] == "base":
        return f"base:{int(choice['extreme_id']):03d}"
    return f"{choice['kind']}:{int(choice['extreme_id']):03d}:{choice['method']}:a{float(choice['alpha']):.3f}"


def choice_source_id(choice: dict[str, Any]) -> int:
    return int(choice["extreme_id"])


def best_scenario_row(rows: list[dict[str, Any]], scenario_name: str) -> dict[str, Any]:
    matches = [row for row in rows if row["scenario"]["name"] == scenario_name]
    if not matches:
        raise ValueError(f"No rows for scenario {scenario_name}")
    return min(matches, key=lambda row: (float(row["overall"]), float(row["robust_score"])))


def source_usage(row: dict[str, Any]) -> dict[str, Any]:
    source_counter: Counter[int] = Counter()
    bridge_counter: Counter[int] = Counter()
    target_predominant: dict[str, dict[str, Any]] = {}
    total = 0
    q240 = 0
    non_base = 0
    for target, choices in row["choices"].items():
        keys = Counter(choice_key(choice) for choice in choices)
        predominant_key, predominant_count = keys.most_common(1)[0]
        target_predominant[target] = {
            "choice": predominant_key,
            "count": int(predominant_count),
            "n_folds": len(choices),
        }
        for choice in choices:
            source_id = choice_source_id(choice)
            source_counter[source_id] += 1
            total += 1
            if source_id in Q240_IDS:
                q240 += 1
            if choice["kind"] != "base":
                non_base += 1
                bridge_counter[source_id] += 1
    return {
        "total_fold_target_choices": total,
        "source_counts": {str(key): int(value) for key, value in sorted(source_counter.items())},
        "bridge_source_counts": {str(key): int(value) for key, value in sorted(bridge_counter.items())},
        "q240_choice_rate": float(q240 / total) if total else 0.0,
        "non_base_choice_rate": float(non_base / total) if total else 0.0,
        "target_predominant_choices": target_predominant,
    }


def target_gap_rows(all_row: dict[str, Any], qfree_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target, all_loss in all_row["target_scores"].items():
        qfree_loss = float(qfree_row["target_scores"][target])
        gap = qfree_loss - float(all_loss)
        rows.append(
            {
                "target": target,
                "all_extreme_loss": float(all_loss),
                "q240_free_loss": qfree_loss,
                "q240_free_minus_all": float(gap),
            }
        )
    return sorted(rows, key=lambda row: row["q240_free_minus_all"], reverse=True)


def split_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, rows in report["splits"].items():
        all_best = min(rows, key=lambda row: (float(row["overall"]), float(row["robust_score"])))
        qfree_best = best_scenario_row(rows, "no_q240_extremes")
        q240_only_best = best_scenario_row(rows, "q240_only_stress")
        safe_best = best_scenario_row(rows, "safe_caps_all_extremes")
        summary[split_name] = {
            "all_extreme_best": {
                "scenario": all_best["scenario"]["name"],
                "cap": float(all_best["scenario"]["cap"]),
                "overall": float(all_best["overall"]),
                "robust_score": float(all_best["robust_score"]),
                "usage": source_usage(all_best),
            },
            "q240_free_best": {
                "scenario": qfree_best["scenario"]["name"],
                "cap": float(qfree_best["scenario"]["cap"]),
                "overall": float(qfree_best["overall"]),
                "robust_score": float(qfree_best["robust_score"]),
                "usage": source_usage(qfree_best),
            },
            "q240_only_best": {
                "scenario": q240_only_best["scenario"]["name"],
                "cap": float(q240_only_best["scenario"]["cap"]),
                "overall": float(q240_only_best["overall"]),
                "robust_score": float(q240_only_best["robust_score"]),
                "usage": source_usage(q240_only_best),
            },
            "safe_all_extreme_best": {
                "scenario": safe_best["scenario"]["name"],
                "cap": float(safe_best["scenario"]["cap"]),
                "overall": float(safe_best["overall"]),
                "robust_score": float(safe_best["robust_score"]),
                "usage": source_usage(safe_best),
            },
            "q240_free_gap_vs_all": {
                "overall": float(qfree_best["overall"] - all_best["overall"]),
                "robust_score": float(qfree_best["robust_score"] - all_best["robust_score"]),
                "target_rows": target_gap_rows(all_best, qfree_best),
            },
        }
    return summary


def build_recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    subject = summary["subject_hole_fold_internal"]
    loso = summary["loso_fold_internal"]
    qfree_subject = subject["q240_free_best"]
    qfree_loso = loso["q240_free_best"]
    all_subject = subject["all_extreme_best"]
    all_loso = loso["all_extreme_best"]
    return [
        {
            "rank": 1,
            "lane": "A1_q240_free_source_layer_rebuild",
            "source_ids": [238, 239, 241],
            "cap": 0.30,
            "why": (
                "This is the cleanest immediate A1 lane: it gives up about "
                f"{qfree_subject['overall'] - all_subject['overall']:.6f} subject-hole and "
                f"{qfree_loso['overall'] - all_loso['overall']:.6f} LOSO local score versus the "
                "q240-dominated all-extreme row, but avoids direct ID242/243 dependence."
            ),
            "target_map": qfree_subject["usage"]["target_predominant_choices"],
            "allowed_next_action": (
                "Rebuild source alternatives 238/239/241 inside frozen outer folds, then replay the "
                "cap0.30 target-wise bridge selector report-only."
            ),
            "blocked_action": "Do not create a DACON CSV from the existing q240-free OOF artifacts.",
        },
        {
            "rank": 2,
            "lane": "A1_s4_s3_micro_ablation",
            "source_ids": [238, 239],
            "cap": 0.30,
            "why": (
                "S4 consistently selects ID239 and S3 mostly stays on base with occasional ID238. "
                "This is lower-upside than the q240-free full target map, but cheaper and useful as "
                "a sanity check before a larger source rebuild."
            ),
            "target_map": {
                "S3": qfree_subject["usage"]["target_predominant_choices"].get("S3"),
                "S4": qfree_subject["usage"]["target_predominant_choices"].get("S4"),
            },
            "allowed_next_action": "Run a report-only nested S3/S4 ablation if compute budget is tight.",
            "blocked_action": "Do not tune S3/S4 by public-anchor movement alone.",
        },
        {
            "rank": 3,
            "lane": "A1_q240_branch_hold",
            "source_ids": [242, 243],
            "cap": 0.30,
            "why": (
                "The all-extreme row is strongest, with q240 choices in roughly "
                f"{all_subject['usage']['q240_choice_rate']:.1%} of subject-hole fold-target choices "
                f"and {all_loso['usage']['q240_choice_rate']:.1%} of LOSO choices. That is signal, "
                "but it is also exactly the blocked full-OOF q240 calibration chain."
            ),
            "target_map": all_subject["usage"]["target_predominant_choices"],
            "allowed_next_action": (
                "Keep as an explanatory stress target only; revisit after lower calibration/source "
                "generation is rebuilt without full-OOF shortcuts."
            ),
            "blocked_action": "Do not submit or descend into ID242/252 artifacts from this evidence.",
        },
    ]


def build_report() -> dict[str, Any]:
    input_report = load_input()
    summary = split_summary(input_report)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "a1_bridge_source_rebuild_priority_post_id338",
        "input_report": str(INPUT_PATH.relative_to(ROOT)),
        "source_artifacts_are_preexisting": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "source_catalog": SOURCE_CATALOG,
        "split_summary": summary,
        "recommendations": build_recommendations(summary),
        "decision": {
            "primary_next_lane": "A1_q240_free_source_layer_rebuild",
            "public_058_plausible": False,
            "create_submission_artifact": False,
            "reason": (
                "The A1 signal survives, but the strongest all-extreme selector is dominated by the "
                "blocked q240 chain. The safest useful next work is a q240-free source-layer rebuild, "
                "not a leaderboard submission."
            ),
        },
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 bridge source rebuild priority — post ID338",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook, base prediction, submission-log row, or DACON submission.",
        "",
        "## Split summary",
        "",
        "| split | all-extreme best | q240-free best | q240-free gap | all q240-choice rate | qfree sources |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split_name, split in report["split_summary"].items():
        all_best = split["all_extreme_best"]
        qfree = split["q240_free_best"]
        gap = split["q240_free_gap_vs_all"]
        qfree_sources = ", ".join(f"{k}:{v}" for k, v in qfree["usage"]["bridge_source_counts"].items())
        lines.append(
            f"| `{split_name}` | {fmt(all_best['overall'])} / robust {fmt(all_best['robust_score'])} | "
            f"{fmt(qfree['overall'])} / robust {fmt(qfree['robust_score'])} | "
            f"+{fmt(gap['overall'])} / robust +{fmt(gap['robust_score'])} | "
            f"{all_best['usage']['q240_choice_rate']:.1%} | `{qfree_sources}` |"
        )

    lines.extend(
        [
            "",
            "## q240-free target map",
            "",
            "| target | predominant q240-free choice | target loss gap vs all-extreme | read |",
            "|---|---|---:|---|",
        ]
    )
    subject = report["split_summary"]["subject_hole_fold_internal"]
    qfree_choices = subject["q240_free_best"]["usage"]["target_predominant_choices"]
    gap_by_target = {
        row["target"]: row["q240_free_minus_all"] for row in subject["q240_free_gap_vs_all"]["target_rows"]
    }
    for target, choice in qfree_choices.items():
        choice_text = choice["choice"]
        source_id = int(choice_text.split(":")[1])
        source_read = report["source_catalog"].get(source_id, {}).get("role", "unknown")
        lines.append(
            f"| `{target}` | `{choice_text}` ({choice['count']}/{choice['n_folds']} folds) | "
            f"{gap_by_target[target]:+.6f} | {source_read} |"
        )

    lines.extend(["", "## Recommendations", ""])
    for rec in report["recommendations"]:
        lines.extend(
            [
                f"### {rec['rank']}. `{rec['lane']}`",
                "",
                f"- Source IDs: `{rec['source_ids']}`",
                f"- Why: {rec['why']}",
                f"- Allowed next action: {rec['allowed_next_action']}",
                f"- Blocked action: {rec['blocked_action']}",
                "",
            ]
        )

    decision = report["decision"]
    lines.extend(
        [
            "## Decision",
            "",
            f"Primary next lane: `{decision['primary_next_lane']}`.",
            "",
            decision["reason"],
            "",
            "No submission artifact or DACON submission was created.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_report()
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_PATH.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
