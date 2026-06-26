#!/usr/bin/env python3
"""Prioritize post-A1 sequence pivots.

After the A1 q240-free replay failed, this report consolidates the surviving
non-calibration sequence evidence. It is intentionally report-only: no
submission CSV, notebook, base predictions, or DACON submission are created.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ANCHOR_IDS = [178, 179, 190, 191]
OUTPUT_PATH = REPORTS_DIR / "post_a1_sequence_pivot_priority_20260618.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostic_path(anchor_id: int) -> Path:
    return REPORTS_DIR / f"id{anchor_id}_sequence_summary_embedding_post_a1_20260618.json"


def best_info(target_report: dict[str, Any]) -> dict[str, Any]:
    best_name = target_report["best"]
    for name, info in target_report["top10"]:
        if name == best_name:
            return {"name": best_name, **info}
    return {"name": best_name, "kind": "unknown"}


def summarize_anchor(anchor_id: int) -> dict[str, Any]:
    report = load_json(diagnostic_path(anchor_id))
    target_rows = {}
    modified_targets = []
    max_delta_vs_base = 0.0
    sum_improvement = 0.0
    for target in TARGETS:
        target_report = report["targets"][target]
        info = best_info(target_report)
        improvement = float(target_report["improvement_vs_base"])
        sum_improvement += improvement
        if target_report["best"] != "base" and improvement > 0:
            modified_targets.append(target)
            max_delta_vs_base = max(max_delta_vs_base, float(info.get("test_max_abs_delta", 0.0)))
        target_rows[target] = {
            "best": target_report["best"],
            "base_loss": float(target_report["base_loss"]),
            "best_loss": float(target_report["best_loss"]),
            "improvement": improvement,
            "test_max_abs_delta_vs_base": float(info.get("test_max_abs_delta", 0.0)),
            "test_mean": float(info.get("test_mean", 0.0)) if "test_mean" in info else None,
        }
    return {
        "base_id": anchor_id,
        "base_mean": float(report["base_mean"]),
        "targetwise_best_mean": float(report["targetwise_best_mean"]),
        "mean_improvement": float(report["base_mean"] - report["targetwise_best_mean"]),
        "sum_target_improvement": sum_improvement,
        "modified_targets": modified_targets,
        "n_modified_targets": len(modified_targets),
        "max_test_delta_vs_base": max_delta_vs_base,
        "target_rows": target_rows,
        "diagnostic_report": str(diagnostic_path(anchor_id).relative_to(ROOT)),
    }


def load_s1_probe() -> dict[str, Any]:
    report = load_json(REPORTS_DIR / "recent_s1_probe_stress_review_20260618.json")
    ranked = {int(row["id"]): row for row in report["ranked_candidates"]}
    return {
        "safety_first_id": int(report["decision"]["safety_first_candidate"]),
        "local_gain_sibling": int(report["decision"]["local_gain_sibling"]),
        "id337": ranked[337],
        "id338": ranked[338],
        "decision": report["decision"],
    }


def combined_id178_q123_plus_id337_s1_estimate(anchor178: dict[str, Any], s1_probe: dict[str, Any]) -> dict[str, Any]:
    q123_improvement = sum(anchor178["target_rows"][target]["improvement"] for target in ["Q1", "Q2", "Q3"])
    # ID337 is exactly ID178 + S1 random-conv w0.05; use its measured local gain.
    id337_gain = float(s1_probe["id337"]["comparison"]["local_gain_vs_178"])
    estimated_local = float(anchor178["base_mean"] - (q123_improvement + id337_gain * len(TARGETS)) / len(TARGETS))
    return {
        "name": "id178_q123_sequence_summary_plus_id337_s1_safety_estimate",
        "base": 178,
        "components": [
            "Q1/Q2/Q3 sequence-summary embedding from ID178 post-A1 replay",
            "S1 ID337 low-movement random-conv safety probe",
        ],
        "estimated_local": estimated_local,
        "q123_improvement_sum": q123_improvement,
        "id337_mean_gain": id337_gain,
        "modified_targets": ["Q1", "Q2", "Q3", "S1"],
        "caution": (
            "This is only an additive estimate from existing diagnostics. It is not a generated artifact and "
            "has no test-geometry/projection gate yet."
        ),
    }


def build_report() -> dict[str, Any]:
    a1 = load_json(REPORTS_DIR / "a1_q240_free_nested_source_replay_post_id338_20260618.json")
    anchors = [summarize_anchor(anchor_id) for anchor_id in ANCHOR_IDS]
    anchors_ranked = sorted(anchors, key=lambda row: (row["targetwise_best_mean"], row["max_test_delta_vs_base"]))
    s1_probe = load_s1_probe()
    anchor178 = next(row for row in anchors if row["base_id"] == 178)
    combined_estimate = combined_id178_q123_plus_id337_s1_estimate(anchor178, s1_probe)
    recommendations = [
        {
            "rank": 1,
            "lane": "sequence_summary_anchor_gate",
            "read": (
                "Sequence-summary embeddings survive on ID178/179/190/191 after A1 failed. This is now the "
                "best non-calibration modeling lane, but it needs a movement/projection gate before artifact promotion."
            ),
            "preferred_screen": "ID178 Q1/Q2/Q3 only, then optionally combine with ID337 S1 safety.",
            "do_next": (
                "Build a report-only geometry gate for an ID178 Q1/Q2/Q3 sequence-summary overlay and the "
                "ID178 Q1/Q2/Q3 + ID337-S1 combined estimate before creating ID339."
            ),
        },
        {
            "rank": 2,
            "lane": "id337_controlled_public_probe_hold",
            "read": "ID337 remains the safest exact public probe, but it is not validation repair and needs exact approval.",
            "do_next": "Keep as explicit-approval-only; do not submit automatically.",
        },
        {
            "rank": 3,
            "lane": "a1_q240_free_deprioritized",
            "read": a1["decision"]["reason"],
            "do_next": "Do not spend heavy Kaggle/model-level compute on this exact A1 q240-free stack.",
        },
    ]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_a1_sequence_pivot_priority",
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "a1_q240_free_decision": a1["decision"],
        "sequence_summary_anchor_replay": anchors_ranked,
        "s1_probe": {
            "safety_first_id": s1_probe["safety_first_id"],
            "local_gain_sibling": s1_probe["local_gain_sibling"],
            "id337_local": float(s1_probe["id337"]["overall"]),
            "id337_proxy": float(s1_probe["id337"]["conservative_public_risk_proxy"]),
            "id338_local": float(s1_probe["id338"]["overall"]),
            "id338_proxy": float(s1_probe["id338"]["conservative_public_risk_proxy"]),
        },
        "combined_estimate": combined_estimate,
        "recommendations": recommendations,
        "decision": {
            "primary_next_lane": "sequence_summary_anchor_gate",
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "reason": (
                "After A1 q240-free collapsed, sequence-summary embeddings are the strongest surviving "
                "non-calibration signal. They improve all four checked anchors, but broad target movement means "
                "a geometry gate is required before promoting any ID339 artifact."
            ),
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-A1 sequence pivot priority",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No CSV, notebook, base predictions, submission-log row, or DACON submission was created.",
        "",
        "## Sequence-summary anchor replay",
        "",
        "| rank | base | base local | best local | gain | modified targets | max test Δ vs base |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(report["sequence_summary_anchor_replay"], start=1):
        lines.append(
            f"| {rank} | {row['base_id']} | {row['base_mean']:.6f} | "
            f"{row['targetwise_best_mean']:.6f} | {row['mean_improvement']:+.6f} | "
            f"`{row['modified_targets']}` | {row['max_test_delta_vs_base']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## ID178 target improvements",
            "",
            "| target | best | improvement | max test Δ vs ID178 base |",
            "|---|---|---:|---:|",
        ]
    )
    id178 = next(row for row in report["sequence_summary_anchor_replay"] if row["base_id"] == 178)
    for target in TARGETS:
        row = id178["target_rows"][target]
        lines.append(
            f"| `{target}` | `{row['best']}` | {row['improvement']:+.6f} | {row['test_max_abs_delta_vs_base']:.6f} |"
        )
    combined = report["combined_estimate"]
    lines.extend(
        [
            "",
            "## Combined candidate estimate",
            "",
            f"- Name: `{combined['name']}`",
            f"- Estimated local: `{combined['estimated_local']:.6f}`",
            f"- Modified targets: `{combined['modified_targets']}`",
            f"- Caution: {combined['caution']}",
            "",
            "## Recommendations",
            "",
        ]
    )
    for rec in report["recommendations"]:
        lines.extend(
            [
                f"### {rec['rank']}. `{rec['lane']}`",
                "",
                f"- Read: {rec['read']}",
                f"- Next: {rec['do_next']}",
                "",
            ]
        )
    lines.extend(["## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_PATH.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
