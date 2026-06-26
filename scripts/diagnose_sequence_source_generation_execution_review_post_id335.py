#!/usr/bin/env python3
"""Review executed report-only sequence source-generation diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EXECUTED_REPORTS = [
    {
        "lane": "Q2_sequence_summary_embedding",
        "path": REPORTS_DIR / "id191_sequence_summary_embedding_post_id335_20260617.json",
        "targets_from_plan": ["Q2", "Q3"],
    },
    {
        "lane": "S3_smartphone_light_compact",
        "path": REPORTS_DIR / "id193_smartphone_light_compact_post_id335_20260617.json",
        "targets_from_plan": ["S3"],
    },
    {
        "lane": "Q3_S2_S4_wavelet_spectral",
        "path": REPORTS_DIR / "id203_wavelet_spectral_sequence_post_id335_20260617.json",
        "targets_from_plan": ["Q3", "S2", "S4"],
    },
]


def load_lane(row: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(row["path"].read_text(encoding="utf-8"))
    target_rows = []
    for target in TARGETS:
        target_info = report["targets"][target]
        base_loss = float(target_info["base_loss"])
        best_loss = float(target_info["best_loss"])
        best_name = str(target_info["best"])
        target_rows.append(
            {
                "target": target,
                "base_loss": base_loss,
                "best_loss": best_loss,
                "improvement": base_loss - best_loss,
                "best": best_name,
                "changed": best_name != "base",
            }
        )
    return {
        "lane": row["lane"],
        "path": str(row["path"]),
        "targets_from_plan": row["targets_from_plan"],
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "mean_improvement": float(report["base_mean"] - report["targetwise_best_mean"]),
        "target_rows": target_rows,
        "changed_targets": [target_row["target"] for target_row in target_rows if target_row["changed"]],
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sequence source-generation execution review — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It summarizes diagnostics already executed without `--promote`; no submission artifact or DACON submission was created.",
        "",
        "## Lane summary",
        "",
        "| lane | base id | base mean | best mean | improvement | changed targets |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for lane in report["lanes"]:
        lines.append(
            f"| `{lane['lane']}` | {lane['base_id']} | {lane['base_mean']:.6f} | "
            f"{lane['targetwise_best_mean']:.6f} | {lane['mean_improvement']:+.6f} | "
            f"`{lane['changed_targets']}` |"
        )
    lines.extend(["", "## Target-level details", ""])
    for lane in report["lanes"]:
        lines.extend(
            [
                f"### {lane['lane']}",
                "",
                "| target | best | base loss | best loss | improvement |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for target_row in lane["target_rows"]:
            lines.append(
                f"| {target_row['target']} | `{target_row['best']}` | {target_row['base_loss']:.6f} | "
                f"{target_row['best_loss']:.6f} | {target_row['improvement']:+.6f} |"
            )
        lines.append("")
    lines.extend(["## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    lanes = [load_lane(row) for row in EXECUTED_REPORTS]
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sequence_source_generation_execution_review_post_id335",
        "lanes": lanes,
        "decision": {
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "reason": (
                "Executed report-only source diagnostics reproduce the existing sequence-source gains, but the gains "
                "are small and remain selected under public-structure diagnostics rather than true model-level nested "
                "validation. Use these results to guide future Kaggle source generation, not artifact promotion."
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
        default=REPORTS_DIR / "sequence_source_generation_execution_review_post_id335_20260617.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    for lane in report["lanes"]:
        print(
            f"{lane['lane']}: base={lane['base_mean']:.6f} best={lane['targetwise_best_mean']:.6f} "
            f"improvement={lane['mean_improvement']:+.6f} changed={lane['changed_targets']}"
        )
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
