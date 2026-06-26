#!/usr/bin/env python3
"""Report-only stress review for recent S1 probe artifacts.

This diagnostic uses existing OOF/test artifacts only. It does not train models,
create submissions, or call DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    align,
    geometry,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.diagnose_public_anchor_rerank_post_id335 import conservative_proxy, load_log, log_row  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
DEFAULT_CANDIDATE_IDS = [178, 179, 336, 337, 338, 334, 335]
ANCHOR_IDS = [178, 287, 329, 335]
PUBLIC_ID178 = 0.6014113009
ID178_LOCAL_REFERENCE = 0.5386382667903545
ID178_PUBLIC_GAP = PUBLIC_ID178 - ID178_LOCAL_REFERENCE


def review_candidate(
    candidate_id: int,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    anchor_tests: dict[int, pd.DataFrame],
    log: pd.DataFrame,
) -> dict[str, Any]:
    oof = align(load_artifact("oof", candidate_id), train)
    test_pred = align(load_artifact("test", candidate_id), test)
    overall, target_scores = local_scores(train, oof)
    slices = slice_summary(train, oof)
    geom = geometry(test_pred, anchor_tests)
    log_info = log_row(log, candidate_id)
    proxy_row = {
        "local_cv": overall,
        "mean_delta_vs_178": geom["mean_delta_vs_178"],
        "max_delta_vs_178": geom["max_delta_vs_178"],
        "mean_delta_vs_329": geom["mean_delta_vs_329"],
        "max_delta_vs_329": geom["max_delta_vs_329"],
        "projection_failed": geom["projection_failed"],
    }
    return {
        "id": candidate_id,
        "submission_file": log_info.get("submission_file", ""),
        "status_clean": log_info.get("status_clean", ""),
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "geometry": geom,
        "conservative_public_risk_proxy": conservative_proxy(proxy_row),
        "naive_id178_gap_projected_public": float(overall + ID178_PUBLIC_GAP),
    }


def add_comparison_metrics(rows: list[dict[str, Any]]) -> None:
    by_id = {row["id"]: row for row in rows}
    id178 = by_id[178]
    id337 = by_id.get(337)
    for row in rows:
        row["comparison"] = {
            "local_gain_vs_178": float(id178["overall"] - row["overall"]),
            "robust_gain_vs_178": float(id178["robust_score"] - row["robust_score"]),
            "proxy_delta_vs_337": (
                None
                if id337 is None
                else float(row["conservative_public_risk_proxy"] - id337["conservative_public_risk_proxy"])
            ),
            "local_delta_vs_337": None if id337 is None else float(row["overall"] - id337["overall"]),
        }


def classify(row: dict[str, Any]) -> str:
    geom = row["geometry"]
    comp = row["comparison"]
    if row["id"] == 337:
        return "safety-first S1 probe"
    if row["id"] == 338:
        return "local-gain sibling; higher public-anchor movement"
    if row["id"] == 336:
        return "local-best ID178 S1 weight; movement higher than ID337"
    if row["id"] == 179:
        return "same-family anchor reference"
    if geom["projection_failed"] > 0.02:
        return "reject/watch: failed-direction projection"
    if comp["robust_gain_vs_178"] > 0 and comp["local_gain_vs_178"] > 0:
        return "research-only positive local/robust signal"
    return "reference/watch"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Recent S1 probe stress review",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. Existing OOF/test artifacts were scanned; no model was trained and no DACON submission was made.",
        "",
        "## Candidate comparison",
        "",
        "| rank | id | local | robust | gain vs 178 | robust gain | max Δ178 | failed proj | proxy | worst target | read |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(report["ranked_candidates"], start=1):
        comp = row["comparison"]
        geom = row["geometry"]
        slices = row["slice_summary"]
        lines.append(
            f"| {rank} | {row['id']} | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{comp['local_gain_vs_178']:+.6f} | {comp['robust_gain_vs_178']:+.6f} | "
            f"{geom['max_delta_vs_178']:.6f} | {geom['projection_failed']:+.6f} | "
            f"{row['conservative_public_risk_proxy']:.6f} | "
            f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | {row['review_read']} |"
        )
    lines.extend(
        [
            "",
            "## S1 target losses",
            "",
            "| id | S1 loss | Δ S1 vs 178 | Δ S1 vs 337 |",
            "|---:|---:|---:|---:|",
        ]
    )
    by_id = {row["id"]: row for row in report["ranked_candidates"]}
    s1_178 = by_id[178]["target_scores"]["S1"]
    s1_337 = by_id[337]["target_scores"]["S1"]
    for row in report["ranked_candidates"]:
        s1_loss = row["target_scores"]["S1"]
        lines.append(f"| {row['id']} | {s1_loss:.6f} | {s1_loss - s1_178:+.6f} | {s1_loss - s1_337:+.6f} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"]["summary"],
            "",
        ]
    )
    return "\n".join(lines)


def run(output_json: Path, candidate_ids: list[int]) -> dict[str, Any]:
    train, test = load_train_test()
    log = load_log()
    anchor_tests = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in ANCHOR_IDS}
    rows = [
        review_candidate(candidate_id, train=train, test=test, anchor_tests=anchor_tests, log=log)
        for candidate_id in candidate_ids
    ]
    add_comparison_metrics(rows)
    for row in rows:
        row["review_read"] = classify(row)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["conservative_public_risk_proxy"],
            row["geometry"]["max_delta_vs_178"],
            row["overall"],
        ),
    )
    id337 = next(row for row in rows if row["id"] == 337)
    id338 = next(row for row in rows if row["id"] == 338)
    decision = {
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "public_058_plausible": False,
        "safety_first_candidate": 337,
        "local_gain_sibling": 338,
        "summary": (
            "ID337 remains the safety-first S1 probe: it has the best conservative proxy and lowest movement from ID178. "
            f"ID338 is locally stronger by {id337['overall'] - id338['overall']:.6f}, but its proxy is worse by "
            f"{id338['conservative_public_risk_proxy'] - id337['conservative_public_risk_proxy']:.6f}. "
            "Neither candidate repairs validation or proves Public 0.58; both remain explicit-approval-only controlled probes."
        ),
    }
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "recent_s1_probe_stress_review",
        "candidate_ids": candidate_ids,
        "anchor_ids": ANCHOR_IDS,
        "ranked_candidates": ranked,
        "decision": decision,
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
        default=REPORTS_DIR / "recent_s1_probe_stress_review_20260618.json",
    )
    parser.add_argument("--candidate-ids", type=int, nargs="+", default=DEFAULT_CANDIDATE_IDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json, args.candidate_ids)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
