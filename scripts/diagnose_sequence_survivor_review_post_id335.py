#!/usr/bin/env python3
"""Review top sequence-temporal survivors after excluding the ID242 chain.

This report-only diagnostic compares the surviving non-ID242 candidates against
confirmed public anchors. It does not create predictions or submission files.
The goal is to decide whether ID179 still deserves to be the cautious manual
review anchor, or whether lower-local sequence descendants justify more work.
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
    TARGETS,
    align,
    geometry,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.diagnose_public_anchor_rerank_post_id335 import load_log, log_row  # noqa: E402

REPORTS_DIR = ROOT / "reports"
SURVIVOR_REPORT = REPORTS_DIR / "surviving_candidates_excluding_id242_post_id335_20260617.json"
KST = timezone(timedelta(hours=9))
DEFAULT_CANDIDATE_IDS = [178, 179, 190, 191, 193, 177, 198, 200, 203, 206, 180, 334, 335]
PUBLIC_ANCHORS = {
    178: 0.6014113009,
    287: 0.6698787318,
    329: 0.6017908577,
}


def experiment_path(candidate_id: int) -> Path | None:
    matches = sorted(REPORTS_DIR.glob(f"experiment_{candidate_id}_*.json"))
    return matches[0] if len(matches) == 1 else None


def dependency_summary(candidate_id: int) -> dict[str, Any]:
    path = experiment_path(candidate_id)
    if path is None:
        return {"experiment_json": None, "method": None, "upstream_hint": None, "selection_contract": None}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    selector = metadata.get("selector_info", {})
    upstream = (
        selector.get("base_source")
        or selector.get("diagnostic_report")
        or selector.get("artifact_ids")
        or selector.get("anchor_id")
    )
    return {
        "experiment_json": str(path.relative_to(ROOT)),
        "method": metadata.get("method"),
        "upstream_hint": upstream,
        "selection_contract": selector.get("selection_contract"),
        "caution": selector.get("caution"),
    }


def target_deltas(candidate: dict[str, Any], anchor: dict[str, Any]) -> dict[str, float]:
    return {target: float(candidate["target_scores"][target] - anchor["target_scores"][target]) for target in TARGETS}


def review_candidate(
    *,
    candidate_id: int,
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
    dep = dependency_summary(candidate_id)
    return {
        "id": candidate_id,
        "submission_file": log_info.get("submission_file", ""),
        "status_clean": log_info.get("status_clean", ""),
        "public_lb": PUBLIC_ANCHORS.get(candidate_id),
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
        "dependency": dep,
    }


def score_review_row(row: dict[str, Any], anchor_row: dict[str, Any]) -> dict[str, Any]:
    local_gain = float(anchor_row["overall"] - row["overall"])
    robust_gain = float(anchor_row["robust_score"] - row["robust_score"])
    movement = float(row["geometry"]["max_delta_vs_178"])
    failed_projection = float(row["geometry"]["projection_failed"])
    risk_score = (
        2.0 * max(0.0, failed_projection)
        + 0.8 * max(0.0, movement - 0.08)
        + 1.0 * max(0.0, -robust_gain)
        + 0.5 * max(0.0, -local_gain)
    )
    efficiency = local_gain / movement if movement > 0 else 0.0
    return {
        "local_gain_vs_178": local_gain,
        "robust_gain_vs_178": robust_gain,
        "movement_efficiency": float(efficiency),
        "risk_score": float(risk_score),
        "review_score": float(-risk_score + 0.2 * local_gain + 0.1 * robust_gain),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sequence survivor review — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Candidate comparison",
        "",
        "| rank | id | local | robust | gain vs 178 | robust gain | max Δ178 | proj failed | worst target | read |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(report["ranked_candidates"], start=1):
        review = row["review_metrics"]
        slices = row["slice_summary"]
        read = row["review_read"]
        lines.append(
            f"| {rank} | {row['id']} | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{review['local_gain_vs_178']:+.6f} | {review['robust_gain_vs_178']:+.6f} | "
            f"{row['geometry']['max_delta_vs_178']:.6f} | {row['geometry']['projection_failed']:+.4f} | "
            f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | {read} |"
        )
    lines.extend(
        [
            "",
            "## Target deltas versus ID178",
            "",
            "Negative is locally better than ID178.",
            "",
            "| id | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["ranked_candidates"]:
        deltas = row["target_delta_vs_178"]
        lines.append(f"| {row['id']} | " + " | ".join(f"{deltas[target]:+.6f}" for target in TARGETS) + " |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report["decision"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def classify_candidate(row: dict[str, Any]) -> str:
    review = row["review_metrics"]
    max_delta = row["geometry"]["max_delta_vs_178"]
    failed_projection = row["geometry"]["projection_failed"]
    if row["id"] == 179:
        return "cleanest low-movement same-family review anchor"
    if failed_projection > 0.03:
        return "reject: failed-direction projection too high"
    if max_delta > 0.12 and review["local_gain_vs_178"] < 0.003:
        return "watch-only: more movement than ID179 for tiny gain"
    if max_delta > 0.13:
        return "watch-only: high movement sequence descendant"
    if review["robust_gain_vs_178"] > 0 and review["local_gain_vs_178"] > 0:
        return "research-only: locally better but not public-proof"
    return "reference/watch-only"


def run(output_json: Path, candidate_ids: list[int]) -> dict[str, Any]:
    train, test = load_train_test()
    log = load_log()
    anchor_tests = {
        candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in [178, 287, 329, 335]
    }
    rows = [
        review_candidate(candidate_id=candidate_id, train=train, test=test, anchor_tests=anchor_tests, log=log)
        for candidate_id in candidate_ids
    ]
    anchor_row = next(row for row in rows if row["id"] == 178)
    for row in rows:
        row["review_metrics"] = score_review_row(row, anchor_row)
        row["target_delta_vs_178"] = target_deltas(row, anchor_row)
        row["review_read"] = classify_candidate(row)
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["id"] == 179 else 1,
            row["review_metrics"]["risk_score"],
            -row["review_metrics"]["local_gain_vs_178"],
            row["geometry"]["max_delta_vs_178"],
        ),
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sequence_survivor_review_post_id335",
        "candidate_ids": candidate_ids,
        "public_anchors": PUBLIC_ANCHORS,
        "source_report": str(SURVIVOR_REPORT.relative_to(ROOT)),
        "ranked_candidates": ranked,
        "decision": {
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "manual_probe_anchor": 179,
            "reason": (
                "ID179 remains the cleanest low-movement same-family review anchor after excluding the ID242 chain. "
                "Several sequence descendants improve local CV slightly more, but they move much farther from the "
                "confirmed ID178 public anchor and still use non-strict/OOF-selected sequence overlays. This review "
                "does not authorize a DACON submission."
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
        default=REPORTS_DIR / "sequence_survivor_review_post_id335_20260617.json",
    )
    parser.add_argument("--candidate-ids", type=int, nargs="+", default=DEFAULT_CANDIDATE_IDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json, args.candidate_ids)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    top = report["ranked_candidates"][0]
    print(
        f"top review anchor: id={top['id']} local={top['overall']:.6f} "
        f"robust={top['robust_score']:.6f} max_delta178={top['geometry']['max_delta_vs_178']:.6f}"
    )
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
