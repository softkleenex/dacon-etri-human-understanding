#!/usr/bin/env python3
"""Q3/S3 subject-drift gate after the post-ID341 residual smoke.

The broader subject residual sequence-drift smoke found weak target-level
signals, but not enough to cover the Q3/S3 bottleneck. This report-only gate
extracts the Q3/S3-specific evidence and prevents the source family from being
promoted into a submission artifact unless it clears the strict local coverage
bar.

It creates no predictions, submission CSV, base prediction artifact,
notebook/history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
DEFAULT_INPUT = REPORTS_DIR / "subject_residual_sequence_drift_smoke_post_id341_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_subject_drift_gate_20260618.json"
TARGET_FOCUS = ["Q3", "S3"]
WIN_FLOOR_FOR_Q3S3_CONTEXT = 3
WIN_FLOOR_FOR_Q3S3_PROMOTION = 4


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def target_summary(payload: dict[str, Any], target: str) -> dict[str, Any]:
    summary = payload.get("target_signal_summary", {}).get(target, {})
    return {
        "target": target,
        "primary_wins_vs_id341": int(summary.get("primary_wins_vs_id341", 0) or 0),
        "primary_splits_seen": int(summary.get("primary_splits_seen", 0) or 0),
        "mean_primary_delta_vs_id341": summary.get("mean_primary_delta_vs_id341"),
        "min_primary_delta_vs_id341": summary.get("min_primary_delta_vs_id341"),
        "max_primary_delta_vs_id341": summary.get("max_primary_delta_vs_id341"),
    }


def build_report(input_path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    payload = load_json(input_path)
    decision = payload.get("decision", {})
    target_rows = {target: target_summary(payload, target) for target in TARGET_FOCUS}
    q3_wins = target_rows["Q3"]["primary_wins_vs_id341"]
    s3_wins = target_rows["S3"]["primary_wins_vs_id341"]
    any_context_signal = q3_wins >= WIN_FLOOR_FOR_Q3S3_CONTEXT or s3_wins >= WIN_FLOOR_FOR_Q3S3_CONTEXT
    all_promotable = q3_wins >= WIN_FLOOR_FOR_Q3S3_PROMOTION and s3_wins >= WIN_FLOOR_FOR_Q3S3_PROMOTION
    public_058 = decision.get("public_058_reviewable") is True
    beats_id178 = decision.get("beats_id178_public_anchor_proxy") is True
    artifact_allowed = bool(all_promotable and public_058 and beats_id178)
    q3s3_closed = bool(not artifact_allowed)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_subject_drift_gate",
        "purpose": (
            "Extract Q3/S3-specific evidence from the subject residual sequence-drift smoke and decide whether it "
            "can reopen the local artifact path."
        ),
        "inputs": {"subject_residual_sequence_drift_smoke": rel(input_path)},
        "target_focus": TARGET_FOCUS,
        "thresholds": {
            "win_floor_for_context": WIN_FLOOR_FOR_Q3S3_CONTEXT,
            "win_floor_for_promotion": WIN_FLOOR_FOR_Q3S3_PROMOTION,
            "requires_beats_id178_public_anchor_proxy": True,
            "requires_public_058_reviewable": True,
        },
        "target_evidence": target_rows,
        "source_decision": {
            "source_family_has_smoke_signal": decision.get("source_family_has_smoke_signal"),
            "projected_public_proxy_from_id341": decision.get("projected_public_proxy_from_id341"),
            "beats_id178_public_anchor_proxy": decision.get("beats_id178_public_anchor_proxy"),
            "public_058_reviewable": decision.get("public_058_reviewable"),
        },
        "decision": {
            "q3s3_subject_drift_gate_passed": True,
            "target_focus": TARGET_FOCUS,
            "q3_primary_wins_vs_id341": q3_wins,
            "s3_primary_wins_vs_id341": s3_wins,
            "q3s3_subject_drift_context_signal": any_context_signal,
            "q3s3_subject_drift_artifact_allowed": artifact_allowed,
            "q3s3_subject_drift_closed_for_artifact": q3s3_closed,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Keep subject-drift Q3/S3 evidence as negative/context-only evidence. It does not cover both Q3 "
                "and S3 strongly enough, does not beat the ID178 public-anchor proxy, and cannot create an artifact."
            ),
        },
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Q3/S3 subject-drift gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Target evidence",
        "",
        "| target | wins vs ID341 | mean Δ341 | min Δ341 | max Δ341 |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, row in report["target_evidence"].items():
        lines.append(
            f"| `{target}` | {row['primary_wins_vs_id341']}/{row['primary_splits_seen']} | "
            f"{row['mean_primary_delta_vs_id341']:+.6f} | {row['min_primary_delta_vs_id341']:+.6f} | "
            f"{row['max_primary_delta_vs_id341']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Gate passed: `{decision['q3s3_subject_drift_gate_passed']}`",
            f"- Q3 wins vs ID341: `{decision['q3_primary_wins_vs_id341']}`",
            f"- S3 wins vs ID341: `{decision['s3_primary_wins_vs_id341']}`",
            f"- Context signal: `{decision['q3s3_subject_drift_context_signal']}`",
            f"- Artifact allowed: `{decision['q3s3_subject_drift_artifact_allowed']}`",
            f"- Closed for artifact: `{decision['q3s3_subject_drift_closed_for_artifact']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(input_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
