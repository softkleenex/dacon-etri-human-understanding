#!/usr/bin/env python3
"""Decide the next action after post-ID341 Kaggle remote result summary.

This local no-submit gate consumes the remote result summary and output-intake
report. It decides whether to keep waiting for Kaggle outputs, close the remote
lane, or allow a separate artifact-contract review. It never creates predictions,
submission CSVs, base prediction artifacts, notebook history, submission-log rows,
Kaggle runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_REMOTE_SUMMARY = REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json"
DEFAULT_INTAKE_REPORT = REPORTS_DIR / "post_id341_kaggle_output_intake_gate_20260618.json"
KST = timezone(timedelta(hours=9))

ID178_PUBLIC_ANCHOR = 0.6014113009
PUBLIC_058_TARGET = 0.58


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_for(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "error": None}
    try:
        payload = load_json(path)
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": True,
            "bytes": path.stat().st_size,
            "decision": payload.get("decision", {}),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def best_proxy_margin(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_proxy_row") or {}
    min_proxy = best.get("min_proxy")
    if not isinstance(min_proxy, int | float):
        return {
            "filename": best.get("filename"),
            "min_proxy": None,
            "delta_vs_id178_public_anchor": None,
            "delta_vs_public_058": None,
        }
    return {
        "filename": best.get("filename"),
        "min_proxy": float(min_proxy),
        "delta_vs_id178_public_anchor": float(min_proxy) - ID178_PUBLIC_ANCHOR,
        "delta_vs_public_058": float(min_proxy) - PUBLIC_058_TARGET,
    }


def build_report(remote_summary_path: Path, intake_report_path: Path) -> dict[str, Any]:
    summary_status = status_for(remote_summary_path)
    intake_status = status_for(intake_report_path)
    summary = load_json(remote_summary_path) if summary_status.get("parse_ok") else {}
    intake = load_json(intake_report_path) if intake_status.get("parse_ok") else {}
    summary_decision = summary.get("decision", {})
    intake_decision = intake.get("decision", {})
    counts = summary.get("counts", {})
    margin = best_proxy_margin(summary)

    output_pending = summary_decision.get("output_pending") is True or intake_decision.get("output_pending") is True
    parse_or_missing_problem = bool(
        not summary_status.get("parse_ok")
        or not intake_status.get("parse_ok")
        or counts.get("missing_reports", 0)
        or counts.get("parse_errors", 0)
    )
    forbidden_problem = bool(
        counts.get("forbidden_action_rows", 0) or intake_decision.get("remote_forbidden_actions_present")
    )
    intake_allows_review = intake_decision.get("artifact_contract_review_allowed") is True
    summary_reviewable = summary_decision.get("remote_summary_reviewable") is True
    public_058_evidence = summary_decision.get("public_058_evidence_present") is True
    beats_id178_evidence = summary_decision.get("beats_id178_public_anchor_evidence_present") is True
    allow_artifact_contract_review = bool(
        summary_status.get("parse_ok")
        and intake_status.get("parse_ok")
        and not output_pending
        and not parse_or_missing_problem
        and not forbidden_problem
        and intake_allows_review
        and summary_reviewable
        and beats_id178_evidence
    )
    close_remote_lane = bool(
        summary_status.get("parse_ok")
        and intake_status.get("parse_ok")
        and not output_pending
        and not allow_artifact_contract_review
    )
    if output_pending:
        next_action = (
            "Wait for an explicitly approved Kaggle no-submit run and pulled outputs; do not create artifacts."
        )
    elif allow_artifact_contract_review:
        next_action = (
            "Open a separate artifact-contract review. Do not create a CSV until that review names an exact candidate, "
            "base predictions, notebook history, and submission-log row."
        )
    elif close_remote_lane:
        next_action = (
            "Close this remote no-submit lane for artifact creation and move to a materially different raw/sequence source "
            "design; do not promote ID342 from these reports."
        )
    else:
        next_action = "Fix malformed or missing intake/summary evidence before deciding next modeling action."
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_remote_next_action_decision",
        "purpose": (
            "Decide whether pulled no-submit Kaggle evidence should remain pending, close the lane, or open a separate "
            "artifact-contract review. This report itself never creates artifacts or submissions."
        ),
        "inputs": {
            "remote_summary": summary_status,
            "intake_report": intake_status,
        },
        "remote_summary_counts": counts,
        "best_proxy_margin": margin,
        "evidence_flags": {
            "output_pending": output_pending,
            "parse_or_missing_problem": parse_or_missing_problem,
            "forbidden_problem": forbidden_problem,
            "intake_allows_review": intake_allows_review,
            "summary_reviewable": summary_reviewable,
            "public_058_evidence_present": public_058_evidence,
            "beats_id178_public_anchor_evidence_present": beats_id178_evidence,
        },
        "decision": {
            "remote_next_action_ready": bool(summary_status.get("parse_ok") and intake_status.get("parse_ok")),
            "output_pending": output_pending,
            "allow_separate_artifact_contract_review": allow_artifact_contract_review,
            "close_remote_lane_for_artifact_creation": close_remote_lane,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": next_action,
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    margin = report["best_proxy_margin"]
    lines = [
        "# Post-ID341 remote next-action decision",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It decides whether remote no-submit evidence is pending, closed, or eligible for a separate artifact-contract review.",
        "",
        "## Decision",
        "",
        f"- Output pending: `{decision['output_pending']}`",
        f"- Allow separate artifact-contract review: `{decision['allow_separate_artifact_contract_review']}`",
        f"- Close remote lane for artifact creation: `{decision['close_remote_lane_for_artifact_creation']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Evidence flags",
        "",
    ]
    for key, value in report["evidence_flags"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Best proxy margin",
            "",
            f"- Filename: `{margin['filename']}`",
            f"- Min proxy: `{fmt(margin['min_proxy'])}`",
            f"- Delta vs ID178 Public: `{fmt(margin['delta_vs_id178_public_anchor'])}`",
            f"- Delta vs Public 0.58: `{fmt(margin['delta_vs_public_058'])}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-summary", type=Path, default=DEFAULT_REMOTE_SUMMARY)
    parser.add_argument("--intake-report", type=Path, default=DEFAULT_INTAKE_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_remote_next_action_decision_20260618.json",
    )
    args = parser.parse_args()
    remote_summary = args.remote_summary if args.remote_summary.is_absolute() else ROOT / args.remote_summary
    intake_report = args.intake_report if args.intake_report.is_absolute() else ROOT / args.intake_report
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(remote_summary, intake_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
