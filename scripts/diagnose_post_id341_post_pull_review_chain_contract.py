#!/usr/bin/env python3
"""Audit the local post-pull review chain for the post-ID341 Kaggle path.

This report-only diagnostic checks the four local post-pull review reports:

1. output intake,
2. remote result summary,
3. remote next-action decision, and
4. Public-0.58 artifact-review threshold.

Before any approved Kaggle run has produced outputs, the safe state is that the
chain is parseable, consistently output-pending, and blocks every artifact,
Kaggle, and DACON action. This script never creates predictions, submission
CSVs, base prediction artifacts, notebook-history artifacts, submission-log
rows, Kaggle pushes/runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MAKEFILE = ROOT / "Makefile"
KST = timezone(timedelta(hours=9))

OUTPUT_INTAKE = REPORTS_DIR / "post_id341_kaggle_output_intake_gate_20260618.json"
REMOTE_SUMMARY = REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json"
REMOTE_NEXT_ACTION = REPORTS_DIR / "post_id341_remote_next_action_decision_20260618.json"
REMOTE_THRESHOLD = REPORTS_DIR / "post_id341_remote_artifact_review_threshold_20260618.json"

EXPECTED_TARGET_ORDER = [
    "scripts/diagnose_post_id341_kaggle_output_intake_gate.py",
    "scripts/summarize_post_id341_kaggle_remote_results.py",
    "scripts/decide_post_id341_remote_next_action.py",
    "scripts/decide_post_id341_remote_artifact_review_threshold.py",
]
FORBIDDEN_TRUE_KEYS = [
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": safe_relative(path),
            "exists": False,
            "parse_ok": False,
            "decision": {},
            "error": None,
        }
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "error": None,
    }


def make_target_body(target: str) -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(target)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t.*\n)*)", re.MULTILINE)
    match = pattern.search(text)
    return match.group(0) if match else ""


def ordered_target_scripts_present(body: str) -> dict[str, Any]:
    positions = {script: body.find(script) for script in EXPECTED_TARGET_ORDER}
    present = {script: position >= 0 for script, position in positions.items()}
    ordered = all(present.values()) and [positions[script] for script in EXPECTED_TARGET_ORDER] == sorted(
        positions[script] for script in EXPECTED_TARGET_ORDER
    )
    return {
        "target": "kaggle-review-upstream-rebuild-output",
        "required_scripts_present": present,
        "required_scripts_in_order": ordered,
        "target_has_no_remote_guard": "ALLOW_KAGGLE_REMOTE=true" not in body
        and "require-kaggle-remote-approval" not in body,
        "target_has_no_dacon_submit": "dacon-submit" not in body,
    }


def forbidden_true(decision: dict[str, Any]) -> dict[str, bool]:
    return {key: decision.get(key) is True for key in FORBIDDEN_TRUE_KEYS}


def build_report() -> dict[str, Any]:
    output_intake = report_status(OUTPUT_INTAKE)
    remote_summary = report_status(REMOTE_SUMMARY)
    next_action = report_status(REMOTE_NEXT_ACTION)
    threshold = report_status(REMOTE_THRESHOLD)
    reports = {
        "output_intake": output_intake,
        "remote_summary": remote_summary,
        "remote_next_action": next_action,
        "remote_artifact_review_threshold": threshold,
    }
    target_contract = ordered_target_scripts_present(make_target_body("kaggle-review-upstream-rebuild-output"))
    all_reports_parse = all(row["parse_ok"] for row in reports.values())
    output_pending_consistent = bool(
        output_intake["decision"].get("output_pending") is True
        and remote_summary["decision"].get("output_pending") is True
        and next_action["decision"].get("output_pending") is True
        and threshold["decision"].get("output_pending") is True
    )
    review_blocked_consistent = bool(
        output_intake["decision"].get("artifact_contract_review_allowed") is False
        and remote_summary["decision"].get("remote_summary_reviewable") is False
        and remote_summary["decision"].get("public_058_evidence_present") is False
        and remote_summary["decision"].get("beats_id178_public_anchor_evidence_present") is False
        and next_action["decision"].get("allow_separate_artifact_contract_review") is False
        and threshold["decision"].get("allow_public_058_artifact_contract_review") is False
        and threshold["decision"].get("record_id178_improvement_lead_only") is False
    )
    forbidden_by_report = {name: forbidden_true(row["decision"]) for name, row in reports.items()}
    no_forbidden_true = all(not any(values.values()) for values in forbidden_by_report.values())
    target_contract_pass = bool(
        all(target_contract["required_scripts_present"].values())
        and target_contract["required_scripts_in_order"]
        and target_contract["target_has_no_remote_guard"]
        and target_contract["target_has_no_dacon_submit"]
    )
    pass_gate = bool(
        all_reports_parse
        and output_pending_consistent
        and review_blocked_consistent
        and no_forbidden_true
        and target_contract_pass
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_post_pull_review_chain_contract",
        "purpose": (
            "Verify the local post-pull review chain is parseable, ordered, local-only, and currently "
            "pending-safe before explicit-approved Kaggle outputs exist."
        ),
        "inputs": reports,
        "target_contract": target_contract,
        "checks": {
            "all_reports_parse": all_reports_parse,
            "output_pending_consistent": output_pending_consistent,
            "review_blocked_consistent": review_blocked_consistent,
            "no_forbidden_true": no_forbidden_true,
            "target_contract_pass": target_contract_pass,
        },
        "forbidden_true_by_report": forbidden_by_report,
        "decision": {
            "post_pull_review_chain_contract_pass": pass_gate,
            "output_pending": output_pending_consistent,
            "artifact_contract_review_allowed": False,
            "public_058_reviewable": False,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Post-pull review chain is pending-safe; explicit approval is still required before any Kaggle "
                "network step."
                if pass_gate
                else "Fix post-pull review-chain drift before consuming Kaggle approval."
            ),
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


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 post-pull review-chain contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-submit. It checks that the output-intake, remote-summary, next-action, and Public-0.58 threshold reports are currently pending-safe.",
        "",
        "## Decision",
        "",
        f"- Post-pull review-chain contract pass: `{decision['post_pull_review_chain_contract_pass']}`",
        f"- Output pending: `{decision['output_pending']}`",
        f"- Artifact-contract review allowed: `{decision['artifact_contract_review_allowed']}`",
        f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Target contract", ""])
    for key, value in report["target_contract"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_post_pull_review_chain_contract_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
