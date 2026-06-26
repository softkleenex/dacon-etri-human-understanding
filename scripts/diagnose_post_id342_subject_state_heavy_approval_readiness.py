#!/usr/bin/env python3
"""Post-ID342 subject-state heavy approval readiness audit.

This report-only diagnostic consolidates the remaining heavy route contracts
into a single fail-closed readiness check. It does not create runner files,
read raw payload rows, fit embeddings, write features, push/run Kaggle, edit
submission logs, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_subject_state_heavy_approval_readiness_20260624.json"
KST = timezone(timedelta(hours=9))

HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"

INPUT_REPORTS = {
    "goal_state_audit": REPORTS_DIR / "post_id342_goal_state_audit_20260624.json",
    "heavy_compute_contract": REPORTS_DIR / "post_id342_self_supervised_heavy_compute_contract_20260624.json",
    "route_research_packet": REPORTS_DIR / "post_id342_self_supervised_route_research_packet_20260624.json",
    "runner_spec": REPORTS_DIR / "post_id342_self_supervised_heavy_runner_spec_20260624.json",
    "domain_token_contract": REPORTS_DIR / "post_id342_subject_state_domain_token_contract_20260624.json",
    "raw_schema_contract": REPORTS_DIR / "post_id342_subject_state_raw_schema_contract_20260624.json",
    "docs_records_audit": REPORTS_DIR / "docs_records_audit.json",
}

FORBIDDEN_ACTIONS_NOW = {
    "create_runner_files": False,
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_downstream_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "submit_dacon": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "decision": payload.get("decision", {}),
        "findings": payload.get("findings", []),
    }


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    goal = reports["goal_state_audit"].get("decision", {})
    heavy = reports["heavy_compute_contract"].get("decision", {})
    research = reports["route_research_packet"].get("decision", {})
    runner = reports["runner_spec"].get("decision", {})
    domain = reports["domain_token_contract"].get("decision", {})
    schema = reports["raw_schema_contract"].get("decision", {})
    docs_findings = reports["docs_records_audit"].get("findings", [])
    return {
        "all_input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "goal_not_achieved": goal.get("goal_achieved") is False and goal.get("public_055_reached") is False,
        "current_action_gates_closed": goal.get("run_kaggle_now") is False
        and goal.get("dacon_submit_now") is False
        and goal.get("create_submission_artifact_now") is False,
        "heavy_contract_ready": heavy.get("post_id342_heavy_compute_contract_ready") is True,
        "heavy_compute_blocked_now": heavy.get("heavy_compute_allowed_now") is False
        and heavy.get("run_kaggle_now") is False,
        "research_packet_ready": research.get("research_packet_ready") is True,
        "runner_spec_ready": runner.get("runner_spec_ready") is True,
        "runner_execution_blocked_now": runner.get("run_kaggle_now") is False,
        "domain_token_contract_ready": domain.get("domain_token_contract_ready") is True,
        "raw_schema_contract_ready": schema.get("raw_schema_contract_ready") is True
        and schema.get("raw_payload_rows_read") == 0,
        "docs_audit_clean": reports["docs_records_audit"].get("parse_ok") is True
        and isinstance(docs_findings, list)
        and not docs_findings,
    }


def after_approval_sequence() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "action": "refresh gates",
            "command": "make post-id342-subject-state-raw-schema-contract",
            "allowed_without_exact_approval": True,
            "requires_exact_approval": False,
            "purpose": "Refresh report-only prerequisites and confirm payload reads remain zero.",
        },
        {
            "step": 2,
            "action": "create no-submit runner files",
            "command": "create notebooks/kaggle_self_supervised_subject_state_runner/ and runner script",
            "allowed_without_exact_approval": False,
            "requires_exact_approval": True,
            "purpose": "Materialize the dedicated runner defined by the spec.",
        },
        {
            "step": 3,
            "action": "local no-network runner smoke",
            "command": "make post-id342-self-supervised-subject-state-local-smoke",
            "allowed_without_exact_approval": False,
            "requires_exact_approval": True,
            "purpose": "Verify the runner produces reports only and cannot produce submissions.",
        },
        {
            "step": 4,
            "action": "push no-submit Kaggle runner",
            "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-push-self-supervised-subject-state-runner",
            "allowed_without_exact_approval": False,
            "requires_exact_approval": True,
            "purpose": "Run remote heavy pretraining only after fail-closed local smoke.",
        },
        {
            "step": 5,
            "action": "wait and pull reports only",
            "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-self-supervised-subject-state-output",
            "allowed_without_exact_approval": False,
            "requires_exact_approval": True,
            "purpose": "Review embedding/downstream smoke reports before any future artifact decision.",
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in INPUT_REPORTS.items()}
    prerequisites = prerequisite_status(reports)
    prerequisites_ready = all(prerequisites.values())
    approval_observed = args.approval_phrase_observed
    manual_kaggle_status = args.manual_kaggle_status.strip()
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_heavy_approval_readiness",
        "purpose": "Consolidate fail-closed readiness for the remaining post-ID342 heavy route.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "manual_kaggle_status_observation": {
            "status": manual_kaggle_status,
            "observed_complete": "COMPLETE" in manual_kaggle_status.upper(),
            "note": "Status is operator-observed context only; it does not authorize a new run.",
        },
        "after_approval_sequence": after_approval_sequence(),
        "decision": {
            "heavy_approval_readiness_audit_ready": prerequisites_ready,
            "approval_prerequisites_ready": prerequisites_ready,
            "exact_heavy_approval_observed": approval_observed,
            "approval_missing_only_blocker_for_runner_creation": prerequisites_ready and not approval_observed,
            "requires_exact_heavy_compute_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "create_runner_files_now": False,
            "read_raw_parquet_payloads_now": False,
            "write_token_cache_now": False,
            "write_feature_matrix_now": False,
            "fit_embedding_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Exact heavy approval is the only missing runner-creation gate; keep all execution actions closed."
                if prerequisites_ready and not approval_observed
                else "Refresh prerequisite reports or inspect approval state before any implementation."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_runner_files_created": True,
        "no_raw_payload_rows_read": True,
        "no_token_cache_created": True,
        "no_feature_matrix_created": True,
        "no_embedding_fit": True,
        "no_prediction_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 subject-state heavy approval readiness",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report consolidates the remaining heavy route gates. It is report-only and creates no runner, token cache, feature matrix, embedding, prediction, Kaggle run, submission artifact, log row, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Readiness audit ready: `{decision['heavy_approval_readiness_audit_ready']}`",
        f"- Approval prerequisites ready: `{decision['approval_prerequisites_ready']}`",
        f"- Exact heavy approval observed: `{decision['exact_heavy_approval_observed']}`",
        f"- Approval missing only blocker: `{decision['approval_missing_only_blocker_for_runner_creation']}`",
        f"- Create runner files now: `{decision['create_runner_files_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Required phrase: `{decision['heavy_approval_phrase']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Manual Kaggle Status Observation",
        "",
        "```json",
        json.dumps(report["manual_kaggle_status_observation"], indent=2, sort_keys=True),
        "```",
        "",
        "## Prerequisites",
        "",
        "```json",
        json.dumps(report["prerequisites"], indent=2, sort_keys=True),
        "```",
        "",
        "## After-Approval Sequence",
        "",
        "| step | action | command | allowed without exact approval | purpose |",
        "|---:|---|---|---:|---|",
    ]
    for row in report["after_approval_sequence"]:
        lines.append(
            f"| {row['step']} | {row['action']} | `{row['command']}` | "
            f"`{row['allowed_without_exact_approval']}` | {row['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden Actions Now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manual-kaggle-status", default="not_checked_by_script")
    parser.add_argument("--approval-phrase-observed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Readiness audit ready: {report['decision']['heavy_approval_readiness_audit_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
