#!/usr/bin/env python3
"""Post-ID342 morning closeout decision report.

This report-only diagnostic consolidates the current autonomous-work window
state: public score gap, Kaggle health, approval gates, and next action. It
does not create runner files, train models, push/run Kaggle, create artifacts,
edit submission logs, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_morning_closeout_20260624.json"
KST = timezone(timedelta(hours=9))

INPUT_REPORTS = {
    "goal_state_audit": REPORTS_DIR / "post_id342_goal_state_audit_20260624.json",
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json",
    "heavy_approval_readiness": REPORTS_DIR / "post_id342_subject_state_heavy_approval_readiness_20260624.json",
    "heavy_success_thresholds": REPORTS_DIR / "post_id342_subject_state_heavy_success_thresholds_20260624.json",
    "kaggle_input_availability": REPORTS_DIR / "post_id342_kaggle_input_availability_20260624.json",
    "kaggle_cli_health": REPORTS_DIR / "post_id342_kaggle_cli_health_20260624.json",
    "docs_records_audit": REPORTS_DIR / "docs_records_audit.json",
}

FORBIDDEN_ACTIONS_NOW = {
    "create_runner_files": False,
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "pull_kaggle_output": False,
    "submit_dacon": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": rel(path),
            "exists": False,
            "parse_ok": False,
            "decision": {},
            "findings": [],
            "error": "missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "findings": [],
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
    synthesis = reports["remaining_route_synthesis"].get("decision", {})
    readiness = reports["heavy_approval_readiness"].get("decision", {})
    thresholds = reports["heavy_success_thresholds"].get("decision", {})
    input_availability = reports["kaggle_input_availability"].get("decision", {})
    cli = reports["kaggle_cli_health"].get("decision", {})
    docs_findings = reports["docs_records_audit"].get("findings", [])
    return {
        "all_input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "goal_not_achieved": goal.get("goal_achieved") is False and goal.get("public_055_reached") is False,
        "id342_anchor_current": synthesis.get("current_public_anchor_id") == 342,
        "no_dacon_submission_meaningful_now": synthesis.get("dacon_submit_meaningful_now") is False,
        "no_kaggle_wait_state": synthesis.get("kaggle_upstream_complete_no_wait_state") is True,
        "approval_prerequisites_ready": readiness.get("approval_prerequisites_ready") is True,
        "exact_heavy_approval_missing": readiness.get("exact_heavy_approval_observed") is False,
        "success_thresholds_ready": thresholds.get("heavy_success_threshold_contract_ready") is True,
        "kaggle_inputs_ready": input_availability.get("kaggle_input_availability_ready") is True,
        "kaggle_cli_usable": cli.get("kaggle_cli_health_ready") is True,
        "docs_audit_clean": reports["docs_records_audit"].get("parse_ok") is True
        and isinstance(docs_findings, list)
        and not docs_findings,
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in INPUT_REPORTS.items()}
    prerequisites = prerequisite_status(reports)
    synthesis = reports["remaining_route_synthesis"].get("decision", {})
    readiness = reports["heavy_approval_readiness"].get("decision", {})
    thresholds = reports["heavy_success_thresholds"].get("decision", {})
    cli = reports["kaggle_cli_health"].get("decision", {})
    closeout_ready = all(prerequisites.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_morning_closeout",
        "purpose": "Consolidate the morning operating state after post-ID342 no-approval work.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "score_state": {
            "best_public_anchor_id": synthesis.get("current_public_anchor_id"),
            "best_public_score": synthesis.get("current_public_score"),
            "target_public_score": synthesis.get("target_public_score"),
            "gap_to_public_055": synthesis.get("gap_to_public_055"),
            "closed_routes": synthesis.get("closed_route_count"),
            "route_count": synthesis.get("route_count"),
            "deferred_routes": synthesis.get("deferred_route_count"),
        },
        "kaggle_state": {
            "cli_health_ready": cli.get("kaggle_cli_health_ready"),
            "installed_kaggle_version": cli.get("installed_kaggle_version"),
            "outdated_warning_seen": cli.get("outdated_warning_seen"),
            "dataset_listing_ok": cli.get("dataset_listing_ok"),
            "upstream_kernel_complete": cli.get("upstream_kernel_complete"),
            "auth_network_slug_blocker_observed": cli.get("auth_network_slug_blocker_observed"),
        },
        "approval_state": {
            "approval_prerequisites_ready": readiness.get("approval_prerequisites_ready"),
            "exact_heavy_approval_observed": readiness.get("exact_heavy_approval_observed"),
            "heavy_approval_phrase": readiness.get("heavy_approval_phrase"),
            "approval_missing_only_blocker_for_runner_creation": readiness.get(
                "approval_missing_only_blocker_for_runner_creation"
            ),
        },
        "future_review_thresholds": {
            "artifact_review_public_proxy": thresholds.get("artifact_review_threshold_public_proxy"),
            "dacon_candidate_public_proxy": thresholds.get("dacon_candidate_threshold_public_proxy"),
        },
        "decision": {
            "morning_closeout_ready": closeout_ready,
            "goal_achieved": False,
            "public_055_reached": False,
            "dacon_submit_now": False,
            "dacon_submit_meaningful_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_runner_files_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "kaggle_waiting_now": False,
            "kaggle_cli_blocker_now": False,
            "next_action": (
                "No DACON/Kaggle action now. The only score-relevant execution path is the exact-approved "
                "self-supervised subject-state no-submit heavy route."
                if closeout_ready
                else "Refresh missing prerequisite reports before making an operating decision."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_runner_files_created": True,
        "no_model_trained": True,
        "no_prediction_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_kaggle_output_pull_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 morning closeout",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report consolidates the operating state. It performs no model training, Kaggle push/run/pull, artifact creation, submission-log edit, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Morning closeout ready: `{decision['morning_closeout_ready']}`",
        f"- Goal achieved: `{decision['goal_achieved']}`",
        f"- Public 0.55 reached: `{decision['public_055_reached']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- DACON submit meaningful now: `{decision['dacon_submit_meaningful_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Kaggle waiting now: `{decision['kaggle_waiting_now']}`",
        f"- Kaggle CLI blocker now: `{decision['kaggle_cli_blocker_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Score State",
        "",
        "```json",
        json.dumps(report["score_state"], indent=2, sort_keys=True),
        "```",
        "",
        "## Kaggle State",
        "",
        "```json",
        json.dumps(report["kaggle_state"], indent=2, sort_keys=True),
        "```",
        "",
        "## Approval State",
        "",
        "```json",
        json.dumps(report["approval_state"], indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Future Review Thresholds",
        "",
        "```json",
        json.dumps(report["future_review_thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "## Prerequisites",
        "",
        "```json",
        json.dumps(report["prerequisites"], indent=2, sort_keys=True),
        "```",
        "",
        "## Forbidden Actions Now",
        "",
        "```json",
        json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Morning closeout ready: {report['decision']['morning_closeout_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
