#!/usr/bin/env python3
"""Build the post-ID341 operator handoff contract.

This local report-only diagnostic consolidates the no-approval frontier, Kaggle
prelaunch gate, launch approval note, execution packet, guarded remote audit,
and local smoke freshness state into one handoff object. It is intended to make
the next operator action unambiguous: without the exact approval phrase, keep the
workspace in report-only mode; with the exact approval phrase, run only the
pre-registered no-submit Kaggle command sequence and then stop at local review
gates.

It creates no predictions, submission CSVs, base prediction artifacts,
notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or DACON
submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_operator_handoff_contract_20260619.json"
APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
PRIMARY_REMOTE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
CONSUMED_MANUAL_PROBE = "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"

REPORT_PATHS = {
    "no_approval_frontier": "post_id341_no_approval_frontier_20260619.json",
    "prelaunch": "post_id341_kaggle_prelaunch_readiness_gate_20260618.json",
    "launch_approval_note": "post_id341_kaggle_launch_approval_note_20260618.json",
    "execution_packet": "kaggle_upstream_execution_packet_post_id341_20260618.json",
    "guarded_remote_contract": "kaggle_guarded_remote_contract_audit_20260618.json",
    "local_smoke_freshness": "post_id341_local_smoke_freshness_gate_20260618.json",
    "local_runner_smoke": "post_id341_kaggle_runner_local_smoke_20260618.json",
    "remote_next_action": "post_id341_remote_next_action_decision_20260618.json",
    "remote_artifact_threshold": "post_id341_remote_artifact_review_threshold_20260618.json",
    "runner_score_relevance": "post_id341_runner_score_relevance_contract_20260618.json",
}

EXPECTED_NETWORK_COMMANDS = [
    "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output",
]
EXPECTED_LOCAL_REVIEW_COMMAND = "make kaggle-review-upstream-rebuild-output"
EXPECTED_LOCAL_PREAPPROVAL_COMMANDS = [
    "make kaggle-approval-contract-test",
    "make kaggle-upstream-prelaunch-local",
]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        return {"exists": False, "parse_ok": False, "decision": {}, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "path": rel(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload.setdefault("path", rel(path))
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(payloads: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return payloads[name].get("decision", {})


def flag(payloads: dict[str, dict[str, Any]], name: str, key: str, expected: Any = True) -> bool:
    return decision(payloads, name).get(key) is expected


def commands(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("command")) for row in rows]


def required_commands(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("command")) for row in rows if row.get("preferred_wrapper") is not True]


def preferred_wrapper_commands(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("command")) for row in rows if row.get("preferred_wrapper") is True]


def submission_log_snapshot() -> dict[str, Any]:
    if not SUBMISSION_LOG.exists():
        return {"exists": False, "rows": 0, "max_id": None, "manual_probe_row_ok": False}
    with SUBMISSION_LOG.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [int(row["id"]) for row in rows if row.get("id")]
    manual_rows = [row for row in rows if row.get("submission_file") == CONSUMED_MANUAL_PROBE]
    manual_probe_row_ok = bool(
        len(manual_rows) == 1
        and str(manual_rows[0].get("is_submitted_bool", "")).lower() == "true"
        and not str(manual_rows[0].get("public_lb_score_num", "")).strip()
    )
    return {
        "exists": True,
        "rows": len(rows),
        "max_id": max(ids) if ids else None,
        "manual_probe_row_count": len(manual_rows),
        "manual_probe_row_ok": manual_probe_row_ok,
    }


def action_flags_false(payloads: dict[str, dict[str, Any]], name: str) -> bool:
    row = decision(payloads, name)
    return all(
        row.get(key) is False
        for key in [
            "create_submission_artifact",
            "submit_now",
            "push_kaggle_now",
            "run_kaggle_now",
        ]
        if key in row
    )


def build_report() -> dict[str, Any]:
    payloads = {key: load_report(name) for key, name in REPORT_PATHS.items()}
    all_reports_ready = all(payload.get("exists") and payload.get("parse_ok") for payload in payloads.values())
    approval_note = payloads["launch_approval_note"]
    execution_packet = payloads["execution_packet"]
    guarded = payloads["guarded_remote_contract"]
    local_preapproval_rows = approval_note.get("local_pre_approval_commands", [])
    local_preapproval_commands = required_commands(local_preapproval_rows)
    preferred_local_ready_wrappers = preferred_wrapper_commands(local_preapproval_rows)
    approval_required_commands = commands(approval_note.get("exact_approval_required_commands", []))
    packet_required_commands = commands(execution_packet.get("approval_required_commands", []))
    expected_full_sequence = EXPECTED_NETWORK_COMMANDS + [EXPECTED_LOCAL_REVIEW_COMMAND]
    sequence_checks = {
        "local_preapproval_matches_expected": local_preapproval_commands == EXPECTED_LOCAL_PREAPPROVAL_COMMANDS,
        "preferred_local_ready_wrapper_present": preferred_local_ready_wrappers == ["make kaggle-upstream-ready-local"],
        "approval_required_matches_expected": approval_required_commands == expected_full_sequence,
        "packet_required_matches_expected": packet_required_commands == expected_full_sequence,
        "approval_and_packet_sequences_match": approval_required_commands == packet_required_commands,
        "network_steps_are_guarded": all(
            command.startswith("ALLOW_KAGGLE_REMOTE=true ") for command in EXPECTED_NETWORK_COMMANDS
        ),
        "local_review_step_is_not_guarded": approval_required_commands[-1:] == [EXPECTED_LOCAL_REVIEW_COMMAND],
        "guarded_audit_sequences_pass": all(guarded.get("sequence_checks", {}).values()),
        "guarded_audit_local_preapproval_pass": all(guarded.get("local_preapproval_checks", {}).values()),
        "guarded_audit_threshold_contract_pass": all(guarded.get("strict_threshold_contract_checks", {}).values()),
        "guarded_audit_handoff_contract_pass": all(guarded.get("approval_packet_handoff_contract_checks", {}).values()),
    }
    state_checks = {
        "no_approval_frontier_ready": flag(payloads, "no_approval_frontier", "no_approval_frontier_ready", True),
        "all_local_no_approval_score_routes_closed": flag(
            payloads, "no_approval_frontier", "all_local_no_approval_score_routes_closed", True
        ),
        "no_approval_score_route_exists_false": flag(
            payloads, "no_approval_frontier", "no_approval_score_route_exists", False
        ),
        "only_score_relevant_route_matches": decision(payloads, "no_approval_frontier").get(
            "only_score_relevant_open_route"
        )
        == PRIMARY_REMOTE_ROUTE,
        "prelaunch_ready": flag(payloads, "prelaunch", "prelaunch_ready_for_explicit_approval", True),
        "prelaunch_requires_approval": flag(payloads, "prelaunch", "requires_explicit_user_approval", True),
        "approval_note_ready": flag(payloads, "launch_approval_note", "approval_note_ready", True),
        "execution_packet_ready": flag(payloads, "execution_packet", "ready_for_explicit_approval", True),
        "guarded_remote_contract_pass": flag(payloads, "guarded_remote_contract", "guarded_remote_contract_pass", True),
        "local_smoke_fresh": flag(payloads, "local_smoke_freshness", "local_runner_smoke_fresh", True),
        "local_smoke_rerun_not_required": flag(
            payloads, "local_smoke_freshness", "requires_local_runner_smoke_rerun", False
        ),
        "local_runner_smoke_passed": flag(payloads, "local_runner_smoke", "local_runner_smoke_passed", True),
        "remote_outputs_pending": flag(payloads, "remote_next_action", "output_pending", True)
        and flag(payloads, "remote_artifact_threshold", "output_pending", True),
        "remote_artifact_review_closed_until_outputs": flag(
            payloads, "remote_artifact_threshold", "allow_public_058_artifact_contract_review", False
        ),
        "runner_score_relevance_pass": flag(
            payloads, "runner_score_relevance", "runner_score_relevance_contract_pass", True
        ),
    }
    no_action_checks = {
        name: action_flags_false(payloads, name)
        for name in [
            "no_approval_frontier",
            "prelaunch",
            "launch_approval_note",
            "execution_packet",
            "guarded_remote_contract",
            "local_smoke_freshness",
            "local_runner_smoke",
            "remote_next_action",
            "remote_artifact_threshold",
            "runner_score_relevance",
        ]
    }
    log_snapshot = submission_log_snapshot()
    artifact_boundary_checks = {
        "submission_log_exists": bool(log_snapshot.get("exists")),
        "manual_probe_row_ok": bool(log_snapshot.get("manual_probe_row_ok")),
        "no_report_allows_actions": all(no_action_checks.values()),
        "manual_probe_matches_frontier": decision(payloads, "no_approval_frontier").get(
            "manual_probe_if_user_spends_submission"
        )
        is None,
        "manual_probe_is_not_public_058_evidence": flag(
            payloads, "no_approval_frontier", "manual_probe_is_public_058_evidence", False
        ),
    }
    handoff_ready = bool(
        all_reports_ready
        and all(sequence_checks.values())
        and all(state_checks.values())
        and all(artifact_boundary_checks.values())
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_operator_handoff_contract",
        "purpose": (
            "Consolidate the no-approval state and exact approved Kaggle no-submit sequence into one operator "
            "handoff contract."
        ),
        "inputs": {key: payloads[key].get("path", rel(REPORTS_DIR / name)) for key, name in REPORT_PATHS.items()},
        "report_status": {
            key: {"exists": bool(payload.get("exists")), "parse_ok": bool(payload.get("parse_ok"))}
            for key, payload in payloads.items()
        },
        "approval_phrase": APPROVAL_PHRASE,
        "local_preapproval_commands": local_preapproval_commands,
        "preferred_local_ready_wrappers": preferred_local_ready_wrappers,
        "approval_required_commands": approval_required_commands,
        "sequence_checks": sequence_checks,
        "state_checks": state_checks,
        "artifact_boundary_checks": artifact_boundary_checks,
        "submission_log_snapshot": log_snapshot,
        "manual_probe_if_user_spends_submission": None,
        "primary_score_relevant_route": PRIMARY_REMOTE_ROUTE,
        "decision": {
            "operator_handoff_ready": handoff_ready,
            "safe_wait_state_without_user_action": handoff_ready,
            "requires_exact_approval_phrase": True,
            "approval_phrase": APPROVAL_PHRASE,
            "local_preapproval_required_before_network": True,
            "score_relevant_route": PRIMARY_REMOTE_ROUTE,
            "manual_probe_if_user_spends_submission": None,
            "manual_probe_is_public_058_evidence": False,
            "remote_outputs_pending": state_checks["remote_outputs_pending"],
            "artifact_review_open": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Wait for the exact approval phrase before any Kaggle network command. If approved, run local "
                "preapproval checks, then only the registered no-submit command sequence, and stop at local output "
                "review gates."
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
    decision_row = report["decision"]
    lines = [
        "# Post-ID341 operator handoff contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Operator handoff ready: `{decision_row['operator_handoff_ready']}`",
        f"- Safe wait state without user action: `{decision_row['safe_wait_state_without_user_action']}`",
        f"- Requires exact approval phrase: `{decision_row['requires_exact_approval_phrase']}`",
        f"- Approval phrase: `{decision_row['approval_phrase']}`",
        f"- Score-relevant route: `{decision_row['score_relevant_route']}`",
        f"- Manual probe if user spends submission: `{decision_row['manual_probe_if_user_spends_submission']}`",
        f"- Manual probe is Public 0.58 evidence: `{decision_row['manual_probe_is_public_058_evidence']}`",
        f"- Artifact review open: `{decision_row['artifact_review_open']}`",
        f"- Create submission artifact: `{decision_row['create_submission_artifact']}`",
        f"- Submit now: `{decision_row['submit_now']}`",
        f"- Push Kaggle now: `{decision_row['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_row['run_kaggle_now']}`",
        f"- Next action: {decision_row['next_action']}",
        "",
        "## Local preapproval commands",
        "",
    ]
    for command in report["local_preapproval_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Approval-required command sequence", ""])
    for command in report["approval_required_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Checks", "", "### Sequence checks", ""])
    for key, value in report["sequence_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "### State checks", ""])
    for key, value in report["state_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "### Artifact boundary checks", ""])
    for key, value in report["artifact_boundary_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0 if report["decision"]["operator_handoff_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
