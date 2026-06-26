#!/usr/bin/env python3
"""Audit the guarded Kaggle remote command contract.

This local report-only audit checks that:
- Kaggle network Make targets fail closed behind `ALLOW_KAGGLE_REMOTE=true`.
- The approval note, execution packet, and prelaunch gate share the same guarded
  no-submit command sequence.
- Local post-pull review remains non-network and no-submit.

It never runs Kaggle network commands, creates predictions, creates submission
artifacts, or submits to DACON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
MAKEFILE = ROOT / "Makefile"
APPROVAL_NOTE = REPORTS_DIR / "post_id341_kaggle_launch_approval_note_20260618.json"
EXECUTION_PACKET = REPORTS_DIR / "kaggle_upstream_execution_packet_post_id341_20260618.json"
PRELAUNCH_GATE = REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json"
PRELAUNCH_SCRIPT = ROOT / "scripts" / "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py"

EXPECTED_SEQUENCE = [
    "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output",
    "make kaggle-review-upstream-rebuild-output",
]
NETWORK_COMMAND_COUNT = 6
GUARDED_TARGETS = [
    "kaggle-push-code",
    "kaggle-push-upstream-patch-dataset",
    "kaggle-push-only",
    "kaggle-check-base-dataset-files",
    "kaggle-check-upstream-patch-dataset-files",
    "kaggle-kernel-status-upstream-rebuild",
    "kaggle-pull-output",
    "kaggle-wait-and-pull-upstream-rebuild-output",
    "kaggle-wait-and-pull",
]
LOCAL_ONLY_TARGETS = [
    "kaggle-approval-contract-test",
    "kaggle-upstream-prelaunch-local",
    "kaggle-upstream-ready-local",
    "kaggle-review-upstream-rebuild-output",
]
EXPECTED_LOCAL_PRE_APPROVAL_SEQUENCE = [
    "make kaggle-approval-contract-test",
    "make kaggle-upstream-prelaunch-local",
]
LOCAL_REVIEW_REQUIRED_SCRIPTS = [
    "scripts/diagnose_post_id341_kaggle_output_intake_gate.py",
    "scripts/summarize_post_id341_kaggle_remote_results.py",
    "scripts/decide_post_id341_remote_next_action.py",
    "scripts/decide_post_id341_remote_artifact_review_threshold.py",
]
LOCAL_PRELAUNCH_REQUIRED_SCRIPTS = [
    "scripts/diagnose_post_id341_kaggle_runner_intake_contract.py",
    "scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py",
    "scripts/diagnose_post_id341_local_source_closure_synthesis.py",
    "scripts/diagnose_kaggle_source_zip_selftest_post_id341.py",
    "scripts/diagnose_post_id341_kaggle_launch_approval_note.py",
    "scripts/diagnose_post_id341_kaggle_output_intake_gate.py",
    "scripts/summarize_post_id341_kaggle_remote_results.py",
    "scripts/decide_post_id341_remote_next_action.py",
    "scripts/decide_post_id341_remote_artifact_review_threshold.py",
    "scripts/diagnose_post_id341_post_pull_review_chain_contract.py",
    "scripts/diagnose_post_id341_public_local_gap.py",
    "scripts/diagnose_post_id341_data_source_coverage.py",
    "scripts/audit_kaggle_guarded_remote_contract.py",
    "scripts/audit_kaggle_remote_fail_closed_dryrun.py",
    "scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
]
LOCAL_PRELAUNCH_Q3S3_CHAIN_SCRIPTS = [
    "diagnose_post_id341_q3s3_negative_evidence_map.py",
    "diagnose_post_id341_q3s3_remaining_route_inventory.py",
    "diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py",
]
LOCAL_PRELAUNCH_APPROVAL_ARTIFACT_SCRIPTS = [
    "diagnose_post_id341_kaggle_runner_intake_contract.py",
    "diagnose_kaggle_upstream_execution_packet_post_id341.py",
    "diagnose_post_id341_local_source_closure_synthesis.py",
    "diagnose_kaggle_source_zip_selftest_post_id341.py",
    "diagnose_post_id341_kaggle_launch_approval_note.py",
    "audit_kaggle_guarded_remote_contract.py",
    "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
]
PRELAUNCH_SCRIPT_REQUIRED_HANDOFF_TOKENS = [
    "q3s3_negative_map_handoff_contract",
    "q3s3_negative_map_handoff_pass",
    "remaining_inventory_uses_negative_map_arg",
    "negative_map_asserts_eight_closed_lanes",
    "runner_q3s3_negative_map_handoff_ready",
    "runner_q3s3_negative_map_handoff_contract",
    "and runner_q3s3_negative_map_handoff_ready",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_body(makefile_text: str, target: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(target)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t.*\n)*)", re.MULTILINE)
    match = pattern.search(makefile_text)
    return match.group(0) if match else None


def command_sequence_from_report(path: Path, key: str) -> list[str]:
    payload = load_json(path)
    value = payload.get(key, [])
    if key == "exact_command_sequence_after_explicit_approval":
        return list(value)
    return [row["command"] for row in value]


def command_sequence_from_script_constant(path: Path, constant: str) -> list[str]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, constant, []))


def ordered_before(body: str, before_scripts: list[str], after_scripts: list[str]) -> bool:
    before_positions = [body.find(script) for script in before_scripts]
    after_positions = [body.find(script) for script in after_scripts]
    return bool(
        before_positions
        and after_positions
        and all(position >= 0 for position in before_positions + after_positions)
        and before_positions == sorted(before_positions)
        and max(before_positions) < min(after_positions)
    )


def build_report() -> dict[str, Any]:
    makefile_text = MAKEFILE.read_text(encoding="utf-8")
    prelaunch_script_text = PRELAUNCH_SCRIPT.read_text(encoding="utf-8")
    local_review_body = target_body(makefile_text, "kaggle-review-upstream-rebuild-output") or ""
    local_prelaunch_body = target_body(makefile_text, "kaggle-upstream-prelaunch-local") or ""
    approval_payload = load_json(APPROVAL_NOTE)
    packet_payload = load_json(EXECUTION_PACKET)
    makefile_checks = {
        "allow_variable_defaults_false": "ALLOW_KAGGLE_REMOTE ?= false" in makefile_text,
        "require_target_present": "require-kaggle-remote-approval:" in makefile_text,
        "guarded_targets": {
            target: bool((body := target_body(makefile_text, target)) and "require-kaggle-remote-approval" in body)
            for target in GUARDED_TARGETS
        },
        "local_only_targets": {
            target: bool((body := target_body(makefile_text, target)) and "require-kaggle-remote-approval" not in body)
            for target in LOCAL_ONLY_TARGETS
        },
    }
    approval_sequence = [row["command"] for row in approval_payload.get("exact_approval_required_commands", [])]
    local_preapproval_rows = approval_payload.get("local_pre_approval_commands", [])
    local_preapproval_sequence = [
        row["command"] for row in local_preapproval_rows if row.get("preferred_wrapper") is not True
    ]
    preferred_local_ready_wrappers = [
        row["command"] for row in local_preapproval_rows if row.get("preferred_wrapper") is True
    ]
    packet_sequence = [row["command"] for row in packet_payload.get("approval_required_commands", [])]
    prelaunch_script_sequence = command_sequence_from_script_constant(PRELAUNCH_SCRIPT, "EXACT_COMMAND_SEQUENCE")
    prelaunch_script_local_sequence = command_sequence_from_script_constant(
        PRELAUNCH_SCRIPT, "LOCAL_PRE_APPROVAL_COMMAND_SEQUENCE"
    )
    prelaunch_report_sequence = (
        command_sequence_from_report(PRELAUNCH_GATE, "exact_command_sequence_after_explicit_approval")
        if PRELAUNCH_GATE.exists()
        else []
    )
    sequence_checks = {
        "approval_matches_expected": approval_sequence == EXPECTED_SEQUENCE,
        "packet_matches_expected": packet_sequence == EXPECTED_SEQUENCE,
        "prelaunch_script_matches_expected": prelaunch_script_sequence == EXPECTED_SEQUENCE,
        "all_authoritative_sequences_match_each_other": approval_sequence
        == packet_sequence
        == prelaunch_script_sequence,
        "network_commands_are_guarded": all(
            command.startswith("ALLOW_KAGGLE_REMOTE=true make ")
            for command in EXPECTED_SEQUENCE[:NETWORK_COMMAND_COUNT]
        ),
        "local_review_command_is_not_guarded": EXPECTED_SEQUENCE[-1] == "make kaggle-review-upstream-rebuild-output",
    }
    local_preapproval_checks = {
        "approval_local_preapproval_sequence_matches_expected": local_preapproval_sequence
        == EXPECTED_LOCAL_PRE_APPROVAL_SEQUENCE,
        "approval_preferred_local_ready_wrapper_present": preferred_local_ready_wrappers
        == ["make kaggle-upstream-ready-local"],
        "prelaunch_local_preapproval_sequence_matches_expected": prelaunch_script_local_sequence
        == EXPECTED_LOCAL_PRE_APPROVAL_SEQUENCE,
        "approval_stop_rule_requires_local_preapproval_commands": any(
            "kaggle-approval-contract-test" in rule and "kaggle-upstream-prelaunch-local" in rule
            for rule in approval_payload.get("stop_rules", [])
        ),
        "approval_stop_rule_prefers_ready_wrapper": any(
            "kaggle-upstream-ready-local" in rule for rule in approval_payload.get("stop_rules", [])
        ),
    }
    local_review_checks = {
        "required_scripts_present": {script: script in local_review_body for script in LOCAL_REVIEW_REQUIRED_SCRIPTS},
        "threshold_gate_is_last_review_step": bool(
            local_review_body.rfind("scripts/decide_post_id341_remote_artifact_review_threshold.py")
            > local_review_body.rfind("scripts/decide_post_id341_remote_next_action.py")
        ),
        "local_review_has_no_remote_guard": "ALLOW_KAGGLE_REMOTE=true" not in local_review_body,
        "local_review_has_no_dacon_submit": "dacon-submit" not in local_review_body,
    }
    local_prelaunch_checks = {
        "required_scripts_present": {
            script: script in local_prelaunch_body for script in LOCAL_PRELAUNCH_REQUIRED_SCRIPTS
        },
        "q3s3_negative_chain_before_approval_artifacts": ordered_before(
            local_prelaunch_body,
            LOCAL_PRELAUNCH_Q3S3_CHAIN_SCRIPTS,
            LOCAL_PRELAUNCH_APPROVAL_ARTIFACT_SCRIPTS,
        ),
        "prelaunch_has_no_remote_guard": "ALLOW_KAGGLE_REMOTE=true" not in local_prelaunch_body,
        "prelaunch_has_no_dacon_submit": "dacon-submit" not in local_prelaunch_body,
    }
    strict_threshold_contract_checks = {
        "approval_step6_mentions_threshold_gate": "Public-0.58"
        in approval_payload["exact_approval_required_commands"][-1]["why"]
        and "threshold gate" in approval_payload["exact_approval_required_commands"][-1]["why"],
        "packet_step6_mentions_threshold_gate": "Public-0.58" in packet_payload["approval_required_commands"][-1]["why"]
        and "threshold gate" in packet_payload["approval_required_commands"][-1]["why"],
        "approval_stop_rules_require_min_proxy_058": any(
            "min_proxy <= 0.58" in rule and "ID178" in rule for rule in approval_payload.get("stop_rules", [])
        ),
        "packet_stop_rules_require_min_proxy_058": any(
            "min_proxy <= 0.58" in rule and "ID178" in rule for rule in packet_payload.get("stop_rules", [])
        ),
        "approval_stop_rules_block_id178_only_id342": any(
            "only beat ID178" in rule and "do not create ID342" in rule
            for rule in approval_payload.get("stop_rules", [])
        ),
        "packet_stop_rules_block_id178_only_id342": any(
            "only beat ID178" in rule and "do not create ID342" in rule for rule in packet_payload.get("stop_rules", [])
        ),
    }
    approval_packet_handoff_contract_checks = {
        "approval_stop_rules_require_q3s3_negative_map_handoff": any(
            "Q3/S3 negative-map handoff" in rule
            and "--negative-evidence-map" in rule
            and "eight closed local lanes" in rule
            for rule in approval_payload.get("stop_rules", [])
        ),
        "packet_stop_rules_require_q3s3_negative_map_handoff": any(
            "Q3/S3 negative-map handoff" in rule
            and "--negative-evidence-map" in rule
            and "eight closed local lanes" in rule
            for rule in packet_payload.get("stop_rules", [])
        ),
    }
    prelaunch_script_handoff_contract_checks = {
        "required_handoff_tokens_present": {
            token: token in prelaunch_script_text for token in PRELAUNCH_SCRIPT_REQUIRED_HANDOFF_TOKENS
        },
        "handoff_ready_check_exposed": '"runner_q3s3_negative_map_handoff_ready":' in prelaunch_script_text,
        "handoff_contract_input_exposed": '"runner_q3s3_negative_map_handoff_contract":' in prelaunch_script_text,
        "handoff_check_part_of_prelaunch_ready": "and runner_q3s3_negative_map_handoff_ready" in prelaunch_script_text,
    }
    all_makefile_checks = bool(
        makefile_checks["allow_variable_defaults_false"]
        and makefile_checks["require_target_present"]
        and all(makefile_checks["guarded_targets"].values())
        and all(makefile_checks["local_only_targets"].values())
    )
    all_sequence_checks = all(sequence_checks.values())
    all_local_preapproval_checks = all(local_preapproval_checks.values())
    all_local_review_checks = bool(
        all(local_review_checks["required_scripts_present"].values())
        and local_review_checks["threshold_gate_is_last_review_step"]
        and local_review_checks["local_review_has_no_remote_guard"]
        and local_review_checks["local_review_has_no_dacon_submit"]
    )
    all_local_prelaunch_checks = bool(
        all(local_prelaunch_checks["required_scripts_present"].values())
        and local_prelaunch_checks["q3s3_negative_chain_before_approval_artifacts"]
        and local_prelaunch_checks["prelaunch_has_no_remote_guard"]
        and local_prelaunch_checks["prelaunch_has_no_dacon_submit"]
    )
    all_strict_threshold_contract_checks = all(strict_threshold_contract_checks.values())
    all_approval_packet_handoff_contract_checks = all(approval_packet_handoff_contract_checks.values())
    all_prelaunch_script_handoff_contract_checks = bool(
        all(prelaunch_script_handoff_contract_checks["required_handoff_tokens_present"].values())
        and prelaunch_script_handoff_contract_checks["handoff_ready_check_exposed"]
        and prelaunch_script_handoff_contract_checks["handoff_contract_input_exposed"]
        and prelaunch_script_handoff_contract_checks["handoff_check_part_of_prelaunch_ready"]
    )
    audit_pass = bool(
        all_makefile_checks
        and all_sequence_checks
        and all_local_preapproval_checks
        and all_local_review_checks
        and all_local_prelaunch_checks
        and all_strict_threshold_contract_checks
        and all_approval_packet_handoff_contract_checks
        and all_prelaunch_script_handoff_contract_checks
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "kaggle_guarded_remote_contract_audit",
        "purpose": (
            "Ensure Kaggle remote commands fail closed and approval/packet/prelaunch-script sources use one "
            "guarded sequence without depending on a stale prelaunch report."
        ),
        "inputs": {
            "makefile": safe_relative(MAKEFILE),
            "approval_note": safe_relative(APPROVAL_NOTE),
            "execution_packet": safe_relative(EXECUTION_PACKET),
            "prelaunch_script": safe_relative(PRELAUNCH_SCRIPT),
            "prelaunch_gate_report_observed_not_authoritative": safe_relative(PRELAUNCH_GATE),
        },
        "expected_sequence": EXPECTED_SEQUENCE,
        "observed_sequences": {
            "approval_note_local_preapproval": local_preapproval_sequence,
            "approval_note": approval_sequence,
            "execution_packet": packet_sequence,
            "prelaunch_script_local_preapproval": prelaunch_script_local_sequence,
            "prelaunch_script": prelaunch_script_sequence,
            "prelaunch_gate_report_observed_not_authoritative": prelaunch_report_sequence,
        },
        "makefile_checks": makefile_checks,
        "sequence_checks": sequence_checks,
        "local_preapproval_checks": local_preapproval_checks,
        "local_review_checks": local_review_checks,
        "local_prelaunch_checks": local_prelaunch_checks,
        "strict_threshold_contract_checks": strict_threshold_contract_checks,
        "approval_packet_handoff_contract_checks": approval_packet_handoff_contract_checks,
        "prelaunch_script_handoff_contract_checks": prelaunch_script_handoff_contract_checks,
        "decision": {
            "guarded_remote_contract_pass": audit_pass,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Guarded remote contract is consistent; explicit approval is still required for any Kaggle network step."
                if audit_pass
                else "Fix guarded remote command drift before consuming Kaggle approval."
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
        "# Kaggle guarded remote contract audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-submit. It checks the Makefile guard plus approval/packet/prelaunch-script command sequence.",
        "",
        "## Decision",
        "",
        f"- Guarded remote contract pass: `{decision['guarded_remote_contract_pass']}`",
        f"- Execute Kaggle now: `{decision['execute_kaggle_now']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Expected guarded sequence",
        "",
    ]
    for index, command in enumerate(report["expected_sequence"], start=1):
        lines.append(f"{index}. `{command}`")
    lines.extend(["", "## Makefile checks", ""])
    for key, value in report["makefile_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Sequence checks", ""])
    for key, value in report["sequence_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Local pre-approval checks", ""])
    for key, value in report["local_preapproval_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Local review checks", ""])
    for key, value in report["local_review_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Local prelaunch checks", ""])
    for key, value in report["local_prelaunch_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Strict threshold contract checks", ""])
    for key, value in report["strict_threshold_contract_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Approval/packet handoff contract checks", ""])
    for key, value in report["approval_packet_handoff_contract_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Prelaunch script handoff contract checks", ""])
    for key, value in report["prelaunch_script_handoff_contract_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "kaggle_guarded_remote_contract_audit_20260618.json",
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
