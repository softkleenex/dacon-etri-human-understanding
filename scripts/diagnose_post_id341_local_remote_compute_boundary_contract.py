#!/usr/bin/env python3
"""Validate the local-vs-Kaggle compute boundary for post-ID341.

This report-only contract prevents the local prelaunch path from regressing
back into executing the full upstream rebuild notebook on the Mac. Local work is
limited to fast packaging/contract-staging checks; the long upstream runner is a
guarded Kaggle no-submit action that still requires exact user approval.

It creates no predictions, submission CSVs, base prediction artifacts,
submission-log rows, Kaggle runs, or DACON submissions.
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
KST = timezone(timedelta(hours=9))

MAKEFILE = ROOT / "Makefile"
LOCAL_SMOKE_SCRIPT = ROOT / "scripts" / "run_post_id341_kaggle_runner_local_smoke.py"
RUNNER_NOTEBOOK = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "Upstream_Rebuild_Post_ID341_Runner.ipynb"
RUNNER_METADATA = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "kernel-metadata.json"

EXPECTED_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def make_target_body(makefile_text: str, target: str) -> str:
    match = re.search(rf"^{re.escape(target)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t.*\n)*)", makefile_text, re.MULTILINE)
    return match.group("body") if match else ""


def make_target_block(makefile_text: str, target: str) -> str:
    match = re.search(
        rf"^(?P<header>{re.escape(target)}(?:\s*:[^\n]*)?\n)(?P<body>(?:\t.*\n)*)",
        makefile_text,
        re.MULTILINE,
    )
    return match.group("header") + match.group("body") if match else ""


def contains_any(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if pattern in text]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    makefile_text = read_text(MAKEFILE)
    smoke_source = read_text(LOCAL_SMOKE_SCRIPT)
    metadata = load_json(RUNNER_METADATA) if RUNNER_METADATA.exists() else {}
    local_smoke_body = make_target_body(makefile_text, "kaggle-local-runner-smoke")
    upstream_ready_body = make_target_body(makefile_text, "kaggle-upstream-ready-local")
    prelaunch_body = make_target_body(makefile_text, "kaggle-upstream-prelaunch-local")
    push_upstream_body = make_target_body(makefile_text, "kaggle-push-upstream-rebuild-runner")
    push_only_block = make_target_block(makefile_text, "kaggle-push-only")
    remote_guard_body = make_target_body(makefile_text, "require-kaggle-remote-approval")

    local_full_run_tokens = contains_any(
        smoke_source,
        [
            "nbconvert",
            "--execute",
            "jupyter",
            "ExecutePreprocessor",
            "KAGGLE_WORKING_DIR",
        ],
    )
    contract_smoke_checks = {
        "smoke_mode_declared": "local_contract_staging_no_full_notebook_execution" in smoke_source,
        "stages_contract_outputs": "def stage_contract_outputs" in smoke_source,
        "uses_output_intake_gate": "diagnose_post_id341_kaggle_output_intake_gate.py" in smoke_source,
        "does_not_call_local_notebook_execution": not local_full_run_tokens,
        "markdown_explicitly_blocks_full_local_notebook": (
            "It intentionally does not execute the full upstream notebook locally." in smoke_source
        ),
    }
    makefile_checks = {
        "help_describes_contract_smoke": (
            "make kaggle-local-runner-smoke - Validate upstream runner output contract locally without Kaggle network"
            in makefile_text
        ),
        "help_does_not_claim_full_local_execution": (
            "make kaggle-local-runner-smoke - Execute the upstream runner locally without Kaggle network"
            not in makefile_text
        ),
        "local_smoke_target_uses_contract_script": "scripts/run_post_id341_kaggle_runner_local_smoke.py"
        in local_smoke_body,
        "local_smoke_target_has_no_network_flag": "ALLOW_KAGGLE_REMOTE=true" not in local_smoke_body,
        "local_smoke_target_has_no_dacon_submit": "dacon-submit" not in local_smoke_body,
        "upstream_ready_runs_smoke_before_prelaunch": (
            upstream_ready_body.find("$(MAKE) kaggle-local-runner-smoke") >= 0
            and upstream_ready_body.find("$(MAKE) kaggle-upstream-prelaunch-local")
            > upstream_ready_body.find("$(MAKE) kaggle-local-runner-smoke")
        ),
        "prelaunch_records_compute_boundary_before_final_gate": (
            prelaunch_body.find("diagnose_post_id341_local_remote_compute_boundary_contract.py") >= 0
            and prelaunch_body.find("diagnose_post_id341_kaggle_prelaunch_readiness_gate.py")
            > prelaunch_body.find("diagnose_post_id341_local_remote_compute_boundary_contract.py")
        ),
    }
    remote_guard_checks = {
        "remote_guard_target_exists": bool(remote_guard_body),
        "remote_guard_defaults_closed": "ALLOW_KAGGLE_REMOTE" in remote_guard_body and "exit 2" in remote_guard_body,
        "upstream_push_goes_through_guarded_push_only": "$(MAKE) kaggle-push-only" in push_upstream_body,
        "push_only_requires_remote_approval": "require-kaggle-remote-approval" in push_only_block,
        "approval_phrase_recorded": EXPECTED_APPROVAL_PHRASE in makefile_text
        or EXPECTED_APPROVAL_PHRASE in read_text(REPORTS_DIR / "post_id341_operator_handoff_contract_20260619.md"),
    }
    runner_metadata_checks = {
        "runner_notebook_exists": RUNNER_NOTEBOOK.exists(),
        "runner_metadata_exists": RUNNER_METADATA.exists(),
        "runner_cpu_only_gpu_disabled": metadata.get("enable_gpu") is False,
        "runner_cpu_only_tpu_disabled": metadata.get("enable_tpu") is False,
    }
    checks = {
        "contract_smoke": contract_smoke_checks,
        "makefile": makefile_checks,
        "remote_guard": remote_guard_checks,
        "runner_metadata": runner_metadata_checks,
    }
    pass_contract = all(all(section.values()) for section in checks.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_local_remote_compute_boundary_contract",
        "purpose": (
            "Keep local prelaunch bounded to fast contract-staging checks and reserve the full upstream runner "
            "for exact-approved Kaggle no-submit execution."
        ),
        "inputs": {
            "makefile": safe_relative(MAKEFILE),
            "local_smoke_script": safe_relative(LOCAL_SMOKE_SCRIPT),
            "runner_notebook": safe_relative(RUNNER_NOTEBOOK),
            "runner_metadata": safe_relative(RUNNER_METADATA),
        },
        "local_full_run_forbidden_tokens_found": local_full_run_tokens,
        "checks": checks,
        "decision": {
            "local_remote_compute_boundary_contract_pass": pass_contract,
            "local_full_upstream_notebook_execution_allowed_by_default": False,
            "local_contract_smoke_required_before_kaggle_approval": True,
            "full_runner_requires_exact_kaggle_approval": True,
            "approval_phrase": EXPECTED_APPROVAL_PHRASE,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Boundary is healthy: local work remains a fast contract smoke; full runner stays behind exact "
                "Kaggle no-submit approval."
                if pass_contract
                else "Fix the local/Kaggle compute boundary before consuming any Kaggle approval."
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
        "# Post-ID341 local/Kaggle compute boundary contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It ensures local prelaunch uses a fast contract-staging smoke and never runs the full upstream notebook on the Mac by default.",
        "",
        "## Decision",
        "",
        f"- Boundary contract pass: `{decision['local_remote_compute_boundary_contract_pass']}`",
        f"- Local full upstream notebook execution allowed by default: `{decision['local_full_upstream_notebook_execution_allowed_by_default']}`",
        f"- Local contract smoke required before Kaggle approval: `{decision['local_contract_smoke_required_before_kaggle_approval']}`",
        f"- Full runner requires exact Kaggle approval: `{decision['full_runner_requires_exact_kaggle_approval']}`",
        f"- Approval phrase: `{decision['approval_phrase']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Modify submission log: `{decision['modify_submission_log']}`",
        f"- Create ID342/new submission ID: `{decision['create_id342_or_new_submission_id']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(report["checks"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Local full-run forbidden tokens found",
        "",
        "```json",
        json.dumps(report["local_full_run_forbidden_tokens_found"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_local_remote_compute_boundary_contract_20260619.json",
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
    return 0 if report["decision"]["local_remote_compute_boundary_contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
