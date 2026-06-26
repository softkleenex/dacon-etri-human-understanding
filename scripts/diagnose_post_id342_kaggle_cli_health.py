#!/usr/bin/env python3
"""Post-ID342 Kaggle CLI health audit.

This report-only diagnostic checks the project-local Kaggle CLI version,
dataset listing, and existing upstream kernel status. It performs no dataset
push, kernel push/run, output pull, runner creation, submission artifact
creation, submission-log edit, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_kaggle_cli_health_20260624.json"
KST = timezone(timedelta(hours=9))

BASE_DATASET_SLUG = "softkleenex/dacon-etri-human-base"
PATCH_DATASET_SLUG = "softkleenex/dacon-etri-upstream-patch-post-id341"
UPSTREAM_KERNEL_SLUG = "softkleenex/dacon-etri-upstream-rebuild-post-id341"

INPUT_REPORTS = {
    "goal_state_audit": REPORTS_DIR / "post_id342_goal_state_audit_20260624.json",
    "kaggle_input_availability": REPORTS_DIR / "post_id342_kaggle_input_availability_20260624.json",
    "heavy_approval_readiness": REPORTS_DIR / "post_id342_subject_state_heavy_approval_readiness_20260624.json",
}

FORBIDDEN_ACTIONS_NOW = {
    "push_kaggle_dataset": False,
    "push_kaggle_kernel": False,
    "run_kaggle_kernel": False,
    "pull_kaggle_output": False,
    "create_runner_files": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
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
    return {"path": rel(path), "exists": True, "parse_ok": True, "decision": payload.get("decision", {})}


def run_command(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)  # noqa: S603
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
        }
    return {
        "ok": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error_type": None,
    }


def command_excerpt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result["ok"],
        "command": result["command"],
        "returncode": result["returncode"],
        "stdout_excerpt": result.get("stdout", "")[:1200],
        "stderr_excerpt": result.get("stderr", "")[:500],
        "error_type": result.get("error_type"),
    }


def parse_kaggle_version(output: str) -> str | None:
    for token in output.replace("\n", " ").split():
        if token.count(".") >= 1 and token[0].isdigit():
            return token
    return None


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    goal = reports["goal_state_audit"].get("decision", {})
    inputs = reports["kaggle_input_availability"].get("decision", {})
    readiness = reports["heavy_approval_readiness"].get("decision", {})
    return {
        "all_input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "goal_not_achieved": goal.get("goal_achieved") is False and goal.get("public_055_reached") is False,
        "action_gates_closed": goal.get("run_kaggle_now") is False and goal.get("dacon_submit_now") is False,
        "input_availability_ready": inputs.get("kaggle_input_availability_ready") is True,
        "heavy_readiness_ready": readiness.get("heavy_approval_readiness_audit_ready") is True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    allow_remote = args.allow_remote or os.environ.get("ALLOW_KAGGLE_REMOTE") == "true"
    reports = {key: read_json(path) for key, path in INPUT_REPORTS.items()}
    prerequisites = prerequisite_status(reports)

    version_result = run_command([args.kaggle, "--version"], args.timeout_seconds)
    version_output = f"{version_result.get('stdout', '')}\n{version_result.get('stderr', '')}"
    base_files = None
    patch_files = None
    kernel_status = None
    if allow_remote:
        base_files = run_command([args.kaggle, "datasets", "files", BASE_DATASET_SLUG], args.timeout_seconds)
        patch_files = run_command([args.kaggle, "datasets", "files", PATCH_DATASET_SLUG], args.timeout_seconds)
        kernel_status = run_command([args.kaggle, "kernels", "status", UPSTREAM_KERNEL_SLUG], args.timeout_seconds)

    remote_results = [result for result in [base_files, patch_files, kernel_status] if result is not None]
    combined_remote_output = "\n".join(
        f"{result.get('stdout', '')}\n{result.get('stderr', '')}" for result in remote_results
    )
    remote_ok = allow_remote and all(result["ok"] for result in remote_results)
    dataset_ok = (
        allow_remote and base_files is not None and patch_files is not None and base_files["ok"] and patch_files["ok"]
    )
    upstream_complete = kernel_status is not None and "KernelWorkerStatus.COMPLETE" in kernel_status.get("stdout", "")
    outdated_warning = "outdated `kaggle` version" in combined_remote_output
    installed_version = parse_kaggle_version(version_output)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_kaggle_cli_health",
        "purpose": "Check project-local Kaggle CLI health without any push/run/pull/submit action.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "allow_remote": allow_remote,
        "kaggle_binary": args.kaggle,
        "version_check": command_excerpt(version_result),
        "remote_checks": {
            "base_dataset_files": None if base_files is None else command_excerpt(base_files),
            "patch_dataset_files": None if patch_files is None else command_excerpt(patch_files),
            "upstream_kernel_status": None if kernel_status is None else command_excerpt(kernel_status),
        },
        "decision": {
            "kaggle_cli_health_ready": all(prerequisites.values()) and version_result["ok"] and remote_ok,
            "remote_checked": allow_remote,
            "installed_kaggle_version": installed_version,
            "latest_warning_version": "2.2.2" if outdated_warning else None,
            "outdated_warning_seen": outdated_warning,
            "dataset_listing_ok": dataset_ok,
            "upstream_kernel_complete": upstream_complete,
            "auth_network_slug_blocker_observed": allow_remote and not remote_ok,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "pull_kaggle_output_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Kaggle CLI is usable for required status/list checks; only the outdated-version warning remains."
                if all(prerequisites.values()) and version_result["ok"] and remote_ok
                else "Kaggle CLI health is not fully proven; inspect command excerpts before any approved remote run."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_kaggle_push_or_run_made": True,
        "no_kaggle_output_pull_made": True,
        "no_runner_files_created": True,
        "no_prediction_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 Kaggle CLI health",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report checks Kaggle CLI health only. It performs no Kaggle push, kernel run, output pull, submission artifact creation, log edit, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Kaggle CLI health ready: `{decision['kaggle_cli_health_ready']}`",
        f"- Remote checked: `{decision['remote_checked']}`",
        f"- Installed version: `{decision['installed_kaggle_version']}`",
        f"- Outdated warning seen: `{decision['outdated_warning_seen']}`",
        f"- Dataset listing ok: `{decision['dataset_listing_ok']}`",
        f"- Upstream kernel complete: `{decision['upstream_kernel_complete']}`",
        f"- Auth/network/slug blocker observed: `{decision['auth_network_slug_blocker_observed']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
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
    parser.add_argument("--kaggle", default=os.environ.get("KAGGLE", ".venv/bin/kaggle"))
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Kaggle CLI health ready: {report['decision']['kaggle_cli_health_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
