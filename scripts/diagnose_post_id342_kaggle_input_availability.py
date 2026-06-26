#!/usr/bin/env python3
"""Post-ID342 Kaggle input availability audit.

This report-only diagnostic checks whether the Kaggle datasets needed by the
future no-submit subject-state runner are visible through the Kaggle CLI. It
does not push datasets, push/run kernels, pull outputs, create runner files,
create predictions, edit submission logs, or submit to DACON.
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_kaggle_input_availability_20260624.json"
KST = timezone(timedelta(hours=9))

DATASETS = {
    "base": "softkleenex/dacon-etri-human-base",
    "upstream_patch": "softkleenex/dacon-etri-upstream-patch-post-id341",
}

INPUT_REPORTS = {
    "heavy_approval_readiness": REPORTS_DIR / "post_id342_subject_state_heavy_approval_readiness_20260624.json",
    "raw_schema_contract": REPORTS_DIR / "post_id342_subject_state_raw_schema_contract_20260624.json",
    "success_thresholds": REPORTS_DIR / "post_id342_subject_state_heavy_success_thresholds_20260624.json",
}

FORBIDDEN_ACTIONS_NOW = {
    "push_kaggle_dataset": False,
    "push_kaggle_kernel": False,
    "run_kaggle_kernel": False,
    "pull_kaggle_output": False,
    "create_runner_files": False,
    "read_raw_parquet_payloads": False,
    "write_feature_matrix": False,
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


def parse_dataset_files(output: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    data_lines = [
        line
        for line in lines
        if not line.startswith("name ")
        and not line.startswith("-")
        and not line.startswith("Next Page Token")
        and not line.startswith("Warning:")
    ]
    return {
        "listed_file_count_first_page": len(data_lines),
        "has_next_page_token": any(line.startswith("Next Page Token") for line in lines),
        "first_files": [line.split()[0] for line in data_lines[:10]],
    }


def dataset_observation(name: str, slug: str, kaggle_bin: str, timeout: int, allow_remote: bool) -> dict[str, Any]:
    if not allow_remote:
        return {
            "name": name,
            "slug": slug,
            "remote_checked": False,
            "available": False,
            "reason": "ALLOW_KAGGLE_REMOTE is not true; remote dataset listing intentionally skipped.",
            "command_result": None,
            "parsed": {},
        }
    command = [kaggle_bin, "datasets", "files", slug]
    command_result = run_command(command, timeout)
    combined_output = f"{command_result.get('stdout', '')}\n{command_result.get('stderr', '')}"
    parsed = parse_dataset_files(combined_output) if command_result["ok"] else {}
    return {
        "name": name,
        "slug": slug,
        "remote_checked": True,
        "available": command_result["ok"] and parsed.get("listed_file_count_first_page", 0) > 0,
        "kaggle_cli_outdated_warning": "outdated `kaggle` version" in combined_output,
        "command_result": {
            "ok": command_result["ok"],
            "returncode": command_result["returncode"],
            "stderr_excerpt": command_result.get("stderr", "")[:500],
            "stdout_excerpt": command_result.get("stdout", "")[:1200],
        },
        "parsed": parsed,
    }


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    readiness = reports["heavy_approval_readiness"].get("decision", {})
    raw_schema = reports["raw_schema_contract"].get("decision", {})
    thresholds = reports["success_thresholds"].get("decision", {})
    return {
        "all_input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "heavy_readiness_ready": readiness.get("heavy_approval_readiness_audit_ready") is True,
        "execution_closed": readiness.get("run_kaggle_now") is False and readiness.get("dacon_submit_now") is False,
        "raw_schema_ready": raw_schema.get("raw_schema_contract_ready") is True,
        "success_thresholds_ready": thresholds.get("heavy_success_threshold_contract_ready") is True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    allow_remote = args.allow_remote or os.environ.get("ALLOW_KAGGLE_REMOTE") == "true"
    reports = {key: read_json(path) for key, path in INPUT_REPORTS.items()}
    prerequisites = prerequisite_status(reports)
    observations = {
        name: dataset_observation(name, slug, args.kaggle, args.timeout_seconds, allow_remote)
        for name, slug in DATASETS.items()
    }
    available = all(row.get("available") is True for row in observations.values())
    outdated_warning = any(row.get("kaggle_cli_outdated_warning") is True for row in observations.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_kaggle_input_availability",
        "purpose": "Check Kaggle input dataset visibility for the future no-submit subject-state runner.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "allow_remote": allow_remote,
        "kaggle_binary": args.kaggle,
        "dataset_observations": observations,
        "decision": {
            "kaggle_input_availability_ready": all(prerequisites.values()) and available,
            "remote_checked": allow_remote,
            "required_dataset_count": len(DATASETS),
            "available_dataset_count": sum(1 for row in observations.values() if row.get("available") is True),
            "kaggle_cli_outdated_warning_seen": outdated_warning,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_runner_files_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Both required Kaggle datasets are visible; input availability is not the current blocker."
                if all(prerequisites.values()) and available
                else "Remote input availability is not fully proven; rerun with ALLOW_KAGGLE_REMOTE=true and inspect errors."
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
        "# Post-ID342 Kaggle input availability",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report checks Kaggle dataset visibility only. It creates no runner, prediction, submission artifact, Kaggle run, output pull, log row, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Kaggle input availability ready: `{decision['kaggle_input_availability_ready']}`",
        f"- Remote checked: `{decision['remote_checked']}`",
        f"- Available datasets: `{decision['available_dataset_count']}/{decision['required_dataset_count']}`",
        f"- Kaggle CLI outdated warning seen: `{decision['kaggle_cli_outdated_warning_seen']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Dataset Observations",
        "",
        "| name | slug | checked | available | first page files | next page |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["dataset_observations"].values():
        parsed = row.get("parsed", {})
        lines.append(
            f"| `{row['name']}` | `{row['slug']}` | `{row['remote_checked']}` | `{row['available']}` | "
            f"`{parsed.get('listed_file_count_first_page', 0)}` | `{parsed.get('has_next_page_token', False)}` |"
        )
    lines.extend(
        [
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
    )
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
    print(f"Kaggle input availability ready: {report['decision']['kaggle_input_availability_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
