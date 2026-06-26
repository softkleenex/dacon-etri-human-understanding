#!/usr/bin/env python3
"""Runtime fail-closed audit for guarded Kaggle remote Make targets.

This local audit intentionally calls the Kaggle-facing Make targets without
`ALLOW_KAGGLE_REMOTE=true` and with `KAGGLE=/usr/bin/false`. A passing result is
that each target fails at the approval guard before any Kaggle CLI command can
run. It never performs a Kaggle network action, creates predictions, creates
submission artifacts, edits the submission log, or submits to DACON.
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
KST = timezone(timedelta(hours=9))

REMOTE_TARGETS = [
    "kaggle-push-code",
    "kaggle-push-upstream-patch-dataset",
    "kaggle-check-base-dataset-files",
    "kaggle-check-upstream-patch-dataset-files",
    "kaggle-push-upstream-rebuild-runner",
    "kaggle-kernel-status-upstream-rebuild",
    "kaggle-pull-upstream-rebuild-output",
    "kaggle-wait-and-pull-upstream-rebuild-output",
]
EXPECTED_GUARD_MESSAGE = "Kaggle remote action blocked"
FORBIDDEN_LEAK_MARKERS = [
    "datasets version",
    "datasets files",
    "kernels push",
    "kernels status",
    "kernels output",
    "Submitting ",
    "dacon_submit_api",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_blocked_target(target: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "ALLOW_KAGGLE_REMOTE": "false",
            "KAGGLE": "/usr/bin/false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    proc = subprocess.run(
        ["make", "--no-print-directory", target],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    forbidden_matches = [marker for marker in FORBIDDEN_LEAK_MARKERS if marker in combined]
    guard_message_present = EXPECTED_GUARD_MESSAGE in combined
    failed_before_remote = bool(proc.returncode != 0 and guard_message_present and not forbidden_matches)
    return {
        "target": target,
        "cmd": f"ALLOW_KAGGLE_REMOTE=false KAGGLE=/usr/bin/false make --no-print-directory {target}",
        "returncode": proc.returncode,
        "guard_message_present": guard_message_present,
        "forbidden_leak_markers": forbidden_matches,
        "failed_before_remote": failed_before_remote,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }


def build_report() -> dict[str, Any]:
    results = [run_blocked_target(target) for target in REMOTE_TARGETS]
    all_failed_before_remote = all(row["failed_before_remote"] for row in results)
    all_returned_nonzero = all(row["returncode"] != 0 for row in results)
    all_guard_messages_present = all(row["guard_message_present"] for row in results)
    no_forbidden_leak_markers = all(not row["forbidden_leak_markers"] for row in results)
    pass_gate = bool(
        all_failed_before_remote and all_returned_nonzero and all_guard_messages_present and no_forbidden_leak_markers
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "kaggle_remote_fail_closed_dryrun_audit",
        "purpose": (
            "Prove that the Kaggle-facing Make targets fail closed without explicit approval before any "
            "Kaggle CLI command can run."
        ),
        "tested_targets": REMOTE_TARGETS,
        "expected_guard_message": EXPECTED_GUARD_MESSAGE,
        "forbidden_leak_markers": FORBIDDEN_LEAK_MARKERS,
        "results": results,
        "checks": {
            "all_failed_before_remote": all_failed_before_remote,
            "all_returned_nonzero": all_returned_nonzero,
            "all_guard_messages_present": all_guard_messages_present,
            "no_forbidden_leak_markers": no_forbidden_leak_markers,
        },
        "decision": {
            "remote_fail_closed_dryrun_pass": pass_gate,
            "requires_explicit_user_approval": True,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Remote targets fail closed without approval; explicit approval is still required before any "
                "Kaggle network step."
                if pass_gate
                else "Fix remote target guard leakage before consuming any Kaggle approval."
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
        "# Kaggle remote fail-closed dry-run audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-submit. It intentionally calls remote Make targets without approval and expects the guard to block before Kaggle CLI execution.",
        "",
        "## Decision",
        "",
        f"- Remote fail-closed dry-run pass: `{decision['remote_fail_closed_dryrun_pass']}`",
        f"- Requires explicit user approval: `{decision['requires_explicit_user_approval']}`",
        f"- Execute Kaggle now: `{decision['execute_kaggle_now']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Target results", ""])
    for row in report["results"]:
        lines.extend(
            [
                f"### `{row['target']}`",
                "",
                f"- Return code: `{row['returncode']}`",
                f"- Guard message present: `{row['guard_message_present']}`",
                f"- Forbidden leak markers: `{row['forbidden_leak_markers']}`",
                f"- Failed before remote: `{row['failed_before_remote']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "kaggle_remote_fail_closed_dryrun_audit_20260618.json",
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
