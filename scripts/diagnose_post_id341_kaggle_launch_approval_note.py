#!/usr/bin/env python3
"""Build the exact post-ID341 Kaggle launch approval note.

This report-only diagnostic turns the ready upstream packet into a user-facing
approval contract. It does not push or run Kaggle, does not create predictions,
and does not submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

SYNTHESIS_PATH = REPORTS_DIR / "post_id341_local_source_closure_synthesis_20260618.json"
PACKET_PATH = REPORTS_DIR / "kaggle_upstream_execution_packet_post_id341_20260618.json"
SELFTEST_PATH = REPORTS_DIR / "kaggle_source_zip_selftest_post_id341_20260618.json"
KERNEL_METADATA_PATH = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "kernel-metadata.json"
DATASET_METADATA_PATH = ROOT / "dataset-metadata.json"
MAKEFILE_PATH = ROOT / "Makefile"
KAGGLE_CLI = ROOT / ".venv" / "bin" / "kaggle"
EXPECTED_DATASET_SOURCES = [
    "softkleenex/dacon-etri-human-base",
    "softkleenex/dacon-etri-upstream-patch-post-id341",
]

REQUIRED_MAKE_TARGETS = [
    "kaggle-approval-contract-test",
    "kaggle-upstream-prelaunch-local",
    "kaggle-upstream-ready-local",
    "kaggle-push-upstream-patch-dataset",
    "kaggle-check-base-dataset-files",
    "kaggle-check-upstream-patch-dataset-files",
    "kaggle-push-upstream-rebuild-runner",
    "kaggle-kernel-status-upstream-rebuild",
    "kaggle-wait-and-pull-upstream-rebuild-output",
    "kaggle-review-upstream-rebuild-output",
    "require-kaggle-remote-approval",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_capture(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }


def make_target_status() -> dict[str, bool]:
    text = MAKEFILE_PATH.read_text(encoding="utf-8")
    return {target: f"{target}:" in text for target in REQUIRED_MAKE_TARGETS}


def build_report() -> dict[str, Any]:
    synthesis = load_json(SYNTHESIS_PATH)
    packet = load_json(PACKET_PATH)
    selftest = load_json(SELFTEST_PATH)
    kernel_metadata = load_json(KERNEL_METADATA_PATH)
    dataset_metadata = load_json(DATASET_METADATA_PATH)
    make_targets = make_target_status()
    kaggle_version = run_capture([str(KAGGLE_CLI), "--version"]) if KAGGLE_CLI.exists() else {"passed": False}
    synthesis_decision = synthesis.get("decision", {})
    packet_decision = packet.get("decision", {})
    selftest_decision = selftest.get("decision", {})
    metadata_safe = bool(
        kernel_metadata.get("id") == "softkleenex/dacon-etri-upstream-rebuild-post-id341"
        and kernel_metadata.get("enable_gpu") is False
        and kernel_metadata.get("enable_tpu") is False
        and kernel_metadata.get("dataset_sources") == EXPECTED_DATASET_SOURCES
        and dataset_metadata.get("id") == "softkleenex/dacon-etri-human-base"
    )
    prerequisites = {
        "local_source_stack_closed": synthesis_decision.get("local_source_stack_closed_for_submission_artifact")
        is True,
        "no_local_lane_promotable": synthesis_decision.get("any_local_lane_promotable") is False,
        "closure_reports_present": synthesis_decision.get("all_required_reports_present") is True,
        "source_zip_selftest_ready": selftest_decision.get("ready_for_explicit_kaggle_approval") is True,
        "upstream_packet_ready": packet_decision.get("ready_for_explicit_approval") is True,
        "kernel_metadata_safe": metadata_safe,
        "kaggle_cli_available": kaggle_version.get("passed") is True,
        "approval_contract_test_target_present": make_targets.get("kaggle-approval-contract-test") is True,
        "prelaunch_local_target_present": make_targets.get("kaggle-upstream-prelaunch-local") is True,
        "ready_local_target_present": make_targets.get("kaggle-upstream-ready-local") is True,
        "make_targets_present": all(make_targets.values()),
    }
    approval_ready = all(prerequisites.values())
    dataset_slug = dataset_metadata["id"]
    kernel_slug = kernel_metadata["id"]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_kaggle_launch_approval_note",
        "purpose": (
            "Freeze the exact approval contract for the next no-submit Kaggle/Linux upstream rebuild parity run."
        ),
        "inputs": {
            "synthesis": str(SYNTHESIS_PATH.relative_to(ROOT)),
            "upstream_packet": str(PACKET_PATH.relative_to(ROOT)),
            "source_zip_selftest": str(SELFTEST_PATH.relative_to(ROOT)),
            "kernel_metadata": str(KERNEL_METADATA_PATH.relative_to(ROOT)),
            "dataset_metadata": str(DATASET_METADATA_PATH.relative_to(ROOT)),
        },
        "prerequisites": prerequisites,
        "make_targets": make_targets,
        "kaggle_cli": {"path": str(KAGGLE_CLI.relative_to(ROOT)), "version_check": kaggle_version},
        "dataset_slug": dataset_slug,
        "dataset_slugs": EXPECTED_DATASET_SOURCES,
        "kernel_slug": kernel_slug,
        "local_pre_approval_commands": [
            {
                "step": 0,
                "command": "make kaggle-upstream-ready-local",
                "why": (
                    "Preferred one-command local readiness wrapper: focused approval contract checks, local runner "
                    "smoke, freshness gate, and final prelaunch readiness. This remains local and no-network."
                ),
                "network": False,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
                "preferred_wrapper": True,
            },
            {
                "step": 1,
                "command": "make kaggle-approval-contract-test",
                "why": "Run the focused local approval/guard contract lint, format-check, and pytest suite.",
                "network": False,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
                "preferred_wrapper": False,
            },
            {
                "step": 2,
                "command": "make kaggle-upstream-prelaunch-local",
                "why": "Refresh all no-submit local prelaunch reports after the focused contract suite passes.",
                "network": False,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
                "preferred_wrapper": False,
            },
        ],
        "exact_approval_required_commands": [
            {
                "step": 1,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset",
                "why": "Push the small no-submit patch dataset with refreshed scripts, reports, submissions, and post-ID341 artifacts.",
                "network": True,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
            {
                "step": 2,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files",
                "why": "Verify Kaggle sees the large base dataset used for raw/processed competition files.",
                "network": True,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
            {
                "step": 3,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files",
                "why": "Verify Kaggle sees the patch dataset before pushing the runner.",
                "network": True,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
            {
                "step": 4,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner",
                "why": "Push the CPU-only upstream rebuild/parity runner notebook.",
                "network": True,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
            {
                "step": 5,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild",
                "why": "Poll until the runner is complete before pulling outputs.",
                "network": True,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
            {
                "step": 6,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output",
                "why": "Pull report-only outputs after the Kaggle runner completes.",
                "network": True,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
            {
                "step": 7,
                "command": "make kaggle-review-upstream-rebuild-output",
                "why": (
                    "Run local no-submit intake, remote summary, next-action decision, and the strict Public-0.58 "
                    "artifact-review threshold gate on pulled outputs."
                ),
                "network": False,
                "creates_dacon_submission": False,
                "creates_submission_artifact": False,
            },
        ],
        "stop_rules": [
            "Do not execute any command above unless the user explicitly approves this exact command sequence.",
            "Immediately before any Kaggle network command, prefer make kaggle-upstream-ready-local; at minimum run make kaggle-approval-contract-test and make kaggle-upstream-prelaunch-local locally, and stop if any local gate fails.",
            "Every Kaggle network command must include ALLOW_KAGGLE_REMOTE=true; without it, Makefile targets fail closed.",
            "Do not DACON-submit from Kaggle or local during this run.",
            "Do not create a submission CSV, base_oof/base_test artifact, notebook history artifact, or submission_log row from pulled outputs automatically.",
            "If the base or patch dataset file listings do not show the expected payloads, stop before pushing the runner.",
            "If the runner fails or outputs are missing, stop and write a failure report instead of creating ID342.",
            (
                "Stop unless the Q3/S3 negative-map handoff is proven: the remote negative map is generated and "
                "staged, remaining-route inventory uses --negative-evidence-map, and eight closed local lanes are "
                "asserted."
            ),
            "Only after pulled reports pass safe intake, next-action, public/local gap, data/source coverage, no-forbidden-action, ID178-beating, and min_proxy <= 0.58 gates may a separate Public-0.58 artifact-contract review begin.",
            "If pulled reports only beat ID178 but remain above 0.58, record an improvement lead only; do not create ID342.",
        ],
        "decision": {
            "approval_note_ready": approval_ready,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "requires_explicit_user_approval": True,
            "next_action": (
                "Ask for explicit approval of the exact command sequence before running Kaggle. Without approval, keep "
                "the repo in no-submit report-only mode."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Kaggle launch approval note",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It does not push/run Kaggle, create predictions, create submission artifacts, or submit to DACON.",
        "",
        "## Decision",
        "",
        f"- Approval note ready: `{decision['approval_note_ready']}`",
        f"- Requires explicit user approval: `{decision['requires_explicit_user_approval']}`",
        f"- Execute Kaggle now: `{decision['execute_kaggle_now']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Prerequisites",
        "",
        "| prerequisite | pass |",
        "|---|---|",
    ]
    for key, value in report["prerequisites"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Local pre-approval commands", ""])
    for row in report["local_pre_approval_commands"]:
        lines.extend(
            [
                f"{row['step']}. `{row['command']}`",
                f"   - Why: {row['why']}",
                f"   - Network: `{row['network']}`",
                f"   - Creates DACON submission: `{row['creates_dacon_submission']}`",
                f"   - Creates submission artifact: `{row['creates_submission_artifact']}`",
                "",
            ]
        )
    lines.extend(["## Exact commands requiring approval", ""])
    for row in report["exact_approval_required_commands"]:
        lines.extend(
            [
                f"{row['step']}. `{row['command']}`",
                f"   - Why: {row['why']}",
                f"   - Network: `{row['network']}`",
                f"   - Creates DACON submission: `{row['creates_dacon_submission']}`",
                f"   - Creates submission artifact: `{row['creates_submission_artifact']}`",
                "",
            ]
        )
    lines.extend(["## Stop rules", ""])
    for rule in report["stop_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_kaggle_launch_approval_note_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
