#!/usr/bin/env python3
"""Run a lightweight post-ID341 Kaggle upstream runner contract smoke.

This local smoke builds the Kaggle source.zip, stages the current local
report-only evidence under the same filenames the Kaggle runner would emit, and
validates that staged output with the same output-intake gate used after Kaggle
pulls. It intentionally does not execute the full upstream notebook locally:
that runner is long enough that it belongs on Kaggle after explicit approval.

It never pushes/runs Kaggle, creates submission CSVs, edits the submission log,
creates base prediction artifacts, or submits to DACON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
NOTEBOOK = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "Upstream_Rebuild_Post_ID341_Runner.ipynb"
SMOKE_ROOT = ROOT / "build" / "post_id341_kaggle_runner_local_smoke"
WORK_DIR = SMOKE_ROOT / "work"
OUTPUT_DIR = SMOKE_ROOT / "kaggle_outputs"
EXECUTED_NOTEBOOK_DIR = SMOKE_ROOT / "executed_notebook"
LOCAL_NOTEBOOK = SMOKE_ROOT / NOTEBOOK.name
LOCAL_SOURCE_ZIP = SMOKE_ROOT / "source.zip"
INTAKE_REPORT = REPORTS_DIR / "post_id341_kaggle_local_runner_smoke_intake_20260618.json"
RUNNER_MANIFEST = "post_id341_upstream_rebuild_runner_manifest.json"

REMOTE_TO_LOCAL_REPORT_OVERRIDES = {
    "remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json": (
        "phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json"
    ),
    "remote_phase2_block_A_q1_proxy_gap_20260618.json": "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "remote_phase2_block_B_s2_sleep_recovery_retry_20260618.json": (
        "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json"
    ),
    "remote_phase3_usage_temporal_microstructure_smoke_20260618.json": (
        "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json"
    ),
}


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_capture(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-5000:],
        "stderr_tail": proc.stderr[-5000:],
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_input_fingerprints() -> dict[str, dict[str, Any]]:
    paths = {
        NOTEBOOK,
        ROOT / "Makefile",
        ROOT / "dataset-metadata.json",
        ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "kernel-metadata.json",
        ROOT / "scripts" / "diagnose_post_id341_kaggle_output_intake_gate.py",
        ROOT / "scripts" / "diagnose_post_id341_kaggle_runner_intake_contract.py",
        ROOT / "scripts" / "diagnose_kaggle_source_zip_selftest_post_id341.py",
        ROOT / "scripts" / "diagnose_kaggle_upstream_execution_packet_post_id341.py",
    }
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        for rel in sorted(set(re.findall(r"scripts/[A-Za-z0-9_./-]+?\.py", text))):
            paths.add(ROOT / rel)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        rel = safe_relative(path)
        if not path.exists():
            out[rel] = {"exists": False, "sha256": None, "size_bytes": None, "mtime_ns": None}
            continue
        stat = path.stat()
        out[rel] = {
            "exists": True,
            "sha256": sha256_file(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return out


def output_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
    by_suffix: dict[str, int] = {}
    suspicious: list[str] = []
    for file_path in files:
        suffix = file_path.suffix.lower() or "<none>"
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        lower = file_path.name.lower()
        if lower.endswith(".csv") or "submission" in lower or "base_oof" in lower or "base_test" in lower:
            suspicious.append(safe_relative(file_path))
    return {
        "exists": path.exists(),
        "file_count": len(files),
        "by_suffix": by_suffix,
        "files": [safe_relative(p) for p in files],
        "suspicious_submission_like_files": suspicious,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intake_module():
    script = ROOT / "scripts" / "diagnose_post_id341_kaggle_output_intake_gate.py"
    spec = importlib.util.spec_from_file_location("post_id341_kaggle_output_intake_gate", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load intake script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_report_path_for_remote(remote_name: str) -> Path:
    local_name = REMOTE_TO_LOCAL_REPORT_OVERRIDES.get(remote_name, remote_name.removeprefix("remote_"))
    return REPORTS_DIR / local_name


def stage_contract_outputs() -> dict[str, Any]:
    """Stage local reports using the remote runner output filenames.

    The output-intake gate deliberately consumes remote-style filenames. For a
    local no-network smoke, copying already-generated local reports into that
    shape verifies the contract without running the long notebook locally.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intake_module = load_intake_module()
    expected_reports = list(intake_module.EXPECTED_REMOTE_REPORTS)
    staged_files: list[str] = []
    missing_local_reports: list[dict[str, str]] = []
    for remote_name in expected_reports:
        src_json = local_report_path_for_remote(remote_name)
        dst_json = OUTPUT_DIR / remote_name
        if not src_json.exists():
            missing_local_reports.append({"remote": remote_name, "local": safe_relative(src_json)})
            continue
        shutil.copy2(src_json, dst_json)
        staged_files.append(str(dst_json))
        src_md = src_json.with_suffix(".md")
        if src_md.exists():
            dst_md = dst_json.with_suffix(".md")
            shutil.copy2(src_md, dst_md)
            staged_files.append(str(dst_md))
    readiness = load_json(REPORTS_DIR / "post_id341_upstream_rebuild_phase1_readiness_20260618.json")
    manifest = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "runner": "post_id341_local_contract_smoke_no_submit",
        "staged_files": staged_files,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "local_new_source_stack_closed": readiness.get("decision", {}).get("local_new_source_stack_closed") is True,
        "next_allowed_action": (
            "This is a local contract smoke only. Full notebook execution remains a Kaggle no-submit action after "
            "explicit approval; do not create or submit a CSV from this runner."
        ),
    }
    manifest_path = OUTPUT_DIR / RUNNER_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    staged_files.append(str(manifest_path))
    return {
        "cmd": "stage local reports as remote-style Kaggle output",
        "returncode": 0 if not missing_local_reports else 1,
        "passed": not missing_local_reports,
        "stdout_tail": json.dumps(
            {
                "staged_file_count": len(staged_files),
                "missing_local_reports": missing_local_reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        "stderr_tail": "",
        "missing_local_reports": missing_local_reports,
        "staged_files": [safe_relative(Path(path)) for path in staged_files],
    }


def cleanup_transient_build() -> dict[str, Any]:
    removed: list[str] = []
    for path in [SMOKE_ROOT, ROOT / "build" / "kaggle_dataset"]:
        if path.exists():
            shutil.rmtree(path)
            removed.append(safe_relative(path))
    build_dir = ROOT / "build"
    build_removed_if_empty = False
    if build_dir.exists():
        try:
            next(build_dir.iterdir())
        except StopIteration:
            build_dir.rmdir()
            build_removed_if_empty = True
    return {
        "removed_paths": removed,
        "build_removed_if_empty": build_removed_if_empty,
        "build_exists_after_cleanup": build_dir.exists(),
    }


def build_report(keep_build: bool, timeout_seconds: int) -> dict[str, Any]:
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    build = run_capture(["make", "build-kaggle-dataset"], timeout=timeout_seconds)
    if build["passed"]:
        shutil.copy2(ROOT / "build" / "kaggle_dataset" / "source.zip", LOCAL_SOURCE_ZIP)
        shutil.copy2(NOTEBOOK, LOCAL_NOTEBOOK)
    stage = stage_contract_outputs() if build["passed"] else {"returncode": 1, "passed": False}
    before_cleanup_snapshot = output_snapshot(OUTPUT_DIR)
    intake = run_capture(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "scripts/diagnose_post_id341_kaggle_output_intake_gate.py",
            "--pull-dir",
            str(OUTPUT_DIR),
            "--output",
            str(INTAKE_REPORT),
        ],
        timeout=timeout_seconds,
    )
    intake_payload = load_json(INTAKE_REPORT) if INTAKE_REPORT.exists() else {}
    intake_decision = intake_payload.get("decision", {})
    smoke_passed = bool(
        build["passed"]
        and stage["passed"]
        and intake["passed"]
        and intake_decision.get("all_expected_reports_present") is True
        and intake_decision.get("runner_manifest_ok") is True
        and intake_decision.get("unexpected_submission_like_files_present") is False
        and intake_decision.get("remote_forbidden_actions_present") is False
        and intake_decision.get("create_submission_artifact") is False
        and intake_decision.get("submit_now") is False
    )
    cleanup = {"skipped": True} if keep_build else cleanup_transient_build()
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_kaggle_runner_local_smoke",
        "purpose": (
            "Build source.zip, stage current local reports as remote-style Kaggle outputs, and validate the "
            "report-only output contract without executing the full notebook locally."
        ),
        "smoke_mode": "local_contract_staging_no_full_notebook_execution",
        "notebook": safe_relative(NOTEBOOK),
        "local_notebook_copy": safe_relative(LOCAL_NOTEBOOK),
        "local_source_zip": safe_relative(LOCAL_SOURCE_ZIP),
        "smoke_root": safe_relative(SMOKE_ROOT),
        "output_dir": safe_relative(OUTPUT_DIR),
        "intake_report": safe_relative(INTAKE_REPORT),
        "input_fingerprints": collect_input_fingerprints(),
        "commands": {
            "build_kaggle_dataset": build,
            "stage_contract_outputs": stage,
            "intake_gate": intake,
        },
        "output_snapshot_before_cleanup": before_cleanup_snapshot,
        "intake_decision": intake_decision,
        "cleanup": cleanup,
        "decision": {
            "local_runner_smoke_passed": smoke_passed,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Local runner contract smoke passed; explicit approval is still required for any full Kaggle run."
                if smoke_passed
                else "Fix local runner contract smoke failure before consuming Kaggle approval."
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
    snapshot = report["output_snapshot_before_cleanup"]
    lines = [
        "# Post-ID341 Kaggle runner local smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-network. It builds source.zip, stages current local reports under remote-style output filenames, validates them with the intake gate, and creates no submission artifacts. It intentionally does not execute the full upstream notebook locally.",
        "",
        "## Decision",
        "",
        f"- Local runner smoke passed: `{decision['local_runner_smoke_passed']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Command results",
        "",
    ]
    for key, row in report["commands"].items():
        lines.append(f"- {key}: passed `{row['passed']}` returncode `{row['returncode']}`")
    lines.extend(
        [
            "",
            "## Output snapshot before cleanup",
            "",
            f"- File count: `{snapshot['file_count']}`",
            f"- By suffix: `{snapshot['by_suffix']}`",
            f"- Suspicious submission-like files: `{snapshot['suspicious_submission_like_files']}`",
            "",
            "## Intake decision",
            "",
            "```json",
            json.dumps(report["intake_decision"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Cleanup",
            "",
            "```json",
            json.dumps(report["cleanup"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=REPORTS_DIR / "post_id341_kaggle_runner_local_smoke_20260618.json"
    )
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(keep_build=args.keep_build, timeout_seconds=args.timeout_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0 if report["decision"]["local_runner_smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
