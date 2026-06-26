#!/usr/bin/env python3
"""Gate pulled post-ID342 self-supervised subject-state Kaggle outputs.

This report-only diagnostic inspects the no-submit Kaggle runner output
directory. It rejects CSV/submission-like artifacts, checks the remote report
and runner manifest, and records whether the output is safe to use as research
evidence.

It never creates predictions, submission CSVs, base prediction artifacts,
submission-log rows, Kaggle pushes/runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_PULL_DIR = ROOT / "build" / "kaggle_outputs" / "self_supervised_subject_state_post_id342"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_self_supervised_subject_state_remote_intake_20260624.json"
REMOTE_REPORT_NAME = "remote_post_id342_self_supervised_subject_state_no_submit_smoke_20260624.json"
REMOTE_FEATURE_GATE_REPORT_NAME = "remote_post_id342_raw_subject_state_feature_gate_20260624.json"
REMOTE_SEQUENCE_GATE_REPORT_NAME = "remote_post_id342_s3q3_raw_sequence_representation_gate_20260624.json"
REMOTE_MANIFEST_NAME = "post_id342_self_supervised_subject_state_runner_manifest.json"
FORBIDDEN_NAME_HINTS = ("submission", "base_oof", "base_test", "candidate")
KST = timezone(timedelta(hours=9))


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_first(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def forbidden_path(path: Path) -> bool:
    lower = path.name.lower()
    return path.suffix.lower() == ".csv" or any(hint in lower for hint in FORBIDDEN_NAME_HINTS)


def unexpected_artifacts(pull_dir: Path) -> list[dict[str, Any]]:
    if not pull_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(pull_dir.rglob("*")):
        if path.is_file() and forbidden_path(path):
            rows.append({"path": safe_relative(path), "bytes": path.stat().st_size})
    return rows


def json_status(path: Path | None, expected_name: str) -> dict[str, Any]:
    if path is None:
        return {
            "filename": expected_name,
            "exists": False,
            "path": None,
            "bytes": 0,
            "parse_ok": False,
            "parse_error": None,
            "payload": None,
        }
    try:
        payload = load_json(path)
        return {
            "filename": expected_name,
            "exists": True,
            "path": safe_relative(path),
            "bytes": path.stat().st_size,
            "parse_ok": True,
            "parse_error": None,
            "payload": payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "filename": expected_name,
            "exists": True,
            "path": safe_relative(path),
            "bytes": path.stat().st_size,
            "parse_ok": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "payload": None,
        }


def remote_report_checks(payload: dict[str, Any] | None) -> dict[str, bool]:
    decision = payload.get("decision", {}) if payload else {}
    return {
        "smoke_passed": bool(decision.get("no_submit_runner_smoke_pass") is True),
        "no_submission_artifact_decision": bool(decision.get("create_submission_artifact") is False),
        "no_submission_log_decision": bool(decision.get("modify_submission_log") is False),
        "no_dacon_submit_decision": bool(decision.get("dacon_submit") is False),
        "no_submission_created": bool(payload and payload.get("no_submission_created") is True),
        "no_submission_artifact_created": bool(payload and payload.get("no_submission_artifact_created") is True),
        "no_submission_log_row_created": bool(payload and payload.get("no_submission_log_row_created") is True),
        "no_dacon_submission_made": bool(payload and payload.get("no_dacon_submission_made") is True),
    }


def manifest_checks(payload: dict[str, Any] | None) -> dict[str, bool]:
    output_files = [str(value) for value in payload.get("output_files", [])] if payload else []
    suspicious_outputs = [name for name in output_files if forbidden_path(Path(name))]
    return {
        "remote_no_submit_mode": bool(payload and payload.get("mode") == "remote_no_submit"),
        "no_submission_created": bool(payload and payload.get("no_submission_created") is True),
        "no_submission_artifact_created": bool(payload and payload.get("no_submission_artifact_created") is True),
        "no_submission_log_row_created": bool(payload and payload.get("no_submission_log_row_created") is True),
        "no_dacon_submission_made": bool(payload and payload.get("no_dacon_submission_made") is True),
        "output_file_list_has_no_forbidden_names": not suspicious_outputs,
    }


def feature_gate_report_checks(payload: dict[str, Any] | None) -> dict[str, bool]:
    decision = payload.get("decision", {}) if payload else {}
    return {
        "reviewed_candidates_present": bool(decision.get("reviewed_candidates", 0) > 0),
        "no_feature_matrix_artifact_decision": bool(decision.get("create_feature_matrix_artifact") is False),
        "no_prediction_artifact_decision": bool(decision.get("create_prediction_artifact") is False),
        "no_submission_artifact_decision": bool(decision.get("create_submission_artifact") is False),
        "no_submission_log_decision": bool(decision.get("modify_submission_log") is False),
        "no_dacon_submit_decision": bool(decision.get("dacon_submit_now") is False),
        "no_feature_matrix_created": bool(payload and payload.get("no_feature_matrix_created") is True),
        "no_predictions_created": bool(payload and payload.get("no_predictions_created") is True),
        "no_submission_created": bool(payload and payload.get("no_submission_created") is True),
        "no_submission_log_row_created": bool(payload and payload.get("no_submission_log_row_created") is True),
        "no_dacon_submission_made": bool(payload and payload.get("no_dacon_submission_made") is True),
    }


def sequence_gate_report_checks(payload: dict[str, Any] | None) -> dict[str, bool]:
    decision = payload.get("decision", {}) if payload else {}
    return {
        "reviewed_candidates_present": bool(decision.get("reviewed_candidates", 0) > 0),
        "no_feature_matrix_artifact_decision": bool(decision.get("create_feature_matrix_artifact") is False),
        "no_prediction_artifact_decision": bool(decision.get("create_prediction_artifact") is False),
        "no_submission_artifact_decision": bool(decision.get("create_submission_artifact") is False),
        "no_submission_log_decision": bool(decision.get("modify_submission_log") is False),
        "no_dacon_submit_decision": bool(decision.get("dacon_submit_now") is False),
        "no_feature_matrix_created": bool(payload and payload.get("no_feature_matrix_created") is True),
        "no_predictions_created": bool(payload and payload.get("no_predictions_created") is True),
        "no_submission_created": bool(payload and payload.get("no_submission_created") is True),
        "no_submission_log_row_created": bool(payload and payload.get("no_submission_log_row_created") is True),
        "no_dacon_submission_made": bool(payload and payload.get("no_dacon_submission_made") is True),
    }


def output_files(pull_dir: Path) -> list[dict[str, Any]]:
    if not pull_dir.exists():
        return []
    return [
        {"path": safe_relative(path), "bytes": path.stat().st_size}
        for path in sorted(pull_dir.rglob("*"))
        if path.is_file()
    ]


def build_report(pull_dir: Path) -> dict[str, Any]:
    remote_status = json_status(find_first(pull_dir, REMOTE_REPORT_NAME), REMOTE_REPORT_NAME)
    feature_gate_status = json_status(
        find_first(pull_dir, REMOTE_FEATURE_GATE_REPORT_NAME),
        REMOTE_FEATURE_GATE_REPORT_NAME,
    )
    sequence_gate_status = json_status(
        find_first(pull_dir, REMOTE_SEQUENCE_GATE_REPORT_NAME),
        REMOTE_SEQUENCE_GATE_REPORT_NAME,
    )
    manifest_status = json_status(find_first(pull_dir, REMOTE_MANIFEST_NAME), REMOTE_MANIFEST_NAME)
    suspicious = unexpected_artifacts(pull_dir)
    files = output_files(pull_dir)
    report_checks = remote_report_checks(remote_status["payload"])
    feature_gate_checks = feature_gate_report_checks(feature_gate_status["payload"])
    sequence_gate_checks = sequence_gate_report_checks(sequence_gate_status["payload"])
    runner_manifest_checks = manifest_checks(manifest_status["payload"])
    output_present = pull_dir.exists() and bool(files)
    remote_report_ok = bool(remote_status["parse_ok"] and all(report_checks.values()))
    feature_gate_report_ok = bool(
        not feature_gate_status["exists"] or (feature_gate_status["parse_ok"] and all(feature_gate_checks.values()))
    )
    sequence_gate_report_ok = bool(
        not sequence_gate_status["exists"] or (sequence_gate_status["parse_ok"] and all(sequence_gate_checks.values()))
    )
    manifest_ok = bool(manifest_status["parse_ok"] and all(runner_manifest_checks.values()))
    remote_intake_ready = bool(
        output_present
        and remote_report_ok
        and feature_gate_report_ok
        and sequence_gate_report_ok
        and manifest_ok
        and not suspicious
    )
    if remote_intake_ready:
        next_action = (
            "Use this as no-submit research evidence only. Do not create a submission artifact; "
            "derive a separate supervised candidate only after a new artifact-contract review."
        )
    elif not output_present:
        next_action = "Keep waiting for pulled Kaggle outputs or inspect the failed remote log."
    else:
        next_action = "Repair malformed or unsafe remote outputs before using this lane as evidence."
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_self_supervised_subject_state_remote_intake",
        "purpose": (
            "Inspect pulled no-submit Kaggle subject-state outputs and decide whether they are safe research evidence."
        ),
        "pull_dir": safe_relative(pull_dir),
        "output_files": files,
        "unexpected_submission_like_artifacts": suspicious,
        "remote_report": {key: value for key, value in remote_status.items() if key != "payload"},
        "remote_report_checks": report_checks,
        "feature_gate_report": {key: value for key, value in feature_gate_status.items() if key != "payload"},
        "feature_gate_report_checks": feature_gate_checks,
        "sequence_gate_report": {key: value for key, value in sequence_gate_status.items() if key != "payload"},
        "sequence_gate_report_checks": sequence_gate_checks,
        "runner_manifest": {key: value for key, value in manifest_status.items() if key != "payload"},
        "runner_manifest_checks": runner_manifest_checks,
        "decision": {
            "remote_intake_ready": remote_intake_ready,
            "output_present": output_present,
            "remote_report_ok": remote_report_ok,
            "feature_gate_report_present": bool(feature_gate_status["exists"]),
            "feature_gate_report_ok": feature_gate_report_ok,
            "sequence_gate_report_present": bool(sequence_gate_status["exists"]),
            "sequence_gate_report_ok": sequence_gate_report_ok,
            "runner_manifest_ok": manifest_ok,
            "unexpected_submission_like_artifacts_present": bool(suspicious),
            "create_submission_artifact": False,
            "submit_now": False,
            "dacon_submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": next_action,
        },
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action_made_by_this_script": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID342 self-supervised subject-state remote intake",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It checks pulled Kaggle no-submit outputs and refuses any submission-like artifact.",
        "",
        "## Decision",
        "",
        f"- Remote intake ready: `{decision['remote_intake_ready']}`",
        f"- Output present: `{decision['output_present']}`",
        f"- Remote report OK: `{decision['remote_report_ok']}`",
        f"- Feature gate report present: `{decision['feature_gate_report_present']}`",
        f"- Feature gate report OK: `{decision['feature_gate_report_ok']}`",
        f"- Sequence gate report present: `{decision['sequence_gate_report_present']}`",
        f"- Sequence gate report OK: `{decision['sequence_gate_report_ok']}`",
        f"- Runner manifest OK: `{decision['runner_manifest_ok']}`",
        (
            "- Unexpected submission-like artifacts present: "
            f"`{decision['unexpected_submission_like_artifacts_present']}`"
        ),
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Output Files",
        "",
    ]
    if report["output_files"]:
        for row in report["output_files"]:
            lines.append(f"- `{row['path']}` ({row['bytes']} bytes)")
    else:
        lines.append("- none")
    lines.extend(["", "## Remote Report Checks", ""])
    for key, value in report["remote_report_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Feature Gate Report Checks", ""])
    for key, value in report["feature_gate_report_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Sequence Gate Report Checks", ""])
    for key, value in report["sequence_gate_report_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Runner Manifest Checks", ""])
    for key, value in report["runner_manifest_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Guardrails", ""])
    for key in [
        "no_predictions_created",
        "no_submission_created",
        "no_base_prediction_artifact_created",
        "no_submission_log_row_created",
        "no_kaggle_action_made_by_this_script",
        "no_dacon_submission_made",
    ]:
        lines.append(f"- {key}: `{report[key]}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-dir", type=Path, default=DEFAULT_PULL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args.pull_dir if args.pull_dir.is_absolute() else ROOT / args.pull_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {safe_relative(output_path)}")
    print(f"Wrote {safe_relative(output_path.with_suffix('.md'))}")
    print(f"Remote intake ready: {report['decision']['remote_intake_ready']}")
    print(f"Create submission artifact: {report['decision']['create_submission_artifact']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
