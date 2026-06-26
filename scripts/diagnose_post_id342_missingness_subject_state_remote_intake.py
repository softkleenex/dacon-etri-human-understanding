#!/usr/bin/env python3
"""Gate pulled post-ID342 missingness subject-state Kaggle outputs.

This report-only diagnostic inspects the no-submit Kaggle runner output
directory for the missingness-aware cross-modal subject-state heavy route. It
rejects CSV/submission-like artifacts and records whether the output is safe to
use as research evidence.

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
DEFAULT_PULL_DIR = ROOT / "build" / "kaggle_outputs" / "missingness_subject_state_post_id342"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_missingness_subject_state_remote_intake_20260624.json"
KST = timezone(timedelta(hours=9))

REQUIRED_REPORTS = {
    "manifest": "post_id342_subject_state_remote_manifest.json",
    "embedding_purity": "post_id342_subject_state_embedding_purity.json",
    "token_coverage": "post_id342_subject_state_token_coverage.json",
    "downstream_smoke": "post_id342_subject_state_downstream_smoke.json",
    "failed_frontier_resistance": "post_id342_subject_state_failed_frontier_resistance.json",
    "public_gap_budget": "post_id342_subject_state_public_gap_budget.json",
    "artifact_separation": "post_id342_subject_state_artifact_separation.json",
    "summary": "post_id342_subject_state_heavy_no_submit_summary.json",
}
FORBIDDEN_NAME_HINTS = ("submission", "base_oof", "base_test", "candidate")


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def forbidden_path(path: Path) -> bool:
    lower = path.name.lower()
    return path.suffix.lower() == ".csv" or any(hint in lower for hint in FORBIDDEN_NAME_HINTS)


def output_files(pull_dir: Path) -> list[dict[str, Any]]:
    if not pull_dir.exists():
        return []
    return [
        {"path": safe_relative(path), "bytes": path.stat().st_size}
        for path in sorted(pull_dir.rglob("*"))
        if path.is_file()
    ]


def unexpected_artifacts(pull_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": safe_relative(path), "bytes": path.stat().st_size}
        for path in sorted(pull_dir.rglob("*"))
        if path.is_file() and forbidden_path(path)
    ] if pull_dir.exists() else []


def load_json_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": safe_relative(path),
            "bytes": 0,
            "parse_ok": False,
            "parse_error": None,
            "payload": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "path": safe_relative(path),
            "bytes": path.stat().st_size,
            "parse_ok": True,
            "parse_error": None,
            "payload": payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "exists": True,
            "path": safe_relative(path),
            "bytes": path.stat().st_size,
            "parse_ok": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "payload": None,
        }


def find_first(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename)) if root.exists() else []
    return matches[0] if matches else direct


def report_statuses(pull_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        key: load_json_status(find_first(pull_dir, filename))
        for key, filename in REQUIRED_REPORTS.items()
    }


def no_submit_checks(statuses: dict[str, dict[str, Any]]) -> dict[str, bool]:
    manifest = statuses["manifest"]["payload"] or {}
    embedding = statuses["embedding_purity"]["payload"] or {}
    downstream = statuses["downstream_smoke"]["payload"] or {}
    public_gap = statuses["public_gap_budget"]["payload"] or {}
    artifact = statuses["artifact_separation"]["payload"] or {}
    summary = statuses["summary"]["payload"] or {}
    return {
        "all_required_reports_parse": all(status["parse_ok"] for status in statuses.values()),
        "remote_no_submit_mode": manifest.get("mode") == "remote_no_submit",
        "manifest_no_submission_created": manifest.get("no_submission_created") is True,
        "manifest_no_prediction_artifacts": manifest.get("no_predictions_created") is True,
        "manifest_no_submission_log_row": manifest.get("no_submission_log_row_created") is True,
        "manifest_no_dacon_submission": manifest.get("no_dacon_submission_made") is True,
        "embedding_label_free": embedding.get("label_free_pretraining") is True,
        "embedding_no_target_labels": embedding.get("uses_target_labels_for_pretraining") is False,
        "downstream_no_submission_artifact": downstream.get("create_submission_artifact") is False,
        "downstream_no_submission_log": downstream.get("modify_submission_log") is False,
        "downstream_no_dacon": downstream.get("dacon_submit_now") is False,
        "artifact_separation_clean": artifact.get("unexpected_submission_like_artifacts") == [],
        "artifact_separation_no_submission": artifact.get("create_submission_artifact") is False,
        "summary_no_submission": summary.get("create_submission_artifact") is False,
        "summary_no_submission_log": summary.get("modify_submission_log") is False,
        "summary_no_dacon": summary.get("dacon_submit_now") is False,
        "public_gap_has_threshold_decision": isinstance(public_gap.get("dacon_candidate_signal"), bool),
    }


def build_report(pull_dir: Path) -> dict[str, Any]:
    statuses = report_statuses(pull_dir)
    checks = no_submit_checks(statuses)
    suspicious = unexpected_artifacts(pull_dir)
    files = output_files(pull_dir)
    summary_payload = statuses["summary"]["payload"] or {}
    token_payload = statuses["token_coverage"]["payload"] or {}
    coverage = token_payload.get("token_coverage", {}) if isinstance(token_payload, dict) else {}
    output_present = pull_dir.exists() and bool(files)
    remote_intake_ready = bool(output_present and all(checks.values()) and not suspicious)
    if remote_intake_ready:
        next_action = (
            "Use this as no-submit research evidence only. Open artifact review only if "
            "artifact_review_allowed is true; DACON still requires exact file/memo approval."
        )
    elif not output_present:
        next_action = "Keep waiting for pulled Kaggle outputs or inspect the remote run log."
    else:
        next_action = "Repair malformed or unsafe remote outputs before treating this route as evidence."
    public_gap = summary_payload.get("gap_to_public_055")
    best_proxy = summary_payload.get("best_projected_public_proxy")
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_missingness_subject_state_remote_intake",
        "purpose": "Inspect pulled no-submit missingness subject-state Kaggle outputs.",
        "pull_dir": safe_relative(pull_dir),
        "output_files": files,
        "unexpected_submission_like_artifacts": suspicious,
        "required_reports": {
            key: {name: value for name, value in status.items() if name != "payload"}
            for key, status in statuses.items()
        },
        "no_submit_checks": checks,
        "decision": {
            "remote_intake_ready": remote_intake_ready,
            "output_present": output_present,
            "unexpected_submission_like_artifacts_present": bool(suspicious),
            "stable_gate_pass_count": summary_payload.get("stable_gate_pass_count"),
            "best_projected_public_proxy": best_proxy,
            "gap_to_public_055": public_gap,
            "artifact_review_allowed": summary_payload.get("artifact_review_allowed"),
            "dacon_candidate_signal": summary_payload.get("dacon_candidate_signal"),
            "raw_rows_read": coverage.get("raw_rows_read"),
            "mean_modality_key_coverage": coverage.get("mean_modality_key_coverage"),
            "next_action": next_action,
        },
        "safe_for_research_evidence": remote_intake_ready,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "dacon_submit_now": False,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    decision = report["decision"]
    lines = [
        "# Post-ID342 missingness subject-state remote intake",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "## Decision",
        "",
        f"- Remote intake ready: `{decision['remote_intake_ready']}`",
        f"- Stable gate pass count: `{decision['stable_gate_pass_count']}`",
        f"- Best projected public proxy: `{decision['best_projected_public_proxy']}`",
        f"- Gap to Public 0.55: `{decision['gap_to_public_055']}`",
        f"- Artifact review allowed: `{decision['artifact_review_allowed']}`",
        f"- DACON candidate signal: `{decision['dacon_candidate_signal']}`",
        f"- Raw rows read: `{decision['raw_rows_read']}`",
        f"- Mean modality key coverage: `{decision['mean_modality_key_coverage']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Output Contract",
        "",
        f"- Unexpected submission-like artifacts: `{len(report['unexpected_submission_like_artifacts'])}`",
        f"- Create submission artifact: `{report['create_submission_artifact']}`",
        f"- Modify submission log: `{report['modify_submission_log']}`",
        f"- DACON submit now: `{report['dacon_submit_now']}`",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-dir", type=Path, default=DEFAULT_PULL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pull_dir = args.pull_dir if args.pull_dir.is_absolute() else ROOT / args.pull_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(pull_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, output)
    print(f"Wrote {safe_relative(output)}")
    print(f"Remote intake ready: {report['decision']['remote_intake_ready']}")
    print(f"Best projected public proxy: {report['decision']['best_projected_public_proxy']}")


if __name__ == "__main__":
    main()
