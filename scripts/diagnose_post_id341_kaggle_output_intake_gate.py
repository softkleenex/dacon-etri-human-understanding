#!/usr/bin/env python3
"""Gate pulled post-ID341 Kaggle upstream outputs.

This report-only diagnostic is intended to run after
`make kaggle-wait-and-pull-upstream-rebuild-output`. It inspects the pulled report files,
checks for accidental submission-like artifacts, and decides whether a separate
artifact-contract review may be started.

It never creates submission CSVs, base prediction artifacts, notebook/history
artifacts, submission-log rows, Kaggle pushes/runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_PULL_DIR = ROOT / "build" / "kaggle_outputs" / "upstream_rebuild_post_id341"
KST = timezone(timedelta(hours=9))

APPROVAL_NOTE_PATH = REPORTS_DIR / "post_id341_kaggle_launch_approval_note_20260618.json"
LOCAL_SYNTHESIS_PATH = REPORTS_DIR / "post_id341_local_source_closure_synthesis_20260618.json"

EXPECTED_REMOTE_REPORTS = [
    "remote_post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "remote_external_etri_prior_smoke_post_id341_20260618.json",
    "remote_combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "remote_post_id341_phase2_raw_source_execution_spec_20260618.json",
    "remote_post_id341_q2q3s3_source_hypothesis_matrix_20260618.json",
    "remote_post_id341_q2q3s3_local_fallback_chain_contract_20260618.json",
    "remote_post_id341_q3s3_negative_evidence_map_20260618.json",
    "remote_post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json",
    "remote_post_id341_q3s3_remaining_route_inventory_20260618.json",
    "remote_phase2_raw_multimodal_execution_plan_post_id341_20260618.json",
    "remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json",
    "remote_phase2_block_A_q1_proxy_gap_20260618.json",
    "remote_phase2_block_B_s2_sleep_recovery_retry_20260618.json",
    "remote_phase3_usage_temporal_microstructure_smoke_20260618.json",
    "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json",
    "remote_post_id341_upstream_rebuild_blueprint_20260618.json",
    "remote_post_id341_upstream_rebuild_phase1_readiness_20260618.json",
    "remote_post_id341_upstream_source_priority_matrix_20260618.json",
    "remote_post_id341_clean_backbone_rebuild_design_20260618.json",
    "remote_post_id341_clean_id221_feature_similarity_smoke_20260618.json",
    "remote_post_id341_clean_id221_inner_selection_probe_20260618.json",
    "remote_post_id341_clean_backbone_next_step_decision_20260618.json",
    "remote_post_id341_clean_id226_calibration_smoke_20260618.json",
    "remote_post_id341_clean_backbone_lane_closure_20260618.json",
]
EXPECTED_RUNNER_MANIFEST = "post_id341_upstream_rebuild_runner_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_first(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def decision_summary(report: dict[str, Any]) -> dict[str, Any]:
    decision = report.get("decision", {})
    return {
        "selected_targets": decision.get("selected_targets"),
        "create_submission_artifact": decision.get("create_submission_artifact"),
        "submit_now": decision.get("submit_now"),
        "push_kaggle_now": decision.get("push_kaggle_now"),
        "run_kaggle_now": decision.get("run_kaggle_now"),
        "public_058_reviewable": decision.get("public_058_reviewable"),
        "beats_id178_public_anchor_proxy": decision.get("beats_id178_public_anchor_proxy"),
        "final_block_a_gate_pass": decision.get("final_block_a_gate_pass"),
        "final_block_b_gate_pass": decision.get("final_block_b_gate_pass"),
        "final_usage_temporal_gate_pass": decision.get("final_usage_temporal_gate_pass"),
        "validation_stop_passed": decision.get("validation_stop_passed"),
        "design_ready": decision.get("design_ready"),
        "source_family": decision.get("source_family"),
        "score_route_opened": decision.get("score_route_opened"),
        "train_model_now": decision.get("train_model_now"),
        "feature_matrix_allowed_now": decision.get("feature_matrix_allowed_now"),
        "public_058_evidence": decision.get("public_058_evidence"),
        "phase1_ready": decision.get("phase1_ready"),
        "local_new_source_stack_closed": decision.get("local_new_source_stack_closed"),
        "close_clean_backbone_lane_now": decision.get("close_clean_backbone_lane_now"),
        "projected_public_proxy_from_id341": decision.get("projected_public_proxy_from_id341"),
    }


def report_status(pull_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for filename in EXPECTED_REMOTE_REPORTS:
        path = find_first(pull_dir, filename)
        if path is None:
            rows.append(
                {
                    "filename": filename,
                    "exists": False,
                    "path": None,
                    "bytes": 0,
                    "decision": None,
                    "parse_error": None,
                }
            )
            continue
        parse_error = None
        summary = None
        try:
            summary = decision_summary(load_json(path))
        except Exception as exc:  # noqa: BLE001
            parse_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "filename": filename,
                "exists": True,
                "path": safe_relative(path),
                "bytes": path.stat().st_size,
                "decision": summary,
                "parse_error": parse_error,
            }
        )
    return rows


def unexpected_artifacts(pull_dir: Path) -> list[dict[str, Any]]:
    if not pull_dir.exists():
        return []
    suspicious: list[dict[str, Any]] = []
    for path in sorted(pull_dir.rglob("*")):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(".csv") or "submission" in lower or "base_oof" in lower or "base_test" in lower:
            suspicious.append(
                {
                    "path": safe_relative(path),
                    "bytes": path.stat().st_size,
                }
            )
    return suspicious


def runner_manifest_status(pull_dir: Path) -> dict[str, Any]:
    path = find_first(pull_dir, EXPECTED_RUNNER_MANIFEST)
    if path is None:
        return {
            "filename": EXPECTED_RUNNER_MANIFEST,
            "exists": False,
            "path": None,
            "bytes": 0,
            "parse_error": None,
            "checks": {},
            "missing_staged_expected_reports": EXPECTED_REMOTE_REPORTS,
            "suspicious_staged_files": [],
            "ok_for_artifact_review": False,
        }
    parse_error = None
    payload: dict[str, Any] | None = None
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        parse_error = f"{type(exc).__name__}: {exc}"
    staged_files = list(payload.get("staged_files", [])) if payload else []
    staged_basenames = {Path(str(value)).name for value in staged_files}
    missing_staged = sorted(set(EXPECTED_REMOTE_REPORTS) - staged_basenames)
    suspicious_staged = sorted(
        name
        for name in staged_basenames
        if name.lower().endswith(".csv")
        or "submission" in name.lower()
        or "base_oof" in name.lower()
        or "base_test" in name.lower()
    )
    checks = {
        "no_submission_created": bool(payload and payload.get("no_submission_created") is True),
        "no_dacon_submission_made": bool(payload and payload.get("no_dacon_submission_made") is True),
        "local_new_source_stack_closed": bool(payload and payload.get("local_new_source_stack_closed") is True),
        "expected_reports_staged": not missing_staged,
        "no_submission_like_staged_files": not suspicious_staged,
    }
    ok = parse_error is None and all(checks.values())
    return {
        "filename": EXPECTED_RUNNER_MANIFEST,
        "exists": True,
        "path": safe_relative(path),
        "bytes": path.stat().st_size,
        "parse_error": parse_error,
        "runner": payload.get("runner") if payload else None,
        "staged_files_count": len(staged_files),
        "checks": checks,
        "missing_staged_expected_reports": missing_staged,
        "suspicious_staged_files": suspicious_staged,
        "ok_for_artifact_review": ok,
    }


def build_report(pull_dir: Path) -> dict[str, Any]:
    approval_note = load_json(APPROVAL_NOTE_PATH)
    local_synthesis = load_json(LOCAL_SYNTHESIS_PATH)
    rows = report_status(pull_dir)
    runner_manifest = runner_manifest_status(pull_dir)
    suspicious = unexpected_artifacts(pull_dir)
    output_present = pull_dir.exists() and any(path.is_file() for path in pull_dir.rglob("*"))
    all_expected_present = all(row["exists"] and not row["parse_error"] for row in rows)
    missing = [row for row in rows if not row["exists"]]
    parse_errors = [row for row in rows if row["parse_error"]]
    remote_review_signals = []
    remote_forbidden_actions = []
    for row in rows:
        decision = row.get("decision") or {}
        if any(
            decision.get(key) is True
            for key in [
                "public_058_reviewable",
                "beats_id178_public_anchor_proxy",
                "final_block_a_gate_pass",
                "final_block_b_gate_pass",
                "final_usage_temporal_gate_pass",
            ]
        ):
            remote_review_signals.append(row)
        if any(
            decision.get(key) is True
            for key in [
                "create_submission_artifact",
                "submit_now",
                "push_kaggle_now",
                "run_kaggle_now",
            ]
        ):
            remote_forbidden_actions.append(row)
    artifact_review_allowed = bool(
        output_present
        and all_expected_present
        and runner_manifest["ok_for_artifact_review"]
        and remote_review_signals
        and not suspicious
        and not remote_forbidden_actions
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_kaggle_output_intake_gate",
        "purpose": (
            "Inspect pulled no-submit Kaggle upstream outputs and decide whether a separate artifact-contract review "
            "may start. This report itself never creates artifacts or submissions."
        ),
        "pull_dir": safe_relative(pull_dir),
        "approval_note": {
            "path": safe_relative(APPROVAL_NOTE_PATH),
            "approval_note_ready": approval_note.get("decision", {}).get("approval_note_ready"),
            "execute_kaggle_now": approval_note.get("decision", {}).get("execute_kaggle_now"),
        },
        "local_synthesis": {
            "path": safe_relative(LOCAL_SYNTHESIS_PATH),
            "local_source_stack_closed_for_submission_artifact": local_synthesis.get("decision", {}).get(
                "local_source_stack_closed_for_submission_artifact"
            ),
            "any_local_lane_promotable": local_synthesis.get("decision", {}).get("any_local_lane_promotable"),
        },
        "output_present": output_present,
        "expected_report_status": rows,
        "expected_runner_manifest": runner_manifest,
        "missing_expected_reports": missing,
        "parse_error_reports": parse_errors,
        "unexpected_submission_like_files": suspicious,
        "remote_review_signal_reports": [
            {"filename": row["filename"], "path": row["path"], "decision": row["decision"]}
            for row in remote_review_signals
        ],
        "remote_forbidden_action_reports": [
            {"filename": row["filename"], "path": row["path"], "decision": row["decision"]}
            for row in remote_forbidden_actions
        ],
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "output_pending": not output_present,
            "all_expected_reports_present": all_expected_present,
            "runner_manifest_required": output_present,
            "runner_manifest_present": runner_manifest["exists"],
            "runner_manifest_ok": bool(output_present and runner_manifest["ok_for_artifact_review"]),
            "runner_manifest_missing_or_invalid": bool(
                output_present and not runner_manifest["ok_for_artifact_review"]
            ),
            "unexpected_submission_like_files_present": bool(suspicious),
            "remote_forbidden_actions_present": bool(remote_forbidden_actions),
            "remote_review_signal_present": bool(remote_review_signals),
            "artifact_contract_review_allowed": artifact_review_allowed,
            "public_058_reviewable": False,
            "next_action": (
                "Outputs are pending; do not create artifacts."
                if not output_present
                else (
                    "A separate artifact-contract review may start, but this intake gate still creates nothing."
                    if artifact_review_allowed
                    else "Do not create artifacts; inspect missing reports, runner manifest, suspicious files, or failed remote gates first."
                )
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
        "# Post-ID341 Kaggle output intake gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It inspects pulled Kaggle outputs if present; it creates no predictions, artifacts, logs, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Output pending: `{decision['output_pending']}`",
        f"- All expected reports present: `{decision['all_expected_reports_present']}`",
        f"- Runner manifest required: `{decision['runner_manifest_required']}`",
        f"- Runner manifest present: `{decision['runner_manifest_present']}`",
        f"- Runner manifest ok: `{decision['runner_manifest_ok']}`",
        f"- Unexpected submission-like files present: `{decision['unexpected_submission_like_files_present']}`",
        f"- Remote forbidden actions present: `{decision['remote_forbidden_actions_present']}`",
        f"- Remote review signal present: `{decision['remote_review_signal_present']}`",
        f"- Artifact-contract review allowed: `{decision['artifact_contract_review_allowed']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Expected reports",
        "",
        "| filename | exists | parse error | path |",
        "|---|---|---|---|",
    ]
    for row in report["expected_report_status"]:
        lines.append(f"| `{row['filename']}` | `{row['exists']}` | `{row['parse_error']}` | `{row['path']}` |")
    manifest = report["expected_runner_manifest"]
    lines.extend(
        [
            "",
            "## Runner manifest",
            "",
            f"- Filename: `{manifest['filename']}`",
            f"- Exists: `{manifest['exists']}`",
            f"- Parse error: `{manifest['parse_error']}`",
            f"- OK for artifact review: `{manifest['ok_for_artifact_review']}`",
            f"- Missing staged expected reports: `{manifest['missing_staged_expected_reports']}`",
            f"- Suspicious staged files: `{manifest['suspicious_staged_files']}`",
        ]
    )
    if report["unexpected_submission_like_files"]:
        lines.extend(["", "## Unexpected submission-like files", ""])
        for row in report["unexpected_submission_like_files"]:
            lines.append(f"- `{row['path']}` ({row['bytes']} bytes)")
    if report["remote_review_signal_reports"]:
        lines.extend(["", "## Remote review signals", ""])
        for row in report["remote_review_signal_reports"]:
            lines.append(f"- `{row['filename']}` → `{row['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-dir", type=Path, default=DEFAULT_PULL_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_kaggle_output_intake_gate_20260618.json",
    )
    args = parser.parse_args()
    pull_dir = args.pull_dir if args.pull_dir.is_absolute() else ROOT / args.pull_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(pull_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
