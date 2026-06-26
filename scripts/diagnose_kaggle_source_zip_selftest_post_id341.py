#!/usr/bin/env python3
"""Self-test the Kaggle source.zip for the post-ID341 upstream runner.

This is a local report-only smoke. It rebuilds `build/kaggle_dataset/source.zip`,
verifies required entries, extracts it into a temporary directory, and runs the
phase-1 readiness audit from the extracted project. It then removes the transient
`build/kaggle_dataset` package directory unless `--keep-build` is passed.

No Kaggle push/run, DACON submission, submission CSV, base prediction artifact,
notebook artifact, or submission-log row is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SOURCE_ZIP = ROOT / "build" / "kaggle_dataset" / "source.zip"
KST = timezone(timedelta(hours=9))

REQUIRED_ENTRIES = [
    "scripts/diagnose_post_id341_upstream_rebuild_phase1_readiness.py",
    "scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py",
    "scripts/diagnose_post_id341_transfer_gate_v2_with_joint_risk.py",
    "scripts/diagnose_external_etri_prior_smoke_post_id341.py",
    "scripts/diagnose_combined_sleep_evening_external_prior_gate_post_id341.py",
    "scripts/diagnose_post_id341_phase2_raw_source_execution_spec.py",
    "scripts/diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py",
    "scripts/diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py",
    "scripts/diagnose_post_id341_q3s3_remaining_route_inventory.py",
    "scripts/diagnose_phase2_raw_multimodal_execution_plan_post_id341.py",
    "scripts/diagnose_phase2_block_a_q1_s4_sleep_residual_bridge_post_id341.py",
    "scripts/diagnose_phase2_block_a_q1_proxy_gap_post_id341.py",
    "scripts/diagnose_phase2_block_b_s2_sleep_recovery_retry_post_id341.py",
    "scripts/diagnose_phase3_usage_temporal_microstructure_smoke_post_id341.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_design.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_metadata_smoke.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_spec.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_generation_dryrun.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract.py",
    "scripts/write_post_id341_usage_semantic_burst_markov_feature_matrix.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop.py",
    "scripts/diagnose_post_id341_self_supervised_subject_state_embedding_design.py",
    "scripts/diagnose_post_id341_local_source_closure_synthesis.py",
    "scripts/diagnose_post_id341_target_gap_budget.py",
    "scripts/diagnose_post_id341_missing_target_signal_inventory.py",
    "scripts/diagnose_post_id341_q3s3_physio_light_onset_smoke.py",
    "scripts/diagnose_post_id341_q3s3_local_lane_closure.py",
    "scripts/diagnose_post_id341_q3s3_event_cadence_smoke.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_v2_design.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_metadata_smoke.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_resource_gate.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_model_smoke.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_lane_closure.py",
    "scripts/diagnose_post_id341_q3s3_negative_evidence_map.py",
    "scripts/diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py",
    "Makefile",
    "dataset-metadata.json",
    "reports/post_id341_upstream_rebuild_phase1_readiness_20260618.json",
    "reports/public_joint_distribution_risk_post_id341_20260618.json",
    "reports/joint_risk_candidate_rerank_post_id341_20260618.json",
    "reports/post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "reports/external_etri_prior_smoke_post_id341_20260618.json",
    "reports/combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "reports/post_id341_phase2_raw_source_execution_spec_20260618.json",
    "reports/post_id341_q2q3s3_source_hypothesis_matrix_20260618.json",
    "reports/post_id341_q2q3s3_local_fallback_chain_contract_20260618.json",
    "reports/post_id341_q3s3_remaining_route_inventory_20260618.json",
    "reports/phase2_raw_multimodal_execution_plan_post_id341_20260618.json",
    "reports/phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json",
    "reports/phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "reports/phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "reports/phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "reports/post_id341_usage_semantic_burst_markov_design_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_spec_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_features_v1_schema.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "reports/post_id341_self_supervised_subject_state_embedding_design_20260619.json",
    "reports/post_id341_local_source_closure_synthesis_20260618.json",
    "reports/post_id341_target_gap_budget_20260618.json",
    "reports/post_id341_missing_target_signal_inventory_20260618.json",
    "reports/post_id341_q3s3_physio_light_onset_smoke_20260618.json",
    "reports/post_id341_q3s3_local_lane_closure_20260618.json",
    "reports/post_id341_q3s3_event_cadence_smoke_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_v2_design_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_metadata_smoke_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_resource_gate_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_model_smoke_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_lane_closure_20260618.json",
    "reports/post_id341_q3s3_negative_evidence_map_20260618.json",
    "reports/post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json",
    "reports/combined_sequence_sleep_evening_gate_post_id341_20260618.json",
    "reports/place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "reports/routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "reports/sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "reports/id340_validation_split_manifest_20260618.json",
    "data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet",
    "data/external/etri_lifelog/",
    "artifacts/base_predictions/",
    "submissions/",
    "submissions/submission_log.csv",
    "submissions/submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv",
    "submissions/submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv",
    "submissions/submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv",
    "submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv",
]


def run_capture(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
    }


def zip_entry_check(zip_path: Path) -> dict[str, Any]:
    if not zip_path.exists():
        return {"exists": False, "required_entries": dict.fromkeys(REQUIRED_ENTRIES, False)}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    required_entries: dict[str, bool] = {}
    for entry in REQUIRED_ENTRIES:
        required_entries[entry] = (
            any(name.startswith(entry) for name in names) if entry.endswith("/") else entry in names
        )
    return {
        "exists": True,
        "size_bytes": zip_path.stat().st_size,
        "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 3),
        "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "required_entries": required_entries,
        "all_required_present": all(required_entries.values()),
    }


def run_packaged_readiness(zip_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dacon_upstream_zip_selftest_") as tmp:
        tmp_path = Path(tmp)
        project_dir = tmp_path / "project"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(project_dir)
        output_arg = "reports/selftest_post_id341_upstream_rebuild_phase1_readiness_20260618.json"
        output_path = project_dir / output_arg
        proc = run_capture(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "scripts/diagnose_post_id341_upstream_rebuild_phase1_readiness.py",
                "--output",
                output_arg,
            ],
            cwd=project_dir,
        )
        decision = None
        if output_path.exists():
            decision = json.loads(output_path.read_text(encoding="utf-8")).get("decision", {})
        return {
            "extracted_project_required_paths": {
                "scripts": (project_dir / "scripts").exists(),
                "reports": (project_dir / "reports").exists(),
                "base_predictions": (project_dir / "artifacts" / "base_predictions").exists(),
                "makefile": (project_dir / "Makefile").exists(),
                "dataset_metadata": (project_dir / "dataset-metadata.json").exists(),
                "submissions_dir": (project_dir / "submissions").exists(),
                "submission_log": (project_dir / "submissions" / "submission_log.csv").exists(),
                "anchor_submission_id178": (
                    project_dir
                    / "submissions"
                    / "submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv"
                ).exists(),
                "anchor_submission_id287": (
                    project_dir
                    / "submissions"
                    / "submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv"
                ).exists(),
                "anchor_submission_id329": (
                    project_dir
                    / "submissions"
                    / "submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv"
                ).exists(),
                "anchor_submission_id341": (
                    project_dir
                    / "submissions"
                    / "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"
                ).exists(),
            },
            "readiness_command": proc,
            "readiness_decision": decision,
            "selftest_passed": bool(
                proc["passed"]
                and decision
                and decision.get("phase1_ready") is True
                and decision.get("local_new_source_stack_closed") is True
                and decision.get("create_submission_artifact") is False
                and decision.get("submit_now") is False
                and decision.get("push_kaggle_now") is False
                and decision.get("run_heavy_rebuild_now") is False
            ),
        }


def build_report(keep_build: bool) -> dict[str, Any]:
    build = run_capture(["make", "build-kaggle-dataset"], cwd=ROOT)
    zip_check = zip_entry_check(SOURCE_ZIP)
    packaged_readiness = run_packaged_readiness(SOURCE_ZIP) if zip_check.get("all_required_present") else None
    build_removed = False
    package_dir = SOURCE_ZIP.parent
    if not keep_build and package_dir.exists():
        shutil.rmtree(package_dir)
        build_removed = not package_dir.exists()
    selftest_passed = bool(build["passed"] and zip_check.get("all_required_present") and packaged_readiness)
    if packaged_readiness:
        selftest_passed = selftest_passed and packaged_readiness["selftest_passed"]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "kaggle_source_zip_selftest_post_id341",
        "purpose": "Verify the local Kaggle source.zip is self-contained for the post-ID341 no-submit upstream runner.",
        "build_command": build,
        "source_zip": zip_check,
        "packaged_readiness": packaged_readiness,
        "transient_package_dir": str(package_dir.relative_to(ROOT)),
        "transient_build_removed": build_removed,
        "decision": {
            "selftest_passed": selftest_passed,
            "ready_for_explicit_kaggle_approval": selftest_passed,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "If the user explicitly approves, run the guarded sequence: "
                "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset; "
                "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files; "
                "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files; "
                "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner; "
                "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild; "
                "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output; then "
                "make kaggle-review-upstream-rebuild-output. Otherwise keep working locally without creating ID342."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    source_zip = report["source_zip"]
    readiness = report["packaged_readiness"] or {}
    lines = [
        "# Kaggle source.zip self-test — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It rebuilds and extracts the local Kaggle source zip, then removes only the transient package directory. It does not push/run Kaggle or submit to DACON.",
        "",
        "## Result",
        "",
        f"- Self-test passed: `{decision['selftest_passed']}`",
        f"- Ready for explicit Kaggle approval: `{decision['ready_for_explicit_kaggle_approval']}`",
        f"- Build command passed: `{report['build_command']['passed']}`",
        f"- Source zip entries present: `{source_zip.get('all_required_present')}`",
        f"- Source zip size MB: `{source_zip.get('size_mb')}`",
        f"- Source zip SHA256: `{source_zip.get('sha256')}`",
        f"- Packaged readiness passed: `{readiness.get('selftest_passed')}`",
        f"- Transient build removed: `{report['transient_build_removed']}`",
        "",
        "## Required entries",
        "",
        "| entry | present |",
        "|---|---|",
    ]
    for entry, present in source_zip.get("required_entries", {}).items():
        lines.append(f"| `{entry}` | `{present}` |")
    lines.extend(
        [
            "",
            "## Packaged readiness decision",
            "",
            "```json",
            json.dumps(readiness.get("readiness_decision"), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Decision",
            "",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "kaggle_source_zip_selftest_post_id341_20260618.json",
    )
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(keep_build=args.keep_build)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
