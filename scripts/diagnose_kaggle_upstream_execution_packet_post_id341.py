#!/usr/bin/env python3
"""Prepare a no-submit Kaggle execution packet for post-ID341 upstream runner.

This is a local, report-only readiness packet. It does not push to Kaggle, run a
kernel, create a DACON submission, create a submission CSV, or modify the
submission log. It records the exact command sequence that requires explicit user
approval, verifies that the local source.zip contains the reports required by the
upstream runner, and checks the runner pre-flight contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

KAGGLE_CLI = ROOT / ".venv" / "bin" / "kaggle"
SOURCE_ZIP = ROOT / "build" / "kaggle_dataset" / "source.zip"
NOTEBOOK_DIR = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner"
KERNEL_METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"
READINESS_PATH = REPORTS_DIR / "post_id341_upstream_rebuild_phase1_readiness_20260618.json"
SOURCE_ZIP_SELFTEST_PATH = REPORTS_DIR / "kaggle_source_zip_selftest_post_id341_20260618.json"
REQUIRED_NOTEBOOK_SNIPPETS = [
    "diagnose_post_id341_transfer_gate_v2_with_joint_risk.py",
    "diagnose_external_etri_prior_smoke_post_id341.py",
    "diagnose_combined_sleep_evening_external_prior_gate_post_id341.py",
    "diagnose_post_id341_phase2_raw_source_execution_spec.py",
    "diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py",
    "diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py",
    "diagnose_post_id341_q3s3_negative_evidence_map.py",
    "diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py",
    "diagnose_post_id341_q3s3_remaining_route_inventory.py",
    "diagnose_phase2_raw_multimodal_execution_plan_post_id341.py",
    "diagnose_phase2_block_a_q1_s4_sleep_residual_bridge_post_id341.py",
    "diagnose_phase2_block_a_q1_proxy_gap_post_id341.py",
    "diagnose_phase2_block_b_s2_sleep_recovery_retry_post_id341.py",
    "diagnose_phase3_usage_temporal_microstructure_smoke_post_id341.py",
    "diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop.py",
    "diagnose_post_id341_self_supervised_subject_state_embedding_design.py",
    "remote_post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "remote_external_etri_prior_smoke_post_id341_20260618.json",
    "remote_combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "remote_phase2_block_A_q1_proxy_gap_20260618.json",
    "remote_phase2_block_B_s2_sleep_recovery_retry_20260618.json",
    "remote_phase3_usage_temporal_microstructure_smoke_20260618.json",
    "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json",
    "remote_post_id341_phase2_raw_source_execution_spec_20260618.json",
    "remote_post_id341_q2q3s3_source_hypothesis_matrix_20260618.json",
    "remote_post_id341_q2q3s3_local_fallback_chain_contract_20260618.json",
    "remote_post_id341_q3s3_negative_evidence_map_20260618.json",
    "remote_post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json",
    "remote_post_id341_q3s3_remaining_route_inventory_20260618.json",
    "remote_phase2_raw_multimodal_execution_plan_post_id341_20260618.json",
    "remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json",
    "post_id341_phase2_and_clean_backbone_no_submit_runner",
]

REQUIRED_SOURCE_ZIP_ENTRIES = [
    "scripts/diagnose_post_id341_upstream_rebuild_phase1_readiness.py",
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
    "reports/id340_validation_split_manifest_20260618.json",
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


def run_capture(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "passed": proc.returncode == 0,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_source_zip_selftest() -> dict[str, Any] | None:
    if not SOURCE_ZIP_SELFTEST_PATH.exists():
        return None
    report = load_json(SOURCE_ZIP_SELFTEST_PATH)
    required_entries = report.get("source_zip", {}).get("required_entries", {})
    current_required_entries = {}
    for entry in REQUIRED_SOURCE_ZIP_ENTRIES:
        if entry.endswith("/"):
            current_required_entries[entry] = any(
                str(key).startswith(entry) and value for key, value in required_entries.items()
            )
        else:
            current_required_entries[entry] = bool(required_entries.get(entry))
    return {
        "path": str(SOURCE_ZIP_SELFTEST_PATH.relative_to(ROOT)),
        "decision": report.get("decision", {}),
        "source_zip": report.get("source_zip", {}),
        "current_required_entries": current_required_entries,
        "current_required_entries_present": all(current_required_entries.values()),
        "transient_build_removed": report.get("transient_build_removed"),
    }


def source_zip_check() -> dict[str, Any]:
    if not SOURCE_ZIP.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "required_entries": dict.fromkeys(REQUIRED_SOURCE_ZIP_ENTRIES, False),
            "all_required_present": False,
        }
    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        names = set(zf.namelist())
    required_entries = {}
    for entry in REQUIRED_SOURCE_ZIP_ENTRIES:
        if entry.endswith("/"):
            required_entries[entry] = any(name.startswith(entry) for name in names)
        else:
            required_entries[entry] = entry in names
    return {
        "exists": True,
        "size_bytes": SOURCE_ZIP.stat().st_size,
        "size_mb": round(SOURCE_ZIP.stat().st_size / (1024 * 1024), 3),
        "sha256": hashlib.sha256(SOURCE_ZIP.read_bytes()).hexdigest(),
        "required_entries": required_entries,
        "all_required_present": all(required_entries.values()),
    }


def makefile_package_includes_reports() -> bool:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for line in makefile.splitlines():
        if "source.zip" in line and "zip -r" in line:
            return " reports " in f" {line} "
    return False


def notebook_contract_check() -> dict[str, Any]:
    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebook_paths:
        return {
            "notebook_exists": False,
            "path": None,
            "required_snippets": dict.fromkeys(REQUIRED_NOTEBOOK_SNIPPETS, False),
            "all_required_snippets_present": False,
        }
    notebook_path = notebook_paths[0]
    text = notebook_path.read_text(encoding="utf-8")
    snippets = {snippet: snippet in text for snippet in REQUIRED_NOTEBOOK_SNIPPETS}
    return {
        "notebook_exists": True,
        "path": str(notebook_path.relative_to(ROOT)),
        "required_snippets": snippets,
        "all_required_snippets_present": all(snippets.values()),
    }


def build_report(run_preflight: bool) -> dict[str, Any]:
    readiness = load_json(READINESS_PATH)
    kernel_metadata = load_json(KERNEL_METADATA_PATH)
    preflight = (
        run_capture(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "scripts/pre_flight_check.py",
                "--notebook_dir",
                "notebooks/kaggle_upstream_rebuild_runner",
                "--research",
            ]
        )
        if run_preflight
        else {"passed": None, "cmd": "not run"}
    )
    kaggle_version = run_capture([str(KAGGLE_CLI), "--version"]) if KAGGLE_CLI.exists() else {"passed": False}
    zip_check = source_zip_check()
    zip_selftest = latest_source_zip_selftest()
    notebook_contract = notebook_contract_check()
    zip_evidence_passed = bool(
        zip_check["all_required_present"]
        or (
            zip_selftest
            and zip_selftest["decision"].get("ready_for_explicit_kaggle_approval") is True
            and zip_selftest.get("current_required_entries_present") is True
        )
    )
    metadata_safe = bool(
        kernel_metadata.get("enable_gpu") is False
        and kernel_metadata.get("enable_tpu") is False
        and kernel_metadata.get("id") == "softkleenex/dacon-etri-upstream-rebuild-post-id341"
    )
    readiness_decision = readiness.get("decision", {})
    ready_for_approval = bool(
        preflight.get("passed") is True
        and zip_evidence_passed
        and notebook_contract["all_required_snippets_present"]
        and makefile_package_includes_reports()
        and metadata_safe
        and readiness_decision.get("phase1_ready") is True
        and readiness_decision.get("local_new_source_stack_closed") is True
        and readiness_decision.get("create_submission_artifact") is False
        and readiness_decision.get("submit_now") is False
        and readiness_decision.get("push_kaggle_now") is False
        and readiness_decision.get("run_heavy_rebuild_now") is False
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "kaggle_upstream_execution_packet_post_id341",
        "purpose": "Freeze the exact no-submit Kaggle runner execution packet after local new-source closure.",
        "kaggle_cli": {
            "path": str(KAGGLE_CLI.relative_to(ROOT)),
            "exists": KAGGLE_CLI.exists(),
            "version_check": kaggle_version,
            "credential_file_exists": (Path.home() / ".kaggle" / "kaggle.json").exists(),
        },
        "kernel_metadata": {
            "path": str(KERNEL_METADATA_PATH.relative_to(ROOT)),
            "metadata": kernel_metadata,
            "metadata_safe": metadata_safe,
        },
        "source_zip": zip_check,
        "source_zip_selftest": zip_selftest,
        "source_zip_evidence_passed": zip_evidence_passed,
        "notebook_contract": notebook_contract,
        "makefile_package_includes_reports": makefile_package_includes_reports(),
        "preflight": preflight,
        "readiness_decision": readiness_decision,
        "approval_required_commands": [
            {
                "step": 1,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset",
                "why": "Version the small patch dataset so Kaggle overlays refreshed scripts, reports, submissions, and post-ID341 artifacts.",
                "network": True,
                "creates_dacon_submission": False,
            },
            {
                "step": 2,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files",
                "why": "Verify the large base dataset is visible for raw/processed files.",
                "network": True,
                "creates_dacon_submission": False,
            },
            {
                "step": 3,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files",
                "why": "Verify the patch dataset is visible before pushing the runner.",
                "network": True,
                "creates_dacon_submission": False,
            },
            {
                "step": 4,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner",
                "why": "Push the CPU-only no-submit upstream runner notebook.",
                "network": True,
                "creates_dacon_submission": False,
            },
            {
                "step": 5,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild",
                "why": "Poll the CPU-only upstream runner status.",
                "network": True,
                "creates_dacon_submission": False,
            },
            {
                "step": 6,
                "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output",
                "why": "Pull report-only runner outputs after Kaggle finishes.",
                "network": True,
                "creates_dacon_submission": False,
            },
            {
                "step": 7,
                "command": "make kaggle-review-upstream-rebuild-output",
                "why": (
                    "Run local no-submit intake, remote summary, next-action decision, and the strict Public-0.58 "
                    "artifact-review threshold gate."
                ),
                "network": False,
                "creates_dacon_submission": False,
            },
        ],
        "stop_rules": [
            "Do not run any approval_required_command unless the user explicitly approves Kaggle push/run.",
            "Every Kaggle network approval command must include ALLOW_KAGGLE_REMOTE=true; Makefile network targets fail closed without it.",
            "Do not submit to DACON from this runner.",
            "Do not create a submission CSV, base prediction artifact, notebook history artifact, or submission-log row.",
            "Stop if pre-flight fails, kernel metadata enables GPU/TPU, source.zip misses required reports, or readiness phase1/local closure is false.",
            "Stop if the upstream notebook no longer regenerates and stages the latest closure reports.",
            (
                "Stop unless the Q3/S3 negative-map handoff is proven: the remote negative map is generated and "
                "staged, remaining-route inventory uses --negative-evidence-map, and eight closed local lanes are "
                "asserted."
            ),
            "After pulling Kaggle outputs, run intake, remote summary, next-action, and Public-0.58 threshold gates locally before adding any heavier rebuild cell.",
            "Only open a Public-0.58 artifact-contract review if the pulled reports are safe, beat ID178, and show min_proxy <= 0.58.",
            "If pulled reports only beat ID178 but remain above 0.58, record an improvement lead only; do not create ID342.",
        ],
        "decision": {
            "ready_for_explicit_approval": ready_for_approval,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Await explicit approval for the exact Kaggle command sequence, or continue designing a materially "
                "different local raw source without creating ID342."
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
    notebook_contract = report["notebook_contract"]
    lines = [
        "# Kaggle upstream execution packet — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It does not push to Kaggle, run a Kaggle kernel, create a submission artifact, or submit to DACON.",
        "",
        "## Readiness",
        "",
        f"- Ready for explicit approval: `{decision['ready_for_explicit_approval']}`",
        f"- Pre-flight passed: `{report['preflight'].get('passed')}`",
        f"- Source zip exists: `{source_zip['exists']}`",
        f"- Source zip evidence passed: `{report['source_zip_evidence_passed']}`",
        f"- Source zip size MB: `{source_zip.get('size_mb')}`",
        f"- Source zip required entries present: `{source_zip['all_required_present']}`",
        f"- Latest source.zip self-test: `{None if report['source_zip_selftest'] is None else report['source_zip_selftest']['decision'].get('selftest_passed')}`",
        f"- Latest self-test covers current required entries: `{None if report['source_zip_selftest'] is None else report['source_zip_selftest'].get('current_required_entries_present')}`",
        f"- Notebook latest-closure contract passed: `{notebook_contract['all_required_snippets_present']}`",
        f"- Makefile packages reports: `{report['makefile_package_includes_reports']}`",
        f"- Kernel metadata safe: `{report['kernel_metadata']['metadata_safe']}`",
        f"- Phase-1 ready: `{report['readiness_decision'].get('phase1_ready')}`",
        f"- Local new-source stack closed: `{report['readiness_decision'].get('local_new_source_stack_closed')}`",
        "",
        "## Required source.zip entries",
        "",
        "| entry | present |",
        "|---|---|",
    ]
    for entry, present in source_zip["required_entries"].items():
        lines.append(f"| `{entry}` | `{present}` |")
    lines.extend(
        [
            "",
            "## Required notebook snippets",
            "",
            "| snippet | present |",
            "|---|---|",
        ]
    )
    for snippet, present in notebook_contract["required_snippets"].items():
        lines.append(f"| `{snippet}` | `{present}` |")
    lines.extend(
        [
            "",
            "## Approval-required command sequence",
            "",
            "| step | command | why |",
            "|---:|---|---|",
        ]
    )
    for row in report["approval_required_commands"]:
        lines.append(f"| {row['step']} | `{row['command']}` | {row['why']} |")
    lines.extend(["", "## Stop rules", ""])
    lines.extend(f"- {rule}" for rule in report["stop_rules"])
    lines.extend(
        [
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
        default=REPORTS_DIR / "kaggle_upstream_execution_packet_post_id341_20260618.json",
    )
    parser.add_argument("--no-preflight", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(run_preflight=not args.no_preflight)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
