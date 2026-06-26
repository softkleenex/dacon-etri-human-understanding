#!/usr/bin/env python3
"""Final local prelaunch readiness gate for the post-ID341 Kaggle no-submit run.

This report-only diagnostic aggregates the approval note, execution packet,
source.zip self-test, runner/intake contract, output-intake gate, material
source remote coverage contract, kernel
metadata, public/local gap diagnosis, data/source coverage audit,
missing-target inventory, Q2/Q3/S3 source-hypothesis matrix, local-smoke
freshness, fail-closed remote dry-run, and Makefile dry-run commands.
It decides whether the repo is ready to wait for explicit user approval of the
exact Kaggle command sequence.

It never pushes/runs Kaggle, trains models, creates predictions, creates a
submission CSV, edits the submission log, or submits to DACON.
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
EXPECTED_DATASET_SOURCES = [
    "softkleenex/dacon-etri-human-base",
    "softkleenex/dacon-etri-upstream-patch-post-id341",
]

APPROVAL_NOTE = REPORTS_DIR / "post_id341_kaggle_launch_approval_note_20260618.json"
EXECUTION_PACKET = REPORTS_DIR / "kaggle_upstream_execution_packet_post_id341_20260618.json"
SOURCE_SELFTEST = REPORTS_DIR / "kaggle_source_zip_selftest_post_id341_20260618.json"
RUNNER_INTAKE_CONTRACT = REPORTS_DIR / "post_id341_kaggle_runner_intake_contract_20260618.json"
OUTPUT_INTAKE_GATE = REPORTS_DIR / "post_id341_kaggle_output_intake_gate_20260618.json"
POST_PULL_REVIEW_CHAIN_CONTRACT = REPORTS_DIR / "post_id341_post_pull_review_chain_contract_20260618.json"
RUNNER_SCORE_RELEVANCE_CONTRACT = REPORTS_DIR / "post_id341_runner_score_relevance_contract_20260618.json"
MATERIAL_SOURCE_REMOTE_COVERAGE_CONTRACT = (
    REPORTS_DIR / "post_id341_material_source_remote_coverage_contract_20260619.json"
)
GUARDED_REMOTE_CONTRACT_AUDIT = REPORTS_DIR / "kaggle_guarded_remote_contract_audit_20260618.json"
REMOTE_FAIL_CLOSED_DRYRUN_AUDIT = REPORTS_DIR / "kaggle_remote_fail_closed_dryrun_audit_20260618.json"
PUBLIC_LOCAL_GAP_DIAGNOSIS = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"
DATA_SOURCE_COVERAGE_AUDIT = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"
MISSING_TARGET_SIGNAL_INVENTORY = REPORTS_DIR / "post_id341_missing_target_signal_inventory_20260618.json"
Q2Q3S3_SOURCE_HYPOTHESIS_MATRIX = REPORTS_DIR / "post_id341_q2q3s3_source_hypothesis_matrix_20260618.json"
Q2Q3S3_LOCAL_FALLBACK_CHAIN = REPORTS_DIR / "post_id341_q2q3s3_local_fallback_chain_contract_20260618.json"
Q3S3_LOCAL_LANE_CLOSURE = REPORTS_DIR / "post_id341_q3s3_local_lane_closure_20260618.json"
Q3S3_SEMANTIC_ROUTINE_LANE_CLOSURE = REPORTS_DIR / "post_id341_q3s3_semantic_routine_lane_closure_20260618.json"
Q3S3_REMAINING_ROUTE_INVENTORY = REPORTS_DIR / "post_id341_q3s3_remaining_route_inventory_20260618.json"
USAGE_SEMANTIC_BURST_MARKOV_VALIDATION_STOP = (
    REPORTS_DIR / "post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
)
SELF_SUPERVISED_SUBJECT_STATE_DESIGN = (
    REPORTS_DIR / "post_id341_self_supervised_subject_state_embedding_design_20260619.json"
)
LOCAL_SMOKE_FRESHNESS_GATE = REPORTS_DIR / "post_id341_local_smoke_freshness_gate_20260618.json"
REMOTE_RESULT_SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_post_id341_kaggle_remote_results.py"
REMOTE_NEXT_ACTION_SCRIPT = ROOT / "scripts" / "decide_post_id341_remote_next_action.py"
KERNEL_METADATA = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "kernel-metadata.json"
NOTEBOOK = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "Upstream_Rebuild_Post_ID341_Runner.ipynb"

EXACT_COMMAND_SEQUENCE = [
    "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild",
    "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output",
    "make kaggle-review-upstream-rebuild-output",
]
LOCAL_PRE_APPROVAL_COMMAND_SEQUENCE = [
    "make kaggle-approval-contract-test",
    "make kaggle-upstream-prelaunch-local",
]
DRY_RUN_COMMANDS = [
    ["make", "-n", "kaggle-approval-contract-test"],
    ["make", "-n", "kaggle-upstream-prelaunch-local"],
    ["make", "-n", "ALLOW_KAGGLE_REMOTE=true", "kaggle-push-upstream-patch-dataset"],
    ["make", "-n", "ALLOW_KAGGLE_REMOTE=true", "kaggle-check-base-dataset-files"],
    ["make", "-n", "ALLOW_KAGGLE_REMOTE=true", "kaggle-check-upstream-patch-dataset-files"],
    ["make", "-n", "ALLOW_KAGGLE_REMOTE=true", "kaggle-push-upstream-rebuild-runner"],
    ["make", "-n", "ALLOW_KAGGLE_REMOTE=true", "kaggle-kernel-status-upstream-rebuild"],
    ["make", "-n", "ALLOW_KAGGLE_REMOTE=true", "kaggle-wait-and-pull-upstream-rebuild-output"],
    ["make", "-n", "kaggle-review-upstream-rebuild-output"],
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": None, "error": None}
    try:
        payload = load_json(path)
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": True,
            "bytes": path.stat().st_size,
            "decision": payload.get("decision", {}),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_capture(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def metadata_status() -> dict[str, Any]:
    status = report_file_status(KERNEL_METADATA)
    if not status["parse_ok"]:
        return status
    payload = load_json(KERNEL_METADATA)
    status.update(
        {
            "id": payload.get("id"),
            "code_file": payload.get("code_file"),
            "enable_gpu": payload.get("enable_gpu"),
            "enable_tpu": payload.get("enable_tpu"),
            "dataset_sources": payload.get("dataset_sources"),
            "is_private": payload.get("is_private"),
        }
    )
    status["safe_for_cpu_no_submit_runner"] = bool(
        payload.get("id") == "softkleenex/dacon-etri-upstream-rebuild-post-id341"
        and payload.get("code_file") == NOTEBOOK.name
        and payload.get("enable_gpu") is False
        and payload.get("enable_tpu") is False
        and payload.get("dataset_sources") == EXPECTED_DATASET_SOURCES
    )
    return status


def build_report(run_dry_run: bool) -> dict[str, Any]:
    approval = report_file_status(APPROVAL_NOTE)
    packet = report_file_status(EXECUTION_PACKET)
    selftest = report_file_status(SOURCE_SELFTEST)
    contract = report_file_status(RUNNER_INTAKE_CONTRACT)
    output_gate = report_file_status(OUTPUT_INTAKE_GATE)
    post_pull_review_chain = report_file_status(POST_PULL_REVIEW_CHAIN_CONTRACT)
    runner_score_relevance = report_file_status(RUNNER_SCORE_RELEVANCE_CONTRACT)
    material_source_remote_coverage = report_file_status(MATERIAL_SOURCE_REMOTE_COVERAGE_CONTRACT)
    guarded_contract = report_file_status(GUARDED_REMOTE_CONTRACT_AUDIT)
    remote_fail_closed = report_file_status(REMOTE_FAIL_CLOSED_DRYRUN_AUDIT)
    public_gap = report_file_status(PUBLIC_LOCAL_GAP_DIAGNOSIS)
    data_coverage = report_file_status(DATA_SOURCE_COVERAGE_AUDIT)
    missing_target_inventory = report_file_status(MISSING_TARGET_SIGNAL_INVENTORY)
    q2q3s3_source_hypothesis_matrix = report_file_status(Q2Q3S3_SOURCE_HYPOTHESIS_MATRIX)
    q2q3s3_local_fallback_chain = report_file_status(Q2Q3S3_LOCAL_FALLBACK_CHAIN)
    q3s3_local_lane_closure = report_file_status(Q3S3_LOCAL_LANE_CLOSURE)
    q3s3_semantic_routine_lane_closure = report_file_status(Q3S3_SEMANTIC_ROUTINE_LANE_CLOSURE)
    q3s3_remaining_route_inventory = report_file_status(Q3S3_REMAINING_ROUTE_INVENTORY)
    usage_semantic_burst_markov_validation_stop = report_file_status(USAGE_SEMANTIC_BURST_MARKOV_VALIDATION_STOP)
    self_supervised_subject_state_design = report_file_status(SELF_SUPERVISED_SUBJECT_STATE_DESIGN)
    local_smoke_freshness = report_file_status(LOCAL_SMOKE_FRESHNESS_GATE)
    metadata = metadata_status()
    dry_runs = [run_capture(cmd) for cmd in DRY_RUN_COMMANDS] if run_dry_run else []
    contract_payload = load_json(RUNNER_INTAKE_CONTRACT) if contract.get("parse_ok") else {}
    q3s3_negative_map_handoff = contract_payload.get("q3s3_negative_map_handoff_contract", {})

    approval_ready = approval.get("decision", {}).get("approval_note_ready") is True
    explicit_required = approval.get("decision", {}).get("requires_explicit_user_approval") is True
    packet_ready = packet.get("decision", {}).get("ready_for_explicit_approval") is True
    selftest_ready = (
        selftest.get("decision", {}).get("selftest_passed") is True
        and selftest.get("decision", {}).get("ready_for_explicit_kaggle_approval") is True
    )
    contract_pass = contract.get("decision", {}).get("runner_intake_contract_pass") is True
    runner_q3s3_negative_map_handoff_ready = bool(
        q3s3_negative_map_handoff.get("q3s3_negative_map_handoff_pass") is True
        and q3s3_negative_map_handoff.get("negative_map_name_present") is True
        and q3s3_negative_map_handoff.get("fold_nested_sibling_name_present") is True
        and q3s3_negative_map_handoff.get("remaining_inventory_name_present") is True
        and q3s3_negative_map_handoff.get("negative_map_declared_before_inventory") is True
        and q3s3_negative_map_handoff.get("fold_nested_sibling_declared_between_negative_and_inventory") is True
        and q3s3_negative_map_handoff.get("negative_map_generation_command_present") is True
        and q3s3_negative_map_handoff.get("fold_nested_sibling_generation_command_present") is True
        and q3s3_negative_map_handoff.get("remaining_inventory_uses_negative_map_arg") is True
        and q3s3_negative_map_handoff.get("remaining_inventory_uses_fold_nested_sibling_arg") is True
        and q3s3_negative_map_handoff.get("negative_map_staged_in_report_roots") is True
        and q3s3_negative_map_handoff.get("fold_nested_sibling_staged_in_report_roots") is True
        and q3s3_negative_map_handoff.get("negative_map_asserts_eight_closed_lanes") is True
    )
    guarded_remote_contract_pass = guarded_contract.get("decision", {}).get("guarded_remote_contract_pass") is True
    remote_fail_closed_dryrun_pass = bool(
        remote_fail_closed.get("decision", {}).get("remote_fail_closed_dryrun_pass") is True
        and remote_fail_closed.get("decision", {}).get("execute_kaggle_now") is False
        and remote_fail_closed.get("decision", {}).get("push_kaggle_now") is False
        and remote_fail_closed.get("decision", {}).get("run_kaggle_now") is False
        and remote_fail_closed.get("decision", {}).get("create_submission_artifact") is False
        and remote_fail_closed.get("decision", {}).get("submit_now") is False
    )
    public_local_gap_ready = public_gap.get("decision", {}).get("public_local_gap_diagnosis_ready") is True
    public_local_gap_blocks_local_only = bool(
        public_gap.get("decision", {}).get("any_watch_candidate_passes_public_058_recent_floor_proxy") is False
        and public_gap.get("decision", {}).get("any_watch_candidate_passes_public_058_recent_stress_proxy") is False
    )
    data_source_coverage_ready = data_coverage.get("decision", {}).get("data_source_coverage_ready") is True
    data_source_external_concat_blocked = (
        data_coverage.get("decision", {}).get("external_direct_concat_allowed") is False
    )
    data_source_local_stack_closed = (
        data_coverage.get("decision", {}).get("local_source_stack_closed_for_submission_artifact") is True
    )
    missing_target_inventory_ready = bool(
        missing_target_inventory.get("decision", {}).get("local_q2_q3_s3_promotable") is False
        and missing_target_inventory.get("decision", {}).get("focus_missing_locally") == ["Q2", "Q3", "S3"]
        and missing_target_inventory.get("decision", {}).get("focus_has_upstream_backbone_signal") == ["Q2", "Q3", "S3"]
        and missing_target_inventory.get("decision", {}).get("external_direct_concat_allowed") is False
        and missing_target_inventory.get("decision", {}).get("create_submission_artifact") is False
        and missing_target_inventory.get("decision", {}).get("submit_now") is False
    )
    q2q3s3_source_hypothesis_matrix_ready = bool(
        q2q3s3_source_hypothesis_matrix.get("decision", {}).get("q2q3s3_source_hypothesis_matrix_ready") is True
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("all_focus_targets_missing_locally") is True
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("all_focus_targets_have_historical_id226_signal")
        is True
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("local_q2q3s3_artifact_promotable") is False
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("external_direct_concat_allowed") is False
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("create_submission_artifact") is False
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("submit_now") is False
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("push_kaggle_now") is False
        and q2q3s3_source_hypothesis_matrix.get("decision", {}).get("run_kaggle_now") is False
    )
    q3s3_local_lane_closure_ready = bool(
        q3s3_local_lane_closure.get("decision", {}).get("local_q2_component_keep_as_context") is True
        and q3s3_local_lane_closure.get("decision", {}).get("q3_s3_uncovered_after_local_fallback") == ["Q3", "S3"]
        and q3s3_local_lane_closure.get("decision", {}).get("local_fallback_closed_for_submission_artifact") is True
        and q3s3_local_lane_closure.get("decision", {}).get("local_artifact_allowed") is False
        and q3s3_local_lane_closure.get("decision", {}).get("primary_next_score_relevant_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and q3s3_local_lane_closure.get("decision", {}).get("requires_explicit_kaggle_approval") is True
        and q3s3_local_lane_closure.get("decision", {}).get("create_submission_artifact") is False
        and q3s3_local_lane_closure.get("decision", {}).get("submit_now") is False
    )
    q3s3_semantic_routine_lane_closure_ready = bool(
        q3s3_semantic_routine_lane_closure.get("decision", {}).get("semantic_routine_lane_closure_passed") is True
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("q3_s3_still_uncovered") == ["Q3", "S3"]
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("local_semantic_routine_artifact_allowed")
        is False
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("public_058_evidence") is False
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("primary_next_score_relevant_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("requires_explicit_kaggle_approval") is True
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("create_submission_artifact") is False
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("submit_now") is False
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("push_kaggle_now") is False
        and q3s3_semantic_routine_lane_closure.get("decision", {}).get("run_kaggle_now") is False
    )
    q2q3s3_local_fallback_chain_ready = bool(
        q2q3s3_local_fallback_chain.get("decision", {}).get("local_fallback_chain_contract_pass") is True
        and q2q3s3_local_fallback_chain.get("decision", {}).get("primary_score_relevant_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and q2q3s3_local_fallback_chain.get("decision", {}).get("selected_local_context_targets") == ["Q2"]
        and q2q3s3_local_fallback_chain.get("decision", {}).get("q3_s3_still_uncovered") is True
        and q2q3s3_local_fallback_chain.get("decision", {}).get("local_q2_lead_context_only") is True
        and q2q3s3_local_fallback_chain.get("decision", {}).get("public_058_evidence") is False
        and q2q3s3_local_fallback_chain.get("decision", {}).get("local_artifact_allowed") is False
        and q2q3s3_local_fallback_chain.get("decision", {}).get("requires_explicit_kaggle_approval") is True
        and q2q3s3_local_fallback_chain.get("decision", {}).get("create_submission_artifact") is False
        and q2q3s3_local_fallback_chain.get("decision", {}).get("submit_now") is False
        and q2q3s3_local_fallback_chain.get("decision", {}).get("push_kaggle_now") is False
        and q2q3s3_local_fallback_chain.get("decision", {}).get("run_kaggle_now") is False
    )
    q3s3_remaining_route_inventory_ready = bool(
        q3s3_remaining_route_inventory.get("decision", {}).get("q3s3_remaining_route_inventory_ready") is True
        and q3s3_remaining_route_inventory.get("decision", {}).get("q3_s3_still_uncovered") is True
        and q3s3_remaining_route_inventory.get("decision", {}).get("fold_nested_sibling_bridge_closed") is True
        and q3s3_remaining_route_inventory.get("decision", {}).get("local_no_approval_score_route_exists") is False
        and q3s3_remaining_route_inventory.get("decision", {}).get("only_score_relevant_open_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and q3s3_remaining_route_inventory.get("decision", {}).get("requires_explicit_kaggle_approval") is True
        and q3s3_remaining_route_inventory.get("decision", {}).get("approval_phrase")
        == "Kaggle upstream rebuild no-submit 실행 승인"
        and q3s3_remaining_route_inventory.get("decision", {}).get("public_058_evidence") is False
        and q3s3_remaining_route_inventory.get("decision", {}).get("create_submission_artifact") is False
        and q3s3_remaining_route_inventory.get("decision", {}).get("submit_now") is False
        and q3s3_remaining_route_inventory.get("decision", {}).get("push_kaggle_now") is False
        and q3s3_remaining_route_inventory.get("decision", {}).get("run_kaggle_now") is False
    )
    usage_semantic_burst_markov_validation_stop_ready = bool(
        usage_semantic_burst_markov_validation_stop.get("decision", {}).get("validation_stop_passed") is True
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("public_058_evidence") is False
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("score_route_opened") is False
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("train_model_now") is False
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("create_submission_artifact") is False
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("submit_now") is False
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("push_kaggle_now") is False
        and usage_semantic_burst_markov_validation_stop.get("decision", {}).get("run_kaggle_now") is False
    )
    self_supervised_subject_state_design_ready = bool(
        self_supervised_subject_state_design.get("decision", {}).get("design_ready") is True
        and self_supervised_subject_state_design.get("decision", {}).get("source_family")
        == "self_supervised_subject_state_embedding_v1"
        and self_supervised_subject_state_design.get("decision", {}).get("local_no_approval_allowed") is False
        and self_supervised_subject_state_design.get("decision", {}).get("requires_explicit_heavy_compute_approval")
        is True
        and self_supervised_subject_state_design.get("decision", {}).get("public_058_evidence") is False
        and self_supervised_subject_state_design.get("decision", {}).get("score_route_opened") is False
        and self_supervised_subject_state_design.get("decision", {}).get("feature_matrix_allowed_now") is False
        and self_supervised_subject_state_design.get("decision", {}).get("train_model_now") is False
        and self_supervised_subject_state_design.get("decision", {}).get("create_submission_artifact") is False
        and self_supervised_subject_state_design.get("decision", {}).get("submit_now") is False
        and self_supervised_subject_state_design.get("decision", {}).get("push_kaggle_now") is False
        and self_supervised_subject_state_design.get("decision", {}).get("run_kaggle_now") is False
    )
    output_pending_clean = bool(
        output_gate.get("decision", {}).get("output_pending") is True
        and output_gate.get("decision", {}).get("artifact_contract_review_allowed") is False
        and output_gate.get("decision", {}).get("unexpected_submission_like_files_present") is False
        and output_gate.get("decision", {}).get("remote_forbidden_actions_present") is False
    )
    post_pull_review_chain_ready = bool(
        post_pull_review_chain.get("decision", {}).get("post_pull_review_chain_contract_pass") is True
        and post_pull_review_chain.get("decision", {}).get("output_pending") is True
        and post_pull_review_chain.get("decision", {}).get("artifact_contract_review_allowed") is False
        and post_pull_review_chain.get("decision", {}).get("public_058_reviewable") is False
        and post_pull_review_chain.get("decision", {}).get("create_submission_artifact") is False
        and post_pull_review_chain.get("decision", {}).get("submit_now") is False
        and post_pull_review_chain.get("decision", {}).get("push_kaggle_now") is False
        and post_pull_review_chain.get("decision", {}).get("run_kaggle_now") is False
    )
    runner_score_relevance_ready = bool(
        runner_score_relevance.get("decision", {}).get("runner_score_relevance_contract_pass") is True
        and runner_score_relevance.get("decision", {}).get("score_relevant_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and runner_score_relevance.get("decision", {}).get("missing_targets") == ["Q2", "Q3", "S3"]
        and runner_score_relevance.get("decision", {}).get("create_submission_artifact") is False
        and runner_score_relevance.get("decision", {}).get("submit_now") is False
        and runner_score_relevance.get("decision", {}).get("push_kaggle_now") is False
        and runner_score_relevance.get("decision", {}).get("run_kaggle_now") is False
    )
    material_source_remote_coverage_ready = bool(
        material_source_remote_coverage.get("decision", {}).get("material_source_remote_coverage_contract_pass") is True
        and material_source_remote_coverage.get("decision", {}).get("all_queue_items_covered") is True
        and material_source_remote_coverage.get("decision", {}).get("no_forbidden_actions") is True
        and material_source_remote_coverage.get("decision", {}).get("only_score_relevant_open_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and material_source_remote_coverage.get("decision", {}).get("requires_explicit_kaggle_approval") is True
        and material_source_remote_coverage.get("decision", {}).get("approval_phrase")
        == "Kaggle upstream rebuild no-submit 실행 승인"
        and material_source_remote_coverage.get("decision", {}).get("public_058_evidence") is False
        and material_source_remote_coverage.get("decision", {}).get("create_submission_artifact") is False
        and material_source_remote_coverage.get("decision", {}).get("submit_now") is False
        and material_source_remote_coverage.get("decision", {}).get("push_kaggle_now") is False
        and material_source_remote_coverage.get("decision", {}).get("run_kaggle_now") is False
    )
    local_smoke_fresh = bool(
        local_smoke_freshness.get("decision", {}).get("local_runner_smoke_fresh") is True
        and local_smoke_freshness.get("decision", {}).get("requires_local_runner_smoke_rerun") is False
        and local_smoke_freshness.get("decision", {}).get("create_submission_artifact") is False
        and local_smoke_freshness.get("decision", {}).get("submit_now") is False
        and local_smoke_freshness.get("decision", {}).get("push_kaggle_now") is False
        and local_smoke_freshness.get("decision", {}).get("run_kaggle_now") is False
    )
    metadata_safe = metadata.get("safe_for_cpu_no_submit_runner") is True
    remote_summary_script_exists = REMOTE_RESULT_SUMMARY_SCRIPT.exists()
    remote_next_action_script_exists = REMOTE_NEXT_ACTION_SCRIPT.exists()
    dry_runs_pass = all(row["passed"] for row in dry_runs) if run_dry_run else None
    no_forbidden_local_decision = all(
        (row.get("decision") or {}).get(key) is not True
        for row in [
            approval,
            packet,
            selftest,
            contract,
            output_gate,
            post_pull_review_chain,
            runner_score_relevance,
            material_source_remote_coverage,
            remote_fail_closed,
            public_gap,
            data_coverage,
            missing_target_inventory,
            q2q3s3_source_hypothesis_matrix,
            q2q3s3_local_fallback_chain,
            q3s3_local_lane_closure,
            q3s3_semantic_routine_lane_closure,
            q3s3_remaining_route_inventory,
            usage_semantic_burst_markov_validation_stop,
            self_supervised_subject_state_design,
            local_smoke_freshness,
        ]
        for key in ["push_kaggle_now", "run_kaggle_now", "create_submission_artifact", "submit_now"]
    )
    prelaunch_ready = bool(
        approval_ready
        and explicit_required
        and packet_ready
        and selftest_ready
        and contract_pass
        and runner_q3s3_negative_map_handoff_ready
        and guarded_remote_contract_pass
        and remote_fail_closed_dryrun_pass
        and public_local_gap_ready
        and public_local_gap_blocks_local_only
        and data_source_coverage_ready
        and data_source_external_concat_blocked
        and data_source_local_stack_closed
        and missing_target_inventory_ready
        and q2q3s3_source_hypothesis_matrix_ready
        and q2q3s3_local_fallback_chain_ready
        and q3s3_local_lane_closure_ready
        and q3s3_semantic_routine_lane_closure_ready
        and q3s3_remaining_route_inventory_ready
        and usage_semantic_burst_markov_validation_stop_ready
        and self_supervised_subject_state_design_ready
        and local_smoke_fresh
        and output_pending_clean
        and post_pull_review_chain_ready
        and runner_score_relevance_ready
        and material_source_remote_coverage_ready
        and metadata_safe
        and remote_summary_script_exists
        and remote_next_action_script_exists
        and no_forbidden_local_decision
        and (dry_runs_pass is not False)
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_kaggle_prelaunch_readiness_gate",
        "purpose": (
            "Aggregate all local no-submit readiness evidence before an explicit-approved Kaggle/Linux parity run. "
            "This gate does not authorize remote execution by itself."
        ),
        "inputs": {
            "approval_note": approval,
            "execution_packet": packet,
            "source_zip_selftest": selftest,
            "runner_intake_contract": contract,
            "runner_q3s3_negative_map_handoff_contract": q3s3_negative_map_handoff,
            "guarded_remote_contract": guarded_contract,
            "remote_fail_closed_dryrun_audit": remote_fail_closed,
            "public_local_gap_diagnosis": public_gap,
            "data_source_coverage_audit": data_coverage,
            "missing_target_signal_inventory": missing_target_inventory,
            "q2q3s3_source_hypothesis_matrix": q2q3s3_source_hypothesis_matrix,
            "q2q3s3_local_fallback_chain": q2q3s3_local_fallback_chain,
            "q3s3_local_lane_closure": q3s3_local_lane_closure,
            "q3s3_semantic_routine_lane_closure": q3s3_semantic_routine_lane_closure,
            "q3s3_remaining_route_inventory": q3s3_remaining_route_inventory,
            "usage_semantic_burst_markov_validation_stop": usage_semantic_burst_markov_validation_stop,
            "self_supervised_subject_state_design": self_supervised_subject_state_design,
            "local_smoke_freshness_gate": local_smoke_freshness,
            "output_intake_gate": output_gate,
            "post_pull_review_chain_contract": post_pull_review_chain,
            "runner_score_relevance_contract": runner_score_relevance,
            "material_source_remote_coverage_contract": material_source_remote_coverage,
            "kernel_metadata": metadata,
        },
        "dry_runs": dry_runs,
        "checks": {
            "approval_note_ready": approval_ready,
            "requires_explicit_user_approval": explicit_required,
            "execution_packet_ready": packet_ready,
            "source_zip_selftest_ready": selftest_ready,
            "runner_intake_contract_pass": contract_pass,
            "runner_q3s3_negative_map_handoff_ready": runner_q3s3_negative_map_handoff_ready,
            "guarded_remote_contract_pass": guarded_remote_contract_pass,
            "remote_fail_closed_dryrun_pass": remote_fail_closed_dryrun_pass,
            "public_local_gap_ready": public_local_gap_ready,
            "public_local_gap_blocks_local_cv_only_promotion": public_local_gap_blocks_local_only,
            "data_source_coverage_ready": data_source_coverage_ready,
            "data_source_external_concat_blocked": data_source_external_concat_blocked,
            "data_source_local_stack_closed": data_source_local_stack_closed,
            "missing_target_inventory_ready": missing_target_inventory_ready,
            "q2q3s3_source_hypothesis_matrix_ready": q2q3s3_source_hypothesis_matrix_ready,
            "q2q3s3_local_fallback_chain_ready": q2q3s3_local_fallback_chain_ready,
            "q3s3_local_lane_closure_ready": q3s3_local_lane_closure_ready,
            "q3s3_semantic_routine_lane_closure_ready": q3s3_semantic_routine_lane_closure_ready,
            "q3s3_remaining_route_inventory_ready": q3s3_remaining_route_inventory_ready,
            "usage_semantic_burst_markov_validation_stop_ready": usage_semantic_burst_markov_validation_stop_ready,
            "self_supervised_subject_state_design_ready": self_supervised_subject_state_design_ready,
            "local_runner_smoke_fresh": local_smoke_fresh,
            "output_intake_pending_and_clean": output_pending_clean,
            "post_pull_review_chain_ready": post_pull_review_chain_ready,
            "runner_score_relevance_ready": runner_score_relevance_ready,
            "material_source_remote_coverage_ready": material_source_remote_coverage_ready,
            "kernel_metadata_safe": metadata_safe,
            "remote_result_summary_script_exists": remote_summary_script_exists,
            "remote_next_action_script_exists": remote_next_action_script_exists,
            "dry_runs_pass": dry_runs_pass,
            "no_forbidden_local_decision": no_forbidden_local_decision,
        },
        "exact_command_sequence_after_explicit_approval": EXACT_COMMAND_SEQUENCE,
        "local_pre_approval_command_sequence": LOCAL_PRE_APPROVAL_COMMAND_SEQUENCE,
        "decision": {
            "prelaunch_ready_for_explicit_approval": prelaunch_ready,
            "requires_explicit_user_approval": True,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "artifact_contract_review_allowed": False,
            "next_action": (
                "Await explicit approval of the exact command sequence, then run the no-submit Kaggle/Linux parity path."
                if prelaunch_ready
                else "Fix failed local prelaunch checks before asking for or consuming Kaggle approval."
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
        "# Post-ID341 Kaggle prelaunch readiness gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is a local no-submit gate. It aggregates the approval note, source.zip self-test, runner/intake contract, Q3/S3 negative-map handoff, output-intake gate, post-pull review-chain contract, runner score-relevance contract, material source remote coverage contract, public/local gap diagnosis, data/source coverage audit, missing-target inventory, Q2/Q3/S3 source-hypothesis matrix, Q2/Q3/S3 local fallback chain, Q3/S3 local-lane closure, Q3/S3 semantic-routine lane closure, Q3/S3 remaining-route inventory, usage semantic burst/Markov validation-stop, self-supervised subject-state design gate, local-smoke freshness, fail-closed remote dry-run, kernel metadata, and Makefile dry-runs.",
        "",
        "## Decision",
        "",
        f"- Prelaunch ready for explicit approval: `{decision['prelaunch_ready_for_explicit_approval']}`",
        f"- Requires explicit user approval: `{decision['requires_explicit_user_approval']}`",
        f"- Execute Kaggle now: `{decision['execute_kaggle_now']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Artifact-contract review allowed: `{decision['artifact_contract_review_allowed']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Local pre-approval command sequence", ""])
    for index, command in enumerate(report["local_pre_approval_command_sequence"], start=1):
        lines.append(f"{index}. `{command}`")
    lines.extend(["", "## Exact command sequence after explicit approval", ""])
    for index, command in enumerate(report["exact_command_sequence_after_explicit_approval"], start=1):
        lines.append(f"{index}. `{command}`")
    lines.extend(["", "## Dry-run commands", ""])
    for row in report["dry_runs"]:
        lines.append(f"- `{row['cmd']}` → passed `{row['passed']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json",
    )
    parser.add_argument("--skip-dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(run_dry_run=not args.skip_dry_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
