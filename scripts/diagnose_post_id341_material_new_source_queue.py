#!/usr/bin/env python3
"""Queue materially new post-ID341 source-family work.

This is a report-only diagnostic. It does not train a model, create test
predictions, create a submission CSV, modify the submission log, run Kaggle, or
submit to DACON. Its purpose is to prevent the no-approval workflow from looping
back into already-closed Q3/S3 lanes while preserving one safe local design step
and the explicit-approved Kaggle upstream rebuild route.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
SCORE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
SELF_SUPERVISED_DESIGN_REPORT = "post_id341_self_supervised_subject_state_embedding_design_20260619.json"

REQUIRED_CONTEXT_REPORTS = {
    "no_approval_frontier": "post_id341_no_approval_frontier_20260619.json",
    "operator_handoff": "post_id341_operator_handoff_contract_20260619.json",
    "q3s3_remaining_route_inventory": "post_id341_q3s3_remaining_route_inventory_20260618.json",
    "q3s3_negative_evidence_map": "post_id341_q3s3_negative_evidence_map_20260618.json",
    "target_gap_budget": "post_id341_target_gap_budget_20260618.json",
    "missing_target_inventory": "post_id341_missing_target_signal_inventory_20260618.json",
    "sleep_boundary_prepromotion": "sleep_boundary_prepromotion_gate_post_id341_20260618.json",
    "evening_load_recovery": "evening_load_recovery_smoke_post_id341_20260618.json",
    "combined_sleep_evening_external": "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "routine_stability_place_entropy": "routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "sensor_adherence_dropout": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "phase3_usage_temporal": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "phase3_ambience_context": "phase3_ambience_context_closure_post_id341_20260618.json",
    "external_public05917_residual": "post_id341_external_public05917_residual_audit_20260619.json",
    "gps_path_topology_model_smoke": "post_id341_gps_path_topology_model_smoke_20260619.json",
    "circadian_hr_light_phase_metadata": "post_id341_circadian_hr_light_phase_metadata_probe_20260619.json",
    "circadian_hr_light_phase_feature_spec": "post_id341_circadian_hr_light_phase_feature_spec_20260619.json",
    "circadian_hr_light_phase_feature_generation_dryrun": (
        "post_id341_circadian_hr_light_phase_feature_generation_dryrun_20260619.json"
    ),
    "circadian_hr_light_phase_feature_matrix_smoke_contract": (
        "post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract_20260619.json"
    ),
    "circadian_hr_light_phase_feature_matrix_schema": "post_id341_circadian_hr_light_phase_features_v1_schema.json",
    "circadian_hr_light_phase_model_smoke_contract": (
        "post_id341_circadian_hr_light_phase_model_smoke_contract_20260619.json"
    ),
    "circadian_hr_light_phase_model_smoke": "post_id341_circadian_hr_light_phase_model_smoke_20260619.json",
    "usage_semantic_burst_markov_design": "post_id341_usage_semantic_burst_markov_design_20260619.json",
    "usage_semantic_burst_markov_metadata_smoke": (
        "post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.json"
    ),
    "usage_semantic_burst_markov_feature_spec": "post_id341_usage_semantic_burst_markov_feature_spec_20260619.json",
    "usage_semantic_burst_markov_feature_generation_dryrun": (
        "post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.json"
    ),
    "usage_semantic_burst_markov_feature_matrix_smoke_contract": (
        "post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.json"
    ),
    "usage_semantic_burst_markov_feature_matrix_schema": (
        "post_id341_usage_semantic_burst_markov_features_v1_schema.json"
    ),
    "usage_semantic_burst_markov_validation_stop": (
        "post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
    ),
    "proximity_ambience_cooccurrence_design": ("post_id341_proximity_ambience_cooccurrence_design_20260619.json"),
    "proximity_ambience_cooccurrence_metadata_smoke": (
        "post_id341_proximity_ambience_cooccurrence_metadata_smoke_20260619.json"
    ),
    "proximity_ambience_cooccurrence_feature_spec": (
        "post_id341_proximity_ambience_cooccurrence_feature_spec_20260619.json"
    ),
    "proximity_ambience_cooccurrence_feature_generation_dryrun": (
        "post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun_20260619.json"
    ),
    "proximity_ambience_cooccurrence_feature_matrix_smoke_contract": (
        "post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract_20260619.json"
    ),
    "proximity_ambience_cooccurrence_feature_matrix_schema": (
        "post_id341_proximity_ambience_cooccurrence_features_v1_schema.json"
    ),
    "proximity_ambience_cooccurrence_validation_stop": (
        "post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop_20260619.json"
    ),
    "phone_wearable_discordance_design": "post_id341_phone_wearable_discordance_design_20260619.json",
    "phone_wearable_discordance_feature_spec": "post_id341_phone_wearable_discordance_feature_spec_20260619.json",
    "phone_wearable_discordance_feature_generation_dryrun": (
        "post_id341_phone_wearable_discordance_feature_generation_dryrun_20260619.json"
    ),
    "phone_wearable_discordance_feature_matrix_smoke_contract": (
        "post_id341_phone_wearable_discordance_feature_matrix_smoke_contract_20260619.json"
    ),
    "phone_wearable_discordance_feature_matrix_schema": (
        "post_id341_phone_wearable_discordance_features_v1_schema.json"
    ),
    "phone_wearable_discordance_validation_stop": (
        "post_id341_phone_wearable_discordance_feature_matrix_validation_stop_20260619.json"
    ),
    "sleep_proxy_threshold_residual_design": "post_id341_sleep_proxy_threshold_residual_design_20260619.json",
}


def read_json_report(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(ROOT))}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "exists": True,
            "path": str(path.relative_to(ROOT)),
            "parse_error": str(exc),
        }
    data["exists"] = True
    data["path"] = str(path.relative_to(ROOT))
    return data


def decision(report: dict[str, Any]) -> dict[str, Any]:
    raw = report.get("decision", {})
    return raw if isinstance(raw, dict) else {}


def context_status(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    no_approval = decision(reports["no_approval_frontier"])
    operator = decision(reports["operator_handoff"])
    remaining = decision(reports["q3s3_remaining_route_inventory"])
    target_gap = decision(reports["target_gap_budget"])
    gps = decision(reports["gps_path_topology_model_smoke"])
    circadian = decision(reports["circadian_hr_light_phase_metadata"])
    circadian_spec = decision(reports["circadian_hr_light_phase_feature_spec"])
    circadian_dryrun = decision(reports["circadian_hr_light_phase_feature_generation_dryrun"])
    circadian_matrix_contract = decision(reports["circadian_hr_light_phase_feature_matrix_smoke_contract"])
    circadian_matrix_schema = decision(reports["circadian_hr_light_phase_feature_matrix_schema"])
    circadian_model_contract = decision(reports["circadian_hr_light_phase_model_smoke_contract"])
    circadian_model_smoke = decision(reports["circadian_hr_light_phase_model_smoke"])
    usage_design = decision(reports["usage_semantic_burst_markov_design"])
    usage_metadata = decision(reports["usage_semantic_burst_markov_metadata_smoke"])
    usage_feature_spec = decision(reports["usage_semantic_burst_markov_feature_spec"])
    usage_dryrun = decision(reports["usage_semantic_burst_markov_feature_generation_dryrun"])
    usage_matrix_contract = decision(reports["usage_semantic_burst_markov_feature_matrix_smoke_contract"])
    usage_matrix_schema = decision(reports["usage_semantic_burst_markov_feature_matrix_schema"])
    usage_validation_stop = decision(reports["usage_semantic_burst_markov_validation_stop"])
    proximity_design = decision(reports["proximity_ambience_cooccurrence_design"])
    proximity_metadata = decision(reports["proximity_ambience_cooccurrence_metadata_smoke"])
    proximity_feature_spec = decision(reports["proximity_ambience_cooccurrence_feature_spec"])
    proximity_dryrun = decision(reports["proximity_ambience_cooccurrence_feature_generation_dryrun"])
    proximity_matrix_contract = decision(reports["proximity_ambience_cooccurrence_feature_matrix_smoke_contract"])
    proximity_matrix_schema = decision(reports["proximity_ambience_cooccurrence_feature_matrix_schema"])
    proximity_validation_stop = decision(reports["proximity_ambience_cooccurrence_validation_stop"])
    phone_wearable_design = decision(reports["phone_wearable_discordance_design"])
    phone_wearable_feature_spec = decision(reports["phone_wearable_discordance_feature_spec"])
    phone_wearable_dryrun = decision(reports["phone_wearable_discordance_feature_generation_dryrun"])
    phone_wearable_matrix_contract = decision(reports["phone_wearable_discordance_feature_matrix_smoke_contract"])
    phone_wearable_matrix_schema = decision(reports["phone_wearable_discordance_feature_matrix_schema"])
    phone_wearable_validation_stop = decision(reports["phone_wearable_discordance_validation_stop"])
    sleep_proxy_design = decision(reports["sleep_proxy_threshold_residual_design"])
    circadian_matrix_contract_action_safe = (
        circadian_matrix_contract.get("feature_matrix_smoke_contract_ready") is True
        and circadian_matrix_contract.get("source_family") == "circadian_hr_light_phase_shape_v1"
        and circadian_matrix_contract.get("feature_matrix_write_allowed_now") is False
        and circadian_matrix_contract.get("public_058_evidence") is False
        and circadian_matrix_contract.get("score_route_opened") is False
        and circadian_matrix_contract.get("train_model_now") is False
        and circadian_matrix_contract.get("create_submission_artifact") is False
        and circadian_matrix_contract.get("submit_now") is False
    )
    usage_matrix_contract_action_safe = (
        usage_matrix_contract.get("feature_matrix_smoke_contract_ready") is True
        and usage_matrix_contract.get("source_family") == "usage_semantic_burst_markov_v1"
        and usage_matrix_contract.get("target_focus") == ["Q2", "Q3"]
        and usage_matrix_contract.get("feature_matrix_write_allowed_now") is False
        and usage_matrix_contract.get("public_058_evidence") is False
        and usage_matrix_contract.get("score_route_opened") is False
        and usage_matrix_contract.get("train_model_now") is False
        and usage_matrix_contract.get("create_submission_artifact") is False
        and usage_matrix_contract.get("submit_now") is False
    )
    gps_lane_closed = bool(
        gps.get("model_smoke_completed", False)
        and gps.get("source_family") == "gps_path_topology_sequence_v1"
        and gps.get("selected_targets") == []
        and gps.get("any_gps_topology_signal_found") is False
        and gps.get("public_058_evidence") is False
        and gps.get("score_route_opened") is False
        and gps.get("create_submission_artifact") is False
        and gps.get("submit_now") is False
    )
    return {
        "all_context_reports_present": all(
            report.get("exists") and not report.get("parse_error") for report in reports.values()
        ),
        "no_approval_score_route_exists": bool(no_approval.get("no_approval_score_route_exists", True)),
        "only_score_relevant_open_route": no_approval.get("only_score_relevant_open_route"),
        "prelaunch_ready_for_explicit_approval": bool(no_approval.get("prelaunch_ready_for_explicit_approval", False)),
        "operator_handoff_ready": bool(operator.get("operator_handoff_ready", False)),
        "q3_s3_still_uncovered": bool(remaining.get("q3_s3_still_uncovered", True)),
        "target_gap_has_no_public_058_budget": not bool(
            target_gap.get("strict_budget_public_058_reviewable", False)
            or target_gap.get("watch_budget_public_058_reviewable", False)
        ),
        "gps_path_topology_model_lane_closed": gps_lane_closed,
        "circadian_hr_light_phase_metadata_passed": bool(
            circadian.get("metadata_probe_passed") is True
            and circadian.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian.get("next_local_allowed_step") == "report_only_feature_spec_no_model"
            and circadian.get("public_058_evidence") is False
            and circadian.get("create_submission_artifact") is False
            and circadian.get("submit_now") is False
        ),
        "circadian_hr_light_phase_feature_spec_ready": bool(
            circadian_spec.get("feature_spec_ready") is True
            and circadian_spec.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian_spec.get("next_local_allowed_step") == "report_only_feature_generation_dryrun_no_matrix"
            and circadian_spec.get("feature_matrix_allowed_now") is False
            and circadian_spec.get("public_058_evidence") is False
            and circadian_spec.get("create_submission_artifact") is False
            and circadian_spec.get("submit_now") is False
        ),
        "circadian_hr_light_phase_feature_generation_dryrun_passed": bool(
            circadian_dryrun.get("dryrun_passed") is True
            and circadian_dryrun.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian_dryrun.get("next_local_allowed_step") == "report_only_feature_matrix_smoke_contract"
            and circadian_dryrun.get("feature_matrix_allowed_now") is False
            and circadian_dryrun.get("public_058_evidence") is False
            and circadian_dryrun.get("create_submission_artifact") is False
            and circadian_dryrun.get("submit_now") is False
        ),
        "circadian_hr_light_phase_feature_matrix_smoke_contract_ready": bool(
            circadian_matrix_contract_action_safe
            and (
                (
                    circadian_matrix_contract.get("next_local_allowed_step")
                    == "write_circadian_feature_matrix_under_contract_no_model_no_submission"
                    and circadian_matrix_contract.get("feature_matrix_write_allowed_next_under_contract") is True
                )
                or (
                    circadian_matrix_contract.get("next_local_allowed_step")
                    == "validate_circadian_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
                    and circadian_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
                )
            )
        ),
        "circadian_hr_light_phase_feature_matrix_materialized": bool(
            circadian_matrix_contract_action_safe
            and circadian_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
            and circadian_matrix_schema.get("feature_matrix_materialized") is True
            and circadian_matrix_schema.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian_matrix_schema.get("row_count") == 700
            and circadian_matrix_schema.get("feature_column_count") == 48
            and circadian_matrix_schema.get("public_058_evidence") is False
            and circadian_matrix_schema.get("score_route_opened") is False
            and circadian_matrix_schema.get("train_model_now") is False
            and circadian_matrix_schema.get("create_submission_artifact") is False
            and circadian_matrix_schema.get("submit_now") is False
        ),
        "circadian_hr_light_phase_model_smoke_contract_ready": bool(
            circadian_model_contract.get("model_smoke_contract_ready") is True
            and circadian_model_contract.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian_model_contract.get("model_smoke_run_allowed_now") is False
            and (
                (
                    circadian_model_contract.get("next_local_allowed_step")
                    == "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact"
                    and circadian_model_contract.get("model_smoke_run_allowed_next_under_contract") is True
                    and circadian_model_contract.get("model_smoke_already_completed_under_contract") is False
                )
                or (
                    circadian_model_contract.get("next_local_allowed_step")
                    == "close_circadian_phase_model_smoke_or_continue_usage_design"
                    and circadian_model_contract.get("model_smoke_run_allowed_next_under_contract") is False
                    and circadian_model_contract.get("model_smoke_already_completed_under_contract") is True
                )
            )
            and circadian_model_contract.get("public_058_evidence") is False
            and circadian_model_contract.get("score_route_opened") is False
            and circadian_model_contract.get("train_model_now") is False
            and circadian_model_contract.get("create_submission_artifact") is False
            and circadian_model_contract.get("submit_now") is False
        ),
        "circadian_hr_light_phase_model_smoke_completed": bool(
            circadian_model_smoke.get("model_smoke_completed") is True
            and circadian_model_smoke.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian_model_smoke.get("public_058_evidence") is False
            and circadian_model_smoke.get("score_route_opened") is False
            and circadian_model_smoke.get("create_submission_artifact") is False
            and circadian_model_smoke.get("submit_now") is False
        ),
        "circadian_hr_light_phase_model_lane_closed": bool(
            circadian_model_smoke.get("model_smoke_completed") is True
            and circadian_model_smoke.get("source_family") == "circadian_hr_light_phase_shape_v1"
            and circadian_model_smoke.get("selected_targets") == []
            and circadian_model_smoke.get("any_circadian_phase_signal_found") is False
            and circadian_model_smoke.get("public_058_evidence") is False
            and circadian_model_smoke.get("score_route_opened") is False
            and circadian_model_smoke.get("create_submission_artifact") is False
            and circadian_model_smoke.get("submit_now") is False
        ),
        "usage_semantic_burst_markov_design_ready": bool(
            usage_design.get("design_ready") is True
            and usage_design.get("source_family") == "usage_semantic_burst_markov_v1"
            and usage_design.get("target_focus") == ["Q2", "Q3"]
            and usage_design.get("next_local_allowed_step")
            == "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features"
            and usage_design.get("public_058_evidence") is False
            and usage_design.get("score_route_opened") is False
            and usage_design.get("feature_matrix_allowed_now") is False
            and usage_design.get("train_model_now") is False
            and usage_design.get("create_submission_artifact") is False
            and usage_design.get("submit_now") is False
        ),
        "usage_semantic_burst_markov_metadata_smoke_passed": bool(
            usage_metadata.get("metadata_smoke_passed") is True
            and usage_metadata.get("source_family") == "usage_semantic_burst_markov_v1"
            and usage_metadata.get("target_focus") == ["Q2", "Q3"]
            and usage_metadata.get("next_local_allowed_step")
            == "report_only_usage_semantic_burst_markov_feature_spec_no_matrix"
            and usage_metadata.get("public_058_evidence") is False
            and usage_metadata.get("score_route_opened") is False
            and usage_metadata.get("feature_matrix_allowed_now") is False
            and usage_metadata.get("train_model_now") is False
            and usage_metadata.get("create_submission_artifact") is False
            and usage_metadata.get("submit_now") is False
        ),
        "usage_semantic_burst_markov_feature_spec_ready": bool(
            usage_feature_spec.get("feature_spec_ready") is True
            and usage_feature_spec.get("source_family") == "usage_semantic_burst_markov_v1"
            and usage_feature_spec.get("target_focus") == ["Q2", "Q3"]
            and usage_feature_spec.get("next_local_allowed_step")
            == "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix"
            and usage_feature_spec.get("public_058_evidence") is False
            and usage_feature_spec.get("score_route_opened") is False
            and usage_feature_spec.get("feature_matrix_allowed_now") is False
            and usage_feature_spec.get("train_model_now") is False
            and usage_feature_spec.get("create_submission_artifact") is False
            and usage_feature_spec.get("submit_now") is False
        ),
        "usage_semantic_burst_markov_feature_generation_dryrun_passed": bool(
            usage_dryrun.get("feature_generation_dryrun_passed") is True
            and usage_dryrun.get("source_family") == "usage_semantic_burst_markov_v1"
            and usage_dryrun.get("target_focus") == ["Q2", "Q3"]
            and usage_dryrun.get("next_local_allowed_step")
            == "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract"
            and usage_dryrun.get("public_058_evidence") is False
            and usage_dryrun.get("score_route_opened") is False
            and usage_dryrun.get("feature_matrix_allowed_now") is False
            and usage_dryrun.get("train_model_now") is False
            and usage_dryrun.get("create_submission_artifact") is False
            and usage_dryrun.get("submit_now") is False
        ),
        "usage_semantic_burst_markov_feature_matrix_smoke_contract_ready": bool(
            usage_matrix_contract_action_safe
            and (
                (
                    usage_matrix_contract.get("next_local_allowed_step")
                    == "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission"
                    and usage_matrix_contract.get("feature_matrix_write_allowed_next_under_contract") is True
                )
                or (
                    usage_matrix_contract.get("next_local_allowed_step")
                    == "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
                    and usage_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
                )
            )
        ),
        "usage_semantic_burst_markov_feature_matrix_materialized": bool(
            usage_matrix_contract_action_safe
            and usage_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
            and usage_matrix_schema.get("feature_matrix_materialized") is True
            and usage_matrix_schema.get("source_family") == "usage_semantic_burst_markov_v1"
            and usage_matrix_schema.get("row_count") == 700
            and usage_matrix_schema.get("feature_column_count") == 77
            and usage_matrix_schema.get("public_058_evidence") is False
            and usage_matrix_schema.get("score_route_opened") is False
            and usage_matrix_schema.get("train_model_now") is False
            and usage_matrix_schema.get("create_submission_artifact") is False
            and usage_matrix_schema.get("submit_now") is False
        ),
        "usage_semantic_burst_markov_validation_stop_passed": bool(
            usage_validation_stop.get("validation_stop_passed") is True
            and usage_validation_stop.get("source_family") == "usage_semantic_burst_markov_v1"
            and usage_validation_stop.get("target_focus") == ["Q2", "Q3"]
            and usage_validation_stop.get("selected_targets") == []
            and usage_validation_stop.get("public_058_evidence") is False
            and usage_validation_stop.get("score_route_opened") is False
            and usage_validation_stop.get("train_model_now") is False
            and usage_validation_stop.get("create_submission_artifact") is False
            and usage_validation_stop.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_design_ready": bool(
            proximity_design.get("design_ready") is True
            and proximity_design.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_design.get("target_focus") == ["Q3", "S3"]
            and proximity_design.get("public_058_evidence") is False
            and proximity_design.get("score_route_opened") is False
            and proximity_design.get("feature_matrix_allowed_now") is False
            and proximity_design.get("train_model_now") is False
            and proximity_design.get("create_submission_artifact") is False
            and proximity_design.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_metadata_smoke_passed": bool(
            proximity_metadata.get("metadata_smoke_passed") is True
            and proximity_metadata.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_metadata.get("target_focus") == ["Q3", "S3"]
            and proximity_metadata.get("public_058_evidence") is False
            and proximity_metadata.get("score_route_opened") is False
            and proximity_metadata.get("feature_matrix_allowed_now") is False
            and proximity_metadata.get("train_model_now") is False
            and proximity_metadata.get("create_submission_artifact") is False
            and proximity_metadata.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_feature_spec_ready": bool(
            proximity_feature_spec.get("feature_spec_ready") is True
            and proximity_feature_spec.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_feature_spec.get("target_focus") == ["Q3", "S3"]
            and proximity_feature_spec.get("feature_count") == 80
            and proximity_feature_spec.get("public_058_evidence") is False
            and proximity_feature_spec.get("score_route_opened") is False
            and proximity_feature_spec.get("feature_matrix_allowed_now") is False
            and proximity_feature_spec.get("train_model_now") is False
            and proximity_feature_spec.get("create_submission_artifact") is False
            and proximity_feature_spec.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_feature_generation_dryrun_passed": bool(
            proximity_dryrun.get("feature_generation_dryrun_passed") is True
            and proximity_dryrun.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_dryrun.get("target_focus") == ["Q3", "S3"]
            and proximity_dryrun.get("feature_count") == 80
            and proximity_dryrun.get("public_058_evidence") is False
            and proximity_dryrun.get("score_route_opened") is False
            and proximity_dryrun.get("feature_matrix_allowed_now") is False
            and proximity_dryrun.get("train_model_now") is False
            and proximity_dryrun.get("create_submission_artifact") is False
            and proximity_dryrun.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_feature_matrix_smoke_contract_ready": bool(
            proximity_matrix_contract.get("feature_matrix_smoke_contract_ready") is True
            and proximity_matrix_contract.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_matrix_contract.get("target_focus") == ["Q3", "S3"]
            and proximity_matrix_contract.get("feature_matrix_write_allowed_now") is False
            and (
                proximity_matrix_contract.get("feature_matrix_write_allowed_next_under_contract") is True
                or proximity_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
            )
            and proximity_matrix_contract.get("public_058_evidence") is False
            and proximity_matrix_contract.get("score_route_opened") is False
            and proximity_matrix_contract.get("train_model_now") is False
            and proximity_matrix_contract.get("create_submission_artifact") is False
            and proximity_matrix_contract.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_feature_matrix_materialized": bool(
            proximity_matrix_schema.get("feature_matrix_materialized") is True
            and proximity_matrix_schema.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_matrix_schema.get("target_focus") == ["Q3", "S3"]
            and proximity_matrix_schema.get("row_count") == 700
            and proximity_matrix_schema.get("feature_column_count") == 80
            and proximity_matrix_schema.get("public_058_evidence") is False
            and proximity_matrix_schema.get("score_route_opened") is False
            and proximity_matrix_schema.get("train_model_now") is False
            and proximity_matrix_schema.get("create_submission_artifact") is False
            and proximity_matrix_schema.get("submit_now") is False
        ),
        "proximity_ambience_cooccurrence_validation_stop_passed": bool(
            proximity_validation_stop.get("validation_stop_passed") is True
            and proximity_validation_stop.get("source_family") == "proximity_ambience_cooccurrence_v1"
            and proximity_validation_stop.get("target_focus") == ["Q3", "S3"]
            and proximity_validation_stop.get("selected_targets") == []
            and proximity_validation_stop.get("public_058_evidence") is False
            and proximity_validation_stop.get("score_route_opened") is False
            and proximity_validation_stop.get("train_model_now") is False
            and proximity_validation_stop.get("create_submission_artifact") is False
            and proximity_validation_stop.get("submit_now") is False
        ),
        "phone_wearable_discordance_design_ready": bool(
            phone_wearable_design.get("design_ready") is True
            and phone_wearable_design.get("source_family") == "phone_wearable_discordance_v1"
            and phone_wearable_design.get("target_focus") == ["Q2", "Q3", "S3"]
            and phone_wearable_design.get("public_058_evidence") is False
            and phone_wearable_design.get("score_route_opened") is False
            and phone_wearable_design.get("feature_matrix_allowed_now") is False
            and phone_wearable_design.get("train_model_now") is False
            and phone_wearable_design.get("create_submission_artifact") is False
            and phone_wearable_design.get("submit_now") is False
        ),
        "phone_wearable_discordance_feature_spec_ready": bool(
            phone_wearable_feature_spec.get("feature_spec_ready") is True
            and phone_wearable_feature_spec.get("source_family") == "phone_wearable_discordance_v1"
            and phone_wearable_feature_spec.get("target_focus") == ["Q2", "Q3", "S3"]
            and phone_wearable_feature_spec.get("feature_count") == 85
            and phone_wearable_feature_spec.get("next_local_allowed_step")
            == "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix"
            and phone_wearable_feature_spec.get("public_058_evidence") is False
            and phone_wearable_feature_spec.get("score_route_opened") is False
            and phone_wearable_feature_spec.get("feature_matrix_allowed_now") is False
            and phone_wearable_feature_spec.get("train_model_now") is False
            and phone_wearable_feature_spec.get("create_submission_artifact") is False
            and phone_wearable_feature_spec.get("submit_now") is False
        ),
        "phone_wearable_discordance_feature_generation_dryrun_passed": bool(
            phone_wearable_dryrun.get("feature_generation_dryrun_passed") is True
            and phone_wearable_dryrun.get("source_family") == "phone_wearable_discordance_v1"
            and phone_wearable_dryrun.get("target_focus") == ["Q2", "Q3", "S3"]
            and phone_wearable_dryrun.get("feature_count") == 85
            and phone_wearable_dryrun.get("next_local_allowed_step")
            == "report_only_phone_wearable_discordance_feature_matrix_smoke_contract"
            and phone_wearable_dryrun.get("public_058_evidence") is False
            and phone_wearable_dryrun.get("score_route_opened") is False
            and phone_wearable_dryrun.get("feature_matrix_allowed_now") is False
            and phone_wearable_dryrun.get("train_model_now") is False
            and phone_wearable_dryrun.get("create_submission_artifact") is False
            and phone_wearable_dryrun.get("submit_now") is False
        ),
        "phone_wearable_discordance_feature_matrix_smoke_contract_ready": bool(
            phone_wearable_matrix_contract.get("feature_matrix_smoke_contract_ready") is True
            and phone_wearable_matrix_contract.get("source_family") == "phone_wearable_discordance_v1"
            and phone_wearable_matrix_contract.get("target_focus") == ["Q2", "Q3", "S3"]
            and phone_wearable_matrix_contract.get("feature_matrix_write_allowed_now") is False
            and (
                (
                    phone_wearable_matrix_contract.get("next_local_allowed_step")
                    == "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission"
                    and phone_wearable_matrix_contract.get("feature_matrix_write_allowed_next_under_contract") is True
                )
                or (
                    phone_wearable_matrix_contract.get("next_local_allowed_step")
                    == "validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
                    and phone_wearable_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
                )
            )
            and phone_wearable_matrix_contract.get("public_058_evidence") is False
            and phone_wearable_matrix_contract.get("score_route_opened") is False
            and phone_wearable_matrix_contract.get("train_model_now") is False
            and phone_wearable_matrix_contract.get("create_submission_artifact") is False
            and phone_wearable_matrix_contract.get("submit_now") is False
        ),
        "phone_wearable_discordance_feature_matrix_materialized": bool(
            phone_wearable_matrix_contract.get("feature_matrix_smoke_contract_ready") is True
            and phone_wearable_matrix_contract.get("feature_matrix_already_materialized_under_contract") is True
            and phone_wearable_matrix_schema.get("feature_matrix_materialized") is True
            and phone_wearable_matrix_schema.get("source_family") == "phone_wearable_discordance_v1"
            and phone_wearable_matrix_schema.get("target_focus") == ["Q2", "Q3", "S3"]
            and phone_wearable_matrix_schema.get("row_count") == 700
            and phone_wearable_matrix_schema.get("feature_column_count") == 85
            and phone_wearable_matrix_schema.get("public_058_evidence") is False
            and phone_wearable_matrix_schema.get("score_route_opened") is False
            and phone_wearable_matrix_schema.get("train_model_now") is False
            and phone_wearable_matrix_schema.get("create_submission_artifact") is False
            and phone_wearable_matrix_schema.get("submit_now") is False
        ),
        "phone_wearable_discordance_validation_stop_passed": bool(
            phone_wearable_validation_stop.get("validation_stop_passed") is True
            and phone_wearable_validation_stop.get("source_family") == "phone_wearable_discordance_v1"
            and phone_wearable_validation_stop.get("target_focus") == ["Q2", "Q3", "S3"]
            and phone_wearable_validation_stop.get("selected_targets") == []
            and phone_wearable_validation_stop.get("public_058_evidence") is False
            and phone_wearable_validation_stop.get("score_route_opened") is False
            and phone_wearable_validation_stop.get("train_model_now") is False
            and phone_wearable_validation_stop.get("create_submission_artifact") is False
            and phone_wearable_validation_stop.get("submit_now") is False
        ),
        "sleep_proxy_threshold_residual_design_ready": bool(
            sleep_proxy_design.get("sleep_proxy_threshold_residual_design_ready") is True
            and sleep_proxy_design.get("source_family") == "sleep_proxy_threshold_residual_v1"
            and sleep_proxy_design.get("target_focus") == ["S2", "S3", "S4", "Q1"]
            and sleep_proxy_design.get("metadata_feasible") is True
            and sleep_proxy_design.get("design_backlog_recorded") is True
            and sleep_proxy_design.get("local_no_approval_score_route_exists") is False
            and sleep_proxy_design.get("open_design_lane_now") is False
            and sleep_proxy_design.get("feature_matrix_allowed_now") is False
            and sleep_proxy_design.get("train_model_now") is False
            and sleep_proxy_design.get("compute_target_associations_now") is False
            and sleep_proxy_design.get("public_058_evidence") is False
            and sleep_proxy_design.get("public_058_reviewable") is False
            and sleep_proxy_design.get("create_feature_matrix") is False
            and sleep_proxy_design.get("create_submission_artifact") is False
            and sleep_proxy_design.get("modify_submission_log") is False
            and sleep_proxy_design.get("create_id342_or_new_submission_id") is False
            and sleep_proxy_design.get("submit_now") is False
            and sleep_proxy_design.get("push_kaggle_now") is False
            and sleep_proxy_design.get("run_kaggle_now") is False
        ),
    }


def closed_repeat_lanes() -> list[dict[str, Any]]:
    return [
        {
            "lane": "gps_path_topology_sequence_v1",
            "status": "closed_for_artifact",
            "repeat_risk": "high",
            "reason": (
                "The GPS path-topology feature matrix was materialized and the fold-internal model smoke selected "
                "no Q3/S3 target, opened no score route, and produced no Public 0.58 evidence."
            ),
            "do_not_repeat_as": "GPS topology metadata/spec/matrix/model-smoke replay with only tiny weight changes",
        },
        {
            "lane": "sleep_boundary_micro_blend",
            "status": "feature_lead_only",
            "repeat_risk": "high",
            "reason": (
                "Sleep-boundary v3 scale 0.05 is clean and low-movement, but the prepromotion proxy remains above "
                "the ID178 public anchor and far above Public 0.58."
            ),
            "do_not_repeat_as": "another Q1/S1/S2/S4 micro-scale local blend",
        },
        {
            "lane": "evening_load_recovery",
            "status": "closed_for_artifact",
            "repeat_risk": "high",
            "reason": "The best evening-load rows have only 4 primary wins and no reviewable low-movement candidate.",
            "do_not_repeat_as": "S4-only or weak evening-load scale scan",
        },
        {
            "lane": "routine_stability_place_entropy",
            "status": "closed_for_artifact",
            "repeat_risk": "high",
            "reason": "Routine/place entropy selected no Q3/S3 target.",
            "do_not_repeat_as": "daily aggregate place/routine entropy scan",
        },
        {
            "lane": "sensor_adherence_dropout",
            "status": "closed_for_artifact",
            "repeat_risk": "high",
            "reason": "Adherence/dropout selected no target as a standalone local lane.",
            "do_not_repeat_as": "coverage/missingness-only feature family",
        },
        {
            "lane": "paper_family_processed_features",
            "status": "closed_for_artifact",
            "repeat_risk": "high",
            "reason": "Paper-family processed-feature smoke selected no Q2/Q3/S3 target.",
            "do_not_repeat_as": "same processed-column family with only small weight changes",
        },
        {
            "lane": "local_oof_combo_or_second_stage_calibration",
            "status": "quarantined",
            "repeat_risk": "critical",
            "reason": "ID287 and ID329 showed local OOF/frontier gains can collapse or plateau publicly.",
            "do_not_repeat_as": "same-family local CV cascade or full-OOF calibration",
        },
    ]


def source_queue(
    circadian_feature_spec_ready: bool = False,
    circadian_feature_generation_dryrun_passed: bool = False,
    circadian_feature_matrix_smoke_contract_ready: bool = False,
    circadian_feature_matrix_materialized: bool = False,
    circadian_model_smoke_contract_ready: bool = False,
    circadian_model_smoke_completed: bool = False,
    circadian_model_lane_closed: bool = False,
    usage_semantic_burst_markov_design_ready: bool = False,
    usage_semantic_burst_markov_metadata_smoke_passed: bool = False,
    usage_semantic_burst_markov_feature_spec_ready: bool = False,
    usage_semantic_burst_markov_feature_generation_dryrun_passed: bool = False,
    usage_semantic_burst_markov_feature_matrix_smoke_contract_ready: bool = False,
    usage_semantic_burst_markov_feature_matrix_materialized: bool = False,
    usage_semantic_burst_markov_validation_stop_passed: bool = False,
    proximity_ambience_cooccurrence_design_ready: bool = False,
    proximity_ambience_cooccurrence_metadata_smoke_passed: bool = False,
    proximity_ambience_cooccurrence_feature_spec_ready: bool = False,
    proximity_ambience_cooccurrence_feature_generation_dryrun_passed: bool = False,
    proximity_ambience_cooccurrence_feature_matrix_smoke_contract_ready: bool = False,
    proximity_ambience_cooccurrence_feature_matrix_materialized: bool = False,
    proximity_ambience_cooccurrence_validation_stop_passed: bool = False,
    phone_wearable_discordance_design_ready: bool = False,
    phone_wearable_discordance_feature_spec_ready: bool = False,
    phone_wearable_discordance_feature_generation_dryrun_passed: bool = False,
    phone_wearable_discordance_feature_matrix_smoke_contract_ready: bool = False,
    phone_wearable_discordance_feature_matrix_materialized: bool = False,
    phone_wearable_discordance_validation_stop_passed: bool = False,
    self_supervised_subject_state_embedding_design_ready: bool = False,
    sleep_proxy_threshold_residual_design_ready: bool = False,
) -> list[dict[str, Any]]:
    if circadian_model_lane_closed:
        circadian_status = "closed_after_feature_matrix_and_model_smoke_no_signal"
        circadian_first_safe_probe = (
            "completed: metadata, feature spec, feature-generation dry run, feature-matrix contract, matrix write, "
            "model-smoke contract, and bounded in-memory model smoke. No target was selected and no score route "
            "opened."
        )
        circadian_next_allowed_step = "closed_no_artifact"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = False
    elif circadian_model_smoke_completed:
        circadian_status = "model_smoke_completed_review_needed_no_artifact"
        circadian_first_safe_probe = (
            "completed through bounded in-memory model smoke; review selected targets before any further contract."
        )
        circadian_next_allowed_step = "review_model_smoke_report_no_artifact"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = False
    elif circadian_model_smoke_contract_ready:
        circadian_status = "model_smoke_contract_ready_run_only_under_contract"
        circadian_first_safe_probe = (
            "completed: metadata feasibility, feature spec, feature-generation dry run, feature-matrix smoke "
            "contract, registered label-free feature matrix, and report-only model-smoke contract; no fold model "
            "has run yet."
        )
        circadian_next_allowed_step = "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = True
    elif circadian_feature_matrix_materialized:
        circadian_status = "feature_matrix_materialized_under_contract_no_model"
        circadian_first_safe_probe = (
            "completed: metadata feasibility, feature spec, feature-generation dry run, feature-matrix smoke "
            "contract, and registered label-free feature matrix; no fold model or submission artifact."
        )
        circadian_next_allowed_step = "report_only_model_smoke_contract_no_model_no_submission"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = False
    elif circadian_feature_matrix_smoke_contract_ready:
        circadian_status = "feature_matrix_smoke_contract_ready_write_only_under_contract"
        circadian_first_safe_probe = (
            "completed: metadata feasibility, report-only feature spec, report-only feature-generation dry run, "
            "and report-only feature-matrix smoke contract; no feature matrix or fold model yet."
        )
        circadian_next_allowed_step = "write_circadian_feature_matrix_under_contract_no_model_no_submission"
        circadian_feature_matrix_write_allowed_next = True
        circadian_model_smoke_allowed_next = False
    elif circadian_feature_generation_dryrun_passed:
        circadian_status = "feature_generation_dryrun_passed_contract_only_without_approval"
        circadian_first_safe_probe = (
            "completed: metadata feasibility, report-only feature spec, and report-only feature-generation dry "
            "run; no feature matrix or fold model."
        )
        circadian_next_allowed_step = "report_only_feature_matrix_smoke_contract"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = False
    elif circadian_feature_spec_ready:
        circadian_status = "feature_spec_ready_dryrun_contract_only_without_approval"
        circadian_first_safe_probe = (
            "completed: metadata feasibility and report-only feature spec; no feature matrix or fold model."
        )
        circadian_next_allowed_step = "report_only_feature_generation_dryrun_no_matrix"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = False
    else:
        circadian_status = "metadata_passed_feature_spec_only_allowed_without_approval"
        circadian_first_safe_probe = (
            "completed: report-only missingness and time-grid feasibility audit; no fold model."
        )
        circadian_next_allowed_step = "report_only_feature_spec_no_model"
        circadian_feature_matrix_write_allowed_next = False
        circadian_model_smoke_allowed_next = False
    if usage_semantic_burst_markov_validation_stop_passed:
        usage_status = "validated_stop_completed_no_model"
        usage_first_safe_probe = (
            "completed: design, metadata/cardinality smoke, feature spec, in-memory dry run, feature-matrix "
            "smoke contract, one registered label-free feature matrix, schema report, and validation stop. No "
            "model, submission artifact, Kaggle run, or DACON submission is authorized."
        )
        usage_next_allowed_step = "closed_stop_or_wait_for_explicit_kaggle_approval"
        usage_local_no_approval_allowed = False
    elif usage_semantic_burst_markov_feature_matrix_materialized:
        usage_status = "feature_matrix_materialized_under_contract_no_model"
        usage_first_safe_probe = (
            "completed: design, metadata/cardinality smoke, feature spec, in-memory dry run, feature-matrix "
            "smoke contract, and one registered label-free feature matrix plus schema report. No model, "
            "submission artifact, Kaggle run, or DACON submission is authorized."
        )
        usage_next_allowed_step = (
            "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
        )
        usage_local_no_approval_allowed = True
    elif usage_semantic_burst_markov_feature_matrix_smoke_contract_ready:
        usage_status = "feature_matrix_smoke_contract_ready_write_only_under_contract"
        usage_first_safe_probe = (
            "completed: design, metadata/cardinality smoke, feature spec, in-memory dry run, and report-only "
            "feature-matrix smoke contract. The contract permits only one registered label-free matrix writer, "
            "then a hard stop for schema validation."
        )
        usage_next_allowed_step = (
            "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission"
        )
        usage_local_no_approval_allowed = True
    elif usage_semantic_burst_markov_feature_generation_dryrun_passed:
        usage_status = "feature_generation_dryrun_passed_contract_only"
        usage_first_safe_probe = (
            "completed: design, metadata/cardinality smoke, feature spec, and in-memory dry run. The dry run "
            "assembled 700 aligned rows, 77 bounded feature columns, finite values, and no persisted feature matrix "
            "or model."
        )
        usage_next_allowed_step = "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract"
        usage_local_no_approval_allowed = True
    elif usage_semantic_burst_markov_feature_spec_ready:
        usage_status = "feature_spec_ready_dryrun_only"
        usage_first_safe_probe = (
            "completed: design, metadata/cardinality smoke, and report-only feature spec with exact predeclared "
            "states, row contract, leakage guards, and a bounded 77-column first-pass schema. No feature matrix or "
            "model has run."
        )
        usage_next_allowed_step = "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix"
        usage_local_no_approval_allowed = True
    elif usage_semantic_burst_markov_metadata_smoke_passed:
        usage_status = "metadata_cardinality_smoke_passed_feature_spec_only"
        usage_first_safe_probe = (
            "completed: report-only design plus metadata/cardinality smoke over predeclared app categories, Markov "
            "pairs, session bursts, and screen coupling. No feature matrix or model."
        )
        usage_next_allowed_step = "report_only_usage_semantic_burst_markov_feature_spec_no_matrix"
        usage_local_no_approval_allowed = True
    elif usage_semantic_burst_markov_design_ready:
        usage_status = "design_ready_metadata_cardinality_smoke_only"
        usage_first_safe_probe = (
            "completed: report-only design for app/session burst states; next may only check metadata/cardinality."
        )
        usage_next_allowed_step = "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features"
        usage_local_no_approval_allowed = True
    elif circadian_model_lane_closed:
        usage_status = "open_report_only_design_after_circadian_model_smoke_closure"
        usage_first_safe_probe = (
            "Start with a report-only design/spec for app/session burst states; do not generate features or train."
        )
        usage_next_allowed_step = "report_only_usage_semantic_burst_markov_design_no_model"
        usage_local_no_approval_allowed = True
    else:
        usage_status = "hold_until_circadian_feature_spec_review"
        usage_first_safe_probe = "Only inspect app token/session cardinality if the first two probes fail."
        usage_next_allowed_step = "defer"
        usage_local_no_approval_allowed = False
    if proximity_ambience_cooccurrence_validation_stop_passed:
        proximity_status = "validated_stop_completed_no_model"
        proximity_first_safe_probe = (
            "completed: report-only design, metadata/co-occurrence smoke, bounded 80-column feature spec, "
            "in-memory feature-generation dry run, report-only feature-matrix smoke contract, one registered "
            "label-free feature matrix, schema report, and deterministic validation stop. No model, submission "
            "artifact, Kaggle run, or DACON submission is authorized."
        )
        proximity_next_allowed_step = "closed_stop_or_wait_for_explicit_kaggle_approval"
        proximity_local_no_approval_allowed = False
    elif proximity_ambience_cooccurrence_feature_matrix_materialized:
        proximity_status = "feature_matrix_materialized_under_contract_no_model"
        proximity_first_safe_probe = (
            "completed: report-only design, metadata/co-occurrence smoke, bounded 80-column feature spec, "
            "in-memory feature-generation dry run, report-only feature-matrix smoke contract, and one registered "
            "label-free feature matrix plus schema report. No model or submission artifact is authorized."
        )
        proximity_next_allowed_step = (
            "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
        )
        proximity_local_no_approval_allowed = True
    elif proximity_ambience_cooccurrence_feature_matrix_smoke_contract_ready:
        proximity_status = "feature_matrix_smoke_contract_ready_write_only_under_contract"
        proximity_first_safe_probe = (
            "completed: report-only design, metadata/co-occurrence smoke, bounded 80-column feature spec, "
            "in-memory feature-generation dry run, and report-only feature-matrix smoke contract. The contract "
            "permits only one registered label-free matrix writer, then a hard stop for schema validation."
        )
        proximity_next_allowed_step = (
            "write_proximity_ambience_cooccurrence_feature_matrix_under_contract_no_model_no_submission"
        )
        proximity_local_no_approval_allowed = True
    elif proximity_ambience_cooccurrence_feature_generation_dryrun_passed:
        proximity_status = "feature_generation_dryrun_passed_contract_only"
        proximity_first_safe_probe = (
            "completed: report-only design, metadata/co-occurrence smoke, bounded 80-column feature spec, and "
            "in-memory feature-generation dry run. The dry run assembled 700 aligned rows, finite values, and no "
            "persisted feature matrix or model."
        )
        proximity_next_allowed_step = "report_only_proximity_ambience_cooccurrence_feature_matrix_smoke_contract"
        proximity_local_no_approval_allowed = True
    elif proximity_ambience_cooccurrence_feature_spec_ready:
        proximity_status = "feature_spec_ready_dryrun_only"
        proximity_first_safe_probe = (
            "completed: report-only design, metadata/co-occurrence smoke, and bounded 80-column feature spec. "
            "No feature matrix or model has run."
        )
        proximity_next_allowed_step = "report_only_proximity_ambience_cooccurrence_feature_generation_dryrun_no_matrix"
        proximity_local_no_approval_allowed = True
    elif proximity_ambience_cooccurrence_metadata_smoke_passed:
        proximity_status = "metadata_smoke_passed_feature_spec_only"
        proximity_first_safe_probe = (
            "completed: design plus metadata/co-occurrence smoke. No feature matrix or model has run."
        )
        proximity_next_allowed_step = "report_only_proximity_ambience_cooccurrence_feature_spec_no_matrix"
        proximity_local_no_approval_allowed = True
    elif proximity_ambience_cooccurrence_design_ready:
        proximity_status = "design_ready_metadata_smoke_only"
        proximity_first_safe_probe = (
            "completed: report-only design for proximity/social ambience co-occurrence. No feature generation."
        )
        proximity_next_allowed_step = "proximity_ambience_cooccurrence_metadata_smoke_no_features"
        proximity_local_no_approval_allowed = True
    elif usage_semantic_burst_markov_validation_stop_passed:
        proximity_status = "open_report_only_design_after_usage_validation_stop"
        proximity_first_safe_probe = (
            "Start with a report-only design for WiFi/BLE proximity plus ambience social/place co-occurrence; "
            "do not generate features or train."
        )
        proximity_next_allowed_step = "report_only_proximity_ambience_cooccurrence_design_no_model"
        proximity_local_no_approval_allowed = True
    else:
        proximity_status = "hold_until_usage_validation_stop"
        proximity_first_safe_probe = "Defer until usage semantic burst/Markov validation stop passes."
        proximity_next_allowed_step = "defer"
        proximity_local_no_approval_allowed = False

    if phone_wearable_discordance_validation_stop_passed:
        phone_wearable_status = "validated_stop_completed_no_model"
        phone_wearable_first_safe_probe = (
            "completed: design, bounded feature spec, in-memory dry run, feature-matrix smoke contract, one "
            "registered label-free feature matrix, schema report, and deterministic validation stop. No model, "
            "submission artifact, Kaggle run, or DACON submission is authorized."
        )
        phone_wearable_next_allowed_step = "closed_stop_or_wait_for_explicit_kaggle_approval"
        phone_wearable_local_no_approval_allowed = False
    elif phone_wearable_discordance_feature_matrix_materialized:
        phone_wearable_status = "feature_matrix_materialized_under_contract_no_model"
        phone_wearable_first_safe_probe = (
            "completed: report-only design, bounded 85-column feature spec, in-memory dry run, feature-matrix "
            "smoke contract, and one registered label-free feature matrix plus schema report. No model or "
            "submission artifact is authorized."
        )
        phone_wearable_next_allowed_step = (
            "validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
        )
        phone_wearable_local_no_approval_allowed = True
    elif phone_wearable_discordance_feature_matrix_smoke_contract_ready:
        phone_wearable_status = "feature_matrix_smoke_contract_ready_write_only_under_contract"
        phone_wearable_first_safe_probe = (
            "completed: report-only design, bounded 85-column feature spec, in-memory feature-generation dry run, "
            "and report-only feature-matrix smoke contract. The contract permits only one registered label-free "
            "matrix writer, then a hard stop for schema validation. No model, submission artifact, Kaggle run, or "
            "DACON submission is authorized."
        )
        phone_wearable_next_allowed_step = (
            "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission"
        )
        phone_wearable_local_no_approval_allowed = True
    elif phone_wearable_discordance_feature_generation_dryrun_passed:
        phone_wearable_status = "feature_generation_dryrun_passed_contract_only"
        phone_wearable_first_safe_probe = (
            "completed: report-only design, bounded 85-column feature spec, and in-memory feature-generation "
            "dry run over 700 aligned rows. No feature matrix, model, submission artifact, Kaggle run, or DACON "
            "submission is authorized."
        )
        phone_wearable_next_allowed_step = "report_only_phone_wearable_discordance_feature_matrix_smoke_contract"
        phone_wearable_local_no_approval_allowed = True
    elif phone_wearable_discordance_feature_spec_ready:
        phone_wearable_status = "feature_spec_ready_dryrun_only"
        phone_wearable_first_safe_probe = (
            "completed: report-only design plus bounded 85-column feature spec over phone state, wearable state, "
            "and fixed discordance windows. No feature matrix, model, submission artifact, Kaggle run, or DACON "
            "submission is authorized."
        )
        phone_wearable_next_allowed_step = "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix"
        phone_wearable_local_no_approval_allowed = True
    elif phone_wearable_discordance_design_ready:
        phone_wearable_status = "design_ready_feature_spec_only"
        phone_wearable_first_safe_probe = (
            "completed: report-only design plus raw coverage/discordance metadata over phone state and wearable "
            "state. No feature matrix, model, submission artifact, Kaggle run, or DACON submission is authorized."
        )
        phone_wearable_next_allowed_step = "report_only_phone_wearable_discordance_feature_spec_no_matrix"
        phone_wearable_local_no_approval_allowed = True
    elif proximity_ambience_cooccurrence_validation_stop_passed:
        phone_wearable_status = "open_report_only_design_after_proximity_validation_stop"
        phone_wearable_first_safe_probe = (
            "Start with a report-only design for phone/wearable discordance; do not generate features or train."
        )
        phone_wearable_next_allowed_step = "report_only_phone_wearable_discordance_design_no_model"
        phone_wearable_local_no_approval_allowed = True
    else:
        phone_wearable_status = "hold_until_proximity_validation_stop"
        phone_wearable_first_safe_probe = "Defer until proximity/ambience validation stop passes."
        phone_wearable_next_allowed_step = "defer"
        phone_wearable_local_no_approval_allowed = False

    if self_supervised_subject_state_embedding_design_ready:
        self_supervised_status = "design_gate_ready_heavy_deferred"
        self_supervised_first_safe_probe = (
            "completed: metadata-only design gate over all 12 raw sensor streams and the 450/250 row contract. "
            "No full raw-stream read, embedding fit, feature matrix, model, submission artifact, Kaggle run, or "
            "DACON submission is authorized."
        )
    else:
        self_supervised_status = "requires_separate_kaggle_approval_if_ever_executed"
        self_supervised_first_safe_probe = "Design document only; no local training and no automatic Kaggle run."

    if sleep_proxy_threshold_residual_design_ready:
        sleep_proxy_status = "design_backlog_recorded_closed_no_local_lane"
        sleep_proxy_first_safe_probe = (
            "completed: design-only schema/context feasibility for sleep-stage, sleep-efficiency, fragmentation, "
            "sleep HR, pedometer, and light proxy blocks. It is recorded as backlog only; no target associations, "
            "feature matrix, model, submission artifact, Kaggle run, or DACON submission is authorized."
        )
    else:
        sleep_proxy_status = "backlog_unverified_do_not_open"
        sleep_proxy_first_safe_probe = (
            "Backlog idea only; refresh the design report before considering even a no-artifact design lane."
        )

    return [
        {
            "rank": 0,
            "name": "approved_clean_upstream_id221_to_id226",
            "type": "score_relevant_remote_no_submit",
            "status": "open_requires_exact_approval",
            "target_focus": ["Q2", "Q3", "S3"],
            "raw_inputs": ["existing upstream base predictions", "clean rebuild packet"],
            "why_materially_new": (
                "This is not another local recombination; it rebuilds the historical ID226-style Q2/Q3/S3 signal "
                "through the guarded Kaggle/Linux no-submit path."
            ),
            "next_allowed_step": "only_after_exact_approval_phrase",
            "approval_phrase": APPROVAL_PHRASE,
            "score_relevant": True,
            "local_no_approval_allowed": False,
            "train_model_now": False,
            "create_artifact_allowed": False,
        },
        {
            "rank": 1,
            "name": "gps_path_topology_sequence_v1",
            "type": "materially_new_local_design_probe",
            "status": "closed_after_feature_matrix_and_model_smoke",
            "target_focus": ["Q3", "S3"],
            "raw_inputs": ["mGps", "mWifi", "mBle", "mActivity"],
            "why_materially_new": (
                "Closed place/routine lanes used daily entropy/novelty summaries. This would inspect route topology, "
                "home-work transition motifs, dwell-segment ordering, and late-day path disruption as sequences. "
                "That probe has now been executed through feature-matrix materialization and a fold-internal model "
                "smoke, which selected no target."
            ),
            "first_safe_probe": "completed: metadata, feature spec, feature matrix contract, matrix write, model smoke",
            "next_allowed_step": "closed_no_artifact",
            "score_relevant": False,
            "local_no_approval_allowed": False,
            "train_model_now": False,
            "create_artifact_allowed": False,
        },
        {
            "rank": 2,
            "name": "circadian_hr_light_phase_shape_v1",
            "type": "materially_new_local_design_probe",
            "status": circadian_status,
            "target_focus": ["Q3", "S3", "S2"],
            "raw_inputs": ["wHr", "wLight", "mLight", "wPedo"],
            "why_materially_new": (
                "Sleep-boundary features used windows and transitions; this would test label-free 24h phase-shape "
                "summaries such as phase delay, amplitude, curve roughness, and light/HR phase mismatch."
            ),
            "first_safe_probe": circadian_first_safe_probe,
            "next_allowed_step": circadian_next_allowed_step,
            "score_relevant": False,
            "local_no_approval_allowed": circadian_next_allowed_step
            in {
                "report_only_feature_spec_no_model",
                "report_only_feature_generation_dryrun_no_matrix",
                "report_only_feature_matrix_smoke_contract",
                "write_circadian_feature_matrix_under_contract_no_model_no_submission",
                "report_only_model_smoke_contract_no_model_no_submission",
                "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact",
            },
            "train_model_now": False,
            "create_artifact_allowed": circadian_feature_matrix_write_allowed_next,
            "submission_artifact_allowed": False,
            "feature_matrix_write_allowed_next_under_contract": circadian_feature_matrix_write_allowed_next,
            "model_smoke_allowed_next_under_contract": circadian_model_smoke_allowed_next,
        },
        {
            "rank": 3,
            "name": "usage_semantic_burst_markov_v1",
            "type": "design_hold_due_overlap",
            "status": usage_status,
            "target_focus": ["Q2", "Q3"],
            "raw_inputs": ["mUsageStats", "mScreenStatus", "mAmbience"],
            "why_materially_new": (
                "Would require app-burst Markov/session semantics rather than aggregate usage totals, but overlaps "
                "with the phase3 usage-temporal closure enough that it should not be first."
            ),
            "first_safe_probe": usage_first_safe_probe,
            "next_allowed_step": usage_next_allowed_step,
            "score_relevant": False,
            "local_no_approval_allowed": usage_local_no_approval_allowed,
            "train_model_now": False,
            "create_artifact_allowed": usage_next_allowed_step
            == "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
            "submission_artifact_allowed": False,
            "feature_matrix_materialized_under_contract": usage_semantic_burst_markov_feature_matrix_materialized,
        },
        {
            "rank": 4,
            "name": "proximity_ambience_cooccurrence_v1",
            "type": "materially_new_local_design_probe",
            "status": proximity_status,
            "target_focus": ["Q3", "S3"],
            "raw_inputs": ["mWifi", "mBle", "mAmbience", "mScreenStatus"],
            "why_materially_new": (
                "Combines WiFi/BLE proximity richness with speech/crowd/inside/outside ambience in fixed "
                "10-minute windows. This is not GPS route topology, usage Markov app semantics, ambience-only "
                "daily context, or missingness/event cadence."
            ),
            "first_safe_probe": proximity_first_safe_probe,
            "next_allowed_step": proximity_next_allowed_step,
            "score_relevant": False,
            "local_no_approval_allowed": proximity_local_no_approval_allowed,
            "train_model_now": False,
            "create_artifact_allowed": proximity_next_allowed_step
            == "write_proximity_ambience_cooccurrence_feature_matrix_under_contract_no_model_no_submission",
            "submission_artifact_allowed": False,
            "feature_matrix_write_allowed_next_under_contract": proximity_next_allowed_step
            == "write_proximity_ambience_cooccurrence_feature_matrix_under_contract_no_model_no_submission",
            "feature_matrix_materialized_under_contract": proximity_ambience_cooccurrence_feature_matrix_materialized,
        },
        {
            "rank": 5,
            "name": "phone_wearable_discordance_v1",
            "type": "materially_new_local_design_probe",
            "status": phone_wearable_status,
            "target_focus": ["Q2", "Q3", "S3"],
            "raw_inputs": ["mScreenStatus", "mACStatus", "mActivity", "wPedo", "wHr", "wLight", "mLight"],
            "why_materially_new": (
                "Uses fixed cross-device disagreement between phone rest/charging/screen/activity state and "
                "wearable activation/light/heart-rate state. This is not coverage-only adherence, sleep-boundary "
                "micro-blending, circadian phase-shape, usage Markov semantics, GPS topology, or proximity/ambience "
                "co-occurrence."
            ),
            "first_safe_probe": phone_wearable_first_safe_probe,
            "next_allowed_step": phone_wearable_next_allowed_step,
            "score_relevant": False,
            "local_no_approval_allowed": phone_wearable_local_no_approval_allowed,
            "train_model_now": False,
            "create_artifact_allowed": phone_wearable_next_allowed_step
            == "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission",
            "submission_artifact_allowed": False,
            "feature_matrix_write_allowed_next_under_contract": phone_wearable_next_allowed_step
            == "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission",
            "feature_matrix_materialized_under_contract": phone_wearable_discordance_feature_matrix_materialized,
        },
        {
            "rank": 6,
            "name": "self_supervised_subject_state_embedding_v1",
            "type": "heavy_design_only",
            "status": self_supervised_status,
            "target_focus": ["Q2", "Q3", "S3"],
            "raw_inputs": ["all raw sensor streams"],
            "why_materially_new": (
                "Could learn label-free day-state embeddings and then test a small fold-internal model, but it is "
                "heavy and risks becoming another opaque local selector."
            ),
            "first_safe_probe": self_supervised_first_safe_probe,
            "next_allowed_step": "defer_until_explicit_heavy_compute_approval",
            "score_relevant": False,
            "local_no_approval_allowed": False,
            "train_model_now": False,
            "create_artifact_allowed": False,
            "submission_artifact_allowed": False,
            "design_gate_ready": self_supervised_subject_state_embedding_design_ready,
            "feature_matrix_write_allowed_next_under_contract": False,
        },
        {
            "rank": 7,
            "name": "sleep_proxy_threshold_residual_v1",
            "type": "design_backlog_not_open",
            "status": sleep_proxy_status,
            "target_focus": ["S2", "S3", "S4", "Q1"],
            "raw_inputs": [
                "processed sleep-stage duration proxies",
                "processed sleep efficiency / fragmentation proxies",
                "sleep HR / pedometer / light proxies",
            ],
            "why_materially_new": (
                "Would test subject-normalized residuals around sleep-guideline thresholds for S2/S3/S4/Q1, but it "
                "overlaps enough with closed sleep-boundary/circadian lanes that it remains a recorded design backlog "
                "instead of a local no-approval score lane."
            ),
            "first_safe_probe": sleep_proxy_first_safe_probe,
            "next_allowed_step": "closed_design_backlog_only",
            "score_relevant": False,
            "local_no_approval_allowed": False,
            "train_model_now": False,
            "create_artifact_allowed": False,
            "submission_artifact_allowed": False,
            "feature_matrix_write_allowed_next_under_contract": False,
            "design_backlog_recorded": sleep_proxy_threshold_residual_design_ready,
            "open_design_lane_now": False,
            "compute_target_associations_now": False,
        },
    ]


def build_report() -> dict[str, Any]:
    reports = {name: read_json_report(filename) for name, filename in REQUIRED_CONTEXT_REPORTS.items()}
    self_supervised_report = read_json_report(SELF_SUPERVISED_DESIGN_REPORT)
    self_supervised_decision = decision(self_supervised_report)
    self_supervised_design_ready = bool(
        self_supervised_report.get("exists") is True
        and not self_supervised_report.get("parse_error")
        and self_supervised_decision.get("design_ready") is True
        and self_supervised_decision.get("source_family") == "self_supervised_subject_state_embedding_v1"
        and self_supervised_decision.get("target_focus") == ["Q2", "Q3", "S3"]
        and self_supervised_decision.get("materially_new_but_heavy") is True
        and self_supervised_decision.get("public_058_evidence") is False
        and self_supervised_decision.get("score_route_opened") is False
        and self_supervised_decision.get("feature_matrix_allowed_now") is False
        and self_supervised_decision.get("train_model_now") is False
        and self_supervised_decision.get("create_submission_artifact") is False
        and self_supervised_decision.get("submit_now") is False
        and self_supervised_decision.get("push_kaggle_now") is False
        and self_supervised_decision.get("run_kaggle_now") is False
    )
    status = context_status(reports)
    status["self_supervised_subject_state_embedding_design_ready"] = self_supervised_design_ready
    status["self_supervised_subject_state_embedding_design_report_present"] = bool(
        self_supervised_report.get("exists") is True and not self_supervised_report.get("parse_error")
    )
    queue = source_queue(
        status["circadian_hr_light_phase_feature_spec_ready"],
        status["circadian_hr_light_phase_feature_generation_dryrun_passed"],
        status["circadian_hr_light_phase_feature_matrix_smoke_contract_ready"],
        status["circadian_hr_light_phase_feature_matrix_materialized"],
        status["circadian_hr_light_phase_model_smoke_contract_ready"],
        status["circadian_hr_light_phase_model_smoke_completed"],
        status["circadian_hr_light_phase_model_lane_closed"],
        status["usage_semantic_burst_markov_design_ready"],
        status["usage_semantic_burst_markov_metadata_smoke_passed"],
        status["usage_semantic_burst_markov_feature_spec_ready"],
        status["usage_semantic_burst_markov_feature_generation_dryrun_passed"],
        status["usage_semantic_burst_markov_feature_matrix_smoke_contract_ready"],
        status["usage_semantic_burst_markov_feature_matrix_materialized"],
        status["usage_semantic_burst_markov_validation_stop_passed"],
        status["proximity_ambience_cooccurrence_design_ready"],
        status["proximity_ambience_cooccurrence_metadata_smoke_passed"],
        status["proximity_ambience_cooccurrence_feature_spec_ready"],
        status["proximity_ambience_cooccurrence_feature_generation_dryrun_passed"],
        status["proximity_ambience_cooccurrence_feature_matrix_smoke_contract_ready"],
        status["proximity_ambience_cooccurrence_feature_matrix_materialized"],
        status["proximity_ambience_cooccurrence_validation_stop_passed"],
        status["phone_wearable_discordance_design_ready"],
        status["phone_wearable_discordance_feature_spec_ready"],
        status["phone_wearable_discordance_feature_generation_dryrun_passed"],
        status["phone_wearable_discordance_feature_matrix_smoke_contract_ready"],
        status["phone_wearable_discordance_feature_matrix_materialized"],
        status["phone_wearable_discordance_validation_stop_passed"],
        status["self_supervised_subject_state_embedding_design_ready"],
        status["sleep_proxy_threshold_residual_design_ready"],
    )
    next_local = next(
        (
            item
            for item in queue
            if item["local_no_approval_allowed"]
            and item["next_allowed_step"]
            in {
                "report_only_feature_spec_no_model",
                "report_only_feature_generation_dryrun_no_matrix",
                "report_only_feature_matrix_smoke_contract",
                "write_circadian_feature_matrix_under_contract_no_model_no_submission",
                "report_only_model_smoke_contract_no_model_no_submission",
                "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact",
                "report_only_usage_semantic_burst_markov_design_no_model",
                "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features",
                "report_only_usage_semantic_burst_markov_feature_spec_no_matrix",
                "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix",
                "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract",
                "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
                "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
                "report_only_proximity_ambience_cooccurrence_design_no_model",
                "proximity_ambience_cooccurrence_metadata_smoke_no_features",
                "report_only_proximity_ambience_cooccurrence_feature_spec_no_matrix",
                "report_only_proximity_ambience_cooccurrence_feature_generation_dryrun_no_matrix",
                "report_only_proximity_ambience_cooccurrence_feature_matrix_smoke_contract",
                "write_proximity_ambience_cooccurrence_feature_matrix_under_contract_no_model_no_submission",
                "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
                "report_only_phone_wearable_discordance_design_no_model",
                "report_only_phone_wearable_discordance_feature_spec_no_matrix",
                "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix",
                "report_only_phone_wearable_discordance_feature_matrix_smoke_contract",
                "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission",
                "validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
            }
        ),
        None,
    )
    next_no_approval_step = next_local["name"] if next_local else "none_local_no_approval_lanes_closed"
    next_no_approval_step_type = (
        next_local["next_allowed_step"]
        if next_local
        else "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval"
    )
    blockers = closed_repeat_lanes()
    ready = (
        status["all_context_reports_present"]
        and not status["no_approval_score_route_exists"]
        and status["only_score_relevant_open_route"] == SCORE_ROUTE
        and status["q3_s3_still_uncovered"]
        and status["gps_path_topology_model_lane_closed"]
        and status["circadian_hr_light_phase_metadata_passed"]
        and status["sleep_proxy_threshold_residual_design_ready"]
    )
    if status["phone_wearable_discordance_validation_stop_passed"]:
        local_next_action = (
            "The phone/wearable discordance feature matrix validation stop passed. Local no-approval score work "
            "remains closed; the sleep-proxy threshold residual idea is recorded as closed design backlog only. Wait "
            "for exact Kaggle upstream rebuild no-submit approval or a genuinely new report-only source design. Do "
            "not train a model, create ID342, run Kaggle, create a submission artifact, modify the submission log, or "
            "submit."
        )
    elif status["phone_wearable_discordance_feature_matrix_materialized"]:
        local_next_action = (
            "The phone/wearable discordance feature matrix is registered under contract. Without exact Kaggle "
            "approval, stop at deterministic schema validation; do not train a model, create ID342, run Kaggle, "
            "create a submission artifact, modify the submission log, or submit."
        )
    elif status["phone_wearable_discordance_feature_matrix_smoke_contract_ready"]:
        local_next_action = (
            "The phone/wearable discordance feature-matrix smoke contract is ready. Without exact Kaggle approval, "
            "the next local step may only write exactly one registered label-free feature matrix plus schema report, "
            "then stop. Do not train a model, create ID342, run Kaggle, create a submission artifact, modify the "
            "submission log, or submit."
        )
    elif status["phone_wearable_discordance_feature_generation_dryrun_passed"]:
        local_next_action = (
            "The phone/wearable discordance in-memory feature-generation dry run passed. Without exact Kaggle "
            "approval, the next local step may only be a report-only feature-matrix smoke contract. Do not write "
            "the feature matrix, train a model, create ID342, run Kaggle, create a submission artifact, modify "
            "the submission log, or submit."
        )
    elif status["phone_wearable_discordance_feature_spec_ready"]:
        local_next_action = (
            "The phone/wearable discordance feature spec is ready. Without exact Kaggle approval, the next local "
            "step may only be an in-memory feature-generation dry run. Do not write a feature matrix, train a "
            "model, create ID342, run Kaggle, create a submission artifact, modify the submission log, or submit."
        )
    elif status["phone_wearable_discordance_design_ready"]:
        local_next_action = (
            "The phone/wearable discordance design is ready. Without exact Kaggle approval, the next local step may "
            "only be a report-only feature spec. Do not write a feature matrix, train a model, create ID342, run "
            "Kaggle, create a submission artifact, modify the submission log, or submit."
        )
    elif status["proximity_ambience_cooccurrence_validation_stop_passed"]:
        local_next_action = (
            "The proximity/ambience feature matrix validation stop passed. Without exact Kaggle approval, the next "
            "local step may only be a report-only phone/wearable discordance design. Do not write a feature matrix, "
            "train a model, create ID342, run Kaggle, create a submission artifact, modify the submission log, or submit."
        )
    elif status["proximity_ambience_cooccurrence_feature_matrix_materialized"]:
        local_next_action = (
            "The proximity/ambience co-occurrence feature matrix is registered under contract. Without exact Kaggle "
            "approval, stop at deterministic schema validation; do not train a model, create ID342, run Kaggle, "
            "create a submission artifact, modify the submission log, or submit."
        )
    elif status["proximity_ambience_cooccurrence_feature_matrix_smoke_contract_ready"]:
        local_next_action = (
            "The proximity/ambience co-occurrence feature-matrix smoke contract is ready. Without exact Kaggle "
            "approval, the next local step may only write exactly one registered label-free feature matrix plus "
            "schema report, then stop. Do not train a model, create ID342, run Kaggle, or submit."
        )
    elif status["proximity_ambience_cooccurrence_feature_generation_dryrun_passed"]:
        local_next_action = (
            "The proximity/ambience co-occurrence in-memory dry run passed. Without exact Kaggle approval, the next "
            "local step may only be a report-only feature-matrix smoke contract. Do not write a feature matrix, "
            "train a model, create ID342, run Kaggle, or submit."
        )
    elif status["proximity_ambience_cooccurrence_feature_spec_ready"]:
        local_next_action = (
            "The proximity/ambience co-occurrence feature spec is ready. Without exact Kaggle approval, the next "
            "local step may only be an in-memory feature-generation dry run that checks 700-row alignment, finite "
            "bounded columns, and privacy guards. Do not write a feature matrix, train a model, create ID342, run "
            "Kaggle, or submit."
        )
    elif status["proximity_ambience_cooccurrence_metadata_smoke_passed"]:
        local_next_action = (
            "The proximity/ambience metadata smoke passed. Without exact Kaggle approval, the next local step may "
            "only be a report-only feature spec. Do not write a feature matrix, train a model, create ID342, run "
            "Kaggle, or submit."
        )
    elif status["proximity_ambience_cooccurrence_design_ready"]:
        local_next_action = (
            "The proximity/ambience co-occurrence design is ready. Without exact Kaggle approval, the next local "
            "step may only be a metadata/co-occurrence smoke. Do not generate features, train a model, create "
            "ID342, run Kaggle, or submit."
        )
    elif status["usage_semantic_burst_markov_validation_stop_passed"]:
        local_next_action = (
            "The usage semantic burst/Markov matrix validation stop passed. Without exact Kaggle approval, the next "
            "local step may only be a report-only proximity/ambience co-occurrence design. Do not generate features, "
            "train a model, create ID342, run Kaggle, or submit."
        )
    elif status["usage_semantic_burst_markov_feature_matrix_materialized"]:
        local_next_action = (
            "The usage semantic burst/Markov feature matrix is registered under contract. Without exact Kaggle "
            "approval, stop at schema validation: do not train a model, create ID342, run Kaggle, create a "
            "submission artifact, modify the submission log, or submit."
        )
    elif status["usage_semantic_burst_markov_feature_matrix_smoke_contract_ready"]:
        local_next_action = (
            "The usage semantic burst/Markov feature-matrix smoke contract is ready. Without exact Kaggle approval, "
            "the next local step may only write exactly one registered label-free feature matrix plus schema report, "
            "then stop. Do not train a model, create ID342, run Kaggle, or submit."
        )
    elif status["usage_semantic_burst_markov_feature_generation_dryrun_passed"]:
        local_next_action = (
            "The usage semantic burst/Markov in-memory feature-generation dry run passed. Without exact Kaggle "
            "approval, the next local step may only be a report-only feature-matrix smoke contract. Do not write "
            "the feature matrix, train a model, create ID342, run Kaggle, or submit."
        )
    elif status["usage_semantic_burst_markov_feature_spec_ready"]:
        local_next_action = (
            "The usage semantic burst/Markov feature spec is ready. Without exact Kaggle approval, the next local "
            "step may only be a report-only in-memory feature-generation dry run that checks exact row alignment, "
            "bounded columns, missingness flags, and leakage guards. Do not write a feature matrix, train a model, "
            "create ID342, run Kaggle, or submit."
        )
    elif status["usage_semantic_burst_markov_metadata_smoke_passed"]:
        local_next_action = (
            "The usage semantic burst/Markov metadata smoke passed. Without exact Kaggle approval, the next local "
            "step may only be a report-only feature spec with exact predeclared states and leakage guards. Do not "
            "generate a feature matrix, train a model, create ID342, run Kaggle, or submit."
        )
    elif status["usage_semantic_burst_markov_design_ready"]:
        local_next_action = (
            "The usage semantic burst/Markov design is ready. Without exact Kaggle approval, the next local step may "
            "only be a metadata/cardinality smoke over app states, Markov pairs, sessions, and screen coupling. Do "
            "not generate features, train a model, create ID342, run Kaggle, or submit."
        )
    elif status["circadian_hr_light_phase_model_lane_closed"]:
        local_next_action = (
            "The circadian HR/light lane is closed after a bounded model smoke selected no targets. Without exact "
            "Kaggle approval, the next local step may only be a report-only usage-semantic burst/Markov design. "
            "Do not generate features, train a model, create ID342, run Kaggle, or submit."
        )
    elif status["circadian_hr_light_phase_model_smoke_contract_ready"]:
        local_next_action = (
            "Without exact Kaggle approval, do not replay closed GPS or OOF-combo lanes. If continuing locally, "
            "run only the bounded in-memory circadian model smoke under contract, then stop. Do not persist models, "
            "write OOF/test predictions, create ID342, run Kaggle, or submit."
        )
    elif status["circadian_hr_light_phase_feature_matrix_materialized"]:
        local_next_action = (
            "Without exact Kaggle approval, do not replay closed GPS or OOF-combo lanes. The circadian feature "
            "matrix is registered as label-free infrastructure only. If continuing locally, write only a "
            "report-only model-smoke contract; do not train the model, create ID342, run Kaggle, or submit."
        )
    elif status["circadian_hr_light_phase_feature_matrix_smoke_contract_ready"]:
        local_next_action = (
            "Without exact Kaggle approval, do not replay closed GPS or OOF-combo lanes. If continuing locally, "
            "write only the registered circadian feature matrix and schema report under the smoke contract, then "
            "stop for validation. Do not create ID342, train a fold model, run Kaggle, or submit."
        )
    elif status["circadian_hr_light_phase_feature_generation_dryrun_passed"]:
        local_next_action = (
            "Without exact Kaggle approval, do not replay closed GPS or OOF-combo lanes. If continuing locally, "
            "write only a report-only circadian feature-matrix smoke contract. Do not write the matrix itself, "
            "create ID342, train a fold model, or submit."
        )
    elif status["circadian_hr_light_phase_feature_spec_ready"]:
        local_next_action = (
            "Without exact Kaggle approval, do not replay closed GPS or OOF-combo lanes. If continuing locally, "
            "run only a report-only circadian HR/light feature-generation dry run that checks row alignment, "
            "coverage, and schema bounds. Do not write a feature matrix, create ID342, train a fold model, or submit."
        )
    else:
        local_next_action = (
            "Without exact Kaggle approval, do not replay the closed GPS topology lane. If continuing locally, "
            "run only the report-only circadian HR/light phase-shape feature spec. Do not create "
            "ID342, do not train a fold model, and do not reopen closed micro-blend lanes."
        )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_material_new_source_queue",
        "report_only": True,
        "context_status": status,
        "closed_repeat_lanes": blockers,
        "source_queue": queue,
        "decision": {
            "material_new_source_queue_ready": ready,
            "local_no_approval_score_route_exists": False,
            "only_score_relevant_open_route": SCORE_ROUTE,
            "requires_explicit_kaggle_approval": True,
            "approval_phrase": APPROVAL_PHRASE,
            "next_no_approval_step": next_no_approval_step,
            "next_no_approval_step_type": next_no_approval_step_type,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "train_model_now": False,
            "next_action": local_next_action,
        },
        "context_reports": {
            name: {
                "path": report.get("path"),
                "exists": report.get("exists", False),
                "parse_error": report.get("parse_error"),
            }
            for name, report in reports.items()
        }
        | {
            "self_supervised_subject_state_embedding_design": {
                "path": self_supervised_report.get("path"),
                "exists": self_supervised_report.get("exists", False),
                "parse_error": self_supervised_report.get("parse_error"),
            }
        },
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 materially-new source queue",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no predictions, submission CSV, base prediction artifact, "
            "notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Queue ready: `{dec['material_new_source_queue_ready']}`",
        f"- Local no-approval score route exists: `{dec['local_no_approval_score_route_exists']}`",
        f"- Only score-relevant open route: `{dec['only_score_relevant_open_route']}`",
        f"- Requires explicit Kaggle approval: `{dec['requires_explicit_kaggle_approval']}`",
        f"- Approval phrase: `{dec['approval_phrase']}`",
        f"- Next no-approval step: `{dec['next_no_approval_step']}`",
        f"- Next no-approval step type: `{dec['next_no_approval_step_type']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Context status",
        "",
    ]
    for key, value in report["context_status"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Closed/repeat lanes not to reopen", ""])
    lines.append("| lane | status | repeat risk | do not repeat as | reason |")
    lines.append("|---|---|---|---|---|")
    for lane in report["closed_repeat_lanes"]:
        lines.append(
            f"| `{lane['lane']}` | `{lane['status']}` | `{lane['repeat_risk']}` | "
            f"{lane['do_not_repeat_as']} | {lane['reason']} |"
        )
    lines.extend(["", "## Material source queue", ""])
    lines.append(
        "| rank | source family | status | targets | local no-approval allowed | score relevant | next allowed step |"
    )
    lines.append("|---:|---|---|---|---|---|---|")
    for item in report["source_queue"]:
        lines.append(
            f"| {item['rank']} | `{item['name']}` | `{item['status']}` | "
            f"`{','.join(item['target_focus'])}` | `{item['local_no_approval_allowed']}` | "
            f"`{item['score_relevant']}` | `{item['next_allowed_step']}` |"
        )
    lines.extend(["", "## Why the next local step is limited", ""])
    if dec["next_no_approval_step_type"] == "report_only_feature_matrix_smoke_contract":
        limit_note = (
            "The next local step is deliberately only a report-only circadian feature-matrix smoke contract. "
            "The dry run proved 700-row in-memory alignment and a bounded 48-column schema, but it is still not "
            "score evidence. Q3/S3 remain uncovered, the GPS topology lane is closed after its model smoke, and "
            "every current local no-approval score route is closed. A feature parquet, fold model, or ID342 would "
            "violate the current frontier unless that later contract explicitly permits it."
        )
    elif dec["next_no_approval_step_type"] == "write_circadian_feature_matrix_under_contract_no_model_no_submission":
        limit_note = (
            "The next local step is deliberately only the registered circadian feature-matrix writer. The smoke "
            "contract permits a single label-free parquet plus schema report, then a hard stop for validation. "
            "This still is not score evidence: no fold model, submission CSV, submission-log row, Kaggle run, "
            "DACON submission, or ID342 may be created from the matrix write itself."
        )
    elif dec["next_no_approval_step_type"] == "report_only_model_smoke_contract_no_model_no_submission":
        limit_note = (
            "The circadian feature matrix is now materialized, but this is still not score evidence. The only "
            "local no-approval follow-up is a report-only model-smoke contract that defines gates before any "
            "fold model is trained. No model, submission CSV, submission-log row, Kaggle run, DACON submission, "
            "or ID342 is authorized by the matrix alone."
        )
    elif dec["next_no_approval_step_type"] == "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact":
        limit_note = (
            "The circadian model-smoke contract is ready, but it authorizes only a bounded in-memory fold smoke. "
            "The run may report targetwise split losses, but it must not persist models, write OOF/test "
            "predictions, create a submission CSV, modify the submission log, run Kaggle, submit to DACON, or "
            "create ID342."
        )
    elif dec["next_no_approval_step_type"] == "report_only_usage_semantic_burst_markov_design_no_model":
        limit_note = (
            "The circadian HR/light feature lane is closed after its model smoke selected no targets. The next "
            "local no-approval step is deliberately only report-only usage-semantic burst/Markov design. Feature "
            "generation, model training, ID342, Kaggle, and DACON submission remain closed."
        )
    elif dec["next_no_approval_step_type"] == "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features":
        limit_note = (
            "The usage semantic burst/Markov design is ready, but this still permits only a metadata/cardinality "
            "smoke over fixed app states, Markov transition coverage, session bursts, and screen coupling. Feature "
            "generation, model training, ID342, Kaggle, and DACON submission remain closed."
        )
    elif dec["next_no_approval_step_type"] == "report_only_usage_semantic_burst_markov_feature_spec_no_matrix":
        limit_note = (
            "The usage semantic burst/Markov metadata smoke passed, but it authorizes only a report-only feature "
            "spec with exact predeclared states and leakage guards. No feature matrix, model, submission CSV, "
            "submission-log row, Kaggle run, DACON submission, or ID342 is authorized."
        )
    elif (
        dec["next_no_approval_step_type"]
        == "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix"
    ):
        limit_note = (
            "The usage semantic burst/Markov feature spec is ready, but it authorizes only a report-only in-memory "
            "feature-generation dry run. The dry run may check 700-row alignment, bounded feature columns, "
            "missingness flags, and leakage guards, but it must not write a feature matrix, train a model, create "
            "a submission CSV, modify the submission log, run Kaggle, submit to DACON, or create ID342."
        )
    elif dec["next_no_approval_step_type"] == "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract":
        limit_note = (
            "The usage semantic burst/Markov in-memory dry run passed, but it authorizes only a report-only "
            "feature-matrix smoke contract. The contract may decide whether a single registered label-free matrix "
            "writer is reviewable later, but no matrix, model, submission CSV, submission-log row, Kaggle run, "
            "DACON submission, or ID342 is authorized by the dry run itself."
        )
    elif (
        dec["next_no_approval_step_type"]
        == "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission"
    ):
        limit_note = (
            "The usage semantic burst/Markov feature-matrix smoke contract is ready, but it authorizes only one "
            "registered label-free matrix writer plus schema report. The writer must stop immediately after schema "
            "validation and must not train a model, create a submission CSV, modify the submission log, run Kaggle, "
            "submit to DACON, or create ID342."
        )
    elif (
        dec["next_no_approval_step_type"]
        == "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
    ):
        limit_note = (
            "The usage semantic burst/Markov feature matrix is registered under contract and has a schema report, "
            "but this is still infrastructure rather than leaderboard evidence. The no-approval lane stops here: "
            "no model, prediction file, submission CSV, submission-log row, Kaggle run, DACON submission, or ID342 "
            "is authorized from this matrix alone."
        )
    elif dec["next_no_approval_step_type"] == "report_only_phone_wearable_discordance_feature_spec_no_matrix":
        limit_note = (
            "The phone/wearable discordance design is ready, but it authorizes only a report-only feature spec. "
            "The design observed cross-device disagreement bins and adequate raw coverage, yet it is not score "
            "evidence. No feature matrix, model, prediction file, submission CSV, submission-log row, Kaggle run, "
            "DACON submission, or ID342 is authorized."
        )
    elif (
        dec["next_no_approval_step_type"]
        == "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix"
    ):
        limit_note = (
            "The phone/wearable discordance feature spec is ready, but it authorizes only an in-memory "
            "feature-generation dry run. The dry run may check exact 700-row alignment, bounded feature columns, "
            "finite values, and leakage/action guards, but it must not write a feature matrix, train a model, "
            "create a submission CSV, modify the submission log, run Kaggle, submit to DACON, or create ID342."
        )
    elif dec["next_no_approval_step_type"] == "report_only_phone_wearable_discordance_feature_matrix_smoke_contract":
        limit_note = (
            "The phone/wearable discordance in-memory dry run passed, but it authorizes only a report-only "
            "feature-matrix smoke contract. The contract may decide whether a future single label-free matrix "
            "writer is reviewable; the dry run itself is still not score evidence and authorizes no matrix, model, "
            "submission CSV, submission-log row, Kaggle run, DACON submission, or ID342."
        )
    elif (
        dec["next_no_approval_step_type"]
        == "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission"
    ):
        limit_note = (
            "The phone/wearable discordance feature-matrix smoke contract is ready, but it authorizes only one "
            "registered label-free matrix writer plus schema report. The writer must stop immediately after schema "
            "validation and must not train a model, create a submission CSV, modify the submission log, run Kaggle, "
            "submit to DACON, or create ID342."
        )
    elif (
        dec["next_no_approval_step_type"]
        == "validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
    ):
        limit_note = (
            "The phone/wearable discordance feature matrix is registered under contract and has a schema report, "
            "but this is still label-free infrastructure rather than leaderboard evidence. The no-approval lane "
            "must stop at deterministic schema validation: no model, prediction file, submission CSV, submission-log "
            "row, Kaggle run, DACON submission, or ID342 is authorized from this matrix alone."
        )
    elif dec["next_no_approval_step_type"] == "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval":
        limit_note = (
            "The phone/wearable discordance feature matrix validation stop has passed, and no local no-approval "
            "lane remains open. The only score-relevant route is the explicit-approved Kaggle upstream rebuild "
            "no-submit path; otherwise the workspace may continue only with a new report-only source design. "
            "No model, prediction file, submission CSV, submission-log row, Kaggle run, DACON submission, or ID342 "
            "is authorized from the validated label-free matrices."
        )
    elif dec["next_no_approval_step_type"] == "report_only_phone_wearable_discordance_design_no_model":
        limit_note = (
            "The next local step is deliberately only a report-only phone/wearable discordance design. It may check "
            "raw coverage and fixed cross-device disagreement feasibility, but must not write a feature matrix, train, "
            "create ID342, run Kaggle, or submit."
        )
    elif dec["next_no_approval_step_type"] == "report_only_feature_generation_dryrun_no_matrix":
        limit_note = (
            "The next local step is deliberately only a report-only circadian feature-generation dry run with no "
            "feature-matrix write. Q3/S3 remain uncovered, the GPS topology lane is now closed after its model "
            "smoke, the circadian metadata/spec reports are feasible but not score evidence, and every current "
            "local no-approval score route is closed. A new fold model, feature parquet, or ID342 would violate "
            "the current frontier unless a later contract separately proves a materially different, transfer-safe "
            "source family."
        )
    else:
        limit_note = (
            "The next local step is deliberately only report-only design work. Q3/S3 remain uncovered, the GPS "
            "topology lane is closed after its model smoke, and every current local no-approval score route is "
            "closed. A new fold model, feature parquet, or ID342 would violate the current frontier."
        )
    lines.append(limit_note)
    lines.extend(["", "## Context reports", "", "```json", json.dumps(report["context_reports"], indent=2), "```", ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output.with_suffix(".md").write_text(markdown(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    write_report(report, args.output)
    dec = report["decision"]
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    print(
        json.dumps(
            {
                "material_new_source_queue_ready": dec["material_new_source_queue_ready"],
                "next_no_approval_step": dec["next_no_approval_step"],
                "only_score_relevant_open_route": dec["only_score_relevant_open_route"],
                "create_submission_artifact": dec["create_submission_artifact"],
                "submit_now": dec["submit_now"],
                "push_kaggle_now": dec["push_kaggle_now"],
                "run_kaggle_now": dec["run_kaggle_now"],
                "train_model_now": dec["train_model_now"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
