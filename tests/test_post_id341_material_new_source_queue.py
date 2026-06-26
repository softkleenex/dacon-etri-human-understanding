from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_material_new_source_queue import build_report, write_report


def test_material_new_source_queue_is_report_only_and_keeps_score_route() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["material_new_source_queue_ready"] is True
    assert dec["local_no_approval_score_route_exists"] is False
    assert dec["only_score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert dec["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert dec["train_model_now"] is False


def test_material_new_source_queue_closes_old_lanes_then_closes_phone_wearable_after_validation_stop() -> None:
    report = build_report()
    dec = report["decision"]
    queue = report["source_queue"]
    by_name = {item["name"]: item for item in queue}

    assert dec["next_no_approval_step"] == "none_local_no_approval_lanes_closed"
    assert dec["next_no_approval_step_type"] == "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval"
    assert report["context_status"]["gps_path_topology_model_lane_closed"] is True
    assert report["context_status"]["circadian_hr_light_phase_metadata_passed"] is True
    assert report["context_status"]["circadian_hr_light_phase_feature_spec_ready"] is True
    assert report["context_status"]["circadian_hr_light_phase_feature_generation_dryrun_passed"] is True
    assert report["context_status"]["circadian_hr_light_phase_feature_matrix_smoke_contract_ready"] is True
    assert report["context_status"]["circadian_hr_light_phase_feature_matrix_materialized"] is True
    assert report["context_status"]["circadian_hr_light_phase_model_smoke_contract_ready"] is True
    assert report["context_status"]["circadian_hr_light_phase_model_smoke_completed"] is True
    assert report["context_status"]["circadian_hr_light_phase_model_lane_closed"] is True
    assert report["context_status"]["usage_semantic_burst_markov_design_ready"] is True
    assert report["context_status"]["usage_semantic_burst_markov_metadata_smoke_passed"] is True
    assert report["context_status"]["usage_semantic_burst_markov_feature_spec_ready"] is True
    assert report["context_status"]["usage_semantic_burst_markov_feature_generation_dryrun_passed"] is True
    assert report["context_status"]["usage_semantic_burst_markov_feature_matrix_smoke_contract_ready"] is True
    assert report["context_status"]["usage_semantic_burst_markov_feature_matrix_materialized"] is True
    assert report["context_status"]["usage_semantic_burst_markov_validation_stop_passed"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_design_ready"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_metadata_smoke_passed"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_feature_spec_ready"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_feature_generation_dryrun_passed"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_feature_matrix_smoke_contract_ready"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_feature_matrix_materialized"] is True
    assert report["context_status"]["proximity_ambience_cooccurrence_validation_stop_passed"] is True
    assert report["context_status"]["phone_wearable_discordance_design_ready"] is True
    assert report["context_status"]["phone_wearable_discordance_feature_spec_ready"] is True
    assert report["context_status"]["phone_wearable_discordance_feature_generation_dryrun_passed"] is True
    assert report["context_status"]["phone_wearable_discordance_feature_matrix_smoke_contract_ready"] is True
    assert report["context_status"]["phone_wearable_discordance_feature_matrix_materialized"] is True
    assert report["context_status"]["phone_wearable_discordance_validation_stop_passed"] is True
    assert report["context_status"]["sleep_proxy_threshold_residual_design_ready"] is True
    assert report["context_status"]["self_supervised_subject_state_embedding_design_report_present"] is True
    assert report["context_status"]["self_supervised_subject_state_embedding_design_ready"] is True
    assert by_name["gps_path_topology_sequence_v1"]["status"] == "closed_after_feature_matrix_and_model_smoke"
    assert by_name["gps_path_topology_sequence_v1"]["local_no_approval_allowed"] is False
    assert by_name["gps_path_topology_sequence_v1"]["train_model_now"] is False
    assert by_name["gps_path_topology_sequence_v1"]["create_artifact_allowed"] is False
    assert set(by_name["gps_path_topology_sequence_v1"]["target_focus"]) == {"Q3", "S3"}
    assert by_name["circadian_hr_light_phase_shape_v1"]["local_no_approval_allowed"] is False
    assert (
        by_name["circadian_hr_light_phase_shape_v1"]["status"]
        == "closed_after_feature_matrix_and_model_smoke_no_signal"
    )
    assert by_name["circadian_hr_light_phase_shape_v1"]["next_allowed_step"] == "closed_no_artifact"
    assert by_name["circadian_hr_light_phase_shape_v1"]["feature_matrix_write_allowed_next_under_contract"] is False
    assert by_name["circadian_hr_light_phase_shape_v1"]["model_smoke_allowed_next_under_contract"] is False
    assert by_name["circadian_hr_light_phase_shape_v1"]["submission_artifact_allowed"] is False
    assert by_name["usage_semantic_burst_markov_v1"]["local_no_approval_allowed"] is False
    assert by_name["usage_semantic_burst_markov_v1"]["status"] == "validated_stop_completed_no_model"
    assert (
        by_name["usage_semantic_burst_markov_v1"]["next_allowed_step"]
        == "closed_stop_or_wait_for_explicit_kaggle_approval"
    )
    assert by_name["proximity_ambience_cooccurrence_v1"]["local_no_approval_allowed"] is False
    assert by_name["proximity_ambience_cooccurrence_v1"]["status"] == "validated_stop_completed_no_model"
    assert (
        by_name["proximity_ambience_cooccurrence_v1"]["next_allowed_step"]
        == "closed_stop_or_wait_for_explicit_kaggle_approval"
    )
    assert by_name["proximity_ambience_cooccurrence_v1"]["create_artifact_allowed"] is False
    assert by_name["proximity_ambience_cooccurrence_v1"]["feature_matrix_write_allowed_next_under_contract"] is False
    assert by_name["proximity_ambience_cooccurrence_v1"]["submission_artifact_allowed"] is False
    assert by_name["phone_wearable_discordance_v1"]["local_no_approval_allowed"] is False
    assert by_name["phone_wearable_discordance_v1"]["status"] == "validated_stop_completed_no_model"
    assert (
        by_name["phone_wearable_discordance_v1"]["next_allowed_step"]
        == "closed_stop_or_wait_for_explicit_kaggle_approval"
    )
    assert by_name["phone_wearable_discordance_v1"]["score_relevant"] is False
    assert by_name["phone_wearable_discordance_v1"]["train_model_now"] is False
    assert by_name["phone_wearable_discordance_v1"]["create_artifact_allowed"] is False
    assert by_name["phone_wearable_discordance_v1"]["submission_artifact_allowed"] is False
    assert by_name["phone_wearable_discordance_v1"]["feature_matrix_write_allowed_next_under_contract"] is False
    assert by_name["phone_wearable_discordance_v1"]["feature_matrix_materialized_under_contract"] is True
    assert by_name["self_supervised_subject_state_embedding_v1"]["status"] == "design_gate_ready_heavy_deferred"
    assert by_name["self_supervised_subject_state_embedding_v1"]["design_gate_ready"] is True
    assert by_name["self_supervised_subject_state_embedding_v1"]["local_no_approval_allowed"] is False
    assert by_name["self_supervised_subject_state_embedding_v1"]["train_model_now"] is False
    assert by_name["self_supervised_subject_state_embedding_v1"]["create_artifact_allowed"] is False
    assert by_name["self_supervised_subject_state_embedding_v1"]["submission_artifact_allowed"] is False
    assert (
        by_name["self_supervised_subject_state_embedding_v1"]["feature_matrix_write_allowed_next_under_contract"]
        is False
    )
    assert by_name["sleep_proxy_threshold_residual_v1"]["status"] == "design_backlog_recorded_closed_no_local_lane"
    assert by_name["sleep_proxy_threshold_residual_v1"]["target_focus"] == ["S2", "S3", "S4", "Q1"]
    assert by_name["sleep_proxy_threshold_residual_v1"]["next_allowed_step"] == "closed_design_backlog_only"
    assert by_name["sleep_proxy_threshold_residual_v1"]["score_relevant"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["local_no_approval_allowed"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["train_model_now"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["create_artifact_allowed"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["submission_artifact_allowed"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["feature_matrix_write_allowed_next_under_contract"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["design_backlog_recorded"] is True
    assert by_name["sleep_proxy_threshold_residual_v1"]["open_design_lane_now"] is False
    assert by_name["sleep_proxy_threshold_residual_v1"]["compute_target_associations_now"] is False
    assert by_name["approved_clean_upstream_id221_to_id226"]["score_relevant"] is True
    assert by_name["approved_clean_upstream_id221_to_id226"]["local_no_approval_allowed"] is False


def test_material_new_source_queue_records_repeat_lanes_and_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "queue.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    repeat_lanes = {lane["lane"] for lane in written["closed_repeat_lanes"]}

    assert "local_oof_combo_or_second_stage_calibration" in repeat_lanes
    assert "sleep_boundary_micro_blend" in repeat_lanes
    assert "gps_path_topology_sequence_v1" in repeat_lanes
    assert "creates no predictions" in md
    assert "Train model now: `False`" in md
    assert "phone/wearable discordance feature matrix validation stop has passed" in md
    assert "validated label-free matrices" in md
    assert "self_supervised_subject_state_embedding_design" in md
    assert "sleep_proxy_threshold_residual_design" in md
    assert "sleep-proxy threshold residual idea is recorded as closed design backlog only" in md
    assert "train a model" in md.lower()
    assert "id342" in md.lower()
