from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_design import build_report as build_design
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun import (
    build_report as build_dryrun,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract import (
    build_report as build_matrix_contract,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop import (
    build_report as build_validation_stop,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_spec import build_report as build_spec
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_metadata_smoke import build_report as build_metadata
from scripts.write_post_id341_proximity_ambience_cooccurrence_feature_matrix import (
    MATRIX_PATH,
    SCHEMA_REPORT_PATH,
    build_feature_matrix,
    validate_feature_matrix,
)


def test_proximity_ambience_design_is_report_only_and_ready() -> None:
    report = build_design()
    dec = report["decision"]

    assert dec["design_ready"] is True
    assert dec["source_family"] == "proximity_ambience_cooccurrence_v1"
    assert dec["target_focus"] == ["Q3", "S3"]
    assert dec["selected_targets"] == []
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert report["context_status"]["usage_validation_stop_passed"] is True
    assert report["no_submission_log_row_created"] is True


def test_proximity_ambience_metadata_smoke_has_cross_sensor_coverage_and_blocks_actions() -> None:
    report = build_metadata()
    dec = report["decision"]
    profile = report["bin_profile"]
    coverage = report["coverage_by_split"]

    assert dec["metadata_smoke_passed"] is True
    assert dec["source_family"] == "proximity_ambience_cooccurrence_v1"
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert profile["proximity_social_cooccur_bins"] >= 500
    assert profile["evening_proximity_social_cooccur_bins"] >= 100
    assert profile["unique_wifi_tokens_count_only"] > 0
    assert profile["unique_ble_tokens_count_only"] > 0
    assert coverage["proximity_social_cooccur_bins"]["train"]["coverage_rate"] >= 0.40
    assert coverage["proximity_social_cooccur_bins"]["test"]["coverage_rate"] >= 0.40
    assert "BSSID" not in json.dumps(report, ensure_ascii=False)


def test_proximity_ambience_feature_spec_is_bounded_and_markdown_safe(tmp_path: Path) -> None:
    report = build_spec()
    dec = report["decision"]

    assert dec["feature_spec_ready"] is True
    assert dec["feature_count"] == 80
    assert report["schema_bounds"]["actual_feature_columns"] <= report["schema_bounds"]["max_feature_columns"]
    assert report["schema_bounds"]["all_features_prefixed"] is True
    assert report["schema_bounds"]["no_raw_identifier_features"] is True
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False

    output = tmp_path / "proxamb_spec.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert json.loads(output.read_text())["decision"]["feature_spec_ready"] is True


def test_proximity_ambience_feature_generation_dryrun_aligns_rows_and_blocks_actions() -> None:
    report = build_dryrun()
    dec = report["decision"]
    summary = report["feature_frame_summary"]
    checks = report["checks"]

    assert dec["feature_generation_dryrun_passed"] is True
    assert dec["source_family"] == "proximity_ambience_cooccurrence_v1"
    assert dec["target_focus"] == ["Q3", "S3"]
    assert dec["feature_count"] == 80
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert summary["row_count"] == 700
    assert summary["train_rows"] == 450
    assert summary["test_rows"] == 250
    assert summary["feature_column_count"] == 80
    assert summary["feature_columns_match_spec"] is True
    assert summary["duplicate_key_rows"] == 0
    assert summary["finite_all_values"] is True
    assert summary["nan_count"] == 0
    assert summary["nonzero_feature_count"] >= 60
    assert checks["privacy_guard_passed"] is True
    assert report["no_feature_matrix_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert "BSSID" not in json.dumps(report, ensure_ascii=False)


def test_proximity_ambience_feature_matrix_smoke_contract_allows_only_future_registered_matrix() -> None:
    report = build_matrix_contract()
    dec = report["decision"]
    checks = report["checks"]
    writer = report["future_writer_contract"]
    paths = report["registered_paths"]

    assert dec["feature_matrix_smoke_contract_ready"] is True
    assert dec["source_family"] == "proximity_ambience_cooccurrence_v1"
    assert dec["target_focus"] == ["Q3", "S3"]
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["feature_matrix_write_allowed_now"] is False
    assert (
        dec["feature_matrix_write_allowed_next_under_contract"]
        or dec["feature_matrix_already_materialized_under_contract"]
    )
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert checks["spec_contract_ready"] is True
    assert checks["dryrun_contract_ready"] is True
    assert checks["dryrun_schema_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert checks["privacy_contract_ready"] is True
    assert checks["does_not_write_feature_matrix"] is True
    assert paths["feature_matrix_exists_now"] in {True, False}
    assert paths["schema_report_exists_now"] in {True, False}
    assert writer["expected_row_count"] == 700
    assert writer["expected_feature_column_count"] == 80
    assert writer["model_training_allowed"] is False
    assert writer["submission_artifact_allowed"] is False
    assert report["no_feature_matrix_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_proximity_ambience_feature_matrix_writer_schema_is_label_free_and_aligned() -> None:
    assert MATRIX_PATH.exists()
    assert SCHEMA_REPORT_PATH.exists()

    frame, expected = build_feature_matrix()
    validation = validate_feature_matrix(frame, expected)
    schema = json.loads(SCHEMA_REPORT_PATH.read_text())
    dec = schema["decision"]

    assert validation["allowed_boundary_passed"] is True
    assert dec["feature_matrix_materialized"] is True
    assert dec["source_family"] == "proximity_ambience_cooccurrence_v1"
    assert dec["target_focus"] == ["Q3", "S3"]
    assert dec["row_count"] == 700
    assert dec["feature_column_count"] == 80
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert schema["artifact"]["matrix_exists"] is True
    assert schema["validation"]["allowed_boundary_passed"] is True
    assert schema["validation"]["forbidden_columns"] == []
    assert schema["no_submission_log_row_created"] is True


def test_proximity_ambience_feature_matrix_validation_stop_closes_local_lane() -> None:
    report = build_validation_stop()
    dec = report["decision"]
    checks = report["checks"]
    match = report["deterministic_rebuild"]

    assert dec["validation_stop_passed"] is True
    assert dec["source_family"] == "proximity_ambience_cooccurrence_v1"
    assert dec["target_focus"] == ["Q3", "S3"]
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert checks["matrix_shape_exact"] is True
    assert checks["schema_materialized_and_safe"] is True
    assert checks["contract_records_materialized_stop"] is True
    assert checks["matrix_validation_passed"] is True
    assert checks["deterministic_rebuild_match"] is True
    assert checks["submission_log_has_no_source_family_rows"] is True
    assert checks["no_source_family_submission_artifacts"] is True
    assert match["deterministic_rebuild_match"] is True
    assert report["safety_contract"]["model_trained"] is False
    assert report["safety_contract"]["submission_csv_created"] is False
