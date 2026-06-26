from scripts.write_post_id341_phone_wearable_discordance_feature_matrix import (
    MATRIX_PATH,
    SCHEMA_REPORT_PATH,
    build_feature_matrix,
    feature_names,
    validate_feature_matrix,
)


def test_phone_wearable_feature_matrix_writer_schema_is_label_free_and_aligned() -> None:
    assert MATRIX_PATH.exists()
    assert SCHEMA_REPORT_PATH.exists()

    frame, expected = build_feature_matrix()
    validation = validate_feature_matrix(frame, expected)

    assert validation["allowed_boundary_passed"] is True
    assert validation["row_alignment"]["row_count"] == 700
    assert validation["row_alignment"]["train_rows"] == 450
    assert validation["row_alignment"]["test_rows"] == 250
    assert validation["feature_columns"]["feature_column_count"] == 85
    assert validation["feature_columns"]["features"] == feature_names()
    assert validation["forbidden_columns"] == []
    assert validation["checks"]["all_feature_values_finite"] is True
    assert validation["checks"]["no_target_or_prediction_columns"] is True


def test_phone_wearable_schema_report_records_no_score_route() -> None:
    import json

    schema = json.loads(SCHEMA_REPORT_PATH.read_text())
    dec = schema["decision"]

    assert dec["feature_matrix_materialized"] is True
    assert dec["source_family"] == "phone_wearable_discordance_v1"
    assert dec["target_focus"] == ["Q2", "Q3", "S3"]
    assert dec["row_count"] == 700
    assert dec["feature_column_count"] == 85
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert schema["no_submission_log_row_created"] is True
