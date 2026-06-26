from scripts.diagnose_post_id341_phone_wearable_discordance_feature_matrix_validation_stop import build_report


def test_phone_wearable_feature_matrix_validation_stop_closes_local_lane() -> None:
    report = build_report()
    dec = report["decision"]
    checks = report["checks"]
    match = report["deterministic_rebuild"]

    assert dec["validation_stop_passed"] is True
    assert dec["source_family"] == "phone_wearable_discordance_v1"
    assert dec["target_focus"] == ["Q2", "Q3", "S3"]
    assert dec["selected_targets"] == []
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert checks["matrix_shape_exact"] is True
    assert checks["schema_materialized_and_safe"] is True
    assert checks["contract_records_materialized_stop"] is True
    assert checks["queue_records_safe_boundary"] is True
    assert checks["matrix_validation_passed"] is True
    assert checks["deterministic_rebuild_match"] is True
    assert checks["submission_log_has_no_source_family_rows"] is True
    assert checks["no_source_family_submission_artifacts"] is True
    assert match["deterministic_rebuild_match"] is True
    assert report["safety_contract"]["model_trained"] is False
    assert report["safety_contract"]["submission_csv_created"] is False
