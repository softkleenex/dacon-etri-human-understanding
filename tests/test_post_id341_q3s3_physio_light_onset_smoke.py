import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports" / "post_id341_q3s3_physio_light_onset_smoke_20260618.json"


def load_report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_q3s3_physio_light_onset_smoke_is_report_only_and_closed():
    report = load_report()
    decision = report["decision"]

    assert decision["model_smoke_completed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["selected_targets"] == []
    assert decision["q3_signal_found"] is False
    assert decision["s3_signal_found"] is False
    assert decision["next_local_allowed_step"] == "close_physio_light_onset_lane"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q3s3_physio_light_onset_smoke_has_full_key_coverage_and_no_artifacts():
    report = load_report()
    coverage = report["feature_coverage"]

    assert coverage["n_train_rows"] == 450
    assert coverage["n_test_rows"] == 250
    assert coverage["raw_feature_columns"] > 50
    assert coverage["all_null_raw_features"] == []
    assert coverage["train_any_nonnull_row_share"] == 1.0
    assert coverage["test_any_nonnull_row_share"] == 1.0

    assert report["no_full_test_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_run_made"] is True
    assert report["no_dacon_submission_made"] is True
