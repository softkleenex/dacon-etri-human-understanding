import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports" / "post_id341_q3s3_local_lane_closure_20260618.json"


def load_report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_q3s3_local_lane_closure_blocks_local_artifacts_and_points_upstream():
    report = load_report()
    decision = report["decision"]
    evidence = report["evidence"]

    assert evidence["q3s3_physio_light_onset_closed"] is True
    assert evidence["q3s3_calendar_sleep_prior_closed"] is True
    assert evidence["q3s3_subject_drift_closed"] is True
    assert decision["local_q2_component_keep_as_context"] is True
    assert decision["q3_s3_uncovered_after_local_fallback"] == ["Q3", "S3"]
    assert decision["local_fallback_closed_for_submission_artifact"] is True
    assert decision["local_artifact_allowed"] is False
    assert decision["public_058_evidence"] is False
    assert decision["primary_next_score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_explicit_kaggle_approval"] is True
    assert decision["prelaunch_ready_for_explicit_approval"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q3s3_local_lane_closure_is_report_only():
    report = load_report()

    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_made"] is True
    assert report["no_kaggle_run_made"] is True
    assert report["no_dacon_submission_made"] is True
