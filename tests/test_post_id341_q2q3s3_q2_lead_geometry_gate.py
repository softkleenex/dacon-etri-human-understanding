import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports" / "post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json"


def load_report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_q2_lead_geometry_gate_records_reviewable_but_not_public_058_artifact():
    report = load_report()
    decision = report["decision"]

    assert decision["q2_config_available"] is True
    assert decision["reviewable_low_movement_candidate_exists"] is True
    assert decision["public_058_plausible"] is False
    assert decision["q3_s3_still_uncovered"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q2_lead_geometry_gate_keeps_predictions_in_memory_only():
    report = load_report()

    assert report["full_test_predictions_in_memory_only"] is True
    assert report["no_full_test_predictions_written"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_run_made"] is True
    assert report["no_dacon_submission_made"] is True
    assert report["best_by_proxy"]["test_geometry"]["proxy_public_risk"] > 0.58
