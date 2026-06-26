import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_subject_drift_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_subject_drift_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q3s3_subject_drift_gate_closes_artifact_path():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["q3s3_subject_drift_gate_passed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["q3_primary_wins_vs_id341"] == 1
    assert decision["s3_primary_wins_vs_id341"] == 2
    assert decision["q3s3_subject_drift_artifact_allowed"] is False
    assert decision["q3s3_subject_drift_closed_for_artifact"] is True
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q3s3_subject_drift_gate_is_report_only():
    module = load_module()
    report = module.build_report()

    assert "subject_residual_sequence_drift_smoke" in report["inputs"]
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True
