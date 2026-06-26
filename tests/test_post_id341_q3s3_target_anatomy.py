import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_target_anatomy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_target_anatomy", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q3s3_target_anatomy_is_ready_and_report_only():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["q3s3_target_anatomy_ready"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["current_local_sources_still_closed"] is True
    assert decision["q3_s3_still_uncovered"] is True
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_q3s3_target_anatomy_records_expected_label_structure():
    module = load_module()
    report = module.build_report()

    assert report["dataset"]["rows"] == 450
    assert report["dataset"]["subjects"] == 10
    assert report["target_summaries"]["Q3"]["positive_rate"] == 0.6
    assert round(report["target_summaries"]["S3"]["positive_rate"], 6) == 0.662222
    assert report["target_summaries"]["Q3"]["lag1"]["pairs"] > 0
    assert report["target_summaries"]["S3"]["lag1"]["pairs"] > 0
    assert report["same_day_correlations"]["Q3"]
    assert report["same_day_correlations"]["S3"]
    assert all(report["evidence_checks"].values())


def test_q3s3_target_anatomy_markdown_keeps_no_artifact_boundary():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Q3/S3 target anatomy ready: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
