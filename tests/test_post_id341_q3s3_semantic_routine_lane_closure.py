import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_semantic_routine_lane_closure.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_semantic_routine_lane_closure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_routine_lane_closure_blocks_id342_artifact():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["semantic_routine_lane_closure_passed"] is True
    assert decision["q3_s3_still_uncovered"] == ["Q3", "S3"]
    assert decision["local_semantic_routine_artifact_allowed"] is False
    assert decision["public_058_evidence"] is False
    assert decision["primary_next_score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_explicit_kaggle_approval"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert "local_smoke_freshness" in report["inputs"]
    assert "prelaunch" not in report["inputs"]


def test_semantic_routine_lane_closure_records_failed_selection_and_no_artifacts():
    module = load_module()
    report = module.build_report()

    assert report["checks"]["design_metadata_resource_gates_passed"] is True
    assert report["checks"]["model_smoke_closed_without_selected_targets"] is True
    assert "prelaunch_ready_pending_safe" in report["checks"]
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True
