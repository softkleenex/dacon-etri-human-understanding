from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_local_source_closure_synthesis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("local_source_closure_synthesis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_source_closure_synthesis_blocks_local_id342_and_keeps_remote_route() -> None:
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["local_source_closure_synthesis_ready"] is True
    assert decision["local_no_approval_score_route_exists"] is False
    assert decision["only_score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["local_source_stack_closed_for_submission_artifact"] is True
    assert decision["any_local_lane_promotable"] is False
    assert decision["all_required_reports_present"] is True
    assert decision["kaggle_source_zip_selftest_ready"] is True
    assert decision["kaggle_upstream_packet_ready"] is True
    assert decision["requires_explicit_approval_for_kaggle"] is True
    assert decision["requires_exact_kaggle_approval_phrase"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["public_058_reviewable"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["modify_submission_log"] is False
    assert decision["create_id342_or_new_submission_id"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["train_model_now"] is False


def test_local_source_closure_synthesis_has_no_promotable_or_missing_lanes() -> None:
    module = load_module()
    report = module.build_report()

    assert report["missing_required_reports"] == []
    assert report["local_promotable_rows"] == []
    assert len(report["source_rows"]) >= 10
    assert any(row["lane"] == "kaggle_source_zip_selftest" for row in report["source_rows"])
    assert any(row["lane"] == "kaggle_upstream_packet" for row in report["source_rows"])
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_notebook_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_local_source_closure_synthesis_markdown_is_operator_safe() -> None:
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Local source closure synthesis ready: `True`" in markdown
    assert "Local no-approval score route exists: `False`" in markdown
    assert "Only score-relevant open route: `explicit_approved_clean_upstream_id221_to_id226`" in markdown
    assert "Create ID342/new submission ID: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`" in markdown
