import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_no_approval_frontier.py"


def load_module():
    spec = importlib.util.spec_from_file_location("no_approval_frontier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_approval_frontier_closes_local_routes_and_keeps_remote_approval_required():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["no_approval_frontier_ready"] is True
    assert decision["all_local_no_approval_score_routes_closed"] is True
    assert decision["no_approval_score_route_exists"] is False
    assert decision["only_score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_explicit_kaggle_approval"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["prelaunch_ready_for_explicit_approval"] is True
    assert decision["public_058_reviewable"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["local_closure_checks"]["external_public05917_direct_route_closed"] is True
    assert report["local_closure_checks"]["q2q3s3_paper_family_local_lane_closed"] is True
    assert report["local_closure_checks"]["q3s3_oof_gain_direct_submit_blocked"] is True
    assert report["local_closure_checks"]["usage_semantic_burst_markov_matrix_validated_stop"] is True
    assert report["report_status"]["external_public05917_residual"]["exists"] is True
    assert report["report_status"]["q2q3s3_paper_family"]["exists"] is True
    assert report["report_status"]["q3s3_anchor_failure_profile"]["exists"] is True
    assert report["report_status"]["usage_semantic_burst_markov_validation_stop"]["exists"] is True


def test_no_approval_frontier_records_public_anchor_and_report_only_contract():
    module = load_module()
    report = module.build_report()
    anchor = report["public_anchor_summary"]["best_public_anchor"]

    assert anchor["id"] == 337
    assert anchor["public"] == 0.6006206286219425
    assert all(report["local_closure_checks"].values())
    assert all(report["prelaunch_checks"].values())
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_no_approval_frontier_markdown_is_unambiguous():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "No-approval score route exists: `False`" in markdown
    assert "Only score-relevant open route: `explicit_approved_clean_upstream_id221_to_id226`" in markdown
    assert "Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "external_public05917_direct_route_closed: `True`" in markdown
    assert "q2q3s3_paper_family_local_lane_closed: `True`" in markdown
    assert "q3s3_oof_gain_direct_submit_blocked: `True`" in markdown
    assert "usage_semantic_burst_markov_matrix_validated_stop: `True`" in markdown
