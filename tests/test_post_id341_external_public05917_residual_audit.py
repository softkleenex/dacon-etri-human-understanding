import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_external_public05917_residual_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("post_id341_external_public05917_residual_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_public05917_residual_audit_closes_direct_notebook_route():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert report["diagnostic"] == "post_id341_external_public05917_residual_audit"
    assert decision["audit_ready"] is True
    assert decision["residual_direct_notebook_route_exists"] is False
    assert decision["residual_direct_routes"] == []
    assert decision["public_058_evidence_present"] is False
    assert decision["q3_s3_still_uncovered"] is True
    assert decision["only_score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["operator_handoff_ready"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_external_public05917_residual_audit_maps_all_major_notebook_ideas():
    module = load_module()
    report = module.build_report()
    idea_status = {row["idea"]: row for row in report["idea_status"]}

    assert report["reference_notebook"]["exists"] is True
    assert report["external_review_summary"]["claimed_public_score"] == 0.5917
    assert set(idea_status) == {
        "anchor_blending_and_subject_hole_cv",
        "stability_selected_lgbm_xgb_side_models",
        "window_pair_screen_interactions",
        "smartphone_light_screen_compact_s3",
        "target_specific_blending",
        "subject_adaptive_or_subject_profile_calibration",
    }
    assert all(row["artifact_allowed"] is False for row in idea_status.values())
    assert idea_status["stability_selected_lgbm_xgb_side_models"]["status"] == "tested_and_closed_as_direct_overlay"
    assert idea_status["target_specific_blending"]["status"] == "q2_context_only_q3s3_missing"


def test_external_public05917_residual_audit_markdown_is_no_submit_clear():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "This is report-only" in markdown
    assert "Residual direct notebook route exists: `False`" in markdown
    assert "Public 0.58 evidence present: `False`" in markdown
    assert "Only score-relevant open route: `explicit_approved_clean_upstream_id221_to_id226`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Push Kaggle now: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
