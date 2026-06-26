import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_joint_risk_candidate_rerank_post_id341.py"


def load_module():
    spec = importlib.util.spec_from_file_location("joint_risk_candidate_rerank_post_id341", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_risk_rerank_is_report_only_and_prefers_id341_low_warning_candidate():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert report["diagnostic"] == "joint_risk_candidate_rerank_post_id341"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["best_low_joint_warning_id"] == 341
    assert (
        decision["best_low_joint_warning_file"]
        == "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"
    )
    assert decision["frontier_candidates_have_joint_warnings"] is True
    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_dacon_submission_made"] is True


def test_joint_risk_rerank_keeps_public_collapse_and_frontier_warnings_visible():
    module = load_module()
    report = module.build_report()
    recent = {row["id"]: row for row in report["recent_rows_ge_330"]}
    controls = {row["id"]: row for row in report["controls"]}

    assert controls[287]["joint_risk_tier"] == "already_scored_reference"
    assert controls[329]["joint_risk_tier"] == "already_scored_reference"
    assert recent[333]["joint_risk_tier"] == "reject_high_joint_risk"
    assert recent[333]["joint_warning_count"] >= 4
    assert recent[331]["joint_risk_tier"] == "frontier_overfit_warning"
    assert recent[341]["joint_risk_tier"] == "low_joint_warning"
    assert recent[341]["joint_warning_count"] == 0


def test_joint_risk_rerank_markdown_separates_warning_overlay_from_submission_decision():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "This is report-only" in markdown
    assert "Best low-warning ID: `341`" in markdown
    assert "Frontier candidates have joint warnings: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
