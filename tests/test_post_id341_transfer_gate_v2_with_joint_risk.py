import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_transfer_gate_v2_with_joint_risk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("post_id341_transfer_gate_v2_with_joint_risk", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transfer_gate_v2_is_report_only_and_blocks_public_058_promotion():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert report["diagnostic"] == "post_id341_transfer_gate_v2_with_joint_risk"
    assert decision["any_candidate_passes_combined_public_058_gate"] is False
    assert decision["public_058_plausible_candidate_exists"] is False
    assert decision["best_combined_pass_id"] is None
    assert decision["best_combined_pass_file"] is None
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_notebook_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_transfer_gate_v2_keeps_recent_controls_visible_and_classified():
    module = load_module()
    report = module.build_report()
    recent = {row["id"]: row for row in report["recent_control_rows"]}

    assert recent[178]["public_lb"] == 0.6014113009
    assert recent[287]["combined_gate_pass"] is False
    assert recent[287]["joint_warning_count"] >= 4
    assert recent[329]["combined_gate_pass"] is False

    assert recent[340]["joint_gate_pass"] is True
    assert recent[340]["transfer_gate_pass"] is False
    assert recent[340]["combined_tier"] == "joint_safe_but_transfer_blocked"
    assert recent[341]["joint_gate_pass"] is True
    assert recent[341]["transfer_gate_pass"] is False
    assert recent[341]["combined_tier"] == "joint_safe_but_transfer_blocked"
    assert "transfer_proxy_above_public_058_gate" in recent[341]["block_reasons"]


def test_transfer_gate_v2_prioritizes_transfer_proxy_but_does_not_promote_id340():
    module = load_module()
    report = module.build_report()
    top_proxy = report["top_transfer_proxy_rows"][0]

    assert top_proxy["id"] == 340
    assert (
        top_proxy["submission_file"]
        == "submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv"
    )
    assert top_proxy["joint_gate_pass"] is True
    assert top_proxy["transfer_gate_pass"] is False
    assert top_proxy["combined_gate_pass"] is False
    assert top_proxy["combined_tier"] == "joint_safe_but_transfer_blocked"
    assert report["combined_pass_rows"] == []


def test_transfer_gate_v2_markdown_is_unambiguous_about_no_submit():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "This is report-only" in markdown
    assert "Any candidate passes combined Public 0.58 gate: `False`" in markdown
    assert "Public 0.58 plausible candidate exists: `False`" in markdown
    assert "Best combined-pass ID: `None`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Push Kaggle now: `False`" in markdown
