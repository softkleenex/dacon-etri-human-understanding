import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_sleep_proxy_threshold_residual_design.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("sleep_proxy_threshold_residual_design", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sleep_proxy_threshold_residual_design_is_backlog_only():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["sleep_proxy_threshold_residual_design_ready"] is True
    assert decision["source_family"] == "sleep_proxy_threshold_residual_v1"
    assert decision["target_focus"] == ["S2", "S3", "S4", "Q1"]
    assert decision["metadata_feasible"] is True
    assert decision["design_backlog_recorded"] is True
    assert decision["local_no_approval_score_route_exists"] is False
    assert decision["open_design_lane_now"] is False
    assert decision["feature_matrix_allowed_now"] is False
    assert decision["train_model_now"] is False
    assert decision["compute_target_associations_now"] is False
    assert decision["public_058_evidence"] is False
    assert decision["public_058_reviewable"] is False
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["manual_probe_if_user_spends_submission"] is None
    assert decision["manual_probe_family_status"] == "consumed_after_id341_id340_id337_public_micro_probe_slate"


def test_sleep_proxy_design_schema_blocks_are_available_but_no_artifacts_allowed():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert report["processed_schema"]["train"]["shape"] == [450, 242]
    assert report["processed_schema"]["test"]["shape"] == [250, 242]
    assert all(row["schema_ready"] for row in report["feature_block_rows"])
    assert all(row["design_feasible"] for row in report["target_design_rows"])
    assert all(report["context_checks"].values())
    for key in [
        "create_feature_matrix",
        "create_submission_artifact",
        "modify_submission_log",
        "create_id342_or_new_submission_id",
        "submit_now",
        "push_kaggle_now",
        "run_kaggle_now",
    ]:
        assert decision[key] is False
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_feature_matrix_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_sleep_proxy_design_markdown_is_unambiguous():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Design ready: `True`" in markdown
    assert "Design backlog recorded: `True`" in markdown
    assert "Local no-approval score route exists: `False`" in markdown
    assert "Open design lane now: `False`" in markdown
    assert "Feature matrix allowed now: `False`" in markdown
    assert "Compute target associations now: `False`" in markdown
    assert "Manual probe if user spends submission: `None`" in markdown
    assert "Submit now: `False`" in markdown
