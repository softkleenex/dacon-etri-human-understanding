import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_source_route.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_source_route", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_q2q3s3_source_route_prioritizes_upstream_and_blocks_artifacts():
    module = load_module()
    report = module.build_report()

    decision = report["decision"]
    assert decision["missing_targets"] == ["Q2", "Q3", "S3"]
    assert decision["primary_score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["fallback_no_approval_route"] == "local_or_kaggle_q2q3s3_env_social_routine_v2_design"
    assert decision["remote_no_submit_prelaunch_ready"] is True
    assert decision["local_q2q3s3_artifact_allowed"] is False
    assert decision["more_q1_s4_tuning_allowed"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert "local_smoke_freshness" in report["inputs"]
    assert "kaggle_prelaunch" not in report["inputs"]


def test_q2q3s3_source_route_markdown_names_missing_targets_and_no_submit_path():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(report)

    assert "Q2, Q3, S3" in markdown
    assert "explicit_approved_clean_upstream_id221_to_id226" in markdown
    assert "local_or_kaggle_q2q3s3_env_social_routine_v2_design" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
