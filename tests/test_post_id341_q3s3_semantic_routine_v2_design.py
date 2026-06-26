import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_semantic_routine_v2_design.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_semantic_routine_v2_design", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q3s3_semantic_routine_design_is_report_only_and_blocks_artifacts():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["design_ready"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["selected_targets"] == []
    assert decision["public_058_evidence"] is False
    assert decision["materially_different_from_physio_light_onset"] is True
    assert decision["external_direct_concat_allowed"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q3s3_semantic_routine_design_uses_raw_event_sources_without_direct_concat():
    module = load_module()
    report = module.build_report()
    schemas = report["raw_schema_summary"]["schemas"]
    checks = report["checks"]

    assert checks["closed_physio_light_onset_lane_confirmed"] is True
    assert checks["source_matrix_ready"] is True
    assert checks["data_sources_ready"] is True
    assert checks["external_direct_concat_blocked"] is True
    for source in ["mUsageStats", "mWifi", "mBle", "mGps", "mScreenStatus", "mActivity", "mACStatus"]:
        assert schemas[source]["exists"] is True
        assert "subject_id" in schemas[source]["columns"]
        assert "timestamp" in schemas[source]["columns"]


def test_q3s3_semantic_routine_design_names_distinct_feature_families():
    module = load_module()
    report = module.build_report()
    families = {row["target"]: row["family"] for row in report["feature_blueprint"]}
    markdown = module.build_markdown(report)

    assert families == {
        "Q3": "semantic_app_social_place_novelty",
        "S3": "sleep_onset_state_machine",
    }
    assert "semantic novelty" in markdown
    assert "ordered transitions" in markdown
    assert "Create submission artifact: `False`" in markdown
