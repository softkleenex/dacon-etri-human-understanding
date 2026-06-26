import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_post_id341_q3s3_semantic_routine_metadata_smoke.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_semantic_routine_metadata_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_routine_metadata_smoke_is_bounded_and_report_only():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["metadata_smoke_passed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["selected_targets"] == []
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["sample_policy"]["full_file_load_allowed"] is False
    assert report["checks"]["bounded_sample_only"] is True


def test_semantic_routine_metadata_smoke_covers_q3_and_s3_sources():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]
    summaries = report["source_summaries"]

    assert set(decision["q3_ready_sources"]) >= {"mUsageStats", "mWifi", "mBle", "mGps"}
    assert set(decision["s3_ready_sources"]) >= {"mScreenStatus", "mActivity", "mACStatus", "mLight", "wLight"}
    for source, summary in summaries.items():
        assert summary["exists"] is True, source
        assert summary["sample_rows"] <= summary["max_sample_rows"], source
        assert summary["bounded_sample_only"] is True, source
        assert summary["non_null_row_share"] > 0, source


def test_semantic_routine_metadata_smoke_does_not_emit_raw_identifiers():
    module = load_module()
    report = module.build_report()
    rendered = module.build_markdown(report)

    assert report["checks"]["privacy_no_raw_identifier_examples"] is True
    assert "raw identifiers/coordinates are summarized only as counts" in str(report["source_summaries"])
    for forbidden_key in report["sample_policy"]["raw_identifier_keys_not_emitted"]:
        assert f"`{forbidden_key}`" in rendered
    assert "Create submission artifact: `False`" in rendered
