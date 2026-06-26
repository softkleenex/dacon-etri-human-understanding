import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_feature_generation_metadata_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report_bundle():
    module = load_module()
    report = module.build_report()
    return report, module.build_markdown(report)


def test_q2q3s3_feature_generation_metadata_smoke_passes_and_is_report_only(report_bundle):
    report, _markdown = report_bundle

    decision = report["decision"]
    assert decision["metadata_smoke_passed"] is True
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["next_local_allowed_step"] == "fold_internal_q2q3s3_model_smoke_only"
    assert decision["train_model_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False

    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_made"] is True
    assert report["no_kaggle_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_q2q3s3_feature_generation_metadata_smoke_covers_rows_blocks_and_guards_ids(report_bundle):
    report, _markdown = report_bundle

    rows = report["row_coverage"]
    assert rows["train_rows"] == 450
    assert rows["test_rows"] == 250
    assert rows["expected_rows_ok"] is True
    assert rows["duplicate_key_rows"] == 0

    summary = report["feature_summary"]
    assert summary["raw_feature_columns"] >= 50
    assert summary["all_null_raw_feature_count"] == 0
    assert summary["duplicate_columns"] == []

    blocks = summary["block_summary"]
    assert blocks["routine_stability_clock_v2"]["raw_feature_columns"] > 0
    assert blocks["place_social_novelty_v2"]["raw_feature_columns"] > 0
    assert blocks["ambient_load_recovery_context_v2"]["raw_feature_columns"] > 0
    assert blocks["routine_stability_clock_v2"]["train_any_nonnull_row_share"] > 0.95
    assert blocks["place_social_novelty_v2"]["test_any_nonnull_row_share"] > 0.95

    guards = report["leakage_guards"]
    assert guards["forbidden_feature_columns"] == []
    assert guards["target_columns_present_in_feature_frame"] == []
    assert guards["direct_high_cardinality_identifier_columns_exported"] is False
    assert guards["exact_lat_lon_means_exported"] is False
    assert guards["uses_target_values"] is False
    assert guards["leakage_guard_ok"] is True


def test_q2q3s3_feature_generation_metadata_markdown_records_no_submit_contract(report_bundle):
    _report, markdown = report_bundle

    assert "Q2, Q3, S3" in markdown
    assert "Metadata smoke passed: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Push Kaggle now: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
