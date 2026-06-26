import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("q3s3_fold_nested_sibling_bridge_smoke", SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def report(module):
    return module.build_report(
        model_specs=(module.MODEL_SPECS[0],),
        split_names=["actual_test_band_holdout"],
    )


def test_fold_nested_sibling_bridge_is_report_only_and_closed(report):
    decision = report["decision"]

    assert report["diagnostic"] == "post_id341_q3s3_fold_nested_sibling_bridge_smoke"
    assert decision["fold_nested_sibling_smoke_completed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["selected_targets"] == []
    assert decision["q3_signal_found"] is False
    assert decision["s3_signal_found"] is False
    assert decision["all_q3s3_targets_covered"] is False
    assert decision["public_058_evidence"] is False
    assert decision["local_artifact_allowed"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_fold_nested_sibling_bridge_keeps_expected_leakage_boundary(report):
    assert report["target_focus"] == ["Q3", "S3"]
    assert report["sibling_targets"] == {"Q3": "Q2", "S3": "S2"}
    assert all(report["checks"].values())
    assert report["config"]["split_names"] == ["actual_test_band_holdout"]
    assert report["config"]["model_specs"][0]["name"] == "lr_c0_02"
    assert set(report["top_rows_by_target"]) == {"Q3", "S3"}
    assert report["scan_row_count"] > 0

    for target, rows in report["top_rows_by_target"].items():
        assert rows, target
        assert rows[0]["sibling_target"] == report["sibling_targets"][target]
        assert rows[0]["primary_wins_vs_id341"] == 0
        assert rows[0]["mean_delta_vs_id341"] > 0


def test_fold_nested_sibling_bridge_markdown_blocks_shortcuts(module, report):
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Q3 uses only fold-nested predictions of Q2" in markdown
    assert "S3 uses only fold-nested predictions of S2" in markdown
    assert "not Q2 labels from the same row" in markdown
    assert "not S2 labels from the same row" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
    assert "do not create ID342" in markdown


def test_fold_nested_sibling_bridge_creates_no_artifact_side_effects(report):
    assert report["no_full_test_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_notebook_history_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True
