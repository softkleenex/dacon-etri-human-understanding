import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_post_id341_q3s3_semantic_routine_model_smoke.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_semantic_routine_model_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quick_report():
    module = load_module()
    return module.build_report(model_specs=(module.MODEL_SPECS[0],), split_names=["actual_test_band_holdout"])


def test_semantic_routine_model_smoke_is_report_only(quick_report):
    decision = quick_report["decision"]

    assert decision["model_smoke_completed"] is True
    assert decision["resource_gate_ready"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert quick_report["no_submission_created"] is True
    assert quick_report["no_base_prediction_artifact_created"] is True
    assert quick_report["no_submission_log_row_created"] is True


def test_semantic_routine_model_smoke_feature_frame_is_aligned_and_constrained(quick_report):
    coverage = quick_report["feature_coverage"]

    assert coverage["n_rows"] == 700
    assert coverage["n_train_rows"] == 450
    assert coverage["n_test_rows"] == 250
    assert coverage["raw_feature_columns"] <= 64
    assert coverage["feature_columns"] <= 64
    assert coverage["train_any_nonnull_row_share"] > 0.95
    assert coverage["test_any_nonnull_row_share"] > 0.95
    assert coverage["raw_identifier_columns_exported"] == []
    assert coverage["exact_lat_lon_columns_exported"] == []


def test_semantic_routine_model_smoke_scans_q3_s3_against_anchors(quick_report):
    assert set(quick_report["top_rows_by_target"]) == {"Q3", "S3"}
    assert quick_report["scan_row_count"] > 0
    assert quick_report["inputs"]["id178_oof"].startswith("artifacts/base_predictions/base_oof_id_178")
    assert quick_report["inputs"]["id341_oof"].startswith("artifacts/base_predictions/base_oof_id_341")
