import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_fold_internal_model_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_fold_internal_model_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quick_report():
    module = load_module()
    return module.build_report(model_specs=(module.MODEL_SPECS[0],), split_names=["actual_test_band_holdout"])


def test_q2q3s3_fold_internal_model_smoke_is_report_only(quick_report):
    decision = quick_report["decision"]

    assert decision["model_smoke_completed"] is True
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False

    assert quick_report["no_full_test_predictions_created"] is True
    assert quick_report["no_submission_created"] is True
    assert quick_report["no_base_prediction_artifact_created"] is True
    assert quick_report["no_submission_log_row_created"] is True
    assert quick_report["no_kaggle_run_made"] is True
    assert quick_report["no_dacon_submission_made"] is True


def test_q2q3s3_fold_internal_model_smoke_uses_aligned_features_and_primary_schema(quick_report):
    feature_metadata = quick_report["feature_metadata"]

    assert feature_metadata["aligned_train_rows"] == 450
    assert feature_metadata["feature_columns"] >= 50
    assert set(quick_report["top_rows_by_target"]) == {"Q2", "Q3", "S3"}
    assert quick_report["scan_row_count"] > 0
    assert quick_report["inputs"]["id178_oof"].startswith("artifacts/base_predictions/base_oof_id_178")
    assert quick_report["inputs"]["id341_oof"].startswith("artifacts/base_predictions/base_oof_id_341")
