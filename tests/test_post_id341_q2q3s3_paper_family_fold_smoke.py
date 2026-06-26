import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_paper_family_fold_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_paper_family_fold_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quick_report():
    module = load_module()
    return module.build_report(model_specs=(module.MODEL_SPECS[0],), split_names=["actual_test_band_holdout"])


def test_q2q3s3_paper_family_fold_smoke_is_report_only(quick_report):
    decision = quick_report["decision"]

    assert quick_report["diagnostic"] == "post_id341_q2q3s3_paper_family_fold_smoke"
    assert decision["paper_family_smoke_completed"] is True
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False

    assert quick_report["no_full_test_predictions_created"] is True
    assert quick_report["no_submission_created"] is True
    assert quick_report["no_base_prediction_artifact_created"] is True
    assert quick_report["no_notebook_artifact_created"] is True
    assert quick_report["no_submission_log_row_created"] is True
    assert quick_report["no_kaggle_push_made"] is True
    assert quick_report["no_kaggle_run_made"] is True
    assert quick_report["no_dacon_submission_made"] is True


def test_q2q3s3_paper_family_sources_cover_reference_schema(quick_report):
    assert quick_report["target_focus"] == ["Q2", "Q3", "S3"]
    families = quick_report["paper_families"]

    assert families["Q2"]["required_source_groups"] == ["mUsageStats", "DW"]
    assert families["Q3"]["required_source_groups"] == ["wPedo", "mUsageStats", "DW"]
    assert families["S3"]["required_source_groups"] == ["wPedo", "mLight", "wLight", "DW"]

    metadata = quick_report["feature_metadata"]
    assert metadata["aligned_train_rows"] == 450
    assert metadata["processed_train_shape"] == [450, 242]
    assert metadata["processed_test_shape"] == [250, 242]
    for group in ["mUsageStats", "wPedo", "mLight", "wLight", "DW"]:
        assert metadata["source_group_columns"][group] > 0
        assert metadata["source_group_test_columns"][group] > 0
    for target in ["Q2", "Q3", "S3"]:
        assert quick_report["family_feature_counts"][target] > 0
        assert target in quick_report["top_rows_by_target"]
    assert quick_report["scan_row_count"] > 0


def test_q2q3s3_paper_family_uses_existing_anchors_without_artifact_contract(quick_report):
    assert quick_report["inputs"]["id178_oof"].startswith("artifacts/base_predictions/base_oof_id_178")
    assert quick_report["inputs"]["id341_oof"].startswith("artifacts/base_predictions/base_oof_id_341")
    assert quick_report["decision"]["manual_probe_if_spending_submission"] is None
    assert quick_report["decision"]["manual_probe_family_status"] == (
        "consumed_after_id341_id340_id337_public_micro_probe_slate"
    )
