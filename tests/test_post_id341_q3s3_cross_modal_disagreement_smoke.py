import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_cross_modal_disagreement_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_cross_modal_disagreement_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cross_modal_disagreement_smoke_is_report_only_and_closed():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["model_smoke_completed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["selected_targets"] == []
    assert decision["q3_signal_found"] is False
    assert decision["s3_signal_found"] is False
    assert decision["public_058_evidence"] is False
    assert decision["cross_modal_artifact_allowed"] is False
    assert decision["negative_map_extension_recommended"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_cross_modal_features_are_aligned_and_nonempty():
    module = load_module()
    train = module.load_train_sorted()
    features, coverage = module.build_cross_modal_features()
    aligned = module.align_train_features(train, features)
    cols = module.feature_columns(aligned)

    assert coverage["n_train_rows"] == 450
    assert coverage["n_test_rows"] == 250
    assert coverage["raw_feature_columns"] > 20
    assert coverage["feature_columns"] > coverage["raw_feature_columns"]
    assert coverage["train_any_nonnull_row_share"] == 1.0
    assert len(aligned) == len(train)
    assert len(cols) > 30
    assert any(col.startswith("xmod_screen_minus_dark") for col in cols)
    assert any(col.startswith("xmod_phone_watch_dark_absdiff") for col in cols)
