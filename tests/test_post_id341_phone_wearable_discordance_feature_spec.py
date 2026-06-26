import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_phone_wearable_discordance_feature_spec.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phone_wearable_discordance_feature_spec", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phone_wearable_discordance_feature_spec_is_ready_and_no_submit():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["feature_spec_ready"] is True
    assert decision["source_family"] == "phone_wearable_discordance_v1"
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["selected_targets"] == []
    assert decision["public_058_evidence"] is False
    assert decision["score_route_opened"] is False
    assert decision["feature_matrix_allowed_now"] is False
    assert decision["train_model_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_phone_wearable_discordance_feature_spec_schema_contract():
    module = load_module()
    report = module.build_report()
    bounds = report["schema_bounds"]

    assert report["row_contract"]["expected_train_rows"] == 450
    assert report["row_contract"]["expected_test_rows"] == 250
    assert report["row_contract"]["expected_total_rows"] == 700
    assert report["row_contract"]["label_columns_allowed"] is False
    assert report["row_contract"]["prediction_columns_allowed"] is False
    assert report["row_contract"]["submission_columns_allowed"] is False
    assert report["feature_count"] == 85
    assert bounds["actual_feature_columns"] == 85
    assert bounds["actual_feature_columns"] <= bounds["max_feature_columns"]
    assert bounds["all_features_prefixed"] is True
    assert bounds["duplicate_feature_count"] == 0
    assert bounds["forbidden_feature_token_hits"] == []
    assert bounds["raw_identifier_columns_allowed"] is False
    assert bounds["fold_internal_preprocessing_required_for_any_model"] is True
    assert all(name.startswith("pwd_") for name in report["feature_names"])


def test_phone_wearable_discordance_feature_spec_markdown_boundary():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "phone/wearable discordance feature spec" in markdown
    assert "Feature matrix allowed now: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
    assert "in-memory feature-generation dry run" in markdown
    assert "ID342" in markdown
