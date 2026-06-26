import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_phone_wearable_discordance_feature_matrix_smoke_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phone_wearable_discordance_feature_matrix_smoke_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phone_wearable_feature_matrix_smoke_contract_is_report_only_and_ready():
    module = load_module()
    report = module.build_report()
    dec = report["decision"]
    checks = report["checks"]
    writer = report["future_writer_contract"]
    paths = report["registered_paths"]

    assert dec["feature_matrix_smoke_contract_ready"] is True
    assert dec["source_family"] == "phone_wearable_discordance_v1"
    assert dec["target_focus"] == ["Q2", "Q3", "S3"]
    assert dec["selected_targets"] == []
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["feature_matrix_write_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert checks["spec_contract_ready"] is True
    assert checks["dryrun_contract_ready"] is True
    assert checks["dryrun_schema_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert checks["does_not_write_feature_matrix"] is True
    assert checks["does_not_train_model"] is True
    assert checks["does_not_create_submission_artifact"] is True
    assert paths["feature_matrix_exists_now"] in {True, False}
    assert paths["schema_report_exists_now"] in {True, False}
    assert writer["expected_row_count"] == 700
    assert writer["expected_train_rows"] == 450
    assert writer["expected_test_rows"] == 250
    assert writer["expected_feature_column_count"] == 85
    assert writer["model_training_allowed"] is False
    assert writer["submission_artifact_allowed"] is False
    assert writer["kaggle_or_dacon_allowed"] is False
    assert report["no_feature_matrix_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_dacon_submission_made"] is True


def test_phone_wearable_feature_matrix_smoke_contract_markdown_boundary():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "phone/wearable discordance feature-matrix smoke contract" in markdown
    assert "Feature matrix write allowed now: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
    assert "does not create ID342" in markdown
    assert "registered label-free matrix writer" in markdown
