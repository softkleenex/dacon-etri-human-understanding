import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_phone_wearable_discordance_feature_generation_dryrun.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phone_wearable_discordance_feature_generation_dryrun", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phone_wearable_discordance_dryrun_passes_and_blocks_actions():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["feature_generation_dryrun_passed"] is True
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
    assert report["no_feature_matrix_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_dacon_submission_made"] is True


def test_phone_wearable_discordance_dryrun_frame_contract():
    module = load_module()
    report = module.build_report()
    summary = report["feature_frame_summary"]
    checks = report["checks"]

    assert checks["spec_contract_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert checks["frame_contract_ready"] is True
    assert checks["no_feature_matrix_written"] is True
    assert summary["row_count"] == 700
    assert summary["train_rows"] == 450
    assert summary["test_rows"] == 250
    assert summary["feature_column_count"] == 85
    assert summary["expected_feature_column_count"] == 85
    assert summary["feature_columns_match_spec"] is True
    assert summary["duplicate_key_rows"] == 0
    assert summary["finite_all_values"] is True
    assert summary["nan_count"] == 0
    assert summary["feature_min"] >= 0
    assert summary["feature_max"] <= 24
    assert summary["nonzero_feature_count"] >= 70
    assert summary["zero_variance_feature_count"] <= 10


def test_phone_wearable_discordance_dryrun_markdown_boundary():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "phone/wearable discordance feature-generation dry run" in markdown
    assert "in-memory" in markdown
    assert "Feature matrix allowed now: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
    assert "feature-matrix smoke contract" in markdown
