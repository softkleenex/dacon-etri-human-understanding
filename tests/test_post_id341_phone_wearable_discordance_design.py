import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_phone_wearable_discordance_design.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phone_wearable_discordance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phone_wearable_discordance_design_is_ready_and_no_submit():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["design_ready"] is True
    assert decision["source_family"] == "phone_wearable_discordance_v1"
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["coverage_ready"] is True
    assert decision["discordance_observed"] is True
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


def test_phone_wearable_discordance_has_row_contract_and_feasibility():
    module = load_module()
    report = module.build_report()

    assert report["row_contract"]["actual_train_rows"] == 450
    assert report["row_contract"]["actual_test_rows"] == 250
    assert report["row_contract"]["actual_total_rows"] == 700
    assert report["row_contract"]["hour_grid_rows"] == 700 * 24
    assert report["source_coverage"]["screen"]["train_day_rate"] >= 0.95
    assert report["source_coverage"]["charge"]["train_day_rate"] >= 0.95
    assert report["source_coverage"]["pedo"]["train_day_rate"] >= 0.85
    assert report["source_coverage"]["hr"]["train_day_rate"] >= 0.50
    assert report["discordance_feasibility"]["phone_rest_hour_bins"] > 0
    assert report["discordance_feasibility"]["wearable_active_hour_bins"] > 0
    assert report["discordance_feasibility"]["any_discordance_observed"] is True


def test_phone_wearable_discordance_markdown_mentions_forbidden_actions():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "phone/wearable discordance" in markdown
    assert "Feature matrix allowed now: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
