import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_semantic_routine_resource_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_semantic_routine_resource_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_routine_resource_gate_allows_only_small_local_no_artifact_smoke():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["resource_gate_passed"] is True
    assert decision["local_no_artifact_model_smoke_allowed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["public_058_evidence"] is False
    assert decision["next_local_allowed_step"] == "fold_internal_semantic_routine_model_smoke_only"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_semantic_routine_resource_gate_has_conservative_limits():
    module = load_module()
    report = module.build_report()
    constraints = report["plan_constraints"]
    budget = report["resource_budget"]

    assert report["checks"]["metadata_smoke_ready"] is True
    assert report["checks"]["raw_size_within_local_soft_limit"] is True
    assert constraints["stream_raw_sources_one_at_a_time"] is True
    assert constraints["do_not_one_hot_raw_app_wifi_ble_ids"] is True
    assert constraints["max_engineered_feature_columns"] <= 64
    assert constraints["max_model_smoke_targets"] == 2
    assert constraints["max_model_specs"] <= 2
    assert constraints["max_blend_weights"] <= 4
    assert budget["total_raw_bytes"] <= budget["local_raw_bytes_soft_limit"]


def test_semantic_routine_resource_gate_markdown_records_no_submit_contract():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(report)

    assert "Resource gate passed: `True`" in markdown
    assert "Local no-artifact model smoke allowed: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
