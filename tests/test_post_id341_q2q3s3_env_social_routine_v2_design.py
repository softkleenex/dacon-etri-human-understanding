import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_env_social_routine_v2_design.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_env_social_routine_v2_design", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_q2q3s3_env_social_routine_design_is_report_only_and_targeted():
    module = load_module()
    report = module.build_report()

    decision = report["decision"]
    assert decision["design_ready"] is True
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["next_local_allowed_step"] == "feature_generation_metadata_smoke_only"
    assert decision["train_model_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q2q3s3_env_social_routine_design_blocks_high_cardinality_shortcuts():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(report)

    assert report["raw_source_profiles"]["mWifi"]["cardinality"]["m_wifi"]["bssid"]["unique"] > 1000
    assert report["raw_source_profiles"]["mBle"]["cardinality"]["m_ble"]["address"]["unique"] > 1000
    assert "direct BSSID/address/app-name one-hot columns fitted on full train+test" in markdown
    assert "Q2, Q3, S3" in markdown
    assert "Create submission artifact: `False`" in markdown
