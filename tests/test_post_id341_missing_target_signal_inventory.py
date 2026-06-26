import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_missing_target_signal_inventory.py"


def load_module():
    spec = importlib.util.spec_from_file_location("missing_target_signal_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_target_inventory_blocks_local_artifact():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["local_q2_q3_s3_promotable"] is False
    assert decision["focus_missing_locally"] == ["Q2", "Q3", "S3"]


def test_missing_target_inventory_routes_focus_targets_to_clean_upstream_rebuild():
    module = load_module()
    report = module.build_report()

    for target in ["Q2", "Q3", "S3"]:
        inventory = report["target_inventory"][target]
        assert inventory["local"]["local_status"] == "missing"
        assert inventory["local"]["local_selected_row_count"] == 0
        assert inventory["upstream"]["clean_backbone_id226_row"] is not None
        assert inventory["recommended_action"] == "route_to_clean_id221_then_id226_no_submit_upstream_rebuild"

    assert report["evidence_flags"]["all_major_sources_present"] is True
