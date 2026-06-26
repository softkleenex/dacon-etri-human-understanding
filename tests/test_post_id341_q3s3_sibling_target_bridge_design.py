import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_sibling_target_bridge_design.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_sibling_bridge", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q3s3_sibling_bridge_design_is_ready_but_no_artifact():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["sibling_bridge_design_ready"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["material_sibling_structure_found"] is True
    assert decision["requires_fold_nested_rebuild"] is True
    assert decision["recommended_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["local_artifact_allowed"] is False
    assert decision["public_058_evidence"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q3s3_sibling_bridge_names_q2_and_s2_with_leakage_controls():
    module = load_module()
    report = module.build_report()
    rows = {row["target"]: row for row in report["bridge_rows"]}

    assert rows["Q3"]["sibling_target"] == "Q2"
    assert rows["S3"]["sibling_target"] == "S2"
    assert rows["Q3"]["label_corr"] >= 0.30
    assert rows["S3"]["label_corr"] >= 0.30
    assert rows["Q3"]["requires_fold_nested_oof"] is True
    assert rows["S3"]["requires_test_time_sibling_prediction"] is True
    assert all(report["checks"].values())
    assert any("out-of-fold" in item for item in report["leakage_controls"])


def test_q3s3_sibling_bridge_markdown_blocks_shortcuts():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Sibling bridge design ready: `True`" in markdown
    assert "Requires fold-nested rebuild: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Do not create ID342" in markdown
