import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_runner_score_relevance_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("runner_score_relevance_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_score_relevance_contract_passes_current_repo_state():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["runner_score_relevance_contract_pass"] is True
    assert decision["score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["missing_targets"] == ["Q2", "Q3", "S3"]
    assert decision["execute_kaggle_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert "q3s3_semantic_routine_lane_closure" in report["inputs"]
    assert "q3s3_remaining_route_inventory" in report["inputs"]
    assert report["inputs"]["q3s3_remaining_route_inventory"]["decision"]["fold_nested_sibling_bridge_closed"] is True
    assert report["checks"]["q3s3_semantic_routine_closure_contract"] is True
    assert report["checks"]["q3s3_remaining_route_inventory_contract"] is True
    assert all(report["checks"].values())
    assert all(report["runner_script_coverage"].values())


def test_runner_score_relevance_markdown_names_missing_targets_and_route():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Runner score-relevance contract pass: `True`" in markdown
    assert "explicit_approved_clean_upstream_id221_to_id226" in markdown
    assert "Q2" in markdown and "Q3" in markdown and "S3" in markdown
    assert "q3s3_semantic_routine_closure_contract: `True`" in markdown
    assert "q3s3_remaining_route_inventory_contract: `True`" in markdown
