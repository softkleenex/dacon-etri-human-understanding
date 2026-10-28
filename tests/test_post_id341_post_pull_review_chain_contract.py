import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_post_pull_review_chain_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("post_pull_review_chain_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_post_pull_review_chain_contract_passes_current_pending_state():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["post_pull_review_chain_contract_pass"] is True
    assert decision["output_pending"] is True
    assert decision["artifact_contract_review_allowed"] is False
    assert decision["public_058_reviewable"] is False
    assert decision["execute_kaggle_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert all(report["checks"].values())


def test_post_pull_review_chain_contract_verifies_target_order_and_local_only():
    module = load_module()
    report = module.build_report()
    target = report["target_contract"]

    assert all(target["required_scripts_present"].values())
    assert target["required_scripts_in_order"] is True
    assert target["target_has_no_remote_guard"] is True
    assert target["target_has_no_dacon_submit"] is True


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_post_pull_review_chain_contract_markdown_records_pending_safe_state():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Post-pull review-chain contract pass: `True`" in markdown
    assert "Artifact-contract review allowed: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
