from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_local_remote_compute_boundary_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("local_remote_compute_boundary", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_remote_compute_boundary_passes_and_blocks_actions() -> None:
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["local_remote_compute_boundary_contract_pass"] is True
    assert decision["local_full_upstream_notebook_execution_allowed_by_default"] is False
    assert decision["local_contract_smoke_required_before_kaggle_approval"] is True
    assert decision["full_runner_requires_exact_kaggle_approval"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["create_submission_artifact"] is False
    assert decision["modify_submission_log"] is False
    assert decision["create_id342_or_new_submission_id"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["local_full_run_forbidden_tokens_found"] == []


def test_local_remote_compute_boundary_enforces_contract_staging_smoke() -> None:
    module = load_module()
    report = module.build_report()

    assert report["checks"]["contract_smoke"]["smoke_mode_declared"] is True
    assert report["checks"]["contract_smoke"]["stages_contract_outputs"] is True
    assert report["checks"]["contract_smoke"]["uses_output_intake_gate"] is True
    assert report["checks"]["contract_smoke"]["does_not_call_local_notebook_execution"] is True
    assert report["checks"]["makefile"]["help_describes_contract_smoke"] is True
    assert report["checks"]["makefile"]["help_does_not_claim_full_local_execution"] is True
    assert report["checks"]["makefile"]["prelaunch_records_compute_boundary_before_final_gate"] is True


def test_local_remote_compute_boundary_markdown_is_clear() -> None:
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Boundary contract pass: `True`" in markdown
    assert "Local full upstream notebook execution allowed by default: `False`" in markdown
    assert "Full runner requires exact Kaggle approval: `True`" in markdown
    assert "Create ID342/new submission ID: `False`" in markdown
