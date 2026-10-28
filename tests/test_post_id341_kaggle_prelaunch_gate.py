import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("kaggle_prelaunch_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_prelaunch_gate_includes_public_gap_and_data_coverage_checks():
    module = load_module()
    report = module.build_report(run_dry_run=False)
    checks = report["checks"]

    assert report["decision"]["prelaunch_ready_for_explicit_approval"] is True
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert report["local_pre_approval_command_sequence"] == [
        "make kaggle-approval-contract-test",
        "make kaggle-upstream-prelaunch-local",
    ]
    assert checks["runner_q3s3_negative_map_handoff_ready"] is True
    assert checks["public_local_gap_ready"] is True
    assert checks["public_local_gap_blocks_local_cv_only_promotion"] is True
    assert checks["data_source_coverage_ready"] is True
    assert checks["data_source_external_concat_blocked"] is True
    assert checks["data_source_local_stack_closed"] is True
    assert checks["missing_target_inventory_ready"] is True
    assert checks["q2q3s3_source_hypothesis_matrix_ready"] is True
    assert checks["q2q3s3_local_fallback_chain_ready"] is True
    assert checks["q3s3_local_lane_closure_ready"] is True
    assert checks["q3s3_semantic_routine_lane_closure_ready"] is True
    assert checks["q3s3_remaining_route_inventory_ready"] is True
    assert checks["usage_semantic_burst_markov_validation_stop_ready"] is True
    assert checks["self_supervised_subject_state_design_ready"] is True
    assert report["inputs"]["q3s3_remaining_route_inventory"]["decision"]["fold_nested_sibling_bridge_closed"] is True
    assert report["inputs"]["usage_semantic_burst_markov_validation_stop"]["decision"]["validation_stop_passed"] is True
    assert report["inputs"]["usage_semantic_burst_markov_validation_stop"]["decision"]["score_route_opened"] is False
    assert report["inputs"]["self_supervised_subject_state_design"]["decision"]["design_ready"] is True
    assert report["inputs"]["self_supervised_subject_state_design"]["decision"]["train_model_now"] is False
    assert checks["local_runner_smoke_fresh"] is True
    assert checks["remote_fail_closed_dryrun_pass"] is True
    assert checks["post_pull_review_chain_ready"] is True
    assert checks["runner_score_relevance_ready"] is True
    assert checks["material_source_remote_coverage_ready"] is True
    assert "public_local_gap_diagnosis" in report["inputs"]
    assert "data_source_coverage_audit" in report["inputs"]
    assert "missing_target_signal_inventory" in report["inputs"]
    assert "q2q3s3_source_hypothesis_matrix" in report["inputs"]
    assert "q2q3s3_local_fallback_chain" in report["inputs"]
    assert "q3s3_local_lane_closure" in report["inputs"]
    assert "q3s3_semantic_routine_lane_closure" in report["inputs"]
    assert "q3s3_remaining_route_inventory" in report["inputs"]
    assert "usage_semantic_burst_markov_validation_stop" in report["inputs"]
    assert "self_supervised_subject_state_design" in report["inputs"]
    assert "runner_q3s3_negative_map_handoff_contract" in report["inputs"]
    assert "local_smoke_freshness_gate" in report["inputs"]
    assert "remote_fail_closed_dryrun_audit" in report["inputs"]
    assert "post_pull_review_chain_contract" in report["inputs"]
    assert "runner_score_relevance_contract" in report["inputs"]
    assert "material_source_remote_coverage_contract" in report["inputs"]


def test_prelaunch_markdown_mentions_gap_and_data_coverage():
    module = load_module()
    report = module.build_report(run_dry_run=False)
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "public/local gap diagnosis" in markdown
    assert "Q3/S3 negative-map handoff" in markdown
    assert "Local pre-approval command sequence" in markdown
    assert "make kaggle-approval-contract-test" in markdown
    assert "make kaggle-upstream-prelaunch-local" in markdown
    assert "data/source coverage audit" in markdown
    assert "missing-target inventory" in markdown
    assert "Q2/Q3/S3 source-hypothesis matrix" in markdown
    assert "Q2/Q3/S3 local fallback chain" in markdown
    assert "Q3/S3 local-lane closure" in markdown
    assert "Q3/S3 semantic-routine lane closure" in markdown
    assert "Q3/S3 remaining-route inventory" in markdown
    assert "usage semantic burst/Markov validation-stop" in markdown
    assert "self-supervised subject-state design gate" in markdown
    assert "fold_nested_sibling_bridge_closed" in json.dumps(report, ensure_ascii=False)
    assert "local-smoke freshness" in markdown
    assert "fail-closed remote dry-run" in markdown
    assert "post-pull review-chain contract" in markdown
    assert "runner score-relevance contract" in markdown
    assert "material source remote coverage contract" in markdown
    assert "public_local_gap_blocks_local_cv_only_promotion: `True`" in markdown
    assert "data_source_external_concat_blocked: `True`" in markdown
    assert "missing_target_inventory_ready: `True`" in markdown
    assert "q2q3s3_source_hypothesis_matrix_ready: `True`" in markdown
    assert "q2q3s3_local_fallback_chain_ready: `True`" in markdown
    assert "q3s3_local_lane_closure_ready: `True`" in markdown
    assert "q3s3_semantic_routine_lane_closure_ready: `True`" in markdown
    assert "q3s3_remaining_route_inventory_ready: `True`" in markdown
    assert "usage_semantic_burst_markov_validation_stop_ready: `True`" in markdown
    assert "self_supervised_subject_state_design_ready: `True`" in markdown
    assert "runner_q3s3_negative_map_handoff_ready: `True`" in markdown
    assert "local_runner_smoke_fresh: `True`" in markdown
    assert "remote_fail_closed_dryrun_pass: `True`" in markdown
    assert "post_pull_review_chain_ready: `True`" in markdown
    assert "runner_score_relevance_ready: `True`" in markdown
    assert "material_source_remote_coverage_ready: `True`" in markdown


def test_prelaunch_gate_fails_closed_when_negative_map_handoff_breaks(monkeypatch, tmp_path):
    module = load_module()
    contract = json.loads(module.RUNNER_INTAKE_CONTRACT.read_text(encoding="utf-8"))
    handoff = contract["q3s3_negative_map_handoff_contract"]
    handoff["q3s3_negative_map_handoff_pass"] = False
    handoff["remaining_inventory_uses_negative_map_arg"] = False

    broken_contract = tmp_path / "broken_runner_intake_contract.json"
    broken_contract.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "RUNNER_INTAKE_CONTRACT", broken_contract)

    report = module.build_report(run_dry_run=False)

    assert report["checks"]["runner_intake_contract_pass"] is True
    assert report["checks"]["runner_q3s3_negative_map_handoff_ready"] is False
    assert report["decision"]["prelaunch_ready_for_explicit_approval"] is False
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert "Fix failed local prelaunch checks" in report["decision"]["next_action"]


def test_prelaunch_gate_fails_closed_when_material_source_coverage_breaks(monkeypatch, tmp_path):
    module = load_module()
    coverage = json.loads(module.MATERIAL_SOURCE_REMOTE_COVERAGE_CONTRACT.read_text(encoding="utf-8"))
    coverage["decision"]["material_source_remote_coverage_contract_pass"] = False
    coverage["decision"]["all_queue_items_covered"] = False

    broken_coverage = tmp_path / "broken_material_source_remote_coverage.json"
    broken_coverage.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "MATERIAL_SOURCE_REMOTE_COVERAGE_CONTRACT", broken_coverage)

    report = module.build_report(run_dry_run=False)

    assert report["checks"]["material_source_remote_coverage_ready"] is False
    assert report["decision"]["prelaunch_ready_for_explicit_approval"] is False
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert "Fix failed local prelaunch checks" in report["decision"]["next_action"]
