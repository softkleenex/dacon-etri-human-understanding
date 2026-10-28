from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_material_source_remote_coverage_contract.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("material_source_remote_coverage_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_material_source_remote_coverage_contract_passes_and_is_report_only() -> None:
    module = load_module()
    report = module.build_report()
    decision = report["decision"]
    rows = {row["name"]: row for row in report["coverage_rows"]}

    assert decision["material_source_remote_coverage_contract_pass"] is True
    assert decision["all_queue_items_covered"] is True
    assert decision["only_score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert rows["approved_clean_upstream_id221_to_id226"]["coverage_mode"] == "explicit_approval_remote_score_route"
    assert rows["usage_semantic_burst_markov_v1"]["coverage_mode"] == "remote_required_report_only_gate"
    assert rows["self_supervised_subject_state_embedding_v1"]["coverage_mode"] == "remote_required_report_only_gate"
    assert rows["phone_wearable_discordance_v1"]["coverage_mode"] == "local_closed_evidence_only"
    assert rows["proximity_ambience_cooccurrence_v1"]["coverage_mode"] == "local_closed_evidence_only"
    assert rows["sleep_proxy_threshold_residual_v1"]["coverage_mode"] == "local_closed_evidence_only"


def test_remote_required_sources_are_generated_staged_and_expected() -> None:
    module = load_module()
    rows = {row["name"]: row for row in module.build_report()["coverage_rows"]}

    for source in ["usage_semantic_burst_markov_v1", "self_supervised_subject_state_embedding_v1"]:
        checks = rows[source]["remote_checks"]
        assert len(checks) == 1
        assert checks[0]["runner_generates"] is True
        assert checks[0]["runner_stages"] is True
        assert checks[0]["runner_intake_expects"] is True
        assert checks[0]["output_intake_expects"] is True
        assert checks[0]["covered"] is True


def test_contract_fails_closed_when_self_supervised_remote_staging_is_missing(tmp_path: Path) -> None:
    module = load_module()
    runner = json.loads(module.DEFAULT_RUNNER_CONTRACT.read_text(encoding="utf-8"))
    missing = "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
    runner["runner_generated_reports"] = [name for name in runner["runner_generated_reports"] if name != missing]
    runner_path = tmp_path / "runner_missing_self_supervised.json"
    runner_path.write_text(json.dumps(runner, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = module.build_report(runner_contract_path=runner_path)
    rows = {row["name"]: row for row in report["coverage_rows"]}

    assert rows["self_supervised_subject_state_embedding_v1"]["covered"] is False
    assert report["decision"]["all_queue_items_covered"] is False
    assert report["decision"]["material_source_remote_coverage_contract_pass"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_local_closed_lanes_use_local_evidence_not_remote_requirements() -> None:
    module = load_module()
    rows = {row["name"]: row for row in module.build_report()["coverage_rows"]}

    for source in [
        "gps_path_topology_sequence_v1",
        "circadian_hr_light_phase_shape_v1",
        "proximity_ambience_cooccurrence_v1",
        "phone_wearable_discordance_v1",
        "sleep_proxy_threshold_residual_v1",
    ]:
        row = rows[source]
        assert row["coverage_mode"] == "local_closed_evidence_only"
        assert row["covered"] is True
        assert row["remote_checks"] == []
        assert row["local_evidence"]
        assert all(item["exists"] and item["parse_ok"] for item in row["local_evidence"])


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_material_source_remote_coverage_markdown_is_unambiguous() -> None:
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Coverage contract pass: `True`" in markdown
    assert "All queue items covered: `True`" in markdown
    assert "remote_required_report_only_gate" in markdown
    assert "local_closed_evidence_only" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
