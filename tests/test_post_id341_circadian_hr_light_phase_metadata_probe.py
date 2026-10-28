from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.diagnose_post_id341_circadian_hr_light_phase_metadata_probe import build_report, write_report


def test_circadian_hr_light_phase_metadata_probe_is_report_only_and_feasible() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["metadata_probe_passed"] is True
    assert dec["source_family"] == "circadian_hr_light_phase_shape_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3", "S2"}
    assert dec["strong_phase_shape_train_rate"] >= 0.50
    assert dec["strong_phase_shape_test_rate"] >= 0.50
    assert dec["next_local_allowed_step"] == "report_only_feature_spec_no_model"
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_circadian_hr_light_phase_metadata_probe_uses_queue_and_raw_phase_sources() -> None:
    report = build_report()

    assert report["queue_report"]["next_no_approval_step"] in {
        "usage_semantic_burst_markov_v1",
        "proximity_ambience_cooccurrence_v1",
    }
    assert report["queue_report"]["circadian_queue_status"] in {
        "metadata_passed_feature_spec_only_allowed_without_approval",
        "feature_spec_ready_dryrun_contract_only_without_approval",
        "feature_generation_dryrun_passed_contract_only_without_approval",
        "feature_matrix_smoke_contract_ready_write_only_under_contract",
        "feature_matrix_materialized_under_contract_no_model",
        "model_smoke_contract_ready_run_only_under_contract",
        "model_smoke_completed_review_needed_no_artifact",
        "closed_after_feature_matrix_and_model_smoke_no_signal",
    }
    assert set(report["source_summaries"]) == {"wHr", "wLight", "mLight", "wPedo"}
    assert report["source_summaries"]["wHr"]["coverage_by_split"]["test"]["coverage_rate"] > 0.90
    assert report["source_summaries"]["mLight"]["coverage_by_split"]["train"]["coverage_rate"] == 1.0
    assert report["phase_feasibility"]["by_split"]["train"]["weak_phase_shape_rate"] >= 0.70
    assert report["heart_rate_payload_summary"]["nonempty_payload_rate"] > 0.90


def test_circadian_hr_light_phase_metadata_probe_markdown_blocks_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "circadian_probe.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["public_058_evidence"] is False
    assert "creates no predictions" in md
    assert "Train model now: `False`" in md
    assert "does not open a submission artifact" in md
