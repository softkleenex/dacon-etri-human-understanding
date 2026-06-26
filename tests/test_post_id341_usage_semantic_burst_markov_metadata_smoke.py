from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_usage_semantic_burst_markov_metadata_smoke import (
    PREDECLARED_APP_CATEGORIES,
    build_markdown,
    build_report,
    write_report,
)


def test_usage_semantic_burst_markov_metadata_smoke_is_report_only() -> None:
    report = build_report()
    decision = report["decision"]

    assert decision["metadata_smoke_passed"] is True
    assert decision["source_family"] == "usage_semantic_burst_markov_v1"
    assert decision["target_focus"] == ["Q2", "Q3"]
    assert decision["selected_targets"] == []
    assert decision["public_058_evidence"] is False
    assert decision["score_route_opened"] is False
    assert decision["feature_matrix_allowed_now"] is False
    assert decision["train_model_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_feature_matrix_created"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_usage_semantic_burst_markov_metadata_smoke_checks_contracts_and_state_space() -> None:
    report = build_report()
    checks = report["checks"]
    usage = report["usage_cardinality_profile"]
    transition = report["transition_profile"]
    session = report["session_profile"]
    screen = report["screen_coupling_profile"]

    assert checks["design_contract_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert checks["usage_train_test_coverage_ready"] is True
    assert checks["category_state_space_ready"] is True
    assert checks["session_state_ready"] is True
    assert checks["screen_coupling_ready"] is True
    assert usage["usage_rows"] == 45197
    assert usage["coverage_by_split"]["train"]["coverage_rate"] >= 0.95
    assert usage["coverage_by_split"]["test"]["coverage_rate"] >= 0.95
    assert usage["nonzero_dominant_categories"] >= 5
    assert transition["unique_transition_pairs"] >= 20
    assert transition["transition_pair_coverage_rate"] >= 0.40
    assert session["sessions"] >= 1000
    assert screen["usage_rows_with_screen_match_rate_5min"] >= 0.95


def test_usage_semantic_burst_markov_metadata_smoke_keeps_predeclared_category_boundary() -> None:
    report = build_report()
    usage = report["usage_cardinality_profile"]
    transition = report["transition_profile"]
    ambience = report["ambience_schema_profile"]

    assert set(usage["dominant_category_counts"]) == set(PREDECLARED_APP_CATEGORIES)
    assert set(usage["category_presence_counts"]) == set(PREDECLARED_APP_CATEGORIES)
    assert transition["all_possible_pairs"] == len(PREDECLARED_APP_CATEGORIES) ** 2
    assert transition["valid_transition_rows"] > transition["unique_transition_pairs"]
    assert ambience["usage_as_guard_only"] is True
    assert ambience["rows"] > 400_000


def test_usage_semantic_burst_markov_metadata_smoke_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "usage_metadata_smoke.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["decision"]["create_submission_artifact"] is False
    assert (
        written["decision"]["next_local_allowed_step"]
        == "report_only_usage_semantic_burst_markov_feature_spec_no_matrix"
    )
    assert "Top transition pairs" in md
    assert "does not create ID342" in direct_md
