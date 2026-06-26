from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_usage_semantic_burst_markov_design import (
    PREDECLARED_APP_CATEGORIES,
    build_markdown,
    build_report,
    write_report,
)


def test_usage_semantic_burst_markov_design_is_report_only_and_blocks_artifacts() -> None:
    report = build_report()
    decision = report["decision"]

    assert decision["design_ready"] is True
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


def test_usage_semantic_burst_markov_design_respects_queue_and_prior_closures() -> None:
    report = build_report()
    checks = report["checks"]

    assert checks["queue_allows_report_only_design"] is True
    assert checks["prior_usage_aggregate_closed"] is True
    assert checks["ambience_guard_only"] is True
    assert checks["semantic_routine_artifact_closed"] is True
    assert checks["raw_cardinality_bounded_enough_for_design"] is True
    assert checks["uses_predeclared_app_categories_only"] is True
    assert checks["does_not_create_feature_matrix"] is True
    assert checks["does_not_train_model"] is True


def test_usage_semantic_burst_markov_design_profiles_raw_sources_and_categories() -> None:
    report = build_report()
    schemas = report["raw_schema_summary"]["schemas"]
    usage = report["usage_payload_profile"]
    screen = report["screen_status_profile"]
    families = {item["name"] for item in report["feature_blueprint"]}

    assert schemas["mUsageStats"]["exists"] is True
    assert schemas["mScreenStatus"]["exists"] is True
    assert schemas["mAmbience"]["exists"] is True
    assert usage["rows"] == 45197
    assert usage["unique_app_names"] >= 500
    assert usage["rows_with_payload_rate"] > 0.90
    assert set(PREDECLARED_APP_CATEGORIES).issuperset(usage["predeclared_category_event_counts"])
    assert screen["rows"] > 900_000
    assert screen["coverage_by_split"]["train"]["coverage_rate"] > 0.95
    assert families == {
        "app_burst_sessionization",
        "semantic_app_category_markov_chain",
        "ambience_guarded_context_burst",
    }


def test_usage_semantic_burst_markov_design_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "usage_design.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["decision"]["create_submission_artifact"] is False
    assert "Markov" in md
    assert "Anti-repeat rule" in md
    assert "does not create ID342" in direct_md
