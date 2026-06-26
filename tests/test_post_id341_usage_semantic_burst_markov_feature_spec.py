from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_spec import (
    FEATURE_COLUMN_BOUND,
    NEXT_ALLOWED_STEP,
    PREDECLARED_APP_CATEGORIES,
    build_markdown,
    build_report,
    write_report,
)


def test_usage_semantic_burst_markov_feature_spec_is_report_only() -> None:
    report = build_report()
    decision = report["decision"]

    assert decision["feature_spec_ready"] is True
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
    assert decision["next_local_allowed_step"] == NEXT_ALLOWED_STEP
    assert report["no_feature_matrix_created"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_usage_semantic_burst_markov_feature_spec_contracts_and_bounds() -> None:
    report = build_report()
    checks = report["checks"]
    schema = report["feature_schema"]
    row_contract = report["row_contract"]

    assert checks["design_contract_ready"] is True
    assert checks["metadata_contract_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert checks["metadata_state_space_ready"] is True
    assert checks["feature_schema_bounded"] is True
    assert checks["uses_predeclared_app_categories_only"] is True
    assert checks["raw_app_name_features_allowed"] is False
    assert row_contract["expected_row_count"] == 700
    assert row_contract["expected_train_rows"] == 450
    assert row_contract["expected_test_rows"] == 250
    assert schema["feature_group_count"] == 7
    assert schema["feature_column_count"] == 77
    assert schema["feature_column_count"] <= FEATURE_COLUMN_BOUND
    assert len(schema["feature_columns"]) == len(set(schema["feature_columns"]))


def test_usage_semantic_burst_markov_feature_spec_freezes_state_and_leakage_boundary() -> None:
    report = build_report()
    state = report["state_space_contract"]
    validation = report["validation_and_leakage_contract"]
    groups = {group["name"]: group for group in report["feature_schema"]["feature_groups"]}

    assert state["predeclared_app_categories"] == PREDECLARED_APP_CATEGORIES
    assert state["predeclared_category_count"] == 7
    assert state["predeclared_pair_space_size"] == 49
    assert state["raw_app_name_features_allowed"] is False
    assert state["target_guided_vocabulary_allowed"] is False
    assert "semantic_category_distribution" in groups
    assert "markov_transition_shape" in groups
    assert "screen_coupled_bursts" in groups
    assert "ambience_context_guards" in groups
    assert "raw app-name one-hot" in validation["forbidden_inputs"]
    assert "public leaderboard score" in validation["forbidden_inputs"]
    assert "fit any scaling/imputation" in validation["preprocessing_fit_scope"]


def test_usage_semantic_burst_markov_feature_spec_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "usage_feature_spec.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["decision"]["create_submission_artifact"] is False
    assert written["decision"]["next_local_allowed_step"] == NEXT_ALLOWED_STEP
    assert "Feature groups" in md
    assert "Raw app-name features allowed: `False`" in md
    assert "does not create ID342" in direct_md
    assert "No feature matrix, model" in direct_md
