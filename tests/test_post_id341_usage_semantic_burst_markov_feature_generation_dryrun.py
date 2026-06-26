from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_generation_dryrun import (
    NEXT_ALLOWED_STEP,
    assemble_feature_frame,
    build_markdown,
    build_report,
    write_report,
)
from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_spec import all_feature_columns, feature_groups


def test_usage_semantic_burst_markov_feature_generation_dryrun_is_report_only() -> None:
    report = build_report()
    decision = report["decision"]

    assert decision["feature_generation_dryrun_passed"] is True
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


def test_usage_semantic_burst_markov_feature_generation_dryrun_alignment_and_schema() -> None:
    report = build_report()
    checks = report["checks"]
    summary = report["feature_frame_summary"]

    assert checks["spec_contract_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert checks["row_alignment_passed"] is True
    assert checks["schema_passed"] is True
    assert checks["values_passed"] is True
    assert summary["row_count"] == 700
    assert summary["train_rows"] == 450
    assert summary["test_rows"] == 250
    assert summary["feature_column_count"] == 77
    assert summary["feature_column_count"] == len(all_feature_columns(feature_groups()))
    assert summary["finite_all_values"] is True
    assert summary["nan_count"] == 0
    assert summary["zero_variance_feature_count"] < 20


def test_usage_semantic_burst_markov_feature_frame_columns_are_exact_and_safe() -> None:
    frame = assemble_feature_frame()
    feature_columns = all_feature_columns(feature_groups())

    assert list(frame.columns) == ["subject_id", "lifelog_date", "split", *feature_columns]
    assert len(frame) == 700
    assert frame["split"].value_counts().to_dict() == {"train": 450, "test": 250}
    assert not frame[feature_columns].isna().any().any()
    forbidden_tokens = ["app_name", "raw_app", "target", "label", "public_lb", "submission", "prediction", "oof"]
    assert all(not any(token in column.lower() for token in forbidden_tokens) for column in feature_columns)


def test_usage_semantic_burst_markov_feature_generation_dryrun_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "usage_feature_generation_dryrun.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["decision"]["create_submission_artifact"] is False
    assert written["decision"]["next_local_allowed_step"] == NEXT_ALLOWED_STEP
    assert "In-memory feature frame summary" in md
    assert "creates no feature matrix/parquet" in md
    assert "does not create ID342" in direct_md
