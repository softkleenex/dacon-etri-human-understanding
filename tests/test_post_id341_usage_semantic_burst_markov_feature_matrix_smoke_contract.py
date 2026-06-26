from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract import (
    NEXT_ALLOWED_STEP,
    build_markdown,
    build_report,
    write_report,
)


def test_usage_semantic_burst_markov_feature_matrix_contract_is_report_only() -> None:
    report = build_report()
    decision = report["decision"]

    assert decision["feature_matrix_smoke_contract_ready"] is True
    assert decision["source_family"] == "usage_semantic_burst_markov_v1"
    assert decision["target_focus"] == ["Q2", "Q3"]
    assert decision["selected_targets"] == []
    assert decision["public_058_evidence"] is False
    assert decision["score_route_opened"] is False
    assert decision["feature_matrix_write_allowed_now"] is False
    assert (
        decision["feature_matrix_write_allowed_next_under_contract"] is True
        or decision["feature_matrix_already_materialized_under_contract"] is True
    )
    assert decision["train_model_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["next_local_allowed_step"] in {
        NEXT_ALLOWED_STEP,
        "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
    }
    assert report["no_feature_matrix_created"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_usage_semantic_burst_markov_feature_matrix_contract_checks_dryrun_and_paths() -> None:
    report = build_report()
    checks = report["checks"]
    paths = report["registered_paths"]
    dry = report["dryrun_summary"]
    writer = report["future_writer_contract"]

    assert checks["spec_contract_ready"] is True
    assert checks["dryrun_contract_ready"] is True
    assert checks["dryrun_schema_ready"] is True
    assert checks["queue_contract_ready"] is True
    assert (
        checks["registered_matrix_absent_before_write"] is True
        and checks["registered_schema_absent_before_write"] is True
    ) or (
        paths["feature_matrix_exists_now"] is True
        and paths["schema_report_exists_now"] is True
        and report["decision"]["feature_matrix_already_materialized_under_contract"] is True
    )
    assert dry["row_count"] == 700
    assert dry["train_rows"] == 450
    assert dry["test_rows"] == 250
    assert dry["feature_column_count"] == 77
    assert dry["finite_all_values"] is True
    assert dry["nan_count"] == 0
    assert paths["feature_matrix_path"] == "data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet"
    assert paths["schema_report_path"] == "reports/post_id341_usage_semantic_burst_markov_features_v1_schema.json"
    assert writer["writer_must_stop_after_schema_report"] is True
    assert writer["model_training_allowed"] is False
    assert writer["submission_artifact_allowed"] is False


def test_usage_semantic_burst_markov_feature_matrix_contract_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "usage_matrix_contract.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["decision"]["create_submission_artifact"] is False
    assert written["decision"]["next_local_allowed_step"] in {
        NEXT_ALLOWED_STEP,
        "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
    }
    assert "Feature matrix write allowed now: `False`" in md
    assert "Feature matrix write allowed next under contract:" in md
    assert "Feature matrix already materialized under contract:" in md
    assert "single registered label-free matrix writer" in direct_md
    assert "does not create ID342" in direct_md
