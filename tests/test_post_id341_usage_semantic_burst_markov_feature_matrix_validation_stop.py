from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop import (
    build_markdown,
    build_report,
    write_report,
)


def test_usage_semantic_burst_markov_validation_stop_passes_and_blocks_actions() -> None:
    report = build_report()
    dec = report["decision"]
    checks = report["checks"]
    safety = report["safety_contract"]

    assert dec["validation_stop_passed"] is True
    assert dec["source_family"] == "usage_semantic_burst_markov_v1"
    assert dec["target_focus"] == ["Q2", "Q3"]
    assert dec["selected_targets"] == []
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert dec["next_local_allowed_step"] == "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval"
    assert all(checks.values())
    assert safety["model_trained"] is False
    assert safety["predictions_created"] is False
    assert safety["submission_log_modified"] is False
    assert safety["kaggle_run_executed"] is False
    assert safety["dacon_submission_executed"] is False


def test_usage_semantic_burst_markov_validation_stop_checks_matrix_and_rebuild_match() -> None:
    report = build_report()
    matrix = report["inputs"]["feature_matrix"]
    validation = report["validation"]
    rebuild = report["deterministic_rebuild"]
    submission_log = report["submission_log_status"]

    assert matrix["path"] == "data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet"
    assert matrix["exists"] is True
    assert matrix["shape"] == [700, 80]
    assert matrix["sha256"]
    assert validation["allowed_boundary_passed"] is True
    assert validation["feature_columns"]["feature_column_count"] == 77
    assert rebuild["same_shape"] is True
    assert rebuild["same_columns"] is True
    assert rebuild["keys_match"] is True
    assert rebuild["values_match_with_tolerance_1e_12"] is True
    assert rebuild["deterministic_rebuild_match"] is True
    assert rebuild["numeric_delta_max"] == 0.0
    assert submission_log["row_count"] >= 341
    assert submission_log["max_id"] >= 341
    assert submission_log["source_family_row_count"] == 0
    assert report["source_family_submission_artifacts"] == []


def test_usage_semantic_burst_markov_validation_stop_markdown_boundary(tmp_path: Path) -> None:
    output = tmp_path / "usage_validation_stop.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["decision"]["validation_stop_passed"] is True
    assert written["decision"]["create_submission_artifact"] is False
    assert "Validation stop passed: `True`" in md
    assert "Train model now: `False`" in md
    assert "Run Kaggle now: `False`" in md
    assert "creates no model" in direct_md
    assert "Do not use it to train a local model" in direct_md
