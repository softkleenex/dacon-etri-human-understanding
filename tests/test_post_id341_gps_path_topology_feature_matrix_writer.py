from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.write_post_id341_gps_path_topology_feature_matrix import (
    FIRST_DRYRUN_FEATURES,
    KEYS,
    TARGET_FOCUS,
    build_feature_matrix,
    build_schema_report,
    validate_feature_matrix,
    verify_contract_allows_write,
)


@pytest.fixture(scope="module")
def gps_matrix_bundle() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    return build_feature_matrix()


def test_gps_feature_matrix_is_label_free_aligned_and_bounded(
    gps_matrix_bundle: tuple[pd.DataFrame, dict[str, Any], pd.DataFrame],
) -> None:
    frame, metadata, expected = gps_matrix_bundle
    validation = validate_feature_matrix(frame, metadata, expected)
    checks = validation["checks"]

    assert validation["allowed_boundary_passed"] is True
    assert len(frame) == 700
    assert list(frame.columns[: len(KEYS)]) == KEYS
    assert [column for column in frame.columns if column not in KEYS] == FIRST_DRYRUN_FEATURES
    assert validation["feature_columns"]["feature_column_count"] == 32
    assert validation["feature_columns"]["feature_column_count"] <= 64
    assert checks["row_alignment_exact"] is True
    assert checks["duplicate_keys_absent"] is True
    assert checks["train_rows_exact"] is True
    assert checks["test_rows_exact"] is True
    assert checks["no_target_or_prediction_columns"] is True
    assert checks["all_feature_values_finite"] is True
    assert validation["coverage"]["by_split"]["train"]["gps_topology_ready_rate"] >= 0.90
    assert validation["coverage"]["by_split"]["test"]["gps_topology_ready_rate"] >= 0.90


def test_gps_feature_matrix_schema_report_keeps_score_actions_closed(
    gps_matrix_bundle: tuple[pd.DataFrame, dict[str, Any], pd.DataFrame], tmp_path: Path
) -> None:
    frame, metadata, expected = gps_matrix_bundle
    matrix_path = tmp_path / "post_id341_gps_path_topology_features_v1.parquet"
    frame.to_parquet(matrix_path, index=False)
    schema_path = tmp_path / "post_id341_gps_path_topology_features_v1_schema.json"
    contract_checks = {
        "contract_report_parse_ok": True,
        "contract_ready_or_already_materialized": True,
        "source_family_matches": True,
        "target_focus_matches": True,
        "matrix_path_matches_contract": True,
        "schema_path_matches_contract": True,
        "row_bound_matches_contract": True,
        "column_bound_matches_contract": True,
        "no_model_allowed_by_contract": True,
        "no_submission_allowed_by_contract": True,
        "no_submission_log_allowed_by_contract": True,
        "no_kaggle_allowed_by_contract": True,
        "no_dacon_allowed_by_contract": True,
    }
    report = build_schema_report(frame, metadata, expected, contract_checks, matrix_path, schema_path, True)
    dec = report["decision"]
    safety = report["safety_contract"]

    assert dec["feature_matrix_materialized"] is True
    assert dec["matrix_written_now"] is True
    assert dec["target_focus"] == TARGET_FOCUS
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert safety["uses_target_labels"] is False
    assert safety["uses_oof_predictions"] is False
    assert safety["submission_log_modified"] is False


def test_gps_feature_matrix_materialize_to_tmp_uses_contract_and_writes_only_allowed_shapes(tmp_path: Path) -> None:
    contract = json.loads(Path("reports/post_id341_gps_path_topology_feature_matrix_smoke_contract_20260619.json").read_text())
    matrix_path = Path("data/processed/post_id341_gps_path_topology_features_v1.parquet")
    schema_path = Path("reports/post_id341_gps_path_topology_features_v1_schema.json")
    checks = verify_contract_allows_write(contract, matrix_path, schema_path)

    assert all(checks.values())

    # The real CLI is path-locked by the contract. This test validates the pure writer
    # through the same build/validation path without touching submissions or logs.
    frame, metadata, expected = build_feature_matrix()
    tmp_matrix = tmp_path / "matrix.parquet"
    tmp_schema = tmp_path / "schema.json"
    frame.to_parquet(tmp_matrix, index=False)
    report = build_schema_report(frame, metadata, expected, dict.fromkeys(checks, True), tmp_matrix, tmp_schema, True)
    loaded = pd.read_parquet(tmp_matrix)

    assert loaded.shape == (700, 35)
    assert report["decision"]["feature_column_count"] == 32
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_gps_feature_matrix_real_artifact_if_present_is_valid() -> None:
    matrix_path = Path("data/processed/post_id341_gps_path_topology_features_v1.parquet")
    if not matrix_path.exists():
        pytest.skip("Real GPS feature matrix has not been materialized in this workspace yet.")

    frame, metadata, expected = build_feature_matrix()
    existing = pd.read_parquet(matrix_path)
    validation = validate_feature_matrix(existing, metadata, expected)

    assert existing.shape == frame.shape
    assert list(existing.columns) == list(frame.columns)
    assert validation["allowed_boundary_passed"] is True
