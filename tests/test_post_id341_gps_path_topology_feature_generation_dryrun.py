from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.diagnose_post_id341_gps_path_topology_feature_generation_dryrun import build_report, write_report


@pytest.fixture(scope="module")
def gps_dryrun_report() -> dict[str, Any]:
    return build_report()


def test_gps_path_topology_feature_generation_dryrun_passes_but_is_report_only(
    gps_dryrun_report: dict[str, Any],
) -> None:
    report = gps_dryrun_report
    dec = report["decision"]
    safety = report["safety_contract"]

    assert dec["dryrun_passed"] is True
    assert dec["source_family"] == "gps_path_topology_sequence_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3"}
    assert dec["row_alignment_exact"] is True
    assert dec["feature_column_bound_passed"] is True
    assert dec["feature_column_count"] <= 64
    assert dec["train_gps_topology_ready_rate"] >= 0.90
    assert dec["test_gps_topology_ready_rate"] >= 0.90
    assert dec["public_058_evidence"] is False
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert safety["feature_matrix_written"] is False
    assert safety["submission_log_modified"] is False
    assert safety["uses_target_labels"] is False


def test_gps_path_topology_feature_generation_dryrun_alignment_and_cardinality(
    gps_dryrun_report: dict[str, Any],
) -> None:
    report = gps_dryrun_report

    assert report["row_alignment"]["expected_rows"] == report["row_alignment"]["dryrun_rows"]
    assert report["row_alignment"]["duplicate_keys"] == 0
    assert report["coverage"]["by_split"]["train"]["gps_topology_ready_rows"] > 0
    assert report["coverage"]["by_split"]["test"]["gps_topology_ready_rows"] > 0
    assert report["cardinality_bounds"]["wifi_cardinality"]["max_tokens_after_capping"] <= 64
    assert report["cardinality_bounds"]["ble_cardinality"]["max_tokens_after_capping"] <= 64


def test_gps_path_topology_feature_generation_dryrun_markdown_blocks_artifacts(
    tmp_path: Path,
    gps_dryrun_report: dict[str, Any],
) -> None:
    output = tmp_path / "gps_dryrun.json"
    report = gps_dryrun_report
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["feature_matrix_allowed_now"] is False
    assert "creates no feature matrix/parquet" in md
    assert "Feature matrix allowed now: `False`" in md
    assert "does not open ID342" in md
