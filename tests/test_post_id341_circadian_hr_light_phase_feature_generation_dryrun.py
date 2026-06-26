from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.diagnose_post_id341_circadian_hr_light_phase_feature_generation_dryrun import (
    build_report,
    write_report,
)


@pytest.fixture(scope="module")
def circadian_dryrun_report() -> dict[str, Any]:
    return build_report()


def test_circadian_feature_generation_dryrun_passes_but_is_report_only(
    circadian_dryrun_report: dict[str, Any],
) -> None:
    report = circadian_dryrun_report
    dec = report["decision"]
    safety = report["safety_contract"]

    assert dec["dryrun_passed"] is True
    assert dec["source_family"] == "circadian_hr_light_phase_shape_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3", "S2"}
    assert dec["row_alignment_exact"] is True
    assert dec["feature_column_bound_passed"] is True
    assert dec["feature_column_count"] == 48
    assert dec["train_strong_phase_shape_rate"] >= 0.50
    assert dec["test_strong_phase_shape_rate"] >= 0.50
    assert dec["train_weak_phase_shape_rate"] >= 0.70
    assert dec["test_weak_phase_shape_rate"] >= 0.70
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
    assert safety["uses_oof_predictions"] is False
    assert safety["uses_public_scores"] is False


def test_circadian_feature_generation_dryrun_alignment_and_schema(
    circadian_dryrun_report: dict[str, Any],
) -> None:
    report = circadian_dryrun_report

    assert report["row_alignment"]["expected_rows"] == 700
    assert report["row_alignment"]["dryrun_rows"] == 700
    assert report["row_alignment"]["duplicate_keys"] == 0
    assert report["row_alignment"]["train_rows"] == 450
    assert report["row_alignment"]["test_rows"] == 250
    assert report["feature_columns"]["missing_declared_features"] == []
    assert report["feature_columns"]["extra_features"] == []
    assert report["feature_columns"]["all_values_finite"] is True
    assert report["coverage"]["by_split"]["train"]["strong_phase_shape_rows"] > 0
    assert report["coverage"]["by_split"]["test"]["strong_phase_shape_rows"] > 0


def test_circadian_feature_generation_dryrun_markdown_blocks_artifacts(
    tmp_path: Path,
    circadian_dryrun_report: dict[str, Any],
) -> None:
    output = tmp_path / "circadian_dryrun.json"
    write_report(circadian_dryrun_report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["feature_matrix_allowed_now"] is False
    assert "creates no feature matrix/parquet" in md
    assert "Feature matrix allowed now: `False`" in md
    assert "does not open ID342" in md
