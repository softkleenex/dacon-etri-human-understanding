from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.diagnose_post_id341_gps_path_topology_model_smoke import build_report, write_report


def test_gps_path_topology_model_smoke_is_no_submit_and_bounded() -> None:
    report = build_report()
    decision = report["decision"]
    safety = report["safety_contract"]
    metadata = report["feature_metadata"]

    assert decision["model_smoke_completed"] is True
    assert decision["source_family"] == "gps_path_topology_sequence_v1"
    assert set(decision["target_focus"]) == {"Q3", "S3"}
    assert decision["public_058_evidence"] is False
    assert decision["score_route_opened"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert safety["in_memory_models_trained"] is True
    assert safety["persisted_model_created"] is False
    assert safety["full_test_predictions_created"] is False
    assert safety["submission_csv_created"] is False
    assert safety["submission_log_modified"] is False
    assert metadata["aligned_train_rows"] == 450
    assert metadata["feature_columns"] == 32
    assert metadata["forbidden_feature_columns"] == []


def test_gps_path_topology_model_smoke_checks_contract_and_reports_targets() -> None:
    report = build_report()
    contract_checks = report["contract"]["checks"]

    assert report["contract"]["contract_ready_for_model_smoke"] is True
    assert all(contract_checks.values())
    assert set(report["top_rows_by_target"]) == {"Q3", "S3"}
    assert report["scan_row_count"] > 0
    for target, rows in report["top_rows_by_target"].items():
        assert target in {"Q3", "S3"}
        assert rows
        assert {"candidate", "primary_wins_vs_id341", "mean_delta_vs_id341", "max_delta_vs_id341"}.issubset(rows[0])


def test_gps_path_topology_model_smoke_markdown_keeps_artifact_boundary(tmp_path: Path) -> None:
    output = tmp_path / "gps_model_smoke.json"
    report: dict[str, Any] = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["create_submission_artifact"] is False
    assert written["decision"]["submit_now"] is False
    assert "no-submit, in-memory model smoke" in md
    assert "does not create ID342" in md
