from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_gps_path_topology_metadata_probe import build_report, write_report


def test_gps_path_topology_metadata_probe_is_report_only_and_feasible() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["metadata_probe_passed"] is True
    assert dec["source_family"] == "gps_path_topology_sequence_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3"}
    assert dec["next_local_allowed_step"] == "design_feature_spec_only"
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False


def test_gps_path_topology_metadata_probe_uses_queue_and_raw_place_sources() -> None:
    report = build_report()

    assert report["queue_report"]["gps_queue_status"] == "closed_after_feature_matrix_and_model_smoke"
    assert report["queue_report"]["gps_next_allowed_step"] == "closed_no_artifact"
    assert set(report["source_summaries"]) == {"mGps", "mWifi", "mBle", "mActivity"}
    assert report["source_summaries"]["mGps"]["coverage_by_split"]["test"]["coverage_rate"] > 0.5
    assert report["gps_feasibility"]["payload"]["rows_with_valid_coordinate_payload"] > 0
    assert report["gps_feasibility"]["feasibility_by_split"]["train"]["strong_topology_days"] > 0


def test_gps_path_topology_metadata_probe_markdown_blocks_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "gps_probe.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["public_058_evidence"] is False
    assert "creates no predictions" in md
    assert "Train model now: `False`" in md
    assert "does not open a submission artifact" in md
