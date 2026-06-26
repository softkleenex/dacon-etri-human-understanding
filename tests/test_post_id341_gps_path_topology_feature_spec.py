from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_gps_path_topology_feature_spec import build_report, write_report


def test_gps_path_topology_feature_spec_is_ready_but_report_only() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["feature_spec_ready"] is True
    assert dec["source_family"] == "gps_path_topology_sequence_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3"}
    assert dec["next_local_allowed_step"] == "report_only_feature_generation_dryrun"
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False


def test_gps_path_topology_feature_spec_has_material_blocks_and_leakage_guards() -> None:
    report = build_report()
    block_names = {block["block"] for block in report["feature_blocks"]}
    guard_text = "\n".join(report["leakage_guards"]).lower()

    assert {
        "gps_payload_quality",
        "path_motion_shape",
        "dwell_segment_topology",
        "route_novelty_and_regularization",
        "place_social_context_alignment",
    }.issubset(block_names)
    assert "no target labels" in guard_text
    assert "training fold only" in guard_text
    assert report["resource_bounds"]["max_daily_feature_columns_for_first_matrix"] <= 64


def test_gps_path_topology_feature_spec_markdown_blocks_feature_matrix_and_id342(tmp_path: Path) -> None:
    output = tmp_path / "feature_spec.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["feature_matrix_allowed_now"] is False
    assert "creates no feature matrix" in md
    assert "Feature matrix allowed now: `False`" in md
    assert "does not allow a feature parquet yet" in md
