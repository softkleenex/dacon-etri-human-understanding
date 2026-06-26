from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_circadian_hr_light_phase_feature_spec import build_report, write_report


def test_circadian_hr_light_phase_feature_spec_is_ready_but_report_only() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["feature_spec_ready"] is True
    assert dec["source_family"] == "circadian_hr_light_phase_shape_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3", "S2"}
    assert dec["metadata_probe_passed"] is True
    assert dec["next_local_allowed_step"] == "report_only_feature_generation_dryrun_no_matrix"
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False


def test_circadian_hr_light_phase_feature_spec_has_phase_blocks_and_leakage_guards() -> None:
    report = build_report()
    block_names = {block["block"] for block in report["feature_blocks"]}
    guard_text = "\n".join(report["leakage_guards"]).lower()

    assert {
        "phase_source_quality",
        "wearable_hr_phase_shape",
        "wearable_light_phase_shape",
        "phone_light_dark_continuity",
        "rest_activity_phase_shape",
        "cross_modal_phase_alignment",
    }.issubset(block_names)
    assert "no target labels" in guard_text
    assert "predeclared hour bins" in guard_text
    assert "no feature parquet" in guard_text
    assert report["resource_bounds"]["planned_first_pass_feature_columns"] <= 56


def test_circadian_hr_light_phase_feature_spec_markdown_blocks_matrix_and_id342(tmp_path: Path) -> None:
    output = tmp_path / "feature_spec.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()

    assert written["decision"]["feature_matrix_allowed_now"] is False
    assert "creates no feature matrix" in md
    assert "Feature matrix allowed now: `False`" in md
    assert "does not allow a feature parquet yet" in md
    assert "ID342" in md
