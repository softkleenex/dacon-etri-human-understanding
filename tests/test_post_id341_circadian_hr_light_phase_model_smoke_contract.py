from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_circadian_hr_light_phase_model_smoke_contract import (
    build_markdown,
    build_report,
    write_report,
)


def test_circadian_model_smoke_contract_ready_but_trains_nothing() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["model_smoke_contract_ready"] is True
    assert dec["source_family"] == "circadian_hr_light_phase_shape_v1"
    assert dec["target_focus"] == ["Q3", "S3", "S2"]
    assert dec["model_smoke_run_allowed_now"] is False
    assert dec["model_smoke_run_allowed_next_under_contract"] is False
    assert dec["model_smoke_already_completed_under_contract"] is True
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_circadian_model_smoke_contract_checks_matrix_queue_and_gap_gates() -> None:
    report = build_report()
    checks = report["checks"]

    assert all(checks["schema_contract"].values())
    assert all(checks["feature_matrix_contract"].values())
    assert all(checks["queue_contract"].values())
    assert all(checks["frontier_contract"].values())
    assert all(checks["public_gap_contract"].values())
    assert all(checks["local_smoke_contract"].values())
    assert checks["schema_contract"]["schema_row_count_exact"] is True
    assert checks["schema_contract"]["schema_feature_column_count_exact"] is True
    assert checks["public_gap_contract"]["no_watch_candidate_passes_floor"] is True
    assert checks["frontier_contract"]["frontier_no_approval_score_route_absent"] is True


def test_circadian_model_smoke_contract_future_boundary_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "circadian_model_contract.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)
    future = written["allowed_future_model_smoke_contract"]

    assert future["allowed_train_rows_exact"] == 450
    assert future["allowed_feature_columns_exact"] == 48
    assert future["must_fit_preprocessing_inside_each_outer_train_fold"] is True
    assert future["must_not_create_full_test_predictions"] is True
    assert future["must_not_write_oof_predictions"] is True
    assert future["must_not_persist_model"] is True
    assert future["must_not_create_submission_artifact"] is True
    assert "trains no model" in md
    assert "Model-smoke run allowed now: `False`" in md
    assert "Model-smoke run allowed next under contract: `False`" in md
    assert "Model-smoke already completed under contract: `True`" in md
    assert "does not create ID342" in direct_md
