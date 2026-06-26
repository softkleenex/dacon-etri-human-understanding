from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_post_id341_gps_path_topology_feature_matrix_smoke_contract import (
    build_markdown,
    build_report,
    write_report,
)


def test_gps_feature_matrix_smoke_contract_ready_but_writes_nothing() -> None:
    report = build_report()
    dec = report["decision"]

    assert dec["feature_matrix_smoke_contract_ready"] is True
    assert dec["source_family"] == "gps_path_topology_sequence_v1"
    assert set(dec["target_focus"]) == {"Q3", "S3"}
    assert dec["feature_matrix_write_allowed_now"] is False
    assert (
        dec["feature_matrix_write_allowed_next_under_contract"] is True
        or dec["feature_matrix_already_materialized_under_contract"] is True
    )
    assert dec["allowed_future_matrix_path"] == "data/processed/post_id341_gps_path_topology_features_v1.parquet"
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert report["no_feature_matrix_written"] is True
    assert report["no_submission_log_row_created"] is True


def test_gps_feature_matrix_smoke_contract_checks_dryrun_frontier_and_paths() -> None:
    report = build_report()
    checks = report["checks"]

    assert all(checks["input_contract"].values())
    assert all(checks["dryrun_contract"].values())
    assert all(checks["frontier_contract"].values())
    assert all(checks["smoke_contract"].values())
    assert all(checks["path_contract"].values())
    assert "existing_registered_output_contract" in checks
    assert checks["dryrun_contract"]["row_alignment_exact"] is True
    assert checks["dryrun_contract"]["feature_columns_bounded"] is True
    assert checks["dryrun_contract"]["coverage_passes"] is True
    assert checks["frontier_contract"]["no_approval_score_route_absent"] is True


def test_gps_feature_matrix_smoke_contract_future_boundary_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "gps_contract.json"
    report = build_report()
    write_report(report, output)

    written = json.loads(output.read_text())
    md = output.with_suffix(".md").read_text()
    direct_md = build_markdown(report)

    assert written["allowed_future_write_contract"]["must_not_train_model"] is True
    assert written["allowed_future_write_contract"]["must_not_modify_submission_log"] is True
    assert written["allowed_future_write_contract"]["must_revalidate_after_write"] is True
    assert "writes no feature matrix/parquet" in md
    assert "Feature matrix write allowed now: `False`" in md
    assert "Feature matrix write allowed next under contract:" in md
    assert "Feature matrix already materialized under contract:" in md
    assert "does not create ID342" in direct_md
