from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_self_supervised_heavy_compute_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("self_supervised_heavy_compute_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_self_supervised_heavy_compute_contract_ready_but_actions_blocked() -> None:
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["heavy_compute_contract_ready"] is True
    assert decision["source_family"] == "self_supervised_subject_state_embedding_v1"
    assert decision["target_focus"] == ["Q2", "Q3", "S3"]
    assert decision["heavy_compute_allowed_now"] is False
    assert decision["requires_explicit_heavy_compute_approval"] is True
    assert decision["local_no_approval_allowed"] is False
    assert decision["public_058_evidence"] is False
    assert decision["score_route_opened"] is False
    assert decision["feature_matrix_allowed_now"] is False
    assert decision["train_model_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["modify_submission_log"] is False
    assert decision["create_id342_or_new_submission_id"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_full_raw_stream_read"] is True
    assert report["no_feature_matrix_created"] is True
    assert report["no_model_trained"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_self_supervised_heavy_compute_uses_separate_approval_phrase() -> None:
    module = load_module()
    report = module.build_report()
    phrases = report["approval_phrases"]
    decision = report["decision"]

    assert phrases["phrases_are_separate"] is True
    assert phrases["self_supervised_heavy_compute_no_submit"] == (
        "Self-supervised subject-state heavy compute no-submit 실행 승인"
    )
    assert phrases["kaggle_upstream_rebuild_no_submit"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["approval_phrase"] == phrases["self_supervised_heavy_compute_no_submit"]
    assert decision["approval_phrase"] != phrases["kaggle_upstream_rebuild_no_submit"]
    assert decision["current_kaggle_upstream_approval_phrase_only_applies_to_upstream_rebuild"] is True


def test_self_supervised_heavy_compute_prerequisites_allow_recorded_id342_state() -> None:
    module = load_module()
    report = module.build_report()
    prerequisites = report["prerequisites"]
    raw = prerequisites["raw_manifest_summary"]
    row = prerequisites["row_contract_summary"]
    log = prerequisites["submission_log_status"]

    assert prerequisites["design_report_ready"] is True
    assert prerequisites["raw_manifest_ready"] is True
    assert prerequisites["row_contract_ready"] is True
    assert prerequisites["queue_records_heavy_deferred"] is True
    assert prerequisites["no_approval_frontier_closed"] is True
    assert prerequisites["prelaunch_freshness_ready"] is True
    assert prerequisites["submission_log_ready_for_existing_id342_state"] is True
    assert raw["raw_modality_count"] == 12
    assert raw["expected_raw_modality_count"] == 12
    assert raw["total_raw_rows"] >= 6_000_000
    assert raw["all_streams_have_subject_and_timestamp"] is True
    assert row["official_train_rows"] == 450
    assert row["official_test_rows"] == 250
    assert row["processed_train_rows"] == 450
    assert row["processed_test_rows"] == 250
    assert row["row_contract_exact"] is True
    assert log["max_id"] == 342
    assert log["ids_are_unique"] is True
    assert log["has_id342_row"] is True
    assert log["id342_row_count"] == 1
    assert log["id342_existing_generated_unsubmitted"] is True
    assert log["log_ready_for_future_unique_id"] is True
    assert log["next_numeric_id_if_later_authorized"] == 343


def test_self_supervised_heavy_compute_plan_is_label_free_and_no_submit() -> None:
    module = load_module()
    report = module.build_report()
    plan = report["bounded_heavy_plan"]
    forbidden = report["forbidden_actions_now"]

    assert len(plan) >= 6
    assert all(stage["allowed_now"] is False for stage in plan)
    assert any("fixed vocabularies" in stage["after_exact_heavy_approval"] for stage in plan)
    assert any("masked reconstruction" in stage["after_exact_heavy_approval"] for stage in plan)
    assert any("subject-hole" in stage["after_exact_heavy_approval"] for stage in plan)
    assert all(value is False for value in forbidden.values())


def test_self_supervised_heavy_compute_markdown_forbids_training_and_submission() -> None:
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "self-supervised heavy-compute contract" in markdown
    assert "Heavy-compute contract ready: `True`" in markdown
    assert "Heavy compute allowed now: `False`" in markdown
    assert "Approval phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`" in markdown
    assert "Kaggle upstream phrase only applies to upstream rebuild: `True`" in markdown
    assert "Feature matrix allowed now: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Create new submission id: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
    assert "Submit now: `False`" in markdown
