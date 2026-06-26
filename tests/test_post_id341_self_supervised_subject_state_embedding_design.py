import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_self_supervised_subject_state_embedding_design.py"


def load_module():
    spec = importlib.util.spec_from_file_location("self_supervised_subject_state_embedding_design", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_self_supervised_subject_state_design_is_ready_but_execution_blocked() -> None:
    module = load_module()
    report = module.build_report()
    dec = report["decision"]

    assert dec["design_ready"] is True
    assert dec["source_family"] == "self_supervised_subject_state_embedding_v1"
    assert dec["target_focus"] == ["Q2", "Q3", "S3"]
    assert dec["materially_new_but_heavy"] is True
    assert dec["local_no_approval_allowed"] is False
    assert dec["requires_explicit_heavy_compute_approval"] is True
    assert dec["public_058_evidence"] is False
    assert dec["score_route_opened"] is False
    assert dec["feature_matrix_allowed_now"] is False
    assert dec["train_model_now"] is False
    assert dec["create_submission_artifact"] is False
    assert dec["submit_now"] is False
    assert dec["push_kaggle_now"] is False
    assert dec["run_kaggle_now"] is False
    assert report["no_feature_matrix_created"] is True
    assert report["no_model_trained"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_dacon_submission_made"] is True


def test_self_supervised_subject_state_design_metadata_contract() -> None:
    module = load_module()
    report = module.build_report()
    raw = report["raw_modality_manifest"]
    row = report["row_contract"]
    context = report["context_status"]

    assert raw["raw_modality_count"] == 12
    assert raw["expected_raw_modality_count"] == 12
    assert raw["all_expected_modalities_present"] is True
    assert raw["all_streams_have_subject_and_timestamp"] is True
    assert raw["total_raw_rows"] >= 6_000_000
    assert row["official_train_rows"] == 450
    assert row["official_test_rows"] == 250
    assert row["processed_train_rows"] == 450
    assert row["processed_test_rows"] == 250
    assert row["train_key_duplicates"] == 0
    assert row["test_key_duplicates"] == 0
    assert context["all_context_reports_present"] is True
    assert context["queue_records_self_supervised_heavy_deferred"] is True
    assert context["all_local_no_approval_score_routes_closed"] is True
    assert context["local_source_stack_closed"] is True


def test_self_supervised_subject_state_markdown_forbids_heavy_actions() -> None:
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "self-supervised subject-state embedding design" in markdown
    assert "reads metadata, not full raw streams" in markdown
    assert "Local no-approval allowed: `False`" in markdown
    assert "Requires explicit heavy-compute approval: `True`" in markdown
    assert "Feature matrix allowed now: `False`" in markdown
    assert "Train model now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
    assert "No full-OOF calibration" in markdown
