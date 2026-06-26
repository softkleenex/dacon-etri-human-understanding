import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_kaggle_output_intake_gate.py"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("output_intake_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_local_gate_inputs(module, tmp_path, monkeypatch):
    approval = tmp_path / "approval.json"
    synthesis = tmp_path / "synthesis.json"
    write_json(approval, {"decision": {"approval_note_ready": True, "execute_kaggle_now": False}})
    write_json(
        synthesis,
        {
            "decision": {
                "local_source_stack_closed_for_submission_artifact": True,
                "any_local_lane_promotable": False,
            }
        },
    )
    monkeypatch.setattr(module, "APPROVAL_NOTE_PATH", approval)
    monkeypatch.setattr(module, "LOCAL_SYNTHESIS_PATH", synthesis)


def seed_remote_reports(module, pull_dir: Path, *, review_signal: bool):
    for index, filename in enumerate(module.EXPECTED_REMOTE_REPORTS):
        decision = {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
        }
        if index == 0 and review_signal:
            decision["public_058_reviewable"] = True
        write_json(pull_dir / filename, {"decision": decision})


def seed_runner_manifest(module, pull_dir: Path):
    write_json(
        pull_dir / module.EXPECTED_RUNNER_MANIFEST,
        {
            "runner": "post_id341_phase2_and_clean_backbone_no_submit_runner",
            "staged_files": [f"kaggle_outputs/{name}" for name in module.EXPECTED_REMOTE_REPORTS],
            "no_submission_created": True,
            "no_dacon_submission_made": True,
            "local_new_source_stack_closed": True,
        },
    )


def test_manifest_is_required_once_outputs_exist(tmp_path, monkeypatch):
    module = load_gate_module()
    assert "remote_post_id341_q3s3_negative_evidence_map_20260618.json" in module.EXPECTED_REMOTE_REPORTS
    assert "remote_post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json" in module.EXPECTED_REMOTE_REPORTS
    assert "remote_post_id341_q3s3_remaining_route_inventory_20260618.json" in module.EXPECTED_REMOTE_REPORTS
    assert (
        "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
        in module.EXPECTED_REMOTE_REPORTS
    )
    assert (
        "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
        in module.EXPECTED_REMOTE_REPORTS
    )
    seed_local_gate_inputs(module, tmp_path, monkeypatch)
    pull_dir = tmp_path / "pulled"
    seed_remote_reports(module, pull_dir, review_signal=True)

    report = module.build_report(pull_dir)

    assert report["decision"]["output_pending"] is False
    assert report["decision"]["all_expected_reports_present"] is True
    assert report["decision"]["runner_manifest_present"] is False
    assert report["decision"]["runner_manifest_missing_or_invalid"] is True
    assert report["decision"]["artifact_contract_review_allowed"] is False


def test_valid_manifest_and_review_signal_allow_separate_artifact_review(tmp_path, monkeypatch):
    module = load_gate_module()
    seed_local_gate_inputs(module, tmp_path, monkeypatch)
    pull_dir = tmp_path / "pulled"
    seed_remote_reports(module, pull_dir, review_signal=True)
    seed_runner_manifest(module, pull_dir)

    report = module.build_report(pull_dir)

    assert report["decision"]["runner_manifest_present"] is True
    assert report["decision"]["runner_manifest_ok"] is True
    assert report["expected_runner_manifest"]["missing_staged_expected_reports"] == []
    assert report["decision"]["artifact_contract_review_allowed"] is True
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_usage_validation_stop_is_required_but_not_a_review_signal(tmp_path, monkeypatch):
    module = load_gate_module()
    seed_local_gate_inputs(module, tmp_path, monkeypatch)
    pull_dir = tmp_path / "pulled"
    seed_remote_reports(module, pull_dir, review_signal=False)
    seed_runner_manifest(module, pull_dir)
    usage_file = pull_dir / "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
    write_json(
        usage_file,
        {
            "decision": {
                "validation_stop_passed": True,
                "public_058_evidence": False,
                "score_route_opened": False,
                "train_model_now": False,
                "create_submission_artifact": False,
                "submit_now": False,
                "push_kaggle_now": False,
                "run_kaggle_now": False,
            }
        },
    )

    report = module.build_report(pull_dir)
    usage_row = [
        row
        for row in report["expected_report_status"]
        if row["filename"]
        == "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
    ][0]

    assert usage_row["exists"] is True
    assert usage_row["decision"]["validation_stop_passed"] is True
    assert usage_row["decision"]["score_route_opened"] is False
    assert report["decision"]["remote_review_signal_present"] is False
    assert report["decision"]["artifact_contract_review_allowed"] is False
    assert report["decision"]["create_submission_artifact"] is False


def test_self_supervised_design_is_required_but_not_a_review_signal(tmp_path, monkeypatch):
    module = load_gate_module()
    seed_local_gate_inputs(module, tmp_path, monkeypatch)
    pull_dir = tmp_path / "pulled"
    seed_remote_reports(module, pull_dir, review_signal=False)
    seed_runner_manifest(module, pull_dir)
    design_file = pull_dir / "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
    write_json(
        design_file,
        {
            "decision": {
                "design_ready": True,
                "source_family": "self_supervised_subject_state_embedding_v1",
                "public_058_evidence": False,
                "score_route_opened": False,
                "feature_matrix_allowed_now": False,
                "train_model_now": False,
                "create_submission_artifact": False,
                "submit_now": False,
                "push_kaggle_now": False,
                "run_kaggle_now": False,
            }
        },
    )

    report = module.build_report(pull_dir)
    design_row = [
        row
        for row in report["expected_report_status"]
        if row["filename"] == "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
    ][0]

    assert design_row["exists"] is True
    assert design_row["decision"]["design_ready"] is True
    assert design_row["decision"]["score_route_opened"] is False
    assert report["decision"]["remote_review_signal_present"] is False
    assert report["decision"]["artifact_contract_review_allowed"] is False
    assert report["decision"]["create_submission_artifact"] is False


def test_submission_like_file_blocks_artifact_review_even_with_valid_manifest(tmp_path, monkeypatch):
    module = load_gate_module()
    seed_local_gate_inputs(module, tmp_path, monkeypatch)
    pull_dir = tmp_path / "pulled"
    seed_remote_reports(module, pull_dir, review_signal=True)
    seed_runner_manifest(module, pull_dir)
    (pull_dir / "accidental_submission.csv").write_text("id,Q1\n0,0.5\n", encoding="utf-8")

    report = module.build_report(pull_dir)

    assert report["decision"]["runner_manifest_ok"] is True
    assert report["decision"]["unexpected_submission_like_files_present"] is True
    assert report["decision"]["artifact_contract_review_allowed"] is False
