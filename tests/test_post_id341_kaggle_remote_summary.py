import importlib.util
import json
from pathlib import Path

SUMMARY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_post_id341_kaggle_remote_results.py"
INTAKE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_kaggle_output_intake_gate.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_intake_inputs(intake_module, tmp_path, monkeypatch):
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
    monkeypatch.setattr(intake_module, "APPROVAL_NOTE_PATH", approval)
    monkeypatch.setattr(intake_module, "LOCAL_SYNTHESIS_PATH", synthesis)


def seed_remote_bundle(intake_module, pull_dir: Path, *, review_signal: bool, forbidden: bool = False):
    for index, filename in enumerate(intake_module.EXPECTED_REMOTE_REPORTS):
        decision = {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
        }
        if index == 0 and review_signal:
            decision.update(
                {
                    "public_058_reviewable": True,
                    "projected_public_proxy_from_id341": 0.579,
                    "selected_targets": ["Q1"],
                    "selected_target_count": 1,
                }
            )
        if index == 1 and forbidden:
            decision["submit_now"] = True
        write_json(pull_dir / filename, {"decision": decision})
    write_json(
        pull_dir / intake_module.EXPECTED_RUNNER_MANIFEST,
        {
            "runner": "post_id341_phase2_and_clean_backbone_no_submit_runner",
            "staged_files": [f"kaggle_outputs/{name}" for name in intake_module.EXPECTED_REMOTE_REPORTS],
            "no_submission_created": True,
            "no_dacon_submission_made": True,
            "local_new_source_stack_closed": True,
        },
    )


def build_intake_report(tmp_path, monkeypatch, *, review_signal: bool, forbidden: bool = False):
    intake_module = load_module(INTAKE_SCRIPT, "intake_gate_for_summary_test")
    seed_intake_inputs(intake_module, tmp_path, monkeypatch)
    pull_dir = tmp_path / "pulled"
    seed_remote_bundle(intake_module, pull_dir, review_signal=review_signal, forbidden=forbidden)
    intake_report = intake_module.build_report(pull_dir)
    intake_path = tmp_path / "intake.json"
    write_json(intake_path, intake_report)
    return pull_dir, intake_path


def test_summary_reports_pending_state_from_current_empty_outputs(tmp_path):
    intake = load_module(INTAKE_SCRIPT, "remote_summary_pending_intake_contract")
    summary = load_module(SUMMARY_SCRIPT, "remote_summary_pending_test")
    report = summary.build_report(tmp_path / "missing", tmp_path / "missing_intake.json")

    assert report["decision"]["output_pending"] is True
    assert (
        "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
        in intake.EXPECTED_REMOTE_REPORTS
    )
    assert (
        "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
        in intake.EXPECTED_REMOTE_REPORTS
    )
    assert report["decision"]["remote_summary_reviewable"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_summary_allows_review_only_when_intake_allows(tmp_path, monkeypatch):
    pull_dir, intake_path = build_intake_report(tmp_path, monkeypatch, review_signal=True)
    summary = load_module(SUMMARY_SCRIPT, "remote_summary_review_test")

    report = summary.build_report(pull_dir, intake_path)

    assert report["decision"]["output_pending"] is False
    assert report["decision"]["artifact_contract_review_allowed_by_intake"] is True
    assert report["decision"]["remote_summary_reviewable"] is True
    assert report["decision"]["public_058_evidence_present"] is True
    assert report["decision"]["beats_id178_public_anchor_evidence_present"] is True
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_summary_blocks_review_when_remote_forbidden_action_exists(tmp_path, monkeypatch):
    pull_dir, intake_path = build_intake_report(tmp_path, monkeypatch, review_signal=True, forbidden=True)
    summary = load_module(SUMMARY_SCRIPT, "remote_summary_forbidden_test")

    report = summary.build_report(pull_dir, intake_path)

    assert report["counts"]["forbidden_action_rows"] == 1
    assert report["decision"]["artifact_contract_review_allowed_by_intake"] is False
    assert report["decision"]["remote_summary_reviewable"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
