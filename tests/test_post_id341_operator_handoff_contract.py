import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_operator_handoff_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("operator_handoff", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_operator_handoff_is_ready_but_requires_exact_approval():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["operator_handoff_ready"] is True
    assert decision["safe_wait_state_without_user_action"] is True
    assert decision["requires_exact_approval_phrase"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["artifact_review_open"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_operator_handoff_sequences_are_guarded_and_match_authoritative_packets():
    module = load_module()
    report = module.build_report()

    assert report["local_preapproval_commands"] == [
        "make kaggle-approval-contract-test",
        "make kaggle-upstream-prelaunch-local",
    ]
    assert report["approval_required_commands"] == [
        "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset",
        "ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files",
        "ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files",
        "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner",
        "ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild",
        "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output",
        "make kaggle-review-upstream-rebuild-output",
    ]
    assert all(report["sequence_checks"].values())
    assert all(report["state_checks"].values())
    assert all(report["artifact_boundary_checks"].values())


def test_operator_handoff_preserves_report_only_artifact_boundary_and_markdown():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_notebook_history_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True
    assert "Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Run Kaggle now: `False`" in markdown
