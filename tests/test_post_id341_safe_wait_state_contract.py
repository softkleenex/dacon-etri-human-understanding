import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_safe_wait_state_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("safe_wait_state_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_wait_state_is_ready_and_blocks_all_score_actions():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["safe_wait_state_contract_ready"] is True
    assert decision["safe_wait_state_ready"] is True
    assert decision["default_action"] == "do_not_submit"
    assert decision["score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_exact_kaggle_approval_phrase"] is True
    assert decision["kaggle_approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["separate_heavy_compute_phrase_required"] is True
    assert (
        decision["heavy_compute_approval_phrase"] == "Self-supervised subject-state heavy compute no-submit 실행 승인"
    )
    assert decision["manual_probe_id"] is None
    assert decision["manual_probe_if_user_spends_submission"] is None
    assert decision["manual_probe_sha256_12"] == "1c0237b4fd37"
    assert decision["manual_probe_is_public_058_evidence"] is False
    assert decision["public_058_reviewable"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["modify_submission_log"] is False
    assert decision["create_id342_or_new_submission_id"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_safe_wait_state_proves_log_and_manual_probe_artifact_integrity():
    module = load_module()
    report = module.build_report()
    log = report["submission_log_snapshot"]
    csv_info = report["artifact_inventory"]["manual_probe_submission_csv"]
    notebook_info = report["artifact_inventory"]["manual_probe_notebook"]

    assert log["rows"] == 342
    assert log["max_id"] == 342
    assert log["ids_are_unique"] is True
    assert log["manual_probe_row_count"] == 1
    assert log["manual_probe_row_ok"] is True
    assert log["manual_probe_fresh_for_submission"] is False
    assert log["manual_probe_already_submitted"] is True
    assert log["recorded_id342_row_count"] == 1
    assert log["recorded_id342_row_ok"] is True
    assert log["ready_for_existing_id342_state"] is True
    assert log["next_numeric_id_if_later_authorized"] == 343
    assert csv_info["exists"] is True
    assert csv_info["shape"] == [250, 10]
    assert csv_info["columns_ok"] is True
    assert csv_info["row_count_ok"] is True
    assert csv_info["has_blank_cells"] is False
    assert csv_info["duplicate_key_rows"] == 0
    assert notebook_info["exists"] is True
    assert report["artifact_inventory"]["id342_controlled_artifacts_accounted_for"] is True
    assert report["artifact_inventory"]["recorded_id342_core_artifacts_exist"] is True
    assert all(report["source_decision_checks"].values())
    assert all(report["artifact_checks"].values())
    assert all(report["action_guard_checks"].values())


def test_safe_wait_state_markdown_is_operator_unambiguous():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Safe-wait state ready: `True`" in markdown
    assert "Default action: `do_not_submit`" in markdown
    assert "Kaggle approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`" in markdown
    assert "Manual probe if user spends submission: `None`" in markdown
    assert "Manual probe hash: `1c0237b4fd37`" in markdown
    assert "Public 0.58 reviewable: `False`" in markdown
    assert "Create new submission ID: `False`" in markdown
    assert "Submit now: `False`" in markdown
