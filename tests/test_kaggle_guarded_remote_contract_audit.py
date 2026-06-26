import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_kaggle_guarded_remote_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("guarded_contract_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guarded_contract_audit_passes_current_repo_state():
    module = load_module()
    report = module.build_report()

    assert report["decision"]["guarded_remote_contract_pass"] is True
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert report["makefile_checks"]["allow_variable_defaults_false"] is True
    assert all(report["makefile_checks"]["guarded_targets"].values())
    assert all(report["makefile_checks"]["local_only_targets"].values())
    assert all(report["sequence_checks"].values())
    assert all(report["local_preapproval_checks"].values())
    assert all(report["local_review_checks"]["required_scripts_present"].values())
    assert report["local_review_checks"]["threshold_gate_is_last_review_step"] is True
    assert report["local_review_checks"]["local_review_has_no_remote_guard"] is True
    assert all(report["local_prelaunch_checks"]["required_scripts_present"].values())
    assert report["local_prelaunch_checks"]["q3s3_negative_chain_before_approval_artifacts"] is True
    assert report["local_prelaunch_checks"]["prelaunch_has_no_remote_guard"] is True
    assert all(report["strict_threshold_contract_checks"].values())
    assert all(report["approval_packet_handoff_contract_checks"].values())
    assert all(report["prelaunch_script_handoff_contract_checks"]["required_handoff_tokens_present"].values())
    assert report["prelaunch_script_handoff_contract_checks"]["handoff_ready_check_exposed"] is True
    assert report["prelaunch_script_handoff_contract_checks"]["handoff_contract_input_exposed"] is True
    assert report["prelaunch_script_handoff_contract_checks"]["handoff_check_part_of_prelaunch_ready"] is True
    assert "prelaunch_script" in report["inputs"]
    assert (
        report["observed_sequences"]["approval_note_local_preapproval"] == module.EXPECTED_LOCAL_PRE_APPROVAL_SEQUENCE
    )
    assert (
        report["observed_sequences"]["prelaunch_script_local_preapproval"]
        == module.EXPECTED_LOCAL_PRE_APPROVAL_SEQUENCE
    )
    assert report["observed_sequences"]["prelaunch_script"] == module.EXPECTED_SEQUENCE
    required_prelaunch = report["local_prelaunch_checks"]["required_scripts_present"]
    assert required_prelaunch["scripts/diagnose_kaggle_source_zip_selftest_post_id341.py"] is True
    assert required_prelaunch["scripts/diagnose_post_id341_kaggle_launch_approval_note.py"] is True
    assert required_prelaunch["scripts/diagnose_post_id341_kaggle_output_intake_gate.py"] is True
    assert required_prelaunch["scripts/summarize_post_id341_kaggle_remote_results.py"] is True
    assert required_prelaunch["scripts/decide_post_id341_remote_next_action.py"] is True
    assert required_prelaunch["scripts/decide_post_id341_remote_artifact_review_threshold.py"] is True
    assert required_prelaunch["scripts/diagnose_post_id341_post_pull_review_chain_contract.py"] is True
    assert required_prelaunch["scripts/audit_kaggle_remote_fail_closed_dryrun.py"] is True


def test_expected_sequence_is_guarded_for_network_and_local_for_review():
    module = load_module()
    assert len(module.EXPECTED_SEQUENCE) == 7
    for command in module.EXPECTED_SEQUENCE[: module.NETWORK_COMMAND_COUNT]:
        assert command.startswith("ALLOW_KAGGLE_REMOTE=true make ")
    assert module.EXPECTED_SEQUENCE[-1] == "make kaggle-review-upstream-rebuild-output"


def test_report_roundtrip_markdown_contains_guarded_sequence():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Guarded remote contract pass: `True`" in markdown
    assert "ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset" in markdown
    assert "make kaggle-review-upstream-rebuild-output" in markdown
    assert "scripts/decide_post_id341_remote_artifact_review_threshold.py" in markdown
    assert "Strict threshold contract checks" in markdown
    assert "Local pre-approval checks" in markdown
    assert "kaggle-approval-contract-test" in markdown
    assert "Approval/packet handoff contract checks" in markdown
    assert "Prelaunch script handoff contract checks" in markdown
    assert "runner_q3s3_negative_map_handoff_ready" in markdown
    assert "approval_stop_rules_require_q3s3_negative_map_handoff" in markdown
    assert "q3s3_negative_chain_before_approval_artifacts: `True`" in markdown


def test_prelaunch_report_is_not_authoritative_for_sequence(monkeypatch, tmp_path):
    module = load_module()
    monkeypatch.setattr(module, "PRELAUNCH_GATE", tmp_path / "missing_prelaunch_report.json")

    report = module.build_report()

    assert report["decision"]["guarded_remote_contract_pass"] is True
    assert report["sequence_checks"]["prelaunch_script_matches_expected"] is True
    assert report["sequence_checks"]["all_authoritative_sequences_match_each_other"] is True
    assert report["observed_sequences"]["prelaunch_gate_report_observed_not_authoritative"] == []


def test_local_preapproval_commands_are_required_by_approval_note(monkeypatch, tmp_path):
    module = load_module()
    approval_payload = json.loads(module.APPROVAL_NOTE.read_text(encoding="utf-8"))
    approval_payload["local_pre_approval_commands"] = []
    approval_payload["stop_rules"] = [
        rule for rule in approval_payload["stop_rules"] if "kaggle-approval-contract-test" not in rule
    ]
    broken_approval = tmp_path / "approval_without_local_preapproval.json"
    broken_approval.write_text(json.dumps(approval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "APPROVAL_NOTE", broken_approval)

    report = module.build_report()

    assert report["local_preapproval_checks"]["approval_local_preapproval_sequence_matches_expected"] is False
    assert report["local_preapproval_checks"]["approval_stop_rule_requires_local_preapproval_commands"] is False
    assert report["decision"]["guarded_remote_contract_pass"] is False
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_prelaunch_handoff_tokens_are_authoritative_not_optional(monkeypatch, tmp_path):
    module = load_module()
    replacement_script = tmp_path / "minimal_prelaunch_without_handoff.py"
    replacement_script.write_text(
        "EXACT_COMMAND_SEQUENCE = " + repr(module.EXPECTED_SEQUENCE) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PRELAUNCH_SCRIPT", replacement_script)

    report = module.build_report()

    assert report["sequence_checks"]["prelaunch_script_matches_expected"] is True
    assert report["prelaunch_script_handoff_contract_checks"]["handoff_ready_check_exposed"] is False
    assert report["prelaunch_script_handoff_contract_checks"]["handoff_check_part_of_prelaunch_ready"] is False
    assert report["decision"]["guarded_remote_contract_pass"] is False
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_approval_packet_handoff_stop_rule_is_required(monkeypatch, tmp_path):
    module = load_module()
    approval_payload = json.loads(module.APPROVAL_NOTE.read_text(encoding="utf-8"))
    approval_payload["stop_rules"] = [
        rule for rule in approval_payload["stop_rules"] if "Q3/S3 negative-map handoff" not in rule
    ]
    broken_approval = tmp_path / "approval_without_handoff_rule.json"
    broken_approval.write_text(json.dumps(approval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "APPROVAL_NOTE", broken_approval)

    report = module.build_report()

    assert (
        report["approval_packet_handoff_contract_checks"]["approval_stop_rules_require_q3s3_negative_map_handoff"]
        is False
    )
    assert (
        report["approval_packet_handoff_contract_checks"]["packet_stop_rules_require_q3s3_negative_map_handoff"] is True
    )
    assert report["decision"]["guarded_remote_contract_pass"] is False
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
