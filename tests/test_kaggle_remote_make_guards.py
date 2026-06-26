import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
PRELAUNCH_SCRIPT = ROOT / "scripts" / "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py"


def target_header(name: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(name)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t.*\n)*)", re.MULTILINE)


def target_body(makefile_text: str, name: str) -> str:
    match = target_header(name).search(makefile_text)
    assert match, f"missing target {name}"
    return match.group(0)


def load_prelaunch_module():
    spec = importlib.util.spec_from_file_location("prelaunch_gate", PRELAUNCH_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_position(body: str, script_name: str) -> int:
    position = body.rfind(script_name)
    assert position >= 0, f"missing script in kaggle-upstream-prelaunch-local: {script_name}"
    return position


def test_kaggle_network_targets_fail_closed_by_make_guard():
    text = MAKEFILE.read_text(encoding="utf-8")
    guarded_targets = [
        "kaggle-push-code",
        "kaggle-push-upstream-patch-dataset",
        "kaggle-push-only",
        "kaggle-check-base-dataset-files",
        "kaggle-check-upstream-patch-dataset-files",
        "kaggle-kernel-status-upstream-rebuild",
        "kaggle-pull-output",
        "kaggle-wait-and-pull-upstream-rebuild-output",
        "kaggle-wait-and-pull",
    ]
    assert "ALLOW_KAGGLE_REMOTE ?= false" in text
    assert "require-kaggle-remote-approval:" in text
    for target in guarded_targets:
        assert "require-kaggle-remote-approval" in target_body(text, target)


def test_prelaunch_network_sequence_uses_explicit_allow_flag():
    module = load_prelaunch_module()
    network_commands = module.EXACT_COMMAND_SEQUENCE[:-1]
    assert network_commands
    for command in network_commands:
        assert command.startswith("ALLOW_KAGGLE_REMOTE=true make "), command
    assert module.EXACT_COMMAND_SEQUENCE[-1] == "make kaggle-review-upstream-rebuild-output"


def test_upstream_prelaunch_local_rebuilds_full_q2q3s3_chain_before_score_contract():
    body = target_body(MAKEFILE.read_text(encoding="utf-8"), "kaggle-upstream-prelaunch-local")
    ordered_scripts = [
        "diagnose_post_id341_q2q3s3_source_route.py",
        "diagnose_post_id341_q2q3s3_env_social_routine_v2_design.py",
        "diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke.py",
        "diagnose_post_id341_q2q3s3_fold_internal_model_smoke.py",
        "diagnose_post_id341_q2q3s3_q2_lead_geometry_gate.py",
        "diagnose_post_id341_q2q3s3_paper_family_fold_smoke.py",
        "diagnose_post_id341_q3s3_anchor_failure_profile.py",
        "diagnose_post_id341_q3s3_calendar_sleep_prior_smoke.py",
        "diagnose_post_id341_q3s3_subject_drift_gate.py",
        "diagnose_post_id341_q3s3_event_cadence_smoke.py",
        "diagnose_post_id341_q3s3_local_lane_closure.py",
        "diagnose_post_id341_q3s3_negative_evidence_map.py",
        "diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py",
        "diagnose_post_id341_q3s3_remaining_route_inventory.py",
        "diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py",
        "diagnose_post_id341_runner_score_relevance_contract.py",
        "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
    ]

    positions = []
    for script_name in ordered_scripts:
        position = body.find(script_name)
        assert position >= 0, f"missing script in kaggle-upstream-prelaunch-local: {script_name}"
        positions.append(position)

    assert positions == sorted(positions)


def test_upstream_prelaunch_refreshes_q3s3_negative_map_before_kaggle_approval_artifacts():
    body = target_body(MAKEFILE.read_text(encoding="utf-8"), "kaggle-upstream-prelaunch-local")
    q3s3_negative = body.find("diagnose_post_id341_q3s3_negative_evidence_map.py")
    q3s3_sibling = body.find("diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py")
    q3s3_remaining = body.find("diagnose_post_id341_q3s3_remaining_route_inventory.py")
    local_fallback = body.find("diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py")
    assert q3s3_negative >= 0
    assert q3s3_sibling > q3s3_negative
    assert q3s3_remaining > q3s3_sibling
    assert local_fallback > q3s3_remaining

    approval_artifact_scripts = [
        "diagnose_post_id341_kaggle_runner_intake_contract.py",
        "diagnose_kaggle_upstream_execution_packet_post_id341.py",
        "diagnose_post_id341_local_source_closure_synthesis.py",
        "diagnose_kaggle_source_zip_selftest_post_id341.py",
        "diagnose_post_id341_kaggle_launch_approval_note.py",
        "audit_kaggle_guarded_remote_contract.py",
        "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
    ]
    for script_name in approval_artifact_scripts:
        position = body.find(script_name)
        assert position > local_fallback, f"{script_name} must run after refreshed Q3/S3 negative evidence chain"


def test_upstream_prelaunch_final_material_chain_is_not_stale():
    body = target_body(MAKEFILE.read_text(encoding="utf-8"), "kaggle-upstream-prelaunch-local")
    ordered_final_scripts = [
        "diagnose_post_id341_self_supervised_subject_state_embedding_design.py",
        "diagnose_post_id341_sleep_proxy_threshold_residual_design.py",
        "diagnose_post_id341_material_new_source_queue.py",
        "diagnose_post_id341_material_source_remote_coverage_contract.py",
        "diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
        "diagnose_post_id341_prelaunch_freshness_contract.py",
        "diagnose_post_id341_self_supervised_heavy_compute_contract.py",
        "diagnose_post_id341_safe_wait_state_contract.py",
        "diagnose_post_id341_next_material_route_triage.py",
    ]

    positions = [final_position(body, script_name) for script_name in ordered_final_scripts]

    assert positions == sorted(positions)


def test_lint_paths_include_kaggle_approval_guard_scripts():
    text = MAKEFILE.read_text(encoding="utf-8")
    lint_line = next(line for line in text.splitlines() if line.startswith("LINT_PATHS :="))
    for script_path in [
        "scripts/audit_kaggle_guarded_remote_contract.py",
        "scripts/audit_kaggle_remote_fail_closed_dryrun.py",
        "scripts/diagnose_kaggle_source_zip_selftest_post_id341.py",
        "scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py",
        "scripts/diagnose_post_id341_kaggle_launch_approval_note.py",
        "scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
        "scripts/diagnose_post_id341_kaggle_runner_intake_contract.py",
    ]:
        assert script_path in lint_line


def test_kaggle_approval_contract_test_target_is_local_focused_suite():
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "kaggle-approval-contract-test" in text.splitlines()[0]
    assert "make kaggle-approval-contract-test - Run local focused Kaggle approval/guard contract checks" in text

    lint_line = next(line for line in text.splitlines() if line.startswith("KAGGLE_APPROVAL_CONTRACT_LINT_PATHS :="))
    test_line = next(line for line in text.splitlines() if line.startswith("KAGGLE_APPROVAL_CONTRACT_TESTS :="))
    for path in [
        "tests/test_kaggle_remote_make_guards.py",
        "tests/test_kaggle_guarded_remote_contract_audit.py",
        "tests/test_kaggle_remote_fail_closed_dryrun_audit.py",
        "tests/test_post_id341_kaggle_output_intake_gate.py",
        "tests/test_post_id341_kaggle_prelaunch_gate.py",
        "tests/test_post_id341_kaggle_remote_summary.py",
        "tests/test_post_id341_kaggle_runner_intake_contract.py",
        "tests/test_post_id341_kaggle_source_package_contract.py",
        "tests/test_post_id341_post_pull_review_chain_contract.py",
        "tests/test_post_id341_remote_artifact_review_threshold.py",
        "tests/test_post_id341_remote_next_action.py",
    ]:
        assert path in lint_line
        assert path in test_line

    for path in [
        "scripts/audit_kaggle_guarded_remote_contract.py",
        "scripts/audit_kaggle_remote_fail_closed_dryrun.py",
        "scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py",
        "scripts/diagnose_post_id341_kaggle_launch_approval_note.py",
        "scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py",
    ]:
        assert path in lint_line

    body = target_body(text, "kaggle-approval-contract-test")
    assert ".venv/bin/ruff check $(KAGGLE_APPROVAL_CONTRACT_LINT_PATHS)" in body
    assert ".venv/bin/ruff format --check $(KAGGLE_APPROVAL_CONTRACT_LINT_PATHS)" in body
    assert "PYTHONPATH=. .venv/bin/pytest -q $(KAGGLE_APPROVAL_CONTRACT_TESTS)" in body
    assert "ALLOW_KAGGLE_REMOTE=true" not in body
    assert "require-kaggle-remote-approval" not in body
    assert "dacon-submit" not in body
