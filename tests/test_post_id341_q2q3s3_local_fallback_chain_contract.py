import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_local_fallback_chain_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q2q3s3_local_fallback_chain_is_context_only_and_blocks_artifacts():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["local_fallback_chain_contract_pass"] is True
    assert decision["primary_score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["fallback_no_approval_route"] == "local_or_kaggle_q2q3s3_env_social_routine_v2_design"
    assert decision["missing_targets"] == ["Q2", "Q3", "S3"]
    assert decision["selected_local_context_targets"] == ["Q2"]
    assert decision["q3_s3_still_uncovered"] is True
    assert decision["local_q2_lead_context_only"] is True
    assert decision["public_058_evidence"] is False
    assert decision["local_artifact_allowed"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q2q3s3_local_fallback_chain_checks_full_report_sequence():
    module = load_module()
    report = module.build_report()
    checks = report["checks"]

    assert checks["parse_contract"] is True
    assert checks["route_contract"] is True
    assert checks["design_contract"] is True
    assert checks["feature_metadata_contract"] is True
    assert checks["fold_internal_contract"] is True
    assert checks["q2_geometry_contract"] is True
    assert checks["paper_family_contract"] is True
    assert checks["q3s3_closure_contract"] is True
    assert checks["q3s3_negative_evidence_contract"] is True
    assert checks["no_forbidden_decisions"] is True
    assert "q2_lead_geometry_gate" in report["inputs"]
    assert "paper_family_fold_smoke" in report["inputs"]
    assert "q3s3_local_lane_closure" in report["inputs"]
    assert "q3s3_negative_evidence_map" in report["inputs"]


def test_q2q3s3_local_fallback_chain_markdown_names_context_only_boundary():
    module = load_module()
    markdown = module.build_markdown(module.build_report())

    assert "context-only" in markdown
    assert "Q3/S3 still uncovered: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Public 0.58 evidence: `False`" in markdown
