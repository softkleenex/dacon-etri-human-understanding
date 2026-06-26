import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_local_smoke_freshness_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("local_smoke_freshness_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_smoke_freshness_gate_blocks_artifacts_and_requires_matching_fingerprints():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["checks"]["smoke_report_passed"] is True
    assert report["checks"]["smoke_intake_safe"] is True
    assert report["checks"]["fingerprints_recorded"] is True
    assert report["checks"]["fingerprints_match_current_inputs"] is True
    assert decision["local_runner_smoke_fresh"] is True
