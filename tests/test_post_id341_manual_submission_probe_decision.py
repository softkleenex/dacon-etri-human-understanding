import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_manual_submission_probe_decision.py"


def load_module():
    spec = importlib.util.spec_from_file_location("manual_submission_probe_decision", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_submission_probe_decision_is_no_submit_and_prefers_id341_safety_probe():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["manual_submission_probe_decision_ready"] is True
    assert decision["public_058_plausible_candidate_exists"] is False
    assert decision["default_action"] == "do_not_submit"
    assert decision["submit_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["lowest_risk_probe_id"] == 341
    assert decision["lowest_risk_probe_artifact_contract_ready"] is False
    assert decision["lowest_proxy_focus_id"] == 340
    assert decision["lowest_local_cv_focus_id"] == 339
    assert decision["requires_explicit_user_approval_before_dacon_submit"] is True
    assert report["focus_candidate_ids"] == [337, 338, 339, 340, 341]

    id341 = next(row for row in report["focus_candidates"] if row["id"] == 341)
    assert id341["artifact_contract_ready"] is False
    assert id341["is_submitted_bool"] is True
    assert id341["public_lb"] is None
    assert id341["csv_status"]["columns_ok"] is True
    assert id341["csv_status"]["row_count_ok"] is True
    assert id341["notebook_status"]["exists"] is True


def test_manual_submission_probe_markdown_separates_probe_from_public_058_claim():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Public 0.58 plausible candidate exists: `False`" in markdown
    assert "Default action: `do_not_submit`" in markdown
    assert "Lowest-risk probe ID: `341`" in markdown
    assert "Lowest proxy focus ID: `340`" in markdown
    assert "Lowest-local-CV focus ID: `339`" in markdown
    assert "Proxy-ranked focus candidates" in markdown
    assert "Requires explicit DACON approval: `True`" in markdown
