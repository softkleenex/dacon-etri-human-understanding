import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_dacon_submit_approval_contract.py"
ID341 = "submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"
ID340 = "submissions/submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv"


def load_module():
    spec = importlib.util.spec_from_file_location("dacon_submit_approval_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_allows_only_current_id341_manual_sanity_probe():
    module = load_module()
    report = module.build_report(ID341, "manual sanity probe id341", module.DEFAULT_POLICY_REPORT)

    assert report["approval_contract_passed"] is True
    assert report["errors"] == []
    assert report["policy"]["mode"] == "manual_public_sanity_probe"
    assert report["policy"]["freshness"]["fresh"] is True
    assert report["policy"]["score_improvement_submission_recommended"] is False
    assert report["artifact"]["shape"] == [250, 10]
    assert report["artifact"]["na_total"] == 0
    assert report["artifact"]["log_match_count"] == 1
    assert report["artifact"]["duplicate_file_count"] == 1
    assert report["artifact"]["is_submitted_bool"] is False
    assert report["artifact"]["public_lb"] is None
    assert report["artifact"]["notebook_exists"] is True
    assert report["no_dacon_submission_made"] is True


def test_contract_rejects_other_unsubmitted_candidate_even_with_valid_artifact():
    module = load_module()
    report = module.build_report(ID340, "manual sanity probe id340 should fail", module.DEFAULT_POLICY_REPORT)

    assert report["approval_contract_passed"] is False
    assert "sub_path_not_allowed_by_current_submission_spend_policy" in report["errors"]
    assert report["policy"]["mode"] == "blocked_by_submission_spend_policy"
    assert report["artifact"]["shape"] == [250, 10]
    assert report["artifact"]["notebook_exists"] is True


def test_contract_rejects_default_submission_memo():
    module = load_module()
    report = module.build_report(ID341, "Automated submission", module.DEFAULT_POLICY_REPORT)

    assert report["approval_contract_passed"] is False
    assert "sub_memo_must_not_be_default" in report["errors"]


def test_contract_rejects_stale_policy_report_even_for_id341(tmp_path):
    module = load_module()
    stale_policy = tmp_path / "stale_policy.json"
    shutil.copy2(module.DEFAULT_POLICY_REPORT, stale_policy)
    os.utime(stale_policy, (1, 1))

    report = module.build_report(ID341, "manual sanity probe id341", stale_policy)

    assert report["approval_contract_passed"] is False
    assert "policy_report_stale_or_missing_dependencies" in report["errors"]
    assert report["policy"]["freshness"]["fresh"] is False
    assert report["policy"]["freshness"]["stale_dependencies"]


def test_dacon_submit_make_target_calls_contract_before_api_attempt():
    sub_path = ID340
    proc = subprocess.run(
        [
            "make",
            "dacon-submit",
            "ALLOW_DACON_SUBMIT=true",
            f"SUB_PATH={sub_path}",
            f"DACON_APPROVED_SUB_PATH={sub_path}",
            "SUB_MEMO=manual sanity probe id340 should fail",
            "DACON_APPROVED_SUB_MEMO=manual sanity probe id340 should fail",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "sub_path_not_allowed_by_current_submission_spend_policy" in combined
    assert "Submitting " not in combined
    assert "dacon_submit_api.post_submission_file" not in combined
