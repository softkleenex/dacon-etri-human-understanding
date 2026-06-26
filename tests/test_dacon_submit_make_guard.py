import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def target_header(name: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(name)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t.*\n)*)", re.MULTILINE)


def target_body(makefile_text: str, name: str) -> str:
    match = target_header(name).search(makefile_text)
    assert match, f"missing target {name}"
    return match.group(0)


def test_dacon_submit_target_is_guarded_by_explicit_allow_flag():
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "ALLOW_DACON_SUBMIT ?= false" in text
    assert "DACON_APPROVED_SUB_PATH ?=" in text
    assert "DACON_APPROVED_SUB_MEMO ?=" in text
    assert "DACON_SUBMIT_POLICY_REPORT ?= reports/post_id341_submission_spend_policy_20260619.json" in text
    assert "require-dacon-submit-approval:" in text
    guard = target_body(text, "require-dacon-submit-approval")
    assert "ALLOW_DACON_SUBMIT" in guard
    assert "DACON_APPROVED_SUB_PATH" in guard
    assert "DACON_APPROVED_SUB_MEMO" in guard
    assert "audit_dacon_submit_approval_contract.py" in guard
    assert "--policy-report" in guard
    assert "must exactly equal SUB_PATH" in guard
    assert "must exactly equal SUB_MEMO" in guard
    assert "explicit approval of the exact SUB_PATH and SUB_MEMO" in text
    assert "require-dacon-submit-approval" in target_body(text, "dacon-submit")


def test_dacon_submit_fails_closed_before_python_api_call_without_allow_flag():
    proc = subprocess.run(
        [
            "make",
            "dacon-submit",
            "SUB_PATH=submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv",
            "SUB_MEMO=test guard should not submit",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "DACON submission blocked" in combined
    assert "Submitting " not in combined
    assert "dacon_submit_api.post_submission_file" not in combined


def test_dacon_submit_fails_closed_when_allow_flag_has_no_exact_path_approval():
    proc = subprocess.run(
        [
            "make",
            "dacon-submit",
            "ALLOW_DACON_SUBMIT=true",
            "SUB_PATH=submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv",
            "SUB_MEMO=test guard should not submit",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "DACON exact path approval mismatch" in combined
    assert "Error 3" in combined
    assert "Submitting " not in combined
    assert "dacon_submit_api.post_submission_file" not in combined


def test_dacon_submit_fails_closed_when_memo_approval_mismatches():
    sub_path = "submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"
    proc = subprocess.run(
        [
            "make",
            "dacon-submit",
            "ALLOW_DACON_SUBMIT=true",
            f"SUB_PATH={sub_path}",
            f"DACON_APPROVED_SUB_PATH={sub_path}",
            "SUB_MEMO=test guard should not submit",
            "DACON_APPROVED_SUB_MEMO=different memo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "DACON exact memo approval mismatch" in combined
    assert "Error 4" in combined
    assert "Submitting " not in combined
    assert "dacon_submit_api.post_submission_file" not in combined


def test_manual_submission_probe_decision_has_make_target():
    text = MAKEFILE.read_text(encoding="utf-8")
    body = target_body(text, "manual-submission-probe-decision")

    assert "diagnose_post_id341_manual_submission_probe_decision.py" in body
    assert "reports/post_id341_manual_submission_probe_decision_20260618.json" in body


def test_audit_refreshes_manual_submission_probe_decision():
    text = MAKEFILE.read_text(encoding="utf-8")
    body = target_body(text, "audit")

    assert "diagnose_post_id341_manual_submission_probe_decision.py" in body
    assert "reports/post_id341_manual_submission_probe_decision_20260618.json" in body
