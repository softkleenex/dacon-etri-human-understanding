import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def makefile_text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def target_body(text: str, target: str) -> str:
    match = re.search(rf"^{re.escape(target)}(?:\s*:[^\n]*)?\n(?P<body>(?:\t.*\n)*)", text, re.MULTILINE)
    assert match, f"{target} target is missing"
    return match.group("body")


def test_kaggle_upstream_ready_local_target_is_advertised_and_phony():
    text = makefile_text()

    assert "kaggle-upstream-ready-local" in text.splitlines()[0]
    assert "make kaggle-upstream-ready-local - Run full no-network readiness chain before Kaggle approval" in text


def test_kaggle_upstream_ready_local_target_runs_only_local_gates_in_safe_order():
    body = target_body(makefile_text(), "kaggle-upstream-ready-local")
    expected_order = [
        "$(MAKE) kaggle-local-runner-smoke",
        ".venv/bin/python scripts/diagnose_post_id341_local_smoke_freshness_gate.py --output reports/post_id341_local_smoke_freshness_gate_20260618.json",
        "$(MAKE) kaggle-upstream-prelaunch-local",
        "$(MAKE) kaggle-approval-contract-test",
    ]

    positions = [body.find(command) for command in expected_order]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert "ALLOW_KAGGLE_REMOTE=true" not in body
    assert "require-kaggle-remote-approval" not in body
    assert "$(KAGGLE)" not in body
    assert "dacon-submit" not in body
