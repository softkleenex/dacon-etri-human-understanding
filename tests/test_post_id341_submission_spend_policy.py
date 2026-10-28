import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_submission_spend_policy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("submission_spend_policy", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skip(
    reason="ID341 시점 스냅샷 계약. 제출 로그가 342행에서 350행으로 늘어난 뒤에는 성립하지 않는다 (대회 종료 2026-06-26)."
)
def test_submission_spend_policy_blocks_score_submission_but_preserves_id341_sanity_probe():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]
    flags = report["evidence_flags"]
    probe = report["sanity_probe_candidate"]

    assert decision["submission_spend_policy_ready"] is True
    assert decision["score_improvement_submission_recommended"] is False
    assert decision["public_sanity_probe_recommended_by_default"] is False
    assert decision["public_sanity_probe_if_user_insists_id"] is None
    assert decision["public_sanity_probe_if_user_insists_file"] is None
    assert decision["requires_exact_dacon_csv_approval"] is True
    assert decision["allowed_dacon_submit_now"] is False
    assert decision["allowed_kaggle_network_now"] is False
    assert decision["allowed_new_submission_artifact_now"] is False
    assert decision["public_058_plausible_candidate_exists"] is False

    assert flags["all_required_inputs_parse"] is True
    assert flags["no_public_058_candidate"] is True
    assert flags["no_submit_consensus"] is True
    assert flags["id341_is_manual_probe"] is True
    assert flags["id341_artifact_ready"] is False
    assert flags["submission_log_ready_for_existing_id342_state"] is True
    assert flags["kaggle_route_requires_exact_approval"] is True
    assert report["submission_log_invariants"]["max_id"] == 342
    assert report["submission_log_invariants"]["ids_are_unique"] is True
    assert report["submission_log_invariants"]["recorded_id342_row_ok"] is True
    assert report["submission_log_invariants"]["next_numeric_id_if_later_authorized"] == 343

    assert probe["id"] == 341
    assert probe["artifact"]["artifact_ready"] is False
    assert probe["artifact"]["csv_shape"] == [250, 10]
    assert probe["artifact"]["csv_na_total"] == 0


def test_submission_spend_policy_markdown_says_no_default_submission():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Score-improvement submission recommended: `False`" in markdown
    assert "Public sanity probe recommended by default: `False`" in markdown
    assert "If the user insists on one sanity probe: ID `None`" in markdown
    assert "Allowed DACON submit now: `False`" in markdown
    assert "Allowed Kaggle network now: `False`" in markdown
    assert "Public 0.58 plausible candidate exists: `False`" in markdown
