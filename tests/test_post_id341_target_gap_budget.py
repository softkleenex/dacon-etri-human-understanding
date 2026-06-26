import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_target_gap_budget.py"


def load_module():
    spec = importlib.util.spec_from_file_location("target_gap_budget", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_gap_budget_blocks_artifact_from_current_local_sources():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["strict_budget_beats_id178"] is False
    assert decision["watch_budget_beats_id178"] is False
    assert decision["strict_budget_public_058_reviewable"] is False
    assert decision["watch_budget_public_058_reviewable"] is False
    assert decision["q1_s4_only_is_insufficient"] is True


def test_target_gap_budget_identifies_missing_independent_targets():
    module = load_module()
    report = module.build_report()

    assert report["strict_budget_projection"]["selected_targets"] == ["Q1", "S4"]
    assert report["strict_budget_projection"]["projected_proxy_from_id341"] > module.ID178_PUBLIC
    assert report["strict_budget_projection"]["additional_sum_delta_needed_to_beat_id178"] < 0
    assert {"Q2", "Q3", "S3"}.issubset(report["missing_strict_targets"])
    assert report["decision"]["recommended_next_focus"] == (
        "Q2/Q3/S3 independent raw or upstream source signal before more Q1/S4 tuning"
    )
