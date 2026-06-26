import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_public060_wall_root_cause.py"


def load_module():
    spec = importlib.util.spec_from_file_location("public060_wall_root_cause", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public060_wall_root_cause_closes_local_cv_and_micro_probe_paths():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["public060_wall_root_cause_ready"] is True
    assert decision["wall_explained_by_current_evidence"] is True
    assert decision["local_cv_only_candidate_path_closed"] is True
    assert decision["same_family_micro_probe_path_closed"] is True
    assert decision["q3s3_independent_signal_required"] is True
    assert decision["primary_score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_exact_kaggle_approval"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"

    for key in [
        "create_feature_matrix",
        "train_model_now",
        "create_submission_artifact",
        "modify_submission_log",
        "create_id342_or_new_submission_id",
        "submit_now",
        "push_kaggle_now",
        "run_kaggle_now",
    ]:
        assert decision[key] is False


def test_public060_wall_recent_band_and_collapse_evidence_are_explicit():
    module = load_module()
    report = module.build_report()
    band = report["recent_micro_probe_band"]
    collapse = report["collapse_evidence"]

    assert band["ready"] is True
    assert band["available_ids"] == [178, 329, 337, 340]
    assert band["best_recent_public_id"] == 337
    assert band["best_recent_public"] == 0.600620628622
    assert band["all_recent_public_near_060"] is True
    assert band["local_span"] > 0.02
    assert band["public_span"] < 0.0013
    assert band["local_span_to_public_span_ratio"] > 15

    assert collapse["id"] == 287
    assert collapse["catastrophic_local_public_mismatch"] is True
    assert collapse["public_minus_local"] > 0.29


def test_public060_wall_records_q3s3_gap_and_q2_only_lead():
    module = load_module()
    report = module.build_report()
    target_gap = report["target_gap_evidence"]
    causes = {row["cause"]: row for row in report["cause_rows"]}

    assert target_gap["q3_s3_still_uncovered"] is True
    assert target_gap["q2_only_signal_found"] is True
    assert target_gap["q2_best_proxy"] > 0.60
    assert target_gap["q2_not_public_058"] is True
    assert target_gap["local_no_approval_score_route_exists"] is False

    assert causes["validation_surface_mismatch"]["active"] is True
    assert causes["q3_s3_uncovered"]["active"] is True
    assert causes["q2_only_lead_insufficient"]["active"] is True
    assert causes["leaderboard_probe_family_consumed"]["active"] is True


def test_public060_wall_markdown_is_operator_unambiguous(tmp_path):
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Wall explained by current evidence: `True`" in markdown
    assert "Same-family micro-probe path closed: `True`" in markdown
    assert "Q3/S3 independent signal required: `True`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown

    out = tmp_path / "wall.md"
    out.write_text(markdown, encoding="utf-8")
    assert out.read_text(encoding="utf-8").startswith("# Post-ID341 Public-0.60 wall")
