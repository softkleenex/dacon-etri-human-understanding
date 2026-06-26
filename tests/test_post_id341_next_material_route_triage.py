import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_next_material_route_triage.py"


def load_module():
    spec = importlib.util.spec_from_file_location("next_material_route_triage", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_material_route_triage_keeps_only_approved_upstream_as_score_route():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["next_material_route_triage_ready"] is True
    assert decision["local_no_approval_score_route_exists"] is False
    assert decision["no_new_local_lane_opened"] is True
    assert decision["recommended_primary_next_action"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_explicit_kaggle_approval"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["secondary_heavy_route_requires_separate_approval"] is True
    assert decision["heavy_approval_phrase"] == "Self-supervised subject-state heavy compute no-submit 실행 승인"
    assert decision["open_score_route_count"] == 1

    primary = report["route_rows"][0]
    assert primary["route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert primary["score_relevant"] is True
    assert primary["local_no_approval_allowed"] is False
    assert primary["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"

    manual = next(row for row in report["route_rows"] if row["route"] == "manual_id341_public_sanity_probe")
    assert manual["tier"] == "closed_already_consumed"
    assert manual["status"] == "closed_after_id341_id340_id337_micro_probe_results"
    assert manual["requires_approval"] is False
    assert manual["next_allowed_step"] == "closed"


def test_next_material_route_triage_allows_only_design_backlog_not_artifacts():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]
    sleep_route = next(row for row in report["route_rows"] if row["route"] == "sleep_proxy_threshold_residual_v1")

    assert decision["sleep_proxy_threshold_residual_backlog_feasible"] is True
    assert decision["sleep_proxy_threshold_residual_open_now"] is False
    assert sleep_route["tier"] == "design_backlog_not_open"
    assert sleep_route["metadata_feasible"] is True
    assert sleep_route["local_no_approval_allowed"] is False
    assert sleep_route["score_relevant"] is False
    assert report["processed_schema"]["train"]["shape"] == [450, 242]
    assert report["processed_schema"]["test"]["shape"] == [250, 242]
    assert report["processed_schema"]["train"]["sleep_proxy_column_count"] >= 20
    assert all(report["evidence_flags"].values())

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


def test_next_material_route_triage_markdown_is_unambiguous():
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Local no-approval score route exists: `False`" in markdown
    assert "No new local lane opened: `True`" in markdown
    assert "Recommended primary next action: `explicit_approved_clean_upstream_id221_to_id226`" in markdown
    assert "Sleep-proxy backlog open now: `False`" in markdown
    assert "Create feature matrix: `False`" in markdown
    assert "Submit now: `False`" in markdown
