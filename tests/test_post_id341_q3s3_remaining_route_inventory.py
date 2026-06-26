import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_remaining_route_inventory.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_remaining_route_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q3s3_remaining_route_inventory_closes_local_no_approval_score_routes():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]
    checks = report["checks"]

    assert report["diagnostic"] == "post_id341_q3s3_remaining_route_inventory"
    assert checks["all_inputs_ready"] is True
    assert checks["all_raw_sources_readable"] is True
    assert checks["q3s3_local_lanes_closed"] is True
    assert checks["fold_nested_sibling_bridge_closed"] is True
    assert checks["gps_path_topology_model_lane_closed"] is True
    assert checks["external_public05917_direct_route_closed"] is True
    assert checks["local_no_approval_score_route_count"] == 0
    assert checks["open_score_route_count"] == 1
    assert decision["q3s3_remaining_route_inventory_ready"] is True
    assert decision["q3_s3_still_uncovered"] is True
    assert decision["fold_nested_sibling_bridge_closed"] is True
    assert decision["gps_path_topology_model_lane_closed"] is True
    assert decision["external_public05917_direct_route_closed"] is True
    assert decision["local_no_approval_score_route_exists"] is False
    assert decision["only_score_relevant_open_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["requires_explicit_kaggle_approval"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"
    assert decision["public_058_evidence"] is False
    scout_routes = [row for row in report["closed_route_rows"] if row["status"] == "closed_scout"]
    assert [row["route"] for row in scout_routes] == ["cross_modal_disagreement_scout"]
    assert scout_routes[0]["artifact_allowed"] is False
    sibling_routes = [row for row in report["closed_route_rows"] if row["route"] == "fold_nested_sibling_bridge"]
    assert len(sibling_routes) == 1
    assert sibling_routes[0]["status"] == "closed_for_local_artifact"
    assert sibling_routes[0]["artifact_allowed"] is False
    gps_routes = [row for row in report["closed_route_rows"] if row["route"] == "gps_path_topology_sequence_v1"]
    assert len(gps_routes) == 1
    assert gps_routes[0]["status"] == "closed_after_model_smoke"
    external_routes = [
        row for row in report["closed_route_rows"] if row["route"] == "external_public05917_direct_replay"
    ]
    assert len(external_routes) == 1
    assert external_routes[0]["status"] == "closed_as_direct_route"


def test_q3s3_remaining_route_inventory_is_report_only_and_closes_manual_probe_family():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["manual_probe_if_user_spends_submission"] is None
    assert decision["manual_probe_family_status"] == "consumed_after_id341_id340_id337_public_micro_probe_slate"
    manual_routes = [
        row for row in report["open_or_blocked_route_rows"] if row["route"] == "manual_id341_public_sanity_probe"
    ]
    assert manual_routes[0]["status"] == "closed_after_id341_id340_id337_micro_probe_results"
    assert report["no_model_trained"] is True
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True


def test_q3s3_remaining_route_inventory_markdown_is_unambiguous():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Local no-approval score route exists: `False`" in markdown
    assert "Only score-relevant open route: `explicit_approved_clean_upstream_id221_to_id226`" in markdown
    assert "Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`" in markdown
    assert "fold_nested_sibling_bridge" in markdown
    assert "GPS topology model lane closed: `True`" in markdown
    assert "External Public 0.5917 direct route closed: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown


def test_q3s3_remaining_route_inventory_accepts_explicit_negative_evidence_path():
    module = load_module()
    explicit_path = Path("reports/post_id341_q3s3_negative_evidence_map_20260618.json")
    explicit_sibling_path = Path("reports/post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json")
    report = module.build_report(explicit_path, explicit_sibling_path)

    assert report["inputs"]["negative_evidence_map"]["path"] == str(explicit_path)
    assert report["inputs"]["fold_nested_sibling_bridge"]["path"] == str(explicit_sibling_path)
    assert report["inputs"]["fold_nested_sibling_bridge"]["parse_ok"] is True
    assert report["decision"]["q3s3_remaining_route_inventory_ready"] is True
    assert report["decision"]["fold_nested_sibling_bridge_closed"] is True
    assert report["decision"]["local_no_approval_score_route_exists"] is False
