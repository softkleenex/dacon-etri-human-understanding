import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "decide_post_id341_remote_next_action.py"


def load_module():
    spec = importlib.util.spec_from_file_location("remote_next_action", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_inputs(tmp_path, *, pending=False, reviewable=False, forbidden_rows=0, beats_id178=True):
    intake = tmp_path / "intake.json"
    summary = tmp_path / "summary.json"
    write_json(
        intake,
        {
            "decision": {
                "output_pending": pending,
                "artifact_contract_review_allowed": reviewable,
                "remote_forbidden_actions_present": bool(forbidden_rows),
            }
        },
    )
    write_json(
        summary,
        {
            "decision": {
                "output_pending": pending,
                "remote_summary_reviewable": reviewable,
                "public_058_evidence_present": reviewable,
                "beats_id178_public_anchor_evidence_present": beats_id178 and reviewable,
                "create_submission_artifact": False,
                "submit_now": False,
            },
            "counts": {
                "missing_reports": 0,
                "parse_errors": 0,
                "forbidden_action_rows": forbidden_rows,
            },
            "best_proxy_row": {
                "filename": "remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json",
                "min_proxy": 0.579 if reviewable else 0.603,
            },
        },
    )
    return summary, intake


def test_pending_outputs_wait_without_artifact(tmp_path):
    module = load_module()
    summary, intake = write_inputs(tmp_path, pending=True)

    report = module.build_report(summary, intake)

    assert report["decision"]["output_pending"] is True
    assert report["decision"]["allow_separate_artifact_contract_review"] is False
    assert report["decision"]["close_remote_lane_for_artifact_creation"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_reviewable_remote_evidence_only_allows_separate_contract_review(tmp_path):
    module = load_module()
    summary, intake = write_inputs(tmp_path, reviewable=True)

    report = module.build_report(summary, intake)

    assert report["decision"]["output_pending"] is False
    assert report["decision"]["allow_separate_artifact_contract_review"] is True
    assert report["decision"]["close_remote_lane_for_artifact_creation"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_nonreviewable_finished_outputs_close_remote_lane(tmp_path):
    module = load_module()
    summary, intake = write_inputs(tmp_path, reviewable=False)

    report = module.build_report(summary, intake)

    assert report["decision"]["output_pending"] is False
    assert report["decision"]["allow_separate_artifact_contract_review"] is False
    assert report["decision"]["close_remote_lane_for_artifact_creation"] is True
    assert report["decision"]["create_submission_artifact"] is False


def test_forbidden_remote_action_blocks_review_even_if_summary_claims_reviewable(tmp_path):
    module = load_module()
    summary, intake = write_inputs(tmp_path, reviewable=True, forbidden_rows=1)

    report = module.build_report(summary, intake)

    assert report["evidence_flags"]["forbidden_problem"] is True
    assert report["decision"]["allow_separate_artifact_contract_review"] is False
    assert report["decision"]["close_remote_lane_for_artifact_creation"] is True
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
