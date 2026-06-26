import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "decide_post_id341_remote_artifact_review_threshold.py"


def load_module():
    spec = importlib.util.spec_from_file_location("remote_artifact_threshold", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_common_inputs(tmp_path):
    public_gap = tmp_path / "public_gap.json"
    data = tmp_path / "data.json"
    prelaunch = tmp_path / "prelaunch.json"
    write_json(public_gap, {"decision": {"public_local_gap_diagnosis_ready": True}})
    write_json(
        data,
        {
            "decision": {
                "data_source_coverage_ready": True,
                "external_direct_concat_allowed": False,
                "local_source_stack_closed_for_submission_artifact": True,
            }
        },
    )
    write_json(
        prelaunch,
        {"decision": {"prelaunch_ready_for_explicit_approval": True, "execute_kaggle_now": False}},
    )
    return public_gap, data, prelaunch


def write_remote_inputs(
    tmp_path,
    *,
    pending: bool,
    proxy: float | None,
    next_allows_review: bool,
    public_058: bool,
    beats_id178: bool,
):
    summary = tmp_path / "summary.json"
    next_action = tmp_path / "next.json"
    write_json(
        summary,
        {
            "decision": {
                "output_pending": pending,
                "public_058_evidence_present": public_058,
                "beats_id178_public_anchor_evidence_present": beats_id178,
                "create_submission_artifact": False,
                "submit_now": False,
            },
            "counts": {"missing_reports": 0, "parse_errors": 0, "forbidden_action_rows": 0},
            "best_proxy_row": None
            if proxy is None
            else {
                "filename": "remote_best.json",
                "min_proxy": proxy,
                "selected_targets": ["Q1"],
            },
        },
    )
    write_json(
        next_action,
        {
            "decision": {
                "output_pending": pending,
                "allow_separate_artifact_contract_review": next_allows_review,
                "create_submission_artifact": False,
                "submit_now": False,
            }
        },
    )
    return summary, next_action


def build_report(tmp_path, **remote_kwargs):
    module = load_module()
    public_gap, data, prelaunch = write_common_inputs(tmp_path)
    summary, next_action = write_remote_inputs(tmp_path, **remote_kwargs)
    return module.build_report(summary, next_action, public_gap, data, prelaunch)


def test_pending_outputs_do_not_open_threshold_review(tmp_path):
    report = build_report(
        tmp_path,
        pending=True,
        proxy=None,
        next_allows_review=False,
        public_058=False,
        beats_id178=False,
    )

    assert report["decision"]["threshold_gate_ready"] is True
    assert report["decision"]["output_pending"] is True
    assert report["decision"]["allow_public_058_artifact_contract_review"] is False
    assert report["decision"]["create_submission_artifact"] is False


def test_public_058_remote_evidence_allows_separate_review_only(tmp_path):
    report = build_report(
        tmp_path,
        pending=False,
        proxy=0.579,
        next_allows_review=True,
        public_058=True,
        beats_id178=True,
    )

    assert report["decision"]["allow_public_058_artifact_contract_review"] is True
    assert report["evidence_flags"]["selected_targets_present"] is True
    assert report["evidence_flags"]["local_source_stack_closed"] is True
    assert report["decision"]["record_id178_improvement_lead_only"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_id178_improvement_without_public_058_is_lead_only(tmp_path):
    report = build_report(
        tmp_path,
        pending=False,
        proxy=0.599,
        next_allows_review=True,
        public_058=False,
        beats_id178=True,
    )

    assert report["decision"]["allow_public_058_artifact_contract_review"] is False
    assert report["decision"]["record_id178_improvement_lead_only"] is True
    assert report["evidence_flags"]["proxy_lt_id178_public"] is True
    assert report["evidence_flags"]["proxy_le_public_058"] is False


def test_claimed_public_058_signal_above_threshold_is_consistency_problem(tmp_path):
    report = build_report(
        tmp_path,
        pending=False,
        proxy=0.599,
        next_allows_review=True,
        public_058=True,
        beats_id178=True,
    )

    assert report["decision"]["allow_public_058_artifact_contract_review"] is False
    assert report["decision"]["record_id178_improvement_lead_only"] is False
    assert report["evidence_flags"]["public_058_signal_consistency_problem"] is True
    assert report["evidence_flags"]["proxy_lt_id178_public"] is True
    assert report["evidence_flags"]["proxy_le_public_058"] is False
    assert "conflicts with the numeric proxy threshold" in report["decision"]["next_action"]


def test_forbidden_summary_blocks_even_low_proxy(tmp_path):
    module = load_module()
    public_gap, data, prelaunch = write_common_inputs(tmp_path)
    summary, next_action = write_remote_inputs(
        tmp_path,
        pending=False,
        proxy=0.579,
        next_allows_review=True,
        public_058=True,
        beats_id178=True,
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["counts"]["forbidden_action_rows"] = 1
    write_json(summary, payload)

    report = module.build_report(summary, next_action, public_gap, data, prelaunch)

    assert report["decision"]["allow_public_058_artifact_contract_review"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["evidence_flags"]["forbidden_problem"] is True


def test_low_proxy_without_selected_targets_blocks_public_058_review(tmp_path):
    module = load_module()
    public_gap, data, prelaunch = write_common_inputs(tmp_path)
    summary, next_action = write_remote_inputs(
        tmp_path,
        pending=False,
        proxy=0.579,
        next_allows_review=True,
        public_058=True,
        beats_id178=True,
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["best_proxy_row"]["selected_targets"] = []
    write_json(summary, payload)

    report = module.build_report(summary, next_action, public_gap, data, prelaunch)

    assert report["evidence_flags"]["selected_targets_present"] is False
    assert report["decision"]["allow_public_058_artifact_contract_review"] is False
    assert report["decision"]["record_id178_improvement_lead_only"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert "lacks selected targets" in report["decision"]["next_action"]


def test_unclosed_local_source_stack_blocks_public_058_review(tmp_path):
    module = load_module()
    public_gap, data, prelaunch = write_common_inputs(tmp_path)
    data_payload = json.loads(data.read_text(encoding="utf-8"))
    data_payload["decision"]["local_source_stack_closed_for_submission_artifact"] = False
    write_json(data, data_payload)
    summary, next_action = write_remote_inputs(
        tmp_path,
        pending=False,
        proxy=0.579,
        next_allows_review=True,
        public_058=True,
        beats_id178=True,
    )

    report = module.build_report(summary, next_action, public_gap, data, prelaunch)

    assert report["evidence_flags"]["local_source_stack_closed"] is False
    assert report["decision"]["allow_public_058_artifact_contract_review"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert "local source stack is not closed" in report["decision"]["next_action"]
