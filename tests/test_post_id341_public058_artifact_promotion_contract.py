from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_public058_artifact_promotion_contract.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("public058_artifact_promotion_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_submission_log(path: Path, *, duplicate_id: bool = False) -> None:
    rows = [
        {
            "id": "341",
            "submission_file": "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv",
            "notebook_file": "notebooks/history/submission_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.ipynb",
        }
    ]
    if duplicate_id:
        rows.append(
            {
                "id": "341",
                "submission_file": "submission_id_341_duplicate_forbidden.csv",
                "notebook_file": "notebooks/history/submission_341_duplicate_forbidden.ipynb",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "submission_file", "notebook_file"])
        writer.writeheader()
        writer.writerows(rows)


def seed_inputs(tmp_path: Path, *, review_open: bool, duplicate_id: bool = False):
    threshold = tmp_path / "threshold.json"
    next_action = tmp_path / "next.json"
    summary = tmp_path / "summary.json"
    log = tmp_path / "submission_log.csv"
    write_json(
        threshold,
        {
            "decision": {
                "threshold_gate_ready": True,
                "allow_public_058_artifact_contract_review": review_open,
                "output_pending": not review_open,
            },
            "best_proxy": {
                "filename": "remote_best.json",
                "min_proxy": 0.579 if review_open else None,
                "selected_targets": ["Q2", "Q3"] if review_open else None,
            },
        },
    )
    write_json(
        next_action,
        {
            "decision": {
                "allow_separate_artifact_contract_review": review_open,
                "output_pending": not review_open,
            }
        },
    )
    write_json(
        summary,
        {
            "decision": {
                "remote_summary_reviewable": review_open,
                "public_058_evidence_present": review_open,
                "beats_id178_public_anchor_evidence_present": review_open,
            }
        },
    )
    write_submission_log(log, duplicate_id=duplicate_id)
    return threshold, next_action, summary, log


def test_current_public058_promotion_contract_blocks_artifacts_and_id342() -> None:
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["promotion_contract_ready"] is True
    assert decision["public058_artifact_contract_review_open"] is False
    assert decision["artifact_promotion_allowed_now"] is False
    assert decision["requires_separate_exact_candidate_approval"] is True
    assert decision["requires_exact_csv_notebook_base_log_lineage"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["create_base_prediction_artifacts"] is False
    assert decision["create_notebook_history_artifact"] is False
    assert decision["modify_submission_log"] is False
    assert decision["create_id342_or_new_submission_id"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["inputs"]["submission_log"]["max_id"] >= 341
    assert report["inputs"]["submission_log"]["ids_are_unique"] is True


def test_public058_review_open_still_does_not_allow_promotion(tmp_path: Path) -> None:
    module = load_module()
    threshold, next_action, summary, log = seed_inputs(tmp_path, review_open=True)

    report = module.build_report(threshold, next_action, summary, log)
    decision = report["decision"]

    assert decision["promotion_contract_ready"] is True
    assert decision["public058_artifact_contract_review_open"] is True
    assert decision["artifact_promotion_allowed_now"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["modify_submission_log"] is False
    assert decision["create_id342_or_new_submission_id"] is False
    assert decision["submit_now"] is False
    assert report["evidence_flags"]["threshold_allows_public058_review"] is True
    assert report["evidence_flags"]["submission_log_ready_for_future_unique_id"] is True
    assert "exact_user_csv_approval" in report["required_lineage_fields_before_future_promotion"]
    assert "base_oof_prediction_path" in report["required_lineage_fields_before_future_promotion"]
    assert "base_test_prediction_path" in report["required_lineage_fields_before_future_promotion"]


def test_duplicate_submission_id_blocks_contract_readiness(tmp_path: Path) -> None:
    module = load_module()
    threshold, next_action, summary, log = seed_inputs(tmp_path, review_open=True, duplicate_id=True)

    report = module.build_report(threshold, next_action, summary, log)

    assert report["inputs"]["submission_log"]["ids_are_unique"] is False
    assert report["evidence_flags"]["submission_log_ready_for_future_unique_id"] is False
    assert report["decision"]["promotion_contract_ready"] is False
    assert report["decision"]["public058_artifact_contract_review_open"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["modify_submission_log"] is False


def test_public058_promotion_contract_markdown_names_lineage_requirements(tmp_path: Path) -> None:
    module = load_module()
    threshold, next_action, summary, log = seed_inputs(tmp_path, review_open=True)
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report(threshold, next_action, summary, log))))

    assert "Public-0.58 artifact promotion contract" in markdown
    assert "Artifact promotion allowed now: `False`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Modify submission log: `False`" in markdown
    assert "Create ID342/new submission id: `False`" in markdown
    assert "`unique_submission_csv_path`" in markdown
    assert "`unique_notebook_history_path`" in markdown
    assert "`base_oof_prediction_path`" in markdown
    assert "`base_test_prediction_path`" in markdown
