#!/usr/bin/env python3
"""Report-only heavy-compute contract for the post-ID341 self-supervised route.

This diagnostic does not execute the self-supervised embedding idea. It checks
that the design gate is ready, then freezes a separate approval boundary for any
future heavy compute. It must not read full raw streams, fit embeddings, write
feature matrices, train downstream models, create submission artifacts, modify
the submission log, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "self_supervised_subject_state_embedding_v1"
TARGET_FOCUS = ["Q2", "Q3", "S3"]
HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"
KAGGLE_UPSTREAM_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"

DESIGN_REPORT = REPORTS_DIR / "post_id341_self_supervised_subject_state_embedding_design_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
FRONTIER_REPORT = REPORTS_DIR / "post_id341_no_approval_frontier_20260619.json"
PRELAUNCH_FRESHNESS_REPORT = REPORTS_DIR / "post_id341_prelaunch_freshness_contract_20260619.json"
SUBMISSION_LOG = SUBMISSIONS_DIR / "submission_log.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_self_supervised_heavy_compute_contract_20260619.json"
RECORDED_ID342_FILE = "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv"

FORBIDDEN_ACTIONS_NOW = {
    "read_full_raw_streams_now": False,
    "fit_embedding_now": False,
    "write_feature_matrix_now": False,
    "train_downstream_model_now": False,
    "write_oof_or_test_predictions_now": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "create_id342_or_new_submission_id": False,
    "push_kaggle_now": False,
    "run_kaggle_now": False,
    "submit_now": False,
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": str(exc),
        }
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def self_supervised_queue_item(queue_report: dict[str, Any]) -> dict[str, Any]:
    queue = queue_report.get("source_queue", [])
    if not isinstance(queue, list):
        return {}
    for item in queue:
        if isinstance(item, dict) and item.get("name") == SOURCE_FAMILY:
            return item
    return {}


def submission_log_status(path: Path = SUBMISSION_LOG) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids: list[int] = []
    id342_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            row_id = int(row.get("id", ""))
        except ValueError:
            continue
        ids.append(row_id)
        if row_id == 342:
            id342_rows.append(row)
    max_id = max(ids) if ids else None
    ids_are_unique = len(ids) == len(set(ids))
    id342_row = id342_rows[-1] if id342_rows else {}
    public_value = str(id342_row.get("public_lb_score_num", "")).strip().lower()
    id342_existing_generated_unsubmitted = bool(
        len(id342_rows) == 1
        and id342_row.get("submission_file") == RECORDED_ID342_FILE
        and str(id342_row.get("status_clean", "")).lower() == "generated_unsubmitted"
        and str(id342_row.get("is_submitted_bool", "")).lower() == "false"
        and public_value in {"", "nan"}
    )
    id342_state_ok = bool(len(id342_rows) == 0 or id342_existing_generated_unsubmitted)
    log_ready_for_future_unique_id = bool(ids_are_unique and max_id is not None and max_id >= 341 and id342_state_ok)
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "rows": len(rows),
        "max_id": max_id,
        "ids_are_unique": ids_are_unique,
        "has_id342_row": 342 in ids,
        "id342_row_count": len(id342_rows),
        "id342_submission_files": [row.get("submission_file", "") for row in id342_rows],
        "id342_existing_generated_unsubmitted": id342_existing_generated_unsubmitted,
        "id342_state_ok": id342_state_ok,
        "log_ready_for_future_unique_id": log_ready_for_future_unique_id,
        "next_numeric_id_if_later_authorized": None if max_id is None else max_id + 1,
    }


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    design = reports["design"]
    queue = reports["queue"]
    frontier = reports["frontier"]
    freshness = reports["prelaunch_freshness"]
    design_decision = decision(design)
    queue_decision = decision(queue)
    frontier_decision = decision(frontier)
    freshness_decision = decision(freshness)
    raw_manifest = design.get("raw_modality_manifest", {})
    row_contract = design.get("row_contract", {})
    queue_item = self_supervised_queue_item(queue)
    log_status = submission_log_status()

    return {
        "design_report_ready": bool(
            design.get("parse_ok") is True
            and design_decision.get("design_ready") is True
            and design_decision.get("requires_explicit_heavy_compute_approval") is True
            and design_decision.get("feature_matrix_allowed_now") is False
            and design_decision.get("train_model_now") is False
            and design_decision.get("create_submission_artifact") is False
            and design.get("no_model_trained") is True
            and design.get("no_submission_log_row_created") is True
        ),
        "raw_manifest_ready": bool(
            raw_manifest.get("raw_modality_count") == 12
            and raw_manifest.get("expected_raw_modality_count") == 12
            and raw_manifest.get("all_expected_modalities_present") is True
            and raw_manifest.get("all_streams_have_subject_and_timestamp") is True
            and raw_manifest.get("total_raw_rows", 0) >= 6_000_000
        ),
        "row_contract_ready": bool(
            row_contract.get("official_train_rows") == 450
            and row_contract.get("official_test_rows") == 250
            and row_contract.get("processed_train_rows") == 450
            and row_contract.get("processed_test_rows") == 250
            and row_contract.get("row_contract_exact") is True
        ),
        "queue_records_heavy_deferred": bool(
            queue.get("parse_ok") is True
            and queue_decision.get("material_new_source_queue_ready") is True
            and queue_item.get("type") == "heavy_design_only"
            and queue_item.get("status") == "design_gate_ready_heavy_deferred"
            and queue_item.get("next_allowed_step") == "defer_until_explicit_heavy_compute_approval"
            and queue_item.get("local_no_approval_allowed") is False
            and queue_item.get("train_model_now") is False
            and queue_item.get("create_artifact_allowed") is False
            and queue_item.get("submission_artifact_allowed") is False
        ),
        "no_approval_frontier_closed": bool(
            frontier.get("parse_ok") is True
            and frontier_decision.get("all_local_no_approval_score_routes_closed") is True
            and frontier_decision.get("no_approval_score_route_exists") is False
            and frontier_decision.get("create_submission_artifact") is False
            and frontier_decision.get("run_kaggle_now") is False
            and frontier_decision.get("submit_now") is False
        ),
        "prelaunch_freshness_ready": bool(
            freshness.get("parse_ok") is True
            and freshness_decision.get("prelaunch_freshness_contract_pass") is True
            and freshness_decision.get("requires_explicit_user_approval") is True
            and freshness_decision.get("run_kaggle_now") is False
            and freshness_decision.get("submit_now") is False
        ),
        "submission_log_ready_for_existing_id342_state": log_status.get("log_ready_for_future_unique_id") is True,
        "submission_log_status": log_status,
        "raw_manifest_summary": {
            "raw_modality_count": raw_manifest.get("raw_modality_count"),
            "expected_raw_modality_count": raw_manifest.get("expected_raw_modality_count"),
            "total_raw_rows": raw_manifest.get("total_raw_rows"),
            "total_raw_size_mb": raw_manifest.get("total_raw_size_mb"),
            "all_streams_have_subject_and_timestamp": raw_manifest.get("all_streams_have_subject_and_timestamp"),
        },
        "row_contract_summary": {
            "official_train_rows": row_contract.get("official_train_rows"),
            "official_test_rows": row_contract.get("official_test_rows"),
            "processed_train_rows": row_contract.get("processed_train_rows"),
            "processed_test_rows": row_contract.get("processed_test_rows"),
            "row_contract_exact": row_contract.get("row_contract_exact"),
        },
    }


def bounded_heavy_plan() -> list[dict[str, Any]]:
    return [
        {
            "stage": "approval_boundary",
            "allowed_now": False,
            "after_exact_heavy_approval": "contract_refresh_only",
            "guard": "The Kaggle upstream approval phrase does not authorize this route.",
        },
        {
            "stage": "label_free_tokenization",
            "allowed_now": False,
            "after_exact_heavy_approval": "fixed vocabularies and timestamp bins only",
            "guard": "No target-guided token selection, no public-score feedback, no OOF residual tuning.",
        },
        {
            "stage": "self_supervised_pretraining",
            "allowed_now": False,
            "after_exact_heavy_approval": "masked reconstruction or contrastive objective only",
            "guard": "No target labels, no submission CSV, no leaderboard-informed early stopping.",
        },
        {
            "stage": "temporary_embedding_artifacts",
            "allowed_now": False,
            "after_exact_heavy_approval": "ignored build/artifact/report areas only",
            "guard": "No submission_log row and no promoted feature matrix without a later artifact contract.",
        },
        {
            "stage": "fold_internal_downstream_smoke",
            "allowed_now": False,
            "after_exact_heavy_approval": "subject-hole fold-internal smoke report only",
            "guard": "Outer validation rows cannot influence embedding choice, calibration, or blending.",
        },
        {
            "stage": "submission_artifact_review",
            "allowed_now": False,
            "after_exact_heavy_approval": "still blocked",
            "guard": "Requires a separate future artifact contract and exact CSV approval.",
        },
    ]


def build_report() -> dict[str, Any]:
    reports = {
        "design": read_json_report(DESIGN_REPORT),
        "queue": read_json_report(QUEUE_REPORT),
        "frontier": read_json_report(FRONTIER_REPORT),
        "prelaunch_freshness": read_json_report(PRELAUNCH_FRESHNESS_REPORT),
    }
    prerequisites = prerequisite_status(reports)
    prerequisite_flags = [
        prerequisites["design_report_ready"],
        prerequisites["raw_manifest_ready"],
        prerequisites["row_contract_ready"],
        prerequisites["queue_records_heavy_deferred"],
        prerequisites["no_approval_frontier_closed"],
        prerequisites["prelaunch_freshness_ready"],
        prerequisites["submission_log_ready_for_existing_id342_state"],
    ]
    contract_ready = all(prerequisite_flags)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC%z").replace("+0900", "+09:00"),
        "diagnostic": "post_id341_self_supervised_heavy_compute_contract",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Define the separate no-submit approval boundary for a future label-free self-supervised "
            "subject-state embedding run, without executing heavy compute or opening a score route."
        ),
        "approval_phrases": {
            "self_supervised_heavy_compute_no_submit": HEAVY_APPROVAL_PHRASE,
            "kaggle_upstream_rebuild_no_submit": KAGGLE_UPSTREAM_APPROVAL_PHRASE,
            "phrases_are_separate": HEAVY_APPROVAL_PHRASE != KAGGLE_UPSTREAM_APPROVAL_PHRASE,
        },
        "prerequisites": prerequisites,
        "bounded_heavy_plan": bounded_heavy_plan(),
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "allowed_output_locations_after_explicit_heavy_approval": [
            "build/self_supervised_subject_state_embedding_v1/",
            "artifacts/self_supervised_subject_state_embedding_v1/",
            "reports/post_id341_self_supervised_*",
        ],
        "decision": {
            "heavy_compute_contract_ready": contract_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "heavy_compute_allowed_now": False,
            "requires_explicit_heavy_compute_approval": True,
            "approval_phrase": HEAVY_APPROVAL_PHRASE,
            "current_kaggle_upstream_approval_phrase_only_applies_to_upstream_rebuild": True,
            "kaggle_upstream_approval_phrase": KAGGLE_UPSTREAM_APPROVAL_PHRASE,
            "local_no_approval_allowed": False,
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not execute this route now. If the exact separate heavy-compute approval phrase is given later, "
                "refresh this contract first, then run only label-free no-submit stages; any submission artifact "
                "still requires a later artifact contract and exact CSV approval."
                if contract_ready
                else "Repair prerequisite reports before considering any heavy-compute approval."
            ),
        },
        "no_full_raw_stream_read": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
        "context_reports": {
            name: {"path": report.get("path"), "exists": report.get("exists"), "parse_ok": report.get("parse_ok")}
            for name, report in reports.items()
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision_payload = report["decision"]
    prerequisites = report["prerequisites"]
    lines = [
        "# Post-ID341 self-supervised heavy-compute contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only and no-network. It creates no embedding, feature matrix, model, prediction, "
            "submission CSV, submission-log row, Kaggle run, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Heavy-compute contract ready: `{decision_payload['heavy_compute_contract_ready']}`",
        f"- Heavy compute allowed now: `{decision_payload['heavy_compute_allowed_now']}`",
        f"- Requires explicit heavy-compute approval: `{decision_payload['requires_explicit_heavy_compute_approval']}`",
        f"- Approval phrase: `{decision_payload['approval_phrase']}`",
        (
            "- Kaggle upstream phrase only applies to upstream rebuild: "
            f"`{decision_payload['current_kaggle_upstream_approval_phrase_only_applies_to_upstream_rebuild']}`"
        ),
        f"- Public `0.58` evidence: `{decision_payload['public_058_evidence']}`",
        f"- Score route opened: `{decision_payload['score_route_opened']}`",
        f"- Feature matrix allowed now: `{decision_payload['feature_matrix_allowed_now']}`",
        f"- Train model now: `{decision_payload['train_model_now']}`",
        f"- Create submission artifact: `{decision_payload['create_submission_artifact']}`",
        f"- Modify submission log: `{decision_payload['modify_submission_log']}`",
        f"- Create new submission id: `{decision_payload['create_id342_or_new_submission_id']}`",
        f"- Push Kaggle now: `{decision_payload['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- Submit now: `{decision_payload['submit_now']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Prerequisites",
        "",
    ]
    for key in [
        "design_report_ready",
        "raw_manifest_ready",
        "row_contract_ready",
        "queue_records_heavy_deferred",
        "no_approval_frontier_closed",
        "prelaunch_freshness_ready",
        "submission_log_ready_for_existing_id342_state",
    ]:
        lines.append(f"- {key}: `{prerequisites[key]}`")
    lines.extend(
        [
            "",
            "## Raw and row summary",
            "",
            "```json",
            json.dumps(
                {
                    "raw_manifest_summary": prerequisites["raw_manifest_summary"],
                    "row_contract_summary": prerequisites["row_contract_summary"],
                    "submission_log_status": prerequisites["submission_log_status"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## Bounded heavy plan",
            "",
            "| stage | allowed now | after exact heavy approval | guard |",
            "|---|---:|---|---|",
        ]
    )
    for stage in report["bounded_heavy_plan"]:
        lines.append(
            f"| `{stage['stage']}` | `{stage['allowed_now']}` | "
            f"{stage['after_exact_heavy_approval']} | {stage['guard']} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden actions now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
