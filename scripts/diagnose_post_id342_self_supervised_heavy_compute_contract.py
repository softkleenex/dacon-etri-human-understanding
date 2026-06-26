#!/usr/bin/env python3
"""Post-ID342 self-supervised heavy-compute contract.

This report-only diagnostic refreshes the self-supervised subject-state heavy
route after ID342 became the confirmed public anchor. It accepts the current
submission-log state where ID342 is scored/submitted, while still blocking all
heavy execution until the exact heavy-compute no-submit approval phrase is
provided.

It creates no embedding, feature matrix, model, prediction, submission CSV,
submission-log row, Kaggle push/run, or DACON submission.
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
REMAINING_ROUTE_SYNTHESIS = REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json"
RESEARCH_PACKET = REPORTS_DIR / "post_id342_self_supervised_route_research_packet_20260624.json"
GOAL_STATE_AUDIT = REPORTS_DIR / "post_id342_goal_state_audit_20260624.json"
DOCS_AUDIT = REPORTS_DIR / "docs_records_audit.json"
SUBMISSION_LOG = SUBMISSIONS_DIR / "submission_log.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_self_supervised_heavy_compute_contract_20260624.json"
RECORDED_ID342_FILE = "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv"

FORBIDDEN_ACTIONS_NOW = {
    "read_full_raw_streams_now": False,
    "fit_embedding_now": False,
    "write_feature_matrix_now": False,
    "train_downstream_model_now": False,
    "write_oof_or_test_predictions_now": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "create_new_submission_id": False,
    "push_kaggle_now": False,
    "run_kaggle_now": False,
    "submit_now": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def numeric(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def submission_log_status(path: Path = SUBMISSION_LOG) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ids: list[int] = []
    id342_rows: list[dict[str, str]] = []
    generated_unsubmitted_after_anchor: list[int] = []
    for row in rows:
        try:
            row_id = int(row.get("id", ""))
        except ValueError:
            continue
        ids.append(row_id)
        if row_id == 342:
            id342_rows.append(row)
        if row_id > 342 and str(row.get("is_submitted_bool", "")).lower() == "false":
            generated_unsubmitted_after_anchor.append(row_id)

    max_id = max(ids) if ids else None
    ids_are_unique = len(ids) == len(set(ids))
    id342_row = id342_rows[-1] if id342_rows else {}
    id342_public = numeric(id342_row.get("public_lb_score_num"))
    id342_scored_anchor_ok = bool(
        len(id342_rows) == 1
        and id342_row.get("submission_file") == RECORDED_ID342_FILE
        and str(id342_row.get("is_submitted_bool", "")).lower() == "true"
        and id342_public is not None
        and abs(id342_public - 0.5960832519) < 1e-12
    )
    log_ready_for_future_unique_id = bool(
        ids_are_unique and max_id is not None and max_id >= 344 and id342_scored_anchor_ok
    )
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
        "id342_public_score": id342_public,
        "id342_scored_anchor_ok": id342_scored_anchor_ok,
        "generated_unsubmitted_after_anchor": generated_unsubmitted_after_anchor,
        "log_ready_for_future_unique_id": log_ready_for_future_unique_id,
        "next_numeric_id_if_later_authorized": None if max_id is None else max_id + 1,
    }


def find_route(report: dict[str, Any], route_name: str) -> dict[str, Any]:
    rows = report.get("route_rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("route") == route_name:
            return row
    return {}


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    design = reports["design"]
    synthesis = reports["remaining_route_synthesis"]
    research = reports["research_packet"]
    goal = reports["goal_state_audit"]
    docs = reports["docs_audit"]
    design_decision = decision(design)
    synthesis_decision = decision(synthesis)
    research_decision = decision(research)
    goal_decision = decision(goal)
    raw_manifest = design.get("raw_modality_manifest", {})
    row_contract = design.get("row_contract", {})
    heavy_route = find_route(synthesis, "self_supervised_subject_state_embedding")
    log_status = submission_log_status()
    docs_findings = docs.get("findings", [])

    return {
        "design_report_ready": bool(
            design.get("parse_ok") is True
            and design_decision.get("design_ready") is True
            and design_decision.get("requires_explicit_heavy_compute_approval") is True
            and design_decision.get("feature_matrix_allowed_now") is False
            and design_decision.get("train_model_now") is False
            and design_decision.get("create_submission_artifact") is False
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
        "post_id342_synthesis_ready": bool(
            synthesis.get("parse_ok") is True
            and synthesis_decision.get("closed_route_count") == 11
            and synthesis_decision.get("route_count") == 12
            and synthesis_decision.get("missing_or_review_route_count") == 0
            and synthesis_decision.get("run_kaggle_now") is False
            and heavy_route.get("status") == "deferred_requires_exact_heavy_compute_approval"
        ),
        "research_packet_ready": bool(
            research.get("parse_ok") is True
            and research_decision.get("research_packet_ready") is True
            and research_decision.get("heavy_route_is_only_non_closed_route") is True
            and research_decision.get("heavy_compute_allowed_now") is False
        ),
        "goal_audit_ready": bool(
            goal.get("parse_ok") is True
            and goal_decision.get("goal_achieved") is False
            and goal_decision.get("public_055_reached") is False
            and goal_decision.get("run_kaggle_now") is False
            and goal_decision.get("dacon_submit_now") is False
        ),
        "docs_audit_clean": bool(
            docs.get("parse_ok") is True and isinstance(docs_findings, list) and not docs_findings
        ),
        "submission_log_ready_for_post_id342_anchor": log_status.get("log_ready_for_future_unique_id") is True,
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
            "after_exact_heavy_approval": "refresh_this_contract_with_approval_phrase",
            "guard": "The Kaggle upstream approval phrase does not authorize this route.",
        },
        {
            "stage": "dedicated_runner_spec",
            "allowed_now": False,
            "after_exact_heavy_approval": "write_or_refresh no-submit runner spec only",
            "guard": "The existing upstream rebuild runner is not an embedding-training runner.",
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


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        "design": read_json_report(DESIGN_REPORT),
        "remaining_route_synthesis": read_json_report(REMAINING_ROUTE_SYNTHESIS),
        "research_packet": read_json_report(RESEARCH_PACKET),
        "goal_state_audit": read_json_report(GOAL_STATE_AUDIT),
        "docs_audit": read_json_report(DOCS_AUDIT),
    }
    prerequisites = prerequisite_status(reports)
    approval_phrase_received = args.approval_phrase == HEAVY_APPROVAL_PHRASE
    all_prerequisites_ready = all(
        prerequisites[key] is True
        for key in [
            "design_report_ready",
            "raw_manifest_ready",
            "row_contract_ready",
            "post_id342_synthesis_ready",
            "research_packet_ready",
            "goal_audit_ready",
            "docs_audit_clean",
            "submission_log_ready_for_post_id342_anchor",
        ]
    )
    decision_payload = {
        "post_id342_heavy_compute_contract_ready": all_prerequisites_ready,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "approval_phrase_received": approval_phrase_received,
        "heavy_compute_allowed_now": False,
        "requires_explicit_heavy_compute_approval": True,
        "approval_phrase": HEAVY_APPROVAL_PHRASE,
        "current_kaggle_upstream_approval_phrase_only_applies_to_upstream_rebuild": True,
        "kaggle_upstream_approval_phrase": KAGGLE_UPSTREAM_APPROVAL_PHRASE,
        "dedicated_heavy_runner_required": True,
        "existing_upstream_runner_is_sufficient_for_heavy_execution": False,
        "local_no_approval_allowed": False,
        "score_route_opened": False,
        "feature_matrix_allowed_now": False,
        "train_model_now": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "create_new_submission_id": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "next_action": (
            "Contract is ready for an approval request, but execution remains blocked. "
            "After the exact heavy phrase, refresh this contract with the phrase and only then prepare a dedicated no-submit runner."
            if all_prerequisites_ready
            else "Repair prerequisite reports before considering any heavy-compute approval."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_self_supervised_heavy_compute_contract",
        "purpose": "Refresh the heavy-compute approval boundary after ID342 became the scored public anchor.",
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "context_reports": {
            key: {"path": report.get("path"), "exists": report.get("exists"), "parse_ok": report.get("parse_ok")}
            for key, report in reports.items()
        },
        "prerequisites": prerequisites,
        "bounded_heavy_plan": bounded_heavy_plan(),
        "allowed_output_locations_after_explicit_heavy_approval": [
            "build/self_supervised_subject_state_embedding_v1/",
            "artifacts/self_supervised_subject_state_embedding_v1/",
            "reports/post_id342_self_supervised_*",
        ],
        "approval_phrases": {
            "self_supervised_heavy_compute_no_submit": HEAVY_APPROVAL_PHRASE,
            "kaggle_upstream_rebuild_no_submit_not_sufficient": KAGGLE_UPSTREAM_APPROVAL_PHRASE,
        },
        "decision": decision_payload,
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_full_raw_stream_read": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_payload = report["decision"]
    prereq = report["prerequisites"]
    lines = [
        "# Post-ID342 self-supervised heavy-compute contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only and no-network. It creates no embedding, feature matrix, model, prediction, submission CSV, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Post-ID342 heavy-compute contract ready: `{decision_payload['post_id342_heavy_compute_contract_ready']}`",
        f"- Heavy compute allowed now: `{decision_payload['heavy_compute_allowed_now']}`",
        f"- Approval phrase received: `{decision_payload['approval_phrase_received']}`",
        f"- Required phrase: `{decision_payload['approval_phrase']}`",
        f"- Dedicated heavy runner required: `{decision_payload['dedicated_heavy_runner_required']}`",
        f"- Existing upstream runner sufficient: `{decision_payload['existing_upstream_runner_is_sufficient_for_heavy_execution']}`",
        f"- Push Kaggle now: `{decision_payload['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- Submit now: `{decision_payload['submit_now']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Prerequisites",
        "",
    ]
    for key, value in prereq.items():
        if isinstance(value, bool):
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Submission Log State",
            "",
            "```json",
            json.dumps(prereq["submission_log_status"], indent=2, sort_keys=True),
            "```",
            "",
            "## Bounded Heavy Plan",
            "",
            "| stage | allowed now | after exact heavy approval | guard |",
            "|---|---:|---|---|",
        ]
    )
    for row in report["bounded_heavy_plan"]:
        lines.append(
            f"| `{row['stage']}` | `{row['allowed_now']}` | {row['after_exact_heavy_approval']} | {row['guard']} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden Actions Now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--approval-phrase", default="", help="Exact heavy-compute approval phrase, if explicitly given."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Contract ready: {report['decision']['post_id342_heavy_compute_contract_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
