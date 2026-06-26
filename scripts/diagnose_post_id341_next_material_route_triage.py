#!/usr/bin/env python3
"""Triage the next material route after the post-ID341 safe-wait state.

This report-only diagnostic does not open a new local score lane. It compares
currently available routes after the ID178/287/329/337/340/341 evidence: the explicit
Kaggle no-submit upstream rebuild, the separate heavy self-supervised route,
and any remaining lightweight design-only backlog suggested by the processed
schema. It exists to avoid drifting back into same-family local recombination.

It creates no predictions, submission CSVs, feature matrices, base prediction
artifacts, notebook/history artifacts, submission-log rows, Kaggle pushes/runs,
or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
TRAIN_PROCESSED = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PROCESSED = ROOT / "data" / "processed" / "df_test_raw.parquet"
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_next_material_route_triage_20260619.json"

KAGGLE_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"
PRIMARY_SCORE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
MANUAL_PROBE_FILE = "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"

REQUIRED_REPORTS = {
    "safe_wait": "post_id341_safe_wait_state_contract_20260619.json",
    "material_queue": "post_id341_material_new_source_queue_20260619.json",
    "material_coverage": "post_id341_material_source_remote_coverage_contract_20260619.json",
    "q3s3_remaining": "post_id341_q3s3_remaining_route_inventory_20260618.json",
    "public_local_gap": "post_id341_public_local_gap_diagnosis_20260618.json",
    "manual_probe": "post_id341_manual_submission_probe_decision_20260618.json",
    "self_supervised_heavy": "post_id341_self_supervised_heavy_compute_contract_20260619.json",
}

SLEEP_PROXY_TOKENS = [
    "sleep_duration",
    "sleep_efficiency",
    "fragmentation",
    "slp_",
    "awake",
    "deep",
    "light",
]
RAW_STREAMS = [
    "ch2025_mACStatus.parquet",
    "ch2025_mActivity.parquet",
    "ch2025_mAmbience.parquet",
    "ch2025_mBle.parquet",
    "ch2025_mGps.parquet",
    "ch2025_mLight.parquet",
    "ch2025_mScreenStatus.parquet",
    "ch2025_mUsageStats.parquet",
    "ch2025_mWifi.parquet",
    "ch2025_wHr.parquet",
    "ch2025_wLight.parquet",
    "ch2025_wPedo.parquet",
]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_report(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
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
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "payload": payload,
    }


def decision(reports: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    return reports[key].get("decision", {})


def schema_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "shape": [0, 0], "columns": []}
    df = pd.read_parquet(path)
    columns = list(df.columns)
    sleep_proxy_columns = [col for col in columns if any(token in col.lower() for token in SLEEP_PROXY_TOKENS)]
    target_columns = [col for col in ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"] if col in columns]
    return {
        "path": rel(path),
        "exists": True,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": columns,
        "target_columns": target_columns,
        "sleep_proxy_columns": sleep_proxy_columns,
        "sleep_proxy_column_count": len(sleep_proxy_columns),
    }


def raw_stream_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in RAW_STREAMS:
        path = RAW_DIR / filename
        if not path.exists():
            rows.append({"filename": filename, "exists": False, "rows": 0, "columns": []})
            continue
        df = pd.read_parquet(path)
        rows.append(
            {
                "filename": filename,
                "exists": True,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "has_subject_timestamp": {"subject_id", "timestamp"}.issubset(df.columns),
            }
        )
    return rows


def queue_item(queue: dict[str, Any], name: str) -> dict[str, Any]:
    payload = queue.get("payload", {})
    items = payload.get("source_queue", []) if isinstance(payload.get("source_queue"), list) else []
    return next((item for item in items if item.get("name") == name), {})


def route_rows(
    reports: dict[str, dict[str, Any]], train_schema: dict[str, Any], test_schema: dict[str, Any]
) -> list[dict[str, Any]]:
    material_queue = reports["material_queue"]
    upstream_item = queue_item(material_queue, "approved_clean_upstream_id221_to_id226")
    heavy_decision = decision(reports, "self_supervised_heavy")
    public_gap_decision = decision(reports, "public_local_gap")
    manual_decision = decision(reports, "manual_probe")
    q3s3_decision = decision(reports, "q3s3_remaining")

    sleep_proxy_ready = bool(
        train_schema.get("shape") == [450, 242]
        and test_schema.get("shape") == [250, 242]
        and train_schema.get("sleep_proxy_column_count", 0) >= 20
        and test_schema.get("sleep_proxy_column_count", 0) >= 20
    )
    return [
        {
            "rank": 0,
            "route": PRIMARY_SCORE_ROUTE,
            "tier": "only_score_relevant_open_route",
            "status": upstream_item.get("status") or "open_requires_exact_approval",
            "target_focus": ["Q2", "Q3", "S3"],
            "why": "It rebuilds the historical clean ID221→ID226 path under Kaggle/Linux no-submit controls instead of local OOF recombination.",
            "requires_approval": True,
            "approval_phrase": KAGGLE_APPROVAL_PHRASE,
            "local_no_approval_allowed": False,
            "score_relevant": True,
            "next_allowed_step": "only_after_exact_approval_phrase",
        },
        {
            "rank": 1,
            "route": "self_supervised_subject_state_embedding_v1",
            "tier": "heavy_material_route_deferred",
            "status": "blocked_without_separate_heavy_approval",
            "target_focus": heavy_decision.get("target_focus", ["Q2", "Q3", "S3"]),
            "why": "It can use all raw streams label-free, but it is heavy and must not run under the Kaggle upstream approval phrase.",
            "requires_approval": True,
            "approval_phrase": HEAVY_APPROVAL_PHRASE,
            "local_no_approval_allowed": False,
            "score_relevant": False,
            "next_allowed_step": "defer_until_explicit_heavy_compute_approval",
        },
        {
            "rank": 2,
            "route": "sleep_proxy_threshold_residual_v1",
            "tier": "design_backlog_not_open",
            "status": "metadata_feasible_but_overlaps_existing_sleep_hr_light_lanes",
            "target_focus": ["S2", "S3", "S4", "Q1"],
            "why": "Processed data contains many sleep-stage/efficiency/fragmentation proxy columns, so a future design-only rule residual audit is feasible; however, related sleep/circadian lanes already failed promotion gates, so it should not open a new local score lane now.",
            "requires_approval": False,
            "approval_phrase": None,
            "local_no_approval_allowed": False,
            "score_relevant": False,
            "next_allowed_step": "future_report_only_design_only_if_upstream_or_user_direction_keeps_local_design_open",
            "metadata_feasible": sleep_proxy_ready,
            "train_sleep_proxy_columns": train_schema.get("sleep_proxy_column_count", 0),
            "test_sleep_proxy_columns": test_schema.get("sleep_proxy_column_count", 0),
        },
        {
            "rank": 3,
            "route": "public_score_inverse_inference",
            "tier": "blocked_low_information",
            "status": "do_not_use_for_candidate_creation",
            "target_focus": ["all"],
            "why": "Only aggregate public scores are available, and recent public/local correlation is weak; using public outcomes to infer labels would overfit the leaderboard rather than create robust signal.",
            "requires_approval": False,
            "approval_phrase": None,
            "local_no_approval_allowed": False,
            "score_relevant": False,
            "next_allowed_step": "blocked",
            "recent_public_floor_cv_target": public_gap_decision.get("public_058_local_cv_target_recent_floor"),
        },
        {
            "rank": 4,
            "route": "manual_id341_public_sanity_probe",
            "tier": "closed_already_consumed",
            "status": "closed_after_id341_id340_id337_micro_probe_results",
            "target_focus": ["all"],
            "why": "The ID341/340/337 public sanity-probe family has already been spent and remained near 0.60, so it should not be repeated.",
            "requires_approval": False,
            "approval_phrase": None,
            "local_no_approval_allowed": False,
            "score_relevant": False,
            "next_allowed_step": "closed",
            "submission_file": manual_decision.get("lowest_risk_probe_file", MANUAL_PROBE_FILE),
        },
        {
            "rank": 5,
            "route": "q3_s3_closed_local_lanes",
            "tier": "closed",
            "status": "do_not_reopen_without_new_material_source",
            "target_focus": ["Q3", "S3"],
            "why": "Remaining-route inventory says Q3/S3 local no-approval lanes are closed; same-family local recombination caused the ID287/ID329 failure mode.",
            "requires_approval": False,
            "approval_phrase": None,
            "local_no_approval_allowed": False,
            "score_relevant": False,
            "next_allowed_step": "closed",
            "q3_s3_still_uncovered": q3s3_decision.get("q3_s3_still_uncovered"),
        },
    ]


def build_report() -> dict[str, Any]:
    reports = {key: load_report(filename) for key, filename in REQUIRED_REPORTS.items()}
    train_schema = schema_status(TRAIN_PROCESSED)
    test_schema = schema_status(TEST_PROCESSED)
    raw_rows = raw_stream_status()
    all_reports_ready = all(row.get("exists") and row.get("parse_ok") for row in reports.values())
    safe_wait_ready = decision(reports, "safe_wait").get("safe_wait_state_ready") is True
    material_queue_ready = decision(reports, "material_queue").get("material_new_source_queue_ready") is True
    material_coverage_ready = (
        decision(reports, "material_coverage").get("material_source_remote_coverage_contract_pass") is True
    )
    all_raw_streams_present = all(row.get("exists") and row.get("has_subject_timestamp") for row in raw_rows)
    processed_schema_ready = bool(train_schema.get("shape") == [450, 242] and test_schema.get("shape") == [250, 242])
    routes = route_rows(reports, train_schema, test_schema)

    route_triage_ready = bool(
        all_reports_ready
        and safe_wait_ready
        and material_queue_ready
        and material_coverage_ready
        and all_raw_streams_present
        and processed_schema_ready
    )
    no_approval_routes = [row for row in routes if row["local_no_approval_allowed"]]
    open_score_routes = [
        row for row in routes if row["score_relevant"] and row["status"] == "open_requires_exact_approval"
    ]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_next_material_route_triage",
        "report_only": True,
        "purpose": "Rank the next material routes without opening any new local score lane after the safe-wait state.",
        "inputs": {key: {k: v for k, v in row.items() if k != "payload"} for key, row in reports.items()},
        "processed_schema": {"train": train_schema, "test": test_schema},
        "raw_stream_rows": raw_rows,
        "route_rows": routes,
        "evidence_flags": {
            "all_reports_ready": all_reports_ready,
            "safe_wait_ready": safe_wait_ready,
            "material_queue_ready": material_queue_ready,
            "material_coverage_ready": material_coverage_ready,
            "all_raw_streams_present": all_raw_streams_present,
            "processed_schema_ready": processed_schema_ready,
            "sleep_proxy_backlog_metadata_feasible": bool(routes[2].get("metadata_feasible")),
        },
        "decision": {
            "next_material_route_triage_ready": route_triage_ready,
            "local_no_approval_score_route_exists": bool(no_approval_routes),
            "no_new_local_lane_opened": True,
            "recommended_primary_next_action": PRIMARY_SCORE_ROUTE,
            "requires_explicit_kaggle_approval": True,
            "approval_phrase": KAGGLE_APPROVAL_PHRASE,
            "secondary_heavy_route_requires_separate_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "sleep_proxy_threshold_residual_backlog_feasible": bool(routes[2].get("metadata_feasible")),
            "sleep_proxy_threshold_residual_open_now": False,
            "manual_probe_if_user_spends_submission": None,
            "manual_probe_is_public_058_evidence": False,
            "public_058_reviewable": False,
            "create_feature_matrix": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "open_score_route_count": len(open_score_routes),
            "next_action": "Do not reopen local recombination. Wait for the exact Kaggle no-submit upstream approval, or keep only design-level backlog triage with no feature matrix/model/submission artifacts.",
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision_row = report["decision"]
    lines = [
        "# Post-ID341 next material route triage",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrix, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Triage ready: `{decision_row['next_material_route_triage_ready']}`",
        f"- Local no-approval score route exists: `{decision_row['local_no_approval_score_route_exists']}`",
        f"- No new local lane opened: `{decision_row['no_new_local_lane_opened']}`",
        f"- Recommended primary next action: `{decision_row['recommended_primary_next_action']}`",
        f"- Kaggle approval phrase: `{decision_row['approval_phrase']}`",
        f"- Secondary heavy route requires separate approval: `{decision_row['secondary_heavy_route_requires_separate_approval']}`",
        f"- Sleep-proxy backlog feasible: `{decision_row['sleep_proxy_threshold_residual_backlog_feasible']}`",
        f"- Sleep-proxy backlog open now: `{decision_row['sleep_proxy_threshold_residual_open_now']}`",
        f"- Public 0.58 reviewable: `{decision_row['public_058_reviewable']}`",
        f"- Create feature matrix: `{decision_row['create_feature_matrix']}`",
        f"- Train model now: `{decision_row['train_model_now']}`",
        f"- Create submission artifact: `{decision_row['create_submission_artifact']}`",
        f"- Create ID342/new submission ID: `{decision_row['create_id342_or_new_submission_id']}`",
        f"- Submit now: `{decision_row['submit_now']}`",
        f"- Push Kaggle now: `{decision_row['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_row['run_kaggle_now']}`",
        f"- Next action: {decision_row['next_action']}",
        "",
        "## Route ranking",
        "",
        "| rank | route | tier | status | score relevant | approval | local no-approval | next step |",
        "|---:|---|---|---|---:|---|---:|---|",
    ]
    for row in report["route_rows"]:
        lines.append(
            f"| {row['rank']} | `{row['route']}` | `{row['tier']}` | `{row['status']}` | "
            f"`{row['score_relevant']}` | `{row['approval_phrase']}` | `{row['local_no_approval_allowed']}` | "
            f"`{row['next_allowed_step']}` |"
        )
    lines.extend(["", "## Evidence flags", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in report["evidence_flags"].items())
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
