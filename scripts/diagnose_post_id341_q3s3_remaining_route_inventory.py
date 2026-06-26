#!/usr/bin/env python3
"""Inventory the remaining post-ID341 Q3/S3 score routes.

This report-only gate consolidates the local Q3/S3 closures, data-source
coverage, manual probe decision, and Kaggle prelaunch state into one answer:
is there any no-approval local score route left, or is the only score-relevant
path the explicit-approved no-submit upstream rebuild?

It intentionally creates no predictions, submission CSV, notebook artifact,
submission-log row, Kaggle run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
PROCESSED_TRAIN = ROOT / "data" / "processed" / "df_train_raw.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "df_test_raw.parquet"
KST = timezone(timedelta(hours=9))

NEGATIVE_EVIDENCE_PATH = REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json"
LOCAL_CLOSURE_PATH = REPORTS_DIR / "post_id341_q3s3_local_lane_closure_20260618.json"
FOLD_NESTED_SIBLING_BRIDGE_PATH = REPORTS_DIR / "post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json"
GPS_TOPOLOGY_MODEL_SMOKE_PATH = REPORTS_DIR / "post_id341_gps_path_topology_model_smoke_20260619.json"
EXTERNAL_PUBLIC05917_RESIDUAL_PATH = REPORTS_DIR / "post_id341_external_public05917_residual_audit_20260619.json"
SOURCE_MATRIX_PATH = REPORTS_DIR / "post_id341_q2q3s3_source_hypothesis_matrix_20260618.json"
DATA_COVERAGE_PATH = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"
MANUAL_PROBE_PATH = REPORTS_DIR / "post_id341_manual_submission_probe_decision_20260618.json"

APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
TARGET_FOCUS = ["Q3", "S3"]
RAW_MODALITIES = {
    "mACStatus": "charging/sleep-boundary state",
    "mActivity": "phone activity state",
    "mAmbience": "ambient semantic context",
    "mBle": "nearby-device/social novelty",
    "mGps": "place movement and route novelty",
    "mLight": "phone-light sleep-boundary context",
    "mScreenStatus": "screen-off/on transitions",
    "mUsageStats": "app-use routine and category load",
    "mWifi": "place/social novelty",
    "wHr": "wearable physiology",
    "wLight": "wearable light exposure",
    "wPedo": "movement and step rhythm",
}


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "parse_ok": True,
        "decision": payload.get("decision", {}),
    }


def parquet_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "readable": False}
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "readable": True,
        "rows": int(parquet.metadata.num_rows),
        "columns": list(parquet.schema.names),
        "column_count": len(parquet.schema.names),
        "bytes": path.stat().st_size,
    }


def raw_modality_rows() -> list[dict[str, Any]]:
    rows = []
    for modality, purpose in RAW_MODALITIES.items():
        path = RAW_DIR / f"ch2025_{modality}.parquet"
        status = parquet_status(path)
        rows.append(
            {
                "modality": modality,
                "purpose": purpose,
                "path": status["path"],
                "exists": status["exists"],
                "readable": status["readable"],
                "rows": status.get("rows"),
                "columns": status.get("columns", []),
                "column_count": status.get("column_count"),
                "bytes": status.get("bytes"),
            }
        )
    return rows


def processed_schema_summary() -> dict[str, Any]:
    train = parquet_status(PROCESSED_TRAIN)
    test = parquet_status(PROCESSED_TEST)
    target_cols = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    train_cols = set(train.get("columns", []))
    test_cols = set(test.get("columns", []))
    return {
        "train": {
            "path": train["path"],
            "exists": train["exists"],
            "readable": train["readable"],
            "rows": train.get("rows"),
            "column_count": train.get("column_count"),
        },
        "test": {
            "path": test["path"],
            "exists": test["exists"],
            "readable": test["readable"],
            "rows": test.get("rows"),
            "column_count": test.get("column_count"),
        },
        "train_targets_present": sorted(col for col in target_cols if col in train_cols),
        "test_targets_present": sorted(col for col in target_cols if col in test_cols),
        "shared_feature_count": len((train_cols & test_cols) - set(target_cols)),
    }


def closed_route_rows(
    negative_evidence: dict[str, Any],
    local_closure: dict[str, Any],
    sibling_bridge: dict[str, Any],
    gps_topology: dict[str, Any],
    external_public05917: dict[str, Any],
    sibling_bridge_path: Path = FOLD_NESTED_SIBLING_BRIDGE_PATH,
) -> list[dict[str, Any]]:
    rows = []
    for row in negative_evidence.get("lane_rows", []):
        rows.append(
            {
                "route": row.get("lane"),
                "status": "closed",
                "target_focus": TARGET_FOCUS,
                "reason": row.get("reason"),
                "evidence_report": row.get("report"),
                "artifact_allowed": bool(row.get("artifact_allowed")),
            }
        )
    for row in negative_evidence.get("scout_rows", []):
        rows.append(
            {
                "route": row.get("lane"),
                "status": "closed_scout",
                "target_focus": TARGET_FOCUS,
                "reason": row.get("reason"),
                "evidence_report": row.get("report"),
                "artifact_allowed": bool(row.get("artifact_allowed")),
            }
        )
    if local_closure.get("decision", {}).get("local_fallback_closed_for_submission_artifact") is True:
        rows.append(
            {
                "route": "q2_only_local_fallback",
                "status": "closed_for_q3s3_score_route",
                "target_focus": ["Q2"],
                "reason": "The local fallback leaves Q3/S3 uncovered and cannot justify a new artifact.",
                "evidence_report": safe_relative(LOCAL_CLOSURE_PATH),
                "artifact_allowed": False,
            }
        )
    sibling_decision = sibling_bridge.get("decision", {})
    if (
        sibling_decision.get("fold_nested_sibling_smoke_completed") is True
        and sibling_decision.get("selected_targets") == []
        and sibling_decision.get("q3_signal_found") is False
        and sibling_decision.get("s3_signal_found") is False
        and sibling_decision.get("all_q3s3_targets_covered") is False
        and sibling_decision.get("public_058_evidence") is False
        and sibling_decision.get("local_artifact_allowed") is False
    ):
        rows.append(
            {
                "route": "fold_nested_sibling_bridge",
                "status": "closed_for_local_artifact",
                "target_focus": TARGET_FOCUS,
                "reason": (
                    "Leakage-safe Q3<-Q2 and S3<-S2 nested sibling predictions selected no Q3/S3 target "
                    "and provide no Public-0.58 evidence."
                ),
                "evidence_report": safe_relative(sibling_bridge_path),
                "artifact_allowed": False,
            }
        )
    gps_decision = gps_topology.get("decision", {})
    if (
        gps_decision.get("model_smoke_completed") is True
        and gps_decision.get("source_family") == "gps_path_topology_sequence_v1"
        and gps_decision.get("selected_targets") == []
        and gps_decision.get("any_gps_topology_signal_found") is False
        and gps_decision.get("public_058_evidence") is False
        and gps_decision.get("score_route_opened") is False
        and gps_decision.get("create_submission_artifact") is False
    ):
        rows.append(
            {
                "route": "gps_path_topology_sequence_v1",
                "status": "closed_after_model_smoke",
                "target_focus": TARGET_FOCUS,
                "reason": (
                    "The raw GPS path-topology feature matrix is valid, but fold-internal model smoke selected no "
                    "Q3/S3 target and opened no Public-0.58 score route."
                ),
                "evidence_report": safe_relative(GPS_TOPOLOGY_MODEL_SMOKE_PATH),
                "artifact_allowed": False,
            }
        )
    external_decision = external_public05917.get("decision", {})
    if (
        external_decision.get("audit_ready") is True
        and external_decision.get("residual_direct_notebook_route_exists") is False
        and external_decision.get("public_058_evidence_present") is False
        and external_decision.get("q3_s3_still_uncovered") is True
        and external_decision.get("create_submission_artifact") is False
    ):
        rows.append(
            {
                "route": "external_public05917_direct_replay",
                "status": "closed_as_direct_route",
                "target_focus": TARGET_FOCUS,
                "reason": (
                    "The external Public-0.5917 notebook remains useful as validation philosophy, but it lacks a "
                    "reproducible direct route and still leaves Q3/S3 uncovered."
                ),
                "evidence_report": safe_relative(EXTERNAL_PUBLIC05917_RESIDUAL_PATH),
                "artifact_allowed": False,
            }
        )
    return rows


def route_rows(
    source_matrix: dict[str, Any],
    data_coverage: dict[str, Any],
    manual_probe: dict[str, Any],
) -> list[dict[str, Any]]:
    matrix_decision = source_matrix.get("decision", {})
    coverage_decision = data_coverage.get("decision", {})
    return [
        {
            "route": "explicit_approved_clean_upstream_id221_to_id226",
            "status": "open_but_requires_explicit_approval",
            "score_relevant": True,
            "target_focus": matrix_decision.get("missing_targets", ["Q2", "Q3", "S3"]),
            "approval_phrase": APPROVAL_PHRASE,
            "reason": (
                "Historical ID226-style upstream signal covers Q2/Q3/S3, while local post-ID341 lanes are closed. "
                "This is the only route that can move the missing targets without spending a DACON submission."
            ),
        },
        {
            "route": "local_summary_or_combo_cascade",
            "status": "closed",
            "score_relevant": False,
            "target_focus": TARGET_FOCUS,
            "reason": (
                "ID287 and ID329 showed that local-frontier/full-OOF or same-family combo candidates can look better "
                "locally while staying near or worse than the ID178 public anchor."
            ),
        },
        {
            "route": "external_direct_row_concatenation",
            "status": "blocked",
            "score_relevant": False,
            "target_focus": TARGET_FOCUS,
            "reason": "External ETRI data is allowed as prior/context only; direct train-row concatenation remains blocked.",
            "external_direct_concat_allowed": coverage_decision.get("external_direct_concat_allowed"),
        },
        {
            "route": "manual_id341_public_sanity_probe",
            "status": "closed_after_id341_id340_id337_micro_probe_results",
            "score_relevant": False,
            "target_focus": ["Q2"],
            "submission_file": None,
            "reason": (
                "The ID341/340/337 sanity-probe family has already been spent and remained near 0.60, "
                "so it should not be repeated."
            ),
        },
    ]


def build_report(
    negative_evidence_path: Path = NEGATIVE_EVIDENCE_PATH,
    fold_nested_sibling_bridge_path: Path = FOLD_NESTED_SIBLING_BRIDGE_PATH,
) -> dict[str, Any]:
    negative_evidence_path = (
        negative_evidence_path if negative_evidence_path.is_absolute() else ROOT / negative_evidence_path
    )
    fold_nested_sibling_bridge_path = (
        fold_nested_sibling_bridge_path
        if fold_nested_sibling_bridge_path.is_absolute()
        else ROOT / fold_nested_sibling_bridge_path
    )
    negative_status = json_status(negative_evidence_path)
    local_status = json_status(LOCAL_CLOSURE_PATH)
    sibling_status = json_status(fold_nested_sibling_bridge_path)
    gps_status = json_status(GPS_TOPOLOGY_MODEL_SMOKE_PATH)
    external_status = json_status(EXTERNAL_PUBLIC05917_RESIDUAL_PATH)
    source_status = json_status(SOURCE_MATRIX_PATH)
    data_status = json_status(DATA_COVERAGE_PATH)
    manual_status = json_status(MANUAL_PROBE_PATH)

    negative = load_json(negative_evidence_path)
    local = load_json(LOCAL_CLOSURE_PATH)
    sibling_bridge = load_json(fold_nested_sibling_bridge_path)
    gps_topology = load_json(GPS_TOPOLOGY_MODEL_SMOKE_PATH)
    external_public05917 = load_json(EXTERNAL_PUBLIC05917_RESIDUAL_PATH)
    source_matrix = load_json(SOURCE_MATRIX_PATH)
    data_coverage = load_json(DATA_COVERAGE_PATH)
    manual_probe = load_json(MANUAL_PROBE_PATH)

    raw_rows = raw_modality_rows()
    closed_rows = closed_route_rows(
        negative,
        local,
        sibling_bridge,
        gps_topology,
        external_public05917,
        fold_nested_sibling_bridge_path,
    )
    open_and_blocked_rows = route_rows(source_matrix, data_coverage, manual_probe)
    open_score_routes = [
        row
        for row in open_and_blocked_rows
        if row.get("score_relevant") is True and row["status"] == "open_but_requires_explicit_approval"
    ]
    local_no_approval_routes = [
        row
        for row in open_and_blocked_rows
        if row.get("score_relevant") is True and row["status"] == "open_without_approval"
    ]

    all_inputs_ready = all(
        status["exists"] and status["parse_ok"]
        for status in [
            negative_status,
            local_status,
            sibling_status,
            gps_status,
            external_status,
            source_status,
            data_status,
            manual_status,
        ]
    )
    all_raw_sources_readable = all(row["exists"] and row["readable"] for row in raw_rows)
    q3s3_closed = negative.get("decision", {}).get("all_local_q3s3_lanes_closed") is True
    sibling_decision = sibling_bridge.get("decision", {})
    sibling_bridge_closed = bool(
        sibling_decision.get("fold_nested_sibling_smoke_completed") is True
        and sibling_decision.get("selected_targets") == []
        and sibling_decision.get("q3_signal_found") is False
        and sibling_decision.get("s3_signal_found") is False
        and sibling_decision.get("all_q3s3_targets_covered") is False
        and sibling_decision.get("public_058_evidence") is False
        and sibling_decision.get("local_artifact_allowed") is False
        and sibling_decision.get("create_submission_artifact") is False
        and sibling_decision.get("submit_now") is False
        and sibling_decision.get("push_kaggle_now") is False
        and sibling_decision.get("run_kaggle_now") is False
    )
    gps_decision = gps_topology.get("decision", {})
    gps_topology_model_lane_closed = bool(
        gps_decision.get("model_smoke_completed") is True
        and gps_decision.get("selected_targets") == []
        and gps_decision.get("any_gps_topology_signal_found") is False
        and gps_decision.get("public_058_evidence") is False
        and gps_decision.get("score_route_opened") is False
        and gps_decision.get("create_submission_artifact") is False
        and gps_decision.get("submit_now") is False
        and gps_decision.get("push_kaggle_now") is False
        and gps_decision.get("run_kaggle_now") is False
    )
    external_decision = external_public05917.get("decision", {})
    external_public05917_direct_route_closed = bool(
        external_decision.get("audit_ready") is True
        and external_decision.get("residual_direct_notebook_route_exists") is False
        and external_decision.get("public_058_evidence_present") is False
        and external_decision.get("q3_s3_still_uncovered") is True
        and external_decision.get("create_submission_artifact") is False
        and external_decision.get("submit_now") is False
        and external_decision.get("push_kaggle_now") is False
        and external_decision.get("run_kaggle_now") is False
    )
    inventory_ready = bool(
        all_inputs_ready
        and q3s3_closed
        and sibling_bridge_closed
        and gps_topology_model_lane_closed
        and external_public05917_direct_route_closed
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_remaining_route_inventory",
        "purpose": (
            "Consolidate post-ID341 Q3/S3 closures and open routes so closed local lanes are not repeatedly "
            "re-promoted as score candidates."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "negative_evidence_map": negative_status,
            "local_lane_closure": local_status,
            "fold_nested_sibling_bridge": sibling_status,
            "gps_path_topology_model_smoke": gps_status,
            "external_public05917_residual": external_status,
            "source_hypothesis_matrix": source_status,
            "data_source_coverage": data_status,
            "manual_submission_probe_decision": manual_status,
        },
        "raw_modality_rows": raw_rows,
        "processed_schema_summary": processed_schema_summary(),
        "closed_route_rows": closed_rows,
        "open_or_blocked_route_rows": open_and_blocked_rows,
        "checks": {
            "all_inputs_ready": all_inputs_ready,
            "all_raw_sources_readable": all_raw_sources_readable,
            "q3s3_local_lanes_closed": q3s3_closed,
            "fold_nested_sibling_bridge_closed": sibling_bridge_closed,
            "gps_path_topology_model_lane_closed": gps_topology_model_lane_closed,
            "external_public05917_direct_route_closed": external_public05917_direct_route_closed,
            "local_no_approval_score_route_count": len(local_no_approval_routes),
            "open_score_route_count": len(open_score_routes),
        },
        "decision": {
            "q3s3_remaining_route_inventory_ready": inventory_ready,
            "target_focus": TARGET_FOCUS,
            "q3_s3_still_uncovered": True,
            "fold_nested_sibling_bridge_closed": sibling_bridge_closed,
            "gps_path_topology_model_lane_closed": gps_topology_model_lane_closed,
            "external_public05917_direct_route_closed": external_public05917_direct_route_closed,
            "local_no_approval_score_route_exists": bool(local_no_approval_routes),
            "only_score_relevant_open_route": open_score_routes[0]["route"] if len(open_score_routes) == 1 else None,
            "requires_explicit_kaggle_approval": True,
            "approval_phrase": APPROVAL_PHRASE,
            "manual_probe_if_user_spends_submission": None,
            "manual_probe_family_status": "consumed_after_id341_id340_id337_public_micro_probe_slate",
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not create a local ID342 from closed Q3/S3 lanes. The next score-relevant action is the "
                "explicit-approved no-submit Kaggle upstream rebuild; otherwise continue with report-only design/tests."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Q3/S3 remaining route inventory",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, notebook artifact, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Inventory ready: `{decision['q3s3_remaining_route_inventory_ready']}`",
        f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
        f"- GPS topology model lane closed: `{decision['gps_path_topology_model_lane_closed']}`",
        f"- External Public 0.5917 direct route closed: `{decision['external_public05917_direct_route_closed']}`",
        f"- Local no-approval score route exists: `{decision['local_no_approval_score_route_exists']}`",
        f"- Only score-relevant open route: `{decision['only_score_relevant_open_route']}`",
        f"- Requires explicit Kaggle approval: `{decision['requires_explicit_kaggle_approval']}`",
        f"- Approval phrase: `{decision['approval_phrase']}`",
        f"- Manual probe if spending DACON submission: `{decision['manual_probe_if_user_spends_submission']}`",
        f"- Manual probe family status: `{decision['manual_probe_family_status']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(report["checks"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Closed local routes",
        "",
        "| route | status | targets | artifact allowed | reason |",
        "|---|---|---|---|---|",
    ]
    for row in report["closed_route_rows"]:
        lines.append(
            f"| `{row['route']}` | `{row['status']}` | `{','.join(row.get('target_focus') or [])}` | "
            f"`{row.get('artifact_allowed')}` | {row.get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## Open or blocked routes",
            "",
            "| route | status | score relevant | targets | reason |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["open_or_blocked_route_rows"]:
        lines.append(
            f"| `{row['route']}` | `{row['status']}` | `{row.get('score_relevant')}` | "
            f"`{','.join(row.get('target_focus') or [])}` | {row.get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## Raw modality coverage",
            "",
            "| modality | rows | columns | purpose |",
            "|---|---:|---:|---|",
        ]
    )
    for row in report["raw_modality_rows"]:
        lines.append(
            f"| `{row['modality']}` | {row.get('rows') or ''} | {row.get('column_count') or ''} | {row['purpose']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_q3s3_remaining_route_inventory_20260618.json",
    )
    parser.add_argument(
        "--negative-evidence-map",
        type=Path,
        default=NEGATIVE_EVIDENCE_PATH,
        help="Path to the Q3/S3 negative-evidence map used by this inventory.",
    )
    parser.add_argument(
        "--fold-nested-sibling-bridge",
        type=Path,
        default=FOLD_NESTED_SIBLING_BRIDGE_PATH,
        help="Path to the Q3/S3 fold-nested sibling bridge smoke used by this inventory.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args.negative_evidence_map, args.fold_nested_sibling_bridge)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
