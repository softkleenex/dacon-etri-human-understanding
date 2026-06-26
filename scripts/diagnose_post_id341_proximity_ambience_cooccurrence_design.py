#!/usr/bin/env python3
"""Report-only design for post-ID341 proximity/ambience co-occurrence work.

This source family is a deliberately narrow local continuation after the usage
semantic burst/Markov matrix validation stop. It inspects whether WiFi/BLE
proximity traces can be combined with ambience speech/crowd/inside/outside
context as a materially different Q3/S3 source family. It creates no feature
matrix, trains no model, creates no predictions or submission artifacts, edits
no submission log, runs no Kaggle command, and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "proximity_ambience_cooccurrence_v1"
TARGET_FOCUS = ["Q3", "S3"]
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_design_20260619.json"

CONTEXT_REPORTS = {
    "no_approval_frontier": "post_id341_no_approval_frontier_20260619.json",
    "q3s3_remaining_route_inventory": "post_id341_q3s3_remaining_route_inventory_20260618.json",
    "q3s3_negative_evidence_map": "post_id341_q3s3_negative_evidence_map_20260618.json",
    "usage_matrix_validation_stop": "post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "local_source_closure": "post_id341_local_source_closure_synthesis_20260618.json",
}
RAW_FILES = {
    "mWifi": RAW_ITEMS_DIR / "ch2025_mWifi.parquet",
    "mBle": RAW_ITEMS_DIR / "ch2025_mBle.parquet",
    "mAmbience": RAW_ITEMS_DIR / "ch2025_mAmbience.parquet",
    "mScreenStatus": RAW_ITEMS_DIR / "ch2025_mScreenStatus.parquet",
}
FORBIDDEN_DECISION_KEYS = [
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
    "train_model_now",
    "execute_kaggle_now",
]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_json_report(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "exists": True, "parse_ok": False, "decision": {}, "error": str(exc)}
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    raw = report.get("decision", {})
    return raw if isinstance(raw, dict) else {}


def no_forbidden_actions(report: dict[str, Any]) -> bool:
    dec = decision(report)
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def parquet_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "columns": [], "num_rows": None, "error": None}
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return {
            "path": rel(path),
            "exists": True,
            "columns": list(parquet_file.schema.names),
            "num_rows": int(parquet_file.metadata.num_rows),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "exists": True, "columns": [], "num_rows": None, "error": str(exc)}


def context_status(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    no_approval = decision(reports["no_approval_frontier"])
    remaining = decision(reports["q3s3_remaining_route_inventory"])
    negative = decision(reports["q3s3_negative_evidence_map"])
    usage_stop = decision(reports["usage_matrix_validation_stop"])
    closure = decision(reports["local_source_closure"])
    return {
        "all_context_reports_present": all(
            report.get("exists") and report.get("parse_ok") for report in reports.values()
        ),
        "all_context_reports_action_safe": all(no_forbidden_actions(report) for report in reports.values()),
        "no_approval_score_route_exists_false": no_approval.get("no_approval_score_route_exists") is False,
        "only_score_relevant_open_route": no_approval.get("only_score_relevant_open_route"),
        "prelaunch_ready_for_explicit_approval": no_approval.get("prelaunch_ready_for_explicit_approval") is True,
        "q3_s3_still_uncovered": remaining.get("q3_s3_still_uncovered") is True,
        "all_local_q3s3_lanes_closed": negative.get("all_local_q3s3_lanes_closed") is True,
        "usage_validation_stop_passed": usage_stop.get("validation_stop_passed") is True,
        "usage_stop_opened_no_score_route": usage_stop.get("score_route_opened") is False,
        "local_source_stack_closed": closure.get("local_source_stack_closed_for_submission_artifact") is True,
    }


def source_blueprint() -> list[dict[str, Any]]:
    return [
        {
            "group": "proximity_signature_stability",
            "sources": ["mWifi", "mBle"],
            "feature_intent": [
                "count-only WiFi/BLE scan richness by evening and late windows",
                "hashed-never-persisted proximity signature repeat rate within subject/day",
                "near-device burst density from RSSI thresholds, without storing raw network identifier values",
            ],
            "target_hypothesis": "S3/Q3 may react to social/place density and late-day context shifts rather than raw GPS paths.",
        },
        {
            "group": "speech_crowd_inside_outside_context",
            "sources": ["mAmbience"],
            "feature_intent": [
                "speech/conversation/crowd scores as social context indicators",
                "inside/outside and vehicle/music scores as coarse environment states",
                "late-evening ambience volatility instead of daily aggregate ambience totals",
            ],
            "target_hypothesis": "S3 and Q3 can plausibly depend on social stimulation and environment recovery pressure.",
        },
        {
            "group": "cross_sensor_cooccurrence",
            "sources": ["mWifi", "mBle", "mAmbience", "mScreenStatus"],
            "feature_intent": [
                "10-minute-bin co-occurrence of proximity scans with speech/crowd ambience",
                "screen-on persistence when proximity/social ambience is active",
                "subject-relative novelty of proximity+ambience state sequences",
            ],
            "target_hypothesis": "This is materially different from closed GPS topology, usage Markov, and ambience-only lanes.",
        },
    ]


def build_report() -> dict[str, Any]:
    reports = {name: load_json_report(filename) for name, filename in CONTEXT_REPORTS.items()}
    status = context_status(reports)
    schemas = {name: parquet_schema(path) for name, path in RAW_FILES.items()}
    raw_ready = all(schema["exists"] and not schema["error"] for schema in schemas.values())
    design_ready = bool(
        raw_ready
        and status["all_context_reports_present"]
        and status["all_context_reports_action_safe"]
        and status["no_approval_score_route_exists_false"]
        and status["q3_s3_still_uncovered"]
        and status["all_local_q3s3_lanes_closed"]
        and status["usage_validation_stop_passed"]
        and status["usage_stop_opened_no_score_route"]
        and status["local_source_stack_closed"]
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_design",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "context_status": status,
        "raw_schema": schemas,
        "source_blueprint": source_blueprint(),
        "material_difference": {
            "vs_gps_path_topology": "Uses no coordinate/path topology; proximity IDs are only ephemeral count/signature inputs.",
            "vs_usage_semantic_burst_markov": "Uses WiFi/BLE/social ambience co-occurrence, not app-category Markov chains.",
            "vs_phase3_ambience_context": "Requires cross-sensor proximity+ambience timing, not ambience-only daily context.",
            "vs_event_cadence_or_dropout": "Uses context co-occurrence intensity, not missingness or generic event cadence alone.",
        },
        "privacy_and_leakage_guards": [
            "Do not persist raw WiFi/BLE identifiers or exact proximity token values in reports or feature matrices.",
            "Use predeclared ambience groups and RSSI thresholds; no target-guided vocabulary selection.",
            "Any later model smoke must fit preprocessing fold-internally and may only report split losses.",
            "No submission artifact, base prediction artifact, notebook/history artifact, or submission-log row is authorized.",
        ],
        "decision": {
            "design_ready": design_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "proximity_ambience_cooccurrence_metadata_smoke_no_features",
            "next_action": (
                "Run only the report-only metadata/co-occurrence smoke next; do not write a feature matrix, train, "
                "create ID342, run Kaggle, or submit."
                if design_ready
                else "Fix missing context/schema prerequisites before any next local step."
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


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 proximity/ambience co-occurrence design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no features, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Design ready: `{dec['design_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Context status",
        "",
    ]
    for key, value in report["context_status"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Raw schema", "", "| source | rows | columns | ready |", "|---|---:|---|---|"])
    for name, schema in report["raw_schema"].items():
        lines.append(
            f"| `{name}` | `{schema['num_rows']}` | `{','.join(schema['columns'])}` | `{schema['exists'] and not schema['error']}` |"
        )
    lines.extend(["", "## Blueprint", ""])
    for item in report["source_blueprint"]:
        lines.append(f"### {item['group']}")
        lines.append("")
        lines.append(f"- Sources: `{','.join(item['sources'])}`")
        lines.append(f"- Target hypothesis: {item['target_hypothesis']}")
        for feature in item["feature_intent"]:
            lines.append(f"  - {feature}")
        lines.append("")
    lines.extend(["## Material difference", ""])
    for key, value in report["material_difference"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Guards", ""])
    for guard in report["privacy_and_leakage_guards"]:
        lines.append(f"- {guard}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(md_path)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    if not report["decision"]["design_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
