#!/usr/bin/env python3
"""Report-only design for post-ID341 usage semantic burst/Markov source work.

The material source queue allows only a design step for this lane. This script
therefore inspects lightweight raw metadata/cardinality, documents how this
family differs from the failed aggregate usage-temporal smoke, and stops. It
does not generate features, train models, create predictions, create submission
artifacts, modify the submission log, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_design_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
USAGE_TEMPORAL_SMOKE = REPORTS_DIR / "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json"
AMBIENCE_CLOSURE = REPORTS_DIR / "phase3_ambience_context_closure_post_id341_20260618.json"
SEMANTIC_ROUTINE_CLOSURE = REPORTS_DIR / "post_id341_q3s3_semantic_routine_lane_closure_20260618.json"
PUBLIC_JOINT_RISK = REPORTS_DIR / "public_joint_distribution_risk_post_id341_20260618.json"

SOURCE_FAMILY = "usage_semantic_burst_markov_v1"
TARGET_FOCUS = ["Q2", "Q3"]
EXPECTED_NEXT_STEP_TYPE = "report_only_usage_semantic_burst_markov_design_no_model"
ALLOWED_QUEUE_STEP_TYPES = {
    EXPECTED_NEXT_STEP_TYPE,
    "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features",
    "report_only_usage_semantic_burst_markov_feature_spec_no_matrix",
    "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix",
    "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract",
    "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
    "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
}
RAW_FILES = {
    "mUsageStats": RAW_ITEMS_DIR / "ch2025_mUsageStats.parquet",
    "mScreenStatus": RAW_ITEMS_DIR / "ch2025_mScreenStatus.parquet",
    "mAmbience": RAW_ITEMS_DIR / "ch2025_mAmbience.parquet",
}
PREDECLARED_APP_CATEGORIES = [
    "communication",
    "media_search",
    "finance_shop",
    "system_home",
    "routine_health",
    "mobility_context",
    "other",
]
FORBIDDEN_ACTION_KEYS = [
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


def load_json_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
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


def no_forbidden_actions(report: dict[str, Any]) -> bool:
    dec = decision(report)
    return all(dec.get(key) is not True for key in FORBIDDEN_ACTION_KEYS)


def queue_lane_completed_no_model(queue: dict[str, Any]) -> bool:
    items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    item = next((entry for entry in items if entry.get("name") == SOURCE_FAMILY), {})
    return bool(
        item.get("status") == "validated_stop_completed_no_model"
        and item.get("local_no_approval_allowed") is False
        and item.get("feature_matrix_materialized_under_contract") is True
        and item.get("train_model_now") is False
        and item.get("create_artifact_allowed") is False
        and item.get("submission_artifact_allowed") is False
    )


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


def expected_days() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True).drop_duplicates(["subject_id", "lifelog_date", "split"])


def coverage_from_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> dict[str, Any]:
    expected = expected_days()
    days = df[["subject_id", timestamp_col]].copy()
    days["lifelog_date"] = pd.to_datetime(days[timestamp_col]).dt.strftime("%Y-%m-%d")
    per_day = days.groupby(["subject_id", "lifelog_date"]).size().rename("rows").reset_index()
    joined = expected.merge(per_day, on=["subject_id", "lifelog_date"], how="left")
    joined["rows"] = joined["rows"].fillna(0)
    out: dict[str, Any] = {}
    for split, split_df in joined.groupby("split"):
        out[str(split)] = {
            "expected_days": int(len(split_df)),
            "covered_days": int((split_df["rows"] > 0).sum()),
            "coverage_rate": float((split_df["rows"] > 0).mean()),
            "median_rows_per_expected_day": float(split_df["rows"].median()),
            "p90_rows_per_expected_day": float(split_df["rows"].quantile(0.90)),
        }
    return out


def app_name_to_category(name: str) -> str:
    normalized = str(name).lower().strip()
    if any(token in normalized for token in ["카카오", "톡", "전화", "통화", "메시지", "mail", "message"]):
        return "communication"
    if any(token in normalized for token in ["naver", "youtube", "검색", "브라우저", "chrome", "music", "음악"]):
        return "media_search"
    if any(token in normalized for token in ["토스", "pay", "bank", "쇼핑", "shop", "마트", "카드"]):
        return "finance_shop"
    if any(token in normalized for token in ["one ui", "홈", "launcher", "system", "설정"]):
        return "system_home"
    if any(token in normalized for token in ["health", "헬스", "walk", "캐시워크", "운동", "성경"]):
        return "routine_health"
    if any(token in normalized for token in ["map", "지도", "버스", "지하철", "taxi", "naver map"]):
        return "mobility_context"
    return "other"


def usage_payload_profile() -> dict[str, Any]:
    df = pd.read_parquet(RAW_FILES["mUsageStats"], columns=["subject_id", "timestamp", "m_usage_stats"])
    payload_lengths = df["m_usage_stats"].map(lambda value: len(value) if hasattr(value, "__len__") else 0)
    app_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    total_time_values: list[float] = []
    for payload in df["m_usage_stats"]:
        if isinstance(payload, (str, bytes, dict)) or not hasattr(payload, "__iter__"):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            app_name = str(item.get("app_name", ""))
            app_counter[app_name] += 1
            category_counter[app_name_to_category(app_name)] += 1
            try:
                total_time_values.append(float(item.get("total_time", 0.0)))
            except (TypeError, ValueError):
                pass
    ts = pd.to_datetime(df["timestamp"])
    return {
        "path": rel(RAW_FILES["mUsageStats"]),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "timestamp_span": {"min": str(ts.min()), "max": str(ts.max())},
        "rows_with_payload_rate": float((payload_lengths > 0).mean()),
        "median_apps_per_row": float(payload_lengths.median()),
        "p90_apps_per_row": float(payload_lengths.quantile(0.90)),
        "unique_app_names": int(len(app_counter)),
        "predeclared_category_event_counts": dict(category_counter),
        "top_app_name_counts_sample": dict(app_counter.most_common(12)),
        "total_app_events": int(sum(app_counter.values())),
        "median_total_time_per_app_event": float(pd.Series(total_time_values).median()) if total_time_values else None,
        "coverage_by_split": coverage_from_timestamps(df),
    }


def screen_status_profile() -> dict[str, Any]:
    df = pd.read_parquet(RAW_FILES["mScreenStatus"], columns=["subject_id", "timestamp", "m_screen_use"])
    ts = pd.to_datetime(df["timestamp"])
    ordered = df.assign(_ts=ts).sort_values(["subject_id", "_ts"])
    changed = ordered.groupby("subject_id")["m_screen_use"].diff().fillna(0).ne(0)
    return {
        "path": rel(RAW_FILES["mScreenStatus"]),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "timestamp_span": {"min": str(ts.min()), "max": str(ts.max())},
        "state_counts": {str(key): int(value) for key, value in df["m_screen_use"].value_counts().sort_index().items()},
        "state_change_rows": int(changed.sum()),
        "state_change_rate": float(changed.mean()),
        "coverage_by_split": coverage_from_timestamps(df),
    }


def raw_schema_summary() -> dict[str, Any]:
    schemas = {name: parquet_schema(path) for name, path in RAW_FILES.items()}
    return {
        "schemas": schemas,
        "all_required_files_present": all(schema["exists"] for schema in schemas.values()),
        "all_required_schemas_readable": all(schema["exists"] and not schema["error"] for schema in schemas.values()),
        "total_rows": int(sum(schema["num_rows"] or 0 for schema in schemas.values())),
    }


def blueprint() -> list[dict[str, Any]]:
    return [
        {
            "name": "app_burst_sessionization",
            "target_focus": TARGET_FOCUS,
            "sources": ["mUsageStats", "mScreenStatus"],
            "what_to_measure": [
                "sessionized app bursts from usage-stat timestamp deltas, not daily total usage",
                "late-evening and post-midnight reactivation bursts coupled with screen-on persistence",
                "subject-relative deviation from recent same-subject burst timing and duration",
            ],
            "why_not_repeat_phase3_usage_temporal": (
                "The failed phase3 usage-temporal lane used aggregate daily category totals. This design promotes only "
                "ordered burst/session shape and keeps raw app names out of model features."
            ),
        },
        {
            "name": "semantic_app_category_markov_chain",
            "target_focus": ["Q2", "Q3"],
            "sources": ["mUsageStats"],
            "predeclared_state_space": PREDECLARED_APP_CATEGORIES,
            "what_to_measure": [
                "category transition entropy and self-transition streaks within evening windows",
                "rare transition motifs such as communication→finance/shop or media/search→system_home loops",
                "Markov surprise versus each subject's rolling baseline transition matrix",
            ],
            "leakage_guard": "Use only fixed hand-written app categories; no target-guided app vocabulary or raw app one-hot.",
        },
        {
            "name": "ambience_guarded_context_burst",
            "target_focus": ["Q3"],
            "sources": ["mUsageStats", "mAmbience"],
            "what_to_measure": [
                "speech/music/vehicle ambience flags around high-intensity app bursts",
                "context disagreement: quiet ambience but high app-switching, or noisy ambience plus late reactivation",
            ],
            "reopen_guard": (
                "mAmbience is a guard/context source only because the ambience closure found full-OOF traps and no nested "
                "positive signal."
            ),
        },
    ]


def build_report() -> dict[str, Any]:
    queue = load_json_report(QUEUE_REPORT)
    usage_smoke = load_json_report(USAGE_TEMPORAL_SMOKE)
    ambience = load_json_report(AMBIENCE_CLOSURE)
    semantic_closure = load_json_report(SEMANTIC_ROUTINE_CLOSURE)
    joint_risk = load_json_report(PUBLIC_JOINT_RISK)
    queue_dec = decision(queue)
    usage_dec = decision(usage_smoke)
    ambience_dec = decision(ambience)
    semantic_dec = decision(semantic_closure)
    schemas = raw_schema_summary()
    usage_profile = usage_payload_profile()
    screen_profile = screen_status_profile()

    current_queue_allows_design = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type") in ALLOWED_QUEUE_STEP_TYPES
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
    )
    usage_lane_completed_no_model = bool(queue.get("parse_ok") is True and queue_lane_completed_no_model(queue))
    queue_allows_design = bool(current_queue_allows_design or usage_lane_completed_no_model)
    prior_usage_aggregate_closed = bool(
        usage_smoke.get("parse_ok") is True
        and usage_dec.get("usage_temporal_has_config_signal") is False
        and usage_dec.get("final_usage_temporal_gate_pass") is False
        and usage_dec.get("create_submission_artifact") is False
        and usage_dec.get("submit_now") is False
    )
    ambience_guard_only = bool(
        ambience.get("parse_ok") is True
        and ambience_dec.get("ambience_reopen_now") is False
        and ambience_dec.get("ambience_has_nested_positive_signal") is False
        and ambience_dec.get("create_submission_artifact") is False
        and ambience_dec.get("submit_now") is False
    )
    semantic_routine_artifact_closed = bool(
        semantic_closure.get("parse_ok") is True
        and semantic_dec.get("semantic_routine_lane_closure_passed") is True
        and semantic_dec.get("local_semantic_routine_artifact_allowed") is False
        and semantic_dec.get("create_submission_artifact") is False
        and semantic_dec.get("submit_now") is False
    )
    raw_cardinality_bounded_enough_for_design = bool(
        schemas["all_required_files_present"]
        and schemas["all_required_schemas_readable"]
        and usage_profile["rows_with_payload_rate"] > 0.90
        and usage_profile["unique_app_names"] >= 100
        and screen_profile["subjects"] >= 10
    )
    design_ready = bool(
        queue_allows_design
        and prior_usage_aggregate_closed
        and ambience_guard_only
        and semantic_routine_artifact_closed
        and raw_cardinality_bounded_enough_for_design
        and no_forbidden_actions(joint_risk)
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_usage_semantic_burst_markov_design",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Define the next no-approval local step after circadian closure: a design-only usage semantic burst/Markov "
            "source family that avoids the failed aggregate usage-temporal and ambience trap patterns."
        ),
        "inputs": {
            "material_new_source_queue": queue,
            "phase3_usage_temporal_smoke": usage_smoke,
            "phase3_ambience_context_closure": ambience,
            "semantic_routine_lane_closure": semantic_closure,
            "public_joint_distribution_risk": joint_risk,
        },
        "raw_schema_summary": schemas,
        "usage_payload_profile": usage_profile,
        "screen_status_profile": screen_profile,
        "feature_blueprint": blueprint(),
        "checks": {
            "queue_allows_report_only_design": queue_allows_design,
            "current_queue_allows_report_only_design": current_queue_allows_design,
            "usage_lane_completed_no_model": usage_lane_completed_no_model,
            "prior_usage_aggregate_closed": prior_usage_aggregate_closed,
            "ambience_guard_only": ambience_guard_only,
            "semantic_routine_artifact_closed": semantic_routine_artifact_closed,
            "raw_cardinality_bounded_enough_for_design": raw_cardinality_bounded_enough_for_design,
            "uses_predeclared_app_categories_only": True,
            "does_not_create_feature_matrix": True,
            "does_not_train_model": True,
            "does_not_create_submission_artifact": True,
        },
        "decision": {
            "design_ready": design_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "materially_different_from_failed_usage_temporal": True,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features",
            "next_action": (
                "Do not create ID342. The next no-approval step may only be a metadata/cardinality smoke for session "
                "state counts and Markov state coverage; no feature parquet, model, Kaggle run, or DACON submission."
                if design_ready
                else "Fix queue/prior-closure/raw-cardinality blockers before opening any usage semantic burst work."
            ),
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
    dec = report["decision"]
    usage = report["usage_payload_profile"]
    screen = report["screen_status_profile"]
    lines = [
        "# Post-ID341 usage semantic burst/Markov design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, "
            "notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Design ready: `{dec['design_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{dec['target_focus']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Raw metadata",
        "",
        f"- Usage rows / total app events / unique app names: `{usage['rows']}` / "
        f"`{usage['total_app_events']}` / `{usage['unique_app_names']}`",
        f"- Usage train/test coverage: `{usage['coverage_by_split']['train']['coverage_rate']:.4f}` / "
        f"`{usage['coverage_by_split']['test']['coverage_rate']:.4f}`",
        f"- Screen rows / state-change rate: `{screen['rows']}` / `{screen['state_change_rate']:.6f}`",
        f"- Screen train/test coverage: `{screen['coverage_by_split']['train']['coverage_rate']:.4f}` / "
        f"`{screen['coverage_by_split']['test']['coverage_rate']:.4f}`",
        "",
        "## Blueprint",
        "",
    ]
    for item in report["feature_blueprint"]:
        lines.extend(
            [
                f"### `{item['name']}`",
                "",
                f"- Target focus: `{item['target_focus']}`",
                f"- Sources: `{item['sources']}`",
                "- Measurements:",
            ]
        )
        lines.extend(f"  - {shape}" for shape in item["what_to_measure"])
        if "why_not_repeat_phase3_usage_temporal" in item:
            lines.append(f"- Anti-repeat rule: {item['why_not_repeat_phase3_usage_temporal']}")
        if "leakage_guard" in item:
            lines.append(f"- Leakage guard: {item['leakage_guard']}")
        if "reopen_guard" in item:
            lines.append(f"- Reopen guard: {item['reopen_guard']}")
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            (
                "This design does not create ID342 and does not authorize a submission. Any future feature matrix or "
                "model smoke requires a separate no-artifact contract after metadata/cardinality coverage is checked."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    write_report(report, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
