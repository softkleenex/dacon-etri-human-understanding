#!/usr/bin/env python3
"""Report-only metadata/cardinality smoke for usage semantic burst/Markov work.

This is the first executable check after the usage semantic burst/Markov design.
It inspects raw app-usage and screen-status state coverage, app-category Markov
state coverage, and session/burst feasibility. It creates no feature matrix,
trains no model, creates no predictions or submission artifacts, modifies no
submission log, runs no Kaggle command, and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_usage_semantic_burst_markov_design import (  # noqa: E402
    PREDECLARED_APP_CATEGORIES,
    SOURCE_FAMILY,
    app_name_to_category,
    coverage_from_timestamps,
)

REPORTS_DIR = ROOT / "reports"
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.json"
DESIGN_REPORT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_design_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
USAGE_PATH = RAW_ITEMS_DIR / "ch2025_mUsageStats.parquet"
SCREEN_PATH = RAW_ITEMS_DIR / "ch2025_mScreenStatus.parquet"
AMBIENCE_PATH = RAW_ITEMS_DIR / "ch2025_mAmbience.parquet"

TARGET_FOCUS = ["Q2", "Q3"]
NEXT_ALLOWED_STEP = "report_only_usage_semantic_burst_markov_feature_spec_no_matrix"
ALLOWED_QUEUE_STEP_TYPES = {
    "report_only_usage_semantic_burst_markov_design_no_model",
    "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features",
    NEXT_ALLOWED_STEP,
    "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix",
    "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract",
    "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
    "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
}
FORBIDDEN_DECISION_KEYS = [
    "feature_matrix_allowed_now",
    "train_model_now",
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
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
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def queue_item(queue: dict[str, Any], name: str) -> dict[str, Any]:
    items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    return next((item for item in items if item.get("name") == name), {})


def queue_lane_completed_no_model(queue: dict[str, Any]) -> bool:
    item = queue_item(queue, SOURCE_FAMILY)
    return bool(
        item.get("status") == "validated_stop_completed_no_model"
        and item.get("local_no_approval_allowed") is False
        and item.get("feature_matrix_materialized_under_contract") is True
        and item.get("train_model_now") is False
        and item.get("create_artifact_allowed") is False
        and item.get("submission_artifact_allowed") is False
    )


def iter_usage_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, (str, bytes, dict)) or not hasattr(payload, "__iter__"):
        return []
    return [item for item in payload if isinstance(item, dict)]


def payload_category_summary(payload: Any) -> dict[str, Any]:
    category_time = dict.fromkeys(PREDECLARED_APP_CATEGORIES, 0.0)
    categories_present: set[str] = set()
    app_count = 0
    total_time = 0.0
    top_app_name = ""
    top_app_time = -1.0
    for item in iter_usage_payload(payload):
        app_count += 1
        app_name = str(item.get("app_name", ""))
        category = app_name_to_category(app_name)
        categories_present.add(category)
        try:
            total = float(item.get("total_time", 0.0))
        except (TypeError, ValueError):
            total = 0.0
        total_time += total
        category_time[category] += total
        if total > top_app_time:
            top_app_time = total
            top_app_name = app_name
    if app_count == 0:
        dominant_category = "other"
    else:
        dominant_category = max(PREDECLARED_APP_CATEGORIES, key=lambda category: category_time[category])
    return {
        "app_count": app_count,
        "total_time": total_time,
        "dominant_category": dominant_category,
        "categories_present": sorted(categories_present),
        "top_app_name": top_app_name,
    }


def build_usage_row_frame() -> pd.DataFrame:
    raw = pd.read_parquet(USAGE_PATH, columns=["subject_id", "timestamp", "m_usage_stats"])
    summaries = raw["m_usage_stats"].map(payload_category_summary)
    summary_frame = pd.DataFrame(list(summaries))
    frame = pd.concat([raw[["subject_id", "timestamp"]].copy(), summary_frame], axis=1)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["lifelog_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["hour"] = frame["timestamp"].dt.hour
    frame["has_payload"] = frame["app_count"] > 0
    return frame


def usage_cardinality_profile(frame: pd.DataFrame) -> dict[str, Any]:
    category_counts = frame["dominant_category"].value_counts().reindex(PREDECLARED_APP_CATEGORIES, fill_value=0)
    present_counts: Counter[str] = Counter()
    for values in frame["categories_present"]:
        for category in values:
            present_counts[category] += 1
    return {
        "usage_rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "payload_row_rate": float(frame["has_payload"].mean()),
        "median_apps_per_row": float(frame["app_count"].median()),
        "p90_apps_per_row": float(frame["app_count"].quantile(0.90)),
        "median_total_time_per_row": float(frame["total_time"].median()),
        "dominant_category_counts": {str(key): int(value) for key, value in category_counts.items()},
        "category_presence_counts": {
            category: int(present_counts.get(category, 0)) for category in PREDECLARED_APP_CATEGORIES
        },
        "nonzero_dominant_categories": int((category_counts > 0).sum()),
        "coverage_by_split": coverage_from_timestamps(frame.rename(columns={"timestamp": "timestamp"})),
    }


def transition_profile(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(["subject_id", "timestamp"]).copy()
    ordered["prev_subject"] = ordered["subject_id"].shift()
    ordered["prev_date"] = ordered["lifelog_date"].shift()
    ordered["prev_category"] = ordered["dominant_category"].shift()
    ordered["prev_timestamp"] = ordered["timestamp"].shift()
    ordered["gap_minutes"] = (ordered["timestamp"] - ordered["prev_timestamp"]).dt.total_seconds() / 60.0
    valid = (
        ordered["has_payload"]
        & ordered["prev_category"].notna()
        & ordered["subject_id"].eq(ordered["prev_subject"])
        & ordered["lifelog_date"].eq(ordered["prev_date"])
        & ordered["gap_minutes"].le(360)
    )
    trans = ordered.loc[valid, ["prev_category", "dominant_category", "hour", "lifelog_date", "subject_id"]].copy()
    trans["pair"] = trans["prev_category"].astype(str) + "→" + trans["dominant_category"].astype(str)
    pair_counts = trans["pair"].value_counts()
    evening = trans[trans["hour"].between(18, 23)]
    all_possible_pairs = len(PREDECLARED_APP_CATEGORIES) ** 2
    return {
        "valid_transition_rows": int(len(trans)),
        "unique_transition_pairs": int(pair_counts.size),
        "all_possible_pairs": int(all_possible_pairs),
        "transition_pair_coverage_rate": float(pair_counts.size / all_possible_pairs),
        "evening_transition_rows": int(len(evening)),
        "evening_unique_transition_pairs": int(evening["pair"].nunique()),
        "top_transition_pairs": {str(key): int(value) for key, value in pair_counts.head(12).items()},
        "self_transition_rate": float((trans["prev_category"] == trans["dominant_category"]).mean())
        if len(trans)
        else 0.0,
        "subject_days_with_transitions": int(trans[["subject_id", "lifelog_date"]].drop_duplicates().shape[0]),
    }


def session_profile(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(["subject_id", "timestamp"]).copy()
    same_subject = ordered["subject_id"].eq(ordered["subject_id"].shift())
    same_date = ordered["lifelog_date"].eq(ordered["lifelog_date"].shift())
    gap_minutes = ordered.groupby("subject_id")["timestamp"].diff().dt.total_seconds().div(60.0)
    new_session = (~same_subject) | (~same_date) | gap_minutes.isna() | gap_minutes.gt(30)
    ordered["session_id"] = new_session.cumsum()
    session = (
        ordered.groupby(["subject_id", "lifelog_date", "session_id"])
        .agg(
            rows=("timestamp", "size"),
            app_events=("app_count", "sum"),
            start_hour=("hour", "min"),
            end_hour=("hour", "max"),
            categories=("dominant_category", "nunique"),
            total_time=("total_time", "sum"),
        )
        .reset_index()
    )
    evening_rows = ordered[ordered["hour"].between(18, 23)].groupby(["subject_id", "lifelog_date"]).size()
    late_rows = ordered[(ordered["hour"] >= 22) | (ordered["hour"] <= 3)].groupby(["subject_id", "lifelog_date"]).size()
    return {
        "sessions": int(len(session)),
        "median_rows_per_session": float(session["rows"].median()) if len(session) else 0.0,
        "p90_rows_per_session": float(session["rows"].quantile(0.90)) if len(session) else 0.0,
        "multi_category_session_rate": float((session["categories"] >= 2).mean()) if len(session) else 0.0,
        "evening_burst_subject_days_ge3_rows": int((evening_rows >= 3).sum()),
        "late_reactivation_subject_days_ge1_rows": int((late_rows >= 1).sum()),
        "max_rows_in_single_session": int(session["rows"].max()) if len(session) else 0,
        "max_app_events_in_single_session": int(session["app_events"].max()) if len(session) else 0,
    }


def screen_profile_and_coupling(usage_frame: pd.DataFrame) -> dict[str, Any]:
    screen = pd.read_parquet(SCREEN_PATH, columns=["subject_id", "timestamp", "m_screen_use"])
    screen["timestamp"] = pd.to_datetime(screen["timestamp"])
    screen["lifelog_date"] = screen["timestamp"].dt.strftime("%Y-%m-%d")
    ordered_screen = screen.sort_values(["subject_id", "timestamp"]).copy()
    changed = ordered_screen.groupby("subject_id")["m_screen_use"].diff().fillna(0).ne(0)
    matches = []
    for subject_id, left in usage_frame[["subject_id", "timestamp"]].sort_values("timestamp").groupby("subject_id"):
        right = screen.loc[screen["subject_id"].eq(subject_id), ["timestamp", "m_screen_use"]].sort_values("timestamp")
        if right.empty:
            continue
        joined = pd.merge_asof(
            left.sort_values("timestamp"),
            right,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=5),
        )
        matches.append(joined)
    coupled = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=["m_screen_use"])
    matched = coupled["m_screen_use"].notna()
    return {
        "screen_rows": int(len(screen)),
        "subjects": int(screen["subject_id"].nunique()),
        "state_counts": {
            str(key): int(value) for key, value in screen["m_screen_use"].value_counts().sort_index().items()
        },
        "state_change_rows": int(changed.sum()),
        "state_change_rate": float(changed.mean()),
        "coverage_by_split": coverage_from_timestamps(screen),
        "usage_rows_with_screen_match_rate_5min": float(matched.mean()) if len(coupled) else 0.0,
        "screen_on_at_usage_rate": float(coupled.loc[matched, "m_screen_use"].astype(float).mean())
        if matched.any()
        else 0.0,
    }


def ambience_schema_profile() -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(AMBIENCE_PATH)
    return {
        "path": rel(AMBIENCE_PATH),
        "exists": AMBIENCE_PATH.exists(),
        "rows": int(parquet_file.metadata.num_rows),
        "columns": list(parquet_file.schema.names),
        "usage_as_guard_only": True,
    }


def build_report() -> dict[str, Any]:
    design = load_json_report(DESIGN_REPORT)
    queue = load_json_report(QUEUE_REPORT)
    design_dec = decision(design)
    queue_dec = decision(queue)
    usage_item = queue_item(queue, SOURCE_FAMILY)
    usage_frame = build_usage_row_frame()
    usage_profile = usage_cardinality_profile(usage_frame)
    transition = transition_profile(usage_frame)
    session = session_profile(usage_frame)
    screen = screen_profile_and_coupling(usage_frame)
    ambience = ambience_schema_profile()

    design_contract_ready = bool(
        design.get("parse_ok") is True
        and design_dec.get("design_ready") is True
        and design_dec.get("source_family") == SOURCE_FAMILY
        and design_dec.get("target_focus") == TARGET_FOCUS
        and design_dec.get("feature_matrix_allowed_now") is False
        and design_dec.get("train_model_now") is False
        and design_dec.get("create_submission_artifact") is False
        and design_dec.get("submit_now") is False
        and design_dec.get("next_local_allowed_step")
        == "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features"
    )
    current_queue_contract_ready = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and usage_item.get("name") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type") in ALLOWED_QUEUE_STEP_TYPES
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
    )
    usage_lane_completed_no_model = bool(queue.get("parse_ok") is True and queue_lane_completed_no_model(queue))
    queue_contract_ready = bool(current_queue_contract_ready or usage_lane_completed_no_model)
    category_state_space_ready = bool(
        usage_profile["nonzero_dominant_categories"] >= 5
        and transition["unique_transition_pairs"] >= 20
        and transition["transition_pair_coverage_rate"] >= 0.40
    )
    session_state_ready = bool(
        session["sessions"] >= 1000
        and session["multi_category_session_rate"] >= 0.20
        and session["evening_burst_subject_days_ge3_rows"] >= 150
    )
    screen_coupling_ready = bool(
        screen["coverage_by_split"]["train"]["coverage_rate"] >= 0.99
        and screen["coverage_by_split"]["test"]["coverage_rate"] >= 0.99
        and screen["usage_rows_with_screen_match_rate_5min"] >= 0.95
    )
    metadata_smoke_passed = bool(
        design_contract_ready
        and queue_contract_ready
        and usage_profile["coverage_by_split"]["train"]["coverage_rate"] >= 0.95
        and usage_profile["coverage_by_split"]["test"]["coverage_rate"] >= 0.95
        and category_state_space_ready
        and session_state_ready
        and screen_coupling_ready
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_usage_semantic_burst_markov_metadata_smoke",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Check whether the usage semantic burst/Markov design has enough raw state coverage and cardinality for a "
            "future report-only feature spec. This is not feature generation or model evidence."
        ),
        "inputs": {"design_report": design, "material_new_source_queue": queue},
        "usage_cardinality_profile": usage_profile,
        "transition_profile": transition,
        "session_profile": session,
        "screen_coupling_profile": screen,
        "ambience_schema_profile": ambience,
        "checks": {
            "design_contract_ready": design_contract_ready,
            "queue_contract_ready": queue_contract_ready,
            "current_queue_contract_ready": current_queue_contract_ready,
            "usage_lane_completed_no_model": usage_lane_completed_no_model,
            "usage_train_test_coverage_ready": usage_profile["coverage_by_split"]["train"]["coverage_rate"] >= 0.95
            and usage_profile["coverage_by_split"]["test"]["coverage_rate"] >= 0.95,
            "category_state_space_ready": category_state_space_ready,
            "session_state_ready": session_state_ready,
            "screen_coupling_ready": screen_coupling_ready,
            "ambience_guard_only": ambience["usage_as_guard_only"] is True,
            "does_not_create_feature_matrix": True,
            "does_not_train_model": True,
            "does_not_create_submission_artifact": True,
        },
        "decision": {
            "metadata_smoke_passed": metadata_smoke_passed,
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
            "next_local_allowed_step": NEXT_ALLOWED_STEP
            if metadata_smoke_passed
            else "fix_usage_semantic_burst_markov_metadata_smoke_inputs",
            "next_action": (
                "A future no-approval step may only write a report-only feature spec with exact predeclared states and "
                "leakage guards. Do not generate a feature matrix, train a model, create ID342, run Kaggle, or submit."
                if metadata_smoke_passed
                else "Do not advance usage semantic burst/Markov work until metadata/cardinality blockers are fixed."
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
    usage = report["usage_cardinality_profile"]
    trans = report["transition_profile"]
    session = report["session_profile"]
    screen = report["screen_coupling_profile"]
    lines = [
        "# Post-ID341 usage semantic burst/Markov metadata smoke",
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
        f"- Metadata smoke passed: `{dec['metadata_smoke_passed']}`",
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
        "## Coverage/cardinality",
        "",
        f"- Usage train/test coverage: `{usage['coverage_by_split']['train']['coverage_rate']:.4f}` / "
        f"`{usage['coverage_by_split']['test']['coverage_rate']:.4f}`",
        f"- Dominant app-category states used: `{usage['nonzero_dominant_categories']}` / "
        f"`{len(PREDECLARED_APP_CATEGORIES)}`",
        f"- Valid Markov transitions / unique pairs / pair coverage: `{trans['valid_transition_rows']}` / "
        f"`{trans['unique_transition_pairs']}` / `{trans['transition_pair_coverage_rate']:.4f}`",
        f"- Evening transitions / evening unique pairs: `{trans['evening_transition_rows']}` / "
        f"`{trans['evening_unique_transition_pairs']}`",
        f"- Sessions / multi-category session rate: `{session['sessions']}` / "
        f"`{session['multi_category_session_rate']:.4f}`",
        f"- Evening burst days / late reactivation days: `{session['evening_burst_subject_days_ge3_rows']}` / "
        f"`{session['late_reactivation_subject_days_ge1_rows']}`",
        f"- Screen train/test coverage: `{screen['coverage_by_split']['train']['coverage_rate']:.4f}` / "
        f"`{screen['coverage_by_split']['test']['coverage_rate']:.4f}`",
        f"- Usage rows with screen match within 5min: `{screen['usage_rows_with_screen_match_rate_5min']:.4f}`",
        "",
        "## Top transition pairs",
        "",
    ]
    for pair, count in trans["top_transition_pairs"].items():
        lines.append(f"- `{pair}`: `{count}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "Passing this metadata smoke does not create ID342 and does not authorize feature generation. It only "
                "allows a later report-only feature spec with exact predeclared states, leakage guards, and no model."
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
