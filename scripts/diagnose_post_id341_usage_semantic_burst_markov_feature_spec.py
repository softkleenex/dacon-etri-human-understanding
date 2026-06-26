#!/usr/bin/env python3
"""Report-only feature spec for usage semantic burst/Markov v1.

This is the step after the usage semantic burst/Markov metadata/cardinality
smoke. It freezes the exact label-free feature families, predeclared app-state
space, leakage guards, and future dry-run boundary. It creates no feature
matrix, trains no model, creates no predictions or submission artifacts,
modifies no submission log, runs no Kaggle command, and submits nothing to
DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_usage_semantic_burst_markov_design import (  # noqa: E402
    PREDECLARED_APP_CATEGORIES,
    SOURCE_FAMILY,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_feature_spec_20260619.json"
DESIGN_REPORT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_design_20260619.json"
METADATA_REPORT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"

TARGET_FOCUS = ["Q2", "Q3"]
NEXT_ALLOWED_STEP = "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix"
CURRENT_ALLOWED_QUEUE_STEP = "report_only_usage_semantic_burst_markov_feature_spec_no_matrix"
ALLOWED_QUEUE_STEP_TYPES = {
    "report_only_usage_semantic_burst_markov_design_no_model",
    "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features",
    CURRENT_ALLOWED_QUEUE_STEP,
    NEXT_ALLOWED_STEP,
    "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract",
    "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
    "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
}
FEATURE_COLUMN_BOUND = 96
KEY_COLUMNS = ["subject_id", "lifelog_date", "split"]
TARGET_COLUMNS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FORBIDDEN_FEATURE_TOKENS = [
    "app_name",
    "raw_app",
    "target",
    "label",
    "public_lb",
    "submission",
    "prediction",
    "oof",
]
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


def fixed_category_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{category}" for category in PREDECLARED_APP_CATEGORIES]


def feature_groups() -> list[dict[str, Any]]:
    return [
        {
            "name": "coverage_quality_flags",
            "sources": ["mUsageStats", "mScreenStatus", "mAmbience"],
            "columns": [
                "usage_row_count",
                "usage_payload_row_rate",
                "usage_missing_flag",
                "screen_match_rate_5min",
                "screen_missing_flag",
                "ambience_match_rate_10min",
                "source_coverage_score",
            ],
            "leakage_guard": "Only source-presence and match rates over the current lifelog day; no target labels or predictions.",
        },
        {
            "name": "app_burst_session_shape",
            "sources": ["mUsageStats"],
            "columns": [
                "usage_rows_total",
                "usage_rows_evening",
                "usage_rows_late",
                "payload_event_count_total",
                "usage_total_time_sum",
                "usage_total_time_evening_sum",
                "usage_total_time_late_sum",
                "session_count",
                "session_rows_median",
                "session_rows_p90",
                "max_session_rows",
                "multi_category_session_count",
                "evening_burst_rows_ge3",
                "late_reactivation_rows_ge1",
            ],
            "window_definition": {
                "session_gap_minutes": 30,
                "evening_hours": "18<=hour<=23",
                "late_reactivation_hours": "hour>=22 or hour<=3",
            },
            "leakage_guard": "Sessions are cut by subject and lifelog_date; rolling variants may use only prior lifelog_date rows.",
        },
        {
            "name": "semantic_category_distribution",
            "sources": ["mUsageStats"],
            "columns": [
                *fixed_category_columns("category_time_share"),
                *fixed_category_columns("category_row_presence_count"),
                *fixed_category_columns("category_dominant_count"),
            ],
            "predeclared_state_space": PREDECLARED_APP_CATEGORIES,
            "leakage_guard": "Raw app names are mapped only through the fixed hand-written category function; no learned app vocabulary.",
        },
        {
            "name": "markov_transition_shape",
            "sources": ["mUsageStats"],
            "columns": [
                "transition_count",
                "unique_transition_pair_count",
                "transition_entropy",
                "evening_transition_count",
                "evening_unique_transition_pair_count",
                "self_transition_rate",
                "top_pair_concentration",
                "rare_pair_count",
                "category_switch_rate",
                "markov_surprise_vs_subject_baseline",
            ],
            "predeclared_pair_space_size": len(PREDECLARED_APP_CATEGORIES) ** 2,
            "leakage_guard": "Transition pairs are over fixed categories only; subject baselines are fitted inside each outer fold.",
        },
        {
            "name": "screen_coupled_bursts",
            "sources": ["mUsageStats", "mScreenStatus"],
            "columns": [
                "screen_on_at_usage_rate",
                "screen_state_change_count_near_usage",
                "screen_on_burst_count",
                "screen_off_burst_count",
                "late_screen_on_usage_rate",
                "evening_screen_on_usage_rate",
                "usage_after_screen_on_count",
                "usage_before_screen_off_count",
                "screen_coupling_missing_rate",
                "screen_reactivation_after_idle_count",
            ],
            "join_rule": "Nearest screen-status event within 5 minutes; unmatched rows get explicit missing flags.",
            "leakage_guard": "Screen coupling is computed from raw timestamps only and must not condition on target or public score.",
        },
        {
            "name": "ambience_context_guards",
            "sources": ["mUsageStats", "mAmbience"],
            "columns": [
                "ambience_events_near_burst_count",
                "ambience_unique_element_count_near_burst",
                "quiet_high_switching_flag",
                "noisy_late_reactivation_flag",
                "ambience_missing_rate_near_usage",
                "ambience_evening_context_count",
                "ambience_late_context_count",
                "ambience_guard_conflict_score",
            ],
            "join_rule": "Nearest ambience elements within 10 minutes; ambience stays guard-only because its standalone lane closed.",
            "leakage_guard": "Ambience features may gate burst context but cannot reopen a standalone ambience model lane.",
        },
        {
            "name": "subject_relative_rolling_baseline",
            "sources": ["mUsageStats", "mScreenStatus"],
            "columns": [
                "rolling_7d_burst_count_z",
                "rolling_7d_late_reactivation_z",
                "rolling_7d_transition_entropy_z",
                "rolling_7d_screen_on_rate_z",
                "subject_history_day_count",
                "subject_history_missing_flag",
                "subject_relative_usage_intensity_rank",
            ],
            "window_definition": {
                "history": "prior lifelog_date rows only",
                "min_history_days": 2,
                "max_history_days": 7,
            },
            "leakage_guard": "In validation, rolling/baseline parameters are learned from the outer-train fold only; never from validation labels.",
        },
    ]


def all_feature_columns(groups: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for group in groups:
        columns.extend(group["columns"])
    return columns


def feature_columns_are_safe(columns: list[str]) -> bool:
    lowered = [column.lower() for column in columns]
    return all(not any(token in column for token in FORBIDDEN_FEATURE_TOKENS) for column in lowered)


def build_report() -> dict[str, Any]:
    design = load_json_report(DESIGN_REPORT)
    metadata = load_json_report(METADATA_REPORT)
    queue = load_json_report(QUEUE_REPORT)
    design_dec = decision(design)
    metadata_dec = decision(metadata)
    queue_dec = decision(queue)
    groups = feature_groups()
    columns = all_feature_columns(groups)
    usage_profile = (
        metadata.get("usage_cardinality_profile", {})
        if isinstance(metadata.get("usage_cardinality_profile"), dict)
        else {}
    )
    transition_profile = (
        metadata.get("transition_profile", {}) if isinstance(metadata.get("transition_profile"), dict) else {}
    )
    session_profile = metadata.get("session_profile", {}) if isinstance(metadata.get("session_profile"), dict) else {}
    screen_profile = (
        metadata.get("screen_coupling_profile", {}) if isinstance(metadata.get("screen_coupling_profile"), dict) else {}
    )

    design_contract_ready = bool(
        design.get("parse_ok") is True
        and design_dec.get("design_ready") is True
        and design_dec.get("source_family") == SOURCE_FAMILY
        and design_dec.get("target_focus") == TARGET_FOCUS
        and design_dec.get("feature_matrix_allowed_now") is False
        and design_dec.get("create_submission_artifact") is False
        and design_dec.get("submit_now") is False
        and no_forbidden_actions(design)
    )
    metadata_contract_ready = bool(
        metadata.get("parse_ok") is True
        and metadata_dec.get("metadata_smoke_passed") is True
        and metadata_dec.get("source_family") == SOURCE_FAMILY
        and metadata_dec.get("target_focus") == TARGET_FOCUS
        and metadata_dec.get("next_local_allowed_step") == CURRENT_ALLOWED_QUEUE_STEP
        and metadata_dec.get("feature_matrix_allowed_now") is False
        and metadata_dec.get("train_model_now") is False
        and metadata_dec.get("create_submission_artifact") is False
        and metadata_dec.get("submit_now") is False
        and no_forbidden_actions(metadata)
    )
    current_queue_contract_ready = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type") in ALLOWED_QUEUE_STEP_TYPES
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
        and no_forbidden_actions(queue)
    )
    usage_lane_completed_no_model = bool(queue.get("parse_ok") is True and queue_lane_completed_no_model(queue))
    queue_contract_ready = bool(current_queue_contract_ready or usage_lane_completed_no_model)
    metadata_state_space_ready = bool(
        usage_profile.get("nonzero_dominant_categories") == len(PREDECLARED_APP_CATEGORIES)
        and transition_profile.get("all_possible_pairs") == len(PREDECLARED_APP_CATEGORIES) ** 2
        and transition_profile.get("unique_transition_pairs", 0) >= 40
        and session_profile.get("sessions", 0) >= 1000
        and screen_profile.get("usage_rows_with_screen_match_rate_5min", 0.0) >= 0.95
    )
    feature_schema_bounded = bool(
        len(columns) == len(set(columns))
        and len(columns) <= FEATURE_COLUMN_BOUND
        and feature_columns_are_safe(columns)
        and set(PREDECLARED_APP_CATEGORIES) == set(PREDECLARED_APP_CATEGORIES)
    )
    feature_spec_ready = bool(
        design_contract_ready
        and metadata_contract_ready
        and queue_contract_ready
        and metadata_state_space_ready
        and feature_schema_bounded
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_usage_semantic_burst_markov_feature_spec",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Freeze an exact bounded label-free feature schema for usage semantic burst/Markov v1 after the metadata "
            "smoke passed. This is a contract/specification only, not feature generation or model evidence."
        ),
        "inputs": {"design_report": design, "metadata_smoke_report": metadata, "material_new_source_queue": queue},
        "row_contract": {
            "expected_row_count": 700,
            "expected_train_rows": 450,
            "expected_test_rows": 250,
            "key_columns": KEY_COLUMNS,
            "target_columns_forbidden_in_features": TARGET_COLUMNS,
            "submission_columns_forbidden_in_features": ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"],
        },
        "state_space_contract": {
            "predeclared_app_categories": PREDECLARED_APP_CATEGORIES,
            "predeclared_category_count": len(PREDECLARED_APP_CATEGORIES),
            "predeclared_pair_space_size": len(PREDECLARED_APP_CATEGORIES) ** 2,
            "raw_app_name_features_allowed": False,
            "target_guided_vocabulary_allowed": False,
        },
        "feature_schema": {
            "feature_column_bound": FEATURE_COLUMN_BOUND,
            "feature_column_count": len(columns),
            "feature_group_count": len(groups),
            "feature_groups": groups,
            "feature_columns": columns,
        },
        "validation_and_leakage_contract": {
            "outer_split": "existing subject-hole/fold-internal evaluation only; do not pick candidates on full OOF rows",
            "preprocessing_fit_scope": "fit any scaling/imputation/baseline statistics inside each outer-train fold only",
            "rolling_history_scope": "subject-relative rolling features use only prior lifelog_date rows within the available raw stream",
            "test_transform_scope": "full-train feature parameters may be reused only after the dry-run/model contract is passed",
            "forbidden_inputs": [
                "target labels",
                "OOF/test predictions",
                "public leaderboard score",
                "raw app-name one-hot",
                "learned app vocabulary",
                "same-row sibling target labels",
            ],
            "missingness_policy": "preserve missingness with explicit flags; do not drop expected train/test keys",
        },
        "future_dryrun_contract": {
            "next_local_allowed_step": NEXT_ALLOWED_STEP,
            "dryrun_may_materialize_in_memory": True,
            "feature_parquet_allowed": False,
            "model_training_allowed": False,
            "predictions_allowed": False,
            "submission_artifact_allowed": False,
            "kaggle_or_dacon_allowed": False,
        },
        "checks": {
            "design_contract_ready": design_contract_ready,
            "metadata_contract_ready": metadata_contract_ready,
            "queue_contract_ready": queue_contract_ready,
            "current_queue_contract_ready": current_queue_contract_ready,
            "usage_lane_completed_no_model": usage_lane_completed_no_model,
            "metadata_state_space_ready": metadata_state_space_ready,
            "feature_schema_bounded": feature_schema_bounded,
            "uses_predeclared_app_categories_only": True,
            "raw_app_name_features_allowed": False,
            "does_not_create_feature_matrix": True,
            "does_not_train_model": True,
            "does_not_create_submission_artifact": True,
        },
        "decision": {
            "feature_spec_ready": feature_spec_ready,
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
            if feature_spec_ready
            else "fix_usage_semantic_feature_spec_inputs",
            "next_action": (
                "A future no-approval step may only run a report-only feature-generation dry run in memory. Do not "
                "write a feature matrix/parquet, train a model, create ID342, run Kaggle, or submit."
                if feature_spec_ready
                else "Do not advance usage semantic burst/Markov work until feature-spec contract blockers are fixed."
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
    schema = report["feature_schema"]
    state = report["state_space_contract"]
    lines = [
        "# Post-ID341 usage semantic burst/Markov feature spec",
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
        f"- Feature spec ready: `{dec['feature_spec_ready']}`",
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
        "## State space and feature bound",
        "",
        f"- Predeclared app categories: `{state['predeclared_app_categories']}`",
        f"- Pair-space size: `{state['predeclared_pair_space_size']}`",
        f"- Raw app-name features allowed: `{state['raw_app_name_features_allowed']}`",
        f"- Target-guided vocabulary allowed: `{state['target_guided_vocabulary_allowed']}`",
        f"- Feature groups / columns / bound: `{schema['feature_group_count']}` / "
        f"`{schema['feature_column_count']}` / `{schema['feature_column_bound']}`",
        "",
        "## Feature groups",
        "",
    ]
    for group in schema["feature_groups"]:
        lines.extend(
            [
                f"### `{group['name']}`",
                "",
                f"- Sources: `{group['sources']}`",
                f"- Column count: `{len(group['columns'])}`",
                f"- Leakage guard: {group['leakage_guard']}",
                "- Columns:",
            ]
        )
        lines.extend(f"  - `{column}`" for column in group["columns"])
        lines.append("")
    lines.extend(
        [
            "## Validation/leakage contract",
            "",
            f"- Outer split: {report['validation_and_leakage_contract']['outer_split']}",
            f"- Fit scope: {report['validation_and_leakage_contract']['preprocessing_fit_scope']}",
            f"- Rolling scope: {report['validation_and_leakage_contract']['rolling_history_scope']}",
            f"- Missingness policy: {report['validation_and_leakage_contract']['missingness_policy']}",
            "- Forbidden inputs:",
        ]
    )
    lines.extend(f"  - {item}" for item in report["validation_and_leakage_contract"]["forbidden_inputs"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "Passing this feature spec does not create ID342 and does not authorize feature generation beyond a "
                "future report-only in-memory dry run. No feature matrix, model, submission CSV, submission-log row, "
                "Kaggle run, DACON submission, or Public `0.58` claim is authorized."
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
