#!/usr/bin/env python3
"""Report-only feature spec for phone/wearable discordance v1.

The preceding design report proved that phone rest state and wearable
activation state can be joined on the 700 competition rows with observable
discordance. This script freezes a bounded, label-free feature schema only. It
does not write a feature matrix, train a model, create predictions or
submission artifacts, edit the submission log, run Kaggle, or submit to DACON.
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

from scripts.diagnose_post_id341_phone_wearable_discordance_design import (  # noqa: E402
    FEATURE_GROUPS,
    SOURCE_FAMILY,
    TARGET_FOCUS,
    WINDOWS,
    decision,
    read_json_report,
    rel,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_phone_wearable_discordance_feature_spec_20260619.json"
DESIGN_REPORT_NAME = "post_id341_phone_wearable_discordance_design_20260619.json"
QUEUE_REPORT_NAME = "post_id341_material_new_source_queue_20260619.json"

KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]
CURRENT_ALLOWED_STEP = "report_only_phone_wearable_discordance_feature_spec_no_matrix"
NEXT_ALLOWED_STEP = "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix"
POST_DRYRUN_ALLOWED_STEP = "report_only_phone_wearable_discordance_feature_matrix_smoke_contract"
POST_CONTRACT_ALLOWED_STEP = "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission"
POST_VALIDATION_ALLOWED_STEP = (
    "validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
)
LANE_CLOSED_ALLOWED_STEP = "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval"
FEATURE_PREFIX = "pwd_"
FEATURE_COLUMN_BOUND = 96
GLOBAL_FEATURES = [
    "day_source_coverage_index",
    "phone_rest_total_hour_bins",
    "wearable_active_total_hour_bins",
    "discordance_total_hour_bins",
    "discordance_to_phone_rest_ratio",
    "late_discordance_share",
    "first_discordance_hour",
    "last_discordance_hour",
    "max_contiguous_discordance_hours",
    "source_missingness_guard_count",
]
FORBIDDEN_FEATURE_TOKENS = [
    "target",
    "label",
    "public_lb",
    "submission",
    "prediction",
    "oof",
    "score",
]
FORBIDDEN_DECISION_KEYS = [
    "feature_matrix_allowed_now",
    "train_model_now",
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
]


def load_required_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    return read_json_report(DESIGN_REPORT_NAME), read_json_report(QUEUE_REPORT_NAME)


def window_feature_names() -> list[str]:
    names: list[str] = []
    for window_name in WINDOWS:
        for group_name, metrics in FEATURE_GROUPS.items():
            for metric in metrics:
                names.append(f"{FEATURE_PREFIX}{window_name}_{group_name}_{metric}")
    return names


def feature_names() -> list[str]:
    return window_feature_names() + [f"{FEATURE_PREFIX}global_{name}" for name in GLOBAL_FEATURES]


def no_forbidden_decision_actions(dec: dict[str, Any]) -> bool:
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def queue_allows_feature_spec_or_next(queue_decision: dict[str, Any]) -> bool:
    if not queue_decision:
        return True
    return bool(
        queue_decision.get("next_no_approval_step") in {None, SOURCE_FAMILY, "none_local_no_approval_lanes_closed"}
        and queue_decision.get("next_no_approval_step_type")
        in {
            None,
            CURRENT_ALLOWED_STEP,
            NEXT_ALLOWED_STEP,
            POST_DRYRUN_ALLOWED_STEP,
            POST_CONTRACT_ALLOWED_STEP,
            POST_VALIDATION_ALLOWED_STEP,
            LANE_CLOSED_ALLOWED_STEP,
        }
        and queue_decision.get("only_score_relevant_open_route")
        in {None, "explicit_approved_clean_upstream_id221_to_id226"}
        and queue_decision.get("create_submission_artifact") is not True
        and queue_decision.get("submit_now") is not True
        and queue_decision.get("push_kaggle_now") is not True
        and queue_decision.get("run_kaggle_now") is not True
        and queue_decision.get("train_model_now") is not True
    )


def build_report() -> dict[str, Any]:
    design, queue = load_required_reports()
    design_decision = decision(design)
    queue_decision = decision(queue)
    names = feature_names()
    duplicate_features = sorted({name for name in names if names.count(name) > 1})
    forbidden_feature_hits = [name for name in names for token in FORBIDDEN_FEATURE_TOKENS if token in name.lower()]
    schema_bounds = {
        "feature_prefix": FEATURE_PREFIX,
        "max_feature_columns": FEATURE_COLUMN_BOUND,
        "actual_feature_columns": len(names),
        "all_features_prefixed": all(name.startswith(FEATURE_PREFIX) for name in names),
        "duplicate_feature_count": len(duplicate_features),
        "forbidden_feature_token_hits": sorted(set(forbidden_feature_hits)),
        "raw_identifier_columns_allowed": False,
        "raw_timestamp_columns_allowed": False,
        "fold_internal_preprocessing_required_for_any_model": True,
    }
    feature_spec_ready = bool(
        design.get("exists")
        and design.get("parse_ok")
        and design_decision.get("design_ready") is True
        and design_decision.get("source_family") == SOURCE_FAMILY
        and design_decision.get("target_focus") == TARGET_FOCUS
        and design_decision.get("coverage_ready") is True
        and design_decision.get("discordance_observed") is True
        and design_decision.get("next_local_allowed_step") == CURRENT_ALLOWED_STEP
        and no_forbidden_decision_actions(design_decision)
        and queue_allows_feature_spec_or_next(queue_decision)
        and len(names) <= FEATURE_COLUMN_BOUND
        and schema_bounds["all_features_prefixed"]
        and schema_bounds["duplicate_feature_count"] == 0
        and not schema_bounds["forbidden_feature_token_hits"]
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_phone_wearable_discordance_feature_spec",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "input_reports": {
            "design": {
                "path": rel(REPORTS_DIR / DESIGN_REPORT_NAME),
                "exists": design.get("exists"),
                "parse_ok": design.get("parse_ok"),
            },
            "material_new_source_queue": {
                "path": rel(REPORTS_DIR / QUEUE_REPORT_NAME),
                "exists": queue.get("exists"),
                "parse_ok": queue.get("parse_ok", queue.get("exists") and not queue.get("parse_error")),
            },
        },
        "row_contract": {
            "key_columns": KEY_COLUMNS,
            "expected_train_rows": 450,
            "expected_test_rows": 250,
            "expected_total_rows": 700,
            "label_columns_allowed": False,
            "prediction_columns_allowed": False,
            "submission_columns_allowed": False,
            "raw_timestamp_columns_allowed": False,
        },
        "windows": WINDOWS,
        "feature_groups": FEATURE_GROUPS,
        "global_features": GLOBAL_FEATURES,
        "feature_names": names,
        "feature_count": len(names),
        "schema_bounds": schema_bounds,
        "aggregation_contract": {
            "join_grain": "subject_id + lifelog_date + fixed day window",
            "hour_grid": "24 fixed hourly bins per row; windows are deterministic and predeclared.",
            "phone_rest_definition": "screen-off, charging, or still-phone state from raw phone sensors only.",
            "wearable_activation_definition": "positive steps, observed heart-rate, or wearable light from raw wearable sensors only.",
            "discordance_definition": "fixed cross-device disagreement masks; no threshold fitted to labels.",
            "missing_value_policy": "write explicit source coverage/missingness guards; later imputation must be fold-internal.",
        },
        "leakage_guards": [
            "Build future features only from each row's lifelog_date raw sensor timestamps.",
            "Do not use target labels, public LB scores, OOF predictions, submission files, or base prediction artifacts.",
            "Keep phone/wearable thresholds fixed from the design report; do not tune them on validation residuals.",
            "Any later feature-generation dry run may stay in memory only unless a separate matrix contract is written.",
            "Any future model smoke must fit preprocessing, imputation, selection, and calibration inside each outer fold.",
            "This spec cannot create ID342, a notebook-history artifact, a base prediction artifact, or a submission-log row.",
        ],
        "decision": {
            "feature_spec_ready": feature_spec_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "feature_count": len(names),
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": NEXT_ALLOWED_STEP,
            "next_action": (
                "Feature spec is ready. A future no-approval step may only run an in-memory feature-generation "
                "dry run that checks 700-row alignment and bounded finite columns; do not write a feature matrix, "
                "train a model, create ID342, run Kaggle, or submit."
                if feature_spec_ready
                else "Fix design/queue/schema prerequisites before any next step."
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
    dec = report["decision"]
    lines = [
        "# Post-ID341 phone/wearable discordance feature spec",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Feature spec ready: `{dec['feature_spec_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Feature count: `{dec['feature_count']}`",
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
        "## Row contract",
        "",
    ]
    for key, value in report["row_contract"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Feature groups", ""])
    for group, metrics in report["feature_groups"].items():
        lines.append(f"- `{group}`: `{', '.join(metrics)}`")
    lines.extend(["", "## Global features", ""])
    for name in report["global_features"]:
        lines.append(f"- `{FEATURE_PREFIX}global_{name}`")
    lines.extend(["", "## Schema bounds", ""])
    for key, value in report["schema_bounds"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Aggregation contract", ""])
    for key, value in report["aggregation_contract"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Leakage guards", ""])
    for guard in report["leakage_guards"]:
        lines.append(f"- {guard}")
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
    return 0 if report["decision"]["feature_spec_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
