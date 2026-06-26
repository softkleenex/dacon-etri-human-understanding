#!/usr/bin/env python3
"""Report-only feature-matrix smoke contract for proximity/ambience v1.

This gate follows the in-memory feature-generation dry run. It decides whether a
future writer may materialize exactly one registered label-free feature matrix
path plus schema report, then stop. This script itself writes only JSON/Markdown
reports. It does not write the matrix/parquet, train a model, create predictions
or submission artifacts, modify the submission log, run Kaggle, or submit to
DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "proximity_ambience_cooccurrence_v1"
TARGET_FOCUS = ["Q3", "S3"]
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract_20260619.json"
FEATURE_SPEC_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_spec_20260619.json"
DRYRUN_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
REGISTERED_MATRIX_PATH = ROOT / "data" / "processed" / "post_id341_proximity_ambience_cooccurrence_features_v1.parquet"
REGISTERED_SCHEMA_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_features_v1_schema.json"
CURRENT_QUEUE_STEP = "report_only_proximity_ambience_cooccurrence_feature_matrix_smoke_contract"
NEXT_ALLOWED_STEP = "write_proximity_ambience_cooccurrence_feature_matrix_under_contract_no_model_no_submission"
ALLOWED_QUEUE_STEPS = {
    "report_only_proximity_ambience_cooccurrence_feature_generation_dryrun_no_matrix",
    CURRENT_QUEUE_STEP,
    NEXT_ALLOWED_STEP,
    "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
}
FORBIDDEN_DECISION_KEYS = (
    "train_model_now",
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
)
FORBIDDEN_NAME_TOKENS = ("bssid", "address", "raw_token", "identifier_value")


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
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "diagnostic": payload.get("diagnostic"),
        "decision": payload.get("decision", {}),
        "feature_frame_summary": payload.get("feature_frame_summary", {}),
        "feature_count": payload.get("feature_count"),
        "error": None,
    }


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def no_forbidden_actions(dec: dict[str, Any]) -> bool:
    return all(dec.get(key) is False for key in FORBIDDEN_DECISION_KEYS)


def queue_item(queue_payload: dict[str, Any], name: str) -> dict[str, Any]:
    items = queue_payload.get("source_queue", []) if isinstance(queue_payload.get("source_queue"), list) else []
    return next((item for item in items if item.get("name") == name), {})


def load_queue_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "source_item": {}, "error": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "source_item": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "source_item": queue_item(payload, SOURCE_FAMILY),
        "error": None,
    }


def build_report() -> dict[str, Any]:
    spec = load_json_report(FEATURE_SPEC_REPORT)
    dryrun = load_json_report(DRYRUN_REPORT)
    queue = load_queue_report(QUEUE_REPORT)
    spec_dec = decision(spec)
    dryrun_dec = decision(dryrun)
    queue_dec = decision(queue)
    source_item = queue.get("source_item", {})
    dry_summary = dryrun.get("feature_frame_summary", {})
    registered_matrix_exists = REGISTERED_MATRIX_PATH.exists()
    registered_schema_exists = REGISTERED_SCHEMA_REPORT.exists()

    spec_contract_ready = bool(
        spec.get("parse_ok") is True
        and spec_dec.get("feature_spec_ready") is True
        and spec_dec.get("source_family") == SOURCE_FAMILY
        and spec_dec.get("target_focus") == TARGET_FOCUS
        and spec_dec.get("feature_count") == 80
        and spec_dec.get("public_058_evidence") is False
        and spec_dec.get("score_route_opened") is False
        and spec_dec.get("feature_matrix_allowed_now") is False
        and no_forbidden_actions(spec_dec)
    )
    dryrun_contract_ready = bool(
        dryrun.get("parse_ok") is True
        and dryrun_dec.get("feature_generation_dryrun_passed") is True
        and dryrun_dec.get("source_family") == SOURCE_FAMILY
        and dryrun_dec.get("target_focus") == TARGET_FOCUS
        and dryrun_dec.get("feature_count") == 80
        and dryrun_dec.get("next_local_allowed_step") == CURRENT_QUEUE_STEP
        and dryrun_dec.get("public_058_evidence") is False
        and dryrun_dec.get("score_route_opened") is False
        and dryrun_dec.get("feature_matrix_allowed_now") is False
        and no_forbidden_actions(dryrun_dec)
    )
    dryrun_schema_ready = bool(
        dry_summary.get("row_count") == 700
        and dry_summary.get("train_rows") == 450
        and dry_summary.get("test_rows") == 250
        and dry_summary.get("feature_column_count") == 80
        and dry_summary.get("feature_columns_match_spec") is True
        and dry_summary.get("duplicate_key_rows") == 0
        and dry_summary.get("finite_all_values") is True
        and dry_summary.get("nan_count") == 0
        and dry_summary.get("nonzero_feature_count", 0) >= 60
        and dry_summary.get("zero_variance_feature_count", 99) <= 20
    )
    current_queue_contract_ready = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type") in ALLOWED_QUEUE_STEPS
        and source_item.get("name") == SOURCE_FAMILY
        and source_item.get("status") == "feature_generation_dryrun_passed_contract_only"
        and source_item.get("next_allowed_step") == CURRENT_QUEUE_STEP
        and queue_dec.get("public_058_evidence") is False
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
        and no_forbidden_actions(queue_dec)
    )
    queue_records_contract_ready = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type") == NEXT_ALLOWED_STEP
        and source_item.get("name") == SOURCE_FAMILY
        and source_item.get("status") == "feature_matrix_smoke_contract_ready_write_only_under_contract"
        and source_item.get("next_allowed_step") == NEXT_ALLOWED_STEP
        and queue_dec.get("public_058_evidence") is False
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
        and no_forbidden_actions(queue_dec)
    )
    queue_records_matrix_materialized = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type")
        == "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
        and source_item.get("name") == SOURCE_FAMILY
        and source_item.get("status") == "feature_matrix_materialized_under_contract_no_model"
        and source_item.get("next_allowed_step")
        == "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
        and queue_dec.get("public_058_evidence") is False
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
        and no_forbidden_actions(queue_dec)
    )
    queue_records_lane_closed = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step")
        in {"none_local_no_approval_lanes_closed", "phone_wearable_discordance_v1"}
        and queue_dec.get("next_no_approval_step_type")
        in {
            "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval",
            "report_only_phone_wearable_discordance_feature_spec_no_matrix",
            "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix",
            "report_only_phone_wearable_discordance_feature_matrix_smoke_contract",
            "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission",
        }
        and source_item.get("name") == SOURCE_FAMILY
        and source_item.get("status") == "validated_stop_completed_no_model"
        and source_item.get("next_allowed_step") == "closed_stop_or_wait_for_explicit_kaggle_approval"
        and source_item.get("local_no_approval_allowed") is False
        and queue_dec.get("public_058_evidence") is False
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
        and no_forbidden_actions(queue_dec)
    )
    queue_contract_ready = bool(
        current_queue_contract_ready
        or queue_records_contract_ready
        or queue_records_matrix_materialized
        or queue_records_lane_closed
    )
    privacy_contract_ready = bool(
        not any(token in str(FEATURE_SPEC_REPORT).lower() for token in FORBIDDEN_NAME_TOKENS)
        and not any(token in str(DRYRUN_REPORT).lower() for token in FORBIDDEN_NAME_TOKENS)
        and dryrun_schema_ready
    )
    matrix_write_allowed_next = bool(
        spec_contract_ready
        and dryrun_contract_ready
        and dryrun_schema_ready
        and queue_contract_ready
        and privacy_contract_ready
        and not registered_matrix_exists
        and not registered_schema_exists
    )
    matrix_already_materialized_under_contract = bool(registered_matrix_exists and registered_schema_exists)
    contract_ready = bool(matrix_write_allowed_next or matrix_already_materialized_under_contract)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Gate whether the proximity/ambience co-occurrence v1 dry-run schema may be materialized later as "
            "exactly one registered label-free feature matrix, with no model or submission side effects."
        ),
        "inputs": {
            "feature_spec_report": {key: value for key, value in spec.items() if key != "decision"},
            "feature_generation_dryrun_report": {key: value for key, value in dryrun.items() if key != "decision"},
            "material_new_source_queue": {key: value for key, value in queue.items() if key != "decision"},
        },
        "registered_paths": {
            "feature_matrix_path": rel(REGISTERED_MATRIX_PATH),
            "schema_report_path": rel(REGISTERED_SCHEMA_REPORT),
            "feature_matrix_exists_now": registered_matrix_exists,
            "schema_report_exists_now": registered_schema_exists,
        },
        "dryrun_summary": dry_summary,
        "future_writer_contract": {
            "writer_allowed_next_under_contract": matrix_write_allowed_next,
            "allowed_feature_matrix_path": rel(REGISTERED_MATRIX_PATH),
            "allowed_schema_report_path": rel(REGISTERED_SCHEMA_REPORT),
            "expected_row_count": 700,
            "expected_train_rows": 450,
            "expected_test_rows": 250,
            "expected_feature_column_count": 80,
            "required_key_columns": ["subject_id", "sleep_date", "lifelog_date", "split"],
            "forbidden_columns": ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "prediction", "oof", "submission"],
            "writer_must_stop_after_schema_report": True,
            "model_training_allowed": False,
            "submission_artifact_allowed": False,
            "kaggle_or_dacon_allowed": False,
        },
        "checks": {
            "spec_contract_ready": spec_contract_ready,
            "dryrun_contract_ready": dryrun_contract_ready,
            "dryrun_schema_ready": dryrun_schema_ready,
            "queue_contract_ready": queue_contract_ready,
            "current_queue_contract_ready": current_queue_contract_ready,
            "queue_records_contract_ready": queue_records_contract_ready,
            "queue_records_matrix_materialized": queue_records_matrix_materialized,
            "queue_records_lane_closed": queue_records_lane_closed,
            "privacy_contract_ready": privacy_contract_ready,
            "registered_matrix_absent_before_write": not registered_matrix_exists,
            "registered_schema_absent_before_write": not registered_schema_exists,
            "feature_matrix_write_allowed_next_under_contract": matrix_write_allowed_next,
            "does_not_write_feature_matrix": True,
            "does_not_train_model": True,
            "does_not_create_submission_artifact": True,
        },
        "decision": {
            "feature_matrix_smoke_contract_ready": contract_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_write_allowed_now": False,
            "feature_matrix_write_allowed_next_under_contract": matrix_write_allowed_next,
            "feature_matrix_already_materialized_under_contract": matrix_already_materialized_under_contract,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": NEXT_ALLOWED_STEP
            if matrix_write_allowed_next
            else "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
            if matrix_already_materialized_under_contract
            else "fix_proximity_ambience_feature_matrix_contract_inputs",
            "next_action": (
                "A future no-approval step may write exactly one registered label-free feature matrix plus schema "
                "report, then stop. Do not train a model, create ID342, run Kaggle, or submit."
                if matrix_write_allowed_next
                else "The registered matrix already exists; validate schema/report and stop unless explicit Kaggle approval is given."
                if matrix_already_materialized_under_contract
                else "Do not write the proximity/ambience feature matrix until contract blockers are fixed."
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


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 proximity/ambience co-occurrence feature-matrix smoke contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It writes no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Contract ready: `{dec['feature_matrix_smoke_contract_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Feature matrix write allowed now: `{dec['feature_matrix_write_allowed_now']}`",
        f"- Feature matrix write allowed next under contract: `{dec['feature_matrix_write_allowed_next_under_contract']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Registered paths",
        "",
    ]
    for key, value in report["registered_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Dry-run summary", ""])
    for key, value in report["dryrun_summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Future writer contract", ""])
    for key, value in report["future_writer_contract"].items():
        lines.append(f"- {key}: `{value}`")
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
    if not report["decision"]["feature_matrix_smoke_contract_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
