#!/usr/bin/env python3
"""Validate the post-ID341 proximity/ambience feature matrix, then stop.

This report-only gate follows the registered matrix materialization. It validates
the existing parquet and schema against the deterministic writer, then explicitly
keeps the lane closed for models, ID342 artifacts, Kaggle runs, and DACON
submissions unless a separate exact approval is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.write_post_id341_proximity_ambience_cooccurrence_feature_matrix import (
    KEYS,
    MATRIX_PATH,
    ROOT,
    SCHEMA_REPORT_PATH,
    SOURCE_FAMILY,
    TARGET_FOCUS,
    build_feature_matrix,
    feature_names,
    safe_relative,
    validate_feature_matrix,
)

REPORTS_DIR = ROOT / "reports"
CONTRACT_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop_20260619.json"
KST = timezone(timedelta(hours=9))
FORBIDDEN_DECISION_KEYS = (
    "train_model_now",
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
)
SUBMISSION_ARTIFACT_SEARCH_ROOTS = ["submissions", "notebooks/history", "artifacts/base_predictions"]
VALIDATE_STEP = "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "diagnostic": payload.get("diagnostic"),
        "decision": payload.get("decision", {}),
        "payload": payload,
        "error": None,
    }


def no_forbidden_actions(decision: dict[str, Any]) -> bool:
    return all(decision.get(key) is False for key in FORBIDDEN_DECISION_KEYS)


def queue_lane_completed_no_model(queue_payload: dict[str, Any]) -> bool:
    items = queue_payload.get("source_queue", []) if isinstance(queue_payload.get("source_queue"), list) else []
    item = next((entry for entry in items if entry.get("name") == SOURCE_FAMILY), {})
    return bool(
        item.get("status") == "validated_stop_completed_no_model"
        and item.get("local_no_approval_allowed") is False
        and item.get("feature_matrix_materialized_under_contract") is True
        and item.get("train_model_now") is False
        and item.get("create_artifact_allowed") is False
        and item.get("submission_artifact_allowed") is False
    )


def source_family_submission_artifacts() -> list[str]:
    paths: list[str] = []
    for root in SUBMISSION_ARTIFACT_SEARCH_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for pattern in (f"*{SOURCE_FAMILY}*", "*proximity_ambience_cooccurrence*"):
            paths.extend(safe_relative(path) for path in base.glob(pattern) if path.is_file())
    return sorted(set(paths))


def submission_log_status() -> dict[str, Any]:
    if not SUBMISSION_LOG.exists():
        return {"exists": False, "row_count": None, "max_id": None, "source_family_row_count": None, "safe": False}
    frame = pd.read_csv(SUBMISSION_LOG)
    max_id = int(frame["id"].max()) if "id" in frame.columns and len(frame) else None
    source_family_mask = frame.astype(str).apply(
        lambda column: (
            column.str.contains(SOURCE_FAMILY, case=False, regex=False)
            | column.str.contains("proximity_ambience_cooccurrence", case=False, regex=False)
        )
    )
    source_family_row_count = int(source_family_mask.any(axis=1).sum())
    return {
        "exists": True,
        "row_count": int(len(frame)),
        "max_id": max_id,
        "source_family_row_count": source_family_row_count,
        "safe": source_family_row_count == 0,
    }


def deterministic_match(existing: pd.DataFrame, rebuilt: pd.DataFrame) -> dict[str, Any]:
    names = feature_names()
    same_shape = existing.shape == rebuilt.shape
    same_columns = list(existing.columns) == list(rebuilt.columns)
    keys_match = bool(
        existing[KEYS].astype(str).reset_index(drop=True).equals(rebuilt[KEYS].astype(str).reset_index(drop=True))
    )
    numeric_delta_max: float | None = None
    numeric_delta_mean: float | None = None
    values_match = False
    if same_shape and same_columns:
        delta = (existing[names].astype(float) - rebuilt[names].astype(float)).abs()
        numeric_delta_max = float(delta.to_numpy().max())
        numeric_delta_mean = float(delta.to_numpy().mean())
        values_match = bool(numeric_delta_max <= 1e-12)
    return {
        "same_shape": same_shape,
        "same_columns": same_columns,
        "keys_match": keys_match,
        "numeric_delta_max": numeric_delta_max,
        "numeric_delta_mean": numeric_delta_mean,
        "values_match_with_tolerance_1e_12": values_match,
        "deterministic_rebuild_match": bool(same_shape and same_columns and keys_match and values_match),
    }


def build_report() -> dict[str, Any]:
    schema = load_json_status(SCHEMA_REPORT_PATH)
    contract = load_json_status(CONTRACT_REPORT)
    queue = load_json_status(QUEUE_REPORT)
    submission_log = submission_log_status()
    source_family_artifacts = source_family_submission_artifacts()

    matrix_exists = MATRIX_PATH.exists()
    existing = pd.read_parquet(MATRIX_PATH) if matrix_exists else pd.DataFrame()
    rebuilt, expected = build_feature_matrix()
    validation = (
        validate_feature_matrix(existing, expected)
        if matrix_exists
        else {"allowed_boundary_passed": False, "checks": {}}
    )
    match = deterministic_match(existing, rebuilt) if matrix_exists else {"deterministic_rebuild_match": False}

    schema_decision = schema["decision"]
    contract_decision = contract["decision"]
    queue_decision = queue["decision"]
    current_queue_records_validate_stop = bool(
        queue_decision.get("material_new_source_queue_ready") is True
        and queue_decision.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_decision.get("next_no_approval_step_type") == VALIDATE_STEP
        and queue_decision.get("local_no_approval_score_route_exists") is False
        and queue_decision.get("requires_explicit_kaggle_approval") is True
        and queue_decision.get("public_058_evidence") is False
        and no_forbidden_actions(queue_decision)
    )
    lane_completed_no_model = bool(queue.get("parse_ok") is True and queue_lane_completed_no_model(queue["payload"]))
    matrix_file = {
        "path": safe_relative(MATRIX_PATH),
        "exists": matrix_exists,
        "bytes": MATRIX_PATH.stat().st_size if matrix_exists else None,
        "sha256": sha256_file(MATRIX_PATH),
        "shape": list(existing.shape) if matrix_exists else None,
    }

    checks = {
        "matrix_exists": matrix_exists,
        "matrix_shape_exact": matrix_exists and tuple(existing.shape) == (700, 84),
        "schema_report_parse_ok": schema.get("parse_ok") is True,
        "schema_materialized_and_safe": bool(
            schema_decision.get("feature_matrix_materialized") is True
            and schema_decision.get("source_family") == SOURCE_FAMILY
            and schema_decision.get("target_focus") == TARGET_FOCUS
            and schema_decision.get("allowed_boundary_passed") is True
            and schema_decision.get("contract_passed") is True
            and schema_decision.get("row_count") == 700
            and schema_decision.get("feature_column_count") == 80
            and schema_decision.get("public_058_evidence") is False
            and schema_decision.get("score_route_opened") is False
            and no_forbidden_actions(schema_decision)
        ),
        "contract_report_parse_ok": contract.get("parse_ok") is True,
        "contract_records_materialized_stop": bool(
            contract_decision.get("feature_matrix_smoke_contract_ready") is True
            and contract_decision.get("feature_matrix_already_materialized_under_contract") is True
            and contract_decision.get("feature_matrix_write_allowed_now") is False
            and contract_decision.get("feature_matrix_write_allowed_next_under_contract") is False
            and contract_decision.get("next_local_allowed_step") == VALIDATE_STEP
            and contract_decision.get("public_058_evidence") is False
            and contract_decision.get("score_route_opened") is False
            and no_forbidden_actions(contract_decision)
        ),
        "queue_records_validate_stop": bool(current_queue_records_validate_stop or lane_completed_no_model),
        "matrix_validation_passed": validation.get("allowed_boundary_passed") is True,
        "deterministic_rebuild_match": match.get("deterministic_rebuild_match") is True,
        "submission_log_has_no_source_family_rows": submission_log["safe"] is True,
        "no_source_family_submission_artifacts": len(source_family_artifacts) == 0,
    }
    validation_stop_passed = bool(all(checks.values()))
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Validate the registered label-free proximity/ambience feature matrix and close the no-approval local "
            "lane at schema validation without opening any model, artifact, Kaggle, or DACON action."
        ),
        "inputs": {
            "feature_matrix": matrix_file,
            "schema_report": {key: value for key, value in schema.items() if key != "payload"},
            "contract_report": {key: value for key, value in contract.items() if key != "payload"},
            "material_new_source_queue": {key: value for key, value in queue.items() if key != "payload"},
        },
        "validation": validation,
        "deterministic_rebuild": match,
        "submission_log_status": submission_log,
        "source_family_submission_artifacts": source_family_artifacts,
        "checks": checks,
        "queue_lane_completed_no_model": lane_completed_no_model,
        "decision": {
            "validation_stop_passed": validation_stop_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval",
            "approval_phrase": "Kaggle upstream rebuild no-submit 실행 승인",
            "next_action": (
                "Proximity/ambience matrix is validated as label-free infrastructure. Stop local no-approval work "
                "here; only an exact approved Kaggle no-submit upstream rebuild may proceed, and still may not submit."
                if validation_stop_passed
                else "Fix the proximity/ambience matrix/schema/queue validation blockers before any further local work."
            ),
        },
        "safety_contract": {
            "feature_matrix_written_now": False,
            "model_trained": False,
            "predictions_created": False,
            "submission_csv_created": False,
            "submission_log_modified": False,
            "base_prediction_artifact_created": False,
            "notebook_history_artifact_created": False,
            "kaggle_run_executed": False,
            "dacon_submission_executed": False,
            "uses_target_labels": False,
            "uses_oof_predictions": False,
            "uses_public_scores": False,
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
    matrix = report["inputs"]["feature_matrix"]
    checks = report["checks"]
    match = report["deterministic_rebuild"]
    lines = [
        "# Proximity/ambience feature matrix validation stop — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It validates the registered label-free matrix and then stops the local no-approval "
            "lane. It creates no model, predictions, submission CSV, base prediction artifact, notebook/history "
            "artifact, submission-log row, Kaggle run, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Validation stop passed: `{dec['validation_stop_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{dec['target_focus']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Approval phrase: `{dec['approval_phrase']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Matrix",
        "",
        f"- Path: `{matrix['path']}`",
        f"- Exists: `{matrix['exists']}`",
        f"- Shape: `{matrix['shape']}`",
        f"- SHA256: `{matrix['sha256']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Deterministic rebuild match",
            "",
            f"- Same shape: `{match.get('same_shape')}`",
            f"- Same columns: `{match.get('same_columns')}`",
            f"- Keys match: `{match.get('keys_match')}`",
            f"- Max numeric delta: `{match.get('numeric_delta_max')}`",
            f"- Mean numeric delta: `{match.get('numeric_delta_mean')}`",
            f"- Deterministic rebuild match: `{match.get('deterministic_rebuild_match')}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.output)
    print(f"Wrote {safe_relative(args.output)}")
    print(f"Wrote {safe_relative(args.output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    if not report["decision"]["validation_stop_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
