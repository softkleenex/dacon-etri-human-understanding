#!/usr/bin/env python3
"""Write the registered post-ID341 proximity/ambience feature matrix.

This materialization step is authorized by
`post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract_20260619`.
It may write only the registered label-free feature parquet and schema report.
It must not train a model, create predictions, create a submission CSV, modify
`submissions/submission_log.csv`, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun import (
    assemble_feature_frame,
    expected_key_frame,
    feature_names,
    feature_summary,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract import (
    REGISTERED_MATRIX_PATH,
    REGISTERED_SCHEMA_REPORT,
    SOURCE_FAMILY,
    TARGET_FOCUS,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
CONTRACT_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract_20260619.json"
MATRIX_PATH = REGISTERED_MATRIX_PATH
SCHEMA_REPORT_PATH = REGISTERED_SCHEMA_REPORT
KST = timezone(timedelta(hours=9))
KEYS = ["subject_id", "sleep_date", "lifelog_date", "split"]
TARGET_COLUMNS = {"Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"}
FORBIDDEN_TOKEN_FRAGMENTS = (
    "bssid",
    "address",
    "token",
    "identifier",
    "target",
    "label",
    "oof",
    "pred",
    "prediction",
    "submission",
    "public_lb",
    "leaderboard",
)


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required contract/report is missing: {safe_relative(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_contract_allows_write(contract: dict[str, Any], matrix_path: Path, schema_path: Path) -> dict[str, bool]:
    decision = contract.get("decision", {}) if isinstance(contract.get("decision"), dict) else {}
    writer = (
        contract.get("future_writer_contract", {}) if isinstance(contract.get("future_writer_contract"), dict) else {}
    )
    return {
        "contract_report_parse_ok": contract.get("diagnostic")
        == "post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract",
        "contract_ready_or_already_materialized": decision.get("feature_matrix_write_allowed_next_under_contract")
        is True
        or decision.get("feature_matrix_already_materialized_under_contract") is True,
        "source_family_matches": decision.get("source_family") == SOURCE_FAMILY,
        "target_focus_matches": decision.get("target_focus") == TARGET_FOCUS,
        "matrix_path_matches_contract": writer.get("allowed_feature_matrix_path") == safe_relative(matrix_path),
        "schema_path_matches_contract": writer.get("allowed_schema_report_path") == safe_relative(schema_path),
        "row_bound_matches_contract": writer.get("expected_row_count") == 700,
        "train_bound_matches_contract": writer.get("expected_train_rows") == 450,
        "test_bound_matches_contract": writer.get("expected_test_rows") == 250,
        "column_exact_matches_contract": writer.get("expected_feature_column_count") == 80,
        "key_columns_match_contract": writer.get("required_key_columns") == KEYS,
        "writer_must_stop_after_schema_report": writer.get("writer_must_stop_after_schema_report") is True,
        "no_model_allowed_by_contract": writer.get("model_training_allowed") is False,
        "no_submission_allowed_by_contract": writer.get("submission_artifact_allowed") is False,
        "no_kaggle_or_dacon_allowed_by_contract": writer.get("kaggle_or_dacon_allowed") is False,
        "decision_keeps_score_route_closed": decision.get("public_058_evidence") is False
        and decision.get("score_route_opened") is False,
        "decision_keeps_forbidden_actions_closed": decision.get("train_model_now") is False
        and decision.get("create_submission_artifact") is False
        and decision.get("submit_now") is False
        and decision.get("push_kaggle_now") is False
        and decision.get("run_kaggle_now") is False,
    }


def build_feature_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = expected_key_frame().sort_values(KEYS).reset_index(drop=True)
    frame = assemble_feature_frame().sort_values(KEYS).reset_index(drop=True)
    for key in KEYS:
        frame[key] = frame[key].astype(str)
        expected[key] = expected[key].astype(str)
    names = feature_names()
    for column in names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0.0)
        if not is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype(float)
    return frame[[*KEYS, *names]], expected[[*KEYS]]


def forbidden_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for column in columns:
        lower = column.lower()
        if column in TARGET_COLUMNS or any(fragment in lower for fragment in FORBIDDEN_TOKEN_FRAGMENTS):
            out.append(column)
    return sorted(set(out))


def row_alignment(frame: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    frame_keys = frame[KEYS].astype(str).copy()
    expected_keys = expected[KEYS].astype(str).copy()
    merged = frame_keys.merge(expected_keys.drop_duplicates(), on=KEYS, how="outer", indicator=True)
    return {
        "row_count": int(len(frame)),
        "expected_row_count": int(len(expected)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "duplicate_keys": int(frame.duplicated(KEYS).sum()),
        "missing_expected_keys": int(merged["_merge"].eq("right_only").sum()),
        "unexpected_keys": int(merged["_merge"].eq("left_only").sum()),
        "exact_key_alignment": bool(frame_keys.reset_index(drop=True).equals(expected_keys.reset_index(drop=True))),
    }


def finite_feature_values(frame: pd.DataFrame, feature_cols: list[str]) -> bool:
    if not feature_cols:
        return False
    return bool(np.isfinite(frame[feature_cols].astype(float).to_numpy()).all())


def validate_feature_matrix(frame: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    names = feature_names()
    alignment = row_alignment(frame, expected)
    summary = feature_summary(frame)
    feature_cols = [column for column in frame.columns if column not in KEYS]
    forbidden = forbidden_columns(list(frame.columns))
    numeric_or_bool_cols = [
        column for column in feature_cols if is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column])
    ]
    checks = {
        "row_count_exact": len(frame) == 700,
        "row_alignment_exact": alignment.get("exact_key_alignment") is True,
        "duplicate_keys_absent": alignment.get("duplicate_keys") == 0,
        "train_rows_exact": alignment.get("train_rows") == 450,
        "test_rows_exact": alignment.get("test_rows") == 250,
        "key_columns_first": list(frame.columns[: len(KEYS)]) == KEYS,
        "feature_columns_exact_declared_order": feature_cols == names,
        "feature_column_count_exact": len(feature_cols) == 80,
        "feature_column_count_bounded": len(feature_cols) <= 80,
        "no_target_or_prediction_columns": forbidden == [],
        "no_raw_proximity_identifier_columns": forbidden == [],
        "all_feature_columns_numeric_or_bool": len(numeric_or_bool_cols) == len(feature_cols),
        "no_missing_feature_values": not bool(frame[feature_cols].isna().any().any()),
        "all_feature_values_finite": finite_feature_values(frame, feature_cols),
        "split_values_expected": sorted(frame["split"].unique().tolist()) == ["test", "train"],
        "zero_variance_guard_bounded": summary.get("zero_variance_feature_count", 99) <= 20,
        "nonzero_feature_guard": summary.get("nonzero_feature_count", 0) >= 60,
    }
    return {
        "checks": checks,
        "allowed_boundary_passed": all(checks.values()),
        "row_alignment": alignment,
        "feature_frame_summary": summary,
        "feature_columns": {"feature_column_count": len(feature_cols), "features": feature_cols},
        "forbidden_columns": forbidden,
        "numeric_or_bool_feature_columns": numeric_or_bool_cols,
    }


def column_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        item: dict[str, Any] = {
            "name": column,
            "dtype": str(frame[column].dtype),
            "non_null_count": int(frame[column].notna().sum()),
            "null_count": int(frame[column].isna().sum()),
            "unique_count": int(frame[column].nunique(dropna=True)),
        }
        if column not in KEYS:
            numeric = frame[column].astype(float)
            item.update({"min": float(numeric.min()), "max": float(numeric.max()), "mean": float(numeric.mean())})
        rows.append(item)
    return rows


def build_schema_report(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    contract_checks: dict[str, bool],
    matrix_path: Path,
    schema_path: Path,
    matrix_written_now: bool,
) -> dict[str, Any]:
    validation = validate_feature_matrix(frame, expected)
    matrix_exists = matrix_path.exists()
    contract_passed = all(contract_checks.values())
    materialized = bool(contract_passed and validation["allowed_boundary_passed"] and matrix_exists)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_features_v1_schema",
        "report_only": False,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Validate and register the single allowed no-model/no-submission proximity/ambience feature matrix."
        ),
        "inputs": {
            "contract_report": safe_relative(CONTRACT_REPORT),
            "raw_feature_family": "mWifi + mBle + mAmbience + mScreenStatus co-occurrence aggregates",
        },
        "artifact": {
            "matrix_path": safe_relative(matrix_path),
            "schema_report_path": safe_relative(schema_path),
            "matrix_exists": matrix_exists,
            "matrix_written_now": matrix_written_now,
            "matrix_size_bytes": matrix_path.stat().st_size if matrix_exists else None,
            "matrix_sha256": sha256_file(matrix_path) if matrix_exists else None,
        },
        "contract_checks": contract_checks,
        "validation": validation,
        "column_schema": column_schema(frame),
        "decision": {
            "feature_matrix_materialized": materialized,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "row_count": int(len(frame)),
            "feature_column_count": int(len(feature_names())),
            "contract_passed": contract_passed,
            "allowed_boundary_passed": validation["allowed_boundary_passed"],
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_written_now": matrix_written_now,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": (
                "validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
                if materialized
                else "fix_proximity_ambience_feature_matrix_materialization"
            ),
            "next_action": (
                "Feature matrix materialized and schema-validated. Next local step may only validate deterministic "
                "rebuild/schema and stop; do not train, create ID342, run Kaggle, or submit."
                if materialized
                else "Do not advance until the feature matrix materialization contract is fixed."
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
    artifact = report["artifact"]
    lines = [
        "# Post-ID341 proximity/ambience feature matrix schema",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This registers one label-free feature matrix only. It creates no model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Feature matrix materialized: `{dec['feature_matrix_materialized']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Row count: `{dec['row_count']}`",
        f"- Feature columns: `{dec['feature_column_count']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Artifact",
        "",
    ]
    for key, value in artifact.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Validation checks", ""])
    for key, value in report["validation"]["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Contract checks", ""])
    for key, value in report["contract_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    contract = load_json(CONTRACT_REPORT)
    contract_checks = verify_contract_allows_write(contract, MATRIX_PATH, SCHEMA_REPORT_PATH)
    if not all(contract_checks.values()):
        print(json.dumps({"contract_checks": contract_checks}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    frame, expected = build_feature_matrix()
    validation = validate_feature_matrix(frame, expected)
    if not validation["allowed_boundary_passed"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix_written_now = False
    if not MATRIX_PATH.exists():
        frame.to_parquet(MATRIX_PATH, index=False)
        matrix_written_now = True

    report = build_schema_report(frame, expected, contract_checks, MATRIX_PATH, SCHEMA_REPORT_PATH, matrix_written_now)
    if not report["decision"]["feature_matrix_materialized"]:
        print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
        raise SystemExit(1)

    SCHEMA_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = SCHEMA_REPORT_PATH.with_suffix(".md")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {safe_relative(MATRIX_PATH)}")
    print(f"Wrote {safe_relative(SCHEMA_REPORT_PATH)}")
    print(f"Wrote {safe_relative(md_path)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
