#!/usr/bin/env python3
"""Write the registered post-ID341 usage semantic burst/Markov feature matrix.

This is the narrow materialization step authorized by
`post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619`.
It may write only the registered label-free feature parquet and schema report.
It must not train a model, create predictions, create a submission CSV, modify
`submissions/submission_log.csv`, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from scripts.diagnose_post_id341_usage_semantic_burst_markov_design import PREDECLARED_APP_CATEGORIES
from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_generation_dryrun import (
    FEATURE_COLUMN_BOUND,
    ROOT,
    SOURCE_FAMILY,
    TARGET_FOCUS,
    all_feature_columns,
    assemble_feature_frame,
    columns_are_safe,
    expected_key_frame,
    feature_frame_summary,
    feature_groups,
)

REPORTS_DIR = ROOT / "reports"
CONTRACT_REPORT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.json"
MATRIX_PATH = ROOT / "data" / "processed" / "post_id341_usage_semantic_burst_markov_features_v1.parquet"
SCHEMA_REPORT_PATH = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_features_v1_schema.json"
KST = timezone(timedelta(hours=9))
KEYS = ["subject_id", "lifelog_date", "split"]
FIRST_DRYRUN_FEATURES = all_feature_columns(feature_groups())
TARGET_COLUMNS = {"Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"}
FORBIDDEN_TOKEN_FRAGMENTS = (
    "app_name",
    "raw_app",
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
        == "post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract",
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
        "column_exact_matches_contract": writer.get("expected_feature_column_count") == 77,
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
    expected = expected_key_frame()
    frame = assemble_feature_frame().sort_values(KEYS).reset_index(drop=True)
    frame["lifelog_date"] = frame["lifelog_date"].astype(str)
    frame["split"] = frame["split"].astype(str)
    for column in FIRST_DRYRUN_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0.0)
        if not is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype(float)
    return frame[[*KEYS, *FIRST_DRYRUN_FEATURES]], expected.sort_values(KEYS).reset_index(drop=True)


def forbidden_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for column in columns:
        lower = column.lower()
        if column in TARGET_COLUMNS or any(fragment in lower for fragment in FORBIDDEN_TOKEN_FRAGMENTS):
            out.append(column)
    return sorted(set(out))


def row_alignment(frame: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    frame_keys = frame[KEYS].copy()
    expected_keys = expected[KEYS].copy()
    for key in KEYS:
        frame_keys[key] = frame_keys[key].astype(str)
        expected_keys[key] = expected_keys[key].astype(str)
    duplicate_keys = int(frame.duplicated(KEYS).sum())
    merged = frame_keys.merge(expected_keys.drop_duplicates(), on=KEYS, how="outer", indicator=True)
    return {
        "row_count": int(len(frame)),
        "expected_row_count": int(len(expected)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "duplicate_keys": duplicate_keys,
        "missing_expected_keys": int(merged["_merge"].eq("right_only").sum()),
        "unexpected_keys": int(merged["_merge"].eq("left_only").sum()),
        "exact_key_alignment": bool(frame_keys.reset_index(drop=True).equals(expected_keys.reset_index(drop=True))),
    }


def coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split, part in frame.groupby("split", sort=True):
        by_split[str(split)] = {
            "rows": int(len(part)),
            "usage_present_rate": float(part["usage_row_count"].fillna(0).gt(0).mean()),
            "screen_match_rate_mean": float(part["screen_match_rate_5min"].fillna(0).mean()),
            "ambience_match_rate_mean": float(part["ambience_match_rate_10min"].fillna(0).mean()),
            "source_coverage_score_mean": float(part["source_coverage_score"].fillna(0).mean()),
            "late_reactivation_day_rate": float(part["late_reactivation_rows_ge1"].fillna(0).gt(0).mean()),
            "evening_burst_day_rate": float(part["evening_burst_rows_ge3"].fillna(0).gt(0).mean()),
            "category_switch_rate_mean": float(part["category_switch_rate"].fillna(0).mean()),
            "transition_entropy_mean": float(part["transition_entropy"].fillna(0).mean()),
        }
    return {"by_split": by_split}


def finite_feature_values(frame: pd.DataFrame, feature_cols: list[str]) -> bool:
    if not feature_cols:
        return False
    values = frame[feature_cols].astype(float).to_numpy()
    return bool(np.isfinite(values).all())


def validate_feature_matrix(frame: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    alignment = row_alignment(frame, expected)
    summary = feature_frame_summary(frame)
    coverage = coverage_summary(frame)
    feature_cols = [column for column in frame.columns if column not in KEYS]
    forbidden = forbidden_columns(list(frame.columns))
    numeric_or_bool_cols = [
        column for column in feature_cols if is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column])
    ]
    train = coverage["by_split"].get("train", {})
    test = coverage["by_split"].get("test", {})
    checks = {
        "row_count_exact": len(frame) == 700,
        "row_alignment_exact": alignment.get("exact_key_alignment") is True,
        "duplicate_keys_absent": alignment.get("duplicate_keys") == 0,
        "train_rows_exact": alignment.get("train_rows") == 450,
        "test_rows_exact": alignment.get("test_rows") == 250,
        "key_columns_first": list(frame.columns[: len(KEYS)]) == KEYS,
        "feature_columns_exact_declared_order": feature_cols == FIRST_DRYRUN_FEATURES,
        "feature_column_count_exact": summary.get("feature_column_count") == 77,
        "feature_column_count_bounded": summary.get("feature_column_count", 999) <= FEATURE_COLUMN_BOUND,
        "no_target_or_prediction_columns": forbidden == [],
        "no_raw_app_name_columns": columns_are_safe(feature_cols),
        "all_feature_columns_numeric_or_bool": len(numeric_or_bool_cols) == len(feature_cols),
        "no_missing_feature_values": not bool(frame[feature_cols].isna().any().any()),
        "all_feature_values_finite": finite_feature_values(frame, feature_cols),
        "split_values_expected": sorted(frame["split"].unique().tolist()) == ["test", "train"],
        "zero_variance_guard_bounded": summary.get("zero_variance_feature_count", 99) <= 5,
        "usage_coverage_passes": train.get("usage_present_rate", 0.0) >= 0.95
        and test.get("usage_present_rate", 0.0) >= 0.95,
        "screen_coupling_passes": train.get("screen_match_rate_mean", 0.0) >= 0.95
        and test.get("screen_match_rate_mean", 0.0) >= 0.95,
        "ambience_coupling_passes": train.get("ambience_match_rate_mean", 0.0) >= 0.95
        and test.get("ambience_match_rate_mean", 0.0) >= 0.95,
        "predeclared_category_space_used": len(PREDECLARED_APP_CATEGORIES) == 7,
    }
    return {
        "checks": checks,
        "allowed_boundary_passed": all(checks.values()),
        "row_alignment": alignment,
        "coverage": coverage,
        "feature_frame_summary": summary,
        "feature_columns": {"feature_column_count": len(feature_cols), "features": feature_cols},
        "forbidden_columns": forbidden,
        "numeric_or_bool_feature_columns": numeric_or_bool_cols,
        "predeclared_app_categories": PREDECLARED_APP_CATEGORIES,
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
    file_info = {
        "matrix_path": safe_relative(matrix_path),
        "schema_report_path": safe_relative(schema_path),
        "matrix_exists": matrix_exists,
        "matrix_size_bytes": matrix_path.stat().st_size if matrix_exists else None,
        "matrix_sha256": sha256_file(matrix_path) if matrix_exists else None,
    }
    contract_passed = all(contract_checks.values())
    materialized = bool(contract_passed and validation["allowed_boundary_passed"] and matrix_exists)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_usage_semantic_burst_markov_features_v1_schema",
        "report_only": False,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Validate and register the single allowed no-model/no-submission usage semantic burst/Markov feature matrix."
        ),
        "inputs": {
            "contract_report": safe_relative(CONTRACT_REPORT),
            "raw_feature_family": "mUsageStats + mScreenStatus + mAmbience semantic burst/session/Markov aggregates",
        },
        "artifact": file_info,
        "contract_checks": contract_checks,
        "validation": validation,
        "schema": column_schema(frame),
        "decision": {
            "feature_matrix_materialized": materialized,
            "matrix_written_now": matrix_written_now,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "allowed_boundary_passed": validation["allowed_boundary_passed"],
            "contract_passed": contract_passed,
            "row_count": int(len(frame)),
            "feature_column_count": int(len([column for column in frame.columns if column not in KEYS])),
            "public_058_evidence": False,
            "score_route_opened": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
            "next_action": (
                "Feature matrix is registered as non-score infrastructure only. Do not create ID342, train a model, "
                "submit to DACON, or run Kaggle from this artifact without a separate approved contract."
            ),
        },
        "safety_contract": {
            "feature_matrix_written": materialized,
            "predictions_created": False,
            "submission_csv_created": False,
            "submission_log_modified": False,
            "base_prediction_artifact_created": False,
            "notebook_history_artifact_created": False,
            "model_trained": False,
            "kaggle_run_executed": False,
            "dacon_submission_executed": False,
            "uses_target_labels": False,
            "uses_oof_predictions": False,
            "uses_public_scores": False,
            "raw_app_name_feature_columns_created": False,
            "target_guided_vocabulary_used": False,
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    validation = report["validation"]
    coverage = validation["coverage"]["by_split"]
    artifact = report["artifact"]
    lines = [
        "# Usage semantic burst/Markov feature matrix v1 schema — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This materializes only the registered label-free usage semantic burst/Markov feature matrix. It creates "
            "no model, predictions, submission CSV, base prediction artifact, notebook/history artifact, "
            "submission-log row, Kaggle run, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Feature matrix materialized: `{dec['feature_matrix_materialized']}`",
        f"- Matrix written now: `{dec['matrix_written_now']}`",
        f"- Matrix path: `{artifact['matrix_path']}`",
        f"- Matrix SHA256: `{artifact['matrix_sha256']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Rows: `{dec['row_count']}`",
        f"- Feature columns: `{dec['feature_column_count']}` / `77`",
        f"- Contract passed: `{dec['contract_passed']}`",
        f"- Allowed boundary passed: `{dec['allowed_boundary_passed']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Validation checks",
        "",
    ]
    for key, value in validation["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| split | rows | usage present | screen match | ambience match | source coverage | late reactivation | evening burst |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ["train", "test"]:
        item = coverage.get(split, {})
        lines.append(
            f"| `{split}` | {item.get('rows', 0)} | {item.get('usage_present_rate', 0.0):.4f} | "
            f"{item.get('screen_match_rate_mean', 0.0):.4f} | "
            f"{item.get('ambience_match_rate_mean', 0.0):.4f} | "
            f"{item.get('source_coverage_score_mean', 0.0):.4f} | "
            f"{item.get('late_reactivation_day_rate', 0.0):.4f} | "
            f"{item.get('evening_burst_day_rate', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Feature columns",
            "",
            "```json",
            json.dumps(validation["feature_columns"]["features"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Boundary",
            "",
            (
                "This is still not leaderboard evidence. It only parks a bounded, target-label-free feature matrix "
                "for later review; ID342/modeling/submission/Kaggle actions remain closed without a new explicit contract."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_schema_report(report: dict[str, Any], schema_path: Path) -> None:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    schema_path.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")


def materialize_feature_matrix(matrix_path: Path, schema_path: Path, overwrite: bool = False) -> dict[str, Any]:
    contract = load_json(CONTRACT_REPORT)
    contract_checks = verify_contract_allows_write(contract, matrix_path, schema_path)
    if not all(contract_checks.values()):
        raise RuntimeError(f"Usage semantic burst/Markov feature-matrix contract is not open: {contract_checks}")

    frame, expected = build_feature_matrix()
    validation = validate_feature_matrix(frame, expected)
    if not validation["allowed_boundary_passed"]:
        raise RuntimeError(f"Usage semantic burst/Markov feature matrix failed validation: {validation['checks']}")

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_written_now = False
    if matrix_path.exists() and not overwrite:
        existing = pd.read_parquet(matrix_path)
        existing_validation = validate_feature_matrix(existing, expected)
        if not existing_validation["allowed_boundary_passed"]:
            raise RuntimeError(f"Existing matrix failed validation: {existing_validation['checks']}")
    else:
        frame.to_parquet(matrix_path, index=False)
        matrix_written_now = True

    report = build_schema_report(frame, expected, contract_checks, matrix_path, schema_path, matrix_written_now)
    write_schema_report(report, schema_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-output", type=Path, default=MATRIX_PATH)
    parser.add_argument("--schema-output", type=Path, default=SCHEMA_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = args.matrix_output if args.matrix_output.is_absolute() else ROOT / args.matrix_output
    schema_path = args.schema_output if args.schema_output.is_absolute() else ROOT / args.schema_output
    report = materialize_feature_matrix(matrix_path, schema_path, overwrite=args.overwrite)
    dec = report["decision"]
    print(f"Wrote/validated {safe_relative(matrix_path)}")
    print(f"Wrote {safe_relative(schema_path)}")
    print(f"Wrote {safe_relative(schema_path.with_suffix('.md'))}")
    print(
        json.dumps(
            {
                "feature_matrix_materialized": dec["feature_matrix_materialized"],
                "matrix_written_now": dec["matrix_written_now"],
                "row_count": dec["row_count"],
                "feature_column_count": dec["feature_column_count"],
                "public_058_evidence": dec["public_058_evidence"],
                "train_model_now": dec["train_model_now"],
                "create_submission_artifact": dec["create_submission_artifact"],
                "submit_now": dec["submit_now"],
                "push_kaggle_now": dec["push_kaggle_now"],
                "run_kaggle_now": dec["run_kaggle_now"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
