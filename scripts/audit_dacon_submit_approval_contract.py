#!/usr/bin/env python3
"""Fail-closed local contract for the DACON submit Make target.

This script performs only local checks. It does not import the DACON API, does
not read DACON credentials, and does not submit anything. The Makefile calls it
after exact path/memo approval but before the Python DACON API call.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSION_LOG = SUBMISSIONS_DIR / "submission_log.csv"
DEFAULT_POLICY_REPORT = ROOT / "reports" / "post_id341_submission_spend_policy_20260619.json"
DEFAULT_SUB_MEMO = "Automated submission"
EXPECTED_COLUMNS = ["subject_id", "sleep_date", "lifelog_date", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
POLICY_DEPENDENCY_PATHS = [
    SUBMISSION_LOG,
    ROOT / "reports" / "post_id341_manual_submission_probe_decision_20260619.json",
    ROOT / "reports" / "post_id341_safe_wait_state_contract_20260619.json",
    ROOT / "reports" / "post_id341_public058_artifact_promotion_contract_20260619.json",
    ROOT / "reports" / "post_id341_public_local_gap_diagnosis_20260618.json",
    ROOT / "reports" / "post_id341_next_material_route_triage_20260619.json",
]


def clean_make_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_submission_path(raw_sub_path: str) -> Path:
    cleaned = clean_make_value(raw_sub_path)
    path = Path(cleaned)
    if not path.is_absolute():
        path = ROOT / path
    return path


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_freshness(policy_report_path: Path, extra_dependencies: list[Path] | None = None) -> dict[str, Any]:
    dependencies = [*POLICY_DEPENDENCY_PATHS, *(extra_dependencies or [])]
    if not policy_report_path.exists():
        return {
            "policy_mtime": None,
            "newest_dependency_mtime": None,
            "missing_dependencies": [relpath(path) for path in dependencies if not path.exists()],
            "stale_dependencies": [],
            "fresh": False,
        }

    policy_mtime = policy_report_path.stat().st_mtime
    dependency_rows = []
    missing_dependencies = []
    stale_dependencies = []
    for path in dependencies:
        if not path.exists():
            missing_dependencies.append(relpath(path))
            continue
        mtime = path.stat().st_mtime
        row = {"path": relpath(path), "mtime": mtime, "stale_vs_policy": mtime > policy_mtime}
        dependency_rows.append(row)
        if row["stale_vs_policy"]:
            stale_dependencies.append(row["path"])

    newest_dependency_mtime = max((row["mtime"] for row in dependency_rows), default=None)
    return {
        "policy_mtime": policy_mtime,
        "newest_dependency_mtime": newest_dependency_mtime,
        "missing_dependencies": missing_dependencies,
        "stale_dependencies": stale_dependencies,
        "dependency_rows": dependency_rows,
        "fresh": not missing_dependencies and not stale_dependencies,
    }


def finite_target_frame(df: pd.DataFrame) -> bool:
    targets = df[EXPECTED_COLUMNS[3:]]
    if not all(pd.api.types.is_numeric_dtype(targets[col]) for col in targets.columns):
        return False
    values = targets.to_numpy(dtype=float)
    return bool(math.isfinite(float(values.min())) and math.isfinite(float(values.max())))


def load_log_row(submission_file: str) -> tuple[dict[str, Any] | None, int]:
    log = pd.read_csv(SUBMISSION_LOG)
    matches = log[log["submission_file"].astype(str) == submission_file]
    if matches.empty:
        return None, 0
    return matches.iloc[-1].to_dict(), int(len(matches))


def bool_from_row(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_report(sub_path: str, sub_memo: str, policy_report_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    path = resolve_submission_path(sub_path)
    policy = load_json(policy_report_path) if policy_report_path.exists() else {}
    policy_decision = policy.get("decision", {})
    policy_probe_file = policy_decision.get("public_sanity_probe_if_user_insists_file")
    policy_score_submission_open = policy_decision.get("score_improvement_submission_recommended") is True

    freshness = policy_freshness(policy_report_path)
    if not freshness["fresh"]:
        errors.append("policy_report_stale_or_missing_dependencies")

    if not policy_report_path.exists():
        errors.append("policy_report_missing")
    elif policy_decision.get("submission_spend_policy_ready") is not True:
        errors.append("policy_report_not_ready")

    if not is_under(path, SUBMISSIONS_DIR):
        errors.append("sub_path_must_be_under_submissions_dir")
    if path.suffix.lower() != ".csv":
        errors.append("sub_path_must_be_csv")
    if not path.name.startswith("submission_id_"):
        errors.append("sub_path_must_be_unique_submission_id_file")
    if not path.exists():
        errors.append("sub_path_missing")

    memo = clean_make_value(sub_memo)
    if not memo:
        errors.append("sub_memo_empty")
    if memo == DEFAULT_SUB_MEMO:
        errors.append("sub_memo_must_not_be_default")

    csv_shape: list[int] | None = None
    csv_columns: list[str] | None = None
    csv_na_total: int | None = None
    csv_target_min: float | None = None
    csv_target_max: float | None = None
    if path.exists():
        try:
            csv = pd.read_csv(path)
            csv_shape = [int(csv.shape[0]), int(csv.shape[1])]
            csv_columns = list(csv.columns)
            csv_na_total = int(csv.isna().sum().sum())
            if csv_columns == EXPECTED_COLUMNS and finite_target_frame(csv):
                targets = csv[EXPECTED_COLUMNS[3:]].to_numpy(dtype=float)
                csv_target_min = float(targets.min())
                csv_target_max = float(targets.max())
            else:
                errors.append("csv_schema_or_target_type_invalid")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"csv_read_failed:{type(exc).__name__}")

    if csv_shape != [250, 10]:
        errors.append("csv_shape_must_be_250x10")
    if csv_columns != EXPECTED_COLUMNS:
        errors.append("csv_columns_mismatch")
    if csv_na_total not in (0, None):
        errors.append("csv_must_not_contain_missing_values")

    log_row, log_match_count = load_log_row(path.name) if SUBMISSION_LOG.exists() else (None, 0)
    if log_row is None:
        errors.append("submission_file_missing_from_submission_log")
        notebook_path = None
        is_submitted = None
        public_lb = None
        duplicate_file_count = None
    else:
        notebook_file = log_row.get("notebook_file")
        notebook_path = ROOT / str(notebook_file) if isinstance(notebook_file, str) and notebook_file else None
        is_submitted = bool_from_row(log_row.get("is_submitted_bool"))
        public_lb_value = log_row.get("public_lb_score_num")
        public_lb = None if pd.isna(public_lb_value) else float(public_lb_value)
        duplicate_file_count = int(log_row.get("duplicate_file_count") or 0)

        if log_match_count != 1:
            errors.append("submission_file_must_have_single_log_row")
        if duplicate_file_count != 1:
            errors.append("duplicate_file_count_must_be_one")
        if bool_from_row(log_row.get("artifact_exists")) is not True:
            errors.append("submission_log_artifact_exists_must_be_true")
        if is_submitted is not False:
            errors.append("submission_log_row_must_be_unsubmitted_before_submit")
        if public_lb is not None:
            errors.append("submission_log_row_must_not_already_have_public_score")
        if notebook_path is None or not notebook_path.exists():
            errors.append("matching_notebook_file_missing")

    if policy_score_submission_open:
        allowed_by_policy = True
        policy_mode = "score_improvement"
    elif path.name == policy_probe_file:
        allowed_by_policy = True
        policy_mode = "manual_public_sanity_probe"
        warnings.append("policy_allows_only_manual_sanity_probe_not_score_improvement")
    else:
        allowed_by_policy = False
        policy_mode = "blocked_by_submission_spend_policy"
        errors.append("sub_path_not_allowed_by_current_submission_spend_policy")

    return {
        "contract": "dacon_submit_approval_contract",
        "approval_contract_passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "sub_path": clean_make_value(sub_path),
            "sub_memo": memo,
            "policy_report": relpath(policy_report_path),
        },
        "policy": {
            "score_improvement_submission_recommended": policy_score_submission_open,
            "public_sanity_probe_if_user_insists_file": policy_probe_file,
            "mode": policy_mode,
            "allowed_by_policy": allowed_by_policy,
            "freshness": freshness,
        },
        "artifact": {
            "path": relpath(path),
            "exists": path.exists(),
            "shape": csv_shape,
            "columns": csv_columns,
            "na_total": csv_na_total,
            "target_min": csv_target_min,
            "target_max": csv_target_max,
            "log_match_count": log_match_count,
            "duplicate_file_count": duplicate_file_count,
            "is_submitted_bool": is_submitted,
            "public_lb": public_lb,
            "notebook_path": relpath(notebook_path),
            "notebook_exists": bool(notebook_path and notebook_path.exists()),
        },
        "no_dacon_submission_made": True,
        "no_kaggle_push_or_run_made": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-path", required=True)
    parser.add_argument("--sub-memo", required=True)
    parser.add_argument("--policy-report", type=Path, default=DEFAULT_POLICY_REPORT)
    args = parser.parse_args()

    policy_report = args.policy_report if args.policy_report.is_absolute() else ROOT / args.policy_report
    report = build_report(args.sub_path, args.sub_memo, policy_report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["approval_contract_passed"] else 8


if __name__ == "__main__":
    raise SystemExit(main())
