#!/usr/bin/env python3
"""Contract for the circadian HR/light feature-matrix smoke step.

This report-only diagnostic consumes the circadian feature-generation dry run
and current post-ID341 operating gates. It decides whether a future feature-
matrix writer may be implemented under a narrow no-model/no-submission contract.
This script does not write a feature matrix, train a model, create predictions,
create a submission CSV, modify the submission log, run Kaggle, or submit to
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

SOURCE_FAMILY = "circadian_hr_light_phase_shape_v1"
TARGET_FOCUS = ["Q3", "S3", "S2"]
DRYRUN_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_feature_generation_dryrun_20260619.json"
NO_APPROVAL_FRONTIER = REPORTS_DIR / "post_id341_no_approval_frontier_20260619.json"
LOCAL_SMOKE_FRESHNESS = REPORTS_DIR / "post_id341_local_smoke_freshness_gate_20260618.json"
PRELAUNCH_GATE = REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json"
PROPOSED_MATRIX_PATH = ROOT / "data" / "processed" / "post_id341_circadian_hr_light_phase_features_v1.parquet"
PROPOSED_SCHEMA_PATH = REPORTS_DIR / "post_id341_circadian_hr_light_phase_features_v1_schema.json"

FORBIDDEN_DECISION_KEYS = [
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
    "train_model_now",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload["path"] = safe_relative(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def no_forbidden_action(report: dict[str, Any]) -> bool:
    dec = decision(report)
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def dryrun_contract(dryrun: dict[str, Any]) -> dict[str, bool]:
    dec = decision(dryrun)
    alignment = dryrun.get("row_alignment", {}) if isinstance(dryrun.get("row_alignment"), dict) else {}
    columns = dryrun.get("feature_columns", {}) if isinstance(dryrun.get("feature_columns"), dict) else {}
    coverage = dryrun.get("coverage", {}) if isinstance(dryrun.get("coverage"), dict) else {}
    safety = dryrun.get("safety_contract", {}) if isinstance(dryrun.get("safety_contract"), dict) else {}
    by_split = coverage.get("by_split", {}) if isinstance(coverage.get("by_split"), dict) else {}
    train = by_split.get("train", {}) if isinstance(by_split.get("train"), dict) else {}
    test = by_split.get("test", {}) if isinstance(by_split.get("test"), dict) else {}
    return {
        "dryrun_passed": dec.get("dryrun_passed") is True,
        "source_family_matches": dec.get("source_family") == SOURCE_FAMILY,
        "target_focus_matches": dec.get("target_focus") == TARGET_FOCUS,
        "row_alignment_exact": alignment.get("expected_rows") == 700
        and alignment.get("dryrun_rows") == 700
        and alignment.get("exact_key_alignment") is True
        and alignment.get("duplicate_keys") == 0,
        "feature_columns_bounded": columns.get("feature_column_count") == 48
        and columns.get("feature_column_count") == dec.get("feature_column_count")
        and columns.get("feature_column_count", 999) <= 56
        and columns.get("within_first_dryrun_bound") is True
        and columns.get("missing_declared_features") == []
        and columns.get("extra_features") == []
        and columns.get("all_values_finite") is True,
        "coverage_passes": train.get("strong_phase_shape_rate", 0.0) >= 0.50
        and test.get("strong_phase_shape_rate", 0.0) >= 0.50
        and train.get("weak_phase_shape_rate", 0.0) >= 0.70
        and test.get("weak_phase_shape_rate", 0.0) >= 0.70
        and train.get("whr_covered_rate", 0.0) >= 0.85
        and test.get("whr_covered_rate", 0.0) >= 0.90,
        "dryrun_wrote_no_artifacts": safety.get("feature_matrix_written") is False
        and safety.get("feature_parquet_written") is False
        and safety.get("predictions_created") is False
        and safety.get("submission_csv_created") is False
        and safety.get("model_trained") is False,
        "dryrun_all_action_flags_false": no_forbidden_action(dryrun),
    }


def frontier_contract(frontier: dict[str, Any]) -> dict[str, bool]:
    dec = decision(frontier)
    return {
        "frontier_ready": dec.get("no_approval_frontier_ready") is True,
        "local_score_routes_closed": dec.get("all_local_no_approval_score_routes_closed") is True,
        "no_approval_score_route_absent": dec.get("no_approval_score_route_exists") is False,
        "score_route_requires_kaggle_approval": dec.get("requires_explicit_kaggle_approval") is True,
        "public_058_not_reviewable": dec.get("public_058_reviewable") is False,
        "frontier_all_action_flags_false": no_forbidden_action(frontier),
    }


def smoke_contract(local_smoke: dict[str, Any], prelaunch: dict[str, Any]) -> dict[str, bool]:
    smoke_dec = decision(local_smoke)
    prelaunch_dec = decision(prelaunch)
    return {
        "local_smoke_fresh": smoke_dec.get("local_runner_smoke_fresh") is True,
        "local_smoke_rerun_not_required": smoke_dec.get("requires_local_runner_smoke_rerun") is False,
        "prelaunch_ready_for_explicit_approval": prelaunch_dec.get("prelaunch_ready_for_explicit_approval") is True,
        "prelaunch_still_requires_approval": prelaunch_dec.get("requires_explicit_user_approval") is True,
        "prelaunch_artifact_review_closed": prelaunch_dec.get("artifact_contract_review_allowed") is False,
        "smoke_and_prelaunch_action_flags_false": no_forbidden_action(local_smoke) and no_forbidden_action(prelaunch),
    }


def path_contract() -> dict[str, bool]:
    return {
        "proposed_matrix_path_is_unique": not PROPOSED_MATRIX_PATH.exists(),
        "proposed_schema_path_is_unique": not PROPOSED_SCHEMA_PATH.exists(),
        "proposed_matrix_under_data_processed": PROPOSED_MATRIX_PATH.parent == ROOT / "data" / "processed",
        "proposed_schema_under_reports": PROPOSED_SCHEMA_PATH.parent == REPORTS_DIR,
        "proposed_matrix_name_is_source_specific": "circadian_hr_light_phase" in PROPOSED_MATRIX_PATH.name,
        "proposed_schema_name_is_source_specific": "circadian_hr_light_phase" in PROPOSED_SCHEMA_PATH.name,
    }


def existing_registered_output_contract(schema_report: dict[str, Any]) -> dict[str, bool]:
    dec = decision(schema_report)
    artifact = schema_report.get("artifact", {}) if isinstance(schema_report.get("artifact"), dict) else {}
    validation = schema_report.get("validation", {}) if isinstance(schema_report.get("validation"), dict) else {}
    safety = schema_report.get("safety_contract", {}) if isinstance(schema_report.get("safety_contract"), dict) else {}
    return {
        "schema_report_parse_ok": schema_report.get("exists") is True and schema_report.get("parse_ok") is True,
        "schema_diagnostic_matches": schema_report.get("diagnostic")
        == "post_id341_circadian_hr_light_phase_features_v1_schema",
        "schema_source_family_matches": dec.get("source_family") == SOURCE_FAMILY,
        "schema_target_focus_matches": dec.get("target_focus") == TARGET_FOCUS,
        "schema_matrix_path_matches": artifact.get("matrix_path") == safe_relative(PROPOSED_MATRIX_PATH),
        "schema_matrix_exists": artifact.get("matrix_exists") is True and PROPOSED_MATRIX_PATH.exists(),
        "schema_feature_matrix_materialized": dec.get("feature_matrix_materialized") is True,
        "schema_allowed_boundary_passed": dec.get("allowed_boundary_passed") is True
        and validation.get("allowed_boundary_passed") is True,
        "schema_row_and_column_bounds_match": dec.get("row_count") == 700 and dec.get("feature_column_count") == 48,
        "schema_score_actions_closed": dec.get("public_058_evidence") is False
        and dec.get("score_route_opened") is False
        and dec.get("train_model_now") is False
        and dec.get("create_submission_artifact") is False
        and dec.get("submit_now") is False
        and dec.get("push_kaggle_now") is False
        and dec.get("run_kaggle_now") is False,
        "schema_safety_flags_closed": safety.get("predictions_created") is False
        and safety.get("submission_csv_created") is False
        and safety.get("submission_log_modified") is False
        and safety.get("model_trained") is False
        and safety.get("kaggle_run_executed") is False
        and safety.get("dacon_submission_executed") is False
        and safety.get("uses_target_labels") is False
        and safety.get("uses_oof_predictions") is False
        and safety.get("uses_public_scores") is False,
    }


def build_report() -> dict[str, Any]:
    dryrun = load_report(DRYRUN_REPORT)
    frontier = load_report(NO_APPROVAL_FRONTIER)
    local_smoke = load_report(LOCAL_SMOKE_FRESHNESS)
    prelaunch = load_report(PRELAUNCH_GATE)
    schema_report = load_report(PROPOSED_SCHEMA_PATH)
    inputs = {
        "circadian_feature_generation_dryrun": dryrun,
        "no_approval_frontier": frontier,
        "local_smoke_freshness": local_smoke,
        "prelaunch_gate": prelaunch,
        "existing_feature_matrix_schema_report": schema_report,
    }
    input_contract = {
        name: bool(report.get("exists") and report.get("parse_ok"))
        for name, report in inputs.items()
        if name != "existing_feature_matrix_schema_report"
    }
    dryrun_checks = dryrun_contract(dryrun)
    frontier_checks = frontier_contract(frontier)
    smoke_checks = smoke_contract(local_smoke, prelaunch)
    existing_output_checks = existing_registered_output_contract(schema_report)
    existing_registered_output = bool(all(existing_output_checks.values()))
    proposed_output_absent = not PROPOSED_MATRIX_PATH.exists()
    proposed_schema_absent = not PROPOSED_SCHEMA_PATH.exists()
    raw_path_checks = path_contract()
    path_checks = {
        "proposed_matrix_path_is_unique": True,
        "proposed_schema_path_is_unique": True,
        "proposed_matrix_absent_or_registered_existing": proposed_output_absent or existing_registered_output,
        "proposed_schema_absent_or_registered_existing": proposed_schema_absent or existing_registered_output,
        "proposed_matrix_under_data_processed": raw_path_checks["proposed_matrix_under_data_processed"],
        "proposed_schema_under_reports": raw_path_checks["proposed_schema_under_reports"],
        "proposed_matrix_name_is_source_specific": raw_path_checks["proposed_matrix_name_is_source_specific"],
        "proposed_schema_name_is_source_specific": raw_path_checks["proposed_schema_name_is_source_specific"],
    }
    contract_ready = bool(
        all(input_contract.values())
        and all(dryrun_checks.values())
        and all(frontier_checks.values())
        and all(smoke_checks.values())
        and all(path_checks.values())
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Freeze the contract for a future circadian HR/light feature-matrix writer while keeping the current "
            "turn report-only, no-model, no-submission, and no-Kaggle-network."
        ),
        "inputs": inputs,
        "checks": {
            "input_contract": input_contract,
            "dryrun_contract": dryrun_checks,
            "frontier_contract": frontier_checks,
            "smoke_contract": smoke_checks,
            "existing_registered_output_contract": existing_output_checks,
            "path_contract": path_checks,
        },
        "allowed_future_write_contract": {
            "allowed_matrix_path": safe_relative(PROPOSED_MATRIX_PATH),
            "allowed_schema_report_path": safe_relative(PROPOSED_SCHEMA_PATH),
            "allowed_rows_exact": 700,
            "allowed_feature_columns_exact": 48,
            "allowed_feature_columns_max": 56,
            "required_key_columns": ["subject_id", "lifelog_date", "split"],
            "required_feature_family": SOURCE_FAMILY,
            "required_target_focus": TARGET_FOCUS,
            "must_revalidate_after_write": True,
            "must_not_include_target_labels": True,
            "must_not_include_oof_predictions": True,
            "must_not_include_submission_predictions": True,
            "must_not_train_model": True,
            "must_not_create_submission_artifact": True,
            "must_not_modify_submission_log": True,
            "must_not_run_kaggle": True,
            "must_not_submit_dacon": True,
        },
        "decision": {
            "feature_matrix_smoke_contract_ready": contract_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "feature_matrix_write_allowed_now": False,
            "feature_matrix_write_allowed_next_under_contract": contract_ready and not existing_registered_output,
            "feature_matrix_already_materialized_under_contract": existing_registered_output,
            "allowed_future_matrix_path": safe_relative(PROPOSED_MATRIX_PATH),
            "allowed_future_schema_report_path": safe_relative(PROPOSED_SCHEMA_PATH),
            "public_058_evidence": False,
            "score_route_opened": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": (
                "validate_circadian_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
                if existing_registered_output
                else "write_circadian_feature_matrix_under_contract_no_model_no_submission"
                if contract_ready
                else "fix_circadian_feature_matrix_contract_inputs"
            ),
            "next_action": (
                "The registered circadian feature matrix already exists and passed schema validation. Stop here "
                "unless a separate model-smoke contract is opened."
                if existing_registered_output
                else (
                    "A future writer may create only the registered circadian feature matrix and schema report, "
                    "then must stop for validation. No model, ID342, Kaggle run, or DACON submission is allowed."
                )
                if contract_ready
                else "Do not write a circadian feature matrix until this contract is repaired."
            ),
        },
        "no_feature_matrix_written": True,
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
    checks = report["checks"]
    future = report["allowed_future_write_contract"]
    lines = [
        "# Circadian HR/light feature-matrix smoke contract — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It writes no feature matrix/parquet, predictions, submission CSV, "
            "base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model, "
            "or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Feature-matrix smoke contract ready: `{dec['feature_matrix_smoke_contract_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Feature matrix write allowed now: `{dec['feature_matrix_write_allowed_now']}`",
        f"- Feature matrix write allowed next under contract: `{dec['feature_matrix_write_allowed_next_under_contract']}`",
        f"- Feature matrix already materialized under contract: `{dec['feature_matrix_already_materialized_under_contract']}`",
        f"- Allowed future matrix path: `{dec['allowed_future_matrix_path']}`",
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
        "## Contract checks",
        "",
    ]
    for group, group_checks in checks.items():
        lines.append(f"### {group}")
        lines.append("")
        for key, value in group_checks.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Allowed future write boundary",
            "",
            "```json",
            json.dumps(future, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Boundary",
            "",
            (
                "Passing this contract only permits a future no-model feature-matrix writer for the registered "
                "circadian HR/light file. It is not leaderboard evidence, does not create ID342, and does not bypass "
                "the exact Kaggle approval phrase."
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract_20260619.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    write_report(report, output)
    dec = report["decision"]
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(
        json.dumps(
            {
                "feature_matrix_smoke_contract_ready": dec["feature_matrix_smoke_contract_ready"],
                "source_family": dec["source_family"],
                "feature_matrix_write_allowed_now": dec["feature_matrix_write_allowed_now"],
                "feature_matrix_write_allowed_next_under_contract": dec[
                    "feature_matrix_write_allowed_next_under_contract"
                ],
                "feature_matrix_already_materialized_under_contract": dec[
                    "feature_matrix_already_materialized_under_contract"
                ],
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
    return 0 if dec["feature_matrix_smoke_contract_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
