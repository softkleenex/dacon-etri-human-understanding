#!/usr/bin/env python3
"""Report-only contract for the circadian HR/light model-smoke step.

The circadian feature matrix is now materialized, but it is not score evidence.
This diagnostic decides whether a future tiny fold-internal model smoke may be
run under a narrow no-artifact contract. It does not train a model, create OOF
or test predictions, create a submission CSV, modify the submission log, run
Kaggle, or submit to DACON.
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
SCORE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
MATRIX_PATH = ROOT / "data" / "processed" / "post_id341_circadian_hr_light_phase_features_v1.parquet"
SCHEMA_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_features_v1_schema.json"
FEATURE_MATRIX_CONTRACT = (
    REPORTS_DIR / "post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract_20260619.json"
)
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
NO_APPROVAL_FRONTIER = REPORTS_DIR / "post_id341_no_approval_frontier_20260619.json"
PUBLIC_LOCAL_GAP = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"
LOCAL_SMOKE_FRESHNESS = REPORTS_DIR / "post_id341_local_smoke_freshness_gate_20260618.json"
MODEL_SMOKE_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_model_smoke_20260619.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_model_smoke_contract_20260619.json"

FORBIDDEN_DECISION_KEYS = [
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
    "train_model_now",
]


def safe_relative(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
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


def safety(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("safety_contract", {})
    return value if isinstance(value, dict) else {}


def no_forbidden_action(report: dict[str, Any]) -> bool:
    dec = decision(report)
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def schema_contract(schema: dict[str, Any]) -> dict[str, bool]:
    dec = decision(schema)
    artifact = schema.get("artifact", {}) if isinstance(schema.get("artifact"), dict) else {}
    validation = schema.get("validation", {}) if isinstance(schema.get("validation"), dict) else {}
    safe = safety(schema)
    return {
        "schema_exists_and_parse_ok": schema.get("exists") is True and schema.get("parse_ok") is True,
        "schema_diagnostic_matches": schema.get("diagnostic")
        == "post_id341_circadian_hr_light_phase_features_v1_schema",
        "schema_source_family_matches": dec.get("source_family") == SOURCE_FAMILY,
        "schema_target_focus_matches": dec.get("target_focus") == TARGET_FOCUS,
        "schema_matrix_materialized": dec.get("feature_matrix_materialized") is True,
        "schema_boundary_passed": dec.get("allowed_boundary_passed") is True
        and validation.get("allowed_boundary_passed") is True,
        "schema_row_count_exact": dec.get("row_count") == 700,
        "schema_feature_column_count_exact": dec.get("feature_column_count") == 48,
        "schema_matrix_path_matches": artifact.get("matrix_path") == safe_relative(MATRIX_PATH),
        "schema_matrix_exists": artifact.get("matrix_exists") is True and MATRIX_PATH.exists(),
        "schema_score_actions_closed": no_forbidden_action(schema)
        and dec.get("public_058_evidence") is False
        and dec.get("score_route_opened") is False,
        "schema_safety_flags_closed": safe.get("predictions_created") is False
        and safe.get("submission_csv_created") is False
        and safe.get("submission_log_modified") is False
        and safe.get("model_trained") is False
        and safe.get("kaggle_run_executed") is False
        and safe.get("dacon_submission_executed") is False
        and safe.get("uses_target_labels") is False
        and safe.get("uses_oof_predictions") is False
        and safe.get("uses_public_scores") is False,
    }


def feature_matrix_contract(matrix_contract: dict[str, Any]) -> dict[str, bool]:
    dec = decision(matrix_contract)
    return {
        "feature_matrix_contract_exists_and_parse_ok": matrix_contract.get("exists") is True
        and matrix_contract.get("parse_ok") is True,
        "feature_matrix_contract_diagnostic_matches": matrix_contract.get("diagnostic")
        == "post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract",
        "feature_matrix_contract_ready": dec.get("feature_matrix_smoke_contract_ready") is True,
        "feature_matrix_materialized_under_contract": dec.get("feature_matrix_already_materialized_under_contract")
        is True,
        "feature_matrix_write_closed_now": dec.get("feature_matrix_write_allowed_now") is False
        and dec.get("feature_matrix_write_allowed_next_under_contract") is False,
        "feature_matrix_contract_source_matches": dec.get("source_family") == SOURCE_FAMILY,
        "feature_matrix_contract_target_matches": dec.get("target_focus") == TARGET_FOCUS,
        "feature_matrix_contract_actions_closed": no_forbidden_action(matrix_contract)
        and dec.get("public_058_evidence") is False
        and dec.get("score_route_opened") is False,
    }


def queue_contract(queue: dict[str, Any]) -> dict[str, bool]:
    dec = decision(queue)
    items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    circadian_item = next((item for item in items if item.get("name") == SOURCE_FAMILY), {})
    allowed_steps = {
        "report_only_model_smoke_contract_no_model_no_submission",
        "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact",
        "report_only_usage_semantic_burst_markov_design_no_model",
        "usage_semantic_burst_markov_metadata_cardinality_smoke_no_features",
        "report_only_usage_semantic_burst_markov_feature_spec_no_matrix",
        "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix",
        "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract",
        "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
        "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
    }
    allowed_circadian_statuses = {
        "feature_matrix_materialized_under_contract_no_model",
        "model_smoke_contract_ready_run_only_under_contract",
        "model_smoke_completed_review_needed_no_artifact",
        "closed_after_feature_matrix_and_model_smoke_no_signal",
    }
    return {
        "queue_exists_and_parse_ok": queue.get("exists") is True and queue.get("parse_ok") is True,
        "queue_ready": dec.get("material_new_source_queue_ready") is True,
        "queue_contains_circadian_source": circadian_item.get("name") == SOURCE_FAMILY,
        "queue_circadian_status_valid": circadian_item.get("status") in allowed_circadian_statuses,
        "queue_step_allows_contract_or_future_smoke": dec.get("next_no_approval_step_type") in allowed_steps,
        "queue_no_local_score_route": dec.get("local_no_approval_score_route_exists") is False,
        "queue_score_route_requires_explicit_kaggle": dec.get("only_score_relevant_open_route") == SCORE_ROUTE
        and dec.get("requires_explicit_kaggle_approval") is True,
        "queue_public_058_false": dec.get("public_058_evidence") is False,
        "queue_actions_closed": no_forbidden_action(queue),
    }


def frontier_contract(frontier: dict[str, Any]) -> dict[str, bool]:
    dec = decision(frontier)
    return {
        "frontier_exists_and_parse_ok": frontier.get("exists") is True and frontier.get("parse_ok") is True,
        "frontier_ready": dec.get("no_approval_frontier_ready") is True,
        "frontier_local_score_routes_closed": dec.get("all_local_no_approval_score_routes_closed") is True,
        "frontier_no_approval_score_route_absent": dec.get("no_approval_score_route_exists") is False,
        "frontier_only_score_route_is_explicit_kaggle": dec.get("only_score_relevant_open_route") == SCORE_ROUTE,
        "frontier_public_058_not_reviewable": dec.get("public_058_reviewable") is False,
        "frontier_actions_closed": no_forbidden_action(frontier),
    }


def public_gap_contract(public_gap: dict[str, Any]) -> dict[str, bool]:
    dec = decision(public_gap)
    return {
        "public_gap_exists_and_parse_ok": public_gap.get("exists") is True and public_gap.get("parse_ok") is True,
        "public_gap_ready": dec.get("public_local_gap_diagnosis_ready") is True,
        "no_watch_candidate_passes_floor": dec.get("any_watch_candidate_passes_public_058_recent_floor_proxy") is False,
        "no_watch_candidate_passes_stress": dec.get("any_watch_candidate_passes_public_058_recent_stress_proxy")
        is False,
        "public_gap_actions_closed": no_forbidden_action(public_gap),
    }


def local_smoke_contract(local_smoke: dict[str, Any]) -> dict[str, bool]:
    dec = decision(local_smoke)
    return {
        "local_smoke_exists_and_parse_ok": local_smoke.get("exists") is True and local_smoke.get("parse_ok") is True,
        "local_runner_smoke_fresh": dec.get("local_runner_smoke_fresh") is True,
        "local_runner_smoke_rerun_not_required": dec.get("requires_local_runner_smoke_rerun") is False,
        "local_smoke_actions_closed": no_forbidden_action(local_smoke),
    }


def existing_model_smoke_contract(model_smoke: dict[str, Any]) -> dict[str, bool]:
    dec = decision(model_smoke)
    safe = safety(model_smoke)
    return {
        "model_smoke_exists_and_parse_ok": model_smoke.get("exists") is True and model_smoke.get("parse_ok") is True,
        "model_smoke_diagnostic_matches": model_smoke.get("diagnostic")
        == "post_id341_circadian_hr_light_phase_model_smoke",
        "model_smoke_completed": dec.get("model_smoke_completed") is True,
        "model_smoke_source_family_matches": dec.get("source_family") == SOURCE_FAMILY,
        "model_smoke_target_focus_matches": dec.get("target_focus") == TARGET_FOCUS,
        "model_smoke_actions_closed": dec.get("public_058_evidence") is False
        and dec.get("score_route_opened") is False
        and dec.get("create_submission_artifact") is False
        and dec.get("submit_now") is False
        and dec.get("push_kaggle_now") is False
        and dec.get("run_kaggle_now") is False,
        "model_smoke_safety_flags_closed": safe.get("persisted_model_created") is False
        and safe.get("full_test_predictions_created") is False
        and safe.get("submission_csv_created") is False
        and safe.get("submission_log_modified") is False
        and safe.get("kaggle_run_executed") is False
        and safe.get("dacon_submission_executed") is False
        and safe.get("uses_public_scores") is False,
    }


def build_report() -> dict[str, Any]:
    schema = load_report(SCHEMA_REPORT)
    matrix_contract = load_report(FEATURE_MATRIX_CONTRACT)
    queue = load_report(QUEUE_REPORT)
    frontier = load_report(NO_APPROVAL_FRONTIER)
    public_gap = load_report(PUBLIC_LOCAL_GAP)
    local_smoke = load_report(LOCAL_SMOKE_FRESHNESS)
    model_smoke = load_report(MODEL_SMOKE_REPORT)
    existing_model_smoke_checks = existing_model_smoke_contract(model_smoke)
    existing_model_smoke_completed = bool(all(existing_model_smoke_checks.values()))
    existing_model_smoke_absent = model_smoke.get("exists") is False

    checks = {
        "schema_contract": schema_contract(schema),
        "feature_matrix_contract": feature_matrix_contract(matrix_contract),
        "queue_contract": queue_contract(queue),
        "frontier_contract": frontier_contract(frontier),
        "public_gap_contract": public_gap_contract(public_gap),
        "local_smoke_contract": local_smoke_contract(local_smoke),
        "existing_model_smoke_contract": existing_model_smoke_checks,
    }
    blocking_check_groups = {key: value for key, value in checks.items() if key != "existing_model_smoke_contract"}
    existing_model_smoke_state_ok = existing_model_smoke_absent or existing_model_smoke_completed
    contract_ready = (
        all(all(group.values()) for group in blocking_check_groups.values()) and existing_model_smoke_state_ok
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_circadian_hr_light_phase_model_smoke_contract",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Open only a future bounded, fold-internal, in-memory model-smoke review for the registered circadian "
            "feature matrix. This contract itself trains no model and creates no score/submission artifact."
        ),
        "inputs": {
            "feature_matrix": safe_relative(MATRIX_PATH),
            "schema_report": schema,
            "feature_matrix_contract": matrix_contract,
            "material_source_queue": queue,
            "no_approval_frontier": frontier,
            "public_local_gap": public_gap,
            "local_smoke_freshness": local_smoke,
            "existing_model_smoke": model_smoke,
        },
        "checks": checks,
        "allowed_future_model_smoke_contract": {
            "allowed_future_script": "scripts/diagnose_post_id341_circadian_hr_light_phase_model_smoke.py",
            "allowed_matrix_path": safe_relative(MATRIX_PATH),
            "allowed_schema_report_path": safe_relative(SCHEMA_REPORT),
            "allowed_targets": TARGET_FOCUS,
            "allowed_anchor_ids": [178, 341],
            "allowed_train_rows_exact": 450,
            "allowed_test_rows_present_but_unused": 250,
            "allowed_feature_columns_exact": 48,
            "allowed_model_scope": "tiny_fold_internal_in_memory_only",
            "allowed_model_families": [
                "fixed_logistic_or_ridge_like_linear_smoke",
                "fixed_hist_gradient_boosting_smoke",
            ],
            "must_use_predeclared_splits": True,
            "must_fit_preprocessing_inside_each_outer_train_fold": True,
            "must_compare_against_id178_and_id341_anchors": True,
            "must_report_targetwise_split_losses": True,
            "must_not_use_test_labels": True,
            "must_not_create_full_test_predictions": True,
            "must_not_write_oof_predictions": True,
            "must_not_persist_model": True,
            "must_not_create_submission_artifact": True,
            "must_not_modify_submission_log": True,
            "must_not_run_kaggle": True,
            "must_not_submit_dacon": True,
            "promotion_gate": (
                "A later smoke may select targets only if it beats ID178/ID341 under frozen primary splits without "
                "opening a local-CV-only Public 0.58 claim; any submission artifact still requires a separate contract."
            ),
        },
        "decision": {
            "model_smoke_contract_ready": contract_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "model_smoke_run_allowed_now": False,
            "model_smoke_run_allowed_next_under_contract": contract_ready and not existing_model_smoke_completed,
            "model_smoke_already_completed_under_contract": existing_model_smoke_completed,
            "allowed_future_model_smoke_script": "scripts/diagnose_post_id341_circadian_hr_light_phase_model_smoke.py",
            "public_058_evidence": False,
            "score_route_opened": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": (
                "close_circadian_phase_model_smoke_or_continue_usage_design"
                if existing_model_smoke_completed
                else "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact"
                if contract_ready
                else "fix_circadian_model_smoke_contract_inputs"
            ),
            "next_action": (
                "The bounded in-memory circadian model smoke already ran under contract. Do not rerun by default; "
                "continue only with the next queued report-only source design."
                if existing_model_smoke_completed
                else (
                    "A future script may run only the bounded in-memory circadian model smoke and then stop. No ID342, "
                    "submission artifact, Kaggle run, or DACON submission is allowed."
                )
                if contract_ready
                else "Do not run a circadian model smoke until this contract is repaired."
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
    future = report["allowed_future_model_smoke_contract"]
    lines = [
        "# Circadian HR/light model-smoke contract — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It trains no model and creates no predictions, OOF file, submission CSV, "
            "base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Model-smoke contract ready: `{dec['model_smoke_contract_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Model-smoke run allowed now: `{dec['model_smoke_run_allowed_now']}`",
        f"- Model-smoke run allowed next under contract: `{dec['model_smoke_run_allowed_next_under_contract']}`",
        f"- Model-smoke already completed under contract: `{dec['model_smoke_already_completed_under_contract']}`",
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
    for group, values in report["checks"].items():
        lines.append(f"### {group}")
        lines.append("")
        for key, value in values.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Allowed future model-smoke boundary",
            "",
            "```json",
            json.dumps(future, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Boundary",
            "",
            (
                "Passing this contract does not create ID342 and does not authorize a submission. It only allows a "
                "later bounded, in-memory fold-smoke report over the registered label-free circadian matrix."
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
                "model_smoke_contract_ready": dec["model_smoke_contract_ready"],
                "source_family": dec["source_family"],
                "model_smoke_run_allowed_now": dec["model_smoke_run_allowed_now"],
                "model_smoke_run_allowed_next_under_contract": dec["model_smoke_run_allowed_next_under_contract"],
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
    return 0 if dec["model_smoke_contract_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
