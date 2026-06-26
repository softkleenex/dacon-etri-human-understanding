#!/usr/bin/env python3
"""Design-only sleep-proxy threshold residual backlog after ID341.

This diagnostic freezes a narrowly scoped, design-only backlog item for the
sleep-proxy threshold residual idea identified by the next-material-route triage.
It inspects existing processed schema coverage and overlap with closed lanes,
but it intentionally does not compute target associations, fit a model, create a
feature matrix, create predictions, edit submission_log.csv, run Kaggle, or
submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TRAIN_PROCESSED = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PROCESSED = ROOT / "data" / "processed" / "df_test_raw.parquet"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "sleep_proxy_threshold_residual_v1"
TARGET_FOCUS = ["S2", "S3", "S4", "Q1"]
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_sleep_proxy_threshold_residual_design_20260619.json"
KAGGLE_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"

REQUIRED_CONTEXT_REPORTS = {
    "safe_wait": "post_id341_safe_wait_state_contract_20260619.json",
    "next_route_triage": "post_id341_next_material_route_triage_20260619.json",
    "q3s3_remaining": "post_id341_q3s3_remaining_route_inventory_20260618.json",
    "sleep_boundary": "sleep_boundary_prepromotion_gate_post_id341_20260618.json",
    "circadian_model_smoke": "post_id341_circadian_hr_light_phase_model_smoke_20260619.json",
    "q2q3s3_paper_family": "post_id341_q2q3s3_paper_family_fold_smoke_20260619.json",
    "public_local_gap": "post_id341_public_local_gap_diagnosis_20260618.json",
}

TARGET_RULES = {
    "S2": {
        "definition": "objective sleep efficiency guideline adherence",
        "positive_class": "recommended / high sleep efficiency",
        "candidate_proxy_blocks": ["sleep_efficiency_fragmentation", "sleep_stage_duration"],
        "risk": "processed sleep-efficiency proxies already overlap with prior sleep-boundary/circadian lanes",
    },
    "S3": {
        "definition": "objective sleep onset latency guideline adherence",
        "positive_class": "recommended sleep onset latency",
        "candidate_proxy_blocks": ["sleep_stage_duration", "sleep_pedo_light"],
        "risk": "SOL is not directly observed; threshold residuals would be proxy-only and public-transfer fragile",
    },
    "S4": {
        "definition": "objective wake after sleep onset guideline adherence",
        "positive_class": "low WASO / stable sleep maintenance",
        "candidate_proxy_blocks": ["sleep_efficiency_fragmentation", "sleep_hr_microstructure"],
        "risk": "WASO-like fragmentation was already part of weak/closed sleep-family evidence",
    },
    "Q1": {
        "definition": "subjective sleep quality above subject average",
        "positive_class": "better-than-own-average subjective sleep quality",
        "candidate_proxy_blocks": ["sleep_stage_duration", "sleep_hr_microstructure", "sleep_pedo_light"],
        "risk": "subjective target may not transfer from objective sleep proxy without subject-normalized validation",
    },
}

FEATURE_BLOCK_RULES = {
    "sleep_stage_duration": ["sleep_duration", "deep", "light", "awake"],
    "sleep_efficiency_fragmentation": ["sleep_efficiency", "fragmentation"],
    "sleep_hr_microstructure": ["wHr_sleep_hr", "slp_hr", "rmssd", "spike"],
    "sleep_pedo_light": ["slp_pedo", "slp_wlight", "wlight", "mlight"],
}

FORBIDDEN_NOW = {
    "compute_target_associations": True,
    "fit_model": True,
    "write_feature_matrix": True,
    "create_predictions": True,
    "create_submission_artifact": True,
    "modify_submission_log": True,
    "create_id342_or_new_submission_id": True,
    "push_kaggle_now": True,
    "run_kaggle_now": True,
    "submit_now": True,
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_report(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
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
        "decision": payload.get("decision", {}),
        "payload": payload,
    }


def decision(reports: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    value = reports[key].get("decision", {})
    return value if isinstance(value, dict) else {}


def load_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "shape": [0, 0], "columns": []}
    df = pd.read_parquet(path)
    columns = list(df.columns)
    return {
        "path": rel(path),
        "exists": True,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": columns,
        "key_columns_present": all(col in columns for col in ["subject_id", "sleep_date", "lifelog_date"]),
        "target_columns_present": [col for col in ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"] if col in columns],
    }


def columns_for_block(columns: list[str], tokens: list[str]) -> list[str]:
    lowered = [(col, col.lower()) for col in columns]
    return sorted({col for col, low in lowered if any(token.lower() in low for token in tokens)})


def feature_block_rows(train_schema: dict[str, Any], test_schema: dict[str, Any]) -> list[dict[str, Any]]:
    train_columns = train_schema.get("columns", [])
    test_columns = test_schema.get("columns", [])
    rows: list[dict[str, Any]] = []
    for block, tokens in FEATURE_BLOCK_RULES.items():
        train_matches = columns_for_block(train_columns, tokens)
        test_matches = columns_for_block(test_columns, tokens)
        shared = sorted(set(train_matches) & set(test_matches))
        rows.append(
            {
                "block": block,
                "tokens": tokens,
                "train_column_count": len(train_matches),
                "test_column_count": len(test_matches),
                "shared_column_count": len(shared),
                "shared_columns_preview": shared[:12],
                "schema_ready": len(shared) > 0,
            }
        )
    return rows


def context_checks(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {
        "safe_wait_ready": decision(reports, "safe_wait").get("safe_wait_state_ready") is True,
        "next_route_triage_ready": decision(reports, "next_route_triage").get("next_material_route_triage_ready")
        is True,
        "sleep_backlog_feasible_but_not_open": decision(reports, "next_route_triage").get(
            "sleep_proxy_threshold_residual_backlog_feasible"
        )
        is True
        and decision(reports, "next_route_triage").get("sleep_proxy_threshold_residual_open_now") is False,
        "q3s3_local_lanes_closed": decision(reports, "q3s3_remaining").get("local_no_approval_score_route_exists")
        is False,
        "sleep_boundary_not_promotable": decision(reports, "sleep_boundary").get("promote_submission_artifact") is False
        and decision(reports, "sleep_boundary").get("create_submission_artifact") is False
        and decision(reports, "sleep_boundary").get("submit_now") is False,
        "circadian_lane_closed": decision(reports, "circadian_model_smoke").get("score_route_opened") is False,
        "paper_family_lane_closed": decision(reports, "q2q3s3_paper_family").get("any_paper_family_signal_found")
        is False,
        "public_gap_blocks_cv_only": decision(reports, "public_local_gap").get(
            "any_watch_candidate_passes_public_058_recent_floor_proxy"
        )
        is False,
    }


def design_rows(block_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready_blocks = {row["block"] for row in block_rows if row["schema_ready"]}
    rows: list[dict[str, Any]] = []
    for target, spec in TARGET_RULES.items():
        required_blocks = spec["candidate_proxy_blocks"]
        rows.append(
            {
                "target": target,
                "definition": spec["definition"],
                "positive_class": spec["positive_class"],
                "candidate_proxy_blocks": required_blocks,
                "ready_proxy_blocks": [block for block in required_blocks if block in ready_blocks],
                "missing_proxy_blocks": [block for block in required_blocks if block not in ready_blocks],
                "design_feasible": all(block in ready_blocks for block in required_blocks),
                "risk": spec["risk"],
                "future_validation_requirement": "subject/fold-nested, target-specific, no target-derived preprocessing, no public-guided candidate selection",
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    reports = {key: load_report(filename) for key, filename in REQUIRED_CONTEXT_REPORTS.items()}
    train_schema = load_schema(TRAIN_PROCESSED)
    test_schema = load_schema(TEST_PROCESSED)
    blocks = feature_block_rows(train_schema, test_schema)
    designs = design_rows(blocks)
    context = context_checks(reports)
    all_reports_ready = all(row.get("exists") and row.get("parse_ok") for row in reports.values())
    schema_ready = bool(
        train_schema.get("shape") == [450, 242]
        and test_schema.get("shape") == [250, 242]
        and train_schema.get("key_columns_present")
        and test_schema.get("key_columns_present")
    )
    all_blocks_present = all(row["schema_ready"] for row in blocks)
    any_design_feasible = any(row["design_feasible"] for row in designs)
    design_contract_ready = bool(all_reports_ready and schema_ready and all(context.values()) and all_blocks_present)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_sleep_proxy_threshold_residual_design",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Record a feasible but closed design backlog for subject-normalized sleep-proxy threshold residuals "
            "without opening a local score lane."
        ),
        "inputs": {key: {k: v for k, v in row.items() if k != "payload"} for key, row in reports.items()},
        "processed_schema": {
            "train": {k: v for k, v in train_schema.items() if k != "columns"},
            "test": {k: v for k, v in test_schema.items() if k != "columns"},
        },
        "feature_block_rows": blocks,
        "target_design_rows": designs,
        "context_checks": context,
        "forbidden_actions_now": FORBIDDEN_NOW,
        "future_open_conditions": [
            "refresh safe-wait and next-route triage first",
            "prove this design remains materially different from closed sleep-boundary/circadian lanes",
            "only run a no-artifact design/dry-run gate before any feature matrix",
            "use fold-nested target-specific validation if a future model smoke is explicitly opened",
            "do not create a submission artifact without a separate artifact contract and exact CSV approval",
        ],
        "decision": {
            "sleep_proxy_threshold_residual_design_ready": design_contract_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "metadata_feasible": bool(schema_ready and all_blocks_present and any_design_feasible),
            "design_backlog_recorded": True,
            "local_no_approval_score_route_exists": False,
            "open_design_lane_now": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "compute_target_associations_now": False,
            "public_058_evidence": False,
            "public_058_reviewable": False,
            "requires_explicit_kaggle_approval_for_score_route": True,
            "approval_phrase": KAGGLE_APPROVAL_PHRASE,
            "manual_probe_if_user_spends_submission": None,
            "manual_probe_family_status": "consumed_after_id341_id340_id337_public_micro_probe_slate",
            "create_feature_matrix": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Keep this as design backlog only. The score-relevant route remains the exact-approved Kaggle "
                "no-submit upstream rebuild."
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
    decision_row = report["decision"]
    lines = [
        "# Post-ID341 sleep-proxy threshold residual design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrix, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Design ready: `{decision_row['sleep_proxy_threshold_residual_design_ready']}`",
        f"- Metadata feasible: `{decision_row['metadata_feasible']}`",
        f"- Design backlog recorded: `{decision_row['design_backlog_recorded']}`",
        f"- Local no-approval score route exists: `{decision_row['local_no_approval_score_route_exists']}`",
        f"- Open design lane now: `{decision_row['open_design_lane_now']}`",
        f"- Feature matrix allowed now: `{decision_row['feature_matrix_allowed_now']}`",
        f"- Train model now: `{decision_row['train_model_now']}`",
        f"- Compute target associations now: `{decision_row['compute_target_associations_now']}`",
        f"- Public 0.58 reviewable: `{decision_row['public_058_reviewable']}`",
        f"- Manual probe if user spends submission: `{decision_row['manual_probe_if_user_spends_submission']}`",
        f"- Manual probe family status: `{decision_row['manual_probe_family_status']}`",
        f"- Create ID342/new submission ID: `{decision_row['create_id342_or_new_submission_id']}`",
        f"- Submit now: `{decision_row['submit_now']}`",
        f"- Push Kaggle now: `{decision_row['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_row['run_kaggle_now']}`",
        f"- Next action: {decision_row['next_action']}",
        "",
        "## Feature block schema coverage",
        "",
        "| block | shared columns | schema ready | preview |",
        "|---|---:|---:|---|",
    ]
    for row in report["feature_block_rows"]:
        preview = ", ".join(row["shared_columns_preview"][:5])
        lines.append(f"| `{row['block']}` | {row['shared_column_count']} | `{row['schema_ready']}` | `{preview}` |")
    lines.extend(["", "## Target design rows", "", "| target | feasible | proxy blocks | risk |", "|---|---:|---|---|"])
    for row in report["target_design_rows"]:
        lines.append(
            f"| `{row['target']}` | `{row['design_feasible']}` | `{','.join(row['candidate_proxy_blocks'])}` | {row['risk']} |"
        )
    lines.extend(["", "## Context checks", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in report["context_checks"].items())
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
