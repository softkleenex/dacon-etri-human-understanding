#!/usr/bin/env python3
"""Post-ID342 subject-state heavy success-threshold contract.

This report-only diagnostic defines how a future approved no-submit heavy run
will be judged. It creates no runner files, reads no raw payload rows, writes no
features, fits no models, pushes/runs no Kaggle jobs, edits no submission logs,
and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_subject_state_heavy_success_thresholds_20260624.json"
KST = timezone(timedelta(hours=9))

HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"

INPUT_REPORTS = {
    "goal_state_audit": REPORTS_DIR / "post_id342_goal_state_audit_20260624.json",
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json",
    "public_alignment_lens": REPORTS_DIR / "post_id342_public_alignment_validation_lens_20260624.json",
    "public_response_geometry": REPORTS_DIR / "post_id342_public_response_geometry_20260624.json",
    "heavy_approval_readiness": REPORTS_DIR / "post_id342_subject_state_heavy_approval_readiness_20260624.json",
    "runner_spec": REPORTS_DIR / "post_id342_self_supervised_heavy_runner_spec_20260624.json",
    "docs_records_audit": REPORTS_DIR / "docs_records_audit.json",
}

TARGETS = ["Q2", "Q3", "S3"]
PRIMARY_SCORE_TARGETS = ["Q3", "S3"]

THRESHOLD_TIERS = [
    {
        "tier": "reject",
        "public_proxy_max": None,
        "artifact_review_allowed": False,
        "dacon_candidate_allowed": False,
        "rules": [
            "Any embedding purity failure.",
            "Any submission artifact/log row created by the no-submit runner.",
            "No Q3/S3 stable-lens improvement versus ID342.",
            "Improvement appears only in failed-frontier-inclusive lenses.",
        ],
    },
    {
        "tier": "research_lead_only",
        "public_proxy_max": 0.5960832519,
        "artifact_review_allowed": False,
        "dacon_candidate_allowed": False,
        "rules": [
            "At least one of Q3/S3 beats ID342 on stable failed-frontier-excluded lenses.",
            "Projected public proxy is below ID342 but still above 0.58.",
            "Record as source evidence only; do not create artifacts or submit.",
        ],
    },
    {
        "tier": "artifact_review_candidate",
        "public_proxy_max": 0.58,
        "artifact_review_allowed": True,
        "dacon_candidate_allowed": False,
        "rules": [
            "Q3 or S3 has material stable-lens gain with worst primary split non-degrading.",
            "Other primary score target is neutral or improving.",
            "Projected public proxy is at or below 0.58.",
            "Remote report proves embedding purity, privacy guards, and artifact separation.",
        ],
    },
    {
        "tier": "dacon_candidate_requires_fresh_exact_file_memo_approval",
        "public_proxy_max": 0.55,
        "artifact_review_allowed": True,
        "dacon_candidate_allowed": True,
        "rules": [
            "Projected public proxy is at or below the target 0.55 zone.",
            "Movement is not a post-public ID342-neighborhood tweak.",
            "File/ipynb/result/log matching can be proven for the exact artifact.",
            "DACON still requires exact file and memo approval before submission.",
        ],
    },
]

MANDATORY_REMOTE_REPORTS = [
    "post_id342_subject_state_remote_manifest.json",
    "post_id342_subject_state_embedding_purity.json",
    "post_id342_subject_state_token_coverage.json",
    "post_id342_subject_state_downstream_smoke.json",
    "post_id342_subject_state_failed_frontier_resistance.json",
    "post_id342_subject_state_public_gap_budget.json",
    "post_id342_subject_state_artifact_separation.json",
]

STOP_RULES = [
    {
        "name": "embedding_purity",
        "stop_if": "Any target label, OOF/base prediction, public score, or test-label proxy is used during pretraining.",
    },
    {
        "name": "privacy_guard",
        "stop_if": "Raw app names, BSSID, BLE device IDs, or GPS coordinates are emitted in review artifacts.",
    },
    {
        "name": "id342_stable_lens_failure",
        "stop_if": "Q3 and S3 both fail to beat ID342 on stable failed-frontier-excluded lenses.",
    },
    {
        "name": "failed_frontier_dependence",
        "stop_if": "The apparent gain is driven by ID287/ID329-inclusive lenses or failed-direction geometry.",
    },
    {
        "name": "public_gap_not_material",
        "stop_if": "Projected public proxy stays in the ID342 0.596 band.",
    },
    {
        "name": "artifact_leak",
        "stop_if": "No-submit runner creates submission CSVs, base-test predictions, notebook-history artifacts, or log rows.",
    },
]

MINIMUM_DOWNSTREAM_METRICS = {
    "q3_or_s3_mean_stable_lens_delta_vs_id342_max": -0.001,
    "primary_worst_split_delta_vs_id342_max": 0.0005,
    "other_primary_target_delta_vs_id342_max": 0.00025,
    "failed_frontier_excluded_required": True,
    "failed_frontier_inclusive_report_required": True,
    "recent_slate_order_preservation_required": True,
}

FORBIDDEN_ACTIONS_NOW = {
    "create_runner_files": False,
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_downstream_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "submit_dacon": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
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
        "decision": payload.get("decision", {}),
        "findings": payload.get("findings", []),
    }


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    goal = reports["goal_state_audit"].get("decision", {})
    synthesis = reports["remaining_route_synthesis"].get("decision", {})
    alignment = reports["public_alignment_lens"].get("decision", {})
    readiness = reports["heavy_approval_readiness"].get("decision", {})
    runner = reports["runner_spec"].get("decision", {})
    docs_findings = reports["docs_records_audit"].get("findings", [])
    return {
        "all_input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "goal_not_achieved": goal.get("goal_achieved") is False and goal.get("public_055_reached") is False,
        "id342_anchor_current": synthesis.get("current_public_anchor_id") == 342
        and synthesis.get("current_public_score") == 0.5960832519,
        "stable_lens_policy_ready": alignment.get("validation_repair_required") is True
        and bool(alignment.get("trusted_lens_candidates")),
        "heavy_readiness_ready_but_execution_closed": readiness.get("heavy_approval_readiness_audit_ready") is True
        and readiness.get("run_kaggle_now") is False,
        "runner_spec_ready": runner.get("runner_spec_ready") is True,
        "docs_audit_clean": reports["docs_records_audit"].get("parse_ok") is True
        and isinstance(docs_findings, list)
        and not docs_findings,
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in INPUT_REPORTS.items()}
    prerequisites = prerequisite_status(reports)
    ready = all(prerequisites.values())
    synthesis_decision = reports["remaining_route_synthesis"].get("decision", {})
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_heavy_success_thresholds",
        "purpose": "Define pass/fail and promotion thresholds for a future approved no-submit heavy run.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "current_anchor": {
            "id": synthesis_decision.get("current_public_anchor_id"),
            "public_score": synthesis_decision.get("current_public_score"),
            "target_public_score": synthesis_decision.get("target_public_score"),
            "gap_to_public_055": synthesis_decision.get("gap_to_public_055"),
        },
        "target_focus": TARGETS,
        "primary_score_targets": PRIMARY_SCORE_TARGETS,
        "minimum_downstream_metrics": MINIMUM_DOWNSTREAM_METRICS,
        "threshold_tiers": THRESHOLD_TIERS,
        "mandatory_remote_reports": MANDATORY_REMOTE_REPORTS,
        "stop_rules": STOP_RULES,
        "decision": {
            "heavy_success_threshold_contract_ready": ready,
            "requires_exact_heavy_compute_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "create_runner_files_now": False,
            "read_raw_parquet_payloads_now": False,
            "write_token_cache_now": False,
            "write_feature_matrix_now": False,
            "fit_embedding_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "dacon_submit_now": False,
            "artifact_review_threshold_public_proxy": 0.58,
            "dacon_candidate_threshold_public_proxy": 0.55,
            "next_action": (
                "Use these thresholds to review a future approved no-submit heavy run; do not execute now."
                if ready
                else "Refresh prerequisite reports before using these thresholds."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_runner_files_created": True,
        "no_raw_payload_rows_read": True,
        "no_token_cache_created": True,
        "no_feature_matrix_created": True,
        "no_embedding_fit": True,
        "no_prediction_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 subject-state heavy success thresholds",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report defines how a future approved no-submit heavy run will be judged. It creates no runner, token cache, feature matrix, embedding, prediction, Kaggle run, artifact, log row, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Threshold contract ready: `{decision['heavy_success_threshold_contract_ready']}`",
        f"- Artifact-review public proxy threshold: `{decision['artifact_review_threshold_public_proxy']}`",
        f"- DACON-candidate public proxy threshold: `{decision['dacon_candidate_threshold_public_proxy']}`",
        f"- Create runner files now: `{decision['create_runner_files_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Required phrase: `{decision['heavy_approval_phrase']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Current Anchor",
        "",
        "```json",
        json.dumps(report["current_anchor"], indent=2, sort_keys=True),
        "```",
        "",
        "## Minimum Downstream Metrics",
        "",
        "```json",
        json.dumps(report["minimum_downstream_metrics"], indent=2, sort_keys=True),
        "```",
        "",
        "## Threshold Tiers",
        "",
        "| tier | proxy max | artifact review | DACON candidate | rules |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["threshold_tiers"]:
        lines.append(
            f"| `{row['tier']}` | `{row['public_proxy_max']}` | `{row['artifact_review_allowed']}` | "
            f"`{row['dacon_candidate_allowed']}` | {'; '.join(row['rules'])} |"
        )
    lines.extend(
        [
            "",
            "## Mandatory Remote Reports",
            "",
        ]
    )
    for item in report["mandatory_remote_reports"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Stop Rules",
            "",
            "| rule | stop if |",
            "|---|---|",
        ]
    )
    for row in report["stop_rules"]:
        lines.append(f"| `{row['name']}` | {row['stop_if']} |")
    lines.extend(
        [
            "",
            "## Prerequisites",
            "",
            "```json",
            json.dumps(report["prerequisites"], indent=2, sort_keys=True),
            "```",
            "",
            "## Forbidden Actions Now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Threshold contract ready: {report['decision']['heavy_success_threshold_contract_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
