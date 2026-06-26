#!/usr/bin/env python3
"""Route the missing Q2/Q3/S3 target gap after the post-ID341 closures.

This is a report-only diagnostic. It reads existing evidence and decides which
source family should be pursued next for the missing targets. It does not train
models, create predictions, create submission artifacts, push/run Kaggle, or
submit to DACON.
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

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q2q3s3_source_route_20260618.json"

REPORTS = {
    "missing_target_inventory": "post_id341_missing_target_signal_inventory_20260618.json",
    "target_gap_budget": "post_id341_target_gap_budget_20260618.json",
    "local_source_closure": "post_id341_local_source_closure_synthesis_20260618.json",
    "local_smoke_freshness": "post_id341_local_smoke_freshness_gate_20260618.json",
    "new_source_blueprint": "post_id341_new_source_blueprint_20260618.json",
    "clean_backbone_closure": "post_id341_clean_backbone_lane_closure_20260618.json",
    "phase2_block_a_gap": "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "phase2_block_b_s2_recovery": "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "phase3_usage_temporal": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "place_social": "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "external_prior": "external_etri_prior_smoke_post_id341_20260618.json",
    "combined_external": "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
}

MISSING_TARGETS = ["Q2", "Q3", "S3"]


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rel_report(name: str) -> str:
    return str((REPORTS_DIR / name).relative_to(ROOT))


def target_inventory_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = inventory.get("target_inventory", [])
    if isinstance(rows, list):
        return rows
    if isinstance(rows, dict):
        return list(rows.values())
    return []


def inventory_by_target(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in target_inventory_rows(inventory):
        target = row.get("target")
        if target:
            out[str(target)] = row
    return out


def route_table(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    inventory_rows = inventory_by_target(reports["missing_target_inventory"])
    upstream_rows = [inventory_rows.get(target, {}) for target in MISSING_TARGETS]
    id226_fold_wins = [row.get("ID226_fold_wins") or row.get("id226_fold_wins") for row in upstream_rows]
    id226_deltas = [row.get("ID226_delta_vs_ID178") or row.get("id226_delta_vs_id178") for row in upstream_rows]

    phase3_usage = reports["phase3_usage_temporal"].get("decision", {})
    place = reports["place_social"].get("decision", {})
    sensor = reports["sensor_adherence"].get("decision", {})
    external_prior = reports["external_prior"].get("decision", {})
    combined_external = reports["combined_external"].get("decision", {})
    block_a_gap = reports["phase2_block_a_gap"].get("decision", {})
    block_b = reports["phase2_block_b_s2_recovery"].get("decision", {})

    return [
        {
            "rank": 1,
            "route": "explicit_approved_clean_upstream_id221_to_id226",
            "target_coverage": MISSING_TARGETS,
            "evidence": {
                "missing_targets_have_upstream_backbone_signal": True,
                "id226_fold_wins": id226_fold_wins,
                "id226_delta_vs_id178": id226_deltas,
            },
            "why": (
                "The only current evidence that covers all missing Q2/Q3/S3 targets is the historical clean "
                "ID221→ID226 upstream backbone surface. It must be rebuilt report-first before artifact review."
            ),
            "blocking_condition": "requires_exact_user_approval_for_kaggle_no_submit",
            "allowed_now_without_approval": False,
            "creates_submission": False,
        },
        {
            "rank": 2,
            "route": "local_or_kaggle_q2q3s3_env_social_routine_v2_design",
            "target_coverage": MISSING_TARGETS,
            "evidence": {
                "usage_temporal_selected_targets": phase3_usage.get("selected_targets", []),
                "place_social_selected_targets": place.get("selected_targets", []),
                "sensor_adherence_selected_targets": sensor.get("selected_targets", []),
            },
            "why": (
                "If approval is not given, the only locally meaningful next design is a materially different "
                "Q2/Q3/S3-first environment/social/routine source. Existing usage/place/sensor standalone smokes "
                "selected no targets, so this must be a new pre-registered source design, not wider local tuning."
            ),
            "blocking_condition": "previous_standalone_local_smokes_selected_no_targets",
            "allowed_now_without_approval": True,
            "creates_submission": False,
        },
        {
            "rank": 3,
            "route": "external_prior_regularizer_only",
            "target_coverage": ["Q2"],
            "evidence": {
                "external_prior_reviewable": external_prior.get(
                    "external_prior_reviewable_low_movement_candidate_exists"
                )
                or external_prior.get("reviewable_candidate_exists"),
                "combined_external_public_058": combined_external.get("public_058_plausible"),
                "combined_external_best_proxy": combined_external.get("best_proxy"),
            },
            "why": (
                "External ETRI files are useful as weak priors or bin-boundary regularizers, but current evidence "
                "blocks direct concatenation and does not produce a Public-0.58-reviewable route."
            ),
            "blocking_condition": "external_direct_concat_blocked_and_proxy_too_high",
            "allowed_now_without_approval": True,
            "creates_submission": False,
        },
        {
            "rank": 4,
            "route": "more_q1_s4_or_s2_local_tuning",
            "target_coverage": ["Q1", "S2", "S4"],
            "evidence": {
                "q1_all_primary_signal_confirmed": block_a_gap.get("q1_all_primary_signal_confirmed"),
                "q1_signal_sufficient_for_id178_proxy": block_a_gap.get("q1_signal_sufficient_for_id178_proxy"),
                "additional_independent_target_signal_required": block_a_gap.get(
                    "additional_independent_target_signal_required"
                ),
                "s2_final_block_b_gate_pass": block_b.get("final_block_b_gate_pass"),
            },
            "why": (
                "Q1/S4/S2 evidence cannot close the missing-target budget by itself. More local tuning here "
                "preserves the current failure mode."
            ),
            "blocking_condition": "does_not_cover_q2_q3_s3_gap",
            "allowed_now_without_approval": False,
            "creates_submission": False,
        },
    ]


def build_report() -> dict[str, Any]:
    reports = {key: load_report(name) for key, name in REPORTS.items()}
    missing_decision = reports["missing_target_inventory"].get("decision", {})
    gap_decision = reports["target_gap_budget"].get("decision", {})
    closure_decision = reports["local_source_closure"].get("decision", {})
    freshness_decision = reports["local_smoke_freshness"].get("decision", {})

    routes = route_table(reports)
    primary_route = routes[0]
    fallback_route = routes[1]
    missing_targets_match = missing_decision.get("focus_missing_locally") == MISSING_TARGETS
    local_closed = closure_decision.get("local_source_stack_closed_for_submission_artifact") is True
    q1_s4_insufficient = gap_decision.get("q1_s4_only_is_insufficient") is True
    remote_ready = bool(missing_targets_match and local_closed and q1_s4_insufficient)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_source_route",
        "purpose": (
            "Choose the next no-submit source route for the missing Q2/Q3/S3 target gap after local "
            "post-ID341 lanes failed to produce a promotable candidate."
        ),
        "inputs": {key: rel_report(name) for key, name in REPORTS.items()},
        "evidence_flags": {
            "missing_targets_match_q2_q3_s3": missing_targets_match,
            "q1_s4_only_is_insufficient": q1_s4_insufficient,
            "local_source_stack_closed_for_submission_artifact": local_closed,
            "local_smoke_fresh": freshness_decision.get("local_runner_smoke_fresh") is True,
            "route_ready_for_prelaunch_contract": remote_ready,
        },
        "routes": routes,
        "decision": {
            "primary_score_relevant_route": primary_route["route"],
            "fallback_no_approval_route": fallback_route["route"],
            "missing_targets": MISSING_TARGETS,
            "local_q2q3s3_artifact_allowed": False,
            "more_q1_s4_tuning_allowed": False,
            "remote_no_submit_prelaunch_ready": remote_ready,
            "requires_explicit_user_approval": True,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If the user gives the exact Kaggle approval phrase, run the guarded no-submit upstream rebuild. "
                "Without approval, only design/report a materially different Q2/Q3/S3-first env-social-routine "
                "source; do not create ID342 or tune Q1/S4 further."
            ),
        },
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    flags = report["evidence_flags"]
    lines = [
        "# Post-ID341 Q2/Q3/S3 source route",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, notebook/history artifact, Kaggle run, or DACON submission.",
        "",
        "## Evidence flags",
        "",
    ]
    for key, value in flags.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Ranked routes",
            "",
            "| rank | route | target coverage | allowed without approval | blocking condition |",
            "|---:|---|---|---|---|",
        ]
    )
    for route in report["routes"]:
        lines.append(
            f"| {route['rank']} | `{route['route']}` | `{', '.join(route['target_coverage'])}` | "
            f"`{route['allowed_now_without_approval']}` | `{route['blocking_condition']}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    for route in report["routes"]:
        lines.extend([f"### {route['rank']}. `{route['route']}`", "", route["why"], ""])
    lines.extend(
        [
            "## Decision",
            "",
            f"- Primary score-relevant route: `{decision['primary_score_relevant_route']}`",
            f"- Fallback no-approval route: `{decision['fallback_no_approval_route']}`",
            f"- Missing targets: `{', '.join(decision['missing_targets'])}`",
            f"- Local Q2/Q3/S3 artifact allowed: `{decision['local_q2q3s3_artifact_allowed']}`",
            f"- More Q1/S4 tuning allowed: `{decision['more_q1_s4_tuning_allowed']}`",
            f"- Remote no-submit prelaunch ready: `{decision['remote_no_submit_prelaunch_ready']}`",
            f"- Requires explicit user approval: `{decision['requires_explicit_user_approval']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
