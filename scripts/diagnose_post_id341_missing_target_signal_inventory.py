#!/usr/bin/env python3
"""Post-ID341 missing-target signal inventory.

Report-only diagnostic. It reads existing target-gap, upstream-priority, and
source-closure reports to decide whether the missing Q2/Q3/S3 surface contains
any local promotable signal or should remain assigned to the explicit-approved
no-submit Kaggle upstream rebuild path.

It creates no submission CSV, base prediction artifact, notebook/history
artifact, submission-log row, Kaggle push/run, or DACON submission.
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

FOCUS_TARGETS = ["Q2", "Q3", "S3"]
SECONDARY_TARGETS = ["S1", "S2"]
ID178_PUBLIC = 0.6014113009
PUBLIC_058_REVIEW_CUTOFF = 0.590

TARGET_GAP_PATH = REPORTS_DIR / "post_id341_target_gap_budget_20260618.json"
UPSTREAM_PRIORITY_PATH = REPORTS_DIR / "post_id341_upstream_source_priority_matrix_20260618.json"
PHASE2_SPEC_PATH = REPORTS_DIR / "post_id341_phase2_raw_source_execution_spec_20260618.json"
DATA_COVERAGE_PATH = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"
EXTERNAL_PRIOR_PATH = REPORTS_DIR / "external_etri_prior_smoke_post_id341_20260618.json"
LOCAL_CLOSURE_PATH = REPORTS_DIR / "post_id341_local_source_closure_synthesis_20260618.json"
PRELAUNCH_PATH = REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json"


SOURCE_CLOSURE_PATHS = {
    "phase2_block_a_q1_s4_sleep_residual_bridge": REPORTS_DIR
    / "phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json",
    "phase2_block_b_s2_sleep_recovery_retry": REPORTS_DIR
    / "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "usage_temporal_microstructure": REPORTS_DIR
    / "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "routine_stability_place_entropy": REPORTS_DIR / "routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "place_social_novelty": REPORTS_DIR / "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence_dropout": REPORTS_DIR / "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "external_etri_prior": EXTERNAL_PRIOR_PATH,
    "combined_new_source_gate": REPORTS_DIR / "combined_new_source_gate_post_id341_20260618.json",
    "combined_sleep_evening_external_prior_gate": REPORTS_DIR
    / "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def rows_by_target(target_gap: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {target: [] for target in FOCUS_TARGETS + SECONDARY_TARGETS}
    for row in target_gap.get("ranked_signal_rows", []):
        target = str(row.get("target"))
        if target in out:
            out[target].append(row)
    return out


def summarize_target_gap(target_gap: dict[str, Any]) -> dict[str, Any]:
    strict_targets = set(target_gap.get("strict_budget_projection", {}).get("selected_targets", []))
    watch_targets = set(target_gap.get("watch_budget_projection", {}).get("selected_targets", []))
    row_map = rows_by_target(target_gap)
    summary: dict[str, Any] = {}
    for target in FOCUS_TARGETS + SECONDARY_TARGETS:
        rows = row_map[target]
        best_watch = rows[0] if rows else None
        summary[target] = {
            "strict_selected": target in strict_targets,
            "watch_selected": target in watch_targets,
            "local_selected_row_count": len(rows),
            "best_local_row": best_watch,
            "local_status": (
                "strict" if target in strict_targets else "watch_only" if target in watch_targets else "missing"
            ),
        }
    return summary


def upstream_target_rows(upstream: dict[str, Any], target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in upstream.get("source_records", []):
        full_oof = record.get("full_oof", {})
        delta_vs_178 = full_oof.get("target_delta_vs_178", {}).get(target)
        delta_vs_341 = full_oof.get("target_delta_vs_341", {}).get(target)
        if delta_vs_178 is None and delta_vs_341 is None:
            continue
        rows.append(
            {
                "id": record.get("id"),
                "layer": record.get("layer"),
                "action": record.get("action"),
                "status_clean": record.get("status_clean"),
                "dependency_read": record.get("dependency_read"),
                "local_cv": record.get("local_cv"),
                "artifact_exists": record.get("artifact_exists"),
                "fold_wins_vs_341": record.get("frozen_split", {}).get("fold_wins_vs_341"),
                "fold_count": record.get("frozen_split", {}).get("fold_count"),
                "worst_fold_delta_vs_341": record.get("frozen_split", {}).get("worst_fold_delta_vs_341"),
                "target_delta_vs_178": delta_vs_178,
                "target_delta_vs_341": delta_vs_341,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("target_delta_vs_178")) or 999.0,
            -(int(row.get("fold_wins_vs_341") or 0)),
            int(row.get("id") or 999999),
        ),
    )


def summarize_upstream(upstream: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in FOCUS_TARGETS + SECONDARY_TARGETS:
        rows = upstream_target_rows(upstream, target)
        root_row = next((row for row in rows if row.get("id") == 221), None)
        backbone_row = next((row for row in rows if row.get("id") == 226), None)
        out[target] = {
            "best_historical_upstream_row": rows[0] if rows else None,
            "clean_root_id221_row": root_row,
            "clean_backbone_id226_row": backbone_row,
            "historical_row_count": len(rows),
        }
    return out


def summarize_source_closures() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in SOURCE_CLOSURE_PATHS.items():
        report = maybe_load_json(path)
        if report is None:
            out[name] = {"path": str(path.relative_to(ROOT)), "exists": False}
            continue
        decision = report.get("decision", {})
        out[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "selected_targets": decision.get("selected_targets"),
            "selected_target_count": decision.get("selected_target_count"),
            "best_proxy": decision.get("best_proxy"),
            "projected_public_proxy_from_id341": decision.get("projected_public_proxy_from_id341"),
            "public_058_reviewable": decision.get("public_058_reviewable"),
            "public_058_plausible": decision.get("public_058_plausible"),
            "reviewable_low_movement_candidate_exists": decision.get("reviewable_low_movement_candidate_exists"),
            "create_submission_artifact": decision.get("create_submission_artifact"),
            "submit_now": decision.get("submit_now"),
            "push_kaggle_now": decision.get("push_kaggle_now"),
            "run_kaggle_now": decision.get("run_kaggle_now"),
        }
    return out


def summarize_evidence_flags(
    data_coverage: dict[str, Any] | None,
    phase2_spec: dict[str, Any] | None,
    local_closure: dict[str, Any] | None,
    prelaunch: dict[str, Any] | None,
) -> dict[str, Any]:
    phase2_decision = (phase2_spec or {}).get("decision", {})
    data_decision = (data_coverage or {}).get("decision", {})
    closure_decision = (local_closure or {}).get("decision", {})
    prelaunch_decision = (prelaunch or {}).get("decision", {})
    return {
        "data_source_coverage_ready": data_decision.get("data_source_coverage_ready"),
        "all_major_sources_present": data_decision.get("all_major_sources_present"),
        "external_direct_concat_allowed": data_decision.get("external_direct_concat_allowed"),
        "phase2_ready": phase2_decision.get("phase2_raw_sequence_source_execution_spec_ready")
        or phase2_decision.get("phase2_ready"),
        "local_source_stack_closed_for_submission_artifact": closure_decision.get(
            "local_source_stack_closed_for_submission_artifact"
        ),
        "any_local_lane_promotable": closure_decision.get("any_local_lane_promotable"),
        "prelaunch_ready_for_explicit_approval": prelaunch_decision.get("prelaunch_ready_for_explicit_approval"),
        "requires_explicit_user_approval": prelaunch_decision.get("requires_explicit_user_approval"),
    }


def target_action(target_summary: dict[str, Any], upstream_summary: dict[str, Any]) -> str:
    local_status = target_summary["local_status"]
    if local_status == "strict":
        return "keep_as_control_only_until_combined_budget_beats_id178"
    if local_status == "watch_only":
        return "do_not_promote_watch_only_signal_without_all_primary_confirmation"
    if upstream_summary.get("clean_backbone_id226_row"):
        return "route_to_clean_id221_then_id226_no_submit_upstream_rebuild"
    return "requires_materially_new_raw_source_before_local_artifact"


def build_report() -> dict[str, Any]:
    target_gap = load_json(TARGET_GAP_PATH)
    upstream = load_json(UPSTREAM_PRIORITY_PATH)
    data_coverage = maybe_load_json(DATA_COVERAGE_PATH)
    phase2_spec = maybe_load_json(PHASE2_SPEC_PATH)
    local_closure = maybe_load_json(LOCAL_CLOSURE_PATH)
    prelaunch = maybe_load_json(PRELAUNCH_PATH)

    local_by_target = summarize_target_gap(target_gap)
    upstream_by_target = summarize_upstream(upstream)
    target_inventory: dict[str, Any] = {}
    for target in FOCUS_TARGETS + SECONDARY_TARGETS:
        target_inventory[target] = {
            "local": local_by_target[target],
            "upstream": upstream_by_target[target],
            "recommended_action": target_action(local_by_target[target], upstream_by_target[target]),
        }

    focus_missing_locally = [target for target in FOCUS_TARGETS if local_by_target[target]["local_status"] == "missing"]
    focus_has_upstream_backbone = [
        target for target in FOCUS_TARGETS if upstream_by_target[target].get("clean_backbone_id226_row") is not None
    ]
    secondary_watch_only = [
        target for target in SECONDARY_TARGETS if local_by_target[target]["local_status"] == "watch_only"
    ]
    evidence_flags = summarize_evidence_flags(data_coverage, phase2_spec, local_closure, prelaunch)
    decision = {
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "focus_targets": FOCUS_TARGETS,
        "focus_missing_locally": focus_missing_locally,
        "focus_has_upstream_backbone_signal": focus_has_upstream_backbone,
        "secondary_watch_only_targets": secondary_watch_only,
        "local_q2_q3_s3_promotable": False,
        "external_direct_concat_allowed": evidence_flags.get("external_direct_concat_allowed"),
        "prelaunch_ready_for_explicit_approval": evidence_flags.get("prelaunch_ready_for_explicit_approval"),
        "requires_explicit_user_approval": evidence_flags.get("requires_explicit_user_approval"),
        "recommended_next_action": (
            "Do not create a local ID342. Q2/Q3/S3 have no current local selected rows; the only meaningful "
            "Q2/Q3/S3 surface is the report-first clean ID221→ID226 upstream rebuild, gated by explicit Kaggle approval."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_missing_target_signal_inventory",
        "purpose": (
            "Inventory the Q2/Q3/S3 gap after ID341 and separate local promotable evidence from upstream rebuild leads."
        ),
        "inputs": {
            "target_gap": str(TARGET_GAP_PATH.relative_to(ROOT)),
            "upstream_priority": str(UPSTREAM_PRIORITY_PATH.relative_to(ROOT)),
            "phase2_spec": str(PHASE2_SPEC_PATH.relative_to(ROOT)),
            "data_coverage": str(DATA_COVERAGE_PATH.relative_to(ROOT)),
            "local_closure": str(LOCAL_CLOSURE_PATH.relative_to(ROOT)),
            "prelaunch": str(PRELAUNCH_PATH.relative_to(ROOT)),
        },
        "constants": {
            "id178_public": ID178_PUBLIC,
            "public_058_review_cutoff": PUBLIC_058_REVIEW_CUTOFF,
            "focus_targets": FOCUS_TARGETS,
            "secondary_targets": SECONDARY_TARGETS,
        },
        "target_inventory": target_inventory,
        "source_closures": summarize_source_closures(),
        "evidence_flags": evidence_flags,
        "decision": decision,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    out = safe_float(value)
    return "" if out is None else f"{out:.{digits}f}"


def target_row(target: str, row: dict[str, Any]) -> str:
    local = row["local"]
    upstream = row["upstream"]
    backbone = upstream.get("clean_backbone_id226_row") or {}
    root = upstream.get("clean_root_id221_row") or {}
    return (
        f"| {target} | `{local['local_status']}` | {local['local_selected_row_count']} | "
        f"{fmt((local.get('best_local_row') or {}).get('mean_delta_vs_id341'))} | "
        f"{fmt(root.get('target_delta_vs_178'))} | {fmt(backbone.get('target_delta_vs_178'))} | "
        f"{backbone.get('fold_wins_vs_341', '')}/{backbone.get('fold_count', '')} | "
        f"`{row['recommended_action']}` |"
    )


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    flags = report["evidence_flags"]
    lines = [
        "# Post-ID341 missing-target signal inventory",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Focus targets: `{', '.join(decision['focus_targets'])}`",
        f"- Focus targets missing locally: `{', '.join(decision['focus_missing_locally'])}`",
        f"- Focus targets with upstream backbone signal: `{', '.join(decision['focus_has_upstream_backbone_signal'])}`",
        f"- Secondary watch-only targets: `{', '.join(decision['secondary_watch_only_targets'])}`",
        f"- Local Q2/Q3/S3 promotable: `{decision['local_q2_q3_s3_promotable']}`",
        f"- External direct concat allowed: `{decision['external_direct_concat_allowed']}`",
        f"- Prelaunch ready for explicit approval: `{decision['prelaunch_ready_for_explicit_approval']}`",
        f"- Requires explicit user approval: `{decision['requires_explicit_user_approval']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        "",
        decision["recommended_next_action"],
        "",
        "## Target inventory",
        "",
        "| target | local status | local rows | best local mean Δ341 | ID221 Δ178 | ID226 Δ178 | ID226 fold wins | action |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for target in report["constants"]["focus_targets"] + report["constants"]["secondary_targets"]:
        lines.append(target_row(target, report["target_inventory"][target]))
    lines.extend(
        [
            "",
            "## Evidence flags",
            "",
            f"- Data/source coverage ready: `{flags['data_source_coverage_ready']}`",
            f"- All major sources present: `{flags['all_major_sources_present']}`",
            f"- Local source stack closed for artifact: `{flags['local_source_stack_closed_for_submission_artifact']}`",
            f"- Any local lane promotable: `{flags['any_local_lane_promotable']}`",
            "",
            "Interpretation: Q2/Q3/S3 are not absent because files or external data are missing; they are absent because the local post-ID341 source gates selected no transfer-safe rows. Historical upstream artifacts contain the missing target surface, but they must be rebuilt cleanly before artifact review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_missing_target_signal_inventory_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
