#!/usr/bin/env python3
"""Summarize the post-ID341 no-approval frontier.

This local report-only audit consolidates the current local closures, public
anchors, and Kaggle prelaunch state into one operational decision: whether any
local no-approval score route remains, or whether the only score-relevant path
is the explicit-approved Kaggle no-submit upstream rebuild.

It creates no predictions, submission CSVs, base prediction artifacts,
notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or DACON
submissions.
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
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_no_approval_frontier_20260619.json"
APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
PRIMARY_REMOTE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
CONSUMED_MANUAL_PROBE = "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"

REPORT_PATHS = {
    "public_gap": "post_id341_public_local_gap_diagnosis_20260618.json",
    "transfer_gate": "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "target_gap_budget": "post_id341_target_gap_budget_20260618.json",
    "missing_target_inventory": "post_id341_missing_target_signal_inventory_20260618.json",
    "q3s3_remaining": "post_id341_q3s3_remaining_route_inventory_20260618.json",
    "clean_backbone_closure": "post_id341_clean_backbone_lane_closure_20260618.json",
    "phase2_block_a_gap": "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "phase2_block_b": "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "phase3_usage": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "phase3_ambience": "phase3_ambience_context_closure_post_id341_20260618.json",
    "local_smoke_freshness": "post_id341_local_smoke_freshness_gate_20260618.json",
    "external_public05917_residual": "post_id341_external_public05917_residual_audit_20260619.json",
    "q2q3s3_paper_family": "post_id341_q2q3s3_paper_family_fold_smoke_20260619.json",
    "q3s3_anchor_failure_profile": "post_id341_q3s3_anchor_failure_profile_20260619.json",
    "usage_semantic_burst_markov_validation_stop": (
        "post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
    ),
    "prelaunch": "post_id341_kaggle_prelaunch_readiness_gate_20260618.json",
    "guarded_remote_contract": "kaggle_guarded_remote_contract_audit_20260618.json",
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        return {"exists": False, "parse_ok": False, "decision": {}, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "path": rel(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload.setdefault("path", rel(path))
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def public_anchor_summary() -> dict[str, Any]:
    if not SUBMISSION_LOG.exists():
        return {"submission_log_exists": False, "plausible_public_rows": 0}
    log = pd.read_csv(SUBMISSION_LOG)
    public = log[log["public_lb_score_num"].fillna(0) > 0.1].copy()
    public = public[public["public_lb_score_num"] < 5].copy()
    if public.empty:
        return {"submission_log_exists": True, "plausible_public_rows": 0}
    best = public.sort_values("public_lb_score_num").iloc[0].to_dict()
    recent_ids = [178, 287, 329, 337, 340]
    recent = public[public["id"].isin(recent_ids)].sort_values("id")
    return {
        "submission_log_exists": True,
        "plausible_public_rows": int(len(public)),
        "best_public_anchor": {
            "id": int(best["id"]),
            "submission_file": str(best["submission_file"]),
            "local_cv": float(best["local_cv_score_num"]),
            "public": float(best["public_lb_score_num"]),
            "cv_lb_gap": float(best["cv_lb_gap"]),
        },
        "recent_public_anchors": [
            {
                "id": int(row["id"]),
                "submission_file": str(row["submission_file"]),
                "local_cv": float(row["local_cv_score_num"]),
                "public": float(row["public_lb_score_num"]),
                "cv_lb_gap": float(row["cv_lb_gap"]),
            }
            for _, row in recent.iterrows()
        ],
    }


def flag(payloads: dict[str, dict[str, Any]], name: str, key: str, expected: Any = True) -> bool:
    return payloads[name].get("decision", {}).get(key) is expected


def build_report() -> dict[str, Any]:
    payloads = {key: load_report(name) for key, name in REPORT_PATHS.items()}
    all_required_reports_ready = all(payload.get("exists") and payload.get("parse_ok") for payload in payloads.values())

    local_closure_checks = {
        "public_gap_blocks_local_cv_only_promotion": flag(
            payloads, "public_gap", "any_watch_candidate_passes_public_058_recent_floor_proxy", False
        )
        and flag(payloads, "public_gap", "any_watch_candidate_passes_public_058_recent_stress_proxy", False),
        "transfer_gate_has_no_public_058_candidate": flag(
            payloads, "transfer_gate", "any_candidate_passes_combined_public_058_gate", False
        ),
        "target_gap_has_no_public_058_budget": flag(
            payloads, "target_gap_budget", "strict_budget_public_058_reviewable", False
        )
        and flag(payloads, "target_gap_budget", "watch_budget_public_058_reviewable", False),
        "missing_targets_route_to_upstream": flag(
            payloads, "missing_target_inventory", "local_q2_q3_s3_promotable", False
        ),
        "q3s3_local_no_approval_closed": flag(
            payloads, "q3s3_remaining", "local_no_approval_score_route_exists", False
        ),
        "clean_backbone_closed": flag(payloads, "clean_backbone_closure", "close_clean_backbone_lane_now", True),
        "phase2_block_a_not_public_058": flag(
            payloads, "phase2_block_a_gap", "q1_signal_sufficient_for_public_058", False
        ),
        "phase2_block_b_closed": flag(payloads, "phase2_block_b", "final_block_b_gate_pass", False),
        "phase3_usage_closed": flag(payloads, "phase3_usage", "final_usage_temporal_gate_pass", False),
        "phase3_ambience_closed": flag(payloads, "phase3_ambience", "ambience_reopen_now", False),
        "external_public05917_direct_route_closed": flag(
            payloads, "external_public05917_residual", "residual_direct_notebook_route_exists", False
        )
        and flag(payloads, "external_public05917_residual", "public_058_evidence_present", False),
        "q2q3s3_paper_family_local_lane_closed": flag(
            payloads, "q2q3s3_paper_family", "q3s3_covered_by_paper_family", False
        )
        and flag(payloads, "q2q3s3_paper_family", "public_058_evidence", False)
        and flag(payloads, "q2q3s3_paper_family", "create_submission_artifact", False),
        "q3s3_oof_gain_direct_submit_blocked": flag(
            payloads, "q3s3_anchor_failure_profile", "q3s3_oof_gain_alone_is_submission_evidence", False
        )
        and flag(
            payloads, "q3s3_anchor_failure_profile", "unsubmitted_upstream_requires_rebuild_not_direct_submit", True
        )
        and flag(payloads, "q3s3_anchor_failure_profile", "public_058_evidence", False)
        and flag(payloads, "q3s3_anchor_failure_profile", "create_submission_artifact", False),
        "usage_semantic_burst_markov_matrix_validated_stop": flag(
            payloads, "usage_semantic_burst_markov_validation_stop", "validation_stop_passed", True
        )
        and flag(payloads, "usage_semantic_burst_markov_validation_stop", "public_058_evidence", False)
        and flag(payloads, "usage_semantic_burst_markov_validation_stop", "score_route_opened", False)
        and flag(payloads, "usage_semantic_burst_markov_validation_stop", "train_model_now", False)
        and flag(payloads, "usage_semantic_burst_markov_validation_stop", "create_submission_artifact", False),
    }
    prelaunch_checks = {
        "local_runner_smoke_fresh": flag(payloads, "local_smoke_freshness", "local_runner_smoke_fresh", True),
        "guarded_remote_contract_pass": flag(payloads, "guarded_remote_contract", "guarded_remote_contract_pass", True),
        "prelaunch_ready_for_explicit_approval": flag(
            payloads, "prelaunch", "prelaunch_ready_for_explicit_approval", True
        ),
        "prelaunch_requires_explicit_user_approval": flag(
            payloads, "prelaunch", "requires_explicit_user_approval", True
        ),
    }
    all_local_closed = all(local_closure_checks.values())
    approval_ready = all(prelaunch_checks.values())
    no_approval_score_route_exists = not all_local_closed

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_no_approval_frontier",
        "purpose": (
            "Consolidate the no-approval frontier after ID178/287/329/341 evidence, local closures, and Kaggle "
            "prelaunch checks."
        ),
        "inputs": {key: payloads[key].get("path", rel(REPORTS_DIR / name)) for key, name in REPORT_PATHS.items()},
        "report_status": {
            key: {"exists": bool(payload.get("exists")), "parse_ok": bool(payload.get("parse_ok"))}
            for key, payload in payloads.items()
        },
        "public_anchor_summary": public_anchor_summary(),
        "local_closure_checks": local_closure_checks,
        "prelaunch_checks": prelaunch_checks,
        "decision": {
            "no_approval_frontier_ready": bool(all_required_reports_ready),
            "all_local_no_approval_score_routes_closed": bool(all_local_closed),
            "no_approval_score_route_exists": bool(no_approval_score_route_exists),
            "only_score_relevant_open_route": PRIMARY_REMOTE_ROUTE,
            "requires_explicit_kaggle_approval": True,
            "approval_phrase": APPROVAL_PHRASE,
            "prelaunch_ready_for_explicit_approval": bool(approval_ready),
            "manual_probe_if_user_spends_submission": None,
            "manual_probe_is_public_058_evidence": False,
            "public_058_reviewable": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Without explicit approval, do not create ID342 or submit a local candidate. The ID341/340/337 "
                "manual micro-probe family is already consumed and remains near 0.60. The score-relevant "
                "route is the approved no-submit Kaggle upstream rebuild; otherwise continue only with report-only "
                "design for genuinely new raw/sequence source families."
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
    decision = report["decision"]
    anchor = report["public_anchor_summary"].get("best_public_anchor", {})
    lines = [
        "# Post-ID341 no-approval frontier",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Public anchor read",
        "",
        f"- Plausible public rows: `{report['public_anchor_summary'].get('plausible_public_rows')}`",
    ]
    if anchor:
        lines.extend(
            [
                f"- Best public anchor: ID`{anchor['id']}` public `{anchor['public']:.10f}` local `{anchor['local_cv']:.6f}`",
                f"- File: `{anchor['submission_file']}`",
            ]
        )
    lines.extend(["", "## Local closure checks", ""])
    for key, value in report["local_closure_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Prelaunch checks", ""])
    for key, value in report["prelaunch_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- No-approval frontier ready: `{decision['no_approval_frontier_ready']}`",
            f"- All local no-approval score routes closed: `{decision['all_local_no_approval_score_routes_closed']}`",
            f"- No-approval score route exists: `{decision['no_approval_score_route_exists']}`",
            f"- Only score-relevant open route: `{decision['only_score_relevant_open_route']}`",
            f"- Requires explicit Kaggle approval: `{decision['requires_explicit_kaggle_approval']}`",
            f"- Approval phrase: `{decision['approval_phrase']}`",
            f"- Prelaunch ready for explicit approval: `{decision['prelaunch_ready_for_explicit_approval']}`",
            f"- Manual probe if user spends submission: `{decision['manual_probe_if_user_spends_submission']}`",
            f"- Manual probe is Public 0.58 evidence: `{decision['manual_probe_is_public_058_evidence']}`",
            f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
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
