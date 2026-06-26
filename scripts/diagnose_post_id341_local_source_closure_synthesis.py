#!/usr/bin/env python3
"""Synthesize post-ID341 local-source closure evidence.

This report-only diagnostic reads the post-ID341 local raw/source closure
reports and decides whether another local submission artifact should be created
or whether the next score-relevant path must move to a report-first upstream
rebuild/parity runner.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, no Kaggle push/run, and no DACON submission.
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

REPORTS = {
    "transfer_gate_v2": "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "phase2_block_a_q1_gap": "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "phase2_block_b_s2_recovery": "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "phase3_usage_temporal": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "phase3_ambience_closure": "phase3_ambience_context_closure_post_id341_20260618.json",
    "place_social": "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "external_prior": "external_etri_prior_smoke_post_id341_20260618.json",
    "combined_external": "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "clean_backbone_closure": "post_id341_clean_backbone_lane_closure_20260618.json",
    "upstream_phase1_readiness": "post_id341_upstream_rebuild_phase1_readiness_20260618.json",
    "kaggle_source_zip_selftest": "kaggle_source_zip_selftest_post_id341_20260618.json",
    "kaggle_upstream_packet": "kaggle_upstream_execution_packet_post_id341_20260618.json",
}
ID178_PUBLIC = 0.6014113009
PUBLIC_058_REVIEW_CUTOFF = 0.590
ONLY_SCORE_RELEVANT_OPEN_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
EXPECTED_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"


def load_report(name: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def status_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, name in REPORTS.items():
        path = REPORTS_DIR / name
        report = load_report(name)
        decision = report.get("decision", {}) if report else {}
        proxy_gap_math = report.get("proxy_gap_math", {}) if report else {}
        aggregate = report.get("aggregate", {}) if report else {}
        rows.append(
            {
                "lane": key,
                "path": str(path.relative_to(ROOT)),
                "exists": report is not None,
                "selected_targets": decision.get("selected_targets"),
                "create_submission_artifact": decision.get("create_submission_artifact"),
                "submit_now": decision.get("submit_now"),
                "push_kaggle_now": decision.get("push_kaggle_now"),
                "run_kaggle_now": decision.get("run_kaggle_now"),
                "public_058_reviewable": decision.get("public_058_reviewable"),
                "projected_public_proxy_from_id341": decision.get("projected_public_proxy_from_id341"),
                "final_block_a_gate_pass": decision.get("final_block_a_gate_pass"),
                "final_block_b_gate_pass": decision.get("final_block_b_gate_pass"),
                "final_usage_temporal_gate_pass": decision.get("final_usage_temporal_gate_pass"),
                "ambience_reopen_now": decision.get("ambience_reopen_now"),
                "ambience_has_nested_positive_signal": decision.get("ambience_has_nested_positive_signal"),
                "clean_backbone_closed": decision.get("close_clean_backbone_lane_now"),
                "phase1_ready": decision.get("phase1_ready"),
                "local_new_source_stack_closed": decision.get("local_new_source_stack_closed"),
                "selftest_passed": decision.get("selftest_passed"),
                "ready_for_explicit_approval": decision.get("ready_for_explicit_approval")
                or decision.get("ready_for_explicit_kaggle_approval"),
                "q1_signal_factor_needed_to_beat_id178_if_q1_only": proxy_gap_math.get(
                    "q1_signal_factor_needed_to_beat_id178_if_q1_only"
                ),
                "ambience_best_full_oof_delta": (
                    aggregate.get("best_full_oof_row", {}).get("full_oof_delta_vs_base")
                    if isinstance(aggregate.get("best_full_oof_row"), dict)
                    else None
                ),
                "ambience_worst_nested_delta": (
                    aggregate.get("worst_nested_collapse_row", {}).get("nested_delta_vs_base")
                    if isinstance(aggregate.get("worst_nested_collapse_row"), dict)
                    else None
                ),
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    rows = status_rows()
    missing = [row for row in rows if not row["exists"]]
    local_promotable_rows = [
        row
        for row in rows
        if row["exists"]
        and (
            row["create_submission_artifact"] is True
            or row["submit_now"] is True
            or row["public_058_reviewable"] is True
            or row["final_block_a_gate_pass"] is True
            or row["final_block_b_gate_pass"] is True
            or row["final_usage_temporal_gate_pass"] is True
            or row["ambience_reopen_now"] is True
        )
    ]
    packet = next((row for row in rows if row["lane"] == "kaggle_upstream_packet"), {})
    selftest = next((row for row in rows if row["lane"] == "kaggle_source_zip_selftest"), {})
    readiness = next((row for row in rows if row["lane"] == "upstream_phase1_readiness"), {})
    source_stack_closed = bool(readiness.get("local_new_source_stack_closed") is True)
    packet_ready = bool(packet.get("ready_for_explicit_approval") is True)
    selftest_ready = bool(selftest.get("ready_for_explicit_approval") is True)
    local_closed_for_artifact = bool(not missing and not local_promotable_rows and source_stack_closed)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_local_source_closure_synthesis",
        "purpose": (
            "Combine the post-ID341 local raw/source closure evidence and decide whether another local ID342-style "
            "artifact is justified before upstream rebuild/parity."
        ),
        "id178_public_anchor": ID178_PUBLIC,
        "public_058_review_cutoff": PUBLIC_058_REVIEW_CUTOFF,
        "source_rows": rows,
        "missing_required_reports": missing,
        "local_promotable_rows": local_promotable_rows,
        "decision": {
            "local_source_closure_synthesis_ready": True,
            "local_no_approval_score_route_exists": False,
            "only_score_relevant_open_route": ONLY_SCORE_RELEVANT_OPEN_ROUTE,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "train_model_now": False,
            "local_source_stack_closed_for_submission_artifact": local_closed_for_artifact,
            "any_local_lane_promotable": bool(local_promotable_rows),
            "all_required_reports_present": not missing,
            "kaggle_source_zip_selftest_ready": selftest_ready,
            "kaggle_upstream_packet_ready": packet_ready,
            "recommended_next_lane": "report_first_upstream_rebuild_or_kaggle_linux_parity",
            "requires_explicit_approval_for_kaggle": True,
            "requires_exact_kaggle_approval_phrase": True,
            "approval_phrase": EXPECTED_APPROVAL_PHRASE,
            "public_058_reviewable": False,
            "next_action": (
                "Do not create a local ID342 submission artifact. The only score-relevant next move is explicit-user-"
                f"approved Kaggle/Linux report-first upstream rebuild/parity (`{EXPECTED_APPROVAL_PHRASE}`), then local intake, remote summary, "
                "next-action, and Public-0.58 threshold gates. Only open a Public-0.58 artifact-contract review if "
                "the pulled reports are safe, beat the ID178 public-anchor proxy, and show min_proxy <= 0.58; "
                "ID178-only improvement above 0.58 is an improvement lead, not ID342."
            ),
        },
        "no_model_trained": True,
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
    lines = [
        "# Post-ID341 local-source closure synthesis",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It reads existing closure reports and creates no predictions, submission CSV, base artifacts, notebook artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Local source closure synthesis ready: `{decision['local_source_closure_synthesis_ready']}`",
        f"- Local no-approval score route exists: `{decision['local_no_approval_score_route_exists']}`",
        f"- Only score-relevant open route: `{decision['only_score_relevant_open_route']}`",
        f"- Local source stack closed for submission artifact: `{decision['local_source_stack_closed_for_submission_artifact']}`",
        f"- Any local lane promotable: `{decision['any_local_lane_promotable']}`",
        f"- All required reports present: `{decision['all_required_reports_present']}`",
        f"- Kaggle source.zip self-test ready: `{decision['kaggle_source_zip_selftest_ready']}`",
        f"- Kaggle upstream packet ready: `{decision['kaggle_upstream_packet_ready']}`",
        f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Modify submission log: `{decision['modify_submission_log']}`",
        f"- Create ID342/new submission ID: `{decision['create_id342_or_new_submission_id']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Train model now: `{decision['train_model_now']}`",
        f"- Requires exact Kaggle approval phrase: `{decision['requires_exact_kaggle_approval_phrase']}`",
        f"- Approval phrase: `{decision['approval_phrase']}`",
        f"- Recommended next lane: `{decision['recommended_next_lane']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Source lane table",
        "",
        "| lane | exists | selected | proxy | pass/reopen | ready | report |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in report["source_rows"]:
        pass_bits = [
            f"A={row['final_block_a_gate_pass']}",
            f"B={row['final_block_b_gate_pass']}",
            f"usage={row['final_usage_temporal_gate_pass']}",
            f"amb={row['ambience_reopen_now']}",
            f"p058={row['public_058_reviewable']}",
        ]
        ready_bits = [
            f"phase1={row['phase1_ready']}",
            f"selftest={row['selftest_passed']}",
            f"approval={row['ready_for_explicit_approval']}",
        ]
        selected = row["selected_targets"] if row["selected_targets"] is not None else ""
        proxy = row["projected_public_proxy_from_id341"]
        proxy_text = f"{proxy:.6f}" if isinstance(proxy, int | float) else ""
        lines.append(
            f"| `{row['lane']}` | `{row['exists']}` | `{selected}` | {proxy_text} | "
            f"`{', '.join(pass_bits)}` | `{', '.join(ready_bits)}` | `{row['path']}` |"
        )
    if report["missing_required_reports"]:
        lines.extend(["", "## Missing required reports", ""])
        for row in report["missing_required_reports"]:
            lines.append(f"- `{row['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_local_source_closure_synthesis_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
