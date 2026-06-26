#!/usr/bin/env python3
"""Audit whether the post-ID341 Kaggle runner is still score-relevant.

This report-only diagnostic ties together the missing-target budget, local lane
closure, runner/intake contract, and post-pull review chain. It exists because a
safe runner is not enough: the next approved Kaggle run should also cover the
actual current bottleneck, namely Q2/Q3/S3 missing-target transfer via the clean
ID221→ID226 upstream path.

It never runs Kaggle, trains models, creates predictions, creates submission
artifacts, edits the submission log, or submits to DACON.
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

NOTEBOOK = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "Upstream_Rebuild_Post_ID341_Runner.ipynb"
SOURCE_ROUTE = REPORTS_DIR / "post_id341_q2q3s3_source_route_20260618.json"
TARGET_GAP_BUDGET = REPORTS_DIR / "post_id341_target_gap_budget_20260618.json"
MISSING_TARGET_INVENTORY = REPORTS_DIR / "post_id341_missing_target_signal_inventory_20260618.json"
Q2Q3S3_SOURCE_HYPOTHESIS_MATRIX = REPORTS_DIR / "post_id341_q2q3s3_source_hypothesis_matrix_20260618.json"
Q3S3_CLOSURE = REPORTS_DIR / "post_id341_q3s3_local_lane_closure_20260618.json"
Q3S3_SEMANTIC_ROUTINE_CLOSURE = REPORTS_DIR / "post_id341_q3s3_semantic_routine_lane_closure_20260618.json"
Q3S3_REMAINING_ROUTE_INVENTORY = REPORTS_DIR / "post_id341_q3s3_remaining_route_inventory_20260618.json"
RUNNER_INTAKE = REPORTS_DIR / "post_id341_kaggle_runner_intake_contract_20260618.json"
POST_PULL_CHAIN = REPORTS_DIR / "post_id341_post_pull_review_chain_contract_20260618.json"

REQUIRED_RUNNER_SCRIPTS = {
    "joint_risk_transfer_gate": "scripts/diagnose_post_id341_transfer_gate_v2_with_joint_risk.py",
    "external_prior": "scripts/diagnose_external_etri_prior_smoke_post_id341.py",
    "q2q3s3_source_hypothesis_matrix": "scripts/diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py",
    "q2q3s3_local_fallback_chain": "scripts/diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py",
    "phase2_raw_spec": "scripts/diagnose_post_id341_phase2_raw_source_execution_spec.py",
    "phase2_multimodal_plan": "scripts/diagnose_phase2_raw_multimodal_execution_plan_post_id341.py",
    "phase2_block_a_q1_s4": "scripts/diagnose_phase2_block_a_q1_s4_sleep_residual_bridge_post_id341.py",
    "phase2_block_b_s2": "scripts/diagnose_phase2_block_b_s2_sleep_recovery_retry_post_id341.py",
    "phase3_usage_temporal": "scripts/diagnose_phase3_usage_temporal_microstructure_smoke_post_id341.py",
    "upstream_priority_matrix": "scripts/diagnose_post_id341_upstream_source_priority_matrix.py",
    "clean_backbone_design": "scripts/diagnose_post_id341_clean_backbone_rebuild_design.py",
    "clean_id221_smoke": "scripts/diagnose_post_id341_clean_id221_feature_similarity_smoke.py",
    "clean_id221_inner": "scripts/diagnose_post_id341_clean_id221_inner_selection_probe.py",
    "clean_id226_smoke": "scripts/diagnose_post_id341_clean_id226_calibration_smoke.py",
    "clean_backbone_closure": "scripts/diagnose_post_id341_clean_backbone_lane_closure.py",
}


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "error": None,
    }


def notebook_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_report() -> dict[str, Any]:
    source_route = report_status(SOURCE_ROUTE)
    target_gap = report_status(TARGET_GAP_BUDGET)
    missing_inventory = report_status(MISSING_TARGET_INVENTORY)
    q2q3s3_matrix = report_status(Q2Q3S3_SOURCE_HYPOTHESIS_MATRIX)
    q3s3_closure = report_status(Q3S3_CLOSURE)
    q3s3_semantic_routine_closure = report_status(Q3S3_SEMANTIC_ROUTINE_CLOSURE)
    q3s3_remaining_route_inventory = report_status(Q3S3_REMAINING_ROUTE_INVENTORY)
    runner_intake = report_status(RUNNER_INTAKE)
    post_pull_chain = report_status(POST_PULL_CHAIN)
    text = notebook_text(NOTEBOOK)
    runner_script_coverage = {name: script in text for name, script in REQUIRED_RUNNER_SCRIPTS.items()}

    source_route_decision = source_route["decision"]
    target_gap_decision = target_gap["decision"]
    missing_decision = missing_inventory["decision"]
    q2q3s3_matrix_decision = q2q3s3_matrix["decision"]
    closure_decision = q3s3_closure["decision"]
    semantic_closure_decision = q3s3_semantic_routine_closure["decision"]
    remaining_route_decision = q3s3_remaining_route_inventory["decision"]
    runner_decision = runner_intake["decision"]
    post_pull_decision = post_pull_chain["decision"]

    missing_target_contract = bool(
        missing_decision.get("focus_missing_locally") == ["Q2", "Q3", "S3"]
        and missing_decision.get("focus_has_upstream_backbone_signal") == ["Q2", "Q3", "S3"]
        and missing_decision.get("local_q2_q3_s3_promotable") is False
        and missing_decision.get("external_direct_concat_allowed") is False
    )
    q2q3s3_matrix_contract = bool(
        q2q3s3_matrix_decision.get("q2q3s3_source_hypothesis_matrix_ready") is True
        and q2q3s3_matrix_decision.get("all_focus_targets_missing_locally") is True
        and q2q3s3_matrix_decision.get("all_focus_targets_have_historical_id226_signal") is True
        and q2q3s3_matrix_decision.get("all_focus_targets_id226_win_all_folds_vs_341") is True
        and q2q3s3_matrix_decision.get("processed_feature_coverage_sufficient_for_design") is True
        and q2q3s3_matrix_decision.get("external_direct_concat_allowed") is False
        and q2q3s3_matrix_decision.get("local_q2q3s3_artifact_promotable") is False
        and q2q3s3_matrix_decision.get("create_submission_artifact") is False
        and q2q3s3_matrix_decision.get("submit_now") is False
        and q2q3s3_matrix_decision.get("push_kaggle_now") is False
        and q2q3s3_matrix_decision.get("run_kaggle_now") is False
    )
    target_gap_contract = bool(
        target_gap_decision.get("q1_s4_only_is_insufficient") is True
        and {"Q2", "Q3", "S3"}.issubset(set(target_gap_decision.get("needs_new_independent_targets", [])))
        and target_gap_decision.get("strict_budget_public_058_reviewable") is False
        and target_gap_decision.get("watch_budget_public_058_reviewable") is False
    )
    route_contract = bool(
        source_route_decision.get("primary_score_relevant_route") == "explicit_approved_clean_upstream_id221_to_id226"
        and source_route_decision.get("local_q2q3s3_artifact_allowed") is False
        and source_route_decision.get("remote_no_submit_prelaunch_ready") is True
    )
    closure_contract = bool(
        closure_decision.get("q3_s3_uncovered_after_local_fallback") == ["Q3", "S3"]
        and closure_decision.get("local_fallback_closed_for_submission_artifact") is True
        and closure_decision.get("primary_next_score_relevant_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
    )
    semantic_closure_contract = bool(
        semantic_closure_decision.get("semantic_routine_lane_closure_passed") is True
        and semantic_closure_decision.get("q3_s3_still_uncovered") == ["Q3", "S3"]
        and semantic_closure_decision.get("local_semantic_routine_artifact_allowed") is False
        and semantic_closure_decision.get("public_058_evidence") is False
        and semantic_closure_decision.get("primary_next_score_relevant_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and semantic_closure_decision.get("create_submission_artifact") is False
        and semantic_closure_decision.get("submit_now") is False
        and semantic_closure_decision.get("push_kaggle_now") is False
        and semantic_closure_decision.get("run_kaggle_now") is False
    )
    remaining_route_inventory_contract = bool(
        remaining_route_decision.get("q3s3_remaining_route_inventory_ready") is True
        and remaining_route_decision.get("q3_s3_still_uncovered") is True
        and remaining_route_decision.get("fold_nested_sibling_bridge_closed") is True
        and remaining_route_decision.get("local_no_approval_score_route_exists") is False
        and remaining_route_decision.get("only_score_relevant_open_route")
        == "explicit_approved_clean_upstream_id221_to_id226"
        and remaining_route_decision.get("requires_explicit_kaggle_approval") is True
        and remaining_route_decision.get("approval_phrase") == "Kaggle upstream rebuild no-submit 실행 승인"
        and remaining_route_decision.get("public_058_evidence") is False
        and remaining_route_decision.get("create_submission_artifact") is False
        and remaining_route_decision.get("submit_now") is False
        and remaining_route_decision.get("push_kaggle_now") is False
        and remaining_route_decision.get("run_kaggle_now") is False
    )
    runner_coverage_contract = bool(all(runner_script_coverage.values()))
    runner_intake_contract = runner_decision.get("runner_intake_contract_pass") is True
    post_pull_chain_contract = post_pull_decision.get("post_pull_review_chain_contract_pass") is True
    no_forbidden_decisions = all(
        decision.get(key) is not True
        for decision in [
            source_route_decision,
            target_gap_decision,
            missing_decision,
            q2q3s3_matrix_decision,
            closure_decision,
            runner_decision,
            post_pull_decision,
            semantic_closure_decision,
            remaining_route_decision,
        ]
        for key in ["create_submission_artifact", "submit_now", "push_kaggle_now", "run_kaggle_now"]
    )
    score_relevant = bool(
        all(
            [
                source_route["parse_ok"],
                target_gap["parse_ok"],
                missing_inventory["parse_ok"],
                q2q3s3_matrix["parse_ok"],
                q3s3_closure["parse_ok"],
                q3s3_semantic_routine_closure["parse_ok"],
                q3s3_remaining_route_inventory["parse_ok"],
                runner_intake["parse_ok"],
                post_pull_chain["parse_ok"],
                missing_target_contract,
                q2q3s3_matrix_contract,
                target_gap_contract,
                route_contract,
                closure_contract,
                semantic_closure_contract,
                remaining_route_inventory_contract,
                runner_coverage_contract,
                runner_intake_contract,
                post_pull_chain_contract,
                no_forbidden_decisions,
            ]
        )
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_runner_score_relevance_contract",
        "purpose": (
            "Verify that the approved no-submit Kaggle runner is not merely safe, but aimed at the current "
            "Public-0.58 bottleneck: missing Q2/Q3/S3 signal through the clean ID221→ID226 upstream path."
        ),
        "inputs": {
            "notebook": safe_relative(NOTEBOOK),
            "source_route": source_route,
            "target_gap_budget": target_gap,
            "missing_target_inventory": missing_inventory,
            "q2q3s3_source_hypothesis_matrix": q2q3s3_matrix,
            "q3s3_local_lane_closure": q3s3_closure,
            "q3s3_semantic_routine_lane_closure": q3s3_semantic_routine_closure,
            "q3s3_remaining_route_inventory": q3s3_remaining_route_inventory,
            "runner_intake_contract": runner_intake,
            "post_pull_review_chain_contract": post_pull_chain,
        },
        "runner_script_coverage": runner_script_coverage,
        "checks": {
            "missing_target_contract": missing_target_contract,
            "q2q3s3_source_hypothesis_matrix_contract": q2q3s3_matrix_contract,
            "target_gap_contract": target_gap_contract,
            "route_contract": route_contract,
            "q3s3_closure_contract": closure_contract,
            "q3s3_semantic_routine_closure_contract": semantic_closure_contract,
            "q3s3_remaining_route_inventory_contract": remaining_route_inventory_contract,
            "runner_coverage_contract": runner_coverage_contract,
            "runner_intake_contract": runner_intake_contract,
            "post_pull_chain_contract": post_pull_chain_contract,
            "no_forbidden_decisions": no_forbidden_decisions,
        },
        "decision": {
            "runner_score_relevance_contract_pass": score_relevant,
            "score_relevant_route": "explicit_approved_clean_upstream_id221_to_id226",
            "missing_targets": ["Q2", "Q3", "S3"],
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Runner is score-relevant and pending-safe; explicit approval is still required for Kaggle execution."
                if score_relevant
                else "Fix runner score-relevance drift before consuming Kaggle approval."
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
    lines = [
        "# Post-ID341 runner score-relevance contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-submit. It verifies that the next approved Kaggle runner still targets the missing Q2/Q3/S3 bottleneck through the clean ID221→ID226 route.",
        "",
        "## Decision",
        "",
        f"- Runner score-relevance contract pass: `{decision['runner_score_relevance_contract_pass']}`",
        f"- Score-relevant route: `{decision['score_relevant_route']}`",
        f"- Missing targets: `{decision['missing_targets']}`",
        f"- Execute Kaggle now: `{decision['execute_kaggle_now']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Runner script coverage", ""])
    for key, value in report["runner_script_coverage"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_runner_score_relevance_contract_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
