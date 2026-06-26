#!/usr/bin/env python3
"""Contract for the post-ID341 Q2/Q3/S3 local fallback chain.

This report-only diagnostic makes the local fallback lane explicit: the
Q2/Q3/S3 env-social-routine path produced only a small Q2 context lead, while
Q3/S3 remain uncovered and Public 0.58 evidence is absent. The purpose is to
prevent accidentally promoting the Q2-only local fallback into ID342 while the
score-relevant route remains the explicit-approved clean ID221→ID226 upstream
rebuild.

It never trains models, creates predictions, creates submission artifacts,
edits the submission log, pushes/runs Kaggle, or submits to DACON.
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

SOURCE_ROUTE = REPORTS_DIR / "post_id341_q2q3s3_source_route_20260618.json"
ENV_SOCIAL_DESIGN = REPORTS_DIR / "post_id341_q2q3s3_env_social_routine_v2_design_20260618.json"
FEATURE_METADATA_SMOKE = REPORTS_DIR / "post_id341_q2q3s3_feature_generation_metadata_smoke_20260618.json"
FOLD_INTERNAL_SMOKE = REPORTS_DIR / "post_id341_q2q3s3_fold_internal_model_smoke_20260618.json"
Q2_LEAD_GEOMETRY_GATE = REPORTS_DIR / "post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json"
PAPER_FAMILY_SMOKE = REPORTS_DIR / "post_id341_q2q3s3_paper_family_fold_smoke_20260619.json"
Q3S3_LOCAL_LANE_CLOSURE = REPORTS_DIR / "post_id341_q3s3_local_lane_closure_20260618.json"
Q3S3_NEGATIVE_EVIDENCE_MAP = REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json"

FORBIDDEN_DECISION_KEYS = ["create_submission_artifact", "submit_now", "push_kaggle_now", "run_kaggle_now"]
MISSING_TARGETS = ["Q2", "Q3", "S3"]
UNCOVERED_AFTER_FALLBACK = ["Q3", "S3"]
PRIMARY_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
FALLBACK_ROUTE = "local_or_kaggle_q2q3s3_env_social_routine_v2_design"


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


def selected_targets(status: dict[str, Any]) -> list[str]:
    value = status.get("decision", {}).get("selected_targets")
    return value if isinstance(value, list) else []


def no_forbidden_decisions(statuses: list[dict[str, Any]]) -> bool:
    return all(
        status.get("decision", {}).get(key) is not True for status in statuses for key in FORBIDDEN_DECISION_KEYS
    )


def build_report() -> dict[str, Any]:
    source_route = report_status(SOURCE_ROUTE)
    env_design = report_status(ENV_SOCIAL_DESIGN)
    feature_smoke = report_status(FEATURE_METADATA_SMOKE)
    fold_smoke = report_status(FOLD_INTERNAL_SMOKE)
    q2_geometry = report_status(Q2_LEAD_GEOMETRY_GATE)
    paper_family = report_status(PAPER_FAMILY_SMOKE)
    q3s3_closure = report_status(Q3S3_LOCAL_LANE_CLOSURE)
    q3s3_negative_map = report_status(Q3S3_NEGATIVE_EVIDENCE_MAP)
    statuses = [
        source_route,
        env_design,
        feature_smoke,
        fold_smoke,
        q2_geometry,
        paper_family,
        q3s3_closure,
        q3s3_negative_map,
    ]

    source_decision = source_route["decision"]
    env_decision = env_design["decision"]
    feature_decision = feature_smoke["decision"]
    fold_decision = fold_smoke["decision"]
    q2_decision = q2_geometry["decision"]
    paper_family_decision = paper_family["decision"]
    closure_decision = q3s3_closure["decision"]
    negative_map_decision = q3s3_negative_map["decision"]

    parse_contract = all(status["parse_ok"] for status in statuses)
    route_contract = bool(
        source_decision.get("primary_score_relevant_route") == PRIMARY_ROUTE
        and source_decision.get("fallback_no_approval_route") == FALLBACK_ROUTE
        and source_decision.get("missing_targets") == MISSING_TARGETS
        and source_decision.get("local_q2q3s3_artifact_allowed") is False
        and source_decision.get("more_q1_s4_tuning_allowed") is False
        and source_decision.get("requires_explicit_user_approval") is True
    )
    design_contract = bool(
        env_decision.get("design_ready") is True
        and env_decision.get("target_focus") == MISSING_TARGETS
        and env_decision.get("next_local_allowed_step") == "feature_generation_metadata_smoke_only"
        and env_decision.get("train_model_now") is False
        and env_decision.get("create_submission_artifact") is False
    )
    feature_metadata_contract = bool(
        feature_decision.get("metadata_smoke_passed") is True
        and feature_decision.get("target_focus") == MISSING_TARGETS
        and feature_decision.get("next_local_allowed_step") == "fold_internal_q2q3s3_model_smoke_only"
        and feature_decision.get("train_model_now") is False
        and feature_decision.get("create_submission_artifact") is False
    )
    fold_internal_contract = bool(
        fold_decision.get("model_smoke_completed") is True
        and selected_targets(fold_smoke) == ["Q2"]
        and fold_decision.get("any_q2q3s3_signal_found") is True
        and fold_decision.get("all_q2q3s3_targets_covered") is False
        and fold_decision.get("public_058_evidence") is False
        and fold_decision.get("next_local_allowed_step") == "combined_no_artifact_geometry_gate"
        and fold_decision.get("create_submission_artifact") is False
    )
    q2_geometry_contract = bool(
        q2_decision.get("reviewable_low_movement_candidate_exists") is True
        and q2_decision.get("public_058_plausible") is False
        and q2_decision.get("q3_s3_still_uncovered") is True
        and q2_decision.get("create_submission_artifact") is False
        and q2_decision.get("submit_now") is False
    )
    paper_family_contract = bool(
        paper_family_decision.get("paper_family_smoke_completed") is True
        and selected_targets(paper_family) == []
        and paper_family_decision.get("q3s3_covered_by_paper_family") is False
        and paper_family_decision.get("public_058_evidence") is False
        and paper_family_decision.get("next_local_allowed_step") == "close_paper_family_local_lane"
        and paper_family_decision.get("create_submission_artifact") is False
        and paper_family_decision.get("submit_now") is False
    )
    q3s3_closure_contract = bool(
        closure_decision.get("local_q2_component_keep_as_context") is True
        and closure_decision.get("q3_s3_uncovered_after_local_fallback") == UNCOVERED_AFTER_FALLBACK
        and closure_decision.get("local_fallback_closed_for_submission_artifact") is True
        and closure_decision.get("local_artifact_allowed") is False
        and closure_decision.get("public_058_evidence") is False
        and closure_decision.get("primary_next_score_relevant_route") == PRIMARY_ROUTE
        and closure_decision.get("requires_explicit_kaggle_approval") is True
    )
    q3s3_negative_evidence_contract = bool(
        negative_map_decision.get("q3s3_negative_evidence_map_passed") is True
        and negative_map_decision.get("all_local_q3s3_lanes_closed") is True
        and negative_map_decision.get("q3_s3_still_uncovered") is True
        and negative_map_decision.get("public_058_evidence") is False
        and negative_map_decision.get("local_artifact_allowed") is False
        and negative_map_decision.get("primary_next_score_relevant_route") == PRIMARY_ROUTE
        and negative_map_decision.get("create_submission_artifact") is False
        and negative_map_decision.get("submit_now") is False
    )
    forbidden_contract = no_forbidden_decisions(statuses)
    local_fallback_chain_contract_pass = bool(
        parse_contract
        and route_contract
        and design_contract
        and feature_metadata_contract
        and fold_internal_contract
        and q2_geometry_contract
        and paper_family_contract
        and q3s3_closure_contract
        and q3s3_negative_evidence_contract
        and forbidden_contract
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_local_fallback_chain_contract",
        "purpose": (
            "Verify that the Q2/Q3/S3 local fallback chain remains context-only: Q2 has a small low-movement lead, "
            "Q3/S3 remain uncovered, and no Public-0.58 artifact review or submission artifact is allowed."
        ),
        "inputs": {
            "source_route": source_route,
            "env_social_routine_v2_design": env_design,
            "feature_generation_metadata_smoke": feature_smoke,
            "fold_internal_model_smoke": fold_smoke,
            "q2_lead_geometry_gate": q2_geometry,
            "paper_family_fold_smoke": paper_family,
            "q3s3_local_lane_closure": q3s3_closure,
            "q3s3_negative_evidence_map": q3s3_negative_map,
        },
        "checks": {
            "parse_contract": parse_contract,
            "route_contract": route_contract,
            "design_contract": design_contract,
            "feature_metadata_contract": feature_metadata_contract,
            "fold_internal_contract": fold_internal_contract,
            "q2_geometry_contract": q2_geometry_contract,
            "paper_family_contract": paper_family_contract,
            "q3s3_closure_contract": q3s3_closure_contract,
            "q3s3_negative_evidence_contract": q3s3_negative_evidence_contract,
            "no_forbidden_decisions": forbidden_contract,
        },
        "decision": {
            "local_fallback_chain_contract_pass": local_fallback_chain_contract_pass,
            "primary_score_relevant_route": PRIMARY_ROUTE,
            "fallback_no_approval_route": FALLBACK_ROUTE,
            "missing_targets": MISSING_TARGETS,
            "selected_local_context_targets": ["Q2"] if fold_internal_contract else selected_targets(fold_smoke),
            "q3_s3_still_uncovered": q3s3_closure_contract,
            "local_q2_lead_context_only": bool(fold_internal_contract and q2_geometry_contract),
            "public_058_evidence": False,
            "local_artifact_allowed": False,
            "requires_explicit_kaggle_approval": True,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Keep the Q2 local lead as context only; pursue the explicit-approved clean ID221→ID226 upstream rebuild "
                "for Q2/Q3/S3 coverage, or design a materially different Q3/S3 source without creating ID342."
                if local_fallback_chain_contract_pass
                else "Fix Q2/Q3/S3 local fallback chain drift before relying on this lane as no-submit context."
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
        "# Post-ID341 Q2/Q3/S3 local fallback chain contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It confirms the local Q2/Q3/S3 fallback lane is context-only: Q2 has a small lead, Q3/S3 remain uncovered, and no submission artifact is allowed.",
        "",
        "## Decision",
        "",
        f"- Local fallback chain contract pass: `{decision['local_fallback_chain_contract_pass']}`",
        f"- Primary score-relevant route: `{decision['primary_score_relevant_route']}`",
        f"- Fallback no-approval route: `{decision['fallback_no_approval_route']}`",
        f"- Missing targets: `{decision['missing_targets']}`",
        f"- Selected local context targets: `{decision['selected_local_context_targets']}`",
        f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
        f"- Local Q2 lead context only: `{decision['local_q2_lead_context_only']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Local artifact allowed: `{decision['local_artifact_allowed']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_q2q3s3_local_fallback_chain_contract_20260618.json",
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
