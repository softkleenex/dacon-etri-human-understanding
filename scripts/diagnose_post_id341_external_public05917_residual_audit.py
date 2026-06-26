#!/usr/bin/env python3
"""Post-ID341 residual audit for the external Public 0.5917 notebook.

Report-only diagnostic. This does not execute the external notebook, train a
model, create predictions, create a submission CSV, edit the submission log,
push/run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

REFERENCE_NOTEBOOK = Path(
    "/Volumes/samsd/download_samsd/[Public_0.5917] LGBM+XGB 앵커·Subject-hole CV·안정성 피처선별·블렌드.ipynb"
)

REQUIRED_REPORTS = {
    "external_review": REPORTS_DIR / "external_public05917_notebook_review_20260616.json",
    "feature_similarity": REPORTS_DIR / "public_feature_similarity_review_post_id221_20260616.json",
    "booster_stability": REPORTS_DIR / "public_booster_stability_overlay_fast_scan_20260616.json",
    "window_pair": REPORTS_DIR / "id215_window_pair_sequence_diagnostic_20260616.json",
    "smartphone_light": REPORTS_DIR / "id215_smartphone_light_compact_diagnostic_20260616.json",
    "s3_smartphone_artifact": REPORTS_DIR
    / "experiment_198_id193_s3_smartphone_light_compact_overlay_cv0.536380_unsubmitted.json",
    "q2q3s3_model_smoke": REPORTS_DIR / "post_id341_q2q3s3_fold_internal_model_smoke_20260618.json",
    "q2_lead_geometry": REPORTS_DIR / "post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json",
    "q3s3_inventory": REPORTS_DIR / "post_id341_q3s3_remaining_route_inventory_20260618.json",
    "transfer_gate_v2": REPORTS_DIR / "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "operator_handoff": REPORTS_DIR / "post_id341_operator_handoff_contract_20260619.json",
}

IDEA_KEYWORDS = {
    "anchor_blending": ["V152", "anchor", "앵커", "base_oof", "base_test"],
    "subject_hole_cv": ["Subject-Hole", "subject_hole", "make_subject_hole_folds", "Subject-Aware"],
    "stability_filtering": ["stability", "안정성", "apply_stability_filter", "feature reduction"],
    "window_pair_screen": ["Window-Pair", "mScreenStatus", "screen", "윈도우"],
    "lgbm_xgb_cat": ["LightGBM", "LGBM", "XGBoost", "XGB", "CatBoost", "Cat"],
    "target_specific_blend": ["Target-Specific", "target_specific", "find_target_specific_weights"],
    "subject_adaptive_calibration": ["Subject-adaptive", "subject_profile", "Subject별", "calibration"],
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def target_best_summary(report: dict[str, Any]) -> dict[str, Any]:
    targets = report.get("targets") or {}
    if not isinstance(targets, dict):
        return {"target_count": 0, "non_base_targets": [], "best_by_target": {}}
    best_by_target = {target: payload.get("best") for target, payload in targets.items() if isinstance(payload, dict)}
    non_base = sorted(target for target, best in best_by_target.items() if best and best != "base")
    return {
        "target_count": len(best_by_target),
        "non_base_targets": non_base,
        "best_by_target": best_by_target,
    }


def notebook_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    markdown_cells = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
    code_cells = sum(1 for cell in cells if cell.get("cell_type") == "code")
    keyword_counts = {
        idea: sum(text.lower().count(keyword.lower()) for keyword in keywords)
        for idea, keywords in IDEA_KEYWORDS.items()
    }
    future_work_match = re.search(r"## 13\..*", text, flags=re.S)
    future_work_excerpt = ""
    if future_work_match:
        future_work_excerpt = "\n".join(future_work_match.group(0).splitlines()[:18])
    return {
        "exists": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "cell_count": len(cells),
        "markdown_cell_count": markdown_cells,
        "code_cell_count": code_cells,
        "keyword_counts": keyword_counts,
        "claimed_public_score_from_filename": 0.5917,
        "future_work_excerpt": future_work_excerpt,
    }


def build_report() -> dict[str, Any]:
    reports = {name: safe_report(path) for name, path in REQUIRED_REPORTS.items()}
    missing_reports = [name for name, path in REQUIRED_REPORTS.items() if not path.exists()]

    external_review = reports["external_review"]
    booster = reports["booster_stability"]
    window_pair = reports["window_pair"]
    smartphone_light = reports["smartphone_light"]
    s3_artifact = reports["s3_smartphone_artifact"]
    q2_smoke_decision = reports["q2q3s3_model_smoke"].get("decision", {})
    q2_geometry_decision = reports["q2_lead_geometry"].get("decision", {})
    q3s3_decision = reports["q3s3_inventory"].get("decision", {})
    transfer_decision = reports["transfer_gate_v2"].get("decision", {})
    handoff_decision = reports["operator_handoff"].get("decision", {})

    booster_results = booster.get("results", [])
    booster_targetwise_best = {
        int(row.get("base_id")): {
            target: payload.get("best")
            for target, payload in (row.get("targets") or {}).items()
            if isinstance(payload, dict)
        }
        for row in booster_results
        if row.get("base_id") is not None
    }
    booster_non_base_count = sum(
        1 for row in booster_targetwise_best.values() for best in row.values() if best and best != "base"
    )

    s3_modified_targets = (s3_artifact.get("selector_info") or {}).get("modified_targets", [])
    q2_selected_targets = q2_smoke_decision.get("selected_targets", [])
    q2_best_proxy = q2_geometry_decision.get("best_proxy")

    idea_status = [
        {
            "idea": "anchor_blending_and_subject_hole_cv",
            "source_notebook_signal": "Core claimed recipe: V152 anchor + Subject-Hole CV + conservative blend.",
            "current_evidence": "Absorbed into current artifact discipline and public-transfer gates.",
            "status": "absorbed_as_validation_principle",
            "artifact_allowed": False,
            "reason": "ID178/ID341 anchor controls remain safer than local-CV-only candidates; no new direct notebook artifact exists.",
        },
        {
            "idea": "stability_selected_lgbm_xgb_side_models",
            "source_notebook_signal": "23,177 -> 1,682 stability-selected feature reduction and LGBM/XGB blend.",
            "current_evidence": f"Fast stability-booster scan non-base target choices: {booster_non_base_count}.",
            "status": "tested_and_closed_as_direct_overlay",
            "artifact_allowed": False,
            "reason": "Existing stability-booster report selected base for every target/base combination.",
        },
        {
            "idea": "window_pair_screen_interactions",
            "source_notebook_signal": "Window-pair/screen-status stability features, especially S targets.",
            "current_evidence": target_best_summary(window_pair),
            "status": "tested_as_sequence_window_pair_context",
            "artifact_allowed": False,
            "reason": "Window-pair diagnostic does not currently clear post-ID341 Public-0.58 transfer gates.",
        },
        {
            "idea": "smartphone_light_screen_compact_s3",
            "source_notebook_signal": "mScreenStatus and phone-light style S-target signal.",
            "current_evidence": {
                "id215_diagnostic": target_best_summary(smartphone_light),
                "promoted_context_artifact_id198_modified_targets": s3_modified_targets,
            },
            "status": "partially_used_but_not_promotable_post_id341",
            "artifact_allowed": False,
            "reason": "S3 compact signal was already tried as ID198 context; Q3/S3 remain uncovered post-ID341.",
        },
        {
            "idea": "target_specific_blending",
            "source_notebook_signal": "Target-specific weights such as Q2 XGB-heavy and S3/S4 Cat-heavy.",
            "current_evidence": {
                "q2_selected_targets_post_id341": q2_selected_targets,
                "q2_best_proxy": q2_best_proxy,
                "q3_s3_still_uncovered": q3s3_decision.get("q3_s3_still_uncovered"),
            },
            "status": "q2_context_only_q3s3_missing",
            "artifact_allowed": False,
            "reason": "Post-ID341 local fallback finds Q2 only; Q3/S3 coverage is still missing and proxy is not Public-0.58 plausible.",
        },
        {
            "idea": "subject_adaptive_or_subject_profile_calibration",
            "source_notebook_signal": "Subject profile analysis and future subject-adaptive calibration.",
            "current_evidence": "Subject/demographic/calibration descendants are represented in existing diagnostics and are blocked by transfer/joint-risk gates.",
            "status": "research_only_not_submission_evidence",
            "artifact_allowed": False,
            "reason": "ID287/ID329 demonstrated that local calibration/recombination can collapse or plateau publicly.",
        },
    ]

    residual_direct_routes = [row for row in idea_status if row["artifact_allowed"]]
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_external_public05917_residual_audit",
        "purpose": (
            "Audit whether the external Public 0.5917 reference notebook still contains an untested direct "
            "post-ID341 score route after current Q2/Q3/S3 and Q3/S3 closures."
        ),
        "reference_notebook": notebook_summary(REFERENCE_NOTEBOOK),
        "required_reports": {
            name: {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
            for name, path in REQUIRED_REPORTS.items()
        },
        "missing_reports": missing_reports,
        "external_review_summary": {
            "claimed_public_score": external_review.get("claimed_public_score"),
            "claimed_local_oof": external_review.get("claimed_local_oof"),
            "claimed_local_public_gap": external_review.get("claimed_local_public_gap"),
            "practical_assessment": external_review.get("practical_assessment"),
            "transferable_ideas": external_review.get("transferable_ideas", []),
        },
        "idea_status": idea_status,
        "decision": {
            "audit_ready": not missing_reports and bool(REFERENCE_NOTEBOOK.exists()),
            "residual_direct_notebook_route_exists": bool(residual_direct_routes),
            "residual_direct_routes": [row["idea"] for row in residual_direct_routes],
            "public_058_evidence_present": False,
            "q3_s3_still_uncovered": q3s3_decision.get("q3_s3_still_uncovered"),
            "only_score_relevant_open_route": q3s3_decision.get("only_score_relevant_open_route"),
            "prelaunch_ready_for_explicit_approval": reports["operator_handoff"]
            .get("state_checks", {})
            .get("prelaunch_ready"),
            "operator_handoff_ready": handoff_decision.get("operator_handoff_ready"),
            "approval_phrase": handoff_decision.get("approval_phrase"),
            "transfer_gate_public_058_candidate_exists": transfer_decision.get("public_058_plausible_candidate_exists"),
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not reopen the external Public 0.5917 notebook as a direct local submission route. "
                "Keep its anchor/stability ideas as validation philosophy; the score-relevant route remains "
                "the explicit-approved no-submit Kaggle upstream rebuild."
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
    return report


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 external Public 0.5917 residual audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It does not execute the external notebook, train a model, create predictions, create a submission artifact, push/run Kaggle, or submit to DACON.",
        "",
        "## Decision",
        "",
        f"- Audit ready: `{decision['audit_ready']}`",
        f"- Residual direct notebook route exists: `{decision['residual_direct_notebook_route_exists']}`",
        f"- Public 0.58 evidence present: `{decision['public_058_evidence_present']}`",
        f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
        f"- Only score-relevant open route: `{decision['only_score_relevant_open_route']}`",
        f"- Operator handoff ready: `{decision['operator_handoff_ready']}`",
        f"- Approval phrase: `{decision['approval_phrase']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Reference notebook",
        "",
        f"- Exists: `{report['reference_notebook'].get('exists')}`",
        f"- Cells: `{report['reference_notebook'].get('cell_count')}`",
        f"- SHA256: `{report['reference_notebook'].get('sha256')}`",
        f"- Claimed Public from filename: `{report['reference_notebook'].get('claimed_public_score_from_filename')}`",
        "",
        "## Idea status",
        "",
        "| idea | status | artifact allowed | reason |",
        "|---|---|---:|---|",
    ]
    for row in report["idea_status"]:
        lines.append(f"| `{row['idea']}` | `{row['status']}` | `{row['artifact_allowed']}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## External review summary",
            "",
            "```json",
            json.dumps(report["external_review_summary"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Required reports",
            "",
            "```json",
            json.dumps(report["required_reports"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_external_public05917_residual_audit_20260619.json",
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
    return 0 if report["decision"]["audit_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
