#!/usr/bin/env python3
"""Design a no-artifact Q3/S3 sibling-target bridge contract.

Q3 is most correlated with Q2 in the official labels, and S3 is most correlated
with S2. This report-only diagnostic turns that anatomy finding into a leakage
controlled design contract for future upstream rebuild work. It does not train a
model, produce predictions, create submission artifacts, edit the submission
log, run Kaggle, or submit to DACON.
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_sibling_target_bridge_design_20260619.json"
ANATOMY = REPORTS_DIR / "post_id341_q3s3_target_anatomy_20260619.json"
NEGATIVE_MAP = REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json"
FALLBACK_CHAIN = REPORTS_DIR / "post_id341_q2q3s3_local_fallback_chain_contract_20260618.json"
FRONTIER = REPORTS_DIR / "post_id341_no_approval_frontier_20260619.json"
OPERATOR_HANDOFF = REPORTS_DIR / "post_id341_operator_handoff_contract_20260619.json"
APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
PRIMARY_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "parse_ok": False, "decision": {}, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "parse_ok": False, "decision": {}, "path": rel(path), "error": str(exc)}
    payload.setdefault("path", rel(path))
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("decision", {})


def build_bridge_rows(anatomy: dict[str, Any]) -> list[dict[str, Any]]:
    read = anatomy.get("anatomy_read", {})
    q3_corr = read.get("q3_top_same_day_correlation", {})
    s3_corr = read.get("s3_top_same_day_correlation", {})
    return [
        {
            "bridge": "q3_from_q2_subjective_state_sibling",
            "target": "Q3",
            "sibling_target": q3_corr.get("target"),
            "label_corr": q3_corr.get("pearson_corr"),
            "evidence": "Q3 stress and Q2 fatigue share a subjective evening-state axis.",
            "allowed_use": "future_upstream_fold_nested_design_only",
            "blocked_local_shortcut": "Do not blend current Q2 predictions or Q2 labels into Q3 outside a fold-nested rebuild.",
            "requires_test_time_sibling_prediction": True,
            "requires_fold_nested_oof": True,
            "create_artifact_now": False,
        },
        {
            "bridge": "s3_from_s2_objective_sleep_sibling",
            "target": "S3",
            "sibling_target": s3_corr.get("target"),
            "label_corr": s3_corr.get("pearson_corr"),
            "evidence": "S3 sleep-onset-latency guideline and S2 sleep-efficiency guideline share objective sleep-quality structure.",
            "allowed_use": "future_upstream_fold_nested_design_only",
            "blocked_local_shortcut": "Do not blend current S2 predictions or S2 labels into S3 outside a fold-nested rebuild.",
            "requires_test_time_sibling_prediction": True,
            "requires_fold_nested_oof": True,
            "create_artifact_now": False,
        },
    ]


def build_report() -> dict[str, Any]:
    anatomy = load_report(ANATOMY)
    negative_map = load_report(NEGATIVE_MAP)
    fallback = load_report(FALLBACK_CHAIN)
    frontier = load_report(FRONTIER)
    handoff = load_report(OPERATOR_HANDOFF)
    bridge_rows = build_bridge_rows(anatomy)
    anatomy_decision = decision(anatomy)
    negative_decision = decision(negative_map)
    fallback_decision = decision(fallback)
    frontier_decision = decision(frontier)
    handoff_decision = decision(handoff)
    q3_row = bridge_rows[0]
    s3_row = bridge_rows[1]
    checks = {
        "all_inputs_ready": all(
            report.get("exists") and report.get("parse_ok")
            for report in [anatomy, negative_map, fallback, frontier, handoff]
        ),
        "anatomy_ready": anatomy_decision.get("q3s3_target_anatomy_ready") is True,
        "q3_sibling_is_q2": q3_row.get("sibling_target") == "Q2",
        "s3_sibling_is_s2": s3_row.get("sibling_target") == "S2",
        "q3_corr_is_material": float(q3_row.get("label_corr") or 0.0) >= 0.30,
        "s3_corr_is_material": float(s3_row.get("label_corr") or 0.0) >= 0.30,
        "local_q3s3_lanes_closed": negative_decision.get("all_local_q3s3_lanes_closed") is True,
        "q3_s3_still_uncovered": negative_decision.get("q3_s3_still_uncovered") is True,
        "q2_context_only": fallback_decision.get("local_q2_lead_context_only") is True,
        "no_approval_score_route_closed": frontier_decision.get("no_approval_score_route_exists") is False,
        "operator_handoff_ready": handoff_decision.get("operator_handoff_ready") is True,
        "operator_requires_exact_approval": handoff_decision.get("approval_phrase") == APPROVAL_PHRASE,
    }
    design_ready = all(checks.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_sibling_target_bridge_design",
        "purpose": "Convert Q3/S3 sibling-label anatomy into a leakage-safe upstream design contract.",
        "inputs": {
            "target_anatomy": anatomy.get("path", rel(ANATOMY)),
            "negative_evidence_map": negative_map.get("path", rel(NEGATIVE_MAP)),
            "fallback_chain": fallback.get("path", rel(FALLBACK_CHAIN)),
            "no_approval_frontier": frontier.get("path", rel(FRONTIER)),
            "operator_handoff": handoff.get("path", rel(OPERATOR_HANDOFF)),
        },
        "target_focus": ["Q3", "S3"],
        "bridge_rows": bridge_rows,
        "leakage_controls": [
            "Sibling target labels may not be used as direct features for the same row.",
            "Any sibling-target prediction feature must be generated out-of-fold for train rows and by a separately fitted test-time sibling model.",
            "Current full-OOF/local artifact recombinations are not sufficient evidence for ID342.",
            "A sibling bridge can only be evaluated inside the explicit-approved upstream rebuild or a future no-artifact fold-nested local smoke.",
        ],
        "checks": checks,
        "decision": {
            "sibling_bridge_design_ready": design_ready,
            "target_focus": ["Q3", "S3"],
            "material_sibling_structure_found": checks["q3_corr_is_material"] and checks["s3_corr_is_material"],
            "local_artifact_allowed": False,
            "public_058_evidence": False,
            "requires_fold_nested_rebuild": True,
            "recommended_route": PRIMARY_ROUTE,
            "requires_explicit_kaggle_approval": True,
            "approval_phrase": APPROVAL_PHRASE,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Carry Q3<-Q2 and S3<-S2 sibling bridges only as fold-nested upstream design requirements. "
                "Do not create ID342 or blend sibling predictions locally from existing artifacts."
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
        "# Post-ID341 Q3/S3 sibling-target bridge design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model and creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Sibling bridge design ready: `{decision['sibling_bridge_design_ready']}`",
        f"- Material sibling structure found: `{decision['material_sibling_structure_found']}`",
        f"- Requires fold-nested rebuild: `{decision['requires_fold_nested_rebuild']}`",
        f"- Recommended route: `{decision['recommended_route']}`",
        f"- Requires explicit Kaggle approval: `{decision['requires_explicit_kaggle_approval']}`",
        f"- Approval phrase: `{decision['approval_phrase']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Bridge rows",
        "",
    ]
    for row in report["bridge_rows"]:
        lines.extend(
            [
                f"### {row['bridge']}",
                f"- Target: `{row['target']}`",
                f"- Sibling target: `{row['sibling_target']}`",
                f"- Label correlation: `{row['label_corr']:.6f}`",
                f"- Allowed use: `{row['allowed_use']}`",
                f"- Blocked shortcut: {row['blocked_local_shortcut']}",
                "",
            ]
        )
    lines.extend(["## Leakage controls", ""])
    for item in report["leakage_controls"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
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
    return 0 if report["decision"]["sibling_bridge_design_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
