#!/usr/bin/env python3
"""Post-ID342 deadline-aware operating policy.

This report folds the current verified competition evidence into a deadline
mode. It creates no predictions, feature matrices, submission artifacts,
Kaggle actions, submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_deadline_operating_policy_20260625.json"
KST = timezone(timedelta(hours=9))

CURRENT_ANCHOR_ID = 342
CURRENT_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
PAPER_DEADLINE_DATE = "2026-06-26"
ALGORITHM_CLOSE_DATE = "2026-07-23"
PAPER_DEADLINE_SOURCE = "https://ictc.org/authors_paperSubmission"
PAPER_DEADLINE_CORROBORATING_SOURCE = "https://dacon.io/competitions/official/236690/talkboard/416793"
ALGORITHM_CLOSE_SOURCE = "https://dacon.io/en/competitions/official/236690/overview/description"

SYNTHESIS_PATH = REPORTS_DIR / "post_id342_remaining_route_synthesis_20260625.json"
PRIORITY_PATH = REPORTS_DIR / "post_id342_external_research_priority_20260625.json"
PAPER_PACKAGE_AUDIT_PATH = REPORTS_DIR / "paper_submission_package_audit_20260625.json"
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
PAPER_TEX_PATH = ROOT / "docs" / "paper_submission_20260626" / "paper.tex"
PAPER_FINAL_HANDOFF_PATH = ROOT / "docs" / "paper_submission_20260626" / "paper_submission_final_handoff.md"
PAPER_COMPILE_NOTE_PATH = ROOT / "docs" / "paper_submission_20260626" / "overleaf_compile_note.md"
PAPER_SOURCE_ZIP_PATH = ROOT / "build" / "paper_submission_20260626" / "paper_submission_20260626_source.zip"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": rel(path), "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "path": rel(path), "parse_ok": False, "error": f"{type(exc).__name__}: {exc}"}
    payload["exists"] = True
    payload["path"] = rel(path)
    payload["parse_ok"] = True
    return payload


def submission_summary() -> dict[str, Any]:
    if not SUBMISSION_LOG.exists():
        return {"exists": False, "path": rel(SUBMISSION_LOG)}
    rows: list[dict[str, str]] = []
    with SUBMISSION_LOG.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    submitted = [
        row
        for row in rows
        if str(row.get("is_submitted_bool") or row.get("is_submitted") or row.get("submitted") or "").lower()
        == "true"
    ]
    scored = []
    for row in rows:
        public = (
            row.get("public_score")
            or row.get("public_lb_score_num")
            or row.get("public_lb_score")
            or row.get("lb_score")
            or row.get("leaderboard_score")
        )
        try:
            public_value = float(public) if public not in {None, ""} else None
        except ValueError:
            public_value = None
        submitted_flag = str(row.get("is_submitted_bool") or row.get("is_submitted") or row.get("submitted") or "").lower()
        if public_value is not None and public_value > 0.0 and submitted_flag == "true":
            scored.append({**row, "_public_value": public_value})
    best = min(scored, key=lambda row: row["_public_value"]) if scored else None
    recent = rows[-8:]
    return {
        "exists": True,
        "path": rel(SUBMISSION_LOG),
        "row_count": int(len(rows)),
        "submitted_count": int(len(submitted)),
        "scored_count": int(len(scored)),
        "best_scored_id": best.get("id") if best else None,
        "best_scored_file": (best.get("submission_file") or best.get("filename")) if best else None,
        "best_public_score": best.get("_public_value") if best else None,
        "recent_rows": recent,
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(KST)
    paper_deadline_date = datetime.fromisoformat(PAPER_DEADLINE_DATE).replace(tzinfo=KST)
    algorithm_close_date = datetime.fromisoformat(ALGORITHM_CLOSE_DATE).replace(tzinfo=KST)
    synthesis = read_json(SYNTHESIS_PATH)
    priority = read_json(PRIORITY_PATH)
    paper_package_audit = read_json(PAPER_PACKAGE_AUDIT_PATH)
    synthesis_decision = synthesis.get("decision", {})
    priority_decision = priority.get("decision", {})
    paper_package_decision = paper_package_audit.get("decision", {})
    sub_summary = submission_summary()
    hours_until_paper_date = (paper_deadline_date - now).total_seconds() / 3600.0
    days_until_algorithm_close = (algorithm_close_date - now).total_seconds() / 86400.0

    no_current_candidate = (
        synthesis_decision.get("run_kaggle_now") is False
        and synthesis_decision.get("dacon_submit_now") is False
        and priority_decision.get("run_kaggle_now") is False
        and priority_decision.get("dacon_submit_now") is False
    )
    decision = {
        "deadline_mode_active": True,
        "now_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "paper_deadline_date": PAPER_DEADLINE_DATE,
        "paper_deadline_exact_time_known": False,
        "hours_until_paper_deadline_date_start": hours_until_paper_date,
        "algorithm_close_date": ALGORITHM_CLOSE_DATE,
        "days_until_algorithm_close_date_start": days_until_algorithm_close,
        "current_public_anchor_id": CURRENT_ANCHOR_ID,
        "current_public_score": CURRENT_PUBLIC,
        "target_public_score": PUBLIC_055,
        "gap_to_public_055": CURRENT_PUBLIC - PUBLIC_055,
        "registered_routes_closed": synthesis_decision.get("closed_route_count"),
        "registered_route_count": synthesis_decision.get("route_count"),
        "same_family_or_external_route_available_now": False,
        "no_current_candidate": bool(no_current_candidate),
        "paper_package_audit_pass": paper_package_decision.get("package_audit_pass"),
        "paper_package_human_fillins_remaining": paper_package_decision.get("human_fillins_remaining"),
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "dacon_spend_policy": (
            "Do not spend DACON slots before a candidate clears a fresh local/stable-shape/failed-frontier gate. "
            "If a daily slot is expiring, use it only for an explicitly named information probe whose downside is "
            "accepted; current evidence does not justify such a probe."
        ),
        "paper_handoff_paths": {
            "working_tex": rel(PAPER_TEX_PATH),
            "final_handoff": rel(PAPER_FINAL_HANDOFF_PATH),
            "compile_note": rel(PAPER_COMPILE_NOTE_PATH),
            "source_zip": rel(PAPER_SOURCE_ZIP_PATH),
            "package_audit": rel(PAPER_PACKAGE_AUDIT_PATH),
        },
        "next_two_hours": [
            "Freeze new long Kaggle/heavy runs.",
            "Use docs/paper_submission_20260626/paper.tex as the working IEEE source draft.",
            "Use docs/paper_submission_20260626/paper_submission_final_handoff.md for the exact handoff paths.",
            "Upload build/paper_submission_20260626/paper_submission_20260626_source.zip to Overleaf if a TeX PDF is needed.",
            "Fill human-only author/affiliation/EDAS fields and compile the four-page PDF outside this workspace if needed.",
            "Allow only report-only or local gates expected to finish within 30-60 minutes.",
            "Do not submit ID343/ID344 or other tiny micro-signal files without a new deadline-specific approval.",
        ],
        "next_day": [
            "If paper submission is required by 2026-06-26, prioritize method narrative, validation protocol, and negative-evidence appendix.",
            "Continue leaderboard work only through materially new Q3/S3/S4 hypotheses that can be locally gated.",
            "Keep ID342 as the best confirmed public anchor unless a new scored result beats 0.5960832519.",
        ],
    }
    return {
        "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_deadline_operating_policy",
        "purpose": "Switch post-ID342 work into deadline-aware operating mode.",
        "official_deadline_sources": {
            "algorithm_close": {
                "date": ALGORITHM_CLOSE_DATE,
                "source_url": ALGORITHM_CLOSE_SOURCE,
                "note": "DACON overview search result listed 07.23 Close and 04.21 Start Date.",
            },
            "paper_submission": {
                "date": PAPER_DEADLINE_DATE,
                "exact_time_known": False,
                "source_url": PAPER_DEADLINE_SOURCE,
                "corroborating_source_url": PAPER_DEADLINE_CORROBORATING_SOURCE,
                "note": (
                    "ICTC 2026 paper-submission page lists Paper Submission Deadline: June 26, 2026. "
                    "DACON talkboard also states the ETRI paper deadline follows ICTC 2026 and is 2026-06-26."
                ),
            },
        },
        "evidence_inputs": {
            "synthesis": {"path": rel(SYNTHESIS_PATH), "exists": synthesis.get("exists"), "parse_ok": synthesis.get("parse_ok")},
            "priority": {"path": rel(PRIORITY_PATH), "exists": priority.get("exists"), "parse_ok": priority.get("parse_ok")},
            "paper_package_audit": {
                "path": rel(PAPER_PACKAGE_AUDIT_PATH),
                "exists": paper_package_audit.get("exists"),
                "parse_ok": paper_package_audit.get("parse_ok"),
                "package_audit_pass": paper_package_decision.get("package_audit_pass"),
                "human_fillins_remaining": paper_package_decision.get("human_fillins_remaining"),
            },
            "submission_log": sub_summary,
        },
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID342 deadline operating policy",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Deadline mode active: `{decision['deadline_mode_active']}`",
        f"- Now: `{decision['now_kst']}`",
        f"- Paper deadline date: `{decision['paper_deadline_date']}`",
        f"- Paper deadline exact time known: `{decision['paper_deadline_exact_time_known']}`",
        f"- Algorithm close date: `{decision['algorithm_close_date']}`",
        f"- Current public anchor: `ID{decision['current_public_anchor_id']}` / `{decision['current_public_score']:.10f}`",
        f"- Gap to Public 0.55: `{decision['gap_to_public_055']:+.10f}`",
        f"- Registered routes closed: `{decision['registered_routes_closed']}/{decision['registered_route_count']}`",
        f"- No current candidate: `{decision['no_current_candidate']}`",
        f"- Paper package audit pass: `{decision['paper_package_audit_pass']}`",
        f"- Paper package human fill-ins remaining: `{decision['paper_package_human_fillins_remaining']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- DACON spend policy: {decision['dacon_spend_policy']}",
        "",
        "## Paper Handoff Paths",
        "",
        "```json",
        json.dumps(decision["paper_handoff_paths"], indent=2, sort_keys=True),
        "```",
        "",
        "## Next Two Hours",
        "",
    ]
    lines.extend(f"- {item}" for item in decision["next_two_hours"])
    lines.extend(
        [
            "",
            "## Next Day",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in decision["next_day"])
    lines.extend(
        [
            "",
            "## Official Deadline Sources",
            "",
            "```json",
            json.dumps(report["official_deadline_sources"], indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence Inputs",
            "",
            "```json",
            json.dumps(report["evidence_inputs"], indent=2, sort_keys=True, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Deadline mode active: {decision['deadline_mode_active']} | submit_now={decision['dacon_submit_now']}")


if __name__ == "__main__":
    main()
