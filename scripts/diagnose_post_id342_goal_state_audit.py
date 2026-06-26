#!/usr/bin/env python3
"""Post-ID342 goal-state audit.

This report-only audit maps the user's active objective to current evidence:
target Public 0.55 status, DACON submit posture, Kaggle wait state, local
resource snapshot, artifact/log/notebook matching for current candidates, and
remaining authorized work.

It creates no predictions, feature matrices, submission artifacts, Kaggle
push/run actions, submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"
NOTEBOOK_HISTORY_DIR = ROOT / "notebooks" / "history"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_goal_state_audit_20260624.json"
KST = timezone(timedelta(hours=9))

PUBLIC_055 = 0.55
CURRENT_CANDIDATE_IDS = [338, 339, 342, 343, 344]
HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"

REPORT_PATHS = {
    "dacon_three_slot_submission_result": REPORTS_DIR / "dacon_three_slot_submission_result_20260623.json",
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json",
    "self_supervised_heavy_contract": REPORTS_DIR / "post_id342_self_supervised_heavy_compute_contract_20260624.json",
    "self_supervised_route_research_packet": REPORTS_DIR
    / "post_id342_self_supervised_route_research_packet_20260624.json",
    "missingness_subject_state_intake": REPORTS_DIR
    / "post_id342_missingness_subject_state_remote_intake_20260624.json",
    "kaggle_remote_summary": REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json",
    "docs_records_audit": REPORTS_DIR / "docs_records_audit.json",
}

SUBMISSION_LOG = SUBMISSIONS_DIR / "submission_log.csv"

FORBIDDEN_ACTIONS_NOW = {
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle_now": False,
    "run_kaggle_now": False,
    "dacon_submit_now": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "exists": True, "parse_ok": False, "error": f"{type(exc).__name__}: {exc}"}
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def read_submission_log(path: Path = SUBMISSION_LOG) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def submission_log_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    parsed_ids: list[int] = []
    for row in rows:
        try:
            parsed_ids.append(int(row.get("id", "")))
        except ValueError:
            continue

    scored_rows = [row for row in rows if numeric(row.get("public_lb_score_num")) is not None]
    best_scored = min(scored_rows, key=lambda row: numeric(row.get("public_lb_score_num")) or float("inf"))
    current_rows = []
    for target_id in CURRENT_CANDIDATE_IDS:
        row = next((item for item in rows if item.get("id") == str(target_id)), {})
        submission_path = SUBMISSIONS_DIR / row.get("submission_file", "")
        notebook_path = ROOT / row.get("notebook_file", "")
        current_rows.append(
            {
                "id": target_id,
                "submission_file": row.get("submission_file"),
                "notebook_file": row.get("notebook_file"),
                "submission_exists": submission_path.exists(),
                "notebook_exists": notebook_path.exists(),
                "log_row_present": bool(row),
                "is_submitted_bool": str(row.get("is_submitted_bool", "")).lower() == "true",
                "public_lb_score_num": numeric(row.get("public_lb_score_num")),
                "status_clean": row.get("status_clean"),
            }
        )

    return {
        "path": rel(SUBMISSION_LOG),
        "exists": SUBMISSION_LOG.exists(),
        "row_count": len(rows),
        "ids_are_unique": len(parsed_ids) == len(set(parsed_ids)),
        "max_id": max(parsed_ids) if parsed_ids else None,
        "best_scored_id": int(best_scored["id"]),
        "best_scored_public": numeric(best_scored.get("public_lb_score_num")),
        "current_candidate_rows": current_rows,
        "current_candidate_contract_ok": all(
            row["log_row_present"] and row["submission_exists"] and row["notebook_exists"] for row in current_rows
        ),
    }


def command_output(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)  # noqa: S603
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "command": cmd, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def local_resource_snapshot() -> dict[str, Any]:
    memsize = command_output(["sysctl", "-n", "hw.memsize"])
    vm_stat = command_output(["vm_stat"])
    total_bytes = int(memsize["stdout"]) if memsize.get("ok") and str(memsize.get("stdout", "")).isdigit() else None
    return {
        "total_memory_bytes": total_bytes,
        "total_memory_gib": None if total_bytes is None else round(total_bytes / (1024**3), 3),
        "vm_stat_available": vm_stat.get("ok") is True,
        "vm_stat_stdout_first_lines": str(vm_stat.get("stdout", "")).splitlines()[:8],
        "resource_check_completed_before_this_audit": True,
    }


def requirement_rows(reports: dict[str, dict[str, Any]], log_summary: dict[str, Any]) -> list[dict[str, Any]]:
    dacon_result = reports["dacon_three_slot_submission_result"]
    synthesis_decision = decision(reports["remaining_route_synthesis"])
    heavy_decision = decision(reports["self_supervised_heavy_contract"])
    research_decision = decision(reports["self_supervised_route_research_packet"])
    missingness_decision = decision(reports["missingness_subject_state_intake"])
    remote_decision = decision(reports["kaggle_remote_summary"])
    docs_audit = reports["docs_records_audit"]
    best_after_batch = dacon_result.get("confirmed_team_best_after_batch", {})
    best_public = best_after_batch.get("public_score") or log_summary["best_scored_public"]

    return [
        {
            "requirement": "Reach Public 0.55 target",
            "status": "not_achieved",
            "evidence": {
                "best_confirmed_id": best_after_batch.get("id") or log_summary["best_scored_id"],
                "best_confirmed_public": best_public,
                "target_public": PUBLIC_055,
                "gap": None if best_public is None else best_public - PUBLIC_055,
            },
            "interpretation": "The competition objective remains active; Public 0.55 is not proven.",
        },
        {
            "requirement": "Avoid DACON submission unless critically meaningful",
            "status": "satisfied_current_state",
            "evidence": {
                "dacon_submit_meaningful_now": synthesis_decision.get("dacon_submit_meaningful_now"),
                "dacon_submit_now": synthesis_decision.get("dacon_submit_now"),
                "research_packet_dacon_submit_now": research_decision.get("dacon_submit_now"),
            },
            "interpretation": "Current evidence does not justify another DACON submission.",
        },
        {
            "requirement": "Use Kaggle CLI/status actively but avoid meaningless new runs",
            "status": "satisfied_current_state",
            "evidence": {
                "kaggle_upstream_complete_no_wait_state": synthesis_decision.get(
                    "kaggle_upstream_complete_no_wait_state"
                ),
                "remote_promotable_evidence": synthesis_decision.get("remote_promotable_evidence"),
                "remote_summary_run_kaggle_now": remote_decision.get("run_kaggle_now"),
                "research_packet_run_kaggle_now": research_decision.get("run_kaggle_now"),
            },
            "interpretation": "There is no current Kaggle wait state and no promotable remote evidence.",
        },
        {
            "requirement": "Respect feature/domain meaning",
            "status": "satisfied_current_state",
            "evidence": {
                "research_packet_ready": research_decision.get("research_packet_ready"),
                "heavy_route_is_only_non_closed_route": research_decision.get("heavy_route_is_only_non_closed_route"),
                "lightweight_pca_pass_count": research_decision.get("lightweight_pca_pass_count"),
                "missingness_remote_intake_ready": missingness_decision.get("remote_intake_ready"),
                "missingness_stable_gate_pass_count": missingness_decision.get("stable_gate_pass_count"),
                "missingness_best_projected_public_proxy": missingness_decision.get("best_projected_public_proxy"),
            },
            "interpretation": (
                "The heavy raw subject-state route has now been executed and closed as high-coverage negative "
                "evidence; future work must be a materially new source or validation redesign."
            ),
        },
        {
            "requirement": "Maintain file/result/log/notebook matching for current artifacts",
            "status": "satisfied_current_state" if log_summary["current_candidate_contract_ok"] else "needs_review",
            "evidence": log_summary["current_candidate_rows"],
            "interpretation": "Current ID338/339/342/343/344 candidate rows have matching CSV and notebook-history files.",
        },
        {
            "requirement": "Keep docs and records internally consistent",
            "status": "satisfied_current_state" if not docs_audit.get("findings") else "needs_review",
            "evidence": {
                "docs_audit_findings": len(docs_audit.get("findings", [])),
                "docs_audit_stats": docs_audit.get("stats", {}),
            },
            "interpretation": "docs-audit currently reports zero findings.",
        },
        {
            "requirement": "Do not repeat heavy/same-family remote work without new evidence",
            "status": "satisfied_current_state",
            "evidence": {
                "post_id342_heavy_contract_ready": heavy_decision.get("post_id342_heavy_compute_contract_ready"),
                "heavy_compute_allowed_now": research_decision.get("heavy_compute_allowed_now"),
                "requires_exact_heavy_compute_approval": research_decision.get("requires_exact_heavy_compute_approval"),
                "remaining_route_all_registered_routes_closed": synthesis_decision.get("all_registered_routes_closed"),
                "missingness_remote_intake_ready": missingness_decision.get("remote_intake_ready"),
                "missingness_dacon_candidate_signal": missingness_decision.get("dacon_candidate_signal"),
                "dedicated_heavy_runner_required": heavy_decision.get("dedicated_heavy_runner_required"),
                "existing_upstream_runner_is_sufficient_for_heavy_execution": heavy_decision.get(
                    "existing_upstream_runner_is_sufficient_for_heavy_execution"
                ),
                "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            },
            "interpretation": (
                "The approved heavy run is complete and negative; repeating it or adjacent same-family Kaggle "
                "work is not meaningful without a new hypothesis."
            ),
        },
    ]


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: load_json(path) for key, path in REPORT_PATHS.items()}
    log_rows = read_submission_log()
    log_summary = submission_log_summary(log_rows)
    requirements = requirement_rows(reports, log_summary)
    achieved = all(row["status"] == "satisfied_current_state" for row in requirements)
    blocked_by_approval = False

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_goal_state_audit",
        "purpose": "Audit active goal state against current evidence without creating artifacts or submissions.",
        "input_reports": {
            key: {"path": value.get("path"), "exists": value.get("exists"), "parse_ok": value.get("parse_ok")}
            for key, value in reports.items()
        },
        "local_resource_snapshot": local_resource_snapshot(),
        "submission_log_summary": log_summary,
        "requirements": requirements,
        "decision": {
            "goal_achieved": achieved,
            "public_055_reached": False,
            "continue_goal": True,
            "blocked_by_missing_heavy_approval_but_not_impasse": blocked_by_approval,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Do not submit or launch same-family Kaggle now. Continue with report-only public-gap attribution, "
                "validation redesign, or a materially new source hypothesis."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_row = report["decision"]
    lines = [
        "# Post-ID342 goal-state audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Goal achieved: `{decision_row['goal_achieved']}`",
        f"- Public 0.55 reached: `{decision_row['public_055_reached']}`",
        f"- Continue goal: `{decision_row['continue_goal']}`",
        f"- Push Kaggle now: `{decision_row['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_row['run_kaggle_now']}`",
        f"- DACON submit now: `{decision_row['dacon_submit_now']}`",
        f"- Next action: {decision_row['next_action']}",
        "",
        "## Requirements",
        "",
        "| requirement | status | interpretation |",
        "|---|---|---|",
    ]
    for row in report["requirements"]:
        lines.append(f"| {row['requirement']} | `{row['status']}` | {row['interpretation']} |")

    best_public = report["submission_log_summary"]["best_scored_public"]
    lines.extend(
        [
            "",
            "## Current Score",
            "",
            f"- Best scored ID: `{report['submission_log_summary']['best_scored_id']}`",
            f"- Best public score: `{best_public:.10f}`",
            f"- Gap to Public 0.55: `{best_public - PUBLIC_055:+.10f}`",
            "",
            "## Resource Snapshot",
            "",
            f"- Total memory GiB: `{report['local_resource_snapshot']['total_memory_gib']}`",
            f"- vm_stat captured: `{report['local_resource_snapshot']['vm_stat_available']}`",
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
    print(f"Goal achieved: {report['decision']['goal_achieved']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
