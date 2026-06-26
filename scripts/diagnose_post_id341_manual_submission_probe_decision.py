#!/usr/bin/env python3
"""Manual DACON probe decision after the post-ID341 local gates.

This report-only diagnostic answers a narrow operational question:
if the user insists on spending one DACON submission as a public sanity probe,
which existing CSV is the least-bad controlled option?

It does not submit to DACON, push/run Kaggle, create a submission CSV, create a
notebook/history artifact, create base predictions, or edit submission_log.csv.
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
SUBMISSIONS_DIR = ROOT / "submissions"
KST = timezone(timedelta(hours=9))

COMBINED_GATE_PATH = REPORTS_DIR / "post_id341_transfer_gate_v2_joint_risk_20260618.json"
ID341_RISK_GATE_PATH = REPORTS_DIR / "id341_q2_seqsum_w00025_public_risk_gate_20260618.json"
PRELAUNCH_GATE_PATH = REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json"
SUBMISSION_LOG_PATH = SUBMISSIONS_DIR / "submission_log.csv"

FOCUS_IDS = [337, 338, 339, 340, 341]
SAFETY_FIRST_ID = 341
PROXY_LEADER_ID = 340
LOCAL_LEADER_ID = 339
ANCHOR_ID = 178
EXPECTED_COLUMNS = ["subject_id", "sleep_date", "lifelog_date", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


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


def load_log() -> pd.DataFrame:
    df = pd.read_csv(SUBMISSION_LOG_PATH)
    df["id_num"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df


def find_combined_row(combined_gate: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    sections = [
        "recent_control_rows",
        "top_transfer_proxy_rows",
        "joint_safe_but_transfer_blocked",
        "local_cv_frontier_combined",
        "combined_pass_rows",
    ]
    for section in sections:
        for row in combined_gate.get(section, []):
            if int(row.get("id", -1)) == candidate_id:
                return row
    return {}


def log_row(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"] == candidate_id]
    if rows.empty:
        return {"id": candidate_id, "exists_in_log": False}
    row = rows.iloc[-1].to_dict()
    row["exists_in_log"] = True
    return row


def csv_status(submission_file: str | None) -> dict[str, Any]:
    if not submission_file:
        return {"path": None, "exists": False, "shape": None, "columns_ok": False, "row_count_ok": False}
    path = SUBMISSIONS_DIR / submission_file
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "shape": None, "columns_ok": False, "row_count_ok": False}
    df = pd.read_csv(path)
    return {
        "path": safe_relative(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "columns_ok": list(df.columns) == EXPECTED_COLUMNS,
        "row_count_ok": int(df.shape[0]) == 250,
        "target_min": float(df[EXPECTED_COLUMNS[3:]].min().min()),
        "target_max": float(df[EXPECTED_COLUMNS[3:]].max().max()),
    }


def notebook_status(row: dict[str, Any]) -> dict[str, Any]:
    notebook = row.get("notebook_file")
    if not isinstance(notebook, str) or not notebook:
        return {"path": notebook, "exists": False}
    path = ROOT / notebook
    return {
        "path": safe_relative(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def candidate_card(candidate_id: int, combined_gate: dict[str, Any], log: pd.DataFrame) -> dict[str, Any]:
    combined_row = find_combined_row(combined_gate, candidate_id)
    row = log_row(log, candidate_id)
    submission_file = row.get("submission_file") or combined_row.get("submission_file")
    status = csv_status(submission_file)
    notebook = notebook_status(row)
    public_lb = row.get("public_lb_score_num")
    if pd.isna(public_lb):
        public_lb = None
    is_submitted = bool(row.get("is_submitted_bool")) if row.get("exists_in_log") else None
    return {
        "id": candidate_id,
        "submission_file": submission_file,
        "model_description": row.get("model_description"),
        "local_cv": combined_row.get("local_cv", row.get("local_cv_score_num")),
        "public_lb": public_lb,
        "is_submitted_bool": is_submitted,
        "status_clean": row.get("status_clean"),
        "duplicate_file_count": int(row.get("duplicate_file_count") or 0) if row.get("exists_in_log") else None,
        "artifact_exists_column": bool(row.get("artifact_exists")) if row.get("exists_in_log") else None,
        "csv_status": status,
        "notebook_status": notebook,
        "proxy_public_risk": combined_row.get("proxy_public_risk"),
        "transfer_risk_tier": combined_row.get("transfer_risk_tier"),
        "joint_risk_tier": combined_row.get("joint_risk_tier"),
        "joint_warning_count": combined_row.get("joint_warning_count"),
        "mean_delta_vs_178": combined_row.get("mean_delta_vs_178"),
        "max_delta_vs_178": combined_row.get("max_delta_vs_178"),
        "max_delta_vs_341": combined_row.get("max_delta_vs_341"),
        "combined_tier": combined_row.get("combined_tier"),
        "block_reasons": combined_row.get("block_reasons") or [],
        "artifact_contract_ready": bool(
            row.get("exists_in_log")
            and status["exists"]
            and status["columns_ok"]
            and status["row_count_ok"]
            and notebook["exists"]
            and int(row.get("duplicate_file_count") or 0) == 1
            and not is_submitted
            and public_lb is None
        ),
    }


def sort_for_safety(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row.get("max_delta_vs_178") or 999.0),
        float(row.get("mean_delta_vs_178") or 999.0),
        float(row.get("proxy_public_risk") or 999.0),
        int(row["id"]),
    )


def build_report() -> dict[str, Any]:
    combined_status = report_status(COMBINED_GATE_PATH)
    id341_status = report_status(ID341_RISK_GATE_PATH)
    prelaunch_status = report_status(PRELAUNCH_GATE_PATH)
    combined_gate = load_json(COMBINED_GATE_PATH)
    log = load_log()

    focus_cards = [candidate_card(candidate_id, combined_gate, log) for candidate_id in FOCUS_IDS]
    safety_ranked = sorted(focus_cards, key=sort_for_safety)
    proxy_ranked = sorted(focus_cards, key=lambda row: (float(row.get("proxy_public_risk") or 999.0), row["id"]))

    safety_first = next(row for row in focus_cards if row["id"] == SAFETY_FIRST_ID)
    proxy_leader = next(row for row in focus_cards if row["id"] == PROXY_LEADER_ID)
    local_leader = next(row for row in focus_cards if row["id"] == LOCAL_LEADER_ID)
    public_058_plausible = bool(
        combined_status["decision"].get("public_058_plausible_candidate_exists")
        or id341_status["decision"].get("public_058_plausible")
    )
    safety_first_ready = safety_first["artifact_contract_ready"] and safety_first["joint_warning_count"] == 0
    prelaunch_ready = prelaunch_status["decision"].get("prelaunch_ready_for_explicit_approval") is True

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_manual_submission_probe_decision",
        "purpose": (
            "Freeze the current manual DACON probe choice without submitting. This separates a low-risk public "
            "sanity probe from a Public-0.58 evidence claim."
        ),
        "inputs": {
            "combined_transfer_joint_gate": combined_status,
            "id341_public_risk_gate": id341_status,
            "kaggle_prelaunch_gate": prelaunch_status,
            "submission_log": {
                "path": safe_relative(SUBMISSION_LOG_PATH),
                "exists": SUBMISSION_LOG_PATH.exists(),
                "rows": int(len(log)),
            },
        },
        "focus_candidate_ids": FOCUS_IDS,
        "focus_candidates": focus_cards,
        "safety_ranked_focus_candidates": safety_ranked,
        "proxy_ranked_focus_candidates": proxy_ranked,
        "anchor_public_truth": {
            "id178_public": 0.6014113009,
            "id287_public": 0.6698787318,
            "id329_public": 0.6017908577,
        },
        "decision": {
            "manual_submission_probe_decision_ready": True,
            "public_058_plausible_candidate_exists": public_058_plausible,
            "default_action": "do_not_submit",
            "submit_now": False,
            "create_submission_artifact": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "lowest_risk_probe_id": safety_first["id"],
            "lowest_risk_probe_file": safety_first["submission_file"],
            "lowest_risk_probe_artifact_contract_ready": safety_first_ready,
            "lowest_proxy_focus_id": proxy_leader["id"],
            "lowest_proxy_focus_file": proxy_leader["submission_file"],
            "lowest_proxy_focus_not_recommended_reason": (
                "ID340 has the lowest transfer proxy among the 337-341 focus probes, but ID341 is the safer manual "
                "sanity probe because it has much smaller movement from the confirmed ID178 public anchor."
            ),
            "lowest_local_cv_focus_id": local_leader["id"],
            "lowest_local_cv_focus_file": local_leader["submission_file"],
            "lowest_local_cv_focus_not_recommended_reason": (
                "ID339 has the lowest local CV among the 337-341 focus probes, but it moves much more from ID178 "
                "and has a worse transfer proxy; this is exactly the failure mode seen in ID287/ID329."
            ),
            "requires_explicit_user_approval_before_dacon_submit": True,
            "recommended_primary_next_action": (
                "Use the explicit-approved no-submit Kaggle/Linux upstream rebuild. The previous ID341/340/337 "
                "sanity probes have now shown only the 0.60 public band, so do not spend another DACON submission "
                "on the same micro-probe family."
            ),
            "kaggle_prelaunch_ready_for_explicit_approval": prelaunch_ready,
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{value:.{digits}f}"


def table_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        reasons = ",".join(row.get("block_reasons") or [])
        lines.append(
            f"| {row['id']} | {fmt(row.get('local_cv'))} | {fmt(row.get('proxy_public_risk'))} | "
            f"{fmt(row.get('mean_delta_vs_178'))} | {fmt(row.get('max_delta_vs_178'))} | "
            f"{fmt(row.get('max_delta_vs_341'))} | `{row.get('joint_risk_tier')}` | "
            f"{row.get('joint_warning_count')} | `{row.get('combined_tier')}` | "
            f"`{row.get('artifact_contract_ready')}` | `{reasons}` | `{row.get('submission_file')}` |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 manual submission probe decision",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It does not submit to DACON, push/run Kaggle, create a new CSV, or edit the submission log.",
        "",
        "## Decision",
        "",
        f"- Manual probe decision ready: `{decision['manual_submission_probe_decision_ready']}`",
        f"- Public 0.58 plausible candidate exists: `{decision['public_058_plausible_candidate_exists']}`",
        f"- Default action: `{decision['default_action']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Lowest-risk probe ID: `{decision['lowest_risk_probe_id']}`",
        f"- Lowest-risk probe file: `{decision['lowest_risk_probe_file']}`",
        f"- Lowest-risk artifact contract ready: `{decision['lowest_risk_probe_artifact_contract_ready']}`",
        f"- Lowest proxy focus ID: `{decision['lowest_proxy_focus_id']}`",
        f"- Why not lowest-proxy by default: {decision['lowest_proxy_focus_not_recommended_reason']}",
        f"- Lowest-local-CV focus ID: `{decision['lowest_local_cv_focus_id']}`",
        f"- Why not lowest-local-CV by default: {decision['lowest_local_cv_focus_not_recommended_reason']}",
        f"- Requires explicit DACON approval: `{decision['requires_explicit_user_approval_before_dacon_submit']}`",
        f"- Recommended primary next action: {decision['recommended_primary_next_action']}",
        "",
        "## Focus candidates",
        "",
        "| id | local | proxy | Δ178 mean | Δ178 max | Δ341 max | joint tier | warn | combined tier | artifact ready | block reasons | file |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|",
        *table_rows(report["focus_candidates"]),
        "",
        "## Safety-ranked focus candidates",
        "",
        "| id | local | proxy | Δ178 mean | Δ178 max | Δ341 max | joint tier | warn | combined tier | artifact ready | block reasons | file |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|",
        *table_rows(report["safety_ranked_focus_candidates"]),
        "",
        "## Proxy-ranked focus candidates",
        "",
        "| id | local | proxy | Δ178 mean | Δ178 max | Δ341 max | joint tier | warn | combined tier | artifact ready | block reasons | file |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|",
        *table_rows(report["proxy_ranked_focus_candidates"]),
        "",
        "## Anchor public truth",
        "",
        "```json",
        json.dumps(report["anchor_public_truth"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_manual_submission_probe_decision_20260618.json",
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
