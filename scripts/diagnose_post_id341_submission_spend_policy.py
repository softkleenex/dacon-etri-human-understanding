#!/usr/bin/env python3
"""Report-only DACON submission spend policy after ID341.

This diagnostic answers the operator question "what should we submit now?" while
preserving the post-ID341 guardrails:

- no DACON submission,
- no Kaggle push/run,
- no new submission CSV/notebook/base-prediction artifact,
- no submission_log.csv edit.

It deliberately separates a score-improvement submission from a public sanity
probe. Under the current evidence, the former is closed; the latter is only the
existing ID341 CSV, and only with explicit user approval for that exact file.
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
SUBMISSION_LOG_PATH = SUBMISSIONS_DIR / "submission_log.csv"
KST = timezone(timedelta(hours=9))

MANUAL_PROBE_PATH = REPORTS_DIR / "post_id341_manual_submission_probe_decision_20260619.json"
SAFE_WAIT_PATH = REPORTS_DIR / "post_id341_safe_wait_state_contract_20260619.json"
PUBLIC_058_PROMOTION_PATH = REPORTS_DIR / "post_id341_public058_artifact_promotion_contract_20260619.json"
PUBLIC_LOCAL_GAP_PATH = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"
NEXT_ROUTE_PATH = REPORTS_DIR / "post_id341_next_material_route_triage_20260619.json"

ID178_PUBLIC = 0.6014113009
ID329_PUBLIC = 0.6017908577
ID287_PUBLIC = 0.6698787318
ID337_PUBLIC = 0.6006206286219425
ID340_PUBLIC = 0.601379175253277
SANITY_PROBE_ID = 341
RECORDED_ID342 = 342
RECORDED_ID342_FILE = "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv"
TARGET_PUBLIC = 0.58


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def input_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": relpath(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": relpath(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": relpath(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
    }


def bool_decision(status: dict[str, Any], key: str) -> bool:
    return status.get("decision", {}).get(key) is True


def load_submission_log() -> pd.DataFrame:
    df = pd.read_csv(SUBMISSION_LOG_PATH)
    df["id_num"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df


def row_for_id(df: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = df[df["id_num"] == candidate_id]
    if rows.empty:
        return {"id": candidate_id, "exists": False}
    row = rows.iloc[-1].to_dict()
    row["exists"] = True
    return row


def artifact_status(row: dict[str, Any]) -> dict[str, Any]:
    submission_file = row.get("submission_file")
    notebook_file = row.get("notebook_file")
    csv_path = SUBMISSIONS_DIR / submission_file if isinstance(submission_file, str) else None
    notebook_path = ROOT / notebook_file if isinstance(notebook_file, str) else None

    csv_ok = False
    csv_shape: list[int] | None = None
    csv_columns: list[str] | None = None
    csv_na_total: int | None = None
    if csv_path and csv_path.exists():
        csv = pd.read_csv(csv_path)
        csv_shape = [int(csv.shape[0]), int(csv.shape[1])]
        csv_columns = list(csv.columns)
        csv_na_total = int(csv.isna().sum().sum())
        csv_ok = csv_shape == [250, 10] and csv_na_total == 0

    duplicate_file_count = int(row.get("duplicate_file_count") or 0) if row.get("exists") else None
    is_submitted = bool(row.get("is_submitted_bool")) if row.get("exists") else None
    public_lb = row.get("public_lb_score_num")
    if pd.isna(public_lb):
        public_lb = None

    return {
        "submission_file": submission_file,
        "notebook_file": notebook_file,
        "csv_path": relpath(csv_path) if csv_path else None,
        "csv_exists": bool(csv_path and csv_path.exists()),
        "csv_shape": csv_shape,
        "csv_columns": csv_columns,
        "csv_na_total": csv_na_total,
        "notebook_path": relpath(notebook_path) if notebook_path else None,
        "notebook_exists": bool(notebook_path and notebook_path.exists()),
        "duplicate_file_count": duplicate_file_count,
        "is_submitted_bool": is_submitted,
        "public_lb": public_lb,
        "artifact_ready": bool(
            row.get("exists")
            and csv_ok
            and notebook_path
            and notebook_path.exists()
            and duplicate_file_count == 1
            and is_submitted is False
            and public_lb is None
        ),
    }


def current_log_invariants(df: pd.DataFrame) -> dict[str, Any]:
    ids = df["id_num"].dropna()
    recorded_id342_rows = df[df["id_num"] == RECORDED_ID342]
    recorded_id342_row_ok = False
    if len(recorded_id342_rows) == 1:
        row = recorded_id342_rows.iloc[-1]
        recorded_id342_row_ok = bool(
            row.get("submission_file") == RECORDED_ID342_FILE
            and str(row.get("status_clean", "")).lower() == "generated_unsubmitted"
            and str(row.get("is_submitted_bool", "")).lower() == "false"
            and pd.isna(row.get("public_lb_score_num"))
        )
    max_id = int(ids.max()) if not ids.empty else None
    return {
        "submission_log_path": relpath(SUBMISSION_LOG_PATH),
        "rows": int(len(df)),
        "max_id": max_id,
        "ids_are_unique": bool(ids.is_unique),
        "rows_id_ge_342": int((df["id_num"] >= 342).sum()),
        "rows_id_341": int((df["id_num"] == 341).sum()),
        "recorded_id342_row_count": int(len(recorded_id342_rows)),
        "recorded_id342_row_ok": recorded_id342_row_ok,
        "post_manual_submission_ids": [int(value) for value in sorted(ids[ids > SANITY_PROBE_ID])],
        "ready_for_existing_id342_state": bool(ids.is_unique and max_id == RECORDED_ID342 and recorded_id342_row_ok),
        "next_numeric_id_if_later_authorized": None if max_id is None else max_id + 1,
    }


def build_report() -> dict[str, Any]:
    manual = input_status(MANUAL_PROBE_PATH)
    safe_wait = input_status(SAFE_WAIT_PATH)
    promotion = input_status(PUBLIC_058_PROMOTION_PATH)
    public_gap = input_status(PUBLIC_LOCAL_GAP_PATH)
    next_route = input_status(NEXT_ROUTE_PATH)
    df = load_submission_log()
    id341 = row_for_id(df, SANITY_PROBE_ID)
    id341_artifact = artifact_status(id341)

    manual_decision = manual["decision"]
    safe_wait_decision = safe_wait["decision"]
    promotion_decision = promotion["decision"]
    public_gap_decision = public_gap["decision"]
    next_route_decision = next_route["decision"]

    no_public_058_candidate = (
        manual_decision.get("public_058_plausible_candidate_exists") is False
        and promotion_decision.get("public058_artifact_contract_review_open") is False
        and public_gap_decision.get("any_watch_candidate_passes_public_058_recent_floor_proxy") is False
        and public_gap_decision.get("any_watch_candidate_passes_public_058_recent_stress_proxy") is False
    )
    no_submit_consensus = (
        manual_decision.get("submit_now") is False
        and safe_wait_decision.get("submit_now") is False
        and promotion_decision.get("submit_now") is False
        and public_gap_decision.get("submit_now") is False
        and next_route_decision.get("submit_now") is False
    )
    id341_is_manual_probe = manual_decision.get("lowest_risk_probe_id") == SANITY_PROBE_ID
    kaggle_route_requires_approval = (
        next_route_decision.get("requires_explicit_kaggle_approval") is True
        and next_route_decision.get("approval_phrase") == "Kaggle upstream rebuild no-submit 실행 승인"
    )
    sanity_probe_file = id341.get("submission_file") if id341_artifact["artifact_ready"] else None
    sanity_probe_id = SANITY_PROBE_ID if id341_artifact["artifact_ready"] else None
    log_invariants = current_log_invariants(df)

    reasons = [
        "ID337 is now the best confirmed recent public micro-probe, but only by a tiny margin over ID178 and still near 0.60.",
        "No current local candidate passes the Public 0.58 recent-gap floor or stress proxy.",
        "The Public-0.58 artifact-promotion contract is closed.",
        "The only score-relevant open route is the explicitly approved no-submit Kaggle upstream rebuild.",
    ]
    if id341_artifact["artifact_ready"]:
        reasons.append("ID341 is artifact-ready only as a low-risk public sanity probe, not as Public 0.58 evidence.")
    else:
        reasons.append("ID341 is no longer a fresh sanity-probe CSV because it is already marked submitted locally.")

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_submission_spend_policy",
        "purpose": (
            "Give a deterministic answer to whether a DACON submission should be spent now, without creating or "
            "submitting any artifact."
        ),
        "inputs": {
            "manual_probe": manual,
            "safe_wait": safe_wait,
            "public058_promotion": promotion,
            "public_local_gap": public_gap,
            "next_route": next_route,
        },
        "public_truth_anchors": {
            "id178_public": ID178_PUBLIC,
            "id329_public": ID329_PUBLIC,
            "id287_public": ID287_PUBLIC,
            "id337_public": ID337_PUBLIC,
            "id340_public": ID340_PUBLIC,
            "target_public": TARGET_PUBLIC,
        },
        "submission_log_invariants": log_invariants,
        "sanity_probe_candidate": {
            "id": SANITY_PROBE_ID,
            "submission_file": id341.get("submission_file"),
            "local_cv": None if pd.isna(id341.get("local_cv_score_num")) else float(id341.get("local_cv_score_num")),
            "model_description": id341.get("model_description"),
            "artifact": id341_artifact,
        },
        "evidence_flags": {
            "all_required_inputs_parse": all(
                status.get("parse_ok") for status in [manual, safe_wait, promotion, public_gap, next_route]
            ),
            "no_public_058_candidate": no_public_058_candidate,
            "no_submit_consensus": no_submit_consensus,
            "id341_is_manual_probe": id341_is_manual_probe,
            "id341_artifact_ready": id341_artifact["artifact_ready"],
            "submission_log_ready_for_existing_id342_state": log_invariants["ready_for_existing_id342_state"],
            "kaggle_route_requires_exact_approval": kaggle_route_requires_approval,
        },
        "decision": {
            "submission_spend_policy_ready": True,
            "score_improvement_submission_recommended": False,
            "public_sanity_probe_recommended_by_default": False,
            "public_sanity_probe_if_user_insists_id": sanity_probe_id,
            "public_sanity_probe_if_user_insists_file": sanity_probe_file,
            "requires_exact_dacon_csv_approval": True,
            "allowed_dacon_submit_now": False,
            "allowed_kaggle_network_now": False,
            "allowed_new_submission_artifact_now": False,
            "public_058_plausible_candidate_exists": False,
            "recommended_primary_next_action": "Wait for exact no-submit Kaggle upstream rebuild approval, not DACON submission.",
            "operator_answer": (
                "Do not spend a DACON submission for score improvement now. ID341/337/340 already show that the "
                "current micro-probe family remains near 0.60, not Public 0.58."
            ),
            "reasons": reasons,
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
    flags = report["evidence_flags"]
    probe = report["sanity_probe_candidate"]
    artifact = probe["artifact"]
    lines = [
        "# Post-ID341 submission spend policy",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It does not submit to DACON, run Kaggle, create a new CSV, or edit `submission_log.csv`.",
        "",
        "## Decision",
        "",
        f"- Score-improvement submission recommended: `{decision['score_improvement_submission_recommended']}`",
        f"- Public sanity probe recommended by default: `{decision['public_sanity_probe_recommended_by_default']}`",
        f"- If the user insists on one sanity probe: ID `{decision['public_sanity_probe_if_user_insists_id']}`",
        f"- Sanity probe file: `{decision['public_sanity_probe_if_user_insists_file']}`",
        f"- Requires exact DACON CSV approval: `{decision['requires_exact_dacon_csv_approval']}`",
        f"- Allowed DACON submit now: `{decision['allowed_dacon_submit_now']}`",
        f"- Allowed Kaggle network now: `{decision['allowed_kaggle_network_now']}`",
        f"- Allowed new submission artifact now: `{decision['allowed_new_submission_artifact_now']}`",
        f"- Public 0.58 plausible candidate exists: `{decision['public_058_plausible_candidate_exists']}`",
        f"- Primary next action: {decision['recommended_primary_next_action']}",
        "",
        "## Evidence flags",
        "",
        "| flag | value |",
        "|---|---:|",
    ]
    for key, value in flags.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Sanity probe candidate",
            "",
            f"- ID: `{probe['id']}`",
            f"- File: `{probe['submission_file']}`",
            f"- Local CV: `{probe['local_cv']}`",
            f"- CSV ready: `{artifact['artifact_ready']}`",
            f"- CSV path: `{artifact['csv_path']}`",
            f"- CSV shape: `{artifact['csv_shape']}`",
            f"- Notebook exists: `{artifact['notebook_exists']}`",
            "",
            "## Reasons",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend(
        [
            "",
            "## Public truth anchors",
            "",
            "```json",
            json.dumps(report["public_truth_anchors"], indent=2, ensure_ascii=False),
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
        default=REPORTS_DIR / "post_id341_submission_spend_policy_20260619.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {relpath(output)}")
    print(f"Wrote {relpath(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
