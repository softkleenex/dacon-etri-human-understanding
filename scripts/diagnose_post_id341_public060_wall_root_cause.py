#!/usr/bin/env python3
"""Diagnose the post-ID341 Public-0.60 wall.

This report-only diagnostic turns the recent leaderboard frustration into a
reproducible stop-line:

- ID178/337/340 form a tight public band near 0.60 despite different local
  micro-overlays.
- ID329 and especially ID287 prove that lower local CV can be misleading.
- Q3/S3 remain uncovered by local no-approval lanes.

It creates no predictions, feature matrices, submission CSVs, base-prediction
artifacts, notebook/history artifacts, submission-log rows, Kaggle pushes/runs,
or DACON submissions.
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_public060_wall_root_cause_20260621.json"
KST = timezone(timedelta(hours=9))

TARGET_PUBLIC = 0.58
PUBLIC_WALL = 0.60
RECENT_PUBLIC_IDS = [178, 329, 337, 340]
COLLAPSE_ID = 287
Q3S3_INVENTORY = REPORTS_DIR / "post_id341_q3s3_remaining_route_inventory_20260618.json"
Q2_MODEL_SMOKE = REPORTS_DIR / "post_id341_q2q3s3_fold_internal_model_smoke_20260618.json"
Q2_GEOMETRY_GATE = REPORTS_DIR / "post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json"
PUBLIC_LOCAL_GAP = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"
TRANSFER_AUDIT = REPORTS_DIR / "post_id341_public_transfer_audit_20260618.json"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path).get("decision", {})


def round_or_none(value: Any, digits: int = 9) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def load_log() -> pd.DataFrame:
    log = pd.read_csv(SUBMISSION_LOG)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    for col in ["local_cv_score_num", "public_lb_score_num", "cv_lb_gap"]:
        log[col] = pd.to_numeric(log[col], errors="coerce")
    return log


def real_public_rows(log: pd.DataFrame) -> pd.DataFrame:
    rows = log[
        log["public_lb_score_num"].notna()
        & log["local_cv_score_num"].notna()
        & log["public_lb_score_num"].between(0.3, 1.0, inclusive="neither")
        & log["status_clean"].isin(["scored", "cv_lb_gap_risk"])
    ].copy()
    rows["public_minus_local"] = rows["public_lb_score_num"] - rows["local_cv_score_num"]
    return rows.sort_values("public_lb_score_num")


def row_card(row: pd.Series) -> dict[str, Any]:
    return {
        "id": int(row["id_num"]),
        "submission_file": row.get("submission_file"),
        "model_description": row.get("model_description"),
        "local_cv": round_or_none(row["local_cv_score_num"], 6),
        "public_lb": round_or_none(row["public_lb_score_num"], 12),
        "public_minus_local": round_or_none(row["public_minus_local"], 12),
        "status_clean": row.get("status_clean"),
    }


def cards_for_ids(rows: pd.DataFrame, ids: list[int]) -> list[dict[str, Any]]:
    cards = []
    for candidate_id in ids:
        match = rows[rows["id_num"].eq(candidate_id)]
        if match.empty:
            cards.append({"id": candidate_id, "missing": True})
        else:
            cards.append(row_card(match.iloc[-1]))
    return cards


def recent_band(rows: pd.DataFrame) -> dict[str, Any]:
    recent = rows[rows["id_num"].isin(RECENT_PUBLIC_IDS)].copy()
    if recent.empty:
        return {"ids": RECENT_PUBLIC_IDS, "available_ids": [], "ready": False}
    local_span = float(recent["local_cv_score_num"].max() - recent["local_cv_score_num"].min())
    public_span = float(recent["public_lb_score_num"].max() - recent["public_lb_score_num"].min())
    best = recent.sort_values("public_lb_score_num").iloc[0]
    return {
        "ids": RECENT_PUBLIC_IDS,
        "available_ids": sorted(int(x) for x in recent["id_num"].tolist()),
        "ready": set(RECENT_PUBLIC_IDS).issubset({int(x) for x in recent["id_num"].tolist()}),
        "best_recent_public_id": int(best["id_num"]),
        "best_recent_public": round_or_none(best["public_lb_score_num"], 12),
        "local_span": round(local_span, 12),
        "public_span": round(public_span, 12),
        "local_span_to_public_span_ratio": None if public_span == 0 else round(local_span / public_span, 6),
        "all_recent_public_near_060": bool(
            recent["public_lb_score_num"].between(PUBLIC_WALL, 0.602, inclusive="both").all()
        ),
        "rows": cards_for_ids(recent, RECENT_PUBLIC_IDS),
    }


def collapse_evidence(rows: pd.DataFrame) -> dict[str, Any]:
    collapse = rows[rows["id_num"].eq(COLLAPSE_ID)]
    if collapse.empty:
        return {"id": COLLAPSE_ID, "exists": False}
    row = collapse.iloc[-1]
    return {
        "id": COLLAPSE_ID,
        "exists": True,
        "local_cv": round_or_none(row["local_cv_score_num"], 6),
        "public_lb": round_or_none(row["public_lb_score_num"], 12),
        "public_minus_local": round_or_none(row["public_minus_local"], 12),
        "catastrophic_local_public_mismatch": bool(row["public_minus_local"] >= 0.20),
    }


def build_report() -> dict[str, Any]:
    log = load_log()
    public = real_public_rows(log)
    band = recent_band(public)
    collapse = collapse_evidence(public)
    q3s3_decision = decision(Q3S3_INVENTORY)
    q2_smoke_decision = decision(Q2_MODEL_SMOKE)
    q2_gate_decision = decision(Q2_GEOMETRY_GATE)
    public_gap_decision = decision(PUBLIC_LOCAL_GAP)
    transfer_decision = decision(TRANSFER_AUDIT)

    q3s3_uncovered = q3s3_decision.get("q3_s3_still_uncovered") is True
    q2_only_signal = q2_smoke_decision.get("selected_targets") == ["Q2"]
    q2_not_public058 = (
        q2_gate_decision.get("public_058_plausible") is False
        and float(q2_gate_decision.get("best_proxy", 9.0)) > PUBLIC_WALL
    )
    local_cv_not_sufficient = (
        public_gap_decision.get("any_watch_candidate_passes_public_058_recent_floor_proxy") is False
        and public_gap_decision.get("any_watch_candidate_passes_public_058_recent_stress_proxy") is False
        and transfer_decision.get("any_candidate_passes_public_058_gate") is False
    )

    cause_rows = [
        {
            "cause": "validation_surface_mismatch",
            "evidence": (
                "Recent local CV changes around ID178/337/340 span far more locally than publicly, "
                "while ID329/287 show lower local CV can fail to transfer."
            ),
            "active": bool(band.get("ready") and collapse.get("catastrophic_local_public_mismatch")),
        },
        {
            "cause": "q3_s3_uncovered",
            "evidence": "Remaining-route inventory keeps Q3/S3 uncovered and local no-approval score routes closed.",
            "active": q3s3_uncovered,
        },
        {
            "cause": "q2_only_lead_insufficient",
            "evidence": "The env/social/routine local smoke selected Q2 only; Q3/S3 stay empty and Q2 proxy remains above 0.60.",
            "active": bool(q2_only_signal and q2_not_public058),
        },
        {
            "cause": "leaderboard_probe_family_consumed",
            "evidence": "ID341/340/337 micro-probes have already been spent and stayed near the 0.60 public band.",
            "active": True,
        },
    ]
    all_causes_active = all(row["active"] for row in cause_rows)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_public060_wall_root_cause",
        "report_only": True,
        "target_public": TARGET_PUBLIC,
        "public_wall": PUBLIC_WALL,
        "inputs": {
            "submission_log": rel(SUBMISSION_LOG),
            "q3s3_inventory": rel(Q3S3_INVENTORY),
            "q2_model_smoke": rel(Q2_MODEL_SMOKE),
            "q2_geometry_gate": rel(Q2_GEOMETRY_GATE),
            "public_local_gap": rel(PUBLIC_LOCAL_GAP),
            "transfer_audit": rel(TRANSFER_AUDIT),
        },
        "public_rows_count": int(len(public)),
        "best_public_rows": cards_for_ids(public.sort_values("public_lb_score_num").head(5), [337, 340, 178, 329, 287]),
        "recent_micro_probe_band": band,
        "collapse_evidence": collapse,
        "target_gap_evidence": {
            "q3_s3_still_uncovered": q3s3_uncovered,
            "q2_only_signal_found": q2_only_signal,
            "q2_best_proxy": q2_gate_decision.get("best_proxy"),
            "q2_not_public_058": q2_not_public058,
            "local_no_approval_score_route_exists": q3s3_decision.get("local_no_approval_score_route_exists"),
            "only_score_relevant_open_route": q3s3_decision.get("only_score_relevant_open_route"),
        },
        "cause_rows": cause_rows,
        "decision": {
            "public060_wall_root_cause_ready": True,
            "wall_explained_by_current_evidence": all_causes_active,
            "local_cv_only_candidate_path_closed": local_cv_not_sufficient,
            "same_family_micro_probe_path_closed": True,
            "q3s3_independent_signal_required": q3s3_uncovered,
            "primary_score_relevant_route": "explicit_approved_clean_upstream_id221_to_id226",
            "requires_exact_kaggle_approval": True,
            "approval_phrase": "Kaggle upstream rebuild no-submit 실행 승인",
            "create_feature_matrix": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not spend more DACON submissions or local micro-overlays. The next useful score work is either "
                "the exact-approved no-submit upstream rebuild, or a materially new Q3/S3 independent source that "
                "passes this wall diagnosis and the transfer gates before artifact review."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    band = report["recent_micro_probe_band"]
    lines = [
        "# Post-ID341 Public-0.60 wall root-cause diagnosis",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrix, submission artifact, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Wall explained by current evidence: `{decision['wall_explained_by_current_evidence']}`",
        f"- Local-CV-only path closed: `{decision['local_cv_only_candidate_path_closed']}`",
        f"- Same-family micro-probe path closed: `{decision['same_family_micro_probe_path_closed']}`",
        f"- Q3/S3 independent signal required: `{decision['q3s3_independent_signal_required']}`",
        f"- Primary score route: `{decision['primary_score_relevant_route']}`",
        f"- Requires exact Kaggle approval: `{decision['requires_exact_kaggle_approval']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Recent public band",
        "",
        f"- Best recent public ID: `{band.get('best_recent_public_id')}`",
        f"- Best recent public score: `{band.get('best_recent_public')}`",
        f"- Local span across ID178/329/337/340: `{band.get('local_span')}`",
        f"- Public span across ID178/329/337/340: `{band.get('public_span')}`",
        f"- All recent micro probes near 0.60: `{band.get('all_recent_public_near_060')}`",
        "",
        "| id | local CV | public | public-local gap | file |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in band.get("rows", []):
        lines.append(
            f"| {row['id']} | {row.get('local_cv')} | {row.get('public_lb')} | "
            f"{row.get('public_minus_local')} | `{row.get('submission_file')}` |"
        )
    lines.extend(["", "## Causes", "", "| cause | active | evidence |", "|---|:---:|---|"])
    for row in report["cause_rows"]:
        lines.append(f"| `{row['cause']}` | `{row['active']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Target-gap evidence",
            "",
            "```json",
            json.dumps(report["target_gap_evidence"], indent=2, ensure_ascii=False),
            "```",
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
