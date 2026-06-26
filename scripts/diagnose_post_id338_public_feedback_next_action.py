#!/usr/bin/env python3
"""Post-ID338 public-feedback next-action diagnostic.

Report-only diagnostic. It uses ID338 as a fixed neighborhood/screening anchor
while recording the authenticated correction that ID342 is the current best
public anchor. It creates no submission artifact and makes no network calls.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))

PUBLIC_ID178 = 0.6014113009
PUBLIC_ID337 = 0.6006206286219425
PUBLIC_ID338 = 0.5992145177
PUBLIC_ID342 = 0.5960832519
PUBLIC_ID340 = 0.601379175253277
PUBLIC_ID287 = 0.6698787318


def extract_id(path: Path) -> int | None:
    match = re.search(r"base_test_id_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def find_base_test_files() -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(BASE_DIR.glob("base_test_id_*.csv")):
        candidate_id = extract_id(path)
        if candidate_id is not None and candidate_id not in files:
            files[candidate_id] = path
    return files


def load_log() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    for column in ["local_cv_score_num", "public_lb_score_num", "local_cv_score", "public_lb_score"]:
        if column in log.columns:
            log[column] = pd.to_numeric(log[column], errors="coerce")
    return log


def log_row(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"].eq(candidate_id)]
    if rows.empty:
        return {}
    row = rows.iloc[-1].to_dict()
    return {
        "submission_file": row.get("submission_file", ""),
        "status_clean": row.get("status_clean", ""),
        "risk_reason": row.get("risk_reason", ""),
        "local_cv": safe_float(row.get("local_cv_score_num", row.get("local_cv_score"))),
        "public_lb": safe_float(row.get("public_lb_score_num", row.get("public_lb_score"))),
        "submitted": bool(row.get("submitted")),
    }


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_pred(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return frame[TARGETS].astype(float).to_numpy()


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def family_for(candidate_id: int, text: str) -> str:
    text = text.lower()
    if candidate_id in {173, 177, 178, 179, 334, 335, 336, 337, 338}:
        return "sequence_temporal_anchor"
    if "randomconv" in text or "random_conv" in text or "sequence" in text or "seqsum" in text:
        return "sequence_temporal_anchor"
    if "featuresim" in text or "feature_similarity" in text or "feature-similarity" in text:
        return "independent_feature_similarity"
    if "q3" in text or "s3" in text or "q123" in text:
        return "q3_s3_related_probe"
    if "post328_combo" in text or "failed_direction" in text or "orthogonal" in text or "antivector" in text:
        return "failed_direction_combo"
    if "calibration" in text or "q240" in text or "isotonic" in text:
        return "calibration_frontier"
    if "strict" in text or "compact" in text or "feature" in text:
        return "strict_feature"
    return "other"


def target_mean_abs(arr: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    diff = np.abs(arr - anchor)
    return {target: float(diff[:, idx].mean()) for idx, target in enumerate(TARGETS)}


def dominant_targets(target_deltas: dict[str, float], limit: int = 3) -> list[str]:
    return [
        target
        for target, value in sorted(target_deltas.items(), key=lambda item: item[1], reverse=True)[:limit]
        if value > 0
    ]


def risk_tier(row: dict[str, Any]) -> str:
    if row.get("public_lb") is not None:
        return "known_public_reference"
    if row["max_delta_vs_338"] <= 0.08 and row["mean_delta_vs_338"] <= 0.015 and row["projection_failed"] <= 0.02:
        return "low_id338_distance"
    if row["max_delta_vs_338"] <= 0.14 and row["mean_delta_vs_338"] <= 0.030 and row["projection_failed"] <= 0.04:
        return "reviewable_id338_neighbor"
    if row["projection_failed"] > 0.08 or row["max_delta_vs_338"] > 0.22:
        return "reject_high_public_risk"
    return "diagnostic_only"


def proxy(row: dict[str, Any]) -> float:
    """Screening score only; not a public-score prediction."""
    local = row.get("local_cv")
    local_pressure = 0.0 if local is None else max(0.0, 0.538 - float(local))
    return float(
        PUBLIC_ID338
        + 0.32 * row["mean_delta_vs_338"]
        + 0.030 * row["max_delta_vs_338"]
        + 0.12 * max(0.0, row["projection_failed"])
        + 0.20 * max(0.0, local_pressure - 0.006)
    )


def row_for_candidate(
    candidate_id: int,
    path: Path,
    log: pd.DataFrame,
    arr178: np.ndarray,
    arr287: np.ndarray,
    arr338: np.ndarray,
) -> dict[str, Any]:
    info = log_row(log, candidate_id)
    arr = load_pred(path)
    failed_direction = (arr287 - arr178).reshape(-1)
    vec178 = (arr - arr178).reshape(-1)
    target_deltas = target_mean_abs(arr, arr338)
    row = {
        "id": candidate_id,
        "submission_file": info.get("submission_file", ""),
        "base_test_file": str(path.relative_to(ROOT)),
        "family": family_for(candidate_id, f"{info.get('submission_file', '')} {path.name}"),
        "local_cv": info.get("local_cv"),
        "public_lb": info.get("public_lb"),
        "status_clean": info.get("status_clean", ""),
        "risk_reason": info.get("risk_reason", ""),
        "submitted": bool(info.get("submitted")),
        "mean_delta_vs_338": float(np.mean(np.abs(arr - arr338))),
        "max_delta_vs_338": float(np.max(np.abs(arr - arr338))),
        "rmse_vs_338": float(np.sqrt(np.mean((arr - arr338) ** 2))),
        "mean_delta_vs_178": float(np.mean(np.abs(arr - arr178))),
        "max_delta_vs_178": float(np.max(np.abs(arr - arr178))),
        "projection_failed": projection(vec178, failed_direction),
        "target_mean_delta_vs_338": target_deltas,
        "dominant_targets_vs_338": dominant_targets(target_deltas),
    }
    row["id338_screening_proxy"] = proxy(row)
    row["risk_tier"] = risk_tier(row)
    return row


def short_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "submission_file",
        "family",
        "local_cv",
        "public_lb",
        "status_clean",
        "risk_tier",
        "id338_screening_proxy",
        "mean_delta_vs_338",
        "max_delta_vs_338",
        "mean_delta_vs_178",
        "projection_failed",
        "dominant_targets_vs_338",
        "target_mean_delta_vs_338",
    ]
    return {key: row.get(key) for key in keys}


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return "" if value is None else f"{value:.{digits}f}"


def table_rows(rows: list[dict[str, Any]], limit: int) -> list[str]:
    lines = []
    for row in rows[:limit]:
        targets = ",".join(row.get("dominant_targets_vs_338") or [])
        lines.append(
            f"| {row['id']} | {fmt(row.get('local_cv'))} | {fmt(row.get('public_lb'), 10)} | "
            f"`{row.get('risk_tier')}` | `{row.get('family')}` | "
            f"{fmt(row.get('id338_screening_proxy'))} | {fmt(row.get('mean_delta_vs_338'))} | "
            f"{fmt(row.get('max_delta_vs_338'))} | {fmt(row.get('projection_failed'), 4)} | "
            f"`{targets}` | `{row.get('submission_file')}` |"
        )
    return lines


def build_report() -> dict[str, Any]:
    log = load_log()
    files = find_base_test_files()
    required = [178, 287, 338]
    missing = [candidate_id for candidate_id in required if candidate_id not in files]
    if missing:
        raise FileNotFoundError(f"Missing required base_test ids: {missing}")

    arr178 = load_pred(files[178])
    arr287 = load_pred(files[287])
    arr338 = load_pred(files[338])
    rows = [
        row_for_candidate(candidate_id, path, log, arr178, arr287, arr338)
        for candidate_id, path in sorted(files.items())
    ]
    unscored = [row for row in rows if row.get("public_lb") is None]
    recent = [row for row in rows if row["id"] >= 334]
    pending_same_batch = [row for row in rows if row["id"] in {339, 342}]
    id338_neighbors = sorted(
        [
            row
            for row in unscored
            if row["risk_tier"] in {"low_id338_distance", "reviewable_id338_neighbor"}
            and row["id"] not in {339, 342}
        ],
        key=lambda row: (
            row["id338_screening_proxy"],
            row["mean_delta_vs_338"],
            row.get("local_cv") if row.get("local_cv") is not None else 999.0,
        ),
    )
    q3_s3_neighbors = [
        row
        for row in id338_neighbors
        if {"Q3", "S3"}.intersection(set(row.get("dominant_targets_vs_338") or []))
    ]
    tier_counts = dict(Counter(row["risk_tier"] for row in rows))
    family_counts = dict(Counter(row["family"] for row in rows))

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id338_public_feedback_next_action",
        "purpose": (
            "Use ID338 as a fixed neighborhood screening anchor while recording that authenticated submission "
            "history corrected the current best public anchor to ID342."
        ),
        "leaderboard_source": "https://dacon.io/en/competitions/official/236690/leaderboard",
        "confirmed_public_anchors": {
            "178": PUBLIC_ID178,
            "337": PUBLIC_ID337,
            "338": PUBLIC_ID338,
            "342": PUBLIC_ID342,
            "340": PUBLIC_ID340,
            "287_failed": PUBLIC_ID287,
        },
        "authenticated_correction": {
            "current_best_id": 342,
            "current_best_public": PUBLIC_ID342,
            "id342_dacon_submission_id": 1467937,
            "id342_c_time": "2026-06-23 23:30:37",
            "id339_public": 0.5985187787,
            "id339_dacon_submission_id": 1467938,
            "id339_c_time": "2026-06-23 23:30:48",
            "id338_public": PUBLIC_ID338,
            "id338_dacon_submission_id": 1467936,
            "id338_c_time": "2026-06-23 23:30:25",
        },
        "public_feedback": {
            "best_confirmed_id": 342,
            "best_confirmed_public": PUBLIC_ID342,
            "delta_vs_id337": PUBLIC_ID342 - PUBLIC_ID337,
            "delta_vs_id178": PUBLIC_ID342 - PUBLIC_ID178,
            "interpretation": (
                "Authenticated submission history shows ID342 feature-similarity shrink is the current public "
                "anchor. ID338 remains useful only as a neighborhood screening anchor; ID339 and ID338 are scored "
                "comparison probes."
            ),
        },
        "input_counts": {
            "base_test_rows": len(rows),
            "unscored_rows": len(unscored),
            "recent_rows_ge_334": len(recent),
        },
        "tier_counts": tier_counts,
        "family_counts": family_counts,
        "recent_rows_ge_334": [short_row(row) for row in sorted(recent, key=lambda row: row["id"])],
        "pending_same_batch": [short_row(row) for row in sorted(pending_same_batch, key=lambda row: row["id"])],
        "top_id338_neighbors": [short_row(row) for row in id338_neighbors[:30]],
        "top_q3_s3_neighbors": [short_row(row) for row in q3_s3_neighbors[:20]],
        "decision": {
            "wait_for_kaggle_or_dacon": "no same-batch score wait remains; Kaggle v3 is closed",
            "create_submission_artifact": False,
            "submit_now": False,
            "spend_rule_after_2330_kst": (
                "The user-approved 23:30 KST spend rule has already been executed for 2026-06-23 via IDs 338, 342, and 339."
            ),
            "next_modeling_action": (
                "Do not continue the same S1-only or Q-sequence cascade. If another no-submit cycle is available, "
                "build an ID342-aware Q3/S3 independent-source probe from materially new raw-source features, while "
                "still using ID338 distance and known failed-direction projection as secondary safety screens."
            ),
            "no_submission_created": True,
            "no_base_prediction_artifact_created": True,
            "no_notebook_artifact_created": True,
            "no_submission_log_row_created": True,
            "no_dacon_submission_made": True,
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID338 public-feedback next action",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifact, base prediction, notebook, submission-log row, Kaggle run, or DACON submission.",
        "",
        "Correction: authenticated DACON submission history shows ID342 is the current best public anchor. This report keeps ID338-distance tables only as neighborhood diagnostics.",
        "",
        "## Public feedback",
        "",
        f"- Best confirmed ID: `{report['public_feedback']['best_confirmed_id']}`",
        f"- Best confirmed Public: `{report['public_feedback']['best_confirmed_public']:.10f}`",
        f"- Delta vs ID337: `{report['public_feedback']['delta_vs_id337']:.10f}`",
        f"- Delta vs ID178: `{report['public_feedback']['delta_vs_id178']:.10f}`",
        f"- Interpretation: {report['public_feedback']['interpretation']}",
        "",
        "## Recent candidates",
        "",
        "| id | local | public | tier | family | proxy | mean delta338 | max delta338 | failed proj | targets | file |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["recent_rows_ge_334"], limit=80),
        "",
        "## Same-batch scored rows",
        "",
        "| id | local | public | tier | family | proxy | mean delta338 | max delta338 | failed proj | targets | file |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["pending_same_batch"], limit=10),
        "",
        "## Top ID338 neighbors",
        "",
        "| id | local | public | tier | family | proxy | mean delta338 | max delta338 | failed proj | targets | file |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["top_id338_neighbors"], limit=20),
        "",
        "## Decision",
        "",
        f"- Wait state: {report['decision']['wait_for_kaggle_or_dacon']}",
        f"- 23:30 KST spend rule: {report['decision']['spend_rule_after_2330_kst']}",
        f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
        f"- Submit now: `{report['decision']['submit_now']}`",
        f"- Next modeling action: {report['decision']['next_modeling_action']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id338_public_feedback_next_action_20260623.json",
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
