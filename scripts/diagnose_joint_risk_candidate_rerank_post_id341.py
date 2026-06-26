#!/usr/bin/env python3
"""Rerank existing candidates with post-ID341 joint-risk warnings.

This report-only diagnostic applies the public joint-distribution warning lens to
all logged submission CSVs that can be loaded locally. It is intentionally a
candidate-selection warning, not a public-score model and not a submission
creator. The thresholds are conservative envelopes around the current safer
public anchors ID178/ID329, with ID287 used only as a known collapse contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_joint_distribution_risk_post_id341 import (  # noqa: E402
    LOG_PATH,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    TEST_PATH,
    TRAIN_PATH,
    candidate_metrics,
    load_submission,
    train_reference_stats,
)

KST = timezone(timedelta(hours=9))
PUBLIC_JOINT_RISK_PATH = REPORTS_DIR / "public_joint_distribution_risk_post_id341_20260618.json"
SAFE_ANCHOR_IDS = [178, 329]
COLLAPSE_ANCHOR_ID = 287
CONTROL_IDS = [178, 287, 329, 337, 339, 340, 341]
RECENT_ID_MIN = 330


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def quantile(values: list[float], q: float) -> float | None:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return None
    return float(np.quantile(clean, q))


def compute_thresholds(anchor_rows: list[dict[str, Any]], collapse_row: dict[str, Any] | None) -> dict[str, Any]:
    safe_rows = [row for row in anchor_rows if row["id"] in SAFE_ANCHOR_IDS]
    if len(safe_rows) < 2:
        raise ValueError("Need both ID178 and ID329 to compute safe joint-risk thresholds.")

    def max_safe(metric: str) -> float:
        return max(float(row[metric]) for row in safe_rows if row.get(metric) is not None)

    def min_safe(metric: str) -> float:
        return min(float(row[metric]) for row in safe_rows if row.get(metric) is not None)

    thresholds = {
        "target_mean_l1_to_train": max_safe("target_mean_l1_to_train") + 0.003,
        "target_mean_max_abs_to_train": max_safe("target_mean_max_abs_to_train") + 0.010,
        "pair11_l1_to_train": max_safe("pair11_l1_to_train") + 0.004,
        "extreme_rate_05_95": max_safe("extreme_rate_05_95") + 0.025,
        "extreme_rate_01_99": max_safe("extreme_rate_01_99") + 0.010,
        "prediction_sharpness": max_safe("prediction_sharpness") + 0.035,
        "prediction_entropy_min": min_safe("prediction_entropy") - 0.045,
        "temporal_volatility_ratio_to_train": max_safe("temporal_volatility_ratio_to_train") + 0.160,
        "mean_abs_delta_vs_id178": 0.045,
        "max_abs_delta_vs_id178": 0.300,
    }
    if collapse_row:
        thresholds["collapse_contrast_id"] = collapse_row["id"]
        thresholds["collapse_mean_abs_delta_vs_id178"] = collapse_row.get("mean_abs_delta_vs_id178")
        thresholds["collapse_extreme_rate_05_95"] = collapse_row.get("extreme_rate_05_95")
    return thresholds


def warning_flags(row: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    checks = [
        ("target_mean_l1_to_train", ">"),
        ("target_mean_max_abs_to_train", ">"),
        ("pair11_l1_to_train", ">"),
        ("extreme_rate_05_95", ">"),
        ("extreme_rate_01_99", ">"),
        ("prediction_sharpness", ">"),
        ("temporal_volatility_ratio_to_train", ">"),
        ("mean_abs_delta_vs_id178", ">"),
        ("max_abs_delta_vs_id178", ">"),
    ]
    flags = []
    for metric, _op in checks:
        value = row.get(metric)
        threshold = thresholds.get(metric)
        if value is not None and threshold is not None and float(value) > float(threshold):
            flags.append(metric)
    entropy = row.get("prediction_entropy")
    entropy_min = thresholds.get("prediction_entropy_min")
    if entropy is not None and entropy_min is not None and float(entropy) < float(entropy_min):
        flags.append("prediction_entropy_low")
    return flags


def risk_tier(row: dict[str, Any], flags: list[str]) -> str:
    mean_delta = row.get("mean_abs_delta_vs_id178")
    local = row.get("local_cv")
    if row.get("public_lb") is not None:
        return "already_scored_reference"
    if len(flags) >= 4:
        return "reject_high_joint_risk"
    if mean_delta is not None and float(mean_delta) > 0.080:
        return "reject_high_anchor_distance"
    if local is not None and float(local) < 0.520 and len(flags) >= 2:
        return "frontier_overfit_warning"
    if len(flags) >= 2:
        return "caution_joint_risk"
    if len(flags) == 1:
        return "watch_one_warning"
    return "low_joint_warning"


def load_candidate_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce")
    for col in ["local_cv_score_num", "public_lb_score_num"]:
        log[col] = pd.to_numeric(log[col], errors="coerce")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    return log, train, test


def build_metrics_for_log(
    log: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_key = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    train_stats = train_reference_stats(train)
    loadable = log[
        log["id_num"].notna()
        & log["submission_file"].notna()
        & log["local_cv_score_num"].notna()
        & log["submission_file"].astype(str).str.endswith(".csv")
    ].copy()

    loaded: dict[int, pd.DataFrame] = {}
    skipped = []
    for _idx, row in loadable.iterrows():
        candidate_id = int(row["id_num"])
        path = SUBMISSIONS_DIR / str(row["submission_file"])
        if not path.exists():
            skipped.append({"id": candidate_id, "submission_file": row["submission_file"], "reason": "missing_file"})
            continue
        try:
            loaded[candidate_id] = load_submission(path, reference_key)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": candidate_id, "submission_file": row["submission_file"], "reason": str(exc)})

    raw_predictions = {candidate_id: frame[TARGETS].astype(float).to_numpy() for candidate_id, frame in loaded.items()}
    anchor_predictions = {
        candidate_id: values
        for candidate_id, values in raw_predictions.items()
        if candidate_id in [*SAFE_ANCHOR_IDS, COLLAPSE_ANCHOR_ID, 341]
    }
    rows = []
    for _idx, row in loadable.iterrows():
        candidate_id = int(row["id_num"])
        if candidate_id not in loaded:
            continue
        rows.append(candidate_metrics(row, loaded[candidate_id], train_stats, anchor_predictions))
    return rows, skipped


def decorate_rows(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        flags = warning_flags(row, thresholds)
        decorated = {
            **row,
            "joint_warning_flags": flags,
            "joint_warning_count": len(flags),
            "joint_risk_tier": risk_tier(row, flags),
        }
        out.append(decorated)
    return out


def short_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "id",
        "submission_file",
        "local_cv",
        "public_lb",
        "status_clean",
        "joint_risk_tier",
        "joint_warning_count",
        "joint_warning_flags",
        "target_mean_l1_to_train",
        "pair11_l1_to_train",
        "extreme_rate_05_95",
        "prediction_entropy",
        "prediction_sharpness",
        "temporal_volatility_ratio_to_train",
        "mean_abs_delta_vs_id178",
        "max_abs_delta_vs_id178",
    ]
    return {key: row.get(key) for key in keep if key in row}


def build_report() -> dict[str, Any]:
    public_risk = load_json(PUBLIC_JOINT_RISK_PATH)
    log, train, test = load_candidate_rows()
    rows, skipped = build_metrics_for_log(log, train, test)
    anchor_rows = [row for row in rows if row["id"] in [*SAFE_ANCHOR_IDS, COLLAPSE_ANCHOR_ID]]
    collapse_row = next((row for row in rows if row["id"] == COLLAPSE_ANCHOR_ID), None)
    thresholds = compute_thresholds(anchor_rows, collapse_row)
    decorated = decorate_rows(rows, thresholds)
    unscored = [row for row in decorated if row.get("public_lb") is None]
    recent = [row for row in decorated if row["id"] >= RECENT_ID_MIN]
    controls = [row for row in decorated if row["id"] in CONTROL_IDS]

    ranked_reviewable = sorted(
        [
            row
            for row in unscored
            if row["joint_risk_tier"] in {"low_joint_warning", "watch_one_warning", "caution_joint_risk"}
        ],
        key=lambda row: (
            row["joint_warning_count"],
            row.get("mean_abs_delta_vs_id178") if row.get("mean_abs_delta_vs_id178") is not None else 999.0,
            row.get("local_cv") if row.get("local_cv") is not None else 999.0,
        ),
    )
    local_frontier = sorted(unscored, key=lambda row: row.get("local_cv") if row.get("local_cv") is not None else 999.0)
    tier_counts = pd.Series([row["joint_risk_tier"] for row in decorated]).value_counts().to_dict()
    unscored_tier_counts = pd.Series([row["joint_risk_tier"] for row in unscored]).value_counts().to_dict()

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "joint_risk_candidate_rerank_post_id341",
        "purpose": "Apply post-ID341 joint-distribution warning metrics to existing logged candidates without creating artifacts.",
        "input_counts": {
            "logged_rows": int(len(log)),
            "metric_rows": int(len(decorated)),
            "unscored_rows": int(len(unscored)),
            "recent_rows_ge_330": int(len(recent)),
            "skipped_rows": int(len(skipped)),
        },
        "public_joint_risk_source": str(PUBLIC_JOINT_RISK_PATH.relative_to(ROOT)),
        "strong_signal_metric_names": public_risk.get("decision", {}).get("strong_signal_metric_names", []),
        "thresholds": thresholds,
        "tier_counts": tier_counts,
        "unscored_tier_counts": unscored_tier_counts,
        "controls": [short_row(row) for row in sorted(controls, key=lambda item: item["id"])],
        "recent_rows_ge_330": [short_row(row) for row in sorted(recent, key=lambda item: item["id"])],
        "top_unscored_low_joint_warning": [short_row(row) for row in ranked_reviewable[:30]],
        "top_unscored_by_local_cv": [short_row(row) for row in local_frontier[:30]],
        "skipped": skipped[:80],
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "best_low_joint_warning_id": ranked_reviewable[0]["id"] if ranked_reviewable else None,
            "best_low_joint_warning_file": ranked_reviewable[0]["submission_file"] if ranked_reviewable else None,
            "frontier_candidates_have_joint_warnings": any(
                row["joint_warning_count"] > 0 for row in local_frontier[:10]
            ),
            "next_action": (
                "Use this rerank as a warning overlay before any manual submission decision. It does not override "
                "the post-ID341 transfer gate; candidates still need explicit approval and artifact-contract review."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_num(value)
    return "" if value is None else f"{value:.{digits}f}"


def table_rows(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    lines = []
    for row in rows[:limit]:
        flags = ",".join(row.get("joint_warning_flags") or [])
        lines.append(
            f"| {row['id']} | {fmt(row.get('local_cv'))} | {fmt(row.get('public_lb'))} | "
            f"`{row.get('joint_risk_tier')}` | {row.get('joint_warning_count')} | "
            f"{fmt(row.get('mean_abs_delta_vs_id178'))} | {fmt(row.get('extreme_rate_05_95'))} | "
            f"{fmt(row.get('pair11_l1_to_train'))} | `{flags}` | `{row.get('submission_file')}` |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Joint-risk candidate rerank — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It applies joint-distribution warning metrics to existing logged candidate CSVs. It creates no submission artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Input coverage",
        "",
        f"- Metric rows: `{report['input_counts']['metric_rows']}`",
        f"- Unscored rows: `{report['input_counts']['unscored_rows']}`",
        f"- Recent rows `id >= 330`: `{report['input_counts']['recent_rows_ge_330']}`",
        f"- Skipped rows: `{report['input_counts']['skipped_rows']}`",
        "",
        "## Tier counts",
        "",
        "```json",
        json.dumps(report["unscored_tier_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Controls",
        "",
        "| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["controls"], limit=50),
        "",
        "## Recent candidates `id >= 330`",
        "",
        "| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["recent_rows_ge_330"], limit=80),
        "",
        "## Top unscored low-warning candidates",
        "",
        "| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["top_unscored_low_joint_warning"], limit=30),
        "",
        "## Local-CV frontier with warnings",
        "",
        "| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        *table_rows(report["top_unscored_by_local_cv"], limit=30),
        "",
        "## Decision",
        "",
        f"- Best low-warning ID: `{report['decision']['best_low_joint_warning_id']}`",
        f"- Best low-warning file: `{report['decision']['best_low_joint_warning_file']}`",
        f"- Frontier candidates have joint warnings: `{report['decision']['frontier_candidates_have_joint_warnings']}`",
        f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
        f"- Submit now: `{report['decision']['submit_now']}`",
        f"- Next action: {report['decision']['next_action']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "joint_risk_candidate_rerank_post_id341_20260618.json",
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
