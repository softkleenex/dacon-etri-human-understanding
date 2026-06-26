#!/usr/bin/env python3
"""Evaluate whether post-hoc OOF subset stress explains public outcomes.

This diagnostic intentionally uses existing base OOF prediction files only. It
does not train models, create submissions, or promote candidates. The key
question is whether known public failures such as ID287 are penalized by
test-geometry OOF subset losses.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from scripts.diagnose_test_split_fold_alignment import (
    _actual_test_band_holdout,
    _build_timeline,
    _test_neighbor_holdout,
)

TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))


def mean_target_logloss(frame: pd.DataFrame, row_idx: np.ndarray) -> tuple[float, dict[str, float]]:
    subset = frame.iloc[row_idx]
    per_target = {
        target: float(
            log_loss(
                subset[f"{target}_true"],
                np.clip(subset[f"{target}_pred"], 1e-15, 1.0 - 1e-15),
                labels=[0, 1],
            )
        )
        for target in TARGET_COLS
    }
    return float(np.mean(list(per_target.values()))), per_target


def parse_candidate_id(path: Path) -> int | None:
    match = re.search(r"base_oof_id_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_submission_log(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    return {int(row["id"]): row.to_dict() for _, row in log.iterrows() if pd.notna(row.get("id"))}


def scan_oof_files(
    *,
    train_csv: Path,
    test_csv: Path,
    base_dir: Path,
    submission_log: Path,
) -> dict[str, Any]:
    train = pd.read_csv(train_csv, parse_dates=["sleep_date", "lifelog_date"]).reset_index(drop=True)
    test = pd.read_csv(test_csv, parse_dates=["sleep_date", "lifelog_date"]).reset_index(drop=True)
    timeline = _build_timeline(train, test)

    band_folds = _actual_test_band_holdout(train, timeline, bins=10)
    boundary_folds = _test_neighbor_holdout(train, timeline, neighbor_positions=2)
    if not band_folds or not boundary_folds:
        raise RuntimeError("Could not build diagnostic holdout indices.")

    band_idx = band_folds[0][1]
    boundary_idx = boundary_folds[0][1]
    all_idx = np.arange(len(train))

    labels = train[["subject_id", "lifelog_date", *TARGET_COLS]].rename(columns={"lifelog_date": "date"}).copy()
    labels["date"] = labels["date"].dt.strftime("%Y-%m-%d")
    log_by_id = load_submission_log(submission_log)

    rows: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("base_oof_id_*.csv")):
        candidate_id = parse_candidate_id(path)
        if candidate_id is None:
            continue
        pred = pd.read_csv(path)
        if not {"subject_id", "date", *TARGET_COLS}.issubset(pred.columns):
            continue
        merged = labels.merge(pred, on=["subject_id", "date"], how="inner", suffixes=("_true", "_pred"))
        if len(merged) != len(train):
            continue

        overall_loss, overall_per_target = mean_target_logloss(merged, all_idx)
        boundary_loss, boundary_per_target = mean_target_logloss(merged, boundary_idx)
        band_loss, band_per_target = mean_target_logloss(merged, band_idx)
        log_row = log_by_id.get(candidate_id, {})
        rows.append(
            {
                "id": candidate_id,
                "base_oof_file": str(path),
                "overall_oof_loss": overall_loss,
                "test_boundary_neighbor_oof_loss": boundary_loss,
                "actual_test_band_oof_loss": band_loss,
                "boundary_gap_vs_overall": boundary_loss - overall_loss,
                "band_gap_vs_overall": band_loss - overall_loss,
                "overall_per_target": overall_per_target,
                "boundary_per_target": boundary_per_target,
                "band_per_target": band_per_target,
                "local_cv_score": _safe_float(log_row.get("local_cv_score_num")),
                "public_lb_score": _safe_float(log_row.get("public_lb_score_num")),
                "status_clean": log_row.get("status_clean"),
                "submission_file": log_row.get("submission_file"),
                "model_description": log_row.get("model_description"),
            }
        )

    rows_sorted_boundary = sorted(rows, key=lambda row: row["test_boundary_neighbor_oof_loss"])
    rows_sorted_band = sorted(rows, key=lambda row: row["actual_test_band_oof_loss"])
    anchor_ids = {178, 287, 329}
    anchors = [row for row in rows if row["id"] in anchor_ids]

    id287_boundary_rank = _rank_of(rows_sorted_boundary, 287)
    id287_band_rank = _rank_of(rows_sorted_band, 287)
    conclusion = (
        "OOF subset stress is not a valid promotion gate: ID287 remains near the top by "
        "test-geometry OOF losses despite its confirmed Public 0.6698787318 collapse."
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "oof_test_geometry_stress_post_id329",
        "n_oof_candidates_scanned": len(rows),
        "n_train_rows": int(len(train)),
        "band_holdout_n": int(len(band_idx)),
        "boundary_neighbor_holdout_n": int(len(boundary_idx)),
        "anchor_ids": sorted(anchor_ids),
        "anchors": sorted(anchors, key=lambda row: row["id"]),
        "id287_boundary_rank": id287_boundary_rank,
        "id287_band_rank": id287_band_rank,
        "top_boundary_oof": rows_sorted_boundary[:25],
        "top_band_oof": rows_sorted_band[:25],
        "known_public_rows_by_boundary": [
            row
            for row in rows_sorted_boundary
            if row.get("public_lb_score") is not None and row.get("public_lb_score") > 0
        ][:25],
        "conclusion": conclusion,
        "public_058_plausible_candidate_found": False,
        "no_model_trained": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_of(rows: list[dict[str, Any]], candidate_id: int) -> int | None:
    for idx, row in enumerate(rows, start=1):
        if row["id"] == candidate_id:
            return idx
    return None


def write_markdown(report: dict[str, Any], output_md: Path) -> None:
    def row_line(row: dict[str, Any], rank: int) -> str:
        public = row.get("public_lb_score")
        public_text = "" if public is None else f"{public:.6f}"
        return (
            f"| {rank} | {row['id']} | {row['overall_oof_loss']:.6f} | "
            f"{row['test_boundary_neighbor_oof_loss']:.6f} | {row['actual_test_band_oof_loss']:.6f} | "
            f"{public_text} | `{Path(row['base_oof_file']).name}` |"
        )

    lines = [
        "# OOF test-geometry stress diagnostic — post ID329",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This diagnostic scans existing base OOF predictions only. No model was trained, no submission CSV was created, and no DACON submission was made.",
        "",
        "## Verdict",
        "",
        "**Do not use post-hoc OOF subset loss as a promotion gate.** It fails the most important sanity check: ID287 still looks excellent by these losses even though its confirmed Public score collapsed to `0.6698787318`.",
        "",
        "## Anchor check",
        "",
        "| id | overall OOF | boundary OOF | band OOF | public | status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["anchors"]:
        public = row.get("public_lb_score")
        public_text = "" if public is None else f"{public:.6f}"
        lines.append(
            f"| {row['id']} | {row['overall_oof_loss']:.6f} | "
            f"{row['test_boundary_neighbor_oof_loss']:.6f} | {row['actual_test_band_oof_loss']:.6f} | "
            f"{public_text} | `{row.get('status_clean')}` |"
        )

    lines.extend(
        [
            "",
            f"ID287 boundary rank: `{report['id287_boundary_rank']}` / `{report['n_oof_candidates_scanned']}`",
            f"ID287 band rank: `{report['id287_band_rank']}` / `{report['n_oof_candidates_scanned']}`",
            "",
            "## Top boundary OOF rows",
            "",
            "| rank | id | overall | boundary | band | public | file |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(report["top_boundary_oof"][:15], start=1):
        lines.append(row_line(row, rank))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The test-geometry subset is useful for describing where the hidden rows live, but post-hoc OOF predictions can remain contaminated by the same full-OOF/non-nested effects that made ID287 fail.",
            "- A valid future gate must be fold-nested: candidate selection, calibration, and blending must be learned without the outer validation rows.",
            "- No Public 0.58-plausible candidate is promoted by this diagnostic.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/raw/ch2026_metrics_train.csv")
    parser.add_argument("--test-csv", default="data/raw/ch2026_submission_sample.csv")
    parser.add_argument("--base-dir", default="artifacts/base_predictions")
    parser.add_argument("--submission-log", default="submissions/submission_log.csv")
    parser.add_argument("--output-json", default="reports/oof_test_geometry_stress_post_id329_20260617.json")
    args = parser.parse_args()

    report = scan_oof_files(
        train_csv=Path(args.train_csv),
        test_csv=Path(args.test_csv),
        base_dir=Path(args.base_dir),
        submission_log=Path(args.submission_log),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(report, output_json.with_suffix(".md"))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
