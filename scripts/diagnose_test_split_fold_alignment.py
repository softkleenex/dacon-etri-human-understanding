#!/usr/bin/env python3
"""Diagnose how local CV folds resemble the hidden test split shape.

This is a report-only validation diagnostic. It does not train models, create
submission files, or update the submission log.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.validation import make_subject_hole_folds, make_subject_timeseries_folds

TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))


def _run_lengths(labels: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not labels:
        return runs
    current = labels[0]
    start = 0
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            runs.append({"label": current, "start": start, "end": idx - 1, "length": idx - start})
            current = label
            start = idx
    runs.append({"label": current, "start": start, "end": len(labels) - 1, "length": len(labels) - start})
    return runs


def _histogram(values: np.ndarray, bins: int = 10) -> list[float]:
    if values.size == 0:
        return [0.0] * bins
    clipped = np.clip(values.astype(float), 0.0, 1.0)
    hist, _ = np.histogram(clipped, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return [0.0] * bins
    return (hist / total).astype(float).tolist()


def _l1_distance(a: list[float], b: list[float]) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


def _build_timeline(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_part = train[["subject_id", "sleep_date", "lifelog_date"]].copy()
    train_part["split"] = "train"
    train_part["source_index"] = train_part.index.astype(int)
    test_part = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    test_part["split"] = "test"
    test_part["source_index"] = test_part.index.astype(int)
    all_rows = pd.concat([train_part, test_part], ignore_index=True)
    all_rows = all_rows.sort_values(["subject_id", "sleep_date", "lifelog_date", "split"]).reset_index(drop=True)
    all_rows["subject_pos"] = all_rows.groupby("subject_id").cumcount()
    all_rows["subject_n"] = all_rows.groupby("subject_id")["subject_pos"].transform("max") + 1
    all_rows["relative_pos"] = np.where(
        all_rows["subject_n"] > 1,
        all_rows["subject_pos"] / (all_rows["subject_n"] - 1),
        0.0,
    )
    return all_rows


def _subject_split_summary(timeline: pd.DataFrame, train: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for subject_id, group in timeline.groupby("subject_id", sort=True):
        labels = group["split"].tolist()
        runs = _run_lengths(labels)
        test_rows = group[group["split"] == "test"]
        train_rows = group[group["split"] == "train"]
        train_target_rates = (
            train.loc[train["subject_id"] == subject_id, TARGET_COLS].mean(numeric_only=True).round(6).to_dict()
        )
        summaries.append(
            {
                "subject_id": subject_id,
                "train_n": int((group["split"] == "train").sum()),
                "test_n": int((group["split"] == "test").sum()),
                "all_n": int(len(group)),
                "test_fraction": float((group["split"] == "test").mean()),
                "train_min": str(train_rows["sleep_date"].min().date()) if not train_rows.empty else None,
                "train_max": str(train_rows["sleep_date"].max().date()) if not train_rows.empty else None,
                "test_min": str(test_rows["sleep_date"].min().date()) if not test_rows.empty else None,
                "test_max": str(test_rows["sleep_date"].max().date()) if not test_rows.empty else None,
                "n_runs": int(len(runs)),
                "test_run_count": int(sum(run["label"] == "test" for run in runs)),
                "runs": runs,
                "target_positive_rates_train": train_target_rates,
            }
        )
    return summaries


def _fold_relative_positions(train: pd.DataFrame, timeline: pd.DataFrame, valid_idx: np.ndarray) -> np.ndarray:
    valid_keys = train.loc[valid_idx, ["subject_id", "sleep_date", "lifelog_date"]].copy()
    valid_keys["split"] = "train"
    merged = valid_keys.merge(
        timeline[["subject_id", "sleep_date", "lifelog_date", "split", "relative_pos"]],
        on=["subject_id", "sleep_date", "lifelog_date", "split"],
        how="left",
    )
    return merged["relative_pos"].dropna().to_numpy(dtype=float)


def _fold_run_stats(train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for fold_id, (_train_idx, valid_idx) in enumerate(folds):
        valid_set = {int(x) for x in valid_idx}
        subject_rows: list[dict[str, Any]] = []
        for subject_id, group in train.sort_values(["subject_id", "sleep_date"]).groupby("subject_id", sort=True):
            labels = ["valid" if int(idx) in valid_set else "train" for idx in group.index]
            runs = _run_lengths(labels)
            valid_runs = [run for run in runs if run["label"] == "valid"]
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "valid_n": int(sum(label == "valid" for label in labels)),
                    "valid_run_count": int(len(valid_runs)),
                    "valid_run_lengths": [int(run["length"]) for run in valid_runs],
                }
            )
        stats.append({"fold": fold_id, "subjects": subject_rows})
    return stats


def _actual_test_band_holdout(
    train: pd.DataFrame, timeline: pd.DataFrame, bins: int = 10
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out train rows whose full-timeline positions fall in occupied test bins."""
    test_bins = set(
        np.floor(
            np.clip(timeline.loc[timeline["split"] == "test", "relative_pos"].to_numpy(dtype=float), 0.0, 0.999999)
            * bins
        ).astype(int)
    )
    train_timeline = timeline[timeline["split"] == "train"].copy()
    train_bins = np.floor(np.clip(train_timeline["relative_pos"].to_numpy(dtype=float), 0.0, 0.999999) * bins).astype(
        int
    )
    valid_source_idx = train_timeline.loc[[int(bin_id) in test_bins for bin_id in train_bins], "source_index"].to_numpy(
        dtype=int
    )
    all_idx = train.index.to_numpy(dtype=int)
    train_idx = np.setdiff1d(all_idx, valid_source_idx, assume_unique=False)
    return [(train_idx, valid_source_idx)] if len(train_idx) and len(valid_source_idx) else []


def _test_neighbor_holdout(
    train: pd.DataFrame,
    timeline: pd.DataFrame,
    neighbor_positions: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out train rows adjacent to the known hidden-test blocks."""
    valid_parts: list[np.ndarray] = []
    for _subject_id, group in timeline.groupby("subject_id", sort=True):
        test_pos = group.loc[group["split"] == "test", "subject_pos"].to_numpy(dtype=int)
        train_rows = group[group["split"] == "train"].copy()
        if test_pos.size == 0 or train_rows.empty:
            continue
        train_pos = train_rows["subject_pos"].to_numpy(dtype=int)
        min_dist = np.min(np.abs(train_pos[:, None] - test_pos[None, :]), axis=1)
        valid_parts.append(train_rows.loc[min_dist <= neighbor_positions, "source_index"].to_numpy(dtype=int))
    valid_idx = np.unique(np.concatenate(valid_parts)) if valid_parts else np.array([], dtype=int)
    all_idx = train.index.to_numpy(dtype=int)
    train_idx = np.setdiff1d(all_idx, valid_idx, assume_unique=False)
    return [(train_idx, valid_idx)] if len(train_idx) and len(valid_idx) else []


def _evaluate_strategy(
    name: str,
    fold_fn: Callable[[pd.DataFrame], list[tuple[np.ndarray, np.ndarray]]],
    train: pd.DataFrame,
    timeline: pd.DataFrame,
    test_hist: list[float],
) -> dict[str, Any]:
    folds = fold_fn(train)
    fold_rows: list[dict[str, Any]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        rel = _fold_relative_positions(train, timeline, valid_idx)
        hist = _histogram(rel)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_n": int(len(train_idx)),
                "valid_n": int(len(valid_idx)),
                "valid_fraction_of_train": float(len(valid_idx) / len(train)),
                "relative_pos_hist": hist,
                "l1_distance_to_actual_test_hist": _l1_distance(hist, test_hist),
                "relative_pos_mean": float(rel.mean()) if rel.size else None,
                "relative_pos_std": float(rel.std()) if rel.size else None,
            }
        )
    distances = [row["l1_distance_to_actual_test_hist"] for row in fold_rows]
    return {
        "strategy": name,
        "n_folds": int(len(folds)),
        "mean_l1_distance_to_actual_test_hist": float(np.mean(distances)) if distances else None,
        "min_l1_distance_to_actual_test_hist": float(np.min(distances)) if distances else None,
        "max_l1_distance_to_actual_test_hist": float(np.max(distances)) if distances else None,
        "folds": fold_rows,
        "fold_run_stats": _fold_run_stats(train, folds),
    }


def _format_runs(runs: list[dict[str, Any]]) -> str:
    return "; ".join(f"{run['label']}:{run['start']}-{run['end']}({run['length']})" for run in runs)


def write_markdown(report: dict[str, Any], output_md: Path) -> None:
    lines = [
        "# Test split vs CV fold alignment diagnostic",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is a report-only validation diagnostic. No model was trained, no submission CSV was created, and no DACON submission was made.",
        "",
        "## Actual train/test split shape",
        "",
        "| subject | train | test | test frac | test runs | timeline runs |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["subject_summaries"]:
        row_for_md = {**row, "runs_text": _format_runs(row["runs"])}
        lines.append(
            "| {subject_id} | {train_n} | {test_n} | {test_fraction:.3f} | {test_run_count} | `{runs_text}` |".format(
                **row_for_md,
            )
        )
    lines.extend(
        [
            "",
            "## Fold-shape comparison",
            "",
            f"Actual test relative-position histogram: `{np.round(report['actual_test_relative_pos_hist'], 4).tolist()}`",
            "",
            "| strategy | folds | mean L1 ↓ | min L1 | max L1 | verdict |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in report["strategies"]:
        verdict = (
            "closer to interleaved hidden-test shape"
            if strategy["strategy"] == report["best_strategy_by_histogram_l1"]
            else "less aligned"
        )
        lines.append(
            "| {strategy} | {n_folds} | {mean_l1_distance_to_actual_test_hist:.4f} | "
            "{min_l1_distance_to_actual_test_hist:.4f} | {max_l1_distance_to_actual_test_hist:.4f} | "
            "{verdict} |".format(**strategy, verdict=verdict)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The hidden test rows are not a pure future holdout. Most subjects have two test blocks: one middle block and one later block.",
            "- Some subjects (`id04`, `id05`, `id06`, `id08`) have fragmented later test blocks, so a single chronological split is structurally mismatched.",
            "- Pure histogram alignment can favor late-window validation, but prior public results show that a pure chronological validation story is not enough for this competition.",
            "- The diagnostic holdouts that use the known sample-row date geometry should be used as stress tests, not as a new optimization leaderboard.",
            "- A future candidate should pass subject-hole, test-geometry stress, and LOSO-style robustness, then be checked against ID178/ID329/ID287 geometry before promotion.",
            "",
            "## Decision",
            "",
            "- Public 0.58 plausible candidate from this diagnostic: **no**",
            "- Submission candidate promoted: **no**",
            "- Recommended use: validation design evidence only.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/raw/ch2026_metrics_train.csv")
    parser.add_argument("--test-csv", default="data/raw/ch2026_submission_sample.csv")
    parser.add_argument("--output-json", default="reports/test_split_fold_alignment_post_id329_20260617.json")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, parse_dates=["sleep_date", "lifelog_date"])
    test = pd.read_csv(args.test_csv, parse_dates=["sleep_date", "lifelog_date"])
    timeline = _build_timeline(train, test)
    test_rel = timeline.loc[timeline["split"] == "test", "relative_pos"].to_numpy(dtype=float)
    test_hist = _histogram(test_rel)

    strategies = [
        _evaluate_strategy(
            "subject_hole_5",
            lambda df: make_subject_hole_folds(df, n_folds=5),
            train,
            timeline,
            test_hist,
        ),
        _evaluate_strategy(
            "subject_timeseries_5",
            lambda df: make_subject_timeseries_folds(df, n_folds=5),
            train,
            timeline,
            test_hist,
        ),
        _evaluate_strategy(
            "actual_test_band_holdout_diagnostic",
            lambda df: _actual_test_band_holdout(df, timeline, bins=10),
            train,
            timeline,
            test_hist,
        ),
        _evaluate_strategy(
            "test_boundary_neighbor_holdout_diagnostic",
            lambda df: _test_neighbor_holdout(df, timeline, neighbor_positions=2),
            train,
            timeline,
            test_hist,
        ),
    ]
    best_strategy = min(
        strategies,
        key=lambda item: (
            item["mean_l1_distance_to_actual_test_hist"]
            if item["mean_l1_distance_to_actual_test_hist"] is not None
            else float("inf")
        ),
    )["strategy"]

    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "test_split_fold_alignment_post_id329",
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "target_cols": TARGET_COLS,
        "actual_test_relative_pos_hist": test_hist,
        "subject_summaries": _subject_split_summary(timeline, train),
        "strategies": strategies,
        "best_strategy_by_histogram_l1": best_strategy,
        "no_model_trained": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(report, output_json.with_suffix(".md"))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
