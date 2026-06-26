#!/usr/bin/env python3
"""Q3/S3 calendar and sleep-window prior smoke after post-ID341.

This diagnostic tests a lightweight, materially different no-approval source
from the raw physio-light-onset and semantic-routine lanes: calendar position,
subject progress, demographics, and already-materialized sleep-window summary
features. It reuses the frozen fold-internal anchor comparison machinery and
remains report-only.

It does not create full-test predictions, base prediction artifacts, submission
CSVs, notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or
DACON submissions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (
    ANCHOR_IDS,
    BLEND_WEIGHTS,
    CONFIG_MAX_DELTA_CEILING,
    CONFIG_MEAN_DELTA_CEILING,
    CONFIG_WIN_FLOOR,
    EPS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    find_oof,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    scan_rows_for_target,
    target_loss_on_split,
)

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
PROCESSED_TRAIN = ROOT / "data" / "processed" / "df_train_raw.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "df_test_raw.parquet"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_calendar_sleep_prior_smoke_20260618.json"

TARGET_FOCUS = ["Q3", "S3"]
TARGET_COLUMNS = set(TARGETS)
KEY_COLUMNS = {"subject_id", "sleep_date", "lifelog_date", "date", "row_date", "is_test"}
BASE_FEATURE_PATTERNS = (
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "is_rest_day",
    "demo_",
    "sleep",
    "slp_",
    "presleep",
    "screen_night",
    "screen_eve",
    "act_night",
    "act_eve",
    "mlight_night",
    "mlight_dark",
    "wlight_night",
    "wlight_dark",
    "ac_night",
    "ac_eve",
    "wHr_sleep",
    "wHr_slp",
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        if np.isfinite(value):
            return float(value)
    return None


def selected_base_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in KEY_COLUMNS or col in TARGET_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        lower = col.lower()
        if any(pattern.lower() in lower for pattern in BASE_FEATURE_PATTERNS):
            cols.append(col)
    return sorted(set(cols))


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["row_date"] = pd.to_datetime(out["lifelog_date"]).dt.date
    date_ts = pd.to_datetime(out["row_date"])
    min_date = date_ts.min()
    out["csp_global_day_index"] = (date_ts - min_date).dt.days.astype(float)
    out["csp_global_day_index_sq"] = out["csp_global_day_index"] ** 2
    dayofweek = pd.to_numeric(out.get("dayofweek", date_ts.dt.dayofweek), errors="coerce").astype(float)
    out["csp_dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    out["csp_dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    out = out.sort_values(["subject_id", "row_date"]).reset_index(drop=True)
    out["csp_subject_day_index"] = out.groupby("subject_id").cumcount().astype(float)
    out["csp_subject_day_index_sq"] = out["csp_subject_day_index"] ** 2
    subject_span = out.groupby("subject_id")["csp_subject_day_index"].transform("max").replace(0, np.nan)
    out["csp_subject_progress"] = (out["csp_subject_day_index"] / subject_span).fillna(0.0)
    return out


def add_missing_indicators(frame: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    missing_cols: list[str] = []
    for col in base_cols:
        if out[col].isna().any():
            missing_col = f"{col}_missing"
            out[missing_col] = out[col].isna().astype(float)
            missing_cols.append(missing_col)
    return out, missing_cols


def build_calendar_sleep_prior_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    train = pd.read_parquet(PROCESSED_TRAIN)
    test = pd.read_parquet(PROCESSED_TEST)
    train = train.copy().assign(is_test=0)
    test = test.copy().assign(is_test=1)
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined["subject_id"] = combined["subject_id"].astype(str)
    combined = add_calendar_features(combined)
    base_cols = selected_base_columns(combined)
    generated_cols = [col for col in combined.columns if col.startswith("csp_")]
    candidate_cols = sorted(set(base_cols + generated_cols))
    combined, missing_cols = add_missing_indicators(combined, candidate_cols)
    feature_cols = feature_columns(combined)
    raw_feature_cols = [col for col in feature_cols if not col.endswith("_missing")]
    train_rows = combined[combined["is_test"].eq(0)]
    test_rows = combined[combined["is_test"].eq(1)]
    coverage = {
        "n_rows": int(len(combined)),
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "raw_feature_columns": int(len(raw_feature_cols)),
        "missing_indicator_columns": int(len(missing_cols)),
        "feature_columns": int(len(feature_cols)),
        "mean_missing_rate": safe_float(combined[feature_cols].isna().mean().mean()) if feature_cols else None,
        "train_any_nonnull_row_share": safe_float(train_rows[raw_feature_cols].notna().any(axis=1).mean())
        if raw_feature_cols
        else None,
        "test_any_nonnull_row_share": safe_float(test_rows[raw_feature_cols].notna().any(axis=1).mean())
        if raw_feature_cols
        else None,
        "all_null_raw_features": [col for col in raw_feature_cols if combined[col].notna().sum() == 0],
        "raw_identifier_columns_exported": [
            col for col in raw_feature_cols if col in {"subject_id", "sleep_date", "lifelog_date", "date", "row_date"}
        ],
        "direct_target_columns_exported": [col for col in raw_feature_cols if col in TARGET_COLUMNS],
        "feature_family_counts": {
            "calendar_generated": int(sum(col.startswith("csp_") for col in raw_feature_cols)),
            "demographic": int(sum(col.startswith("demo_") for col in raw_feature_cols)),
            "sleep_window": int(
                sum(
                    any(token in col.lower() for token in ("sleep", "slp_", "presleep", "night", "eve"))
                    for col in raw_feature_cols
                )
            ),
        },
    }
    return combined, coverage


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in KEY_COLUMNS
        and col not in TARGET_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[col])
        and frame[col].notna().any()
    ]


def aligned_feature_frame(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, coverage = build_calendar_sleep_prior_frame()
    train_features = features[features["is_test"].eq(0)].copy()
    train_features["date"] = pd.to_datetime(train_features["row_date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(
        train_features.drop(columns=["is_test"]), on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    if len(aligned) != len(train):
        raise ValueError("Calendar/sleep-prior feature frame does not align with sorted train rows.")
    coverage = dict(coverage)
    coverage["aligned_train_rows"] = int(len(aligned))
    return aligned, coverage


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGET_FOCUS}
    if not rows:
        return out
    frame = pd.DataFrame(rows)
    for target in TARGET_FOCUS:
        target_frame = frame[frame["target"].eq(target)].copy()
        summaries: list[dict[str, Any]] = []
        for candidate, group in target_frame.groupby("candidate", sort=False):
            deltas = group.set_index("split_name")["delta_vs_id341"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            delta_values = [float(v) for v in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "candidate": str(candidate).replace("q2q3s3", "q3s3_calendar_sleep_prior"),
                    "model_spec": row0.get("model_spec"),
                    "anchor": row0.get("anchor"),
                    "weight": row0.get("weight"),
                    "primary_wins_vs_id341": int(sum(delta < 0 for delta in delta_values)),
                    "mean_delta_vs_id341": float(np.mean(delta_values)),
                    "min_delta_vs_id341": float(np.min(delta_values)),
                    "max_delta_vs_id341": float(np.max(delta_values)),
                    "split_deltas_vs_id341": {str(k): float(v) for k, v in deltas.items()},
                    "split_losses": {str(k): float(v) for k, v in losses.items()},
                }
            )
        out[target] = sorted(
            summaries,
            key=lambda row: (
                -row["primary_wins_vs_id341"],
                row["mean_delta_vs_id341"],
                row["max_delta_vs_id341"],
                row["candidate"],
            ),
        )
    return out


def select_target_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in summaries.items():
        for row in rows:
            if (
                row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
                and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
                and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
            ):
                selected[target] = row
                break
    return selected


def split_anchor_table(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    split_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        for split_name in split_names:
            folds = splits[split_name]
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "id178_loss": target_loss_on_split(train, folds, target, references["id178"]),
                    "id341_loss": target_loss_on_split(train, folds, target, references["id341"]),
                }
            )
    return rows


def build_report(
    model_specs=MODEL_SPECS,
    split_names: list[str] | None = None,
) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    train = load_train_sorted()
    features, coverage = aligned_feature_frame(train)
    cols = feature_columns(features)
    splits = load_split_manifest()
    references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        rows = scan_rows_for_target(train, features, cols, splits, references, target, model_specs, split_names)
        for row in rows:
            row["candidate"] = str(row["candidate"]).replace("q2q3s3", "q3s3_calendar_sleep_prior")
        scan_rows.extend(rows)
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    selected_targets = sorted(selected.keys())
    all_targets_covered = selected_targets == TARGET_FOCUS
    any_signal = bool(selected_targets)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_calendar_sleep_prior_smoke",
        "purpose": (
            "Run a lightweight fold-internal smoke on calendar, subject-progress, demographic, and "
            "already-materialized sleep-window summary features for Q3/S3."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "processed_train": rel(PROCESSED_TRAIN),
            "processed_test": rel(PROCESSED_TEST),
            "split_manifest": rel(REPORTS_DIR / "id340_validation_split_manifest_20260618.json"),
            "id178_oof": rel(find_oof(178)),
            "id341_oof": rel(find_oof(341)),
        },
        "feature_coverage": coverage,
        "config": {
            "model_specs": [asdict(spec) for spec in model_specs],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "eps": EPS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "anchor_rows": split_anchor_table(train, splits, references, split_names),
        "selected_target_configs": selected,
        "top_rows_by_target": {target: rows[:8] for target, rows in summaries.items()},
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "model_smoke_completed": True,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "q3_signal_found": "Q3" in selected,
            "s3_signal_found": "S3" in selected,
            "all_q3s3_targets_covered": all_targets_covered,
            "public_058_evidence": False,
            "next_local_allowed_step": "context_only_geometry_review"
            if any_signal
            else "close_calendar_sleep_prior_lane",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Keep any selected Q3/S3 calendar/sleep-prior row as context only; do not create a submission "
                "artifact unless a later independent gate covers Q3/S3 and clears the Public-0.58 threshold. If no "
                "target is selected, keep the explicit-approved upstream rebuild as the score-relevant route."
            ),
        },
        "no_full_test_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    coverage = report["feature_coverage"]
    lines = [
        "# Q3/S3 calendar/sleep-prior smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: train `{coverage['n_train_rows']}`, test `{coverage['n_test_rows']}`",
        f"- Raw feature columns: `{coverage['raw_feature_columns']}`",
        f"- Missing indicators: `{coverage['missing_indicator_columns']}`",
        f"- Mean missing rate: `{coverage['mean_missing_rate']:.4f}`",
        f"- Train/test any-non-null row share: `{coverage['train_any_nonnull_row_share']}` / `{coverage['test_any_nonnull_row_share']}`",
        f"- Direct target columns exported: `{coverage['direct_target_columns_exported']}`",
        "",
        "## Selected target configs",
        "",
        "| target | candidate | wins | mean Δ341 | max Δ341 |",
        "|---|---|---:|---:|---:|",
    ]
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    lines.extend(["", "## Top rows by target", ""])
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | candidate | wins | mean Δ341 | max Δ341 |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows[:5], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Model smoke completed: `{decision['model_smoke_completed']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Q3 signal found: `{decision['q3_signal_found']}`",
            f"- S3 signal found: `{decision['s3_signal_found']}`",
            f"- All Q3/S3 targets covered: `{decision['all_q3s3_targets_covered']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
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
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
