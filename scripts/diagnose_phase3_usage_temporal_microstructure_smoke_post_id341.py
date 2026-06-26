#!/usr/bin/env python3
"""Phase-3 usage temporal microstructure smoke after ID341.

Block A found a clean but too-small Q1 sleep/residual signal and block B closed
the S2 sleep/recovery retry. This report-only diagnostic tests a materially
different raw source family: time-of-day app-usage microstructure from
`mUsageStats`.

To avoid high-cardinality app memorization, this smoke does not use app names as
direct model columns. It converts raw app lists into coarse category/time-bin
load, entropy, concentration, and late-use features, then evaluates tiny
source overlays inside the frozen primary validation splits.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, no Kaggle push/run, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_phase2_block_a_q1_s4_sleep_residual_bridge_post_id341 import (  # noqa: E402
    evaluate_assembled_candidate,
    key_frame,
    predict_full_train_target,
    predict_target_for_split,
    selected_test_geometry,
    target_loss_on_split,
)
from scripts.diagnose_subject_residual_sequence_drift_smoke_post_id341 import (  # noqa: E402
    EPS,
    ID178_PUBLIC,
    ID341_TRANSFER_PROXY,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    PUBLIC_058_REVIEW_CUTOFF,
    TARGETS,
    load_processed_frames,
    load_reference_values,
    load_split_manifest,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_USAGE_PATH = ROOT / "data" / "raw" / "ch2025_data_items" / "ch2025_mUsageStats.parquet"
PHASE2_BLOCK_A_PATH = REPORTS_DIR / "phase2_block_a_q1_proxy_gap_post_id341_20260618.json"
PHASE2_BLOCK_B_PATH = REPORTS_DIR / "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json"
TARGET_SCAN_FOCUS = ["Q2", "Q3", "S1", "S2", "S4"]
ANCHORS = ["id178", "id341"]
MODEL_NAMES = ["lr_c0_03", "lr_c0_12", "hgb_l4_l2_5"]
BLEND_WEIGHTS = [0.005, 0.01, 0.025, 0.05]
CONFIG_WIN_FLOOR = 4
CONFIG_MEAN_DELTA_CEILING = -0.00005
CONFIG_MAX_DELTA_CEILING = 0.00050
FINAL_PRIMARY_WINS_FLOOR = 5
APP_CATEGORIES = ["communication", "media_search", "finance_shop", "routine_health", "system_home", "other"]
TIME_BINS = {
    "early_00_06": (0, 6),
    "morning_06_12": (6, 12),
    "afternoon_12_18": (12, 18),
    "evening_18_22": (18, 22),
    "late_22_24": (22, 24),
}


def normalize_app_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def categorize_app(app_name: Any) -> str:
    text = normalize_app_name(app_name)
    if any(token in text for token in ["카카오", "kakao", "메시지", "message", "전화", "통화", "문자"]):
        return "communication"
    if any(token in text for token in ["naver", "chrome", "youtube", "유튜브", "인터넷", "browser", "google"]):
        return "media_search"
    if any(
        token in text
        for token in [
            "토스",
            "pay",
            "페이",
            "카드",
            "bank",
            "은행",
            "쿠팡",
            "shop",
            "쇼핑",
            "lotte",
            "롯데",
        ]
    ):
        return "finance_shop"
    if any(token in text for token in ["캐시워크", "성경", "캘린더", "calendar", "health", "헬스", "운동"]):
        return "routine_health"
    if any(token in text for token in ["one ui", "홈", "launcher", "설정", "권한", "system", "관리자"]):
        return "system_home"
    return "other"


def time_bin(hour: int) -> str:
    for name, (start, end) in TIME_BINS.items():
        if start <= hour < end:
            return name
    return "unknown"


def shannon_entropy(values: pd.Series) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values.astype(float) / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def aggregate_usage_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    events_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for record_idx, row in enumerate(raw.itertuples(index=False)):
        timestamp = pd.Timestamp(row.timestamp)
        row_date = timestamp.date()
        hour = int(timestamp.hour)
        bin_name = time_bin(hour)
        entries = row.m_usage_stats
        try:
            entry_list = list(entries) if entries is not None else []
        except TypeError:
            entry_list = []
        record_total = 0.0
        record_apps = []
        for entry in entry_list:
            if not isinstance(entry, dict):
                continue
            app_norm = normalize_app_name(entry.get("app_name", ""))
            total_time = float(pd.to_numeric(entry.get("total_time", 0.0), errors="coerce") or 0.0)
            if not math.isfinite(total_time) or total_time < 0:
                total_time = 0.0
            events_rows.append(
                {
                    "subject_id": row.subject_id,
                    "row_date": row_date,
                    "record_idx": record_idx,
                    "time_bin": bin_name,
                    "app_norm": app_norm,
                    "category": categorize_app(app_norm),
                    "total_time": total_time,
                }
            )
            record_total += total_time
            record_apps.append(app_norm)
        record_rows.append(
            {
                "subject_id": row.subject_id,
                "row_date": row_date,
                "record_idx": record_idx,
                "time_bin": bin_name,
                "record_total_time": record_total,
                "record_app_count": len(record_apps),
                "record_unique_app_count": len(set(record_apps)),
            }
        )
    events = pd.DataFrame(events_rows)
    records = pd.DataFrame(record_rows)
    keys = ["subject_id", "row_date"]
    frames: list[pd.DataFrame] = []
    if not records.empty:
        day_records = records.groupby(keys).agg(
            utm_record_count=("record_idx", "nunique"),
            utm_record_total_time_sum=("record_total_time", "sum"),
            utm_record_total_time_mean=("record_total_time", "mean"),
            utm_record_total_time_max=("record_total_time", "max"),
            utm_record_app_count_mean=("record_app_count", "mean"),
            utm_record_unique_app_count_mean=("record_unique_app_count", "mean"),
        )
        frames.append(day_records)
        for bin_name in TIME_BINS:
            subset = records[records["time_bin"].eq(bin_name)]
            if not subset.empty:
                frames.append(
                    subset.groupby(keys).agg(
                        **{
                            f"utm_{bin_name}_record_count": ("record_idx", "nunique"),
                            f"utm_{bin_name}_time_sum": ("record_total_time", "sum"),
                            f"utm_{bin_name}_app_count_mean": ("record_app_count", "mean"),
                        }
                    )
                )
    if not events.empty:
        app_day = events.groupby(keys + ["app_norm"])["total_time"].sum().reset_index()
        cat_day = events.groupby(keys + ["category"])["total_time"].sum().unstack(fill_value=0.0)
        cat_day.columns = [f"utm_cat_{col}_time_sum" for col in cat_day.columns]
        frames.append(cat_day)
        entropy = app_day.groupby(keys)["total_time"].apply(shannon_entropy).rename("utm_app_time_entropy")
        frames.append(entropy.to_frame())
        total_by_day = app_day.groupby(keys)["total_time"].sum().rename("utm_app_time_total")
        top_by_day = app_day.groupby(keys)["total_time"].max().rename("utm_top_app_time")
        unique_by_day = app_day.groupby(keys)["app_norm"].nunique().rename("utm_unique_app_count")
        frames.extend([total_by_day.to_frame(), top_by_day.to_frame(), unique_by_day.to_frame()])
        for bin_name in TIME_BINS:
            subset = events[events["time_bin"].eq(bin_name)]
            if subset.empty:
                continue
            bin_app = subset.groupby(keys + ["app_norm"])["total_time"].sum().reset_index()
            bin_total = bin_app.groupby(keys)["total_time"].sum().rename(f"utm_{bin_name}_app_time_total")
            bin_entropy = bin_app.groupby(keys)["total_time"].apply(shannon_entropy).rename(f"utm_{bin_name}_entropy")
            bin_top = bin_app.groupby(keys)["total_time"].max().rename(f"utm_{bin_name}_top_app_time")
            frames.extend([bin_total.to_frame(), bin_entropy.to_frame(), bin_top.to_frame()])
            bin_cat = subset.groupby(keys + ["category"])["total_time"].sum().unstack(fill_value=0.0)
            bin_cat.columns = [f"utm_{bin_name}_cat_{col}_time_sum" for col in bin_cat.columns]
            frames.append(bin_cat)
    if not frames:
        return pd.DataFrame(columns=["subject_id", "row_date"]), {"raw_rows": int(len(raw)), "event_rows": 0}
    out = pd.concat(frames, axis=1).reset_index()
    numeric_cols = [col for col in out.columns if col.startswith("utm_")]
    if {"utm_top_app_time", "utm_app_time_total"}.issubset(out.columns):
        out["utm_top_app_share"] = out["utm_top_app_time"] / (out["utm_app_time_total"].abs() + 1.0)
    for category in APP_CATEGORIES:
        col = f"utm_cat_{category}_time_sum"
        if col in out.columns and "utm_app_time_total" in out.columns:
            out[f"utm_cat_{category}_share"] = out[col] / (out["utm_app_time_total"].abs() + 1.0)
    for bin_name in TIME_BINS:
        total_col = f"utm_{bin_name}_app_time_total"
        if total_col in out.columns and "utm_app_time_total" in out.columns:
            out[f"utm_{bin_name}_time_share"] = out[total_col] / (out["utm_app_time_total"].abs() + 1.0)
    if {"utm_late_22_24_time_share", "utm_evening_18_22_time_share"}.issubset(out.columns):
        out["utm_late_plus_evening_share"] = out["utm_late_22_24_time_share"] + out["utm_evening_18_22_time_share"]
    numeric_cols = [col for col in out.columns if col.startswith("utm_")]
    coverage = {
        "raw_rows": int(len(raw)),
        "event_rows": int(len(events)),
        "record_rows": int(len(records)),
        "subject_count": int(raw["subject_id"].nunique()),
        "timestamp_min": str(pd.to_datetime(raw["timestamp"]).min()),
        "timestamp_max": str(pd.to_datetime(raw["timestamp"]).max()),
        "unique_app_names": int(events["app_norm"].nunique()) if not events.empty else 0,
        "category_event_counts": (
            {str(k): int(v) for k, v in events["category"].value_counts().items()} if not events.empty else {}
        ),
        "daily_feature_rows": int(len(out)),
        "daily_feature_columns": int(len(numeric_cols)),
    }
    return out, coverage


def build_usage_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(RAW_USAGE_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    daily, raw_coverage = aggregate_usage_features(raw)
    rows = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    )
    rows["row_date"] = pd.to_datetime(rows["date"]).dt.date
    features = rows.merge(daily, on=["subject_id", "row_date"], how="left")
    numeric_cols = [
        col for col in features.columns if col.startswith("utm_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    for col in numeric_cols:
        if col.endswith("_time_sum") or col.endswith("_time_total") or col.endswith("_time") or "_time_" in col:
            features[f"{col}_log1p"] = np.log1p(features[col].clip(lower=0))
    numeric_cols = [
        col for col in features.columns if col.startswith("utm_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    for col in numeric_cols:
        features[f"{col}_missing"] = features[col].isna().astype(float)
    feature_cols = [col for col in features.columns if col.startswith("utm_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    features_test = features[features["is_test"].eq(1)].reset_index(drop=True)
    if not key_frame(train).equals(key_frame(features_train)):
        raise ValueError("Usage train features do not align with processed train rows.")
    if not key_frame(test).equals(key_frame(features_test)):
        raise ValueError("Usage test features do not align with processed test rows.")
    coverage = {
        **raw_coverage,
        "n_train_rows": int(len(features_train)),
        "n_test_rows": int(len(features_test)),
        "usage_feature_columns": int(len(feature_cols)),
        "train_rows_with_any_usage": int(features_train[feature_cols].notna().any(axis=1).sum()),
        "test_rows_with_any_usage": int(features_test[feature_cols].notna().any(axis=1).sum()),
        "train_missing_mean": float(features_train[feature_cols].isna().mean().mean()),
        "test_missing_mean": float(features_test[feature_cols].isna().mean().mean()),
        "top_missing_features": {
            str(k): float(v)
            for k, v in features_train[feature_cols].isna().mean().sort_values(ascending=False).head(12).items()
        },
    }
    return features_train, features_test, coverage


def scan_usage_configs(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    references = {"id178": reference178, "id341": reference341}
    source_cache: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for target in TARGET_SCAN_FOCUS:
        target_idx = TARGETS.index(target)
        id341_losses = {
            split_name: target_loss_on_split(train, folds, target, reference341[:, target_idx])
            for split_name, folds in splits.items()
        }
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            if spec.name not in MODEL_NAMES:
                continue
            for split_name, folds in splits.items():
                source_cache[(split_name, target, spec.name)] = predict_target_for_split(
                    features_train, feature_cols, y, folds, spec.name
                )
            for anchor_name in ANCHORS:
                anchor = references[anchor_name][:, target_idx]
                for weight in BLEND_WEIGHTS:
                    split_deltas: dict[str, float] = {}
                    split_losses: dict[str, float] = {}
                    for split_name, folds in splits.items():
                        source = np.where(
                            np.isnan(source_cache[(split_name, target, spec.name)]),
                            reference341[:, target_idx],
                            source_cache[(split_name, target, spec.name)],
                        )
                        values = np.clip((1.0 - weight) * anchor + weight * source, EPS, 1 - EPS)
                        loss = target_loss_on_split(train, folds, target, values)
                        split_losses[split_name] = loss
                        split_deltas[split_name] = float(loss - id341_losses[split_name])
                    deltas = list(split_deltas.values())
                    rows.append(
                        {
                            "target": target,
                            "model": spec.name,
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "candidate": f"{target}_{anchor_name}_usage_temporal_{spec.name}_w{weight:g}",
                            "primary_wins_vs_id341": int(sum(delta < 0 for delta in deltas)),
                            "mean_delta_vs_id341": float(np.mean(deltas)),
                            "min_delta_vs_id341": float(np.min(deltas)),
                            "max_delta_vs_id341": float(np.max(deltas)),
                            "split_deltas_vs_id341": split_deltas,
                            "split_losses": split_losses,
                        }
                    )
    rows = sorted(
        rows,
        key=lambda row: (
            row["target"],
            -row["primary_wins_vs_id341"],
            row["mean_delta_vs_id341"],
            row["max_delta_vs_id341"],
        ),
    )
    return rows, source_cache


def select_usage_configs(scan_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target in TARGET_SCAN_FOCUS:
        rows = [row for row in scan_rows if row["target"] == target]
        eligible = [
            row
            for row in rows
            if row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
            and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
            and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
        ]
        if eligible:
            selected[target] = sorted(
                eligible,
                key=lambda row: (-row["primary_wins_vs_id341"], row["mean_delta_vs_id341"], row["max_delta_vs_id341"]),
            )[0]
    return selected


def selected_usage_test_geometry(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    feature_cols: list[str],
    selected_configs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    geometry = selected_test_geometry(train, test, features_train, features_test, feature_cols, selected_configs)
    if geometry is None:
        return None
    full_train_sources = {}
    for target, config in selected_configs.items():
        source = predict_full_train_target(train, features_train, features_test, feature_cols, target, config["model"])
        full_train_sources[target] = {
            "model": config["model"],
            "source_mean": float(source.mean()),
            "source_std": float(source.std()),
            "source_min": float(source.min()),
            "source_max": float(source.max()),
        }
    geometry["usage_full_train_source_summary"] = full_train_sources
    return geometry


def build_report() -> dict[str, Any]:
    train, test = load_processed_frames()
    features_train, features_test, feature_coverage = build_usage_features(train, test)
    feature_cols = [col for col in features_train.columns if col.startswith("utm_")]
    splits_all = load_split_manifest()
    splits = {name: splits_all[name] for name in PRIMARY_GATE_SPLITS if name in splits_all}
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows, source_cache = scan_usage_configs(
        train, features_train, feature_cols, splits, reference178, reference341
    )
    selected_configs = select_usage_configs(scan_rows)
    assembled = evaluate_assembled_candidate(train, splits, reference178, reference341, source_cache, selected_configs)
    mean_delta = assembled["mean_primary_delta_vs_id341"]
    projected_proxy = float(ID341_TRANSFER_PROXY + mean_delta) if mean_delta is not None else ID341_TRANSFER_PROXY
    final_gate_pass = bool(
        selected_configs
        and assembled["primary_wins_vs_id341"] >= FINAL_PRIMARY_WINS_FLOOR
        and assembled["all_primary_splits_beat_id341"]
        and (assembled["max_primary_delta_vs_id341"] is not None and assembled["max_primary_delta_vs_id341"] <= 0.0)
        and projected_proxy < ID178_PUBLIC
    )
    public_058_reviewable = bool(final_gate_pass and projected_proxy <= PUBLIC_058_REVIEW_CUTOFF)
    geometry = selected_usage_test_geometry(train, test, features_train, features_test, feature_cols, selected_configs)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "phase3_usage_temporal_microstructure_smoke_post_id341",
        "purpose": (
            "Test a materially different raw source after block-A/block-B closure: coarse category/time-bin "
            "microstructure from mUsageStats without direct app-name columns."
        ),
        "raw_usage_path": str(RAW_USAGE_PATH.relative_to(ROOT)),
        "phase2_block_a_gap_report": str(PHASE2_BLOCK_A_PATH.relative_to(ROOT)),
        "phase2_block_b_report": str(PHASE2_BLOCK_B_PATH.relative_to(ROOT)),
        "feature_coverage": feature_coverage,
        "target_scan_focus": TARGET_SCAN_FOCUS,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS if spec.name in MODEL_NAMES],
        "blend_weights": BLEND_WEIGHTS,
        "selection_rule": {
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "final_primary_wins_floor": FINAL_PRIMARY_WINS_FLOOR,
            "final_projected_proxy_below_id178_public": ID178_PUBLIC,
        },
        "top_scan_by_target": {
            target: [row for row in scan_rows if row["target"] == target][:12] for target in TARGET_SCAN_FOCUS
        },
        "selected_configs": selected_configs,
        "assembled_candidate": assembled,
        "test_geometry": geometry,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "selected_target_count": int(len(selected_configs)),
            "selected_targets": list(selected_configs),
            "usage_temporal_has_config_signal": bool(selected_configs),
            "all_primary_splits_beat_id341": assembled["all_primary_splits_beat_id341"],
            "primary_wins_vs_id341": assembled["primary_wins_vs_id341"],
            "mean_primary_delta_vs_id341": mean_delta,
            "max_primary_delta_vs_id341": assembled["max_primary_delta_vs_id341"],
            "projected_public_proxy_from_id341": projected_proxy,
            "beats_id178_public_anchor_proxy": bool(projected_proxy < ID178_PUBLIC),
            "final_usage_temporal_gate_pass": final_gate_pass,
            "public_058_reviewable": public_058_reviewable,
            "next_action": (
                "If the usage-temporal gate fails, do not create ID342 from mUsageStats microstructure. If it finds "
                "a small target-specific signal, keep it report-only until it can combine with an independent source "
                "and beat ID178 proxy under the transfer gate."
            ),
        },
        "no_direct_app_name_model_columns": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    decision = report["decision"]
    lines = [
        "# Phase-3 usage temporal microstructure smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Raw rows / event rows: `{coverage['raw_rows']}` / `{coverage['event_rows']}`",
        f"- Unique app names observed but not used directly: `{coverage['unique_app_names']}`",
        f"- Daily usage feature rows/columns: `{coverage['daily_feature_rows']}` / `{coverage['usage_feature_columns']}`",
        f"- Train/test rows with any usage: `{coverage['train_rows_with_any_usage']}` / `{coverage['test_rows_with_any_usage']}`",
        f"- Train/test missing mean: `{coverage['train_missing_mean']:.4f}` / `{coverage['test_missing_mean']:.4f}`",
        f"- Category event counts: `{coverage['category_event_counts']}`",
        "",
        "## Selected configs",
        "",
    ]
    if report["selected_configs"]:
        lines.extend(["| target | config | wins | mean Δ | max Δ |", "|---|---|---:|---:|---:|"])
        for target, row in report["selected_configs"].items():
            lines.append(
                f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
    else:
        lines.append("No usage-temporal config passed the fixed eligibility rule.")
    lines.extend(
        [
            "",
            "## Assembled candidate by primary split",
            "",
            "| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for split_name, split_report in report["assembled_candidate"]["split_reports"].items():
        candidate = split_report["candidate"]
        id341 = split_report["reference_id341"]
        id178 = split_report["reference_id178"]
        lines.append(
            f"| `{split_name}` | {candidate['mean_loss']:.6f} | {id341['mean_loss']:.6f} | "
            f"{split_report['delta_vs_id341']:+.6f} | {id178['mean_loss']:.6f} | "
            f"{split_report['delta_vs_id178']:+.6f} | `{split_report['beats_id341']}` |"
        )
    lines.extend(["", "## Best scanned rows by target", ""])
    for target, rows in report["top_scan_by_target"].items():
        lines.extend([f"### `{target}`", "", "| config | wins | mean Δ | max Δ |", "|---|---:|---:|---:|"])
        for row in rows[:6]:
            lines.append(
                f"| `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    if report.get("test_geometry"):
        geometry = report["test_geometry"]
        lines.extend(
            [
                "## In-memory test geometry",
                "",
                f"- Mean absolute delta vs ID341 base test: `{geometry['vs_base']['id341']['mean_abs_delta_all_targets']:.6f}`",
                f"- Mean absolute delta vs ID178 base test: `{geometry['vs_base']['id178']['mean_abs_delta_all_targets']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- Selected targets: `{', '.join(decision['selected_targets']) if decision['selected_targets'] else 'none'}`",
            f"- Usage temporal has config signal: `{decision['usage_temporal_has_config_signal']}`",
            f"- All primary splits beat ID341: `{decision['all_primary_splits_beat_id341']}`",
            f"- Primary wins vs ID341: `{decision['primary_wins_vs_id341']}`",
            f"- Mean primary Δ vs ID341: `{decision['mean_primary_delta_vs_id341']:+.6f}`",
            f"- Max primary Δ vs ID341: `{decision['max_primary_delta_vs_id341']:+.6f}`",
            f"- Projected public proxy from ID341: `{decision['projected_public_proxy_from_id341']:.6f}`",
            f"- Beats ID178 public anchor proxy: `{decision['beats_id178_public_anchor_proxy']}`",
            f"- Final usage-temporal gate pass: `{decision['final_usage_temporal_gate_pass']}`",
            f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
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
