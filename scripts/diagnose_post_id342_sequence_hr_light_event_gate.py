#!/usr/bin/env python3
"""Post-ID342 sequence HR/light/screen event gate for S4/S3.

This report-only diagnostic follows the raw sleep-event feasibility audit. The
public raw schema has no direct SOL/WASO labels, so this gate reconstructs
event-like proxies from overnight HR, light, screen, charging, activity, and
step sequences without materializing a feature parquet or prediction artifact.

It creates no prediction CSV, feature matrix artifact, base prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
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

from scripts.diagnose_post_id342_evening_state_transition_gate import (  # noqa: E402
    ANCHOR_ID,
    BLEND_WEIGHTS,
    EPS,
    FAILED_FRONTIER_IDS,
    ID342_PUBLIC,
    ITEMS_DIR,
    MAX_FAILED_DIRECTION_CORR,
    MODEL_SPECS,
    PRIMARY_SPLITS,
    PUBLIC_055,
    TARGETS,
    add_calendar_features,
    add_previous_day_features,
    direction_metrics,
    load_reference_values,
    load_split_manifest,
    load_test_sorted,
    load_train_sorted,
    merge_feature_frames,
    predict_target_for_split,
    rank_feature_columns,
    rel,
    safe_array,
    weighted_split_loss,
)
from scripts.diagnose_post_id342_processed_sleep_proxy_gate import (  # noqa: E402
    TARGET_MAX_DELTA_CEILING,
    TARGET_MEAN_DELTA_CEILING,
    TARGET_WIN_FLOOR,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_sequence_hr_light_event_gate_20260625.json"

FOCUS_TARGETS = ["S4", "S3"]
WINDOW_START = 18.0
WINDOW_END = 36.0
MAX_EVENT_FEATURE_COLUMNS = 640

EVENT_ZONES: dict[str, tuple[float, float]] = {
    "evening": (18.0, 21.0),
    "prebed": (21.0, 24.0),
    "early_sleep": (24.0, 27.0),
    "sleep_core": (27.0, 33.0),
    "morning_tail": (33.0, 36.0),
}

TARGET_DOMAIN = {
    "S4": "objective WASO / wakefulness after sleep onset guideline",
    "S3": "objective sleep onset latency guideline",
}

FEATURE_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all_sequence_events": (),
    "s4_waso_fragmentation": (
        "s4",
        "waso",
        "awake",
        "wake",
        "sleep_core",
        "morning_tail",
        "interruption",
        "spike",
        "bright",
        "screen_on",
        "moving",
        "activity_active",
        "after_low",
        "hr_rel_low",
        "hr_spike",
    ),
    "s3_onset_settling": (
        "s3",
        "onset",
        "prebed",
        "early_sleep",
        "screen_off",
        "dark",
        "charging",
        "rest",
        "still",
        "first_sustained",
        "last_true",
        "drop",
        "settle",
    ),
    "physio_light_events": ("hr", "wlight", "mlight", "dark", "bright", "low", "spike", "drop"),
    "screen_charge_inactivity": (
        "screen",
        "charging",
        "activity",
        "pedo",
        "rest",
        "still",
        "moving",
        "screen_off",
    ),
}


def anchor_sleep_event_window(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw timestamps to the evening-to-noon lifelog date window."""
    out = raw.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour.astype(float)
    minute = out["timestamp"].dt.minute.astype(float)
    second = out["timestamp"].dt.second.astype(float)
    rel_hour = hour + minute / 60.0 + second / 3600.0
    anchor_date = out["timestamp"].dt.normalize()
    before_window_start = rel_hour < WINDOW_START
    anchor_date.loc[before_window_start] = anchor_date.loc[before_window_start] - pd.Timedelta(days=1)
    out["lifelog_date"] = anchor_date.dt.date
    out["rel_hour"] = rel_hour + before_window_start.astype(float) * 24.0
    return out[(out["rel_hour"] >= WINDOW_START) & (out["rel_hour"] < WINDOW_END)].copy()


def zone_aggregates(
    df: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    aggs: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if df.empty or not value_cols:
        return pd.DataFrame(columns=["subject_id", "lifelog_date"])
    for zone_name, (start, end) in EVENT_ZONES.items():
        part = df[(df["rel_hour"] >= start) & (df["rel_hour"] < end)]
        if part.empty:
            continue
        for agg in aggs:
            grouped = part.groupby(["subject_id", "lifelog_date"], sort=False)[value_cols].agg(agg)
            grouped.columns = [f"est_seq_{prefix}_{zone_name}_{col}_{agg}" for col in grouped.columns]
            frames.append(grouped.reset_index())
        counts = part.groupby(["subject_id", "lifelog_date"], sort=False).size()
        frames.append(counts.rename(f"est_seq_{prefix}_{zone_name}_row_count").reset_index())
    return merge_feature_frames(frames)


def binary_run_summary(
    df: pd.DataFrame,
    condition_col: str,
    prefix: str,
    *,
    after: float = 21.0,
    before: float = 33.0,
    min_samples: int = 10,
) -> pd.DataFrame:
    keys = ["subject_id", "lifelog_date"]
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=keys)
    for key, group in df.sort_values(keys + ["rel_hour"]).groupby(keys, sort=False):
        part = group[(group["rel_hour"] >= after) & (group["rel_hour"] < before)]
        row: dict[str, Any] = {
            "subject_id": key[0],
            "lifelog_date": key[1],
            f"est_seq_{prefix}_{condition_col}_window_rows": int(len(part)),
        }
        if part.empty:
            rows.append(row)
            continue
        hours = part["rel_hour"].to_numpy(dtype=float)
        mask = pd.to_numeric(part[condition_col], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        true_hours = hours[mask]
        row[f"est_seq_{prefix}_{condition_col}_rate"] = float(mask.mean())
        row[f"est_seq_{prefix}_{condition_col}_first_true_hour"] = float(true_hours[0]) if true_hours.size else np.nan
        row[f"est_seq_{prefix}_{condition_col}_last_true_hour"] = float(true_hours[-1]) if true_hours.size else np.nan
        row[f"est_seq_{prefix}_{condition_col}_run_count"] = 0
        row[f"est_seq_{prefix}_{condition_col}_longest_run_minutes"] = 0.0
        row[f"est_seq_{prefix}_{condition_col}_first_sustained_hour"] = np.nan

        run_count = 0
        current_len = 0
        current_start = np.nan
        longest_len = 0
        first_sustained = np.nan
        previous_hour = np.nan
        for hour, active in zip(hours, mask, strict=False):
            gap_break = np.isfinite(previous_hour) and (hour - previous_hour) > (3.0 / 60.0)
            if active and (current_len == 0 or not gap_break):
                if current_len == 0:
                    current_start = hour
                    run_count += 1
                current_len += 1
            elif active:
                if current_len > longest_len:
                    longest_len = current_len
                current_start = hour
                current_len = 1
                run_count += 1
            else:
                if current_len > longest_len:
                    longest_len = current_len
                current_len = 0
                current_start = np.nan
            if current_len >= min_samples and not np.isfinite(first_sustained):
                first_sustained = current_start
            previous_hour = hour
        if current_len > longest_len:
            longest_len = current_len
        row[f"est_seq_{prefix}_{condition_col}_run_count"] = int(run_count)
        row[f"est_seq_{prefix}_{condition_col}_longest_run_minutes"] = float(longest_len)
        row[f"est_seq_{prefix}_{condition_col}_first_sustained_hour"] = float(first_sustained)
        rows.append(row)
    return pd.DataFrame(rows)


def scalar_group_frame(df: pd.DataFrame, prefix: str, summarizer) -> pd.DataFrame:
    rows = []
    keys = ["subject_id", "lifelog_date"]
    for key, group in df.sort_values(keys + ["rel_hour"]).groupby(keys, sort=False):
        row = {"subject_id": key[0], "lifelog_date": key[1]}
        row.update({f"est_seq_{prefix}_{name}": value for name, value in summarizer(group).items()})
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=keys)


def heart_rate_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wHr.parquet")
    df = anchor_sleep_event_window(raw)
    stats = pd.DataFrame(
        [hr_row_stats(value) for value in df["heart_rate"].to_numpy()],
        columns=[
            "hr_mean",
            "hr_min",
            "hr_max",
            "hr_std",
            "hr_low60_rate",
            "hr_high90_rate",
            "hr_count",
        ],
        index=df.index,
    )
    df = pd.concat([df, stats], axis=1)
    keys = ["subject_id", "lifelog_date"]
    key_index = pd.MultiIndex.from_frame(df[keys])
    evening_baseline = df[df["rel_hour"].between(18.0, 22.0, inclusive="left")].groupby(keys)["hr_mean"].median()
    overall_baseline = df.groupby(keys)["hr_mean"].median()
    baseline = pd.Series(key_index.map(evening_baseline), index=df.index, dtype="float64")
    fallback = pd.Series(key_index.map(overall_baseline), index=df.index, dtype="float64")
    df["hr_evening_baseline"] = baseline.fillna(fallback)
    df["hr_drop_from_evening"] = df["hr_evening_baseline"] - df["hr_mean"]
    df["hr_rel_low"] = ((df["hr_drop_from_evening"] >= 8.0) | (df["hr_mean"] <= 60.0)).astype(float)
    df["hr_spike"] = ((df["hr_mean"] >= df["hr_evening_baseline"] + 8.0) | (df["hr_high90_rate"] > 0.0)).astype(float)

    def summarize(group: pd.DataFrame) -> dict[str, float]:
        night = group[(group["rel_hour"] >= 23.0) & (group["rel_hour"] < 33.0)]
        early = group[(group["rel_hour"] >= 21.0) & (group["rel_hour"] < 26.0)]
        if night.empty:
            night = group
        baseline_value = float(pd.to_numeric(group["hr_evening_baseline"], errors="coerce").median())
        min_idx = pd.to_numeric(night["hr_mean"], errors="coerce").idxmin() if night["hr_mean"].notna().any() else None
        min_hour = float(night.loc[min_idx, "rel_hour"]) if min_idx is not None else np.nan
        low_after_23 = night["hr_rel_low"].fillna(0).to_numpy(dtype=float) > 0
        spikes_after_low = 0.0
        if np.isfinite(min_hour):
            after_low = night[night["rel_hour"] > min_hour]
            spikes_after_low = float((after_low["hr_spike"].fillna(0) > 0).sum())
        first_low_early = early.loc[early["hr_rel_low"] > 0, "rel_hour"]
        return {
            "evening_baseline": baseline_value,
            "night_min": float(pd.to_numeric(night["hr_mean"], errors="coerce").min()),
            "night_q10": float(pd.to_numeric(night["hr_mean"], errors="coerce").quantile(0.10)),
            "night_q90": float(pd.to_numeric(night["hr_mean"], errors="coerce").quantile(0.90)),
            "night_std_mean": float(pd.to_numeric(night["hr_mean"], errors="coerce").std()),
            "drop_to_night_min": baseline_value - float(pd.to_numeric(night["hr_mean"], errors="coerce").min()),
            "night_min_hour": min_hour,
            "low_rate_after23": float(low_after_23.mean()) if low_after_23.size else np.nan,
            "spike_after_low_count": spikes_after_low,
            "first_low_early_hour": float(first_low_early.min()) if not first_low_early.empty else np.nan,
        }

    frame = merge_feature_frames(
        [
            zone_aggregates(
                df,
                [
                    "hr_mean",
                    "hr_min",
                    "hr_max",
                    "hr_std",
                    "hr_low60_rate",
                    "hr_high90_rate",
                    "hr_count",
                    "hr_drop_from_evening",
                ],
                "hr",
                ("mean", "min", "max"),
            ),
            binary_run_summary(df, "hr_rel_low", "hr", after=21.0, before=33.0, min_samples=10),
            binary_run_summary(df, "hr_spike", "hr", after=23.0, before=34.0, min_samples=3),
            scalar_group_frame(df, "hr", summarize),
        ]
    )
    return frame, {"source": "wHr", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def hr_row_stats(values: Any) -> tuple[float, float, float, float, float, float, float]:
    arr = safe_array(values)
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0
    return (
        float(np.mean(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.std(arr)),
        float(np.mean(arr <= 60.0)),
        float(np.mean(arr >= 90.0)),
        float(arr.size),
    )


def light_features(filename: str, value_col: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / filename)
    df = anchor_sleep_event_window(raw)
    values = pd.to_numeric(df[value_col], errors="coerce")
    df[f"{prefix}_log1p"] = np.log1p(values.clip(lower=0))
    df[f"{prefix}_dark"] = (values.fillna(np.inf) < 10.0).astype(float)
    df[f"{prefix}_dim"] = (values.fillna(np.inf) < 50.0).astype(float)
    df[f"{prefix}_bright"] = (values.fillna(0.0) > 500.0).astype(float)

    def summarize(group: pd.DataFrame) -> dict[str, float]:
        night = group[(group["rel_hour"] >= 23.0) & (group["rel_hour"] < 33.0)]
        if night.empty:
            night = group
        dark = night[f"{prefix}_dark"].fillna(0) > 0
        bright = night[f"{prefix}_bright"].fillna(0) > 0
        first_dark = night.loc[dark, "rel_hour"]
        first_dark_hour = float(first_dark.min()) if not first_dark.empty else np.nan
        after_dark = night[night["rel_hour"] > first_dark_hour] if np.isfinite(first_dark_hour) else night.iloc[0:0]
        return {
            "night_dark_rate": float(dark.mean()) if len(dark) else np.nan,
            "night_bright_rate": float(bright.mean()) if len(bright) else np.nan,
            "first_dark_after23_hour": first_dark_hour,
            "bright_after_first_dark_count": float((after_dark[f"{prefix}_bright"].fillna(0) > 0).sum()),
            "night_log1p_q90": float(pd.to_numeric(night[f"{prefix}_log1p"], errors="coerce").quantile(0.90)),
        }

    frame = merge_feature_frames(
        [
            zone_aggregates(
                df,
                [f"{prefix}_log1p", f"{prefix}_dark", f"{prefix}_dim", f"{prefix}_bright"],
                prefix,
                ("mean", "max"),
            ),
            binary_run_summary(df, f"{prefix}_dark", prefix, after=21.0, before=33.0, min_samples=30),
            binary_run_summary(df, f"{prefix}_bright", prefix, after=23.0, before=34.0, min_samples=2),
            scalar_group_frame(df, prefix, summarize),
        ]
    )
    return frame, {"source": prefix, "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def screen_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mScreenStatus.parquet")
    df = anchor_sleep_event_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    df["screen_on"] = (pd.to_numeric(df["m_screen_use"], errors="coerce").fillna(0) > 0).astype(float)
    df["screen_off"] = 1.0 - df["screen_on"]
    prev = df.groupby(["subject_id", "lifelog_date"], sort=False)["screen_on"].shift(1).fillna(0)
    df["screen_unlock"] = ((df["screen_on"] > 0) & (prev <= 0)).astype(float)

    def summarize(group: pd.DataFrame) -> dict[str, float]:
        prebed = group[(group["rel_hour"] >= 21.0) & (group["rel_hour"] < 26.0)]
        core = group[(group["rel_hour"] >= 24.0) & (group["rel_hour"] < 33.0)]
        screen_on = prebed.loc[prebed["screen_on"] > 0, "rel_hour"]
        core_on = core["screen_on"].fillna(0) > 0
        return {
            "last_screen_on_pre26_hour": float(screen_on.max()) if not screen_on.empty else np.nan,
            "pre26_screen_on_rate": float((prebed["screen_on"].fillna(0) > 0).mean()) if len(prebed) else np.nan,
            "core_screen_on_rate": float(core_on.mean()) if len(core_on) else np.nan,
            "core_screen_unlock_count": float((core["screen_unlock"].fillna(0) > 0).sum()),
        }

    frame = merge_feature_frames(
        [
            zone_aggregates(df, ["screen_on", "screen_off", "screen_unlock"], "screen", ("mean", "sum")),
            binary_run_summary(df, "screen_off", "screen", after=21.0, before=33.0, min_samples=45),
            binary_run_summary(df, "screen_on", "screen", after=23.0, before=34.0, min_samples=2),
            scalar_group_frame(df, "screen", summarize),
        ]
    )
    return frame, {"source": "mScreenStatus", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def charging_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mACStatus.parquet")
    df = anchor_sleep_event_window(raw)
    df["charging"] = (pd.to_numeric(df["m_charging"], errors="coerce").fillna(0) > 0).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(df, ["charging"], "charge", ("mean", "sum")),
            binary_run_summary(df, "charging", "charge", after=21.0, before=33.0, min_samples=30),
        ]
    )
    return frame, {"source": "mACStatus", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def activity_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mActivity.parquet")
    df = anchor_sleep_event_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    activity = pd.to_numeric(df["m_activity"], errors="coerce")
    df["activity_active"] = activity.isin([1, 2, 7, 8]).astype(float)
    df["activity_still"] = activity.eq(3).astype(float)
    df["activity_inactive"] = activity.isin([3, 4]).astype(float)
    prev = activity.groupby([df["subject_id"], df["lifelog_date"]], sort=False).shift(1)
    df["activity_transition"] = activity.ne(prev).fillna(False).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(
                df,
                ["activity_active", "activity_still", "activity_inactive", "activity_transition"],
                "activity",
                ("mean", "sum"),
            ),
            binary_run_summary(df, "activity_inactive", "activity", after=21.0, before=33.0, min_samples=30),
            binary_run_summary(df, "activity_active", "activity", after=23.0, before=34.0, min_samples=3),
        ]
    )
    return frame, {"source": "mActivity", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def pedo_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wPedo.parquet")
    df = anchor_sleep_event_window(raw)
    for col in ["step", "step_frequency", "distance", "speed", "burned_calories"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["moving"] = (df["step"].fillna(0) > 0).astype(float)
    df["rest"] = 1.0 - df["moving"]
    frame = merge_feature_frames(
        [
            zone_aggregates(
                df,
                ["step", "distance", "speed", "burned_calories", "moving", "rest"],
                "pedo",
                ("mean", "sum", "max"),
            ),
            binary_run_summary(df, "rest", "pedo", after=21.0, before=33.0, min_samples=30),
            binary_run_summary(df, "moving", "pedo", after=23.0, before=34.0, min_samples=2),
        ]
    )
    return frame, {"source": "wPedo", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def add_cross_event_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    pairs = {
        "est_seq_cross_hr_low_after_screen_off": (
            "est_seq_hr_hr_rel_low_first_sustained_hour",
            "est_seq_screen_screen_off_first_sustained_hour",
        ),
        "est_seq_cross_dark_after_screen_off": (
            "est_seq_wlight_w_light_dark_first_sustained_hour",
            "est_seq_screen_screen_off_first_sustained_hour",
        ),
        "est_seq_cross_charge_after_screen_off": (
            "est_seq_charge_charging_first_sustained_hour",
            "est_seq_screen_screen_off_first_sustained_hour",
        ),
        "est_seq_cross_rest_after_screen_off": (
            "est_seq_pedo_rest_first_sustained_hour",
            "est_seq_screen_screen_off_first_sustained_hour",
        ),
        "est_seq_cross_hr_low_after_dark": (
            "est_seq_hr_hr_rel_low_first_sustained_hour",
            "est_seq_wlight_w_light_dark_first_sustained_hour",
        ),
        "est_seq_cross_activity_rest_after_dark": (
            "est_seq_activity_activity_inactive_first_sustained_hour",
            "est_seq_wlight_w_light_dark_first_sustained_hour",
        ),
    }
    added: dict[str, pd.Series] = {}
    for name, (left, right) in pairs.items():
        if left in out.columns and right in out.columns:
            added[name] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
            added[f"{name}_abs"] = added[name].abs()
    interruption_pairs = {
        "est_seq_cross_screen_on_after_hr_low_rate": (
            "est_seq_screen_screen_on_rate",
            "est_seq_hr_hr_rel_low_rate",
        ),
        "est_seq_cross_bright_after_dark_pressure": (
            "est_seq_wlight_bright_after_first_dark_count",
            "est_seq_wlight_night_dark_rate",
        ),
        "est_seq_cross_spike_after_low_pressure": (
            "est_seq_hr_spike_after_low_count",
            "est_seq_hr_low_rate_after23",
        ),
    }
    for name, (left, right) in interruption_pairs.items():
        if left in out.columns and right in out.columns:
            added[name] = pd.to_numeric(out[left], errors="coerce") * pd.to_numeric(out[right], errors="coerce")
    if added:
        out = pd.concat([out, pd.DataFrame(added, index=out.index)], axis=1)
    return out


def build_sequence_event_feature_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    ).rename(columns={"date": "lifelog_date"})

    builders = [
        heart_rate_features,
        lambda: light_features("ch2025_wLight.parquet", "w_light", "wlight"),
        lambda: light_features("ch2025_mLight.parquet", "m_light", "mlight"),
        screen_features,
        charging_features,
        activity_features,
        pedo_features,
    ]
    frames: list[pd.DataFrame] = []
    source_metadata: list[dict[str, Any]] = []
    for builder in builders:
        frame, metadata = builder()
        metadata["feature_count"] = int(max(0, frame.shape[1] - 2))
        frames.append(frame)
        source_metadata.append(metadata)

    raw_features = merge_feature_frames(frames)
    features = keys.merge(raw_features, on=["subject_id", "lifelog_date"], how="left", validate="one_to_one")
    features = add_cross_event_features(features)
    features = add_calendar_features(features)
    features = add_previous_day_features(features)
    numeric_cols = [
        col
        for col in features.columns
        if col.startswith("est_") and pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()
    ]
    ranked = rank_feature_columns(features, numeric_cols, min(MAX_EVENT_FEATURE_COLUMNS, len(numeric_cols)))
    keep_cols = ["subject_id", "lifelog_date", "is_test", *ranked]
    features = features[keep_cols].copy()
    metadata = {
        "window": {"start_rel_hour": WINDOW_START, "end_rel_hour": WINDOW_END},
        "source_metadata": source_metadata,
        "rows": int(len(features)),
        "train_rows": int((features["is_test"] == 0).sum()),
        "test_rows": int((features["is_test"] == 1).sum()),
        "numeric_feature_columns_before_rank": int(len(numeric_cols)),
        "numeric_feature_columns_used": int(len(ranked)),
        "mean_missing_rate": float(features[ranked].isna().mean().mean()) if ranked else None,
        "max_event_feature_columns": MAX_EVENT_FEATURE_COLUMNS,
    }
    return features, metadata


def family_feature_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for family, keywords in FEATURE_FAMILY_KEYWORDS.items():
        if not keywords:
            families[family] = list(feature_cols)
        else:
            families[family] = [col for col in feature_cols if any(keyword in col for keyword in keywords)]
    return {family: cols for family, cols in families.items() if cols}


def scan_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_families: dict[str, list[str]],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    failed_refs: dict[int, np.ndarray],
    target: str,
) -> dict[str, Any]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for family, feature_cols in feature_families.items():
        for spec in MODEL_SPECS:
            split_source_predictions: dict[str, np.ndarray] = {}
            split_valid_counts: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_target_for_split(features, feature_cols, y, splits[split_name], spec)
                split_source_predictions[split_name] = np.where(np.isnan(pred), anchor_values[:, target_idx], pred)
                split_valid_counts[split_name] = int(covered)
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                split_direction_metrics: dict[str, Any] = {}
                wins = 0
                failed_corrs: list[float] = []
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    source_pred = split_source_predictions[split_name]
                    blended = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_pred, EPS, 1 - EPS
                    )
                    anchor_loss = weighted_split_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_split_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_anchor_losses[split_name] = anchor_loss
                    split_losses[split_name] = candidate_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                    metrics = direction_metrics(blended, anchor_values, failed_refs, target_idx)
                    split_direction_metrics[split_name] = metrics
                    corr = metrics.get("max_failed_direction_corr")
                    if corr is not None:
                        failed_corrs.append(float(corr))
                max_failed_corr = float(max(failed_corrs)) if failed_corrs else None
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                eligible = bool(
                    wins >= TARGET_WIN_FLOOR
                    and mean_delta <= TARGET_MEAN_DELTA_CEILING
                    and max_delta <= TARGET_MAX_DELTA_CEILING
                    and (max_failed_corr is None or max_failed_corr <= MAX_FAILED_DIRECTION_CORR)
                )
                rows.append(
                    {
                        "target": target,
                        "target_domain": TARGET_DOMAIN[target],
                        "config": f"{target.lower()}_id342_sequence_hr_light_event_{family}_{spec.name}_w{weight:g}",
                        "feature_family": family,
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_counts": split_valid_counts,
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "split_deltas": split_deltas,
                        "split_losses": split_losses,
                        "split_anchor_losses": split_anchor_losses,
                        "max_failed_direction_corr": max_failed_corr,
                        "direction_metrics": split_direction_metrics,
                        "eligible": eligible,
                    }
                )
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    partial_rows = [row for row in rows if row["mean_delta"] < 0 and row["wins"] >= 3 and row["max_delta"] <= 0.003][:5]
    return {
        "target": target,
        "domain": TARGET_DOMAIN[target],
        "best": rows[0] if rows else None,
        "eligible_count": int(len(eligible_rows)),
        "selected": eligible_rows[0] if eligible_rows else None,
        "partial_signal_rows": partial_rows,
        "top_rows": rows[:10],
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    feature_frame, feature_metadata = build_sequence_event_feature_frame(train, test)
    features_train = feature_frame[feature_frame["is_test"].eq(0)].reset_index(drop=True)
    feature_cols = [
        col
        for col in features_train.columns
        if col.startswith("est_") and pd.api.types.is_numeric_dtype(features_train[col])
    ]
    feature_families = family_feature_columns(feature_cols)
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    anchor_values = load_reference_values(train, ANCHOR_ID)
    failed_refs = {candidate_id: load_reference_values(train, candidate_id) for candidate_id in FAILED_FRONTIER_IDS}

    target_reports = {
        target: scan_target(
            train=train,
            features=features_train,
            feature_families=feature_families,
            splits=splits,
            anchor_values=anchor_values,
            failed_refs=failed_refs,
            target=target,
        )
        for target in FOCUS_TARGETS
    }
    selected_rows = {
        target: target_report["selected"]
        for target, target_report in target_reports.items()
        if target_report.get("selected") is not None
    }
    selected_targets = list(selected_rows)
    target_delta_sum = float(sum(float(row["mean_delta"]) for row in selected_rows.values()))
    estimated_all_target_delta = target_delta_sum / len(TARGETS)
    max_selected_delta = float(max([float(row["max_delta"]) for row in selected_rows.values()], default=0.0))
    projected_public_proxy = ID342_PUBLIC + estimated_all_target_delta
    final_gate_pass = bool(
        bool(selected_targets)
        and estimated_all_target_delta <= -0.00050
        and max_selected_delta <= TARGET_MAX_DELTA_CEILING
        and projected_public_proxy < ID342_PUBLIC
    )
    partial_signal_targets = [
        target
        for target, target_report in target_reports.items()
        if target_report.get("partial_signal_rows") and target not in selected_targets
    ]

    decision = {
        "selected_targets": selected_targets,
        "s4_selected": "S4" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "all_focus_targets_selected": set(selected_targets) == set(FOCUS_TARGETS),
        "target_delta_sum_vs_id342": target_delta_sum,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "max_selected_target_delta_vs_id342": max_selected_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "partial_signal_targets": partial_signal_targets,
        "sequence_hr_light_event_has_s4s3_signal": bool(selected_targets),
        "final_sequence_hr_light_event_gate_pass": final_gate_pass,
        "public_055_evidence": False,
        "create_feature_matrix": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": bool(final_gate_pass),
        "run_kaggle_now": bool(final_gate_pass),
        "submit_now": False,
        "next_action": (
            "If this gate fails, close local raw HR/light/screen event reconstruction as an immediate ID342 "
            "overlay route. If it passes but remains far above Public 0.55, keep it report-only and pivot to "
            "validation/public-gap redesign before any artifact."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_sequence_hr_light_event_gate",
        "purpose": (
            "Test whether sequence-level HR/light/screen/charging/activity/step event proxies add stable S4/S3 "
            "signal beyond ID342 after direct raw sleep-event reconstruction proved unavailable."
        ),
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": list(BLEND_WEIGHTS),
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "max_failed_direction_corr": MAX_FAILED_DIRECTION_CORR,
            "failed_frontier_ids": FAILED_FRONTIER_IDS,
        },
        "feature_contract": {
            "domain_read": (
                "S3 approximates sleep-onset latency via settling into no-screen, darkness, charging, stillness, "
                "and HR drop. S4 approximates WASO via post-onset HR spikes, light/screen/movement interruptions, "
                "and fragmented low-HR/dark/still runs. All features are label-free raw-sequence summaries."
            ),
            "feature_metadata": feature_metadata,
            "feature_families": {
                family: {"feature_count": int(len(cols)), "example_features": cols[:12]}
                for family, cols in feature_families.items()
            },
        },
        "target_reports": target_reports,
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": not bool(final_gate_pass),
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    metadata = report["feature_contract"]["feature_metadata"]
    lines = [
        "# Post-ID342 sequence HR/light event gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Domain Read",
        "",
        f"- {report['feature_contract']['domain_read']}",
        "",
        "## Feature Contract",
        "",
        f"- Window: `{metadata['window']['start_rel_hour']}` to `{metadata['window']['end_rel_hour']}` relative hours",
        f"- Rows: `{metadata['rows']}` (`{metadata['train_rows']}` train, `{metadata['test_rows']}` test)",
        f"- Numeric feature columns before rank: `{metadata['numeric_feature_columns_before_rank']}`",
        f"- Numeric feature columns used: `{metadata['numeric_feature_columns_used']}`",
        f"- Mean missing rate: `{metadata['mean_missing_rate']}`",
        "",
        "| source | raw rows | window rows | feature count |",
        "|---|---:|---:|---:|",
    ]
    for source in metadata["source_metadata"]:
        lines.append(
            f"| `{source['source']}` | `{source['raw_rows']}` | `{source['window_rows']}` | "
            f"`{source['feature_count']}` |"
        )
    lines.extend(["", "## Families", "", "| family | feature count | examples |", "|---|---:|---|"])
    for family, family_row in report["feature_contract"]["feature_families"].items():
        examples = ", ".join(f"`{col}`" for col in family_row["example_features"][:5])
        lines.append(f"| `{family}` | {family_row['feature_count']} | {examples} |")

    lines.extend(
        ["", "## Target Scan", "", "| target | selected | eligible | best config | wins | mean delta | max delta |"]
    )
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for target, target_report in report["target_reports"].items():
        best = target_report.get("best") or {}
        selected = target_report.get("selected")
        lines.append(
            f"| `{target}` | `{selected is not None}` | {target_report.get('eligible_count', 0)} | "
            f"`{best.get('config')}` | {best.get('wins')} | {float(best.get('mean_delta', 0.0)):.9f} | "
            f"{float(best.get('max_delta', 0.0)):.9f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Estimated all-target delta vs ID342: `{decision['estimated_all_target_delta_vs_id342']:.9f}`",
            f"- Projected Public proxy: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55: `{decision['gap_to_public_055_after_proxy']:.9f}`",
            f"- Final gate pass: `{decision['final_sequence_hr_light_event_gate_pass']}`",
            f"- Kaggle now: `{decision['run_kaggle_now']}`",
            f"- DACON submit now: `{decision['submit_now']}`",
            f"- Next action: {decision['next_action']}",
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
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Selected targets: "
        f"{decision['selected_targets']} | projected_public_proxy="
        f"{decision['projected_public_proxy_from_id342']:.9f} | final_gate_pass="
        f"{decision['final_sequence_hr_light_event_gate_pass']}"
    )


if __name__ == "__main__":
    main()
