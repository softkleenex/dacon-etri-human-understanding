#!/usr/bin/env python3
"""Report-only dry run for circadian HR/light phase-shape feature generation.

This validates whether the circadian HR/light feature family can produce an
aligned, bounded, leak-safe daily feature frame in memory. It writes only
JSON/Markdown reports; it does not write a feature matrix/parquet, train a
model, create predictions, create a submission CSV, touch the submission log,
run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
SPEC_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_feature_spec_20260619.json"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "circadian_hr_light_phase_shape_v1"
TARGET_FOCUS = ["Q3", "S3", "S2"]
KEYS = ["subject_id", "lifelog_date", "split"]
RAW_SOURCE_PATHS = {
    "wHr": RAW_DIR / "ch2025_wHr.parquet",
    "wLight": RAW_DIR / "ch2025_wLight.parquet",
    "mLight": RAW_DIR / "ch2025_mLight.parquet",
    "wPedo": RAW_DIR / "ch2025_wPedo.parquet",
}

PHASE_BINS = {
    "night": list(range(0, 6)),
    "morning": list(range(6, 12)),
    "afternoon": list(range(12, 18)),
    "evening": list(range(18, 24)),
}
PHASE_CENTERS = {
    "night": 2.5,
    "morning": 8.5,
    "afternoon": 14.5,
    "evening": 20.5,
}

FIRST_DRYRUN_FEATURES = [
    "phase_strong_day_flag",
    "phase_weak_day_flag",
    "whr_rows",
    "whr_hours",
    "wlight_hours",
    "mlight_hours",
    "wpedo_hours",
    "phase_missing_source_count",
    "hr_night_median",
    "hr_morning_median",
    "hr_afternoon_median",
    "hr_evening_median",
    "hr_day_night_amplitude",
    "hr_morning_rise_slope",
    "hr_evening_decay_slope",
    "hr_curve_roughness",
    "hr_rest_window_low_quantile",
    "hr_high_load_minutes_proxy",
    "wlight_first_nonzero_hour",
    "wlight_last_nonzero_hour",
    "wlight_morning_exposure_sum",
    "wlight_afternoon_exposure_sum",
    "wlight_evening_exposure_sum",
    "wlight_night_exposure_sum",
    "wlight_day_night_contrast",
    "wlight_evening_delay_ratio",
    "mlight_dark_minutes_proxy",
    "mlight_longest_dark_gap_hours",
    "mlight_night_light_burst_count",
    "mlight_evening_light_share",
    "mlight_morning_light_share",
    "mlight_wearable_light_disagreement",
    "steps_morning_sum",
    "steps_afternoon_sum",
    "steps_evening_sum",
    "steps_night_sum",
    "first_active_hour",
    "last_active_hour",
    "activity_fragmentation_count",
    "late_activity_ratio",
    "rest_window_length_proxy",
    "activity_phase_centroid_hour",
    "hr_light_phase_lag_proxy",
    "activity_light_phase_lag_proxy",
    "night_light_with_high_hr_minutes_proxy",
    "evening_activity_with_bright_light_proxy",
    "morning_light_without_activity_proxy",
    "strong_phase_cross_modal_consistency_score",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": rel(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["exists"] = True
    data["path"] = rel(path)
    return data


def expected_days() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    expected = pd.concat([train, test], ignore_index=True)
    expected["lifelog_date"] = expected["lifelog_date"].astype(str)
    return expected.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"])
    out["lifelog_date"] = timestamps.dt.strftime("%Y-%m-%d")
    out["hour"] = timestamps.dt.hour
    out["timestamp_dt"] = timestamps
    out["phase_bin"] = out["hour"].map(hour_to_phase)
    return out


def hour_to_phase(hour: int) -> str:
    for name, hours in PHASE_BINS.items():
        if int(hour) in hours:
            return name
    return "unknown"


def finite_array(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def safe_quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(q))


def add_expected_alignment(expected: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    merged = expected.merge(daily, on=["subject_id", "lifelog_date"], how="left")
    numeric_cols = [col for col in merged.columns if col not in KEYS]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    return merged


def phase_pivot(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    agg: str = "sum",
    suffix: str = "",
) -> pd.DataFrame:
    pivot = (
        df.groupby(["subject_id", "lifelog_date", "phase_bin"])[value_col]
        .agg(agg)
        .unstack("phase_bin")
        .fillna(0)
        .reset_index()
    )
    rename = {phase: f"{prefix}_{phase}{suffix}" for phase in PHASE_BINS}
    return pivot.rename(columns=rename)


def phase_centroid_from_columns(row: pd.Series, columns: dict[str, str]) -> float:
    weights = []
    centers = []
    for phase, col in columns.items():
        value = float(row.get(col, 0.0))
        if value > 0:
            weights.append(value)
            centers.append(PHASE_CENTERS[phase])
    if not weights:
        return 0.0
    return float(np.average(np.asarray(centers), weights=np.asarray(weights)))


def w_hr_daily_features(expected: pd.DataFrame) -> pd.DataFrame:
    hr = pd.read_parquet(RAW_SOURCE_PATHS["wHr"], columns=["subject_id", "timestamp", "heart_rate"])
    hr = add_time_parts(hr)
    arrays = hr["heart_rate"].map(finite_array)
    hr["hr_count_row"] = arrays.map(len)
    hr["hr_mean_row"] = arrays.map(lambda arr: float(np.mean(arr)) if len(arr) else 0.0)
    hr["hr_p10_row"] = arrays.map(lambda arr: float(np.quantile(arr, 0.10)) if len(arr) else 0.0)
    hr["hr_p90_row"] = arrays.map(lambda arr: float(np.quantile(arr, 0.90)) if len(arr) else 0.0)

    daily = (
        hr.groupby(["subject_id", "lifelog_date"])
        .agg(
            whr_rows=("timestamp", "size"),
            whr_hours=("hour", "nunique"),
            hr_rest_window_low_quantile=("hr_p10_row", lambda s: safe_quantile(s, 0.10)),
            hr_high_load_minutes_proxy=("hr_p90_row", lambda s: int((s >= 100).sum())),
        )
        .reset_index()
    )
    bins = phase_pivot(hr, "hr_mean_row", "hr", agg="median", suffix="_median")
    daily = daily.merge(bins, on=["subject_id", "lifelog_date"], how="left")
    for col in ["hr_night_median", "hr_morning_median", "hr_afternoon_median", "hr_evening_median"]:
        daily[col] = daily[col].fillna(0)
    daytime = daily[["hr_morning_median", "hr_afternoon_median", "hr_evening_median"]].max(axis=1)
    daily["hr_day_night_amplitude"] = daytime.sub(daily["hr_night_median"]).clip(lower=0)
    daily["hr_morning_rise_slope"] = daily["hr_morning_median"].sub(daily["hr_night_median"])
    daily["hr_evening_decay_slope"] = daily["hr_evening_median"].sub(daily["hr_afternoon_median"])
    phase_values = daily[["hr_night_median", "hr_morning_median", "hr_afternoon_median", "hr_evening_median"]]
    daily["hr_curve_roughness"] = phase_values.diff(axis=1).abs().fillna(0).sum(axis=1)
    daily["hr_phase_centroid_hour_tmp"] = daily.apply(
        lambda row: phase_centroid_from_columns(
            row,
            {
                "night": "hr_night_median",
                "morning": "hr_morning_median",
                "afternoon": "hr_afternoon_median",
                "evening": "hr_evening_median",
            },
        ),
        axis=1,
    )
    keep = [
        "subject_id",
        "lifelog_date",
        "whr_rows",
        "whr_hours",
        "hr_night_median",
        "hr_morning_median",
        "hr_afternoon_median",
        "hr_evening_median",
        "hr_day_night_amplitude",
        "hr_morning_rise_slope",
        "hr_evening_decay_slope",
        "hr_curve_roughness",
        "hr_rest_window_low_quantile",
        "hr_high_load_minutes_proxy",
        "hr_phase_centroid_hour_tmp",
    ]
    return add_expected_alignment(expected, daily[keep])


def w_light_daily_features(expected: pd.DataFrame) -> pd.DataFrame:
    light = pd.read_parquet(RAW_SOURCE_PATHS["wLight"], columns=["subject_id", "timestamp", "w_light"])
    light = add_time_parts(light)
    light["w_light"] = pd.to_numeric(light["w_light"], errors="coerce").fillna(0).clip(lower=0)
    daily = (
        light.groupby(["subject_id", "lifelog_date"])
        .agg(wlight_hours=("hour", "nunique"), wlight_rows=("timestamp", "size"))
        .reset_index()
    )
    exposure = phase_pivot(light, "w_light", "wlight", agg="sum", suffix="_exposure_sum")
    daily = daily.merge(exposure, on=["subject_id", "lifelog_date"], how="left").fillna(0)
    nonzero = light.loc[light["w_light"] > 0]
    if nonzero.empty:
        first_last = pd.DataFrame(columns=["subject_id", "lifelog_date"])
    else:
        first_last = (
            nonzero.groupby(["subject_id", "lifelog_date"])
            .agg(wlight_first_nonzero_hour=("hour", "min"), wlight_last_nonzero_hour=("hour", "max"))
            .reset_index()
        )
    daily = daily.merge(first_last, on=["subject_id", "lifelog_date"], how="left")
    daily[["wlight_first_nonzero_hour", "wlight_last_nonzero_hour"]] = daily[
        ["wlight_first_nonzero_hour", "wlight_last_nonzero_hour"]
    ].fillna(0)
    day_sum = daily[
        ["wlight_morning_exposure_sum", "wlight_afternoon_exposure_sum", "wlight_evening_exposure_sum"]
    ].sum(axis=1)
    total = day_sum.add(daily["wlight_night_exposure_sum"])
    daily["wlight_day_night_contrast"] = day_sum.sub(daily["wlight_night_exposure_sum"]).div(total.add(1))
    daily["wlight_evening_delay_ratio"] = daily["wlight_evening_exposure_sum"].div(total.add(1))
    daily["wlight_phase_centroid_hour_tmp"] = daily.apply(
        lambda row: phase_centroid_from_columns(
            row,
            {
                "night": "wlight_night_exposure_sum",
                "morning": "wlight_morning_exposure_sum",
                "afternoon": "wlight_afternoon_exposure_sum",
                "evening": "wlight_evening_exposure_sum",
            },
        ),
        axis=1,
    )
    keep = [
        "subject_id",
        "lifelog_date",
        "wlight_hours",
        "wlight_first_nonzero_hour",
        "wlight_last_nonzero_hour",
        "wlight_morning_exposure_sum",
        "wlight_afternoon_exposure_sum",
        "wlight_evening_exposure_sum",
        "wlight_night_exposure_sum",
        "wlight_day_night_contrast",
        "wlight_evening_delay_ratio",
        "wlight_phase_centroid_hour_tmp",
    ]
    return add_expected_alignment(expected, daily[keep])


def longest_dark_gap_hours(group: pd.DataFrame) -> float:
    dark = group.loc[group["m_light"] <= 5].sort_values("timestamp_dt")
    if dark.empty:
        return 0.0
    diffs = dark["timestamp_dt"].diff().dt.total_seconds().div(60).fillna(10)
    run = 0.0
    best = 0.0
    for gap in diffs:
        if gap <= 20:
            run += 10.0
        else:
            best = max(best, run)
            run = 10.0
    best = max(best, run)
    return float(best / 60.0)


def m_light_daily_features(expected: pd.DataFrame) -> pd.DataFrame:
    light = pd.read_parquet(RAW_SOURCE_PATHS["mLight"], columns=["subject_id", "timestamp", "m_light"])
    light = add_time_parts(light)
    light["m_light"] = pd.to_numeric(light["m_light"], errors="coerce").fillna(0).clip(lower=0)
    light["dark_row"] = light["m_light"] <= 5
    light["night_light_burst"] = light["hour"].isin(PHASE_BINS["night"]) & (light["m_light"] > 20)
    daily = (
        light.groupby(["subject_id", "lifelog_date"])
        .agg(
            mlight_hours=("hour", "nunique"),
            mlight_rows=("timestamp", "size"),
            mlight_dark_minutes_proxy=("dark_row", lambda s: int(s.sum()) * 10),
            mlight_night_light_burst_count=("night_light_burst", "sum"),
        )
        .reset_index()
    )
    longest = (
        light.groupby(["subject_id", "lifelog_date"])
        .apply(longest_dark_gap_hours, include_groups=False)
        .rename("mlight_longest_dark_gap_hours")
        .reset_index()
    )
    daily = daily.merge(longest, on=["subject_id", "lifelog_date"], how="left")
    exposure = phase_pivot(light, "m_light", "mlight", agg="sum", suffix="_exposure_sum")
    daily = daily.merge(exposure, on=["subject_id", "lifelog_date"], how="left").fillna(0)
    total = daily[
        [
            "mlight_night_exposure_sum",
            "mlight_morning_exposure_sum",
            "mlight_afternoon_exposure_sum",
            "mlight_evening_exposure_sum",
        ]
    ].sum(axis=1)
    daily["mlight_evening_light_share"] = daily["mlight_evening_exposure_sum"].div(total.add(1))
    daily["mlight_morning_light_share"] = daily["mlight_morning_exposure_sum"].div(total.add(1))
    keep = [
        "subject_id",
        "lifelog_date",
        "mlight_hours",
        "mlight_dark_minutes_proxy",
        "mlight_longest_dark_gap_hours",
        "mlight_night_light_burst_count",
        "mlight_evening_light_share",
        "mlight_morning_light_share",
        "mlight_evening_exposure_sum",
        "mlight_morning_exposure_sum",
    ]
    return add_expected_alignment(expected, daily[keep])


def active_fragmentation_count(group: pd.DataFrame) -> int:
    active = group.sort_values("timestamp_dt")["active_row"].astype(bool).to_numpy()
    if active.size == 0:
        return 0
    starts = active & np.r_[True, ~active[:-1]]
    return int(starts.sum())


def active_centroid_hour(group: pd.DataFrame) -> float:
    weights = pd.to_numeric(group["step"], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = pd.to_numeric(group["speed"], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        return 0.0
    return float(np.average(group["hour"].to_numpy(dtype=float), weights=weights))


def w_pedo_daily_features(expected: pd.DataFrame) -> pd.DataFrame:
    pedo = pd.read_parquet(RAW_SOURCE_PATHS["wPedo"], columns=["subject_id", "timestamp", "step", "speed"])
    pedo = add_time_parts(pedo)
    pedo["step"] = pd.to_numeric(pedo["step"], errors="coerce").fillna(0).clip(lower=0)
    pedo["speed"] = pd.to_numeric(pedo["speed"], errors="coerce").fillna(0).clip(lower=0)
    pedo["active_row"] = (pedo["step"] > 0) | (pedo["speed"] > 0.1)
    daily = (
        pedo.groupby(["subject_id", "lifelog_date"])
        .agg(wpedo_hours=("hour", "nunique"), wpedo_rows=("timestamp", "size"))
        .reset_index()
    )
    steps = phase_pivot(pedo, "step", "steps", agg="sum", suffix="_sum")
    daily = daily.merge(steps, on=["subject_id", "lifelog_date"], how="left").fillna(0)
    active = pedo.loc[pedo["active_row"]]
    if active.empty:
        first_last = pd.DataFrame(columns=["subject_id", "lifelog_date"])
    else:
        first_last = (
            active.groupby(["subject_id", "lifelog_date"])
            .agg(first_active_hour=("hour", "min"), last_active_hour=("hour", "max"), active_hours=("hour", "nunique"))
            .reset_index()
        )
    daily = daily.merge(first_last, on=["subject_id", "lifelog_date"], how="left").fillna(0)
    frag = (
        pedo.groupby(["subject_id", "lifelog_date"])
        .apply(active_fragmentation_count, include_groups=False)
        .rename("activity_fragmentation_count")
        .reset_index()
    )
    centroid = (
        pedo.groupby(["subject_id", "lifelog_date"])
        .apply(active_centroid_hour, include_groups=False)
        .rename("activity_phase_centroid_hour")
        .reset_index()
    )
    daily = daily.merge(frag, on=["subject_id", "lifelog_date"], how="left")
    daily = daily.merge(centroid, on=["subject_id", "lifelog_date"], how="left")
    total_steps = daily[["steps_night_sum", "steps_morning_sum", "steps_afternoon_sum", "steps_evening_sum"]].sum(
        axis=1
    )
    daily["late_activity_ratio"] = daily["steps_evening_sum"].div(total_steps.add(1))
    daily["rest_window_length_proxy"] = 24 - daily["active_hours"].clip(lower=0, upper=24)
    keep = [
        "subject_id",
        "lifelog_date",
        "wpedo_hours",
        "steps_morning_sum",
        "steps_afternoon_sum",
        "steps_evening_sum",
        "steps_night_sum",
        "first_active_hour",
        "last_active_hour",
        "activity_fragmentation_count",
        "late_activity_ratio",
        "rest_window_length_proxy",
        "activity_phase_centroid_hour",
    ]
    return add_expected_alignment(expected, daily[keep])


def assemble_virtual_daily_frame(expected: pd.DataFrame) -> pd.DataFrame:
    frame = expected.copy()
    for daily in [
        w_hr_daily_features(expected),
        w_light_daily_features(expected),
        m_light_daily_features(expected),
        w_pedo_daily_features(expected),
    ]:
        cols = ["subject_id", "lifelog_date"] + [c for c in daily.columns if c not in KEYS]
        frame = frame.merge(daily[cols], on=["subject_id", "lifelog_date"], how="left")

    numeric_cols = [c for c in frame.columns if c not in KEYS]
    frame[numeric_cols] = frame[numeric_cols].fillna(0)
    frame["phase_strong_day_flag"] = (
        (frame["whr_rows"] >= 240)
        & (frame["whr_hours"] >= 8)
        & (frame["wlight_hours"] >= 18)
        & (frame["mlight_hours"] >= 18)
        & (frame["wpedo_hours"] >= 18)
    )
    frame["phase_weak_day_flag"] = (
        (frame["whr_rows"] >= 120)
        & (frame["whr_hours"] >= 6)
        & (frame["wlight_hours"] >= 12)
        & (frame["mlight_hours"] >= 12)
        & (frame["wpedo_hours"] >= 12)
    )
    source_present = [
        frame["whr_rows"] > 0,
        frame["wlight_hours"] > 0,
        frame["mlight_hours"] > 0,
        frame["wpedo_hours"] > 0,
    ]
    frame["phase_missing_source_count"] = 4 - sum(mask.astype(int) for mask in source_present)
    frame["mlight_wearable_light_disagreement"] = (
        frame["mlight_evening_light_share"].sub(frame["wlight_evening_delay_ratio"]).abs()
    )
    frame["hr_light_phase_lag_proxy"] = (
        frame["hr_phase_centroid_hour_tmp"].sub(frame["wlight_phase_centroid_hour_tmp"]).abs().div(24)
    )
    frame["activity_light_phase_lag_proxy"] = (
        frame["activity_phase_centroid_hour"].sub(frame["wlight_phase_centroid_hour_tmp"]).abs().div(24)
    )
    daily_hr_median = frame[
        ["hr_night_median", "hr_morning_median", "hr_afternoon_median", "hr_evening_median"]
    ].median(axis=1)
    frame["night_light_with_high_hr_minutes_proxy"] = np.where(
        frame["hr_night_median"] > daily_hr_median,
        np.log1p(frame["wlight_night_exposure_sum"].add(frame["mlight_night_light_burst_count"])),
        0.0,
    )
    frame["evening_activity_with_bright_light_proxy"] = np.log1p(frame["steps_evening_sum"]).mul(
        np.log1p(frame["wlight_evening_exposure_sum"])
    )
    frame["morning_light_without_activity_proxy"] = np.log1p(frame["wlight_morning_exposure_sum"]).div(
        np.log1p(frame["steps_morning_sum"]).add(1)
    )
    lag_sum = frame["hr_light_phase_lag_proxy"].add(frame["activity_light_phase_lag_proxy"])
    frame["strong_phase_cross_modal_consistency_score"] = (
        frame["phase_strong_day_flag"].astype(float).mul(1 / (1 + lag_sum))
    )
    for col in FIRST_DRYRUN_FEATURES:
        if col not in frame:
            frame[col] = 0.0
    frame = frame[KEYS + FIRST_DRYRUN_FEATURES]
    frame[FIRST_DRYRUN_FEATURES] = frame[FIRST_DRYRUN_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    return frame


def row_alignment(frame: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    expected_keys = expected[KEYS].sort_values(KEYS).reset_index(drop=True)
    frame_keys = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
    return {
        "expected_rows": int(len(expected_keys)),
        "dryrun_rows": int(len(frame_keys)),
        "exact_key_alignment": bool(expected_keys.equals(frame_keys)),
        "duplicate_keys": int(frame.duplicated(KEYS).sum()),
        "train_rows": int((frame["split"] == "train").sum()),
        "test_rows": int((frame["split"] == "test").sum()),
    }


def coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for split, split_df in frame.groupby("split"):
        strong = split_df["phase_strong_day_flag"].astype(bool)
        weak = split_df["phase_weak_day_flag"].astype(bool)
        out[str(split)] = {
            "rows": int(len(split_df)),
            "strong_phase_shape_rows": int(strong.sum()),
            "strong_phase_shape_rate": float(strong.mean()),
            "weak_phase_shape_rows": int(weak.sum()),
            "weak_phase_shape_rate": float(weak.mean()),
            "whr_covered_rate": float((split_df["whr_rows"] > 0).mean()),
            "wlight_covered_rate": float((split_df["wlight_hours"] > 0).mean()),
            "mlight_covered_rate": float((split_df["mlight_hours"] > 0).mean()),
            "wpedo_covered_rate": float((split_df["wpedo_hours"] > 0).mean()),
            "median_whr_rows": float(split_df["whr_rows"].median()),
            "median_wlight_hours": float(split_df["wlight_hours"].median()),
            "median_mlight_hours": float(split_df["mlight_hours"].median()),
            "median_wpedo_hours": float(split_df["wpedo_hours"].median()),
        }
    return {"by_split": out}


def feature_column_summary(frame: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [col for col in frame.columns if col not in KEYS]
    values = frame[feature_cols].to_numpy(dtype=float)
    return {
        "feature_column_count": len(feature_cols),
        "max_allowed_first_dryrun_columns": 56,
        "within_first_dryrun_bound": len(feature_cols) <= 56,
        "missing_declared_features": sorted(set(FIRST_DRYRUN_FEATURES).difference(feature_cols)),
        "extra_features": sorted(set(feature_cols).difference(FIRST_DRYRUN_FEATURES)),
        "all_values_finite": bool(np.isfinite(values).all()),
        "features": feature_cols,
    }


def value_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for col in FIRST_DRYRUN_FEATURES:
        series = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        ranges[col] = {
            "min": float(series.min()),
            "median": float(series.median()),
            "max": float(series.max()),
        }
    return ranges


def build_report() -> dict[str, Any]:
    spec = read_json(SPEC_REPORT)
    spec_dec = spec.get("decision", {}) if isinstance(spec.get("decision"), dict) else {}
    expected = expected_days()
    frame = assemble_virtual_daily_frame(expected)
    alignment = row_alignment(frame, expected)
    coverage = coverage_summary(frame)
    columns = feature_column_summary(frame)
    train = coverage["by_split"].get("train", {})
    test = coverage["by_split"].get("test", {})
    dryrun_passed = bool(
        spec.get("exists", False)
        and spec_dec.get("feature_spec_ready") is True
        and spec_dec.get("next_local_allowed_step") == "report_only_feature_generation_dryrun_no_matrix"
        and alignment["exact_key_alignment"]
        and alignment["duplicate_keys"] == 0
        and columns["within_first_dryrun_bound"]
        and not columns["missing_declared_features"]
        and not columns["extra_features"]
        and columns["all_values_finite"]
        and train.get("strong_phase_shape_rate", 0.0) >= 0.50
        and test.get("strong_phase_shape_rate", 0.0) >= 0.50
        and train.get("weak_phase_shape_rate", 0.0) >= 0.70
        and test.get("weak_phase_shape_rate", 0.0) >= 0.70
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_circadian_hr_light_phase_feature_generation_dryrun",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "inputs": {
            "feature_spec_report": {
                "path": rel(SPEC_REPORT),
                "exists": spec.get("exists", False),
                "feature_spec_ready": spec_dec.get("feature_spec_ready"),
                "next_local_allowed_step": spec_dec.get("next_local_allowed_step"),
            },
            "raw_sources": {name: rel(path) for name, path in RAW_SOURCE_PATHS.items()},
        },
        "row_alignment": alignment,
        "coverage": coverage,
        "feature_columns": columns,
        "value_ranges": value_ranges(frame),
        "safety_contract": {
            "feature_matrix_written": False,
            "feature_parquet_written": False,
            "predictions_created": False,
            "submission_csv_created": False,
            "submission_log_modified": False,
            "model_trained": False,
            "kaggle_run_executed": False,
            "dacon_submission_executed": False,
            "uses_target_labels": False,
            "uses_oof_predictions": False,
            "uses_public_scores": False,
        },
        "decision": {
            "dryrun_passed": dryrun_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "row_alignment_exact": alignment["exact_key_alignment"],
            "feature_column_bound_passed": columns["within_first_dryrun_bound"],
            "feature_column_count": columns["feature_column_count"],
            "train_strong_phase_shape_rate": train.get("strong_phase_shape_rate", 0.0),
            "test_strong_phase_shape_rate": test.get("strong_phase_shape_rate", 0.0),
            "train_weak_phase_shape_rate": train.get("weak_phase_shape_rate", 0.0),
            "test_weak_phase_shape_rate": test.get("weak_phase_shape_rate", 0.0),
            "next_local_allowed_step": "report_only_feature_matrix_smoke_contract",
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Circadian feature dry run passed. The next no-approval step may only be a separate "
                "report-only feature-matrix smoke contract; no matrix write, model, ID342, Kaggle run, "
                "or DACON submission is allowed."
                if dryrun_passed
                else "Fix the circadian feature dry-run gate before any matrix contract review."
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    align = report["row_alignment"]
    columns = report["feature_columns"]
    lines = [
        "# Circadian HR/light phase-shape feature-generation dry run — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, base "
            "prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model "
            "training job, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Dry run passed: `{dec['dryrun_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Row alignment exact: `{dec['row_alignment_exact']}`",
        f"- Feature column count: `{dec['feature_column_count']}`",
        f"- Feature column bound passed: `{dec['feature_column_bound_passed']}`",
        f"- Strong phase-shape train/test rates: `{dec['train_strong_phase_shape_rate']:.4f}` / "
        f"`{dec['test_strong_phase_shape_rate']:.4f}`",
        f"- Weak phase-shape train/test rates: `{dec['train_weak_phase_shape_rate']:.4f}` / "
        f"`{dec['test_weak_phase_shape_rate']:.4f}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Row alignment",
        "",
        "```json",
        json.dumps(align, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Coverage",
        "",
        "| split | rows | strong rows | strong rate | weak rows | weak rate | HR cov | wLight cov | mLight cov | pedo cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in report["coverage"]["by_split"].items():
        lines.append(
            f"| `{split}` | {row['rows']} | {row['strong_phase_shape_rows']} | "
            f"{row['strong_phase_shape_rate']:.4f} | {row['weak_phase_shape_rows']} | "
            f"{row['weak_phase_shape_rate']:.4f} | {row['whr_covered_rate']:.4f} | "
            f"{row['wlight_covered_rate']:.4f} | {row['mlight_covered_rate']:.4f} | "
            f"{row['wpedo_covered_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Feature columns",
            "",
            f"- Feature column count: `{columns['feature_column_count']}`",
            f"- Max allowed first dry-run columns: `{columns['max_allowed_first_dryrun_columns']}`",
            f"- Missing declared features: `{columns['missing_declared_features']}`",
            f"- Extra features: `{columns['extra_features']}`",
            f"- All values finite: `{columns['all_values_finite']}`",
            "",
            "## Safety contract",
            "",
            "```json",
            json.dumps(report["safety_contract"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Boundary",
            "",
            (
                "This dry run proves only that the predeclared circadian feature design can be aligned in memory. "
                "It is not leaderboard evidence, does not open ID342, and does not allow a feature parquet/model "
                "without a separate contract."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_circadian_hr_light_phase_feature_generation_dryrun_20260619.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    write_report(report, output)
    dec = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        json.dumps(
            {
                "dryrun_passed": dec["dryrun_passed"],
                "source_family": dec["source_family"],
                "feature_column_count": dec["feature_column_count"],
                "next_local_allowed_step": dec["next_local_allowed_step"],
                "feature_matrix_allowed_now": dec["feature_matrix_allowed_now"],
                "train_model_now": dec["train_model_now"],
                "create_submission_artifact": dec["create_submission_artifact"],
                "submit_now": dec["submit_now"],
                "push_kaggle_now": dec["push_kaggle_now"],
                "run_kaggle_now": dec["run_kaggle_now"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
