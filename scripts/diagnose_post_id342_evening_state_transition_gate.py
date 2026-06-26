#!/usr/bin/env python3
"""ID342-aware evening state-transition gate for Q3/S3.

This report-only diagnostic tests a compact, domain-driven source that is
different from the prior broad sequence-summary and sleep-boundary smokes. It
summarizes afternoon/evening/pre-sleep/night transitions, then adds
subject-level fold-internal deviations and previous-day label-free deltas.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw"
ITEMS_DIR = RAW_DIR / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_evening_state_transition_gate_20260624.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FOCUS_TARGETS = ["Q3", "S3"]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
ZONES: dict[str, tuple[float, float]] = {
    "afternoon": (12.0, 18.0),
    "evening": (18.0, 21.0),
    "presleep": (21.0, 24.0),
    "night_early": (24.0, 27.0),
    "sleep_tail": (27.0, 33.0),
}
ZONE_DELTA_PAIRS = [
    ("evening", "afternoon"),
    ("presleep", "evening"),
    ("night_early", "presleep"),
    ("sleep_tail", "presleep"),
]

EPS = 1e-6
ANCHOR_ID = 342
FAILED_FRONTIER_IDS = [287, 329]
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55

BLEND_WEIGHTS = (0.005, 0.01, 0.02, 0.03, 0.05)
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075
MAX_FAILED_DIRECTION_CORR = 0.80
MAX_FEATURE_COLUMNS = 520
MAX_SUBJECT_DEV_COLUMNS = 160
MAX_LAG_BASE_COLUMNS = 90
MAX_TRANSITION_DELTA_COLUMNS = 140

FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all_transition": (),
    "q3_arousal_usage": (
        "usage",
        "screen",
        "hr",
        "active",
        "vehicle",
        "unknown",
        "presleep",
        "evening",
        "delta_prev1",
        "dev_past3",
        "calendar",
    ),
    "s3_settling_sol": (
        "screen",
        "charge",
        "light",
        "dark",
        "bright",
        "still",
        "step",
        "distance",
        "moving",
        "night_early",
        "sleep_tail",
        "last",
        "first",
        "delta",
        "calendar",
    ),
}

WEB_RESEARCH_SOURCES = [
    {
        "title": "ETRI Lifelog Dataset 2024",
        "url": "https://arxiv.org/html/2508.03698v1",
        "read": (
            "ETRI defines Q3 as pre-sleep stress and S3 as sleep-onset-latency guideline adherence; "
            "the baseline section reports that task-specific time-zone features beat all-feature bundles."
        ),
    },
    {
        "title": "National Sleep Foundation - What Is Sleep Quality?",
        "url": "https://www.thensf.org/what-is-sleep-quality/",
        "read": (
            "NSF frames sleep quality around latency, awakenings, WASO, and efficiency, with adult latency "
            "screening at falling asleep within 30 minutes."
        ),
    },
    {
        "title": "National Sleep Foundation - Consistent Sleep Schedules",
        "url": "https://www.thensf.org/sleep-schedules-sleep-timing-guideline/",
        "read": (
            "NSF's sleep-timing guidance motivates regularity and previous-day deviation features rather than "
            "only same-day absolute sensor totals."
        ),
    },
    {
        "title": "Sleep Foundation - Electronics and Sleep",
        "url": "https://www.sleepfoundation.org/how-sleep-works/how-electronics-affect-sleep",
        "read": (
            "Pre-bed device light and alerting effects motivate screen/light/usage settling features for S3 "
            "and Q3, while keeping the gate strictly empirical."
        ),
    },
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_leaf_nodes: int | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_10", "lr", c_value=0.10),
    ModelSpec("lr_c0_30", "lr", c_value=0.30),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_array(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def entropy(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_test_sorted() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest["folds"]:
        splits.setdefault(row["split_name"], []).append(
            (
                np.asarray(row["train_idx"], dtype=int),
                np.asarray(row["valid_idx"], dtype=int),
            )
        )
    return splits


def find_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base OOF ID{candidate_id}")
    return paths[0]


def load_reference_values(train: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_oof(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with sorted train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def anchor_relative_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour
    out["anchor_date"] = out["timestamp"].dt.normalize()
    out.loc[hour < 12, "anchor_date"] = out.loc[hour < 12, "anchor_date"] - pd.Timedelta(days=1)
    out["lifelog_date"] = out["anchor_date"].dt.date
    out["rel_hour"] = hour.astype(float) + (hour < 12).astype(float) * 24.0
    out["rel_hour"] += out["timestamp"].dt.minute.astype(float) / 60.0
    return out[(out["rel_hour"] >= 12.0) & (out["rel_hour"] < 33.0)].copy()


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame if out is None else out.merge(frame, on=["subject_id", "lifelog_date"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["subject_id", "lifelog_date"])


def zone_aggregates(
    df: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    aggs: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date"])
    for zone_name, (start, end) in ZONES.items():
        part = df[(df["rel_hour"] >= start) & (df["rel_hour"] < end)]
        if part.empty:
            continue
        for agg in aggs:
            grouped = part.groupby(["subject_id", "lifelog_date"], sort=False)[value_cols].agg(agg)
            grouped.columns = [f"est_{prefix}_{zone_name}_{col}_{agg}" for col in grouped.columns]
            frames.append(grouped.reset_index())
        counts = part.groupby(["subject_id", "lifelog_date"], sort=False).size()
        frames.append(counts.rename(f"est_{prefix}_{zone_name}_row_count").reset_index())
    return merge_feature_frames(frames)


def event_hours(df: pd.DataFrame, condition_col: str, prefix: str) -> pd.DataFrame:
    active = df[df[condition_col] > 0]
    if active.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date"])
    keys = ["subject_id", "lifelog_date"]
    frames = [
        active.groupby(keys, sort=False)["rel_hour"]
        .max()
        .rename(f"est_{prefix}_last_{condition_col}_hour")
        .reset_index(),
        active.groupby(keys, sort=False)["rel_hour"]
        .min()
        .rename(f"est_{prefix}_first_{condition_col}_hour")
        .reset_index(),
    ]
    late = active[active["rel_hour"] >= 21.0]
    if not late.empty:
        frames.append(
            late.groupby(keys, sort=False)["rel_hour"]
            .min()
            .rename(f"est_{prefix}_first_{condition_col}_after21_hour")
            .reset_index()
        )
    return merge_feature_frames(frames)


def screen_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mScreenStatus.parquet")
    df = anchor_relative_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    df["screen_on"] = (pd.to_numeric(df["m_screen_use"], errors="coerce").fillna(0) > 0).astype(float)
    prev = df.groupby(["subject_id", "lifelog_date"], sort=False)["screen_on"].shift(1).fillna(0)
    df["screen_unlock"] = ((df["screen_on"] > 0) & (prev <= 0)).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(df, ["screen_on", "screen_unlock"], "screen", ("mean", "sum")),
            event_hours(df, "screen_on", "screen"),
            event_hours(df, "screen_unlock", "screen"),
        ]
    )
    return frame, {"source": "mScreenStatus", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def charging_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mACStatus.parquet")
    df = anchor_relative_window(raw)
    df["charging"] = (pd.to_numeric(df["m_charging"], errors="coerce").fillna(0) > 0).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(df, ["charging"], "charge", ("mean", "sum")),
            event_hours(df, "charging", "charge"),
        ]
    )
    return frame, {"source": "mACStatus", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def activity_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mActivity.parquet")
    df = anchor_relative_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    activity = pd.to_numeric(df["m_activity"], errors="coerce")
    df["activity_active"] = activity.isin([1, 2, 7, 8]).astype(float)
    df["activity_still"] = activity.eq(3).astype(float)
    df["activity_vehicle"] = activity.eq(0).astype(float)
    df["activity_unknown"] = activity.eq(4).astype(float)
    prev = activity.groupby([df["subject_id"], df["lifelog_date"]], sort=False).shift(1)
    df["activity_transition"] = activity.ne(prev).fillna(False).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(
                df,
                [
                    "activity_active",
                    "activity_still",
                    "activity_vehicle",
                    "activity_unknown",
                    "activity_transition",
                ],
                "activity",
                ("mean", "sum"),
            ),
            event_hours(df, "activity_active", "activity"),
            event_hours(df, "activity_still", "activity"),
        ]
    )
    return frame, {"source": "mActivity", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def pedo_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wPedo.parquet")
    df = anchor_relative_window(raw)
    for col in ["step", "step_frequency", "distance", "speed", "burned_calories"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["moving"] = (df["step"].fillna(0) > 0).astype(float)
    df["fast_motion"] = (df["speed"].fillna(0) > 1.0).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(
                df,
                ["step", "distance", "burned_calories", "moving", "fast_motion"],
                "pedo",
                ("sum", "mean"),
            ),
            zone_aggregates(df, ["speed", "step_frequency"], "pedo", ("mean", "max")),
            event_hours(df, "moving", "pedo"),
        ]
    )
    return frame, {"source": "wPedo", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def light_features(filename: str, value_col: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / filename)
    df = anchor_relative_window(raw)
    values = pd.to_numeric(df[value_col], errors="coerce")
    df[f"{prefix}_log1p"] = np.log1p(values.clip(lower=0))
    df[f"{prefix}_dark"] = (values.fillna(np.inf) < 10).astype(float)
    df[f"{prefix}_bright"] = (values.fillna(0) > 1000).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(df, [f"{prefix}_log1p", f"{prefix}_dark", f"{prefix}_bright"], prefix, ("mean", "max")),
            event_hours(df, f"{prefix}_dark", prefix),
            event_hours(df, f"{prefix}_bright", prefix),
        ]
    )
    return frame, {"source": prefix, "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def hr_row_stats(values: Any) -> tuple[float, float, float, float, float]:
    arr = safe_array(values)
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, 0.0
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.max(arr)),
        float(np.mean(arr >= 90.0)),
        float(arr.size),
    )


def heart_rate_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wHr.parquet")
    df = anchor_relative_window(raw)
    stats = pd.DataFrame(
        [hr_row_stats(value) for value in df["heart_rate"].to_numpy()],
        columns=["hr_mean", "hr_std", "hr_max", "hr_high90_rate", "hr_count"],
        index=df.index,
    )
    df = pd.concat([df, stats], axis=1)
    df["hr_high90"] = (df["hr_high90_rate"].fillna(0) > 0).astype(float)
    frame = merge_feature_frames(
        [
            zone_aggregates(df, ["hr_mean", "hr_std", "hr_max", "hr_high90_rate", "hr_count"], "hr", ("mean", "max")),
            event_hours(df, "hr_high90", "hr"),
        ]
    )
    return frame, {"source": "wHr", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def categorize_app(name: Any) -> str:
    text = str(name).lower()
    if any(token in text for token in ["kakao", "카카오", "talk", "message", "sms", "전화", "통화", "mail"]):
        return "communication"
    if any(token in text for token in ["naver", "google", "chrome", "youtube", "music", "webtoon", "browser"]):
        return "media_search"
    if any(token in text for token in ["toss", "pay", "bank", "card", "shop", "coupang", "마트", "쇼핑"]):
        return "finance_shop"
    if any(token in text for token in ["health", "walk", "calendar", "clock", "weather", "alarm", "캐시워크"]):
        return "routine_health"
    if any(token in text for token in ["one ui", "home", "launcher", "setting", "system", "홈"]):
        return "system_home"
    return "other"


def usage_row_stats(values: Any) -> dict[str, float]:
    categories = ["communication", "media_search", "finance_shop", "routine_health", "system_home", "other"]
    out = {f"usage_{category}_minutes": 0.0 for category in categories}
    out["usage_total_minutes"] = 0.0
    out["usage_app_count"] = 0.0
    if not isinstance(values, (list, tuple, np.ndarray)):
        out["usage_category_entropy"] = 0.0
        return out
    for entry in values:
        if not isinstance(entry, dict):
            continue
        minutes = float(pd.to_numeric(entry.get("total_time", 0.0), errors="coerce") or 0.0) / 60000.0
        category = categorize_app(entry.get("app_name", ""))
        out[f"usage_{category}_minutes"] += max(0.0, minutes)
        out["usage_total_minutes"] += max(0.0, minutes)
        out["usage_app_count"] += 1.0
    out["usage_category_entropy"] = entropy([out[f"usage_{category}_minutes"] for category in categories])
    return out


def usage_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mUsageStats.parquet")
    df = anchor_relative_window(raw)
    stats = pd.DataFrame([usage_row_stats(value) for value in df["m_usage_stats"].to_numpy()], index=df.index)
    df = pd.concat([df, stats], axis=1)
    df["usage_event"] = (df["usage_app_count"] > 0).astype(float)
    value_cols = [col for col in stats.columns if col.startswith("usage_")]
    frame = merge_feature_frames(
        [
            zone_aggregates(df, value_cols, "usage", ("sum", "mean")),
            event_hours(df, "usage_event", "usage"),
        ]
    )
    return frame, {"source": "mUsageStats", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def add_calendar_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["date_dt"] = pd.to_datetime(out["lifelog_date"])
    dow = out["date_dt"].dt.dayofweek.astype(float)
    month = out["date_dt"].dt.month.astype(float)
    out["est_calendar_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["est_calendar_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["est_calendar_is_weekend"] = dow.isin([5.0, 6.0]).astype(float)
    out["est_calendar_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["est_calendar_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out = out.sort_values(["subject_id", "date_dt"]).reset_index(drop=True)
    grouped = out.groupby("subject_id", sort=False)["date_dt"]
    first = grouped.transform("min")
    last = grouped.transform("max")
    span = (last - first).dt.days.clip(lower=1).astype(float)
    day_from_start = (out["date_dt"] - first).dt.days.astype(float)
    rank = out.groupby("subject_id", sort=False).cumcount().astype(float)
    count = out.groupby("subject_id", sort=False)["date_dt"].transform("count").clip(lower=2).astype(float)
    prev_date = grouped.shift(1)
    next_date = grouped.shift(-1)
    out["est_calendar_subject_day_progress"] = day_from_start / span
    out["est_calendar_subject_rank_pct"] = rank / (count - 1.0)
    out["est_calendar_log_days_since_prev"] = np.log1p((out["date_dt"] - prev_date).dt.days.fillna(999.0))
    out["est_calendar_log_days_to_next"] = np.log1p((next_date - out["date_dt"]).dt.days.fillna(999.0))
    return out.drop(columns=["date_dt"])


def add_zone_delta_features(features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in features.columns
        if col.startswith("est_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    added: dict[str, pd.Series] = {}
    for to_zone, from_zone in ZONE_DELTA_PAIRS:
        to_token = f"_{to_zone}_"
        from_token = f"_{from_zone}_"
        for col in numeric_cols:
            if len(added) >= MAX_TRANSITION_DELTA_COLUMNS:
                break
            if to_token not in col:
                continue
            counterpart = col.replace(to_token, from_token, 1)
            if counterpart in features.columns and pd.api.types.is_numeric_dtype(features[counterpart]):
                added[f"{col}_minus_{from_zone}"] = features[col] - features[counterpart]
        if len(added) >= MAX_TRANSITION_DELTA_COLUMNS:
            break
    if not added:
        return features
    return pd.concat([features, pd.DataFrame(added, index=features.index)], axis=1)


def rank_feature_columns(frame: pd.DataFrame, cols: list[str], limit: int) -> list[str]:
    rows: list[tuple[float, str]] = []
    for col in cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        coverage = float(series.notna().mean())
        variance = float(series.var(skipna=True)) if series.notna().sum() > 2 else 0.0
        keyword_bonus = 1.0
        if any(token in col for token in ["presleep", "night_early", "sleep_tail", "usage", "screen", "hr"]):
            keyword_bonus += 0.5
        if any(token in col for token in ["delta", "last", "first", "calendar"]):
            keyword_bonus += 0.25
        rows.append((coverage * np.log1p(abs(variance)) * keyword_bonus, col))
    rows.sort(reverse=True)
    return [col for _score, col in rows[:limit]]


def add_previous_day_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["date_dt_tmp"] = pd.to_datetime(out["lifelog_date"])
    out = out.sort_values(["subject_id", "date_dt_tmp"]).reset_index(drop=True)
    candidate_cols = [
        col
        for col in out.columns
        if col.startswith("est_") and pd.api.types.is_numeric_dtype(out[col])
    ]
    selected = rank_feature_columns(out, candidate_cols, MAX_LAG_BASE_COLUMNS)
    added: dict[str, pd.Series] = {}
    grouped = out.groupby("subject_id", sort=False)
    for col in selected:
        prev = grouped[col].shift(1)
        past3 = grouped[col].transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
        added[f"{col}_prev1"] = prev
        added[f"{col}_delta_prev1"] = out[col] - prev
        added[f"{col}_dev_past3"] = out[col] - past3
    out = pd.concat([out, pd.DataFrame(added, index=out.index)], axis=1)
    return out.drop(columns=["date_dt_tmp"])


def build_transition_feature_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    )
    keys = keys.rename(columns={"date": "lifelog_date"})

    builders = [
        screen_features,
        charging_features,
        activity_features,
        pedo_features,
        lambda: light_features("ch2025_mLight.parquet", "m_light", "m_light"),
        lambda: light_features("ch2025_wLight.parquet", "w_light", "w_light"),
        heart_rate_features,
        usage_features,
    ]
    frames: list[pd.DataFrame] = []
    source_metadata: list[dict[str, Any]] = []
    for builder in builders:
        frame, metadata = builder()
        frames.append(frame)
        metadata["feature_count"] = int(max(0, frame.shape[1] - 2))
        source_metadata.append(metadata)

    raw_features = merge_feature_frames(frames)
    features = keys.merge(raw_features, on=["subject_id", "lifelog_date"], how="left", validate="one_to_one")
    features = add_calendar_features(features)
    features = add_zone_delta_features(features)
    features = add_previous_day_features(features)

    numeric_cols = [
        col
        for col in features.columns
        if col.startswith("est_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    ranked = rank_feature_columns(features, numeric_cols, MAX_FEATURE_COLUMNS)
    keep_cols = ["subject_id", "lifelog_date", "is_test", *ranked]
    features = features[keep_cols].copy()
    metadata = {
        "source_metadata": source_metadata,
        "rows": int(len(features)),
        "train_rows": int((features["is_test"] == 0).sum()),
        "test_rows": int((features["is_test"] == 1).sum()),
        "numeric_feature_columns": int(len(ranked)),
        "mean_missing_rate": float(features[ranked].isna().mean().mean()) if ranked else None,
        "max_feature_columns": MAX_FEATURE_COLUMNS,
        "max_lag_base_columns": MAX_LAG_BASE_COLUMNS,
        "max_transition_delta_columns": MAX_TRANSITION_DELTA_COLUMNS,
    }
    return features, metadata


def family_feature_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for family, keywords in FAMILY_KEYWORDS.items():
        if family == "all_transition":
            families[family] = list(feature_cols)
        else:
            families[family] = [col for col in feature_cols if any(keyword in col for keyword in keywords)]
    return {family: cols for family, cols in families.items() if cols}


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=47),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.04,
                max_leaf_nodes=int(spec.max_leaf_nodes),
                l2_regularization=1.0,
                min_samples_leaf=12,
                random_state=47,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def fold_matrix(
    features: pd.DataFrame,
    feature_cols: list[str],
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = features.iloc[fit_idx].reset_index(drop=True)
    query = features.iloc[query_idx].reset_index(drop=True)
    usable_cols = [
        col
        for col in feature_cols
        if fit[col].notna().any() and float(pd.to_numeric(fit[col], errors="coerce").var(skipna=True) or 0.0) > 0.0
    ]
    dev_cols = rank_feature_columns(fit, usable_cols, min(MAX_SUBJECT_DEV_COLUMNS, len(usable_cols)))
    fit_raw = fit[usable_cols].reset_index(drop=True)
    query_raw = query[usable_cols].reset_index(drop=True)
    fit_devs: dict[str, pd.Series] = {}
    query_devs: dict[str, pd.Series] = {}
    for col in dev_cols:
        global_median = pd.to_numeric(fit[col], errors="coerce").median()
        if pd.isna(global_median):
            global_median = 0.0
        subject_medians = fit.groupby("subject_id", sort=False)[col].median()
        fit_base = fit["subject_id"].map(subject_medians).fillna(global_median).reset_index(drop=True)
        query_base = query["subject_id"].map(subject_medians).fillna(global_median).reset_index(drop=True)
        fit_devs[f"{col}_dev_subject_fold"] = fit[col].reset_index(drop=True) - fit_base
        query_devs[f"{col}_dev_subject_fold"] = query[col].reset_index(drop=True) - query_base
    return pd.concat([fit_raw, pd.DataFrame(fit_devs)], axis=1), pd.concat(
        [query_raw, pd.DataFrame(query_devs)], axis=1
    )


def predict_target_for_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, int]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = 0
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
            model = make_model(spec)
            model.fit(fit_x, y[fit_idx])
            proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
        covered += len(valid_idx)
    return pred, covered


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_split_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def direction_metrics(
    blended: np.ndarray,
    anchor: np.ndarray,
    failed_refs: dict[int, np.ndarray],
    target_idx: int,
) -> dict[str, Any]:
    candidate_delta = blended - anchor[:, target_idx]
    rows: dict[str, Any] = {
        "mean_abs_movement_vs_id342": float(np.mean(np.abs(candidate_delta))),
        "max_abs_movement_vs_id342": float(np.max(np.abs(candidate_delta))),
        "failed_direction_corr": {},
        "failed_direction_sign_agreement": {},
    }
    corrs: list[float] = []
    for candidate_id, values in failed_refs.items():
        failed_delta = values[:, target_idx] - anchor[:, target_idx]
        corr = safe_corr(candidate_delta, failed_delta)
        key = f"id{candidate_id}"
        rows["failed_direction_corr"][key] = corr
        if corr is not None:
            corrs.append(corr)
        active = (np.abs(candidate_delta) > 1e-9) & (np.abs(failed_delta) > 1e-9)
        agreement = float((np.sign(candidate_delta[active]) == np.sign(failed_delta[active])).mean()) if active.any() else None
        rows["failed_direction_sign_agreement"][key] = agreement
    rows["max_failed_direction_corr"] = float(max(corrs)) if corrs else None
    return rows


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
                split_source_predictions[split_name] = np.where(
                    np.isnan(pred),
                    anchor_values[:, target_idx],
                    pred,
                )
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
                    blended = np.clip((1.0 - weight) * anchor_values[:, target_idx] + weight * source_pred, EPS, 1 - EPS)
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
                        "config": f"{target}_id342_evening_state_{family}_{spec.name}_w{weight:g}",
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
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    partial_rows = [
        row
        for row in rows
        if row["mean_delta"] < 0 and row["wins"] >= 3 and row["max_delta"] <= 0.003
    ][:5]
    return {
        "target": target,
        "best": rows[0] if rows else None,
        "eligible_count": int(len(eligible_rows)),
        "selected": eligible_rows[0] if eligible_rows else None,
        "partial_signal_rows": partial_rows,
        "top_rows": rows[:10],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    feature_frame, feature_metadata = build_transition_feature_frame(train, test)
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
        set(selected_targets) == set(FOCUS_TARGETS)
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
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "all_focus_targets_selected": set(selected_targets) == set(FOCUS_TARGETS),
        "target_delta_sum_vs_id342": target_delta_sum,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "max_selected_target_delta_vs_id342": max_selected_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "partial_signal_targets": partial_signal_targets,
        "evening_state_transition_has_q3s3_signal": bool(selected_targets),
        "final_evening_state_transition_gate_pass": final_gate_pass,
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": bool(final_gate_pass),
        "run_kaggle_now": bool(final_gate_pass),
        "submit_now": False,
        "next_action": (
            "If this gate fails, close lightweight evening state-transition features as another ID342-aware "
            "local lane. If both Q3 and S3 pass, use Kaggle no-submit only for parity review before any artifact."
        ),
    }

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_evening_state_transition_gate",
        "purpose": (
            "Test whether fold-internal evening arousal/settling transitions and previous-day deviations add "
            "Q3/S3 signal beyond ID342."
        ),
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "research_sources": WEB_RESEARCH_SOURCES,
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
                "Q3 is pre-sleep stress and S3 is sleep-onset-latency adherence. This gate uses pre-sleep "
                "screen/app/HR activity and settling into darkness/charging/stillness, with previous-day "
                "label-free deviations and fold-internal subject baselines."
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
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    metadata = report["feature_contract"]["feature_metadata"]
    lines = [
        "# Post-ID342 evening state-transition gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Domain read",
        "",
        f"- {report['feature_contract']['domain_read']}",
        "- Web-researched sources:",
    ]
    for source in report["research_sources"]:
        lines.append(f"  - [{source['title']}]({source['url']}): {source['read']}")
    lines.extend(
        [
            "",
            "## Feature contract",
            "",
            f"- Rows: `{metadata['rows']}` (`{metadata['train_rows']}` train, `{metadata['test_rows']}` test)",
            f"- Numeric feature columns used: `{metadata['numeric_feature_columns']}`",
            f"- Mean missing rate: `{metadata['mean_missing_rate']}`",
            "",
            "| family | feature count | examples |",
            "|---|---:|---|",
        ]
    )
    for family, family_row in report["feature_contract"]["feature_families"].items():
        examples = ", ".join(f"`{col}`" for col in family_row["example_features"][:5])
        lines.append(f"| `{family}` | {family_row['feature_count']} | {examples} |")

    lines.extend(["", "## Target scan", "", "| target | selected | eligible | best config | wins | mean delta | max delta |"])
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
            f"- Final gate pass: `{decision['final_evening_state_transition_gate_pass']}`",
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
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(
        "Selected targets: "
        f"{decision['selected_targets']} | projected_public_proxy={decision['projected_public_proxy_from_id342']:.9f} "
        f"| final_gate_pass={decision['final_evening_state_transition_gate_pass']}"
    )


if __name__ == "__main__":
    main()
