#!/usr/bin/env python3
"""ID342-aware raw temporal Q3/S3 sequence-image smoke.

This diagnostic builds a compact, label-free temporal representation from raw
phone/watch streams. Timestamps from 00:00 to 11:59 are assigned to the previous
lifelog date, so a row for date D summarizes D 12:00 through D+1 09:00. The
result is evaluated only in memory against ID342 OOF predictions.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
ITEMS_DIR = RAW_DIR / "ch2025_data_items"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
RESEARCH_PRIORITY_PATH = REPORTS_DIR / "post_id342_research_gap_and_source_priority_20260624.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_raw_temporal_q3s3_sequence_image_smoke_20260624.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FOCUS_TARGETS = ["Q3", "S3"]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
HOURS = list(range(12, 33))
EPS = 1e-6
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_leaf_nodes: int | None = None
    l2_regularization: float | None = None
    k_neighbors: int | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_02", "lr", c_value=0.02),
    ModelSpec("lr_c0_08", "lr", c_value=0.08),
    ModelSpec("lr_c0_20", "lr", c_value=0.20),
    ModelSpec("ssknn_k1", "ss_knn", k_neighbors=1),
    ModelSpec("ssknn_k3", "ss_knn", k_neighbors=3),
    ModelSpec("ssknn_k7", "ss_knn", k_neighbors=7),
)
BLEND_WEIGHTS = (0.005, 0.01, 0.02, 0.05)
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075
MAX_FEATURE_COLUMNS = 420
MAX_SUBJECT_DEV_COLUMNS = 120


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


def safe_len(value: Any) -> float:
    return float(len(value)) if isinstance(value, (list, tuple, np.ndarray)) else 0.0


def entropy_from_values(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    return float(-(probs * np.log(probs + 1e-9)).sum())


def read_train() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    train["sleep_date"] = pd.to_datetime(train["sleep_date"]).dt.date
    return train.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)


def read_test_keys() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["lifelog_date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    test["sleep_date"] = pd.to_datetime(test["sleep_date"]).dt.date
    return test.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)


def anchor_relative_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour
    out["anchor_date"] = out["timestamp"].dt.normalize()
    out.loc[hour < 12, "anchor_date"] = out.loc[hour < 12, "anchor_date"] - pd.Timedelta(days=1)
    out["lifelog_date"] = out["anchor_date"].dt.date
    out["rel_hour"] = hour.astype(float) + (hour < 12).astype(float) * 24.0
    out["rel_hour"] += out["timestamp"].dt.minute / 60.0
    out["hour_bin"] = np.floor(out["rel_hour"]).astype(int)
    return out[out["hour_bin"].isin(HOURS)].copy()


def pivot_hourly(df: pd.DataFrame, value_cols: list[str], prefix: str, agg: str = "mean") -> pd.DataFrame:
    if df.empty or not value_cols:
        return pd.DataFrame(columns=["subject_id", "lifelog_date"])
    if agg == "mean":
        grouped = df.groupby(["subject_id", "lifelog_date", "hour_bin"], sort=False)[value_cols].mean()
    elif agg == "sum":
        grouped = df.groupby(["subject_id", "lifelog_date", "hour_bin"], sort=False)[value_cols].sum()
    elif agg == "max":
        grouped = df.groupby(["subject_id", "lifelog_date", "hour_bin"], sort=False)[value_cols].max()
    else:
        raise ValueError(f"Unknown aggregation: {agg}")
    wide = grouped.unstack("hour_bin")
    wide.columns = [f"rseq_{prefix}_h{int(hour):02d}_{col}_{agg}" for col, hour in wide.columns]
    return wide.reset_index()


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame if out is None else out.merge(frame, on=["subject_id", "lifelog_date"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["subject_id", "lifelog_date"])


def add_hour_differences(frame: pd.DataFrame, max_groups: int = 80) -> pd.DataFrame:
    by_stem: dict[str, dict[int, str]] = {}
    for col in frame.columns:
        if not col.startswith("rseq_"):
            continue
        marker = "_h"
        if marker not in col:
            continue
        left, right = col.split(marker, 1)
        hour_text = right[:2]
        suffix = right[2:]
        if not hour_text.isdigit():
            continue
        stem = f"{left}{suffix}"
        by_stem.setdefault(stem, {})[int(hour_text)] = col
    added = 0
    diff_columns: dict[str, pd.Series] = {}
    for stem, hour_cols in sorted(by_stem.items()):
        if added >= max_groups:
            break
        usable = [hour for hour in HOURS if hour in hour_cols and hour + 1 in hour_cols]
        if not usable:
            continue
        for hour in usable:
            diff_columns[f"{stem}_diff_h{hour:02d}_to_h{hour + 1:02d}"] = (
                frame[hour_cols[hour + 1]] - frame[hour_cols[hour]]
            )
        added += 1
    if not diff_columns:
        return frame.copy()
    return pd.concat([frame, pd.DataFrame(diff_columns, index=frame.index)], axis=1)


def scalar_source_features(filename: str, value_col: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / filename)
    df = anchor_relative_window(raw)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[f"{prefix}_log1p"] = np.log1p(df[value_col].clip(lower=0))
    df[f"{prefix}_dark"] = (df[value_col] < 10).astype(float)
    df[f"{prefix}_bright"] = (df[value_col] > 1000).astype(float)
    frame = merge_frames(
        [
            pivot_hourly(df, [f"{prefix}_log1p", f"{prefix}_dark", f"{prefix}_bright"], prefix, "mean"),
            pivot_hourly(df, [f"{prefix}_log1p"], prefix, "max"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def screen_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mScreenStatus.parquet")
    df = anchor_relative_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    df["screen_on"] = (df["m_screen_use"] > 0).astype(float)
    prev = df.groupby(["subject_id", "lifelog_date"], sort=False)["screen_on"].shift(1)
    df["screen_unlock"] = ((df["screen_on"] > 0) & (prev.fillna(0) <= 0)).astype(float)
    frame = merge_frames(
        [
            pivot_hourly(df, ["screen_on"], "screen", "mean"),
            pivot_hourly(df, ["screen_on", "screen_unlock"], "screen", "sum"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def ac_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mACStatus.parquet")
    df = anchor_relative_window(raw)
    df["charging"] = (df["m_charging"] > 0).astype(float)
    frame = merge_frames(
        [
            pivot_hourly(df, ["charging"], "charge", "mean"),
            pivot_hourly(df, ["charging"], "charge", "sum"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def activity_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mActivity.parquet")
    df = anchor_relative_window(raw)
    df["activity_active"] = df["m_activity"].isin([1, 2, 7, 8]).astype(float)
    df["activity_still"] = (df["m_activity"] == 3).astype(float)
    df["activity_vehicle"] = (df["m_activity"] == 0).astype(float)
    df["activity_unknown"] = (df["m_activity"] == 4).astype(float)
    frame = merge_frames(
        [
            pivot_hourly(
                df,
                ["activity_active", "activity_still", "activity_vehicle", "activity_unknown"],
                "activity",
                "mean",
            ),
            pivot_hourly(df, ["activity_active"], "activity", "sum"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def pedo_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wPedo.parquet")
    df = anchor_relative_window(raw)
    for col in ["step", "distance", "speed", "burned_calories", "walking_step", "running_step"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["step_active"] = (df["step"] > 0).astype(float)
    frame = merge_frames(
        [
            pivot_hourly(
                df,
                ["step", "distance", "burned_calories", "walking_step", "running_step", "step_active"],
                "pedo",
                "sum",
            ),
            pivot_hourly(df, ["speed", "step_active"], "pedo", "mean"),
            pivot_hourly(df, ["speed"], "pedo", "max"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def hr_row_stats(value: Any) -> tuple[float, float, float, float, float]:
    arr = safe_array(value)
    arr = arr[(arr > 0) & (arr < 250)]
    if arr.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    return (
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float((arr > 90).mean()),
    )


def hr_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wHr.parquet")
    df = anchor_relative_window(raw)
    stats = pd.DataFrame(
        [hr_row_stats(value) for value in df["heart_rate"]],
        columns=["hr_mean", "hr_std", "hr_min", "hr_max", "hr_high90"],
        index=df.index,
    )
    df = pd.concat([df, stats], axis=1)
    frame = merge_frames(
        [
            pivot_hourly(df, ["hr_mean", "hr_std", "hr_high90"], "hr", "mean"),
            pivot_hourly(df, ["hr_min"], "hr", "min" if False else "mean"),
            pivot_hourly(df, ["hr_max"], "hr", "max"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def list_payload_features(filename: str, object_col: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / filename)
    df = anchor_relative_window(raw)
    df[f"{prefix}_payload_count"] = df[object_col].map(safe_len)
    frame = merge_frames(
        [
            pivot_hourly(df, [f"{prefix}_payload_count"], prefix, "mean"),
            pivot_hourly(df, [f"{prefix}_payload_count"], prefix, "max"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def gps_row_stats(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return (0.0, math.nan, math.nan, math.nan)
    speeds: list[float] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            speed = float(item.get("speed", math.nan))
        except (TypeError, ValueError):
            speed = math.nan
        if math.isfinite(speed):
            speeds.append(speed)
    if not speeds:
        return (float(len(value)), math.nan, math.nan, math.nan)
    arr = np.asarray(speeds, dtype=float)
    return (float(len(value)), float(arr.mean()), float(arr.max()), float((arr > 1.0).mean()))


def gps_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mGps.parquet")
    df = anchor_relative_window(raw)
    stats = pd.DataFrame(
        [gps_row_stats(value) for value in df["m_gps"]],
        columns=["gps_points", "gps_speed_mean", "gps_speed_max", "gps_moving_share"],
        index=df.index,
    )
    df = pd.concat([df, stats], axis=1)
    frame = merge_frames(
        [
            pivot_hourly(df, ["gps_points", "gps_speed_mean", "gps_moving_share"], "gps", "mean"),
            pivot_hourly(df, ["gps_speed_max"], "gps", "max"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


APP_CATEGORIES = ["communication", "media_search", "finance_shop", "routine_health", "system_home", "other"]


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def app_category(app_name: Any) -> str:
    text = normalize_text(app_name)
    if any(token in text for token in ["kakao", "message", "sms", "call", "phone", "mail", "talk"]):
        return "communication"
    if any(token in text for token in ["youtube", "netflix", "naver", "chrome", "search", "music", "video"]):
        return "media_search"
    if any(token in text for token in ["bank", "pay", "toss", "shop", "store", "cash"]):
        return "finance_shop"
    if any(token in text for token in ["health", "fit", "calendar", "clock", "weather", "maps"]):
        return "routine_health"
    if any(token in text for token in ["home", "system", "settings", "one ui", "launcher"]):
        return "system_home"
    return "other"


def usage_row_stats(value: Any) -> dict[str, float]:
    out = {f"usage_{category}_time": 0.0 for category in APP_CATEGORIES}
    out.update({"usage_total_time": 0.0, "usage_app_count": 0.0, "usage_entropy": 0.0})
    if not isinstance(value, (list, tuple, np.ndarray)):
        return out
    app_times: list[float] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            total = float(item.get("total_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            total = 0.0
        category = app_category(item.get("app_name", ""))
        out[f"usage_{category}_time"] += total
        out["usage_total_time"] += total
        if total > 0:
            out["usage_app_count"] += 1.0
            app_times.append(total)
    out["usage_entropy"] = entropy_from_values(app_times)
    return out


def usage_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mUsageStats.parquet")
    df = anchor_relative_window(raw)
    rows = [usage_row_stats(value) for value in df["m_usage_stats"]]
    stats = pd.DataFrame(rows, index=df.index)
    for col in stats.columns:
        if col.endswith("_time") or col == "usage_total_time":
            stats[col] = np.log1p(stats[col].clip(lower=0) / 60000.0)
    df = pd.concat([df, stats], axis=1)
    value_cols = list(stats.columns)
    frame = merge_frames(
        [
            pivot_hourly(df, value_cols, "usage", "sum"),
            pivot_hourly(df, ["usage_app_count", "usage_entropy"], "usage", "mean"),
        ]
    )
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


AMBIENCE_CATEGORIES = ["speech", "music", "vehicle", "outdoor", "quiet", "other"]


def ambience_category(label: Any) -> str:
    text = normalize_text(label)
    if "speech" in text or "conversation" in text:
        return "speech"
    if "music" in text:
        return "music"
    if "vehicle" in text or "car" in text or "truck" in text or "bus" in text:
        return "vehicle"
    if "outside" in text or "outdoor" in text:
        return "outdoor"
    if "silence" in text or "quiet" in text:
        return "quiet"
    return "other"


def ambience_row_stats(value: Any) -> dict[str, float]:
    out = {f"amb_{category}_score": 0.0 for category in AMBIENCE_CATEGORIES}
    out["amb_entropy"] = 0.0
    if not isinstance(value, (list, tuple, np.ndarray)):
        return out
    scores: list[float] = []
    for item in value:
        if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 2:
            continue
        category = ambience_category(item[0])
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            score = 0.0
        out[f"amb_{category}_score"] = max(out[f"amb_{category}_score"], score)
        if score > 0:
            scores.append(score)
    out["amb_entropy"] = entropy_from_values(scores)
    return out


def ambience_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mAmbience.parquet")
    df = anchor_relative_window(raw)
    rows = [ambience_row_stats(value) for value in df["m_ambience"]]
    stats = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, stats], axis=1)
    frame = pivot_hourly(df, list(stats.columns), "ambience", "mean")
    return frame, {"raw_rows": int(len(raw)), "window_rows": int(len(df)), "features": int(max(0, frame.shape[1] - 2))}


def build_raw_temporal_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    builders = {
        "screen": screen_features,
        "charge": ac_features,
        "activity": activity_features,
        "pedo": pedo_features,
        "hr": hr_features,
        "mlight": lambda: scalar_source_features("ch2025_mLight.parquet", "m_light", "mlight"),
        "wlight": lambda: scalar_source_features("ch2025_wLight.parquet", "w_light", "wlight"),
        "gps": gps_features,
        "wifi": lambda: list_payload_features("ch2025_mWifi.parquet", "m_wifi", "wifi"),
        "ble": lambda: list_payload_features("ch2025_mBle.parquet", "m_ble", "ble"),
        "usage": usage_features,
        "ambience": ambience_features,
    }
    frames: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    for name, builder in builders.items():
        frame, info = builder()
        frames.append(frame)
        coverage[name] = info
    merged = merge_frames(frames)
    merged = add_hour_differences(merged)
    feature_cols = [col for col in merged.columns if col not in {"subject_id", "lifelog_date"}]
    coverage["feature_columns_before_alignment"] = int(len(feature_cols))
    coverage["raw_total_rows"] = int(sum(info.get("raw_rows", 0) for info in coverage.values() if isinstance(info, dict)))
    coverage["window_total_rows"] = int(
        sum(info.get("window_rows", 0) for info in coverage.values() if isinstance(info, dict))
    )
    return merged, coverage


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


def load_reference_values(train: pd.DataFrame, candidate_id: int) -> tuple[Path, np.ndarray]:
    path = find_oof(candidate_id)
    pred = pd.read_csv(path)
    pred["lifelog_date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "lifelog_date"]].copy()
    merged = key.merge(pred.drop(columns=["date"]), on=["subject_id", "lifelog_date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with sorted train rows.")
    return path, merged[TARGETS].astype(float).to_numpy()


def align_features(train: pd.DataFrame, test: pd.DataFrame, raw_temporal: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_keys = train[["subject_id", "lifelog_date"]].copy()
    test_keys = test[["subject_id", "lifelog_date"]].copy()
    all_keys = pd.concat(
        [train_keys.assign(split="train"), test_keys.assign(split="test")],
        ignore_index=True,
    )
    aligned = all_keys.merge(raw_temporal, on=["subject_id", "lifelog_date"], how="left", validate="one_to_one")
    feature_cols = [col for col in aligned.columns if col not in {"subject_id", "lifelog_date", "split"}]
    numeric_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_numeric_dtype(aligned[col]) and aligned[col].notna().any()
    ]
    numeric_cols = [col for col in numeric_cols if aligned[col].nunique(dropna=True) > 1]
    train_part = aligned[aligned["split"].eq("train")]
    test_part = aligned[aligned["split"].eq("test")]
    metadata = {
        "aligned_rows": int(len(aligned)),
        "train_rows": int(len(train_part)),
        "test_rows": int(len(test_part)),
        "feature_columns": int(len(numeric_cols)),
        "train_any_feature_coverage": float(train_part[numeric_cols].notna().any(axis=1).mean()) if numeric_cols else 0.0,
        "test_any_feature_coverage": float(test_part[numeric_cols].notna().any(axis=1).mean()) if numeric_cols else 0.0,
        "train_missing_mean": float(train_part[numeric_cols].isna().mean().mean()) if numeric_cols else 1.0,
        "test_missing_mean": float(test_part[numeric_cols].isna().mean().mean()) if numeric_cols else 1.0,
    }
    return aligned, metadata


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_split_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    total = 0
    weighted = 0.0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def fold_matrix(
    features: pd.DataFrame,
    feature_cols: list[str],
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = features.iloc[fit_idx]
    query = features.iloc[query_idx]
    usable_cols = [col for col in feature_cols if fit[col].notna().any()]
    fit_raw = fit[usable_cols].reset_index(drop=True)
    query_raw = query[usable_cols].reset_index(drop=True)
    dev_cols = usable_cols[:MAX_SUBJECT_DEV_COLUMNS]
    if not dev_cols:
        return fit_raw, query_raw
    global_medians = fit[dev_cols].median().fillna(0.0)
    subject_medians = fit.groupby("subject_id", sort=False)[dev_cols].median()
    fit_base = subject_medians.reindex(fit["subject_id"]).reset_index(drop=True).fillna(global_medians)
    query_base = subject_medians.reindex(query["subject_id"]).reset_index(drop=True).fillna(global_medians)
    fit_dev = fit[dev_cols].reset_index(drop=True) - fit_base
    query_dev = query[dev_cols].reset_index(drop=True) - query_base
    fit_dev.columns = [f"{col}_dev_subject" for col in dev_cols]
    query_dev.columns = [f"{col}_dev_subject" for col in dev_cols]
    return pd.concat([fit_raw, fit_dev], axis=1), pd.concat([query_raw, query_dev], axis=1)


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=45),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def same_subject_knn_fold_predictions(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit_arr = scaler.fit_transform(imputer.fit_transform(fit_x))
    valid_arr = scaler.transform(imputer.transform(valid_x))
    fit_subjects = features.iloc[fit_idx]["subject_id"].astype(str).to_numpy()
    valid_subjects = features.iloc[valid_idx]["subject_id"].astype(str).to_numpy()
    fit_y = y[fit_idx]
    out = np.full(len(valid_idx), float(np.mean(fit_y)), dtype=float)
    k_requested = int(spec.k_neighbors or 1)
    for subject_id in np.unique(valid_subjects):
        valid_mask = valid_subjects == subject_id
        fit_mask = fit_subjects == subject_id
        if not fit_mask.any():
            continue
        fit_subject_arr = fit_arr[fit_mask]
        fit_subject_y = fit_y[fit_mask]
        valid_subject_arr = valid_arr[valid_mask]
        distances = ((valid_subject_arr[:, None, :] - fit_subject_arr[None, :, :]) ** 2).mean(axis=2)
        k = min(k_requested, fit_subject_arr.shape[0])
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        out[valid_mask] = fit_subject_y[nearest].mean(axis=1)
    return np.clip(out, EPS, 1 - EPS)


def select_feature_columns(aligned: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    candidate_cols = [
        col
        for col in aligned.columns
        if col not in {"subject_id", "lifelog_date", "split"}
        and pd.api.types.is_numeric_dtype(aligned[col])
        and aligned[col].notna().any()
        and aligned[col].nunique(dropna=True) > 1
    ]
    train_part = aligned[aligned["split"].eq("train")]
    rows: list[dict[str, Any]] = []
    for col in candidate_cols:
        series = train_part[col]
        coverage = float(series.notna().mean())
        if coverage < 0.50:
            continue
        variance = float(series.var(skipna=True))
        if not math.isfinite(variance) or variance <= 0:
            continue
        rows.append({"column": col, "coverage": coverage, "variance": variance})
    rows = sorted(rows, key=lambda row: (row["coverage"], row["variance"]), reverse=True)
    selected = [row["column"] for row in rows[:MAX_FEATURE_COLUMNS]]
    metadata = {
        "candidate_numeric_columns": int(len(candidate_cols)),
        "eligible_by_coverage_variance": int(len(rows)),
        "selected_feature_columns": int(len(selected)),
        "max_feature_columns": MAX_FEATURE_COLUMNS,
        "max_subject_dev_columns": MAX_SUBJECT_DEV_COLUMNS,
        "top_selected_columns": selected[:20],
    }
    return selected, metadata


def predict_oof_for_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        elif spec.kind == "ss_knn":
            proba = same_subject_knn_fold_predictions(features, feature_cols, y, fit_idx, valid_idx, spec)
        else:
            fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
            model = make_model(spec)
            model.fit(fit_x, y[fit_idx])
            proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
    if np.isnan(pred).any():
        fill_value = float(np.nanmean(pred)) if np.isfinite(np.nanmean(pred)) else float(np.mean(y))
        pred = np.where(np.isnan(pred), fill_value, pred)
    return np.clip(pred, EPS, 1 - EPS)


def scan_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    target: str,
) -> dict[str, Any]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        split_source_predictions: dict[str, np.ndarray] = {}
        for split_name in PRIMARY_SPLITS:
            split_source_predictions[split_name] = predict_oof_for_split(
                features,
                feature_cols,
                y,
                splits[split_name],
                spec,
            )
        for weight in BLEND_WEIGHTS:
            split_deltas: dict[str, float] = {}
            split_losses: dict[str, float] = {}
            split_anchor_losses: dict[str, float] = {}
            wins = 0
            for split_name in PRIMARY_SPLITS:
                folds = splits[split_name]
                source_pred = split_source_predictions[split_name]
                blended = (1.0 - weight) * anchor_values[:, target_idx] + weight * source_pred
                anchor_loss = weighted_split_loss(y, anchor_values[:, target_idx], folds)
                candidate_loss = weighted_split_loss(y, blended, folds)
                delta = candidate_loss - anchor_loss
                split_anchor_losses[split_name] = anchor_loss
                split_losses[split_name] = candidate_loss
                split_deltas[split_name] = delta
                wins += int(delta < 0)
            rows.append(
                {
                    "target": target,
                    "config": f"{target}_id342_raw_temporal_{spec.name}_w{weight:g}",
                    "model_spec": asdict(spec),
                    "weight": weight,
                    "wins": int(wins),
                    "mean_delta": float(np.mean(list(split_deltas.values()))),
                    "max_delta": float(np.max(list(split_deltas.values()))),
                    "min_delta": float(np.min(list(split_deltas.values()))),
                    "split_deltas": split_deltas,
                    "split_losses": split_losses,
                    "split_anchor_losses": split_anchor_losses,
                    "eligible": bool(
                        wins >= TARGET_WIN_FLOOR
                        and float(np.mean(list(split_deltas.values()))) <= TARGET_MEAN_DELTA_CEILING
                        and float(np.max(list(split_deltas.values()))) <= TARGET_MAX_DELTA_CEILING
                    ),
                }
            )
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    best = rows[0] if rows else None
    return {
        "target": target,
        "best": best,
        "eligible_count": len(eligible_rows),
        "selected": eligible_rows[0] if eligible_rows else None,
        "top_rows": rows[:8],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = read_train()
    test = read_test_keys()
    raw_temporal, raw_coverage = build_raw_temporal_frame()
    aligned, alignment = align_features(train, test, raw_temporal)
    train_features = aligned[aligned["split"].eq("train")].reset_index(drop=True)
    feature_cols, feature_selection = select_feature_columns(aligned)
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    anchor_path, anchor_values = load_reference_values(train, 342)
    target_reports = {
        target: scan_target(train, train_features, feature_cols, splits, anchor_values, target)
        for target in FOCUS_TARGETS
    }
    selected_targets = [
        target
        for target, target_report in target_reports.items()
        if target_report.get("selected") is not None
    ]
    selected_rows = {
        target: target_report["selected"]
        for target, target_report in target_reports.items()
        if target_report.get("selected") is not None
    }
    if selected_rows:
        mean_primary_delta = float(np.mean([row["mean_delta"] for row in selected_rows.values()]))
        max_primary_delta = float(np.max([row["max_delta"] for row in selected_rows.values()]))
    else:
        mean_primary_delta = 0.0
        max_primary_delta = 0.0
    partial_signal_rows = {
        target: [
            row
            for row in target_report["top_rows"]
            if row["mean_delta"] < 0 and row["wins"] >= 3 and row["max_delta"] <= 0.003
        ][:3]
        for target, target_report in target_reports.items()
    }
    partial_signal_targets = [target for target, rows in partial_signal_rows.items() if rows]
    projected_public_proxy = ID342_PUBLIC + mean_primary_delta
    final_gate_pass = bool(
        selected_targets
        and len(selected_targets) == len(FOCUS_TARGETS)
        and mean_primary_delta < -0.0005
        and max_primary_delta <= 0.00075
        and projected_public_proxy < ID342_PUBLIC
    )
    decision = {
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "all_focus_targets_selected": len(selected_targets) == len(FOCUS_TARGETS),
        "mean_primary_delta_vs_id342": mean_primary_delta,
        "max_primary_delta_vs_id342": max_primary_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "raw_temporal_has_q3s3_signal": bool(selected_targets),
        "partial_signal_targets": partial_signal_targets,
        "q3_partial_signal_unstable": "Q3" in partial_signal_targets and "Q3" not in selected_targets,
        "s3_partial_signal_unstable": "S3" in partial_signal_targets and "S3" not in selected_targets,
        "final_raw_temporal_gate_pass": final_gate_pass,
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "next_action": (
            "Do not create a DACON artifact from this smoke. "
            "If the final gate passes locally, use Kaggle no-submit for parity and artifact-review evidence only. "
            "If it fails, keep the raw temporal representation as negative evidence and move to a stronger sequence model."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_raw_temporal_q3s3_sequence_image_smoke",
        "purpose": "Test an ID342-aware raw temporal sequence-image representation for Q3/S3 without creating artifacts.",
        "research_priority_report": rel(RESEARCH_PRIORITY_PATH),
        "anchor_oof": rel(anchor_path),
        "time_window_contract": {
            "anchor_rule": "Timestamps 00:00-11:59 are assigned to the previous lifelog date.",
            "relative_hour_min": min(HOURS),
            "relative_hour_max_exclusive": max(HOURS) + 1,
            "hour_bins": HOURS,
            "target_focus": FOCUS_TARGETS,
        },
        "coverage": {
            "raw_sources": raw_coverage,
            "alignment": alignment,
        },
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": list(BLEND_WEIGHTS),
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "feature_columns_used": int(len(feature_cols)),
            "feature_selection": feature_selection,
        },
        "target_reports": target_reports,
        "partial_signal_rows": partial_signal_rows,
        "decision": decision,
        "in_memory_validation_predictions_only": True,
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    alignment = coverage["alignment"]
    decision = report["decision"]
    lines = [
        "# Post-ID342 raw temporal Q3/S3 sequence-image smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Time Window",
        "",
        f"- Anchor rule: {report['time_window_contract']['anchor_rule']}",
        f"- Relative hours: `{report['time_window_contract']['relative_hour_min']}` to `{report['time_window_contract']['relative_hour_max_exclusive']}`",
        f"- Focus targets: `{','.join(report['time_window_contract']['target_focus'])}`",
        "",
        "## Coverage",
        "",
        f"- Raw total rows in source files: `{coverage['raw_sources']['raw_total_rows']}`",
        f"- Rows inside relative window: `{coverage['raw_sources']['window_total_rows']}`",
        f"- Feature columns before alignment: `{coverage['raw_sources']['feature_columns_before_alignment']}`",
        f"- Feature columns used: `{report['validation']['feature_columns_used']}`",
        f"- Train/test any-feature coverage: `{alignment['train_any_feature_coverage']:.4f}` / `{alignment['test_any_feature_coverage']:.4f}`",
        f"- Train/test missing mean: `{alignment['train_missing_mean']:.4f}` / `{alignment['test_missing_mean']:.4f}`",
        "",
        "## Target Results",
        "",
        "| target | selected | best config | wins | mean delta | max delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for target, target_report in report["target_reports"].items():
        best = target_report["best"] or {}
        selected = target_report["selected"] is not None
        lines.append(
            "| {target} | `{selected}` | `{config}` | {wins} | {mean_delta:+.6f} | {max_delta:+.6f} |".format(
                target=target,
                selected=selected,
                config=best.get("config"),
                wins=int(best.get("wins", 0)),
                mean_delta=float(best.get("mean_delta", 0.0)),
                max_delta=float(best.get("max_delta", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Raw temporal has Q3/S3 signal: `{decision['raw_temporal_has_q3s3_signal']}`",
            f"- Partial signal targets: `{decision['partial_signal_targets']}`",
            f"- Final gate pass: `{decision['final_raw_temporal_gate_pass']}`",
            f"- Projected public proxy from ID342: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55 after proxy: `{decision['gap_to_public_055_after_proxy']:+.9f}`",
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
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)} and {rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
