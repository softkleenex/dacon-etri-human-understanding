#!/usr/bin/env python3
"""Q2/Q3/S3 env-social-routine v2 feature-generation metadata smoke.

This diagnostic is intentionally report-only. It exercises the raw feature
construction path for the post-ID341 Q2/Q3/S3 fallback lane and records metadata
about row coverage, missingness, feature blocks, and leakage guards.

It does not train a model, create OOF/base predictions, create a submission CSV,
modify the submission log, push/run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ITEMS_DIR = RAW_DIR / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q2q3s3_feature_generation_metadata_smoke_20260618.json"
TARGET_FOCUS = ["Q2", "Q3", "S3"]
EXPECTED_TRAIN_ROWS = 450
EXPECTED_TEST_ROWS = 250
FORBIDDEN_FEATURE_TOKENS = ["bssid", "address", "app_name"]

FEATURE_BLOCKS = {
    "routine_stability_clock_v2": ["q2rs_screen_", "q2rs_activity_", "q2rs_usage_", "q2rs_roll_"],
    "place_social_novelty_v2": ["q2ps_wifi_", "q2ps_ble_", "q2ps_gps_"],
    "ambient_load_recovery_context_v2": ["q2al_ambience_", "q2al_context_"],
}

TIME_BINS = {
    "afternoon_12_18": (12.0, 18.0),
    "evening_18_24": (18.0, 24.0),
    "night_24_30": (24.0, 30.0),
    "morning_30_36": (30.0, 36.0),
}

APP_CATEGORIES = ["communication", "media_search", "finance_shop", "routine_health", "system_home", "other"]
AMBIENCE_CATEGORIES = ["speech", "music", "vehicle", "outdoor", "indoor_public", "quiet", "other"]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def list_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "as_py"):
        converted = value.as_py()
        return list_items(converted)
    return []


def item_value(item: Any, key: str, index: int | None = None) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    if hasattr(item, "as_py"):
        return item_value(item.as_py(), key, index)
    if index is not None and isinstance(item, (list, tuple, np.ndarray)) and len(item) > index:
        return item[index]
    return None


def entropy_from_counter(counter: Counter[Any]) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return math.nan
    entropy = 0.0
    for count in counter.values():
        p = float(count) / total
        if p > 0:
            entropy -= p * math.log(p)
    return float(entropy)


def top_share_from_counter(counter: Counter[Any]) -> float:
    total = float(sum(counter.values()))
    return float(max(counter.values()) / total) if counter and total > 0 else math.nan


def set_jaccard(current: set[str], reference: set[str]) -> float:
    if not current and not reference:
        return math.nan
    union = current | reference
    return float(len(current & reference) / len(union)) if union else math.nan


def set_new_share(current: set[str], reference: set[str]) -> float:
    return float(len(current - reference) / len(current)) if current else math.nan


def anchored_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out[out["timestamp"].notna()].copy()
    row_start = out["timestamp"].dt.normalize() - pd.to_timedelta((out["timestamp"].dt.hour < 12).astype(int), unit="D")
    out["row_date"] = row_start.dt.date
    out["rel_hour"] = (out["timestamp"] - row_start).dt.total_seconds() / 3600.0
    return out[out["rel_hour"].between(12.0, 36.0, inclusive="left")].copy()


def time_bin(rel_hour: float) -> str:
    for name, (start, end) in TIME_BINS.items():
        if start <= rel_hour < end:
            return name
    return "outside"


def read_keys(path: Path, is_test: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["subject_id", "lifelog_date"])
    frame = frame.rename(columns={"lifelog_date": "row_date"})
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["row_date"] = pd.to_datetime(frame["row_date"]).dt.date
    frame["is_test"] = int(is_test)
    return frame


def key_frame() -> pd.DataFrame:
    train = read_keys(RAW_DIR / "ch2026_metrics_train.csv", is_test=0)
    test = read_keys(RAW_DIR / "ch2026_submission_sample.csv", is_test=1)
    return pd.concat([train, test], ignore_index=True)


def novelty_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.sort_values(["subject_id", "row_date"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _subject, group in out.groupby("subject_id", sort=False):
        prev_set: set[str] = set()
        rolling_sets: deque[set[str]] = deque(maxlen=7)
        for idx, row in group.iterrows():
            current = set(row.get("_ids", set()))
            rolling_union = set().union(*rolling_sets) if rolling_sets else set()
            rows.append(
                {
                    "_row_index": idx,
                    f"{prefix}_jaccard_prev": set_jaccard(current, prev_set),
                    f"{prefix}_new_share_prev": set_new_share(current, prev_set),
                    f"{prefix}_jaccard_roll7": set_jaccard(current, rolling_union),
                    f"{prefix}_new_share_roll7": set_new_share(current, rolling_union),
                    f"{prefix}_repeat_share_roll7": float(len(current & rolling_union) / len(current))
                    if current
                    else math.nan,
                }
            )
            prev_set = current
            rolling_sets.append(current)
    novelty = pd.DataFrame(rows).set_index("_row_index")
    return pd.concat([out, novelty.reindex(out.index)], axis=1)


def daily_device_features(
    filename: str, list_col: str, id_key: str, prefix: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / filename
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", list_col])
    raw = anchored_window(raw)
    accum: dict[tuple[str, Any], dict[str, Any]] = {}
    observed_items = 0
    for subject_id, row_date, rel_hour, values in raw[["subject_id", "row_date", "rel_hour", list_col]].itertuples(
        index=False
    ):
        key = (str(subject_id), row_date)
        rec = accum.setdefault(
            key,
            {
                "scan_count": 0,
                "item_count": 0,
                "evening_items": 0,
                "night_items": 0,
                "strong_items": 0,
                "rssi_sum": 0.0,
                "rssi_count": 0,
                "counter": Counter(),
            },
        )
        rec["scan_count"] += 1
        bin_name = time_bin(float(rel_hour))
        for item in list_items(values):
            ident = item_value(item, id_key)
            if ident is None or str(ident) == "":
                continue
            observed_items += 1
            rec["counter"][str(ident)] += 1
            rec["item_count"] += 1
            if bin_name == "evening_18_24":
                rec["evening_items"] += 1
            if bin_name == "night_24_30":
                rec["night_items"] += 1
            rssi = safe_float(item_value(item, "rssi"))
            if math.isfinite(rssi):
                rec["rssi_sum"] += rssi
                rec["rssi_count"] += 1
                if rssi >= -60:
                    rec["strong_items"] += 1
    rows: list[dict[str, Any]] = []
    for (subject_id, row_date), rec in accum.items():
        counter: Counter[str] = rec["counter"]
        item_count = float(rec["item_count"])
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                "_ids": set(counter.keys()),
                f"{prefix}_scan_count": float(rec["scan_count"]),
                f"{prefix}_item_count": item_count,
                f"{prefix}_unique_count": float(len(counter)),
                f"{prefix}_items_per_scan": item_count / max(float(rec["scan_count"]), 1.0),
                f"{prefix}_entropy": entropy_from_counter(counter),
                f"{prefix}_top_share": top_share_from_counter(counter),
                f"{prefix}_evening_item_share": rec["evening_items"] / item_count if item_count else math.nan,
                f"{prefix}_night_item_share": rec["night_items"] / item_count if item_count else math.nan,
                f"{prefix}_strong_item_share": rec["strong_items"] / item_count if item_count else math.nan,
                f"{prefix}_mean_rssi": rec["rssi_sum"] / rec["rssi_count"] if rec["rssi_count"] else math.nan,
            }
        )
    features = (
        novelty_columns(pd.DataFrame(rows), prefix).drop(columns=["_ids"], errors="ignore") if rows else pd.DataFrame()
    )
    return features, {
        "source": filename,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(features)),
        "observed_nested_items": int(observed_items),
        "feature_columns": int(len([col for col in features.columns if col.startswith(prefix)])),
    }


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def categorize_application(value: Any) -> str:
    text = normalize_text(value)
    if any(token in text for token in ["카카오", "kakao", "메시지", "message", "전화", "통화", "문자"]):
        return "communication"
    if any(token in text for token in ["naver", "chrome", "youtube", "유튜브", "인터넷", "browser", "google"]):
        return "media_search"
    if any(token in text for token in ["토스", "pay", "페이", "카드", "bank", "은행", "쿠팡", "shop", "쇼핑"]):
        return "finance_shop"
    if any(token in text for token in ["캐시워크", "성경", "캘린더", "calendar", "health", "헬스", "운동"]):
        return "routine_health"
    if any(token in text for token in ["one ui", "홈", "launcher", "설정", "system", "관리자"]):
        return "system_home"
    return "other"


def daily_usage_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mUsageStats.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_usage_stats"])
    raw = anchored_window(raw)
    accum: dict[tuple[str, Any], dict[str, Any]] = {}
    observed_items = 0
    for subject_id, row_date, rel_hour, values in raw[
        ["subject_id", "row_date", "rel_hour", "m_usage_stats"]
    ].itertuples(index=False):
        key = (str(subject_id), row_date)
        rec = accum.setdefault(
            key,
            {
                "scan_count": 0,
                "id_counter": Counter(),
                "time_counter": Counter(),
                "category_time": Counter(),
                "total_time": 0.0,
                "evening_time": 0.0,
                "night_time": 0.0,
            },
        )
        rec["scan_count"] += 1
        bin_name = time_bin(float(rel_hour))
        for item in list_items(values):
            app = normalize_text(item_value(item, "app_name"))
            if not app:
                continue
            observed_items += 1
            total_time = max(safe_float(item_value(item, "total_time"), 0.0), 0.0)
            category = categorize_application(app)
            rec["id_counter"][app] += 1
            rec["time_counter"][app] += total_time
            rec["category_time"][category] += total_time
            rec["total_time"] += total_time
            if bin_name == "evening_18_24":
                rec["evening_time"] += total_time
            if bin_name == "night_24_30":
                rec["night_time"] += total_time
    rows: list[dict[str, Any]] = []
    prefix = "q2rs_usage"
    for (subject_id, row_date), rec in accum.items():
        total_time = float(rec["total_time"])
        row = {
            "subject_id": subject_id,
            "row_date": row_date,
            "_ids": set(rec["id_counter"].keys()),
            f"{prefix}_scan_count": float(rec["scan_count"]),
            f"{prefix}_unique_count": float(len(rec["id_counter"])),
            f"{prefix}_log_total_time": float(np.log1p(total_time)),
            f"{prefix}_evening_time_share": rec["evening_time"] / total_time if total_time else math.nan,
            f"{prefix}_night_time_share": rec["night_time"] / total_time if total_time else math.nan,
            f"{prefix}_event_entropy": entropy_from_counter(rec["id_counter"]),
            f"{prefix}_time_entropy": entropy_from_counter(rec["time_counter"]),
            f"{prefix}_top_event_share": top_share_from_counter(rec["id_counter"]),
            f"{prefix}_top_time_share": top_share_from_counter(rec["time_counter"]),
        }
        for category in APP_CATEGORIES:
            row[f"{prefix}_{category}_time_share"] = (
                float(rec["category_time"][category]) / total_time if total_time else math.nan
            )
        rows.append(row)
    features = (
        novelty_columns(pd.DataFrame(rows), prefix).drop(columns=["_ids"], errors="ignore") if rows else pd.DataFrame()
    )
    return features, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(features)),
        "observed_nested_items": int(observed_items),
        "feature_columns": int(len([col for col in features.columns if col.startswith(prefix)])),
    }


def daily_gps_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mGps.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_gps"])
    raw = anchored_window(raw)
    accum: dict[tuple[str, Any], dict[str, Any]] = {}
    observed_items = 0
    for subject_id, row_date, rel_hour, values in raw[["subject_id", "row_date", "rel_hour", "m_gps"]].itertuples(
        index=False
    ):
        key = (str(subject_id), row_date)
        rec = accum.setdefault(
            key,
            {
                "record_count": 0,
                "point_count": 0,
                "speed_sum": 0.0,
                "speed_sq_sum": 0.0,
                "speed_max": math.nan,
                "moving_points": 0,
                "night_moving_points": 0,
                "lat_sum": 0.0,
                "lon_sum": 0.0,
                "lat_sq_sum": 0.0,
                "lon_sq_sum": 0.0,
                "lat_min": math.nan,
                "lat_max": math.nan,
                "lon_min": math.nan,
                "lon_max": math.nan,
                "coord_count": 0,
            },
        )
        rec["record_count"] += 1
        bin_name = time_bin(float(rel_hour))
        for item in list_items(values):
            observed_items += 1
            rec["point_count"] += 1
            speed = safe_float(item_value(item, "speed"))
            if math.isfinite(speed):
                rec["speed_sum"] += speed
                rec["speed_sq_sum"] += speed * speed
                rec["speed_max"] = speed if not math.isfinite(rec["speed_max"]) else max(rec["speed_max"], speed)
                if speed > 0.5:
                    rec["moving_points"] += 1
                    if bin_name == "night_24_30":
                        rec["night_moving_points"] += 1
            lat = safe_float(item_value(item, "latitude"))
            lon = safe_float(item_value(item, "longitude"))
            if math.isfinite(lat) and math.isfinite(lon):
                rec["coord_count"] += 1
                rec["lat_sum"] += lat
                rec["lon_sum"] += lon
                rec["lat_sq_sum"] += lat * lat
                rec["lon_sq_sum"] += lon * lon
                rec["lat_min"] = lat if not math.isfinite(rec["lat_min"]) else min(rec["lat_min"], lat)
                rec["lat_max"] = lat if not math.isfinite(rec["lat_max"]) else max(rec["lat_max"], lat)
                rec["lon_min"] = lon if not math.isfinite(rec["lon_min"]) else min(rec["lon_min"], lon)
                rec["lon_max"] = lon if not math.isfinite(rec["lon_max"]) else max(rec["lon_max"], lon)
    rows: list[dict[str, Any]] = []
    prefix = "q2ps_gps"
    for (subject_id, row_date), rec in accum.items():
        point_count = float(rec["point_count"])
        coord_count = float(rec["coord_count"])
        speed_mean = rec["speed_sum"] / point_count if point_count else math.nan
        speed_var = rec["speed_sq_sum"] / point_count - speed_mean * speed_mean if point_count else math.nan
        if coord_count:
            lat_mean = rec["lat_sum"] / coord_count
            lon_mean = rec["lon_sum"] / coord_count
            lat_var = max(rec["lat_sq_sum"] / coord_count - lat_mean * lat_mean, 0.0)
            lon_var = max(rec["lon_sq_sum"] / coord_count - lon_mean * lon_mean, 0.0)
            lat_range = rec["lat_max"] - rec["lat_min"]
            lon_range = rec["lon_max"] - rec["lon_min"]
            location_spread = math.sqrt(lat_var + lon_var)
        else:
            lat_range = lon_range = location_spread = math.nan
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                f"{prefix}_record_count": float(rec["record_count"]),
                f"{prefix}_point_count": point_count,
                f"{prefix}_points_per_record": point_count / max(float(rec["record_count"]), 1.0),
                f"{prefix}_speed_mean": speed_mean,
                f"{prefix}_speed_std": math.sqrt(max(speed_var, 0.0)) if math.isfinite(speed_var) else math.nan,
                f"{prefix}_speed_max": rec["speed_max"],
                f"{prefix}_moving_share": rec["moving_points"] / point_count if point_count else math.nan,
                f"{prefix}_night_moving_share": rec["night_moving_points"] / point_count if point_count else math.nan,
                f"{prefix}_location_spread": location_spread,
                f"{prefix}_lat_range": lat_range,
                f"{prefix}_lon_range": lon_range,
            }
        )
    features = pd.DataFrame(rows)
    return features, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(features)),
        "observed_nested_items": int(observed_items),
        "feature_columns": int(len([col for col in features.columns if col.startswith(prefix)])),
        "exact_lat_lon_means_exported": False,
    }


def daily_screen_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mScreenStatus.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_screen_use"])
    raw = anchored_window(raw).sort_values(["subject_id", "row_date", "timestamp"])
    raw["screen_use"] = pd.to_numeric(raw["m_screen_use"], errors="coerce")
    raw["is_on"] = (raw["screen_use"] > 0).astype(float)
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    out = grouped.agg(
        q2rs_screen_record_count=("screen_use", "size"),
        q2rs_screen_on_sum=("is_on", "sum"),
        q2rs_screen_on_mean=("is_on", "mean"),
    ).reset_index()
    for bin_name in TIME_BINS:
        share = raw[raw["bin_name"].eq(bin_name)].groupby(["subject_id", "row_date"])["is_on"].mean()
        out = out.merge(
            share.rename(f"q2rs_screen_{bin_name}_on_mean").reset_index(), on=["subject_id", "row_date"], how="left"
        )
    toggle = (
        grouped["is_on"]
        .apply(lambda series: float(series.diff().abs().fillna(0).sum()))
        .rename("q2rs_screen_toggle_count")
    )
    out = out.merge(toggle.reset_index(), on=["subject_id", "row_date"], how="left")
    out["q2rs_screen_toggle_rate"] = out["q2rs_screen_toggle_count"] / out["q2rs_screen_record_count"].clip(lower=1)
    return out, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len([col for col in out.columns if col.startswith("q2rs_screen_")])),
    }


def daily_activity_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mActivity.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_activity"])
    raw = anchored_window(raw).sort_values(["subject_id", "row_date", "timestamp"])
    raw["activity"] = pd.to_numeric(raw["m_activity"], errors="coerce")
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    out = grouped.agg(q2rs_activity_record_count=("activity", "size")).reset_index()
    value_counts = raw.groupby(["subject_id", "row_date", "activity"]).size().rename("count").reset_index()
    total = value_counts.groupby(["subject_id", "row_date"])["count"].transform("sum").clip(lower=1)
    value_counts["share"] = value_counts["count"] / total
    pivot = value_counts.pivot_table(
        index=["subject_id", "row_date"], columns="activity", values="share", fill_value=0.0
    )
    pivot.columns = [f"q2rs_activity_state_{int(col)}_share" for col in pivot.columns]
    out = out.merge(pivot.reset_index(), on=["subject_id", "row_date"], how="left")
    entropy = grouped["activity"].apply(lambda series: entropy_from_counter(Counter(series.dropna().astype(int))))
    transition = grouped["activity"].apply(lambda series: float(series.diff().fillna(0).ne(0).sum()))
    out = out.merge(
        entropy.rename("q2rs_activity_state_entropy").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    out = out.merge(
        transition.rename("q2rs_activity_transition_count").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    out["q2rs_activity_transition_rate"] = out["q2rs_activity_transition_count"] / out[
        "q2rs_activity_record_count"
    ].clip(lower=1)
    for bin_name in TIME_BINS:
        subset = raw[raw["bin_name"].eq(bin_name)]
        active_share = subset.groupby(["subject_id", "row_date"])["activity"].apply(
            lambda series: float(pd.to_numeric(series, errors="coerce").fillna(-1).ne(0).mean())
        )
        out = out.merge(
            active_share.rename(f"q2rs_activity_{bin_name}_nonzero_share").reset_index(),
            on=["subject_id", "row_date"],
            how="left",
        )
    return out, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len([col for col in out.columns if col.startswith("q2rs_activity_")])),
    }


def categorize_ambience(value: Any) -> str:
    text = normalize_text(value)
    if any(token in text for token in ["speech", "conversation", "narration", "monologue", "child speech"]):
        return "speech"
    if any(token in text for token in ["music", "singing", "song", "musical"]):
        return "music"
    if any(token in text for token in ["vehicle", "car", "truck", "bus", "motor", "road", "bicycle"]):
        return "vehicle"
    if any(token in text for token in ["outside", "wind", "rain", "water", "animal", "bird", "rural", "urban"]):
        return "outdoor"
    if any(token in text for token in ["inside", "crowd", "public space", "room", "hall"]):
        return "indoor_public"
    if any(token in text for token in ["silence", "quiet"]):
        return "quiet"
    return "other"


def daily_ambience_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mAmbience.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_ambience"])
    raw = anchored_window(raw)
    accum: dict[tuple[str, Any], dict[str, Any]] = {}
    observed_items = 0
    for subject_id, row_date, rel_hour, values in raw[["subject_id", "row_date", "rel_hour", "m_ambience"]].itertuples(
        index=False
    ):
        key = (str(subject_id), row_date)
        rec = accum.setdefault(
            key,
            {
                "record_count": 0,
                "item_count": 0,
                "label_counter": Counter(),
                "score_counter": Counter(),
                "top_score_sum": 0.0,
                "evening_score_sum": 0.0,
                "night_score_sum": 0.0,
                "score_sum": 0.0,
            },
        )
        rec["record_count"] += 1
        bin_name = time_bin(float(rel_hour))
        row_top_score = 0.0
        for item in list_items(values):
            label = item_value(item, "label", 0)
            score = max(safe_float(item_value(item, "score", 1), 0.0), 0.0)
            if label is None:
                continue
            observed_items += 1
            category = categorize_ambience(label)
            rec["item_count"] += 1
            rec["label_counter"][category] += 1
            rec["score_counter"][category] += score
            rec["score_sum"] += score
            row_top_score = max(row_top_score, score)
            if bin_name == "evening_18_24":
                rec["evening_score_sum"] += score
            if bin_name == "night_24_30":
                rec["night_score_sum"] += score
        rec["top_score_sum"] += row_top_score
    rows: list[dict[str, Any]] = []
    prefix = "q2al_ambience"
    for (subject_id, row_date), rec in accum.items():
        score_sum = float(rec["score_sum"])
        item_count = float(rec["item_count"])
        row = {
            "subject_id": subject_id,
            "row_date": row_date,
            f"{prefix}_record_count": float(rec["record_count"]),
            f"{prefix}_item_count": item_count,
            f"{prefix}_items_per_record": item_count / max(float(rec["record_count"]), 1.0),
            f"{prefix}_category_entropy": entropy_from_counter(rec["label_counter"]),
            f"{prefix}_top_score_mean": rec["top_score_sum"] / max(float(rec["record_count"]), 1.0),
            f"{prefix}_evening_score_share": rec["evening_score_sum"] / score_sum if score_sum else math.nan,
            f"{prefix}_night_score_share": rec["night_score_sum"] / score_sum if score_sum else math.nan,
        }
        for category in AMBIENCE_CATEGORIES:
            row[f"{prefix}_{category}_score_share"] = (
                rec["score_counter"][category] / score_sum if score_sum else math.nan
            )
        rows.append(row)
    features = pd.DataFrame(rows)
    return features, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(features)),
        "observed_nested_items": int(observed_items),
        "feature_columns": int(len([col for col in features.columns if col.startswith(prefix)])),
    }


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame.copy() if out is None else out.merge(frame, on=["subject_id", "row_date"], how="outer")
    if out is None:
        return pd.DataFrame(columns=["subject_id", "row_date"])
    out["subject_id"] = out["subject_id"].astype(str)
    out["row_date"] = pd.to_datetime(out["row_date"]).dt.date
    return out


def add_temporal_roll_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values(["subject_id", "row_date"]).reset_index(drop=True).copy()
    base_cols = [
        col
        for col in out.columns
        if col.startswith(
            ("q2rs_screen_", "q2rs_activity_", "q2rs_usage_", "q2ps_wifi_", "q2ps_ble_", "q2ps_gps_", "q2al_ambience_")
        )
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    # Keep this metadata smoke compact: rolling deltas for the strongest behavior-summary columns only.
    priority_markers = ["record_count", "unique_count", "entropy", "share", "mean", "moving", "spread", "toggle_rate"]
    selected = [col for col in base_cols if any(marker in col for marker in priority_markers)][:80]
    roll_columns: dict[str, pd.Series] = {}
    for col in selected:
        grouped = out.groupby("subject_id", sort=False)[col]
        prev = grouped.shift(1)
        roll3 = grouped.transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
        roll_columns[f"q2rs_roll_{col}_prev_delta"] = out[col] - prev
        roll_columns[f"q2rs_roll_{col}_abs_roll3_delta"] = (out[col] - roll3).abs()
    if not roll_columns:
        return out
    return pd.concat([out, pd.DataFrame(roll_columns, index=out.index)], axis=1).copy()


def safe_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def add_context_interactions(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["q2al_context_speech_x_usage_night"] = safe_col(out, "q2al_ambience_speech_score_share") * safe_col(
        out, "q2rs_usage_night_time_share"
    )
    out["q2al_context_music_x_usage_total"] = safe_col(out, "q2al_ambience_music_score_share") * safe_col(
        out, "q2rs_usage_log_total_time"
    )
    out["q2al_context_vehicle_x_gps_moving"] = safe_col(out, "q2al_ambience_vehicle_score_share") * safe_col(
        out, "q2ps_gps_moving_share"
    )
    out["q2al_context_outdoor_x_place_spread"] = safe_col(out, "q2al_ambience_outdoor_score_share") * safe_col(
        out, "q2ps_gps_location_spread"
    )
    out["q2rs_roll_screen_x_usage_night"] = safe_col(out, "q2rs_screen_on_mean") * safe_col(
        out, "q2rs_usage_night_time_share"
    )
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = tuple(prefix for prefixes in FEATURE_BLOCKS.values() for prefix in prefixes)
    return [col for col in frame.columns if col.startswith(prefixes)]


def add_missing_indicators(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    cols = feature_columns(out)
    missing = {f"{col}_missing": out[col].isna().astype(float) for col in cols}
    if not missing:
        return out
    return pd.concat([out, pd.DataFrame(missing, index=out.index)], axis=1).copy()


def build_feature_metadata_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    source_frames: list[pd.DataFrame] = []
    source_metadata: dict[str, Any] = {}
    builders = {
        "wifi": lambda: daily_device_features("ch2025_mWifi.parquet", "m_wifi", "bssid", "q2ps_wifi"),
        "ble": lambda: daily_device_features("ch2025_mBle.parquet", "m_ble", "address", "q2ps_ble"),
        "usage": daily_usage_features,
        "gps": daily_gps_features,
        "screen": daily_screen_features,
        "activity": daily_activity_features,
        "ambience": daily_ambience_features,
    }
    for name, builder in builders.items():
        frame, metadata = builder()
        source_frames.append(frame)
        source_metadata[name] = metadata
    daily = merge_feature_frames(source_frames)
    daily = add_temporal_roll_features(daily)
    keys = key_frame()
    features = keys.merge(daily, on=["subject_id", "row_date"], how="left")
    features = add_context_interactions(features)
    features = add_missing_indicators(features)
    return features, source_metadata


def block_feature_columns(cols: list[str], block: str, include_missing: bool = True) -> list[str]:
    prefixes = tuple(FEATURE_BLOCKS[block])
    if include_missing:
        return [col for col in cols if col.startswith(prefixes)]
    return [col for col in cols if col.startswith(prefixes) and not col.endswith("_missing")]


def coverage_summary(features: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    train = features[features["is_test"].eq(0)]
    test = features[features["is_test"].eq(1)]
    raw_cols = [col for col in cols if not col.endswith("_missing")]
    if not raw_cols:
        return {
            "feature_columns": 0,
            "raw_feature_columns": 0,
            "train_any_nonnull_row_share": 0.0,
            "test_any_nonnull_row_share": 0.0,
            "train_missing_mean": None,
            "test_missing_mean": None,
        }
    return {
        "feature_columns": int(len(cols)),
        "raw_feature_columns": int(len(raw_cols)),
        "train_any_nonnull_row_share": float(train[raw_cols].notna().any(axis=1).mean()),
        "test_any_nonnull_row_share": float(test[raw_cols].notna().any(axis=1).mean()),
        "train_missing_mean": float(train[raw_cols].isna().mean().mean()),
        "test_missing_mean": float(test[raw_cols].isna().mean().mean()),
        "top_train_missing_features": {
            str(k): float(v) for k, v in train[raw_cols].isna().mean().sort_values(ascending=False).head(10).items()
        },
    }


def build_report() -> dict[str, Any]:
    features, source_metadata = build_feature_metadata_frame()
    cols = feature_columns(features)
    raw_cols = [col for col in cols if not col.endswith("_missing")]
    forbidden_feature_columns = [col for col in cols if any(token in col.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    duplicate_columns = features.columns[features.columns.duplicated()].tolist()
    all_null_raw_features = [col for col in raw_cols if features[col].isna().all()]
    target_columns_present = [
        target for target in ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"] if target in features.columns
    ]
    block_summary = {block: coverage_summary(features, block_feature_columns(cols, block)) for block in FEATURE_BLOCKS}
    train_rows = int(features["is_test"].eq(0).sum())
    test_rows = int(features["is_test"].eq(1).sum())
    expected_rows_ok = train_rows == EXPECTED_TRAIN_ROWS and test_rows == EXPECTED_TEST_ROWS
    feature_count_ok = len(raw_cols) >= 50
    block_coverage_ok = all(summary["raw_feature_columns"] > 0 for summary in block_summary.values())
    leakage_guard_ok = not forbidden_feature_columns and not target_columns_present and not duplicate_columns
    metadata_smoke_passed = bool(expected_rows_ok and feature_count_ok and block_coverage_ok and leakage_guard_ok)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_feature_generation_metadata_smoke",
        "purpose": (
            "Verify that the Q2/Q3/S3-first env/social/routine v2 feature lane can generate fold-safe, "
            "label-free metadata features for the 450 train and 250 test keys before any model training."
        ),
        "target_focus": TARGET_FOCUS,
        "input_files": {
            "train_keys": rel(RAW_DIR / "ch2026_metrics_train.csv"),
            "test_keys": rel(RAW_DIR / "ch2026_submission_sample.csv"),
            "raw_items_dir": rel(ITEMS_DIR),
        },
        "row_coverage": {
            "total_rows": int(len(features)),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "expected_train_rows": EXPECTED_TRAIN_ROWS,
            "expected_test_rows": EXPECTED_TEST_ROWS,
            "expected_rows_ok": expected_rows_ok,
            "duplicate_key_rows": int(features.duplicated(["subject_id", "row_date", "is_test"]).sum()),
            "subject_count": int(features["subject_id"].nunique()),
            "date_min": str(min(features["row_date"])),
            "date_max": str(max(features["row_date"])),
        },
        "source_metadata": source_metadata,
        "feature_summary": {
            "total_feature_columns": int(len(cols)),
            "raw_feature_columns": int(len(raw_cols)),
            "missing_indicator_columns": int(len([col for col in cols if col.endswith("_missing")])),
            "feature_count_ok": feature_count_ok,
            "all_null_raw_feature_count": int(len(all_null_raw_features)),
            "all_null_raw_features_sample": all_null_raw_features[:20],
            "duplicate_columns": duplicate_columns,
            "block_summary": block_summary,
        },
        "leakage_guards": {
            "forbidden_feature_tokens": FORBIDDEN_FEATURE_TOKENS,
            "forbidden_feature_columns": forbidden_feature_columns,
            "target_columns_present_in_feature_frame": target_columns_present,
            "direct_high_cardinality_identifier_columns_exported": False,
            "exact_lat_lon_means_exported": False,
            "uses_target_values": False,
            "model_training_performed": False,
            "fold_fit_required_for_future_encoders": True,
            "future_training_contract": [
                "fit any top-k/hash/category vocabulary inside each outer fold only",
                "keep direct BSSID/address/app-name strings out of model columns",
                "keep rolling/routine transforms target-free and row-date causal",
                "validate first on subject_hole_5, actual_test_band_holdout, boundary-neighbor, and loso stress splits",
            ],
            "leakage_guard_ok": leakage_guard_ok,
        },
        "decision": {
            "metadata_smoke_passed": metadata_smoke_passed,
            "target_focus": TARGET_FOCUS,
            "expected_rows_ok": expected_rows_ok,
            "block_coverage_ok": block_coverage_ok,
            "leakage_guard_ok": leakage_guard_ok,
            "next_local_allowed_step": "fold_internal_q2q3s3_model_smoke_only"
            if metadata_smoke_passed
            else "fix_metadata_generation_first",
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If this metadata smoke passes, the next safe step is a fold-internal Q2/Q3/S3-only model smoke "
                "against ID178/ID341 anchors; still no submission artifact without a separate promotion gate."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    row = report["row_coverage"]
    feat = report["feature_summary"]
    decision = report["decision"]
    lines = [
        "# Q2/Q3/S3 feature-generation metadata smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, no base prediction artifact, no notebook/history artifact, no submission-log row, no Kaggle push/run, and no DACON submission.",
        "",
        "## Row coverage",
        "",
        f"- Rows: `{row['total_rows']}` total / `{row['train_rows']}` train / `{row['test_rows']}` test",
        f"- Expected rows OK: `{row['expected_rows_ok']}`",
        f"- Duplicate key rows: `{row['duplicate_key_rows']}`",
        f"- Subjects: `{row['subject_count']}`, date range `{row['date_min']}` → `{row['date_max']}`",
        "",
        "## Feature summary",
        "",
        f"- Raw feature columns: `{feat['raw_feature_columns']}`",
        f"- Missing-indicator columns: `{feat['missing_indicator_columns']}`",
        f"- Total feature columns: `{feat['total_feature_columns']}`",
        f"- All-null raw feature count: `{feat['all_null_raw_feature_count']}`",
        "",
        "| block | raw cols | all cols | train coverage | test coverage | train missing | test missing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for block, summary in feat["block_summary"].items():
        lines.append(
            f"| `{block}` | {summary['raw_feature_columns']} | {summary['feature_columns']} | "
            f"{summary['train_any_nonnull_row_share']:.3f} | {summary['test_any_nonnull_row_share']:.3f} | "
            f"{summary['train_missing_mean']:.3f} | {summary['test_missing_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Source metadata",
            "",
            "| source | raw rows | daily rows | nested/items | feature cols |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, metadata in report["source_metadata"].items():
        lines.append(
            f"| `{name}` | {metadata['raw_rows']} | {metadata['daily_rows']} | "
            f"{metadata.get('observed_nested_items', '')} | {metadata['feature_columns']} |"
        )
    guard = report["leakage_guards"]
    lines.extend(
        [
            "",
            "## Leakage guards",
            "",
            f"- Forbidden direct identifier columns: `{guard['forbidden_feature_columns']}`",
            f"- Target columns present in feature frame: `{guard['target_columns_present_in_feature_frame']}`",
            f"- Direct high-cardinality identifier columns exported: `{guard['direct_high_cardinality_identifier_columns_exported']}`",
            f"- Model training performed: `{guard['model_training_performed']}`",
            f"- Leakage guard OK: `{guard['leakage_guard_ok']}`",
            "",
            "## Decision",
            "",
            f"- Metadata smoke passed: `{decision['metadata_smoke_passed']}`",
            f"- Target focus: `{', '.join(decision['target_focus'])}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Train model now: `{decision['train_model_now']}`",
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
