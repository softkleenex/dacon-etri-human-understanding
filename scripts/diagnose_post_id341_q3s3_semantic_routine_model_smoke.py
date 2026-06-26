#!/usr/bin/env python3
"""Fold-internal Q3/S3 semantic-routine model smoke after post-ID341.

This local diagnostic builds a small 700-row daily feature frame from raw event
streams using constrained, non-identifier aggregates. It then compares tiny
fold-internal source models against ID178/ID341 OOF anchors on frozen validation
lenses. It remains report-only: no full-test predictions, submission CSVs, base
prediction artifacts, notebook/history artifacts, Kaggle runs, or DACON
submissions are created.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BLEND_WEIGHTS,
    CONFIG_MAX_DELTA_CEILING,
    CONFIG_MEAN_DELTA_CEILING,
    CONFIG_WIN_FLOOR,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_target_for_split,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw"
ITEMS_DIR = RAW_DIR / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_semantic_routine_model_smoke_20260618.json"
RESOURCE_GATE = REPORTS_DIR / "post_id341_q3s3_semantic_routine_resource_gate_20260618.json"

TARGET_FOCUS = ["Q3", "S3"]
SOURCE_PREFIXES = ("sem_",)
DEFAULT_MODEL_SPECS = (MODEL_SPECS[0], MODEL_SPECS[1])


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "exists": True, "parse_ok": False, "decision": {}, "error": str(exc)}
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "error": None,
    }


def list_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "as_py"):
        return list_items(value.as_py())
    return []


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def entropy_from_counter(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return math.nan
    probs = [count / total for count in counter.values() if count > 0]
    return float(-sum(prob * math.log(prob) for prob in probs))


def anchored_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out[out["timestamp"].notna()].copy()
    row_start = out["timestamp"].dt.normalize() - pd.to_timedelta((out["timestamp"].dt.hour < 12).astype(int), unit="D")
    out["row_date"] = row_start.dt.date
    out["rel_hour"] = (out["timestamp"] - row_start).dt.total_seconds() / 3600.0
    return out[out["rel_hour"].between(12.0, 36.0, inclusive="left")].copy()


def read_key_frame() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv", usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv", usecols=["subject_id", "lifelog_date"])
    train = train.rename(columns={"lifelog_date": "row_date"}).assign(is_test=0)
    test = test.rename(columns={"lifelog_date": "row_date"}).assign(is_test=1)
    keys = pd.concat([train, test], ignore_index=True)
    keys["subject_id"] = keys["subject_id"].astype(str)
    keys["row_date"] = pd.to_datetime(keys["row_date"]).dt.date
    return keys


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def aggregate_usage_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mUsageStats.parquet"
    raw = anchored_window(pd.read_parquet(path, columns=["subject_id", "timestamp", "m_usage_stats"]))
    acc: dict[tuple[str, Any], dict[str, Any]] = defaultdict(
        lambda: {
            "record_count": 0,
            "event_count": 0,
            "total_time": 0.0,
            "late_total_time": 0.0,
            "apps": Counter(),
        }
    )
    for row in raw.itertuples(index=False):
        key = (str(row.subject_id), row.row_date)
        bucket = acc[key]
        bucket["record_count"] += 1
        for item in list_items(row.m_usage_stats):
            if not isinstance(item, dict):
                continue
            app = item.get("app_name")
            total_time = safe_float(item.get("total_time")) or 0.0
            bucket["event_count"] += 1
            bucket["total_time"] += total_time
            if 22 <= row.rel_hour < 27:
                bucket["late_total_time"] += total_time
            if app:
                bucket["apps"][str(app)] += max(total_time, 1.0)
    rows = [
        {
            "subject_id": subject_id,
            "row_date": row_date,
            "sem_usage_record_count": values["record_count"],
            "sem_usage_event_count": values["event_count"],
            "sem_usage_unique_apps": len(values["apps"]),
            "sem_usage_log_total_time": math.log1p(values["total_time"]),
            "sem_usage_late_total_share": ratio(values["late_total_time"], values["total_time"]),
            "sem_usage_app_entropy": entropy_from_counter(values["apps"]),
        }
        for (subject_id, row_date), values in acc.items()
    ]
    return pd.DataFrame(rows), {"source": rel(path), "raw_rows": int(len(raw)), "daily_rows": int(len(rows))}


def aggregate_proximity_features(filename: str, value_col: str, id_key: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / filename
    raw = anchored_window(pd.read_parquet(path, columns=["subject_id", "timestamp", value_col]))
    acc: dict[tuple[str, Any], dict[str, Any]] = defaultdict(
        lambda: {"record_count": 0, "item_count": 0, "late_items": 0, "ids": Counter(), "rssi": []}
    )
    for row in raw.itertuples(index=False):
        key = (str(row.subject_id), row.row_date)
        bucket = acc[key]
        bucket["record_count"] += 1
        for item in list_items(getattr(row, value_col)):
            if not isinstance(item, dict):
                continue
            bucket["item_count"] += 1
            if 22 <= row.rel_hour < 27:
                bucket["late_items"] += 1
            if item.get(id_key):
                bucket["ids"][str(item[id_key])] += 1
            rssi = safe_float(item.get("rssi"))
            if rssi is not None:
                bucket["rssi"].append(rssi)
    rows = []
    for (subject_id, row_date), values in acc.items():
        rssi_values = values["rssi"]
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                f"{prefix}_record_count": values["record_count"],
                f"{prefix}_item_count": values["item_count"],
                f"{prefix}_unique_ids": len(values["ids"]),
                f"{prefix}_late_item_share": ratio(values["late_items"], values["item_count"]),
                f"{prefix}_mean_rssi": float(np.mean(rssi_values)) if rssi_values else math.nan,
            }
        )
    return pd.DataFrame(rows), {"source": rel(path), "raw_rows": int(len(raw)), "daily_rows": int(len(rows))}


def aggregate_gps_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mGps.parquet"
    raw = anchored_window(pd.read_parquet(path, columns=["subject_id", "timestamp", "m_gps"]))
    acc: dict[tuple[str, Any], dict[str, Any]] = defaultdict(
        lambda: {"record_count": 0, "point_count": 0, "places": Counter(), "speeds": [], "moving": 0}
    )
    for row in raw.itertuples(index=False):
        key = (str(row.subject_id), row.row_date)
        bucket = acc[key]
        bucket["record_count"] += 1
        for item in list_items(row.m_gps):
            if not isinstance(item, dict):
                continue
            lat = safe_float(item.get("latitude"))
            lon = safe_float(item.get("longitude"))
            speed = safe_float(item.get("speed"))
            bucket["point_count"] += 1
            if lat is not None and lon is not None:
                bucket["places"][f"{round(lat, 3)}:{round(lon, 3)}"] += 1
            if speed is not None:
                bucket["speeds"].append(speed)
                bucket["moving"] += int(speed > 0.5)
    rows = []
    for (subject_id, row_date), values in acc.items():
        speeds = values["speeds"]
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                "sem_gps_record_count": values["record_count"],
                "sem_gps_point_count": values["point_count"],
                "sem_gps_place_bucket_count": len(values["places"]),
                "sem_gps_speed_mean": float(np.mean(speeds)) if speeds else math.nan,
                "sem_gps_moving_share": ratio(values["moving"], len(speeds)),
            }
        )
    return pd.DataFrame(rows), {"source": rel(path), "raw_rows": int(len(raw)), "daily_rows": int(len(rows))}


def aggregate_ambience_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mAmbience.parquet"
    raw = anchored_window(pd.read_parquet(path, columns=["subject_id", "timestamp", "m_ambience"]))
    acc: dict[tuple[str, Any], dict[str, Any]] = defaultdict(
        lambda: {"record_count": 0, "labels": Counter(), "top_scores": [], "stress_audio_scores": []}
    )
    for row in raw.itertuples(index=False):
        key = (str(row.subject_id), row.row_date)
        bucket = acc[key]
        bucket["record_count"] += 1
        best_score: float | None = None
        stress_score = 0.0
        for item in list_items(row.m_ambience):
            if not isinstance(item, list | tuple) or not item:
                continue
            label = str(item[0])
            score = safe_float(item[1] if len(item) > 1 else None) or 0.0
            bucket["labels"][label] += 1
            best_score = max(best_score or 0.0, score)
            if any(token in label.lower() for token in ["speech", "music", "vehicle", "car"]):
                stress_score += score
        if best_score is not None:
            bucket["top_scores"].append(best_score)
            bucket["stress_audio_scores"].append(stress_score)
    rows = []
    for (subject_id, row_date), values in acc.items():
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                "sem_amb_record_count": values["record_count"],
                "sem_amb_unique_label_count": len(values["labels"]),
                "sem_amb_label_entropy": entropy_from_counter(values["labels"]),
                "sem_amb_top_score_mean": float(np.mean(values["top_scores"])) if values["top_scores"] else math.nan,
                "sem_amb_stress_audio_score_mean": float(np.mean(values["stress_audio_scores"]))
                if values["stress_audio_scores"]
                else math.nan,
            }
        )
    return pd.DataFrame(rows), {"source": rel(path), "raw_rows": int(len(raw)), "daily_rows": int(len(rows))}


def first_rel_hour(group: pd.DataFrame, mask: pd.Series) -> float:
    values = group.loc[mask, "rel_hour"]
    return float(values.min()) if len(values) else math.nan


def last_rel_hour(group: pd.DataFrame, mask: pd.Series) -> float:
    values = group.loc[mask, "rel_hour"]
    return float(values.max()) if len(values) else math.nan


def bin_mean(group: pd.DataFrame, value_col: str, start: float, end: float) -> float:
    values = group.loc[group["rel_hour"].between(start, end, inclusive="left"), value_col]
    return float(values.mean()) if len(values) else math.nan


def aggregate_scalar_state_features(
    filename: str,
    value_col: str,
    prefix: str,
    state_transform: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / filename
    raw = anchored_window(pd.read_parquet(path))
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    if state_transform == "positive":
        raw["state"] = raw[value_col].gt(0).astype(float)
    elif state_transform == "dark":
        raw["state"] = raw[value_col].le(10).astype(float)
    elif state_transform == "moving":
        raw["state"] = (raw["step"].fillna(0).gt(0) | raw.get("speed", 0).fillna(0).gt(0.1)).astype(float)
    else:
        raise ValueError(f"Unknown state transform: {state_transform}")
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    rows = []
    for (subject_id, row_date), group in grouped:
        state = group["state"]
        row: dict[str, Any] = {
            "subject_id": str(subject_id),
            "row_date": row_date,
            f"{prefix}_record_count": len(group),
            f"{prefix}_state_share": float(state.mean()),
            f"{prefix}_late_state_share": bin_mean(group, "state", 22, 24),
            f"{prefix}_onset_state_share": bin_mean(group, "state", 24, 27),
        }
        if prefix in {"sem_screen", "sem_activity"}:
            row[f"{prefix}_last_state_rel_hour"] = last_rel_hour(group, state.gt(0))
        if prefix in {"sem_charge", "sem_wlight"}:
            row[f"{prefix}_first_state_after20"] = first_rel_hour(group, state.gt(0) & group["rel_hour"].ge(20))
        if prefix == "sem_pedo":
            row["sem_pedo_late_step_sum"] = float(
                group.loc[group["rel_hour"].between(22, 24, inclusive="left"), "step"].sum()
            )
            row["sem_pedo_deep_state_share"] = bin_mean(group, "state", 27, 31)
            row["sem_pedo_last_movement_rel_hour"] = last_rel_hour(group, state.gt(0))
        rows.append(row)
    return pd.DataFrame(rows), {"source": rel(path), "raw_rows": int(len(raw)), "daily_rows": int(len(rows))}


def aggregate_hr_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_wHr.parquet"
    raw = anchored_window(pd.read_parquet(path, columns=["subject_id", "timestamp", "heart_rate"]))
    raw["hr_mean"] = raw["heart_rate"].map(
        lambda values: np.mean([safe_float(value) for value in list_items(values) if safe_float(value) is not None])
        if list_items(values)
        else math.nan
    )
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    rows = [
        {
            "subject_id": str(subject_id),
            "row_date": row_date,
            "sem_hr_late_mean": bin_mean(group, "hr_mean", 22, 24),
            "sem_hr_onset_mean": bin_mean(group, "hr_mean", 24, 27),
        }
        for (subject_id, row_date), group in grouped
    ]
    return pd.DataFrame(rows), {"source": rel(path), "raw_rows": int(len(raw)), "daily_rows": int(len(rows))}


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


def safe_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(math.nan, index=frame.index, dtype="float64")


def add_interactions(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["sem_q3_context_novelty_load"] = np.log1p(
        safe_col(out, "sem_usage_unique_apps")
        + safe_col(out, "sem_wifi_unique_ids")
        + safe_col(out, "sem_ble_unique_ids")
        + safe_col(out, "sem_gps_place_bucket_count")
    )
    out["sem_q3_ambient_app_switch"] = safe_col(out, "sem_amb_unique_label_count") * safe_col(
        out, "sem_usage_unique_apps"
    )
    out["sem_s3_quiet_dark_late"] = (1 - safe_col(out, "sem_screen_late_state_share")) * safe_col(
        out, "sem_wlight_late_state_share"
    )
    out["sem_s3_movement_drop_late_to_deep"] = safe_col(out, "sem_pedo_late_state_share") - safe_col(
        out, "sem_pedo_deep_state_share"
    )
    return out


def add_source_missing_indicators(features: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "sem_usage_missing": ["sem_usage_"],
        "sem_social_missing": ["sem_wifi_", "sem_ble_"],
        "sem_gps_missing": ["sem_gps_"],
        "sem_amb_missing": ["sem_amb_"],
        "sem_screen_activity_missing": ["sem_screen_", "sem_activity_"],
        "sem_onset_state_missing": ["sem_charge_", "sem_wlight_", "sem_mlight_", "sem_pedo_", "sem_hr_"],
    }
    missing: dict[str, pd.Series] = {}
    for name, prefixes in groups.items():
        cols = [col for col in features.columns if any(col.startswith(prefix) for prefix in prefixes)]
        if cols:
            missing[name] = features[cols].isna().all(axis=1).astype(float)
    if not missing:
        return features
    return pd.concat([features, pd.DataFrame(missing, index=features.index)], axis=1).copy()


def build_semantic_routine_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    builders = {
        "usage": aggregate_usage_features,
        "wifi": lambda: aggregate_proximity_features("ch2025_mWifi.parquet", "m_wifi", "bssid", "sem_wifi"),
        "ble": lambda: aggregate_proximity_features("ch2025_mBle.parquet", "m_ble", "address", "sem_ble"),
        "gps": aggregate_gps_features,
        "ambience": aggregate_ambience_features,
        "screen": lambda: aggregate_scalar_state_features(
            "ch2025_mScreenStatus.parquet", "m_screen_use", "sem_screen", "positive"
        ),
        "activity": lambda: aggregate_scalar_state_features(
            "ch2025_mActivity.parquet", "m_activity", "sem_activity", "positive"
        ),
        "charge": lambda: aggregate_scalar_state_features(
            "ch2025_mACStatus.parquet", "m_charging", "sem_charge", "positive"
        ),
        "mlight": lambda: aggregate_scalar_state_features("ch2025_mLight.parquet", "m_light", "sem_mlight", "dark"),
        "wlight": lambda: aggregate_scalar_state_features("ch2025_wLight.parquet", "w_light", "sem_wlight", "dark"),
        "pedo": lambda: aggregate_scalar_state_features("ch2025_wPedo.parquet", "step", "sem_pedo", "moving"),
        "hr": aggregate_hr_features,
    }
    frames: list[pd.DataFrame] = []
    source_metadata: dict[str, Any] = {}
    for name, builder in builders.items():
        frame, metadata = builder()
        frames.append(frame)
        source_metadata[name] = metadata
    daily = add_source_missing_indicators(add_interactions(merge_feature_frames(frames)))
    keys = read_key_frame()
    features = keys.merge(daily, on=["subject_id", "row_date"], how="left")
    raw_cols = [
        col
        for col in features.columns
        if col.startswith(SOURCE_PREFIXES) and not col.endswith("_missing") and pd.api.types.is_numeric_dtype(features[col])
    ]
    total_feature_cols = [
        col for col in features.columns if col.startswith(SOURCE_PREFIXES) and pd.api.types.is_numeric_dtype(features[col])
    ]
    usable_feature_cols = [col for col in total_feature_cols if features[col].notna().any()]
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int((features["is_test"] == 0).sum()),
        "n_test_rows": int((features["is_test"] == 1).sum()),
        "raw_feature_columns": int(len(raw_cols)),
        "feature_columns": int(len(usable_feature_cols)),
        "total_semantic_numeric_columns": int(len(total_feature_cols)),
        "train_any_nonnull_row_share": float(features.loc[features["is_test"].eq(0), raw_cols].notna().any(axis=1).mean()),
        "test_any_nonnull_row_share": float(features.loc[features["is_test"].eq(1), raw_cols].notna().any(axis=1).mean()),
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), raw_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), raw_cols].isna().mean().mean()),
        "all_null_raw_features": [col for col in raw_cols if features[col].isna().all()],
        "source_metadata": source_metadata,
        "raw_identifier_columns_exported": [],
        "exact_lat_lon_columns_exported": [],
    }
    return features, coverage


def align_train_features(train: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train_features = features[features["is_test"].eq(0)].copy()
    train_features["date"] = pd.to_datetime(train_features["row_date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(
        train_features.drop(columns=["is_test"]), on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    if len(aligned) != len(train):
        raise ValueError("Semantic-routine features do not align with sorted train rows.")
    return aligned


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features.columns
        if col.startswith(SOURCE_PREFIXES)
        and pd.api.types.is_numeric_dtype(features[col])
        and features[col].notna().any()
    ]


def scan_rows_for_target(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    target: str,
    model_specs: tuple[Any, ...],
    split_names: list[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered = predict_target_for_split(features_train, feature_cols, y, folds, spec)
            source_values = references["id341"].copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), references["id341"][:, target_idx], pred)
            id178_loss = target_loss_on_split(train, folds, target, references["id178"])
            id341_loss = target_loss_on_split(train, folds, target, references["id341"])
            source_loss = target_loss_on_split(train, folds, target, source_values)
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"sem_source_{spec.name}",
                    "model_spec": asdict(spec),
                    "anchor": None,
                    "weight": None,
                    "valid_rows": int(covered),
                    "loss": source_loss,
                    "id178_loss": id178_loss,
                    "id341_loss": id341_loss,
                    "delta_vs_id178": float(source_loss - id178_loss),
                    "delta_vs_id341": float(source_loss - id341_loss),
                }
            )
            for anchor_name, anchor_values in references.items():
                for weight in BLEND_WEIGHTS:
                    blended = anchor_values.copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                        1e-6,
                        1 - 1e-6,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    anchor_loss = id178_loss if anchor_name == "id178" else id341_loss
                    rows.append(
                        {
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"blend_{anchor_name}_sem_{spec.name}_w{weight:g}",
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "valid_rows": int(covered),
                            "loss": loss,
                            "id178_loss": id178_loss,
                            "id341_loss": id341_loss,
                            "delta_vs_anchor": float(loss - anchor_loss),
                            "delta_vs_id178": float(loss - id178_loss),
                            "delta_vs_id341": float(loss - id341_loss),
                        }
                    )
    return rows


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return out
    for target in TARGET_FOCUS:
        summaries: list[dict[str, Any]] = []
        for candidate, group in frame[frame["target"].eq(target)].groupby("candidate", sort=False):
            deltas = group.set_index("split_name")["delta_vs_id341"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            delta_values = [float(v) for v in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "candidate": str(candidate),
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


def build_report(
    model_specs: tuple[Any, ...] = DEFAULT_MODEL_SPECS,
    split_names: list[str] | None = None,
) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    gate = report_status(RESOURCE_GATE)
    gate_decision = gate["decision"]
    train = load_train_sorted()
    features, coverage = build_semantic_routine_features()
    features_train = align_train_features(train, features)
    cols = feature_columns(features_train)
    splits = load_split_manifest()
    references = {"id178": load_reference_values(train, 178), "id341": load_reference_values(train, 341)}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(
            scan_rows_for_target(train, features_train, cols, splits, references, target, model_specs, split_names)
        )
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    selected_targets = sorted(selected.keys())
    gate_ready = bool(
        gate["parse_ok"]
        and gate_decision.get("local_no_artifact_model_smoke_allowed") is True
        and gate_decision.get("create_submission_artifact") is False
        and gate_decision.get("submit_now") is False
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_semantic_routine_model_smoke",
        "purpose": (
            "Run a constrained fold-internal model smoke for Q3/S3 semantic novelty and sleep-onset state-machine "
            "features against ID178/ID341 OOF anchors."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "resource_gate": gate,
            "id178_oof": rel(Path("artifacts/base_predictions") / "base_oof_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638.csv"),
            "id341_oof": rel(Path("artifacts/base_predictions") / "base_oof_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv"),
        },
        "feature_coverage": coverage,
        "config": {
            "model_specs": [asdict(spec) for spec in model_specs],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "selected_target_configs": selected,
        "top_rows_by_target": {target: rows[:8] for target, rows in summaries.items()},
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "model_smoke_completed": True,
            "resource_gate_ready": gate_ready,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "q3_signal_found": "Q3" in selected,
            "s3_signal_found": "S3" in selected,
            "all_q3s3_targets_covered": selected_targets == TARGET_FOCUS,
            "public_058_evidence": False,
            "next_local_allowed_step": "no_artifact_semantic_routine_geometry_gate"
            if selected_targets
            else "close_semantic_routine_model_smoke",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If selected targets exist, run only a no-artifact geometry gate. Otherwise close this local lane and "
                "return to the explicit-approved upstream rebuild path for Q3/S3."
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
        "# Q3/S3 semantic-routine model smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Raw feature columns: `{coverage['raw_feature_columns']}`; total feature columns: `{coverage['feature_columns']}`",
        f"- Train/test non-null row share: `{coverage['train_any_nonnull_row_share']:.3f}` / `{coverage['test_any_nonnull_row_share']:.3f}`",
        f"- Train/test missing mean: `{coverage['train_missing_mean']:.3f}` / `{coverage['test_missing_mean']:.3f}`",
        f"- All-null raw features: `{coverage['all_null_raw_features']}`",
        f"- Raw identifier columns exported: `{coverage['raw_identifier_columns_exported']}`",
        f"- Exact lat/lon columns exported: `{coverage['exact_lat_lon_columns_exported']}`",
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
            f"- Resource gate ready: `{decision['resource_gate_ready']}`",
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
