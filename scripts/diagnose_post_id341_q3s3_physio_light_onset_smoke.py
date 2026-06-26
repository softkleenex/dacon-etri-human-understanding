#!/usr/bin/env python3
"""Q3/S3 physio-light-onset source smoke after the post-ID341 Q2 lead.

This diagnostic tests a materially different Q3/S3 source family from the
Q2-focused env/social/routine lead: wearable heart-rate, wearable/mobile light,
pedometer movement, and charging/onset transition features. It remains
report-only and compares small fold-internal source models against ID178/ID341
OOF anchors on frozen validation lenses.

It does not create full-test predictions, base prediction artifacts, submission
CSVs, notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or
DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_physio_light_onset_smoke_20260618.json"
TARGET_FOCUS = ["Q3", "S3"]
SOURCE_PREFIXES = ("plo_hr_", "plo_pedo_", "plo_wlight_", "plo_mlight_", "plo_charge_", "plo_onset_")
TIME_BINS = {
    "afternoon_12_18": (12.0, 18.0),
    "evening_18_22": (18.0, 22.0),
    "late_22_24": (22.0, 24.0),
    "onset_24_27": (24.0, 27.0),
    "deep_27_31": (27.0, 31.0),
    "morning_31_36": (31.0, 36.0),
}


def rel(path: Path) -> str:
    if not path.is_absolute():
        path = ROOT / path
    return str(path.relative_to(ROOT))


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
        return list_items(value.as_py())
    return []


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


def read_key_frame() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv", usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv", usecols=["subject_id", "lifelog_date"])
    train = train.rename(columns={"lifelog_date": "row_date"}).assign(is_test=0)
    test = test.rename(columns={"lifelog_date": "row_date"}).assign(is_test=1)
    keys = pd.concat([train, test], ignore_index=True)
    keys["subject_id"] = keys["subject_id"].astype(str)
    keys["row_date"] = pd.to_datetime(keys["row_date"]).dt.date
    return keys


def numeric_stats(values: list[float]) -> dict[str, float]:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return {"count": 0.0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": float(len(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def first_rel_hour(group: pd.DataFrame, mask: pd.Series) -> float:
    values = group.loc[mask, "rel_hour"]
    return float(values.min()) if len(values) else math.nan


def last_rel_hour(group: pd.DataFrame, mask: pd.Series) -> float:
    values = group.loc[mask, "rel_hour"]
    return float(values.max()) if len(values) else math.nan


def aggregate_hr_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_wHr.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "heart_rate"])
    raw = anchored_window(raw)
    row_stats = raw["heart_rate"].map(lambda values: numeric_stats([float(x) for x in list_items(values)]))
    stats_frame = pd.DataFrame(row_stats.tolist(), index=raw.index)
    raw = pd.concat([raw, stats_frame.add_prefix("hr_row_")], axis=1)
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    out = grouped.agg(
        plo_hr_record_count=("hr_row_count", "size"),
        plo_hr_sample_count=("hr_row_count", "sum"),
        plo_hr_mean=("hr_row_mean", "mean"),
        plo_hr_std_mean=("hr_row_std", "mean"),
        plo_hr_min=("hr_row_min", "min"),
        plo_hr_max=("hr_row_max", "max"),
    ).reset_index()
    raw["plo_hr_high_row"] = (raw["hr_row_mean"] >= 90).astype(float)
    raw["plo_hr_low_row"] = (raw["hr_row_mean"] <= 60).astype(float)
    high_low = grouped.agg(
        plo_hr_high_row_share=("plo_hr_high_row", "mean"), plo_hr_low_row_share=("plo_hr_low_row", "mean")
    )
    out = out.merge(high_low.reset_index(), on=["subject_id", "row_date"], how="left")
    for bin_name in TIME_BINS:
        subset = raw[raw["bin_name"].eq(bin_name)]
        if subset.empty:
            continue
        stats = subset.groupby(["subject_id", "row_date"], sort=False).agg(
            **{
                f"plo_hr_{bin_name}_mean": ("hr_row_mean", "mean"),
                f"plo_hr_{bin_name}_std_mean": ("hr_row_std", "mean"),
                f"plo_hr_{bin_name}_high_share": ("plo_hr_high_row", "mean"),
            }
        )
        out = out.merge(stats.reset_index(), on=["subject_id", "row_date"], how="left")
    out["plo_onset_hr_late_minus_deep"] = out.get("plo_hr_late_22_24_mean", np.nan) - out.get(
        "plo_hr_deep_27_31_mean", np.nan
    )
    out["plo_onset_hr_evening_minus_onset"] = out.get("plo_hr_evening_18_22_mean", np.nan) - out.get(
        "plo_hr_onset_24_27_mean", np.nan
    )
    return out, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len([col for col in out.columns if col.startswith(("plo_hr_", "plo_onset_hr_"))])),
        "hr_samples_observed": int(raw["hr_row_count"].sum()),
    }


def aggregate_pedo_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_wPedo.parquet"
    cols = ["subject_id", "timestamp", "step", "step_frequency", "running_step", "walking_step", "distance", "speed"]
    raw = pd.read_parquet(path, columns=cols)
    raw = anchored_window(raw)
    for col in cols[2:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["moving"] = ((raw["step"].fillna(0) > 0) | (raw["speed"].fillna(0) > 0.1)).astype(float)
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    out = grouped.agg(
        plo_pedo_record_count=("step", "size"),
        plo_pedo_step_sum=("step", "sum"),
        plo_pedo_step_frequency_mean=("step_frequency", "mean"),
        plo_pedo_distance_sum=("distance", "sum"),
        plo_pedo_speed_mean=("speed", "mean"),
        plo_pedo_moving_share=("moving", "mean"),
    ).reset_index()
    last_move = grouped.apply(lambda group: last_rel_hour(group, group["moving"].gt(0)), include_groups=False)
    out = out.merge(
        last_move.rename("plo_pedo_last_movement_rel_hour").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    for bin_name in TIME_BINS:
        subset = raw[raw["bin_name"].eq(bin_name)]
        if subset.empty:
            continue
        stats = subset.groupby(["subject_id", "row_date"], sort=False).agg(
            **{
                f"plo_pedo_{bin_name}_step_sum": ("step", "sum"),
                f"plo_pedo_{bin_name}_moving_share": ("moving", "mean"),
                f"plo_pedo_{bin_name}_speed_mean": ("speed", "mean"),
            }
        )
        out = out.merge(stats.reset_index(), on=["subject_id", "row_date"], how="left")
    out["plo_onset_late_movement_minus_deep"] = out.get("plo_pedo_late_22_24_moving_share", np.nan) - out.get(
        "plo_pedo_deep_27_31_moving_share", np.nan
    )
    return out, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len([col for col in out.columns if col.startswith(("plo_pedo_", "plo_onset_late_"))])),
    }


def aggregate_light_features(filename: str, value_col: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / filename
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", value_col])
    raw = anchored_window(raw)
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw["dark"] = (raw[value_col].fillna(np.inf) <= 10).astype(float)
    raw["dim"] = (raw[value_col].fillna(np.inf) <= 50).astype(float)
    raw["bright"] = (raw[value_col].fillna(-np.inf) >= 100).astype(float)
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    out = grouped.agg(
        **{
            f"{prefix}_record_count": (value_col, "size"),
            f"{prefix}_mean": (value_col, "mean"),
            f"{prefix}_max": (value_col, "max"),
            f"{prefix}_dark_share": ("dark", "mean"),
            f"{prefix}_dim_share": ("dim", "mean"),
            f"{prefix}_bright_share": ("bright", "mean"),
        }
    ).reset_index()
    first_dark = grouped.apply(
        lambda group: first_rel_hour(group, group["dark"].gt(0) & group["rel_hour"].ge(20)), include_groups=False
    )
    last_bright = grouped.apply(
        lambda group: last_rel_hour(group, group["bright"].gt(0) & group["rel_hour"].le(30)), include_groups=False
    )
    out = out.merge(
        first_dark.rename(f"{prefix}_first_dark_after20").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    out = out.merge(
        last_bright.rename(f"{prefix}_last_bright_before30").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    for bin_name in TIME_BINS:
        subset = raw[raw["bin_name"].eq(bin_name)]
        if subset.empty:
            continue
        stats = subset.groupby(["subject_id", "row_date"], sort=False).agg(
            **{
                f"{prefix}_{bin_name}_mean": (value_col, "mean"),
                f"{prefix}_{bin_name}_dark_share": ("dark", "mean"),
                f"{prefix}_{bin_name}_bright_share": ("bright", "mean"),
            }
        )
        out = out.merge(stats.reset_index(), on=["subject_id", "row_date"], how="left")
    return out, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len([col for col in out.columns if col.startswith(prefix)])),
    }


def aggregate_charge_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / "ch2025_mACStatus.parquet"
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_charging"])
    raw = anchored_window(raw)
    raw["charging"] = (pd.to_numeric(raw["m_charging"], errors="coerce").fillna(0) > 0).astype(float)
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    grouped = raw.groupby(["subject_id", "row_date"], sort=False)
    out = grouped.agg(
        plo_charge_record_count=("charging", "size"),
        plo_charge_share=("charging", "mean"),
    ).reset_index()
    first_charge = grouped.apply(
        lambda group: first_rel_hour(group, group["charging"].gt(0) & group["rel_hour"].ge(18)), include_groups=False
    )
    out = out.merge(
        first_charge.rename("plo_charge_first_after18").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    transition = grouped["charging"].apply(lambda series: float(series.diff().abs().fillna(0).sum()))
    out = out.merge(
        transition.rename("plo_charge_transition_count").reset_index(), on=["subject_id", "row_date"], how="left"
    )
    for bin_name in TIME_BINS:
        subset = raw[raw["bin_name"].eq(bin_name)]
        if subset.empty:
            continue
        stats = subset.groupby(["subject_id", "row_date"], sort=False).agg(
            **{f"plo_charge_{bin_name}_share": ("charging", "mean")}
        )
        out = out.merge(stats.reset_index(), on=["subject_id", "row_date"], how="left")
    return out, {
        "source": path.name,
        "raw_rows": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len([col for col in out.columns if col.startswith("plo_charge_")])),
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


def safe_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def add_interactions(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["plo_onset_light_late_x_hr_late"] = safe_col(out, "plo_wlight_late_22_24_bright_share") * safe_col(
        out, "plo_hr_late_22_24_mean"
    )
    out["plo_onset_light_onset_x_hr_onset"] = safe_col(out, "plo_wlight_onset_24_27_bright_share") * safe_col(
        out, "plo_hr_onset_24_27_mean"
    )
    out["plo_onset_movement_x_light_late"] = safe_col(out, "plo_pedo_late_22_24_moving_share") * safe_col(
        out, "plo_wlight_late_22_24_bright_share"
    )
    out["plo_onset_charge_x_dark"] = safe_col(out, "plo_charge_late_22_24_share") * safe_col(
        out, "plo_wlight_late_22_24_dark_share"
    )
    out["plo_onset_hr_drop_x_dark"] = safe_col(out, "plo_onset_hr_late_minus_deep") * safe_col(
        out, "plo_wlight_deep_27_31_dark_share"
    )
    return out


def add_missing_indicators(features: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = {f"{col}_missing": features[col].isna().astype(float) for col in cols}
    if not missing:
        return features
    return pd.concat([features, pd.DataFrame(missing, index=features.index)], axis=1).copy()


def build_physio_light_onset_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    source_frames: list[pd.DataFrame] = []
    source_metadata: dict[str, Any] = {}
    builders = {
        "hr": aggregate_hr_features,
        "pedo": aggregate_pedo_features,
        "wlight": lambda: aggregate_light_features("ch2025_wLight.parquet", "w_light", "plo_wlight"),
        "mlight": lambda: aggregate_light_features("ch2025_mLight.parquet", "m_light", "plo_mlight"),
        "charge": aggregate_charge_features,
    }
    for name, builder in builders.items():
        frame, metadata = builder()
        source_frames.append(frame)
        source_metadata[name] = metadata
    daily = add_interactions(merge_feature_frames(source_frames))
    keys = read_key_frame()
    features = keys.merge(daily, on=["subject_id", "row_date"], how="left")
    raw_cols = [col for col in features.columns if col.startswith(SOURCE_PREFIXES)]
    features = add_missing_indicators(features, raw_cols)
    feature_cols = [col for col in features.columns if col.startswith(SOURCE_PREFIXES)]
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int((features["is_test"] == 0).sum()),
        "n_test_rows": int((features["is_test"] == 1).sum()),
        "raw_feature_columns": int(len(raw_cols)),
        "feature_columns": int(len(feature_cols)),
        "train_any_nonnull_row_share": float(
            features.loc[features["is_test"].eq(0), raw_cols].notna().any(axis=1).mean()
        ),
        "test_any_nonnull_row_share": float(
            features.loc[features["is_test"].eq(1), raw_cols].notna().any(axis=1).mean()
        ),
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), raw_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), raw_cols].isna().mean().mean()),
        "all_null_raw_features": [col for col in raw_cols if features[col].isna().all()],
        "source_metadata": source_metadata,
    }
    return features, coverage


def align_train_features(train: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train_features = features[features["is_test"].eq(0)].copy()
    train_features["date"] = pd.to_datetime(train_features["row_date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(
        train_features.drop(columns=["is_test"]), on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    if len(aligned) != len(train):
        raise ValueError("Physio-light-onset features do not align with sorted train rows.")
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
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for split_name in PRIMARY_GATE_SPLITS:
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
                    "candidate": f"plo_source_{spec.name}",
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
                            "candidate": f"blend_{anchor_name}_plo_{spec.name}_w{weight:g}",
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


def build_report() -> dict[str, Any]:
    train = load_train_sorted()
    features, coverage = build_physio_light_onset_features()
    features_train = align_train_features(train, features)
    cols = feature_columns(features_train)
    splits = load_split_manifest()
    references = {"id178": load_reference_values(train, 178), "id341": load_reference_values(train, 341)}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(scan_rows_for_target(train, features_train, cols, splits, references, target))
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    selected_targets = sorted(selected.keys())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_physio_light_onset_smoke",
        "purpose": (
            "Test wearable heart-rate, light, movement, and charging/onset transition features as a new Q3/S3 source "
            "after the Q2-only env/social/routine lead."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "raw_sources": [
                rel(ITEMS_DIR / "ch2025_wHr.parquet"),
                rel(ITEMS_DIR / "ch2025_wPedo.parquet"),
                rel(ITEMS_DIR / "ch2025_wLight.parquet"),
                rel(ITEMS_DIR / "ch2025_mLight.parquet"),
                rel(ITEMS_DIR / "ch2025_mACStatus.parquet"),
            ],
            "id178_oof": rel(
                Path(
                    "artifacts/base_predictions/base_oof_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638.csv"
                )
            ),
            "id341_oof": rel(
                Path(
                    "artifacts/base_predictions/base_oof_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv"
                )
            ),
        },
        "feature_coverage": coverage,
        "config": {
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "split_names": PRIMARY_GATE_SPLITS,
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
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "q3_signal_found": "Q3" in selected,
            "s3_signal_found": "S3" in selected,
            "all_q3s3_targets_covered": selected_targets == TARGET_FOCUS,
            "public_058_evidence": False,
            "next_local_allowed_step": "no_artifact_geometry_gate"
            if selected_targets
            else "close_physio_light_onset_lane",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If a selected target appears, run only a no-artifact geometry gate. Otherwise close this local lane and "
                "return to the explicit-approved upstream rebuild path for missing Q3/S3 coverage."
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
    coverage = report["feature_coverage"]
    decision = report["decision"]
    lines = [
        "# Q3/S3 physio-light-onset smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Raw feature columns: `{coverage['raw_feature_columns']}`; total with missing flags: `{coverage['feature_columns']}`",
        f"- Train/test non-null row share: `{coverage['train_any_nonnull_row_share']:.3f}` / `{coverage['test_any_nonnull_row_share']:.3f}`",
        f"- Train/test missing mean: `{coverage['train_missing_mean']:.3f}` / `{coverage['test_missing_mean']:.3f}`",
        f"- All-null raw features: `{coverage['all_null_raw_features']}`",
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
