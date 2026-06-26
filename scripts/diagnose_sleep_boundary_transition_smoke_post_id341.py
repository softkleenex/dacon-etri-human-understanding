#!/usr/bin/env python3
"""Report-only sleep-boundary transition source smoke after ID341.

This diagnostic implements the first genuinely new post-ID341 source family from
`post_id341_new_source_blueprint_20260618`: sleep_boundary_transition_v1.
It extracts raw, label-free event-boundary features from screen/activity/charge/
light/heart-rate/pedometer logs, evaluates small source models inside the frozen
ID340 split manifest, and compares against ID178/ID341 OOF references.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, and no DACON submission.
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

from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
TRANSFER_AUDIT_PATH = REPORTS_DIR / "post_id341_public_transfer_audit_20260618.json"
NEW_SOURCE_BLUEPRINT_PATH = REPORTS_DIR / "post_id341_new_source_blueprint_20260618.json"
EPS = 1e-6
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
BLEND_WEIGHTS = [0.01, 0.03, 0.05, 0.10, 0.20]
SLEEP_TARGETS = ["Q1", "S1", "S2", "S3", "S4"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_iter: int | None = None
    max_leaf_nodes: int | None = None
    l2_regularization: float | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_05", "lr", c_value=0.05),
    ModelSpec("lr_c0_2", "lr", c_value=0.2),
    ModelSpec("hgb_l5_l2_2", "hgb", max_iter=70, max_leaf_nodes=5, l2_regularization=2.0),
)


def read_required_reports() -> dict[str, Any]:
    return {
        "split_manifest_exists": SPLIT_MANIFEST_PATH.exists(),
        "transfer_audit_exists": TRANSFER_AUDIT_PATH.exists(),
        "new_source_blueprint_exists": NEW_SOURCE_BLUEPRINT_PATH.exists(),
    }


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


def add_window_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    row_start = out["timestamp"].dt.normalize() - pd.to_timedelta((out["timestamp"].dt.hour < 12).astype(int), unit="D")
    out["row_date"] = row_start.dt.date
    out["rel_hour"] = (out["timestamp"] - row_start).dt.total_seconds() / 3600.0
    return out[out["rel_hour"].between(12.0, 36.0, inclusive="left")]


def empty_indexed_frame() -> pd.DataFrame:
    return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["subject_id", "row_date"]))


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = empty_indexed_frame()
    for frame in frames:
        if frame.empty:
            continue
        if out.empty:
            out = frame.copy()
        else:
            out = out.join(frame, how="outer")
    return out.reset_index()


def bin_mask(df: pd.DataFrame, start: float, end: float) -> pd.Series:
    return df["rel_hour"].ge(start) & df["rel_hour"].lt(end)


def grouped_mean_sum_count(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    grouped = df.groupby(["subject_id", "row_date"])[value_col]
    return grouped.agg(
        **{
            f"{prefix}_count": "count",
            f"{prefix}_sum": "sum",
            f"{prefix}_mean": "mean",
        }
    )


def aggregate_binary_bins(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    frames = [grouped_mean_sum_count(df, value_col, f"{prefix}_all_12_36")]
    for name, start, end in [
        ("eve_18_22", 18, 22),
        ("late_22_24", 22, 24),
        ("night_24_30", 24, 30),
        ("morn_30_36", 30, 36),
    ]:
        subset = df[bin_mask(df, start, end)]
        if not subset.empty:
            frames.append(grouped_mean_sum_count(subset, value_col, f"{prefix}_{name}"))
    return merge_indexed(frames)


def merge_indexed(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame.copy() if out is None else out.join(frame, how="outer")
    return out if out is not None else empty_indexed_frame()


def longest_true_run_minutes(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def aggregate_screen() -> pd.DataFrame:
    path = RAW_DIR / "ch2025_mScreenStatus.parquet"
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_screen_use"])
    df = add_window_keys(df)
    df["screen_on"] = df["m_screen_use"].astype(float)
    frames = [aggregate_binary_bins(df, "screen_on", "sbt_screen")]
    evening_on = df[df["screen_on"].gt(0) & bin_mask(df, 18, 30)]
    morning_on = df[df["screen_on"].gt(0) & bin_mask(df, 27, 36)]
    if not evening_on.empty:
        frames.append(
            evening_on.groupby(["subject_id", "row_date"])["rel_hour"].max().to_frame("sbt_screen_last_on_18_30")
        )
    if not morning_on.empty:
        frames.append(
            morning_on.groupby(["subject_id", "row_date"])["rel_hour"].min().to_frame("sbt_screen_first_on_27_36")
        )
    out = merge_indexed(frames)
    if {"sbt_screen_first_on_27_36", "sbt_screen_last_on_18_30"}.issubset(out.columns):
        out["sbt_screen_quiet_gap_hours"] = out["sbt_screen_first_on_27_36"] - out["sbt_screen_last_on_18_30"]
    return out


def aggregate_activity() -> pd.DataFrame:
    path = RAW_DIR / "ch2025_mActivity.parquet"
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_activity"])
    df = add_window_keys(df)
    df["still"] = df["m_activity"].eq(3).astype(float)
    df["active"] = df["m_activity"].isin([0, 1, 2, 7, 8]).astype(float)
    df["unknown"] = df["m_activity"].eq(4).astype(float)
    frames = [
        aggregate_binary_bins(df, "still", "sbt_act_still"),
        aggregate_binary_bins(df, "active", "sbt_act_active"),
        aggregate_binary_bins(df, "unknown", "sbt_act_unknown"),
    ]
    night = df[bin_mask(df, 20, 34)].sort_values(["subject_id", "row_date", "timestamp"])
    if not night.empty:
        longest = night.groupby(["subject_id", "row_date"])["still"].apply(
            lambda series: longest_true_run_minutes(series.to_numpy())
        )
        frames.append(longest.to_frame("sbt_act_longest_still_run_minutes_20_34"))
    return merge_indexed(frames)


def aggregate_charge() -> pd.DataFrame:
    path = RAW_DIR / "ch2025_mACStatus.parquet"
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "m_charging"])
    df = add_window_keys(df)
    df["charging"] = df["m_charging"].astype(float)
    frames = [aggregate_binary_bins(df, "charging", "sbt_charge")]
    charging = df[df["charging"].gt(0)]
    if not charging.empty:
        frames.append(
            charging[bin_mask(charging, 18, 30)]
            .groupby(["subject_id", "row_date"])["rel_hour"]
            .min()
            .to_frame("sbt_charge_first_on_18_30")
        )
        frames.append(
            charging[bin_mask(charging, 18, 36)]
            .groupby(["subject_id", "row_date"])["rel_hour"]
            .max()
            .to_frame("sbt_charge_last_on_18_36")
        )
    return merge_indexed(frames)


def aggregate_pedo() -> pd.DataFrame:
    path = RAW_DIR / "ch2025_wPedo.parquet"
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "step", "step_frequency"])
    df = add_window_keys(df)
    df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(float)
    df["has_step"] = df["step"].gt(0).astype(float)
    df["step_frequency"] = pd.to_numeric(df["step_frequency"], errors="coerce")
    frames = []
    grouped = df.groupby(["subject_id", "row_date"])
    frames.append(
        grouped.agg(
            sbt_pedo_count=("step", "count"),
            sbt_pedo_step_sum_12_36=("step", "sum"),
            sbt_pedo_step_max_12_36=("step", "max"),
            sbt_pedo_step_freq_mean_12_36=("step_frequency", "mean"),
            sbt_pedo_has_step_ratio_12_36=("has_step", "mean"),
        )
    )
    for name, start, end in [
        ("eve_18_22", 18, 22),
        ("late_22_24", 22, 24),
        ("night_24_30", 24, 30),
        ("morn_30_36", 30, 36),
    ]:
        subset = df[bin_mask(df, start, end)]
        if not subset.empty:
            frames.append(
                subset.groupby(["subject_id", "row_date"]).agg(
                    **{
                        f"sbt_pedo_step_sum_{name}": ("step", "sum"),
                        f"sbt_pedo_has_step_ratio_{name}": ("has_step", "mean"),
                    }
                )
            )
    stepped = df[df["has_step"].gt(0)]
    if not stepped.empty:
        frames.append(
            stepped[bin_mask(stepped, 18, 30)]
            .groupby(["subject_id", "row_date"])["rel_hour"]
            .max()
            .to_frame("sbt_pedo_last_step_18_30")
        )
        frames.append(
            stepped[bin_mask(stepped, 27, 36)]
            .groupby(["subject_id", "row_date"])["rel_hour"]
            .min()
            .to_frame("sbt_pedo_first_step_27_36")
        )
    out = merge_indexed(frames)
    if {"sbt_pedo_first_step_27_36", "sbt_pedo_last_step_18_30"}.issubset(out.columns):
        out["sbt_pedo_low_step_gap_hours"] = out["sbt_pedo_first_step_27_36"] - out["sbt_pedo_last_step_18_30"]
    return out


def aggregate_light_file(filename: str, value_col: str, prefix: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", value_col])
    df = add_window_keys(df)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df["dark"] = df[value_col].le(10).astype(float)
    df["bright"] = df[value_col].ge(100).astype(float)
    frames = []
    grouped = df.groupby(["subject_id", "row_date"])
    frames.append(
        grouped.agg(
            **{
                f"{prefix}_count": (value_col, "count"),
                f"{prefix}_mean_12_36": (value_col, "mean"),
                f"{prefix}_median_12_36": (value_col, "median"),
                f"{prefix}_min_12_36": (value_col, "min"),
                f"{prefix}_max_12_36": (value_col, "max"),
                f"{prefix}_dark_ratio_12_36": ("dark", "mean"),
                f"{prefix}_bright_ratio_12_36": ("bright", "mean"),
            }
        )
    )
    for name, start, end in [
        ("eve_18_22", 18, 22),
        ("late_22_24", 22, 24),
        ("night_24_30", 24, 30),
        ("morn_30_36", 30, 36),
    ]:
        subset = df[bin_mask(df, start, end)]
        if not subset.empty:
            frames.append(
                subset.groupby(["subject_id", "row_date"]).agg(
                    **{
                        f"{prefix}_mean_{name}": (value_col, "mean"),
                        f"{prefix}_dark_ratio_{name}": ("dark", "mean"),
                    }
                )
            )
    return merge_indexed(frames)


def heart_rate_mean(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.nan
    if arr.size == 0:
        return np.nan
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= 30) & (arr <= 220)]
    return float(np.mean(arr)) if arr.size else np.nan


def aggregate_hr() -> pd.DataFrame:
    path = RAW_DIR / "ch2025_wHr.parquet"
    df = pd.read_parquet(path, columns=["subject_id", "timestamp", "heart_rate"])
    df = add_window_keys(df)
    df["hr_mean"] = df["heart_rate"].map(heart_rate_mean)
    df = df[df["hr_mean"].notna()]
    frames = []
    grouped = df.groupby(["subject_id", "row_date"])
    frames.append(
        grouped.agg(
            sbt_hr_count=("hr_mean", "count"),
            sbt_hr_mean_12_36=("hr_mean", "mean"),
            sbt_hr_std_12_36=("hr_mean", "std"),
            sbt_hr_min_12_36=("hr_mean", "min"),
            sbt_hr_max_12_36=("hr_mean", "max"),
        )
    )
    for name, start, end in [
        ("eve_18_22", 18, 22),
        ("late_22_24", 22, 24),
        ("night_24_30", 24, 30),
        ("morn_30_36", 30, 36),
    ]:
        subset = df[bin_mask(df, start, end)]
        if not subset.empty:
            frames.append(
                subset.groupby(["subject_id", "row_date"]).agg(
                    **{
                        f"sbt_hr_mean_{name}": ("hr_mean", "mean"),
                        f"sbt_hr_min_{name}": ("hr_mean", "min"),
                        f"sbt_hr_std_{name}": ("hr_mean", "std"),
                    }
                )
            )
    min_rows = df.loc[df.groupby(["subject_id", "row_date"])["hr_mean"].idxmin()]
    if not min_rows.empty:
        frames.append(
            min_rows.set_index(["subject_id", "row_date"])[["rel_hour"]].rename(
                columns={"rel_hour": "sbt_hr_min_rel_hour"}
            )
        )
    out = merge_indexed(frames)
    if {"sbt_hr_mean_morn_30_36", "sbt_hr_mean_night_24_30"}.issubset(out.columns):
        out["sbt_hr_morning_minus_night"] = out["sbt_hr_mean_morn_30_36"] - out["sbt_hr_mean_night_24_30"]
    if {"sbt_hr_mean_eve_18_22", "sbt_hr_mean_night_24_30"}.issubset(out.columns):
        out["sbt_hr_evening_minus_night"] = out["sbt_hr_mean_eve_18_22"] - out["sbt_hr_mean_night_24_30"]
    return out


def build_sleep_boundary_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    )
    rows["row_date"] = pd.to_datetime(rows["date"]).dt.date
    feature_frames = [
        aggregate_screen(),
        aggregate_activity(),
        aggregate_charge(),
        aggregate_pedo(),
        aggregate_light_file("ch2025_mLight.parquet", "m_light", "sbt_mlight"),
        aggregate_light_file("ch2025_wLight.parquet", "w_light", "sbt_wlight"),
        aggregate_hr(),
    ]
    raw_features = merge_feature_frames(feature_frames)
    features = rows.merge(raw_features, on=["subject_id", "row_date"], how="left")
    numeric_cols = [
        col for col in features.columns if col.startswith("sbt_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    for col in numeric_cols:
        features[f"{col}_missing"] = features[col].isna().astype(float)
    feature_cols = [col for col in features.columns if col.startswith("sbt_")]
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int((features["is_test"] == 0).sum()),
        "n_test_rows": int((features["is_test"] == 1).sum()),
        "n_feature_columns": int(len(feature_cols)),
        "n_raw_numeric_columns": int(len(numeric_cols)),
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), feature_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), feature_cols].isna().mean().mean()),
        "top_missing_features": {
            str(k): float(v)
            for k, v in features.loc[features["is_test"].eq(0), feature_cols]
            .isna()
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .items()
        },
    }
    return features, coverage


def find_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base OOF ID{candidate_id}")
    return paths[0]


def load_reference_values(train: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_oof(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.date
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=400, random_state=43),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=int(spec.max_iter or 70),
                learning_rate=0.03,
                max_leaf_nodes=int(spec.max_leaf_nodes or 5),
                l2_regularization=float(spec.l2_regularization or 2.0),
                min_samples_leaf=12,
                random_state=43,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


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
    fit_devs: dict[str, pd.Series] = {}
    query_devs: dict[str, pd.Series] = {}
    for col in usable_cols:
        global_median = fit[col].median()
        if pd.isna(global_median):
            global_median = 0.0
        subject_medians = fit.groupby("subject_id")[col].median()
        fit_subject_base = fit["subject_id"].map(subject_medians).fillna(global_median).reset_index(drop=True)
        query_subject_base = query["subject_id"].map(subject_medians).fillna(global_median).reset_index(drop=True)
        fit_devs[f"{col}_dev_subject"] = fit[col].reset_index(drop=True) - fit_subject_base
        query_devs[f"{col}_dev_subject"] = query[col].reset_index(drop=True) - query_subject_base
    fit_x = pd.concat([fit_raw, pd.DataFrame(fit_devs)], axis=1)
    query_x = pd.concat([query_raw, pd.DataFrame(query_devs)], axis=1)
    return fit_x, query_x


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


def evaluate_values_on_split(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    values: np.ndarray,
) -> dict[str, Any]:
    target_losses: dict[str, float] = {}
    target_valid_n: dict[str, int] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        total = 0
        weighted = 0.0
        for _fit_idx, valid_idx in folds:
            pred = values[valid_idx, target_idx]
            if np.isnan(pred).any():
                raise ValueError(f"NaN predictions for target {target} on split evaluation.")
            loss = fold_loss(y[valid_idx], pred)
            weighted += loss * len(valid_idx)
            total += len(valid_idx)
        target_losses[target] = float(weighted / total)
        target_valid_n[target] = int(total)
    return {
        "mean_loss": float(np.mean(list(target_losses.values()))),
        "target_losses": target_losses,
        "target_valid_n": target_valid_n,
    }


def evaluate_model_on_split(
    split_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> dict[str, Any]:
    id178_eval = evaluate_values_on_split(train, folds, reference178)
    id341_eval = evaluate_values_on_split(train, folds, reference341)
    candidate_rows: list[dict[str, Any]] = []
    target_best: dict[str, dict[str, Any]] = {
        target: {
            "target": target,
            "best_candidate": "ID341_reference",
            "best_loss": id341_eval["target_losses"][target],
            "delta_vs_id341": 0.0,
        }
        for target in TARGETS
    }
    for spec in MODEL_SPECS:
        source_values = np.zeros((len(train), len(TARGETS)), dtype=float)
        covered_by_target: dict[str, int] = {}
        for target_idx, target in enumerate(TARGETS):
            y = train[target].astype(int).to_numpy()
            pred, covered = predict_target_for_split(features_train, feature_cols, y, folds, spec)
            # Non-covered rows are irrelevant for this split's validation folds; keep anchor fallback for safety.
            source_values[:, target_idx] = np.where(np.isnan(pred), reference341[:, target_idx], pred)
            covered_by_target[target] = int(covered)
        source_eval = evaluate_values_on_split(train, folds, source_values)
        candidate_rows.append(
            {
                "candidate": f"sleep_boundary_source_{spec.name}",
                "model_spec": asdict(spec),
                "kind": "raw_sleep_boundary_source",
                "anchor": None,
                "weight": None,
                "mean_loss": source_eval["mean_loss"],
                "delta_vs_id178": float(source_eval["mean_loss"] - id178_eval["mean_loss"]),
                "delta_vs_id341": float(source_eval["mean_loss"] - id341_eval["mean_loss"]),
                "target_losses": source_eval["target_losses"],
                "covered_by_target": covered_by_target,
            }
        )
        for anchor_name, anchor_values, _anchor_eval in [
            ("id178", reference178, id178_eval),
            ("id341", reference341, id341_eval),
        ]:
            for weight in BLEND_WEIGHTS:
                blend_values = np.clip((1.0 - weight) * anchor_values + weight * source_values, EPS, 1 - EPS)
                blend_eval = evaluate_values_on_split(train, folds, blend_values)
                candidate_rows.append(
                    {
                        "candidate": f"blend_{anchor_name}_sleep_boundary_{spec.name}_w{weight:g}",
                        "model_spec": asdict(spec),
                        "kind": "fixed_anchor_blend",
                        "anchor": anchor_name,
                        "weight": weight,
                        "mean_loss": blend_eval["mean_loss"],
                        "delta_vs_id178": float(blend_eval["mean_loss"] - id178_eval["mean_loss"]),
                        "delta_vs_id341": float(blend_eval["mean_loss"] - id341_eval["mean_loss"]),
                        "target_losses": blend_eval["target_losses"],
                    }
                )
                for target in TARGETS:
                    loss = blend_eval["target_losses"][target]
                    if loss < target_best[target]["best_loss"]:
                        target_best[target] = {
                            "target": target,
                            "best_candidate": f"blend_{anchor_name}_sleep_boundary_{spec.name}_w{weight:g}",
                            "best_loss": float(loss),
                            "delta_vs_id341": float(loss - id341_eval["target_losses"][target]),
                        }
        for target in TARGETS:
            loss = source_eval["target_losses"][target]
            if loss < target_best[target]["best_loss"]:
                target_best[target] = {
                    "target": target,
                    "best_candidate": f"sleep_boundary_source_{spec.name}",
                    "best_loss": float(loss),
                    "delta_vs_id341": float(loss - id341_eval["target_losses"][target]),
                }
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["mean_loss"], row["candidate"]))
    best = candidate_rows[0]
    return {
        "split": split_name,
        "n_folds": len(folds),
        "reference_id178": id178_eval,
        "reference_id341": id341_eval,
        "best": best,
        "best_delta_vs_id178": float(best["mean_loss"] - id178_eval["mean_loss"]),
        "best_delta_vs_id341": float(best["mean_loss"] - id341_eval["mean_loss"]),
        "best_beats_id341": bool(best["mean_loss"] < id341_eval["mean_loss"]),
        "target_best_vs_id341": target_best,
        "top_rows": candidate_rows[:15],
    }


def target_signal_summary(split_reports: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in TARGETS:
        primary_deltas = []
        primary_wins = 0
        for split_name in PRIMARY_GATE_SPLITS:
            if split_name not in split_reports:
                continue
            delta = float(split_reports[split_name]["target_best_vs_id341"][target]["delta_vs_id341"])
            primary_deltas.append(delta)
            if delta < 0:
                primary_wins += 1
        out[target] = {
            "primary_wins_vs_id341": int(primary_wins),
            "primary_splits_seen": int(len(primary_deltas)),
            "mean_primary_delta_vs_id341": float(np.mean(primary_deltas)) if primary_deltas else None,
            "min_primary_delta_vs_id341": float(np.min(primary_deltas)) if primary_deltas else None,
            "max_primary_delta_vs_id341": float(np.max(primary_deltas)) if primary_deltas else None,
        }
    return out


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    features, feature_coverage = build_sleep_boundary_features(train, test)
    feature_cols = [col for col in features.columns if col.startswith("sbt_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    split_reports = {
        split_name: evaluate_model_on_split(
            split_name, folds, train, features_train, feature_cols, reference178, reference341
        )
        for split_name, folds in splits.items()
    }
    primary_passes = {
        split_name: split_reports[split_name]["best_beats_id341"]
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_reports
    }
    primary_best_deltas = {
        split_name: split_reports[split_name]["best_delta_vs_id341"]
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_reports
    }
    target_summary = target_signal_summary(split_reports)
    any_target_three_primary_wins = any(
        row["primary_wins_vs_id341"] >= 3
        and row["mean_primary_delta_vs_id341"] is not None
        and row["mean_primary_delta_vs_id341"] < 0
        for row in target_summary.values()
    )
    all_primary_pass = all(primary_passes.values()) if primary_passes else False
    best_primary_count = sum(bool(value) for value in primary_passes.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sleep_boundary_transition_smoke_post_id341",
        "purpose": "Evaluate sleep_boundary_transition_v1 raw-source features under frozen post-ID341 validation gates.",
        "required_report_status": read_required_reports(),
        "feature_coverage": feature_coverage,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "sleep_target_focus": SLEEP_TARGETS,
        "split_reports": split_reports,
        "primary_passes_vs_id341": primary_passes,
        "primary_best_deltas_vs_id341": primary_best_deltas,
        "target_signal_summary": target_summary,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "all_primary_splits_beat_id341": all_primary_pass,
            "primary_split_wins_vs_id341": int(best_primary_count),
            "any_target_has_three_primary_wins": bool(any_target_three_primary_wins),
            "source_family_has_smoke_signal": bool(all_primary_pass or any_target_three_primary_wins),
            "next_action": (
                "If smoke signal is true, run a target-restricted v2 with geometry/transfer audit and no submission artifact. "
                "If false, keep sleep-boundary features as diagnostics and prioritize the Kaggle upstream rebuild."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    lines = [
        "# Sleep-boundary transition smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Feature columns: `{coverage['n_feature_columns']}` (`{coverage['n_raw_numeric_columns']}` raw numeric + missing indicators)",
        f"- Mean missing rate: train `{coverage['train_missing_mean']:.4f}`, test `{coverage['test_missing_mean']:.4f}`",
        "",
        "Top train missing features:",
        "",
    ]
    for name, value in coverage["top_missing_features"].items():
        lines.append(f"- `{name}`: {value:.4f}")
    lines.extend(
        [
            "",
            "## Best by frozen split",
            "",
            "| split | best candidate | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for split_name, split_report in report["split_reports"].items():
        best = split_report["best"]
        id341_loss = split_report["reference_id341"]["mean_loss"]
        id178_loss = split_report["reference_id178"]["mean_loss"]
        lines.append(
            f"| `{split_name}` | `{best['candidate']}` | {best['mean_loss']:.6f} | {id341_loss:.6f} | "
            f"{split_report['best_delta_vs_id341']:+.6f} | {id178_loss:.6f} | {split_report['best_delta_vs_id178']:+.6f} | "
            f"`{split_report['best_beats_id341']}` |"
        )
    lines.extend(
        [
            "",
            "## Target-level smoke signal vs ID341 on primary splits",
            "",
            "| target | primary wins | mean Δ | min Δ | max Δ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target, row in report["target_signal_summary"].items():
        mean_delta = row["mean_primary_delta_vs_id341"]
        min_delta = row["min_primary_delta_vs_id341"]
        max_delta = row["max_primary_delta_vs_id341"]
        lines.append(
            f"| `{target}` | {row['primary_wins_vs_id341']}/{row['primary_splits_seen']} | "
            f"{mean_delta:+.6f} | {min_delta:+.6f} | {max_delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- All primary splits beat ID341: `{report['decision']['all_primary_splits_beat_id341']}`",
            f"- Primary split wins vs ID341: `{report['decision']['primary_split_wins_vs_id341']}`",
            f"- Any target has three primary wins: `{report['decision']['any_target_has_three_primary_wins']}`",
            f"- Source family has smoke signal: `{report['decision']['source_family_has_smoke_signal']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "sleep_boundary_transition_smoke_post_id341_20260618.json",
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
