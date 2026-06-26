#!/usr/bin/env python3
"""Missingness-aware cross-modal subject-state no-submit runner.

This is the post-ID342 heavy-route runner. It reads bounded raw phone/watch
payload batches, builds subject-date modality/window tokens with inherited
missingness masks, trains a small label-free masked autoencoder, freezes the
embeddings, and evaluates Q3/S3 downstream heads against ID342 on frozen folds.

It writes report files only. It never creates submission CSVs, base prediction
artifacts, notebook-history artifacts, submission-log rows, or DACON
submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BLEND_WEIGHTS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
ITEMS_DIR = RAW_DIR / "ch2025_data_items"
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
FOCUS_TARGETS = ("Q3", "S3")
SANITY_TARGETS = ("Q2",)
EPS = 1e-6

REMOTE_REPORT_NAMES = {
    "manifest": "post_id342_subject_state_remote_manifest.json",
    "embedding_purity": "post_id342_subject_state_embedding_purity.json",
    "token_coverage": "post_id342_subject_state_token_coverage.json",
    "downstream_smoke": "post_id342_subject_state_downstream_smoke.json",
    "failed_frontier_resistance": "post_id342_subject_state_failed_frontier_resistance.json",
    "public_gap_budget": "post_id342_subject_state_public_gap_budget.json",
    "artifact_separation": "post_id342_subject_state_artifact_separation.json",
    "summary": "post_id342_subject_state_heavy_no_submit_summary.json",
}

FORBIDDEN_OUTPUT_HINTS = (
    "submission",
    "submission_log",
    "base_oof",
    "base_test",
    "candidate",
)

RAW_MODALITIES = {
    "mACStatus": ITEMS_DIR / "ch2025_mACStatus.parquet",
    "mActivity": ITEMS_DIR / "ch2025_mActivity.parquet",
    "mAmbience": ITEMS_DIR / "ch2025_mAmbience.parquet",
    "mBle": ITEMS_DIR / "ch2025_mBle.parquet",
    "mGps": ITEMS_DIR / "ch2025_mGps.parquet",
    "mLight": ITEMS_DIR / "ch2025_mLight.parquet",
    "mScreenStatus": ITEMS_DIR / "ch2025_mScreenStatus.parquet",
    "mUsageStats": ITEMS_DIR / "ch2025_mUsageStats.parquet",
    "mWifi": ITEMS_DIR / "ch2025_mWifi.parquet",
    "wHr": ITEMS_DIR / "ch2025_wHr.parquet",
    "wLight": ITEMS_DIR / "ch2025_wLight.parquet",
    "wPedo": ITEMS_DIR / "ch2025_wPedo.parquet",
}

WINDOW_SPECS = {
    "day_grid": (12.0, 36.0),
    "anchored_evening_grid": (18.0, 27.0),
    "sleep_transition_grid": (20.0, 33.0),
    "morning_recovery_grid": (28.0, 36.0),
}


@dataclass
class WindowAccumulator:
    count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    positive_count: int = 0
    payload_sum: float = 0.0
    hour_counts: Counter[int] = field(default_factory=Counter)

    def add(self, rel_hour: float, value: float, payload_count: float) -> None:
        self.count += 1
        self.value_sum += value
        self.value_sq_sum += value * value
        self.positive_count += int(value > 0)
        self.payload_sum += payload_count
        self.hour_counts.update([int(math.floor(rel_hour))])

    def entropy(self) -> float:
        total = sum(self.hour_counts.values())
        if total <= 0:
            return 0.0
        return float(
            -sum((count / total) * math.log((count / total) + 1e-12) for count in self.hour_counts.values())
        )

    def as_features(self, prefix: str) -> dict[str, float]:
        if self.count == 0:
            return {
                f"{prefix}__observed": 0.0,
                f"{prefix}__missing": 1.0,
                f"{prefix}__count_log1p": 0.0,
                f"{prefix}__value_mean": np.nan,
                f"{prefix}__value_std": np.nan,
                f"{prefix}__positive_ratio": 0.0,
                f"{prefix}__payload_log1p": 0.0,
                f"{prefix}__hour_entropy": 0.0,
            }
        mean = self.value_sum / self.count
        var = max(self.value_sq_sum / self.count - mean * mean, 0.0)
        return {
            f"{prefix}__observed": 1.0,
            f"{prefix}__missing": 0.0,
            f"{prefix}__count_log1p": math.log1p(self.count),
            f"{prefix}__value_mean": mean,
            f"{prefix}__value_std": math.sqrt(var),
            f"{prefix}__positive_ratio": self.positive_count / self.count,
            f"{prefix}__payload_log1p": math.log1p(self.payload_sum),
            f"{prefix}__hour_entropy": self.entropy(),
        }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    c_value: float


class MaskedSubjectStateAutoencoder(nn.Module):
    def __init__(self, in_dim: int, value_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(32, min(256, in_dim * 2))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, value_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return self.decoder(embedding), embedding


MODEL_SPECS = (
    ModelSpec("lr_c0_02", 0.02),
    ModelSpec("lr_c0_08", 0.08),
    ModelSpec("lr_c0_20", 0.20),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_dir_for_mode(mode: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir if output_dir.is_absolute() else ROOT / output_dir
    if mode == "remote_no_submit":
        return Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/kaggle_outputs"))
    return REPORTS_DIR / "post_id342_subject_state_heavy_no_submit_local_smoke"


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def sequence_payload(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def nested_numeric(value: Any, modality: str) -> tuple[float, float]:
    payload = sequence_payload(value)
    if not payload:
        return 0.0, 0.0
    if modality == "mUsageStats":
        total = 0.0
        active = 0
        for item in payload:
            if isinstance(item, dict):
                total += safe_float(item.get("total_time", 0.0))
                active += int(safe_float(item.get("total_time", 0.0)) > 0)
        return math.log1p(max(total, 0.0) / 60000.0), float(active)
    if modality in {"mBle", "mWifi"}:
        rssi = [safe_float(item.get("rssi")) for item in payload if isinstance(item, dict) and item.get("rssi") is not None]
        return (float(np.mean(rssi)) if rssi else 0.0), float(len(payload))
    if modality == "mGps":
        speeds = [
            safe_float(item.get("speed"))
            for item in payload
            if isinstance(item, dict) and item.get("speed") is not None
        ]
        return (float(np.mean(speeds)) if speeds else 0.0), float(len(payload))
    if modality == "mAmbience":
        scores = []
        for item in payload:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                scores.append(safe_float(item[1]))
        return (max(scores) if scores else 0.0), float(len(payload))
    if modality == "wHr":
        values = [safe_float(item) for item in payload if item is not None]
        values = [value for value in values if 0 < value < 250]
        return (float(np.mean(values)) if values else 0.0), float(len(values))
    return float(len(payload)), float(len(payload))


def row_signal(row: pd.Series, modality: str) -> tuple[float, float]:
    if modality == "wPedo":
        numeric_cols = ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"]
        values = [safe_float(row[col]) for col in numeric_cols if col in row and pd.notna(row[col])]
        return (float(np.mean(values)) if values else 0.0), float(len(values))
    value_columns = [col for col in row.index if col not in {"subject_id", "timestamp"}]
    if not value_columns:
        return 0.0, 0.0
    value = row[value_columns[0]]
    if isinstance(value, (list, tuple, np.ndarray)):
        return nested_numeric(value, modality)
    if value is None or pd.isna(value):
        return 0.0, 0.0
    return safe_float(value), 1.0


def anchored_timestamp(timestamp: pd.Timestamp) -> tuple[str, float]:
    hour = timestamp.hour
    anchor = timestamp.normalize()
    if hour < 12:
        anchor -= pd.Timedelta(days=1)
    rel_hour = float(hour + (24 if hour < 12 else 0)) + timestamp.minute / 60.0
    return anchor.strftime("%Y-%m-%d"), rel_hour


def active_windows(rel_hour: float) -> list[str]:
    windows = []
    for name, (start, end) in WINDOW_SPECS.items():
        if start <= rel_hour < end:
            windows.append(name)
    return windows


def load_test_sorted() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def base_key_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_keys = train[["subject_id", "date"]].copy()
    train_keys["split"] = "train"
    test_keys = test[["subject_id", "date"]].copy()
    test_keys["split"] = "test"
    out = pd.concat([train_keys, test_keys], ignore_index=True)
    out["subject_key"] = out["subject_id"].astype(str)
    out["date_key"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def stream_modality(
    modality: str,
    path: Path,
    key_set: set[tuple[str, str]],
    *,
    max_rows: int | None,
    batch_size: int,
    accumulators: dict[tuple[str, str, str, str], WindowAccumulator],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "modality": modality,
        "path": rel(path),
        "exists": path.exists(),
        "max_rows": max_rows,
        "batch_size": batch_size,
        "batches_read": 0,
        "rows_read": 0,
        "rows_after_key_filter": 0,
        "covered_keys": 0,
        "subject_count": 0,
        "date_count": 0,
        "error": None,
    }
    if not path.exists():
        summary["error"] = "missing"
        return summary
    rows_remaining = max_rows
    covered: set[tuple[str, str]] = set()
    subjects: set[str] = set()
    dates: set[str] = set()
    try:
        scanner = ds.dataset(path, format="parquet").scanner(batch_size=batch_size)
        for batch in scanner.to_batches():
            if rows_remaining is not None and rows_remaining <= 0:
                break
            take = batch.num_rows if rows_remaining is None else min(batch.num_rows, rows_remaining)
            if take <= 0:
                break
            frame = batch.slice(0, take).to_pandas()
            if rows_remaining is not None:
                rows_remaining -= take
            summary["batches_read"] += 1
            summary["rows_read"] += int(len(frame))
            if "subject_id" not in frame.columns or "timestamp" not in frame.columns:
                continue
            timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame = frame.assign(_timestamp=timestamps)
            frame = frame[frame["_timestamp"].notna()].copy()
            if frame.empty:
                continue
            derived = [anchored_timestamp(value) for value in frame["_timestamp"]]
            frame["_date_key"] = [row[0] for row in derived]
            frame["_rel_hour"] = [row[1] for row in derived]
            frame["_subject_key"] = frame["subject_id"].astype(str)
            mask = [
                (subject_key, date_key) in key_set
                for subject_key, date_key in zip(frame["_subject_key"], frame["_date_key"], strict=False)
            ]
            if not any(mask):
                continue
            frame = frame.loc[mask].copy()
            summary["rows_after_key_filter"] += int(len(frame))
            for _, row in frame.iterrows():
                subject_key = str(row["_subject_key"])
                date_key = str(row["_date_key"])
                rel_hour = float(row["_rel_hour"])
                value, payload_count = row_signal(row.drop(labels=["_timestamp", "_date_key", "_rel_hour", "_subject_key"]), modality)
                windows = active_windows(rel_hour)
                if not windows:
                    continue
                covered.add((subject_key, date_key))
                subjects.add(subject_key)
                dates.add(date_key)
                for window in windows:
                    accumulators[(subject_key, date_key, modality, window)].add(rel_hour, value, payload_count)
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["covered_keys"] = len(covered)
    summary["subject_count"] = len(subjects)
    summary["date_count"] = len(dates)
    return summary


def build_subject_state_features(args: argparse.Namespace, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = base_key_frame(train, test)
    key_set = set(zip(keys["subject_key"], keys["date_key"], strict=False))
    max_rows = None if args.max_rows_per_modality <= 0 else int(args.max_rows_per_modality)
    accumulators: dict[tuple[str, str, str, str], WindowAccumulator] = defaultdict(WindowAccumulator)
    modality_summaries = [
        stream_modality(
            modality,
            path,
            key_set,
            max_rows=max_rows,
            batch_size=args.batch_size,
            accumulators=accumulators,
        )
        for modality, path in RAW_MODALITIES.items()
    ]
    rows: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    expected_prefixes = [
        (modality, window)
        for modality in RAW_MODALITIES
        for window in WINDOW_SPECS
    ]
    for subject_key, date_key in key_set:
        row_values: dict[str, float] = {}
        for modality, window in expected_prefixes:
            accumulator = accumulators.get((subject_key, date_key, modality, window), WindowAccumulator())
            row_values.update(accumulator.as_features(f"{modality}__{window}"))
        rows[(subject_key, date_key)] = row_values
    feature_records = [
        {"subject_key": subject_key, "date_key": date_key, **values}
        for (subject_key, date_key), values in rows.items()
    ]
    feature_frame = pd.DataFrame.from_records(feature_records)
    aligned = keys.merge(feature_frame, on=["subject_key", "date_key"], how="left", validate="one_to_one")
    feature_cols = [
        col
        for col in aligned.columns
        if col not in {"subject_id", "date", "split", "subject_key", "date_key"}
        and pd.api.types.is_numeric_dtype(aligned[col])
    ]
    coverage_by_modality = {
        row["modality"]: row["covered_keys"] / max(len(key_set), 1)
        for row in modality_summaries
    }
    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "key_count": int(len(key_set)),
        "max_rows_per_modality": max_rows,
        "batch_size": int(args.batch_size),
        "raw_rows_read": int(sum(row["rows_read"] for row in modality_summaries)),
        "rows_after_key_filter": int(sum(row["rows_after_key_filter"] for row in modality_summaries)),
        "feature_columns": int(len(feature_cols)),
        "modality_summaries": modality_summaries,
        "coverage_by_modality": coverage_by_modality,
        "mean_modality_key_coverage": float(np.mean(list(coverage_by_modality.values()))),
        "train_any_observed": float(
            aligned[aligned["split"].eq("train")][[col for col in feature_cols if col.endswith("__observed")]]
            .sum(axis=1)
            .gt(0)
            .mean()
        ),
        "test_any_observed": float(
            aligned[aligned["split"].eq("test")][[col for col in feature_cols if col.endswith("__observed")]]
            .sum(axis=1)
            .gt(0)
            .mean()
        ),
    }
    return aligned, metadata


def prepare_autoencoder_input(aligned: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    feature_cols = [
        col
        for col in aligned.columns
        if col not in {"subject_id", "date", "split", "subject_key", "date_key"}
        and pd.api.types.is_numeric_dtype(aligned[col])
        and aligned[col].notna().any()
    ]
    value_cols = [col for col in feature_cols if not col.endswith("__observed") and not col.endswith("__missing")]
    mask_cols = [col for col in feature_cols if col.endswith("__observed")]
    if value_cols:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        values = scaler.fit_transform(imputer.fit_transform(aligned[value_cols]))
    else:
        values = np.zeros((len(aligned), 0), dtype=np.float32)
    if mask_cols:
        masks = aligned[mask_cols].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        masks = np.zeros((len(aligned), 0), dtype=np.float32)
    arr = np.concatenate([values.astype(np.float32), masks.astype(np.float32)], axis=1)
    metadata = {
        "value_columns": int(len(value_cols)),
        "mask_columns": int(len(mask_cols)),
        "input_dim": int(arr.shape[1]),
        "rows": int(arr.shape[0]),
        "uses_unlabeled_test_raw_features": True,
        "imputer_fit_on_labels": False,
        "scaler_fit_on_labels": False,
    }
    return arr, value_cols, mask_cols, metadata


def train_masked_autoencoder(
    x: np.ndarray,
    value_dim: int,
    *,
    embedding_dim: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if x.shape[0] < 8 or value_dim < 1:
        return np.zeros((x.shape[0], max(1, embedding_dim)), dtype=np.float32), {
            "trained": False,
            "reason": "not_enough_rows_or_values",
            "rows": int(x.shape[0]),
            "value_dim": int(value_dim),
        }
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tensor = torch.tensor(x, dtype=torch.float32)
    model = MaskedSubjectStateAutoencoder(tensor.shape[1], value_dim, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    losses = []
    for _epoch in range(epochs):
        artificial_mask = torch.rand((tensor.shape[0], value_dim), dtype=torch.float32) < 0.25
        masked = tensor.clone()
        masked[:, :value_dim][artificial_mask] = 0.0
        decoded, _embedding = model(masked)
        loss = ((decoded - tensor[:, :value_dim]) ** 2)[artificial_mask].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        _decoded, embedding = model(tensor)
    return embedding.detach().cpu().numpy().astype(np.float32), {
        "trained": True,
        "epochs": int(epochs),
        "embedding_dim": int(embedding_dim),
        "input_dim": int(tensor.shape[1]),
        "value_dim": int(value_dim),
        "losses": losses,
        "final_masked_mse": losses[-1] if losses else None,
        "persistent_model_artifact_created": False,
    }


def make_downstream_model(spec: ModelSpec):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=spec.c_value, solver="liblinear", max_iter=500, random_state=47),
    )


def predict_split_embeddings(
    x: np.ndarray,
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
        else:
            model = make_downstream_model(spec)
            model.fit(x[fit_idx], y[fit_idx])
            proba = model.predict_proba(x[valid_idx])[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1.0 - EPS)
    if np.isnan(pred).any():
        fill_value = float(np.nanmean(pred)) if np.isfinite(np.nanmean(pred)) else float(np.mean(y))
        pred = np.where(np.isnan(pred), fill_value, pred)
    return np.clip(pred, EPS, 1.0 - EPS)


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_split_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    summary = []
    for config, config_rows in grouped.items():
        primary_rows = [row for row in config_rows if row["split_name"] in PRIMARY_GATE_SPLITS]
        deltas = [row["delta_all_vs_id342"] for row in primary_rows]
        target_deltas = [row["delta_target_vs_id342"] for row in primary_rows]
        first = config_rows[0]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        max_delta = float(np.max(deltas)) if deltas else 0.0
        proxy = float(ID342_PUBLIC + mean_delta)
        summary.append(
            {
                "config": config,
                "target": first["target"],
                "model_spec": first["model_spec"],
                "weight": first["weight"],
                "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)),
                "primary_mean_delta_all_vs_id342": mean_delta,
                "primary_max_delta_all_vs_id342": max_delta,
                "primary_mean_delta_target_vs_id342": float(np.mean(target_deltas)) if target_deltas else 0.0,
                "primary_max_delta_target_vs_id342": float(np.max(target_deltas)) if target_deltas else 0.0,
                "projected_public_proxy": proxy,
                "gap_to_public_055": float(proxy - PUBLIC_055),
                "stable_gate_pass": bool(
                    int(sum(delta < 0 for delta in deltas)) >= 4
                    and mean_delta <= -0.00005
                    and max_delta <= 0.0005
                ),
                "split_rows": primary_rows,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            not row["stable_gate_pass"],
            row["primary_mean_delta_all_vs_id342"],
            row["primary_max_delta_all_vs_id342"],
            row["config"],
        ),
    )


def downstream_smoke(train: pd.DataFrame, train_embeddings: np.ndarray, anchor_values: np.ndarray) -> dict[str, Any]:
    splits = load_split_manifest()
    rows: list[dict[str, Any]] = []
    for target in [*FOCUS_TARGETS, *SANITY_TARGETS]:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            split_predictions = {
                split_name: predict_split_embeddings(train_embeddings, y, splits[split_name], spec)
                for split_name in PRIMARY_GATE_SPLITS
            }
            for weight in BLEND_WEIGHTS:
                for split_name in PRIMARY_GATE_SPLITS:
                    folds = splits[split_name]
                    source = split_predictions[split_name]
                    blended_target = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source,
                        EPS,
                        1.0 - EPS,
                    )
                    anchor_loss = target_loss_on_split(train, folds, target, anchor_values)
                    candidate_loss = weighted_split_loss(y, blended_target, folds)
                    delta = float(candidate_loss - anchor_loss)
                    rows.append(
                        {
                            "target": target,
                            "split_name": split_name,
                            "config": f"missingness_subject_state_{target.lower()}_{spec.name}_w{weight:g}",
                            "model_spec": spec.name,
                            "weight": weight,
                            "target_loss": candidate_loss,
                            "anchor_target_loss": anchor_loss,
                            "delta_target_vs_id342": delta,
                            "delta_all_vs_id342": float(delta / len(TARGETS)),
                        }
                    )
    summary = summarize_rows(rows)
    focus_summary = [row for row in summary if row["target"] in FOCUS_TARGETS]
    best = focus_summary[0] if focus_summary else None
    return {
        "target_focus": list(FOCUS_TARGETS),
        "sanity_targets": list(SANITY_TARGETS),
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": list(BLEND_WEIGHTS),
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "reviewed_split_rows": int(len(rows)),
        "reviewed_configs": int(len(summary)),
        "stable_gate_pass_count": int(sum(row["stable_gate_pass"] for row in focus_summary)),
        "top_rows": summary[:40],
        "best_focus_row": best,
    }


def scan_forbidden_outputs(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []
    suspicious = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if path.suffix.lower() == ".csv" or any(hint in lower for hint in FORBIDDEN_OUTPUT_HINTS):
            suspicious.append({"path": rel(path), "bytes": path.stat().st_size})
    return suspicious


def build_reports(args: argparse.Namespace, output_dir: Path) -> dict[str, dict[str, Any]]:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = load_train_sorted()
    test = load_test_sorted()
    aligned, token_coverage_metadata = build_subject_state_features(args, train, test)
    ae_input, value_cols, mask_cols, input_metadata = prepare_autoencoder_input(aligned)
    embeddings, reconstruction = train_masked_autoencoder(
        ae_input,
        value_dim=len(value_cols),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        seed=seed,
    )
    train_embeddings = embeddings[aligned["split"].eq("train").to_numpy()]
    anchor_values = load_reference_values(train, ANCHOR_ID)
    downstream = downstream_smoke(train, train_embeddings, anchor_values)
    best = downstream["best_focus_row"]
    best_proxy = best["projected_public_proxy"] if best else ID342_PUBLIC
    artifact_review_allowed = bool(
        best
        and best["stable_gate_pass"]
        and best_proxy <= args.artifact_review_proxy_threshold
    )
    dacon_candidate_signal = bool(
        artifact_review_allowed and best_proxy <= args.dacon_candidate_proxy_threshold
    )
    embedding_purity = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_embedding_purity",
        "label_free_pretraining": True,
        "uses_target_labels_for_pretraining": False,
        "uses_oof_or_base_predictions_for_pretraining": False,
        "uses_public_scores_for_pretraining": False,
        "uses_unlabeled_test_raw_features": input_metadata["uses_unlabeled_test_raw_features"],
        "raw_privacy_outputs_created": False,
        "persistent_model_artifact_created": False,
        "embedding_matrix_artifact_created": False,
        "embedding_purity_pass": True,
    }
    token_coverage = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_token_coverage",
        "token_coverage_ready": token_coverage_metadata["raw_rows_read"] > 0,
        "token_coverage": token_coverage_metadata,
        "autoencoder_input": input_metadata,
        "reconstruction": reconstruction,
    }
    downstream_report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_downstream_smoke",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "downstream": downstream,
        "create_prediction_artifact": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "dacon_submit_now": False,
    }
    failed_frontier_resistance = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_failed_frontier_resistance",
        "independent_of_failed_frontier_donors": True,
        "uses_id287_or_id329_predictions": False,
        "uses_public_scored_target_swaps": False,
        "stable_gate_pass_count": downstream["stable_gate_pass_count"],
        "best_focus_row": best,
        "failed_frontier_resistance_pass": bool(best and best["stable_gate_pass"]),
    }
    public_gap_budget = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_public_gap_budget",
        "anchor_public": ID342_PUBLIC,
        "target_public": PUBLIC_055,
        "best_projected_public_proxy": best_proxy,
        "gap_to_public_055": float(best_proxy - PUBLIC_055),
        "artifact_review_proxy_threshold": args.artifact_review_proxy_threshold,
        "dacon_candidate_proxy_threshold": args.dacon_candidate_proxy_threshold,
        "artifact_review_allowed": artifact_review_allowed,
        "dacon_candidate_signal": dacon_candidate_signal,
    }
    artifact_separation = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_artifact_separation",
        "output_dir": str(output_dir),
        "unexpected_submission_like_artifacts": scan_forbidden_outputs(output_dir),
        "create_feature_matrix_artifact": False,
        "create_prediction_artifact": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "dacon_submit_now": False,
    }
    manifest = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_remote_manifest",
        "mode": args.mode,
        "runner": "post_id342_missingness_subject_state_heavy_no_submit",
        "config": {
            "max_rows_per_modality": args.max_rows_per_modality,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "seed": seed,
        },
        "report_files": REMOTE_REPORT_NAMES,
        "no_feature_matrix_created": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }
    summary = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_heavy_no_submit_summary",
        "mode": args.mode,
        "best_focus_row": best,
        "best_projected_public_proxy": best_proxy,
        "gap_to_public_055": float(best_proxy - PUBLIC_055),
        "stable_gate_pass_count": downstream["stable_gate_pass_count"],
        "artifact_review_allowed": artifact_review_allowed,
        "dacon_candidate_signal": dacon_candidate_signal,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "dacon_submit_now": False,
        "next_action": (
            "Open separate artifact review only; DACON still requires exact file/memo approval."
            if artifact_review_allowed
            else "Keep as no-submit research evidence; do not create artifacts or submit."
        ),
    }
    return {
        "manifest": manifest,
        "embedding_purity": embedding_purity,
        "token_coverage": token_coverage,
        "downstream_smoke": downstream_report,
        "failed_frontier_resistance": failed_frontier_resistance,
        "public_gap_budget": public_gap_budget,
        "artifact_separation": artifact_separation,
        "summary": summary,
    }


def write_markdown_summary(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    summary = reports["summary"]
    token_coverage = reports["token_coverage"]["token_coverage"]
    reconstruction = reports["token_coverage"]["reconstruction"]
    best = summary["best_focus_row"] or {}
    lines = [
        "# Post-ID342 missingness-aware subject-state heavy no-submit",
        "",
        f"Created: `{summary['created_at_kst']}`",
        "",
        "This runner writes reports only. It creates no feature matrix, prediction artifact, submission CSV, "
        "submission-log row, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Stable gate pass count: `{summary['stable_gate_pass_count']}`",
        f"- Best config: `{best.get('config')}`",
        f"- Best target: `{best.get('target')}`",
        f"- Best projected public proxy: `{summary['best_projected_public_proxy']:.9f}`",
        f"- Gap to Public 0.55: `{summary['gap_to_public_055']:+.9f}`",
        f"- Artifact review allowed: `{summary['artifact_review_allowed']}`",
        f"- DACON candidate signal: `{summary['dacon_candidate_signal']}`",
        f"- Create submission artifact: `{summary['create_submission_artifact']}`",
        f"- DACON submit now: `{summary['dacon_submit_now']}`",
        f"- Next action: {summary['next_action']}",
        "",
        "## Token Coverage",
        "",
        f"- Raw rows read: `{token_coverage['raw_rows_read']}`",
        f"- Rows after key filter: `{token_coverage['rows_after_key_filter']}`",
        f"- Feature columns: `{token_coverage['feature_columns']}`",
        f"- Mean modality key coverage: `{token_coverage['mean_modality_key_coverage']:.6f}`",
        f"- Train/test any observed: `{token_coverage['train_any_observed']:.6f}` / `{token_coverage['test_any_observed']:.6f}`",
        "",
        "## Reconstruction",
        "",
        f"- Trained: `{reconstruction.get('trained')}`",
        f"- Final masked MSE: `{reconstruction.get('final_masked_mse')}`",
        f"- Persistent model artifact created: `{reconstruction.get('persistent_model_artifact_created')}`",
        "",
        "## Top Downstream Rows",
        "",
        "| rank | pass | config | target | wins | mean all delta | max all delta | proxy | gap 0.55 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(reports["downstream_smoke"]["downstream"]["top_rows"][:20], start=1):
        lines.append(
            "| {rank} | `{passed}` | `{config}` | `{target}` | {wins} | {mean:+.6f} | "
            "{max_delta:+.6f} | {proxy:.6f} | {gap:+.6f} |".format(
                rank=rank,
                passed=row["stable_gate_pass"],
                config=row["config"],
                target=row["target"],
                wins=row["primary_wins_vs_id342"],
                mean=row["primary_mean_delta_all_vs_id342"],
                max_delta=row["primary_max_delta_all_vs_id342"],
                proxy=row["projected_public_proxy"],
                gap=row["gap_to_public_055"],
            )
        )
    (output_dir / "post_id342_subject_state_heavy_no_submit_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_reports(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in REMOTE_REPORT_NAMES.items():
        path = output_dir / filename
        path.write_text(json.dumps(reports[key], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_summary(reports, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local_smoke", "remote_no_submit"], default="local_smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows-per-modality", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=45502)
    parser.add_argument("--artifact-review-proxy-threshold", type=float, default=0.58)
    parser.add_argument("--dacon-candidate-proxy-threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for_mode(args.mode, args.output_dir)
    reports = build_reports(args, output_dir)
    write_reports(reports, output_dir)
    summary = reports["summary"]
    print(f"Wrote reports to {rel(output_dir)}")
    print(f"Stable gate pass count: {summary['stable_gate_pass_count']}")
    print(f"Best projected public proxy: {summary['best_projected_public_proxy']:.9f}")
    print(f"Artifact review allowed: {summary['artifact_review_allowed']}")
    print(f"DACON submit now: {summary['dacon_submit_now']}")


if __name__ == "__main__":
    main()
