#!/usr/bin/env python3
"""Report-only raw subject-state feature gate against ID342.

This diagnostic turns the successful post-ID342 no-submit raw-stream smoke into
an ID342-aware validation question. It builds row-aligned, interpretable daily
subject-state features in memory from raw lifelog streams, then reuses the
fold-internal ID342 gate to ask whether the signal actually helps Q3/S3 or
nearby targets.

It writes JSON/Markdown reports only. It does not persist feature matrices,
predictions, submission CSVs, base-prediction artifacts, notebook-history
artifacts, submission-log rows, Kaggle pushes/runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BLEND_WEIGHTS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    fold_matrix,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    make_model,
    target_loss_on_split,
)
from scripts.run_post_id342_self_supervised_subject_state_no_submit import (  # noqa: E402
    RAW_MODALITIES,
    row_signal,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_raw_subject_state_feature_gate_20260624.json"
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
TARGET_FOCUS = ["Q1", "Q2", "Q3", "S2", "S3", "S4"]
KEY_COLUMNS = {"subject_id", "date", "subject_key", "date_key"}
MIN_PRIMARY_WINS_VS_ID342 = 4
MAX_PRIMARY_MEAN_DELTA_VS_ID342 = -0.00005
MAX_PRIMARY_WORST_DELTA_VS_ID342 = 0.00025
MAX_PUBLIC_PROXY_FOR_REVIEW = 0.5920
MAX_GAP_TO_PUBLIC_055_FOR_REVIEW = 0.0400

RESEARCH_PRIORS = [
    {
        "source": "ETRI Lifelog Dataset 2024 paper",
        "url": "https://arxiv.org/abs/2508.03698",
        "read": (
            "Q1/S3 were strongest with 3-hour step-count features, Q2/Q3 with daily mobile app usage, and "
            "demographics/weekday helped across models."
        ),
    },
    {
        "source": "DACON 4th ETRI data page",
        "url": "https://dacon.io/en/competitions/official/236468/data",
        "read": (
            "The challenge exposes multimodal smartphone, wearable, proximity, GPS, light, app, and sleep-related "
            "lifelog streams for the seven Q/S targets."
        ),
    },
    {
        "source": "existing local domain notes",
        "path": "docs/domain/etri_lifelog_2024_paper.md",
        "read": (
            "Q2/Q3 are expected to respond to mobile app usage and daily routine, while Q1/S3 are expected to "
            "respond to 3-hour step, light, and sleep-boundary features."
        ),
    },
    {
        "source": "existing local domain notes",
        "path": "docs/domain/nsf_sleep_guidelines.md",
        "read": (
            "S3 is tied to sleep onset latency, so evening and sleep-boundary phone/light/activity states are "
            "more meaningful than generic all-day averages."
        ),
    },
]


@dataclass
class DailyWindowAccumulator:
    count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    value_min: float | None = None
    value_max: float | None = None
    payload_sum: float = 0.0
    positive_count: int = 0
    hour_counts: Counter[int] = field(default_factory=Counter)

    def add(self, hour: int, value: float, payload_count: float) -> None:
        self.count += 1
        self.value_sum += value
        self.value_sq_sum += value * value
        self.value_min = value if self.value_min is None else min(self.value_min, value)
        self.value_max = value if self.value_max is None else max(self.value_max, value)
        self.payload_sum += payload_count
        self.positive_count += int(value > 0)
        self.hour_counts.update([hour])

    def entropy(self) -> float:
        if not self.hour_counts:
            return 0.0
        total = float(sum(self.hour_counts.values()))
        return float(-sum((count / total) * math.log((count / total) + 1e-12) for count in self.hour_counts.values()))

    def as_features(self, prefix: str) -> dict[str, float]:
        if self.count == 0:
            return {
                f"{prefix}__count_log1p": 0.0,
                f"{prefix}__active_hours": 0.0,
                f"{prefix}__value_mean": np.nan,
                f"{prefix}__value_std": np.nan,
                f"{prefix}__value_min": np.nan,
                f"{prefix}__value_max": np.nan,
                f"{prefix}__payload_mean": 0.0,
                f"{prefix}__payload_log1p_sum": 0.0,
                f"{prefix}__positive_ratio": 0.0,
                f"{prefix}__hour_entropy": 0.0,
            }
        mean = self.value_sum / self.count
        var = max(self.value_sq_sum / self.count - mean * mean, 0.0)
        return {
            f"{prefix}__count_log1p": math.log1p(self.count),
            f"{prefix}__active_hours": float(len(self.hour_counts)),
            f"{prefix}__value_mean": mean,
            f"{prefix}__value_std": math.sqrt(var),
            f"{prefix}__value_min": float(self.value_min if self.value_min is not None else mean),
            f"{prefix}__value_max": float(self.value_max if self.value_max is not None else mean),
            f"{prefix}__payload_mean": self.payload_sum / self.count,
            f"{prefix}__payload_log1p_sum": math.log1p(self.payload_sum),
            f"{prefix}__positive_ratio": self.positive_count / self.count,
            f"{prefix}__hour_entropy": self.entropy(),
        }


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def row_windows(hour: int) -> list[str]:
    windows = ["all"]
    if 0 <= hour <= 5:
        windows.append("night")
    elif 6 <= hour <= 11:
        windows.append("morning")
    elif 12 <= hour <= 17:
        windows.append("afternoon")
    else:
        windows.append("evening")
    if hour in {20, 21, 22, 23}:
        windows.append("pre_sleep")
    if hour in {0, 1, 2, 20, 21, 22, 23}:
        windows.append("sleep_boundary")
    if hour in {4, 5, 6, 7, 8}:
        windows.append("wake_boundary")
    return windows


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return numeric


def train_key_frame(train: pd.DataFrame) -> pd.DataFrame:
    keys = train[["subject_id", "date"]].copy()
    keys["subject_key"] = keys["subject_id"].astype(str)
    keys["date_key"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    return keys


def stream_modality_features(
    modality: str,
    path: Path,
    train_keys: set[tuple[str, str]],
    *,
    max_rows: int | None,
    batch_size: int,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    summary: dict[str, Any] = {
        "modality": modality,
        "path": rel(path),
        "exists": path.exists(),
        "max_rows": max_rows,
        "batches_read": 0,
        "rows_read": 0,
        "rows_after_train_key_filter": 0,
        "covered_train_keys": 0,
        "subject_count": 0,
        "date_count": 0,
        "error": None,
    }
    if not path.exists():
        summary["error"] = "missing"
        return {}, summary

    accumulators: dict[tuple[str, str, str], DailyWindowAccumulator] = defaultdict(DailyWindowAccumulator)
    subjects: set[str] = set()
    dates: set[str] = set()
    covered_keys: set[tuple[str, str]] = set()
    rows_remaining = max_rows
    try:
        scanner = ds.dataset(path, format="parquet").scanner(batch_size=batch_size)
        for batch in scanner.to_batches():
            if rows_remaining is not None and rows_remaining <= 0:
                break
            take = batch.num_rows if rows_remaining is None else min(batch.num_rows, rows_remaining)
            if take <= 0:
                break
            frame = batch.slice(0, take).to_pandas()
            raw_columns = list(frame.columns)
            if rows_remaining is not None:
                rows_remaining -= take
            summary["batches_read"] += 1
            summary["rows_read"] += int(len(frame))
            if "subject_id" not in frame.columns or "timestamp" not in frame.columns:
                continue

            timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame["_subject_key"] = frame["subject_id"].astype(str)
            frame["_date_key"] = timestamp.dt.strftime("%Y-%m-%d")
            frame["_hour"] = timestamp.dt.hour
            valid_mask = frame["_date_key"].notna() & frame["_hour"].notna()
            if not valid_mask.any():
                continue
            frame = frame.loc[valid_mask].copy()
            key_mask = [
                (subject_key, date_key) in train_keys
                for subject_key, date_key in zip(frame["_subject_key"], frame["_date_key"], strict=False)
            ]
            if not any(key_mask):
                continue
            frame = frame.loc[key_mask]
            summary["rows_after_train_key_filter"] += int(len(frame))

            for _, row in frame.iterrows():
                subject_key = str(row["_subject_key"])
                date_key = str(row["_date_key"])
                hour = int(row["_hour"])
                value, payload_count = row_signal(row[raw_columns], modality)
                value = safe_float(value)
                payload_count = safe_float(payload_count)
                covered_keys.add((subject_key, date_key))
                subjects.add(subject_key)
                dates.add(date_key)
                for window in row_windows(hour):
                    accumulators[(subject_key, date_key, window)].add(hour, value, payload_count)
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"

    rows: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (subject_key, date_key, window), accumulator in accumulators.items():
        rows[(subject_key, date_key)].update(accumulator.as_features(f"{modality}__{window}"))

    summary["covered_train_keys"] = len(covered_keys)
    summary["subject_count"] = len(subjects)
    summary["date_count"] = len(dates)
    return dict(rows), summary


def merge_modality_rows(
    base_keys: pd.DataFrame,
    modality_rows: dict[str, dict[tuple[str, str], dict[str, float]]],
) -> pd.DataFrame:
    features = base_keys.copy()
    for rows_by_key in modality_rows.values():
        records = [
            {"subject_key": subject_key, "date_key": date_key, **values}
            for (subject_key, date_key), values in rows_by_key.items()
        ]
        if not records:
            continue
        modality_frame = pd.DataFrame.from_records(records)
        features = features.merge(
            modality_frame,
            on=["subject_key", "date_key"],
            how="left",
            validate="one_to_one",
        )
    return features


def add_difference(features: pd.DataFrame, left: str, right: str, name: str) -> None:
    if left in features.columns and right in features.columns:
        features[name] = features[left] - features[right]


def add_ratio(features: pd.DataFrame, numerator: str, denominator: str, name: str) -> None:
    if numerator in features.columns and denominator in features.columns:
        features[name] = features[numerator] / (features[denominator].abs() + 1e-6)


def add_domain_composites(features: pd.DataFrame) -> list[str]:
    before = set(features.columns)
    add_difference(
        features,
        "mLight__sleep_boundary__value_mean",
        "wLight__sleep_boundary__value_mean",
        "domain__sleep_boundary_phone_watch_light_gap",
    )
    add_difference(
        features,
        "mLight__pre_sleep__value_mean",
        "mLight__morning__value_mean",
        "domain__phone_light_presleep_minus_morning",
    )
    add_difference(
        features,
        "wHr__night__value_mean",
        "wHr__morning__value_mean",
        "domain__hr_night_minus_morning",
    )
    add_difference(
        features,
        "mScreenStatus__sleep_boundary__count_log1p",
        "mScreenStatus__morning__count_log1p",
        "domain__screen_sleep_boundary_minus_morning_count",
    )
    add_difference(
        features,
        "mUsageStats__pre_sleep__payload_log1p_sum",
        "mUsageStats__morning__payload_log1p_sum",
        "domain__app_payload_presleep_minus_morning",
    )
    add_difference(
        features,
        "wPedo__evening__value_mean",
        "wPedo__night__value_mean",
        "domain__pedo_evening_minus_night",
    )
    add_difference(
        features,
        "mWifi__all__hour_entropy",
        "mBle__all__hour_entropy",
        "domain__wifi_ble_hour_entropy_gap",
    )
    add_ratio(
        features,
        "mScreenStatus__sleep_boundary__count_log1p",
        "mUsageStats__sleep_boundary__count_log1p",
        "domain__screen_to_app_sleep_boundary_count_ratio",
    )
    add_ratio(
        features,
        "wPedo__evening__count_log1p",
        "mUsageStats__evening__count_log1p",
        "domain__evening_pedo_to_app_count_ratio",
    )
    return sorted(set(features.columns) - before)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.columns
        if column not in KEY_COLUMNS
        and pd.api.types.is_numeric_dtype(features[column])
        and features[column].notna().any()
    ]


def predict_target_for_split_safe(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: Any,
) -> tuple[np.ndarray, int]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = 0
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
            if fit_x.shape[1] == 0:
                proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
            else:
                model = make_model(spec)
                model.fit(fit_x, y[fit_idx])
                proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, 1e-6, 1.0 - 1e-6)
        covered += len(valid_idx)
    return pred, covered


def scan_split_rows_for_source(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            for split_name in PRIMARY_GATE_SPLITS:
                folds = splits[split_name]
                source_pred, covered = predict_target_for_split_safe(features, feature_cols, y, folds, spec)
                id342_target_loss = target_loss_on_split(train, folds, target, references["id342"])
                source_values = references["id342"].copy()
                source_values[:, target_idx] = np.where(np.isnan(source_pred), references["id342"][:, target_idx], source_pred)
                source_loss = target_loss_on_split(train, folds, target, source_values)
                rows.append(
                    {
                        "source": source_name,
                        "target": target,
                        "split_name": split_name,
                        "candidate": f"{source_name}_{target.lower()}_{spec.name}_replace",
                        "model_spec": spec.name,
                        "weight": 1.0,
                        "valid_rows": int(covered),
                        "target_loss": source_loss,
                        "delta_target_vs_id342": float(source_loss - id342_target_loss),
                        "delta_all_vs_id342": float((source_loss - id342_target_loss) / len(TARGETS)),
                    }
                )
                for weight in BLEND_WEIGHTS:
                    blended = references["id342"].copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * references["id342"][:, target_idx]
                        + weight * source_values[:, target_idx],
                        1e-6,
                        1.0 - 1e-6,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    rows.append(
                        {
                            "source": source_name,
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"{source_name}_{target.lower()}_{spec.name}_w{weight:g}",
                            "model_spec": spec.name,
                            "weight": weight,
                            "valid_rows": int(covered),
                            "target_loss": loss,
                            "delta_target_vs_id342": float(loss - id342_target_loss),
                            "delta_all_vs_id342": float((loss - id342_target_loss) / len(TARGETS)),
                        }
                    )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    summary = []
    for candidate, candidate_rows in grouped.items():
        split_rows = {row["split_name"]: row for row in candidate_rows}
        primary_rows = [split_rows[name] for name in PRIMARY_GATE_SPLITS if name in split_rows]
        deltas = [row["delta_all_vs_id342"] for row in primary_rows]
        target_deltas = [row["delta_target_vs_id342"] for row in primary_rows]
        first = candidate_rows[0]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        public_proxy = float(ID342_PUBLIC + mean_delta)
        checks = {
            "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)) >= MIN_PRIMARY_WINS_VS_ID342,
            "primary_mean_delta_material": mean_delta <= MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "primary_worst_delta_within_ceiling": (float(np.max(deltas)) if deltas else 0.0)
            <= MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "public_proxy_low_enough": public_proxy <= MAX_PUBLIC_PROXY_FOR_REVIEW,
            "public_055_gap_low_enough": (public_proxy - PUBLIC_055) <= MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        }
        summary.append(
            {
                "candidate": candidate,
                "source": first["source"],
                "target": first["target"],
                "model_spec": first["model_spec"],
                "weight": first["weight"],
                "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)),
                "primary_mean_delta_all_vs_id342": mean_delta,
                "primary_max_delta_all_vs_id342": float(np.max(deltas)) if deltas else 0.0,
                "primary_mean_delta_target_vs_id342": float(np.mean(target_deltas)) if target_deltas else 0.0,
                "primary_max_delta_target_vs_id342": float(np.max(target_deltas)) if target_deltas else 0.0,
                "projected_public_proxy": public_proxy,
                "gap_to_public_055": float(public_proxy - PUBLIC_055),
                "checks": checks,
                "local_gate_pass": bool(all(checks.values())),
                "split_rows": primary_rows,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            not row["local_gate_pass"],
            row["primary_mean_delta_all_vs_id342"],
            row["primary_max_delta_all_vs_id342"],
        ),
    )


def build_feature_frame(args: argparse.Namespace, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = train_key_frame(train)
    key_set = set(zip(keys["subject_key"], keys["date_key"], strict=False))
    max_rows = None if args.max_rows_per_modality <= 0 else int(args.max_rows_per_modality)
    modality_rows: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    modality_summaries = []
    for modality, path in RAW_MODALITIES.items():
        rows, summary = stream_modality_features(
            modality,
            path,
            key_set,
            max_rows=max_rows,
            batch_size=args.batch_size,
        )
        modality_rows[modality] = rows
        modality_summaries.append(summary)

    features = merge_modality_rows(keys, modality_rows)
    domain_composites = add_domain_composites(features)
    cols = feature_columns(features)
    coverage_by_modality = {
        row["modality"]: float(row["covered_train_keys"] / max(len(key_set), 1)) for row in modality_summaries
    }
    metadata = {
        "max_rows_per_modality": max_rows,
        "batch_size": int(args.batch_size),
        "train_rows": int(len(train)),
        "train_key_count": int(len(key_set)),
        "raw_rows_read": int(sum(row["rows_read"] for row in modality_summaries)),
        "rows_after_train_key_filter": int(sum(row["rows_after_train_key_filter"] for row in modality_summaries)),
        "modality_summaries": modality_summaries,
        "coverage_by_modality": coverage_by_modality,
        "mean_modality_train_key_coverage": float(np.mean(list(coverage_by_modality.values()))),
        "feature_columns": int(len(cols)),
        "domain_composite_columns": domain_composites,
        "feature_column_sample": cols[:20],
        "mean_missing_rate": float(features[cols].isna().mean().mean()) if cols else None,
    }
    return features, metadata


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    features, feature_metadata = build_feature_frame(args, train)
    cols = feature_columns(features)
    splits = load_split_manifest()
    references = {"id342": load_reference_values(train, ANCHOR_ID)}

    split_rows = scan_split_rows_for_source(
        train=train,
        features=features,
        feature_cols=cols,
        splits=splits,
        references=references,
        source_name="raw_subject_state_daily_windows",
    )
    summary = summarize_rows(split_rows)
    passing = [row for row in summary if row["local_gate_pass"]]
    best = summary[0] if summary else None
    bounded_scan = args.max_rows_per_modality > 0
    coverage = feature_metadata["mean_modality_train_key_coverage"]
    projected_public_proxy = best["projected_public_proxy"] if best else ID342_PUBLIC
    next_action = (
        "Bounded local scan found a gate-passing row; require a full Kaggle no-submit rerun and artifact-contract "
        "review before any candidate generation."
        if passing and bounded_scan
        else "Full scan found a gate-passing row; run a separate artifact-contract review before creating predictions."
        if passing
        else "Bounded local scan is coverage-limited; run the same gate on Kaggle with a full raw scan."
        if bounded_scan and coverage < 0.90
        else "No row passed the strict ID342/Public-0.55 gate; keep this as negative evidence, not a candidate route."
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_raw_subject_state_feature_gate",
        "purpose": "Gate in-memory raw subject-state daily/window features against ID342 without creating artifacts.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "public_target": {"score": PUBLIC_055, "gap_from_anchor": ID342_PUBLIC - PUBLIC_055},
        "research_priors": RESEARCH_PRIORS,
        "target_focus": TARGET_FOCUS,
        "model_specs": [spec.name for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "thresholds": {
            "min_primary_wins_vs_id342": MIN_PRIMARY_WINS_VS_ID342,
            "max_primary_mean_delta_vs_id342": MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "max_primary_worst_delta_vs_id342": MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "max_public_proxy_for_review": MAX_PUBLIC_PROXY_FOR_REVIEW,
            "max_gap_to_public_055_for_review": MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        },
        "feature_metadata": feature_metadata,
        "top_rows": summary[:40],
        "decision": {
            "reviewed_split_rows": int(len(split_rows)),
            "reviewed_candidates": int(len(summary)),
            "local_gate_pass_count": int(len(passing)),
            "best_candidate": best["candidate"] if best else None,
            "best_target": best["target"] if best else None,
            "best_primary_mean_delta_all_vs_id342": best["primary_mean_delta_all_vs_id342"] if best else None,
            "best_primary_max_delta_all_vs_id342": best["primary_max_delta_all_vs_id342"] if best else None,
            "best_projected_public_proxy": projected_public_proxy,
            "best_gap_to_public_055": float(projected_public_proxy - PUBLIC_055),
            "bounded_scan": bounded_scan,
            "kaggle_full_scan_recommended": bool(bounded_scan and (passing or coverage < 0.90)),
            "artifact_review_allowed": False,
            "create_feature_matrix_artifact": False,
            "create_prediction_artifact": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "submit_now": False,
            "dacon_submit_now": False,
            "next_action": next_action,
        },
        "no_feature_matrix_created": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action_made_by_this_script": True,
        "no_dacon_submission_made": True,
    }


def format_optional_float(value: float | None, digits: int = 9) -> str:
    return "None" if value is None else f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    feature_metadata = report["feature_metadata"]
    lines = [
        "# Post-ID342 raw subject-state feature gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It builds raw daily/window features in memory and creates no feature matrix, "
        "prediction, submission artifact, Kaggle action, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate pass count: `{decision['local_gate_pass_count']}`",
        f"- Best candidate: `{decision['best_candidate']}`",
        f"- Best target: `{decision['best_target']}`",
        (
            "- Best mean/max all-target delta vs ID342: "
            f"`{format_optional_float(decision['best_primary_mean_delta_all_vs_id342'])}` / "
            f"`{format_optional_float(decision['best_primary_max_delta_all_vs_id342'])}`"
        ),
        f"- Best projected public proxy: `{decision['best_projected_public_proxy']:.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
        f"- Bounded scan: `{decision['bounded_scan']}`",
        f"- Kaggle full scan recommended: `{decision['kaggle_full_scan_recommended']}`",
        f"- Create feature matrix artifact: `{decision['create_feature_matrix_artifact']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Feature Frame",
        "",
        f"- Raw rows read: `{feature_metadata['raw_rows_read']}`",
        f"- Rows after train-key filter: `{feature_metadata['rows_after_train_key_filter']}`",
        f"- Mean modality train-key coverage: `{feature_metadata['mean_modality_train_key_coverage']:.6f}`",
        f"- Feature columns: `{feature_metadata['feature_columns']}`",
        f"- Domain composite columns: `{len(feature_metadata['domain_composite_columns'])}`",
        f"- Mean missing rate: `{feature_metadata['mean_missing_rate']:.6f}`",
        "",
        "## Top Rows",
        "",
        "| rank | pass | candidate | target | weight | wins | mean all delta | max all delta | proxy | gap 0.55 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["top_rows"][:25], start=1):
        lines.append(
            "| {rank} | `{passed}` | `{candidate}` | `{target}` | {weight} | {wins} | "
            "{mean:+.6f} | {max_delta:+.6f} | {proxy:.6f} | {gap:+.6f} |".format(
                rank=rank,
                passed=row["local_gate_pass"],
                candidate=row["candidate"],
                target=row["target"],
                weight=row["weight"],
                wins=row["primary_wins_vs_id342"],
                mean=row["primary_mean_delta_all_vs_id342"],
                max_delta=row["primary_max_delta_all_vs_id342"],
                proxy=row["projected_public_proxy"],
                gap=row["gap_to_public_055"],
            )
        )
    lines.extend(["", "## Modality Coverage", ""])
    for row in feature_metadata["modality_summaries"]:
        lines.append(
            "- `{modality}`: rows `{rows}`, matched `{matched}`, covered train keys `{covered}`, error `{error}`".format(
                modality=row["modality"],
                rows=row["rows_read"],
                matched=row["rows_after_train_key_filter"],
                covered=row["covered_train_keys"],
                error=row["error"],
            )
        )
    lines.extend(["", "## Guardrails", ""])
    for key in [
        "no_feature_matrix_created",
        "no_predictions_created",
        "no_submission_created",
        "no_base_prediction_artifact_created",
        "no_notebook_history_artifact_created",
        "no_submission_log_row_created",
        "no_kaggle_action_made_by_this_script",
        "no_dacon_submission_made",
    ]:
        lines.append(f"- {key}: `{report[key]}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-rows-per-modality",
        type=int,
        default=32768,
        help="Maximum raw rows to scan per modality. Use 0 for a full stream scan.",
    )
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output_path)}")
    print(f"Wrote {rel(output_path.with_suffix('.md'))}")
    print(f"Local gate pass count: {decision['local_gate_pass_count']}")
    print(f"Kaggle full scan recommended: {decision['kaggle_full_scan_recommended']}")
    print(f"Create submission artifact: {decision['create_submission_artifact']}")
    print(f"DACON submit now: {decision['dacon_submit_now']}")


if __name__ == "__main__":
    main()
