#!/usr/bin/env python3
"""ID342-aware sequence-summary distance/kernel smoke for Q3/S3.

This report-only diagnostic follows the ETRI paper's time-zone idea more
directly than the first raw-temporal smoke: it converts half-hour raw sensor
sequences into deterministic sequence-summary features, adds label-free
calendar/progress features, then tests small fold-internal models and
distance-kernel predictors against the ID342 OOF anchor.

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
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke import (  # noqa: E402
    EPS,
    FOCUS_TARGETS,
    ID342_PUBLIC,
    PRIMARY_SPLITS,
    PUBLIC_055,
    TARGETS,
    fold_loss,
    load_split_manifest,
    weighted_split_loss,
)
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_sequence_summary_distance_kernel_smoke_20260624.json"
RAW_TEMPORAL_SMOKE_PATH = REPORTS_DIR / "post_id342_raw_temporal_q3s3_sequence_image_smoke_20260624.json"

BLEND_WEIGHTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07)
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075
MAX_FAMILY_FEATURES = 1800

FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all": (),
    "paper_core": ("step", "distance", "pedo", "hr", "watch_light", "phone_light", "usage", "calendar"),
    "q3_stress_activity_usage": (
        "step",
        "distance",
        "speed",
        "usage",
        "screen",
        "activity",
        "active",
        "still",
        "hr",
        "amb_",
        "calendar",
    ),
    "s3_sol_light_motion": (
        "watch_light",
        "phone_light",
        "light",
        "screen",
        "charging",
        "step",
        "distance",
        "speed",
        "hr",
        "activity",
        "still",
        "calendar",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_leaf_nodes: int | None = None
    k_neighbors: int | None = None
    sigma_scale: float | None = None
    temperature: float | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_10", "lr", c_value=0.10),
    ModelSpec("lr_c0_30", "lr", c_value=0.30),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
    ModelSpec("rbf_k15_s0_7", "rbf", k_neighbors=15, sigma_scale=0.7),
    ModelSpec("rbf_k25_s1_0", "rbf", k_neighbors=25, sigma_scale=1.0),
    ModelSpec("proto_t1_0", "prototype", temperature=1.0),
    ModelSpec("ssknn_k1", "same_subject_knn", k_neighbors=1),
    ModelSpec("ssknn_k3", "same_subject_knn", k_neighbors=3),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=46),
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
                random_state=46,
            ),
        )
    raise ValueError(f"Unknown sklearn model kind: {spec.kind}")


def calendar_progress_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_keys = pd.concat(
        [
            train[["subject_id", "date"]].assign(split="train"),
            test[["subject_id", "date"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    all_keys["date_dt"] = pd.to_datetime(all_keys["date"])
    all_keys["dow"] = all_keys["date_dt"].dt.dayofweek.astype(float)
    all_keys["month"] = all_keys["date_dt"].dt.month.astype(float)
    all_keys = all_keys.sort_values(["subject_id", "date_dt"]).reset_index(drop=True)
    grouped = all_keys.groupby("subject_id", sort=False)["date_dt"]
    first = grouped.transform("min")
    last = grouped.transform("max")
    span = (last - first).dt.days.clip(lower=1).astype(float)
    day_from_start = (all_keys["date_dt"] - first).dt.days.astype(float)
    day_to_end = (last - all_keys["date_dt"]).dt.days.astype(float)
    rank = all_keys.groupby("subject_id", sort=False).cumcount().astype(float)
    count = all_keys.groupby("subject_id", sort=False)["date_dt"].transform("count").clip(lower=2).astype(float)

    features = pd.DataFrame(
        {
            "calendar_dow_sin": np.sin(2.0 * np.pi * all_keys["dow"] / 7.0),
            "calendar_dow_cos": np.cos(2.0 * np.pi * all_keys["dow"] / 7.0),
            "calendar_is_weekend": all_keys["dow"].isin([5.0, 6.0]).astype(float),
            "calendar_month_sin": np.sin(2.0 * np.pi * all_keys["month"] / 12.0),
            "calendar_month_cos": np.cos(2.0 * np.pi * all_keys["month"] / 12.0),
            "calendar_subject_day_progress": day_from_start / span,
            "calendar_subject_reverse_progress": day_to_end / span,
            "calendar_subject_rank_pct": rank / (count - 1.0),
        },
        index=all_keys.index,
    )
    train_values = features.loc[all_keys["split"].eq("train")].to_numpy(dtype=np.float32)
    test_values = features.loc[all_keys["split"].eq("test")].to_numpy(dtype=np.float32)
    return train_values, test_values, list(features.columns)


def build_feature_matrix() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)
    cal_train, cal_test, calendar_names = calendar_progress_features(train, test)
    x_train = np.concatenate([x_train, cal_train], axis=1)
    x_test = np.concatenate([x_test, cal_test], axis=1)
    feature_names = [*feature_names, *calendar_names]
    metadata = {
        "sequence_rows": int(len(seq_table)),
        "sequence_value_channels": int(len(channel_names) // 2),
        "sequence_total_channels": int(len(channel_names)),
        "sequence_summary_features": int(len(feature_names) - len(calendar_names)),
        "calendar_progress_features": int(len(calendar_names)),
        "total_features": int(x_train.shape[1]),
        "raw_sequence_table_features": int(len(feature_cols)),
    }
    return x_train.astype(np.float32), x_test.astype(np.float32), feature_names, metadata


def family_feature_indices(x_train: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    indices: dict[str, np.ndarray] = {}
    variance = np.var(x_train, axis=0)
    for family, keywords in FAMILY_KEYWORDS.items():
        if family == "all":
            family_idx = np.arange(len(feature_names), dtype=int)
        else:
            family_idx = np.asarray(
                [
                    idx
                    for idx, name in enumerate(feature_names)
                    if any(keyword in name for keyword in keywords)
                ],
                dtype=int,
            )
        if family_idx.size > MAX_FAMILY_FEATURES:
            ranked = family_idx[np.argsort(variance[family_idx])[::-1]]
            family_idx = ranked[:MAX_FAMILY_FEATURES]
        indices[family] = family_idx
    return indices


def preprocess_fit_query(
    x: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    feature_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit_x = x[np.asarray(fit_idx), :][:, feature_idx]
    query_x = x[np.asarray(query_idx), :][:, feature_idx]
    fit_arr = scaler.fit_transform(imputer.fit_transform(fit_x))
    query_arr = scaler.transform(imputer.transform(query_x))
    fit_arr = np.nan_to_num(fit_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    query_arr = np.nan_to_num(query_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return fit_arr, query_arr


def pairwise_mean_squared(query_arr: np.ndarray, fit_arr: np.ndarray) -> np.ndarray:
    distances = ((query_arr[:, None, :] - fit_arr[None, :, :]) ** 2).mean(axis=2)
    return np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=1e6)


def fit_distance_scale(fit_arr: np.ndarray) -> float:
    if fit_arr.shape[0] <= 1:
        return 1.0
    pair = pairwise_mean_squared(fit_arr, fit_arr)
    upper = pair[np.triu_indices(pair.shape[0], k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0)]
    return float(np.median(upper)) if upper.size else 1.0


def rbf_predictions(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    feature_idx: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    fit_arr, valid_arr = preprocess_fit_query(x, fit_idx, valid_idx, feature_idx)
    fit_y = y[fit_idx].astype(float)
    distances = pairwise_mean_squared(valid_arr, fit_arr)
    k = min(int(spec.k_neighbors or fit_arr.shape[0]), fit_arr.shape[0])
    if k < fit_arr.shape[0]:
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        masked = np.full_like(distances, np.inf, dtype=float)
        row_idx = np.arange(distances.shape[0])[:, None]
        masked[row_idx, nearest] = distances[row_idx, nearest]
        distances = masked
    scale = max(fit_distance_scale(fit_arr) * float(spec.sigma_scale or 1.0), 1e-6)
    weights = np.exp(-distances / (2.0 * scale))
    weight_sum = weights.sum(axis=1)
    prior = float(np.mean(fit_y))
    pred = np.divide(weights @ fit_y, weight_sum, out=np.full(len(valid_idx), prior, dtype=float), where=weight_sum > 0)
    return np.clip(pred, EPS, 1.0 - EPS)


def prototype_predictions(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    feature_idx: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    fit_arr, valid_arr = preprocess_fit_query(x, fit_idx, valid_idx, feature_idx)
    fit_y = y[fit_idx].astype(int)
    if len(np.unique(fit_y)) < 2:
        return np.full(len(valid_idx), float(np.mean(fit_y)), dtype=float)
    center0 = fit_arr[fit_y == 0].mean(axis=0)
    center1 = fit_arr[fit_y == 1].mean(axis=0)
    d0 = ((valid_arr - center0.reshape(1, -1)) ** 2).mean(axis=1)
    d1 = ((valid_arr - center1.reshape(1, -1)) ** 2).mean(axis=1)
    temperature = max(float(spec.temperature or 1.0), 1e-6)
    logits = np.clip((d0 - d1) / temperature, -20.0, 20.0)
    return np.clip(1.0 / (1.0 + np.exp(-logits)), EPS, 1.0 - EPS)


def same_subject_knn_predictions(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    feature_idx: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    fit_arr, valid_arr = preprocess_fit_query(x, fit_idx, valid_idx, feature_idx)
    fit_y = y[fit_idx].astype(float)
    fit_subjects = subjects[fit_idx].astype(str)
    valid_subjects = subjects[valid_idx].astype(str)
    out = np.full(len(valid_idx), float(np.mean(fit_y)), dtype=float)
    k_requested = int(spec.k_neighbors or 1)
    for subject_id in np.unique(valid_subjects):
        valid_mask = valid_subjects == subject_id
        fit_mask = fit_subjects == subject_id
        if not fit_mask.any():
            continue
        distances = pairwise_mean_squared(valid_arr[valid_mask], fit_arr[fit_mask])
        k = min(k_requested, distances.shape[1])
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        out[valid_mask] = fit_y[fit_mask][nearest].mean(axis=1)
    return np.clip(out, EPS, 1.0 - EPS)


def predict_oof_for_split(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_idx: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        elif spec.kind in {"lr", "hgb"}:
            model = clone(make_model(spec))
            model.fit(x[fit_idx, :][:, feature_idx], y[fit_idx])
            proba = model.predict_proba(x[valid_idx, :][:, feature_idx])[:, 1]
        elif spec.kind == "rbf":
            proba = rbf_predictions(x, y, fit_idx, valid_idx, feature_idx, spec)
        elif spec.kind == "prototype":
            proba = prototype_predictions(x, y, fit_idx, valid_idx, feature_idx, spec)
        elif spec.kind == "same_subject_knn":
            proba = same_subject_knn_predictions(x, y, subjects, fit_idx, valid_idx, feature_idx, spec)
        else:
            raise ValueError(f"Unknown model kind: {spec.kind}")
        pred[valid_idx] = np.clip(proba, EPS, 1.0 - EPS)
    if np.isnan(pred).any():
        fill_value = float(np.nanmean(pred)) if np.isfinite(np.nanmean(pred)) else float(np.mean(y))
        pred = np.where(np.isnan(pred), fill_value, pred)
    return np.clip(pred, EPS, 1.0 - EPS)


def scan_target(
    train: pd.DataFrame,
    x_train: np.ndarray,
    family_indices: dict[str, np.ndarray],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    target: str,
) -> dict[str, Any]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    subjects = train["subject_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for family, feature_idx in family_indices.items():
        if feature_idx.size == 0:
            continue
        for spec in MODEL_SPECS:
            split_source_predictions: dict[str, np.ndarray] = {}
            for split_name in PRIMARY_SPLITS:
                split_source_predictions[split_name] = predict_oof_for_split(
                    x_train,
                    y,
                    subjects,
                    feature_idx,
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
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                rows.append(
                    {
                        "target": target,
                        "config": f"{target}_id342_seqdist_{family}_{spec.name}_w{weight:g}",
                        "feature_family": family,
                        "feature_count": int(feature_idx.size),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "split_deltas": split_deltas,
                        "split_losses": split_losses,
                        "split_anchor_losses": split_anchor_losses,
                        "eligible": bool(
                            wins >= TARGET_WIN_FLOOR
                            and mean_delta <= TARGET_MEAN_DELTA_CEILING
                            and max_delta <= TARGET_MAX_DELTA_CEILING
                        ),
                    }
                )
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    return {
        "target": target,
        "anchor_primary_losses": {
            split_name: weighted_split_loss(y, anchor_values[:, target_idx], splits[split_name])
            for split_name in PRIMARY_SPLITS
        },
        "best": rows[0] if rows else None,
        "eligible_count": int(len(eligible_rows)),
        "selected": eligible_rows[0] if eligible_rows else None,
        "top_rows": rows[:10],
    }


def full_id342_local_score(train: pd.DataFrame, anchor_values: np.ndarray) -> dict[str, Any]:
    target_losses = {
        target: fold_loss(train[target].astype(int).to_numpy(), anchor_values[:, TARGETS.index(target)])
        for target in TARGETS
    }
    return {"target_losses": target_losses, "mean_loss": float(np.mean(list(target_losses.values())))}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    base_oof, _base_test, base_source = load_artifact_pair(train, test, 342)
    anchor_values = base_oof[TARGETS].astype(float).to_numpy()
    x_train, _x_test, feature_names, feature_metadata = build_feature_matrix()
    if x_train.shape[0] != len(train):
        raise ValueError("Sequence feature matrix does not align with train rows.")
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    family_indices = family_feature_indices(x_train, feature_names)
    family_metadata = {
        family: {
            "feature_count": int(indices.size),
            "example_features": [feature_names[idx] for idx in indices[:12]],
        }
        for family, indices in family_indices.items()
    }
    target_reports = {
        target: scan_target(train, x_train, family_indices, splits, anchor_values, target)
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
        and max_primary_delta <= TARGET_MAX_DELTA_CEILING
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
        "sequence_distance_has_q3s3_signal": bool(selected_targets),
        "partial_signal_targets": partial_signal_targets,
        "q3_partial_signal_unstable": "Q3" in partial_signal_targets and "Q3" not in selected_targets,
        "s3_partial_signal_unstable": "S3" in partial_signal_targets and "S3" not in selected_targets,
        "final_sequence_distance_gate_pass": final_gate_pass,
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": bool(final_gate_pass),
        "run_kaggle_now": bool(final_gate_pass),
        "next_action": (
            "Use Kaggle no-submit only if the final gate passes; otherwise keep this as negative/partial "
            "sequence-distance evidence and do not create a DACON artifact."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_sequence_summary_distance_kernel_smoke",
        "purpose": "Test ID342-aware Q3/S3 sequence-summary distance/kernel signal without creating artifacts.",
        "raw_temporal_smoke_report": rel(RAW_TEMPORAL_SMOKE_PATH),
        "anchor_oof": base_source,
        "anchor_public": ID342_PUBLIC,
        "id342_local_score": full_id342_local_score(train, anchor_values),
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": list(BLEND_WEIGHTS),
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "feature_contract": {
            "source": "44 half-hour raw-sensor sequence table plus label-free calendar/progress features",
            "paper_domain_read": (
                "Q3 stress is screened through activity, phone usage, screen state, ambience, and heart-rate "
                "shape; S3 sleep-onset-latency is screened through evening/pre-sleep light, movement, screen, "
                "charging, and heart-rate shape."
            ),
            "metadata": feature_metadata,
            "feature_families": family_metadata,
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
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    metadata = report["feature_contract"]["metadata"]
    lines = [
        "# Post-ID342 sequence-summary distance/kernel smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Feature Contract",
        "",
        f"- Source: {report['feature_contract']['source']}",
        f"- Domain read: {report['feature_contract']['paper_domain_read']}",
        f"- Sequence rows: `{metadata['sequence_rows']}`",
        f"- Total channels: `{metadata['sequence_total_channels']}`",
        f"- Sequence summary features: `{metadata['sequence_summary_features']}`",
        f"- Calendar/progress features: `{metadata['calendar_progress_features']}`",
        f"- Total features: `{metadata['total_features']}`",
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
    lines.extend(["", "## Top Rows", ""])
    for target, target_report in report["target_reports"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | config | wins | mean delta | max delta |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(target_report["top_rows"][:5], start=1):
            lines.append(
                "| {rank} | `{config}` | {wins} | {mean_delta:+.6f} | {max_delta:+.6f} |".format(
                    rank=rank,
                    config=row["config"],
                    wins=row["wins"],
                    mean_delta=row["mean_delta"],
                    max_delta=row["max_delta"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Sequence distance has Q3/S3 signal: `{decision['sequence_distance_has_q3s3_signal']}`",
            f"- Partial signal targets: `{decision['partial_signal_targets']}`",
            f"- Final gate pass: `{decision['final_sequence_distance_gate_pass']}`",
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
