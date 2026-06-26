#!/usr/bin/env python3
"""Report-only subject residual sequence-drift smoke after ID341.

This diagnostic implements the `subject_residual_sequence_drift_v1` source from
the post-ID341 phase-2 contract. It builds label-free, past-only per-subject
drift/volatility features from processed numeric covariates, evaluates compact
source models on the frozen primary validation splits, and compares them against
ID178/ID341 OOF references.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, no Kaggle push/run, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    EPS,
    load_reference_values,
    load_split_manifest,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import load_base_test_values  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
PROCESSED_TRAIN = ROOT / "data" / "processed" / "df_train_raw.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "df_test_raw.parquet"
PHASE2_SPEC_PATH = REPORTS_DIR / "post_id341_phase2_raw_source_execution_spec_20260618.json"
TRANSFER_GATE_V2_PATH = REPORTS_DIR / "post_id341_transfer_gate_v2_joint_risk_20260618.json"

PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
BLEND_WEIGHTS = [0.005, 0.01, 0.025, 0.05, 0.10]
ANCHOR_IDS = [178, 287, 329, 341]
ID178_PUBLIC = 0.6014113009
ID341_TRANSFER_PROXY = 0.6038297835
PUBLIC_058_REVIEW_CUTOFF = 0.590
MAX_BASE_COLUMNS = 56


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_iter: int | None = None
    max_leaf_nodes: int | None = None
    l2_regularization: float | None = None
    max_depth: int | None = None
    min_samples_leaf: int | None = None
    n_estimators: int | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_12", "lr", c_value=0.12),
    ModelSpec("hgb_l4_l2_5", "hgb", max_iter=55, max_leaf_nodes=4, l2_regularization=5.0),
    ModelSpec("et_d4_l25", "et", max_depth=4, min_samples_leaf=25, n_estimators=120),
)


def read_required_reports() -> dict[str, Any]:
    return {
        "phase2_spec_exists": PHASE2_SPEC_PATH.exists(),
        "transfer_gate_v2_exists": TRANSFER_GATE_V2_PATH.exists(),
    }


def load_processed_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PROCESSED_TRAIN.exists() or not PROCESSED_TEST.exists():
        raise FileNotFoundError("Missing processed df_train_raw/df_test_raw parquet caches.")
    train = pd.read_parquet(PROCESSED_TRAIN).sort_values(["subject_id", "date"]).reset_index(drop=True)
    test = pd.read_parquet(PROCESSED_TEST).sort_values(["subject_id", "date"]).reset_index(drop=True)
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date
    return train, test


def numeric_source_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    excluded = set(TARGETS + ["subject_id", "sleep_date", "lifelog_date", "date"])
    cols: list[str] = []
    for col in train.columns:
        if col in excluded or col not in test.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        train_values = pd.to_numeric(train[col], errors="coerce")
        if train_values.notna().sum() < max(8, int(0.12 * len(train))):
            continue
        if train_values.nunique(dropna=True) <= 1:
            continue
        cols.append(col)
    return cols


def select_base_columns(
    train: pd.DataFrame, test: pd.DataFrame, max_cols: int = MAX_BASE_COLUMNS
) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for col in numeric_source_columns(train, test):
        train_values = pd.to_numeric(train[col], errors="coerce")
        coverage = float(train_values.notna().mean())
        variance = float(np.nanvar(train_values.to_numpy(dtype=float)))
        finite_abs_mean = float(np.nanmean(np.abs(train_values.to_numpy(dtype=float))))
        if not np.isfinite(variance) or variance <= 0:
            continue
        # Penalize direct subject-code-like columns if they appear later; this smoke
        # is about within-subject drift rather than memorizing subject identity.
        subject_penalty = 0.25 if re.search(r"subject.*code|user.*code", col, flags=re.IGNORECASE) else 1.0
        missing_penalty = 0.65 if coverage < 0.5 else 1.0
        score = float(
            subject_penalty * missing_penalty * coverage * np.log1p(variance) / np.log1p(finite_abs_mean + 1.0)
        )
        rows.append(
            {
                "column": col,
                "coverage": coverage,
                "variance": variance,
                "abs_mean": finite_abs_mean,
                "score": score,
            }
        )
    rows = sorted(rows, key=lambda row: (row["score"], row["coverage"], row["variance"]), reverse=True)
    selected = [row["column"] for row in rows[:max_cols]]
    return selected, rows[: min(80, len(rows))]


def safe_alias(index: int, col: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z]+", "_", col).strip("_").lower()
    return f"c{index:03d}_{stem[:54]}"


def prior_rolling(series: pd.Series, subjects: pd.Series, window: int, op: str) -> pd.Series:
    shifted = series.groupby(subjects, sort=False).shift(1)
    roller = shifted.groupby(subjects, sort=False).rolling(window, min_periods=1)
    if op == "mean":
        out = roller.mean()
    elif op == "std":
        out = roller.std()
    else:
        raise ValueError(f"Unsupported rolling op: {op}")
    return out.reset_index(level=0, drop=True)


def prior_expanding(series: pd.Series, subjects: pd.Series, op: str) -> pd.Series:
    shifted = series.groupby(subjects, sort=False).shift(1)
    expander = shifted.groupby(subjects, sort=False).expanding(min_periods=1)
    if op == "mean":
        out = expander.mean()
    elif op == "std":
        out = expander.std()
    else:
        raise ValueError(f"Unsupported expanding op: {op}")
    return out.reset_index(level=0, drop=True)


def build_sequence_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    work = df[["subject_id", "date", *base_cols]].copy()
    work["_row_order"] = np.arange(len(work))
    work["date_dt"] = pd.to_datetime(work["date"])
    work = work.sort_values(["subject_id", "date_dt", "_row_order"]).reset_index(drop=True)
    subjects = work["subject_id"]

    calendar = pd.DataFrame(
        {
            "sr_dayofweek": work["date_dt"].dt.dayofweek.astype(float),
            "sr_dayofmonth": work["date_dt"].dt.day.astype(float),
            "sr_month": work["date_dt"].dt.month.astype(float),
            "sr_sin_dayofyear": np.sin(2 * np.pi * work["date_dt"].dt.dayofyear.astype(float) / 365.25),
            "sr_cos_dayofyear": np.cos(2 * np.pi * work["date_dt"].dt.dayofyear.astype(float) / 365.25),
            "sr_subject_prior_row_count": work.groupby("subject_id").cumcount().astype(float),
            "sr_days_since_prev_row": work.groupby("subject_id")["date_dt"].diff().dt.days.astype(float),
        }
    )
    parts = [calendar]
    for idx, col in enumerate(base_cols):
        alias = safe_alias(idx, col)
        values = pd.to_numeric(work[col], errors="coerce").astype(float)
        prev = values.groupby(subjects, sort=False).shift(1)
        past3_mean = prior_rolling(values, subjects, 3, "mean")
        past3_std = prior_rolling(values, subjects, 3, "std")
        past5_mean = prior_rolling(values, subjects, 5, "mean")
        exp_mean = prior_expanding(values, subjects, "mean")
        exp_std = prior_expanding(values, subjects, "std")
        parts.append(
            pd.DataFrame(
                {
                    f"sr_{alias}_value": values,
                    f"sr_{alias}_missing": values.isna().astype(float),
                    f"sr_{alias}_delta_prev": values - prev,
                    f"sr_{alias}_minus_past3": values - past3_mean,
                    f"sr_{alias}_minus_past5": values - past5_mean,
                    f"sr_{alias}_z_expanding": (values - exp_mean) / (exp_std.abs() + 1e-3),
                    f"sr_{alias}_vol_past3": past3_std,
                }
            )
        )
    features = pd.concat([work[["subject_id", "date", "_row_order"]], *parts], axis=1)
    features = features.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
    feature_cols = [col for col in features.columns if col.startswith("sr_")]
    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    return features


def build_subject_residual_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_cols, ranked_cols = select_base_columns(train, test)
    train_features = build_sequence_features(train, base_cols)
    test_features = build_sequence_features(test, base_cols)
    feature_cols = [col for col in train_features.columns if col.startswith("sr_")]
    coverage = {
        "selected_base_columns": base_cols,
        "selected_base_column_count": int(len(base_cols)),
        "ranked_base_column_preview": ranked_cols[:20],
        "n_train_rows": int(len(train_features)),
        "n_test_rows": int(len(test_features)),
        "n_feature_columns": int(len(feature_cols)),
        "train_missing_mean": float(train_features[feature_cols].isna().mean().mean()),
        "test_missing_mean": float(test_features[feature_cols].isna().mean().mean()),
        "top_train_missing_features": {
            str(k): float(v)
            for k, v in train_features[feature_cols].isna().mean().sort_values(ascending=False).head(12).items()
        },
    }
    return train_features, test_features, coverage


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
                max_iter=int(spec.max_iter or 55),
                learning_rate=0.03,
                max_leaf_nodes=int(spec.max_leaf_nodes or 4),
                l2_regularization=float(spec.l2_regularization or 5.0),
                min_samples_leaf=14,
                random_state=43,
            ),
        )
    if spec.kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=int(spec.n_estimators or 120),
                max_depth=int(spec.max_depth or 4),
                min_samples_leaf=int(spec.min_samples_leaf or 25),
                max_features="sqrt",
                random_state=43,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def residual_matrix(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        fit_devs[f"{col}_fit_subject_dev"] = fit[col].reset_index(drop=True) - fit_subject_base
        query_devs[f"{col}_fit_subject_dev"] = query[col].reset_index(drop=True) - query_subject_base
    fit_x = pd.concat([fit_raw, pd.DataFrame(fit_devs)], axis=1)
    query_x = pd.concat([query_raw, pd.DataFrame(query_devs)], axis=1)
    return fit_x, query_x


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


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
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            fit_x, valid_x = residual_matrix(features.iloc[fit_idx], features.iloc[valid_idx], feature_cols)
            model = make_model(spec)
            model.fit(fit_x, y[fit_idx])
            if isinstance(model, DummyClassifier):
                proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
            else:
                proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
        covered += len(valid_idx)
    return pred, covered


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
            source_values[:, target_idx] = np.where(np.isnan(pred), reference341[:, target_idx], pred)
            covered_by_target[target] = int(covered)
        source_eval = evaluate_values_on_split(train, folds, source_values)
        candidate_rows.append(
            {
                "candidate": f"subject_residual_source_{spec.name}",
                "model_spec": asdict(spec),
                "kind": "subject_residual_source",
                "anchor": None,
                "weight": None,
                "mean_loss": source_eval["mean_loss"],
                "delta_vs_id178": float(source_eval["mean_loss"] - id178_eval["mean_loss"]),
                "delta_vs_id341": float(source_eval["mean_loss"] - id341_eval["mean_loss"]),
                "target_losses": source_eval["target_losses"],
                "covered_by_target": covered_by_target,
            }
        )
        for target in TARGETS:
            loss = source_eval["target_losses"][target]
            if loss < target_best[target]["best_loss"]:
                target_best[target] = {
                    "target": target,
                    "best_candidate": f"subject_residual_source_{spec.name}",
                    "best_loss": float(loss),
                    "delta_vs_id341": float(loss - id341_eval["target_losses"][target]),
                }
        for anchor_name, anchor_values in [("id178", reference178), ("id341", reference341)]:
            for weight in BLEND_WEIGHTS:
                blend_values = np.clip((1.0 - weight) * anchor_values + weight * source_values, EPS, 1 - EPS)
                blend_eval = evaluate_values_on_split(train, folds, blend_values)
                candidate_rows.append(
                    {
                        "candidate": f"blend_{anchor_name}_subject_residual_{spec.name}_w{weight:g}",
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
                            "best_candidate": f"blend_{anchor_name}_subject_residual_{spec.name}_w{weight:g}",
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
        "top_rows": candidate_rows[:12],
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


def predict_full_train_source_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    feature_cols: list[str],
    spec: ModelSpec,
) -> np.ndarray:
    values = np.zeros((len(test), len(TARGETS)), dtype=float)
    fit_x, test_x = residual_matrix(features_train, features_test, feature_cols)
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            proba = np.full(len(test), float(np.mean(y)), dtype=float)
        else:
            model = make_model(spec)
            model.fit(fit_x, y)
            proba = model.predict_proba(test_x)[:, 1]
        values[:, target_idx] = np.clip(proba, EPS, 1 - EPS)
    return values


def geometry_vs_base_tests(test: pd.DataFrame, source_values: np.ndarray) -> dict[str, Any]:
    base_values = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}
    target_mean = {target: float(source_values[:, idx].mean()) for idx, target in enumerate(TARGETS)}
    out: dict[str, Any] = {
        "source_target_mean": target_mean,
        "source_target_std": {target: float(source_values[:, idx].std()) for idx, target in enumerate(TARGETS)},
        "vs_base": {},
    }
    for candidate_id, values in base_values.items():
        corr_by_target = {}
        mean_abs_delta_by_target = {}
        mean_delta_by_target = {}
        for idx, target in enumerate(TARGETS):
            a = source_values[:, idx]
            b = values[:, idx]
            if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
                corr = float(np.corrcoef(a, b)[0, 1])
            else:
                corr = None
            corr_by_target[target] = corr
            mean_abs_delta_by_target[target] = float(np.mean(np.abs(a - b)))
            mean_delta_by_target[target] = float(np.mean(a - b))
        out["vs_base"][f"id{candidate_id}"] = {
            "mean_abs_delta_all_targets": float(np.mean(np.abs(source_values - values))),
            "mean_delta_all_targets": float(np.mean(source_values - values)),
            "mean_abs_delta_by_target": mean_abs_delta_by_target,
            "mean_delta_by_target": mean_delta_by_target,
            "corr_by_target": corr_by_target,
        }
    return out


def build_report() -> dict[str, Any]:
    train, test = load_processed_frames()
    features_train, features_test, feature_coverage = build_subject_residual_features(train, test)
    feature_cols = [col for col in features_train.columns if col.startswith("sr_")]
    splits = load_split_manifest()
    primary_splits = {name: splits[name] for name in PRIMARY_GATE_SPLITS if name in splits}
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    split_reports = {
        split_name: evaluate_model_on_split(
            split_name,
            folds,
            train,
            features_train,
            feature_cols,
            reference178,
            reference341,
        )
        for split_name, folds in primary_splits.items()
    }
    primary_passes = {split_name: row["best_beats_id341"] for split_name, row in split_reports.items()}
    primary_best_deltas = {split_name: row["best_delta_vs_id341"] for split_name, row in split_reports.items()}
    primary_split_wins = sum(bool(value) for value in primary_passes.values())
    all_primary_pass = all(primary_passes.values()) if primary_passes else False
    mean_primary_delta = float(np.mean(list(primary_best_deltas.values()))) if primary_best_deltas else None
    target_summary = target_signal_summary(split_reports)
    any_target_three_primary_wins = any(
        row["primary_wins_vs_id341"] >= 3
        and row["mean_primary_delta_vs_id341"] is not None
        and row["mean_primary_delta_vs_id341"] < 0
        for row in target_summary.values()
    )
    source_family_has_smoke_signal = bool(all_primary_pass or any_target_three_primary_wins)
    projected_proxy = (
        float(ID341_TRANSFER_PROXY + mean_primary_delta) if mean_primary_delta is not None else ID341_TRANSFER_PROXY
    )
    public_058_reviewable = bool(
        source_family_has_smoke_signal and all_primary_pass and projected_proxy <= PUBLIC_058_REVIEW_CUTOFF
    )

    # Full-train test geometry is in-memory only. It helps spot whether this
    # source is just another copy of an existing anchor before any promotion.
    best_model_name = None
    if split_reports:
        best_rows = sorted(
            [row["best"] for row in split_reports.values()],
            key=lambda row: (row["mean_loss"], row["candidate"]),
        )
        for spec in MODEL_SPECS:
            if any(spec.name in row["candidate"] for row in best_rows[:3]):
                best_model_name = spec.name
                break
    geometry: dict[str, Any] | None = None
    if best_model_name is not None:
        best_spec = next(spec for spec in MODEL_SPECS if spec.name == best_model_name)
        source_test_values = predict_full_train_source_test(
            train, test, features_train, features_test, feature_cols, best_spec
        )
        geometry = {
            "source_model": best_model_name,
            "computed_in_memory_only": True,
            **geometry_vs_base_tests(test, source_test_values),
        }

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "subject_residual_sequence_drift_smoke_post_id341",
        "purpose": (
            "Evaluate fold-internal per-subject residual trend/volatility features from processed numeric covariates "
            "under frozen primary post-ID341 validation gates."
        ),
        "required_report_status": read_required_reports(),
        "feature_coverage": feature_coverage,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "split_reports": split_reports,
        "primary_passes_vs_id341": primary_passes,
        "primary_best_deltas_vs_id341": primary_best_deltas,
        "target_signal_summary": target_summary,
        "test_geometry": geometry,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "all_primary_splits_beat_id341": all_primary_pass,
            "primary_split_wins_vs_id341": int(primary_split_wins),
            "mean_primary_delta_vs_id341": mean_primary_delta,
            "any_target_has_three_primary_wins": bool(any_target_three_primary_wins),
            "source_family_has_smoke_signal": source_family_has_smoke_signal,
            "projected_public_proxy_from_id341": projected_proxy,
            "beats_id178_public_anchor_proxy": bool(projected_proxy < ID178_PUBLIC),
            "public_058_reviewable": public_058_reviewable,
            "next_action": (
                "If smoke signal is true, combine only the winning target/split configs with transfer-v2 and joint-risk "
                "gates before any artifact promotion. If false, do not promote this source; keep it as negative evidence "
                "and prioritize the Kaggle phase-2 raw multimodal runner."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    lines = [
        "# Subject residual sequence-drift smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Base numeric columns selected: `{coverage['selected_base_column_count']}`",
        f"- Rows: `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Sequence/residual feature columns: `{coverage['n_feature_columns']}`",
        f"- Mean missing rate: train `{coverage['train_missing_mean']:.4f}`, test `{coverage['test_missing_mean']:.4f}`",
        "",
        "Top selected base columns:",
        "",
    ]
    for row in coverage["ranked_base_column_preview"][:12]:
        lines.append(
            f"- `{row['column']}` — score `{row['score']:.4f}`, coverage `{row['coverage']:.3f}`, variance `{row['variance']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Best by primary frozen split",
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
            f"{split_report['best_delta_vs_id341']:+.6f} | {id178_loss:.6f} | "
            f"{split_report['best_delta_vs_id178']:+.6f} | `{split_report['best_beats_id341']}` |"
        )
    lines.extend(
        [
            "",
            "## Target-level smoke signal vs ID341",
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
    if report.get("test_geometry"):
        geometry = report["test_geometry"]
        lines.extend(
            [
                "",
                "## In-memory test geometry",
                "",
                f"- Source model: `{geometry['source_model']}`",
                f"- Mean absolute delta vs ID341 base test: `{geometry['vs_base']['id341']['mean_abs_delta_all_targets']:.6f}`",
                f"- Mean absolute delta vs ID178 base test: `{geometry['vs_base']['id178']['mean_abs_delta_all_targets']:.6f}`",
            ]
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- All primary splits beat ID341: `{decision['all_primary_splits_beat_id341']}`",
            f"- Primary split wins vs ID341: `{decision['primary_split_wins_vs_id341']}`",
            f"- Mean primary Δ vs ID341: `{decision['mean_primary_delta_vs_id341']:+.6f}`",
            f"- Any target has three primary wins: `{decision['any_target_has_three_primary_wins']}`",
            f"- Source family has smoke signal: `{decision['source_family_has_smoke_signal']}`",
            f"- Projected public proxy from ID341: `{decision['projected_public_proxy_from_id341']:.6f}`",
            f"- Beats ID178 public anchor proxy: `{decision['beats_id178_public_anchor_proxy']}`",
            f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "subject_residual_sequence_drift_smoke_post_id341_20260618.json",
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
