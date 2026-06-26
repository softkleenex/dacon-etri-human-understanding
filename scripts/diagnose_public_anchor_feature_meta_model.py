#!/usr/bin/env python3
"""Diagnose small public-structure meta models over same-subject anchor features."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_multi_anchor_probe_experiment import predict_anchor  # noqa: E402
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    make_interleaved_subject_folds,
)
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    kind: str
    config: dict[str, Any]


ANCHOR_SPECS: tuple[AnchorSpec, ...] = (
    AnchorSpec("subject_mean", "subject_mean", {"kind": "subject_mean"}),
    AnchorSpec("weekday_mean", "weekday_mean", {"kind": "weekday_mean"}),
    AnchorSpec("nearest_h0_25", "nearest_two", {"kind": "nearest_two", "half_life": 0.25}),
    AnchorSpec("nearest_h2", "nearest_two", {"kind": "nearest_two", "half_life": 2.0}),
    AnchorSpec("nearest_h14", "nearest_two", {"kind": "nearest_two", "half_life": 14.0}),
    AnchorSpec("nearest_h84", "nearest_two", {"kind": "nearest_two", "half_life": 84.0}),
    AnchorSpec("exp_h0_25", "exp_all", {"kind": "exp_all", "half_life": 0.25}),
    AnchorSpec("exp_h2", "exp_all", {"kind": "exp_all", "half_life": 2.0}),
    AnchorSpec("exp_h14", "exp_all", {"kind": "exp_all", "half_life": 14.0}),
    AnchorSpec("exp_h84", "exp_all", {"kind": "exp_all", "half_life": 84.0}),
    AnchorSpec("knn_mean_k1", "knn_mean", {"kind": "knn_mean", "k": 1}),
    AnchorSpec("knn_mean_k3", "knn_mean", {"kind": "knn_mean", "k": 3}),
    AnchorSpec("knn_mean_k5", "knn_mean", {"kind": "knn_mean", "k": 5}),
    AnchorSpec("knn_exp_k5_h2", "knn_exp", {"kind": "knn_exp", "k": 5, "half_life": 2.0}),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    l2_regularization: float | None = None
    max_leaf_nodes: int | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_1", "lr", c_value=0.1),
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("lr_c1", "lr", c_value=1.0),
    ModelSpec("hgb_l3_l2_1", "hgb", l2_regularization=1.0, max_leaf_nodes=3),
    ModelSpec("hgb_l7_l2_1", "hgb", l2_regularization=1.0, max_leaf_nodes=7),
)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_date_dt(frame)
    out["weekday"] = out["date_dt"].dt.weekday
    out["date_ordinal"] = out["date_dt"].map(pd.Timestamp.toordinal).astype(float)
    return out


def fit_model(spec: ModelSpec, x_train: np.ndarray, y_train: np.ndarray):
    if len(np.unique(y_train)) < 2:
        return None
    if spec.kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", random_state=43, max_iter=300),
        ).fit(x_train, y_train)
    if spec.kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=int(spec.max_leaf_nodes),
            l2_regularization=float(spec.l2_regularization),
            min_samples_leaf=12,
            random_state=43,
        ).fit(x_train, y_train)
    raise ValueError(f"Unknown model kind: {spec.kind}")


def predict_model(model, x_query: np.ndarray, fallback_mean: float) -> np.ndarray:
    if model is None:
        return np.full(x_query.shape[0], fallback_mean, dtype=float)
    return np.clip(model.predict_proba(x_query)[:, 1], EPS, 1.0 - EPS)


def base_feature_frame(query_df: pd.DataFrame, base_pred: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=query_df.index)
    for target in TARGETS:
        values = np.clip(base_pred.loc[query_df.index, target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        features[f"base_{target}"] = values
        features[f"base_{target}_logit"] = np.log(values / (1.0 - values))
    date_min = float(query_df["date_ordinal"].min())
    date_span = max(float(query_df["date_ordinal"].max()) - date_min, 1.0)
    features["date_scaled_local"] = (query_df["date_ordinal"].astype(float).to_numpy() - date_min) / date_span
    for weekday in range(7):
        features[f"weekday_{weekday}"] = (query_df["weekday"].to_numpy() == weekday).astype(float)
    for subject_id in sorted(query_df["subject_id"].astype(str).unique()):
        features[f"subject_{subject_id}"] = (query_df["subject_id"].astype(str).to_numpy() == subject_id).astype(float)
    return features


def add_anchor_features(
    features: pd.DataFrame,
    known_df: pd.DataFrame,
    query_df: pd.DataFrame,
    known_labels: pd.DataFrame,
    prefix: str,
) -> None:
    for label_target in TARGETS:
        y_known = known_labels[label_target].astype(int).to_numpy()
        for spec in ANCHOR_SPECS:
            values = predict_anchor(spec.kind, known_df, y_known, query_df, spec.config)
            features[f"{prefix}_{label_target}_{spec.name}"] = values


def query_feature_frame(
    known_df: pd.DataFrame,
    query_df: pd.DataFrame,
    known_labels: pd.DataFrame,
    base_pred: pd.DataFrame,
) -> pd.DataFrame:
    features = base_feature_frame(query_df, base_pred)
    add_anchor_features(features, known_df, query_df, known_labels, "anchor")
    return features


def oof_feature_frame(
    train: pd.DataFrame,
    indices: np.ndarray,
    fold_ids: np.ndarray,
    base_pred: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in sorted(np.unique(fold_ids[indices])):
        fit_idx = indices[fold_ids[indices] != fold]
        query_idx = indices[fold_ids[indices] == fold]
        if len(query_idx) == 0:
            continue
        part = query_feature_frame(
            known_df=train.iloc[fit_idx],
            query_df=train.iloc[query_idx],
            known_labels=train.iloc[fit_idx][TARGETS],
            base_pred=base_pred,
        )
        parts.append(part)
    out = pd.concat(parts, axis=0).sort_index()
    return out.loc[indices]


def align_columns(train_x: pd.DataFrame, query_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(train_x.columns) | set(query_x.columns))
    return train_x.reindex(columns=cols, fill_value=0.0), query_x.reindex(columns=cols, fill_value=0.0)


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train_raw, test_raw, _metrics_sorted, _sample_original = build_feature_frames()
    train = prepare_frame(train_raw)
    test = prepare_frame(test_raw)
    base_oof, base_test, base_source = load_artifact_pair(train_raw, test_raw, base_id)
    base_oof.index = train.index
    base_test.index = test.index
    fold_ids = make_interleaved_subject_folds(train)
    all_idx = np.arange(len(train))

    candidate_oof: dict[str, pd.DataFrame] = {
        spec.name: pd.DataFrame(index=train.index, columns=TARGETS, dtype=float) for spec in MODEL_SPECS
    }
    fold_details: dict[str, dict[str, list[dict[str, Any]]]] = {
        target: {spec.name: [] for spec in MODEL_SPECS} for target in TARGETS
    }

    for outer_fold in sorted(np.unique(fold_ids)):
        train_idx = all_idx[fold_ids != outer_fold]
        valid_idx = all_idx[fold_ids == outer_fold]
        x_train = oof_feature_frame(train, train_idx, fold_ids, base_oof)
        x_valid = query_feature_frame(
            known_df=train.iloc[train_idx],
            query_df=train.iloc[valid_idx],
            known_labels=train.iloc[train_idx][TARGETS],
            base_pred=base_oof,
        )
        x_train, x_valid = align_columns(x_train, x_valid)
        x_train_np = x_train.to_numpy(dtype=float)
        x_valid_np = x_valid.to_numpy(dtype=float)
        for target in TARGETS:
            y_train = train.iloc[train_idx][target].astype(int).to_numpy()
            y_valid = train.iloc[valid_idx][target].astype(int).to_numpy()
            fallback_mean = float(np.mean(y_train))
            for spec in MODEL_SPECS:
                model = fit_model(spec, x_train_np, y_train)
                pred = predict_model(model, x_valid_np, fallback_mean)
                candidate_oof[spec.name].loc[valid_idx, target] = pred
                fold_details[target][spec.name].append(
                    {
                        "outer_fold": int(outer_fold),
                        "n_train": int(len(train_idx)),
                        "n_valid": int(len(valid_idx)),
                        "valid_loss": float(log_loss(y_valid, pred, labels=[0, 1])),
                        "valid_prediction_mean": float(np.mean(pred)),
                    }
                )

    report: dict[str, Any] = {
        "diagnostic": "public_anchor_feature_meta_model",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "outer_interleaved_within_subject_blocked_anchor_feature_meta_model",
        "anchor_specs": [asdict(spec) for spec in ANCHOR_SPECS],
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "targets": {},
    }
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report["base_scores"] = base_scores
    report["base_mean"] = float(np.mean(list(base_scores.values())))
    model_means: dict[str, float] = {}
    for spec in MODEL_SPECS:
        scores = {
            target: float(
                log_loss(
                    train[target].astype(int),
                    clip_probabilities(candidate_oof[spec.name][[target]].rename(columns={target: target}), [target])[
                        target
                    ],
                    labels=[0, 1],
                )
            )
            for target in TARGETS
        }
        model_means[spec.name] = float(np.mean(list(scores.values())))
    report["model_means"] = model_means

    best_target_scores = []
    for target in TARGETS:
        candidate_losses = {"base": base_scores[target]}
        for spec in MODEL_SPECS:
            preds = np.clip(candidate_oof[spec.name][target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            candidate_losses[spec.name] = float(log_loss(train[target].astype(int), preds, labels=[0, 1]))
        best_name, best_loss = min(candidate_losses.items(), key=lambda item: (item[1], item[0]))
        best_target_scores.append(best_loss)
        report["targets"][target] = {
            "best": best_name,
            "best_loss": float(best_loss),
            "improvement_vs_base": float(base_scores[target] - best_loss),
            "candidate_losses": candidate_losses,
            "fold_details": {name: fold_details[target].get(name, []) for name in candidate_losses if name != "base"},
        }
    report["targetwise_best_mean"] = float(np.mean(best_target_scores))

    # Full-train selections for possible later promotion. These are not used to score this diagnostic.
    full_train_x = oof_feature_frame(train, all_idx, fold_ids, base_oof)
    test_x = query_feature_frame(train, test, train[TARGETS], base_test)
    full_train_x, test_x = align_columns(full_train_x, test_x)
    report["feature_count"] = int(full_train_x.shape[1])
    report["full_train_candidate_test_prediction_summary"] = {}
    for target, info in report["targets"].items():
        best_name = str(info["best"])
        if best_name == "base":
            pred = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        else:
            spec = next(spec for spec in MODEL_SPECS if spec.name == best_name)
            model = fit_model(spec, full_train_x.to_numpy(dtype=float), train[target].astype(int).to_numpy())
            pred = predict_model(model, test_x.to_numpy(dtype=float), float(train[target].mean()))
        report["full_train_candidate_test_prediction_summary"][target] = {
            "selected": best_name,
            "test_prediction_mean": float(np.mean(pred)),
            "test_prediction_std": float(np.std(pred)),
            "test_prediction_min": float(np.min(pred)),
            "test_prediction_max": float(np.max(pred)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=170)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "id170_anchor_feature_meta_model_diagnostic.json")
    args = parser.parse_args()
    report = run_diagnostic(args.base_id, args.output)
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "model_means": report["model_means"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "feature_count": report["feature_count"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
