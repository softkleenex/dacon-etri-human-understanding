#!/usr/bin/env python3
"""Run a strict Subject-Hole diagnostic for lightweight hourly-sequence overlays."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_sequence_s34_overlay_experiment import (  # noqa: E402
    align_sequence_features,
    build_raw_hourly_sequence_features,
    make_sequence_model,
)
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    REPORTS_DIR,
    TARGETS,
    build_feature_frames,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

EPS = 1e-6
DEFAULT_TARGETS = ["Q2", "Q3", "S3", "S4"]
DEFAULT_MODELS = ["hgb_l7_l2", "et_d6_l12"]
DEFAULT_BLEND_WEIGHTS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]


def load_base_oof(train: pd.DataFrame, base_id: int) -> tuple[pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF artifact.")
    return align_candidate(pd.read_csv(oof_paths[0]), train), oof_paths[0].name


def fit_sequence_oof(
    train: pd.DataFrame,
    train_seq: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    model_name: str,
) -> np.ndarray:
    y = train[target].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train), dtype=float)
    has_prediction = np.zeros(len(train), dtype=bool)
    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), clone(make_sequence_model(model_name)))
        model.fit(train_seq.iloc[train_idx][feature_cols], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(train_seq.iloc[valid_idx][feature_cols])[:, 1]
        has_prediction[valid_idx] = True
    if not has_prediction.all():
        raise ValueError(f"Folds did not produce predictions for every {target} row.")
    return np.clip(oof, EPS, 1.0 - EPS)


def evaluate_target(
    train: pd.DataFrame,
    train_seq: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_oof: pd.DataFrame,
    target: str,
    model_names: list[str],
    blend_weights: list[float],
) -> dict:
    y = train[target].astype(int).to_numpy()
    base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    candidates = {"base": base_values}
    candidate_details = {"base": {"kind": "base"}}

    for model_name in model_names:
        raw_oof = fit_sequence_oof(train, train_seq, feature_cols, folds, target, model_name)
        raw_name = f"raw_{model_name}"
        candidates[raw_name] = raw_oof
        candidate_details[raw_name] = {"kind": "raw_sequence", "model": model_name}
        for weight in blend_weights:
            blend_name = f"blend_{model_name}_{weight:.2f}"
            candidates[blend_name] = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
            candidate_details[blend_name] = {"kind": "blend", "model": model_name, "weight": weight}

    candidate_losses = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in candidates.items()}
    selected = np.zeros(len(train), dtype=float)
    nested_choices = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidates.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected[valid_idx] = candidates[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_best_name, full_best_loss = min(candidate_losses.items(), key=lambda item: item[1])
    nested_loss = float(log_loss(y, selected, labels=[0, 1]))
    return {
        "base_loss": candidate_losses["base"],
        "full_best_candidate": full_best_name,
        "full_best_loss": float(full_best_loss),
        "nested_loss": nested_loss,
        "nested_improves": bool(nested_loss < candidate_losses["base"]),
        "candidate_losses": candidate_losses,
        "candidate_details": candidate_details,
        "nested_choices": nested_choices,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=118)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, base_source = load_base_oof(train, args.base_id)
    sequence_features = build_raw_hourly_sequence_features(ROOT / "data" / "raw" / "ch2025_data_items")
    train_seq, _test_seq, feature_cols = align_sequence_features(train, test, sequence_features)
    folds = make_subject_hole_folds(train, n_folds=5)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }

    report = {
        "base_id": args.base_id,
        "base_source": base_source,
        "diagnostic": f"strict_hourly_sequence_overlay_id{args.base_id:03d}",
        "validation": "Subject-Hole nested candidate selection diagnostic only; no artifact is promoted here.",
        "targets_checked": args.targets,
        "models_checked": args.models,
        "blend_weights": DEFAULT_BLEND_WEIGHTS,
        "n_sequence_rows": int(len(sequence_features)),
        "n_sequence_features": int(len(feature_cols)),
        "fold_sizes": [[int(len(train_idx)), int(len(valid_idx))] for train_idx, valid_idx in folds],
        "base_target_scores": base_scores,
        "base_mean_score": float(np.mean(list(base_scores.values()))),
        "targets": {},
    }
    for target in args.targets:
        result = evaluate_target(
            train=train,
            train_seq=train_seq,
            feature_cols=feature_cols,
            folds=folds,
            base_oof=base_oof,
            target=target,
            model_names=args.models,
            blend_weights=DEFAULT_BLEND_WEIGHTS,
        )
        result["mean_if_promoted_by_nested"] = float(
            (sum(base_scores.values()) - base_scores[target] + result["nested_loss"]) / len(TARGETS)
        )
        report["targets"][target] = result
        print(
            target,
            "base",
            f"{result['base_loss']:.6f}",
            "full_best",
            result["full_best_candidate"],
            f"{result['full_best_loss']:.6f}",
            "nested",
            f"{result['nested_loss']:.6f}",
            "improves",
            result["nested_improves"],
        )

    output_path = Path(args.output) if args.output else REPORTS_DIR / f"strict_hourly_sequence_overlay_diagnostic_id{args.base_id:03d}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
