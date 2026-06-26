#!/usr/bin/env python3
"""Run a strict, fixed-hyperparameter meta-stacking experiment.

This script promotes a uniquely tagged set of base OOF/submission files into a
strict-contract experiment row. It intentionally avoids Optuna, global clip
tuning, temperature tuning, and same-OOF post-processing searches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DataLoader
from src.features.demographics import DEMOGRAPHIC_FEATURE_COLUMNS, add_official_demographics
from src.utils.validation import clip_probabilities, make_subject_hole_folds, validate_submission

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
NOTEBOOK_DIR = ROOT / "notebooks" / "history"
REPORTS_DIR = ROOT / "reports"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
BASE_MODELS = ["catboost", "lgbm", "xgb", "et", "rf", "lr", "dl"]


def extract_date(df: pd.DataFrame) -> pd.Series:
    for col in ["date", "lifelog_date", "sleep_date"]:
        if col in df.columns:
            return pd.to_datetime(df[col]).dt.date
    raise ValueError("No usable date/lifelog_date/sleep_date column found.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def load_base_predictions(
    base_tags: list[str], metrics_sorted: pd.DataFrame, sample_sorted: pd.DataFrame
) -> tuple[list[str], dict, dict]:
    id_col = "subject_id"
    oof_by_model: dict[str, pd.DataFrame] = {}
    sub_by_model: dict[str, pd.DataFrame] = {}
    loaded_models: list[str] = []

    metric_key = metrics_sorted[[id_col, "date"]].reset_index(drop=True)
    sample_key = sample_sorted[[id_col, "date"]].reset_index(drop=True)

    for base_tag in base_tags:
        for model_name in BASE_MODELS:
            feature_name = f"{model_name}_{base_tag}"
            oof_name = f"oof_{model_name}_{base_tag}.csv"
            sub_name = f"submission_{model_name}_{base_tag}.csv"
            oof_path = first_existing([SUBMISSIONS_DIR / oof_name, BASE_PREDICTION_DIR / oof_name])
            sub_path = first_existing([BASE_PREDICTION_DIR / sub_name, SUBMISSIONS_DIR / sub_name])
            if oof_path is None or sub_path is None:
                continue

            oof = pd.read_csv(oof_path)
            sub = pd.read_csv(sub_path)
            oof["date"] = extract_date(oof)
            sub["date"] = extract_date(sub)
            oof = oof.sort_values([id_col, "date"]).reset_index(drop=True)
            sub = sub.sort_values([id_col, "date"]).reset_index(drop=True)

            if not oof[[id_col, "date"]].reset_index(drop=True).equals(metric_key):
                raise ValueError(f"OOF alignment mismatch: {oof_path.name}")
            if not sub[[id_col, "date"]].reset_index(drop=True).equals(sample_key):
                raise ValueError(f"Submission alignment mismatch: {sub_path.name}")

            loaded_models.append(feature_name)
            oof_by_model[feature_name] = oof
            sub_by_model[feature_name] = sub

    if len(loaded_models) < 2:
        raise ValueError(f"Need at least two paired base models for tags {base_tags}; found {loaded_models}")

    return loaded_models, oof_by_model, sub_by_model


def make_meta_features(
    target: str,
    loaded_models: list[str],
    oof_by_model: dict[str, pd.DataFrame],
    sub_by_model: dict[str, pd.DataFrame],
    train_demo: pd.DataFrame,
    test_demo: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-6
    train_probs = np.column_stack(
        [oof_by_model[name][target].astype(float).fillna(0.5).clip(eps, 1.0 - eps).values for name in loaded_models]
    )
    test_probs = np.column_stack(
        [sub_by_model[name][target].astype(float).fillna(0.5).clip(eps, 1.0 - eps).values for name in loaded_models]
    )
    train_logits = np.log(train_probs / (1.0 - train_probs))
    test_logits = np.log(test_probs / (1.0 - test_probs))
    return (
        np.column_stack([train_logits, train_demo[DEMOGRAPHIC_FEATURE_COLUMNS].values]),
        np.column_stack([test_logits, test_demo[DEMOGRAPHIC_FEATURE_COLUMNS].values]),
    )


def fixed_meta_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.2, solver="lbfgs", max_iter=1000, random_state=42)),
        ]
    )


def predict_target(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    has_prediction = np.zeros(len(y), dtype=bool)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_count = 0

    for train_idx, valid_idx in folds:
        y_train = y[train_idx]
        has_prediction[valid_idx] = True
        if len(np.unique(y_train)) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = fixed_meta_model()
        model.fit(X_train[train_idx], y_train)
        oof[valid_idx] = model.predict_proba(X_train[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1]
        fold_count += 1

    if not has_prediction.all():
        raise ValueError("Meta folds did not produce predictions for every training row.")

    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / fold_count, 1e-6, 1.0 - 1e-6)


def trimmed_mean(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, axis=1)
    if sorted_values.shape[1] > 4:
        return sorted_values[:, 2:-2].mean(axis=1)
    if sorted_values.shape[1] > 2:
        return sorted_values[:, 1:-1].mean(axis=1)
    return sorted_values.mean(axis=1)


def predict_trimmed_mean(
    target: str,
    loaded_models: list[str],
    oof_by_model: dict[str, pd.DataFrame],
    sub_by_model: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    train_probs = np.column_stack(
        [oof_by_model[name][target].astype(float).fillna(0.5).clip(1e-6, 1.0 - 1e-6).values for name in loaded_models]
    )
    test_probs = np.column_stack(
        [sub_by_model[name][target].astype(float).fillna(0.5).clip(1e-6, 1.0 - 1e-6).values for name in loaded_models]
    )
    return (
        np.clip(trimmed_mean(train_probs), 1e-6, 1.0 - 1e-6),
        np.clip(trimmed_mean(test_probs), 1e-6, 1.0 - 1e-6),
    )


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    base_tag: str,
    local_cv: float,
    loaded_models: list[str],
    method: str,
) -> None:
    tag_arg = "--base-tags" if "," in base_tag else "--base-tag"
    method_note = (
        "The generating script uses a fixed trimmed mean with no learned weights."
        if method == "trimmed-mean"
        else "The generating script uses fixed logistic meta-stacking and official paper demographics only."
    )
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: strict fixed meta-stacking\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base tag: `{base_tag}`\n"
            f"- Method: `{method}`\n"
            f"- Loaded base models: `{', '.join(loaded_models)}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for the immutable artifact. "
            f"{method_note} It does not use DACON public labels or same-OOF hyperparameter searches."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/run_strict_meta_stacking_experiment.py "
            f"--experiment-id {experiment_id} {tag_arg} {base_tag} --method {method}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    log_path: Path,
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
    model_description: str,
) -> None:
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    if (log_df["id"].astype(str) == str(experiment_id)).any():
        raise ValueError(f"submission_log.csv already has id={experiment_id}")

    row = dict.fromkeys(log_df.columns, "")
    row.update(
        {
            "id": str(experiment_id),
            "date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "submission_file": submission_path.name,
            "model_description": model_description,
            "local_cv_score": f"{local_cv:.6f}",
            "public_lb_score": "",
            "is_submitted": "False",
            "notes": notes,
            "is_submitted_bool": "False",
            "local_cv_score_num": f"{local_cv:.6f}",
            "public_lb_score_num": "",
            "cv_lb_gap": "",
            "artifact_exists": "True",
            "status_clean": "generated_unsubmitted",
            "risk_reason": "Generated locally but not submitted/scored.",
            "safe_for_default_ensemble": "False",
            "artifact_role": "submission",
            "duplicate_file_count": "1",
            "notebook_file": str(notebook_path.relative_to(ROOT)),
        }
    )
    pd.concat([log_df, pd.DataFrame([row])], ignore_index=True).to_csv(log_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--base-tag", default="")
    parser.add_argument("--base-tags", default="")
    parser.add_argument("--slug", default="demographics_v32_strict_meta")
    parser.add_argument("--method", choices=["fixed-logistic", "trimmed-mean"], default="fixed-logistic")
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    data_loader = DataLoader(data_dir=str(ROOT / "data"))
    metrics = data_loader.load_metrics()
    sample = data_loader.load_sample_submission()
    if metrics is None or sample is None:
        raise ValueError("Missing metrics or sample submission.")

    id_col = "subject_id"
    metrics = metrics.copy()
    sample_original = sample.copy()
    metrics["date"] = extract_date(metrics)
    sample_sorted = sample.copy()
    sample_sorted["date"] = extract_date(sample_sorted)
    metrics_sorted = metrics.sort_values([id_col, "date"]).reset_index(drop=True)
    sample_sorted = sample_sorted.sort_values([id_col, "date"]).reset_index(drop=True)

    base_tags = [tag.strip() for tag in args.base_tags.split(",") if tag.strip()]
    if not base_tags and args.base_tag:
        base_tags = [args.base_tag]
    if not base_tags:
        raise ValueError("Pass --base-tag or comma-separated --base-tags.")

    loaded_models, oof_by_model, sub_by_model = load_base_predictions(base_tags, metrics_sorted, sample_sorted)

    folds = make_subject_hole_folds(metrics_sorted, n_folds=5)

    final_oof: dict[str, np.ndarray] = {}
    final_test: dict[str, np.ndarray] = {}
    target_scores: dict[str, float] = {}

    for target in TARGETS:
        y = metrics_sorted[target].astype(int).values
        if args.method == "trimmed-mean":
            oof_pred, test_pred = predict_trimmed_mean(target, loaded_models, oof_by_model, sub_by_model)
        else:
            train_demo = add_official_demographics(metrics_sorted[[id_col]].copy())
            test_demo = add_official_demographics(sample_sorted[[id_col]].copy())
            X_train, X_test = make_meta_features(
                target, loaded_models, oof_by_model, sub_by_model, train_demo, test_demo
            )
            oof_pred, test_pred = predict_target(X_train, y, X_test, folds)
        final_oof[target] = oof_pred
        final_test[target] = test_pred
        target_scores[target] = float(log_loss(y, oof_pred, labels=[0, 1]))

    local_cv = float(np.mean(list(target_scores.values())))
    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.json"
    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name

    for path in [submission_path, notebook_path, metadata_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

    sample_sorted_out = sample_sorted.copy()
    for target in TARGETS:
        sample_sorted_out[target] = final_test[target]

    pred_map = {target: sample_sorted_out.set_index([id_col, "date"])[target].to_dict() for target in TARGETS}
    out = sample_original.copy()
    original_dates = extract_date(out)
    keys = list(zip(out[id_col], original_dates, strict=True))
    for target in TARGETS:
        out[target] = [pred_map[target][key] for key in keys]

    out = clip_probabilities(out, TARGETS)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(submission_path, index=False)

    metadata = {
        "experiment_id": args.experiment_id,
        "base_tags": base_tags,
        "slug": args.slug,
        "method": args.method,
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "loaded_models": loaded_models,
        "targets": TARGETS,
        "validation": (
            "Fixed trimmed mean; no learned weights or post-hoc tuning."
            if args.method == "trimmed-mean"
            else "Subject-Hole outer folds; fixed LogisticRegression(C=0.2); no Optuna/post-hoc tuning."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        base_tag=",".join(base_tags),
        local_cv=local_cv,
        loaded_models=loaded_models,
        method=args.method,
    )

    feature_note = (
        "No learned meta features were used."
        if args.method == "trimmed-mean"
        else "Official ETRI paper demographics were used as static meta features."
    )
    notes = (
        f"Strict {args.method} meta-stacker over {','.join(base_tags)} base OOF/submission files, "
        f"{feature_note} No DACON submission yet."
    )
    model_description = (
        "Strict fixed trimmed-mean stacker over recovered base predictions"
        if args.method == "trimmed-mean"
        else "Strict fixed logistic meta-stacker with official ETRI demographics"
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=model_description,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
