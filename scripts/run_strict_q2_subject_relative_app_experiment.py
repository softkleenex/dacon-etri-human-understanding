#!/usr/bin/env python3
"""Diagnose/promote a compact Q2 subject-relative app-token overlay.

This combines the two currently productive strict feature families:
- Q2 top-app-token usage features from raw mUsageStats.
- Label-free per-subject robust/rank normalization that helped Q3 q23 strain.

The default mode writes only a diagnostic JSON. Use --promote with a fresh
--experiment-id to create immutable submission/notebook/log/base artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_app_token_compact_experiment import build_q2_app_frames  # noqa: E402
from scripts.run_strict_q2_booster_compact_experiment import load_base_artifact  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "Q2"
DEFAULT_BASE_ID = 116
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def app_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [col for col in train.columns if col.startswith("app_") and col in test.columns]


def add_subject_relative_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_cols: list[str],
    prefix: str = "rel_app",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add per-subject robust-z and percentile-rank features from unlabeled covariates."""

    combined = pd.concat(
        [
            train[["subject_id", *source_cols]].assign(_part="train"),
            test[["subject_id", *source_cols]].assign(_part="test"),
        ],
        ignore_index=True,
    )
    relative = pd.DataFrame(index=combined.index)
    relative["_part"] = combined["_part"]

    for col in source_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        grouped = values.groupby(combined["subject_id"])
        median = grouped.transform("median")
        q25 = grouped.transform(lambda series: series.quantile(0.25))
        q75 = grouped.transform(lambda series: series.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        rank = values.groupby(combined["subject_id"]).rank(pct=True, method="average")
        safe_name = col.replace("app_", "", 1)
        relative[f"{prefix}_iqr_{safe_name}"] = ((values - median) / (iqr + 1e-6)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        relative[f"{prefix}_rank_{safe_name}"] = rank.fillna(0.5)

    rel_train = relative[relative["_part"] == "train"].drop(columns=["_part"]).reset_index(drop=True)
    rel_test = relative[relative["_part"] == "test"].drop(columns=["_part"]).reset_index(drop=True)
    rel_cols = list(rel_train.columns)
    train_out = pd.concat([train.reset_index(drop=True), rel_train], axis=1)
    test_out = pd.concat([test.reset_index(drop=True), rel_test], axis=1)
    return train_out, test_out, rel_cols


def compact_relative_app_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=140,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=18,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=25.0,
        random_state=42,
        verbosity=-1,
    )


def fit_predict_relative_app_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train, n_folds=5)
    y = train[TARGET].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train), dtype=float)
    has_prediction = np.zeros(len(train), dtype=bool)
    test_pred = np.zeros(len(test), dtype=float)

    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train.iloc[train_idx][feature_cols])
        x_valid = imputer.transform(train.iloc[valid_idx][feature_cols])
        x_test = imputer.transform(test[feature_cols])
        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(compact_relative_app_model())
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q2 row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q2_subject_relative_app_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original, top_apps = build_q2_app_frames()
    app_cols = app_feature_columns(train, test)
    if not app_cols:
        raise ValueError("No app token feature columns were generated.")
    train_rel, test_rel, rel_cols = add_subject_relative_features(train, test, app_cols)
    base_oof, base_test, base_source = load_base_artifact(train_rel, test_rel, base_id)

    y = train_rel[TARGET].astype(int).values
    base_candidate_name = f"id{base_id:03d}_base"
    raw_candidate_name = "rel_app_iqr_rank__lgbm_l7_lr025_n140_l25"
    raw_oof, raw_test = fit_predict_relative_app_model(train_rel, test_rel, rel_cols)

    candidate_oof = {
        base_candidate_name: base_oof[TARGET].astype(float).to_numpy(),
        raw_candidate_name: raw_oof,
    }
    candidate_test = {
        base_candidate_name: base_test[TARGET].astype(float).to_numpy(),
        raw_candidate_name: raw_test,
    }
    candidate_details = {
        base_candidate_name: {"source": "base"},
        raw_candidate_name: {
            "feature_set": "subject_relative_app_iqr_rank",
            "model": "lgbm_l7_lr025_n140_l25",
            "n_features": len(rel_cols),
            "source_app_features": len(app_cols),
            "blend": False,
        },
    }
    for weight in BLEND_WEIGHTS:
        blend_name = f"blend_base{1.0 - weight:.2f}_{raw_candidate_name}_{weight:.2f}"
        candidate_oof[blend_name] = np.clip(
            (1.0 - weight) * candidate_oof[base_candidate_name] + weight * raw_oof,
            1e-6,
            1.0 - 1e-6,
        )
        candidate_test[blend_name] = np.clip(
            (1.0 - weight) * candidate_test[base_candidate_name] + weight * raw_test,
            1e-6,
            1.0 - 1e-6,
        )
        candidate_details[blend_name] = {
            "blend_candidate": raw_candidate_name,
            "base_weight": 1.0 - weight,
            "candidate_weight": weight,
        }

    candidate_scores = {
        name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())
    }
    folds = make_subject_hole_folds(train_rel, n_folds=5)
    selected_q2_oof = np.zeros(len(train_rel), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {
            name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()
        }
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q2_oof[valid_idx] = candidate_oof[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_fold_log_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_losses = {name: log_loss(y, values, labels=[0, 1]) for name, values in candidate_oof.items()}
    full_best_name, full_best_loss = min(full_losses.items(), key=lambda item: item[1])

    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[TARGET] = np.clip(selected_q2_oof, 1e-6, 1.0 - 1e-6)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], 1e-6, 1.0 - 1e-6)

    target_scores = {
        target: float(log_loss(train_rel[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "subject_relative_top_app_token_usage",
        "top_n_apps": len(top_apps),
        "top_apps": top_apps,
        "n_source_app_features": len(app_cols),
        "n_features": len(rel_cols),
        "candidate_pool_size": len(candidate_oof),
        "candidate_pool_origin": (
            "New ID116 diagnostic: combine top-app-token Q2 signal with label-free per-subject robust/rank "
            "normalization. This is a compact one-family pool to avoid broad selector overfit."
        ),
        "q2_candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(full_losses.items())},
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q2. Q2 validation rows receive "
            "the subject-relative app candidate selected using only each outer fold's training rows. Final test Q2 "
            "uses the candidate with the best full-training OOF log loss."
        ),
        "contamination_note": (
            "Top-app vocabulary and per-subject relative app statistics are computed from unlabeled train/test "
            "covariate rows. This is label-free transductive normalization and uses no target labels."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: Q2 subject-relative app-token overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q2 target and replaces Q2 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q2_subject_relative_app_experiment.py "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
    base_id: int,
) -> None:
    log_path = SUBMISSIONS_DIR / "submission_log.csv"
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    if (log_df["id"].astype(str) == str(experiment_id)).any():
        raise ValueError(f"submission_log.csv already has id={experiment_id}")

    row = dict.fromkeys(log_df.columns, "")
    row.update(
        {
            "id": str(experiment_id),
            "date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "submission_file": submission_path.name,
            "model_description": f"Strict Q2 subject-relative app-token overlay on ID{base_id:03d}",
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


def write_diagnostic_report(
    base_id: int,
    local_cv: float,
    target_scores: dict[str, float],
    selector_info: dict,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q2_subject_relative_app_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q2_subject_relative_app_diagnostic_id{base_id:03d}_{suffix}.json"
    report = {
        "base_id": base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "selector_info": selector_info,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def promote_artifacts(
    experiment_id: int,
    slug: str,
    base_id: int,
    oof: pd.DataFrame,
    test_pred: pd.DataFrame,
    target_scores: dict[str, float],
    selector_info: dict,
    sample_original: pd.DataFrame,
) -> dict:
    if experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")
    local_cv = float(np.mean(list(target_scores.values())))
    status = "unsubmitted"
    filename = f"submission_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name

    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "slug": slug,
        "method": "Q2 compact subject-relative app-token overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with a Q2-only compact LGBM blend over subject-relative top app-token usage features.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=experiment_id,
        slug=slug,
        local_cv=local_cv,
        base_id=base_id,
    )

    notes = (
        f"Strict Q2-only subject-relative app-token overlay on ID{base_id:03d}. Uses per-subject IQR/rank "
        "normalization of top-app duration/share features from raw mUsageStats; compact one-family LGBM blend. "
        "No DACON submission was made."
    )
    append_submission_log(
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        base_id=base_id,
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q2_subject_relative_app_compact_overlay")
    args = parser.parse_args()

    oof, test_pred, target_scores, selector_info, sample_original = predict_q2_subject_relative_app_overlay(
        base_id=args.base_id,
    )
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)

    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "q2_candidate_scores": selector_info["q2_candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }

    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = promote_artifacts(
            experiment_id=args.experiment_id,
            slug=args.slug,
            base_id=args.base_id,
            oof=oof,
            test_pred=test_pred,
            target_scores=target_scores,
            selector_info=selector_info,
            sample_original=sample_original,
        )
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
