#!/usr/bin/env python3
"""Diagnose/promote a compact Q3 semantic app-category overlay."""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_app_category_compact_experiment import (  # noqa: E402
    app_category_feature_sets,
    build_q2_app_category_frames,
    sha256_file,
)
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

TARGET = "Q3"
DEFAULT_BASE_ID = 116
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]


def compact_model_pool() -> dict[str, object]:
    return {
        "lgbm_l7_lr02_n180_l30": LGBMClassifier(
            n_estimators=180,
            learning_rate=0.02,
            num_leaves=7,
            max_depth=3,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_lambda=30.0,
            random_state=42,
            verbosity=-1,
        ),
        "lgbm_l15_lr015_n220_l50": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.65,
            reg_lambda=50.0,
            random_state=43,
            verbosity=-1,
        ),
        "lr_l2_c02": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=1000, solver="liblinear", random_state=42),
        ),
    }


def compact_candidate_pairs() -> list[tuple[str, str]]:
    return [
        ("appcat_abs", "lgbm_l7_lr02_n180_l30"),
        ("appcat_abs_rel", "lgbm_l15_lr015_n220_l50"),
        ("appcat_abs", "lr_l2_c02"),
    ]


def fit_predict_one_candidate(
    model_proto: object,
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
            model = clone(model_proto)
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q3 row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q3_app_category_overlay(base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original, category_report = build_q2_app_category_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    feature_sets = app_category_feature_sets(train, test)
    model_pool = compact_model_pool()

    y = train[TARGET].astype(int).values
    base_candidate_name = f"id{base_id:03d}_base"
    candidate_oof = {base_candidate_name: base_oof[TARGET].astype(float).to_numpy()}
    candidate_test = {base_candidate_name: base_test[TARGET].astype(float).to_numpy()}
    candidate_details = {base_candidate_name: {"source": "base"}}

    for feature_set_name, model_name in compact_candidate_pairs():
        feature_cols = feature_sets[feature_set_name]
        if not feature_cols:
            raise ValueError(f"No features generated for {feature_set_name}.")
        raw_oof, raw_test = fit_predict_one_candidate(model_pool[model_name], train, test, feature_cols)
        raw_name = f"{feature_set_name}__{model_name}"
        candidate_oof[raw_name] = raw_oof
        candidate_test[raw_name] = raw_test
        candidate_details[raw_name] = {
            "feature_set": feature_set_name,
            "model": model_name,
            "n_features": len(feature_cols),
            "blend": False,
        }
        for weight in BLEND_WEIGHTS:
            blend_name = f"blend_base{1.0 - weight:.2f}_{raw_name}_{weight:.2f}"
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
                "blend_candidate": raw_name,
                "base_weight": 1.0 - weight,
                "candidate_weight": weight,
            }

    candidate_scores = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())}
    folds = make_subject_hole_folds(train, n_folds=5)
    selected_q3_oof = np.zeros(len(train), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q3_oof[valid_idx] = candidate_oof[best_name][valid_idx]
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
    oof[TARGET] = np.clip(selected_q3_oof, 1e-6, 1.0 - 1e-6)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], 1e-6, 1.0 - 1e-6)

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "semantic_app_category_usage",
        "category_report": category_report,
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "candidate_pool_size": len(candidate_oof),
        "q3_candidate_scores": candidate_scores,
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
            f"ID{base_id:03d} predictions are preserved for all targets except Q3. Q3 validation rows receive "
            "the compact semantic app-category candidate selected using only each outer fold's training rows. "
            "Final test Q3 uses the candidate with the best full-training OOF log loss."
        ),
        "contamination_note": (
            "App category features are constructed from unlabeled train/test mUsageStats app names and durations. "
            "The category dictionary is hand-written before target fitting and uses no target labels."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q3_app_category_compact_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q3_app_category_compact_diagnostic_id{base_id:03d}_{suffix}.json"
    path.write_text(
        json.dumps({"base_id": base_id, "local_cv": local_cv, "target_scores": target_scores, "selector_info": selector_info}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


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
            f"# Submission {experiment_id:03d}: Q3 semantic app-category compact overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q3 target and replaces Q3 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q3_app_category_compact_experiment.py --promote "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nDo not submit before reviewing metadata and audit reports."),
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
            "model_description": f"Strict Q3 semantic app-category compact overlay on ID{base_id:03d}",
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
        "method": "Q3 compact semantic app-category overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with a Q3-only compact blend over semantic app-category features.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id)
    notes = (
        f"Strict Q3-only compact semantic app-category overlay on ID{base_id:03d}. Compresses raw mUsageStats "
        "app names into auditable categories and time windows; no DACON submission was made."
    )
    append_submission_log(experiment_id, submission_path, notebook_path, local_cv, notes, base_id)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q3_app_category_compact_overlay")
    args = parser.parse_args()

    oof, test_pred, target_scores, selector_info, sample_original = predict_q3_app_category_overlay(base_id=args.base_id)
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "q3_candidate_scores": selector_info["q3_candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = promote_artifacts(
            args.experiment_id,
            args.slug,
            args.base_id,
            oof,
            test_pred,
            target_scores,
            selector_info,
            sample_original,
        )
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
