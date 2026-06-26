#!/usr/bin/env python3
"""Promote a Q1-only morning-recovery overlay on top of ID106."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
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

from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.features.morning_recovery import (  # noqa: E402
    build_morning_recovery_features,
    morning_recovery_feature_columns,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

TARGET = "Q1"
DEFAULT_BASE_ID = 106
OVERLAY_WEIGHT = 0.25


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base_artifact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF/test artifact.")
    base_oof = align_candidate(pd.read_csv(oof_paths[0]), train)
    base_test = align_candidate(pd.read_csv(test_paths[0]), test)
    return base_oof, base_test, f"{oof_paths[0].name}; {test_paths[0].name}"


def build_q1_morning_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    morning = build_morning_recovery_features(ROOT / "data" / "raw" / "ch2025_data_items")
    morning = morning.copy()
    morning["date"] = pd.to_datetime(morning["date"]).dt.date

    train_aug = train.copy()
    test_aug = test.copy()
    train_aug["date"] = pd.to_datetime(train_aug["date"]).dt.date
    test_aug["date"] = pd.to_datetime(test_aug["date"]).dt.date
    train_aug = train_aug.merge(morning, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_aug = test_aug.merge(morning, on=["subject_id", "date"], how="left", validate="one_to_one")
    return train_aug, test_aug, sample_original


def make_morning_lr() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42),
    )


def fit_predict_morning_lr(
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
            model = clone(make_morning_lr())
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q1 row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q1_morning_recovery_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original = build_q1_morning_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    feature_cols = morning_recovery_feature_columns(train, test)
    if not feature_cols:
        raise ValueError("No numeric morning-recovery feature columns were generated.")

    y = train[TARGET].astype(int).values
    morning_oof, morning_test = fit_predict_morning_lr(train=train, test=test, feature_cols=feature_cols)
    base_candidate_name = f"id{base_id:03d}_base"
    overlay_candidate_name = "blend_base0.75_morning_lr_l2_c1_0.25"
    candidate_oof = {
        base_candidate_name: base_oof[TARGET].astype(float).to_numpy(),
        overlay_candidate_name: np.clip(
            (1.0 - OVERLAY_WEIGHT) * base_oof[TARGET].astype(float).to_numpy() + OVERLAY_WEIGHT * morning_oof,
            1e-6,
            1.0 - 1e-6,
        ),
    }
    candidate_test = {
        base_candidate_name: base_test[TARGET].astype(float).to_numpy(),
        overlay_candidate_name: np.clip(
            (1.0 - OVERLAY_WEIGHT) * base_test[TARGET].astype(float).to_numpy() + OVERLAY_WEIGHT * morning_test,
            1e-6,
            1.0 - 1e-6,
        ),
    }
    candidate_scores = {
        name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())
    }

    folds = make_subject_hole_folds(train, n_folds=5)
    selected_q1_oof = np.zeros(len(train), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {
            name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()
        }
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q1_oof[valid_idx] = candidate_oof[best_name][valid_idx]
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
    oof[TARGET] = np.clip(selected_q1_oof, 1e-6, 1.0 - 1e-6)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], 1e-6, 1.0 - 1e-6)

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "n_features": len(feature_cols),
        "overlay_weight": OVERLAY_WEIGHT,
        "morning_lr_full_oof_log_loss": float(log_loss(y, morning_oof, labels=[0, 1])),
        "q1_candidate_scores": candidate_scores,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(full_losses.items())},
            }
        },
        "candidate_pool_origin": (
            "Promoted from the Q1 morning-recovery diagnostic direction in "
            "reports/strict_q1_morning_recovery_diagnostic.json. The final strict script uses a narrower core "
            "morning-recovery feature module and a fixed 75/25 ID106+morning-LR overlay; OOF selection between "
            "ID106 base and the fixed overlay remains fold-nested."
        ),
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q1. Q1 validation rows receive "
            "the candidate selected using only each outer fold's training rows. Final test Q1 uses the candidate "
            "with the best full-training OOF log loss."
        ),
        "feature_contract": (
            "Morning-recovery features use label-free post-sleep 04:00-12:00 sensor windows assigned back to the "
            "previous lifelog date, matching Q1's post-wake self-report timing. No future target interpolation, "
            "Optuna, same-OOF clipping, or DACON submission is used."
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
            f"# Submission {experiment_id:03d}: Q1 morning-recovery overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q1 target and replaces Q1 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q1_morning_recovery_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
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
            "model_description": f"Strict Q1 morning-recovery overlay on ID{base_id:03d}",
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
    parser.add_argument("--slug", default="q1_morning_recovery_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_q1_morning_recovery_overlay(
        base_id=args.base_id,
    )
    local_cv = float(np.mean(list(target_scores.values())))

    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"

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
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "Q1 morning-recovery overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with a Q1-only fixed core morning-recovery overlay promoted after a prior diagnostic.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        base_id=args.base_id,
    )

    notes = (
        f"Strict Q1-only morning-recovery overlay on ID{args.base_id:03d}. Uses post-sleep 04:00-12:00 "
        "sensor windows assigned to the previous lifelog date and a fixed 75/25 blend between ID106 Q1 and "
        "morning-only logistic predictions. Candidate was chosen after a local diagnostic; no future target "
        "interpolation, Optuna, same-OOF clipping, or DACON submission."
    )
    append_submission_log(
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        base_id=args.base_id,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
