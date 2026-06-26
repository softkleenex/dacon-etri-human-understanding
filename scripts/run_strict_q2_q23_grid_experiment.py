#!/usr/bin/env python3
"""Promote a Q2-only q23-strain GB grid overlay on top of ID105."""

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
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q23_strain_experiment import (  # noqa: E402
    attach_q23_strain_features,
    strain_feature_columns,
)
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
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

TARGET = "Q2"
DEFAULT_BASE_ID = 105


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q2_q23_model_pool() -> dict[str, object]:
    """Compact GB/HGB neighborhood found promising in diagnostic search.

    The OOF score reported by the artifact remains fold-nested over this pool.
    The pool itself is a research choice, so metadata records the diagnostic
    origin instead of pretending this is an untouched independent proof.
    """

    models: dict[str, object] = {}
    for n_estimators in [140, 180]:
        for learning_rate in [0.02, 0.03, 0.05]:
            for max_depth in [1, 2]:
                for min_leaf in [8, 12, 20]:
                    name = f"gb_n{n_estimators}_lr{learning_rate:g}_d{max_depth}_l{min_leaf}"
                    models[name] = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        subsample=0.9,
                        random_state=42,
                    )
    for max_iter in [140, 200]:
        for learning_rate in [0.02, 0.03]:
            for l2 in [5.0, 10.0]:
                name = f"hgb_i{max_iter}_lr{learning_rate:g}_l2{l2:g}"
                models[name] = HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=l2,
                    max_leaf_nodes=7,
                    random_state=42,
                )
    return models


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


def fit_predict_one_candidate(
    model_proto: object,
    train_q23: pd.DataFrame,
    test_q23: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train_q23, n_folds=5)
    y = train_q23[TARGET].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train_q23), dtype=float)
    has_prediction = np.zeros(len(train_q23), dtype=bool)
    test_pred = np.zeros(len(test_q23), dtype=float)

    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train_q23.iloc[train_idx][feature_cols])
        x_valid = imputer.transform(train_q23.iloc[valid_idx][feature_cols])
        x_test = imputer.transform(test_q23[feature_cols])

        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(model_proto)
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every row for Q2.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q2_q23_grid_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    train_q23, test_q23, feature_info = attach_q23_strain_features(train, test)
    feature_cols = strain_feature_columns(train_q23, test_q23)
    if not feature_cols:
        raise ValueError("No numeric q23 strain feature columns were generated.")

    base_candidate_name = f"id{base_id:03d}_base"
    candidate_oof = {base_candidate_name: base_oof[TARGET].astype(float).to_numpy()}
    candidate_test = {base_candidate_name: base_test[TARGET].astype(float).to_numpy()}
    y = train_q23[TARGET].astype(int).values
    candidate_scores: dict[str, float] = {
        base_candidate_name: float(log_loss(y, candidate_oof[base_candidate_name], labels=[0, 1]))
    }

    for model_name, model_proto in q2_q23_model_pool().items():
        oof_values, test_values = fit_predict_one_candidate(
            model_proto=model_proto,
            train_q23=train_q23,
            test_q23=test_q23,
            feature_cols=feature_cols,
        )
        candidate_oof[model_name] = oof_values
        candidate_test[model_name] = test_values
        candidate_scores[model_name] = float(log_loss(y, oof_values, labels=[0, 1]))

    folds = make_subject_hole_folds(train_q23, n_folds=5)
    selected_q2_oof = np.zeros(len(train_q23), dtype=float)
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
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_info": feature_info,
        "n_features": len(feature_cols),
        "candidate_pool_size": len(candidate_oof),
        "candidate_pool_origin": "Narrowed from reports/strict_q2_q23_grid_diagnostic.json; OOF selection remains fold-nested.",
        "q2_candidate_scores": candidate_scores,
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
            "the q23 candidate selected using only each outer fold's training rows. Final test Q2 uses the "
            "candidate with the best full-training OOF log loss."
        ),
        "contamination_note": (
            "Q23 features are label-free same-day 12:00-24:00 context covariates. WiFi home-anchor features "
            "inherit the id102 label-free transductive caveat. The model neighborhood was chosen after a local "
            "diagnostic, so treat this as a strong local candidate rather than independent proof."
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
            f"# Submission {experiment_id:03d}: Q2 q23 GB-grid overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q2 target and replaces Q2 only."
        ),
        nbf.v4.new_code_cell(
            f".venv/bin/python scripts/run_strict_q2_q23_grid_experiment.py "
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
            "model_description": f"Strict Q2 q23 GB-grid overlay on ID{base_id:03d}",
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
    parser.add_argument("--slug", default="q2_q23_grid_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_q2_q23_grid_overlay(base_id=args.base_id)
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
        "method": "Q2 q23 GB-grid overlay",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "feature_contract": (
            f"Q2-only overlay on ID{args.base_id:03d} using q23 pre-sleep strain context features and a compact "
            "GB/HGB candidate neighborhood. Q2 candidate selection is fold-nested; no future target interpolation, "
            "Optuna, same-OOF clipping, or DACON submission."
        ),
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
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
        f"Strict Q2-only q23 GB-grid overlay on ID{args.base_id:03d}. Candidate choice for Q2 OOF is fold-nested; "
        "no future target interpolation, Optuna, same-OOF clipping, or DACON submission. Candidate neighborhood "
        "was narrowed after diagnostic search and is recorded in metadata."
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
