#!/usr/bin/env python3
"""Promote Q2/Q3 pre-sleep strain features into the strict artifact contract."""

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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

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
from src.features.q_context import build_q23_strain_features  # noqa: E402
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

Q23_TARGETS = ["Q2", "Q3"]
RAW_CONTEXT_DIR = ROOT / "data" / "raw" / "ch2025_data_items"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q23_model_pool() -> dict[str, object]:
    """Fixed model family for the Q2/Q3 strain overlay.

    The pool is intentionally small and non-Optuna. Fold selection is nested,
    so the validation rows are not used to choose the candidate that predicts
    those same rows.
    """

    return {
        "et8": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=42,
            n_jobs=-1,
        ),
        "et5": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "rf6": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.9,
            random_state=42,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.03,
            l2_regularization=2.0,
            max_leaf_nodes=7,
            random_state=42,
        ),
    }


def strain_feature_columns(train_q23: pd.DataFrame, test_q23: pd.DataFrame) -> list[str]:
    excluded = set(TARGETS + ["subject_id", "date"])
    cols: list[str] = []
    for col in train_q23.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(train_q23[col]) and (
            train_q23[col].notna().any() or test_q23[col].notna().any()
        ):
            cols.append(col)
    return cols


def attach_q23_strain_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_context_dir: Path = RAW_CONTEXT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not raw_context_dir.exists():
        raise FileNotFoundError(f"Missing raw context directory: {raw_context_dir.relative_to(ROOT)}")

    strain = build_q23_strain_features(raw_context_dir)
    strain = strain.copy()
    strain["date"] = pd.to_datetime(strain["date"]).dt.date
    if strain.duplicated(["subject_id", "date"]).any():
        duplicated = strain.loc[strain.duplicated(["subject_id", "date"], keep=False), ["subject_id", "date"]]
        raise ValueError(f"Q23 strain features are not unique by subject/date:\n{duplicated.head()}")

    train_key = train[["subject_id", "date", *TARGETS]].copy()
    test_key = test[["subject_id", "date"]].copy()
    train_key["date"] = pd.to_datetime(train_key["date"]).dt.date
    test_key["date"] = pd.to_datetime(test_key["date"]).dt.date

    train_q23 = train_key.merge(strain, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_q23 = test_key.merge(strain, on=["subject_id", "date"], how="left", validate="one_to_one")
    missing_train = int(train_q23.filter(like="strain_").isna().all(axis=1).sum())
    missing_test = int(test_q23.filter(like="strain_").isna().all(axis=1).sum())
    if missing_train or missing_test:
        raise ValueError(f"Missing all Q23 strain features for train={missing_train}, test={missing_test} rows.")

    info = {
        "raw_context_dir": str(raw_context_dir.relative_to(ROOT)),
        "strain_shape": list(strain.shape),
        "train_shape": list(train_q23.shape),
        "test_shape": list(test_q23.shape),
        "missing_all_strain_rows": {"train": missing_train, "test": missing_test},
    }
    return train_q23, test_q23, info


def fit_predict_one_q23_candidate(
    model_proto: object,
    train_q23: pd.DataFrame,
    test_q23: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train_q23, n_folds=5)
    y = train_q23[target].astype(int).reset_index(drop=True)
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
        raise ValueError(f"Folds did not produce predictions for every row for target {target}.")
    test_pred /= len(folds)
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred, 1e-6, 1.0 - 1e-6)


def predict_q23_strain_candidates(
    train_q23: pd.DataFrame,
    test_q23: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    candidate_oof = {name: np.zeros((len(train_q23), len(Q23_TARGETS)), dtype=float) for name in q23_model_pool()}
    candidate_test = {name: np.zeros((len(test_q23), len(Q23_TARGETS)), dtype=float) for name in q23_model_pool()}
    model_scores = {name: {} for name in q23_model_pool()}

    for candidate_name, model_proto in q23_model_pool().items():
        for target_idx, target in enumerate(Q23_TARGETS):
            oof_values, test_values = fit_predict_one_q23_candidate(
                model_proto=model_proto,
                train_q23=train_q23,
                test_q23=test_q23,
                feature_cols=feature_cols,
                target=target,
            )
            candidate_oof[candidate_name][:, target_idx] = oof_values
            candidate_test[candidate_name][:, target_idx] = test_values
            model_scores[candidate_name][target] = float(
                log_loss(train_q23[target].astype(int), oof_values, labels=[0, 1])
            )

    return candidate_oof, candidate_test, model_scores


def load_id101_base(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob("base_oof_id_101_shutdown_feature_set_selector_cv*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob("base_test_id_101_shutdown_feature_set_selector_cv*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError("Expected exactly one ID101 base OOF/test artifact.")
    base_oof = align_candidate(pd.read_csv(oof_paths[0]), train)
    base_test = align_candidate(pd.read_csv(test_paths[0]), test)
    return base_oof, base_test, f"{oof_paths[0].name}; {test_paths[0].name}"


def predict_q23_strain_overlay() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_id101_base(train, test)
    train_q23, test_q23, feature_info = attach_q23_strain_features(train, test)
    feature_cols = strain_feature_columns(train_q23, test_q23)
    if not feature_cols:
        raise ValueError("No numeric Q23 strain feature columns were generated.")

    q23_candidate_oof, q23_candidate_test, model_scores = predict_q23_strain_candidates(
        train_q23=train_q23,
        test_q23=test_q23,
        feature_cols=feature_cols,
    )

    candidate_oof = {"id101_base": base_oof[Q23_TARGETS].astype(float).to_numpy(), **q23_candidate_oof}
    candidate_test = {"id101_base": base_test[Q23_TARGETS].astype(float).to_numpy(), **q23_candidate_test}

    folds = make_subject_hole_folds(train_q23, n_folds=5)
    selected_q23_oof = np.zeros((len(train_q23), len(Q23_TARGETS)), dtype=float)
    nested_choices: dict[str, list[dict]] = {target: [] for target in Q23_TARGETS}

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        for target_idx, target in enumerate(Q23_TARGETS):
            y = train_q23[target].astype(int).values
            losses = {
                name: log_loss(y[train_idx], values[train_idx, target_idx], labels=[0, 1])
                for name, values in candidate_oof.items()
            }
            best_name, best_loss = min(losses.items(), key=lambda item: item[1])
            selected_q23_oof[valid_idx, target_idx] = candidate_oof[best_name][valid_idx, target_idx]
            nested_choices[target].append(
                {
                    "fold": fold_idx,
                    "candidate": best_name,
                    "train_fold_log_loss": float(best_loss),
                    "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
                }
            )

    selected_q23_test = np.zeros((len(test_q23), len(Q23_TARGETS)), dtype=float)
    full_train_test_choices: dict[str, dict] = {}
    for target_idx, target in enumerate(Q23_TARGETS):
        y = train_q23[target].astype(int).values
        losses = {name: log_loss(y, values[:, target_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q23_test[:, target_idx] = candidate_test[best_name][:, target_idx]
        full_train_test_choices[target] = {
            "candidate": best_name,
            "full_oof_log_loss": float(best_loss),
            "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
        }

    oof = base_oof.copy()
    test_pred = base_test.copy()
    for target_idx, target in enumerate(Q23_TARGETS):
        oof[target] = np.clip(selected_q23_oof[:, target_idx], 1e-6, 1.0 - 1e-6)
        test_pred[target] = np.clip(selected_q23_test[:, target_idx], 1e-6, 1.0 - 1e-6)

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_info": feature_info,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "q23_candidate_scores": model_scores,
        "nested_choices": nested_choices,
        "full_train_test_choices": full_train_test_choices,
        "selection_contract": (
            "ID101 predictions are preserved for Q1/S1/S2/S3/S4. Q2/Q3 validation rows receive the candidate "
            "chosen using only each outer fold's training rows. Final test predictions use the candidate with "
            "the best full-training OOF log loss per Q target."
        ),
        "contamination_note": (
            "Q23 strain features use label-free same-day 12:00-24:00 raw context sensors. No post-midnight "
            "sensor windows or target-derived future interpolation are used. WiFi home-anchor features are "
            "computed from all available unlabeled context rows, so they are transductive covariates rather "
            "than fold-fitted preprocessing."
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
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: Q2/Q3 pre-sleep strain overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            "The generating script keeps ID101 predictions for Q1/S1/S2/S3/S4 and replaces only Q2/Q3 "
            "with a fold-nested selector over pre-sleep strain candidates."
        ),
        nbf.v4.new_code_cell(
            f".venv/bin/python scripts/run_strict_q23_strain_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap. "
            "The metadata explicitly flags the label-free transductive WiFi home-anchor feature."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
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
            "model_description": "Strict Q2/Q3 pre-sleep strain overlay on ID101",
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
    parser.add_argument("--slug", default="q23_strain_overlay")
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_q23_strain_overlay()
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
        "method": "q23 pre-sleep strain overlay",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "feature_contract": (
            "Overlay ID101 with Q2/Q3-only pre-sleep strain features. Q2/Q3 candidate selection is fold-nested; "
            "no future target interpolation, Optuna, same-OOF clipping, or DACON submission."
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
    )

    notes = (
        "Strict Q2/Q3 pre-sleep strain overlay on ID101. Uses same-day 12:00-24:00 GPS/WiFi/usage/"
        "ambience/BLE context features and fold-nested Q-target candidate selection. No future target "
        "interpolation, Optuna, same-OOF clipping, or DACON submission. WiFi home anchor is label-free "
        "transductive and recorded in metadata."
    )
    append_submission_log(
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
