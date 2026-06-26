#!/usr/bin/env python3
"""Select between baseline shutdown and subject-relative shutdown candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
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
    map_predictions_to_original_sample,
    numeric_feature_columns,
    predict_nested_raw_selector,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_experiment import build_shutdown_frames  # noqa: E402
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_subject_relative_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    base_cols = numeric_feature_columns(train, test)
    combined = pd.concat([train.assign(_is_test=0), test.assign(_is_test=1)], ignore_index=True, sort=False)
    added = 0
    for col in base_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        if values.notna().mean() < 0.2:
            continue
        subject_mean = values.groupby(combined["subject_id"]).transform("mean")
        subject_std = values.groupby(combined["subject_id"]).transform("std").replace(0, np.nan)
        combined[f"{col}_subj_z_all"] = ((values - subject_mean) / (subject_std + 1e-6)).replace(
            [np.inf, -np.inf], np.nan
        )
        combined[f"{col}_subj_dev_all"] = (values - subject_mean).replace([np.inf, -np.inf], np.nan)
        added += 2
    train_aug = combined[combined["_is_test"] == 0].drop(columns=["_is_test"]).reset_index(drop=True)
    test_aug = combined[combined["_is_test"] == 1].drop(columns=["_is_test"]).reset_index(drop=True)
    return train_aug, test_aug, added


def load_or_build_base_shutdown_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob("base_oof_id_100_shutdown_nested_selector_cv*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob("base_test_id_100_shutdown_nested_selector_cv*.csv"))
    if len(oof_paths) == 1 and len(test_paths) == 1:
        return pd.read_csv(oof_paths[0]), pd.read_csv(test_paths[0]), "loaded_id100_base_artifacts"

    feature_cols = numeric_feature_columns(train, test)
    oof, test_pred, _scores, _selector_info = predict_nested_raw_selector(train, test, feature_cols)
    return oof, test_pred, "rebuilt_shutdown_base_candidate"


def align_candidate(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.sort_values(["subject_id", "date"]).reset_index(drop=True)
    ref_key = reference[["subject_id", "date"]].copy()
    ref_key["date"] = pd.to_datetime(ref_key["date"]).dt.date
    if not out[["subject_id", "date"]].equals(ref_key.reset_index(drop=True)):
        raise ValueError("Candidate frame does not align with reference subject/date order.")
    return out


def predict_feature_set_selector() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original = build_shutdown_frames()
    base_oof, base_test, base_source = load_or_build_base_shutdown_candidate(train, test)
    base_oof = align_candidate(base_oof, train)
    base_test = align_candidate(base_test, test)

    train_sr, test_sr, added_cols = add_subject_relative_features(train, test)
    sr_feature_cols = numeric_feature_columns(train_sr, test_sr)
    sr_oof, sr_test, sr_scores, sr_selector_info = predict_nested_raw_selector(train_sr, test_sr, sr_feature_cols)
    sr_oof = align_candidate(sr_oof, train)
    sr_test = align_candidate(sr_test, test)

    candidate_oof = {
        "shutdown_base_id100": base_oof[TARGETS].astype(float).to_numpy(),
        "shutdown_subject_relative": sr_oof[TARGETS].astype(float).to_numpy(),
    }
    candidate_test = {
        "shutdown_base_id100": base_test[TARGETS].astype(float).to_numpy(),
        "shutdown_subject_relative": sr_test[TARGETS].astype(float).to_numpy(),
    }

    folds = make_subject_hole_folds(train, n_folds=5)
    selected_oof = np.zeros((len(train), len(TARGETS)), dtype=float)
    nested_choices: dict[str, list[dict]] = {target: [] for target in TARGETS}
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        for target_idx, target in enumerate(TARGETS):
            y = train[target].astype(int).values
            losses = {
                name: log_loss(y[train_idx], values[train_idx, target_idx], labels=[0, 1])
                for name, values in candidate_oof.items()
            }
            best_name, best_loss = min(losses.items(), key=lambda item: item[1])
            selected_oof[valid_idx, target_idx] = candidate_oof[best_name][valid_idx, target_idx]
            nested_choices[target].append(
                {"fold": fold_idx, "candidate": best_name, "train_fold_log_loss": float(best_loss)}
            )

    target_scores = {
        target: float(log_loss(train[target].astype(int), selected_oof[:, target_idx], labels=[0, 1]))
        for target_idx, target in enumerate(TARGETS)
    }

    selected_test = np.zeros((len(test), len(TARGETS)), dtype=float)
    full_train_choices: dict[str, dict] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).values
        losses = {name: log_loss(y, values[:, target_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_test[:, target_idx] = candidate_test[best_name][:, target_idx]
        full_train_choices[target] = {"candidate": best_name, "full_oof_log_loss": float(best_loss)}

    oof = pd.DataFrame({"subject_id": train["subject_id"].values, "date": pd.to_datetime(train["date"]).dt.date})
    test_pred = pd.DataFrame({"subject_id": test["subject_id"].values, "date": pd.to_datetime(test["date"]).dt.date})
    for target_idx, target in enumerate(TARGETS):
        oof[target] = np.clip(selected_oof[:, target_idx], 1e-6, 1.0 - 1e-6)
        test_pred[target] = np.clip(selected_test[:, target_idx], 1e-6, 1.0 - 1e-6)

    selector_info = {
        "base_source": base_source,
        "subject_relative_added_columns": added_cols,
        "subject_relative_scores": sr_scores,
        "nested_choices": nested_choices,
        "full_train_test_choices": full_train_choices,
        "subject_relative_inner_selector": sr_selector_info,
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
            f"# Submission {experiment_id:03d}: shutdown feature-set selector\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            "The generating script selects between baseline shutdown and subject-relative shutdown candidates "
            "with fold-nested OOF selection."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/run_strict_shutdown_feature_set_selector_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug}"
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
            "model_description": "Strict shutdown feature-set nested selector",
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
    parser.add_argument("--slug", default="shutdown_feature_set_selector")
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_feature_set_selector()
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

    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)

    metadata = {
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "shutdown feature-set selector",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "feature_contract": (
            "Fold-nested selector between ID100-style shutdown features and subject-relative shutdown features. "
            "No future target interpolation, Optuna, or same-OOF clipping."
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
        "Strict feature-set selector between baseline shutdown and subject-relative shutdown candidates. "
        "OOF feature-set choice is fold-nested; no future target interpolation, Optuna, same-OOF clipping, "
        "or DACON submission."
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
