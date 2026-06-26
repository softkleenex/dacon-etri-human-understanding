#!/usr/bin/env python3
"""Promote a refined strict S3 date-similarity anchor on top of ID141."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_temporal_anchor_probe_experiment import add_date_dt, load_base_artifact  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    numeric_feature_columns,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 141
TARGET = "S3"
DATE_TIME_TOKENS = ["date", "day", "week", "rank", "frac", "subject_code"]
S3_REFINED_CANDIDATES: list[dict[str, float | int | str]] = [
    {"name": "dt_k5_dw0.1_w0.3", "k": 5, "date_weight": 0.10, "blend_weight": 0.30},
    {"name": "dt_k6_dw0.1_w0.3", "k": 6, "date_weight": 0.10, "blend_weight": 0.30},
    {"name": "dt_k7_dw0.05_w0.3", "k": 7, "date_weight": 0.05, "blend_weight": 0.30},
    {"name": "dt_k7_dw0.05_w0.4", "k": 7, "date_weight": 0.05, "blend_weight": 0.40},
    {"name": "dt_k7_dw0.05_w0.5", "k": 7, "date_weight": 0.05, "blend_weight": 0.50},
    {"name": "dt_k7_dw0.05_w0.6", "k": 7, "date_weight": 0.05, "blend_weight": 0.60},
    {"name": "dt_k7_dw0.1_w0.3", "k": 7, "date_weight": 0.10, "blend_weight": 0.30},
    {"name": "dt_k7_dw0.1_w0.4", "k": 7, "date_weight": 0.10, "blend_weight": 0.40},
]


def date_time_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    cols = [
        col for col in numeric_feature_columns(train, test) if any(token in col.lower() for token in DATE_TIME_TOKENS)
    ]
    if not cols:
        raise ValueError("No date-time feature columns found.")
    return cols


def scaled_date_time_features(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    combined = pd.concat([train[cols], test[cols]], ignore_index=True)
    combined_scaled = scaler.fit_transform(imputer.fit_transform(combined))
    return combined_scaled[: len(train)], combined_scaled[len(train) :]


def date_similarity_anchor_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    known_idx: np.ndarray,
    query_idx: np.ndarray,
    known_y: np.ndarray,
    config: dict[str, float | int | str],
    query_is_test: bool = False,
) -> np.ndarray:
    k = int(config["k"])
    date_weight = float(config["date_weight"])
    x_known = x_train[known_idx]
    x_query = x_test[query_idx] if query_is_test else x_train[query_idx]
    known_df = train.iloc[known_idx].reset_index(drop=True)
    query_df = (test if query_is_test else train).iloc[query_idx].reset_index(drop=True)
    global_mean = float(np.mean(known_y)) if len(known_y) else 0.5
    values: list[float] = []

    for query_position, row in enumerate(query_df.itertuples(index=False)):
        subject_mask = known_df["subject_id"].to_numpy() == row.subject_id
        if not subject_mask.any():
            values.append(global_mean)
            continue
        subject_features = x_known[subject_mask]
        subject_y = known_y[subject_mask]
        difference = subject_features - x_query[query_position]
        distances = np.sqrt(np.mean(difference * difference, axis=1))
        if date_weight > 0.0:
            known_dates = known_df.loc[subject_mask, "date_dt"].to_numpy()
            day_distances = np.abs((known_dates - np.datetime64(row.date_dt)) / np.timedelta64(1, "D")).astype(float)
            distances = distances + date_weight * np.log1p(day_distances)
        nearest = np.argsort(distances)[: min(k, len(distances))]
        values.append(float(np.mean(subject_y[nearest])))
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def candidate_name(config: dict[str, float | int | str]) -> str:
    return str(config["name"])


def apply_candidate_blend(
    base_values: np.ndarray,
    anchor_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    blend_weight = float(config["blend_weight"])
    return np.clip((1.0 - blend_weight) * base_values + blend_weight * anchor_values, EPS, 1.0 - EPS)


def blocked_candidate_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    base_values: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, float | int | str],
) -> tuple[np.ndarray, list[dict]]:
    selected = np.zeros(len(train), dtype=float)
    fold_details: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        anchor = date_similarity_anchor_prediction(
            train=train,
            test=test,
            x_train=x_train,
            x_test=x_test,
            known_idx=train_idx,
            query_idx=valid_idx,
            known_y=y[train_idx],
            config=config,
            query_is_test=False,
        )
        selected[valid_idx] = apply_candidate_blend(base_values[valid_idx], anchor, config)
        fold_details.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_loss": float(log_loss(y[valid_idx], selected[valid_idx], labels=[0, 1])),
            }
        )
    return selected, fold_details


def candidate_test_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    base_test_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    anchor = date_similarity_anchor_prediction(
        train=train,
        test=test,
        x_train=x_train,
        x_test=x_test,
        known_idx=np.arange(len(train)),
        query_idx=np.arange(len(test)),
        known_y=y,
        config=config,
        query_is_test=True,
    )
    return apply_candidate_blend(base_test_values, anchor, config)


def predict_s3_refined_date_similarity_anchor(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    cols = date_time_feature_columns(train_raw, test_raw)
    x_train, x_test = scaled_date_time_features(train_raw, test_raw, cols)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]

    oof = base_oof.copy()
    test_pred = base_test.copy()
    y = train_raw[TARGET].astype(int).to_numpy()
    base_values = np.clip(base_oof[TARGET].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test_values = np.clip(base_test[TARGET].astype(float).to_numpy(), EPS, 1.0 - EPS)
    candidate_oof: dict[str, np.ndarray] = {"base": base_values}
    candidate_test: dict[str, np.ndarray] = {"base": base_test_values}
    candidate_fold_details: dict[str, list[dict]] = {}

    for config in S3_REFINED_CANDIDATES:
        name = candidate_name(config)
        candidate_oof[name], candidate_fold_details[name] = blocked_candidate_oof(
            train=train,
            test=test,
            x_train=x_train,
            x_test=x_test,
            y=y,
            base_values=base_values,
            folds=folds,
            config=config,
        )
        candidate_test[name] = candidate_test_prediction(
            train=train,
            test=test,
            x_train=x_train,
            x_test=x_test,
            y=y,
            base_test_values=base_test_values,
            config=config,
        )

    candidate_losses = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in candidate_oof.items()}
    selected_oof = np.zeros(len(train), dtype=float)
    fold_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_losses = {
            name: float(log_loss(y[train_idx], values[train_idx], labels=[0, 1]))
            for name, values in candidate_oof.items()
        }
        selected_name = min(train_losses, key=train_losses.get)
        selected_oof[valid_idx] = candidate_oof[selected_name][valid_idx]
        fold_choices.append(
            {
                "fold": int(fold_idx),
                "selected": selected_name,
                "selected_train_loss": train_losses[selected_name],
                "selected_valid_loss": float(
                    log_loss(y[valid_idx], candidate_oof[selected_name][valid_idx], labels=[0, 1])
                ),
                "all_train_losses": train_losses,
            }
        )

    nested_selector_loss = float(log_loss(y, selected_oof, labels=[0, 1]))
    full_oof_selected_for_test = min(candidate_losses, key=candidate_losses.get)
    guarded_to_base = nested_selector_loss >= candidate_losses["base"]
    applied_for_test = "base" if guarded_to_base else full_oof_selected_for_test
    if guarded_to_base:
        selected_oof = base_values

    oof[TARGET] = np.clip(selected_oof, EPS, 1.0 - EPS)
    test_pred[TARGET] = candidate_test[applied_for_test]
    target_details = {
        TARGET: {
            "base_loss": candidate_losses["base"],
            "nested_selector_loss_before_guard": nested_selector_loss,
            "final_oof_loss_after_guard": float(log_loss(y, selected_oof, labels=[0, 1])),
            "full_oof_best_loss": candidate_losses[full_oof_selected_for_test],
            "full_oof_selected_for_test": full_oof_selected_for_test,
            "applied_for_test": applied_for_test,
            "guarded_to_base": bool(guarded_to_base),
            "candidate_losses": dict(sorted(candidate_losses.items(), key=lambda item: item[1])),
            "fold_choices": fold_choices,
            "candidate_fold_details": candidate_fold_details,
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(test_pred[TARGET].astype(float))),
            "test_prediction_max_abs_delta": float(
                np.max(np.abs(test_pred[TARGET].astype(float).to_numpy() - base_test_values))
            ),
        }
    }

    target_scores = {
        target: float(log_loss(train_raw[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "strict_subject_hole_nested_s3_refined_date_similarity_anchor",
        "fold_rule": (
            "Uses src.utils.validation.make_subject_hole_folds. S3 date-similarity anchors use only same-subject "
            "labels outside the validation hole. Feature scaling/imputation uses label-free train+test covariates."
        ),
        "modified_targets": [TARGET],
        "unchanged_targets": [target for target in TARGETS if target != TARGET],
        "date_time_tokens": DATE_TIME_TOKENS,
        "date_time_feature_count": len(cols),
        "s3_refined_candidates": S3_REFINED_CANDIDATES,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved except S3, where a fold-nested selector chooses among a "
            "refined date-time same-subject KNN anchor neighborhood plus base."
        ),
        "caution": (
            "Strict Subject-Hole comparable inside this repository. The candidate neighborhood was narrowed after "
            "a local feature-similarity diagnostic."
        ),
    }
    return (
        oof.drop(columns=["date_dt"], errors="ignore"),
        test_pred.drop(columns=["date_dt"], errors="ignore"),
        target_scores,
        selector_info,
        sample_original,
    )


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
            f"# Submission {experiment_id:03d}: strict S3 refined date-similarity anchor\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Strict Subject-Hole nested CV: `{local_cv:.6f}`\n\n"
            "This artifact applies a refined same-subject date-similarity anchor to S3 only. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_s3_refined_date_similarity_anchor_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The CSV, notebook, metadata, base OOF, base test predictions, and submission log row are expected "
            "to share this immutable experiment id and slug."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="s3_refined_date_similarity_anchor")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_s3_refined_date_similarity_anchor(
        args.base_id
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
        "method": "Strict S3 refined date-similarity anchor",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "strict_subject_hole_nested_s3_refined_date_similarity_anchor",
        "strict_subject_hole_comparable": True,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fold-nested selector over base plus refined same-subject date-time feature KNN anchors for S3 using "
            "make_subject_hole_folds. This is comparable with the repository's strict Subject-Hole score lens."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Strict Subject-Hole S3 refined date-similarity anchor on ID{args.base_id:03d}. Uses a fold-nested "
        "same-subject date-time KNN anchor neighborhood for S3 only; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict S3 refined date-similarity anchor on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
