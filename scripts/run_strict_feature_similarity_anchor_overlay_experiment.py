#!/usr/bin/env python3
"""Promote strict same-subject feature-similarity anchors on top of ID140."""

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
DEFAULT_BASE_ID = 140
FEATURE_FAMILY_TOKENS: dict[str, list[str]] = {
    "date_time": ["date", "day", "week", "rank", "frac", "subject_code"],
    "screen_app": ["screen", "usage", "app", "unlock", "charge"],
}
FEATURE_SIMILARITY_CANDIDATES: dict[str, list[dict[str, float | int | str]]] = {
    "S2": [
        {
            "name": "screen_app_k3_w0.1",
            "family": "screen_app",
            "k": 3,
            "date_weight": 0.0,
            "blend_weight": 0.10,
        },
        {
            "name": "screen_app_k3_w0.2",
            "family": "screen_app",
            "k": 3,
            "date_weight": 0.0,
            "blend_weight": 0.20,
        },
        {
            "name": "screen_app_k14_dw0.05_w0.3",
            "family": "screen_app",
            "k": 14,
            "date_weight": 0.05,
            "blend_weight": 0.30,
        },
    ],
    "S3": [
        {
            "name": "date_time_k5_w0.2",
            "family": "date_time",
            "k": 5,
            "date_weight": 0.0,
            "blend_weight": 0.20,
        },
        {
            "name": "date_time_k5_w0.3",
            "family": "date_time",
            "k": 5,
            "date_weight": 0.0,
            "blend_weight": 0.30,
        },
        {
            "name": "date_time_k7_w0.3",
            "family": "date_time",
            "k": 7,
            "date_weight": 0.0,
            "blend_weight": 0.30,
        },
    ],
}


def feature_families(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(train, test)
    families: dict[str, list[str]] = {}
    for family, tokens in FEATURE_FAMILY_TOKENS.items():
        cols = [col for col in numeric_cols if any(token in col.lower() for token in tokens)]
        if not cols:
            raise ValueError(f"Feature family {family!r} has no columns.")
        families[family] = cols
    return families


def scaled_feature_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    families: dict[str, list[str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family, cols in families.items():
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        combined = pd.concat([train[cols], test[cols]], ignore_index=True)
        combined_scaled = scaler.fit_transform(imputer.fit_transform(combined))
        scaled[family] = (combined_scaled[: len(train)], combined_scaled[len(train) :])
    return scaled


def feature_similarity_anchor_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scaled: dict[str, tuple[np.ndarray, np.ndarray]],
    known_idx: np.ndarray,
    query_idx: np.ndarray,
    known_y: np.ndarray,
    config: dict[str, float | int | str],
    query_is_test: bool = False,
) -> np.ndarray:
    family = str(config["family"])
    k = int(config["k"])
    date_weight = float(config["date_weight"])
    x_train, x_test = scaled[family]
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


def apply_feature_similarity_blend(
    base_values: np.ndarray,
    anchor_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    blend_weight = float(config["blend_weight"])
    return np.clip((1.0 - blend_weight) * base_values + blend_weight * anchor_values, EPS, 1.0 - EPS)


def blocked_feature_similarity_candidate_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scaled: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    base_values: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, float | int | str],
) -> tuple[np.ndarray, list[dict]]:
    selected = np.zeros(len(train), dtype=float)
    fold_details: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        anchor = feature_similarity_anchor_prediction(
            train=train,
            test=test,
            scaled=scaled,
            known_idx=train_idx,
            query_idx=valid_idx,
            known_y=y[train_idx],
            config=config,
            query_is_test=False,
        )
        selected[valid_idx] = apply_feature_similarity_blend(base_values[valid_idx], anchor, config)
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
    scaled: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    base_test_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    anchor = feature_similarity_anchor_prediction(
        train=train,
        test=test,
        scaled=scaled,
        known_idx=np.arange(len(train)),
        query_idx=np.arange(len(test)),
        known_y=y,
        config=config,
        query_is_test=True,
    )
    return apply_feature_similarity_blend(base_test_values, anchor, config)


def predict_feature_similarity_anchor_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    families = feature_families(train_raw, test_raw)
    scaled = scaled_feature_matrices(train_raw, test_raw, families)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target, configs in FEATURE_SIMILARITY_CANDIDATES.items():
        y = train_raw[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidate_oof: dict[str, np.ndarray] = {"base": base_values}
        candidate_test: dict[str, np.ndarray] = {"base": base_test_values}
        candidate_fold_details: dict[str, list[dict]] = {}

        for config in configs:
            name = candidate_name(config)
            candidate_oof[name], candidate_fold_details[name] = blocked_feature_similarity_candidate_oof(
                train=train,
                test=test,
                scaled=scaled,
                y=y,
                base_values=base_values,
                folds=folds,
                config=config,
            )
            candidate_test[name] = candidate_test_prediction(
                train=train,
                test=test,
                scaled=scaled,
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

        oof[target] = np.clip(selected_oof, EPS, 1.0 - EPS)
        test_pred[target] = candidate_test[applied_for_test]
        target_details[target] = {
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
            "test_prediction_mean_after": float(np.mean(test_pred[target].astype(float))),
            "test_prediction_max_abs_delta": float(
                np.max(np.abs(test_pred[target].astype(float).to_numpy() - base_test_values))
            ),
        }

    target_scores = {
        target: float(log_loss(train_raw[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "strict_subject_hole_nested_feature_similarity_anchor_overlay",
        "fold_rule": (
            "Uses src.utils.validation.make_subject_hole_folds. For validation holes, feature-similarity anchors "
            "use only same-subject labels outside the validation hole. Feature scaling/imputation uses label-free "
            "train+test covariates."
        ),
        "modified_targets": sorted(FEATURE_SIMILARITY_CANDIDATES),
        "unchanged_targets": [target for target in TARGETS if target not in FEATURE_SIMILARITY_CANDIDATES],
        "feature_family_tokens": FEATURE_FAMILY_TOKENS,
        "feature_family_sizes": {family: len(cols) for family, cols in families.items()},
        "feature_similarity_candidates": FEATURE_SIMILARITY_CANDIDATES,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved except S2/S3, where a fold-nested selector chooses among "
            "narrow same-subject feature-similarity KNN anchors plus base. Final test predictions use the lowest "
            "full-training OOF candidate only when the nested selector improves over base."
        ),
        "caution": (
            "Strict Subject-Hole comparable inside this repository. The feature scaling is transductive over "
            "known train/test covariates but does not use test labels."
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
            f"# Submission {experiment_id:03d}: strict feature-similarity anchor overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Strict Subject-Hole nested CV: `{local_cv:.6f}`\n\n"
            "This artifact applies guarded same-subject feature-similarity anchors to S2/S3 only. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_feature_similarity_anchor_overlay_experiment.py "
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
    parser.add_argument("--slug", default="feature_similarity_anchor_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_feature_similarity_anchor_overlay(
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
        "method": "Strict same-subject feature-similarity anchor overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "strict_subject_hole_nested_feature_similarity_anchor_overlay",
        "strict_subject_hole_comparable": True,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fold-nested selector over base plus same-subject feature-similarity KNN anchors using "
            "make_subject_hole_folds. This is comparable with the repository's strict Subject-Hole score lens."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Strict Subject-Hole feature-similarity anchor overlay on ID{args.base_id:03d}. Uses fold-nested "
        "same-subject feature-KNN anchors for S2/S3 with non-improving targets guarded to base; no DACON "
        "submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict same-subject feature-similarity anchor overlay on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
