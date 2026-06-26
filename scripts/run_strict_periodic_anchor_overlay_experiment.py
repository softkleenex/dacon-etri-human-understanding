#!/usr/bin/env python3
"""Promote strict periodic same-subject anchors on top of ID139."""

from __future__ import annotations

import argparse
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
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 139
PERIODIC_CANDIDATES: dict[str, list[dict[str, float | int | str]]] = {
    "S1": [
        {"name": "weekly_lag_w0.05", "mode": "weekly_lag", "blend_weight": 0.05},
        {"name": "weekly_lag_w0.1", "mode": "weekly_lag", "blend_weight": 0.10},
        {"name": "weekly_lag_w0.2", "mode": "weekly_lag", "blend_weight": 0.20},
        {"name": "weekly_nearest_w0.05", "mode": "weekly_nearest", "blend_weight": 0.05},
        {"name": "weekly_nearest_w0.1", "mode": "weekly_nearest", "blend_weight": 0.10},
        {"name": "weekly_knn3_w0.1", "mode": "weekly_knn", "k": 3, "blend_weight": 0.10},
        {"name": "weekly_knn3_w0.2", "mode": "weekly_knn", "k": 3, "blend_weight": 0.20},
    ],
    "S2": [
        {"name": "subject_weekend_w0.1", "mode": "subject_weekend", "blend_weight": 0.10},
        {"name": "subject_weekend_w0.2", "mode": "subject_weekend", "blend_weight": 0.20},
        {"name": "subject_weekend_w0.3", "mode": "subject_weekend", "blend_weight": 0.30},
        {"name": "subject_weekend_w0.4", "mode": "subject_weekend", "blend_weight": 0.40},
        {"name": "dow_kernel21_w0.2", "mode": "dow_kernel", "half_life": 21.0, "blend_weight": 0.20},
        {"name": "dow_kernel84_w0.1", "mode": "dow_kernel", "half_life": 84.0, "blend_weight": 0.10},
        {"name": "dow_kernel84_w0.2", "mode": "dow_kernel", "half_life": 84.0, "blend_weight": 0.20},
    ],
}


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date_dt"].dt.dayofweek.astype(int)
    out["weekend"] = (out["dow"] >= 5).astype(int)
    return out


def periodic_anchor_prediction(
    known_df: pd.DataFrame,
    known_y: np.ndarray,
    query_df: pd.DataFrame,
    config: dict[str, float | int | str],
) -> np.ndarray:
    mode = str(config["mode"])
    known = known_df[["subject_id", "date_dt", "dow", "weekend"]].copy().reset_index(drop=True)
    known["y"] = np.asarray(known_y, dtype=float)
    query = query_df[["subject_id", "date_dt", "dow", "weekend"]].copy().reset_index(drop=True)
    by_subject = {
        subject_id: group.sort_values("date_dt").reset_index(drop=True)
        for subject_id, group in known.groupby("subject_id", sort=False)
    }
    global_mean = float(np.mean(known_y)) if len(known_y) else 0.5
    subject_mean = known.groupby("subject_id")["y"].mean().to_dict()
    weekend_mean = known.groupby("weekend")["y"].mean().to_dict()
    values: list[float] = []

    for row in query.itertuples(index=False):
        group = by_subject.get(row.subject_id)
        fallback = subject_mean.get(row.subject_id, weekend_mean.get(row.weekend, global_mean))
        if group is None or len(group) == 0:
            values.append(fallback)
            continue

        deltas = (group["date_dt"] - row.date_dt).dt.days.to_numpy(dtype=float)
        distances = np.abs(deltas)
        labels = group["y"].to_numpy(dtype=float)

        if mode == "subject_weekend":
            mask = group["weekend"].to_numpy(dtype=int) == int(row.weekend)
            values.append(float(np.mean(labels[mask])) if mask.any() else fallback)
            continue

        if mode == "weekly_nearest":
            mask = group["dow"].to_numpy(dtype=int) == int(row.dow)
            if mask.any():
                indices = np.where(mask)[0]
                values.append(float(labels[indices[np.argmin(distances[indices])]]))
            else:
                values.append(fallback)
            continue

        if mode == "weekly_knn":
            mask = group["dow"].to_numpy(dtype=int) == int(row.dow)
            k = int(config["k"])
            if mask.any():
                indices = np.where(mask)[0]
                chosen = indices[np.argsort(distances[indices])[: min(k, len(indices))]]
                values.append(float(np.mean(labels[chosen])))
            else:
                values.append(fallback)
            continue

        if mode == "weekly_lag":
            indices: list[int] = []
            for lag in [7.0, 14.0, 21.0]:
                indices.extend(np.where(distances == lag)[0].tolist())
            values.append(float(np.mean(labels[indices])) if indices else fallback)
            continue

        if mode == "dow_kernel":
            half_life = float(config["half_life"])
            same_dow = (group["dow"].to_numpy(dtype=int) == int(row.dow)).astype(float)
            weights = np.exp(-distances / max(half_life, EPS)) * (1.0 + 2.0 * same_dow)
            values.append(float(np.sum(weights * labels) / np.sum(weights)) if np.sum(weights) > 0 else fallback)
            continue

        raise ValueError(f"Unknown periodic anchor mode: {mode}")

    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def candidate_name(config: dict[str, float | int | str]) -> str:
    return str(config["name"])


def apply_periodic_blend(
    base_values: np.ndarray,
    anchor_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    blend_weight = float(config["blend_weight"])
    return np.clip((1.0 - blend_weight) * base_values + blend_weight * anchor_values, EPS, 1.0 - EPS)


def blocked_periodic_candidate_oof(
    train: pd.DataFrame,
    y: np.ndarray,
    base_values: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, float | int | str],
) -> tuple[np.ndarray, list[dict]]:
    selected = np.zeros(len(train), dtype=float)
    fold_details: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        anchor = periodic_anchor_prediction(
            known_df=train.iloc[train_idx],
            known_y=y[train_idx],
            query_df=train.iloc[valid_idx],
            config=config,
        )
        selected[valid_idx] = apply_periodic_blend(base_values[valid_idx], anchor, config)
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
    y: np.ndarray,
    base_test_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    anchor = periodic_anchor_prediction(known_df=train, known_y=y, query_df=test, config=config)
    return apply_periodic_blend(base_test_values, anchor, config)


def predict_periodic_anchor_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_calendar_columns(add_date_dt(train_raw))
    test = add_calendar_columns(add_date_dt(test_raw))
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target, configs in PERIODIC_CANDIDATES.items():
        y = train_raw[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidate_oof: dict[str, np.ndarray] = {"base": base_values}
        candidate_test: dict[str, np.ndarray] = {"base": base_test_values}
        candidate_fold_details: dict[str, list[dict]] = {}

        for config in configs:
            name = candidate_name(config)
            candidate_oof[name], candidate_fold_details[name] = blocked_periodic_candidate_oof(
                train=train,
                y=y,
                base_values=base_values,
                folds=folds,
                config=config,
            )
            candidate_test[name] = candidate_test_prediction(
                train=train,
                test=test,
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
        "validation_lens": "strict_subject_hole_nested_periodic_anchor_overlay",
        "fold_rule": (
            "Uses src.utils.validation.make_subject_hole_folds. For validation holes, periodic anchors use only "
            "same-subject labels outside the validation hole. Fold-level candidate selection is based on fold "
            "training rows only; targets without nested improvement are guarded to base."
        ),
        "modified_targets": sorted(PERIODIC_CANDIDATES),
        "unchanged_targets": [target for target in TARGETS if target not in PERIODIC_CANDIDATES],
        "periodic_candidates": PERIODIC_CANDIDATES,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved except S1/S2, where a fold-nested selector chooses among "
            "narrow weekly/weekday same-subject periodic anchors plus base. Final test predictions use the "
            "lowest full-training OOF candidate only when the nested selector improves over base."
        ),
        "caution": (
            "Strict Subject-Hole comparable inside this repository. The periodic anchors intentionally use known "
            "same-subject labels outside the validation hole to mimic the train/test date-hole structure."
        ),
    }
    return (
        oof.drop(columns=["date_dt", "dow", "weekend"], errors="ignore"),
        test_pred.drop(columns=["date_dt", "dow", "weekend"], errors="ignore"),
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
            f"# Submission {experiment_id:03d}: strict periodic anchor overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Strict Subject-Hole nested CV: `{local_cv:.6f}`\n\n"
            "This artifact applies guarded weekly/weekday same-subject periodic anchors to S1/S2 only. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_periodic_anchor_overlay_experiment.py "
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
    parser.add_argument("--slug", default="periodic_anchor_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_periodic_anchor_overlay(args.base_id)
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
        "method": "Strict periodic same-subject anchor overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "strict_subject_hole_nested_periodic_anchor_overlay",
        "strict_subject_hole_comparable": True,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fold-nested selector over base plus weekly/weekday same-subject periodic anchors using "
            "make_subject_hole_folds. This is comparable with the repository's strict Subject-Hole score lens."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Strict Subject-Hole periodic anchor overlay on ID{args.base_id:03d}. Uses fold-nested weekly/weekday "
        "same-subject anchors for S1/S2 with non-improving targets guarded to base; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict periodic same-subject anchor overlay on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
