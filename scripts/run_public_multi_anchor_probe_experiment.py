#!/usr/bin/env python3
"""Promote a public-structure multi-anchor label-propagation probe.

This script uses the observed DACON split structure: train/test share subjects
but use different dates. Validation is interleaved within each subject, not
Subject-Hole CV. Anchor parameters are fixed from a prior diagnostic and are
recorded as a leaderboard-mapping probe, not strict proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    load_base_artifact,
    make_interleaved_subject_folds,
    temporal_anchor_prediction,
)
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 167

MULTI_ANCHOR_PARAMS: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "nearest_two", "alpha_base": 0.80, "half_life": 0.25},
    "Q2": {"kind": "knn_mean", "alpha_base": 0.50, "k": 3},
    "Q3": {"kind": "nearest_two", "alpha_base": 0.65, "half_life": 84.0},
    "S1": {"kind": "exp_all", "alpha_base": 0.90, "half_life": 0.25},
    "S2": {"kind": "knn_exp", "alpha_base": 0.90, "k": 5, "half_life": 2.0},
    "S3": {"kind": "base", "alpha_base": 1.00},
    "S4": {"kind": "nearest_two", "alpha_base": 0.90, "half_life": 14.0},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_known(known_df: pd.DataFrame, known_y: np.ndarray) -> tuple[dict[str, pd.DataFrame], float]:
    known = known_df[["subject_id", "date_dt", "weekday"]].copy().reset_index(drop=True)
    known["y"] = np.asarray(known_y, dtype=float)
    by_subject = {
        str(subject_id): group.sort_values("date_dt").reset_index(drop=True)
        for subject_id, group in known.groupby("subject_id", sort=False)
    }
    global_mean = float(np.mean(known_y)) if len(known_y) else 0.5
    return by_subject, global_mean


def predict_anchor(
    kind: str,
    known_df: pd.DataFrame,
    known_y: np.ndarray,
    query_df: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    if kind == "base":
        raise ValueError("base anchor should be handled by the caller")
    if kind == "nearest_two":
        return temporal_anchor_prediction(
            known_df=known_df,
            known_y=known_y,
            query_df=query_df,
            half_life=float(config["half_life"]),
        )

    by_subject, global_mean = group_known(known_df, known_y)
    values: list[float] = []
    query = query_df[["subject_id", "date_dt", "weekday"]].copy().reset_index(drop=True)
    for row in query.itertuples(index=False):
        group = by_subject.get(str(row.subject_id))
        if group is None or len(group) == 0:
            values.append(global_mean)
            continue
        y_values = group["y"].to_numpy(dtype=float)
        days = np.abs((group["date_dt"] - row.date_dt).dt.days.to_numpy(dtype=float))
        if kind == "exp_all":
            weights = np.exp(-days / float(config["half_life"]))
            pred = (
                float(np.sum(y_values * weights) / np.sum(weights)) if weights.sum() > 1e-12 else float(y_values.mean())
            )
        elif kind == "knn_mean":
            k = min(int(config["k"]), len(group))
            pred = float(y_values[np.argsort(days)[:k]].mean())
        elif kind == "knn_exp":
            k = min(int(config["k"]), len(group))
            order = np.argsort(days)[:k]
            weights = np.exp(-days[order] / float(config["half_life"]))
            pred = (
                float(np.sum(y_values[order] * weights) / np.sum(weights))
                if weights.sum() > 1e-12
                else float(y_values[order].mean())
            )
        elif kind == "subject_mean":
            pred = float(y_values.mean())
        elif kind == "weekday_mean":
            mask = group["weekday"].to_numpy() == row.weekday
            pred = float(y_values[mask].mean()) if mask.any() else float(y_values.mean())
        else:
            raise ValueError(f"Unknown anchor kind: {kind}")
        values.append(pred)
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def anchor_oof(
    train: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    kind = str(config["kind"])
    if kind == "base":
        raise ValueError("base target should not request anchor_oof")
    oof = np.zeros(len(train), dtype=float)
    fold_details: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        pred = predict_anchor(kind, train.iloc[train_idx], y[train_idx], train.iloc[valid_idx], config)
        oof[valid_idx] = pred
        fold_details.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "anchor_mean": float(np.mean(pred)),
            }
        )
    return np.clip(oof, EPS, 1.0 - EPS), fold_details


def predict_multi_anchor_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    train["weekday"] = train["date_dt"].dt.weekday
    test["weekday"] = test["date_dt"].dt.weekday
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    fold_ids = make_interleaved_subject_folds(train)

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target, config in MULTI_ANCHOR_PARAMS.items():
        kind = str(config["kind"])
        alpha_base = float(config["alpha_base"])
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if kind == "base":
            selected_oof = base_values
            selected_test = base_test_values
            anchor_loss = None
            fold_details: list[dict[str, Any]] = []
            test_anchor_mean = None
            test_anchor_std = None
        else:
            anchor_values, fold_details = anchor_oof(train, y, fold_ids, config)
            test_anchor = predict_anchor(kind, train, y, test, config)
            selected_oof = np.clip(alpha_base * base_values + (1.0 - alpha_base) * anchor_values, EPS, 1.0 - EPS)
            selected_test = np.clip(
                alpha_base * base_test_values + (1.0 - alpha_base) * test_anchor,
                EPS,
                1.0 - EPS,
            )
            anchor_loss = float(log_loss(y, anchor_values, labels=[0, 1]))
            test_anchor_mean = float(np.mean(test_anchor))
            test_anchor_std = float(np.std(test_anchor))
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "anchor_loss": anchor_loss,
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "fold_details": fold_details,
            "test_anchor_mean": test_anchor_mean,
            "test_anchor_std": test_anchor_std,
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "fold_rule": (
            "Within each subject, sorted train dates are assigned to fold = date_rank % 5. For each validation "
            "fold, anchor labels use only non-validation labels from the same subject."
        ),
        "modified_targets": [target for target, config in MULTI_ANCHOR_PARAMS.items() if config["kind"] != "base"],
        "unchanged_targets": [target for target, config in MULTI_ANCHOR_PARAMS.items() if config["kind"] == "base"],
        "multi_anchor_params": MULTI_ANCHOR_PARAMS,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are blended with fixed same-subject label-propagation anchors. "
            "OOF anchor labels exclude the validation fold; final test anchors use all labeled training dates "
            "for the same subject. Parameters are fixed from reports/public_multi_anchor_grid_id167.json."
        ),
        "caution": (
            "This is not strict Subject-Hole CV. It is a public-structure probe for the same-subject train/test "
            "date split and inherits full-grid parameter-selection risk."
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
            f"# Submission {experiment_id:03d}: public multi-anchor probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure blocked CV: `{local_cv:.6f}`\n\n"
            "This artifact uses fixed same-subject temporal/KNN label-propagation anchors. "
            "It is not strict Subject-Hole CV."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_multi_anchor_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before explicitly accepting the public-structure validation risk."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id167_public_multi_anchor_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_multi_anchor_probe(args.base_id)
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
        "method": "Public-structure fixed multi-anchor label-propagation probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fixed multi-anchor parameters under interleaved within-subject blocked CV. "
            "This is not directly comparable to strict Subject-Hole CV."
        ),
        "multi_anchor_params": MULTI_ANCHOR_PARAMS,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Public-structure probe on ID{args.base_id:03d}: fixed target-wise same-subject temporal/KNN "
        "label-propagation anchors under interleaved within-subject blocked CV. S3 remains unchanged. "
        "Not strict Subject-Hole CV; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Public-structure multi-anchor probe on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
