#!/usr/bin/env python3
"""Diagnose/promote a narrow Q1-only probability temperature overlay."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_booster_compact_experiment import load_base_artifact  # noqa: E402
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

warnings.filterwarnings("ignore")

TARGET = "Q1"
DEFAULT_BASE_ID = 117
TEMPERATURES = [0.85, 0.90, 0.95]
EPS = 1e-6


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return np.clip(sigmoid(logit(probabilities) / temperature), EPS, 1.0 - EPS)


def predict_q1_temperature_overlay(base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    y = train[TARGET].astype(int).to_numpy()
    base_q1_oof = base_oof[TARGET].astype(float).to_numpy()
    base_q1_test = base_test[TARGET].astype(float).to_numpy()
    base_candidate_name = f"id{base_id:03d}_base"

    candidate_oof = {base_candidate_name: np.clip(base_q1_oof, EPS, 1.0 - EPS)}
    candidate_test = {base_candidate_name: np.clip(base_q1_test, EPS, 1.0 - EPS)}
    candidate_details = {base_candidate_name: {"source": "base"}}
    for temperature in TEMPERATURES:
        name = f"temp_{temperature:.2f}"
        candidate_oof[name] = apply_temperature(base_q1_oof, temperature)
        candidate_test[name] = apply_temperature(base_q1_test, temperature)
        candidate_details[name] = {
            "source": base_candidate_name,
            "temperature": temperature,
            "transform": "sigmoid(logit(p) / temperature)",
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
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
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

    full_best_name, full_best_loss = min(candidate_scores.items(), key=lambda item: item[1])
    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[TARGET] = np.clip(selected_q1_oof, EPS, 1.0 - EPS)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], EPS, 1.0 - EPS)
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "q1_probability_temperature",
        "candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": candidate_scores,
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q1. "
            "Q1 validation rows use a temperature candidate selected only from each outer fold's training rows; "
            "final test Q1 uses the candidate with the best full-training OOF log loss."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q1_temperature_confirm_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q1_temperature_confirm_diagnostic_id{base_id:03d}_{suffix}.json"
    report = {
        "base_id": base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "selector_info": selector_info,
        "confirm_note": "Narrow Q1-only temperature scaling confirmation over the current strict base artifact.",
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
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
            f"# Submission {experiment_id:03d}: Q1 temperature confirm overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base ID: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This artifact changes only Q1 by applying a narrow temperature transform to the base probabilities. "
            "OOF candidate choice is fold-nested; no DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/run_strict_q1_temperature_confirm_experiment.py "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Review metadata and audit reports before any DACON submission."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


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
    base_q1_score = float(selector_info["candidate_scores"][f"id{base_id:03d}_base"])
    if target_scores[TARGET] >= base_q1_score:
        raise ValueError(f"Refusing to promote non-improving Q1 overlay: {target_scores[TARGET]} >= {base_q1_score}")

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
        "method": "Q1 temperature confirm overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with Q1-only fold-nested temperature candidate selection.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id)
    notes = (
        f"Strict Q1-only probability temperature overlay on ID{base_id:03d}. "
        "Uses fold-nested selection among base and fixed temperature transforms; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict Q1 temperature confirm overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q1_temperature_confirm_overlay")
    args = parser.parse_args()

    oof, test_pred, target_scores, selector_info, sample_original = predict_q1_temperature_overlay(args.base_id)
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "q1_candidate_scores": selector_info["candidate_scores"],
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
