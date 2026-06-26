#!/usr/bin/env python3
"""Promote a fixed public-structure calibration probe on top of ID168."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    make_interleaved_subject_folds,
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
DEFAULT_BASE_ID = 168
CALIBRATION_PARAMS: dict[str, dict[str, Any]] = {
    "Q1": {"method": "temp_shift", "temperature": 0.85, "shift": 0.0},
    "Q2": {"method": "beta2", "C": 1.0},
    "Q3": {"method": "beta2", "C": 1.0},
    "S1": {"method": "temp_shift", "temperature": 0.75, "shift": -0.25},
    "S2": {"method": "temp_shift", "temperature": 0.85, "shift": -0.08},
    "S3": {"method": "temp_shift", "temperature": 0.95, "shift": -0.08},
    "S4": {"method": "temp_shift", "temperature": 0.85, "shift": 0.0},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def apply_temp_shift(values: np.ndarray | pd.Series, temperature: float, shift: float) -> np.ndarray:
    return np.clip(sigmoid(logit(values) / temperature + shift), EPS, 1.0 - EPS)


def beta2_features(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.column_stack([np.log(arr), np.log(1.0 - arr)])


def fit_beta2_predict(
    train_values: np.ndarray, y_train: np.ndarray, query_values: np.ndarray, c_value: float
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(query_values), float(np.mean(y_train)))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, solver="liblinear", random_state=43),
    )
    model.fit(beta2_features(train_values), y_train)
    return np.clip(model.predict_proba(beta2_features(query_values))[:, 1], EPS, 1.0 - EPS)


def calibrated_oof_and_test(
    train: pd.DataFrame,
    y: np.ndarray,
    base_values: np.ndarray,
    base_test_values: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    method = str(config["method"])
    if method == "temp_shift":
        oof = apply_temp_shift(base_values, float(config["temperature"]), float(config["shift"]))
        test_pred = apply_temp_shift(base_test_values, float(config["temperature"]), float(config["shift"]))
        return oof, test_pred, []
    if method != "beta2":
        raise ValueError(f"Unknown calibration method: {method}")

    fold_ids = make_interleaved_subject_folds(add_date_dt(train))
    oof = np.zeros(len(train), dtype=float)
    fold_details: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        pred = fit_beta2_predict(base_values[train_idx], y[train_idx], base_values[valid_idx], float(config["C"]))
        oof[valid_idx] = pred
        fold_details.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_prediction_mean": float(np.mean(pred)),
            }
        )
    test_pred = fit_beta2_predict(base_values, y, base_test_values, float(config["C"]))
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred, EPS, 1.0 - EPS), fold_details


def predict_id168_calibration_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target, config in CALIBRATION_PARAMS.items():
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        calibrated_oof, calibrated_test, fold_details = calibrated_oof_and_test(
            train=train,
            y=y,
            base_values=base_values,
            base_test_values=base_test_values,
            config=config,
        )
        oof[target] = calibrated_oof
        test_pred[target] = calibrated_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "calibrated_loss": float(log_loss(y, calibrated_oof, labels=[0, 1])),
            "fold_details": fold_details,
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(calibrated_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(calibrated_test - base_test_values))),
        }
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "calibration_params": CALIBRATION_PARAMS,
        "target_details": target_details,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv_plus_fixed_calibration",
        "selection_contract": (
            f"ID{base_id:03d} predictions are transformed by fixed calibration choices from "
            "reports/id168_calibration_diagnostic.json. Beta2 targets are fold-fitted for OOF rows and full-fitted "
            "for test rows; temp/shift targets use fixed deterministic transforms."
        ),
        "caution": (
            "Not strict Subject-Hole CV. Calibration choices were selected from a diagnostic on ID168, so this is a "
            "public-structure leaderboard-mapping probe, not independent proof."
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
            f"# Submission {experiment_id:03d}: ID168 fixed calibration public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure calibrated CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for the immutable artifact."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_id168_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id168_fixed_public_calibration_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_id168_calibration_probe(args.base_id)
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
        "method": "Fixed public-structure calibration on ID168 multi-anchor probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_plus_fixed_calibration",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fixed public-structure calibration choices on ID168. Beta2 calibration is fold-fitted for OOF rows; "
            "temp/shift transforms are fixed. This is not strict Subject-Hole CV."
        ),
        "calibration_params": CALIBRATION_PARAMS,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Fixed public-structure calibration on ID{args.base_id:03d}: beta2 for Q2/Q3 and fixed temp/shift for "
        "other targets. Not strict Subject-Hole CV; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Fixed public calibration on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
