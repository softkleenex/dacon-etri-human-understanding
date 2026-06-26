#!/usr/bin/env python3
"""Promote a configurable second-stage full-OOF calibration probe."""

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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_extreme_calibration_probe_experiment import (  # noqa: E402
    apply_quantile_calibrator,
    fit_quantile_calibrator,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
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
DEFAULT_BASE_ID = 226
DEFAULT_TARGET_METHODS = "Q1=isotonic,Q2=isotonic,Q3=isotonic,S1=isotonic,S2=isotonic,S3=base,S4=q160"


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def apply_temp_shift(values: np.ndarray | pd.Series, temperature: float, shift: float) -> np.ndarray:
    return np.clip(sigmoid(logit(values) / temperature + shift), EPS, 1.0 - EPS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def parse_target_methods(value: str) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        target, raw_method = item.split("=", maxsplit=1)
        target = target.strip()
        raw_method = raw_method.strip().lower()
        if target not in TARGETS:
            raise ValueError(f"Unknown target in target methods: {target}")
        if raw_method == "base":
            methods[target] = {"method": "base"}
        elif raw_method == "isotonic":
            methods[target] = {"method": "isotonic"}
        elif raw_method.startswith("q"):
            q = int(raw_method[1:])
            if q < 2:
                raise ValueError("Quantile q must be >= 2.")
            methods[target] = {"method": "quantile", "q": q}
        elif raw_method.startswith("temp_t"):
            # Format: temp_t0.85_s-0.08
            temp_raw, shift_raw = raw_method.removeprefix("temp_t").split("_s", maxsplit=1)
            temperature = float(temp_raw)
            shift = float(shift_raw)
            if temperature <= 0.0:
                raise ValueError("Temperature must be positive.")
            methods[target] = {"method": "temp_shift", "temperature": temperature, "shift": shift}
        else:
            raise ValueError(f"Unknown method for {target}: {raw_method}")
    missing = sorted(set(TARGETS) - set(methods))
    if missing:
        raise ValueError(f"Missing target methods for: {missing}")
    return methods


def fit_apply_target_method(
    train_values: np.ndarray,
    test_values: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = str(config["method"])
    train_values = np.clip(np.asarray(train_values, dtype=float), EPS, 1.0 - EPS)
    test_values = np.clip(np.asarray(test_values, dtype=float), EPS, 1.0 - EPS)
    if method == "base":
        return train_values, test_values, {"method": "base"}
    if method == "temp_shift":
        temperature = float(config["temperature"])
        shift = float(config["shift"])
        return (
            apply_temp_shift(train_values, temperature, shift),
            apply_temp_shift(test_values, temperature, shift),
            {"method": "temp_shift", "temperature": temperature, "shift": shift},
        )
    if method == "quantile":
        q = int(config["q"])
        calibrator = fit_quantile_calibrator(train_values, y, q=q)
        return (
            apply_quantile_calibrator(train_values, calibrator),
            apply_quantile_calibrator(test_values, calibrator),
            {"method": "quantile", "q": q, **calibrator},
        )
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
        model.fit(train_values, y)
        return (
            np.clip(model.predict(train_values), EPS, 1.0 - EPS),
            np.clip(model.predict(test_values), EPS, 1.0 - EPS),
            {
                "method": "isotonic",
                "n_thresholds": int(len(model.X_thresholds_)),
                "x_thresholds": [float(v) for v in model.X_thresholds_],
                "y_thresholds": [float(v) for v in model.y_thresholds_],
            },
        )
    raise ValueError(f"Unknown method: {method}")


def predict_second_stage_calibration_probe(
    base_id: int,
    target_methods: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        config = target_methods[target]
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        calibrated_oof, calibrated_test, fitted = fit_apply_target_method(base_values, base_test_values, y, config)
        oof[target] = calibrated_oof
        test_pred[target] = calibrated_test
        target_details[target] = {
            "config": config,
            "fitted_calibrator": fitted,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "calibrated_loss": float(log_loss(y, calibrated_oof, labels=[0, 1])),
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(calibrated_test)),
            "test_prediction_std_after": float(np.std(calibrated_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(calibrated_test - base_test_values))),
        }
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "modified_targets": [target for target, config in target_methods.items() if config["method"] != "base"],
        "unchanged_targets": [target for target, config in target_methods.items() if config["method"] == "base"],
        "target_methods": target_methods,
        "target_details": target_details,
        "validation_lens": "public_structure_second_stage_full_oof_calibration_not_nested",
        "selection_contract": (
            f"ID{base_id:03d} predictions are transformed by target-wise second-stage full-OOF calibrators. "
            "The same OOF rows are used to select, fit, and score calibration."
        ),
        "caution": (
            "Extreme-risk second-stage leaderboard-response probe. This is not strict Subject-Hole CV, "
            "not fold-nested, and likely to have severe public-gap risk."
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
    target_methods_raw: str,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: second-stage calibration probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Full-OOF public-structure CV: `{local_cv:.6f}`\n\n"
            "This is an extreme-risk immutable artifact, not strict validation proof."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_second_stage_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--target-methods {target_methods_raw}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id226_second_stage_calibration_frontier")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--target-methods", default=DEFAULT_TARGET_METHODS)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    target_methods = parse_target_methods(args.target_methods)

    oof, test_pred, target_scores, selector_info, sample_original = predict_second_stage_calibration_probe(
        args.base_id,
        target_methods,
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
            raise FileExistsError(f"Refusing to overwrite existing artifact: {relpath(path)}")

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
        "method": f"Second-stage full-OOF calibration on ID{args.base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_second_stage_full_oof_calibration_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": (
            f"Second-stage full-OOF calibration on ID{args.base_id:03d}; target methods {target_methods}. "
            "Extreme-risk, non-strict, and not fold-nested."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path,
        submission_path,
        metadata_path,
        args.experiment_id,
        args.slug,
        local_cv,
        args.base_id,
        args.target_methods,
    )

    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Extreme-risk second-stage full-OOF calibration on ID{args.base_id:03d}; "
            f"target methods {target_methods}. Not strict Subject-Hole CV, not fold-nested; "
            "no DACON submission was made."
        ),
        model_description=f"Second-stage full-OOF calibration on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
