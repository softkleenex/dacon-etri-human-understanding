#!/usr/bin/env python3
"""Promote an extreme target-wise q40/isotonic calibration probe on ID163."""

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
DEFAULT_BASE_ID = 163
TARGET_METHODS: dict[str, dict[str, Any]] = {
    "Q1": {"method": "quantile", "q": 40},
    "Q2": {"method": "quantile", "q": 40},
    "Q3": {"method": "quantile", "q": 40},
    "S1": {"method": "quantile", "q": 40},
    "S2": {"method": "quantile", "q": 40},
    "S3": {"method": "isotonic"},
    "S4": {"method": "isotonic"},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_apply_target_method(
    train_values: np.ndarray,
    test_values: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = str(config["method"])
    train_values = np.clip(np.asarray(train_values, dtype=float), EPS, 1.0 - EPS)
    test_values = np.clip(np.asarray(test_values, dtype=float), EPS, 1.0 - EPS)
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
    if method == "base":
        return train_values, test_values, {"method": "base"}
    raise ValueError(f"Unknown target method: {method}")


def predict_targetwise_extreme_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        config = TARGET_METHODS[target]
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
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "target_methods": TARGET_METHODS,
        "target_details": target_details,
        "validation_lens": "public_structure_full_oof_targetwise_extreme_calibration_not_nested",
        "selection_contract": (
            f"ID{base_id:03d} predictions are transformed by target-wise full-OOF calibrators. Methods were selected "
            "from full-OOF losses: q40 for Q1/Q2/Q3/S1/S2 and isotonic for S3/S4."
        ),
        "caution": (
            "Extreme-risk leaderboard-response probe. This is not strict Subject-Hole CV and not fold-nested; the "
            "same OOF rows are used to choose, fit, and score calibration."
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
            f"# Submission {experiment_id:03d}: ID163 target-wise extreme calibration probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Full-OOF public-structure CV: `{local_cv:.6f}`\n\n"
            "This is an extreme-risk immutable artifact, not strict validation proof."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_id163_targetwise_extreme_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id163_targetwise_extreme_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_targetwise_extreme_probe(args.base_id)
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
        "method": f"Target-wise full-OOF extreme calibration on ID{args.base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_full_oof_targetwise_extreme_calibration_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            f"Target-wise full-OOF extreme calibration on ID{args.base_id:03d}; q40 for Q1/Q2/Q3/S1/S2 and "
            "isotonic for S3/S4. Non-strict and not fold-nested."
        ),
        "target_methods": TARGET_METHODS,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Extreme-risk target-wise public-structure calibration on ID{args.base_id:03d}: q40 for Q targets/S1/S2, "
        "isotonic for S3/S4. Not strict Subject-Hole CV, not fold-nested; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Target-wise extreme public calibration on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
