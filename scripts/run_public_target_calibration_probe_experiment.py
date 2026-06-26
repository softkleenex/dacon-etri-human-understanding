#!/usr/bin/env python3
"""Create a high-risk target-wise temperature/shift calibration public probe."""

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
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 124
CALIBRATION_PARAMS: dict[str, dict[str, float]] = {
    "Q1": {"temperature": 0.86, "shift": 0.00},
    "Q2": {"temperature": 0.82, "shift": -0.06},
    "Q3": {"temperature": 0.94, "shift": -0.02},
    "S1": {"temperature": 0.76, "shift": -0.30},
    "S2": {"temperature": 0.94, "shift": -0.06},
    "S3": {"temperature": 0.98, "shift": -0.02},
    "S4": {"temperature": 0.80, "shift": -0.06},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base_artifact(train: pd.DataFrame, test: pd.DataFrame, base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF/test artifact.")
    return (
        align_candidate(pd.read_csv(oof_paths[0]), train),
        align_candidate(pd.read_csv(test_paths[0]), test),
        f"{oof_paths[0].name}; {test_paths[0].name}",
    )


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def calibrate(values: pd.Series | np.ndarray, temperature: float, shift: float) -> np.ndarray:
    return np.clip(sigmoid(logit(values) / temperature + shift), EPS, 1.0 - EPS)


def predict_target_calibration_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target in TARGETS:
        params = CALIBRATION_PARAMS[target]
        temperature = float(params["temperature"])
        shift = float(params["shift"])
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        calibrated_oof = calibrate(base_values, temperature=temperature, shift=shift)
        calibrated_test = calibrate(base_test_values, temperature=temperature, shift=shift)
        oof[target] = calibrated_oof
        test_pred[target] = calibrated_test
        target_details[target] = {
            "temperature": temperature,
            "shift": shift,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "calibrated_loss": float(log_loss(y, calibrated_oof, labels=[0, 1])),
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
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "calibration_params": CALIBRATION_PARAMS,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are replaced for all seven targets by fixed target-wise "
            "temperature/shift transforms. Parameters were selected by a full-OOF public-structure calibration "
            "scan, not fold-nested selection."
        ),
        "caution": (
            "High-risk leaderboard calibration probe: not strict Subject-Hole CV and not nested calibration. "
            "Use only for explicit leaderboard calibration, not as a verified local improvement."
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
            f"# Submission {experiment_id:03d}: target-wise calibration public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Full-OOF calibration probe CV: `{local_cv:.6f}`\n\n"
            "This artifact applies fixed target-wise temperature/shift calibration to all targets. "
            "It is high-risk, non-strict, and not fold-nested."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_target_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before explicitly accepting the non-nested calibration risk."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="target_temperature_shift_calibration_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_target_calibration_probe(args.base_id)
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
        "method": "High-risk target-wise temperature/shift calibration public probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_full_oof_target_calibration_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Full-OOF selected target-wise temperature/shift calibration. This is non-strict, not fold-nested, "
            "and should only be used as a leaderboard calibration probe."
        ),
        "calibration_params": CALIBRATION_PARAMS,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"High-risk public-structure calibration probe on ID{args.base_id:03d}: all seven targets use full-OOF "
        "selected temperature/shift transforms. Not strict Subject-Hole CV, not fold-nested, and no DACON "
        "submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"High-risk public-structure target calibration probe on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
