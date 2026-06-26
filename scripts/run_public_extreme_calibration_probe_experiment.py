#!/usr/bin/env python3
"""Create an extreme-risk full-OOF isotonic/quantile calibration public probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
DEFAULT_BASE_ID = 125
EXTREME_CALIBRATION_METHODS: dict[str, dict[str, float | str]] = {
    "Q1": {"method": "quantile", "q": 20.0},
    "Q2": {"method": "isotonic"},
    "Q3": {"method": "isotonic"},
    "S1": {"method": "isotonic"},
    "S2": {"method": "quantile", "q": 20.0},
    "S3": {"method": "isotonic"},
    "S4": {"method": "quantile", "q": 20.0},
}
NESTED_REFERENCE_SCORES = {
    "Q1": {"method": "beta_C1.0", "score": 0.5785782909498893},
    "Q2": {"method": "beta_C1.0", "score": 0.560379264459373},
    "Q3": {"method": "beta_C1.0", "score": 0.5957298554671148},
    "S1": {"method": "beta_C1.0", "score": 0.49792923937964156},
    "S2": {"method": "beta_C1.0", "score": 0.5577256495008319},
    "S3": {"method": "beta_C1.0", "score": 0.5330408768414289},
    "S4": {"method": "isotonic_p", "score": 0.6132575015337585},
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


def fit_quantile_calibrator(values: np.ndarray, y: np.ndarray, q: int) -> dict:
    values = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, q + 1)))
    if len(edges) <= 2:
        return {"edges": edges.tolist(), "bin_values": [float(np.mean(y))], "global_mean": float(np.mean(y))}
    bins = np.digitize(values, edges[1:-1], right=True)
    bin_values = []
    for bin_idx in range(len(edges) - 1):
        mask = bins == bin_idx
        if mask.any():
            bin_values.append(float((y[mask].sum() + 0.5) / (mask.sum() + 1.0)))
        else:
            bin_values.append(float(np.mean(y)))
    return {"edges": edges.tolist(), "bin_values": bin_values, "global_mean": float(np.mean(y))}


def apply_quantile_calibrator(values: np.ndarray, calibrator: dict) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    edges = np.asarray(calibrator["edges"], dtype=float)
    bin_values = np.asarray(calibrator["bin_values"], dtype=float)
    if len(edges) <= 2 or len(bin_values) == 1:
        return np.full_like(values, float(calibrator["global_mean"]), dtype=float)
    bins = np.digitize(values, edges[1:-1], right=True)
    bins = np.clip(bins, 0, len(bin_values) - 1)
    return np.clip(bin_values[bins], EPS, 1.0 - EPS)


def fit_apply_calibration(
    train_values: np.ndarray,
    test_values: np.ndarray,
    y: np.ndarray,
    method_config: dict[str, float | str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    method = str(method_config["method"])
    train_values = np.clip(np.asarray(train_values, dtype=float), EPS, 1.0 - EPS)
    test_values = np.clip(np.asarray(test_values, dtype=float), EPS, 1.0 - EPS)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
        model.fit(train_values, y)
        train_pred = np.clip(model.predict(train_values), EPS, 1.0 - EPS)
        test_pred = np.clip(model.predict(test_values), EPS, 1.0 - EPS)
        return train_pred, test_pred, {"method": "isotonic", "n_thresholds": int(len(model.X_thresholds_))}
    if method == "quantile":
        q = int(method_config["q"])
        calibrator = fit_quantile_calibrator(train_values, y, q=q)
        return (
            apply_quantile_calibrator(train_values, calibrator),
            apply_quantile_calibrator(test_values, calibrator),
            {"method": "quantile", "q": q, **calibrator},
        )
    raise ValueError(f"Unknown extreme calibration method: {method}")


def predict_extreme_calibration_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        calibrated_oof, calibrated_test, fitted = fit_apply_calibration(
            train_values=base_values,
            test_values=base_test_values,
            y=y,
            method_config=EXTREME_CALIBRATION_METHODS[target],
        )
        oof[target] = calibrated_oof
        test_pred[target] = calibrated_test
        target_details[target] = {
            "method_config": EXTREME_CALIBRATION_METHODS[target],
            "fitted_calibrator": fitted,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "calibrated_loss": float(log_loss(y, calibrated_oof, labels=[0, 1])),
            "nested_reference": NESTED_REFERENCE_SCORES[target],
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(calibrated_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(calibrated_test - base_test_values))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    nested_reference_mean = float(np.mean([item["score"] for item in NESTED_REFERENCE_SCORES.values()]))
    selector_info = {
        "base_source": base_source,
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "extreme_calibration_methods": EXTREME_CALIBRATION_METHODS,
        "target_details": target_details,
        "nested_reference_scores": NESTED_REFERENCE_SCORES,
        "nested_reference_mean": nested_reference_mean,
        "selection_contract": (
            f"ID{base_id:03d} predictions are replaced for all seven targets by full-OOF fitted isotonic or "
            "quantile bin calibrators. This optimizes on the same OOF vector used for reporting."
        ),
        "caution": (
            "Extreme-risk leaderboard calibration probe: not strict Subject-Hole CV and not nested. A nested "
            f"calibration diagnostic averaged {nested_reference_mean:.6f}, worse than the base high-risk ID125, "
            "so this artifact should not be interpreted as a reliable local improvement despite the low score."
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
            f"# Submission {experiment_id:03d}: extreme calibration public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Full-OOF calibration probe CV: `{local_cv:.6f}`\n\n"
            "This artifact applies full-OOF fitted isotonic/quantile calibration. "
            "It is extreme-risk, non-strict, and not fold-nested."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_extreme_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before explicitly accepting the severe non-nested calibration risk."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="extreme_isotonic_quantile_calibration_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_extreme_calibration_probe(args.base_id)
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
        "method": "Extreme-risk full-OOF isotonic/quantile calibration public probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_full_oof_isotonic_quantile_calibration_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Full-OOF fitted isotonic/quantile calibration. This is extreme-risk, non-strict, not fold-nested, "
            "and should only be used as a leaderboard calibration probe."
        ),
        "extreme_calibration_methods": EXTREME_CALIBRATION_METHODS,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Extreme-risk public-structure calibration probe on ID{args.base_id:03d}: all seven targets use full-OOF "
        "fitted isotonic/quantile calibrators. Not strict Subject-Hole CV, not fold-nested; nested calibration "
        f"reference is {selector_info['nested_reference_mean']:.6f}. No DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Extreme-risk public-structure calibration probe on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
