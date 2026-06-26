#!/usr/bin/env python3
"""Create a tempered blend between ID125 and the extreme ID126 calibration probe."""

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
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 125
DEFAULT_EXTREME_ID = 126
DEFAULT_ALPHA = 0.20
ID126_NESTED_REFERENCE_MEAN = 0.5623772397331482


def load_artifact_pair(train: pd.DataFrame, test: pd.DataFrame, artifact_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{artifact_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{artifact_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{artifact_id:03d} base OOF/test artifact.")
    return (
        align_candidate(pd.read_csv(oof_paths[0]), train),
        align_candidate(pd.read_csv(test_paths[0]), test),
        f"{oof_paths[0].name}; {test_paths[0].name}",
    )


def blend_predictions(base_values: pd.Series | np.ndarray, extreme_values: pd.Series | np.ndarray, alpha: float) -> np.ndarray:
    base_arr = np.clip(np.asarray(base_values, dtype=float), EPS, 1.0 - EPS)
    extreme_arr = np.clip(np.asarray(extreme_values, dtype=float), EPS, 1.0 - EPS)
    return np.clip((1.0 - alpha) * base_arr + alpha * extreme_arr, EPS, 1.0 - EPS)


def predict_tempered_extreme_probe(
    base_id: int,
    extreme_id: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1 for a tempered blend artifact.")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    extreme_oof, extreme_test, extreme_source = load_artifact_pair(train, test, extreme_id)

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        extreme_values = np.clip(extreme_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        extreme_test_values = np.clip(extreme_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)

        blended_oof = blend_predictions(base_values, extreme_values, alpha=alpha)
        blended_test = blend_predictions(base_test_values, extreme_test_values, alpha=alpha)
        oof[target] = blended_oof
        test_pred[target] = blended_test

        target_details[target] = {
            "alpha_extreme": alpha,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "extreme_loss": float(log_loss(y, extreme_values, labels=[0, 1])),
            "tempered_loss": float(log_loss(y, blended_oof, labels=[0, 1])),
            "test_prediction_mean_base": float(np.mean(base_test_values)),
            "test_prediction_mean_extreme": float(np.mean(extreme_test_values)),
            "test_prediction_mean_tempered": float(np.mean(blended_test)),
            "test_prediction_max_abs_delta_vs_base": float(np.max(np.abs(blended_test - base_test_values))),
            "test_prediction_max_abs_delta_vs_extreme": float(np.max(np.abs(blended_test - extreme_test_values))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "extreme_source": extreme_source,
        "base_id": base_id,
        "extreme_id": extreme_id,
        "alpha_extreme": alpha,
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "target_details": target_details,
        "id126_nested_reference_mean": ID126_NESTED_REFERENCE_MEAN,
        "selection_contract": (
            f"All seven targets are a fixed convex blend: {1.0 - alpha:.2f} * ID{base_id:03d} + "
            f"{alpha:.2f} * ID{extreme_id:03d}. Alpha was chosen from a light OOF risk-control diagnostic, "
            "not fold-nested selection."
        ),
        "caution": (
            "Tempered leaderboard calibration probe: less extreme than ID126 but still inherits full-OOF, "
            "non-nested calibration risk. Treat as a DACON probe candidate only after explicit approval."
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
    extreme_id: int,
    alpha: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: tempered extreme calibration public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Blend: `{1.0 - alpha:.2f} * ID{base_id:03d} + {alpha:.2f} * ID{extreme_id:03d}`\n"
            f"- Full-OOF public-structure probe CV: `{local_cv:.6f}`\n\n"
            "This artifact tempers the ID126 isotonic/quantile calibration. "
            "It is non-strict and not fold-nested."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_tempered_extreme_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--extreme-id {extreme_id} --alpha {alpha}"
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
    parser.add_argument("--slug", default="tempered_extreme_calibration_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--extreme-id", type=int, default=DEFAULT_EXTREME_ID)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_tempered_extreme_probe(
        base_id=args.base_id,
        extreme_id=args.extreme_id,
        alpha=args.alpha,
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
        "method": (
            f"Tempered blend between ID{args.base_id:03d} and extreme ID{args.extreme_id:03d} "
            "calibration public probe"
        ),
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_tempered_extreme_calibration_blend_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            f"Fixed blend between ID{args.base_id:03d} and ID{args.extreme_id:03d}. This is non-strict, "
            "not fold-nested, and should only be used as a risk-controlled leaderboard calibration probe."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        base_id=args.base_id,
        extreme_id=args.extreme_id,
        alpha=args.alpha,
    )

    notes = (
        f"Tempered public-structure calibration probe: fixed blend {1.0 - args.alpha:.2f} * ID{args.base_id:03d} "
        f"+ {args.alpha:.2f} * ID{args.extreme_id:03d}. Not strict Subject-Hole CV, not fold-nested; "
        f"ID126 nested reference is {ID126_NESTED_REFERENCE_MEAN:.6f}. No DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description="Tempered public-structure extreme calibration blend probe",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
