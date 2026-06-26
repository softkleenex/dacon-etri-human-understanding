#!/usr/bin/env python3
"""Create a tempered blend between a refined clean anchor and ID126 calibration."""

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

from scripts.run_public_temporal_anchor_probe_experiment import load_base_artifact  # noqa: E402
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
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_CLEAN_ID = 132
DEFAULT_EXTREME_ID = 126
DEFAULT_CLEAN_WEIGHT = 0.65


def parse_target_clean_weights(value: str) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        target, raw_weight = item.split("=", maxsplit=1)
        target = target.strip()
        if target not in TARGETS:
            raise ValueError(f"Unknown target in target-clean-weights: {target}")
        weight = float(raw_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("target clean weights must be in [0, 1].")
        weights[target] = weight
    return weights


def predict_refined_anchor_extreme_blend_probe(
    clean_id: int,
    extreme_id: int,
    clean_weight: float,
    target_clean_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    clean_oof, clean_test, clean_source = load_base_artifact(train, test, clean_id)
    extreme_oof, extreme_test, extreme_source = load_base_artifact(train, test, extreme_id)
    oof = clean_oof.copy()
    test_pred = clean_test.copy()
    target_clean_weights = target_clean_weights or {}
    target_details: dict[str, dict] = {}

    for target in TARGETS:
        target_clean_weight = target_clean_weights.get(target, clean_weight)
        extreme_weight = 1.0 - target_clean_weight
        y = train[target].astype(int).to_numpy()
        clean_values = np.clip(clean_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        extreme_values = np.clip(extreme_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        clean_test_values = np.clip(clean_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        extreme_test_values = np.clip(extreme_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        blended_oof = np.clip(target_clean_weight * clean_values + extreme_weight * extreme_values, EPS, 1.0 - EPS)
        blended_test = np.clip(
            target_clean_weight * clean_test_values + extreme_weight * extreme_test_values,
            EPS,
            1.0 - EPS,
        )
        oof[target] = blended_oof
        test_pred[target] = blended_test
        target_details[target] = {
            "clean_loss": float(log_loss(y, clean_values, labels=[0, 1])),
            "extreme_loss": float(log_loss(y, extreme_values, labels=[0, 1])),
            "blended_loss": float(log_loss(y, blended_oof, labels=[0, 1])),
            "clean_weight": target_clean_weight,
            "extreme_weight": extreme_weight,
            "test_prediction_mean_clean": float(np.mean(clean_test_values)),
            "test_prediction_mean_extreme": float(np.mean(extreme_test_values)),
            "test_prediction_mean_after": float(np.mean(blended_test)),
            "test_prediction_max_abs_delta_vs_clean": float(np.max(np.abs(blended_test - clean_test_values))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "clean_source": clean_source,
        "extreme_source": extreme_source,
        "clean_id": clean_id,
        "extreme_id": extreme_id,
        "clean_weight": clean_weight,
        "extreme_weight": 1.0 - clean_weight,
        "target_clean_weights": target_clean_weights,
        "target_extreme_weights": {
            target: 1.0 - target_clean_weights.get(target, clean_weight) for target in TARGETS
        },
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "target_details": target_details,
        "selection_contract": (
            f"All seven targets are fixed blends between ID{clean_id:03d} and ID{extreme_id:03d}. "
            f"Default clean weight is {clean_weight:.2f}; target overrides are {target_clean_weights}."
        ),
        "caution": (
            "Public-structure leaderboard probe only. The extreme component is full-OOF fitted and not nested; "
            "this artifact is not strict Subject-Hole comparable despite its lower public-structure score."
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
    clean_id: int,
    extreme_id: int,
    clean_weight: float,
    target_clean_weights: dict[str, float],
) -> None:
    target_weight_arg = ""
    if target_clean_weights:
        target_weight_arg = " --target-clean-weights " + ",".join(
            f"{target}={target_clean_weights[target]:g}" for target in TARGETS if target in target_clean_weights
        )
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: refined-anchor tempered calibration probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Clean anchor id: `{clean_id:03d}`\n"
            f"- Extreme calibration id: `{extreme_id:03d}`\n"
            f"- Clean weight: `{clean_weight:.2f}`\n"
            f"- Public-structure full-OOF CV: `{local_cv:.6f}`\n\n"
            "This artifact is a non-strict public-structure probe. It should not be submitted without explicit approval."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_refined_anchor_extreme_blend_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --clean-id {clean_id} "
            f"--extreme-id {extreme_id} --clean-weight {clean_weight}{target_weight_arg}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The artifact changes all seven target columns versus the clean anchor."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="refined_anchor_tempered_extreme_public_probe")
    parser.add_argument("--clean-id", type=int, default=DEFAULT_CLEAN_ID)
    parser.add_argument("--extreme-id", type=int, default=DEFAULT_EXTREME_ID)
    parser.add_argument("--clean-weight", type=float, default=DEFAULT_CLEAN_WEIGHT)
    parser.add_argument(
        "--target-clean-weights",
        default="",
        help="Optional comma-separated target overrides, e.g. Q1=0.1,Q2=0.0.",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    if not 0.0 <= args.clean_weight <= 1.0:
        raise ValueError("clean_weight must be in [0, 1].")
    target_clean_weights = parse_target_clean_weights(args.target_clean_weights)

    oof, test_pred, target_scores, selector_info, sample_original = predict_refined_anchor_extreme_blend_probe(
        clean_id=args.clean_id,
        extreme_id=args.extreme_id,
        clean_weight=args.clean_weight,
        target_clean_weights=target_clean_weights,
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
        "method": "Refined-anchor tempered extreme-calibration public probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_full_oof_tempered_calibration_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            f"Fixed blend between ID{args.clean_id:03d} and the non-nested ID{args.extreme_id:03d} extreme "
            "calibration artifact. This is not directly comparable to strict Subject-Hole CV."
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
        args.clean_id,
        args.extreme_id,
        args.clean_weight,
        target_clean_weights,
    )

    if target_clean_weights:
        blend_summary = f"target-wise clean weights {target_clean_weights}"
    else:
        blend_summary = f"fixed blend {args.clean_weight:.2f} * ID{args.clean_id:03d} + {1.0 - args.clean_weight:.2f} * ID{args.extreme_id:03d}"
    notes = (
        f"Public-structure tempered calibration probe: {blend_summary}. Not strict Subject-Hole CV, "
        "not fold-nested; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description="Refined-anchor tempered extreme calibration blend probe",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
