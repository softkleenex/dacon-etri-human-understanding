#!/usr/bin/env python3
"""Create a tempered blend between high-C and lower-variance random-conv branches."""

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
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_PRIMARY_ID = 178
DEFAULT_ANCHOR_ID = 177
DEFAULT_PRIMARY_WEIGHT = 0.80
DEFAULT_TARGET_PRIMARY_WEIGHTS = {
    "Q1": 0.90,
    "Q2": 1.00,
    "Q3": 0.70,
    "S1": 0.55,
    "S2": 0.90,
    "S3": 1.00,
    "S4": 0.85,
}
ID178_PUBLIC_SCORE = 0.6014113009
ID178_PUBLIC_GAP = 0.06277356350964547


def parse_target_primary_weights(value: str) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        target, raw_weight = item.split("=", maxsplit=1)
        target = target.strip()
        if target not in TARGETS:
            raise ValueError(f"Unknown target in target-primary-weights: {target}")
        weight = float(raw_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("target primary weights must be in [0, 1].")
        weights[target] = weight
    return weights


def predict_tempered_random_conv_anchor_blend(
    primary_id: int,
    anchor_id: int,
    primary_weight: float,
    target_primary_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("primary_weight must be in [0, 1].")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    primary_oof, primary_test, primary_source = load_artifact_pair(train, test, primary_id)
    anchor_oof, anchor_test, anchor_source = load_artifact_pair(train, test, anchor_id)

    oof = primary_oof.copy()
    test_pred = primary_test.copy()
    target_details: dict[str, dict] = {}
    max_abs_delta_vs_primary = 0.0

    for target in TARGETS:
        weight = target_primary_weights.get(target, primary_weight)
        anchor_weight = 1.0 - weight
        y = train[target].astype(int).to_numpy()
        primary_values = np.clip(primary_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        anchor_values = np.clip(anchor_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        primary_test_values = np.clip(primary_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        anchor_test_values = np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)

        blended_oof = np.clip(weight * primary_values + anchor_weight * anchor_values, EPS, 1.0 - EPS)
        blended_test = np.clip(weight * primary_test_values + anchor_weight * anchor_test_values, EPS, 1.0 - EPS)
        oof[target] = blended_oof
        test_pred[target] = blended_test
        target_delta = float(np.max(np.abs(blended_test - primary_test_values)))
        max_abs_delta_vs_primary = max(max_abs_delta_vs_primary, target_delta)

        target_details[target] = {
            "primary_weight": weight,
            "anchor_weight": anchor_weight,
            "primary_loss": float(log_loss(y, primary_values, labels=[0, 1])),
            "anchor_loss": float(log_loss(y, anchor_values, labels=[0, 1])),
            "blended_loss": float(log_loss(y, blended_oof, labels=[0, 1])),
            "test_prediction_mean_primary": float(np.mean(primary_test_values)),
            "test_prediction_mean_anchor": float(np.mean(anchor_test_values)),
            "test_prediction_mean_blended": float(np.mean(blended_test)),
            "test_prediction_max_abs_delta_vs_primary": target_delta,
            "test_prediction_max_abs_delta_vs_anchor": float(np.max(np.abs(blended_test - anchor_test_values))),
        }

    if max_abs_delta_vs_primary <= 0.0:
        raise ValueError("Tempered blend did not change test predictions versus the primary artifact.")

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "primary_id": primary_id,
        "anchor_id": anchor_id,
        "primary_source": primary_source,
        "anchor_source": anchor_source,
        "default_primary_weight": primary_weight,
        "target_primary_weights": {target: target_primary_weights.get(target, primary_weight) for target in TARGETS},
        "target_anchor_weights": {
            target: 1.0 - target_primary_weights.get(target, primary_weight) for target in TARGETS
        },
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "target_details": target_details,
        "max_abs_delta_vs_primary_test": max_abs_delta_vs_primary,
        "selection_contract": (
            f"All seven targets are fixed convex blends between ID{primary_id:03d} high-C random-conv predictions "
            f"and ID{anchor_id:03d} lower-variance random-conv predictions. Target weights were chosen from a "
            "light OOF tempering scan after the ID178 public-gap failure."
        ),
        "public_gap_context": {
            "id178_public_score": ID178_PUBLIC_SCORE,
            "id178_public_minus_local_gap": ID178_PUBLIC_GAP,
            "interpretation": (
                "ID178 improved local public-structure CV but scored much worse than its local score on DACON. "
                "This blend slightly reduces high-C residual exposure while keeping or improving local CV."
            ),
        },
        "caution": (
            "Public-structure/non-strict random-conv tempering probe. It is selected from OOF behavior and should "
            "not be treated as strict Subject-Hole evidence or submitted without explicit approval."
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
    primary_id: int,
    anchor_id: int,
    primary_weight: float,
    target_primary_weights: dict[str, float],
) -> None:
    target_weight_arg = ""
    if target_primary_weights:
        target_weight_arg = " --target-primary-weights " + ",".join(
            f"{target}={target_primary_weights[target]:g}" for target in TARGETS if target in target_primary_weights
        )
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: tempered random-conv anchor blend\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Blend sources: ID{primary_id:03d} primary and ID{anchor_id:03d} anchor\n"
            f"- Default primary weight: `{primary_weight:.2f}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact tempers the high-C random-conv branch after the ID178 public-gap result."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_random_conv_tempered_anchor_blend_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --primary-id {primary_id} "
            f"--anchor-id {anchor_id} --primary-weight {primary_weight}{target_weight_arg}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before explicitly accepting the non-strict public-structure risk."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id178_tempered_id177_random_conv_anchor_blend")
    parser.add_argument("--primary-id", type=int, default=DEFAULT_PRIMARY_ID)
    parser.add_argument("--anchor-id", type=int, default=DEFAULT_ANCHOR_ID)
    parser.add_argument("--primary-weight", type=float, default=DEFAULT_PRIMARY_WEIGHT)
    parser.add_argument(
        "--target-primary-weights",
        default=",".join(f"{target}={DEFAULT_TARGET_PRIMARY_WEIGHTS[target]:g}" for target in TARGETS),
        help="Optional comma-separated target overrides, e.g. Q1=0.9,Q3=0.7.",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    target_primary_weights = parse_target_primary_weights(args.target_primary_weights)
    oof, test_pred, target_scores, selector_info, sample_original = predict_tempered_random_conv_anchor_blend(
        primary_id=args.primary_id,
        anchor_id=args.anchor_id,
        primary_weight=args.primary_weight,
        target_primary_weights=target_primary_weights,
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
        "method": f"Tempered random-conv anchor blend between ID{args.primary_id:03d} and ID{args.anchor_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_targetwise_tempered_random_conv_anchor_blend",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Fixed convex blend over existing public-structure OOF/test random-conv artifacts.",
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
        args.primary_id,
        args.anchor_id,
        args.primary_weight,
        target_primary_weights,
    )

    notes = (
        f"Public-structure tempered blend: target-wise convex blend of ID{args.primary_id:03d} high-C random-conv "
        f"and ID{args.anchor_id:03d} lower-variance random-conv after ID178 public gap; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Tempered random-conv anchor blend ID{args.primary_id:03d}/ID{args.anchor_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
