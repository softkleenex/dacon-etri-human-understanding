#!/usr/bin/env python3
"""Promote an outer-supported partial calibration probe on top of ID168."""

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

from scripts.run_public_id168_calibration_probe_experiment import (  # noqa: E402
    CALIBRATION_PARAMS,
    calibrated_oof_and_test,
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
DEFAULT_BASE_ID = 168
MODIFIED_TARGETS = ["Q2", "Q3", "S1"]
OUTER_DIAGNOSTIC = "reports/id168_outer_calibration_selection_diagnostic.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_outer_supported_calibration_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if target in MODIFIED_TARGETS:
            config = CALIBRATION_PARAMS[target]
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
                "status": "modified",
                "config": config,
                "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
                "calibrated_loss": float(log_loss(y, calibrated_oof, labels=[0, 1])),
                "fold_details": fold_details,
                "test_prediction_mean_before": float(np.mean(base_test_values)),
                "test_prediction_mean_after": float(np.mean(calibrated_test)),
                "test_prediction_max_abs_delta": float(np.max(np.abs(calibrated_test - base_test_values))),
            }
        else:
            target_details[target] = {
                "status": "kept_base",
                "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
                "reason": "Outer-fold calibration selection diagnostic did not support promoting this target.",
            }
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "modified_targets": MODIFIED_TARGETS,
        "unchanged_targets": [target for target in TARGETS if target not in MODIFIED_TARGETS],
        "outer_diagnostic": OUTER_DIAGNOSTIC,
        "calibration_params": {target: CALIBRATION_PARAMS[target] for target in MODIFIED_TARGETS},
        "target_details": target_details,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv_plus_outer_supported_target_filter",
        "selection_contract": (
            f"Only Q2/Q3/S1 calibration choices from ID{base_id:03d} are promoted because an outer-fold "
            "selection diagnostic showed those targets retained directional calibration signal. Other targets remain "
            "at the ID168 base. This is a risk-control artifact, not strict Subject-Hole proof."
        ),
        "caution": (
            "Not strict Subject-Hole CV. Target inclusion uses an outer-fold diagnostic, but final calibration "
            "parameters still come from the ID168 calibration diagnostic. Use as a conservative public-structure "
            "probe, not as independent leaderboard evidence."
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
            f"# Submission {experiment_id:03d}: outer-supported partial calibration public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Modified targets: `{', '.join(MODIFIED_TARGETS)}`\n"
            f"- Public-structure calibrated CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for the immutable artifact."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_outer_supported_calibration_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id168_outer_supported_partial_calibration_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_outer_supported_calibration_probe(
        args.base_id
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
        "method": "Outer-supported partial public-structure calibration on ID168",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_partial_calibration",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "modified_targets": MODIFIED_TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Partial fixed calibration choices on ID168 for Q2/Q3/S1 only. Target inclusion is motivated by "
            "outer-fold calibration selection diagnostics; this remains non-strict public-structure CV."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Outer-supported partial public-structure calibration on ID{args.base_id:03d}: modifies Q2/Q3/S1 only; "
        "other targets remain ID168. Not strict Subject-Hole CV; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Outer-supported partial public calibration on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
