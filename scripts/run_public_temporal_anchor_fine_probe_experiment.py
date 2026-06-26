#!/usr/bin/env python3
"""Create a fine-tuned public-structure Q1/Q2/Q3 temporal-anchor probe artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import scripts.run_public_temporal_anchor_probe_experiment as anchor_base  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    map_predictions_to_original_sample,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

FINE_ANCHOR_PARAMS: dict[str, dict[str, float]] = {
    "Q1": {"alpha_base": 0.725, "half_life": 0.5},
    "Q2": {"alpha_base": 0.595, "half_life": 84.0},
    "Q3": {"alpha_base": 0.570, "half_life": 84.0},
}
DEFAULT_BASE_ID = 118


def apply_fine_params() -> None:
    anchor_base.FIXED_ANCHOR_PARAMS = FINE_ANCHOR_PARAMS


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
            f"# Submission {experiment_id:03d}: fine Q1/Q2/Q3 temporal-anchor public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure blocked CV: `{local_cv:.6f}`\n\n"
            "This artifact uses the fine alpha/half-life preset recorded in metadata. "
            "It is not strict Subject-Hole CV and should not be submitted without explicit approval."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_temporal_anchor_fine_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The artifact changes Q1/Q2/Q3 only and keeps S1/S2/S3/S4 from the base ID."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="q123_temporal_anchor_fine_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    apply_fine_params()
    oof, test_pred, target_scores, selector_info, sample_original = anchor_base.predict_temporal_anchor_probe(args.base_id)
    local_cv = float(np.mean(list(target_scores.values())))
    selector_info["fine_preset_origin"] = (
        "Fine fixed alpha/half-life scan after ID119. Kept as a separate script so ID119 remains reproducible."
    )

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
        "method": "Fine Q1/Q2/Q3 temporal-anchor public-structure probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fine fixed Q1/Q2/Q3 temporal-anchor parameters under interleaved within-subject blocked CV; "
            "S targets remain from the base ID. This is not directly comparable to strict Subject-Hole CV."
        ),
        "fine_anchor_params": FINE_ANCHOR_PARAMS,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id)

    notes = (
        f"Fine public-structure probe on ID{args.base_id:03d}: Q1/Q2/Q3 use tuned same-subject temporal-anchor "
        "blends; S1/S2/S3/S4 unchanged. Not strict Subject-Hole CV, not directly comparable with IDs 95-118, "
        "and no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Fine public-structure Q123 temporal-anchor probe on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
