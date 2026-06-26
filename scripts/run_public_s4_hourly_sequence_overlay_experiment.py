#!/usr/bin/env python3
"""Promote an S4-only public hourly-sequence overlay on top of a calibrated base."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_sequence_s34_overlay_experiment import (  # noqa: E402
    align_sequence_features,
    build_raw_hourly_sequence_features,
    fit_public_sequence_candidate,
    load_base_artifact,
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
DEFAULT_BASE_ID = 169
TARGET = "S4"
MODEL_NAME = "et_d6_l12"
BLEND_WEIGHT = 0.18


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_s4_hourly_sequence_overlay(
    base_id: int,
) -> tuple[Any, Any, dict[str, float], dict[str, Any], Any]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    seq = build_raw_hourly_sequence_features(ROOT / "data" / "raw" / "ch2025_data_items")
    train_seq, test_seq, feature_cols = align_sequence_features(train, test, seq)
    raw_oof, raw_test = fit_public_sequence_candidate(
        train=train,
        test_seq=test_seq,
        train_seq=train_seq,
        feature_cols=feature_cols,
        target=TARGET,
        model_name=MODEL_NAME,
    )
    oof = base_oof.copy()
    test_pred = base_test.copy()
    y = train[TARGET].astype(int).to_numpy()
    base_values = np.clip(base_oof[TARGET].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test_values = np.clip(base_test[TARGET].astype(float).to_numpy(), EPS, 1.0 - EPS)
    selected_oof = np.clip((1.0 - BLEND_WEIGHT) * base_values + BLEND_WEIGHT * raw_oof, EPS, 1.0 - EPS)
    selected_test = np.clip((1.0 - BLEND_WEIGHT) * base_test_values + BLEND_WEIGHT * raw_test, EPS, 1.0 - EPS)
    oof[TARGET] = selected_oof
    test_pred[TARGET] = selected_test
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "modified_targets": [TARGET],
        "unchanged_targets": [target for target in TARGETS if target != TARGET],
        "n_sequence_rows": int(len(seq)),
        "n_sequence_features": int(len(feature_cols)),
        "sequence_window": "lifelog date 12:00 through next-day 09:59, flattened by integer hour bins",
        "sensor_families": ["screen", "charging", "activity", "phone_light", "watch_light", "heart_rate", "pedometer"],
        "target_details": {
            TARGET: {
                "model": MODEL_NAME,
                "blend_weight": BLEND_WEIGHT,
                "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
                "raw_sequence_loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                "overlay_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
                "test_prediction_mean_before": float(np.mean(base_test_values)),
                "test_prediction_mean_after": float(np.mean(selected_test)),
                "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all non-S4 targets. S4 receives a fixed {BLEND_WEIGHT:.2f} "
            "blend of the base prediction and the ET hourly-sequence model under public-structure interleaved folds."
        ),
        "caution": "Public-structure/non-strict overlay; blend weight was selected from a local rescan.",
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
            f"# Submission {experiment_id:03d}: S4 hourly sequence overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact changes only S4."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_s4_hourly_sequence_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id169_s4_hourly_sequence_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_s4_hourly_sequence_overlay(args.base_id)
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
        "method": f"S4-only hourly sequence overlay on ID{args.base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_s4_sequence_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "S4-only fixed hourly-sequence overlay under public-structure interleaved within-subject folds.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Public-structure S4-only hourly-sequence overlay on ID{args.base_id:03d}: ET sequence model blended at "
        f"{BLEND_WEIGHT:.2f}; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"S4 hourly sequence overlay on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
