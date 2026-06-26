#!/usr/bin/env python3
"""Create a refined KNN temporal-anchor public-structure probe artifact."""

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

from scripts.run_public_mixed_temporal_anchor_probe_experiment import (  # noqa: E402
    EPS,
    knn_mean_anchor_prediction,
)
from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    load_base_artifact,
    make_interleaved_subject_folds,
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
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

DEFAULT_BASE_ID = 130
KNN_REFINE_CONFIG: dict[str, dict[str, float | int]] = {
    "Q2": {"k": 4, "blend_weight": 0.10},
    "Q3": {"k": 3, "blend_weight": 0.10},
    "S2": {"k": 21, "blend_weight": 0.20},
    "S4": {"k": 21, "blend_weight": 0.20},
}


def blocked_knn_anchor_oof(
    train: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[dict]]:
    selected = np.zeros(len(train), dtype=float)
    fold_details: list[dict] = []
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        selected[valid_idx] = knn_mean_anchor_prediction(
            known_df=train.iloc[train_idx],
            known_y=y[train_idx],
            query_df=train.iloc[valid_idx],
            k=k,
        )
        fold_details.append({"fold": int(fold), "n_valid": int(len(valid_idx))})
    return np.clip(selected, EPS, 1.0 - EPS), fold_details


def predict_knn_temporal_refine_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    fold_ids = make_interleaved_subject_folds(train)

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}
    for target, config in KNN_REFINE_CONFIG.items():
        y = train[target].astype(int).to_numpy()
        k = int(config["k"])
        blend_weight = float(config["blend_weight"])
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        anchor_oof, fold_details = blocked_knn_anchor_oof(train=train, y=y, fold_ids=fold_ids, k=k)
        anchor_test = knn_mean_anchor_prediction(known_df=train, known_y=y, query_df=test, k=k)
        selected_oof = np.clip((1.0 - blend_weight) * base_values + blend_weight * anchor_oof, EPS, 1.0 - EPS)
        selected_test = np.clip((1.0 - blend_weight) * base_test_values + blend_weight * anchor_test, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "k": k,
            "blend_weight": blend_weight,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "anchor_loss": float(log_loss(y, anchor_oof, labels=[0, 1])),
            "refined_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "fold_details": fold_details,
            "test_anchor_mean": float(np.mean(anchor_test)),
            "test_anchor_std": float(np.std(anchor_test)),
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "fold_rule": (
            "Within each subject, sorted train dates are assigned to fold = date_rank % 5. "
            "For validation rows, KNN label anchors use only non-validation labels from the same subject."
        ),
        "modified_targets": sorted(KNN_REFINE_CONFIG),
        "unchanged_targets": [target for target in TARGETS if target not in KNN_REFINE_CONFIG],
        "knn_refine_config": KNN_REFINE_CONFIG,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved except Q2/Q3/S2/S4, where fixed low-weight KNN "
            "same-subject temporal-anchor refinements are applied. Final test anchors use all labeled "
            "training dates for the same subject."
        ),
        "caution": (
            "This is a public-structure probe, not a strict Subject-Hole CV artifact. It is intended as a cleaner "
            "anchor for subsequent tempered leaderboard probes."
        ),
    }
    return (
        oof.drop(columns=["date_dt"], errors="ignore"),
        test_pred.drop(columns=["date_dt"], errors="ignore"),
        target_scores,
        selector_info,
        sample_original,
    )


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
            f"# Submission {experiment_id:03d}: refined KNN temporal-anchor public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure blocked CV: `{local_cv:.6f}`\n\n"
            "This artifact applies fixed low-weight KNN temporal refinements to Q2/Q3/S2/S4. "
            "It is not strict Subject-Hole CV and should not be submitted without explicit approval."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_knn_temporal_refine_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The artifact keeps globally unique CSV/notebook/report/base-prediction filenames."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="knn_temporal_refine_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_knn_temporal_refine_probe(args.base_id)
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
        "method": "Refined KNN temporal-anchor public-structure probe",
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
            "Fixed KNN same-subject temporal refinements under interleaved within-subject blocked CV. "
            "Not directly comparable to strict Subject-Hole CV."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Refined public-structure probe on ID{args.base_id:03d}: Q2/Q3/S2/S4 receive fixed low-weight "
        "KNN same-subject temporal-anchor refinements under interleaved within-subject blocked CV. "
        "Not strict Subject-Hole CV and no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Refined public-structure KNN temporal-anchor probe on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
