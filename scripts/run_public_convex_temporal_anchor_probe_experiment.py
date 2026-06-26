#!/usr/bin/env python3
"""Create a high-risk full-OOF convex temporal-anchor public-structure probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    temporal_anchor_prediction,
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

DEFAULT_BASE_ID = 118
CONVEX_CANDIDATE_SPECS: dict[str, list[tuple[str, float]]] = {
    "Q1": [("nearest", 0.5), ("nearest", 0.25), ("nearest", 1.0), ("knn_mean", 3.0), ("subject_mean", 0.0)],
    "Q2": [("knn_mean", 3.0), ("knn_mean", 4.0), ("knn_mean", 5.0), ("nearest", 336.0), ("subject_mean", 0.0)],
    "Q3": [("nearest", 10000.0), ("nearest", 336.0), ("knn_mean", 3.0), ("subject_mean", 0.0)],
    "S1": [("nearest", 0.25), ("nearest", 0.5), ("knn_mean", 3.0), ("subject_mean", 0.0)],
    "S2": [("knn_mean", 5.0), ("knn_mean", 4.0), ("knn_mean", 7.0), ("nearest", 0.25), ("subject_mean", 0.0)],
    "S3": [("nearest", 0.25), ("nearest", 0.5), ("knn_mean", 5.0), ("subject_mean", 0.0)],
    "S4": [("knn_mean", 7.0), ("knn_mean", 15.0), ("nearest", 10.0), ("nearest", 0.25), ("subject_mean", 0.0)],
}


def subject_mean_anchor_prediction(
    known_df: pd.DataFrame,
    known_y: np.ndarray,
    query_df: pd.DataFrame,
) -> np.ndarray:
    known = known_df[["subject_id"]].copy().reset_index(drop=True)
    known["y"] = np.asarray(known_y, dtype=float)
    subject_means = known.groupby("subject_id")["y"].mean().to_dict()
    global_mean = float(np.mean(known_y)) if len(known_y) else 0.5
    return np.clip(query_df["subject_id"].map(subject_means).fillna(global_mean).to_numpy(dtype=float), EPS, 1.0 - EPS)


def anchor_prediction(
    known_df: pd.DataFrame,
    known_y: np.ndarray,
    query_df: pd.DataFrame,
    spec: tuple[str, float],
) -> np.ndarray:
    mode, param = spec
    if mode == "nearest":
        return temporal_anchor_prediction(
            known_df=known_df,
            known_y=known_y,
            query_df=query_df,
            half_life=float(param),
        )
    if mode == "knn_mean":
        return knn_mean_anchor_prediction(
            known_df=known_df,
            known_y=known_y,
            query_df=query_df,
            k=int(param),
        )
    if mode == "subject_mean":
        return subject_mean_anchor_prediction(known_df=known_df, known_y=known_y, query_df=query_df)
    raise ValueError(f"Unknown convex temporal-anchor component: {spec}")


def blocked_anchor_component_oof(
    train: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    spec: tuple[str, float],
) -> np.ndarray:
    selected = np.zeros(len(train), dtype=float)
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        selected[valid_idx] = anchor_prediction(
            known_df=train.iloc[train_idx],
            known_y=y[train_idx],
            query_df=train.iloc[valid_idx],
            spec=spec,
        )
    return np.clip(selected, EPS, 1.0 - EPS)


def optimize_convex_weights(y: np.ndarray, components: np.ndarray) -> np.ndarray:
    n_components = components.shape[0]

    def objective(theta: np.ndarray) -> float:
        exp_values = np.exp(theta - np.max(theta))
        weights = exp_values / exp_values.sum()
        pred = np.clip(weights @ components, EPS, 1.0 - EPS)
        return float(log_loss(y, pred, labels=[0, 1]))

    starts = [np.zeros(n_components, dtype=float)]
    for component_idx in range(n_components):
        start = np.full(n_components, -2.0, dtype=float)
        start[component_idx] = 2.0
        starts.append(start)

    best_result = None
    for start in starts:
        result = minimize(objective, start, method="BFGS", options={"maxiter": 1000, "gtol": 1e-8})
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("Convex weight optimization did not run.")
    exp_values = np.exp(best_result.x - np.max(best_result.x))
    return exp_values / exp_values.sum()


def predict_convex_temporal_anchor_probe(
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

    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        component_names = ["base"]
        component_oof = [np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)]
        component_test = [np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)]

        for spec in CONVEX_CANDIDATE_SPECS[target]:
            mode, param = spec
            component_names.append(f"{mode}_{param:g}")
            component_oof.append(blocked_anchor_component_oof(train=train, y=y, fold_ids=fold_ids, spec=spec))
            component_test.append(anchor_prediction(known_df=train, known_y=y, query_df=test, spec=spec))

        component_oof_array = np.vstack(component_oof)
        component_test_array = np.vstack(component_test)
        weights = optimize_convex_weights(y=y, components=component_oof_array)
        selected_oof = np.clip(weights @ component_oof_array, EPS, 1.0 - EPS)
        selected_test = np.clip(weights @ component_test_array, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test

        component_scores = {
            name: float(log_loss(y, values, labels=[0, 1]))
            for name, values in zip(component_names, component_oof_array, strict=True)
        }
        weight_map = {
            name: float(weight) for name, weight in zip(component_names, weights, strict=True) if float(weight) > 1e-6
        }
        target_details[target] = {
            "component_names": component_names,
            "component_scores": component_scores,
            "convex_weights": weight_map,
            "full_oof_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "test_prediction_mean_before": float(np.mean(component_test_array[0])),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - component_test_array[0]))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv_full_oof_convex_fit",
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "convex_candidate_specs": CONVEX_CANDIDATE_SPECS,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are replaced for all seven targets by full-OOF fitted convex blends "
            "over base predictions and fixed same-subject temporal-anchor components. Anchor component OOF values "
            "are fold-blocked, but convex weights are optimized on the full OOF vector."
        ),
        "caution": (
            "High-risk leaderboard calibration probe: this is not strict Subject-Hole CV and not fold-nested "
            "weight selection. A stricter nested convex diagnostic was worse than ID122, so this artifact should "
            "not be treated as a verified local improvement."
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
            f"# Submission {experiment_id:03d}: convex temporal-anchor public probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Full-OOF public-structure probe CV: `{local_cv:.6f}`\n\n"
            "This artifact uses full-OOF fitted convex weights over temporal-anchor components. "
            "It is high-risk, non-strict, and not fold-nested."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_convex_temporal_anchor_probe_experiment.py "
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
    parser.add_argument("--slug", default="q123s1234_convex_temporal_anchor_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_convex_temporal_anchor_probe(args.base_id)
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
        "method": "High-risk full-OOF convex all-target temporal-anchor public-structure probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_full_oof_convex_temporal_anchor_fit_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_weight_selection": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Full-OOF fitted convex weights over fold-blocked temporal-anchor components. This is non-strict, "
            "not fold-nested, and should only be used as a leaderboard calibration probe."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"High-risk public-structure probe on ID{args.base_id:03d}: all seven targets use full-OOF fitted convex "
        "blends over same-subject temporal anchors. Not strict Subject-Hole CV, not fold-nested, and no DACON "
        "submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"High-risk public-structure convex temporal-anchor probe on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
