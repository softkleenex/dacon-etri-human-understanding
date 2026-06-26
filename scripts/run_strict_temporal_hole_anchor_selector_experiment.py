#!/usr/bin/env python3
"""Promote a strict subject-hole temporal-anchor selector on top of ID129."""

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

from scripts.run_public_mixed_temporal_anchor_probe_experiment import knn_mean_anchor_prediction  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    load_base_artifact,
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
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 129

ANCHOR_CANDIDATES: dict[str, list[dict[str, float | int | str]]] = {
    "Q1": [
        {"name": "exp_a0.8_h0.5", "mode": "exp", "alpha_base": 0.80, "half_life": 0.5},
        {"name": "exp_a0.8_h1", "mode": "exp", "alpha_base": 0.80, "half_life": 1.0},
        {"name": "knn_w0.2_k1", "mode": "knn", "blend_weight": 0.20, "k": 1},
    ],
    "Q2": [
        {"name": "knn_w0.1_k4", "mode": "knn", "blend_weight": 0.10, "k": 4},
        {"name": "knn_w0.2_k3", "mode": "knn", "blend_weight": 0.20, "k": 3},
        {"name": "knn_w0.2_k4", "mode": "knn", "blend_weight": 0.20, "k": 4},
        {"name": "knn_w0.2_k7", "mode": "knn", "blend_weight": 0.20, "k": 7},
    ],
    "Q3": [
        {"name": "exp_a0.95_h0.25", "mode": "exp", "alpha_base": 0.95, "half_life": 0.25},
        {"name": "exp_a0.95_h0.5", "mode": "exp", "alpha_base": 0.95, "half_life": 0.5},
        {"name": "knn_w0.05_k1", "mode": "knn", "blend_weight": 0.05, "k": 1},
    ],
    "S1": [
        {"name": "knn_w0.05_k1", "mode": "knn", "blend_weight": 0.05, "k": 1},
        {"name": "exp_a0.95_h0.25", "mode": "exp", "alpha_base": 0.95, "half_life": 0.25},
    ],
    "S2": [
        {"name": "knn_w0.1_k1", "mode": "knn", "blend_weight": 0.10, "k": 1},
        {"name": "exp_a0.9_h0.25", "mode": "exp", "alpha_base": 0.90, "half_life": 0.25},
        {"name": "knn_w0.2_k1", "mode": "knn", "blend_weight": 0.20, "k": 1},
    ],
    "S3": [
        {"name": "exp_a0.95_h0.25", "mode": "exp", "alpha_base": 0.95, "half_life": 0.25},
        {"name": "knn_w0.05_k1", "mode": "knn", "blend_weight": 0.05, "k": 1},
        {"name": "exp_a0.9_h0.25", "mode": "exp", "alpha_base": 0.90, "half_life": 0.25},
    ],
    "S4": [
        {"name": "knn_w0.2_k14", "mode": "knn", "blend_weight": 0.20, "k": 14},
        {"name": "knn_w0.2_k21", "mode": "knn", "blend_weight": 0.20, "k": 21},
        {"name": "knn_w0.3_k14", "mode": "knn", "blend_weight": 0.30, "k": 14},
        {"name": "knn_w0.3_k21", "mode": "knn", "blend_weight": 0.30, "k": 21},
    ],
}


def candidate_name(config: dict[str, float | int | str]) -> str:
    return str(config["name"])


def apply_anchor_config(
    base_values: np.ndarray,
    anchor_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    if str(config["mode"]) == "exp":
        alpha_base = float(config["alpha_base"])
        return np.clip(alpha_base * base_values + (1.0 - alpha_base) * anchor_values, EPS, 1.0 - EPS)
    if str(config["mode"]) == "knn":
        blend_weight = float(config["blend_weight"])
        return np.clip((1.0 - blend_weight) * base_values + blend_weight * anchor_values, EPS, 1.0 - EPS)
    raise ValueError(f"Unknown anchor mode: {config['mode']}")


def anchor_prediction(
    known_df: pd.DataFrame,
    known_y: np.ndarray,
    query_df: pd.DataFrame,
    config: dict[str, float | int | str],
) -> np.ndarray:
    mode = str(config["mode"])
    if mode == "exp":
        return temporal_anchor_prediction(
            known_df=known_df,
            known_y=known_y,
            query_df=query_df,
            half_life=float(config["half_life"]),
        )
    if mode == "knn":
        return knn_mean_anchor_prediction(
            known_df=known_df,
            known_y=known_y,
            query_df=query_df,
            k=int(config["k"]),
        )
    raise ValueError(f"Unknown anchor mode: {mode}")


def blocked_candidate_oof(
    train: pd.DataFrame,
    y: np.ndarray,
    base_values: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, float | int | str],
) -> tuple[np.ndarray, list[dict]]:
    selected = np.zeros(len(train), dtype=float)
    fold_details: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        anchor = anchor_prediction(
            known_df=train.iloc[train_idx],
            known_y=y[train_idx],
            query_df=train.iloc[valid_idx],
            config=config,
        )
        selected[valid_idx] = apply_anchor_config(base_values[valid_idx], anchor, config)
        fold_details.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_loss": float(log_loss(y[valid_idx], selected[valid_idx], labels=[0, 1])),
            }
        )
    return selected, fold_details


def candidate_test_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    base_test_values: np.ndarray,
    config: dict[str, float | int | str],
) -> np.ndarray:
    anchor = anchor_prediction(known_df=train, known_y=y, query_df=test, config=config)
    return apply_anchor_config(base_test_values, anchor, config)


def predict_temporal_hole_anchor_selector(
    base_id: int,
    guard_non_improving_targets: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target in TARGETS:
        y = train_raw[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidate_oof: dict[str, np.ndarray] = {"base": base_values}
        candidate_test: dict[str, np.ndarray] = {"base": base_test_values}
        candidate_fold_details: dict[str, list[dict]] = {}

        for config in ANCHOR_CANDIDATES[target]:
            name = candidate_name(config)
            candidate_oof[name], candidate_fold_details[name] = blocked_candidate_oof(
                train=train,
                y=y,
                base_values=base_values,
                folds=folds,
                config=config,
            )
            candidate_test[name] = candidate_test_prediction(
                train=train,
                test=test,
                y=y,
                base_test_values=base_test_values,
                config=config,
            )

        candidate_losses = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in candidate_oof.items()}
        selected_oof = np.zeros(len(train), dtype=float)
        fold_choices: list[dict] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            train_losses = {
                name: float(log_loss(y[train_idx], values[train_idx], labels=[0, 1]))
                for name, values in candidate_oof.items()
            }
            selected_name = min(train_losses, key=train_losses.get)
            selected_oof[valid_idx] = candidate_oof[selected_name][valid_idx]
            fold_choices.append(
                {
                    "fold": int(fold_idx),
                    "selected": selected_name,
                    "selected_train_loss": train_losses[selected_name],
                    "selected_valid_loss": float(
                        log_loss(y[valid_idx], candidate_oof[selected_name][valid_idx], labels=[0, 1])
                    ),
                    "all_train_losses": train_losses,
                }
            )

        nested_selector_loss = float(log_loss(y, selected_oof, labels=[0, 1]))
        full_oof_selected_for_test = min(candidate_losses, key=candidate_losses.get)
        applied_for_test = full_oof_selected_for_test
        guarded_to_base = bool(guard_non_improving_targets and nested_selector_loss >= candidate_losses["base"])
        if guarded_to_base:
            selected_oof = base_values
            applied_for_test = "base"

        oof[target] = np.clip(selected_oof, EPS, 1.0 - EPS)
        test_pred[target] = candidate_test[applied_for_test]
        target_details[target] = {
            "base_loss": candidate_losses["base"],
            "nested_selector_loss_before_guard": nested_selector_loss,
            "final_oof_loss_after_guard": float(log_loss(y, selected_oof, labels=[0, 1])),
            "full_oof_best_loss": candidate_losses[full_oof_selected_for_test],
            "full_oof_selected_for_test": full_oof_selected_for_test,
            "applied_for_test": applied_for_test,
            "guarded_to_base": guarded_to_base,
            "candidate_losses": dict(sorted(candidate_losses.items(), key=lambda item: item[1])),
            "fold_choices": fold_choices,
            "candidate_fold_details": candidate_fold_details,
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(test_pred[target].astype(float))),
            "test_prediction_max_abs_delta": float(
                np.max(np.abs(test_pred[target].astype(float).to_numpy() - base_test_values))
            ),
        }

    target_scores = {
        target: float(log_loss(train_raw[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "strict_subject_hole_nested_temporal_anchor_selector",
        "fold_rule": (
            "Uses src.utils.validation.make_subject_hole_folds. Candidate anchors for a validation hole use only "
            "same-subject labels outside that validation hole; fold-level candidate selection is based on losses "
            "computed on the fold training rows only."
        ),
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "anchor_candidates": ANCHOR_CANDIDATES,
        "guard_non_improving_targets": guard_non_improving_targets,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are replaced for each target by a fold-nested selector over a narrow "
            "same-subject temporal-anchor candidate pool plus the base candidate. Final test predictions use the "
            "candidate with the lowest full-training OOF loss per target, matching the existing strict-artifact "
            "promotion convention."
        ),
        "caution": (
            "The OOF score is strict Subject-Hole comparable inside this repository, but the final test candidate "
            "choice still uses full OOF losses, so this is a leaderboard candidate rather than a proof of Public "
            "score improvement."
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
    guard_non_improving_targets: bool,
) -> None:
    guard_arg = " --guard-non-improving-targets" if guard_non_improving_targets else ""
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: strict temporal-hole anchor selector\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Guard non-improving targets: `{guard_non_improving_targets}`\n"
            f"- Strict Subject-Hole nested CV: `{local_cv:.6f}`\n\n"
            "This artifact applies a fold-nested selector over same-subject temporal-anchor candidates. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_temporal_hole_anchor_selector_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}{guard_arg}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The CSV, notebook, metadata, base OOF, base test predictions, and submission log row are expected "
            "to share this immutable experiment id and slug."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="temporal_hole_anchor_selector")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument(
        "--guard-non-improving-targets",
        action="store_true",
        help="Keep base predictions for targets where the nested selector does not beat the base OOF loss.",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_temporal_hole_anchor_selector(
        base_id=args.base_id,
        guard_non_improving_targets=args.guard_non_improving_targets,
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
        "method": "Strict temporal-hole same-subject anchor selector",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "strict_subject_hole_nested_temporal_anchor_selector",
        "strict_subject_hole_comparable": True,
        "guard_non_improving_targets": args.guard_non_improving_targets,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Fold-nested selector over base plus same-subject temporal-anchor candidates using "
            "make_subject_hole_folds. This is comparable with the repository's strict Subject-Hole score lens."
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
        args.base_id,
        args.guard_non_improving_targets,
    )

    notes = (
        f"Strict Subject-Hole temporal-anchor selector on ID{args.base_id:03d}. Uses a fold-nested selector over "
        "base plus narrow same-subject exp/KNN label-anchor candidates"
        f"{' with non-improving targets guarded to base' if args.guard_non_improving_targets else ''}; "
        "no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict temporal-hole same-subject anchor selector on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
