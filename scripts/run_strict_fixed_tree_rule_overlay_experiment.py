#!/usr/bin/env python3
"""Promote fixed shallow decision-tree rule overlays on top of a base artifact."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_temporal_anchor_probe_experiment import load_base_artifact  # noqa: E402
from scripts.run_strict_fixed_threshold_rule_overlay_experiment import add_rule_proxy_features  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    numeric_feature_columns,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

EPS = 1e-6
DEFAULT_BASE_ID = 166

TREE_RULE_PRESETS: dict[str, dict[str, dict[str, float | int | str | None]]] = {
    "id166_s12_shallow_tree_rules": {
        "S1": {
            "name": "dt_d2_l15_gini_plain",
            "max_depth": 2,
            "min_samples_leaf": 15,
            "criterion": "gini",
            "class_weight": None,
            "blend_weight": 0.10,
        },
        "S2": {
            "name": "dt_d2_l60_entropy_plain",
            "max_depth": 2,
            "min_samples_leaf": 60,
            "criterion": "entropy",
            "class_weight": None,
            "blend_weight": 0.15,
        },
    }
}


def make_tree(config: dict[str, float | int | str | None], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        criterion=str(config["criterion"]),
        class_weight=config["class_weight"],
        random_state=seed,
    )


def fit_tree_oof_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    config: dict[str, float | int | str | None],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y = train[target].astype(int).to_numpy()
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_details = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        if len(np.unique(y[train_idx])) < 2:
            valid_pred = np.full(len(valid_idx), float(np.mean(y[train_idx])))
            fold_test_pred = np.full(len(test), float(np.mean(y[train_idx])))
            root_feature = None
            n_leaves = 1
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), make_tree(config, seed=2000 + fold_idx))
            model.fit(train.iloc[train_idx][feature_cols], y[train_idx])
            valid_pred = model.predict_proba(train.iloc[valid_idx][feature_cols])[:, 1]
            fold_test_pred = model.predict_proba(test[feature_cols])[:, 1]
            tree = model.named_steps["decisiontreeclassifier"]
            root_feature = feature_cols[int(tree.tree_.feature[0])] if tree.tree_.feature[0] >= 0 else None
            n_leaves = int(tree.get_n_leaves())
        oof[valid_idx] = valid_pred
        test_pred += fold_test_pred / len(folds)
        fold_details.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_prediction_mean": float(np.mean(valid_pred)),
                "test_prediction_mean": float(np.mean(fold_test_pred)),
                "root_feature": root_feature,
                "n_leaves": n_leaves,
            }
        )
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred, EPS, 1.0 - EPS), fold_details


def predict_fixed_tree_rule_overlay(
    base_id: int,
    candidate_preset: str,
    include_targets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_rule_proxy_features(train_raw)
    test = add_rule_proxy_features(test_raw)
    if candidate_preset not in TREE_RULE_PRESETS:
        raise ValueError(f"Unknown tree-rule preset: {candidate_preset}")
    candidate_pool = TREE_RULE_PRESETS[candidate_preset]
    selected_candidates = candidate_pool
    if include_targets is not None:
        unknown = sorted(set(include_targets) - set(candidate_pool))
        if unknown:
            raise ValueError(f"Unknown tree-rule target(s): {unknown}")
        selected_candidates = {target: candidate_pool[target] for target in include_targets}

    feature_cols = numeric_feature_columns(train, test)
    feature_cols = [col for col in feature_cols if pd.to_numeric(train[col], errors="coerce").nunique(dropna=True) > 3]
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target, config in selected_candidates.items():
        y = train_raw[target].astype(int).to_numpy()
        raw_oof, raw_test, fold_details = fit_tree_oof_and_test(
            train=train,
            test=test,
            feature_cols=feature_cols,
            target=target,
            config=config,
            folds=folds,
        )
        weight = float(config["blend_weight"])
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        blended_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
        blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        oof[target] = blended_oof
        test_pred[target] = blended_test
        target_details[target] = {
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "tree_oof_loss": float(log_loss(y, raw_oof, labels=[0, 1])),
            "blended_oof_loss": float(log_loss(y, blended_oof, labels=[0, 1])),
            "candidate": config,
            "fold_details": fold_details,
            "n_features": len(feature_cols),
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(blended_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
        }

    target_scores = {
        target: float(log_loss(train_raw[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "subject_hole_fixed_full_oof_shallow_tree_rule_overlay",
        "fold_rule": (
            "Shallow tree OOF predictions are fit inside each Subject-Hole training fold. "
            "Tree config selection is fixed from a prior full-OOF diagnostic, not freshly nested."
        ),
        "modified_targets": sorted(selected_candidates),
        "unchanged_targets": [target for target in TARGETS if target not in selected_candidates],
        "candidate_preset": candidate_preset,
        "fixed_tree_rule_candidates": selected_candidates,
        "target_details": target_details,
        "caution": "Locally tuned leaderboard probe; tree config selection is full-OOF and not fold-nested.",
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
    include_targets: list[str],
    candidate_preset: str,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: fixed shallow tree-rule overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local fixed Subject-Hole OOF CV: `{local_cv:.6f}`\n\n"
            "This artifact applies fixed shallow decision-tree rule overlays. No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_fixed_tree_rule_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--include-targets {','.join(include_targets)} --candidate-preset {candidate_preset}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The fixed tree/config choices are locally tuned and not fold-nested."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="fixed_tree_rule_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--include-targets", default="S1,S2")
    parser.add_argument(
        "--candidate-preset",
        choices=sorted(TREE_RULE_PRESETS),
        default="id166_s12_shallow_tree_rules",
    )
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    include_targets = [target.strip() for target in args.include_targets.split(",") if target.strip()]
    oof, test_pred, target_scores, selector_info, sample_original = predict_fixed_tree_rule_overlay(
        base_id=args.base_id,
        candidate_preset=args.candidate_preset,
        include_targets=include_targets,
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
        "method": "Fixed full-OOF-confirmed shallow tree-rule overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "subject_hole_fixed_full_oof_shallow_tree_rule_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "candidate_preset": args.candidate_preset,
        "validation": (
            f"Fixed shallow tree rules from preset {args.candidate_preset!r}. OOF tree predictions respect "
            "Subject-Hole folds, but candidate selection is not freshly fold-nested."
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
        include_targets,
        args.candidate_preset,
    )

    notes = (
        f"Fixed shallow tree-rule overlay on ID{args.base_id:03d}: {','.join(include_targets)} only. "
        "Candidate choices are locally tuned; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Fixed shallow tree-rule overlay on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
