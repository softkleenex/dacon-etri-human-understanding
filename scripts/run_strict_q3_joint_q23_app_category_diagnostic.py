#!/usr/bin/env python3
"""Diagnose a narrow Q3 joint subject-relative q23 + semantic app-category overlay."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_app_category_compact_experiment import (  # noqa: E402
    app_category_feature_sets,
    build_q2_app_category_frames,
)
from scripts.run_strict_q2_booster_compact_experiment import load_base_artifact  # noqa: E402
from scripts.run_strict_q3_app_category_compact_experiment import fit_predict_one_candidate  # noqa: E402
from scripts.run_strict_q3_subject_relative_compact_experiment import add_subject_relative_q23_features  # noqa: E402
from scripts.run_strict_q23_strain_experiment import RAW_CONTEXT_DIR, build_q23_strain_features  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "Q3"
DEFAULT_BASE_ID = 118
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20]


def build_joint_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, test, sample_original, category_report = build_q2_app_category_frames()
    strain = build_q23_strain_features(RAW_CONTEXT_DIR).copy()
    strain["date"] = pd.to_datetime(strain["date"]).dt.date
    if strain.duplicated(["subject_id", "date"]).any():
        duplicated = strain.loc[strain.duplicated(["subject_id", "date"], keep=False), ["subject_id", "date"]]
        raise ValueError(f"Q23 strain features are not unique by subject/date:\n{duplicated.head()}")

    train_joint = train.copy()
    test_joint = test.copy()
    train_joint["date"] = pd.to_datetime(train_joint["date"]).dt.date
    test_joint["date"] = pd.to_datetime(test_joint["date"]).dt.date
    train_joint = train_joint.merge(strain, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_joint = test_joint.merge(strain, on=["subject_id", "date"], how="left", validate="one_to_one")

    strain_cols = [col for col in strain.columns if col.startswith("strain_")]
    missing_train = int(train_joint[strain_cols].isna().all(axis=1).sum())
    missing_test = int(test_joint[strain_cols].isna().all(axis=1).sum())
    if missing_train or missing_test:
        raise ValueError(f"Missing all Q23 strain features for train={missing_train}, test={missing_test} rows.")

    train_joint, test_joint, rel_q23_cols = add_subject_relative_q23_features(train_joint, test_joint)
    joint_report = {
        "category_report": category_report,
        "strain_shape": list(strain.shape),
        "missing_all_strain_rows": {"train": missing_train, "test": missing_test},
        "n_relative_q23_features": len(rel_q23_cols),
    }
    return train_joint, test_joint, sample_original, joint_report


def joint_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=220,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.70,
        reg_lambda=40.0,
        random_state=42,
        verbosity=-1,
    )


def predict_joint_overlay(base_id: int) -> tuple[dict[str, float], dict]:
    train, test, _sample_original, joint_report = build_joint_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    app_sets = app_category_feature_sets(train, test)
    rel_q23_cols = [col for col in train.columns if col.startswith("rel_") and col.startswith(("rel_iqr_strain_", "rel_rank_strain_"))]
    feature_cols = [*rel_q23_cols, *app_sets["appcat_abs"]]
    if not rel_q23_cols or not app_sets["appcat_abs"]:
        raise ValueError("Joint Q3 features are missing relative q23 or app-category columns.")

    y = train[TARGET].astype(int).to_numpy()
    base_candidate_name = f"id{base_id:03d}_base"
    raw_name = "relq23_appcat_abs__lgbm_l7_lr02_n220_l40"
    raw_oof, raw_test = fit_predict_one_candidate(joint_model(), train, test, feature_cols)

    candidate_oof = {
        base_candidate_name: base_oof[TARGET].astype(float).to_numpy(),
        raw_name: raw_oof,
    }
    candidate_test = {
        base_candidate_name: base_test[TARGET].astype(float).to_numpy(),
        raw_name: raw_test,
    }
    candidate_details = {
        base_candidate_name: {"source": "base"},
        raw_name: {
            "feature_set": "subject_relative_q23_plus_appcat_abs",
            "model": "lgbm_l7_lr02_n220_l40",
            "n_features": len(feature_cols),
            "n_relative_q23_features": len(rel_q23_cols),
            "n_appcat_abs_features": len(app_sets["appcat_abs"]),
            "blend": False,
        },
    }
    for weight in BLEND_WEIGHTS:
        blend_name = f"blend_base{1.0 - weight:.2f}_{raw_name}_{weight:.2f}"
        candidate_oof[blend_name] = np.clip(
            (1.0 - weight) * candidate_oof[base_candidate_name] + weight * raw_oof,
            1e-6,
            1.0 - 1e-6,
        )
        candidate_test[blend_name] = np.clip(
            (1.0 - weight) * candidate_test[base_candidate_name] + weight * raw_test,
            1e-6,
            1.0 - 1e-6,
        )
        candidate_details[blend_name] = {
            "blend_candidate": raw_name,
            "base_weight": 1.0 - weight,
            "candidate_weight": weight,
        }

    candidate_scores = {
        name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())
    }
    folds = make_subject_hole_folds(train, n_folds=5)
    selected_q3_oof = np.zeros(len(train), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q3_oof[valid_idx] = candidate_oof[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_fold_log_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_losses = {name: log_loss(y, values, labels=[0, 1]) for name, values in candidate_oof.items()}
    full_best_name, full_best_loss = min(full_losses.items(), key=lambda item: item[1])

    oof = base_oof.copy()
    oof[TARGET] = np.clip(selected_q3_oof, 1e-6, 1.0 - 1e-6)
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "joint_subject_relative_q23_semantic_app_category",
        "joint_report": joint_report,
        "candidate_pool_size": len(candidate_oof),
        "q3_candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(full_losses.items())},
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q3. Q3 validation rows receive "
            "the joint q23/app-category candidate selected using only each outer fold's training rows. "
            "This diagnostic does not create a submission artifact."
        ),
        "contamination_note": (
            "Subject-relative q23 statistics and app-category aggregates are computed from unlabeled train/test "
            "covariate rows only. The semantic category dictionary is hand-written before target fitting."
        ),
    }
    return target_scores, selector_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    target_scores, selector_info = predict_joint_overlay(args.base_id)
    local_cv = float(np.mean(list(target_scores.values())))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q3_joint_q23_app_category_diagnostic_id{args.base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q3_joint_q23_app_category_diagnostic_id{args.base_id:03d}_{suffix}.json"
    report = {
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "selector_info": selector_info,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(path.relative_to(ROOT)),
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "q3_candidate_scores": selector_info["q3_candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
