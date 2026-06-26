#!/usr/bin/env python3
"""Promote fixed feature-similarity anchor candidates on top of ID142.

This script intentionally differs from the fold-nested strict promotion scripts:
the target/config pairs are fixed from a prior full-OOF confirmation diagnostic.
OOF predictions still respect the Subject-Hole label boundary, but the candidate
choice itself is locally tuned and should be treated as a leaderboard probe.
"""

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

from scripts.run_public_temporal_anchor_probe_experiment import add_date_dt, load_base_artifact  # noqa: E402
from scripts.run_strict_feature_similarity_anchor_overlay_experiment import (  # noqa: E402
    blocked_feature_similarity_candidate_oof,
    candidate_test_prediction,
    scaled_feature_matrices,
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
    numeric_feature_columns,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 142

BASE_FEATURE_FAMILY_TOKENS: dict[str, list[str]] = {
    "date_time": ["date", "day", "week", "rank", "frac", "subject_code"],
    "light_hr": ["light", "hr", "sleep"],
    "move_gps": ["gps", "pedo", "step", "distance", "calorie", "activity"],
    "screen_app": ["screen", "usage", "app", "unlock", "charge"],
}
ENV_FAMILY_TOKEN_PRESETS: dict[str, list[str]] = {
    "env24": ["ambience", "wifi", "ble"],
    "legacy_light47": ["temp", "temperature", "humid", "humidity", "co2", "pm", "dust", "noise", "lux", "light"],
}

FIXED_FEATURE_SIMILARITY_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q1": {
        "name": "date_time_k3_dw0.03_w0.1",
        "family": "date_time",
        "k": 3,
        "date_weight": 0.03,
        "blend_weight": 0.10,
    },
    "Q2": {
        "name": "screen_app_k1_dw0.03_w0.15",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.03,
        "blend_weight": 0.15,
    },
    "Q3": {
        "name": "screen_app_k10_dw0.05_w0.35",
        "family": "screen_app",
        "k": 10,
        "date_weight": 0.05,
        "blend_weight": 0.35,
    },
    "S4": {
        "name": "env_k2_dw0.1_w0.1",
        "family": "env",
        "k": 2,
        "date_weight": 0.10,
        "blend_weight": 0.10,
    },
    "S1": {
        "name": "light_hr_k1_dw0.05_w0.05",
        "family": "light_hr",
        "k": 1,
        "date_weight": 0.05,
        "blend_weight": 0.05,
    },
}
REFINED_GRID_ID146_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q2": {
        "name": "all_k1_dw0_w0.1",
        "family": "all",
        "k": 1,
        "date_weight": 0.0,
        "blend_weight": 0.10,
    },
    "Q3": {
        "name": "all_k1_dw0.1_w0.2",
        "family": "all",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.20,
    },
    "S2": {
        "name": "move_gps_k10_dw0.1_w0.2",
        "family": "move_gps",
        "k": 10,
        "date_weight": 0.10,
        "blend_weight": 0.20,
    },
    "S3": {
        "name": "screen_app_k3_dw0.05_w0.2",
        "family": "screen_app",
        "k": 3,
        "date_weight": 0.05,
        "blend_weight": 0.20,
    },
    "S4": {
        "name": "light_hr_k3_dw0.03_w0.1",
        "family": "light_hr",
        "k": 3,
        "date_weight": 0.03,
        "blend_weight": 0.10,
    },
}
REFINED_GRID_ID147_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q2": {
        "name": "all_k3_dw0.1_w0.1",
        "family": "all",
        "k": 3,
        "date_weight": 0.10,
        "blend_weight": 0.10,
    },
    "Q3": {
        "name": "move_gps_k3_dw0.03_w0.15",
        "family": "move_gps",
        "k": 3,
        "date_weight": 0.03,
        "blend_weight": 0.15,
    },
    "S2": {
        "name": "all_k1_dw0_w0.05",
        "family": "all",
        "k": 1,
        "date_weight": 0.0,
        "blend_weight": 0.05,
    },
    "S3": {
        "name": "move_gps_k1_dw0_w0.05",
        "family": "move_gps",
        "k": 1,
        "date_weight": 0.0,
        "blend_weight": 0.05,
    },
    "S4": {
        "name": "screen_app_k10_dw0.1_w0.15",
        "family": "screen_app",
        "k": 10,
        "date_weight": 0.10,
        "blend_weight": 0.15,
    },
}
REFINED_GRID_ID148_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q1": {
        "name": "screen_app_k1_dw0_w0.05",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.0,
        "blend_weight": 0.05,
    },
    "Q2": {
        "name": "screen_app_k1_dw0.1_w0.03",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.03,
    },
    "Q3": {
        "name": "screen_app_k1_dw0.1_w0.08",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.08,
    },
    "S1": {
        "name": "screen_app_k1_dw0.1_w0.08",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.08,
    },
    "S2": {
        "name": "screen_app_k1_dw0.1_w0.03",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.03,
    },
    "S3": {
        "name": "date_time_k7_dw0.05_w0.15",
        "family": "date_time",
        "k": 7,
        "date_weight": 0.05,
        "blend_weight": 0.15,
    },
    "S4": {
        "name": "env_k1_dw0.03_w0.03",
        "family": "env",
        "k": 1,
        "date_weight": 0.03,
        "blend_weight": 0.03,
    },
}
REFINED_GRID_ID151_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q1": {
        "name": "light_hr_k1_dw0.1_w0.05",
        "family": "light_hr",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.05,
    },
    "Q2": {
        "name": "date_time_k7_dw0_w0.08",
        "family": "date_time",
        "k": 7,
        "date_weight": 0.0,
        "blend_weight": 0.08,
    },
    "Q3": {
        "name": "env_k5_dw0.03_w0.05",
        "family": "env",
        "k": 5,
        "date_weight": 0.03,
        "blend_weight": 0.05,
    },
    "S1": {
        "name": "env_k1_dw0.1_w0.05",
        "family": "env",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.05,
    },
    "S2": {
        "name": "light_hr_k1_dw0_w0.03",
        "family": "light_hr",
        "k": 1,
        "date_weight": 0.0,
        "blend_weight": 0.03,
    },
    "S3": {
        "name": "screen_app_k5_dw0_w0.08",
        "family": "screen_app",
        "k": 5,
        "date_weight": 0.0,
        "blend_weight": 0.08,
    },
    "S4": {
        "name": "light_hr_k3_dw0.05_w0.02",
        "family": "light_hr",
        "k": 3,
        "date_weight": 0.05,
        "blend_weight": 0.02,
    },
}
REFINED_GRID_ID153_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q1": {
        "name": "date_time_k1_dw0_w0.03",
        "family": "date_time",
        "k": 1,
        "date_weight": 0.0,
        "blend_weight": 0.03,
    },
    "Q2": {
        "name": "move_gps_k10_dw0.1_w0.05",
        "family": "move_gps",
        "k": 10,
        "date_weight": 0.10,
        "blend_weight": 0.05,
    },
    "Q3": {
        "name": "move_gps_k2_dw0_w0.02",
        "family": "move_gps",
        "k": 2,
        "date_weight": 0.0,
        "blend_weight": 0.02,
    },
    "S1": {
        "name": "screen_app_k5_dw0.03_w0.05",
        "family": "screen_app",
        "k": 5,
        "date_weight": 0.03,
        "blend_weight": 0.05,
    },
    "S2": {
        "name": "screen_app_k3_dw0_w0.03",
        "family": "screen_app",
        "k": 3,
        "date_weight": 0.0,
        "blend_weight": 0.03,
    },
    "S3": {
        "name": "env_k1_dw0.03_w0.03",
        "family": "env",
        "k": 1,
        "date_weight": 0.03,
        "blend_weight": 0.03,
    },
    "S4": {
        "name": "date_time_k7_dw0.05_w0.01",
        "family": "date_time",
        "k": 7,
        "date_weight": 0.05,
        "blend_weight": 0.01,
    },
}
REFINED_GRID_ID155_Q1S1S3_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q1": {
        "name": "env_k1_dw0.1_w0.015",
        "family": "env",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.015,
    },
    "S1": {
        "name": "screen_app_k1_dw0.2_w0.015",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.20,
        "blend_weight": 0.015,
    },
    "S3": {
        "name": "screen_app_k1_dw0.03_w0.05",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.03,
        "blend_weight": 0.05,
    },
}
REFINED_GRID_ID158_Q23S24_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q2": {
        "name": "move_gps_k2_dw0.2_w0.05",
        "family": "move_gps",
        "k": 2,
        "date_weight": 0.20,
        "blend_weight": 0.05,
    },
    "Q3": {
        "name": "move_gps_k10_dw0.05_w0.03",
        "family": "move_gps",
        "k": 10,
        "date_weight": 0.05,
        "blend_weight": 0.03,
    },
    "S2": {
        "name": "screen_app_k1_dw0.2_w0.05",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.20,
        "blend_weight": 0.05,
    },
    "S4": {
        "name": "move_gps_k1_dw0.2_w0.02",
        "family": "move_gps",
        "k": 1,
        "date_weight": 0.20,
        "blend_weight": 0.02,
    },
}
REFINED_GRID_ID160_Q23S4_CANDIDATES: dict[str, dict[str, float | int | str]] = {
    "Q2": {
        "name": "screen_app_k1_dw0.1_w0.03",
        "family": "screen_app",
        "k": 1,
        "date_weight": 0.10,
        "blend_weight": 0.03,
    },
    "Q3": {
        "name": "env_k2_dw0.2_w0.015",
        "family": "env",
        "k": 2,
        "date_weight": 0.20,
        "blend_weight": 0.015,
    },
    "S4": {
        "name": "env_k1_dw0.2_w0.015",
        "family": "env",
        "k": 1,
        "date_weight": 0.20,
        "blend_weight": 0.015,
    },
}
CANDIDATE_PRESETS: dict[str, dict[str, dict[str, float | int | str]]] = {
    "id142_confirmation": FIXED_FEATURE_SIMILARITY_CANDIDATES,
    "id146_grid_refined": REFINED_GRID_ID146_CANDIDATES,
    "id147_grid_refined": REFINED_GRID_ID147_CANDIDATES,
    "id148_grid_refined": REFINED_GRID_ID148_CANDIDATES,
    "id151_grid_refined": REFINED_GRID_ID151_CANDIDATES,
    "id153_grid_refined": REFINED_GRID_ID153_CANDIDATES,
    "id155_q1s1s3_grid_refined": REFINED_GRID_ID155_Q1S1S3_CANDIDATES,
    "id158_q23s24_grid_refined": REFINED_GRID_ID158_Q23S24_CANDIDATES,
    "id160_q23s4_grid_refined": REFINED_GRID_ID160_Q23S4_CANDIDATES,
}


def feature_family_tokens(env_family: str) -> dict[str, list[str]]:
    if env_family not in ENV_FAMILY_TOKEN_PRESETS:
        raise ValueError(f"Unknown env family preset: {env_family}")
    return {**BASE_FEATURE_FAMILY_TOKENS, "env": ENV_FAMILY_TOKEN_PRESETS[env_family]}


def feature_families(
    train: pd.DataFrame,
    test: pd.DataFrame,
    family_names: set[str] | None = None,
    env_family: str = "env24",
) -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(train, test)
    families: dict[str, list[str]] = {}
    if family_names is None or "all" in family_names:
        families["all"] = numeric_cols
    for family, tokens in feature_family_tokens(env_family).items():
        if family_names is not None and family not in family_names:
            continue
        cols = [col for col in numeric_cols if any(token in col.lower() for token in tokens)]
        if not cols:
            raise ValueError(f"Feature family {family!r} has no columns.")
        families[family] = cols
    return families


def predict_fixed_feature_similarity_overlay(
    base_id: int,
    include_targets: list[str] | None = None,
    env_family: str = "env24",
    candidate_preset: str = "id142_confirmation",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    if candidate_preset not in CANDIDATE_PRESETS:
        raise ValueError(f"Unknown candidate preset: {candidate_preset}")
    candidate_pool = CANDIDATE_PRESETS[candidate_preset]
    selected_candidates = candidate_pool
    if include_targets is not None:
        unknown = sorted(set(include_targets) - set(candidate_pool))
        if unknown:
            raise ValueError(f"Unknown fixed feature-similarity target(s): {unknown}")
        selected_candidates = {target: candidate_pool[target] for target in include_targets}
    selected_family_names = {str(config["family"]) for config in selected_candidates.values()}
    families = feature_families(train_raw, test_raw, selected_family_names, env_family=env_family)
    scaled = scaled_feature_matrices(train_raw, test_raw, families)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target, config in selected_candidates.items():
        y = train_raw[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        fixed_oof, fold_details = blocked_feature_similarity_candidate_oof(
            train=train,
            test=test,
            scaled=scaled,
            y=y,
            base_values=base_values,
            folds=folds,
            config=config,
        )
        fixed_test = candidate_test_prediction(
            train=train,
            test=test,
            scaled=scaled,
            y=y,
            base_test_values=base_test_values,
            config=config,
        )
        oof[target] = np.clip(fixed_oof, EPS, 1.0 - EPS)
        test_pred[target] = np.clip(fixed_test, EPS, 1.0 - EPS)
        target_details[target] = {
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "fixed_oof_loss": float(log_loss(y, fixed_oof, labels=[0, 1])),
            "candidate": config,
            "fold_details": fold_details,
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
        "validation_lens": "subject_hole_fixed_full_oof_confirmed_feature_similarity_overlay",
        "fold_rule": (
            "OOF rows use src.utils.validation.make_subject_hole_folds; each validation hole receives an anchor "
            "computed from same-subject labels outside that hole. Feature scaling/imputation uses only label-free "
            "train+test covariates."
        ),
        "modified_targets": sorted(selected_candidates),
        "unchanged_targets": [target for target in TARGETS if target not in selected_candidates],
        "feature_family_tokens": {
            family: tokens
            for family, tokens in feature_family_tokens(env_family).items()
            if family in selected_family_names
        },
        "env_family_preset": env_family,
        "candidate_preset": candidate_preset,
        "feature_family_sizes": {family: len(cols) for family, cols in families.items()},
        "fixed_feature_similarity_candidates": selected_candidates,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved except {sorted(selected_candidates)}. The fixed "
            f"target/config pairs come from preset {candidate_preset!r}, not from a fresh fold-nested selector."
        ),
        "caution": (
            "This is a locally tuned leaderboard probe. It is useful for DACON mapping, but weaker evidence than "
            "ID142 because candidate choice was made after inspecting full-OOF confirmations."
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
    include_targets: list[str],
    env_family: str,
    candidate_preset: str,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: fixed feature-similarity overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local fixed Subject-Hole OOF CV: `{local_cv:.6f}`\n\n"
            "This artifact applies fixed full-OOF-confirmed feature-similarity anchors to Q2/Q3/S4. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_fixed_feature_similarity_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--include-targets {','.join(include_targets)} --env-family {env_family} "
            f"--candidate-preset {candidate_preset}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The CSV, notebook, metadata, base OOF, base test predictions, and submission log row are expected "
            "to share this immutable experiment id and slug. Candidate selection is locally tuned rather than "
            "fully fold-nested."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="fixed_feature_similarity_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument(
        "--include-targets",
        default="Q2,Q3,S4",
        help="Comma-separated fixed target subset to apply. Default: Q2,Q3,S4.",
    )
    parser.add_argument(
        "--env-family",
        choices=sorted(ENV_FAMILY_TOKEN_PRESETS),
        default="env24",
        help="Environment feature-family preset. Use legacy_light47 only to reproduce ID143.",
    )
    parser.add_argument(
        "--candidate-preset",
        choices=sorted(CANDIDATE_PRESETS),
        default="id142_confirmation",
        help="Fixed candidate preset to apply.",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    include_targets = [target.strip() for target in args.include_targets.split(",") if target.strip()]
    oof, test_pred, target_scores, selector_info, sample_original = predict_fixed_feature_similarity_overlay(
        args.base_id,
        include_targets=include_targets,
        env_family=args.env_family,
        candidate_preset=args.candidate_preset,
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
        "method": "Fixed full-OOF-confirmed feature-similarity overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "subject_hole_fixed_full_oof_confirmed_feature_similarity_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "candidate_preset": args.candidate_preset,
        "validation": (
            f"Fixed target/config pairs from preset {args.candidate_preset!r}. OOF anchor generation respects "
            "Subject-Hole labels, but candidate selection is not freshly fold-nested."
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
        args.env_family,
        args.candidate_preset,
    )

    notes = (
        f"Fixed full-OOF-confirmed feature-similarity overlay on ID{args.base_id:03d}: "
        f"{','.join(include_targets)} only. OOF "
        "anchors obey Subject-Hole label boundaries, but target/config choices are locally tuned from the "
        f"{args.candidate_preset} preset; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Fixed feature-similarity overlay on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
