#!/usr/bin/env python3
"""Promote fixed sleep-proxy threshold-rule overlays on top of a base artifact.

The fixed rules are inspired by external ETRI sleep-summary priors, but they use
only current competition covariates. OOF rule probabilities are fit inside each
Subject-Hole training fold; the rule/config selection itself is full-OOF tuned.
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

from scripts.run_public_temporal_anchor_probe_experiment import load_base_artifact  # noqa: E402
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
DEFAULT_BASE_ID = 162

THRESHOLD_RULE_PRESETS: dict[str, dict[str, dict[str, float | str]]] = {
    "id162_external_sleep_proxy_thresholds": {
        "S2": {
            "feature": "screen_presleep_ratio",
            "threshold": 0.05,
            "direction": "ge",
            "blend_weight": 0.05,
        },
        "S3": {
            "feature": "mUsage_night_time_ratio",
            "threshold": 0.03553448235273544,
            "direction": "ge",
            "blend_weight": 0.10,
        },
        "S4": {
            "feature": "pedo_total_steps",
            "threshold": 4902.0,
            "direction": "ge",
            "blend_weight": 0.15,
        },
    },
    "id164_broad_threshold_rules": {
        "Q1": {
            "feature": "slp_act_active_ratio",
            "threshold": 0.003703703703703704,
            "direction": "ge",
            "blend_weight": 0.03,
        },
        "Q2": {
            "feature": "act_morn_active",
            "threshold": 0.011111111111111112,
            "direction": "ge",
            "blend_weight": 0.07,
        },
        "Q3": {
            "feature": "circadian_light_ratio_roll7_mean",
            "threshold": 0.0010767513069716067,
            "direction": "ge",
            "blend_weight": 0.25,
        },
        "S2": {
            "feature": "screen_on_ratio_roll7_mean",
            "threshold": 0.179101760898395,
            "direction": "ge",
            "blend_weight": 0.07,
        },
        "S3": {
            "feature": "weekofyear",
            "threshold": 35.0,
            "direction": "ge",
            "blend_weight": 0.03,
        },
        "S4": {
            "feature": "wHr_sleep_to_all_mean_ratio",
            "threshold": 0.961257322664951,
            "direction": "ge",
            "blend_weight": 0.05,
        },
    },
    "id165_q3_broad_threshold_residual": {
        "Q3": {
            "feature": "wlight_all_q75",
            "threshold": 236.0,
            "direction": "ge",
            "blend_weight": 0.10,
        },
    },
}


def add_rule_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"wHr_sleep_duration_deep", "wHr_sleep_duration_light"}.issubset(out.columns):
        out["sleep_total_proxy"] = out[["wHr_sleep_duration_deep", "wHr_sleep_duration_light"]].sum(axis=1)
    if {"wHr_sleep_duration_deep", "wHr_sleep_duration_light", "wHr_sleep_duration_awake"}.issubset(out.columns):
        out["time_in_bed_proxy"] = out[
            ["wHr_sleep_duration_deep", "wHr_sleep_duration_light", "wHr_sleep_duration_awake"]
        ].sum(axis=1)
        out["eff_calc"] = out["sleep_total_proxy"] / out["time_in_bed_proxy"].replace(0, np.nan)
    return out


def feature_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if feature not in train.columns or feature not in test.columns:
        raise KeyError(f"Missing threshold-rule feature: {feature}")
    train_values = pd.to_numeric(train[feature], errors="coerce").astype(float)
    test_values = pd.to_numeric(test[feature], errors="coerce").astype(float)
    fill_value = float(pd.concat([train_values, test_values], ignore_index=True).median())
    if not np.isfinite(fill_value):
        fill_value = 0.0
    return train_values.fillna(fill_value).to_numpy(), test_values.fillna(fill_value).to_numpy(), fill_value


def threshold_flags(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "ge":
        return values >= threshold
    if direction == "lt":
        return values < threshold
    raise ValueError(f"Unknown threshold direction: {direction}")


def smoothed_side_probabilities(y: np.ndarray, flags: np.ndarray) -> tuple[float, float]:
    if flags.any():
        true_prob = float((y[flags].sum() + 0.5) / (flags.sum() + 1.0))
    else:
        true_prob = float(np.mean(y))
    false_flags = ~flags
    if false_flags.any():
        false_prob = float((y[false_flags].sum() + 0.5) / (false_flags.sum() + 1.0))
    else:
        false_prob = float(np.mean(y))
    return true_prob, false_prob


def threshold_rule_oof_and_test(
    y: np.ndarray,
    train_values: np.ndarray,
    test_values: np.ndarray,
    threshold: float,
    direction: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    oof = np.zeros(len(y), dtype=float)
    fold_details = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_flags = threshold_flags(train_values[train_idx], threshold, direction)
        valid_flags = threshold_flags(train_values[valid_idx], threshold, direction)
        true_prob, false_prob = smoothed_side_probabilities(y[train_idx], train_flags)
        oof[valid_idx] = np.where(valid_flags, true_prob, false_prob)
        fold_details.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "train_flag_rate": float(np.mean(train_flags)),
                "valid_flag_rate": float(np.mean(valid_flags)),
                "true_side_probability": true_prob,
                "false_side_probability": false_prob,
            }
        )

    full_flags = threshold_flags(train_values, threshold, direction)
    full_true_prob, full_false_prob = smoothed_side_probabilities(y, full_flags)
    test_flags = threshold_flags(test_values, threshold, direction)
    test_pred = np.where(test_flags, full_true_prob, full_false_prob)
    full_fit = {
        "train_flag_rate": float(np.mean(full_flags)),
        "test_flag_rate": float(np.mean(test_flags)),
        "true_side_probability": full_true_prob,
        "false_side_probability": full_false_prob,
    }
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred, EPS, 1.0 - EPS), fold_details, full_fit


def predict_fixed_threshold_rule_overlay(
    base_id: int,
    candidate_preset: str,
    include_targets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_rule_proxy_features(train_raw)
    test = add_rule_proxy_features(test_raw)
    if candidate_preset not in THRESHOLD_RULE_PRESETS:
        raise ValueError(f"Unknown threshold-rule preset: {candidate_preset}")
    candidate_pool = THRESHOLD_RULE_PRESETS[candidate_preset]
    selected_candidates = candidate_pool
    if include_targets is not None:
        unknown = sorted(set(include_targets) - set(candidate_pool))
        if unknown:
            raise ValueError(f"Unknown threshold-rule target(s): {unknown}")
        selected_candidates = {target: candidate_pool[target] for target in include_targets}

    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target, config in selected_candidates.items():
        y = train_raw[target].astype(int).to_numpy()
        feature = str(config["feature"])
        threshold = float(config["threshold"])
        direction = str(config["direction"])
        weight = float(config["blend_weight"])
        train_values, test_values, fill_value = feature_values(train, test, feature)
        rule_oof, rule_test, fold_details, full_fit = threshold_rule_oof_and_test(
            y=y,
            train_values=train_values,
            test_values=test_values,
            threshold=threshold,
            direction=direction,
            folds=folds,
        )
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        blended_oof = np.clip((1.0 - weight) * base_values + weight * rule_oof, EPS, 1.0 - EPS)
        blended_test = np.clip((1.0 - weight) * base_test_values + weight * rule_test, EPS, 1.0 - EPS)
        oof[target] = blended_oof
        test_pred[target] = blended_test
        target_details[target] = {
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "rule_oof_loss": float(log_loss(y, rule_oof, labels=[0, 1])),
            "blended_oof_loss": float(log_loss(y, blended_oof, labels=[0, 1])),
            "candidate": config,
            "fill_value": fill_value,
            "fold_details": fold_details,
            "full_fit": full_fit,
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
        "validation_lens": "subject_hole_fixed_full_oof_threshold_rule_overlay",
        "fold_rule": (
            "Rule side probabilities are fit inside each Subject-Hole training fold for OOF rows. "
            "Threshold/config selection is fixed from a prior full-OOF diagnostic, not freshly nested."
        ),
        "modified_targets": sorted(selected_candidates),
        "unchanged_targets": [target for target in TARGETS if target not in selected_candidates],
        "candidate_preset": candidate_preset,
        "fixed_threshold_rule_candidates": selected_candidates,
        "target_details": target_details,
        "external_note": (
            "External ETRI 2018-2020 summaries motivated the sleep/activity proxy search, but no external rows "
            "or labels are concatenated into current training."
        ),
        "caution": "Locally tuned leaderboard probe; threshold selection is full-OOF and not fold-nested.",
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
            f"# Submission {experiment_id:03d}: fixed threshold-rule overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local fixed Subject-Hole OOF CV: `{local_cv:.6f}`\n\n"
            "This artifact applies fixed sleep/activity proxy threshold rules. No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_fixed_threshold_rule_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--include-targets {','.join(include_targets)} --candidate-preset {candidate_preset}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The fixed threshold/config choices are locally tuned and not fold-nested."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="fixed_threshold_rule_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--include-targets", default="S2,S3,S4")
    parser.add_argument(
        "--candidate-preset",
        choices=sorted(THRESHOLD_RULE_PRESETS),
        default="id162_external_sleep_proxy_thresholds",
    )
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    include_targets = [target.strip() for target in args.include_targets.split(",") if target.strip()]
    oof, test_pred, target_scores, selector_info, sample_original = predict_fixed_threshold_rule_overlay(
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
        "method": "Fixed full-OOF-confirmed sleep/activity proxy threshold-rule overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "subject_hole_fixed_full_oof_threshold_rule_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "candidate_preset": args.candidate_preset,
        "validation": (
            f"Fixed threshold rules from preset {args.candidate_preset!r}. Rule probabilities respect "
            "Subject-Hole folds for OOF rows, but candidate selection is not freshly fold-nested."
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
        f"Fixed sleep/activity proxy threshold-rule overlay on ID{args.base_id:03d}: "
        f"{','.join(include_targets)} only. External ETRI summaries motivated the proxy search, but no external "
        "rows were concatenated. Candidate choices are locally tuned; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Fixed threshold-rule overlay on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
