#!/usr/bin/env python3
"""Promote deterministic sequence-summary embedding overlays."""

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

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    EPS,
    MODEL_SPECS,
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import (  # noqa: E402
    EXTRA_MODEL_SPECS,
    sequence_summary_features,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
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

DEFAULT_BASE_ID = 191
SELECTED_CONFIGS_ID193: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "model": "lr_c0_3", "weight": 0.10},
    "Q2": {"kind": "blend", "model": "lr_c0_3", "weight": 0.15},
    "Q3": {"kind": "blend", "model": "lr_c0_3", "weight": 0.07},
    "S1": {"kind": "base"},
    "S2": {"kind": "base"},
    "S3": {"kind": "base"},
    "S4": {"kind": "base"},
}
DIAGNOSTIC_REPORTS_ID193 = ["reports/id191_sequence_summary_embedding_diagnostic.json"]
PRESETS: dict[str, tuple[dict[str, dict[str, Any]], list[str]]] = {
    "id193": (SELECTED_CONFIGS_ID193, DIAGNOSTIC_REPORTS_ID193),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_lookup() -> dict[str, Any]:
    return {spec.name: spec for spec in (*MODEL_SPECS, *EXTRA_MODEL_SPECS)}


def predict_sequence_summary_embedding_overlay(
    base_id: int,
    selected_configs: dict[str, dict[str, Any]],
    diagnostic_reports: list[str],
) -> tuple[Any, Any, dict[str, float], dict[str, Any], Any]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=np.asarray(train["date"], dtype="datetime64[ns]")))
    specs = model_lookup()

    needed_models = sorted(
        {str(config["model"]) for config in selected_configs.values() if config.get("kind") == "blend"}
    )
    model_oof: dict[str, dict[str, np.ndarray]] = {model_name: {} for model_name in needed_models}
    model_test: dict[str, dict[str, np.ndarray]] = {model_name: {} for model_name in needed_models}
    for model_name in needed_models:
        if model_name not in specs:
            raise ValueError(f"Unknown sequence-summary model: {model_name}")
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            oof_pred, test_pred = fit_oof_model(x_train, x_test, y, fold_ids, specs[model_name])
            model_oof[model_name][target] = oof_pred
            model_test[model_name][target] = test_pred

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        config = selected_configs[target]
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if config["kind"] == "base":
            selected_oof = base_values
            selected_test = base_test_values
            target_details[target] = {
                "config": config,
                "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
                "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            }
        else:
            model_name = str(config["model"])
            weight = float(config["weight"])
            raw_oof = model_oof[model_name][target]
            raw_test = model_test[model_name][target]
            selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
            selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
            target_details[target] = {
                "config": config,
                "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
                "raw_sequence_summary_loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
                "test_prediction_mean_before": float(np.mean(base_test_values)),
                "test_prediction_mean_after": float(np.mean(selected_test)),
                "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
            }
        oof[target] = selected_oof
        test_pred[target] = selected_test

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "modified_targets": [target for target, config in selected_configs.items() if config["kind"] != "base"],
        "unchanged_targets": [target for target, config in selected_configs.items() if config["kind"] == "base"],
        "selected_configs": selected_configs,
        "diagnostic_reports": diagnostic_reports,
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_summary_features": int(x_train.shape[1]),
        "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are blended target-wise with deterministic sequence-summary embedding "
            "models. Target/model/weight choices are selected from listed local public-structure diagnostics."
        ),
        "caution": "Public-structure/non-strict sequence-summary overlay; not strict Subject-Hole evidence.",
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
    preset: str,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: sequence-summary embedding overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses deterministic window/slope/FFT/autocorrelation summaries over raw daily sequences."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_sequence_summary_embedding_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} --preset {preset}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id191_sequence_summary_embedding_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="id193")
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    selected_configs, diagnostic_reports = PRESETS[args.preset]
    oof, test_pred, target_scores, selector_info, sample_original = predict_sequence_summary_embedding_overlay(
        args.base_id,
        selected_configs=selected_configs,
        diagnostic_reports=diagnostic_reports,
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
        "method": f"Target-wise sequence-summary embedding overlay on ID{args.base_id:03d}",
        "preset": args.preset,
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_sequence_summary_embedding",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Target-wise sequence-summary embedding overlay under public-structure interleaved CV.",
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
        args.preset,
    )

    notes = (
        f"Public-structure deterministic sequence-summary embedding overlay on ID{args.base_id:03d}; "
        f"target-wise selected with preset {args.preset}. Not strict Subject-Hole CV; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Sequence-summary embedding overlay on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
