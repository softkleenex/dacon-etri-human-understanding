#!/usr/bin/env python3
"""Promote a target-wise multi-seed random-conv sequence overlay."""

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
    MODEL_SPECS,
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
    make_kernels,
    random_conv_features,
    table_to_tensor,
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

EPS = 1e-6
DEFAULT_BASE_ID = 173
N_KERNELS = 192
PROMOTION_EXTRA_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_7", "lr", c_value=0.7),
    ModelSpec("lr_c50", "lr", c_value=50.0),
    ModelSpec("lr_c100", "lr", c_value=100.0),
)
SELECTED_CONFIGS_ID174: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "seed": 43017, "model": "lr_c0_1", "weight": 0.15},
    "Q2": {"kind": "blend", "seed": 43019, "model": "lr_c0_1", "weight": 0.10},
    "Q3": {"kind": "blend", "seed": 43017, "model": "lr_c0_03", "weight": 0.01},
    "S1": {"kind": "blend", "seed": 43019, "model": "hgb_l7", "weight": 0.10},
    "S2": {"kind": "blend", "seed": 43017, "model": "lr_c0_1", "weight": 0.03},
    "S3": {"kind": "base"},
    "S4": {"kind": "blend", "seed": 43017, "model": "lr_c0_1", "weight": 0.10},
}
SELECTED_CONFIGS_ID175: dict[str, dict[str, Any]] = {
    **SELECTED_CONFIGS_ID174,
    "S1": {"kind": "blend", "seed": 43020, "model": "lr_c0_1", "weight": 0.10},
}
SELECTED_CONFIGS_ID176: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "seed": 43017, "model": "lr_c0_1", "weight": 0.15},
    "Q2": {"kind": "blend", "seed": 43025, "model": "lr_c0_1", "weight": 0.15},
    "Q3": {"kind": "blend", "seed": 43024, "model": "lr_c0_1", "weight": 0.10},
    "S1": {"kind": "blend", "seed": 43020, "model": "lr_c0_1", "weight": 0.10},
    "S2": {"kind": "blend", "seed": 43023, "model": "hgb_l7", "weight": 0.10},
    "S3": {"kind": "base"},
    "S4": {"kind": "blend", "seed": 43024, "model": "lr_c0_03", "weight": 0.15},
}
SELECTED_CONFIGS_ID177: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "seed": 43103, "n_kernels": 384, "model": "lr_c0_1", "weight": 0.20},
    "Q2": {"kind": "blend", "seed": 43025, "n_kernels": 192, "model": "lr_c0_1", "weight": 0.15},
    "Q3": {"kind": "blend", "seed": 43102, "n_kernels": 384, "model": "lr_c0_1", "weight": 0.10},
    "S1": {"kind": "blend", "seed": 43029, "n_kernels": 192, "model": "lr_c0_1", "weight": 0.15},
    "S2": {"kind": "blend", "seed": 43023, "n_kernels": 192, "model": "hgb_l7", "weight": 0.10},
    "S3": {"kind": "base"},
    "S4": {"kind": "blend", "seed": 43030, "n_kernels": 192, "model": "lr_c0_1", "weight": 0.20},
}
SELECTED_CONFIGS_ID178: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "seed": 43103, "n_kernels": 384, "model": "lr_c100", "weight": 0.165},
    "Q2": {"kind": "blend", "seed": 43025, "n_kernels": 192, "model": "lr_c100", "weight": 0.125},
    "Q3": {"kind": "blend", "seed": 43037, "n_kernels": 192, "model": "lr_c50", "weight": 0.110},
    "S1": {"kind": "blend", "seed": 43034, "n_kernels": 192, "model": "lr_c50", "weight": 0.120},
    "S2": {"kind": "blend", "seed": 43023, "n_kernels": 192, "model": "lr_c100", "weight": 0.115},
    "S3": {"kind": "base"},
    "S4": {"kind": "blend", "seed": 43030, "n_kernels": 192, "model": "lr_c0_7", "weight": 0.140},
}
SELECTED_CONFIGS_ID190: dict[str, dict[str, Any]] = {
    **SELECTED_CONFIGS_ID178,
    "S1": {"kind": "blend", "seed": 43203, "n_kernels": 512, "model": "lr_c0_1", "weight": 0.20},
}
DIAGNOSTIC_REPORTS_ID174 = [
    "reports/id173_random_conv_sequence_diagnostic.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43018.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43019.json",
]
DIAGNOSTIC_REPORTS_ID175 = [
    *DIAGNOSTIC_REPORTS_ID174,
    "reports/id173_random_conv_sequence_diagnostic_seed43020.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43021.json",
]
DIAGNOSTIC_REPORTS_ID176 = [
    *DIAGNOSTIC_REPORTS_ID175,
    "reports/id173_random_conv_sequence_diagnostic_seed43022.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43023.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43024.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43025.json",
]
DIAGNOSTIC_REPORTS_ID177 = [
    *DIAGNOSTIC_REPORTS_ID176,
    "reports/id173_random_conv_sequence_diagnostic_seed43026.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43027.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43028.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43029.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43030.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43031.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43032.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43033.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43034.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43035.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43036.json",
    "reports/id173_random_conv_sequence_diagnostic_seed43037.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43101.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43102.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43103.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43104.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43105.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43106.json",
]
DIAGNOSTIC_REPORTS_ID178 = [
    *DIAGNOSTIC_REPORTS_ID177,
    "reports/id173_random_conv_fine_selection_diagnostic.json",
    "reports/id173_random_conv_fine_selection_lrwide_diagnostic.json",
    "reports/id173_random_conv_fine_selection_lrultra_diagnostic.json",
    "reports/id173_random_conv_fine_selection_highc_diagnostic.json",
]
DIAGNOSTIC_REPORTS_ID190 = [
    *DIAGNOSTIC_REPORTS_ID178,
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43107.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43108.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43109.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43110.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43111.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43112.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43113.json",
    "reports/id173_random_conv_sequence_diagnostic_nk384_seed43114.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43201.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43202.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43203.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43204.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43205.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43206.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43207.json",
    "reports/id173_random_conv_sequence_diagnostic_nk512_seed43208.json",
    "reports/id173_random_conv_sequence_diagnostic_nk768_seed43301.json",
    "reports/id173_random_conv_sequence_diagnostic_nk768_seed43302.json",
]
PRESETS: dict[str, tuple[dict[str, dict[str, Any]], list[str]]] = {
    "id174": (SELECTED_CONFIGS_ID174, DIAGNOSTIC_REPORTS_ID174),
    "id175": (SELECTED_CONFIGS_ID175, DIAGNOSTIC_REPORTS_ID175),
    "id176": (SELECTED_CONFIGS_ID176, DIAGNOSTIC_REPORTS_ID176),
    "id177": (SELECTED_CONFIGS_ID177, DIAGNOSTIC_REPORTS_ID177),
    "id178": (SELECTED_CONFIGS_ID178, DIAGNOSTIC_REPORTS_ID178),
    "id190": (SELECTED_CONFIGS_ID190, DIAGNOSTIC_REPORTS_ID190),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_lookup() -> dict[str, Any]:
    return {spec.name: spec for spec in [*MODEL_SPECS, *PROMOTION_EXTRA_MODEL_SPECS]}


def predict_multiseed_random_conv_overlay(
    base_id: int,
    selected_configs: dict[str, dict[str, Any]],
    diagnostic_reports: list[str],
) -> tuple[Any, Any, dict[str, float], dict[str, Any], Any]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=np.asarray(train["date"], dtype="datetime64[ns]")))
    specs = model_lookup()

    tensor_features: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
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
            seed = int(config["seed"])
            n_kernels = int(config.get("n_kernels", N_KERNELS))
            model_name = str(config["model"])
            weight = float(config["weight"])
            feature_key = (seed, n_kernels)
            if feature_key not in tensor_features:
                kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
                tensor_features[feature_key] = (
                    random_conv_features(train_tensor, kernels),
                    random_conv_features(test_tensor, kernels),
                )
            x_train, x_test = tensor_features[feature_key]
            raw_oof, raw_test = fit_oof_model(x_train, x_test, y, fold_ids, specs[model_name])
            selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
            selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
            target_details[target] = {
                "config": config,
                "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
                "raw_sequence_loss": float(log_loss(y, raw_oof, labels=[0, 1])),
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
    used_seed_kernel_pairs = sorted(
        {
            (int(config["seed"]), int(config.get("n_kernels", N_KERNELS)))
            for config in selected_configs.values()
            if config["kind"] != "base"
        }
    )
    used_kernel_counts = sorted({n_kernels for _seed, n_kernels in used_seed_kernel_pairs})
    selector_info = {
        "base_source": base_source,
        "modified_targets": [target for target, config in selected_configs.items() if config["kind"] != "base"],
        "unchanged_targets": [target for target, config in selected_configs.items() if config["kind"] == "base"],
        "selected_configs": selected_configs,
        "diagnostic_reports": diagnostic_reports,
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_kernels_per_seed": used_kernel_counts[0] if len(used_kernel_counts) == 1 else used_kernel_counts,
        "n_random_conv_features": int(max(used_kernel_counts or [N_KERNELS]) * 4),
        "n_random_conv_features_by_kernel_count": {
            str(n_kernels): int(n_kernels * 4) for n_kernels in used_kernel_counts
        },
        "used_seeds": sorted({seed for seed, _n_kernels in used_seed_kernel_pairs}),
        "used_seed_kernel_pairs": [
            {"seed": seed, "n_kernels": n_kernels} for seed, n_kernels in used_seed_kernel_pairs
        ],
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are blended target-wise with deterministic random-convolution sequence "
            "models. Target/seed/kernel-count/model/weight choices are selected from listed local public-structure "
            "diagnostics, so this is a tuned public-structure probe, not nested evidence."
        ),
        "caution": "Public-structure/non-strict random-conv sequence overlay with target-wise full-diagnostic selection.",
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
            f"# Submission {experiment_id:03d}: multiseed random-conv sequence overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses deterministic ROCKET-style random convolution features over raw daily sensor sequences."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_multiseed_random_conv_sequence_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} --preset {preset}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id173_multiseed_random_conv_sequence_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="id174")
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    selected_configs, diagnostic_reports = PRESETS[args.preset]
    oof, test_pred, target_scores, selector_info, sample_original = predict_multiseed_random_conv_overlay(
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
        "method": f"Target-wise multiseed random-conv sequence overlay on ID{args.base_id:03d}",
        "preset": args.preset,
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_multiseed_random_conv_sequence",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Target-wise random-conv sequence overlay under public-structure interleaved within-subject CV.",
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
        f"Public-structure multiseed random-conv sequence overlay on ID{args.base_id:03d}; target-wise selected "
        f"with preset {args.preset}. Not strict Subject-Hole CV; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Multiseed random-conv sequence overlay on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
