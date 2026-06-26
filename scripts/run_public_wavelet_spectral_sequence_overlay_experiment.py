#!/usr/bin/env python3
"""Diagnose/promote wavelet/spectral sequence overlays.

This deterministic representation summarizes each aligned 44-bin daily sensor
sequence with multiscale Haar-like contrasts, lagged roughness, detailed FFT
shares, and window-level volatility. It is deliberately anchor-preserving: the
promotion path selects target-wise light blends into an existing public-structure
anchor rather than replacing the anchor with high-variance raw side models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    EPS,
    MODEL_SPECS,
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
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

DEFAULT_BASE_ID = 203
BLEND_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]
EXTRA_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("hgb_l15", "hgb", max_leaf_nodes=15),
)
WAVELET_SCALES = [1, 2, 4, 8, 11]
DIFF_LAGS = [1, 2, 3, 4, 6, 8, 12]
WINDOWS: dict[str, tuple[int, int]] = {
    "aft": (0, 12),
    "eve": (12, 18),
    "pre": (18, 24),
    "late": (24, 30),
    "deep": (30, 36),
    "wake": (36, 42),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def add_channel_block(
    blocks: list[np.ndarray],
    names: list[str],
    block: np.ndarray,
    channel_names: list[str],
    suffix: str,
) -> None:
    blocks.append(np.asarray(block, dtype=np.float32))
    names.extend(f"{channel}__{suffix}" for channel in channel_names)


def wavelet_spectral_features(tensor: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(tensor, dtype=np.float32)
    n_rows, n_channels, n_bins = values.shape
    centered = values - values.mean(axis=2, keepdims=True)
    blocks: list[np.ndarray] = []
    names: list[str] = []

    for scale in WAVELET_SCALES:
        width = scale * 2
        n_segments = n_bins // width
        if n_segments < 1:
            continue
        trimmed = values[:, :, : n_segments * width].reshape(n_rows, n_channels, n_segments, width)
        left = trimmed[:, :, :, :scale].mean(axis=3)
        right = trimmed[:, :, :, scale:].mean(axis=3)
        detail = left - right
        add_channel_block(blocks, names, detail.mean(axis=2), channel_names, f"haar_s{scale}_mean")
        add_channel_block(blocks, names, detail.std(axis=2), channel_names, f"haar_s{scale}_std")
        add_channel_block(blocks, names, np.abs(detail).mean(axis=2), channel_names, f"haar_s{scale}_abs_mean")
        add_channel_block(blocks, names, (detail**2).mean(axis=2), channel_names, f"haar_s{scale}_energy")
        add_channel_block(blocks, names, np.max(np.abs(detail), axis=2), channel_names, f"haar_s{scale}_maxabs")

    for lag in DIFF_LAGS:
        diff = values[:, :, lag:] - values[:, :, :-lag]
        add_channel_block(blocks, names, diff.mean(axis=2), channel_names, f"diff_lag{lag}_mean")
        add_channel_block(blocks, names, diff.std(axis=2), channel_names, f"diff_lag{lag}_std")
        add_channel_block(blocks, names, np.abs(diff).mean(axis=2), channel_names, f"diff_lag{lag}_abs_mean")
        add_channel_block(blocks, names, (diff**2).mean(axis=2), channel_names, f"diff_lag{lag}_energy")

    fft_power = (np.abs(np.fft.rfft(centered, axis=2)) ** 2).astype(np.float32)
    nonzero_power = fft_power[:, :, 1:]
    total_power = nonzero_power.sum(axis=2) + 1e-8
    freqs = np.arange(1, nonzero_power.shape[2] + 1, dtype=np.float32)
    add_channel_block(blocks, names, total_power, channel_names, "fft_total_power")
    add_channel_block(
        blocks,
        names,
        (nonzero_power * freqs.reshape(1, 1, -1)).sum(axis=2) / total_power,
        channel_names,
        "fft_centroid",
    )
    probs = nonzero_power / total_power[:, :, None]
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=2) / np.log(max(nonzero_power.shape[2], 2))
    add_channel_block(blocks, names, entropy, channel_names, "fft_entropy")
    for coef_idx in range(1, min(12, fft_power.shape[2])):
        add_channel_block(
            blocks,
            names,
            fft_power[:, :, coef_idx] / total_power,
            channel_names,
            f"fft_coef{coef_idx}_share",
        )
    bands = {
        "b1_2": (1, 3),
        "b3_5": (3, 6),
        "b6_10": (6, 11),
        "b11_plus": (11, fft_power.shape[2]),
    }
    for band_name, (start, end) in bands.items():
        if start < fft_power.shape[2]:
            add_channel_block(
                blocks,
                names,
                fft_power[:, :, start:end].sum(axis=2) / total_power,
                channel_names,
                f"fft_{band_name}_share",
            )

    for window_name, (start, end) in WINDOWS.items():
        window_values = values[:, :, start:end]
        if end - start > 1:
            window_diff = np.diff(window_values, axis=2)
        else:
            window_diff = np.zeros((n_rows, n_channels, 1), dtype=np.float32)
        add_channel_block(blocks, names, window_values.std(axis=2), channel_names, f"{window_name}_std")
        add_channel_block(
            blocks,
            names,
            np.abs(window_diff).mean(axis=2),
            channel_names,
            f"{window_name}_rough_abs",
        )

    features = np.concatenate([block.reshape(n_rows, n_channels) for block in blocks], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if features.shape[1] != len(names):
        raise ValueError(f"Feature/name mismatch: {features.shape[1]} vs {len(names)}")
    return features, names


def build_wavelet_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]
]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = wavelet_spectral_features(train_tensor, channel_names)
    x_test, _ = wavelet_spectral_features(test_tensor, channel_names)
    info = {
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_bins": int(train_tensor.shape[2]),
        "n_wavelet_spectral_features": int(x_train.shape[1]),
        "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        "feature_name_sample": feature_names[:40],
        "representation": "Haar-like multiscale contrasts, lagged roughness, detailed FFT shares, and window volatility over the aligned raw tensor.",
    }
    return train, test, sample_original, x_train, x_test, channel_names, info


def model_specs() -> tuple[ModelSpec, ...]:
    return (*MODEL_SPECS, *EXTRA_MODEL_SPECS)


def compute_model_predictions(
    train: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    fold_ids: np.ndarray,
    needed_models: set[str] | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    spec_lookup = {spec.name: spec for spec in model_specs()}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    specs_used = []
    for model_name, spec in spec_lookup.items():
        if needed_models is not None and model_name not in needed_models:
            continue
        specs_used.append(asdict(spec))
        predictions[model_name] = {}
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            oof_pred, test_pred = fit_oof_model(x_train, x_test, y, fold_ids, spec)
            predictions[model_name][target] = {"oof": oof_pred, "test": test_pred}
    return predictions, specs_used


def candidate_key(model_name: str, weight: float) -> str:
    return f"blend__wavelet_spectral__{model_name}__w{weight:g}"


def parse_candidate_key(key: str) -> dict[str, Any]:
    if key == "base":
        return {"kind": "base"}
    if key.startswith("raw__"):
        return {"kind": "raw", "feature_set": "wavelet_spectral", "model": key.split("__", 1)[1]}
    kind, feature_set, model_name, weight_raw = key.split("__", 3)
    if kind != "blend" or feature_set != "wavelet_spectral" or not weight_raw.startswith("w"):
        raise ValueError(f"Unexpected candidate key: {key}")
    return {"kind": "blend", "feature_set": feature_set, "model": model_name, "weight": float(weight_raw[1:])}


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train, test, _sample_original, x_train, x_test, _channel_names, feature_info = build_wavelet_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report: dict[str, Any] = {
        "diagnostic": "public_wavelet_spectral_sequence_overlay",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "feature_info": feature_info,
        "model_specs": specs_used,
        "blend_weights": BLEND_WEIGHTS,
        "base_scores": base_scores,
        "base_mean": float(np.mean(list(base_scores.values()))),
        "targets": {},
    }
    target_best_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidates: dict[str, dict[str, Any]] = {
            "base": {"kind": "base", "loss": base_scores[target], "test_mean": float(np.mean(base_test_values))}
        }
        for model_name, by_target in predictions.items():
            raw_oof = by_target[target]["oof"]
            raw_test = by_target[target]["test"]
            candidates[f"raw__{model_name}"] = {
                "kind": "raw",
                "feature_set": "wavelet_spectral",
                "model": model_name,
                "loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                "test_mean": float(np.mean(raw_test)),
            }
            for weight in BLEND_WEIGHTS:
                pred = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                candidates[candidate_key(model_name, weight)] = {
                    "kind": "blend",
                    "feature_set": "wavelet_spectral",
                    "model": model_name,
                    "weight": weight,
                    "loss": float(log_loss(y, pred, labels=[0, 1])),
                    "test_mean": float(np.mean(blended_test)),
                    "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                }
        selectable = {name: info for name, info in candidates.items() if name == "base" or info.get("kind") == "blend"}
        best_name, best_info = min(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "best": best_name,
            "best_config": parse_candidate_key(best_name),
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top10_selectable": sorted(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
            "top10_all": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
        }
    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict_from_report(
    base_id: int, report_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if int(report["base_id"]) != base_id:
        raise ValueError(f"Report base_id={report['base_id']} does not match requested base_id={base_id}.")
    train, test, sample_original, x_train, x_test, _channel_names, feature_info = build_wavelet_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    selected_configs = {target: report["targets"][target]["best_config"] for target in TARGETS}
    needed_models = {config["model"] for config in selected_configs.values() if config.get("kind") in {"blend", "raw"}}
    predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids, needed_models)
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
        else:
            side = predictions[config["model"]][target]
            raw_oof = side["oof"]
            raw_test = side["test"]
            if config["kind"] == "raw":
                selected_oof = raw_oof
                selected_test = raw_test
            else:
                weight = float(config["weight"])
                selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
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
        "diagnostic_report": str(relpath(report_path)),
        "selected_configs": selected_configs,
        "modified_targets": [target for target, config in selected_configs.items() if config["kind"] != "base"],
        "unchanged_targets": [target for target, config in selected_configs.items() if config["kind"] == "base"],
        "feature_info": feature_info,
        "model_specs_used": specs_used,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are lightly blended target-wise with deterministic wavelet/spectral "
            "sequence models selected from the listed local public-structure diagnostic. Raw side models are diagnostic-only."
        ),
        "caution": "Public-structure/non-strict wavelet/spectral sequence overlay; not strict Subject-Hole evidence.",
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
    diagnostic_report: Path,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: wavelet/spectral sequence overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses deterministic Haar-like, lagged-roughness, FFT, and window-volatility summaries."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_wavelet_spectral_sequence_overlay_experiment.py "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--diagnostic-report {relpath(diagnostic_report)}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def promote_from_report(experiment_id: int, slug: str, base_id: int, diagnostic_report: Path) -> dict[str, Any]:
    diagnostic_report = diagnostic_report.resolve()
    if experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    oof, test_pred, target_scores, selector_info, sample_original = predict_from_report(base_id, diagnostic_report)
    local_cv = float(np.mean(list(target_scores.values())))
    status = "unsubmitted"
    filename = f"submission_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name
    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {relpath(path)}")
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
        "experiment_id": experiment_id,
        "slug": slug,
        "method": f"Target-wise wavelet/spectral sequence overlay on ID{base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_wavelet_spectral_sequence",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise wavelet/spectral sequence overlay under public-structure interleaved CV.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id, diagnostic_report
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Public-structure wavelet/spectral sequence overlay on ID{base_id:03d}; target-wise selected from "
            f"{relpath(diagnostic_report)}. Raw side models were diagnostic-only; not strict Subject-Hole CV; "
            "no DACON submission was made."
        ),
        model_description=f"wavelet/spectral sequence overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id203_wavelet_spectral_sequence_diagnostic.json",
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="id203_wavelet_spectral_sequence_overlay")
    parser.add_argument("--diagnostic-report", type=Path)
    args = parser.parse_args()
    if args.promote:
        if args.experiment_id is None or args.diagnostic_report is None:
            raise ValueError("--promote requires --experiment-id and --diagnostic-report.")
        metadata = promote_from_report(args.experiment_id, args.slug, args.base_id, args.diagnostic_report)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    report = run_diagnostic(args.base_id, args.output)
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "feature_info": report["feature_info"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
