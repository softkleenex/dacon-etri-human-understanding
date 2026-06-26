#!/usr/bin/env python3
"""Diagnose deterministic sequence-summary embeddings over daily sensor tensors.

This is a lightweight alternative to the random-convolution sequence diagnostics.
It reuses the same aligned 44-bin daily raw-sensor tensor, then extracts explicit
time-shape summaries: window means, slopes, difference statistics, autocorrelation,
and coarse FFT energy. The goal is to find a new signal family without adding a
heavy local memory footprint.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    BLEND_WEIGHTS,
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
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402

EXTRA_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("hgb_l15", "hgb", max_leaf_nodes=15),
)
WINDOWS: dict[str, tuple[int, int]] = {
    "afternoon": (0, 12),
    "evening": (12, 18),
    "prebed": (18, 24),
    "late": (24, 30),
    "deep": (30, 36),
    "wake": (36, 42),
    "tail": (42, 44),
}
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("prebed", "late"),
    ("late", "deep"),
    ("deep", "wake"),
    ("evening", "prebed"),
)


def safe_corr_lag(values: np.ndarray, lag: int) -> np.ndarray:
    left = values[:, :, :-lag]
    right = values[:, :, lag:]
    left_centered = left - left.mean(axis=2, keepdims=True)
    right_centered = right - right.mean(axis=2, keepdims=True)
    denom = np.sqrt((left_centered**2).sum(axis=2) * (right_centered**2).sum(axis=2))
    return np.divide(
        (left_centered * right_centered).sum(axis=2),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 1e-8,
    )


def sequence_summary_features(tensor: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(tensor, dtype=float)
    n_rows, n_channels, n_bins = values.shape
    time = np.linspace(-1.0, 1.0, n_bins)
    time_norm = float(np.sum(time**2))
    centered = values - values.mean(axis=2, keepdims=True)
    diff = np.diff(values, axis=2)

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    def add_block(block: np.ndarray, suffix: str) -> None:
        feature_blocks.append(np.asarray(block, dtype=np.float32))
        feature_names.extend([f"{name}__{suffix}" for name in channel_names])

    add_block(values.mean(axis=2), "mean")
    add_block(values.std(axis=2), "std")
    add_block(values.min(axis=2), "min")
    add_block(values.max(axis=2), "max")
    add_block(np.quantile(values, 0.25, axis=2), "q25")
    add_block(np.quantile(values, 0.75, axis=2), "q75")
    add_block(values[:, :, 0], "first")
    add_block(values[:, :, -1], "last")
    add_block((centered * time.reshape(1, 1, -1)).sum(axis=2) / max(time_norm, 1e-8), "slope")
    add_block(diff.mean(axis=2), "diff_mean")
    add_block(diff.std(axis=2), "diff_std")
    add_block(np.abs(diff).mean(axis=2), "diff_abs_mean")
    add_block(np.argmax(values, axis=2) / max(n_bins - 1, 1), "argmax_norm")
    add_block(np.argmin(values, axis=2) / max(n_bins - 1, 1), "argmin_norm")

    window_means: dict[str, np.ndarray] = {}
    for window_name, (start, end) in WINDOWS.items():
        window_means[window_name] = values[:, :, start:end].mean(axis=2)
        add_block(window_means[window_name], f"{window_name}_mean")
    for first, second in CONTRASTS:
        add_block(window_means[first] - window_means[second], f"{first}_minus_{second}")

    for lag in [1, 2, 4]:
        add_block(safe_corr_lag(values, lag), f"autocorr_lag{lag}")

    fft_power = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    total_power = fft_power[:, :, 1:].sum(axis=2) + 1e-8
    bands = {
        "fft_low": (1, 3),
        "fft_mid": (3, 8),
        "fft_high": (8, fft_power.shape[2]),
    }
    for band_name, (start, end) in bands.items():
        add_block(fft_power[:, :, start:end].sum(axis=2) / total_power, band_name)

    features = np.concatenate([block.reshape(n_rows, n_channels) for block in feature_blocks], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return features, feature_names


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    model_specs = (*MODEL_SPECS, *EXTRA_MODEL_SPECS)

    report: dict[str, Any] = {
        "diagnostic": "public_sequence_summary_embedding",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_summary_features": int(x_train.shape[1]),
        "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        "model_specs": [asdict(spec) for spec in model_specs],
        "blend_weights": BLEND_WEIGHTS,
        "targets": {},
    }
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report["base_scores"] = base_scores
    report["base_mean"] = float(np.mean(list(base_scores.values())))

    model_oof: dict[str, pd.DataFrame] = {
        spec.name: pd.DataFrame(index=train.index, columns=TARGETS, dtype=float) for spec in model_specs
    }
    model_test: dict[str, pd.DataFrame] = {
        spec.name: pd.DataFrame(index=test.index, columns=TARGETS, dtype=float) for spec in model_specs
    }
    for spec in model_specs:
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            oof_pred, test_pred = fit_oof_model(x_train, x_test, y, fold_ids, spec)
            model_oof[spec.name][target] = oof_pred
            model_test[spec.name][target] = test_pred

    target_best_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = base_oof[target].astype(float).to_numpy()
        base_test_values = base_test[target].astype(float).to_numpy()
        candidates: dict[str, dict[str, Any]] = {
            "base": {"kind": "base", "loss": base_scores[target], "test_mean": float(base_test[target].mean())}
        }
        for spec in model_specs:
            raw = model_oof[spec.name][target].astype(float).to_numpy()
            raw_test = model_test[spec.name][target].astype(float).to_numpy()
            candidates[f"raw_{spec.name}"] = {
                "kind": "raw",
                "model": spec.name,
                "loss": float(log_loss(y, raw, labels=[0, 1])),
                "test_mean": float(np.mean(raw_test)),
            }
            for weight in BLEND_WEIGHTS:
                pred = np.clip((1.0 - weight) * base_values + weight * raw, EPS, 1.0 - EPS)
                blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                candidates[f"blend_{spec.name}_{weight:g}"] = {
                    "kind": "blend",
                    "model": spec.name,
                    "weight": weight,
                    "loss": float(log_loss(y, pred, labels=[0, 1])),
                    "test_mean": float(np.mean(blended_test)),
                    "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                }
        best_name, best_info = min(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "best": best_name,
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top10": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
        }
    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=191)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id191_sequence_summary_embedding_diagnostic.json",
    )
    args = parser.parse_args()
    report = run_diagnostic(args.base_id, args.output)
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "n_summary_features": report["n_summary_features"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
