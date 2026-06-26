#!/usr/bin/env python3
"""Fine-scan target-wise random-conv residual blends.

This is report-only. It reuses the strongest ID173 random-conv diagnostic pool,
then scans a denser LR regularization/weight neighborhood around the best
target-wise residuals. The purpose is to see whether the ID177 branch still has
local headroom before launching another broad Kaggle seed sweep.
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
    EPS,
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
    make_kernels,
    random_conv_features,
    table_to_tensor,
)
from scripts.run_public_multiseed_random_conv_sequence_overlay_experiment import (  # noqa: E402
    SELECTED_CONFIGS_ID177,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402

FINE_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_005", "lr", c_value=0.005),
    ModelSpec("lr_c0_01", "lr", c_value=0.01),
    ModelSpec("lr_c0_02", "lr", c_value=0.02),
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_05", "lr", c_value=0.05),
    ModelSpec("lr_c0_07", "lr", c_value=0.07),
    ModelSpec("lr_c0_1", "lr", c_value=0.1),
    ModelSpec("lr_c0_15", "lr", c_value=0.15),
    ModelSpec("lr_c0_2", "lr", c_value=0.2),
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("lr_c0_5", "lr", c_value=0.5),
    ModelSpec("lr_c0_7", "lr", c_value=0.7),
    ModelSpec("lr_c1", "lr", c_value=1.0),
    ModelSpec("lr_c1_5", "lr", c_value=1.5),
    ModelSpec("lr_c2", "lr", c_value=2.0),
    ModelSpec("lr_c3", "lr", c_value=3.0),
    ModelSpec("lr_c5", "lr", c_value=5.0),
    ModelSpec("lr_c7", "lr", c_value=7.0),
    ModelSpec("lr_c10", "lr", c_value=10.0),
    ModelSpec("lr_c20", "lr", c_value=20.0),
    ModelSpec("lr_c50", "lr", c_value=50.0),
    ModelSpec("lr_c100", "lr", c_value=100.0),
)


def load_report_candidates(base_id: int, max_pairs_per_target: int) -> dict[str, list[dict[str, Any]]]:
    """Load top seed/kernel pairs per target from existing random-conv reports."""

    per_target: dict[str, dict[tuple[int, int], dict[str, Any]]] = {target: {} for target in TARGETS}
    for path in sorted(REPORTS_DIR.glob(f"id{base_id}_random_conv_sequence_diagnostic*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("diagnostic") != "public_random_conv_sequence" or int(report.get("base_id", -1)) != base_id:
            continue
        seed = int(report["seed"])
        n_kernels = int(report["n_kernels"])
        for target in TARGETS:
            rows = report["targets"][target].get("top10", [])
            best_blend_loss = None
            best_blend_name = None
            for name, candidate in rows:
                if candidate.get("kind") != "blend":
                    continue
                loss = float(candidate["loss"])
                if best_blend_loss is None or loss < best_blend_loss:
                    best_blend_loss = loss
                    best_blend_name = name
            if best_blend_loss is None:
                continue
            key = (seed, n_kernels)
            existing = per_target[target].get(key)
            if existing is None or best_blend_loss < float(existing["loss"]):
                per_target[target][key] = {
                    "seed": seed,
                    "n_kernels": n_kernels,
                    "loss": best_blend_loss,
                    "source_best": best_blend_name,
                    "source_report": str(path.relative_to(ROOT)),
                }

    for target, config in SELECTED_CONFIGS_ID177.items():
        if config["kind"] == "base":
            continue
        key = (int(config["seed"]), int(config.get("n_kernels", 192)))
        per_target[target].setdefault(
            key,
            {
                "seed": key[0],
                "n_kernels": key[1],
                "loss": float("inf"),
                "source_best": "id177_selected",
                "source_report": "reports/experiment_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.json",
            },
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    for target, candidates in per_target.items():
        rows = sorted(
            candidates.values(), key=lambda row: (float(row["loss"]), int(row["seed"]), int(row["n_kernels"]))
        )
        selected[target] = rows[:max_pairs_per_target]
    return selected


def run_diagnostic(
    base_id: int,
    max_pairs_per_target: int,
    max_weight: float,
    weight_step: float,
    output_path: Path,
) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, _base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    target_pairs = load_report_candidates(base_id, max_pairs_per_target)
    weights = [float(round(value, 4)) for value in np.arange(0.0, max_weight + weight_step / 2.0, weight_step)]

    feature_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report: dict[str, Any] = {
        "diagnostic": "public_random_conv_fine_selection",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "model_specs": [asdict(spec) for spec in FINE_MODEL_SPECS],
        "weights": weights,
        "max_pairs_per_target": max_pairs_per_target,
        "base_scores": base_scores,
        "base_mean": float(np.mean(list(base_scores.values()))),
        "targets": {},
    }

    target_best_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidates: dict[str, dict[str, Any]] = {
            "base": {"kind": "base", "loss": base_scores[target]},
        }
        if target != "S3":
            for pair in target_pairs[target]:
                seed = int(pair["seed"])
                n_kernels = int(pair["n_kernels"])
                feature_key = (seed, n_kernels)
                if feature_key not in feature_cache:
                    kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
                    feature_cache[feature_key] = (
                        random_conv_features(train_tensor, kernels),
                        random_conv_features(test_tensor, kernels),
                    )
                x_train, x_test = feature_cache[feature_key]
                for spec in FINE_MODEL_SPECS:
                    raw_oof, _raw_test = fit_oof_model(x_train, x_test, y, fold_ids, spec)
                    raw_loss = float(log_loss(y, raw_oof, labels=[0, 1]))
                    candidates[f"raw_seed{seed}_nk{n_kernels}_{spec.name}"] = {
                        "kind": "raw",
                        "seed": seed,
                        "n_kernels": n_kernels,
                        "model": spec.name,
                        "loss": raw_loss,
                        "source_pair": pair,
                    }
                    for weight in weights:
                        pred = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                        candidates[f"blend_seed{seed}_nk{n_kernels}_{spec.name}_{weight:g}"] = {
                            "kind": "blend",
                            "seed": seed,
                            "n_kernels": n_kernels,
                            "model": spec.name,
                            "weight": weight,
                            "loss": float(log_loss(y, pred, labels=[0, 1])),
                            "source_pair": pair,
                        }
        best_name, best_info = min(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "searched_pairs": target_pairs[target],
            "best": best_name,
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top20": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:20],
        }

    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    report["selected_configs"] = {}
    for target in TARGETS:
        best_name = report["targets"][target]["best"]
        best_info = report["targets"][target]["top20"][0][1]
        if best_name == "base" or best_info["kind"] == "base":
            report["selected_configs"][target] = {"kind": "base"}
        else:
            report["selected_configs"][target] = {
                "kind": "blend",
                "seed": int(best_info["seed"]),
                "n_kernels": int(best_info["n_kernels"]),
                "model": str(best_info["model"]),
                "weight": float(best_info["weight"]),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=173)
    parser.add_argument("--max-pairs-per-target", type=int, default=6)
    parser.add_argument("--max-weight", type=float, default=0.40)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id173_random_conv_fine_selection_diagnostic.json",
    )
    args = parser.parse_args()
    report = run_diagnostic(
        base_id=args.base_id,
        max_pairs_per_target=args.max_pairs_per_target,
        max_weight=args.max_weight,
        weight_step=args.weight_step,
        output_path=args.output,
    )
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "selected_configs": report["selected_configs"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
