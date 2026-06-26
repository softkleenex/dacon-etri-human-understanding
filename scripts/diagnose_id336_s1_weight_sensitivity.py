#!/usr/bin/env python3
"""Report-only S1 weight sensitivity for a random-conv source over a chosen anchor."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_anchor_rerank_post_id335 import conservative_proxy, projection  # noqa: E402
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    MODEL_SPECS,
    RAW_DIR,
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
    REPORTS_DIR,
    TARGETS,
    build_feature_frames,
)

KST = timezone(timedelta(hours=9))
EPS = 1e-6
DEFAULT_BASE_ID = 178
DEFAULT_TARGET = "S1"
DEFAULT_SEED = 45502
DEFAULT_N_KERNELS = 512
DEFAULT_MODEL = "lr_c0_1"
DEFAULT_WEIGHTS = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20]
PUBLIC_ID178 = 0.6014113009
ID178_LOCAL_REFERENCE = 0.5386382667903545
ID178_PUBLIC_GAP = PUBLIC_ID178 - ID178_LOCAL_REFERENCE


def parse_weights(raw: str) -> list[float]:
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not weights:
        raise ValueError("At least one weight is required.")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Weights must be in [0, 1].")
    return weights


def model_lookup() -> dict[str, Any]:
    return {spec.name: spec for spec in MODEL_SPECS}


def find_base_test(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{candidate_id:03d}_*.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one base_test artifact for ID{candidate_id:03d}, found {paths}")
    return paths[0]


def mean_logloss(train: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    scores = {
        target: float(
            log_loss(train[target].astype(int), pred[target].astype(float).clip(EPS, 1.0 - EPS), labels=[0, 1])
        )
        for target in TARGETS
    }
    return float(np.mean(list(scores.values()))), scores


def load_test_array(candidate_id: int) -> np.ndarray:
    return pd.read_csv(find_base_test(candidate_id))[TARGETS].astype(float).to_numpy()


def run_sensitivity(
    base_id: int, target: str, seed: int, n_kernels: int, model_name: str, weights: list[float]
) -> dict:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    specs = model_lookup()
    if model_name not in specs:
        raise ValueError(f"Unknown model spec: {model_name}")

    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
    x_train = random_conv_features(train_tensor, kernels)
    x_test = random_conv_features(test_tensor, kernels)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=np.asarray(train["date"], dtype="datetime64[ns]")))

    y = train[target].astype(int).to_numpy()
    raw_oof, raw_test = fit_oof_model(x_train, x_test, y, fold_ids, specs[model_name])
    base_cv, base_target_scores = mean_logloss(train, base_oof)
    base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    raw_loss = float(log_loss(y, raw_oof, labels=[0, 1]))
    base_loss = float(log_loss(y, base_values, labels=[0, 1]))

    arr178 = load_test_array(178)
    arr287 = load_test_array(287)
    arr329 = load_test_array(329)
    arr335 = load_test_array(335)
    failed_direction = (arr287 - arr178).reshape(-1)
    neutral_direction = (arr329 - arr178).reshape(-1)
    current_direction = (arr335 - arr178).reshape(-1)

    rows = []
    for weight in weights:
        pred_oof = base_oof.copy()
        pred_test = base_test.copy()
        selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
        selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        pred_oof[target] = selected_oof
        pred_test[target] = selected_test
        local_cv, target_scores = mean_logloss(train, pred_oof)
        arr = pred_test[TARGETS].astype(float).to_numpy()
        vec178 = (arr - arr178).reshape(-1)
        row = {
            "weight": weight,
            "local_cv": local_cv,
            "local_gain_vs_base": base_cv - local_cv,
            "target_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "target_improvement_vs_base": base_loss - float(log_loss(y, selected_oof, labels=[0, 1])),
            "raw_sequence_loss": raw_loss,
            "target_scores": target_scores,
            "test_s1_mean_before": float(np.mean(base_test_values)),
            "test_s1_mean_after": float(np.mean(selected_test)),
            "test_s1_max_delta_vs_base": float(np.max(np.abs(selected_test - base_test_values))),
            "test_s1_mean_delta_vs_base": float(np.mean(np.abs(selected_test - base_test_values))),
            "mean_delta_vs_178": float(np.mean(np.abs(arr - arr178))),
            "max_delta_vs_178": float(np.max(np.abs(arr - arr178))),
            "rmse_vs_178": float(np.sqrt(np.mean((arr - arr178) ** 2))),
            "mean_delta_vs_329": float(np.mean(np.abs(arr - arr329))),
            "max_delta_vs_329": float(np.max(np.abs(arr - arr329))),
            "mean_delta_vs_335": float(np.mean(np.abs(arr - arr335))),
            "max_delta_vs_335": float(np.max(np.abs(arr - arr335))),
            "projection_failed": projection(vec178, failed_direction),
            "projection_329": projection(vec178, neutral_direction),
            "projection_335": projection(vec178, current_direction),
            "naive_id178_gap_projected_public": float(local_cv + ID178_PUBLIC_GAP),
        }
        row["conservative_public_risk_proxy"] = conservative_proxy({**row, "local_cv": local_cv})
        row["reviewable_low_movement_gate"] = bool(
            row["max_delta_vs_178"] <= 0.12 and row["max_delta_vs_329"] <= 0.17 and row["projection_failed"] <= 0.02
        )
        rows.append(row)

    rows_by_local = sorted(rows, key=lambda item: (item["local_cv"], item["max_delta_vs_178"]))
    rows_by_proxy = sorted(rows, key=lambda item: (item["conservative_public_risk_proxy"], item["local_cv"]))
    rows_by_balanced = sorted(
        rows,
        key=lambda item: (
            item["conservative_public_risk_proxy"] + 0.35 * max(0.0, item["local_cv"] - rows_by_local[0]["local_cv"]),
            item["max_delta_vs_178"],
        ),
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": f"id{base_id:03d}_{target.lower()}_weight_sensitivity",
        "base_id": base_id,
        "base_source": base_source,
        "target": target,
        "seed": seed,
        "n_kernels": n_kernels,
        "model": model_name,
        "weights": weights,
        "base_cv": base_cv,
        "base_target_scores": base_target_scores,
        "base_target_loss": base_loss,
        "raw_sequence_loss": raw_loss,
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_random_conv_features": int(x_train.shape[1]),
        "rows": rows,
        "best_by_local": rows_by_local[0],
        "best_by_proxy": rows_by_proxy[0],
        "balanced_choice": rows_by_balanced[0],
        "reference_weight": 0.15,
        "no_submission_created": True,
        "recommendation": {
            "do_not_auto_submit": True,
            "summary": (
                "Choose the local-best weight only if local gain dominates; choose the proxy/balanced weight if public-anchor "
                "movement risk dominates. This diagnostic is not public validation."
            ),
        },
    }
    return report


def write_markdown(report: dict) -> str:
    lines = [
        f"# ID{report['base_id']:03d} {report['target']} weight sensitivity",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        f"Base ID: `{report['base_id']}`; target: `{report['target']}`; seed/kernels/model: "
        f"`{report['seed']}` / `{report['n_kernels']}` / `{report['model']}`.",
        "",
        "## Ranked by local CV",
        "",
        "| weight | local CV | local gain | S1 gain | mean Δ178 | max Δ178 | failed proj | proxy | naive public |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["rows"], key=lambda item: item["local_cv"]):
        lines.append(
            f"| {row['weight']:.2f} | {row['local_cv']:.6f} | {row['local_gain_vs_base']:.6f} | "
            f"{row['target_improvement_vs_base']:.6f} | {row['mean_delta_vs_178']:.6f} | "
            f"{row['max_delta_vs_178']:.6f} | {row['projection_failed']:+.6f} | "
            f"{row['conservative_public_risk_proxy']:.6f} | {row['naive_id178_gap_projected_public']:.6f} |"
        )
    best = report["best_by_local"]
    proxy = report["best_by_proxy"]
    balanced = report["balanced_choice"]
    lines.extend(
        [
            "",
            "## Picks",
            "",
            f"- Best local: weight `{best['weight']:.2f}`, local `{best['local_cv']:.6f}`.",
            f"- Best proxy: weight `{proxy['weight']:.2f}`, proxy `{proxy['conservative_public_risk_proxy']:.6f}`.",
            f"- Balanced choice: weight `{balanced['weight']:.2f}`, local `{balanced['local_cv']:.6f}`, "
            f"proxy `{balanced['conservative_public_risk_proxy']:.6f}`.",
            "",
            "## Decision",
            "",
            "This is report-only. It does not create a DACON submission or new artifact ID. Any promoted candidate from this sweep must use a new immutable artifact ID unless that exact artifact already exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-kernels", type=int, default=DEFAULT_N_KERNELS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weights", default=",".join(f"{weight:g}" for weight in DEFAULT_WEIGHTS))
    parser.add_argument("--output-json", type=Path, default=REPORTS_DIR / "id336_s1_weight_sensitivity_20260618.json")
    args = parser.parse_args()
    report = run_sensitivity(
        args.base_id, args.target, args.seed, args.n_kernels, args.model, parse_weights(args.weights)
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_json.with_suffix(".md").write_text(write_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_by_local": report["best_by_local"],
                "best_by_proxy": report["best_by_proxy"],
                "balanced_choice": report["balanced_choice"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
