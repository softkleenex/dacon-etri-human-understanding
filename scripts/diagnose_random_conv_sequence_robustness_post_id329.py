#!/usr/bin/env python3
"""Stress-test the ID178 random-conv sequence settings under stricter folds.

This report-only diagnostic retrains the selected ID178 random-conv sequence
models under alternative validation lenses. It does not create a submission,
append the submission log, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    EPS,
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    make_kernels,
    make_model,
    random_conv_features,
    table_to_tensor,
)
from scripts.diagnose_test_split_fold_alignment import _build_timeline, _test_neighbor_holdout  # noqa: E402
from scripts.run_public_multiseed_random_conv_sequence_overlay_experiment import (  # noqa: E402
    SELECTED_CONFIGS_ID178,
    model_lookup,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

KST = timezone(timedelta(hours=9))


def fold_ids_from_folds(n_rows: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    fold_ids = np.full(n_rows, -1, dtype=int)
    for fold, (_train_idx, valid_idx) in enumerate(folds):
        fold_ids[np.asarray(valid_idx, dtype=int)] = fold
    if np.any(fold_ids < 0):
        missing = int(np.sum(fold_ids < 0))
        raise ValueError(f"Folds do not cover every row exactly once; missing={missing}")
    return fold_ids


def make_loso_fold_ids(train: pd.DataFrame) -> np.ndarray:
    fold_ids = np.zeros(len(train), dtype=int)
    for fold, (_subject_id, group) in enumerate(train.groupby("subject_id", sort=True)):
        fold_ids[group.index.to_numpy(dtype=int)] = fold
    return fold_ids


def fit_oof_by_fold_ids(x_train: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, model_spec: Any) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(model_spec))
        model.fit(x_train[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x_train[valid_idx])[:, 1]
    return np.clip(oof, EPS, 1.0 - EPS)


def fit_holdout(
    x_train: np.ndarray, y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, model_spec: Any
) -> np.ndarray:
    if len(np.unique(y[train_idx])) < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = clone(make_model(model_spec))
    model.fit(x_train[train_idx], y[train_idx])
    pred = model.predict_proba(x_train[valid_idx])[:, 1]
    return np.clip(pred, EPS, 1.0 - EPS)


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y, np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def evaluate_full_fold_strategy(
    *,
    strategy_name: str,
    fold_ids: np.ndarray,
    train: pd.DataFrame,
    base_oof: pd.DataFrame,
    conv_features: dict[tuple[int, int], np.ndarray],
    specs: dict[str, Any],
) -> dict[str, Any]:
    target_rows: dict[str, Any] = {}
    base_losses: list[float] = []
    raw_losses: list[float] = []
    blend_losses: list[float] = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_loss = target_loss(y, base_values)
        config = SELECTED_CONFIGS_ID178[target]
        base_losses.append(base_loss)
        if config["kind"] == "base":
            raw_loss = None
            blend_loss = base_loss
            target_rows[target] = {
                "config": config,
                "base_loss": base_loss,
                "raw_sequence_loss": None,
                "selected_blend_loss": blend_loss,
                "blend_improvement_vs_base": 0.0,
            }
        else:
            feature_key = (int(config["seed"]), int(config.get("n_kernels", 192)))
            raw_pred = fit_oof_by_fold_ids(conv_features[feature_key], y, fold_ids, specs[str(config["model"])])
            raw_loss = target_loss(y, raw_pred)
            weight = float(config["weight"])
            blend_pred = np.clip((1.0 - weight) * base_values + weight * raw_pred, EPS, 1.0 - EPS)
            blend_loss = target_loss(y, blend_pred)
            target_rows[target] = {
                "config": config,
                "base_loss": base_loss,
                "raw_sequence_loss": raw_loss,
                "selected_blend_loss": blend_loss,
                "blend_improvement_vs_base": base_loss - blend_loss,
            }
        raw_losses.append(raw_loss if raw_loss is not None else base_loss)
        blend_losses.append(blend_loss)
    return {
        "strategy": strategy_name,
        "n_folds": int(len(np.unique(fold_ids))),
        "base_mean": float(np.mean(base_losses)),
        "raw_or_base_mean": float(np.mean(raw_losses)),
        "selected_blend_mean": float(np.mean(blend_losses)),
        "blend_improvement_vs_base_mean": float(np.mean(base_losses) - np.mean(blend_losses)),
        "targets": target_rows,
    }


def evaluate_boundary_holdout(
    *,
    train: pd.DataFrame,
    test_sample: pd.DataFrame,
    base_oof: pd.DataFrame,
    conv_features: dict[tuple[int, int], np.ndarray],
    specs: dict[str, Any],
) -> dict[str, Any]:
    timeline = _build_timeline(train, test_sample)
    holdout_folds = _test_neighbor_holdout(
        train,
        timeline,
        neighbor_positions=2,
    )
    train_idx, valid_idx = holdout_folds[0]
    target_rows: dict[str, Any] = {}
    base_losses: list[float] = []
    raw_losses: list[float] = []
    blend_losses: list[float] = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_loss = target_loss(y[valid_idx], base_values[valid_idx])
        config = SELECTED_CONFIGS_ID178[target]
        base_losses.append(base_loss)
        if config["kind"] == "base":
            raw_loss = None
            blend_loss = base_loss
            target_rows[target] = {
                "config": config,
                "base_loss": base_loss,
                "raw_sequence_loss": None,
                "selected_blend_loss": blend_loss,
                "blend_improvement_vs_base": 0.0,
            }
        else:
            feature_key = (int(config["seed"]), int(config.get("n_kernels", 192)))
            raw_valid = fit_holdout(conv_features[feature_key], y, train_idx, valid_idx, specs[str(config["model"])])
            raw_loss = target_loss(y[valid_idx], raw_valid)
            weight = float(config["weight"])
            blend_valid = np.clip((1.0 - weight) * base_values[valid_idx] + weight * raw_valid, EPS, 1.0 - EPS)
            blend_loss = target_loss(y[valid_idx], blend_valid)
            target_rows[target] = {
                "config": config,
                "base_loss": base_loss,
                "raw_sequence_loss": raw_loss,
                "selected_blend_loss": blend_loss,
                "blend_improvement_vs_base": base_loss - blend_loss,
            }
        raw_losses.append(raw_loss if raw_loss is not None else base_loss)
        blend_losses.append(blend_loss)
    return {
        "strategy": "test_boundary_neighbor_holdout",
        "n_folds": 1,
        "train_n": int(len(train_idx)),
        "valid_n": int(len(valid_idx)),
        "base_mean": float(np.mean(base_losses)),
        "raw_or_base_mean": float(np.mean(raw_losses)),
        "selected_blend_mean": float(np.mean(blend_losses)),
        "blend_improvement_vs_base_mean": float(np.mean(base_losses) - np.mean(blend_losses)),
        "targets": target_rows,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Random-conv sequence robustness diagnostic — post ID329",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV was created, no submission-log row was appended, and no DACON submission was made.",
        "",
        "## Strategy summary",
        "",
        "| strategy | folds | base mean | raw/base mean | selected blend mean | blend improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["strategies"]:
        lines.append(
            f"| {row['strategy']} | {row['n_folds']} | {row['base_mean']:.6f} | "
            f"{row['raw_or_base_mean']:.6f} | {row['selected_blend_mean']:.6f} | "
            f"{row['blend_improvement_vs_base_mean']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            report["verdict"],
            "",
            "## Interpretation",
            "",
            "- If the selected ID178 sequence overlay only helps under the public-interleaved lens but not under subject-hole/LOSO/boundary stress, it should not be expanded into another public probe.",
            "- A robust sequence branch should show target-level improvements outside the original selection lens before any new artifact is promoted.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_json: Path, base_id: int = 173) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    test_sample = pd.read_csv(
        ROOT / "data" / "raw" / "ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"]
    )
    base_oof, _base_test, base_source = load_artifact_pair(train, test, base_id)

    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, _test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    needed_feature_keys = sorted(
        {
            (int(config["seed"]), int(config.get("n_kernels", 192)))
            for config in SELECTED_CONFIGS_ID178.values()
            if config["kind"] != "base"
        }
    )
    conv_features = {}
    for seed, n_kernels in needed_feature_keys:
        kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
        conv_features[(seed, n_kernels)] = random_conv_features(train_tensor, kernels)

    subject_hole_fold_ids = fold_ids_from_folds(len(train), make_subject_hole_folds(train, n_folds=5))
    loso_fold_ids = make_loso_fold_ids(train)
    specs = model_lookup()
    strategies = [
        evaluate_full_fold_strategy(
            strategy_name="subject_hole_5",
            fold_ids=subject_hole_fold_ids,
            train=train,
            base_oof=base_oof,
            conv_features=conv_features,
            specs=specs,
        ),
        evaluate_full_fold_strategy(
            strategy_name="leave_one_subject_out_10",
            fold_ids=loso_fold_ids,
            train=train,
            base_oof=base_oof,
            conv_features=conv_features,
            specs=specs,
        ),
        evaluate_boundary_holdout(
            train=train,
            test_sample=test_sample,
            base_oof=base_oof,
            conv_features=conv_features,
            specs=specs,
        ),
    ]
    robust_positive = [
        row for row in strategies if row["strategy"] != "subject_hole_5" and row["blend_improvement_vs_base_mean"] > 0
    ]
    verdict = (
        "No submission candidate is promoted. Treat ID178 sequence expansion as hypothesis-only unless a future "
        "fold-nested branch shows consistent improvements under both LOSO and test-boundary stress."
        if len(robust_positive) < 2
        else "Sequence signal shows some robustness, but this diagnostic still does not create a Public 0.58-plausible candidate."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "random_conv_sequence_robustness_post_id329",
        "base_id": base_id,
        "base_source": base_source,
        "id178_selected_configs": SELECTED_CONFIGS_ID178,
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "feature_keys": [{"seed": seed, "n_kernels": n_kernels} for seed, n_kernels in needed_feature_keys],
        "strategies": strategies,
        "verdict": verdict,
        "public_058_plausible_candidate_found": False,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    output_json.with_suffix(".md").write_text(build_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=173)
    parser.add_argument(
        "--output-json", type=Path, default=REPORTS_DIR / "random_conv_sequence_robustness_post_id329_20260617.json"
    )
    args = parser.parse_args()
    report = run(args.output_json, base_id=args.base_id)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(
        json.dumps(
            {
                "base_id": report["base_id"],
                "strategies": [
                    {
                        "strategy": row["strategy"],
                        "base_mean": row["base_mean"],
                        "selected_blend_mean": row["selected_blend_mean"],
                        "improvement": row["blend_improvement_vs_base_mean"],
                    }
                    for row in report["strategies"]
                ],
                "public_058_plausible_candidate_found": report["public_058_plausible_candidate_found"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
