#!/usr/bin/env python3
"""Report-only alpha scan around the ID178 random-conv sequence overlay.

This diagnostic asks a narrow question: if ID178's sequence overlay signal is
real but partially over-trusted, is a target-wise attenuation more robust than
the binary keep/revert rollback promoted as ID334?

It creates only JSON/Markdown reports. It does not create submission CSVs,
notebooks, base prediction artifacts, submission-log rows, or DACON
submissions.
"""

from __future__ import annotations

import argparse
import itertools
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
EPS = 1e-6
SEQUENCE_TARGETS = [target for target in TARGETS if SELECTED_CONFIGS_ID178[target]["kind"] != "base"]


def fold_ids_from_folds(n_rows: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    fold_ids = np.full(n_rows, -1, dtype=int)
    for fold, (_train_idx, valid_idx) in enumerate(folds):
        fold_ids[np.asarray(valid_idx, dtype=int)] = fold
    if np.any(fold_ids < 0):
        raise ValueError("Folds do not cover every row exactly once.")
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
    return np.clip(model.predict_proba(x_train[valid_idx])[:, 1], EPS, 1.0 - EPS)


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def blend_at_alpha(base_values: np.ndarray, full_overlay_values: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(base_values + alpha * (full_overlay_values - base_values), EPS, 1.0 - EPS)


def sequence_blend_at_alpha(
    base_values: np.ndarray, raw_values: np.ndarray, selected_weight: float, alpha: float
) -> np.ndarray:
    return np.clip(base_values + alpha * selected_weight * (raw_values - base_values), EPS, 1.0 - EPS)


def build_conv_features(train: pd.DataFrame, test: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, _test_tensor, _channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
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
    return conv_features


def stress_prediction_tables(
    train: pd.DataFrame,
    test_sample: pd.DataFrame,
    base_oof: pd.DataFrame,
    conv_features: dict[tuple[int, int], np.ndarray],
) -> dict[str, dict[str, Any]]:
    specs = model_lookup()
    subject_hole_fold_ids = fold_ids_from_folds(len(train), make_subject_hole_folds(train, n_folds=5))
    loso_fold_ids = make_loso_fold_ids(train)
    timeline = _build_timeline(train, test_sample)
    boundary_train_idx, boundary_valid_idx = _test_neighbor_holdout(train, timeline, neighbor_positions=2)[0]

    strategies: dict[str, dict[str, Any]] = {
        "subject_hole_5": {"mode": "full", "fold_ids": subject_hole_fold_ids, "targets": {}},
        "leave_one_subject_out_10": {"mode": "full", "fold_ids": loso_fold_ids, "targets": {}},
        "test_boundary_neighbor_holdout": {
            "mode": "holdout",
            "train_idx": boundary_train_idx,
            "valid_idx": boundary_valid_idx,
            "targets": {},
        },
    }

    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        config = SELECTED_CONFIGS_ID178[target]
        for _strategy_name, strategy in strategies.items():
            if strategy["mode"] == "full":
                valid_idx = np.arange(len(train))
                strategy_base = base_values
                strategy_y = y
                if config["kind"] == "base":
                    raw_values = base_values
                    selected_weight = 0.0
                else:
                    feature_key = (int(config["seed"]), int(config.get("n_kernels", 192)))
                    raw_values = fit_oof_by_fold_ids(
                        conv_features[feature_key], y, strategy["fold_ids"], specs[str(config["model"])]
                    )
                    selected_weight = float(config["weight"])
            else:
                valid_idx = np.asarray(strategy["valid_idx"], dtype=int)
                strategy_base = base_values[valid_idx]
                strategy_y = y[valid_idx]
                if config["kind"] == "base":
                    raw_values = strategy_base
                    selected_weight = 0.0
                else:
                    feature_key = (int(config["seed"]), int(config.get("n_kernels", 192)))
                    raw_values = fit_holdout(
                        conv_features[feature_key],
                        y,
                        np.asarray(strategy["train_idx"], dtype=int),
                        valid_idx,
                        specs[str(config["model"])],
                    )
                    selected_weight = float(config["weight"])
            strategy["targets"][target] = {
                "y": strategy_y,
                "base": strategy_base,
                "raw": raw_values,
                "selected_weight": selected_weight,
                "valid_n": int(len(valid_idx)),
            }
    return strategies


def precompute_losses(
    train: pd.DataFrame,
    base_oof: pd.DataFrame,
    id178_oof: pd.DataFrame,
    strategies: dict[str, dict[str, Any]],
    alphas: list[float],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]], dict[str, float], dict[str, float]]:
    local_losses: dict[str, dict[str, float]] = {}
    base_local_losses: dict[str, float] = {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        full_values = np.clip(id178_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_local_losses[target] = target_loss(y, base_values)
        local_losses[target] = {
            f"{alpha:.2f}": target_loss(y, blend_at_alpha(base_values, full_values, alpha)) for alpha in alphas
        }

    stress_losses: dict[str, dict[str, dict[str, float]]] = {}
    stress_base_means: dict[str, float] = {}
    for strategy_name, strategy in strategies.items():
        stress_losses[strategy_name] = {}
        base_losses = []
        for target in TARGETS:
            info = strategy["targets"][target]
            base_values = np.asarray(info["base"], dtype=float)
            raw_values = np.asarray(info["raw"], dtype=float)
            y = np.asarray(info["y"], dtype=int)
            selected_weight = float(info["selected_weight"])
            base_loss = target_loss(y, base_values)
            base_losses.append(base_loss)
            stress_losses[strategy_name][target] = {
                f"{alpha:.2f}": target_loss(y, sequence_blend_at_alpha(base_values, raw_values, selected_weight, alpha))
                for alpha in alphas
            }
        stress_base_means[strategy_name] = float(np.mean(base_losses))
    return local_losses, stress_losses, base_local_losses, stress_base_means


def geometry_for_alphas(
    alpha_by_target: dict[str, float],
    base_test: pd.DataFrame,
    id178_test: pd.DataFrame,
    id287_test: pd.DataFrame,
    id329_test: pd.DataFrame,
) -> dict[str, float]:
    pred = base_test[TARGETS].astype(float).to_numpy().copy()
    base_arr = base_test[TARGETS].astype(float).to_numpy()
    id178_arr = id178_test[TARGETS].astype(float).to_numpy()
    for col, target in enumerate(TARGETS):
        alpha = alpha_by_target.get(target, 0.0)
        pred[:, col] = blend_at_alpha(base_arr[:, col], id178_arr[:, col], alpha)
    id287_arr = id287_test[TARGETS].astype(float).to_numpy()
    id329_arr = id329_test[TARGETS].astype(float).to_numpy()
    failed_vec = (id287_arr - id178_arr).reshape(-1)
    vec = (pred - id178_arr).reshape(-1)
    denom = float(np.dot(failed_vec, failed_vec))
    projection_failed = float(np.dot(vec, failed_vec) / denom) if denom > 0 else 0.0
    return {
        "mean_abs_delta_vs_id173": float(np.mean(np.abs(pred - base_arr))),
        "max_abs_delta_vs_id173": float(np.max(np.abs(pred - base_arr))),
        "mean_abs_delta_vs_id178": float(np.mean(np.abs(pred - id178_arr))),
        "max_abs_delta_vs_id178": float(np.max(np.abs(pred - id178_arr))),
        "rmse_vs_id178": float(np.sqrt(np.mean((pred - id178_arr) ** 2))),
        "mean_abs_delta_vs_id329": float(np.mean(np.abs(pred - id329_arr))),
        "max_abs_delta_vs_id329": float(np.max(np.abs(pred - id329_arr))),
        "projection_to_id287_failed_direction": projection_failed,
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
    }


def summarize_combo(
    alpha_by_target: dict[str, float],
    local_losses: dict[str, dict[str, float]],
    stress_losses: dict[str, dict[str, dict[str, float]]],
    stress_base_means: dict[str, float],
    base_test: pd.DataFrame,
    id178_test: pd.DataFrame,
    id287_test: pd.DataFrame,
    id329_test: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    local_per_target = {
        target: float(local_losses[target][f"{alpha_by_target.get(target, 0.0):.2f}"]) for target in TARGETS
    }
    strategy_scores = {}
    for strategy_name, target_losses in stress_losses.items():
        per_target = {
            target: float(target_losses[target][f"{alpha_by_target.get(target, 0.0):.2f}"]) for target in TARGETS
        }
        mean_loss = float(np.mean(list(per_target.values())))
        base_mean = stress_base_means[strategy_name]
        strategy_scores[strategy_name] = {
            "mean_loss": mean_loss,
            "base_mean": base_mean,
            "improvement_vs_base": base_mean - mean_loss,
            "per_target": per_target,
        }
    return {
        "label": label,
        "alpha_by_target": {target: float(alpha_by_target.get(target, 0.0)) for target in TARGETS},
        "local_oof_loss": float(np.mean(list(local_per_target.values()))),
        "local_per_target": local_per_target,
        "stress": {
            "strategy_scores": strategy_scores,
            "mean_across_stress_strategies": float(np.mean([row["mean_loss"] for row in strategy_scores.values()])),
            "min_improvement_vs_base": float(min(row["improvement_vs_base"] for row in strategy_scores.values())),
        },
        "geometry": geometry_for_alphas(alpha_by_target, base_test, id178_test, id287_test, id329_test),
    }


def compact_combo(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row["label"],
        "alpha_by_target": row["alpha_by_target"],
        "local_oof_loss": row["local_oof_loss"],
        "stress_mean": row["stress"]["mean_across_stress_strategies"],
        "min_stress_improvement": row["stress"]["min_improvement_vs_base"],
        "max_delta_vs_id178": row["geometry"]["max_abs_delta_vs_id178"],
        "max_delta_vs_id173": row["geometry"]["max_abs_delta_vs_id173"],
        "projection_to_id287_failed_direction": row["geometry"]["projection_to_id287_failed_direction"],
    }


def scan_combinations(
    alphas: list[float],
    local_losses: dict[str, dict[str, float]],
    stress_losses: dict[str, dict[str, dict[str, float]]],
    stress_base_means: dict[str, float],
    base_test: pd.DataFrame,
    id178_test: pd.DataFrame,
    id287_test: pd.DataFrame,
    id329_test: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for values in itertools.product(alphas, repeat=len(SEQUENCE_TARGETS)):
        alpha_by_target = dict(zip(SEQUENCE_TARGETS, values, strict=True))
        for target in TARGETS:
            alpha_by_target.setdefault(target, 0.0)
        rows.append(
            summarize_combo(
                alpha_by_target,
                local_losses,
                stress_losses,
                stress_base_means,
                base_test,
                id178_test,
                id287_test,
                id329_test,
                label="grid_combo",
            )
        )
    return rows


def select_named_rows(rows: list[dict[str, Any]], id334: dict[str, Any]) -> dict[str, Any]:
    positive = [row for row in rows if row["stress"]["min_improvement_vs_base"] > 0]
    low_move_positive = [
        row
        for row in positive
        if row["geometry"]["max_abs_delta_vs_id178"] <= id334["geometry"]["max_abs_delta_vs_id178"] + 1e-12
        and row["geometry"]["projection_to_id287_failed_direction"]
        <= id334["geometry"]["projection_to_id287_failed_direction"] + 1e-12
    ]
    return {
        "best_stress_positive": compact_combo(
            min(
                positive,
                key=lambda row: (
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                    row["local_oof_loss"],
                ),
            )
        )
        if positive
        else None,
        "best_local_among_positive": compact_combo(
            min(
                positive,
                key=lambda row: (
                    row["local_oof_loss"],
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                ),
            )
        )
        if positive
        else None,
        "best_low_movement_positive": compact_combo(
            min(
                low_move_positive,
                key=lambda row: (
                    row["local_oof_loss"],
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                ),
            )
        )
        if low_move_positive
        else None,
        "top_10_stress_positive": [
            compact_combo(row)
            for row in sorted(
                positive,
                key=lambda row: (
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                    row["local_oof_loss"],
                ),
            )[:10]
        ],
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID178 sequence alpha scan — post ID334",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Anchors",
        "",
        "| label | local OOF | stress mean | min stress improvement | max Δ178 | max Δ173 | failed projection | alphas |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["anchors"]:
        lines.append(
            f"| `{row['label']}` | {row['local_oof_loss']:.6f} | "
            f"{row['stress']['mean_across_stress_strategies']:.6f} | "
            f"{row['stress']['min_improvement_vs_base']:+.6f} | "
            f"{row['geometry']['max_abs_delta_vs_id178']:.6f} | "
            f"{row['geometry']['max_abs_delta_vs_id173']:.6f} | "
            f"{row['geometry']['projection_to_id287_failed_direction']:+.4f} | "
            f"`{row['alpha_by_target']}` |"
        )
    lines.extend(["", "## Selected grid rows", ""])
    selected = report["selected_rows"]
    for key in ["best_stress_positive", "best_local_among_positive", "best_low_movement_positive"]:
        row = selected.get(key)
        if row is None:
            lines.append(f"- `{key}`: none")
        else:
            lines.append(
                f"- `{key}`: local `{row['local_oof_loss']:.6f}`, stress `{row['stress_mean']:.6f}`, "
                f"min stress improvement `{row['min_stress_improvement']:+.6f}`, "
                f"max Δ178 `{row['max_delta_vs_id178']:.6f}`, failed projection "
                f"`{row['projection_to_id287_failed_direction']:+.4f}`, alphas `{row['alpha_by_target']}`"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            report["decision"]["reason"],
            "",
            "Do not submit any alpha-scan row directly. If a row is promoted later, it needs a new immutable ID and a full artifact contract.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_json: Path, alpha_grid: list[float]) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    test_sample = pd.read_csv(
        ROOT / "data" / "raw" / "ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"]
    )
    base_oof, base_test, base_source = load_artifact_pair(train, test, 173)
    id178_oof, id178_test, id178_source = load_artifact_pair(train, test, 178)
    _id287_oof, id287_test, id287_source = load_artifact_pair(train, test, 287)
    _id329_oof, id329_test, id329_source = load_artifact_pair(train, test, 329)

    conv_features = build_conv_features(train, test)
    strategies = stress_prediction_tables(train, test_sample, base_oof, conv_features)
    local_losses, stress_losses, base_local_losses, stress_base_means = precompute_losses(
        train, base_oof, id178_oof, strategies, alpha_grid
    )

    alpha_base = dict.fromkeys(TARGETS, 0.0)
    alpha_id178 = {target: (1.0 if target in SEQUENCE_TARGETS else 0.0) for target in TARGETS}
    alpha_id334 = {target: (1.0 if target in ["Q1", "Q2", "Q3", "S2"] else 0.0) for target in TARGETS}
    anchors = [
        summarize_combo(
            alpha_base,
            local_losses,
            stress_losses,
            stress_base_means,
            base_test,
            id178_test,
            id287_test,
            id329_test,
            label="id173_base",
        ),
        summarize_combo(
            alpha_id178,
            local_losses,
            stress_losses,
            stress_base_means,
            base_test,
            id178_test,
            id287_test,
            id329_test,
            label="id178_keep_all",
        ),
        summarize_combo(
            alpha_id334,
            local_losses,
            stress_losses,
            stress_base_means,
            base_test,
            id178_test,
            id287_test,
            id329_test,
            label="id334_rollback_s1_s4",
        ),
    ]
    id334 = anchors[2]
    rows = scan_combinations(
        alpha_grid, local_losses, stress_losses, stress_base_means, base_test, id178_test, id287_test, id329_test
    )
    selected_rows = select_named_rows(rows, id334)

    best_low_movement = selected_rows["best_low_movement_positive"]
    promote_candidate = bool(
        best_low_movement
        and best_low_movement["local_oof_loss"] < id334["local_oof_loss"] - 0.0005
        and best_low_movement["stress_mean"] <= id334["stress"]["mean_across_stress_strategies"] + 0.0005
    )
    reason = (
        "A low-movement alpha row improves local OOF over ID334 without worsening stress materially; this is still "
        "a same-family report-only hypothesis and needs a separate promotion script before any review."
        if promote_candidate
        else "No alpha row is promoted. The scan is useful for understanding ID178's sequence signal, but it does not create a Public 0.58-plausible or submission-ready candidate."
    )

    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "id178_sequence_alpha_scan_post_id334",
        "alpha_grid": alpha_grid,
        "sequence_targets": SEQUENCE_TARGETS,
        "sources": {
            "id173": base_source,
            "id178": id178_source,
            "id287": id287_source,
            "id329": id329_source,
        },
        "base_local_per_target": base_local_losses,
        "stress_base_means": stress_base_means,
        "anchors": anchors,
        "selected_rows": selected_rows,
        "decision": {
            "promote_submission_candidate": promote_candidate,
            "public_058_plausible": False,
            "reason": reason,
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "id178_sequence_alpha_scan_post_id334_20260617.json",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated alpha multipliers for each ID178 sequence target.",
    )
    args = parser.parse_args()
    alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    report = run(args.output_json, alpha_grid)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected_rows": report["selected_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
