#!/usr/bin/env python3
"""Post-ID342 validation-weighted pseudo-public split audit.

This report-only diagnostic builds a conservative review lens from existing OOF
predictions, public-scored rows, stable-shape penalties, subject-hole/LOSO
splits, and test-neighbor stress splits. It is a validation audit, not a
candidate generator.

It creates no prediction CSV, feature matrix artifact, base prediction
artifact, submission CSV, submission-log row, Kaggle action, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
import math
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

from scripts.diagnose_post_id342_public_alignment_validation_lens import (  # noqa: E402
    PRIMARY_SPLITS,
    RECENT_SLATE_IDS,
    load_split_manifest,
)
from scripts.diagnose_post_id342_public_gap_penalty_proxy_redesign import (  # noqa: E402
    PENALTY_FEATURES,
    lens_label,
    load_stable_shape_envelope,
)
from scripts.diagnose_post_id342_public_gap_risk_lens import (  # noqa: E402
    ANCHOR_ID,
    BASE_PREDICTION_DIR,
    FAILED_PUBLIC_IDS,
    PUBLIC_055,
    STABLE_PUBLIC_IDS,
    TARGETS,
    load_oof,
    load_train_sorted,
    public_scores,
    rel,
    row_features,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_validation_weighted_pseudo_public_split_audit_20260625.json"
PUBLIC_REVIEW_THRESHOLD = 0.58
EPS = 1e-6

LENS_GROUPS = [
    {
        "name": "full_oof",
        "kind": "weighted_mean",
        "members": {"full_oof": 1.0},
        "description": "Full OOF loss by target.",
    },
    {
        "name": "primary_mean",
        "kind": "weighted_mean",
        "members": {split_name: 1.0 / len(PRIMARY_SPLITS) for split_name in PRIMARY_SPLITS},
        "description": "Mean of subject-hole, test-neighbor, and LOSO primary split losses.",
    },
    {
        "name": "subject_hole_loso_mean",
        "kind": "weighted_mean",
        "members": {"subject_hole_5": 0.5, "loso_10": 0.5},
        "description": "Subject-generalization mean over subject-hole and LOSO.",
    },
    {
        "name": "test_neighbor_mean",
        "kind": "weighted_mean",
        "members": {
            "actual_test_band_holdout": 1.0 / 3.0,
            "test_boundary_neighbor2_holdout": 1.0 / 3.0,
            "test_boundary_neighbor5_holdout": 1.0 / 3.0,
        },
        "description": "Temporal/test-neighbor stress mean.",
    },
    {
        "name": "primary_worst",
        "kind": "max",
        "members": dict.fromkeys(PRIMARY_SPLITS, 1.0),
        "description": "Worst target loss across the primary validation stresses.",
    },
]

TARGET_WEIGHT_MODES = [
    "uniform",
    "stable_target_reliability",
    "id342_target_bottleneck",
]


def finite_or_none(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def target_log_loss(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def weighted_split_target_loss(
    train: pd.DataFrame,
    values: np.ndarray,
    target: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    total = 0
    weighted = 0.0
    for _train_idx, valid_idx in folds:
        loss = target_log_loss(y[valid_idx], values[valid_idx, target_idx])
        weighted += loss * int(len(valid_idx))
        total += int(len(valid_idx))
    if total == 0:
        raise ValueError(f"Validation split for {target} has no rows.")
    return float(weighted / total)


def target_lens_losses(
    train: pd.DataFrame,
    values: np.ndarray,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {
        "full_oof": {
            target: target_log_loss(train[target].to_numpy(), values[:, target_idx])
            for target_idx, target in enumerate(TARGETS)
        }
    }
    for split_name, folds in splits.items():
        out[split_name] = {
            target: weighted_split_target_loss(train, values, target, folds)
            for target in TARGETS
        }
    return out


def mean_target_loss(target_losses: dict[str, float]) -> float:
    return float(np.mean([target_losses[target] for target in TARGETS]))


def target_vector(row: dict[str, Any], lens_group: dict[str, Any]) -> dict[str, float]:
    target_losses = row["target_lens_losses"]
    if lens_group["kind"] == "max":
        return {
            target: float(max(target_losses[split_name][target] for split_name in lens_group["members"]))
            for target in TARGETS
        }
    return {
        target: float(
            sum(float(weight) * target_losses[split_name][target] for split_name, weight in lens_group["members"].items())
        )
        for target in TARGETS
    }


def pairwise_agreement(rows: list[dict[str, Any]], score_key: str) -> float | None:
    if len(rows) < 2:
        return None
    total = 0
    agree = 0
    for left_idx, left in enumerate(rows):
        for right in rows[left_idx + 1 :]:
            public_sign = np.sign(float(left["public_score"]) - float(right["public_score"]))
            score_sign = np.sign(float(left[score_key]) - float(right[score_key]))
            if public_sign == 0 or score_sign == 0:
                continue
            total += 1
            agree += int(public_sign == score_sign)
    return float(agree / total) if total else None


def pairwise_agreement_from_values(public_values: list[float], score_values: list[float]) -> float | None:
    rows = [
        {"public_score": public_score, "score": score}
        for public_score, score in zip(public_values, score_values, strict=True)
    ]
    return pairwise_agreement(rows, "score")


def spearman(public_values: list[float], score_values: list[float]) -> float | None:
    if len(public_values) < 3:
        return None
    public_series = pd.Series(public_values)
    score_series = pd.Series(score_values)
    if public_series.nunique() <= 1 or score_series.nunique() <= 1:
        return None
    return finite_or_none(public_series.corr(score_series, method="spearman"))


def rank_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "spearman": None,
            "pearson": None,
            "pairwise_agreement": None,
            "rank_mae": None,
            "top1_match": False,
            "public_order": [],
            "score_order": [],
        }
    public_values = [float(row["public_score"]) for row in rows]
    score_values = [float(row[score_key]) for row in rows]
    public_series = pd.Series(public_values, index=[row["id"] for row in rows])
    score_series = pd.Series(score_values, index=[row["id"] for row in rows])
    public_ranks = public_series.rank(method="average", ascending=True)
    score_ranks = score_series.rank(method="average", ascending=True)
    public_order = [int(idx) for idx in public_series.sort_values().index]
    score_order = [int(idx) for idx in score_series.sort_values().index]
    pearson = public_series.corr(score_series, method="pearson") if score_series.nunique() > 1 else None
    return {
        "n": int(len(rows)),
        "spearman": finite_or_none(public_series.corr(score_series, method="spearman")),
        "pearson": finite_or_none(pearson),
        "pairwise_agreement": pairwise_agreement(rows, score_key),
        "rank_mae": float(np.mean(np.abs(public_ranks - score_ranks))),
        "top1_match": bool(public_order[0] == score_order[0]),
        "public_order": public_order,
        "score_order": score_order,
        "best_by_score": score_order[0],
        "best_by_public": public_order[0],
    }


def normalize_target_weights(raw: dict[str, float]) -> dict[str, float]:
    values = {target: max(0.0, float(raw.get(target, 0.0))) for target in TARGETS}
    total = float(sum(values.values()))
    if total <= 0.0 or not np.isfinite(total):
        return {target: 1.0 / len(TARGETS) for target in TARGETS}
    return {target: float(values[target] / total) for target in TARGETS}


def derive_target_weights(
    scored_rows: list[dict[str, Any]],
    lens_group: dict[str, Any],
    mode: str,
) -> dict[str, float]:
    stable_rows = [
        row for row in scored_rows if row["id"] in STABLE_PUBLIC_IDS and row["id"] not in FAILED_PUBLIC_IDS
    ]
    if mode == "uniform" or len(stable_rows) < 3:
        return {target: 1.0 / len(TARGETS) for target in TARGETS}
    if mode == "id342_target_bottleneck":
        anchor_rows = [row for row in stable_rows if row["id"] == ANCHOR_ID]
        if not anchor_rows:
            return {target: 1.0 / len(TARGETS) for target in TARGETS}
        return normalize_target_weights(target_vector(anchor_rows[0], lens_group))
    if mode == "stable_target_reliability":
        weights = {}
        public_values = [float(row["public_score"]) for row in stable_rows]
        for target in TARGETS:
            score_values = [target_vector(row, lens_group)[target] for row in stable_rows]
            sp = spearman(public_values, score_values)
            pairwise = pairwise_agreement_from_values(public_values, score_values)
            reliability = max(0.0, sp or 0.0) * max(0.0, pairwise or 0.0)
            weights[target] = 0.02 + reliability
        return normalize_target_weights(weights)
    raise ValueError(f"Unknown target weight mode: {mode}")


def pseudo_loss(row: dict[str, Any], lens_group: dict[str, Any], target_weights: dict[str, float]) -> float:
    vector = target_vector(row, lens_group)
    return float(sum(float(target_weights[target]) * vector[target] for target in TARGETS))


def shape_excess(row: dict[str, Any], envelope: dict[str, float]) -> tuple[float, int, dict[str, float]]:
    values: dict[str, float] = {}
    for feature, envelope_key in PENALTY_FEATURES:
        threshold = max(float(envelope[envelope_key]), 1e-12)
        values[feature] = max(0.0, float(row[feature]) - threshold) / threshold
    positive = [value for value in values.values() if value > 0.0]
    if not positive:
        return 0.0, 0, values
    return float(np.mean(positive)), int(len(positive)), values


def fit_projection_params(
    scored_rows: list[dict[str, Any]],
    lens_group: dict[str, Any],
    target_weight_mode: str,
) -> dict[str, Any]:
    if not scored_rows:
        raise ValueError("Cannot fit projection params without public-scored rows.")
    target_weights = derive_target_weights(scored_rows, lens_group, target_weight_mode)
    raw_losses = {row["id"]: pseudo_loss(row, lens_group, target_weights) for row in scored_rows}
    stable_rows = [
        row for row in scored_rows if row["id"] in STABLE_PUBLIC_IDS and row["id"] not in FAILED_PUBLIC_IDS
    ]
    gap_source = stable_rows if stable_rows else scored_rows
    stable_gap = float(np.median([float(row["public_score"]) - raw_losses[row["id"]] for row in gap_source]))
    xs = np.asarray([float(row["shape_excess_score"]) for row in scored_rows], dtype=float)
    ys = np.asarray(
        [
            max(0.0, float(row["public_score"]) - (raw_losses[row["id"]] + stable_gap))
            for row in scored_rows
        ],
        dtype=float,
    )
    denom = float(xs @ xs)
    penalty_scale = 0.0 if denom <= 0.0 else float((xs @ ys) / denom)
    penalty_scale = float(np.clip(penalty_scale, 0.0, 0.50))
    return {
        "target_weights": target_weights,
        "stable_gap": stable_gap,
        "shape_penalty_scale": penalty_scale,
    }


def projected_public(row: dict[str, Any], lens_group: dict[str, Any], params: dict[str, Any]) -> float:
    return float(
        pseudo_loss(row, lens_group, params["target_weights"])
        + float(params["stable_gap"])
        + float(params["shape_penalty_scale"]) * float(row["shape_excess_score"])
    )


def error_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    if not rows:
        return {"mae": None, "rmse": None, "max_abs_error": None}
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "mae": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max_abs_error": float(values.max()),
    }


def loo_rows(
    scored_rows: list[dict[str, Any]],
    lens_group: dict[str, Any],
    target_weight_mode: str,
) -> list[dict[str, Any]]:
    out = []
    for held_out in scored_rows:
        train_rows = [row for row in scored_rows if row["id"] != held_out["id"]]
        stable_train = [
            row for row in train_rows if row["id"] in STABLE_PUBLIC_IDS and row["id"] not in FAILED_PUBLIC_IDS
        ]
        if len(stable_train) < 3:
            continue
        params = fit_projection_params(train_rows, lens_group, target_weight_mode)
        projection = projected_public(held_out, lens_group, params)
        out.append(
            {
                "id": int(held_out["id"]),
                "public_score": float(held_out["public_score"]),
                "projection": projection,
                "abs_error": abs(projection - float(held_out["public_score"])),
                "is_stable_public": bool(held_out["id"] in STABLE_PUBLIC_IDS and held_out["id"] not in FAILED_PUBLIC_IDS),
                "stable_gap": float(params["stable_gap"]),
                "shape_penalty_scale": float(params["shape_penalty_scale"]),
            }
        )
    return out


def recent_slate_order(scored_rows: list[dict[str, Any]], score_key: str) -> list[int]:
    rows = [row for row in scored_rows if row["id"] in RECENT_SLATE_IDS]
    if len(rows) != len(RECENT_SLATE_IDS):
        return []
    return [int(row["id"]) for row in sorted(rows, key=lambda row: float(row[score_key]))]


def evaluate_config(
    scored_rows: list[dict[str, Any]],
    lens_group: dict[str, Any],
    target_weight_mode: str,
) -> dict[str, Any]:
    params = fit_projection_params(scored_rows, lens_group, target_weight_mode)
    projected_rows = []
    for row in scored_rows:
        projected = projected_public(row, lens_group, params)
        projected_rows.append({**row, "projection": projected})
    stable_rows = [
        row for row in projected_rows if row["id"] in STABLE_PUBLIC_IDS and row["id"] not in FAILED_PUBLIC_IDS
    ]
    failed_rows = [row for row in projected_rows if row["id"] in FAILED_PUBLIC_IDS]
    loo = loo_rows(scored_rows, lens_group, target_weight_mode)
    stable_loo = [row for row in loo if row["is_stable_public"]]
    failed_loo = [row for row in loo if not row["is_stable_public"]]
    stable_metrics = rank_metrics(stable_rows, "projection")
    all_metrics = rank_metrics(projected_rows, "projection")
    failed_public_min_projection = min((float(row["projection"]) for row in failed_rows), default=None)
    stable_public_max_projection = max((float(row["projection"]) for row in stable_rows), default=None)
    failed_separation_margin = None
    if failed_public_min_projection is not None and stable_public_max_projection is not None:
        failed_separation_margin = float(failed_public_min_projection - stable_public_max_projection)
    slate_order = recent_slate_order(projected_rows, "projection")
    stable_loo_error = error_summary(stable_loo, "abs_error")
    all_loo_error = error_summary(loo, "abs_error")
    review_lens_pass = bool(
        (stable_metrics["spearman"] is not None and stable_metrics["spearman"] >= 0.80)
        and (
            stable_metrics["pairwise_agreement"] is not None
            and stable_metrics["pairwise_agreement"] >= 0.80
        )
        and stable_metrics["top1_match"]
        and slate_order == RECENT_SLATE_IDS
        and (
            stable_loo_error["mae"] is not None
            and stable_loo_error["mae"] <= 0.006
        )
        and (failed_separation_margin is None or failed_separation_margin >= 0.0)
    )
    return {
        "config_name": f"{lens_group['name']}__{target_weight_mode}",
        "lens_group": lens_group["name"],
        "target_weight_mode": target_weight_mode,
        "target_weights": params["target_weights"],
        "stable_gap": float(params["stable_gap"]),
        "shape_penalty_scale": float(params["shape_penalty_scale"]),
        "stable_metrics": stable_metrics,
        "all_scored_metrics": all_metrics,
        "stable_loo_error": stable_loo_error,
        "all_loo_error": all_loo_error,
        "failed_loo_error": error_summary(failed_loo, "abs_error"),
        "recent_slate_projection_order": slate_order,
        "recent_slate_order_match": bool(slate_order == RECENT_SLATE_IDS),
        "failed_separation_margin": failed_separation_margin,
        "review_lens_pass": review_lens_pass,
    }


def config_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stable_loo_mae = row["stable_loo_error"]["mae"]
    stable_pairwise = row["stable_metrics"]["pairwise_agreement"]
    stable_spearman = row["stable_metrics"]["spearman"]
    all_pairwise = row["all_scored_metrics"]["pairwise_agreement"]
    failed_margin = row["failed_separation_margin"]
    return (
        not row["review_lens_pass"],
        -(stable_pairwise if stable_pairwise is not None else -1.0),
        -(stable_spearman if stable_spearman is not None else -1.0),
        not row["recent_slate_order_match"],
        math.inf if stable_loo_mae is None else stable_loo_mae,
        -(all_pairwise if all_pairwise is not None else -1.0),
        -(failed_margin if failed_margin is not None else -math.inf),
        row["config_name"],
    )


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    train = load_train_sorted()
    splits = load_split_manifest()
    y_true = train[TARGETS].astype(int).to_numpy()
    train_target_rate = y_true.mean(axis=0)
    anchor_path = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{ANCHOR_ID}_*.csv"))[0]
    anchor = load_oof(anchor_path, train)
    scores = public_scores()
    envelope = load_stable_shape_envelope()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(BASE_PREDICTION_DIR.glob("base_oof_id_*.csv")):
        try:
            values = load_oof(path, train)
            row = row_features(path, values, anchor, train_target_rate, y_true)
            row["target_lens_losses"] = target_lens_losses(train, values, splits)
            row["lens_losses"] = {
                lens_name: mean_target_loss(target_losses)
                for lens_name, target_losses in row["target_lens_losses"].items()
            }
            if row["id"] in scores:
                row["public_score"] = float(scores[row["id"]])
                row["public_minus_local"] = float(row["public_score"] - row["local_loss"])
            excess_score, excess_count, excess_by_feature = shape_excess(row, envelope)
            row["stable_shape_label"] = lens_label(row, envelope)
            row["shape_excess_score"] = excess_score
            row["shape_excess_feature_count"] = excess_count
            row["shape_excess_by_feature"] = excess_by_feature
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel(path), "error": f"{type(exc).__name__}: {exc}"})
    return rows, errors


def trim_config(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_name": row["config_name"],
        "lens_group": row["lens_group"],
        "target_weight_mode": row["target_weight_mode"],
        "target_weights": row["target_weights"],
        "stable_gap": row["stable_gap"],
        "shape_penalty_scale": row["shape_penalty_scale"],
        "stable_metrics": row["stable_metrics"],
        "all_scored_metrics": row["all_scored_metrics"],
        "stable_loo_error": row["stable_loo_error"],
        "all_loo_error": row["all_loo_error"],
        "failed_loo_error": row["failed_loo_error"],
        "recent_slate_projection_order": row["recent_slate_projection_order"],
        "failed_separation_margin": row["failed_separation_margin"],
        "review_lens_pass": row["review_lens_pass"],
    }


def trim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "filename",
        "local_loss",
        "public_score",
        "public_minus_local",
        "weighted_pseudo_loss",
        "weighted_public_projection",
        "weighted_gap_to_public055",
        "weighted_gap_to_review058",
        "stable_shape_label",
        "shape_excess_score",
        "shape_excess_feature_count",
        "legacy_public_guided_family",
        "weighted_public055_pass",
        "credible_weighted_public055_pass",
        "weighted_review058_pass",
        "credible_weighted_review058_pass",
    ]
    return {key: row.get(key) for key in keys if key in row}


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    rows, errors = collect_rows()
    scored_rows = [row for row in rows if "public_score" in row]
    if not scored_rows:
        raise ValueError("No public-scored OOF rows found.")
    config_rows = [
        evaluate_config(scored_rows, lens_group, target_weight_mode)
        for lens_group in LENS_GROUPS
        for target_weight_mode in TARGET_WEIGHT_MODES
    ]
    config_rows = sorted(config_rows, key=config_sort_key)
    primary_config = config_rows[0]
    primary_lens_group = next(group for group in LENS_GROUPS if group["name"] == primary_config["lens_group"])
    primary_params = {
        "target_weights": primary_config["target_weights"],
        "stable_gap": primary_config["stable_gap"],
        "shape_penalty_scale": primary_config["shape_penalty_scale"],
    }
    for row in rows:
        row["weighted_pseudo_loss"] = pseudo_loss(row, primary_lens_group, primary_params["target_weights"])
        row["weighted_public_projection"] = projected_public(row, primary_lens_group, primary_params)
        row["weighted_gap_to_public055"] = float(row["weighted_public_projection"] - PUBLIC_055)
        row["weighted_gap_to_review058"] = float(row["weighted_public_projection"] - PUBLIC_REVIEW_THRESHOLD)
        row["weighted_public055_pass"] = bool(row["weighted_public_projection"] <= PUBLIC_055)
        row["weighted_review058_pass"] = bool(row["weighted_public_projection"] <= PUBLIC_REVIEW_THRESHOLD)
        stable_shape = row["stable_shape_label"] == "stable_public_shape_envelope"
        row["credible_weighted_public055_pass"] = bool(
            row["weighted_public055_pass"]
            and primary_config["review_lens_pass"]
            and stable_shape
            and not row["legacy_public_guided_family"]
            and row["id"] not in FAILED_PUBLIC_IDS
        )
        row["credible_weighted_review058_pass"] = bool(
            row["weighted_review058_pass"]
            and primary_config["review_lens_pass"]
            and stable_shape
            and not row["legacy_public_guided_family"]
            and row["id"] not in FAILED_PUBLIC_IDS
        )
    top_weighted_rows = sorted(rows, key=lambda row: row["weighted_public_projection"])[:30]
    stable_shape_rows = [
        row
        for row in rows
        if row["stable_shape_label"] == "stable_public_shape_envelope"
        and not row["legacy_public_guided_family"]
        and row["id"] not in FAILED_PUBLIC_IDS
    ]
    top_credible_shape_rows = sorted(stable_shape_rows, key=lambda row: row["weighted_public_projection"])[:30]
    public055_rows = [row for row in rows if row["weighted_public055_pass"]]
    credible_public055_rows = [row for row in rows if row["credible_weighted_public055_pass"]]
    review058_rows = [row for row in rows if row["weighted_review058_pass"]]
    credible_review058_rows = [row for row in rows if row["credible_weighted_review058_pass"]]
    anchor_rows = [row for row in rows if row["id"] == ANCHOR_ID]
    anchor_projection = anchor_rows[0]["weighted_public_projection"] if anchor_rows else None
    decision = {
        "oof_files_scanned": int(len(rows)),
        "scan_errors": int(len(errors)),
        "scored_rows": int(len(scored_rows)),
        "config_count": int(len(config_rows)),
        "review_lens_pass_count": int(sum(row["review_lens_pass"] for row in config_rows)),
        "primary_config": trim_config(primary_config),
        "anchor_weighted_public_projection": anchor_projection,
        "weighted_public055_pass_count": int(len(public055_rows)),
        "credible_weighted_public055_pass_count": int(len(credible_public055_rows)),
        "weighted_review058_pass_count": int(len(review058_rows)),
        "credible_weighted_review058_pass_count": int(len(credible_review058_rows)),
        "best_weighted_candidate": trim_candidate(top_weighted_rows[0]) if top_weighted_rows else None,
        "best_credible_shape_candidate": trim_candidate(top_credible_shape_rows[0])
        if top_credible_shape_rows
        else None,
        "validation_weighted_pseudo_public_gate_pass": bool(
            primary_config["review_lens_pass"] and credible_public055_rows
        ),
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Use the primary validation-weighted lens as a stricter audit lens only. If it finds no credible "
            "Public-0.55 or Public-0.58 stable-shape row, close this validation-redesign branch for candidate "
            "creation and require a materially new source hypothesis."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_validation_weighted_pseudo_public_split_audit",
        "purpose": (
            "Derive and stress-check target-weighted pseudo-public validation lenses before any new post-ID342 "
            "candidate creation."
        ),
        "lens_groups": LENS_GROUPS,
        "target_weight_modes": TARGET_WEIGHT_MODES,
        "stable_public_ids": sorted(STABLE_PUBLIC_IDS),
        "failed_public_ids": sorted(FAILED_PUBLIC_IDS),
        "recent_slate_ids": RECENT_SLATE_IDS,
        "config_summaries": [trim_config(row) for row in config_rows],
        "top_weighted_rows": [trim_candidate(row) for row in top_weighted_rows],
        "top_credible_shape_rows": [trim_candidate(row) for row in top_credible_shape_rows],
        "weighted_public055_pass_rows": [trim_candidate(row) for row in public055_rows[:30]],
        "credible_weighted_public055_pass_rows": [trim_candidate(row) for row in credible_public055_rows[:30]],
        "credible_weighted_review058_pass_rows": [trim_candidate(row) for row in credible_review058_rows[:30]],
        "scan_errors": errors[:20],
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def format_optional(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    primary = decision["primary_config"]
    lines = [
        "# Post-ID342 validation-weighted pseudo-public split audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- OOF files scanned: `{decision['oof_files_scanned']}`",
        f"- Scored rows: `{decision['scored_rows']}`",
        f"- Lens configs evaluated: `{decision['config_count']}`",
        f"- Review-lens pass count: `{decision['review_lens_pass_count']}`",
        f"- Primary config: `{primary['config_name']}`",
        f"- Primary stable LOO MAE: `{primary['stable_loo_error']['mae']}`",
        f"- Primary stable pairwise: `{primary['stable_metrics']['pairwise_agreement']}`",
        f"- Primary all-scored pairwise: `{primary['all_scored_metrics']['pairwise_agreement']}`",
        f"- Primary recent slate order: `{primary['recent_slate_projection_order']}`",
        f"- Anchor weighted projection: `{decision['anchor_weighted_public_projection']}`",
        f"- Public-0.55 weighted pass count: `{decision['weighted_public055_pass_count']}`",
        f"- Credible Public-0.55 weighted pass count: `{decision['credible_weighted_public055_pass_count']}`",
        f"- Public-0.58 review pass count: `{decision['weighted_review058_pass_count']}`",
        f"- Credible Public-0.58 review pass count: `{decision['credible_weighted_review058_pass_count']}`",
        f"- Validation-weighted gate pass: `{decision['validation_weighted_pseudo_public_gate_pass']}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Lens Configs",
        "",
        "| rank | config | pass | stable LOO MAE | stable pairwise | all pairwise | slate order | failed margin |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(report["config_summaries"][:12], start=1):
        lines.append(
            f"| `{rank}` | `{row['config_name']}` | `{row['review_lens_pass']}` | "
            f"`{format_optional(row['stable_loo_error']['mae'], 9)}` | "
            f"`{format_optional(row['stable_metrics']['pairwise_agreement'], 3)}` | "
            f"`{format_optional(row['all_scored_metrics']['pairwise_agreement'], 3)}` | "
            f"`{row['recent_slate_projection_order']}` | "
            f"`{format_optional(row['failed_separation_margin'], 9)}` |"
        )
    lines.extend(
        [
            "",
            "## Target Weights",
            "",
            "| target | weight |",
            "|---|---:|",
        ]
    )
    for target, weight in primary["target_weights"].items():
        lines.append(f"| `{target}` | `{weight:.6f}` |")
    lines.extend(
        [
            "",
            "## Top Weighted Rows",
            "",
            "| id | projection | gap to 0.55 | label | file |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for row in report["top_weighted_rows"][:12]:
        lines.append(
            f"| `{row['id']}` | `{row['weighted_public_projection']:.9f}` | "
            f"`{row['weighted_gap_to_public055']:+.9f}` | `{row['stable_shape_label']}` | "
            f"`{row['filename']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Credible Shape Rows",
            "",
            "| id | projection | gap to 0.55 | file |",
            "|---:|---:|---:|---|",
        ]
    )
    for row in report["top_credible_shape_rows"][:12]:
        lines.append(
            f"| `{row['id']}` | `{row['weighted_public_projection']:.9f}` | "
            f"`{row['weighted_gap_to_public055']:+.9f}` | `{row['filename']}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Credible weighted passes: "
        f"0.55={decision['credible_weighted_public055_pass_count']} | "
        f"0.58={decision['credible_weighted_review058_pass_count']} | "
        f"gate={decision['validation_weighted_pseudo_public_gate_pass']}"
    )


if __name__ == "__main__":
    main()
