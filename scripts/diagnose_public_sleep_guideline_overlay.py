#!/usr/bin/env python3
"""Diagnose lightweight sleep-guideline overlays on existing public-structure artifacts.

This is report-only. It does not create a submission artifact. The goal is to
test whether objective sleep proxies (duration, efficiency, awake/fragmentation,
and sleep-window activity/screen/light signals) add a stable low-weight signal
on top of existing anchors.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    load_base_artifact,
    make_interleaved_subject_folds,
)
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

EPS = 1e-6
DEFAULT_BASE_IDS = [221, 226, 231]
SLEEP_TARGETS = ["S1", "S2", "S3", "S4"]
BLEND_WEIGHTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_leaf_nodes: int | None = None


# Keep the default diagnostic intentionally light. HGB-style sleep overlays can
# be reintroduced later on Kaggle if this linear probe finds a real signal.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_1", "lr", c_value=0.10),
)


def add_sleep_guideline_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    duration_cols = ["wHr_sleep_duration_deep", "wHr_sleep_duration_light", "wHr_sleep_duration_awake"]
    if set(duration_cols).issubset(out.columns):
        deep = pd.to_numeric(out["wHr_sleep_duration_deep"], errors="coerce")
        light = pd.to_numeric(out["wHr_sleep_duration_light"], errors="coerce")
        awake = pd.to_numeric(out["wHr_sleep_duration_awake"], errors="coerce")
        total = deep + light
        time_in_bed = total + awake
        out["sleep_total_proxy"] = total
        out["time_in_bed_proxy"] = time_in_bed
        out["sleep_awake_ratio_calc"] = awake / time_in_bed.replace(0, np.nan)
        out["sleep_efficiency_calc"] = total / time_in_bed.replace(0, np.nan)
        out["sleep_total_log1p"] = np.log1p(total.clip(lower=0))
        out["sleep_awake_log1p"] = np.log1p(awake.clip(lower=0))
        out["sleep_tib_log1p"] = np.log1p(time_in_bed.clip(lower=0))
    if {"wHr_sleep_duration_deep", "wHr_sleep_duration_light"}.issubset(out.columns):
        deep = pd.to_numeric(out["wHr_sleep_duration_deep"], errors="coerce")
        light = pd.to_numeric(out["wHr_sleep_duration_light"], errors="coerce")
        out["sleep_deep_to_light_ratio_calc"] = deep / light.replace(0, np.nan)
    if {"slp_screen_on", "slp_act_active_ratio"}.issubset(out.columns):
        out["sleep_screen_activity_product"] = (
            pd.to_numeric(out["slp_screen_on"], errors="coerce")
            * pd.to_numeric(out["slp_act_active_ratio"], errors="coerce")
        )
    return out


def sleep_feature_sets(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    core_candidates = [
        "sleep_total_proxy",
        "time_in_bed_proxy",
        "sleep_efficiency_calc",
        "sleep_awake_ratio_calc",
        "sleep_total_log1p",
        "sleep_awake_log1p",
        "sleep_tib_log1p",
        "sleep_deep_to_light_ratio_calc",
        "wHr_sleep_efficiency_proxy",
        "wHr_sleep_fragmentation_index",
        "wHr_deep_sleep_ratio",
        "wHr_sleep_hr_mean",
        "wHr_sleep_hr_std",
    ]
    presleep_candidates = [
        "act_presleep_active",
        "screen_presleep_on",
        "screen_presleep_ratio",
        "ac_presleep_charging",
        "screen_night_ratio",
        "screen_eve_ratio",
        "mlight_night_mean",
        "wlight_night_mean",
        "mlight_dark_ratio",
        "wlight_dark_ratio",
    ]
    sleep_window_candidates = [
        col
        for col in train.columns
        if col.startswith(("slp_", "wHr_slp_")) or col in {"sleep_screen_activity_product"}
    ]
    feature_sets = {
        "guideline_core": core_candidates,
        "presleep_context": core_candidates + presleep_candidates,
        "sleep_window": core_candidates + sleep_window_candidates,
        "sleep_all": core_candidates + presleep_candidates + sleep_window_candidates,
    }
    out: dict[str, list[str]] = {}
    for name, columns in feature_sets.items():
        filtered = []
        for col in dict.fromkeys(columns):
            if col not in train.columns or col not in test.columns:
                continue
            if not pd.api.types.is_numeric_dtype(train[col]):
                continue
            train_values = pd.to_numeric(train[col], errors="coerce")
            test_values = pd.to_numeric(test[col], errors="coerce")
            if train_values.nunique(dropna=True) <= 2:
                continue
            if not (train_values.notna().any() or test_values.notna().any()):
                continue
            filtered.append(col)
        out[name] = filtered
    return out


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=300, random_state=2026),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=70,
                learning_rate=0.03,
                max_leaf_nodes=int(spec.max_leaf_nodes),
                min_samples_leaf=18,
                l2_regularization=5.0,
                random_state=2026,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def positive_proba(model: Any, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    if 1 not in classes:
        return np.full(len(features), 1.0 if classes[0] == 1 else 0.0)
    return proba[:, int(np.where(classes == 1)[0][0])]


def fit_predict(
    train_features: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    pred_idx: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    if len(np.unique(y[fit_idx])) < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = clone(make_model(spec))
    model.fit(train_features.iloc[fit_idx], y[fit_idx])
    return np.clip(positive_proba(model, train_features.iloc[pred_idx]), EPS, 1.0 - EPS)


def fit_predict_test_average(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(test_features), dtype=float)
    for train_idx, valid_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(spec))
        model.fit(train_features.iloc[train_idx], y[train_idx])
        oof[valid_idx] = positive_proba(model, train_features.iloc[valid_idx])
        test_pred += positive_proba(model, test_features) / len(folds)
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred, EPS, 1.0 - EPS)


def fold_ids_to_pairs(fold_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    all_idx = np.arange(len(fold_ids))
    for fold in sorted(np.unique(fold_ids)):
        valid_idx = np.where(fold_ids == fold)[0]
        train_idx = np.setdiff1d(all_idx, valid_idx, assume_unique=False)
        pairs.append((train_idx, valid_idx))
    return pairs


def subject_hole_fold_ids(train: pd.DataFrame) -> np.ndarray:
    fold_ids = np.full(len(train), -1, dtype=int)
    for fold, (_train_idx, valid_idx) in enumerate(make_subject_hole_folds(train, n_folds=5)):
        fold_ids[np.asarray(valid_idx)] = fold
    if np.any(fold_ids < 0):
        raise ValueError("Subject-hole folds did not cover every row.")
    return fold_ids


def candidate_name(feature_set: str, model_name: str, weight: float) -> str:
    if weight == 0.0:
        return "base"
    return f"{feature_set}__{model_name}__w{weight:g}"


def blend_values(base: np.ndarray, side: np.ndarray, weight: float) -> np.ndarray:
    if weight == 0.0:
        return np.clip(base, EPS, 1.0 - EPS)
    return np.clip((1.0 - weight) * base + weight * side, EPS, 1.0 - EPS)


def diagnose_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    target: str,
    feature_sets: dict[str, list[str]],
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = train[target].astype(int).to_numpy()
    base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    fold_pairs = fold_ids_to_pairs(fold_ids)

    # Full-fold OOF/test candidates for final test-choice diagnostics.
    side_predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for feature_set, columns in feature_sets.items():
        if not columns:
            continue
        train_features = train[columns].apply(pd.to_numeric, errors="coerce")
        test_features = test[columns].apply(pd.to_numeric, errors="coerce")
        for spec in MODEL_SPECS:
            side_predictions[(feature_set, spec.name)] = fit_predict_test_average(
                train_features=train_features,
                test_features=test_features,
                y=y,
                folds=fold_pairs,
                spec=spec,
            )

    full_losses = {"base": float(log_loss(y, base_values, labels=[0, 1]))}
    full_candidate_values: dict[str, np.ndarray] = {"base": base_values}
    full_candidate_test: dict[str, np.ndarray] = {"base": base_test_values}
    for (feature_set, model_name), (side_oof, side_test) in side_predictions.items():
        for weight in BLEND_WEIGHTS:
            name = candidate_name(feature_set, model_name, weight)
            if name == "base":
                continue
            full_candidate_values[name] = blend_values(base_values, side_oof, weight)
            full_candidate_test[name] = blend_values(base_test_values, side_test, weight)
            full_losses[name] = float(log_loss(y, full_candidate_values[name], labels=[0, 1]))
    full_best_name = min(full_losses, key=full_losses.get)

    # Outer-fold candidate selection. Within each outer train segment, side OOF
    # predictions are rebuilt using only outer-train rows.
    nested_oof = np.zeros(len(y), dtype=float)
    outer_choices = []
    for outer_fold in sorted(np.unique(fold_ids)):
        outer_valid_idx = np.where(fold_ids == outer_fold)[0]
        outer_train_idx = np.where(fold_ids != outer_fold)[0]
        candidate_train_values: dict[str, np.ndarray] = {"base": base_values[outer_train_idx]}
        candidate_valid_values: dict[str, np.ndarray] = {"base": base_values[outer_valid_idx]}

        inner_fold_values = sorted(np.unique(fold_ids[outer_train_idx]))
        for feature_set, columns in feature_sets.items():
            if not columns:
                continue
            train_features = train[columns].apply(pd.to_numeric, errors="coerce")
            for spec in MODEL_SPECS:
                inner_oof = np.zeros(len(outer_train_idx), dtype=float)
                for inner_fold in inner_fold_values:
                    inner_valid_global = outer_train_idx[fold_ids[outer_train_idx] == inner_fold]
                    inner_fit_global = outer_train_idx[fold_ids[outer_train_idx] != inner_fold]
                    pred = fit_predict(
                        train_features=train_features,
                        y=y,
                        fit_idx=inner_fit_global,
                        pred_idx=inner_valid_global,
                        spec=spec,
                    )
                    mask = np.isin(outer_train_idx, inner_valid_global)
                    inner_oof[mask] = pred
                outer_valid_side = fit_predict(
                    train_features=train_features,
                    y=y,
                    fit_idx=outer_train_idx,
                    pred_idx=outer_valid_idx,
                    spec=spec,
                )
                for weight in BLEND_WEIGHTS:
                    name = candidate_name(feature_set, spec.name, weight)
                    if name == "base":
                        continue
                    candidate_train_values[name] = blend_values(base_values[outer_train_idx], inner_oof, weight)
                    candidate_valid_values[name] = blend_values(base_values[outer_valid_idx], outer_valid_side, weight)

        train_losses = {
            name: float(log_loss(y[outer_train_idx], values, labels=[0, 1]))
            for name, values in candidate_train_values.items()
        }
        best_name = min(train_losses, key=train_losses.get)
        nested_oof[outer_valid_idx] = candidate_valid_values[best_name]
        outer_choices.append(
            {
                "outer_fold": int(outer_fold),
                "selected": best_name,
                "train_loss": float(train_losses[best_name]),
                "valid_loss": float(log_loss(y[outer_valid_idx], candidate_valid_values[best_name], labels=[0, 1])),
                "base_valid_loss": float(log_loss(y[outer_valid_idx], base_values[outer_valid_idx], labels=[0, 1])),
                "top_train_losses": dict(sorted(train_losses.items(), key=lambda item: item[1])[:8]),
            }
        )

    selected_test = full_candidate_test[full_best_name]
    details = {
        "base_loss": full_losses["base"],
        "full_oof_best": full_best_name,
        "full_oof_best_loss": full_losses[full_best_name],
        "nested_loss": float(log_loss(y, nested_oof, labels=[0, 1])),
        "nested_delta_vs_base": float(log_loss(y, nested_oof, labels=[0, 1]) - full_losses["base"]),
        "full_oof_delta_vs_base": float(full_losses[full_best_name] - full_losses["base"]),
        "outer_choices": outer_choices,
        "top_full_oof_losses": dict(sorted(full_losses.items(), key=lambda item: item[1])[:12]),
        "selected_test_mean_before": float(np.mean(base_test_values)),
        "selected_test_mean_after": float(np.mean(selected_test)),
        "selected_test_max_abs_delta_vs_base": float(np.max(np.abs(selected_test - base_test_values))),
    }
    return nested_oof, selected_test, details


def diagnose_base(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_id: int,
    fold_ids: np.ndarray,
    fold_scheme: str,
    feature_sets: dict[str, list[str]],
) -> dict[str, Any]:
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    nested_oof = base_oof.copy()
    selected_test = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in SLEEP_TARGETS:
        target_oof, target_test, details = diagnose_target(
            train=train,
            test=test,
            base_oof=base_oof,
            base_test=base_test,
            target=target,
            feature_sets=feature_sets,
            fold_ids=fold_ids,
        )
        nested_oof[target] = target_oof
        selected_test[target] = target_test
        target_details[target] = details

    base_target_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    nested_target_scores = {
        target: float(log_loss(train[target].astype(int), nested_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    test_delta = {
        target: float(
            np.max(
                np.abs(
                    selected_test[target].astype(float).to_numpy() - base_test[target].astype(float).to_numpy()
                )
            )
        )
        for target in SLEEP_TARGETS
    }
    return {
        "base_id": base_id,
        "base_source": base_source,
        "fold_scheme": fold_scheme,
        "base_overall": float(np.mean(list(base_target_scores.values()))),
        "nested_overall": float(np.mean(list(nested_target_scores.values()))),
        "nested_delta_vs_base": float(np.mean(list(nested_target_scores.values())) - np.mean(list(base_target_scores.values()))),
        "base_target_scores": base_target_scores,
        "nested_target_scores": nested_target_scores,
        "target_details": target_details,
        "sleep_target_test_max_abs_delta_vs_base": test_delta,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Public sleep-guideline overlay diagnostic",
        "",
        f"- Fold scheme: `{report['fold_scheme']}`",
        f"- Base IDs: `{report['base_ids']}`",
        "- This is report-only; no submission files were generated.",
        "",
        "## Summary",
        "",
    ]
    for result in report["results"]:
        lines.append(
            f"- ID{result['base_id']:03d}: base `{result['base_overall']:.6f}` -> nested "
            f"`{result['nested_overall']:.6f}` (delta `{result['nested_delta_vs_base']:+.6f}`)."
        )
        for target in SLEEP_TARGETS:
            details = result["target_details"][target]
            lines.append(
                f"  - {target}: base `{details['base_loss']:.6f}`, nested `{details['nested_loss']:.6f}`, "
                f"full-OOF best `{details['full_oof_best']}` = `{details['full_oof_best_loss']:.6f}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "Full-OOF best rows are optimistic diagnostics. Promotion should require nested improvement, "
            "small test movement, and a clean artifact-contract run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ids", default=",".join(str(value) for value in DEFAULT_BASE_IDS))
    parser.add_argument("--fold-scheme", choices=["public_interleaved", "subject_hole"], default="public_interleaved")
    parser.add_argument("--report-name", default="public_sleep_guideline_overlay_diagnostic_20260616")
    args = parser.parse_args()

    train_raw, test_raw, _metrics_sorted, _sample_original = build_feature_frames()
    train = add_sleep_guideline_features(train_raw)
    test = add_sleep_guideline_features(test_raw)
    if args.fold_scheme == "public_interleaved":
        train_for_folds = add_date_dt(train)
        fold_ids = make_interleaved_subject_folds(train_for_folds)
    else:
        fold_ids = subject_hole_fold_ids(train)

    feature_sets = sleep_feature_sets(train, test)
    base_ids = [int(value.strip()) for value in args.base_ids.split(",") if value.strip()]
    results = [
        diagnose_base(
            train=train,
            test=test,
            base_id=base_id,
            fold_ids=fold_ids,
            fold_scheme=args.fold_scheme,
            feature_sets=feature_sets,
        )
        for base_id in base_ids
    ]
    report = {
        "purpose": "Report-only diagnostic for low-weight objective sleep-guideline overlays.",
        "fold_scheme": args.fold_scheme,
        "base_ids": base_ids,
        "targets_modified_in_diagnostic": SLEEP_TARGETS,
        "blend_weights": BLEND_WEIGHTS,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "feature_sets": {name: {"n_features": len(cols), "columns": cols} for name, cols in feature_sets.items()},
        "results": results,
        "promotion_rule": (
            "Do not promote from full-OOF score alone. Consider promotion only if nested_overall improves, "
            "sleep target deltas are directionally consistent, and test max movement remains modest."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{args.report_name}.json"
    md_path = REPORTS_DIR / f"{args.report_name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path.relative_to(ROOT)), "md": str(md_path.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
