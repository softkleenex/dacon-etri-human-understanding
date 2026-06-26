#!/usr/bin/env python3
"""Report-only evening-load/recovery source smoke after ID341.

This diagnostic implements the second post-ID341 new-source family from
`post_id341_new_source_blueprint_20260618`: evening_load_recovery_v1. It builds
label-free load/recovery features from daily lifelog aggregates, evaluates small
source models inside the frozen ID340 split manifest, then runs a conservative
scale/geometry gate against ID178/ID341 public anchors.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    BLEND_WEIGHTS,
    EPS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    TARGETS,
    evaluate_values_on_split,
    fold_matrix,
    load_reference_values,
    load_split_manifest,
    make_model,
    predict_target_for_split,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import (  # noqa: E402
    ID341_PROXY,
    geometry_report,
    load_base_test_values,
)
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
TARGET_FOCUS = ["Q2", "Q3", "S2", "S4"]
ANCHOR_IDS = [178, 287, 329, 341]
SCALES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
CONFIG_WIN_FLOOR = 4
CONFIG_MAX_DELTA_CEILING = 0.0010
CONFIG_MEAN_DELTA_CEILING = -0.00020
REVIEW_MAX_DELTA_VS_341 = 0.040
REVIEW_PROXY_MARGIN = 0.00010
PUBLIC_058_GATE = 0.590

BASE_COLS = [
    "pedo_total_steps",
    "pedo_total_distance",
    "pedo_total_calories",
    "pedo_evening_steps",
    "pedo_active_hours",
    "pedo_step_freq_mean",
    "mUsage_total_time_mins",
    "mUsage_night_time_ratio",
    "mUsage_dopamine_ratio",
    "mUsage_app_entropy",
    "mUsage_hourly_volatility",
    "mUsage_sessions_count",
    "usage_total_time",
    "usage_late_time",
    "usage_late_ratio",
    "usage_eve_time",
    "usage_eve_ratio",
    "screen_on_total",
    "screen_on_ratio",
    "screen_night_ratio",
    "screen_eve_ratio",
    "screen_presleep_ratio",
    "mlight_eve_mean",
    "mlight_night_mean",
    "mlight_dark_ratio",
    "mlight_bright_ratio",
    "wlight_eve_mean",
    "wlight_night_mean",
    "wlight_dark_ratio",
    "wlight_bright_ratio",
    "gps_moving_ratio",
    "gps_total_disp",
    "mGps_radius_of_gyration",
    "mGps_spatial_entropy",
    "mGps_homestay_ratio",
    "amb_music_mean",
    "amb_speech_mean",
    "amb_silence_mean",
    "amb_n_records",
    "wHr_hr_day_mean",
    "wHr_hr_night_mean",
    "wHr_hr_rmssd_mean",
    "wHr_hr_mean_mean",
    "wHr_sleep_efficiency_proxy",
    "wHr_sleep_fragmentation_index",
    "wHr_sleep_hr_mean",
    "wHr_sleep_hr_std",
    "slp_hr_mean",
    "slp_hr_std",
    "slp_hr_rmssd",
    "slp_hr_range",
    "slp_hr_spike_count",
]


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator.abs() + 1.0)


def build_evening_load_recovery_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    )
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    features = rows.copy()
    for col in BASE_COLS:
        if col in combined.columns:
            features[f"elr_raw_{col}"] = pd.to_numeric(combined[col], errors="coerce")
    # Load side: daytime exertion, mobility, app/session pressure.
    steps = safe_col(combined, "pedo_total_steps")
    evening_steps = safe_col(combined, "pedo_evening_steps")
    usage_total = safe_col(combined, "usage_total_time").fillna(safe_col(combined, "mUsage_total_time_mins"))
    usage_late = safe_col(combined, "usage_late_time")
    usage_eve = safe_col(combined, "usage_eve_time")
    screen_total = safe_col(combined, "screen_on_total")
    screen_night_ratio = safe_col(combined, "screen_night_ratio")
    screen_presleep_ratio = safe_col(combined, "screen_presleep_ratio")
    dopamine_ratio = safe_col(combined, "mUsage_dopamine_ratio")
    app_entropy = safe_col(combined, "mUsage_app_entropy")
    moving_ratio = safe_col(combined, "gps_moving_ratio")
    total_disp = safe_col(combined, "gps_total_disp")
    homestay = safe_col(combined, "mGps_homestay_ratio")
    speech = safe_col(combined, "amb_speech_mean")
    music = safe_col(combined, "amb_music_mean")
    silence = safe_col(combined, "amb_silence_mean")
    hr_day = safe_col(combined, "wHr_hr_day_mean")
    hr_night = safe_col(combined, "wHr_hr_night_mean")
    hr_sleep = safe_col(combined, "wHr_sleep_hr_mean").fillna(safe_col(combined, "slp_hr_mean"))
    hr_std = safe_col(combined, "wHr_sleep_hr_std").fillna(safe_col(combined, "slp_hr_std"))
    hr_rmssd = safe_col(combined, "wHr_hr_rmssd_mean").fillna(safe_col(combined, "slp_hr_rmssd"))
    frag = safe_col(combined, "wHr_sleep_fragmentation_index").fillna(safe_col(combined, "slp_hr_spike_count"))
    sleep_eff = safe_col(combined, "wHr_sleep_efficiency_proxy")
    mlight_eve = safe_col(combined, "mlight_eve_mean")
    mlight_night = safe_col(combined, "mlight_night_mean")
    wlight_eve = safe_col(combined, "wlight_eve_mean")
    wlight_night = safe_col(combined, "wlight_night_mean")

    features["elr_log_steps"] = safe_log1p(steps)
    features["elr_log_evening_steps"] = safe_log1p(evening_steps)
    features["elr_evening_step_share"] = ratio(evening_steps, steps)
    features["elr_log_usage_total"] = safe_log1p(usage_total)
    features["elr_late_usage_share"] = ratio(usage_late, usage_total)
    features["elr_evening_usage_share"] = ratio(usage_eve, usage_total)
    features["elr_log_screen_total"] = safe_log1p(screen_total)
    features["elr_screen_late_pressure"] = screen_night_ratio.fillna(0) + screen_presleep_ratio.fillna(0)
    features["elr_dopamine_late_pressure"] = dopamine_ratio * features["elr_late_usage_share"]
    features["elr_app_entropy_pressure"] = app_entropy * features["elr_log_usage_total"]
    features["elr_mobility_load"] = moving_ratio * safe_log1p(total_disp)
    features["elr_home_recovery_proxy"] = homestay - moving_ratio
    features["elr_social_audio_load"] = speech.fillna(0) + music.fillna(0) - silence.fillna(0)
    features["elr_hr_day_minus_night"] = hr_day - hr_night
    features["elr_hr_day_minus_sleep"] = hr_day - hr_sleep
    features["elr_hr_recovery_quality"] = hr_rmssd - hr_std
    features["elr_hr_fragment_pressure"] = frag
    features["elr_sleep_efficiency_proxy"] = sleep_eff
    features["elr_light_evening_minus_night"] = mlight_eve.fillna(wlight_eve) - mlight_night.fillna(wlight_night)
    features["elr_dark_recovery_proxy"] = safe_col(combined, "mlight_dark_ratio").fillna(0) + safe_col(
        combined, "wlight_dark_ratio"
    ).fillna(0)
    features["elr_bright_pressure_proxy"] = safe_col(combined, "mlight_bright_ratio").fillna(0) + safe_col(
        combined, "wlight_bright_ratio"
    ).fillna(0)
    # Interaction terms: load followed by recovery or failed recovery.
    features["elr_load_x_hr_recovery"] = features["elr_log_steps"] * features["elr_hr_day_minus_night"]
    features["elr_late_screen_x_fragment"] = features["elr_screen_late_pressure"] * features["elr_hr_fragment_pressure"]
    features["elr_usage_x_sleep_eff"] = features["elr_log_usage_total"] * features["elr_sleep_efficiency_proxy"]
    features["elr_dopamine_x_hr_recovery"] = features["elr_dopamine_late_pressure"] * features["elr_hr_day_minus_sleep"]
    features["elr_social_audio_x_late_usage"] = features["elr_social_audio_load"] * features["elr_late_usage_share"]

    feature_cols = [col for col in features.columns if col.startswith("elr_")]
    for col in list(feature_cols):
        features[f"{col}_missing"] = features[col].isna().astype(float)
    feature_cols = [col for col in features.columns if col.startswith("elr_")]
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int((features["is_test"] == 0).sum()),
        "n_test_rows": int((features["is_test"] == 1).sum()),
        "n_feature_columns": int(len(feature_cols)),
        "base_columns_available": [col for col in BASE_COLS if col in combined.columns],
        "base_columns_missing": [col for col in BASE_COLS if col not in combined.columns],
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), feature_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), feature_cols].isna().mean().mean()),
        "top_missing_features": {
            str(k): float(v)
            for k, v in features.loc[features["is_test"].eq(0), feature_cols]
            .isna()
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .items()
        },
    }
    return features, coverage


def target_loss_on_split(
    train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], target: str, values: np.ndarray
) -> float:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    total = 0
    weighted = 0.0
    from sklearn.metrics import log_loss

    for _fit_idx, valid_idx in folds:
        pred = values[valid_idx, target_idx]
        weighted += float(log_loss(y[valid_idx], np.clip(pred, EPS, 1 - EPS), labels=[0, 1])) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def scan_target_configs(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    source_cache: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    references = {"id178": reference178, "id341": reference341}
    for target in TARGET_FOCUS:
        target_idx = TARGETS.index(target)
        id341_losses = {
            split_name: target_loss_on_split(train, splits[split_name], target, reference341)
            for split_name in PRIMARY_GATE_SPLITS
        }
        for spec in MODEL_SPECS:
            for split_name in PRIMARY_GATE_SPLITS:
                pred, _covered = predict_target_for_split(
                    features_train,
                    feature_cols,
                    train[target].astype(int).to_numpy(),
                    splits[split_name],
                    spec,
                )
                source_values = reference341.copy()
                source_values[:, target_idx] = np.where(np.isnan(pred), reference341[:, target_idx], pred)
                source_cache[(split_name, target, spec.name)] = source_values
            for anchor_name, anchor_values in references.items():
                for weight in BLEND_WEIGHTS:
                    split_deltas: dict[str, float] = {}
                    split_losses: dict[str, float] = {}
                    for split_name in PRIMARY_GATE_SPLITS:
                        source_values = source_cache[(split_name, target, spec.name)]
                        values = reference341.copy()
                        values[:, target_idx] = np.clip(
                            (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                            EPS,
                            1 - EPS,
                        )
                        loss = target_loss_on_split(train, splits[split_name], target, values)
                        split_losses[split_name] = loss
                        split_deltas[split_name] = float(loss - id341_losses[split_name])
                    deltas = list(split_deltas.values())
                    rows.append(
                        {
                            "target": target,
                            "model": spec.name,
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "candidate": f"{target}_{anchor_name}_{spec.name}_w{weight:g}",
                            "primary_wins_vs_id341": int(sum(delta < 0 for delta in deltas)),
                            "mean_delta_vs_id341": float(np.mean(deltas)),
                            "min_delta_vs_id341": float(np.min(deltas)),
                            "max_delta_vs_id341": float(np.max(deltas)),
                            "split_deltas_vs_id341": split_deltas,
                            "split_losses": split_losses,
                        }
                    )
    rows = sorted(
        rows,
        key=lambda row: (
            row["target"],
            -row["primary_wins_vs_id341"],
            row["mean_delta_vs_id341"],
            row["max_delta_vs_id341"],
        ),
    )
    return rows, source_cache


def select_target_configs(scan_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target in TARGET_FOCUS:
        target_rows = [row for row in scan_rows if row["target"] == target]
        eligible = [
            row
            for row in target_rows
            if row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
            and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
            and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
        ]
        if not eligible:
            continue
        selected[target] = sorted(eligible, key=lambda row: (row["mean_delta_vs_id341"], row["max_delta_vs_id341"]))[0]
    return selected


def source_predictions_for_selected_splits(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    selected_configs: dict[str, dict[str, Any]],
    reference341: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for split_name, folds in splits.items():
        for target, config in selected_configs.items():
            target_idx = TARGETS.index(target)
            spec = next(spec for spec in MODEL_SPECS if spec.name == config["model"])
            pred, _covered = predict_target_for_split(
                features_train, feature_cols, train[target].astype(int).to_numpy(), folds, spec
            )
            source_values = reference341.copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), reference341[:, target_idx], pred)
            out[(split_name, target)] = source_values[:, target_idx]
    return out


def full_test_sources_for_selected(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    selected_configs: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    n_train = len(train)
    n_total = len(features)
    fit_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n_total, dtype=int)
    out: dict[str, np.ndarray] = {}
    for target, config in selected_configs.items():
        spec = next(spec for spec in MODEL_SPECS if spec.name == config["model"])
        y = train[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            out[target] = np.full(len(test_idx), float(np.mean(y)), dtype=float)
            continue
        fit_x, test_x = fold_matrix(features, feature_cols, fit_idx, test_idx)
        model = make_model(spec)
        model.fit(fit_x, y)
        out[target] = np.clip(model.predict_proba(test_x)[:, 1], EPS, 1 - EPS)
    return out


def assemble_split_values(
    scale: float,
    split_name: str,
    selected_configs: dict[str, dict[str, Any]],
    split_sources: dict[tuple[str, str], np.ndarray],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> np.ndarray:
    values = reference341.copy()
    anchors = {"id178": reference178, "id341": reference341}
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        anchor = anchors[config["anchor"]][:, target_idx]
        source = split_sources[(split_name, target)]
        effective_weight = float(config["weight"] * scale)
        values[:, target_idx] = np.clip((1.0 - effective_weight) * anchor + effective_weight * source, EPS, 1 - EPS)
    return values


def assemble_test_values(
    scale: float,
    selected_configs: dict[str, dict[str, Any]],
    test_sources: dict[str, np.ndarray],
    base_tests: dict[int, np.ndarray],
) -> np.ndarray:
    values = base_tests[341].copy()
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        anchor_id = 178 if config["anchor"] == "id178" else 341
        anchor = base_tests[anchor_id][:, target_idx]
        source = test_sources[target]
        effective_weight = float(config["weight"] * scale)
        values[:, target_idx] = np.clip((1.0 - effective_weight) * anchor + effective_weight * source, EPS, 1 - EPS)
    return values


def evaluate_scale(
    scale: float,
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    selected_configs: dict[str, dict[str, Any]],
    split_sources: dict[tuple[str, str], np.ndarray],
    test_sources: dict[str, np.ndarray],
    reference178: np.ndarray,
    reference341: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    split_rows: dict[str, Any] = {}
    for split_name, folds in splits.items():
        values = assemble_split_values(scale, split_name, selected_configs, split_sources, reference178, reference341)
        candidate_eval = evaluate_values_on_split(train, folds, values)
        id341_eval = evaluate_values_on_split(train, folds, reference341)
        id178_eval = evaluate_values_on_split(train, folds, reference178)
        split_rows[split_name] = {
            "candidate_loss": candidate_eval["mean_loss"],
            "id341_loss": id341_eval["mean_loss"],
            "id178_loss": id178_eval["mean_loss"],
            "delta_vs_id341": float(candidate_eval["mean_loss"] - id341_eval["mean_loss"]),
            "delta_vs_id178": float(candidate_eval["mean_loss"] - id178_eval["mean_loss"]),
            "beats_id341": bool(candidate_eval["mean_loss"] < id341_eval["mean_loss"]),
        }
    primary_passes = {
        split_name: split_rows[split_name]["beats_id341"]
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_rows
    }
    local_cv = float(split_rows["subject_hole_5"]["candidate_loss"])
    test_values = assemble_test_values(scale, selected_configs, test_sources, base_tests)
    geometry = geometry_report(test_values, base_tests, local_cv)
    reviewable = bool(
        primary_passes
        and all(primary_passes.values())
        and geometry["proxy_public_risk"] < ID341_PROXY - REVIEW_PROXY_MARGIN
        and geometry["max_delta_vs_341"] <= REVIEW_MAX_DELTA_VS_341
    )
    return {
        "scale": scale,
        "effective_weights": {target: float(config["weight"] * scale) for target, config in selected_configs.items()},
        "split_rows": split_rows,
        "primary_passes_vs_id341": primary_passes,
        "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
        "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
        "local_cv_subject_hole_5": local_cv,
        "test_geometry": geometry,
        "reviewable_low_movement_candidate": reviewable,
    }


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    features, feature_coverage = build_evening_load_recovery_features(train, test)
    feature_cols = [col for col in features.columns if col.startswith("elr_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows, _source_cache = scan_target_configs(
        train, features_train, feature_cols, splits, reference178, reference341
    )
    selected_configs = select_target_configs(scan_rows)
    scale_rows: list[dict[str, Any]] = []
    if selected_configs:
        base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}
        split_sources = source_predictions_for_selected_splits(
            train, features_train, feature_cols, splits, selected_configs, reference341
        )
        test_sources = full_test_sources_for_selected(train, features, feature_cols, selected_configs)
        scale_rows = [
            evaluate_scale(
                scale,
                train,
                splits,
                selected_configs,
                split_sources,
                test_sources,
                reference178,
                reference341,
                base_tests,
            )
            for scale in SCALES
        ]
    reviewable_rows = [row for row in scale_rows if row["reviewable_low_movement_candidate"]]
    best_reviewable = (
        sorted(reviewable_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["scale"]))[0]
        if reviewable_rows
        else None
    )
    best_proxy = (
        sorted(scale_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"]))[
            0
        ]
        if scale_rows
        else None
    )
    top_scan_rows = {target: [row for row in scan_rows if row["target"] == target][:10] for target in TARGET_FOCUS}
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "evening_load_recovery_smoke_post_id341",
        "purpose": "Evaluate independent evening_load_recovery_v1 features under frozen post-ID341 validation gates.",
        "feature_coverage": feature_coverage,
        "config_rules": {
            "target_focus": TARGET_FOCUS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "scales": SCALES,
            "blend_weights": BLEND_WEIGHTS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        },
        "selected_target_configs": selected_configs,
        "top_scan_rows_by_target": top_scan_rows,
        "scale_rows": scale_rows,
        "best_reviewable_low_movement": best_reviewable,
        "best_by_proxy": best_proxy,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "selected_targets": sorted(selected_configs.keys()),
            "reviewable_low_movement_candidate_exists": bool(best_reviewable is not None),
            "best_reviewable_scale": None if best_reviewable is None else best_reviewable["scale"],
            "best_proxy": None if best_proxy is None else best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_scale": None if best_proxy is None else best_proxy["scale"],
            "public_058_plausible": False
            if best_proxy is None
            else bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "next_action": (
                "If evening-load has a reviewable row, combine it with the sleep-boundary scale-0.05 lead in a no-artifact combined gate. "
                "Otherwise keep it as rejected context and try routine-stability/place entropy."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    lines = [
        "# Evening-load/recovery smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Feature columns: `{coverage['n_feature_columns']}`",
        f"- Mean missing rate: train `{coverage['train_missing_mean']:.4f}`, test `{coverage['test_missing_mean']:.4f}`",
        f"- Available base columns: `{len(coverage['base_columns_available'])}`; missing base columns: `{len(coverage['base_columns_missing'])}`",
        "",
        "## Selected target configs",
        "",
        "| target | candidate | wins | mean Δ | max Δ |",
        "|---|---|---:|---:|---:|",
    ]
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    if report["scale_rows"]:
        lines.extend(
            [
                "",
                "## Scale scan",
                "",
                "| scale | primary wins | all primary | local | proxy | Δ341 mean/max | failed proj | reviewable |",
                "|---:|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in report["scale_rows"]:
            geom = row["test_geometry"]
            lines.append(
                f"| {row['scale']:.2f} | {row['primary_wins_vs_id341']} | `{row['all_primary_splits_beat_id341']}` | "
                f"{row['local_cv_subject_hole_5']:.6f} | {geom['proxy_public_risk']:.6f} | "
                f"{geom['mean_delta_vs_341']:.6f}/{geom['max_delta_vs_341']:.6f} | "
                f"{geom['projection_failed']:+.6f} | `{row['reviewable_low_movement_candidate']}` |"
            )
    lines.extend(["", "## Decision", ""])
    best_reviewable = report["best_reviewable_low_movement"]
    if best_reviewable is None:
        lines.append("- Reviewable low-movement candidate: `False`")
    else:
        geom = best_reviewable["test_geometry"]
        lines.extend(
            [
                "- Reviewable low-movement candidate: `True`",
                f"- Best reviewable scale: `{best_reviewable['scale']:.2f}`",
                f"- Best reviewable proxy: `{geom['proxy_public_risk']:.6f}`",
                f"- Best reviewable max Δ vs ID341: `{geom['max_delta_vs_341']:.6f}`",
            ]
        )
    if report["best_by_proxy"] is not None:
        lines.append(
            f"- Best proxy: `{report['best_by_proxy']['test_geometry']['proxy_public_risk']:.6f}` at scale `{report['best_by_proxy']['scale']:.2f}`"
        )
    lines.extend(
        [
            f"- Public `0.58` plausible: `{report['decision']['public_058_plausible']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "evening_load_recovery_smoke_post_id341_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
