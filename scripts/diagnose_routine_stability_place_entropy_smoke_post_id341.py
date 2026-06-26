#!/usr/bin/env python3
"""Report-only routine-stability/place-entropy source smoke after ID341.

This diagnostic implements the third post-ID341 new-source family from
`post_id341_new_source_blueprint_20260618`: routine_stability_place_entropy_v1.
It builds label-free previous-day/rolling routine deviation features from
place, usage, screen, activity, and movement aggregates, evaluates small source
models inside the frozen ID340 split manifest, then runs a conservative
scale/geometry gate against ID178/ID341 anchors.

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

from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (  # noqa: E402
    SCALES,
    evaluate_scale,
    full_test_sources_for_selected,
    safe_col,
    source_predictions_for_selected_splits,
    target_loss_on_split,
)
from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    BLEND_WEIGHTS,
    EPS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    load_reference_values,
    load_split_manifest,
    predict_target_for_split,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import load_base_test_values  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
TARGET_FOCUS = ["Q1", "Q2", "Q3", "S3"]
ANCHOR_IDS = [178, 287, 329, 341]
CONFIG_WIN_FLOOR = 4
CONFIG_MAX_DELTA_CEILING = 0.0010
CONFIG_MEAN_DELTA_CEILING = -0.00020

BASE_COLS = [
    "mGps_unique_locations",
    "mGps_radius_of_gyration",
    "mGps_spatial_entropy",
    "mGps_homestay_ratio",
    "gps_moving_ratio",
    "gps_total_disp",
    "gps_mean_speed",
    "wifi_n_unique",
    "wifi_mean_rssi",
    "wifi_max_rssi",
    "ble_n_unique",
    "ble_n_scans",
    "mUsage_total_time_mins",
    "mUsage_night_time_ratio",
    "mUsage_dopamine_ratio",
    "mUsage_app_entropy",
    "mUsage_hourly_volatility",
    "mUsage_sessions_count",
    "usage_total_time",
    "usage_late_ratio",
    "usage_eve_ratio",
    "screen_on_total",
    "screen_on_ratio",
    "screen_night_ratio",
    "screen_eve_ratio",
    "screen_presleep_ratio",
    "act_active_ratio",
    "act_still_ratio",
    "act_night_active",
    "act_night_still",
    "pedo_total_steps",
    "pedo_evening_steps",
    "pedo_active_hours",
    "amb_music_mean",
    "amb_speech_mean",
    "amb_silence_mean",
    "dayofweek",
    "is_weekend",
]


def build_routine_stability_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = pd.concat(
        [train[["subject_id", "date"]].assign(is_test=0), test[["subject_id", "date"]].assign(is_test=1)],
        ignore_index=True,
    )
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    features = rows.copy()
    features["date_dt"] = pd.to_datetime(features["date"])
    for col in BASE_COLS:
        if col in combined.columns:
            values = pd.to_numeric(combined[col], errors="coerce")
            if col in {
                "mUsage_total_time_mins",
                "usage_total_time",
                "screen_on_total",
                "gps_total_disp",
                "pedo_total_steps",
                "pedo_evening_steps",
            }:
                values = np.log1p(values.clip(lower=0))
            features[f"rsp_raw_{col}"] = values
    raw_cols = [col for col in features.columns if col.startswith("rsp_raw_")]
    features = features.sort_values(["subject_id", "date_dt"]).reset_index(drop=True)
    for col in raw_cols:
        grouped = features.groupby("subject_id", sort=False)[col]
        prev = grouped.shift(1)
        roll3 = grouped.transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
        roll7 = grouped.transform(lambda series: series.shift(1).rolling(7, min_periods=1).mean())
        features[f"{col}_prev_delta"] = features[col] - prev
        features[f"{col}_abs_prev_delta"] = (features[col] - prev).abs()
        features[f"{col}_roll3_delta"] = features[col] - roll3
        features[f"{col}_abs_roll3_delta"] = (features[col] - roll3).abs()
        features[f"{col}_roll7_delta"] = features[col] - roll7
        features[f"{col}_abs_roll7_delta"] = (features[col] - roll7).abs()
    # Compact semantic composites.
    features["rsp_place_entropy_shift"] = safe_col(features, "rsp_raw_mGps_spatial_entropy_abs_roll3_delta") + safe_col(
        features, "rsp_raw_wifi_n_unique_abs_roll3_delta"
    ).fillna(0)
    features["rsp_home_routine_break"] = -safe_col(features, "rsp_raw_mGps_homestay_ratio_roll3_delta") + safe_col(
        features, "rsp_raw_gps_moving_ratio_roll3_delta"
    ).fillna(0)
    features["rsp_screen_usage_routine_shift"] = safe_col(
        features, "rsp_raw_screen_on_total_abs_roll3_delta"
    ) + safe_col(features, "rsp_raw_usage_total_time_abs_roll3_delta").fillna(0)
    features["rsp_late_routine_shift"] = safe_col(features, "rsp_raw_usage_late_ratio_abs_roll3_delta") + safe_col(
        features, "rsp_raw_screen_night_ratio_abs_roll3_delta"
    ).fillna(0)
    features["rsp_activity_routine_shift"] = safe_col(features, "rsp_raw_act_active_ratio_abs_roll3_delta") + safe_col(
        features, "rsp_raw_pedo_total_steps_abs_roll3_delta"
    ).fillna(0)
    features["rsp_social_context_shift"] = safe_col(features, "rsp_raw_amb_speech_mean_abs_roll3_delta") + safe_col(
        features, "rsp_raw_amb_music_mean_abs_roll3_delta"
    ).fillna(0)
    features["rsp_home_x_late_shift"] = features["rsp_home_routine_break"] * features["rsp_late_routine_shift"]
    features["rsp_entropy_x_screen_shift"] = (
        features["rsp_place_entropy_shift"] * features["rsp_screen_usage_routine_shift"]
    )
    feature_cols = [col for col in features.columns if col.startswith("rsp_")]
    for col in list(feature_cols):
        features[f"{col}_missing"] = features[col].isna().astype(float)
    feature_cols = [col for col in features.columns if col.startswith("rsp_")]
    features = features.sort_index().reset_index(drop=True)
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int((features["is_test"] == 0).sum()),
        "n_test_rows": int((features["is_test"] == 1).sum()),
        "n_feature_columns": int(len(feature_cols)),
        "raw_columns": int(len(raw_cols)),
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


def scan_target_configs(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    references = {"id178": reference178, "id341": reference341}
    for target in TARGET_FOCUS:
        target_idx = TARGETS.index(target)
        id341_losses = {
            split_name: target_loss_on_split(train, splits[split_name], target, reference341)
            for split_name in PRIMARY_GATE_SPLITS
        }
        for spec in MODEL_SPECS:
            source_cache: dict[str, np.ndarray] = {}
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
                source_cache[split_name] = source_values
            for anchor_name, anchor_values in references.items():
                for weight in BLEND_WEIGHTS:
                    split_deltas: dict[str, float] = {}
                    split_losses: dict[str, float] = {}
                    for split_name in PRIMARY_GATE_SPLITS:
                        source_values = source_cache[split_name]
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
    return sorted(
        rows,
        key=lambda row: (
            row["target"],
            -row["primary_wins_vs_id341"],
            row["mean_delta_vs_id341"],
            row["max_delta_vs_id341"],
        ),
    )


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
        if eligible:
            selected[target] = sorted(
                eligible, key=lambda row: (row["mean_delta_vs_id341"], row["max_delta_vs_id341"])
            )[0]
    return selected


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    features, feature_coverage = build_routine_stability_features(train, test)
    feature_cols = [col for col in features.columns if col.startswith("rsp_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows = scan_target_configs(train, features_train, feature_cols, splits, reference178, reference341)
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
        "diagnostic": "routine_stability_place_entropy_smoke_post_id341",
        "purpose": "Evaluate routine_stability_place_entropy_v1 features under frozen post-ID341 validation gates.",
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
            "next_action": "If routine-stability has no proxy-below-ID178 row, return to upstream rebuild or design a stronger raw source before artifact promotion.",
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
        "# Routine-stability/place-entropy smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Feature columns: `{coverage['n_feature_columns']}` from `{coverage['raw_columns']}` raw routine columns",
        f"- Mean missing rate: train `{coverage['train_missing_mean']:.4f}`, test `{coverage['test_missing_mean']:.4f}`",
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
        default=REPORTS_DIR / "routine_stability_place_entropy_smoke_post_id341_20260618.json",
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
