#!/usr/bin/env python3
"""Post-ID342 AirKorea 2024 air-quality context gate.

This report-only diagnostic tests the downloaded historical AirKorea hourly
Daejeon cache against ID342. It builds date-aligned pollution features in
memory and creates no persisted feature matrix, prediction file, submission
CSV, submission-log row, Kaggle action, or DACON submission.
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

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import TARGETS  # noqa: E402
from scripts.diagnose_post_id342_processed_sleep_proxy_gate import (  # noqa: E402
    ANCHOR_ID,
    EPS,
    ID342_PUBLIC,
    MODEL_SPECS,
    PRIMARY_SPLITS,
    PUBLIC_055,
    TARGET_MAX_DELTA_CEILING,
    TARGET_MEAN_DELTA_CEILING,
    TARGET_WIN_FLOOR,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_split,
    rel,
    weighted_loss,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_AIRKOREA_CACHE = (
    ROOT / "data" / "external" / "weather" / "airkorea_2024" / "airkorea_2024_daejeon_hourly.csv"
)
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_airkorea_2024_context_gate_20260625.json"

FOCUS_TARGETS = ("Q3", "S3", "S4")
BLEND_WEIGHTS = (0.005, 0.01, 0.025, 0.05)
REVIEW_PUBLIC_THRESHOLD = 0.58
DACON_PUBLIC_TARGET = 0.55
FAILED_FRONTIER_IDS = (287, 329)
MAX_FAILED_DIRECTION_CORR = 0.20
MAX_FEATURES_PER_FAMILY = 160
HGB_MAX_FEATURES = 120

POLLUTANTS = ("so2", "co", "o3", "no2", "pm10", "pm25")
SELECTED_STATIONS = {
    "dunsan": "둔산동",
    "guseong": "구성동",
    "wolpyeong": "월평동",
}
POLLUTION_THRESHOLDS = {
    "pm10": (50.0, 80.0),
    "pm25": (25.0, 35.0),
    "o3": (0.06, 0.08),
    "no2": (0.03, 0.05),
    "co": (0.5, 0.8),
    "so2": (0.003, 0.005),
}
WINDOWS = {
    "day_12_23": (0, 12, 0, 23),
    "night_00_11": (1, 0, 1, 11),
    "sleep_core_22_06": (0, 22, 1, 6),
    "full_lifelog_12_11": (0, 12, 1, 11),
}


def numeric_cols(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in {"subject_id", "date"}
        and pd.api.types.is_numeric_dtype(frame[col])
        and frame[col].notna().any()
    ]


def feature_priority(col: str) -> tuple[int, str]:
    score = 0
    for rank, token in enumerate(("pm25", "pm10", "o3", "no2", "_co_", "so2")):
        if token in col:
            score += (6 - rank) * 100
            break
    for rank, token in enumerate(("sleep_core", "full_lifelog", "night_00_11", "day_12_23")):
        if token in col:
            score += (4 - rank) * 40
            break
    for rank, token in enumerate(("allmean", "dunsan", "guseong", "wolpyeong", "allmax", "station_spread")):
        if token in col:
            score += (6 - rank) * 20
            break
    for rank, token in enumerate(("mean", "max", "p95", "hours_ge", "std", "range", "__lag1", "__dev_past3")):
        if token in col:
            score += (8 - rank) * 5
            break
    return -score, col


def compact_feature_cols(features: pd.DataFrame, cols: list[str], max_cols: int) -> list[str]:
    usable = [col for col in cols if col in features and features[col].notna().any()]
    ranked = sorted(usable, key=feature_priority)
    if len(ranked) <= max_cols:
        return ranked
    return ranked[:max_cols]


def load_airkorea(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    air = pd.read_csv(path)
    required = {"station_code", "station_name", "datetime", *POLLUTANTS}
    missing = sorted(required - set(air.columns))
    if missing:
        raise ValueError(f"AirKorea cache is missing required columns: {missing}")
    air = air.copy()
    air["datetime"] = pd.to_datetime(air["datetime"], errors="coerce")
    air = air.loc[air["datetime"].notna()].reset_index(drop=True)
    air["station_code"] = air["station_code"].astype(str)
    for col in POLLUTANTS:
        air[col] = pd.to_numeric(air[col], errors="coerce")
    window = air.loc[(air["datetime"] >= "2024-06-01") & (air["datetime"] <= "2024-11-22 23:00:00")]
    metadata = {
        "airkorea_cache": rel(path),
        "cache_rows": int(len(air)),
        "columns": list(air.columns),
        "time_min": None if air.empty else str(air["datetime"].min()),
        "time_max": None if air.empty else str(air["datetime"].max()),
        "station_count": int(air["station_code"].nunique()),
        "station_names": sorted(str(value) for value in air["station_name"].dropna().unique()),
        "pollutants": list(POLLUTANTS),
        "competition_window_rows": int(len(window)),
        "competition_window_unique_hours": int(window["datetime"].nunique()),
        "competition_window_station_count": int(window["station_code"].nunique()),
        "source_note": "Historical 2024 AirKorea hourly Daejeon rows extracted from the user-provided archive.",
        "secret_values_stored": False,
    }
    return air, metadata


def safe_std(series: pd.Series) -> float:
    value = series.std()
    if pd.isna(value):
        return 0.0
    return float(value)


def add_hourly_pollutant_features(row: dict[str, float], hourly: pd.DataFrame, prefix: str) -> None:
    for pollutant in POLLUTANTS:
        if pollutant not in hourly:
            continue
        series = hourly[pollutant].dropna()
        if series.empty:
            continue
        row[f"air_{prefix}_{pollutant}_mean"] = float(series.mean())
        row[f"air_{prefix}_{pollutant}_max"] = float(series.max())
        row[f"air_{prefix}_{pollutant}_min"] = float(series.min())
        row[f"air_{prefix}_{pollutant}_std"] = safe_std(series)
        row[f"air_{prefix}_{pollutant}_p95"] = float(series.quantile(0.95))
        row[f"air_{prefix}_{pollutant}_range"] = float(series.max() - series.min())
        for threshold in POLLUTION_THRESHOLDS[pollutant]:
            threshold_tag = str(threshold).replace(".", "p")
            row[f"air_{prefix}_{pollutant}_hours_ge_{threshold_tag}"] = float((series >= threshold).sum())


def aggregate_one_window(air: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> dict[str, float]:
    subset = air.loc[(air["datetime"] >= start) & (air["datetime"] <= end)].copy()
    row: dict[str, float] = {}
    if subset.empty:
        return row

    hourly_mean = subset.groupby("datetime", sort=True)[list(POLLUTANTS)].mean()
    add_hourly_pollutant_features(row, hourly_mean, f"{prefix}_allmean")
    hourly_max = subset.groupby("datetime", sort=True)[list(POLLUTANTS)].max()
    add_hourly_pollutant_features(row, hourly_max, f"{prefix}_allmax")

    for pollutant in POLLUTANTS:
        station_hourly = subset.pivot_table(
            index="datetime",
            columns="station_code",
            values=pollutant,
            aggfunc="mean",
        )
        if station_hourly.empty:
            continue
        spread = station_hourly.max(axis=1) - station_hourly.min(axis=1)
        station_std = station_hourly.std(axis=1)
        row[f"air_{prefix}_{pollutant}_station_spread_mean"] = float(spread.mean())
        row[f"air_{prefix}_{pollutant}_station_spread_max"] = float(spread.max())
        row[f"air_{prefix}_{pollutant}_station_std_mean"] = float(station_std.mean())
        row[f"air_{prefix}_{pollutant}_coverage_rate"] = float(station_hourly.notna().mean().mean())

    for tag, station_name in SELECTED_STATIONS.items():
        station = subset.loc[subset["station_name"] == station_name]
        if station.empty:
            continue
        hourly_station = station.groupby("datetime", sort=True)[list(POLLUTANTS)].mean()
        add_hourly_pollutant_features(row, hourly_station, f"{prefix}_{tag}")
    return row


def build_daily_airkorea_features(keys: pd.DataFrame, air: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    unique_dates = sorted(pd.to_datetime(keys["date"]).dt.date.unique())
    date_rows: list[dict[str, Any]] = []
    for date_value in unique_dates:
        base = pd.Timestamp(date_value)
        row: dict[str, Any] = {"date": date_value}
        for prefix, (start_day, start_hour, end_day, end_hour) in WINDOWS.items():
            start = base + pd.Timedelta(days=start_day, hours=start_hour)
            end = base + pd.Timedelta(days=end_day, hours=end_hour)
            row.update(aggregate_one_window(air, start, end, prefix))
        date_rows.append(row)

    daily = pd.DataFrame(date_rows)
    out = keys[["subject_id", "date"]].copy().reset_index(drop=True)
    out = out.merge(daily, on="date", how="left", validate="many_to_one")
    out = out.sort_values(["subject_id", "date"]).reset_index(drop=True)

    base_cols = numeric_cols(out)
    lag_data: dict[str, pd.Series] = {}
    for col in base_cols:
        lag = out.groupby("subject_id", sort=False)[col].shift(1)
        past3 = lag.groupby(out["subject_id"], sort=False).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        lag_data[f"{col}__lag1"] = lag
        lag_data[f"{col}__dev_past3"] = out[col] - past3
    out = pd.concat([out, pd.DataFrame(lag_data)], axis=1)

    all_cols = numeric_cols(out)
    particulate_cols = [col for col in all_cols if any(token in col for token in ("pm10", "pm25"))]
    oxidant_traffic_cols = [col for col in all_cols if any(token in col for token in ("o3", "no2", "_co_"))]
    station_selected_cols = [col for col in all_cols if any(f"_{tag}" in col for tag in SELECTED_STATIONS)]
    dunsan_cols = [col for col in all_cols if "_dunsan" in col]
    dispersion_cols = [
        col
        for col in all_cols
        if any(token in col for token in ("station_spread", "station_std", "coverage_rate", "allmax"))
    ]
    families = {
        "airkorea_particulate_burden": particulate_cols,
        "airkorea_oxidant_traffic": oxidant_traffic_cols,
        "airkorea_selected_station": station_selected_cols,
        "airkorea_dunsan_station": dunsan_cols,
        "airkorea_station_dispersion": dispersion_cols,
        "airkorea_all": all_cols,
    }
    metadata = {
        "rows": int(len(out)),
        "unique_dates": int(pd.Series(out["date"]).nunique()),
        "windows": WINDOWS,
        "selected_stations": SELECTED_STATIONS,
        "thresholds": POLLUTION_THRESHOLDS,
        "base_feature_columns": int(len(base_cols)),
        "lag_deviation_columns": int(len(lag_data)),
        "families": {
            name: {
                "feature_count": int(len(cols)),
                "feature_sample": cols[:16],
                "mean_missing_rate": float(out[cols].isna().mean().mean()) if cols else None,
            }
            for name, cols in families.items()
        },
    }
    return out, metadata


def movement_corr(candidate: np.ndarray, anchor: np.ndarray, failed: np.ndarray) -> float | None:
    movement = candidate - anchor
    failed_movement = failed - anchor
    if float(np.nanstd(movement)) <= 1e-12 or float(np.nanstd(failed_movement)) <= 1e-12:
        return None
    corr = np.corrcoef(movement, failed_movement)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def load_failed_refs(train: pd.DataFrame) -> tuple[dict[int, np.ndarray], list[int]]:
    refs: dict[int, np.ndarray] = {}
    missing: list[int] = []
    for failed_id in FAILED_FRONTIER_IDS:
        try:
            refs[failed_id] = load_reference_values(train, failed_id)
        except FileNotFoundError:
            missing.append(failed_id)
    return refs, missing


def scan_family(
    family_name: str,
    features: pd.DataFrame,
    feature_cols: list[str],
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    failed_refs: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not feature_cols:
        return rows
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            if spec.kind == "hgb" and len(feature_cols) > HGB_MAX_FEATURES:
                model_feature_cols = feature_cols[:HGB_MAX_FEATURES]
            else:
                model_feature_cols = feature_cols
            split_predictions: dict[str, np.ndarray] = {}
            split_coverage: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_split(features, model_feature_cols, y, splits[split_name], spec)
                split_predictions[split_name] = pred
                split_coverage[split_name] = int(covered)
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_failed_corr: dict[str, dict[str, float | None]] = {}
                wins = 0
                max_failed_corr = -1.0
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    blended = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * split_predictions[split_name],
                        EPS,
                        1.0 - EPS,
                    )
                    anchor_loss = weighted_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                    corr_by_id: dict[str, float | None] = {}
                    for failed_id, failed_values in failed_refs.items():
                        corr = movement_corr(blended, anchor_values[:, target_idx], failed_values[:, target_idx])
                        corr_by_id[f"id{failed_id}"] = corr
                        if corr is not None:
                            max_failed_corr = max(max_failed_corr, corr)
                    split_failed_corr[split_name] = corr_by_id
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                projected_public_proxy = float(ID342_PUBLIC + mean_delta / len(TARGETS))
                failed_corr_for_gate = 0.0 if max_failed_corr < -0.5 else float(max_failed_corr)
                local_gate_pass = bool(
                    wins >= TARGET_WIN_FLOOR
                    and mean_delta <= TARGET_MEAN_DELTA_CEILING
                    and max_delta <= TARGET_MAX_DELTA_CEILING
                    and failed_corr_for_gate <= MAX_FAILED_DIRECTION_CORR
                )
                review_gate_pass = bool(local_gate_pass and projected_public_proxy <= REVIEW_PUBLIC_THRESHOLD)
                dacon_gate_pass = bool(local_gate_pass and projected_public_proxy <= DACON_PUBLIC_TARGET)
                rows.append(
                    {
                        "family_name": family_name,
                        "target": target,
                        "config": f"{family_name}_{target.lower()}_{spec.name}_w{weight:g}",
                        "feature_count": int(len(feature_cols)),
                        "model_feature_count": int(len(model_feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_rows_by_split": split_coverage,
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "projected_public_proxy": projected_public_proxy,
                        "gap_to_public_055": float(projected_public_proxy - PUBLIC_055),
                        "split_deltas": split_deltas,
                        "failed_direction_corr": split_failed_corr,
                        "max_failed_direction_corr": None if max_failed_corr < -0.5 else float(max_failed_corr),
                        "local_gate_pass": local_gate_pass,
                        "review_gate_pass": review_gate_pass,
                        "dacon_gate_pass": dacon_gate_pass,
                    }
                )
    return rows


def missing_cache_report(airkorea_cache: Path) -> dict[str, Any]:
    decision = {
        "airkorea_cache_available": False,
        "reviewed_candidates": 0,
        "local_gate_pass_count": 0,
        "review_gate_pass_count": 0,
        "dacon_gate_pass_count": 0,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": "Restore the ignored AirKorea 2024 Daejeon hourly cache, then rerun this report-only gate.",
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_airkorea_2024_context_gate",
        "purpose": "Gate historical Daejeon AirKorea pollution context against ID342.",
        "airkorea_cache": rel(airkorea_cache),
        "airkorea_cache_exists": False,
        "decision": decision,
        "top_rows": [],
        "no_prediction_file_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    airkorea_cache = args.airkorea_cache if args.airkorea_cache.is_absolute() else ROOT / args.airkorea_cache
    if not airkorea_cache.exists():
        return missing_cache_report(airkorea_cache)

    train = load_train_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    failed_refs, missing_failed_refs = load_failed_refs(train)
    air, air_metadata = load_airkorea(airkorea_cache)
    features, feature_metadata = build_daily_airkorea_features(train[["subject_id", "date"]], air)
    all_feature_cols = numeric_cols(features)
    family_columns = {
        "airkorea_particulate_burden": [col for col in all_feature_cols if any(token in col for token in ("pm10", "pm25"))],
        "airkorea_oxidant_traffic": [
            col for col in all_feature_cols if any(token in col for token in ("o3", "no2", "_co_"))
        ],
        "airkorea_selected_station": [col for col in all_feature_cols if any(f"_{tag}" in col for tag in SELECTED_STATIONS)],
        "airkorea_dunsan_station": [col for col in all_feature_cols if "_dunsan" in col],
        "airkorea_station_dispersion": [
            col
            for col in all_feature_cols
            if any(token in col for token in ("station_spread", "station_std", "coverage_rate", "allmax"))
        ],
        "airkorea_all": all_feature_cols,
    }
    family_columns = {
        family_name: compact_feature_cols(features, feature_cols, MAX_FEATURES_PER_FAMILY)
        for family_name, feature_cols in family_columns.items()
    }
    for family_name, compact_cols in family_columns.items():
        if family_name in feature_metadata["families"]:
            feature_metadata["families"][family_name]["model_feature_count"] = int(len(compact_cols))
            feature_metadata["families"][family_name]["model_feature_sample"] = compact_cols[:16]

    rows: list[dict[str, Any]] = []
    for family_name, feature_cols in family_columns.items():
        rows.extend(scan_family(family_name, features, feature_cols, train, splits, anchor_values, failed_refs))
    rows = sorted(
        rows,
        key=lambda row: (
            not row["dacon_gate_pass"],
            not row["review_gate_pass"],
            not row["local_gate_pass"],
            row["projected_public_proxy"],
            row["mean_delta"],
            row["max_delta"],
        ),
    )
    local_pass = [row for row in rows if row["local_gate_pass"]]
    review_pass = [row for row in rows if row["review_gate_pass"]]
    dacon_pass = [row for row in rows if row["dacon_gate_pass"]]
    best = rows[0] if rows else None
    decision = {
        "airkorea_cache_available": True,
        "reviewed_candidates": int(len(rows)),
        "local_gate_pass_count": int(len(local_pass)),
        "review_gate_pass_count": int(len(review_pass)),
        "dacon_gate_pass_count": int(len(dacon_pass)),
        "best_candidate": best["config"] if best else None,
        "best_family": best["family_name"] if best else None,
        "best_target": best["target"] if best else None,
        "best_wins": best["wins"] if best else None,
        "best_mean_delta": best["mean_delta"] if best else None,
        "best_max_delta": best["max_delta"] if best else None,
        "best_projected_public_proxy": best["projected_public_proxy"] if best else ID342_PUBLIC,
        "best_gap_to_public_055": best["gap_to_public_055"] if best else ID342_PUBLIC - PUBLIC_055,
        "missing_failed_frontier_refs": missing_failed_refs,
        "airkorea_2024_context_gate_pass": bool(review_pass),
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Promote nothing unless a row clears the Public-0.58 review proxy and a separate artifact contract. "
            "This gate is external-source evidence only."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_airkorea_2024_context_gate",
        "purpose": "Gate historical Daejeon AirKorea pollution context against ID342.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "airkorea_cache": rel(airkorea_cache),
        "airkorea_cache_exists": True,
        "airkorea_metadata": air_metadata,
        "feature_metadata": feature_metadata,
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "focus_targets": list(FOCUS_TARGETS),
            "blend_weights": list(BLEND_WEIGHTS),
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "max_failed_direction_corr": MAX_FAILED_DIRECTION_CORR,
            "review_public_threshold": REVIEW_PUBLIC_THRESHOLD,
            "dacon_public_target": DACON_PUBLIC_TARGET,
        },
        "top_rows": rows[:40],
        "decision": decision,
        "no_prediction_file_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID342 AirKorea 2024 context gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- AirKorea cache available: `{decision['airkorea_cache_available']}`",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate passes: `{decision['local_gate_pass_count']}`",
        f"- Review gate passes: `{decision['review_gate_pass_count']}`",
        f"- DACON gate passes: `{decision['dacon_gate_pass_count']}`",
        f"- Best candidate: `{decision.get('best_candidate')}`",
        f"- Best target/family: `{decision.get('best_target')}` / `{decision.get('best_family')}`",
        f"- Best projected Public proxy: `{decision.get('best_projected_public_proxy', ID342_PUBLIC):.9f}`",
        f"- Gate pass: `{decision.get('airkorea_2024_context_gate_pass', False)}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
    ]
    if not report.get("airkorea_cache_exists"):
        return "\n".join(lines)
    lines.extend(
        [
            "## AirKorea Cache",
            "",
            "```json",
            json.dumps(report["airkorea_metadata"], indent=2, sort_keys=True, ensure_ascii=False),
            "```",
            "",
            "## Feature Families",
            "",
            "| family | features | mean missing rate | sample |",
            "|---|---:|---:|---|",
        ]
    )
    for family_name, metadata in report["feature_metadata"]["families"].items():
        sample = ", ".join(f"`{col}`" for col in metadata["feature_sample"][:8])
        missing = metadata["mean_missing_rate"]
        missing_text = "" if missing is None else f"{missing:.6f}"
        lines.append(f"| `{family_name}` | {metadata['feature_count']} | {missing_text} | {sample} |")
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| rank | local | review | dacon | family | target | config | wins | mean delta | max delta | proxy | gap 0.55 | max failed corr |",
            "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(report["top_rows"][:24], start=1):
        max_failed_corr = row["max_failed_direction_corr"]
        corr_text = "" if max_failed_corr is None else f"{max_failed_corr:+.6f}"
        lines.append(
            f"| {rank} | `{row['local_gate_pass']}` | `{row['review_gate_pass']}` | `{row['dacon_gate_pass']}` | "
            f"`{row['family_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
            f"{row['mean_delta']:+.9f} | {row['max_delta']:+.9f} | {row['projected_public_proxy']:.9f} | "
            f"{row['gap_to_public_055']:+.9f} | {corr_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airkorea-cache", type=Path, default=DEFAULT_AIRKOREA_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "AirKorea cache available: "
        f"{decision['airkorea_cache_available']} | review_pass={decision.get('airkorea_2024_context_gate_pass', False)}"
    )


if __name__ == "__main__":
    main()
