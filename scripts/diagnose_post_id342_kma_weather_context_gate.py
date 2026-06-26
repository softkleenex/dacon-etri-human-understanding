#!/usr/bin/env python3
"""Post-ID342 KMA ASOS weather-context gate.

This report-only diagnostic tests a materially external weather source against
ID342. It reads an ignored local KMA ASOS hourly cache produced by
``scripts/fetch_kma_asos_hourly_context.py`` and creates no feature artifact,
prediction file, submission CSV, submission-log row, Kaggle action, or DACON
submission.
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
DEFAULT_WEATHER_CACHE = ROOT / "data" / "external" / "weather" / "kma_asos_hourly_20240601_20241122.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_kma_weather_context_gate_20260625.json"

FOCUS_TARGETS = ("Q3", "S3", "S4")
BLEND_WEIGHTS = (0.005, 0.01, 0.025, 0.05)
REVIEW_PUBLIC_THRESHOLD = 0.58
DACON_PUBLIC_TARGET = 0.55
FAILED_FRONTIER_IDS = (287, 329)
MAX_FAILED_DIRECTION_CORR = 0.20

WEATHER_NUMERIC_COLUMNS = ("ta", "rn", "ws", "hm", "pa", "ps", "ss", "icsr", "dsnw", "ts")
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


def load_weather(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    weather = pd.read_csv(path)
    if "tm" not in weather.columns:
        raise ValueError(f"KMA ASOS cache is missing tm column: {rel(path)}")
    weather = weather.copy()
    weather["tm"] = pd.to_datetime(weather["tm"], errors="coerce")
    weather = weather.loc[weather["tm"].notna()].reset_index(drop=True)
    for col in WEATHER_NUMERIC_COLUMNS:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
    metadata = {
        "weather_cache": rel(path),
        "cache_rows": int(len(weather)),
        "columns": list(weather.columns),
        "station_ids": sorted(str(value) for value in weather.get("stnId", pd.Series(dtype=str)).dropna().unique()),
        "station_names": sorted(str(value) for value in weather.get("stnNm", pd.Series(dtype=str)).dropna().unique()),
        "time_min": None if weather.empty else str(weather["tm"].min()),
        "time_max": None if weather.empty else str(weather["tm"].max()),
        "secret_values_stored": False,
    }
    return weather, metadata


def aggregate_one_window(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> dict[str, float]:
    subset = weather.loc[(weather["tm"] >= start) & (weather["tm"] <= end)].copy()
    row: dict[str, float] = {}
    if subset.empty:
        return row
    if "ta" in subset:
        ta = subset["ta"]
        row[f"kma_{prefix}_temp_mean"] = float(ta.mean())
        row[f"kma_{prefix}_temp_max"] = float(ta.max())
        row[f"kma_{prefix}_temp_min"] = float(ta.min())
        row[f"kma_{prefix}_temp_std"] = float(ta.std())
        row[f"kma_{prefix}_temp_range"] = float(ta.max() - ta.min())
        row[f"kma_{prefix}_hot_hours_ge26"] = float((ta >= 26.0).sum())
        row[f"kma_{prefix}_hot_hours_ge28"] = float((ta >= 28.0).sum())
        row[f"kma_{prefix}_cool_hours_lt15"] = float((ta < 15.0).sum())
    if "hm" in subset:
        hm = subset["hm"]
        row[f"kma_{prefix}_humidity_mean"] = float(hm.mean())
        row[f"kma_{prefix}_humidity_max"] = float(hm.max())
        row[f"kma_{prefix}_humid_hours_ge70"] = float((hm >= 70.0).sum())
    if "ta" in subset and "hm" in subset:
        row[f"kma_{prefix}_temp_humidity_load_mean"] = float((subset["ta"] * subset["hm"] / 100.0).mean())
        row[f"kma_{prefix}_hot_humid_hours"] = float(((subset["ta"] >= 26.0) & (subset["hm"] >= 70.0)).sum())
    if "rn" in subset:
        rn = subset["rn"].fillna(0.0)
        row[f"kma_{prefix}_rain_sum"] = float(rn.sum())
        row[f"kma_{prefix}_rain_max"] = float(rn.max())
        row[f"kma_{prefix}_rain_hours"] = float((rn > 0.0).sum())
    if "ws" in subset:
        row[f"kma_{prefix}_wind_mean"] = float(subset["ws"].mean())
        row[f"kma_{prefix}_wind_max"] = float(subset["ws"].max())
    for col in ("pa", "ps"):
        if col in subset:
            series = subset[col]
            row[f"kma_{prefix}_{col}_mean"] = float(series.mean())
            row[f"kma_{prefix}_{col}_range"] = float(series.max() - series.min())
    for col in ("ss", "icsr"):
        if col in subset:
            row[f"kma_{prefix}_{col}_sum"] = float(subset[col].fillna(0.0).sum())
    if "ts" in subset:
        row[f"kma_{prefix}_ground_temp_mean"] = float(subset["ts"].mean())
    return row


def build_daily_weather_features(keys: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    unique_dates = sorted(pd.to_datetime(keys["date"]).dt.date.unique())
    date_rows: list[dict[str, Any]] = []
    for date_value in unique_dates:
        base = pd.Timestamp(date_value)
        row: dict[str, Any] = {"date": date_value}
        for prefix, (start_day, start_hour, end_day, end_hour) in WINDOWS.items():
            start = base + pd.Timedelta(days=start_day, hours=start_hour)
            end = base + pd.Timedelta(days=end_day, hours=end_hour)
            row.update(aggregate_one_window(weather, start, end, prefix))
        date_rows.append(row)
    daily = pd.DataFrame(date_rows)
    out = keys[["subject_id", "date"]].copy().reset_index(drop=True)
    out = out.merge(daily, on="date", how="left", validate="many_to_one")
    weather_cols = numeric_cols(out)
    out = out.sort_values(["subject_id", "date"]).reset_index(drop=True)
    lag_data: dict[str, pd.Series] = {}
    for col in weather_cols:
        lag = out.groupby("subject_id", sort=False)[col].shift(1)
        past3 = lag.groupby(out["subject_id"], sort=False).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        lag_data[f"{col}__lag1"] = lag
        lag_data[f"{col}__dev_past3"] = out[col] - past3
    out = pd.concat([out, pd.DataFrame(lag_data)], axis=1)
    all_cols = numeric_cols(out)
    thermal_cols = [col for col in all_cols if any(token in col for token in ("temp", "humidity", "humid", "hot"))]
    precip_cols = [col for col in all_cols if any(token in col for token in ("rain", "wind", "_pa_", "_ps_"))]
    solar_ground_cols = [col for col in all_cols if any(token in col for token in ("_ss_", "icsr", "ground"))]
    families = {
        "kma_weather_thermal_load": thermal_cols,
        "kma_weather_precip_pressure": precip_cols,
        "kma_weather_solar_ground": solar_ground_cols,
        "kma_weather_all": all_cols,
    }
    metadata = {
        "rows": int(len(out)),
        "unique_dates": int(pd.Series(out["date"]).nunique()),
        "windows": WINDOWS,
        "base_weather_columns": int(len(weather_cols)),
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
            split_predictions: dict[str, np.ndarray] = {}
            split_coverage: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_split(features, feature_cols, y, splits[split_name], spec)
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


def missing_cache_report(args: argparse.Namespace, weather_cache: Path) -> dict[str, Any]:
    decision = {
        "weather_cache_available": False,
        "reviewed_candidates": 0,
        "local_gate_pass_count": 0,
        "review_gate_pass_count": 0,
        "dacon_gate_pass_count": 0,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Fetch the ignored KMA ASOS cache with KMA_ASOS_SERVICE_KEY or PUBLIC_DATA_SERVICE_KEY, then rerun this "
            "report-only gate. Do not put the service key in tracked files."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_kma_weather_context_gate",
        "purpose": "Gate external KMA ASOS weather context against ID342.",
        "weather_cache": rel(weather_cache),
        "weather_cache_exists": False,
        "fetch_command": (
            ".venv/bin/python scripts/fetch_kma_asos_hourly_context.py "
            f"--output {rel(weather_cache)}"
        ),
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
    weather_cache = args.weather_cache if args.weather_cache.is_absolute() else ROOT / args.weather_cache
    if not weather_cache.exists():
        return missing_cache_report(args, weather_cache)

    train = load_train_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    failed_refs, missing_failed_refs = load_failed_refs(train)
    weather, weather_metadata = load_weather(weather_cache)
    features, feature_metadata = build_daily_weather_features(train[["subject_id", "date"]], weather)
    rows: list[dict[str, Any]] = []
    for family_name, family in feature_metadata["families"].items():
        rows.extend(
            scan_family(
                family_name=family_name,
                features=features,
                feature_cols=family["feature_sample"] if family["feature_count"] <= 16 else [
                    col for col in numeric_cols(features) if col in family["feature_sample"]
                ],
                train=train,
                splits=splits,
                anchor_values=anchor_values,
                failed_refs=failed_refs,
            )
        )
    # Re-scan with full family columns; keep this separate so metadata samples stay compact.
    rows = []
    for family_name in feature_metadata["families"]:
        if family_name == "kma_weather_thermal_load":
            cols = [col for col in numeric_cols(features) if any(token in col for token in ("temp", "humidity", "humid", "hot"))]
        elif family_name == "kma_weather_precip_pressure":
            cols = [col for col in numeric_cols(features) if any(token in col for token in ("rain", "wind", "_pa_", "_ps_"))]
        elif family_name == "kma_weather_solar_ground":
            cols = [col for col in numeric_cols(features) if any(token in col for token in ("_ss_", "icsr", "ground"))]
        else:
            cols = numeric_cols(features)
        rows.extend(scan_family(family_name, features, cols, train, splits, anchor_values, failed_refs))

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
        "weather_cache_available": True,
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
        "weather_context_gate_pass": bool(review_pass),
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Promote nothing unless a row clears the Public-0.58 review proxy and a separate artifact contract. "
            "This gate is weather-source evidence only."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_kma_weather_context_gate",
        "purpose": "Gate external KMA ASOS weather context against ID342.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "weather_cache": rel(weather_cache),
        "weather_cache_exists": True,
        "weather_metadata": weather_metadata,
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
        "# Post-ID342 KMA weather context gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Weather cache available: `{decision['weather_cache_available']}`",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate passes: `{decision['local_gate_pass_count']}`",
        f"- Review gate passes: `{decision['review_gate_pass_count']}`",
        f"- DACON gate passes: `{decision['dacon_gate_pass_count']}`",
        f"- Best candidate: `{decision.get('best_candidate')}`",
        f"- Best target/family: `{decision.get('best_target')}` / `{decision.get('best_family')}`",
        f"- Best projected Public proxy: `{decision.get('best_projected_public_proxy', ID342_PUBLIC):.9f}`",
        f"- Gate pass: `{decision.get('weather_context_gate_pass', False)}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
    ]
    if not report.get("weather_cache_exists"):
        lines.extend(
            [
                "## Fetch",
                "",
                "Set `KMA_ASOS_SERVICE_KEY` or `PUBLIC_DATA_SERVICE_KEY` locally, then run:",
                "",
                "```bash",
                report["fetch_command"],
                "```",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "## Weather Cache",
            "",
            "```json",
            json.dumps(report["weather_metadata"], indent=2, sort_keys=True),
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
    parser.add_argument("--weather-cache", type=Path, default=DEFAULT_WEATHER_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Weather cache available: "
        f"{decision['weather_cache_available']} | review_pass={decision.get('weather_context_gate_pass', False)}"
    )


if __name__ == "__main__":
    main()
