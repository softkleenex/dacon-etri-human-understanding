#!/usr/bin/env python3
"""Report-only dry run for GPS path-topology feature generation.

This validates whether the GPS topology feature family can produce an aligned,
bounded, leak-safe daily feature frame in memory. It writes only JSON/Markdown
reports; it does not write a feature matrix/parquet, train a model, create
predictions, create a submission CSV, touch the submission log, run Kaggle, or
submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
SPEC_REPORT = REPORTS_DIR / "post_id341_gps_path_topology_feature_spec_20260619.json"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "gps_path_topology_sequence_v1"
KEYS = ["subject_id", "lifelog_date", "split"]
RAW_SOURCE_PATHS = {
    "mGps": RAW_DIR / "ch2025_mGps.parquet",
    "mWifi": RAW_DIR / "ch2025_mWifi.parquet",
    "mBle": RAW_DIR / "ch2025_mBle.parquet",
    "mActivity": RAW_DIR / "ch2025_mActivity.parquet",
}

FIRST_DRYRUN_FEATURES = [
    "gps_rows",
    "gps_payload_points",
    "gps_coordinate_payload_rows",
    "gps_active_hours",
    "gps_evening_rows_18_23",
    "gps_night_rows_00_05",
    "gps_gap_count_gt15m",
    "gps_longest_gap_minutes",
    "gps_observed_span_minutes",
    "gps_speed_p50",
    "gps_speed_p90",
    "gps_speed_max",
    "gps_low_speed_payload_share",
    "gps_high_speed_payload_share",
    "gps_lat_span",
    "gps_lon_span",
    "gps_bbox_area_proxy",
    "gps_payload_quality_flag",
    "activity_rows",
    "activity_active_code_count",
    "activity_top_code_share",
    "activity_late_rows_18_23",
    "wifi_rows",
    "wifi_token_count_capped64",
    "wifi_top_token_share",
    "wifi_payload_points",
    "ble_rows",
    "ble_token_count_capped64",
    "ble_top_token_share",
    "ble_payload_points",
    "place_context_rows",
    "place_context_token_count_capped128",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(ROOT))}
    data = json.loads(path.read_text())
    data["exists"] = True
    data["path"] = str(path.relative_to(ROOT))
    return data


def expected_days() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    expected = pd.concat([train, test], ignore_index=True)
    expected["lifelog_date"] = expected["lifelog_date"].astype(str)
    return expected.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)


def date_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"])
    out["lifelog_date"] = timestamps.dt.strftime("%Y-%m-%d")
    out["hour"] = timestamps.dt.hour
    out["timestamp_dt"] = timestamps
    return out


def payload_len(value: Any) -> int:
    try:
        return int(len(value))
    except TypeError:
        return 0


def iter_payload(value: Any) -> Iterable[Any]:
    try:
        if len(value) == 0:
            return ()
    except TypeError:
        return ()
    return value


def payload_numeric_values(value: Any, key: str) -> list[float]:
    values: list[float] = []
    for item in iter_payload(value):
        if isinstance(item, dict):
            raw = item.get(key)
            if raw is not None and np.isfinite(raw):
                values.append(float(raw))
    return values


def payload_tokens(value: Any, key: str) -> list[str]:
    tokens: list[str] = []
    for item in iter_payload(value):
        if isinstance(item, dict):
            raw = item.get(key)
            if raw:
                tokens.append(str(raw))
    return tokens


def safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def add_expected_alignment(expected: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    merged = expected.merge(daily, on=["subject_id", "lifelog_date"], how="left")
    numeric_cols = merged.select_dtypes(include=["number", "bool"]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    return merged


def gps_daily_features(expected: pd.DataFrame) -> pd.DataFrame:
    gps = pd.read_parquet(RAW_SOURCE_PATHS["mGps"], columns=["subject_id", "timestamp", "m_gps"])
    gps = date_parts(gps)
    gps["payload_points"] = gps["m_gps"].map(payload_len)
    gps["coordinate_payload"] = gps["payload_points"] > 0
    gps["speed_values"] = gps["m_gps"].map(lambda value: payload_numeric_values(value, "speed"))
    gps["lat_values"] = gps["m_gps"].map(lambda value: payload_numeric_values(value, "latitude"))
    gps["lon_values"] = gps["m_gps"].map(lambda value: payload_numeric_values(value, "longitude"))
    gps["speed_p50_row"] = gps["speed_values"].map(lambda values: safe_quantile(values, 0.50))
    gps["speed_p90_row"] = gps["speed_values"].map(lambda values: safe_quantile(values, 0.90))
    gps["speed_max_row"] = gps["speed_values"].map(lambda values: max(values) if values else 0.0)
    gps["low_speed_row"] = gps["speed_max_row"] <= 0.5
    gps["high_speed_row"] = gps["speed_p90_row"] >= 4.0
    gps["lat_min_row"] = gps["lat_values"].map(lambda values: min(values) if values else np.nan)
    gps["lat_max_row"] = gps["lat_values"].map(lambda values: max(values) if values else np.nan)
    gps["lon_min_row"] = gps["lon_values"].map(lambda values: min(values) if values else np.nan)
    gps["lon_max_row"] = gps["lon_values"].map(lambda values: max(values) if values else np.nan)

    daily = (
        gps.groupby(["subject_id", "lifelog_date"])
        .agg(
            gps_rows=("timestamp", "size"),
            gps_payload_points=("payload_points", "sum"),
            gps_coordinate_payload_rows=("coordinate_payload", "sum"),
            gps_active_hours=("hour", "nunique"),
            gps_evening_rows_18_23=("hour", lambda s: int(s.isin([18, 19, 20, 21, 22, 23]).sum())),
            gps_night_rows_00_05=("hour", lambda s: int(s.isin([0, 1, 2, 3, 4, 5]).sum())),
            gps_speed_p50=("speed_p50_row", "median"),
            gps_speed_p90=("speed_p90_row", "quantile"),
            gps_speed_max=("speed_max_row", "max"),
            gps_low_speed_payload_share=("low_speed_row", "mean"),
            gps_high_speed_payload_share=("high_speed_row", "mean"),
            gps_lat_min=("lat_min_row", "min"),
            gps_lat_max=("lat_max_row", "max"),
            gps_lon_min=("lon_min_row", "min"),
            gps_lon_max=("lon_max_row", "max"),
            gps_first_ts=("timestamp_dt", "min"),
            gps_last_ts=("timestamp_dt", "max"),
        )
        .reset_index()
    )

    gaps = []
    for (subject_id, lifelog_date), group in gps.sort_values("timestamp_dt").groupby(["subject_id", "lifelog_date"]):
        diffs = group["timestamp_dt"].diff().dt.total_seconds().div(60).dropna()
        gaps.append(
            {
                "subject_id": subject_id,
                "lifelog_date": lifelog_date,
                "gps_gap_count_gt15m": int((diffs > 15).sum()),
                "gps_longest_gap_minutes": float(diffs.max()) if not diffs.empty else 0.0,
            }
        )
    gap_df = pd.DataFrame(gaps)
    daily = daily.merge(gap_df, on=["subject_id", "lifelog_date"], how="left")
    daily["gps_observed_span_minutes"] = (
        daily["gps_last_ts"].sub(daily["gps_first_ts"]).dt.total_seconds().div(60).fillna(0)
    )
    daily["gps_lat_span"] = daily["gps_lat_max"].sub(daily["gps_lat_min"]).fillna(0)
    daily["gps_lon_span"] = daily["gps_lon_max"].sub(daily["gps_lon_min"]).fillna(0)
    daily["gps_bbox_area_proxy"] = daily["gps_lat_span"].mul(daily["gps_lon_span"])
    daily["gps_payload_quality_flag"] = (daily["gps_rows"] >= 60) & (daily["gps_coordinate_payload_rows"] >= 30)
    keep = ["subject_id", "lifelog_date"] + [c for c in FIRST_DRYRUN_FEATURES if c.startswith("gps_")]
    return add_expected_alignment(expected, daily[keep])


def activity_daily_features(expected: pd.DataFrame) -> pd.DataFrame:
    act = pd.read_parquet(RAW_SOURCE_PATHS["mActivity"], columns=["subject_id", "timestamp", "m_activity"])
    act = date_parts(act)
    daily = (
        act.groupby(["subject_id", "lifelog_date"])
        .agg(
            activity_rows=("timestamp", "size"),
            activity_active_code_count=("m_activity", "nunique"),
            activity_late_rows_18_23=("hour", lambda s: int(s.isin([18, 19, 20, 21, 22, 23]).sum())),
        )
        .reset_index()
    )
    top_share = (
        act.groupby(["subject_id", "lifelog_date", "m_activity"])
        .size()
        .rename("count")
        .reset_index()
        .groupby(["subject_id", "lifelog_date"])
        .apply(lambda g: float(g["count"].max() / g["count"].sum()), include_groups=False)
        .rename("activity_top_code_share")
        .reset_index()
    )
    daily = daily.merge(top_share, on=["subject_id", "lifelog_date"], how="left")
    return add_expected_alignment(expected, daily)


def token_daily_features(
    expected: pd.DataFrame,
    source_name: str,
    payload_col: str,
    token_key: str,
    prefix: str,
    cap: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(RAW_SOURCE_PATHS[source_name], columns=["subject_id", "timestamp", payload_col])
    df = date_parts(df)
    df["payload_points"] = df[payload_col].map(payload_len)
    df["tokens"] = df[payload_col].map(lambda value: payload_tokens(value, token_key))

    rows = []
    subject_tokens: dict[str, set[str]] = {}
    for (subject_id, lifelog_date), group in df.groupby(["subject_id", "lifelog_date"]):
        tokens = [token for tokens_in_row in group["tokens"] for token in tokens_in_row]
        token_counts = pd.Series(tokens).value_counts() if tokens else pd.Series(dtype=int)
        rows.append(
            {
                "subject_id": subject_id,
                "lifelog_date": lifelog_date,
                f"{prefix}_rows": int(len(group)),
                f"{prefix}_token_count_capped{cap}": int(min(len(token_counts), cap)),
                f"{prefix}_top_token_share": float(token_counts.iloc[0] / token_counts.sum())
                if len(token_counts)
                else 0.0,
                f"{prefix}_payload_points": int(group["payload_points"].sum()),
            }
        )
        subject_tokens.setdefault(str(subject_id), set()).update(tokens)
    daily = pd.DataFrame(rows)
    cardinality = {
        "source": source_name,
        "prefix": prefix,
        "cap": cap,
        "subjects": len(subject_tokens),
        "max_raw_tokens_per_subject": max((len(tokens) for tokens in subject_tokens.values()), default=0),
        "subjects_over_cap_before_capping": sum(len(tokens) > cap for tokens in subject_tokens.values()),
        "max_tokens_after_capping": min(max((len(tokens) for tokens in subject_tokens.values()), default=0), cap),
    }
    return add_expected_alignment(expected, daily), cardinality


def assemble_virtual_daily_frame(expected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    gps = gps_daily_features(expected)
    activity = activity_daily_features(expected)
    wifi, wifi_cardinality = token_daily_features(expected, "mWifi", "m_wifi", "bssid", "wifi", 64)
    ble, ble_cardinality = token_daily_features(expected, "mBle", "m_ble", "address", "ble", 64)

    frame = expected.copy()
    for daily in [gps, activity, wifi, ble]:
        cols = ["subject_id", "lifelog_date"] + [c for c in daily.columns if c not in KEYS]
        frame = frame.merge(daily[cols], on=["subject_id", "lifelog_date"], how="left")

    numeric_cols = [c for c in frame.columns if c not in KEYS]
    frame[numeric_cols] = frame[numeric_cols].fillna(0)
    frame["place_context_rows"] = frame["wifi_rows"].add(frame["ble_rows"])
    frame["place_context_token_count_capped128"] = frame["wifi_token_count_capped64"].add(
        frame["ble_token_count_capped64"]
    )
    frame["place_context_token_count_capped128"] = frame["place_context_token_count_capped128"].clip(upper=128)
    frame = frame[KEYS + FIRST_DRYRUN_FEATURES]
    metadata = {
        "wifi_cardinality": wifi_cardinality,
        "ble_cardinality": ble_cardinality,
    }
    return frame, metadata


def row_alignment(frame: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    expected_keys = expected[KEYS].sort_values(KEYS).reset_index(drop=True)
    frame_keys = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
    exact = expected_keys.equals(frame_keys)
    duplicates = int(frame.duplicated(KEYS).sum())
    return {
        "expected_rows": int(len(expected_keys)),
        "dryrun_rows": int(len(frame_keys)),
        "exact_key_alignment": bool(exact),
        "duplicate_keys": duplicates,
        "train_rows": int((frame["split"] == "train").sum()),
        "test_rows": int((frame["split"] == "test").sum()),
    }


def coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    gps_ready = frame["gps_payload_quality_flag"] > 0
    rows_with_any_place = frame["place_context_rows"] > 0
    out = {}
    for split, split_df in frame.groupby("split"):
        split_gps_ready = split_df["gps_payload_quality_flag"] > 0
        split_place = split_df["place_context_rows"] > 0
        out[str(split)] = {
            "rows": int(len(split_df)),
            "gps_topology_ready_rows": int(split_gps_ready.sum()),
            "gps_topology_ready_rate": float(split_gps_ready.mean()),
            "place_context_ready_rows": int(split_place.sum()),
            "place_context_ready_rate": float(split_place.mean()),
            "median_gps_rows": float(split_df["gps_rows"].median()),
            "median_gps_payload_points": float(split_df["gps_payload_points"].median()),
        }
    return {
        "by_split": out,
        "overall_gps_topology_ready_rate": float(gps_ready.mean()),
        "overall_place_context_ready_rate": float(rows_with_any_place.mean()),
    }


def feature_column_summary(frame: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in frame.columns if c not in KEYS]
    missing_cols = sorted(set(FIRST_DRYRUN_FEATURES).difference(feature_cols))
    extra_cols = sorted(set(feature_cols).difference(FIRST_DRYRUN_FEATURES))
    return {
        "feature_column_count": len(feature_cols),
        "max_allowed_first_matrix_columns": 64,
        "within_first_matrix_bound": len(feature_cols) <= 64,
        "missing_declared_features": missing_cols,
        "extra_features": extra_cols,
        "features": feature_cols,
    }


def build_report() -> dict[str, Any]:
    spec = read_json(SPEC_REPORT)
    spec_dec = spec.get("decision", {}) if isinstance(spec.get("decision"), dict) else {}
    expected = expected_days()
    frame, metadata = assemble_virtual_daily_frame(expected)
    alignment = row_alignment(frame, expected)
    coverage = coverage_summary(frame)
    columns = feature_column_summary(frame)
    train_rate = coverage["by_split"].get("train", {}).get("gps_topology_ready_rate", 0.0)
    test_rate = coverage["by_split"].get("test", {}).get("gps_topology_ready_rate", 0.0)
    dryrun_passed = bool(
        spec.get("exists", False)
        and spec_dec.get("feature_spec_ready") is True
        and spec_dec.get("next_local_allowed_step") == "report_only_feature_generation_dryrun"
        and alignment["exact_key_alignment"]
        and alignment["duplicate_keys"] == 0
        and columns["within_first_matrix_bound"]
        and train_rate >= 0.90
        and test_rate >= 0.90
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_gps_path_topology_feature_generation_dryrun",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "inputs": {
            "feature_spec_report": {
                "path": str(SPEC_REPORT.relative_to(ROOT)),
                "exists": spec.get("exists", False),
                "feature_spec_ready": spec_dec.get("feature_spec_ready"),
                "next_local_allowed_step": spec_dec.get("next_local_allowed_step"),
            },
            "raw_sources": {name: str(path.relative_to(ROOT)) for name, path in RAW_SOURCE_PATHS.items()},
        },
        "row_alignment": alignment,
        "coverage": coverage,
        "feature_columns": columns,
        "cardinality_bounds": metadata,
        "safety_contract": {
            "feature_matrix_written": False,
            "feature_parquet_written": False,
            "predictions_created": False,
            "submission_csv_created": False,
            "submission_log_modified": False,
            "model_trained": False,
            "kaggle_run_executed": False,
            "dacon_submission_executed": False,
            "uses_target_labels": False,
            "uses_oof_predictions": False,
            "uses_public_scores": False,
        },
        "decision": {
            "dryrun_passed": dryrun_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": ["Q3", "S3"],
            "row_alignment_exact": alignment["exact_key_alignment"],
            "feature_column_count": columns["feature_column_count"],
            "feature_column_bound_passed": columns["within_first_matrix_bound"],
            "train_gps_topology_ready_rate": train_rate,
            "test_gps_topology_ready_rate": test_rate,
            "public_058_evidence": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "feature_matrix_smoke_contract_or_kaggle_upstream_rebuild",
            "next_action": (
                "GPS topology can be materialized in memory with aligned rows and bounded columns, but this remains "
                "non-score evidence. Do not create ID342 or train locally; either write a separate feature-matrix "
                "smoke contract or use the explicit-approved clean upstream Kaggle rebuild for score-relevant work."
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    align = report["row_alignment"]
    coverage = report["coverage"]["by_split"]
    cols = report["feature_columns"]
    lines = [
        "# GPS path-topology feature-generation dry run — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, "
            "base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, "
            "model training job, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Dry run passed: `{dec['dryrun_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Row alignment exact: `{dec['row_alignment_exact']}`",
        f"- Feature columns: `{dec['feature_column_count']}` / `64`",
        f"- Train/test GPS topology ready rates: `{dec['train_gps_topology_ready_rate']:.4f}` / "
        f"`{dec['test_gps_topology_ready_rate']:.4f}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Row alignment",
        "",
        "```json",
        json.dumps(align, indent=2),
        "```",
        "",
        "## Coverage",
        "",
        "| split | rows | GPS-ready rows | GPS-ready rate | place-ready rate | median GPS rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "test"]:
        item = coverage.get(split, {})
        lines.append(
            f"| `{split}` | {item.get('rows', 0)} | {item.get('gps_topology_ready_rows', 0)} | "
            f"{item.get('gps_topology_ready_rate', 0.0):.4f} | "
            f"{item.get('place_context_ready_rate', 0.0):.4f} | {item.get('median_gps_rows', 0.0):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Feature columns",
            "",
            f"- Declared dry-run feature count: `{cols['feature_column_count']}`",
            f"- Within first-matrix bound: `{cols['within_first_matrix_bound']}`",
            "",
            "```json",
            json.dumps(cols["features"], indent=2),
            "```",
            "",
            "## Cardinality bounds",
            "",
            "```json",
            json.dumps(report["cardinality_bounds"], indent=2),
            "```",
            "",
            "## Safety contract",
            "",
            "```json",
            json.dumps(report["safety_contract"], indent=2),
            "```",
            "",
            "## Boundary",
            "",
            (
                "This proves only that a GPS topology feature family can be aligned and bounded in memory. It is not "
                "leaderboard evidence, does not open ID342, and does not change the explicit-approval requirement for "
                "Kaggle or DACON actions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output.with_suffix(".md").write_text(markdown(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_gps_path_topology_feature_generation_dryrun_20260619.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    write_report(report, args.output)
    dec = report["decision"]
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    print(
        json.dumps(
            {
                "dryrun_passed": dec["dryrun_passed"],
                "row_alignment_exact": dec["row_alignment_exact"],
                "feature_column_count": dec["feature_column_count"],
                "train_gps_topology_ready_rate": dec["train_gps_topology_ready_rate"],
                "test_gps_topology_ready_rate": dec["test_gps_topology_ready_rate"],
                "feature_matrix_allowed_now": dec["feature_matrix_allowed_now"],
                "train_model_now": dec["train_model_now"],
                "create_submission_artifact": dec["create_submission_artifact"],
                "submit_now": dec["submit_now"],
                "push_kaggle_now": dec["push_kaggle_now"],
                "run_kaggle_now": dec["run_kaggle_now"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
