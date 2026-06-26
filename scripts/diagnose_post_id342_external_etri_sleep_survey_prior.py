#!/usr/bin/env python3
"""Post-ID342 external ETRI sleep/survey prior audit.

This report-only diagnostic checks whether the repository's historical ETRI
sleep/survey CSVs provide a materially new source mechanism for Q3/S3/S4 after
weather, AirKorea, registered material, and same-family routes closed.

It reads only existing local CSV/parquet metadata, creates no feature matrix,
fits no model, writes no predictions, creates no submission artifact, edits no
submission log, runs no Kaggle job, and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EXTERNAL_DIR = ROOT / "data" / "external" / "etri_lifelog"
PROCESSED_TRAIN_PATH = ROOT / "data" / "processed" / "df_train_raw.parquet"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_external_etri_sleep_survey_prior_20260625.json"
KST = timezone(timedelta(hours=9))

ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55

SURVEY_2020 = EXTERNAL_DIR / "user_survey_2020.csv"
SLEEP_2020 = EXTERNAL_DIR / "user_sleep_2020.csv"
SURVEY_2019 = EXTERNAL_DIR / "user_survey_2019_2018.csv"
SLEEP_2019 = EXTERNAL_DIR / "user_sleep_2019_2018.csv"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def frame_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    frame = pd.read_csv(path)
    return {
        "path": rel(path),
        "exists": True,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": list(frame.columns),
        "mean_missing_rate": float(frame.isna().mean().mean()),
    }


def to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_2020_pm_next_sleep_pairs() -> pd.DataFrame:
    survey = pd.read_csv(SURVEY_2020)
    sleep = pd.read_csv(SLEEP_2020)
    pm = survey.loc[survey["amPm"] == "pm"].copy()
    pm["pm_date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm["sleep_date_next"] = (pm["pm_date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    sleep = sleep.copy()
    sleep["date"] = pd.to_datetime(sleep["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sleep_numeric_cols = [col for col in sleep.columns if col not in {"userId", "timezone", "date"}]
    sleep = to_numeric(sleep, sleep_numeric_cols)
    sleep = (
        sleep.groupby(["userId", "date"], as_index=False)
        .agg({col: "mean" for col in sleep_numeric_cols if col in sleep.columns})
        .reset_index(drop=True)
    )
    merged = pm.merge(
        sleep,
        left_on=["userId", "sleep_date_next"],
        right_on=["userId", "date"],
        how="inner",
        suffixes=("_survey", "_sleep"),
        validate="one_to_one",
    )
    numeric = [
        "pmStress",
        "pmFatigue",
        "pmEmotion",
        "caffeine",
        "cAmount(ml)",
        "alcohol",
        "aAmount(ml)",
        "durationtosleep",
        "wakeupduration",
        "sleep_score",
        "wakeupcount",
        "hr_average",
        "breathing_disturbances_intensity",
        "snoring",
        "snoringepisodecount",
    ]
    merged = to_numeric(merged, numeric)
    merged["q3_positive"] = merged["pmStress"] <= merged.groupby("userId")["pmStress"].transform("mean")
    merged["q2_positive"] = merged["pmFatigue"] <= merged.groupby("userId")["pmFatigue"].transform("mean")
    merged["s3_sol_0_30m"] = merged["durationtosleep"].between(0, 1800, inclusive="both")
    merged["s3_sol_15_30m"] = merged["durationtosleep"].between(900, 1800, inclusive="both")
    merged["s4_waso_lt20m"] = merged["wakeupduration"] < 1200
    merged["sleep_score_positive"] = merged["sleep_score"] >= merged.groupby("userId")["sleep_score"].transform("mean")
    return merged


def binary_rate(frame: pd.DataFrame, col: str) -> float | None:
    if col not in frame or frame[col].dropna().empty:
        return None
    return float(frame[col].mean())


def conditional_summary(frame: pd.DataFrame, label: str, metrics: list[str]) -> dict[str, Any]:
    if label not in frame or frame[label].dropna().empty:
        return {"available": False}
    grouped = frame.groupby(label, dropna=True)[metrics].mean(numeric_only=True)
    out: dict[str, Any] = {"available": True, "groups": {}}
    for key, row in grouped.iterrows():
        out["groups"][str(bool(key))] = {col: safe_float(value) for col, value in row.items()}
    if False in grouped.index and True in grouped.index:
        out["positive_minus_negative"] = {
            col: safe_float(grouped.loc[True, col] - grouped.loc[False, col]) for col in metrics
        }
    return out


def correlation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    x_cols = ["pmStress", "pmFatigue", "pmEmotion", "caffeine", "alcohol"]
    y_cols = [
        "durationtosleep",
        "wakeupduration",
        "sleep_score",
        "wakeupcount",
        "hr_average",
        "breathing_disturbances_intensity",
        "snoringepisodecount",
    ]
    rows: list[dict[str, Any]] = []
    for x_col in x_cols:
        for y_col in y_cols:
            if x_col not in frame or y_col not in frame:
                continue
            subset = frame[[x_col, y_col]].dropna()
            if len(subset) < 30 or subset[x_col].nunique() <= 1 or subset[y_col].nunique() <= 1:
                continue
            corr = subset[x_col].corr(subset[y_col], method="spearman")
            if pd.isna(corr):
                continue
            rows.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "n": int(len(subset)),
                    "spearman": float(corr),
                    "abs_spearman": float(abs(corr)),
                }
            )
    return sorted(rows, key=lambda row: (-row["abs_spearman"], row["x"], row["y"]))


def historical_sleep_distribution() -> dict[str, Any]:
    sleep_2020 = to_numeric(pd.read_csv(SLEEP_2020), ["durationtosleep", "wakeupduration", "sleep_score"])
    sleep_2019 = to_numeric(pd.read_csv(SLEEP_2019), ["total_sleep_time", "time_in_bed", "waso", "sleep_score"])
    return {
        "sleep_2020": {
            "rows": int(len(sleep_2020)),
            "users": int(sleep_2020["userId"].nunique()),
            "sol_0_30m_rate": binary_rate(sleep_2020.assign(flag=sleep_2020["durationtosleep"].between(0, 1800)), "flag"),
            "sol_15_30m_rate": binary_rate(
                sleep_2020.assign(flag=sleep_2020["durationtosleep"].between(900, 1800)), "flag"
            ),
            "waso_lt20m_rate": binary_rate(sleep_2020.assign(flag=sleep_2020["wakeupduration"] < 1200), "flag"),
            "sleep_score_mean": safe_float(sleep_2020["sleep_score"].mean()),
        },
        "sleep_2019_2018": {
            "rows": int(len(sleep_2019)),
            "users": int(sleep_2019["userId"].nunique()),
            "waso_non_null_rows": int(sleep_2019["waso"].notna().sum()),
            "waso_lt20m_rate": binary_rate(sleep_2019.assign(flag=sleep_2019["waso"] < 1200), "flag"),
            "sleep_score_mean": safe_float(sleep_2019["sleep_score"].mean()),
        },
    }


def current_schema_matches() -> dict[str, Any]:
    if not PROCESSED_TRAIN_PATH.exists():
        return {"processed_train_exists": False}
    train = pd.read_parquet(PROCESSED_TRAIN_PATH)
    numeric_cols = [
        col
        for col in train.columns
        if pd.api.types.is_numeric_dtype(train[col]) and col not in {"Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"}
    ]
    patterns = {
        "sleep_hr_fragmentation": ["wHr_sleep", "wHr_slp", "slp_hr", "fragmentation"],
        "pre_sleep_settling": ["presleep", "slp_screen", "slp_act", "slp_pedo", "slp_wlight"],
        "usage_stress_proxy": ["mUsageStats", "screen", "youtube", "kakao", "instagram", "netflix", "tiktok"],
        "social_routine_proxy": ["mGps", "gps", "mWifi", "wifi", "mBle", "ble", "mAmbience", "ambience"],
    }
    family_rows = {}
    for name, tokens in patterns.items():
        matches = sorted({col for col in numeric_cols for token in tokens if token in col})
        family_rows[name] = {
            "match_count": int(len(matches)),
            "examples": matches[:20],
            "mean_missing_rate": float(train[matches].isna().mean().mean()) if matches else None,
        }
    return {
        "processed_train_exists": True,
        "processed_train_shape": [int(train.shape[0]), int(train.shape[1])],
        "numeric_feature_columns": int(len(numeric_cols)),
        "family_matches": family_rows,
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    pairs = build_2020_pm_next_sleep_pairs()
    metrics = [
        "pmStress",
        "pmFatigue",
        "durationtosleep",
        "wakeupduration",
        "sleep_score",
        "s3_sol_0_30m",
        "s3_sol_15_30m",
        "s4_waso_lt20m",
        "sleep_score_positive",
    ]
    q2_conditional = conditional_summary(pairs, "q2_positive", metrics)
    q3_conditional = conditional_summary(pairs, "q3_positive", metrics)
    q2_delta = q2_conditional.get("positive_minus_negative", {})
    q3_delta = q3_conditional.get("positive_minus_negative", {})
    q2_sleep_coupling = bool(
        (q2_delta.get("durationtosleep") or 0.0) < 0
        and (q2_delta.get("wakeupduration") or 0.0) < 0
        and (q2_delta.get("s3_sol_0_30m") or 0.0) > 0
    )
    q3_sleep_coupling = bool(
        (q3_delta.get("durationtosleep") or 0.0) < 0
        and (q3_delta.get("wakeupduration") or 0.0) < 0
        and (q3_delta.get("s3_sol_0_30m") or 0.0) > 0
    )
    decision = {
        "external_etri_prior_analysis_complete": True,
        "external_rows_paired_2020_pm_to_next_sleep": int(len(pairs)),
        "external_users_paired_2020": int(pairs["userId"].nunique()),
        "q2_fatigue_sleep_coupling_supported": q2_sleep_coupling,
        "q3_stress_sleep_coupling_supported": q3_sleep_coupling,
        "direct_external_concat_allowed": False,
        "direct_2026_feature_route_available": False,
        "external_prior_gate_pass": False,
        "projected_public_proxy": ID342_PUBLIC,
        "gap_to_public_055": ID342_PUBLIC - PUBLIC_055,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Use the external ETRI sleep/survey files as domain prior only. Q2 fatigue has a coherent next-sleep "
            "coupling, but Q3 stress does not support a simple sleep-quality proxy and no direct 2026 feature route "
            "is opened. Do not create artifacts, Kaggle runs, or DACON submissions from this prior."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_external_etri_sleep_survey_prior",
        "purpose": "Audit historical ETRI sleep/survey priors for a materially new Q3/S3/S4 source mechanism.",
        "source_files": {
            "survey_2020": frame_summary(SURVEY_2020),
            "sleep_2020": frame_summary(SLEEP_2020),
            "survey_2019_2018": frame_summary(SURVEY_2019),
            "sleep_2019_2018": frame_summary(SLEEP_2019),
        },
        "pairing_policy": {
            "pm_survey_to_sleep": "2020 PM survey date is matched to the same user on next-day sleep.date.",
            "q2_positive": "pmFatigue <= per-user PM fatigue mean.",
            "q3_positive": "pmStress <= per-user PM stress mean.",
            "s3_sol_0_30m": "durationtosleep between 0 and 30 minutes.",
            "s3_sol_15_30m": "durationtosleep between 15 and 30 minutes.",
            "s4_waso_lt20m": "wakeupduration < 20 minutes.",
        },
        "paired_2020_summary": {
            "rows": int(len(pairs)),
            "users": int(pairs["userId"].nunique()),
            "rates": {
                "q2_positive": binary_rate(pairs, "q2_positive"),
                "q3_positive": binary_rate(pairs, "q3_positive"),
                "s3_sol_0_30m": binary_rate(pairs, "s3_sol_0_30m"),
                "s3_sol_15_30m": binary_rate(pairs, "s3_sol_15_30m"),
                "s4_waso_lt20m": binary_rate(pairs, "s4_waso_lt20m"),
                "sleep_score_positive": binary_rate(pairs, "sleep_score_positive"),
            },
            "q2_conditional": q2_conditional,
            "q3_conditional": q3_conditional,
            "top_correlations_abs_ge_0p08": [row for row in correlation_rows(pairs) if row["abs_spearman"] >= 0.08],
        },
        "historical_sleep_distribution": historical_sleep_distribution(),
        "current_2026_schema_matches": current_schema_matches(),
        "decision": decision,
        "no_raw_competition_payload_read": True,
        "no_feature_matrix_created": True,
        "no_model_fit": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    paired = report["paired_2020_summary"]
    q2_delta = paired["q2_conditional"].get("positive_minus_negative", {})
    q3_delta = paired["q3_conditional"].get("positive_minus_negative", {})
    lines = [
        "# Post-ID342 external ETRI sleep/survey prior",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Paired 2020 PM-to-next-sleep rows/users: `{decision['external_rows_paired_2020_pm_to_next_sleep']}` / `{decision['external_users_paired_2020']}`",
        f"- Q2 fatigue sleep-coupling supported: `{decision['q2_fatigue_sleep_coupling_supported']}`",
        f"- Q3 stress sleep-coupling supported: `{decision['q3_stress_sleep_coupling_supported']}`",
        f"- Direct 2026 feature route available: `{decision['direct_2026_feature_route_available']}`",
        f"- External prior gate pass: `{decision['external_prior_gate_pass']}`",
        f"- Projected Public proxy: `{decision['projected_public_proxy']:.9f}`",
        f"- Gap to Public 0.55: `{decision['gap_to_public_055']:+.9f}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Paired 2020 Rates",
        "",
        "```json",
        json.dumps(paired["rates"], indent=2, sort_keys=True),
        "```",
        "",
        "## Positive Minus Negative Deltas",
        "",
        "| label | duration-to-sleep | wakeupduration | sleep score | S3 0-30m | S3 15-30m | S4 WASO<20m |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| `q2_positive` | {q2_delta.get('durationtosleep', 0.0):+.3f} | "
            f"{q2_delta.get('wakeupduration', 0.0):+.3f} | {q2_delta.get('sleep_score', 0.0):+.3f} | "
            f"{q2_delta.get('s3_sol_0_30m', 0.0):+.6f} | {q2_delta.get('s3_sol_15_30m', 0.0):+.6f} | "
            f"{q2_delta.get('s4_waso_lt20m', 0.0):+.6f} |"
        ),
        (
            f"| `q3_positive` | {q3_delta.get('durationtosleep', 0.0):+.3f} | "
            f"{q3_delta.get('wakeupduration', 0.0):+.3f} | {q3_delta.get('sleep_score', 0.0):+.3f} | "
            f"{q3_delta.get('s3_sol_0_30m', 0.0):+.6f} | {q3_delta.get('s3_sol_15_30m', 0.0):+.6f} | "
            f"{q3_delta.get('s4_waso_lt20m', 0.0):+.6f} |"
        ),
        "",
        "## Top Correlations",
        "",
        "| x | y | n | Spearman |",
        "|---|---|---:|---:|",
    ]
    for row in paired["top_correlations_abs_ge_0p08"][:16]:
        lines.append(f"| `{row['x']}` | `{row['y']}` | {row['n']} | {row['spearman']:+.6f} |")
    lines.extend(
        [
            "",
            "## Current 2026 Schema Matches",
            "",
            "| family | columns | missing rate | examples |",
            "|---|---:|---:|---|",
        ]
    )
    for family_name, metadata in report["current_2026_schema_matches"].get("family_matches", {}).items():
        examples = ", ".join(f"`{col}`" for col in metadata["examples"][:8])
        missing = metadata["mean_missing_rate"]
        missing_text = "" if missing is None else f"{missing:.6f}"
        lines.append(f"| `{family_name}` | {metadata['match_count']} | {missing_text} | {examples} |")
    lines.extend(
        [
            "",
            "## Source Files",
            "",
            "```json",
            json.dumps(report["source_files"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "External ETRI prior complete: "
        f"{decision['external_etri_prior_analysis_complete']} | gate_pass={decision['external_prior_gate_pass']}"
    )


if __name__ == "__main__":
    main()
