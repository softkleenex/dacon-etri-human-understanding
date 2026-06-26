#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Report-only in-memory feature-generation dry run for usage semantic burst/Markov v1.

This follows the feature spec and proves that the bounded label-free feature
schema can be assembled over the exact 450 train + 250 test keys. It writes only
JSON/Markdown reports. It does not write a feature matrix/parquet, train a
model, create predictions or submission artifacts, modify the submission log,
run Kaggle, or submit to DACON.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_usage_semantic_burst_markov_design import (  # noqa: E402
    PREDECLARED_APP_CATEGORIES,
    SOURCE_FAMILY,
    app_name_to_category,
)
from scripts.diagnose_post_id341_usage_semantic_burst_markov_feature_spec import (  # noqa: E402
    FEATURE_COLUMN_BOUND,
    NEXT_ALLOWED_STEP as SPEC_ALLOWED_STEP,
    TARGET_FOCUS,
    all_feature_columns,
    feature_groups,
)

REPORTS_DIR = ROOT / "reports"
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.json"
FEATURE_SPEC_REPORT = REPORTS_DIR / "post_id341_usage_semantic_burst_markov_feature_spec_20260619.json"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
USAGE_PATH = RAW_ITEMS_DIR / "ch2025_mUsageStats.parquet"
SCREEN_PATH = RAW_ITEMS_DIR / "ch2025_mScreenStatus.parquet"
AMBIENCE_PATH = RAW_ITEMS_DIR / "ch2025_mAmbience.parquet"

NEXT_ALLOWED_STEP = "report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract"
CURRENT_QUEUE_STEP = "report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix"
ALLOWED_QUEUE_STEP_TYPES = {
    "report_only_usage_semantic_burst_markov_feature_spec_no_matrix",
    CURRENT_QUEUE_STEP,
    NEXT_ALLOWED_STEP,
    "write_usage_semantic_burst_markov_feature_matrix_under_contract_no_model_no_submission",
    "validate_usage_semantic_burst_markov_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval",
}
FORBIDDEN_FEATURE_TOKENS = ["app_name", "raw_app", "target", "label", "public_lb", "submission", "prediction", "oof"]
FORBIDDEN_DECISION_KEYS = [
    "feature_matrix_allowed_now",
    "train_model_now",
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_json_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def no_forbidden_actions(report: dict[str, Any]) -> bool:
    dec = decision(report)
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def queue_lane_completed_no_model(queue: dict[str, Any]) -> bool:
    items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    item = next((entry for entry in items if entry.get("name") == SOURCE_FAMILY), {})
    return bool(
        item.get("status") == "validated_stop_completed_no_model"
        and item.get("local_no_approval_allowed") is False
        and item.get("feature_matrix_materialized_under_contract") is True
        and item.get("train_model_now") is False
        and item.get("create_artifact_allowed") is False
        and item.get("submission_artifact_allowed") is False
    )


def expected_key_frame() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    frame = pd.concat([train, test], ignore_index=True).drop_duplicates(["subject_id", "lifelog_date", "split"])
    frame["lifelog_date"] = frame["lifelog_date"].astype(str)
    return frame.sort_values(["split", "subject_id", "lifelog_date"]).reset_index(drop=True)


def iter_usage_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, (str, bytes, dict)) or not hasattr(payload, "__iter__"):
        return []
    return [item for item in payload if isinstance(item, dict)]


def summarize_usage_payload(payload: Any) -> dict[str, Any]:
    category_time = dict.fromkeys(PREDECLARED_APP_CATEGORIES, 0.0)
    categories_present: set[str] = set()
    total_time = 0.0
    app_count = 0
    for item in iter_usage_payload(payload):
        app_count += 1
        category = app_name_to_category(str(item.get("app_name", "")))
        categories_present.add(category)
        try:
            value = float(item.get("total_time", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        total_time += value
        category_time[category] += value
    dominant_category = max(PREDECLARED_APP_CATEGORIES, key=lambda category: category_time[category])
    return {
        "app_count": int(app_count),
        "total_time": float(total_time),
        "dominant_category": dominant_category if app_count else "other",
        "categories_present": sorted(categories_present),
        "category_time": category_time,
    }


def build_usage_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(USAGE_PATH, columns=["subject_id", "timestamp", "m_usage_stats"])
    summaries = raw["m_usage_stats"].map(summarize_usage_payload)
    summary_frame = pd.DataFrame(list(summaries))
    rows = pd.concat([raw[["subject_id", "timestamp"]].copy(), summary_frame], axis=1)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"])
    rows["lifelog_date"] = rows["timestamp"].dt.strftime("%Y-%m-%d")
    rows["hour"] = rows["timestamp"].dt.hour
    rows["has_payload"] = rows["app_count"] > 0
    rows["usage_row_id"] = np.arange(len(rows))

    event_records: list[dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        for item in iter_usage_payload(row.m_usage_stats):
            app_name = str(item.get("app_name", ""))
            try:
                total_time = float(item.get("total_time", 0.0))
            except (TypeError, ValueError):
                total_time = 0.0
            event_records.append(
                {
                    "subject_id": row.subject_id,
                    "timestamp": timestamp,
                    "lifelog_date": timestamp.strftime("%Y-%m-%d"),
                    "hour": timestamp.hour,
                    "category": app_name_to_category(app_name),
                    "total_time": total_time,
                }
            )
    events = pd.DataFrame.from_records(
        event_records,
        columns=["subject_id", "timestamp", "lifelog_date", "hour", "category", "total_time"],
    )
    return rows, events


def safe_entropy(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(float) / total
    return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())


def nearest_by_subject(left: pd.DataFrame, right: pd.DataFrame, tolerance: pd.Timedelta) -> pd.DataFrame:
    matches: list[pd.DataFrame] = []
    for subject_id, left_part in left.sort_values("timestamp").groupby("subject_id", sort=False):
        right_part = right.loc[right["subject_id"].eq(subject_id)].sort_values("timestamp")
        if right_part.empty:
            part = left_part.copy()
            for column in right.columns:
                if column not in {"subject_id", "timestamp"}:
                    part[column] = np.nan
            matches.append(part)
            continue
        joined = pd.merge_asof(
            left_part.sort_values("timestamp"),
            right_part,
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
            suffixes=("", "_matched"),
        )
        matches.append(joined)
    return pd.concat(matches, ignore_index=True) if matches else left.head(0).copy()


def daily_base(expected: pd.DataFrame) -> pd.DataFrame:
    return expected[["subject_id", "lifelog_date", "split"]].copy()


def add_group_features(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return base.merge(features, on=["subject_id", "lifelog_date"], how="left")


def usage_daily_features(usage_rows: pd.DataFrame, usage_events: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    base = daily_base(expected)
    row_daily = (
        usage_rows.groupby(["subject_id", "lifelog_date"])
        .agg(
            usage_row_count=("timestamp", "size"),
            usage_payload_row_rate=("has_payload", "mean"),
            usage_rows_total=("timestamp", "size"),
            payload_event_count_total=("app_count", "sum"),
            usage_total_time_sum=("total_time", "sum"),
        )
        .reset_index()
    )
    usage_rows = usage_rows.copy()
    usage_rows["is_evening"] = usage_rows["hour"].between(18, 23)
    usage_rows["is_late"] = usage_rows["hour"].ge(22) | usage_rows["hour"].le(3)
    window_daily = (
        usage_rows.groupby(["subject_id", "lifelog_date"])
        .agg(
            usage_rows_evening=("is_evening", "sum"),
            usage_rows_late=("is_late", "sum"),
            usage_total_time_evening_sum=(
                "total_time",
                lambda s: float(s[usage_rows.loc[s.index, "is_evening"]].sum()),
            ),
            usage_total_time_late_sum=("total_time", lambda s: float(s[usage_rows.loc[s.index, "is_late"]].sum())),
        )
        .reset_index()
    )
    out = add_group_features(base, row_daily)
    out = add_group_features(out, window_daily)

    if usage_events.empty:
        category_features = pd.DataFrame(columns=["subject_id", "lifelog_date"])
    else:
        category_time = usage_events.pivot_table(
            index=["subject_id", "lifelog_date"],
            columns="category",
            values="total_time",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(columns=PREDECLARED_APP_CATEGORIES, fill_value=0.0)
        category_share = category_time.div(category_time.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        category_share.columns = [f"category_time_share_{category}" for category in category_share.columns]
        category_presence = (
            usage_events.assign(present=1)
            .pivot_table(
                index=["subject_id", "lifelog_date"],
                columns="category",
                values="present",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=PREDECLARED_APP_CATEGORIES, fill_value=0)
        )
        category_presence.columns = [
            f"category_row_presence_count_{category}" for category in category_presence.columns
        ]
        dominant_count = (
            usage_rows.assign(value=1)
            .pivot_table(
                index=["subject_id", "lifelog_date"],
                columns="dominant_category",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=PREDECLARED_APP_CATEGORIES, fill_value=0)
        )
        dominant_count.columns = [f"category_dominant_count_{category}" for category in dominant_count.columns]
        category_features = pd.concat([category_share, category_presence, dominant_count], axis=1).reset_index()
    out = add_group_features(out, category_features)
    return out


def session_features(usage_rows: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    base = daily_base(expected)
    if usage_rows.empty:
        return base
    ordered = usage_rows.sort_values(["subject_id", "lifelog_date", "timestamp"]).copy()
    same_subject = ordered["subject_id"].eq(ordered["subject_id"].shift())
    same_date = ordered["lifelog_date"].eq(ordered["lifelog_date"].shift())
    gap_minutes = ordered.groupby("subject_id")["timestamp"].diff().dt.total_seconds().div(60.0)
    ordered["new_session"] = (~same_subject) | (~same_date) | gap_minutes.isna() | gap_minutes.gt(30)
    ordered["session_id"] = ordered["new_session"].cumsum()
    session = (
        ordered.groupby(["subject_id", "lifelog_date", "session_id"])
        .agg(rows=("timestamp", "size"), categories=("dominant_category", "nunique"))
        .reset_index()
    )
    session["is_multi_category"] = session["categories"] >= 2
    daily = (
        session.groupby(["subject_id", "lifelog_date"])
        .agg(
            session_count=("session_id", "size"),
            session_rows_median=("rows", "median"),
            session_rows_p90=("rows", lambda s: float(s.quantile(0.90))),
            max_session_rows=("rows", "max"),
            multi_category_session_count=("is_multi_category", "sum"),
        )
        .reset_index()
    )
    evening_burst = (
        ordered[ordered["hour"].between(18, 23)]
        .groupby(["subject_id", "lifelog_date"])
        .size()
        .rename("evening_burst_rows_ge3")
        .reset_index()
    )
    evening_burst["evening_burst_rows_ge3"] = (evening_burst["evening_burst_rows_ge3"] >= 3).astype(int)
    late_reactivation = (
        ordered[ordered["hour"].ge(22) | ordered["hour"].le(3)]
        .groupby(["subject_id", "lifelog_date"])
        .size()
        .rename("late_reactivation_rows_ge1")
        .reset_index()
    )
    late_reactivation["late_reactivation_rows_ge1"] = (late_reactivation["late_reactivation_rows_ge1"] >= 1).astype(int)
    out = add_group_features(base, daily)
    out = add_group_features(out, evening_burst)
    out = add_group_features(out, late_reactivation)
    return out


def markov_features(usage_rows: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    base = daily_base(expected)
    ordered = usage_rows.sort_values(["subject_id", "timestamp"]).copy()
    ordered["prev_subject"] = ordered["subject_id"].shift()
    ordered["prev_date"] = ordered["lifelog_date"].shift()
    ordered["prev_category"] = ordered["dominant_category"].shift()
    ordered["prev_timestamp"] = ordered["timestamp"].shift()
    ordered["gap_minutes"] = (ordered["timestamp"] - ordered["prev_timestamp"]).dt.total_seconds() / 60.0
    valid = (
        ordered["has_payload"]
        & ordered["prev_category"].notna()
        & ordered["subject_id"].eq(ordered["prev_subject"])
        & ordered["lifelog_date"].eq(ordered["prev_date"])
        & ordered["gap_minutes"].le(360)
    )
    trans = ordered.loc[valid, ["subject_id", "lifelog_date", "hour", "prev_category", "dominant_category"]].copy()
    if trans.empty:
        return base
    trans["pair"] = trans["prev_category"].astype(str) + "→" + trans["dominant_category"].astype(str)
    trans["is_self"] = trans["prev_category"].eq(trans["dominant_category"])
    trans["is_switch"] = ~trans["is_self"]
    trans["is_evening"] = trans["hour"].between(18, 23)
    records = []
    for (subject_id, lifelog_date), part in trans.groupby(["subject_id", "lifelog_date"]):
        pair_counts = part["pair"].value_counts()
        transition_count = int(len(part))
        top_count = int(pair_counts.max()) if transition_count else 0
        evening = part[part["is_evening"]]
        records.append(
            {
                "subject_id": subject_id,
                "lifelog_date": lifelog_date,
                "transition_count": transition_count,
                "unique_transition_pair_count": int(pair_counts.size),
                "transition_entropy": safe_entropy(pair_counts),
                "evening_transition_count": int(len(evening)),
                "evening_unique_transition_pair_count": int(evening["pair"].nunique()),
                "self_transition_rate": float(part["is_self"].mean()),
                "top_pair_concentration": float(top_count / transition_count) if transition_count else 0.0,
                "rare_pair_count": int((pair_counts == 1).sum()),
                "category_switch_rate": float(part["is_switch"].mean()),
            }
        )
    daily = pd.DataFrame.from_records(records)
    out = add_group_features(base, daily)
    out = add_rolling_subject_features(
        out,
        value_column="transition_entropy",
        output_column="markov_surprise_vs_subject_baseline",
        mode="delta",
    )
    return out


def screen_features(usage_rows: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    base = daily_base(expected)
    screen = pd.read_parquet(SCREEN_PATH, columns=["subject_id", "timestamp", "m_screen_use"])
    screen["timestamp"] = pd.to_datetime(screen["timestamp"])
    screen["lifelog_date"] = screen["timestamp"].dt.strftime("%Y-%m-%d")
    screen = screen.sort_values(["subject_id", "timestamp"]).copy()
    screen["screen_changed"] = screen.groupby("subject_id")["m_screen_use"].diff().fillna(0).ne(0)
    screen["prev_use"] = screen.groupby("subject_id")["m_screen_use"].shift()
    screen["reactivation"] = screen["prev_use"].eq(0) & screen["m_screen_use"].eq(1)
    daily_screen = (
        screen.groupby(["subject_id", "lifelog_date"])
        .agg(
            screen_day_rows=("timestamp", "size"),
            screen_reactivation_after_idle_count=("reactivation", "sum"),
        )
        .reset_index()
    )
    left = usage_rows[["usage_row_id", "subject_id", "timestamp", "lifelog_date", "hour"]].copy()
    matched = nearest_by_subject(
        left,
        screen[["subject_id", "timestamp", "m_screen_use", "screen_changed"]],
        pd.Timedelta(minutes=5),
    )
    matched["screen_matched"] = matched["m_screen_use"].notna()
    matched["screen_on"] = matched["m_screen_use"].fillna(0).astype(float)
    matched["screen_changed_near"] = matched["screen_changed"].fillna(False).astype(bool)
    matched["is_late"] = matched["hour"].ge(22) | matched["hour"].le(3)
    matched["is_evening"] = matched["hour"].between(18, 23)
    usage_daily = (
        matched.groupby(["subject_id", "lifelog_date"])
        .agg(
            screen_match_rate_5min=("screen_matched", "mean"),
            screen_on_at_usage_rate=("screen_on", "mean"),
            screen_state_change_count_near_usage=("screen_changed_near", "sum"),
            screen_on_burst_count=("screen_on", "sum"),
            screen_off_burst_count=("screen_on", lambda s: float((s == 0).sum())),
            late_screen_on_usage_rate=(
                "screen_on",
                lambda s: (
                    float(s[matched.loc[s.index, "is_late"]].mean()) if matched.loc[s.index, "is_late"].any() else 0.0
                ),
            ),
            evening_screen_on_usage_rate=(
                "screen_on",
                lambda s: (
                    float(s[matched.loc[s.index, "is_evening"]].mean())
                    if matched.loc[s.index, "is_evening"].any()
                    else 0.0
                ),
            ),
            usage_after_screen_on_count=("screen_on", "sum"),
            usage_before_screen_off_count=("screen_on", lambda s: float((s == 0).sum())),
            screen_coupling_missing_rate=("screen_matched", lambda s: float(1.0 - s.mean())),
        )
        .reset_index()
    )
    out = add_group_features(base, daily_screen)
    out = add_group_features(out, usage_daily)
    out["screen_missing_flag"] = out["screen_day_rows"].fillna(0).eq(0).astype(int)
    out = out.drop(columns=["screen_day_rows"], errors="ignore")
    return out


def ambience_row_summary(payload: Any) -> dict[str, Any]:
    labels: list[str] = []
    if not isinstance(payload, (str, bytes, dict)) and hasattr(payload, "__iter__"):
        for item in payload:
            if isinstance(item, (list, tuple)) and item:
                labels.append(str(item[0]))
    lowered = " ".join(labels).lower()
    return {
        "ambience_label_count": len(labels),
        "ambience_top_element": labels[0] if labels else "",
        "ambience_has_speech": "speech" in lowered or "conversation" in lowered,
        "ambience_has_noise": any(
            token in lowered for token in ["music", "vehicle", "car", "outside", "crowd", "bus", "train"]
        ),
        "ambience_quiet_like": any(
            token in lowered for token in ["inside, small room", "inside, large room", "silence"]
        ),
    }


def ambience_features(usage_rows: pd.DataFrame, expected: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    base = daily_base(expected)
    ambience = pd.read_parquet(AMBIENCE_PATH, columns=["subject_id", "timestamp", "m_ambience"])
    summaries = pd.DataFrame(list(ambience["m_ambience"].map(ambience_row_summary)))
    ambience = pd.concat([ambience[["subject_id", "timestamp"]].copy(), summaries], axis=1)
    ambience["timestamp"] = pd.to_datetime(ambience["timestamp"])
    ambience["lifelog_date"] = ambience["timestamp"].dt.strftime("%Y-%m-%d")
    ambience["hour"] = ambience["timestamp"].dt.hour
    ambience["is_evening"] = ambience["hour"].between(18, 23)
    ambience["is_late"] = ambience["hour"].ge(22) | ambience["hour"].le(3)
    left = usage_rows[["usage_row_id", "subject_id", "timestamp", "lifelog_date", "hour"]].copy()
    matched = nearest_by_subject(
        left,
        ambience[
            [
                "subject_id",
                "timestamp",
                "ambience_top_element",
                "ambience_has_speech",
                "ambience_has_noise",
                "ambience_quiet_like",
            ]
        ],
        pd.Timedelta(minutes=10),
    )
    matched["ambience_matched"] = matched["ambience_top_element"].notna()
    matched["is_evening"] = matched["hour"].between(18, 23)
    matched["is_late"] = matched["hour"].ge(22) | matched["hour"].le(3)
    matched["ambience_has_noise"] = matched["ambience_has_noise"].fillna(False).astype(bool)
    matched["ambience_quiet_like"] = matched["ambience_quiet_like"].fillna(False).astype(bool)
    daily = (
        matched.groupby(["subject_id", "lifelog_date"])
        .agg(
            ambience_match_rate_10min=("ambience_matched", "mean"),
            ambience_events_near_burst_count=("ambience_matched", "sum"),
            ambience_unique_element_count_near_burst=("ambience_top_element", "nunique"),
            ambience_missing_rate_near_usage=("ambience_matched", lambda s: float(1.0 - s.mean())),
            ambience_evening_context_count=("is_evening", "sum"),
            ambience_late_context_count=("is_late", "sum"),
            ambience_noise_near_usage=("ambience_has_noise", "sum"),
            ambience_quiet_near_usage=("ambience_quiet_like", "sum"),
        )
        .reset_index()
    )
    out = add_group_features(base, daily)
    switch = current[
        ["subject_id", "lifelog_date", "unique_transition_pair_count", "late_reactivation_rows_ge1"]
    ].copy()
    out = add_group_features(out, switch)
    out["quiet_high_switching_flag"] = (
        out["ambience_quiet_near_usage"].fillna(0).gt(0) & out["unique_transition_pair_count"].fillna(0).ge(5)
    ).astype(int)
    out["noisy_late_reactivation_flag"] = (
        out["ambience_noise_near_usage"].fillna(0).gt(0) & out["late_reactivation_rows_ge1"].fillna(0).gt(0)
    ).astype(int)
    out["ambience_guard_conflict_score"] = out["quiet_high_switching_flag"] + out["noisy_late_reactivation_flag"]
    out = out.drop(
        columns=[
            "unique_transition_pair_count",
            "late_reactivation_rows_ge1",
            "ambience_noise_near_usage",
            "ambience_quiet_near_usage",
        ],
        errors="ignore",
    )
    return out


def add_rolling_subject_features(
    frame: pd.DataFrame,
    value_column: str,
    output_column: str,
    mode: str = "zscore",
) -> pd.DataFrame:
    out = frame.copy()
    out["_date"] = pd.to_datetime(out["lifelog_date"])
    out = out.sort_values(["subject_id", "_date", "split"]).reset_index(drop=True)
    values = out[value_column].fillna(0.0).astype(float)
    outputs: list[float] = []
    for _, idx in out.groupby("subject_id", sort=False).groups.items():
        subject_values = values.loc[list(idx)].reset_index(drop=True)
        rolling_mean = subject_values.shift(1).rolling(7, min_periods=1).mean()
        rolling_std = subject_values.shift(1).rolling(7, min_periods=2).std().replace(0, np.nan)
        if mode == "delta":
            transformed = subject_values - rolling_mean.fillna(subject_values)
        else:
            transformed = (subject_values - rolling_mean) / rolling_std
            transformed = transformed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        outputs.extend(transformed.astype(float).tolist())
    out[output_column] = outputs
    return out.drop(columns=["_date"])


def add_subject_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for value_column, output_column in [
        ("evening_burst_rows_ge3", "rolling_7d_burst_count_z"),
        ("late_reactivation_rows_ge1", "rolling_7d_late_reactivation_z"),
        ("transition_entropy", "rolling_7d_transition_entropy_z"),
        ("screen_on_at_usage_rate", "rolling_7d_screen_on_rate_z"),
    ]:
        out = add_rolling_subject_features(out, value_column=value_column, output_column=output_column, mode="zscore")
    out["_date"] = pd.to_datetime(out["lifelog_date"])
    out = out.sort_values(["subject_id", "_date", "split"]).reset_index(drop=True)
    history_counts: list[int] = []
    ranks: list[float] = []
    for _, idx in out.groupby("subject_id", sort=False).groups.items():
        values = out.loc[list(idx), "usage_total_time_sum"].fillna(0.0).astype(float).reset_index(drop=True)
        for pos, value in enumerate(values):
            start = max(0, pos - 7)
            history = values.iloc[start:pos]
            history_counts.append(int(len(history)))
            ranks.append(float((history <= value).mean()) if len(history) else 0.5)
    out["subject_history_day_count"] = history_counts
    out["subject_history_missing_flag"] = (out["subject_history_day_count"] < 2).astype(int)
    out["subject_relative_usage_intensity_rank"] = ranks
    return out.drop(columns=["_date"])


def assemble_feature_frame() -> pd.DataFrame:
    expected = expected_key_frame()
    usage_rows, usage_events = build_usage_frames()
    frame = usage_daily_features(usage_rows, usage_events, expected)
    for part in [
        session_features(usage_rows, expected),
        markov_features(usage_rows, expected),
        screen_features(usage_rows, expected),
    ]:
        new_columns = [column for column in part.columns if column not in {"subject_id", "lifelog_date", "split"}]
        frame = frame.merge(
            part[["subject_id", "lifelog_date", *new_columns]], on=["subject_id", "lifelog_date"], how="left"
        )
    frame = frame.merge(
        ambience_features(usage_rows, expected, frame),
        on=["subject_id", "lifelog_date", "split"],
        how="left",
    )
    frame = add_subject_relative_features(frame)
    frame["usage_missing_flag"] = frame["usage_row_count"].fillna(0).eq(0).astype(int)
    coverage_parts = pd.concat(
        [
            frame["usage_row_count"].fillna(0).gt(0).astype(float),
            frame["screen_missing_flag"].fillna(1).eq(0).astype(float),
            frame["ambience_match_rate_10min"].fillna(0).gt(0).astype(float),
        ],
        axis=1,
    )
    frame["source_coverage_score"] = coverage_parts.mean(axis=1)
    expected_columns = all_feature_columns(feature_groups())
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    frame[expected_columns] = frame[expected_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return frame[["subject_id", "lifelog_date", "split", *expected_columns]].sort_values(
        ["split", "subject_id", "lifelog_date"]
    )


def feature_frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    feature_columns = all_feature_columns(feature_groups())
    numeric = frame[feature_columns]
    zero_variance = [column for column in feature_columns if float(numeric[column].std(ddof=0)) == 0.0]
    missing_rates = numeric.isna().mean().sort_values(ascending=False)
    return {
        "row_count": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "feature_column_count": int(len(feature_columns)),
        "finite_all_values": bool(np.isfinite(numeric.to_numpy()).all()),
        "nan_count": int(numeric.isna().sum().sum()),
        "zero_variance_feature_count": int(len(zero_variance)),
        "zero_variance_features_sample": zero_variance[:20],
        "max_abs_value": float(np.abs(numeric.to_numpy()).max()) if len(frame) else 0.0,
        "top_missing_rates": {str(key): float(value) for key, value in missing_rates.head(10).items()},
        "split_feature_mean_abs_delta_mean": float(
            (numeric[frame["split"].eq("train")].mean() - numeric[frame["split"].eq("test")].mean()).abs().mean()
        ),
    }


def columns_are_safe(columns: list[str]) -> bool:
    lowered = [column.lower() for column in columns]
    return all(not any(token in column for token in FORBIDDEN_FEATURE_TOKENS) for column in lowered)


def build_report() -> dict[str, Any]:
    spec = load_json_report(FEATURE_SPEC_REPORT)
    queue = load_json_report(QUEUE_REPORT)
    spec_dec = decision(spec)
    queue_dec = decision(queue)
    expected_columns = all_feature_columns(feature_groups())
    frame = assemble_feature_frame()
    summary = feature_frame_summary(frame)

    spec_contract_ready = bool(
        spec.get("parse_ok") is True
        and spec_dec.get("feature_spec_ready") is True
        and spec_dec.get("source_family") == SOURCE_FAMILY
        and spec_dec.get("target_focus") == TARGET_FOCUS
        and spec_dec.get("next_local_allowed_step") == SPEC_ALLOWED_STEP
        and spec_dec.get("feature_matrix_allowed_now") is False
        and spec_dec.get("create_submission_artifact") is False
        and spec_dec.get("submit_now") is False
        and no_forbidden_actions(spec)
    )
    current_queue_contract_ready = bool(
        queue.get("parse_ok") is True
        and queue_dec.get("material_new_source_queue_ready") is True
        and queue_dec.get("next_no_approval_step") == SOURCE_FAMILY
        and queue_dec.get("next_no_approval_step_type") in ALLOWED_QUEUE_STEP_TYPES
        and queue_dec.get("create_submission_artifact") is False
        and queue_dec.get("submit_now") is False
        and queue_dec.get("run_kaggle_now") is False
        and queue_dec.get("train_model_now") is False
        and no_forbidden_actions(queue)
    )
    usage_lane_completed_no_model = bool(queue.get("parse_ok") is True and queue_lane_completed_no_model(queue))
    queue_contract_ready = bool(current_queue_contract_ready or usage_lane_completed_no_model)
    row_alignment_passed = bool(
        summary["row_count"] == 700 and summary["train_rows"] == 450 and summary["test_rows"] == 250
    )
    schema_passed = bool(
        summary["feature_column_count"] == len(expected_columns)
        and summary["feature_column_count"] == 77
        and summary["feature_column_count"] <= FEATURE_COLUMN_BOUND
        and set(frame.columns) == {"subject_id", "lifelog_date", "split", *expected_columns}
        and columns_are_safe(expected_columns)
    )
    values_passed = bool(
        summary["finite_all_values"] and summary["nan_count"] == 0 and summary["zero_variance_feature_count"] < 20
    )
    dryrun_passed = bool(
        spec_contract_ready and queue_contract_ready and row_alignment_passed and schema_passed and values_passed
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_usage_semantic_burst_markov_feature_generation_dryrun",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Assemble the usage semantic burst/Markov feature schema in memory only and verify exact row alignment, "
            "bounded feature columns, missingness handling, and leakage guards before any matrix contract exists."
        ),
        "inputs": {"feature_spec_report": spec, "material_new_source_queue": queue},
        "raw_inputs": {
            "usage_stats": rel(USAGE_PATH),
            "screen_status": rel(SCREEN_PATH),
            "ambience": rel(AMBIENCE_PATH),
        },
        "feature_frame_summary": summary,
        "feature_groups": [
            {"name": group["name"], "column_count": len(group["columns"]), "columns": group["columns"]}
            for group in feature_groups()
        ],
        "leakage_contract": {
            "labels_loaded": False,
            "oof_or_test_predictions_loaded": False,
            "public_scores_used": False,
            "raw_app_name_feature_columns_created": False,
            "target_guided_vocabulary_used": False,
            "feature_frame_persisted": False,
            "all_missing_values_filled_with_flags_or_zero": summary["nan_count"] == 0,
        },
        "checks": {
            "spec_contract_ready": spec_contract_ready,
            "queue_contract_ready": queue_contract_ready,
            "current_queue_contract_ready": current_queue_contract_ready,
            "usage_lane_completed_no_model": usage_lane_completed_no_model,
            "row_alignment_passed": row_alignment_passed,
            "schema_passed": schema_passed,
            "values_passed": values_passed,
            "feature_column_bound_respected": summary["feature_column_count"] <= FEATURE_COLUMN_BOUND,
            "does_not_write_feature_matrix": True,
            "does_not_train_model": True,
            "does_not_create_submission_artifact": True,
        },
        "decision": {
            "feature_generation_dryrun_passed": dryrun_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": NEXT_ALLOWED_STEP
            if dryrun_passed
            else "fix_usage_feature_generation_dryrun_inputs",
            "next_action": (
                "A future no-approval step may only write a report-only feature-matrix smoke contract. Do not write "
                "the feature matrix/parquet, train a model, create ID342, run Kaggle, or submit."
                if dryrun_passed
                else "Do not advance usage semantic burst/Markov work until dry-run blockers are fixed."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    summary = report["feature_frame_summary"]
    lines = [
        "# Post-ID341 usage semantic burst/Markov feature-generation dry run",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, base prediction "
            "artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON "
            "submission."
        ),
        "",
        "## Decision",
        "",
        f"- Dry run passed: `{dec['feature_generation_dryrun_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{dec['target_focus']}`",
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
        "## In-memory feature frame summary",
        "",
        f"- Rows train/test/all: `{summary['train_rows']}` / `{summary['test_rows']}` / `{summary['row_count']}`",
        f"- Feature columns: `{summary['feature_column_count']}`",
        f"- Finite values: `{summary['finite_all_values']}`",
        f"- NaN count: `{summary['nan_count']}`",
        f"- Zero-variance feature count: `{summary['zero_variance_feature_count']}`",
        f"- Max absolute value: `{summary['max_abs_value']:.6f}`",
        f"- Mean abs train/test feature-mean delta: `{summary['split_feature_mean_abs_delta_mean']:.6f}`",
        "",
        "## Feature groups",
        "",
        "| group | columns |",
        "|---|---:|",
    ]
    for group in report["feature_groups"]:
        lines.append(f"| `{group['name']}` | `{group['column_count']}` |")
    lines.extend(
        [
            "",
            "## Leakage and persistence boundary",
            "",
            "```json",
            json.dumps(report["leakage_contract"], indent=2, ensure_ascii=False),
            "```",
            "",
            (
                "Passing this dry run does not create ID342 and does not authorize feature-matrix persistence, model "
                "training, a submission CSV, submission-log changes, Kaggle execution, DACON submission, or a Public "
                "`0.58` claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    write_report(report, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
