#!/usr/bin/env python3
"""Report-only phone/wearable discordance source design after ID341.

This diagnostic freezes a materially different raw-source family for future
review: cross-device disagreement between phone state (screen/charging/activity)
and wearable state (steps/heart-rate/light). It is intentionally design +
metadata only. It does not write a feature matrix, train a model, create
predictions, create a submission CSV, edit the submission log, run Kaggle, or
submit to DACON.
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
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
TRAIN_PATH = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "phone_wearable_discordance_v1"
TARGET_FOCUS = ["Q2", "Q3", "S3"]
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_phone_wearable_discordance_design_20260619.json"

REQUIRED_CONTEXT_REPORTS = {
    "no_approval_frontier": "post_id341_no_approval_frontier_20260619.json",
    "sensor_adherence_dropout": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "sleep_boundary_prepromotion": "sleep_boundary_prepromotion_gate_post_id341_20260618.json",
    "circadian_validation_stop": "post_id341_circadian_hr_light_phase_model_smoke_20260619.json",
    "usage_validation_stop": "post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "proximity_validation_stop": "post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop_20260619.json",
    "operator_handoff": "post_id341_operator_handoff_contract_20260619.json",
}

RAW_SOURCES = {
    "screen": "ch2025_mScreenStatus.parquet",
    "charge": "ch2025_mACStatus.parquet",
    "activity": "ch2025_mActivity.parquet",
    "pedo": "ch2025_wPedo.parquet",
    "hr": "ch2025_wHr.parquet",
    "wearable_light": "ch2025_wLight.parquet",
    "phone_light": "ch2025_mLight.parquet",
}

WINDOWS = {
    "early_morning": {"hours": [0, 1, 2, 3, 4, 5], "description": "00:00-05:59 nocturnal residual activity"},
    "morning": {"hours": [6, 7, 8, 9, 10, 11], "description": "06:00-11:59 wake/start friction"},
    "daytime": {"hours": [12, 13, 14, 15, 16, 17], "description": "12:00-17:59 daytime alignment"},
    "evening": {"hours": [18, 19, 20, 21], "description": "18:00-21:59 evening settling"},
    "late_evening": {"hours": [22, 23], "description": "22:00-23:59 pre-sleep reactivation"},
}

FEATURE_GROUPS = {
    "phone_rest_state": [
        "screen_off_rate",
        "charging_rate",
        "phone_still_rate",
        "screen_off_charging_bins",
    ],
    "wearable_activation_state": [
        "step_active_rate",
        "step_sum_log1p",
        "hr_observed_rate",
        "wearable_light_log1p",
    ],
    "discordance_state": [
        "screen_off_step_active_bins",
        "charging_step_active_bins",
        "phone_rest_wearable_active_bins",
        "phone_dark_wearable_light_bins",
    ],
    "transition_hysteresis": [
        "last_phone_active_hour",
        "first_wearable_active_hour_after_phone_rest",
        "wearable_after_phone_rest_gap_hours",
    ],
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_report(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"exists": False, "parse_ok": False, "path": rel(path), "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"exists": True, "parse_ok": False, "path": rel(path), "error": str(exc), "decision": {}}
    payload["exists"] = True
    payload["parse_ok"] = True
    payload["path"] = rel(path)
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def load_rows() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH, usecols=["subject_id", "sleep_date", "lifelog_date"])
    test = pd.read_csv(SAMPLE_PATH, usecols=["subject_id", "sleep_date", "lifelog_date"])
    rows = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    rows["lifelog_date"] = pd.to_datetime(rows["lifelog_date"]).dt.date
    rows["row_id"] = np.arange(len(rows), dtype=int)
    return rows


def hour_grid(rows: pd.DataFrame) -> pd.DataFrame:
    hours = pd.DataFrame({"hour": list(range(24))})
    grid = rows[["subject_id", "lifelog_date", "split", "row_id"]].merge(hours, how="cross")
    return grid


def add_day_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["lifelog_date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour.astype(int)
    return out


def safe_read(source: str, columns: list[str]) -> pd.DataFrame:
    path = RAW_DIR / RAW_SOURCES[source]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns)


def aggregate_screen() -> pd.DataFrame:
    df = add_day_hour(safe_read("screen", ["subject_id", "timestamp", "m_screen_use"]))
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date", "hour"])
    df["screen_on"] = pd.to_numeric(df["m_screen_use"], errors="coerce").fillna(0).gt(0).astype(float)
    return (
        df.groupby(["subject_id", "lifelog_date", "hour"])
        .agg(screen_records=("screen_on", "size"), screen_on_rate=("screen_on", "mean"))
        .reset_index()
    )


def aggregate_charge() -> pd.DataFrame:
    df = add_day_hour(safe_read("charge", ["subject_id", "timestamp", "m_charging"]))
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date", "hour"])
    df["charging"] = pd.to_numeric(df["m_charging"], errors="coerce").fillna(0).gt(0).astype(float)
    return (
        df.groupby(["subject_id", "lifelog_date", "hour"])
        .agg(charge_records=("charging", "size"), charging_rate=("charging", "mean"))
        .reset_index()
    )


def aggregate_activity() -> pd.DataFrame:
    df = add_day_hour(safe_read("activity", ["subject_id", "timestamp", "m_activity"]))
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date", "hour"])
    activity = pd.to_numeric(df["m_activity"], errors="coerce")
    df["phone_active"] = activity.isin([0, 1, 2, 7, 8]).astype(float)
    df["phone_still"] = activity.eq(3).astype(float)
    return (
        df.groupby(["subject_id", "lifelog_date", "hour"])
        .agg(
            activity_records=("phone_active", "size"),
            phone_active_rate=("phone_active", "mean"),
            phone_still_rate=("phone_still", "mean"),
        )
        .reset_index()
    )


def aggregate_pedo() -> pd.DataFrame:
    df = add_day_hour(safe_read("pedo", ["subject_id", "timestamp", "step", "distance", "speed"]))
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date", "hour"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["step_active"] = df["step"].gt(0).astype(float)
    return (
        df.groupby(["subject_id", "lifelog_date", "hour"])
        .agg(pedo_records=("step", "size"), step_sum=("step", "sum"), step_active_rate=("step_active", "mean"))
        .reset_index()
    )


def list_mean(value: Any) -> float:
    if isinstance(value, np.ndarray):
        arr = value.astype(float)
    elif isinstance(value, list):
        arr = np.asarray(value, dtype=float)
    else:
        return float("nan")
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def aggregate_hr() -> pd.DataFrame:
    df = add_day_hour(safe_read("hr", ["subject_id", "timestamp", "heart_rate"]))
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date", "hour"])
    df["hr_mean_record"] = df["heart_rate"].map(list_mean)
    return (
        df.groupby(["subject_id", "lifelog_date", "hour"])
        .agg(hr_records=("hr_mean_record", "count"), hr_mean=("hr_mean_record", "mean"))
        .reset_index()
    )


def aggregate_light(source: str, value_col: str, prefix: str) -> pd.DataFrame:
    df = add_day_hour(safe_read(source, ["subject_id", "timestamp", value_col]))
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "lifelog_date", "hour"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").clip(lower=0.0)
    return (
        df.groupby(["subject_id", "lifelog_date", "hour"])
        .agg(**{f"{prefix}_records": (value_col, "count"), f"{prefix}_mean": (value_col, "mean")})
        .reset_index()
    )


def build_hourly_frame(rows: pd.DataFrame) -> pd.DataFrame:
    frame = hour_grid(rows)
    for agg in [
        aggregate_screen(),
        aggregate_charge(),
        aggregate_activity(),
        aggregate_pedo(),
        aggregate_hr(),
        aggregate_light("wearable_light", "w_light", "wlight"),
        aggregate_light("phone_light", "m_light", "mlight"),
    ]:
        frame = frame.merge(agg, on=["subject_id", "lifelog_date", "hour"], how="left")
    return frame


def split_rate(frame: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    aligned_mask = mask.reindex(frame.index, fill_value=False).astype(bool)
    for split in ["train", "test"]:
        split_mask = frame["split"].eq(split)
        part = aligned_mask.loc[split_mask]
        out[split] = float(part.mean()) if len(part) else 0.0
    return out


def source_coverage(rows: pd.DataFrame, hourly: pd.DataFrame) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    source_record_cols = {
        "screen": "screen_records",
        "charge": "charge_records",
        "activity": "activity_records",
        "pedo": "pedo_records",
        "hr": "hr_records",
        "wearable_light": "wlight_records",
        "phone_light": "mlight_records",
    }
    day_keys = rows[["subject_id", "lifelog_date", "split", "row_id"]]
    for source, col in source_record_cols.items():
        if col not in hourly.columns:
            coverage[source] = {"train_day_rate": 0.0, "test_day_rate": 0.0, "observed_hour_bins": 0}
            continue
        by_day = (
            hourly.assign(has_source=hourly[col].fillna(0).gt(0))
            .groupby(["subject_id", "lifelog_date", "split", "row_id"])["has_source"]
            .any()
            .reset_index()
        )
        merged = day_keys.merge(by_day, on=["subject_id", "lifelog_date", "split", "row_id"], how="left")
        merged["has_source"] = merged["has_source"].fillna(False)
        coverage[source] = {
            "train_day_rate": float(merged.loc[merged["split"].eq("train"), "has_source"].mean()),
            "test_day_rate": float(merged.loc[merged["split"].eq("test"), "has_source"].mean()),
            "observed_hour_bins": int(hourly[col].fillna(0).gt(0).sum()),
        }
    return coverage


def discordance_feasibility(hourly: pd.DataFrame) -> dict[str, Any]:
    screen_off = hourly["screen_on_rate"].fillna(0.0).le(0.05) & hourly["screen_records"].fillna(0).gt(0)
    charging = hourly["charging_rate"].fillna(0.0).ge(0.5) & hourly["charge_records"].fillna(0).gt(0)
    phone_still = hourly["phone_still_rate"].fillna(0.0).ge(0.5) & hourly["activity_records"].fillna(0).gt(0)
    step_active = hourly["step_active_rate"].fillna(0.0).ge(0.05) & hourly["pedo_records"].fillna(0).gt(0)
    hr_present = hourly["hr_records"].fillna(0).gt(0)
    wlight_present = hourly["wlight_records"].fillna(0).gt(0)
    wearable_active = step_active | hr_present | wlight_present
    phone_rest = (screen_off | charging | phone_still) & (
        hourly["screen_records"].fillna(0).gt(0)
        | hourly["charge_records"].fillna(0).gt(0)
        | hourly["activity_records"].fillna(0).gt(0)
    )
    discordance_masks = {
        "screen_off_step_active": screen_off & step_active,
        "charging_step_active": charging & step_active,
        "phone_still_hr_present": phone_still & hr_present,
        "phone_rest_wearable_active": phone_rest & wearable_active,
        "screen_off_wearable_light_present": screen_off & wlight_present,
    }
    by_mask = {
        name: {
            "hour_bins_total": int(mask.sum()),
            "train_hour_bin_rate": split_rate(hourly, mask)["train"],
            "test_hour_bin_rate": split_rate(hourly, mask)["test"],
        }
        for name, mask in discordance_masks.items()
    }
    by_window: dict[str, dict[str, int]] = {}
    for window, meta in WINDOWS.items():
        window_mask = hourly["hour"].isin(meta["hours"])
        by_window[window] = {name: int((mask & window_mask).sum()) for name, mask in discordance_masks.items()}
    return {
        "phone_rest_hour_bins": int(phone_rest.sum()),
        "wearable_active_hour_bins": int(wearable_active.sum()),
        "discordance_by_mask": by_mask,
        "discordance_by_window_counts": by_window,
        "any_discordance_observed": any(row["hour_bins_total"] > 0 for row in by_mask.values()),
    }


def context_checks(context: dict[str, dict[str, Any]]) -> dict[str, bool]:
    no_approval = decision(context["no_approval_frontier"])
    sensor = decision(context["sensor_adherence_dropout"])
    sleep = decision(context["sleep_boundary_prepromotion"])
    circadian = decision(context["circadian_validation_stop"])
    usage = decision(context["usage_validation_stop"])
    proximity = decision(context["proximity_validation_stop"])
    operator = decision(context["operator_handoff"])
    return {
        "all_context_reports_present": all(item.get("exists") and item.get("parse_ok") for item in context.values()),
        "local_score_routes_closed_before_design": no_approval.get("no_approval_score_route_exists") is False,
        "sensor_adherence_standalone_closed": sensor.get("reviewable_low_movement_candidate_exists") is False,
        "sleep_boundary_is_lead_not_artifact": (
            sleep.get("reviewable_feature_lead") is True
            or sleep.get("reviewable_low_movement_candidate_exists") is True
            or sleep.get("promote_submission_artifact") is False
        )
        and sleep.get("create_submission_artifact") is False,
        "circadian_matrix_model_closed": circadian.get("public_058_evidence") is False
        and circadian.get("create_submission_artifact") is False,
        "usage_matrix_validation_stopped": usage.get("validation_stop_passed") is True
        and usage.get("score_route_opened") is False,
        "proximity_matrix_validation_stopped": proximity.get("validation_stop_passed") is True
        and proximity.get("score_route_opened") is False,
        "operator_handoff_waiting_for_approval": operator.get("operator_handoff_ready") is True
        and operator.get("requires_exact_approval_phrase") is True,
    }


def build_report() -> dict[str, Any]:
    context = {name: read_json_report(path) for name, path in REQUIRED_CONTEXT_REPORTS.items()}
    checks = context_checks(context)
    rows = load_rows()
    hourly = build_hourly_frame(rows)
    coverage = source_coverage(rows, hourly)
    feasibility = discordance_feasibility(hourly)

    coverage_ready = bool(
        coverage["screen"]["train_day_rate"] >= 0.95
        and coverage["screen"]["test_day_rate"] >= 0.95
        and coverage["charge"]["train_day_rate"] >= 0.95
        and coverage["pedo"]["train_day_rate"] >= 0.85
        and coverage["hr"]["train_day_rate"] >= 0.50
    )
    design_ready = bool(all(checks.values()) and coverage_ready and feasibility["any_discordance_observed"])

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_phone_wearable_discordance_design",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "input_reports": {
            name: {"path": report.get("path"), "exists": report.get("exists"), "parse_ok": report.get("parse_ok")}
            for name, report in context.items()
        },
        "context_checks": checks,
        "row_contract": {
            "expected_train_rows": 450,
            "expected_test_rows": 250,
            "expected_total_rows": 700,
            "actual_train_rows": int(rows["split"].eq("train").sum()),
            "actual_test_rows": int(rows["split"].eq("test").sum()),
            "actual_total_rows": int(len(rows)),
            "hour_grid_rows": int(len(hourly)),
            "key_columns": ["subject_id", "sleep_date", "lifelog_date"],
            "label_columns_allowed": False,
            "prediction_columns_allowed": False,
        },
        "raw_sources": {key: rel(RAW_DIR / filename) for key, filename in RAW_SOURCES.items()},
        "source_coverage": coverage,
        "windows": WINDOWS,
        "feature_groups": FEATURE_GROUPS,
        "discordance_feasibility": feasibility,
        "schema_bounds": {
            "max_future_feature_columns": 96,
            "planned_feature_prefix": "pwd_",
            "raw_identifier_columns_allowed": False,
            "fold_internal_preprocessing_required_for_any_model": True,
        },
        "leakage_guards": [
            "Build future features only from each row's lifelog_date timestamps, never from target labels or public scores.",
            "Use fixed discordance definitions; do not select phone/wearable thresholds from outer validation rows.",
            "Any model smoke must learn imputation/scaling/selection inside outer folds only.",
            "This design cannot create ID342, a base prediction artifact, a notebook-history artifact, or a submission-log row.",
        ],
        "decision": {
            "design_ready": design_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "coverage_ready": coverage_ready,
            "discordance_observed": feasibility["any_discordance_observed"],
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "report_only_phone_wearable_discordance_feature_spec_no_matrix"
            if design_ready
            else "fix_context_or_coverage_before_any_followup",
            "next_action": (
                "Treat this as a new report-only source design for the approved upstream rebuild packet. A later step may "
                "write only a feature spec; do not create a matrix, train, create ID342, run Kaggle, or submit."
                if design_ready
                else "Do not proceed until context and coverage checks pass."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 phone/wearable discordance design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Design ready: `{dec['design_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Coverage ready: `{dec['coverage_ready']}`",
        f"- Discordance observed: `{dec['discordance_observed']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Context checks",
        "",
    ]
    for key, value in report["context_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Source coverage", ""])
    for source, row in report["source_coverage"].items():
        lines.append(
            f"- `{source}`: train day rate `{row['train_day_rate']:.4f}`, "
            f"test day rate `{row['test_day_rate']:.4f}`, observed hour bins `{row['observed_hour_bins']}`"
        )
    lines.extend(["", "## Discordance feasibility", ""])
    feas = report["discordance_feasibility"]
    lines.append(f"- Phone-rest hour bins: `{feas['phone_rest_hour_bins']}`")
    lines.append(f"- Wearable-active hour bins: `{feas['wearable_active_hour_bins']}`")
    for name, row in feas["discordance_by_mask"].items():
        lines.append(
            f"- `{name}`: bins `{row['hour_bins_total']}`, train rate `{row['train_hour_bin_rate']:.4f}`, "
            f"test rate `{row['test_hour_bin_rate']:.4f}`"
        )
    lines.extend(["", "## Feature groups", ""])
    for group, metrics in report["feature_groups"].items():
        lines.append(f"- `{group}`: `{', '.join(metrics)}`")
    lines.extend(["", "## Leakage guards", ""])
    for guard in report["leakage_guards"]:
        lines.append(f"- {guard}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0 if report["decision"]["design_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
