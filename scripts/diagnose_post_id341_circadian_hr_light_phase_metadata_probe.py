#!/usr/bin/env python3
"""Metadata/resource probe for circadian HR/light phase-shape features.

This is report-only. It inspects raw wearable heart-rate, wearable light,
phone light, and pedometer streams to decide whether a future label-free
circadian phase-shape feature family is feasible. It trains no model, creates
no predictions, writes no submission artifact, modifies no submission log, runs
no Kaggle command, and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "circadian_hr_light_phase_shape_v1"
TARGET_FOCUS = ["Q3", "S3", "S2"]
SOURCE_FILES = {
    "wHr": {
        "path": RAW_DIR / "ch2025_wHr.parquet",
        "columns": ["subject_id", "timestamp", "heart_rate"],
        "value_col": "heart_rate",
        "purpose": "wearable autonomic/circadian recovery curve",
    },
    "wLight": {
        "path": RAW_DIR / "ch2025_wLight.parquet",
        "columns": ["subject_id", "timestamp", "w_light"],
        "value_col": "w_light",
        "purpose": "wearable light exposure phase and amplitude",
    },
    "mLight": {
        "path": RAW_DIR / "ch2025_mLight.parquet",
        "columns": ["subject_id", "timestamp", "m_light"],
        "value_col": "m_light",
        "purpose": "phone light context and dark-window continuity",
    },
    "wPedo": {
        "path": RAW_DIR / "ch2025_wPedo.parquet",
        "columns": ["subject_id", "timestamp", "step", "step_frequency", "speed"],
        "value_col": "step",
        "purpose": "movement phase and rest/activity rhythm",
    },
}
PHASE_BINS = {
    "night_00_05": list(range(0, 6)),
    "morning_06_11": list(range(6, 12)),
    "afternoon_12_17": list(range(12, 18)),
    "evening_18_23": list(range(18, 24)),
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_queue() -> dict[str, Any]:
    if not QUEUE_REPORT.exists():
        return {"exists": False, "decision": {}, "source_queue": []}
    data = json.loads(QUEUE_REPORT.read_text(encoding="utf-8"))
    data["exists"] = True
    return data


def expected_days() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True).drop_duplicates(["subject_id", "lifelog_date", "split"])


def load_source(name: str) -> pd.DataFrame:
    spec = SOURCE_FILES[name]
    df = pd.read_parquet(spec["path"], columns=spec["columns"])
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["lifelog_date"] = ts.dt.strftime("%Y-%m-%d")
    df["hour"] = ts.dt.hour
    return df


def coverage_by_split(per_day: pd.DataFrame, expected: pd.DataFrame) -> dict[str, dict[str, Any]]:
    joined = expected.merge(per_day, on=["subject_id", "lifelog_date"], how="left")
    fill_cols = [col for col in joined.columns if col not in {"subject_id", "lifelog_date", "split"}]
    joined[fill_cols] = joined[fill_cols].fillna(0)
    out = {}
    for split, split_df in joined.groupby("split"):
        covered = split_df["rows"] > 0
        out[str(split)] = {
            "expected_days": int(len(split_df)),
            "covered_days": int(covered.sum()),
            "coverage_rate": float(covered.mean()),
            "median_rows": float(split_df["rows"].median()),
            "p25_rows": float(split_df["rows"].quantile(0.25)),
            "p75_rows": float(split_df["rows"].quantile(0.75)),
            "median_hours": float(split_df["hours"].median()),
            "days_with_at_least_12_hours": int((split_df["hours"] >= 12).sum()),
            "days_with_at_least_18_hours": int((split_df["hours"] >= 18).sum()),
            "rate_with_at_least_12_hours": float((split_df["hours"] >= 12).mean()),
            "rate_with_at_least_18_hours": float((split_df["hours"] >= 18).mean()),
        }
    return out


def source_daily_summary(name: str, df: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    per_day = df.groupby(["subject_id", "lifelog_date"]).agg(rows=("timestamp", "size"), hours=("hour", "nunique"))
    for bin_name, hours in PHASE_BINS.items():
        mask = df["hour"].isin(hours)
        bin_counts = df.loc[mask].groupby(["subject_id", "lifelog_date"]).size().rename(bin_name)
        per_day = per_day.join(bin_counts, how="left")
    per_day = per_day.fillna(0).reset_index()
    bin_rates = {}
    joined = expected.merge(per_day, on=["subject_id", "lifelog_date"], how="left").fillna(0)
    for split, split_df in joined.groupby("split"):
        bin_rates[str(split)] = {bin_name: float((split_df[bin_name] > 0).mean()) for bin_name in PHASE_BINS}
    ts = pd.to_datetime(df["timestamp"])
    return {
        "source": name,
        "file": rel(SOURCE_FILES[name]["path"]),
        "purpose": SOURCE_FILES[name]["purpose"],
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "subject_days": int(len(per_day)),
        "timestamp_span": {"min": str(ts.min()), "max": str(ts.max())},
        "coverage_by_split": coverage_by_split(per_day, expected),
        "phase_bin_presence_rate_by_split": bin_rates,
    }


def heart_rate_payload_summary(df: pd.DataFrame) -> dict[str, Any]:
    lengths = df["heart_rate"].map(lambda value: len(value) if hasattr(value, "__len__") else 0)
    return {
        "rows": int(len(lengths)),
        "nonempty_payload_rate": float((lengths > 0).mean()),
        "mean_readings_per_row": float(lengths.mean()),
        "median_readings_per_row": float(lengths.median()),
        "p90_readings_per_row": float(lengths.quantile(0.90)),
    }


def daily_phase_table(frames: dict[str, pd.DataFrame], expected: pd.DataFrame) -> pd.DataFrame:
    table = expected.copy()
    for name, df in frames.items():
        per_day = (
            df.groupby(["subject_id", "lifelog_date"])
            .agg(**{f"{name}_rows": ("timestamp", "size"), f"{name}_hours": ("hour", "nunique")})
            .reset_index()
        )
        table = table.merge(per_day, on=["subject_id", "lifelog_date"], how="left")
    metric_cols = [col for col in table.columns if col.endswith("_rows") or col.endswith("_hours")]
    table[metric_cols] = table[metric_cols].fillna(0)
    table["strong_phase_shape_day"] = (
        (table["wHr_rows"] >= 240)
        & (table["wHr_hours"] >= 8)
        & (table["wLight_hours"] >= 18)
        & (table["mLight_hours"] >= 18)
        & (table["wPedo_hours"] >= 18)
    )
    table["strict_phase_shape_day"] = (
        (table["wHr_rows"] >= 300)
        & (table["wHr_hours"] >= 10)
        & (table["wLight_hours"] >= 18)
        & (table["mLight_hours"] >= 18)
        & (table["wPedo_hours"] >= 18)
    )
    table["weak_phase_shape_day"] = (
        (table["wHr_rows"] >= 120)
        & (table["wHr_hours"] >= 6)
        & (table["wLight_hours"] >= 12)
        & (table["mLight_hours"] >= 12)
        & (table["wPedo_hours"] >= 12)
    )
    table["light_pedo_without_hr_day"] = (
        (table["wLight_hours"] >= 18) & (table["mLight_hours"] >= 18) & (table["wPedo_hours"] >= 18)
    )
    return table


def phase_feasibility(table: pd.DataFrame) -> dict[str, Any]:
    by_split = {}
    for split, split_df in table.groupby("split"):
        by_split[str(split)] = {
            "expected_days": int(len(split_df)),
            "strong_phase_shape_days": int(split_df["strong_phase_shape_day"].sum()),
            "strict_phase_shape_days": int(split_df["strict_phase_shape_day"].sum()),
            "weak_phase_shape_days": int(split_df["weak_phase_shape_day"].sum()),
            "light_pedo_without_hr_days": int(split_df["light_pedo_without_hr_day"].sum()),
            "strong_phase_shape_rate": float(split_df["strong_phase_shape_day"].mean()),
            "strict_phase_shape_rate": float(split_df["strict_phase_shape_day"].mean()),
            "weak_phase_shape_rate": float(split_df["weak_phase_shape_day"].mean()),
            "light_pedo_without_hr_rate": float(split_df["light_pedo_without_hr_day"].mean()),
            "median_whr_hours": float(split_df["wHr_hours"].median()),
            "median_wlight_hours": float(split_df["wLight_hours"].median()),
            "median_mlight_hours": float(split_df["mLight_hours"].median()),
            "median_wpedo_hours": float(split_df["wPedo_hours"].median()),
        }
    subject_ready = table.groupby(["split", "subject_id"])["strong_phase_shape_day"].mean().reset_index()
    return {
        "by_split": by_split,
        "subjects_with_half_days_strong_by_split": {
            split: int((split_df["strong_phase_shape_day"] >= 0.5).sum())
            for split, split_df in subject_ready.groupby("split")
        },
    }


def build_report() -> dict[str, Any]:
    queue = read_queue()
    queue_decision = queue.get("decision", {}) if isinstance(queue.get("decision"), dict) else {}
    queue_items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    circadian_queue_item = next((item for item in queue_items if item.get("name") == SOURCE_FAMILY), {})
    expected = expected_days()
    frames = {name: load_source(name) for name in SOURCE_FILES}
    source_summaries = {name: source_daily_summary(name, frame, expected) for name, frame in frames.items()}
    table = daily_phase_table(frames, expected)
    feasibility = phase_feasibility(table)
    train = feasibility["by_split"]["train"]
    test = feasibility["by_split"]["test"]
    metadata_probe_passed = bool(
        queue.get("exists", False)
        and circadian_queue_item.get("name") == SOURCE_FAMILY
        and circadian_queue_item.get("status") is not None
        and train["strong_phase_shape_rate"] >= 0.50
        and test["strong_phase_shape_rate"] >= 0.50
        and train["weak_phase_shape_rate"] >= 0.70
        and test["weak_phase_shape_rate"] >= 0.70
    )
    next_action = (
        "Circadian HR/light phase-shape metadata is feasible enough for a design spec, but this is not score "
        "evidence. Do not train or create ID342; only a report-only feature-spec step is allowed unless the exact "
        "Kaggle no-submit approval phrase is provided."
        if metadata_probe_passed
        else "Circadian HR/light phase-shape coverage is not strong enough under the current resource gate; keep it as context only."
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_circadian_hr_light_phase_metadata_probe",
        "report_only": True,
        "queue_report": {
            "path": rel(QUEUE_REPORT),
            "exists": queue.get("exists", False),
            "next_no_approval_step": queue_decision.get("next_no_approval_step"),
            "only_score_relevant_open_route": queue_decision.get("only_score_relevant_open_route"),
            "circadian_queue_status": circadian_queue_item.get("status"),
            "circadian_next_allowed_step": circadian_queue_item.get("next_allowed_step"),
        },
        "source_summaries": source_summaries,
        "heart_rate_payload_summary": heart_rate_payload_summary(frames["wHr"]),
        "phase_feasibility": feasibility,
        "decision": {
            "metadata_probe_passed": metadata_probe_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "strong_phase_shape_train_rate": train["strong_phase_shape_rate"],
            "strong_phase_shape_test_rate": test["strong_phase_shape_rate"],
            "weak_phase_shape_train_rate": train["weak_phase_shape_rate"],
            "weak_phase_shape_test_rate": test["weak_phase_shape_rate"],
            "next_local_allowed_step": "report_only_feature_spec_no_model",
            "train_model_now": False,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": next_action,
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


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Circadian HR/light phase-shape metadata probe — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no predictions, submission CSV, base prediction artifact, "
            "notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Metadata probe passed: `{dec['metadata_probe_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Strong phase-shape train/test rates: `{dec['strong_phase_shape_train_rate']:.4f}` / "
        f"`{dec['strong_phase_shape_test_rate']:.4f}`",
        f"- Weak phase-shape train/test rates: `{dec['weak_phase_shape_train_rate']:.4f}` / "
        f"`{dec['weak_phase_shape_test_rate']:.4f}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Source coverage",
        "",
        "| source | rows | subjects | subject-days | train coverage | test coverage | train >=18h | test >=18h | purpose |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for source in report["source_summaries"].values():
        train = source["coverage_by_split"].get("train", {})
        test = source["coverage_by_split"].get("test", {})
        lines.append(
            f"| `{source['source']}` | {source['rows']} | {source['subjects']} | {source['subject_days']} | "
            f"{train.get('coverage_rate', 0.0):.4f} | {test.get('coverage_rate', 0.0):.4f} | "
            f"{train.get('rate_with_at_least_18_hours', 0.0):.4f} | "
            f"{test.get('rate_with_at_least_18_hours', 0.0):.4f} | {source['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Phase feasibility",
            "",
            "| split | expected days | strong days | weak days | strict days | strong rate | weak rate | light+pedo no-HR rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, row in report["phase_feasibility"]["by_split"].items():
        lines.append(
            f"| `{split}` | {row['expected_days']} | {row['strong_phase_shape_days']} | "
            f"{row['weak_phase_shape_days']} | {row['strict_phase_shape_days']} | "
            f"{row['strong_phase_shape_rate']:.4f} | {row['weak_phase_shape_rate']:.4f} | "
            f"{row['light_pedo_without_hr_rate']:.4f} |"
        )
    hr = report["heart_rate_payload_summary"]
    lines.extend(
        [
            "",
            "## Heart-rate payload",
            "",
            f"- Nonempty HR payload rate: `{hr['nonempty_payload_rate']:.4f}`",
            f"- Mean/median readings per HR row: `{hr['mean_readings_per_row']:.3f}` / "
            f"`{hr['median_readings_per_row']:.3f}`",
            "",
            "## Boundary",
            "",
            (
                "This report can justify only a report-only feature-spec step. It is not Public 0.58 evidence, "
                "does not open a submission artifact, and does not authorize Kaggle or DACON actions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_circadian_hr_light_phase_metadata_probe_20260619.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    write_report(report, output)
    dec = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        json.dumps(
            {
                "metadata_probe_passed": dec["metadata_probe_passed"],
                "source_family": dec["source_family"],
                "next_local_allowed_step": dec["next_local_allowed_step"],
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
