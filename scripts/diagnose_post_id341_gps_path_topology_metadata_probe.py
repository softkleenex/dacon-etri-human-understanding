#!/usr/bin/env python3
"""Metadata/resource probe for a GPS path-topology source family.

This is report-only. It reads raw GPS/place-adjacent logs to decide whether a
future path-topology feature family is feasible, but it trains no model, creates
no predictions, writes no submission artifact, modifies no submission log, runs
no Kaggle command, and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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

SOURCE_FILES = {
    "mGps": RAW_DIR / "ch2025_mGps.parquet",
    "mWifi": RAW_DIR / "ch2025_mWifi.parquet",
    "mBle": RAW_DIR / "ch2025_mBle.parquet",
    "mActivity": RAW_DIR / "ch2025_mActivity.parquet",
}


def read_queue() -> dict[str, Any]:
    if not QUEUE_REPORT.exists():
        return {"exists": False}
    data = json.loads(QUEUE_REPORT.read_text())
    data["exists"] = True
    return data


def date_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = df[["subject_id", "timestamp"]].copy()
    out["lifelog_date"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d")
    out["hour"] = pd.to_datetime(out["timestamp"]).dt.hour
    return out


def expected_days() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True).drop_duplicates(["subject_id", "lifelog_date", "split"])


def summarize_source(name: str, path: Path, expected: pd.DataFrame) -> dict[str, Any]:
    df = date_frame(path)
    per_day = df.groupby(["subject_id", "lifelog_date"]).size().rename("rows").reset_index()
    joined = expected.merge(per_day, on=["subject_id", "lifelog_date"], how="left")
    joined["rows"] = joined["rows"].fillna(0)
    by_split = {}
    for split, split_df in joined.groupby("split"):
        by_split[str(split)] = {
            "expected_days": int(len(split_df)),
            "covered_days": int((split_df["rows"] > 0).sum()),
            "coverage_rate": float((split_df["rows"] > 0).mean()),
            "median_rows_on_covered_days": float(split_df.loc[split_df["rows"] > 0, "rows"].median())
            if (split_df["rows"] > 0).any()
            else 0.0,
            "days_with_at_least_30_rows": int((split_df["rows"] >= 30).sum()),
            "days_with_at_least_60_rows": int((split_df["rows"] >= 60).sum()),
        }
    timestamp_span = {
        "min": str(pd.to_datetime(df["timestamp"]).min()),
        "max": str(pd.to_datetime(df["timestamp"]).max()),
    }
    hour_counts = df["hour"].value_counts().sort_index()
    night_rows = int(df["hour"].isin([0, 1, 2, 3, 4, 5]).sum())
    evening_rows = int(df["hour"].isin([18, 19, 20, 21, 22, 23]).sum())
    return {
        "source": name,
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "subject_days": int(len(per_day)),
        "timestamp_span": timestamp_span,
        "coverage_by_split": by_split,
        "night_rows_00_05": night_rows,
        "evening_rows_18_23": evening_rows,
        "hour_coverage_count": int(hour_counts.size),
    }


def payload_length(value: Any) -> int:
    try:
        return int(len(value))
    except TypeError:
        return 0


def gps_point_lengths(values: Iterable[Any]) -> dict[str, Any]:
    lengths = []
    valid_rows = 0
    speed_rows = 0
    for value in values:
        try:
            value_len = len(value)
        except TypeError:
            lengths.append(0)
            continue
        lengths.append(value_len)
        if value_len == 0:
            continue
        first = value[0]
        if isinstance(first, dict) and {"latitude", "longitude"}.issubset(first):
            valid_rows += 1
        if isinstance(first, dict) and "speed" in first:
            speed_rows += 1
    series = pd.Series(lengths)
    return {
        "rows_with_valid_coordinate_payload": int(valid_rows),
        "rows_with_speed_payload": int(speed_rows),
        "mean_points_per_row": float(series.mean()),
        "median_points_per_row": float(series.median()),
        "p90_points_per_row": float(series.quantile(0.90)),
        "nonempty_payload_rate": float((series > 0).mean()),
    }


def gps_daily_feasibility(expected: pd.DataFrame) -> dict[str, Any]:
    gps = pd.read_parquet(SOURCE_FILES["mGps"])
    gps_dates = gps[["subject_id", "timestamp", "m_gps"]].copy()
    gps_dates["lifelog_date"] = pd.to_datetime(gps_dates["timestamp"]).dt.strftime("%Y-%m-%d")
    gps_dates["gps_payload_len"] = gps_dates["m_gps"].map(payload_length)
    gps_dates["has_coordinate_payload"] = gps_dates["gps_payload_len"] > 0
    per_day = (
        gps_dates.groupby(["subject_id", "lifelog_date"])
        .agg(
            gps_rows=("timestamp", "size"),
            gps_payload_rows=("has_coordinate_payload", "sum"),
            gps_points=("gps_payload_len", "sum"),
        )
        .reset_index()
    )
    joined = expected.merge(per_day, on=["subject_id", "lifelog_date"], how="left").fillna(0)
    feasibility_by_split = {}
    for split, split_df in joined.groupby("split"):
        feasible = (split_df["gps_rows"] >= 60) & (split_df["gps_payload_rows"] >= 30)
        weak = (split_df["gps_rows"] >= 30) & (split_df["gps_payload_rows"] >= 10)
        feasibility_by_split[str(split)] = {
            "expected_days": int(len(split_df)),
            "strong_topology_days": int(feasible.sum()),
            "weak_topology_days": int(weak.sum()),
            "strong_topology_rate": float(feasible.mean()),
            "weak_topology_rate": float(weak.mean()),
            "median_gps_rows": float(split_df["gps_rows"].median()),
            "median_gps_points": float(split_df["gps_points"].median()),
        }
    subject_ready = (
        joined.assign(strong=(joined["gps_rows"] >= 60) & (joined["gps_payload_rows"] >= 30))
        .groupby(["split", "subject_id"])["strong"]
        .mean()
        .reset_index()
    )
    return {
        "payload": gps_point_lengths(gps["m_gps"]),
        "feasibility_by_split": feasibility_by_split,
        "subjects_with_half_days_strong_by_split": {
            split: int((split_df["strong"] >= 0.5).sum()) for split, split_df in subject_ready.groupby("split")
        },
    }


def build_report() -> dict[str, Any]:
    queue = read_queue()
    queue_decision = queue.get("decision", {}) if isinstance(queue.get("decision"), dict) else {}
    queue_items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    gps_queue_item = next((item for item in queue_items if item.get("name") == "gps_path_topology_sequence_v1"), {})
    expected = expected_days()
    source_summaries = {name: summarize_source(name, path, expected) for name, path in SOURCE_FILES.items()}
    gps_feasibility = gps_daily_feasibility(expected)
    train_feasible = gps_feasibility["feasibility_by_split"]["train"]["strong_topology_rate"]
    test_feasible = gps_feasibility["feasibility_by_split"]["test"]["strong_topology_rate"]
    metadata_probe_passed = (
        queue.get("exists", False)
        and bool(gps_queue_item)
        and train_feasible >= 0.50
        and test_feasible >= 0.50
    )
    gps_lane_already_closed = str(gps_queue_item.get("status", "")).startswith("closed_")
    if metadata_probe_passed and gps_lane_already_closed:
        next_action = (
            "GPS topology metadata was feasible and has already been consumed by the feature-matrix/model-smoke "
            "lane, which selected no Q3/S3 target. Do not replay it as a new local artifact route; continue only "
            "with the explicit-approved upstream rebuild or another report-only source family."
        )
    elif metadata_probe_passed:
        next_action = (
            "GPS topology metadata is feasible enough for a future design spec, but this is not score evidence. "
            "Do not train or create ID342; either keep this as a design-only source candidate or use the "
            "explicit-approved Kaggle upstream rebuild for score-relevant work."
        )
    else:
        next_action = (
            "GPS topology has enough raw row coverage but not enough validated coordinate-payload feasibility under "
            "the current gate. Keep it as design-only context; do not train, create ID342, or override the approved "
            "upstream rebuild route."
        )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_gps_path_topology_metadata_probe",
        "report_only": True,
        "queue_report": {
            "path": str(QUEUE_REPORT.relative_to(ROOT)),
            "exists": queue.get("exists", False),
            "next_no_approval_step": queue_decision.get("next_no_approval_step"),
            "only_score_relevant_open_route": queue_decision.get("only_score_relevant_open_route"),
            "gps_queue_status": gps_queue_item.get("status"),
            "gps_next_allowed_step": gps_queue_item.get("next_allowed_step"),
        },
        "source_summaries": source_summaries,
        "gps_feasibility": gps_feasibility,
        "decision": {
            "metadata_probe_passed": metadata_probe_passed,
            "source_family": "gps_path_topology_sequence_v1",
            "target_focus": ["Q3", "S3"],
            "strong_topology_train_rate": train_feasible,
            "strong_topology_test_rate": test_feasible,
            "next_local_allowed_step": "design_feature_spec_only",
            "train_model_now": False,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": next_action,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# GPS path-topology metadata probe — post ID341",
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
        f"- Strong topology train/test rates: `{dec['strong_topology_train_rate']:.4f}` / "
        f"`{dec['strong_topology_test_rate']:.4f}`",
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
        "| source | rows | subjects | subject-days | train coverage | test coverage | median train rows/day |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in report["source_summaries"].values():
        train = source["coverage_by_split"].get("train", {})
        test = source["coverage_by_split"].get("test", {})
        lines.append(
            f"| `{source['source']}` | {source['rows']} | {source['subjects']} | {source['subject_days']} | "
            f"{train.get('coverage_rate', 0.0):.4f} | {test.get('coverage_rate', 0.0):.4f} | "
            f"{train.get('median_rows_on_covered_days', 0.0):.1f} |"
        )
    gps = report["gps_feasibility"]
    lines.extend(
        [
            "",
            "## GPS topology feasibility",
            "",
            f"- GPS rows with coordinate payload: `{gps['payload']['rows_with_valid_coordinate_payload']}`",
            f"- Mean/median points per GPS row: `{gps['payload']['mean_points_per_row']:.3f}` / "
            f"`{gps['payload']['median_points_per_row']:.3f}`",
            f"- Nonempty payload rate: `{gps['payload']['nonempty_payload_rate']:.4f}`",
            "",
            "| split | expected days | strong topology days | weak topology days | strong rate | median GPS rows | median GPS points |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, row in gps["feasibility_by_split"].items():
        lines.append(
            f"| `{split}` | {row['expected_days']} | {row['strong_topology_days']} | "
            f"{row['weak_topology_days']} | {row['strong_topology_rate']:.4f} | "
            f"{row['median_gps_rows']:.1f} | {row['median_gps_points']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "Passing this probe only means the raw logs are dense enough to design topology features. It does not "
                "open a submission artifact, does not count as Public 0.58 evidence, and does not supersede the "
                "approved clean upstream ID221→ID226 route."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output.with_suffix(".md").write_text(markdown(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_gps_path_topology_metadata_probe_20260619.json",
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
