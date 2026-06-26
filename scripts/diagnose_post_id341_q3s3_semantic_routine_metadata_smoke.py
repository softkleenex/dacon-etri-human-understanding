#!/usr/bin/env python3
"""Bounded metadata/cardinality smoke for post-ID341 Q3/S3 semantic-routine v2.

This is the first allowed step after the semantic-routine v2 design. It reads
only bounded samples from raw parquet files to verify that the proposed Q3
novelty and S3 state-machine feature families are structurally computable. It
does not train a model, compute OOF predictions, create full-test predictions,
create submission artifacts, push/run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_semantic_routine_metadata_smoke_20260618.json"
DESIGN_REPORT = REPORTS_DIR / "post_id341_q3s3_semantic_routine_v2_design_20260618.json"
MAX_SAMPLE_ROWS_PER_FILE = 4096
BATCH_SIZE = 1024

Q3_SOURCES = {
    "mUsageStats": ("ch2025_mUsageStats.parquet", "m_usage_stats"),
    "mWifi": ("ch2025_mWifi.parquet", "m_wifi"),
    "mBle": ("ch2025_mBle.parquet", "m_ble"),
    "mGps": ("ch2025_mGps.parquet", "m_gps"),
    "mAmbience": ("ch2025_mAmbience.parquet", "m_ambience"),
}
S3_SOURCES = {
    "mScreenStatus": ("ch2025_mScreenStatus.parquet", "m_screen_use"),
    "mActivity": ("ch2025_mActivity.parquet", "m_activity"),
    "mACStatus": ("ch2025_mACStatus.parquet", "m_charging"),
    "mLight": ("ch2025_mLight.parquet", "m_light"),
    "wLight": ("ch2025_wLight.parquet", "w_light"),
    "wPedo": ("ch2025_wPedo.parquet", "step"),
    "wHr": ("ch2025_wHr.parquet", "heart_rate"),
}
SOURCE_FILES = {**Q3_SOURCES, **S3_SOURCES}
RAW_IDENTIFIER_KEYS = {"app_name", "bssid", "address", "latitude", "longitude"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_for(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "error": None,
    }


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "as_py"):
        return as_list(value.as_py())
    return []


def numeric_value(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def anchored_rel_hour(timestamp: Any) -> float | None:
    if not hasattr(timestamp, "hour"):
        return None
    row_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if timestamp.hour < 12:
        row_start -= timedelta(days=1)
    return (timestamp - row_start).total_seconds() / 3600.0


def time_bucket(rel_hour: float | None) -> str:
    if rel_hour is None:
        return "missing"
    if 18 <= rel_hour < 22:
        return "evening_18_22"
    if 22 <= rel_hour < 24:
        return "late_22_24"
    if 24 <= rel_hour < 27:
        return "onset_24_27"
    if 27 <= rel_hour < 31:
        return "deep_27_31"
    if 31 <= rel_hour < 36:
        return "morning_31_36"
    return "other_12_18"


def bounded_rows(path: Path, value_col: str, max_rows: int = MAX_SAMPLE_ROWS_PER_FILE) -> tuple[list[dict[str, Any]], int]:
    parquet_file = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    columns = ["subject_id", "timestamp", value_col]
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        rows.extend(batch.to_pylist())
        if len(rows) >= max_rows:
            return rows[:max_rows], parquet_file.metadata.num_rows
    return rows, parquet_file.metadata.num_rows


def count_nested_items(source: str, value: Any, unique_counter: Counter[str]) -> int:
    items = as_list(value)
    if source == "mUsageStats":
        for item in items:
            if isinstance(item, dict) and item.get("app_name"):
                unique_counter[str(item["app_name"])] += 1
        return len(items)
    if source == "mWifi":
        for item in items:
            if isinstance(item, dict) and item.get("bssid"):
                unique_counter[str(item["bssid"])] += 1
        return len(items)
    if source == "mBle":
        for item in items:
            if isinstance(item, dict) and item.get("address"):
                unique_counter[str(item["address"])] += 1
        return len(items)
    if source == "mGps":
        for item in items:
            if not isinstance(item, dict):
                continue
            lat = numeric_value(item.get("latitude"))
            lon = numeric_value(item.get("longitude"))
            if lat is not None and lon is not None:
                unique_counter[f"{round(lat, 3)}:{round(lon, 3)}"] += 1
        return len(items)
    if source == "mAmbience":
        for item in items:
            if isinstance(item, list | tuple) and item:
                unique_counter[str(item[0])] += 1
        return len(items)
    if source == "wHr":
        return len(items)
    return 0


def source_sample_summary(source: str, filename: str, value_col: str) -> dict[str, Any]:
    path = RAW_ITEMS_DIR / filename
    if not path.exists():
        return {"source": source, "path": rel(path), "exists": False, "sample_rows": 0, "error": None}
    try:
        rows, total_rows = bounded_rows(path, value_col)
    except Exception as exc:  # noqa: BLE001
        return {
            "source": source,
            "path": rel(path),
            "exists": True,
            "sample_rows": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    unique_counter: Counter[str] = Counter()
    nested_lengths: list[int] = []
    scalar_values: list[float] = []
    subjects: set[str] = set()
    buckets: Counter[str] = Counter()
    non_null_rows = 0
    for row in rows:
        if row.get("subject_id") is not None:
            subjects.add(str(row["subject_id"]))
        buckets[time_bucket(anchored_rel_hour(row.get("timestamp")))] += 1
        value = row.get(value_col)
        if value is not None:
            non_null_rows += 1
        nested_len = count_nested_items(source, value, unique_counter)
        if nested_len:
            nested_lengths.append(nested_len)
        num = numeric_value(value)
        if num is not None:
            scalar_values.append(num)

    unique_values = set(scalar_values)
    nested_nonempty_share = len(nested_lengths) / len(rows) if rows else 0.0
    scalar_variation = len(unique_values) >= 2
    temporal_coverage_ready = bool(buckets["evening_18_22"] and buckets["late_22_24"] and buckets["onset_24_27"])
    nested_cardinality_ready = bool(
        source in Q3_SOURCES and len(unique_counter) >= 2 and nested_nonempty_share > 0 and temporal_coverage_ready
    )
    state_source_ready = bool(
        source in S3_SOURCES
        and non_null_rows > 0
        and temporal_coverage_ready
        and (scalar_variation or source == "wHr")
    )
    return {
        "source": source,
        "path": rel(path),
        "exists": True,
        "error": None,
        "total_rows": total_rows,
        "sample_rows": len(rows),
        "max_sample_rows": MAX_SAMPLE_ROWS_PER_FILE,
        "bounded_sample_only": len(rows) <= MAX_SAMPLE_ROWS_PER_FILE,
        "sample_subject_count": len(subjects),
        "non_null_row_share": non_null_rows / len(rows) if rows else 0.0,
        "time_bucket_counts": dict(sorted(buckets.items())),
        "temporal_coverage_ready": temporal_coverage_ready,
        "nested_item_rows": len(nested_lengths),
        "nested_item_mean_per_nonempty_row": float(sum(nested_lengths) / len(nested_lengths)) if nested_lengths else 0.0,
        "unique_nested_token_count": len(unique_counter),
        "scalar_unique_count": len(unique_values),
        "nested_cardinality_ready": nested_cardinality_ready,
        "state_source_ready": state_source_ready,
        "privacy_note": "raw identifiers/coordinates are summarized only as counts; no raw token examples are emitted",
    }


def build_report() -> dict[str, Any]:
    design_status = status_for(DESIGN_REPORT)
    design_decision = design_status["decision"]
    source_summaries = {
        source: source_sample_summary(source, filename, value_col)
        for source, (filename, value_col) in SOURCE_FILES.items()
    }

    q3_ready_sources = [source for source in Q3_SOURCES if source_summaries[source].get("nested_cardinality_ready")]
    s3_ready_sources = [source for source in S3_SOURCES if source_summaries[source].get("state_source_ready")]
    all_bounded = all(summary.get("bounded_sample_only") is True for summary in source_summaries.values())
    all_readable = all(summary.get("exists") is True and summary.get("error") is None for summary in source_summaries.values())
    design_ready = bool(
        design_status["parse_ok"]
        and design_decision.get("design_ready") is True
        and design_decision.get("next_local_allowed_step") == "metadata_cardinality_and_state_transition_smoke_only"
        and design_decision.get("create_submission_artifact") is False
        and design_decision.get("submit_now") is False
    )
    q3_metadata_ready = len(q3_ready_sources) >= 4
    s3_metadata_ready = len(s3_ready_sources) >= 5
    metadata_smoke_passed = bool(design_ready and all_readable and all_bounded and q3_metadata_ready and s3_metadata_ready)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_semantic_routine_metadata_smoke",
        "purpose": (
            "Verify with bounded raw samples that Q3 semantic novelty and S3 ordered state-transition features are "
            "structurally computable before any model or artifact work."
        ),
        "sample_policy": {
            "max_sample_rows_per_file": MAX_SAMPLE_ROWS_PER_FILE,
            "batch_size": BATCH_SIZE,
            "full_file_load_allowed": False,
            "model_training_allowed": False,
            "raw_identifier_keys_not_emitted": sorted(RAW_IDENTIFIER_KEYS),
        },
        "inputs": {
            "semantic_routine_v2_design": design_status,
        },
        "source_summaries": source_summaries,
        "checks": {
            "design_ready": design_ready,
            "all_sources_readable": all_readable,
            "bounded_sample_only": all_bounded,
            "q3_metadata_ready": q3_metadata_ready,
            "s3_state_machine_metadata_ready": s3_metadata_ready,
            "privacy_no_raw_identifier_examples": True,
            "no_artifact_actions": True,
        },
        "decision": {
            "metadata_smoke_passed": metadata_smoke_passed,
            "target_focus": ["Q3", "S3"],
            "q3_ready_sources": q3_ready_sources,
            "s3_ready_sources": s3_ready_sources,
            "selected_targets": [],
            "public_058_evidence": False,
            "next_local_allowed_step": (
                "resource_gated_fold_internal_semantic_routine_model_smoke_only"
                if metadata_smoke_passed
                else "fix_semantic_routine_metadata_coverage_first"
            ),
            "requires_resource_gate_before_model_smoke": True,
            "requires_explicit_kaggle_approval_for_heavy_execution": True,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Metadata/cardinality smoke passed. The next step may only be a no-artifact resource-gated "
                "fold-internal model smoke; do not create ID342 or any submission artifacts."
                if metadata_smoke_passed
                else "Do not model yet; inspect missing source coverage/cardinality before continuing this lane."
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


def fmt_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Q3/S3 semantic-routine metadata smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        report["purpose"],
        "",
        "This is report-only and bounded-sample only. It creates no predictions, CSVs, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Metadata smoke passed: `{decision['metadata_smoke_passed']}`",
        f"- Target focus: `{decision['target_focus']}`",
        f"- Q3 ready sources: `{decision['q3_ready_sources']}`",
        f"- S3 ready sources: `{decision['s3_ready_sources']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
        f"- Requires resource gate before model smoke: `{decision['requires_resource_gate_before_model_smoke']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Sample policy",
        "",
    ]
    for key, value in report["sample_policy"].items():
        if isinstance(value, list):
            rendered = ", ".join(f"`{item}`" for item in value)
            lines.append(f"- {key}: {rendered}")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Source summaries", ""])
    lines.append(
        "| source | sample rows | non-null share | temporal coverage | nested tokens | scalar uniques | ready |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for source, summary in report["source_summaries"].items():
        ready = summary.get("nested_cardinality_ready") or summary.get("state_source_ready")
        lines.append(
            "| "
            f"{source} | {summary.get('sample_rows')} | {fmt_float(summary.get('non_null_row_share'))} | "
            f"{summary.get('temporal_coverage_ready')} | {summary.get('unique_nested_token_count')} | "
            f"{summary.get('scalar_unique_count')} | {ready} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
