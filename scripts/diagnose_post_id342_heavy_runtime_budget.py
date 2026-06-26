#!/usr/bin/env python3
"""Post-ID342 heavy route runtime budget from parquet metadata.

This report-only diagnostic estimates the approved-later heavy runner's input
and token-memory budget using file sizes and parquet footer metadata only. It
does not read raw row payloads.

It creates no runner files, token cache, feature matrix, embedding, prediction,
submission artifact, submission-log row, Kaggle run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_heavy_runtime_budget_20260624.json"
KST = timezone(timedelta(hours=9))

HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"

REPORT_PATHS = {
    "implementation_feasibility": REPORTS_DIR / "post_id342_heavy_implementation_feasibility_20260624.json",
    "external_research_priority": REPORTS_DIR / "post_id342_external_research_priority_20260624.json",
    "raw_schema_contract": REPORTS_DIR / "post_id342_subject_state_raw_schema_contract_20260624.json",
    "success_thresholds": REPORTS_DIR / "post_id342_subject_state_heavy_success_thresholds_20260624.json",
}

RAW_FILES = [
    ("mACStatus", ROOT / "data/raw/ch2025_data_items/ch2025_mACStatus.parquet"),
    ("mActivity", ROOT / "data/raw/ch2025_data_items/ch2025_mActivity.parquet"),
    ("mAmbience", ROOT / "data/raw/ch2025_data_items/ch2025_mAmbience.parquet"),
    ("mBle", ROOT / "data/raw/ch2025_data_items/ch2025_mBle.parquet"),
    ("mGps", ROOT / "data/raw/ch2025_data_items/ch2025_mGps.parquet"),
    ("mLight", ROOT / "data/raw/ch2025_data_items/ch2025_mLight.parquet"),
    ("mScreenStatus", ROOT / "data/raw/ch2025_data_items/ch2025_mScreenStatus.parquet"),
    ("mUsageStats", ROOT / "data/raw/ch2025_data_items/ch2025_mUsageStats.parquet"),
    ("mWifi", ROOT / "data/raw/ch2025_data_items/ch2025_mWifi.parquet"),
    ("wHr", ROOT / "data/raw/ch2025_data_items/ch2025_wHr.parquet"),
    ("wLight", ROOT / "data/raw/ch2025_data_items/ch2025_wLight.parquet"),
    ("wPedo", ROOT / "data/raw/ch2025_data_items/ch2025_wPedo.parquet"),
]

TOKEN_BUDGET_ASSUMPTIONS = {
    "source": "DACON/ETRI data page describes about 700 days of public lifelog records.",
    "assumed_day_records": 700,
    "modalities": 12,
    "day_grid_hourly_slots": 24,
    "evening_grid_half_hour_slots": 18,
    "sleep_transition_event_slots": 32,
    "projection_dim": 64,
    "float32_bytes": 4,
    "float16_bytes": 2,
    "recommended_scanner_batch_rows": 65536,
    "recommended_local_day_batch_for_smoke_after_approval": 16,
    "recommended_kaggle_day_batch_after_approval": 64,
}

FORBIDDEN_ACTIONS_NOW = {
    "create_runner_files": False,
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "pull_kaggle_output": False,
    "submit_dacon": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
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
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "diagnostic": payload.get("diagnostic"),
        "decision": payload.get("decision", {}),
    }


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def run_text(cmd: list[str]) -> str | None:
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return completed.stdout.strip()


def local_resource_snapshot() -> dict[str, Any]:
    mem_text = run_text(["sysctl", "-n", "hw.memsize"])
    cpu_text = run_text(["sysctl", "-n", "hw.logicalcpu"])
    vm_text = run_text(["vm_stat"])
    total_mem_bytes = int(mem_text) if mem_text and mem_text.isdigit() else None
    free_pages = None
    page_size = None
    compressor_pages = None
    if vm_text:
        first_line = vm_text.splitlines()[0] if vm_text.splitlines() else ""
        if "page size of" in first_line:
            page_size = int(first_line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
        for line in vm_text.splitlines():
            if line.startswith("Pages free:"):
                free_pages = int(line.rsplit(" ", 1)[-1].rstrip("."))
            elif line.startswith("Pages occupied by compressor:"):
                compressor_pages = int(line.rsplit(" ", 1)[-1].rstrip("."))
    return {
        "total_mem_bytes": total_mem_bytes,
        "logical_cpu": int(cpu_text) if cpu_text and cpu_text.isdigit() else None,
        "vm_page_size": page_size,
        "vm_free_pages": free_pages,
        "vm_free_bytes_estimate": free_pages * page_size if free_pages is not None and page_size else None,
        "vm_compressor_pages": compressor_pages,
        "vm_compressor_bytes_estimate": compressor_pages * page_size
        if compressor_pages is not None and page_size
        else None,
    }


def parquet_metadata_row(modality: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "modality": modality,
            "path": rel(path),
            "exists": False,
            "metadata_read_ok": False,
            "payload_rows_read": 0,
            "error": "missing",
        }
    try:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        row_group_rows = []
        compressed_bytes = 0
        uncompressed_bytes = 0
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            row_group_rows.append(row_group.num_rows)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                compressed_bytes += int(column.total_compressed_size or 0)
                uncompressed_bytes += int(column.total_uncompressed_size or 0)
    except Exception as exc:  # noqa: BLE001
        return {
            "modality": modality,
            "path": rel(path),
            "exists": True,
            "metadata_read_ok": False,
            "payload_rows_read": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    file_size = path.stat().st_size
    min_row_group = min(row_group_rows) if row_group_rows else None
    max_row_group = max(row_group_rows) if row_group_rows else None
    return {
        "modality": modality,
        "path": rel(path),
        "exists": True,
        "metadata_read_ok": True,
        "payload_rows_read": 0,
        "file_size_bytes": file_size,
        "rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
        "min_row_group_rows": min_row_group,
        "max_row_group_rows": max_row_group,
        "compressed_column_bytes": compressed_bytes,
        "uncompressed_column_bytes": uncompressed_bytes,
        "columns": [metadata.schema.column(i).name for i in range(metadata.num_columns)],
    }


def parquet_metadata_rows() -> list[dict[str, Any]]:
    return [parquet_metadata_row(modality, path) for modality, path in RAW_FILES]


def mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def aggregate_parquet_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("metadata_read_ok") is True]
    total_file_size = sum(int(row.get("file_size_bytes", 0)) for row in ok_rows)
    total_rows = sum(int(row.get("rows", 0)) for row in ok_rows)
    total_row_groups = sum(int(row.get("row_groups", 0)) for row in ok_rows)
    total_compressed = sum(int(row.get("compressed_column_bytes", 0)) for row in ok_rows)
    total_uncompressed = sum(int(row.get("uncompressed_column_bytes", 0)) for row in ok_rows)
    max_file = max(ok_rows, key=lambda row: int(row.get("file_size_bytes", 0))) if ok_rows else {}
    one_group_modalities = [row["modality"] for row in ok_rows if row.get("row_groups") == 1]
    return {
        "metadata_read_ok_count": len(ok_rows),
        "expected_modality_count": len(RAW_FILES),
        "total_file_size_bytes": total_file_size,
        "total_file_size_mib": mib(total_file_size),
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "total_compressed_column_bytes": total_compressed,
        "total_uncompressed_column_bytes": total_uncompressed,
        "total_uncompressed_column_mib": mib(total_uncompressed),
        "largest_file_modality": max_file.get("modality"),
        "largest_file_size_mib": mib(max_file.get("file_size_bytes")),
        "one_row_group_modality_count": len(one_group_modalities),
        "one_row_group_modalities": one_group_modalities,
    }


def token_budget() -> dict[str, Any]:
    assumptions = TOKEN_BUDGET_ASSUMPTIONS
    tokens_per_day = (
        assumptions["modalities"]
        * (assumptions["day_grid_hourly_slots"] + assumptions["evening_grid_half_hour_slots"])
        + assumptions["sleep_transition_event_slots"]
    )
    total_tokens = assumptions["assumed_day_records"] * tokens_per_day
    projection_dim = assumptions["projection_dim"]
    token_tensor_fp32_bytes = total_tokens * projection_dim * assumptions["float32_bytes"]
    token_tensor_fp16_bytes = total_tokens * projection_dim * assumptions["float16_bytes"]
    local_batch_tokens = assumptions["recommended_local_day_batch_for_smoke_after_approval"] * tokens_per_day
    kaggle_batch_tokens = assumptions["recommended_kaggle_day_batch_after_approval"] * tokens_per_day
    return {
        **assumptions,
        "tokens_per_day_record": tokens_per_day,
        "estimated_total_tokens": total_tokens,
        "estimated_projected_tensor_fp32_mib": mib(token_tensor_fp32_bytes),
        "estimated_projected_tensor_fp16_mib": mib(token_tensor_fp16_bytes),
        "local_smoke_batch_tokens_after_approval": local_batch_tokens,
        "kaggle_batch_tokens_after_approval": kaggle_batch_tokens,
        "local_smoke_projected_batch_fp32_mib": mib(
            local_batch_tokens * projection_dim * assumptions["float32_bytes"]
        ),
        "kaggle_projected_batch_fp32_mib": mib(kaggle_batch_tokens * projection_dim * assumptions["float32_bytes"]),
    }


def recommended_runtime_plan(parquet_budget: dict[str, Any]) -> list[dict[str, Any]]:
    scanner_batch_rows = TOKEN_BUDGET_ASSUMPTIONS["recommended_scanner_batch_rows"]
    all_single_row_group = parquet_budget["one_row_group_modality_count"] == parquet_budget["expected_modality_count"]
    return [
        {
            "stage": "metadata_gate",
            "allowed_now": True,
            "after_exact_approval": True,
            "plan": "Use parquet footer metadata and file sizes only.",
            "reason": "This is the current report-only stage and reads no row payload.",
        },
        {
            "stage": "payload_streaming_after_approval",
            "allowed_now": False,
            "after_exact_approval": True,
            "plan": f"Use pyarrow.dataset or polars scan batches around {scanner_batch_rows} rows; avoid whole-file reads.",
            "reason": (
                "All modalities have one row group, so row-group reads would load full modality files."
                if all_single_row_group
                else "Some modalities have large row groups; scanner batches still keep memory bounded."
            ),
        },
        {
            "stage": "token_cache_after_approval",
            "allowed_now": False,
            "after_exact_approval": True,
            "plan": "Write compact day/evening/sleep-transition tokens only after local no-submit smoke proves no CSV/log side effects.",
            "reason": "The token tensor budget is small, but cache creation is still execution work.",
        },
        {
            "stage": "embedding_fit_after_approval",
            "allowed_now": False,
            "after_exact_approval": True,
            "plan": "Prefer Kaggle no-submit execution; local Mac only for tiny smoke batches.",
            "reason": "Local memory shows compressor pressure and the goal explicitly prefers careful resource use.",
        },
    ]


def build_prerequisites(reports: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, bool]:
    feasibility = decision(reports["implementation_feasibility"])
    external = decision(reports["external_research_priority"])
    schema = decision(reports["raw_schema_contract"])
    thresholds = decision(reports["success_thresholds"])
    return {
        "input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "implementation_feasibility_ready": feasibility.get("heavy_implementation_feasibility_ready") is True,
        "external_priority_ready": external.get("external_research_priority_ready") is True,
        "raw_schema_contract_ready": schema.get("raw_schema_contract_ready") is True,
        "raw_payload_rows_read_still_zero": schema.get("raw_payload_rows_read") == 0,
        "success_thresholds_ready": thresholds.get("heavy_success_threshold_contract_ready") is True,
        "metadata_read_ok": all(row.get("metadata_read_ok") is True for row in rows),
        "payload_rows_read_now_zero": all(row.get("payload_rows_read") == 0 for row in rows),
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in REPORT_PATHS.items()}
    parquet_rows = parquet_metadata_rows()
    parquet_budget = aggregate_parquet_budget(parquet_rows)
    token_budget_payload = token_budget()
    prerequisites = build_prerequisites(reports, parquet_rows)
    runtime_budget_ready = all(prerequisites.values())
    thresholds = decision(reports["success_thresholds"])
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_heavy_runtime_budget",
        "purpose": "Estimate metadata-only raw input and token runtime budgets for a future approved heavy run.",
        "input_reports": reports,
        "local_resource_snapshot": local_resource_snapshot(),
        "parquet_metadata_rows": parquet_rows,
        "parquet_budget": parquet_budget,
        "token_budget": token_budget_payload,
        "recommended_runtime_plan": recommended_runtime_plan(parquet_budget),
        "prerequisites": prerequisites,
        "future_review_thresholds": {
            "artifact_review_public_proxy": thresholds.get("artifact_review_threshold_public_proxy"),
            "dacon_candidate_public_proxy": thresholds.get("dacon_candidate_threshold_public_proxy"),
        },
        "decision": {
            "heavy_runtime_budget_ready": runtime_budget_ready,
            "metadata_only_read": True,
            "raw_payload_rows_read": 0,
            "total_raw_file_size_mib": parquet_budget["total_file_size_mib"],
            "total_raw_rows_from_metadata": parquet_budget["total_rows"],
            "all_modalities_single_row_group": parquet_budget["one_row_group_modality_count"]
            == parquet_budget["expected_modality_count"],
            "recommended_scanner_batch_rows_after_approval": TOKEN_BUDGET_ASSUMPTIONS[
                "recommended_scanner_batch_rows"
            ],
            "recommended_local_day_batch_after_approval": TOKEN_BUDGET_ASSUMPTIONS[
                "recommended_local_day_batch_for_smoke_after_approval"
            ],
            "recommended_kaggle_day_batch_after_approval": TOKEN_BUDGET_ASSUMPTIONS[
                "recommended_kaggle_day_batch_after_approval"
            ],
            "requires_exact_heavy_compute_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "create_runner_files_now": False,
            "read_raw_parquet_payloads_now": False,
            "write_token_cache_now": False,
            "fit_embedding_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Use scanner/batch streaming after exact heavy approval; do not row-group read whole modality files."
                if runtime_budget_ready
                else "Refresh missing prerequisite reports or metadata before heavy approval execution."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_runner_files_created": True,
        "no_raw_payload_read": True,
        "no_token_cache_created": True,
        "no_embedding_fit": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_kaggle_output_pull_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_payload = report["decision"]
    lines = [
        "# Post-ID342 heavy runtime budget",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report uses file sizes and parquet footer metadata only. It reads no raw row payloads, creates no token cache, fits no embeddings, runs no Kaggle job, and submits nothing.",
        "",
        "## Decision",
        "",
        f"- Runtime budget ready: `{decision_payload['heavy_runtime_budget_ready']}`",
        f"- Metadata-only read: `{decision_payload['metadata_only_read']}`",
        f"- Raw payload rows read: `{decision_payload['raw_payload_rows_read']}`",
        f"- Total raw file size MiB: `{decision_payload['total_raw_file_size_mib']:.3f}`",
        f"- Total raw rows from metadata: `{decision_payload['total_raw_rows_from_metadata']}`",
        f"- All modalities single row group: `{decision_payload['all_modalities_single_row_group']}`",
        f"- Recommended scanner batch rows after approval: `{decision_payload['recommended_scanner_batch_rows_after_approval']}`",
        f"- Recommended local day batch after approval: `{decision_payload['recommended_local_day_batch_after_approval']}`",
        f"- Recommended Kaggle day batch after approval: `{decision_payload['recommended_kaggle_day_batch_after_approval']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- DACON submit now: `{decision_payload['dacon_submit_now']}`",
        f"- Required phrase: `{decision_payload['heavy_approval_phrase']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Parquet Budget",
        "",
        "```json",
        json.dumps(report["parquet_budget"], indent=2, sort_keys=True),
        "```",
        "",
        "## Token Budget",
        "",
        "```json",
        json.dumps(report["token_budget"], indent=2, sort_keys=True),
        "```",
        "",
        "## Runtime Plan",
        "",
        "| stage | allowed now | after exact approval | plan | reason |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["recommended_runtime_plan"]:
        lines.append(
            f"| `{row['stage']}` | `{row['allowed_now']}` | `{row['after_exact_approval']}` | "
            f"{row['plan']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Raw Metadata Rows",
            "",
            "| modality | file MiB | rows | row groups | compressed MiB | uncompressed MiB | payload rows read |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["parquet_metadata_rows"]:
        lines.append(
            f"| `{row['modality']}` | `{mib(row.get('file_size_bytes')):.3f}` | `{row.get('rows')}` | "
            f"`{row.get('row_groups')}` | `{mib(row.get('compressed_column_bytes')):.3f}` | "
            f"`{mib(row.get('uncompressed_column_bytes')):.3f}` | `{row.get('payload_rows_read')}` |"
        )
    lines.extend(
        [
            "",
            "## Local Resource Snapshot",
            "",
            "```json",
            json.dumps(report["local_resource_snapshot"], indent=2, sort_keys=True),
            "```",
            "",
            "## Prerequisites",
            "",
            "```json",
            json.dumps(report["prerequisites"], indent=2, sort_keys=True),
            "```",
            "",
            "## Forbidden Actions Now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Runtime budget ready: {report['decision']['heavy_runtime_budget_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
