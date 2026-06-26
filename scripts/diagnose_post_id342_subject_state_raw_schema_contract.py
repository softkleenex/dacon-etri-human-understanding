#!/usr/bin/env python3
"""Post-ID342 subject-state raw schema contract.

This report-only diagnostic validates that the raw parquet metadata matches the
domain/token contract for the remaining heavy route. It reads parquet schema
metadata only; it does not read row payloads, write token caches, fit embeddings,
create features, push/run Kaggle, edit submission logs, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_subject_state_raw_schema_contract_20260624.json"
DOMAIN_TOKEN_CONTRACT = REPORTS_DIR / "post_id342_subject_state_domain_token_contract_20260624.json"
HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"
KST = timezone(timedelta(hours=9))

EXPECTED_SCHEMAS = {
    "mACStatus": {
        "path": RAW_DIR / "ch2025_mACStatus.parquet",
        "required_columns": ["subject_id", "timestamp", "m_charging"],
        "value_columns": ["m_charging"],
        "token_role": "charge_bedside_routine",
    },
    "mActivity": {
        "path": RAW_DIR / "ch2025_mActivity.parquet",
        "required_columns": ["subject_id", "timestamp", "m_activity"],
        "value_columns": ["m_activity"],
        "token_role": "phone_activity_state",
    },
    "mAmbience": {
        "path": RAW_DIR / "ch2025_mAmbience.parquet",
        "required_columns": ["subject_id", "timestamp", "m_ambience"],
        "value_columns": ["m_ambience"],
        "token_role": "ambient_sound_context",
    },
    "mBle": {
        "path": RAW_DIR / "ch2025_mBle.parquet",
        "required_columns": ["subject_id", "timestamp", "m_ble"],
        "value_columns": ["m_ble"],
        "token_role": "near_device_social_context",
    },
    "mGps": {
        "path": RAW_DIR / "ch2025_mGps.parquet",
        "required_columns": ["subject_id", "timestamp", "m_gps"],
        "value_columns": ["m_gps"],
        "token_role": "mobility_place_routine",
    },
    "mLight": {
        "path": RAW_DIR / "ch2025_mLight.parquet",
        "required_columns": ["subject_id", "timestamp", "m_light"],
        "value_columns": ["m_light"],
        "token_role": "phone_light_context",
    },
    "mScreenStatus": {
        "path": RAW_DIR / "ch2025_mScreenStatus.parquet",
        "required_columns": ["subject_id", "timestamp", "m_screen_use"],
        "value_columns": ["m_screen_use"],
        "token_role": "sleep_boundary_phone_use",
    },
    "mUsageStats": {
        "path": RAW_DIR / "ch2025_mUsageStats.parquet",
        "required_columns": ["subject_id", "timestamp", "m_usage_stats"],
        "value_columns": ["m_usage_stats"],
        "token_role": "phone_app_load",
    },
    "mWifi": {
        "path": RAW_DIR / "ch2025_mWifi.parquet",
        "required_columns": ["subject_id", "timestamp", "m_wifi"],
        "value_columns": ["m_wifi"],
        "token_role": "place_stability",
    },
    "wHr": {
        "path": RAW_DIR / "ch2025_wHr.parquet",
        "required_columns": ["subject_id", "timestamp", "heart_rate"],
        "value_columns": ["heart_rate"],
        "token_role": "wearable_recovery_arousal",
    },
    "wLight": {
        "path": RAW_DIR / "ch2025_wLight.parquet",
        "required_columns": ["subject_id", "timestamp", "w_light"],
        "value_columns": ["w_light"],
        "token_role": "watch_light_context",
    },
    "wPedo": {
        "path": RAW_DIR / "ch2025_wPedo.parquet",
        "required_columns": [
            "subject_id",
            "timestamp",
            "step",
            "step_frequency",
            "running_step",
            "walking_step",
            "distance",
            "speed",
            "burned_calories",
        ],
        "value_columns": [
            "step",
            "step_frequency",
            "running_step",
            "walking_step",
            "distance",
            "speed",
            "burned_calories",
        ],
        "token_role": "wearable_exertion_inactivity",
    },
}

FORBIDDEN_ACTIONS_NOW = {
    "read_parquet_row_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_downstream_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
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
    return {"path": rel(path), "exists": True, "parse_ok": True, "decision": payload.get("decision", {})}


def schema_for(modality: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = spec["path"]
    if not path.exists():
        return {
            "modality": modality,
            "path": rel(path),
            "exists": False,
            "schema_ok": False,
            "metadata_ok": False,
            "required_columns": spec["required_columns"],
            "missing_columns": spec["required_columns"],
            "extra_columns": [],
            "value_columns": spec["value_columns"],
            "token_role": spec["token_role"],
        }

    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    missing = [col for col in spec["required_columns"] if col not in columns]
    extra = [col for col in columns if col not in spec["required_columns"]]
    schema_fields = {
        field.name: {"type": str(field.type), "nullable": field.nullable}
        for field in parquet_file.schema_arrow
        if field.name in spec["required_columns"]
    }
    return {
        "modality": modality,
        "path": rel(path),
        "exists": True,
        "schema_ok": not missing,
        "metadata_ok": True,
        "row_count_metadata": parquet_file.metadata.num_rows,
        "row_group_count_metadata": parquet_file.metadata.num_row_groups,
        "columns": columns,
        "schema_fields": schema_fields,
        "required_columns": spec["required_columns"],
        "missing_columns": missing,
        "extra_columns": extra,
        "value_columns": spec["value_columns"],
        "token_role": spec["token_role"],
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    domain_contract = read_json(DOMAIN_TOKEN_CONTRACT)
    schemas = [schema_for(modality, spec) for modality, spec in EXPECTED_SCHEMAS.items()]
    missing_files = [row["modality"] for row in schemas if not row["exists"]]
    schema_failures = [row["modality"] for row in schemas if not row["schema_ok"]]
    total_rows = sum(row.get("row_count_metadata", 0) for row in schemas)
    metadata_only_ok = not missing_files and not schema_failures
    domain_decision = domain_contract.get("decision", {})
    domain_ready = domain_decision.get("domain_token_contract_ready") is True
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_raw_schema_contract",
        "purpose": "Validate raw parquet schema metadata for the post-ID342 subject-state heavy route.",
        "input_reports": {"domain_token_contract": domain_contract},
        "raw_schema_rows": schemas,
        "schema_summary": {
            "expected_modality_count": len(EXPECTED_SCHEMAS),
            "present_modality_count": sum(1 for row in schemas if row["exists"]),
            "schema_ok_count": sum(1 for row in schemas if row["schema_ok"]),
            "missing_files": missing_files,
            "schema_failures": schema_failures,
            "total_rows_from_metadata_only": total_rows,
            "metadata_only_read": True,
        },
        "decision": {
            "raw_schema_contract_ready": metadata_only_ok and domain_ready,
            "domain_token_contract_ready": domain_ready,
            "metadata_only_read": True,
            "raw_payload_rows_read": 0,
            "target_focus": ["Q2", "Q3", "S3"],
            "primary_score_targets": ["Q3", "S3"],
            "secondary_sanity_target": "Q2",
            "requires_exact_heavy_compute_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "read_raw_parquet_payloads_now": False,
            "write_token_cache_now": False,
            "write_feature_matrix_now": False,
            "fit_embedding_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Raw schemas match the domain-token contract; implement no-submit tokenization only after exact approval."
                if metadata_only_ok and domain_ready
                else "Repair schema/domain prerequisites before any runner implementation."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_parquet_payload_rows_read": True,
        "no_token_cache_created": True,
        "no_feature_matrix_created": True,
        "no_embedding_fit": True,
        "no_prediction_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 subject-state raw schema contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report reads parquet metadata only. It creates no token cache, feature matrix, embedding, prediction, Kaggle run, submission artifact, log row, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Raw schema contract ready: `{decision['raw_schema_contract_ready']}`",
        f"- Domain/token contract ready: `{decision['domain_token_contract_ready']}`",
        f"- Metadata-only read: `{decision['metadata_only_read']}`",
        f"- Raw payload rows read: `{decision['raw_payload_rows_read']}`",
        f"- Primary score targets: `{decision['primary_score_targets']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Required phrase: `{decision['heavy_approval_phrase']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Schema Summary",
        "",
        "```json",
        json.dumps(report["schema_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Raw Schema Rows",
        "",
        "| modality | rows metadata | columns | schema ok | missing columns | token role |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in report["raw_schema_rows"]:
        lines.append(
            f"| `{row['modality']}` | `{row.get('row_count_metadata', 'missing')}` | "
            f"`{row.get('columns', [])}` | `{row['schema_ok']}` | `{row['missing_columns']}` | "
            f"`{row['token_role']}` |"
        )

    lines.extend(
        [
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Raw schema contract ready: {report['decision']['raw_schema_contract_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
