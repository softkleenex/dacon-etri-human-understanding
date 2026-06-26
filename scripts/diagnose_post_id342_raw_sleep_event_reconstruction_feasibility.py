#!/usr/bin/env python3
"""Post-ID342 raw sleep event reconstruction feasibility audit.

This report-only diagnostic checks whether the public raw files contain direct
sleep-stage, SOL, or WASO event fields that could support a materially new
S4/S3 route after processed sleep fragmentation proxies failed.

It creates no predictions, feature matrices, base prediction artifacts,
submission CSVs, submission-log rows, Kaggle actions, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_raw_sleep_event_reconstruction_feasibility_20260625.json"
S4S3_GATE_REPORT = REPORTS_DIR / "post_id342_s4s3_sleep_fragmentation_gate_20260625.json"
KST = timezone(timedelta(hours=9))

DIRECT_SLEEP_EVENT_TOKENS = (
    "sleep_stage",
    "stage",
    "sleep_state",
    "sol",
    "sleep_onset",
    "latency",
    "waso",
    "wake_after",
    "wakefulness_after",
    "sleep_start",
    "sleep_end",
    "bedtime",
    "wake_time",
)
PROXY_TOKENS = (
    "heart_rate",
    "w_light",
    "m_light",
    "m_screen_use",
    "m_charging",
    "m_activity",
    "step",
    "timestamp",
)
EXPECTED_RAW_FILES = {
    "ch2025_wHr.parquet": "wearable heart rate sequence; proxy for sleep depth/awake state only",
    "ch2025_wLight.parquet": "wearable light sequence; proxy for darkness/light exposure",
    "ch2025_mLight.parquet": "phone light sequence; proxy for environment exposure",
    "ch2025_mScreenStatus.parquet": "phone screen on/off; proxy for pre-sleep usage",
    "ch2025_mACStatus.parquet": "charging state; proxy for bedtime context",
    "ch2025_mActivity.parquet": "phone activity state; proxy for pre-sleep movement",
    "ch2025_wPedo.parquet": "wearable steps; proxy for rest/activity",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def schema_row(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    columns = list(df.columns)
    lower_columns = {column: column.lower() for column in columns}
    direct_hits = [
        column for column, lower in lower_columns.items() if any(token in lower for token in DIRECT_SLEEP_EVENT_TOKENS)
    ]
    proxy_hits = [column for column, lower in lower_columns.items() if any(token in lower for token in PROXY_TOKENS)]
    return {
        "path": rel(path),
        "role": EXPECTED_RAW_FILES.get(path.name, "other raw modality"),
        "rows": int(df.shape[0]),
        "columns": columns,
        "direct_sleep_event_columns": direct_hits,
        "proxy_columns": proxy_hits,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": rel(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["exists"] = True
    payload["path"] = rel(path)
    return payload


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    schema_rows = [schema_row(path) for path in sorted(RAW_ITEMS_DIR.glob("*.parquet"))]
    expected_rows = [row for row in schema_rows if Path(row["path"]).name in EXPECTED_RAW_FILES]
    direct_hits = [
        {"path": row["path"], "columns": row["direct_sleep_event_columns"]}
        for row in schema_rows
        if row["direct_sleep_event_columns"]
    ]
    proxy_available = [
        {"path": row["path"], "proxy_columns": row["proxy_columns"]} for row in expected_rows if row["proxy_columns"]
    ]
    s4s3_gate = load_json(S4S3_GATE_REPORT)
    s4s3_decision = s4s3_gate.get("decision", {}) if isinstance(s4s3_gate.get("decision"), dict) else {}
    decision = {
        "raw_files_scanned": len(schema_rows),
        "expected_sleep_proxy_files_present": len(expected_rows),
        "direct_sleep_event_column_count": sum(len(row["direct_sleep_event_columns"]) for row in schema_rows),
        "direct_sleep_event_route_available": bool(direct_hits),
        "proxy_modalities_available": len(proxy_available),
        "processed_s4s3_gate_exists": s4s3_gate.get("exists", False),
        "processed_s4s3_eligible_rows": s4s3_decision.get("eligible_row_count"),
        "processed_s4s3_selected_targets": s4s3_decision.get("selected_targets"),
        "raw_event_reconstruction_escalate_now": False,
        "create_feature_matrix": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "The public raw files do not expose direct sleep-stage, SOL, or WASO event fields. Existing processed "
            "HR/light/screen proxies have also failed the S4/S3 gate. Do not run another processed-column overlay; "
            "either design a genuinely sequence-level HR/light event model with a new local gate, or prioritize "
            "validation/public-gap redesign."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_raw_sleep_event_reconstruction_feasibility",
        "purpose": "Audit raw schema support for direct S4 WASO / S3 SOL event reconstruction.",
        "direct_sleep_event_tokens": DIRECT_SLEEP_EVENT_TOKENS,
        "proxy_tokens": PROXY_TOKENS,
        "schema_rows": schema_rows,
        "direct_sleep_event_hits": direct_hits,
        "proxy_modalities": proxy_available,
        "input_reports": {
            "s4s3_sleep_fragmentation_gate": {
                "path": rel(S4S3_GATE_REPORT),
                "exists": s4s3_gate.get("exists", False),
            }
        },
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 raw sleep event reconstruction feasibility",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Raw files scanned: `{decision['raw_files_scanned']}`",
        f"- Direct sleep event columns: `{decision['direct_sleep_event_column_count']}`",
        f"- Direct event route available: `{decision['direct_sleep_event_route_available']}`",
        f"- Proxy modalities available: `{decision['proxy_modalities_available']}`",
        f"- Processed S4/S3 eligible rows: `{decision['processed_s4s3_eligible_rows']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Raw Schema",
        "",
        "| file | rows | direct event columns | proxy columns | role |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["schema_rows"]:
        lines.append(
            f"| `{Path(row['path']).name}` | `{row['rows']}` | `{len(row['direct_sleep_event_columns'])}` | "
            f"`{len(row['proxy_columns'])}` | {row['role']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Direct sleep event route available: {decision['direct_sleep_event_route_available']}")


if __name__ == "__main__":
    main()
