#!/usr/bin/env python3
"""Report-only Q3/S3 semantic-routine v2 design after post-ID341.

The previous local Q3/S3 physio-light-onset smoke did not find a selected row.
This diagnostic does not train a model. It documents a materially different
next hypothesis: event/state-transition features from usage, social proximity,
place novelty, screen, activity, charging, and ambience streams.

It creates no predictions, submission CSVs, base prediction artifacts,
notebook/history artifacts, submission-log rows, Kaggle runs, or DACON
submissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_semantic_routine_v2_design_20260618.json"
Q3S3_PHYSIO_LIGHT_SMOKE = REPORTS_DIR / "post_id341_q3s3_physio_light_onset_smoke_20260618.json"
SOURCE_MATRIX = REPORTS_DIR / "post_id341_q2q3s3_source_hypothesis_matrix_20260618.json"
DATA_COVERAGE = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"

TARGET_FOCUS = ["Q3", "S3"]
REQUIRED_RAW_FILES = {
    "mUsageStats": "ch2025_mUsageStats.parquet",
    "mWifi": "ch2025_mWifi.parquet",
    "mBle": "ch2025_mBle.parquet",
    "mGps": "ch2025_mGps.parquet",
    "mAmbience": "ch2025_mAmbience.parquet",
    "mScreenStatus": "ch2025_mScreenStatus.parquet",
    "mActivity": "ch2025_mActivity.parquet",
    "mACStatus": "ch2025_mACStatus.parquet",
    "mLight": "ch2025_mLight.parquet",
    "wLight": "ch2025_wLight.parquet",
    "wPedo": "ch2025_wPedo.parquet",
    "wHr": "ch2025_wHr.parquet",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
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


def parquet_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "columns": [], "num_rows": None, "error": None}
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return {
            "path": rel(path),
            "exists": True,
            "columns": list(parquet_file.schema.names),
            "num_rows": parquet_file.metadata.num_rows,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "columns": [],
            "num_rows": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def raw_schema_summary() -> dict[str, Any]:
    schemas = {name: parquet_schema(RAW_ITEMS_DIR / filename) for name, filename in REQUIRED_RAW_FILES.items()}
    return {
        "schemas": schemas,
        "all_required_files_present": all(schema["exists"] for schema in schemas.values()),
        "all_required_schemas_readable": all(schema["exists"] and not schema["error"] for schema in schemas.values()),
        "total_rows": sum(schema["num_rows"] or 0 for schema in schemas.values()),
    }


def build_feature_blueprint() -> list[dict[str, Any]]:
    return [
        {
            "target": "Q3",
            "family": "semantic_app_social_place_novelty",
            "sources": ["mUsageStats", "mWifi", "mBle", "mGps", "mAmbience"],
            "feature_shapes": [
                "subject-relative evening app-category entropy and late-night category switch count",
                "wifi/ble neighbor novelty count versus the subject's recent same-week baseline",
                "gps place-cluster novelty and evening route-fragmentation proxy without storing raw coordinates",
                "ambience state-change rate around the evening-to-late boundary",
                "cross-modal novelty burst: app switch + new proximity device + place transition within a short window",
            ],
            "why_distinct_from_closed_lane": (
                "The closed physio-light-onset lane used physiological/light/onset aggregates. This family treats stress "
                "as social/context novelty and routine disruption, matching the paper hint that Q3 benefits from "
                "mUsageStats and wPedo-style daily-window context."
            ),
            "fail_fast_rule": (
                "Report-only metadata smoke first; then fold-internal no-artifact model smoke only if feature coverage is "
                "complete. Stop unless Q3 beats ID341 in at least 4/5 primary splits with low joint-risk warnings."
            ),
        },
        {
            "target": "S3",
            "family": "sleep_onset_state_machine",
            "sources": ["mScreenStatus", "mActivity", "mACStatus", "mLight", "wLight", "wPedo", "wHr"],
            "feature_shapes": [
                "first sustained quiet-dark-screen-off run after 20:00 and its duration",
                "screen-off to charging-on latency and charging-state stability before sleep",
                "last movement then first dark interval transition gap",
                "state-machine transition counts among active/screen/charging/light buckets from 20:00 to 03:00",
                "subject-relative onset regularity: deviation from each subject's typical quiet-window timing",
            ],
            "why_distinct_from_closed_lane": (
                "The closed lane summarized independent sensor levels. This one models ordered transitions and sustained "
                "states, which is closer to the sleep-onset-latency label mechanism."
            ),
            "fail_fast_rule": (
                "Run metadata/cardinality smoke first. Stop unless S3 produces a low-movement fold-internal improvement; "
                "do not use Q1/S4 sleep-quality movement as a proxy for S3."
            ),
        },
    ]


def build_report() -> dict[str, Any]:
    physio = report_status(Q3S3_PHYSIO_LIGHT_SMOKE)
    matrix = report_status(SOURCE_MATRIX)
    data_coverage = report_status(DATA_COVERAGE)
    schemas = raw_schema_summary()

    physio_decision = physio["decision"]
    matrix_decision = matrix["decision"]
    data_decision = data_coverage["decision"]

    closed_lane_confirmed = bool(
        physio["parse_ok"]
        and physio_decision.get("selected_targets") == []
        and physio_decision.get("q3_signal_found") is False
        and physio_decision.get("s3_signal_found") is False
        and physio_decision.get("public_058_evidence") is False
        and physio_decision.get("create_submission_artifact") is False
        and physio_decision.get("submit_now") is False
    )
    source_matrix_ready = bool(
        matrix["parse_ok"]
        and matrix_decision.get("q2q3s3_source_hypothesis_matrix_ready") is True
        and matrix_decision.get("local_q2q3s3_artifact_promotable") is False
        and matrix_decision.get("requires_explicit_kaggle_approval") is True
        and matrix_decision.get("create_submission_artifact") is False
        and matrix_decision.get("submit_now") is False
    )
    data_sources_ready = bool(
        data_coverage["parse_ok"]
        and data_decision.get("data_source_coverage_ready") is True
        and data_decision.get("external_direct_concat_allowed") is False
        and schemas["all_required_files_present"]
        and schemas["all_required_schemas_readable"]
    )
    design_ready = bool(closed_lane_confirmed and source_matrix_ready and data_sources_ready)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_semantic_routine_v2_design",
        "purpose": (
            "Define a materially different no-submit Q3/S3 source hypothesis after the physio-light-onset lane failed: "
            "semantic novelty, routine disruption, and ordered sleep-onset state transitions."
        ),
        "inputs": {
            "q3s3_physio_light_onset_smoke": physio,
            "q2q3s3_source_hypothesis_matrix": matrix,
            "data_source_coverage": data_coverage,
        },
        "raw_schema_summary": schemas,
        "feature_blueprint": build_feature_blueprint(),
        "checks": {
            "closed_physio_light_onset_lane_confirmed": closed_lane_confirmed,
            "source_matrix_ready": source_matrix_ready,
            "data_sources_ready": data_sources_ready,
            "materially_different_from_physio_light_onset": True,
            "external_direct_concat_blocked": data_decision.get("external_direct_concat_allowed") is False,
            "no_artifact_actions": True,
        },
        "decision": {
            "design_ready": design_ready,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "q3_signal_found": False,
            "s3_signal_found": False,
            "public_058_evidence": False,
            "materially_different_from_physio_light_onset": True,
            "external_direct_concat_allowed": False,
            "next_local_allowed_step": "metadata_cardinality_and_state_transition_smoke_only",
            "requires_explicit_kaggle_approval_for_heavy_execution": True,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Do not create ID342. If continuing locally, run only a small metadata/cardinality smoke for the semantic "
                "routine/state-machine features. Move to heavy feature extraction only with explicit Kaggle approval or "
                "after a separate no-artifact local resource check."
            )
            if design_ready
            else "Fix missing reports or raw schema coverage before using this Q3/S3 semantic-routine v2 design.",
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
    decision = report["decision"]
    lines = [
        "# Post-ID341 Q3/S3 semantic-routine v2 design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        report["purpose"],
        "",
        "This is report-only. It defines the next Q3/S3 hypothesis without creating predictions, CSVs, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Design ready: `{decision['design_ready']}`",
        f"- Target focus: `{decision['target_focus']}`",
        f"- Selected targets: `{decision['selected_targets']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Materially different from physio-light-onset: `{decision['materially_different_from_physio_light_onset']}`",
        f"- External direct concat allowed: `{decision['external_direct_concat_allowed']}`",
        f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Raw schema coverage", ""])
    schema_summary = report["raw_schema_summary"]
    lines.append(f"- All required files present: `{schema_summary['all_required_files_present']}`")
    lines.append(f"- All required schemas readable: `{schema_summary['all_required_schemas_readable']}`")
    lines.append(f"- Total raw rows across required files: `{schema_summary['total_rows']}`")
    lines.extend(["", "## Feature blueprint", ""])
    for row in report["feature_blueprint"]:
        lines.extend(
            [
                f"### {row['target']} — `{row['family']}`",
                "",
                f"- Sources: `{row['sources']}`",
                f"- Why distinct: {row['why_distinct_from_closed_lane']}",
                f"- Fail-fast rule: {row['fail_fast_rule']}",
                "- Feature shapes:",
            ]
        )
        for shape in row["feature_shapes"]:
            lines.append(f"  - {shape}")
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
