#!/usr/bin/env python3
"""Resource gate for post-ID341 Q3/S3 semantic-routine fold-internal smoke.

This report-only gate decides whether the semantic-routine v2 lane may proceed
from bounded metadata smoke to a local no-artifact fold-internal model smoke. It
does not build features, train models, create predictions, create submission
artifacts, push/run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_semantic_routine_resource_gate_20260618.json"
METADATA_SMOKE = REPORTS_DIR / "post_id341_q3s3_semantic_routine_metadata_smoke_20260618.json"

LOCAL_RAW_BYTES_SOFT_LIMIT = 256 * 1024 * 1024
MAX_ENGINEERED_FEATURE_COLUMNS = 64
MAX_MODEL_SMOKE_TARGETS = 2
MAX_MODEL_SPECS = 2
MAX_BLEND_WEIGHTS = 4
EXPECTED_DAILY_ROWS = 700


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


def raw_file_sizes(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source, summary in metadata.get("source_summaries", {}).items():
        path = ROOT / summary.get("path", "")
        out[source] = {
            "path": rel(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "total_rows": summary.get("total_rows"),
            "sample_rows": summary.get("sample_rows"),
        }
    return out


def build_report() -> dict[str, Any]:
    smoke_status = status_for(METADATA_SMOKE)
    metadata = load_json(METADATA_SMOKE) if smoke_status["parse_ok"] else {}
    decision = metadata.get("decision", {})
    checks = metadata.get("checks", {})
    file_sizes = raw_file_sizes(metadata)
    total_raw_bytes = sum(row["bytes"] for row in file_sizes.values())
    largest_source = max(file_sizes.items(), key=lambda item: item[1]["bytes"])[0] if file_sizes else None
    metadata_ready = bool(
        smoke_status["parse_ok"]
        and decision.get("metadata_smoke_passed") is True
        and decision.get("next_local_allowed_step") == "resource_gated_fold_internal_semantic_routine_model_smoke_only"
        and checks.get("bounded_sample_only") is True
        and decision.get("create_submission_artifact") is False
        and decision.get("submit_now") is False
    )
    raw_size_ok = total_raw_bytes <= LOCAL_RAW_BYTES_SOFT_LIMIT
    plan_constraints = {
        "stream_raw_sources_one_at_a_time": True,
        "build_daily_700_row_feature_frame_only": True,
        "do_not_export_raw_identifiers_or_coordinates": True,
        "do_not_one_hot_raw_app_wifi_ble_ids": True,
        "max_engineered_feature_columns": MAX_ENGINEERED_FEATURE_COLUMNS,
        "max_model_smoke_targets": MAX_MODEL_SMOKE_TARGETS,
        "max_model_specs": MAX_MODEL_SPECS,
        "max_blend_weights": MAX_BLEND_WEIGHTS,
        "expected_daily_rows": EXPECTED_DAILY_ROWS,
    }
    local_no_artifact_model_smoke_allowed = bool(metadata_ready and raw_size_ok and all(plan_constraints.values()))

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_semantic_routine_resource_gate",
        "purpose": (
            "Decide whether the semantic-routine v2 lane may run a small local fold-internal model smoke without "
            "creating artifacts or overloading local memory."
        ),
        "inputs": {
            "metadata_smoke": smoke_status,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mac_m4_environment_primary_explanation": False,
        },
        "raw_file_sizes": file_sizes,
        "resource_budget": {
            "total_raw_bytes": total_raw_bytes,
            "total_raw_mebibytes": round(total_raw_bytes / (1024 * 1024), 3),
            "local_raw_bytes_soft_limit": LOCAL_RAW_BYTES_SOFT_LIMIT,
            "local_raw_mebibytes_soft_limit": round(LOCAL_RAW_BYTES_SOFT_LIMIT / (1024 * 1024), 3),
            "largest_source": largest_source,
        },
        "plan_constraints": plan_constraints,
        "checks": {
            "metadata_smoke_ready": metadata_ready,
            "raw_size_within_local_soft_limit": raw_size_ok,
            "streaming_constraints_required": True,
            "feature_width_limited": MAX_ENGINEERED_FEATURE_COLUMNS <= 64,
            "model_grid_limited": MAX_MODEL_SPECS <= 2 and MAX_BLEND_WEIGHTS <= 4,
            "no_artifact_actions": True,
        },
        "decision": {
            "resource_gate_passed": local_no_artifact_model_smoke_allowed,
            "local_no_artifact_model_smoke_allowed": local_no_artifact_model_smoke_allowed,
            "target_focus": ["Q3", "S3"],
            "selected_targets": [],
            "public_058_evidence": False,
            "next_local_allowed_step": (
                "fold_internal_semantic_routine_model_smoke_only"
                if local_no_artifact_model_smoke_allowed
                else "use_kaggle_or_reduce_feature_plan_before_model_smoke"
            ),
            "kaggle_heavy_execution_recommended": not local_no_artifact_model_smoke_allowed,
            "requires_explicit_kaggle_approval_for_remote_execution": True,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Local no-artifact fold-internal model smoke is allowed only under the listed streaming/width/grid "
                "constraints. Do not create ID342 or any submission artifact."
                if local_no_artifact_model_smoke_allowed
                else "Do not run the local model smoke; use explicit-approved Kaggle or reduce the feature plan first."
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
    decision = report["decision"]
    budget = report["resource_budget"]
    lines = [
        "# Post-ID341 Q3/S3 semantic-routine resource gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        report["purpose"],
        "",
        "This is report-only. It creates no features, predictions, CSVs, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Resource gate passed: `{decision['resource_gate_passed']}`",
        f"- Local no-artifact model smoke allowed: `{decision['local_no_artifact_model_smoke_allowed']}`",
        f"- Target focus: `{decision['target_focus']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
        f"- Kaggle heavy execution recommended: `{decision['kaggle_heavy_execution_recommended']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Resource budget",
        "",
        f"- Total raw size: `{budget['total_raw_mebibytes']} MiB`",
        f"- Local soft limit: `{budget['local_raw_mebibytes_soft_limit']} MiB`",
        f"- Largest source: `{budget['largest_source']}`",
        "",
        "## Constraints",
        "",
    ]
    for key, value in report["plan_constraints"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
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
