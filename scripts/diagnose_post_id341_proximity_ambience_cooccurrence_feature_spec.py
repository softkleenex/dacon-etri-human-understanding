#!/usr/bin/env python3
"""Report-only feature spec for proximity/ambience co-occurrence work.

The metadata smoke proved that WiFi/BLE proximity and ambience social/place
co-occurrence is feasible for a Q3/S3 report-only lane. This script freezes a
bounded, privacy-safe feature schema and row contract only. It creates no
feature matrix, trains no model, creates no predictions or submission artifacts,
edits no submission log, runs no Kaggle command, and submits nothing to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_design import (  # noqa: E402
    SOURCE_FAMILY,
    TARGET_FOCUS,
    decision,
    load_json_report,
    rel,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_spec_20260619.json"
DESIGN_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_design_20260619.json"
METADATA_REPORT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_metadata_smoke_20260619.json"

KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]
WINDOWS = {
    "all_day": {"hours": list(range(24)), "description": "full lifelog day"},
    "morning": {"hours": list(range(6, 12)), "description": "06:00-11:59 recovery/start context"},
    "afternoon": {"hours": list(range(12, 18)), "description": "12:00-17:59 daytime social/place context"},
    "evening": {"hours": list(range(18, 24)), "description": "18:00-23:59 late-day stimulation"},
    "late": {"hours": [22, 23, 0, 1, 2, 3], "description": "22:00-03:59 late reactivation band"},
}
FEATURE_GROUPS = {
    "wifi_proximity_richness": ["wifi_bins", "wifi_devices_sum", "wifi_strong_sum"],
    "ble_proximity_richness": ["ble_bins", "ble_devices_sum", "ble_strong_sum"],
    "ambience_social_place": ["speech_max", "speech_mean", "crowd_max", "place_max", "vehicle_music_max"],
    "cross_sensor_cooccurrence": [
        "proximity_social_bins",
        "proximity_place_bins",
        "screen_proximity_bins",
        "social_screen_proximity_bins",
    ],
}
GLOBAL_FEATURES = [
    "wifi_day_covered_flag",
    "ble_day_covered_flag",
    "ambience_day_covered_flag",
    "proximity_social_bin_ratio",
    "late_to_evening_social_cooccur_ratio",
]


def feature_names() -> list[str]:
    names: list[str] = []
    for window in WINDOWS:
        for group, metrics in FEATURE_GROUPS.items():
            for metric in metrics:
                names.append(f"proxamb_{window}_{group}_{metric}")
    names.extend(f"proxamb_global_{name}" for name in GLOBAL_FEATURES)
    return names


def load_required_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    design = load_json_report(DESIGN_REPORT.name)
    metadata = load_json_report(METADATA_REPORT.name)
    return design, metadata


def build_report() -> dict[str, Any]:
    design, metadata = load_required_reports()
    design_decision = decision(design)
    metadata_decision = decision(metadata)
    names = feature_names()
    feature_spec_ready = bool(
        design.get("exists")
        and design.get("parse_ok")
        and metadata.get("exists")
        and metadata.get("parse_ok")
        and design_decision.get("design_ready") is True
        and metadata_decision.get("metadata_smoke_passed") is True
        and metadata_decision.get("source_family") == SOURCE_FAMILY
        and metadata_decision.get("target_focus") == TARGET_FOCUS
        and metadata_decision.get("public_058_evidence") is False
        and metadata_decision.get("score_route_opened") is False
        and metadata_decision.get("create_submission_artifact") is False
        and metadata_decision.get("submit_now") is False
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_feature_spec",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "input_reports": {
            "design": {"path": rel(DESIGN_REPORT), "exists": design.get("exists"), "parse_ok": design.get("parse_ok")},
            "metadata_smoke": {
                "path": rel(METADATA_REPORT),
                "exists": metadata.get("exists"),
                "parse_ok": metadata.get("parse_ok"),
            },
        },
        "row_contract": {
            "key_columns": KEY_COLUMNS,
            "expected_train_rows": 450,
            "expected_test_rows": 250,
            "expected_total_rows": 700,
            "label_columns_allowed": False,
            "prediction_columns_allowed": False,
            "raw_proximity_token_columns_allowed": False,
        },
        "windows": WINDOWS,
        "feature_groups": FEATURE_GROUPS,
        "global_features": GLOBAL_FEATURES,
        "feature_names": names,
        "feature_count": len(names),
        "schema_bounds": {
            "max_feature_columns": 80,
            "actual_feature_columns": len(names),
            "all_features_prefixed": all(name.startswith("proxamb_") for name in names),
            "no_raw_identifier_features": all("bssid" not in name and "address" not in name for name in names),
        },
        "leakage_guards": [
            "Build rows only from lifelog_date source timestamps and sample/train key contracts.",
            "Do not persist raw WiFi/BLE identifier values; use aggregate counts/ratios only.",
            "Use fixed ambience groups and fixed RSSI thresholds, not target-selected tokens.",
            "Any later preprocessing/model smoke must be fold-internal; this spec alone is not score evidence.",
        ],
        "decision": {
            "feature_spec_ready": feature_spec_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "feature_count": len(names),
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "report_only_proximity_ambience_cooccurrence_feature_generation_dryrun_no_matrix",
            "next_action": (
                "Feature spec is ready. Next may only be an in-memory dry run that checks 700-row alignment and "
                "finite bounded columns; do not write a feature matrix, train, create ID342, run Kaggle, or submit."
                if feature_spec_ready
                else "Fix design/metadata prerequisites before any next step."
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


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 proximity/ambience co-occurrence feature spec",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Feature spec ready: `{dec['feature_spec_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Feature count: `{dec['feature_count']}`",
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
        "## Row contract",
        "",
    ]
    for key, value in report["row_contract"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Feature groups", ""])
    for group, metrics in report["feature_groups"].items():
        lines.append(f"- `{group}`: `{', '.join(metrics)}`")
    lines.extend(["", "## Windows", ""])
    for name, meta in report["windows"].items():
        lines.append(f"- `{name}`: {meta['description']} / hours `{meta['hours']}`")
    lines.extend(["", "## Schema bounds", ""])
    for key, value in report["schema_bounds"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Leakage guards", ""])
    for guard in report["leakage_guards"]:
        lines.append(f"- {guard}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(md_path)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    if not report["decision"]["feature_spec_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
