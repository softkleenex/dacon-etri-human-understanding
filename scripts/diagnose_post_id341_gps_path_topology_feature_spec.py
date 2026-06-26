#!/usr/bin/env python3
"""Design the GPS path-topology feature spec after metadata feasibility.

This script is report-only. It consumes the material-new-source queue and GPS
metadata probe, then writes a feature design specification. It does not generate
feature matrices, train a model, create predictions, create a submission CSV,
modify the submission log, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
METADATA_REPORT = REPORTS_DIR / "post_id341_gps_path_topology_metadata_probe_20260619.json"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "gps_path_topology_sequence_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(ROOT))}
    data = json.loads(path.read_text())
    data["exists"] = True
    data["path"] = str(path.relative_to(ROOT))
    return data


def feature_blocks() -> list[dict[str, Any]]:
    return [
        {
            "block": "gps_payload_quality",
            "purpose": "Expose whether a day has enough usable topology signal without using labels.",
            "raw_inputs": ["mGps"],
            "features": [
                "gps_rows",
                "gps_payload_points",
                "gps_active_minutes",
                "gps_missing_gap_count",
                "gps_longest_missing_gap_minutes",
                "gps_evening_rows_18_23",
                "gps_night_rows_00_05",
            ],
            "fit_scope": "row_local_no_fit",
            "target_focus": ["Q3", "S3"],
        },
        {
            "block": "path_motion_shape",
            "purpose": "Represent movement shape rather than daily aggregate distance alone.",
            "raw_inputs": ["mGps", "mActivity"],
            "features": [
                "trajectory_step_euclidean_sum",
                "trajectory_step_euclidean_p50_p90",
                "speed_p50_p90_max",
                "low_speed_run_count",
                "high_speed_burst_count",
                "late_day_motion_ratio",
                "activity_gps_motion_disagreement_minutes",
            ],
            "fit_scope": "predeclared_numeric_thresholds_only",
            "target_focus": ["Q3", "S3"],
        },
        {
            "block": "dwell_segment_topology",
            "purpose": "Approximate home/away and place-transition motifs from dense GPS points.",
            "raw_inputs": ["mGps", "mWifi"],
            "features": [
                "dwell_segment_count",
                "longest_dwell_minutes",
                "dwell_entropy",
                "dominant_dwell_share",
                "evening_dominant_dwell_share",
                "night_away_from_train_home_proxy_minutes",
                "dwell_transition_count",
            ],
            "fit_scope": "train_fold_subject_baseline_only",
            "target_focus": ["Q3", "S3"],
        },
        {
            "block": "route_novelty_and_regularization",
            "purpose": "Measure whether a day deviates from the subject's normal route/place sequence.",
            "raw_inputs": ["mGps", "mWifi", "mBle"],
            "features": [
                "route_token_novelty_rate",
                "route_token_revisit_rate",
                "wifi_place_token_novelty_rate",
                "ble_social_context_novelty_rate",
                "weekday_adjusted_route_distance",
                "late_day_route_disruption_score",
            ],
            "fit_scope": "train_fold_subject_baseline_only",
            "target_focus": ["Q3", "S3"],
        },
        {
            "block": "place_social_context_alignment",
            "purpose": "Separate physical movement from social/place context density.",
            "raw_inputs": ["mWifi", "mBle", "mActivity"],
            "features": [
                "wifi_scan_count",
                "wifi_unique_token_count",
                "wifi_top_token_share",
                "ble_scan_count",
                "ble_unique_token_count",
                "ble_top_token_share",
                "stationary_but_social_dense_minutes",
            ],
            "fit_scope": "token_hashing_or_train_fold_vocabulary_only",
            "target_focus": ["Q3", "S3"],
        },
    ]


def leakage_guards() -> list[str]:
    return [
        "No target labels, OOF predictions, submission CSVs, public scores, or previous candidate predictions may enter feature generation.",
        "Any subject-level home/place baseline must be fit inside the training fold only for validation.",
        "For final test feature generation, place baselines may use current train raw logs only unless a separate report approves unlabeled-test baselines.",
        "All coordinate bucketing must be predeclared or train-fold-fitted; never tune bucket sizes on validation loss.",
        "BLE/Wi-Fi vocabularies must be hashed or train-fold-fitted with capped cardinality.",
        "Feature generation may write only reports in the next dry run; no feature parquet is allowed until a separate smoke contract opens it.",
        "A fold model is explicitly blocked until the dry-run report proves row alignment, coverage, and cardinality bounds.",
    ]


def resource_bounds() -> dict[str, Any]:
    return {
        "max_gps_rows_expected": 1_000_000,
        "max_daily_feature_columns_for_first_matrix": 64,
        "max_coordinate_bins_per_subject": 32,
        "max_wifi_tokens_per_subject": 64,
        "max_ble_tokens_per_subject": 64,
        "allowed_first_execution": "report_only_feature_generation_dryrun",
        "disallowed_first_execution": [
            "fold_model_training",
            "test_prediction_generation",
            "submission_csv_creation",
            "kaggle_remote_run_without_exact_approval",
        ],
    }


def validation_lenses() -> list[dict[str, Any]]:
    return [
        {
            "lens": "row_alignment_dryrun",
            "required_before_model": True,
            "pass_rule": "700 subject/lifelog rows align exactly with train+sample IDs and dates; no duplicate keys.",
        },
        {
            "lens": "coverage_cardinality_gate",
            "required_before_model": True,
            "pass_rule": "train/test coverage both remain >= 0.90 and feature columns <= 64 in the first dry run.",
        },
        {
            "lens": "subject_hole_5",
            "required_before_artifact": True,
            "pass_rule": "future fold-internal model must beat ID341 on Q3/S3 target slices without moving broad distribution risk upward.",
        },
        {
            "lens": "public_transfer_gate_v2",
            "required_before_artifact": True,
            "pass_rule": "future candidate must pass transfer and joint-risk gates before any artifact-contract review.",
        },
    ]


def build_report() -> dict[str, Any]:
    queue = read_json(QUEUE_REPORT)
    metadata = read_json(METADATA_REPORT)
    queue_dec = queue.get("decision", {}) if isinstance(queue.get("decision"), dict) else {}
    queue_items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    gps_queue_item = next((item for item in queue_items if item.get("name") == SOURCE_FAMILY), {})
    metadata_dec = metadata.get("decision", {}) if isinstance(metadata.get("decision"), dict) else {}
    metadata_passed = bool(metadata_dec.get("metadata_probe_passed", False))
    queue_expected = bool(gps_queue_item)
    spec_ready = bool(queue.get("exists") and metadata.get("exists") and metadata_passed and queue_expected)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_gps_path_topology_feature_spec",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "inputs": {
            "queue_report": {
                "path": str(QUEUE_REPORT.relative_to(ROOT)),
                "exists": queue.get("exists", False),
                "next_no_approval_step": queue_dec.get("next_no_approval_step"),
                "gps_queue_status": gps_queue_item.get("status"),
                "gps_next_allowed_step": gps_queue_item.get("next_allowed_step"),
            },
            "metadata_report": {
                "path": str(METADATA_REPORT.relative_to(ROOT)),
                "exists": metadata.get("exists", False),
                "metadata_probe_passed": metadata_passed,
                "strong_topology_train_rate": metadata_dec.get("strong_topology_train_rate"),
                "strong_topology_test_rate": metadata_dec.get("strong_topology_test_rate"),
            },
        },
        "feature_blocks": feature_blocks(),
        "leakage_guards": leakage_guards(),
        "resource_bounds": resource_bounds(),
        "validation_lenses": validation_lenses(),
        "decision": {
            "feature_spec_ready": spec_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": ["Q3", "S3"],
            "metadata_probe_passed": metadata_passed,
            "next_local_allowed_step": "report_only_feature_generation_dryrun",
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "The GPS topology design spec is ready. The next no-approval step may only be a report-only "
                "feature-generation dry run that checks row alignment, coverage, and cardinality; no model, "
                "feature parquet, ID342, Kaggle run, or DACON submission is allowed."
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    meta = report["inputs"]["metadata_report"]
    lines = [
        "# GPS path-topology feature spec — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, "
            "notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Feature spec ready: `{dec['feature_spec_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Metadata probe passed: `{dec['metadata_probe_passed']}`",
        f"- Strong topology train/test rates: `{meta['strong_topology_train_rate']}` / `{meta['strong_topology_test_rate']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Feature blocks",
        "",
        "| block | raw inputs | fit scope | target focus | feature count |",
        "|---|---|---|---|---:|",
    ]
    for block in report["feature_blocks"]:
        lines.append(
            f"| `{block['block']}` | `{', '.join(block['raw_inputs'])}` | `{block['fit_scope']}` | "
            f"`{', '.join(block['target_focus'])}` | {len(block['features'])} |"
        )
    lines.extend(["", "## Leakage guards", ""])
    lines.extend(f"- {guard}" for guard in report["leakage_guards"])
    lines.extend(["", "## Resource bounds", "", "```json", json.dumps(report["resource_bounds"], indent=2), "```"])
    lines.extend(["", "## Validation lenses", ""])
    lines.append("| lens | required before model/artifact | pass rule |")
    lines.append("|---|---|---|")
    for lens in report["validation_lenses"]:
        required = "model" if lens.get("required_before_model") else "artifact"
        lines.append(f"| `{lens['lens']}` | `{required}` | {lens['pass_rule']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This spec turns a raw-source idea into a controlled dry-run plan. It is not leaderboard evidence, "
                "does not allow a feature parquet yet, and does not change the fact that the only current score-relevant "
                "open route is the explicit-approved clean upstream ID221→ID226 rebuild."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output.with_suffix(".md").write_text(markdown(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_gps_path_topology_feature_spec_20260619.json",
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
                "feature_spec_ready": dec["feature_spec_ready"],
                "source_family": dec["source_family"],
                "next_local_allowed_step": dec["next_local_allowed_step"],
                "feature_matrix_allowed_now": dec["feature_matrix_allowed_now"],
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
