#!/usr/bin/env python3
"""Design the circadian HR/light phase-shape feature spec after metadata feasibility.

This script is report-only. It consumes the material-new-source queue and the
circadian HR/light phase metadata probe, then writes a feature design
specification. It does not generate a feature matrix, train a model, create
predictions, create a submission CSV, modify the submission log, run Kaggle, or
submit to DACON.
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
METADATA_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_metadata_probe_20260619.json"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "circadian_hr_light_phase_shape_v1"
TARGET_FOCUS = ["Q3", "S3", "S2"]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": rel(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["exists"] = True
    data["path"] = rel(path)
    return data


def feature_blocks() -> list[dict[str, Any]]:
    return [
        {
            "block": "phase_source_quality",
            "purpose": "Keep downstream models aware of whether a day has enough circadian signal to trust.",
            "raw_inputs": ["wHr", "wLight", "mLight", "wPedo"],
            "features": [
                "phase_strong_day_flag",
                "phase_weak_day_flag",
                "whr_rows",
                "whr_hours",
                "wlight_hours",
                "mlight_hours",
                "wpedo_hours",
                "phase_missing_source_count",
            ],
            "fit_scope": "row_local_no_fit",
            "target_focus": TARGET_FOCUS,
        },
        {
            "block": "wearable_hr_phase_shape",
            "purpose": "Represent autonomic recovery and intraday HR curve shape without using labels.",
            "raw_inputs": ["wHr"],
            "features": [
                "hr_night_median",
                "hr_morning_median",
                "hr_afternoon_median",
                "hr_evening_median",
                "hr_day_night_amplitude",
                "hr_morning_rise_slope",
                "hr_evening_decay_slope",
                "hr_curve_roughness",
                "hr_rest_window_low_quantile",
                "hr_high_load_minutes_proxy",
            ],
            "fit_scope": "predeclared_hour_bins_only",
            "target_focus": ["Q3", "S3"],
        },
        {
            "block": "wearable_light_phase_shape",
            "purpose": "Capture light timing, amplitude, and delayed-evening exposure as a circadian zeitgeber.",
            "raw_inputs": ["wLight"],
            "features": [
                "wlight_first_nonzero_hour",
                "wlight_last_nonzero_hour",
                "wlight_morning_exposure_sum",
                "wlight_afternoon_exposure_sum",
                "wlight_evening_exposure_sum",
                "wlight_night_exposure_sum",
                "wlight_day_night_contrast",
                "wlight_evening_delay_ratio",
            ],
            "fit_scope": "row_local_no_fit",
            "target_focus": ["Q3", "S3", "S2"],
        },
        {
            "block": "phone_light_dark_continuity",
            "purpose": "Use phone light as a fallback/context signal for darkness continuity and late exposure.",
            "raw_inputs": ["mLight"],
            "features": [
                "mlight_dark_minutes_proxy",
                "mlight_longest_dark_gap_hours",
                "mlight_night_light_burst_count",
                "mlight_evening_light_share",
                "mlight_morning_light_share",
                "mlight_wearable_light_disagreement",
            ],
            "fit_scope": "row_local_no_fit",
            "target_focus": ["Q3", "S3", "S2"],
        },
        {
            "block": "rest_activity_phase_shape",
            "purpose": "Separate circadian phase shifts from ordinary movement and rest/activity fragmentation.",
            "raw_inputs": ["wPedo"],
            "features": [
                "steps_morning_sum",
                "steps_afternoon_sum",
                "steps_evening_sum",
                "steps_night_sum",
                "first_active_hour",
                "last_active_hour",
                "activity_fragmentation_count",
                "late_activity_ratio",
                "rest_window_length_proxy",
                "activity_phase_centroid_hour",
            ],
            "fit_scope": "predeclared_activity_thresholds_only",
            "target_focus": ["Q3", "S3"],
        },
        {
            "block": "cross_modal_phase_alignment",
            "purpose": "Measure whether HR, light, and activity are phase-aligned or internally contradictory.",
            "raw_inputs": ["wHr", "wLight", "mLight", "wPedo"],
            "features": [
                "hr_light_phase_lag_proxy",
                "activity_light_phase_lag_proxy",
                "night_light_with_high_hr_minutes_proxy",
                "evening_activity_with_bright_light_proxy",
                "morning_light_without_activity_proxy",
                "strong_phase_cross_modal_consistency_score",
            ],
            "fit_scope": "row_local_no_fit",
            "target_focus": ["Q3", "S3", "S2"],
        },
    ]


def leakage_guards() -> list[str]:
    return [
        "No target labels, OOF predictions, submission CSVs, public scores, or prior candidate predictions may enter feature generation.",
        "All phase windows must be predeclared hour bins; do not tune time windows on validation loss.",
        "Subject-relative baselines, if added later, must be fit inside each training fold only.",
        "Final test feature generation may use row-local raw sensor values, but must not fit subject baselines from unlabeled test rows without a separate report.",
        "Missingness and coverage flags are allowed only as raw-source reliability features, not as public-score-calibrated switches.",
        "The first execution after this spec may only be a report-only dry run checking row alignment, feature counts, and train/test coverage.",
        "No feature parquet, fold model, base prediction artifact, notebook/history artifact, submission-log row, ID342, Kaggle run, or DACON submission is allowed by this spec.",
    ]


def resource_bounds() -> dict[str, Any]:
    blocks = feature_blocks()
    return {
        "max_daily_feature_columns_for_first_dryrun": 56,
        "planned_first_pass_feature_columns": sum(len(block["features"]) for block in blocks),
        "max_hour_bins": 24,
        "max_phase_summary_bins": 4,
        "max_rows_expected": 700,
        "allowed_first_execution": "report_only_feature_generation_dryrun_no_matrix",
        "disallowed_first_execution": [
            "feature_parquet_write",
            "fold_model_training",
            "test_prediction_generation",
            "submission_csv_creation",
            "kaggle_remote_run_without_exact_approval",
            "dacon_submission",
        ],
    }


def validation_lenses() -> list[dict[str, Any]]:
    return [
        {
            "lens": "row_alignment_dryrun",
            "required_before_matrix": True,
            "pass_rule": "The dry run must align exactly to 450 train + 250 test subject/lifelog rows with no duplicate keys.",
        },
        {
            "lens": "coverage_balance_gate",
            "required_before_matrix": True,
            "pass_rule": "Strong phase-shape rates must remain >=0.50 in train and test; weak rates must remain >=0.70.",
        },
        {
            "lens": "feature_count_gate",
            "required_before_matrix": True,
            "pass_rule": "The first-pass schema must stay within 56 feature columns plus keys and must contain only bounded numeric/boolean features.",
        },
        {
            "lens": "subject_hole_fit_scope",
            "required_before_model": True,
            "pass_rule": "Any later subject-relative phase baseline must be fitted inside each outer train fold only.",
        },
        {
            "lens": "target_specific_promotion_gate",
            "required_before_artifact": True,
            "pass_rule": "A future fold-internal model must beat the ID341/ID178 anchor on Q3/S3/S2 slices without worsening public-transfer risk.",
        },
        {
            "lens": "public_transfer_gate_v2",
            "required_before_artifact": True,
            "pass_rule": "No candidate can be promoted unless the conservative public proxy is reviewable below the current ID178/ID329 plateau.",
        },
    ]


def build_report() -> dict[str, Any]:
    queue = read_json(QUEUE_REPORT)
    metadata = read_json(METADATA_REPORT)
    queue_dec = queue.get("decision", {}) if isinstance(queue.get("decision"), dict) else {}
    queue_items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    queue_item = next((item for item in queue_items if item.get("name") == SOURCE_FAMILY), {})
    metadata_dec = metadata.get("decision", {}) if isinstance(metadata.get("decision"), dict) else {}
    metadata_passed = bool(metadata_dec.get("metadata_probe_passed", False))
    allowed_queue_steps = {
        "report_only_feature_spec_no_model",
        "report_only_feature_generation_dryrun_no_matrix",
        "report_only_feature_matrix_smoke_contract",
        "write_circadian_feature_matrix_under_contract_no_model_no_submission",
        "report_only_model_smoke_contract_no_model_no_submission",
        "run_circadian_feature_matrix_model_smoke_under_contract_no_artifact",
        "closed_no_artifact",
    }
    queue_expected = bool(
        queue.get("exists")
        and queue_item.get("name") == SOURCE_FAMILY
        and queue_item.get("next_allowed_step") in allowed_queue_steps
    )
    feature_spec_ready = bool(metadata.get("exists") and metadata_passed and queue_expected)
    next_action = (
        "The circadian HR/light phase-shape spec is ready. The next no-approval step may only be a "
        "report-only feature-generation dry run that checks row alignment, coverage, and schema bounds; no "
        "feature parquet, model, ID342, Kaggle run, or DACON submission is allowed."
        if feature_spec_ready
        else "Circadian HR/light phase-shape spec is not ready; keep this source family as metadata-only context."
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_circadian_hr_light_phase_feature_spec",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "inputs": {
            "queue_report": {
                "path": rel(QUEUE_REPORT),
                "exists": queue.get("exists", False),
                "next_no_approval_step": queue_dec.get("next_no_approval_step"),
                "circadian_queue_status": queue_item.get("status"),
                "circadian_next_allowed_step": queue_item.get("next_allowed_step"),
            },
            "metadata_report": {
                "path": rel(METADATA_REPORT),
                "exists": metadata.get("exists", False),
                "metadata_probe_passed": metadata_passed,
                "strong_phase_shape_train_rate": metadata_dec.get("strong_phase_shape_train_rate"),
                "strong_phase_shape_test_rate": metadata_dec.get("strong_phase_shape_test_rate"),
                "weak_phase_shape_train_rate": metadata_dec.get("weak_phase_shape_train_rate"),
                "weak_phase_shape_test_rate": metadata_dec.get("weak_phase_shape_test_rate"),
            },
        },
        "feature_blocks": feature_blocks(),
        "leakage_guards": leakage_guards(),
        "resource_bounds": resource_bounds(),
        "validation_lenses": validation_lenses(),
        "decision": {
            "feature_spec_ready": feature_spec_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "metadata_probe_passed": metadata_passed,
            "next_local_allowed_step": "report_only_feature_generation_dryrun_no_matrix",
            "feature_matrix_allowed_now": False,
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
    meta = report["inputs"]["metadata_report"]
    lines = [
        "# Circadian HR/light phase-shape feature spec — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction "
            "artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, "
            "or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Feature spec ready: `{dec['feature_spec_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{', '.join(dec['target_focus'])}`",
        f"- Metadata probe passed: `{dec['metadata_probe_passed']}`",
        f"- Strong phase-shape train/test rates: `{meta['strong_phase_shape_train_rate']}` / "
        f"`{meta['strong_phase_shape_test_rate']}`",
        f"- Weak phase-shape train/test rates: `{meta['weak_phase_shape_train_rate']}` / "
        f"`{meta['weak_phase_shape_test_rate']}`",
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
    lines.extend(["", "## Resource bounds", "", "```json"])
    lines.append(json.dumps(report["resource_bounds"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.extend(["", "## Validation lenses", ""])
    lines.append("| lens | required before | pass rule |")
    lines.append("|---|---|---|")
    for lens in report["validation_lenses"]:
        if lens.get("required_before_matrix"):
            required = "matrix"
        elif lens.get("required_before_model"):
            required = "model"
        else:
            required = "artifact"
        lines.append(f"| `{lens['lens']}` | `{required}` | {lens['pass_rule']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This spec turns a feasible raw-source family into a controlled dry-run plan. It is not "
                "leaderboard evidence, does not allow a feature parquet yet, and does not change the fact that "
                "the only current score-relevant open route is the explicit-approved clean upstream ID221→ID226 "
                "rebuild."
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
        default=REPORTS_DIR / "post_id341_circadian_hr_light_phase_feature_spec_20260619.json",
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
