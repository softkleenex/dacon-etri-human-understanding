#!/usr/bin/env python3
"""Design a Q2/Q3/S3-first env/social/routine v2 source lane.

This diagnostic is intentionally report-only. It inspects raw modality metadata
and existing post-ID341 gate reports, then freezes a source-design contract for
the only local fallback route still allowed without Kaggle execution approval:
a materially different Q2/Q3/S3-first environment/social/routine feature lane.

It does not train a model, create predictions, write submission artifacts,
modify the submission log, push/run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q2q3s3_env_social_routine_v2_design_20260618.json"

TARGETS = ["Q2", "Q3", "S3"]

REQUIRED_REPORTS = {
    "q2q3s3_route": "post_id341_q2q3s3_source_route_20260618.json",
    "missing_target_inventory": "post_id341_missing_target_signal_inventory_20260618.json",
    "target_gap_budget": "post_id341_target_gap_budget_20260618.json",
    "public_local_gap": "post_id341_public_local_gap_diagnosis_20260618.json",
    "local_source_closure": "post_id341_local_source_closure_synthesis_20260618.json",
    "usage_temporal": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "place_social": "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "data_source_coverage": "post_id341_data_source_coverage_audit_20260618.json",
}

SOURCE_FILES = {
    "mWifi": "ch2025_mWifi.parquet",
    "mBle": "ch2025_mBle.parquet",
    "mGps": "ch2025_mGps.parquet",
    "mUsageStats": "ch2025_mUsageStats.parquet",
    "mAmbience": "ch2025_mAmbience.parquet",
    "mActivity": "ch2025_mActivity.parquet",
    "mScreenStatus": "ch2025_mScreenStatus.parquet",
}

CARDINALITY_COLUMNS = {
    "mWifi": ["m_wifi"],
    "mBle": ["m_ble"],
    "mUsageStats": ["m_usage_stats"],
    "mAmbience": ["m_ambience"],
    "mActivity": ["m_activity"],
    "mScreenStatus": ["m_screen_use"],
}

NESTED_LIST_FIELDS = {
    "m_wifi": ["bssid", "rssi"],
    "m_ble": ["address", "device_class", "rssi"],
    "m_usage_stats": ["app_name", "total_time"],
}

NESTED_LABEL_SCORE_COLUMNS = {
    "m_ambience": {"label_index": 0, "score_index": 1},
}


def load_json(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def timestamp_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["subject_id", "timestamp"])
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    return {
        "subject_count": int(frame["subject_id"].nunique()),
        "timestamp_min": str(timestamp.min()),
        "timestamp_max": str(timestamp.max()),
        "date_count": int(timestamp.dt.date.nunique()),
        "null_timestamp_rows": int(timestamp.isna().sum()),
    }


def column_cardinality(path: Path, columns: list[str]) -> dict[str, Any]:
    if not columns:
        return {}
    frame = pd.read_parquet(path, columns=[col for col in columns if col])
    out: dict[str, Any] = {}
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        if col in NESTED_LIST_FIELDS:
            lengths = series.map(lambda value: len(value) if value is not None else 0)
            nested_stats: dict[str, Any] = {
                "nested_list_struct": True,
                "row_count": int(len(series)),
                "empty_rows": int((lengths == 0).sum()),
                "mean_items_per_row": float(lengths.mean()),
                "max_items_per_row": int(lengths.max()),
            }
            for field in NESTED_LIST_FIELDS[col]:
                values: list[Any] = []
                for value in series:
                    if value is None:
                        continue
                    values.extend(
                        item.get(field) for item in value if isinstance(item, dict) and item.get(field) is not None
                    )
                value_series = pd.Series(values, dtype="object")
                top_values = value_series.value_counts(dropna=False).head(5)
                nested_stats[field] = {
                    "unique": int(value_series.nunique(dropna=True)),
                    "observed_items": int(len(value_series)),
                    "top_values": {str(k): int(v) for k, v in top_values.to_dict().items()},
                }
            out[col] = nested_stats
            continue
        if col in NESTED_LABEL_SCORE_COLUMNS:
            lengths = series.map(lambda value: len(value) if value is not None else 0)
            labels: list[Any] = []
            scores: list[float] = []
            label_index = NESTED_LABEL_SCORE_COLUMNS[col]["label_index"]
            score_index = NESTED_LABEL_SCORE_COLUMNS[col]["score_index"]
            for value in series:
                if value is None:
                    continue
                for item in value:
                    if len(item) <= max(label_index, score_index):
                        continue
                    labels.append(item[label_index])
                    try:
                        scores.append(float(item[score_index]))
                    except (TypeError, ValueError):
                        pass
            label_series = pd.Series(labels, dtype="object")
            top_labels = label_series.value_counts(dropna=False).head(5)
            out[col] = {
                "nested_label_score_list": True,
                "row_count": int(len(series)),
                "empty_rows": int((lengths == 0).sum()),
                "mean_items_per_row": float(lengths.mean()),
                "max_items_per_row": int(lengths.max()),
                "label_unique": int(label_series.nunique(dropna=True)),
                "label_observed_items": int(len(label_series)),
                "top_labels": {str(k): int(v) for k, v in top_labels.to_dict().items()},
                "score_mean": float(pd.Series(scores).mean()) if scores else None,
            }
            continue
        top_values = series.value_counts(dropna=False).head(5)
        out[col] = {
            "unique": int(series.nunique(dropna=True)),
            "null_rows": int(series.isna().sum()),
            "top_values": {str(k): int(v) for k, v in top_values.to_dict().items()},
        }
    return out


def raw_source_profile() -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for name, filename in SOURCE_FILES.items():
        path = RAW_DIR / filename
        pf = pq.ParquetFile(path)
        profiles[name] = {
            "path": rel(path),
            "rows": int(pf.metadata.num_rows),
            "columns": list(pf.schema.names),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            "timestamp_summary": timestamp_summary(path),
            "cardinality": column_cardinality(path, CARDINALITY_COLUMNS.get(name, [])),
        }
    return profiles


def evidence_read(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    route = reports["q2q3s3_route"]["decision"]
    missing = reports["missing_target_inventory"]["decision"]
    gap = reports["target_gap_budget"]["decision"]
    public_gap = reports["public_local_gap"]["decision"]
    closure = reports["local_source_closure"]["decision"]
    usage = reports["usage_temporal"]["decision"]
    place = reports["place_social"]["decision"]
    sensor = reports["sensor_adherence"]["decision"]
    coverage = reports["data_source_coverage"]["decision"]
    return {
        "route_primary": route.get("primary_score_relevant_route"),
        "route_fallback": route.get("fallback_no_approval_route"),
        "missing_targets": missing.get("focus_missing_locally"),
        "q1_s4_only_is_insufficient": gap.get("q1_s4_only_is_insufficient"),
        "local_cv_needed_for_public_058_recent_floor": public_gap.get("local_cv_needed_for_public_058_recent_floor"),
        "local_cv_needed_for_public_058_recent_stress": public_gap.get("local_cv_needed_for_public_058_recent_stress"),
        "local_source_stack_closed": closure.get("local_source_stack_closed_for_submission_artifact"),
        "usage_temporal_selected_targets": usage.get("selected_targets", []),
        "place_social_selected_targets": place.get("selected_targets", []),
        "sensor_adherence_selected_targets": sensor.get("selected_targets", []),
        "data_source_coverage_ready": coverage.get("data_source_coverage_ready"),
    }


def feature_blocks() -> list[dict[str, Any]]:
    return [
        {
            "block": "routine_stability_clock_v2",
            "target_focus": ["Q2", "Q3", "S3"],
            "raw_inputs": ["mScreenStatus", "mActivity", "mUsageStats"],
            "feature_families": [
                "hourly screen-on/off rhythm distance from subject training-fold routine",
                "activity-state transition entropy by morning/day/evening/night bins",
                "weekday-adjusted app-use total-time rhythm without direct app-name model columns",
                "rolling 3-day and 7-day routine deviation fitted inside each outer fold",
            ],
            "why_q2q3s3": (
                "Q2/Q3/S3 need behavioral regularity and mental-state context rather than another sleep-boundary "
                "overlay; these features model routine stability without borrowing label-derived OOF movement."
            ),
        },
        {
            "block": "place_social_novelty_v2",
            "target_focus": ["Q2", "Q3", "S3"],
            "raw_inputs": ["mWifi", "mBle", "mGps"],
            "feature_families": [
                "Wi-Fi/BLE novelty counts using fold-trained top-k or hashing buckets only",
                "home-like anchor consistency inferred from within-subject frequent place buckets",
                "GPS movement radius, speed quantiles, and stationary ratio by daypart",
                "place-transition entropy and next-day smoothness features available without labels",
            ],
            "why_q2q3s3": (
                "Previous place/social standalone smoke selected no target, so v2 must combine novelty, regularity, "
                "and daypart transitions instead of direct BSSID/address memorization."
            ),
        },
        {
            "block": "ambient_load_recovery_context_v2",
            "target_focus": ["Q2", "Q3"],
            "raw_inputs": ["mAmbience", "mUsageStats", "mActivity"],
            "feature_families": [
                "speech/music/silence ratios by daypart from ambience element",
                "late-day app-use load and app category entropy using fold-local category maps",
                "ambient-context change points before sleep-date boundary",
                "interaction-free single-family summaries first; no broad stacker until split evidence exists",
            ],
            "why_q2q3s3": (
                "Q2/Q3 are the missing questionnaire targets most plausibly tied to social/context load. "
                "This block is deliberately Q2/Q3-first and should not be evaluated as a generic all-target tuner."
            ),
        },
    ]


def validation_contract() -> dict[str, Any]:
    return {
        "outer_splits_required": [
            "subject_hole_5",
            "actual_test_band_holdout",
            "test_boundary_neighbor2_holdout",
            "test_boundary_neighbor5_holdout",
            "loso_10",
        ],
        "fit_inside_fold_only": [
            "top-k Wi-Fi/BLE/app categories",
            "hash bucket dictionaries when used",
            "subject routine baselines",
            "scalers/imputers",
            "feature selection and target selection",
        ],
        "forbidden_shortcuts": [
            "direct BSSID/address/app-name one-hot columns fitted on full train+test",
            "full-OOF target selection",
            "public-LB-guided thresholding",
            "external row concatenation",
            "creating ID342 from a local-only feature design",
        ],
        "promotion_gate": {
            "selected_targets_must_include_any_of": TARGETS,
            "all_primary_split_wins_required_for_promoted_target": 5,
            "projected_proxy_must_beat_id178_public": 0.6014113009,
            "public_058_review_cutoff": 0.590,
            "artifact_review_is_separate": True,
        },
    }


def build_report() -> dict[str, Any]:
    reports = {key: load_json(name) for key, name in REQUIRED_REPORTS.items()}
    evidence = evidence_read(reports)
    profiles = raw_source_profile()
    design_ready = (
        evidence["route_fallback"] == "local_or_kaggle_q2q3s3_env_social_routine_v2_design"
        and evidence["missing_targets"] == TARGETS
        and evidence["q1_s4_only_is_insufficient"] is True
        and evidence["local_source_stack_closed"] is True
        and evidence["data_source_coverage_ready"] is True
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_env_social_routine_v2_design",
        "purpose": "Freeze a Q2/Q3/S3-first env/social/routine source design before any new local smoke or Kaggle run.",
        "inputs": {key: rel(REPORTS_DIR / name) for key, name in REQUIRED_REPORTS.items()},
        "evidence": evidence,
        "raw_source_profiles": profiles,
        "feature_blocks": feature_blocks(),
        "validation_contract": validation_contract(),
        "decision": {
            "design_ready": design_ready,
            "target_focus": TARGETS,
            "next_local_allowed_step": "feature_generation_metadata_smoke_only",
            "next_kaggle_allowed_step": "only_after_explicit_no_submit_approval",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "train_model_now": False,
            "next_action": (
                "Implement only a Q2/Q3/S3 env-social-routine feature-generation smoke next, or run the approved "
                "clean upstream rebuild if the user gives the exact Kaggle no-submit approval phrase. Do not create "
                "a submission artifact from this design."
            ),
        },
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    evidence = report["evidence"]
    lines = [
        "# Post-ID341 Q2/Q3/S3 env-social-routine v2 design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model and creates no predictions, submission CSV, notebook/history artifact, Kaggle run, or DACON submission.",
        "",
        "## Evidence read",
        "",
    ]
    for key, value in evidence.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Raw source profiles",
            "",
            "| source | rows | subjects | dates | size MB | cardinality highlights |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for source, profile in report["raw_source_profiles"].items():
        card = []
        for col, stats in profile["cardinality"].items():
            if "unique" in stats:
                card.append(f"{col}={stats['unique']}")
            elif stats.get("nested_list_struct"):
                child_summaries = [
                    f"{child}={child_stats['unique']}"
                    for child, child_stats in stats.items()
                    if isinstance(child_stats, dict) and "unique" in child_stats
                ]
                card.append(f"{col}({', '.join(child_summaries)})")
            elif stats.get("nested_label_score_list"):
                card.append(f"{col}(labels={stats['label_unique']})")
            else:
                card.append(f"{col}=nested")
        lines.append(
            f"| `{source}` | {profile['rows']} | {profile['timestamp_summary']['subject_count']} | "
            f"{profile['timestamp_summary']['date_count']} | {profile['size_mb']} | `{', '.join(card)}` |"
        )
    lines.extend(["", "## Feature blocks", ""])
    for block in report["feature_blocks"]:
        lines.extend(
            [
                f"### `{block['block']}`",
                "",
                f"- Target focus: `{', '.join(block['target_focus'])}`",
                f"- Raw inputs: `{', '.join(block['raw_inputs'])}`",
                f"- Why Q2/Q3/S3: {block['why_q2q3s3']}",
                "",
                "Feature families:",
            ]
        )
        lines.extend(f"- {item}" for item in block["feature_families"])
        lines.append("")
    contract = report["validation_contract"]
    lines.extend(
        [
            "## Validation contract",
            "",
            f"- Outer splits required: `{', '.join(contract['outer_splits_required'])}`",
            "- Fit inside fold only:",
        ]
    )
    lines.extend(f"  - {item}" for item in contract["fit_inside_fold_only"])
    lines.append("- Forbidden shortcuts:")
    lines.extend(f"  - {item}" for item in contract["forbidden_shortcuts"])
    gate = contract["promotion_gate"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Design ready: `{decision['design_ready']}`",
            f"- Target focus: `{', '.join(decision['target_focus'])}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Next Kaggle allowed step: `{decision['next_kaggle_allowed_step']}`",
            f"- Promotion target set must include: `{', '.join(gate['selected_targets_must_include_any_of'])}`",
            f"- Projected proxy must beat ID178 public: `{gate['projected_proxy_must_beat_id178_public']}`",
            f"- Public-0.58 review cutoff: `{gate['public_058_review_cutoff']}`",
            f"- Train model now: `{decision['train_model_now']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
