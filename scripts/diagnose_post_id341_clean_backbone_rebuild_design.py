#!/usr/bin/env python3
"""Design the clean ID221/ID226 upstream backbone rebuild.

This report-only diagnostic turns the post-ID341 priority matrix into a concrete
Kaggle CPU rebuild contract. It does not train models; it specifies feature
families, nested validation rules, allowed outputs, and stop gates for a clean
ID221/ID226 backbone before any A1/B1 replay.

It creates no predictions, submission CSV, base prediction artifact, notebook
history artifact, submission-log row, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TRAIN_PATH = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PATH = ROOT / "data" / "processed" / "df_test_raw.parquet"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
DEFAULT_PRIORITY_PATH = REPORTS_DIR / "post_id341_upstream_source_priority_matrix_20260618.json"
KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLUMNS = ["subject_id", "sleep_date", "lifelog_date", "date"]


def feature_group(column: str) -> str:
    if column in ID_COLUMNS:
        return "id_calendar"
    if column in TARGETS:
        return "target"
    if column.startswith(("ch2025_mScreenStatus", "screen_", "slp_screen")):
        return "screen"
    if column.startswith(("ch2025_mActivity", "act_", "slp_act")):
        return "activity"
    if column.startswith(("ch2025_wPedo", "pedo_", "slp_pedo")):
        return "pedo"
    if column.startswith(("ch2025_wHr", "wHr_", "slp_hr")):
        return "heart_rate"
    if column.startswith(("ch2025_mLight", "mlight_", "ch2025_wLight", "wlight_", "slp_wlight")):
        return "light"
    if column.startswith(("ch2025_mGps", "mGps", "gps_")):
        return "gps"
    if column.startswith(("ch2025_mUsageStats", "mUsage", "usage_")):
        return "usage"
    if column.startswith(("ch2025_mWifi", "wifi_")):
        return "wifi"
    if column.startswith(("ch2025_mBle", "ble_")):
        return "ble"
    if column.startswith(("ch2025_mAmbience", "amb_")):
        return "ambience"
    if column.startswith("demo_"):
        return "demographic"
    if column in ["dayofweek", "is_weekend", "is_holiday", "is_rest_day"]:
        return "calendar"
    return "other"


def load_priority(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing priority report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment(candidate_id: int) -> dict[str, Any]:
    paths = sorted(REPORTS_DIR.glob(f"experiment_{candidate_id}_*.json"))
    if not paths:
        return {"id": candidate_id, "exists": False}
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    return {
        "id": candidate_id,
        "exists": True,
        "path": str(paths[0].relative_to(ROOT)),
        "method": data.get("method"),
        "local_cv": data.get("local_cv"),
        "selector_info": data.get("selector_info", {}),
    }


def compact_selector_read(experiment: dict[str, Any]) -> dict[str, Any]:
    selector = experiment.get("selector_info", {})
    selected_configs = selector.get("selected_configs", {})
    target_details = selector.get("target_details", {})
    return {
        "method": experiment.get("method"),
        "local_cv": experiment.get("local_cv"),
        "base_source": selector.get("base_source") or selector.get("clean_source"),
        "extreme_source": selector.get("extreme_source"),
        "selected_configs": selected_configs,
        "target_clean_weights": selector.get("target_clean_weights", {}),
        "target_extreme_weights": selector.get("target_extreme_weights", {}),
        "target_detail_keys": sorted(target_details.keys()) if isinstance(target_details, dict) else [],
        "validation_lens": selector.get("validation_lens"),
        "selection_contract": selector.get("selection_contract"),
    }


def data_readiness() -> dict[str, Any]:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    feature_columns = [column for column in train.columns if column not in TARGETS]
    group_counts = Counter(feature_group(column) for column in feature_columns)
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column]) and column in test.columns
    ]
    all_null_numeric = [
        column
        for column in numeric_columns
        if float(train[column].isna().mean()) == 1.0 and float(test[column].isna().mean()) == 1.0
    ]
    high_missing_numeric = [
        {
            "column": column,
            "train_missing": float(train[column].isna().mean()),
            "test_missing": float(test[column].isna().mean()),
            "group": feature_group(column),
        }
        for column in numeric_columns
        if max(float(train[column].isna().mean()), float(test[column].isna().mean())) >= 0.45
    ]
    target_prevalence = {target: float(train[target].mean()) for target in TARGETS}
    return {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_columns),
        "feature_group_counts": dict(sorted(group_counts.items())),
        "all_null_numeric_count": len(all_null_numeric),
        "all_null_numeric_columns": all_null_numeric,
        "high_missing_numeric_top": sorted(
            high_missing_numeric,
            key=lambda row: max(row["train_missing"], row["test_missing"]),
            reverse=True,
        )[:30],
        "target_prevalence": target_prevalence,
    }


def split_readiness() -> dict[str, Any]:
    data = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    split_counts = Counter(row["split_name"] for row in data.get("folds", []))
    return {
        "exists": True,
        "n_train_rows": data.get("n_train_rows"),
        "n_test_rows": data.get("n_test_rows"),
        "n_folds_total": len(data.get("folds", [])),
        "split_counts": dict(sorted(split_counts.items())),
    }


def target_priority(priority: dict[str, Any]) -> list[dict[str, Any]]:
    id226 = next(row for row in priority["source_records"] if int(row["id"]) == 226)
    deltas = id226["full_oof"]["target_delta_vs_178"]
    return [
        {
            "target": target,
            "id226_delta_vs_178": float(deltas[target]),
            "priority": "high" if abs(float(deltas[target])) >= 0.045 else "medium",
        }
        for target in sorted(TARGETS, key=lambda item: deltas[item])
    ]


def build_stage_contract(priority: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "clean_id221_feature_similarity_root",
            "purpose": "Rebuild the ID221-like feature-similarity signal without public-LB or full-OOF target leakage.",
            "inputs": [
                "processed numeric features only, fitted inside each outer fold",
                "feature families: date_time, screen_app, move_gps, light_hr, plus conservative calendar/demographic controls",
                "train labels from training rows only for each frozen fold",
            ],
            "candidate_grid": {
                "families": ["date_time", "screen_app", "move_gps", "light_hr"],
                "k_values": [1, 3, 7],
                "date_weights": [0.0, 0.03, 0.1],
                "blend_weights": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07],
                "imputation": "median/indicator fitted on training fold only",
                "scaling": "standard/robust scaler fitted on training fold only for distance features",
            },
            "selection_rule": (
                "Select per-target family/k/date/blend configs using only inner training-fold validation. "
                "Outer valid rows are score-only."
            ),
            "must_output": ["clean_id221_oof.parquet", "clean_id221_test.parquet", "clean_id221_metrics.json/.md"],
            "promotion_gate": "Must improve ID178 or ID341 on frozen primary lenses with low test-geometry movement; otherwise hold as diagnostic only.",
        },
        {
            "stage": "clean_id226_targetwise_convex_backbone",
            "purpose": "Rebuild the ID226-like convex backbone without reusing the existing ID172 extreme full-OOF calibration artifact.",
            "inputs": [
                "clean ID221 source from the previous stage",
                "strictly fold-internal calibrated source candidates trained/fitted inside each outer fold",
                "optional conservative ID178 anchor for geometry stress comparison only, not as public-selected truth",
            ],
            "candidate_grid": {
                "calibrator_families": ["none", "temperature", "quantile_smooth", "isotonic_minbin"],
                "convex_weights_clean_id221": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                "convex_weights_calibrated_source": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
                "min_bin_size": [24, 32, 48],
                "targets_first": [row["target"] for row in target_priority(priority)[:4]],
            },
            "selection_rule": (
                "For each target, fit calibrators and choose convex weights inside inner folds only. "
                "No target may use the same row for fitting a calibrator, selecting a weight, and reporting the outer score."
            ),
            "must_output": ["clean_id226_oof.parquet", "clean_id226_test.parquet", "clean_id226_metrics.json/.md"],
            "promotion_gate": "Must beat ID341 proxy after transfer-gate geometry before any local artifact contract is considered.",
        },
        {
            "stage": "hold_calibration_and_replay_layers",
            "purpose": "Prevent the old failure mode from re-entering before a clean backbone exists.",
            "inputs": ["ID227/228/237", "ID238/239/241", "B1 254/255/256/257/258/266"],
            "candidate_grid": {},
            "selection_rule": "No replay or calibration derivative is allowed until clean ID221 and ID226 reports are green.",
            "must_output": ["hold_decision.json/.md"],
            "promotion_gate": "Always no-submit at this stage.",
        },
    ]


def build_report(priority_path: Path) -> dict[str, Any]:
    priority = load_priority(priority_path)
    priority_decision = priority.get("decision", {})
    id221_exp = load_experiment(221)
    id226_exp = load_experiment(226)
    focus = priority_decision.get("first_rebuild_focus", [])
    ready = bool(
        priority_decision.get("any_existing_artifact_promotable") is False
        and any("ID221" in item for item in focus)
        and any("ID226" in item for item in focus)
        and TRAIN_PATH.exists()
        and TEST_PATH.exists()
        and SPLIT_MANIFEST_PATH.exists()
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_clean_backbone_rebuild_design",
        "purpose": "Define the report-first clean ID221/ID226 rebuild contract before any Kaggle heavy run.",
        "priority_report": str(priority_path.relative_to(ROOT))
        if priority_path.is_relative_to(ROOT)
        else str(priority_path),
        "data_readiness": data_readiness(),
        "split_readiness": split_readiness(),
        "source_metadata_reads": {
            "id221": compact_selector_read(id221_exp),
            "id226": compact_selector_read(id226_exp),
        },
        "target_priority_from_id226": target_priority(priority),
        "stage_contract": build_stage_contract(priority),
        "decision": {
            "clean_backbone_design_ready": ready,
            "first_heavy_compute_focus": "clean_id221_then_clean_id226",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "If explicitly approved, push/run the CPU-only Kaggle runner in report-first mode. "
                "The next notebook revision after pulled green reports should implement only clean ID221/ID226 rebuild cells, "
                "not ID227/228/237 or A1/B1 replay."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 clean backbone rebuild design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no prediction artifact, and makes no DACON submission.",
        "",
        "## Data readiness",
        "",
        f"- Train/test shape: `{report['data_readiness']['train_shape']}` / `{report['data_readiness']['test_shape']}`",
        f"- Numeric feature count: `{report['data_readiness']['numeric_feature_count']}`",
        f"- All-null numeric columns: `{report['data_readiness']['all_null_numeric_count']}`",
        "",
        "Feature group counts:",
        "",
        "```json",
        json.dumps(report["data_readiness"]["feature_group_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Target priority from existing ID226 signal",
        "",
        "| target | ID226 delta vs ID178 | priority |",
        "|---|---:|---|",
    ]
    for row in report["target_priority_from_id226"]:
        lines.append(f"| `{row['target']}` | {row['id226_delta_vs_178']:+.6f} | `{row['priority']}` |")
    lines.extend(
        [
            "",
            "## Stage contract",
            "",
        ]
    )
    for stage in report["stage_contract"]:
        lines.extend(
            [
                f"### `{stage['stage']}`",
                "",
                stage["purpose"],
                "",
                f"- Selection rule: {stage['selection_rule']}",
                f"- Must output: `{', '.join(stage['must_output'])}`",
                f"- Promotion gate: {stage['promotion_gate']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- Clean backbone design ready: `{report['decision']['clean_backbone_design_ready']}`",
            f"- First heavy compute focus: `{report['decision']['first_heavy_compute_focus']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
            f"- Run heavy rebuild now: `{report['decision']['run_heavy_rebuild_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority-report", type=Path, default=DEFAULT_PRIORITY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_clean_backbone_rebuild_design_20260618.json",
    )
    args = parser.parse_args()
    priority_path = args.priority_report if args.priority_report.is_absolute() else ROOT / args.priority_report
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(priority_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
