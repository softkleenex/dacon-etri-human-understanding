#!/usr/bin/env python3
"""Phase-2 block B: S2 sleep/recovery retry after ID341.

This report-only diagnostic implements the second block from the post-ID341
phase-2 plan. Block A found a clean but too-small Q1 sleep/residual signal. This
block asks a narrower question: can S2 be improved by a distinct raw
sleep/recovery source without borrowing Q1/S4 movement or widening into a local
blend cascade?

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, no Kaggle push/run, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (  # noqa: E402
    build_evening_load_recovery_features,
)
from scripts.diagnose_phase2_block_a_q1_s4_sleep_residual_bridge_post_id341 import (  # noqa: E402
    build_bridge_features,
    evaluate_assembled_candidate,
    key_frame,
    predict_full_train_target,
    predict_target_for_split,
    selected_test_geometry,
    target_loss_on_split,
)
from scripts.diagnose_subject_residual_sequence_drift_smoke_post_id341 import (  # noqa: E402
    EPS,
    ID178_PUBLIC,
    ID341_TRANSFER_PROXY,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    PUBLIC_058_REVIEW_CUTOFF,
    TARGETS,
    load_processed_frames,
    load_reference_values,
    load_split_manifest,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
PHASE2_PLAN_PATH = REPORTS_DIR / "phase2_raw_multimodal_execution_plan_post_id341_20260618.json"
BLOCK_A_GAP_PATH = REPORTS_DIR / "phase2_block_a_q1_proxy_gap_post_id341_20260618.json"
TARGET_SCAN_FOCUS = ["S2"]
ANCHORS = ["id178", "id341"]
MODEL_NAMES = ["lr_c0_03", "lr_c0_12", "hgb_l4_l2_5"]
BLEND_WEIGHTS = [0.005, 0.01, 0.025, 0.05]
CONFIG_WIN_FLOOR = 4
CONFIG_MEAN_DELTA_CEILING = -0.00005
CONFIG_MAX_DELTA_CEILING = 0.00050
FINAL_PRIMARY_WINS_FLOOR = 5


def build_s2_recovery_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bridge_train, bridge_test, bridge_coverage = build_bridge_features(train, test)
    evening_features, evening_coverage = build_evening_load_recovery_features(train, test)
    evening_train = evening_features[evening_features["is_test"].eq(0)].reset_index(drop=True)
    evening_test = evening_features[evening_features["is_test"].eq(1)].reset_index(drop=True)

    if not key_frame(train).equals(key_frame(bridge_train)) or not key_frame(train).equals(key_frame(evening_train)):
        raise ValueError("S2 recovery train features do not align with processed train rows.")
    if not key_frame(test).equals(key_frame(bridge_test)) or not key_frame(test).equals(key_frame(evening_test)):
        raise ValueError("S2 recovery test features do not align with processed test rows.")

    bridge_cols = [col for col in bridge_train.columns if col.startswith(("sbt_", "sr_"))]
    evening_cols = [col for col in evening_train.columns if col.startswith("elr_")]
    recovery_train = pd.concat(
        [
            train[["subject_id", "date"]].reset_index(drop=True),
            bridge_train[bridge_cols].reset_index(drop=True),
            evening_train[evening_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    recovery_test = pd.concat(
        [
            test[["subject_id", "date"]].reset_index(drop=True),
            bridge_test[bridge_cols].reset_index(drop=True),
            evening_test[evening_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    feature_cols = bridge_cols + evening_cols
    coverage = {
        "n_train_rows": int(len(recovery_train)),
        "n_test_rows": int(len(recovery_test)),
        "sleep_boundary_feature_columns": int(sum(col.startswith("sbt_") for col in bridge_cols)),
        "subject_residual_feature_columns": int(sum(col.startswith("sr_") for col in bridge_cols)),
        "evening_recovery_feature_columns": int(len(evening_cols)),
        "s2_recovery_feature_columns": int(len(feature_cols)),
        "bridge_coverage": {
            key: bridge_coverage.get(key)
            for key in [
                "sleep_feature_columns",
                "residual_feature_columns",
                "bridge_feature_columns",
                "train_missing_mean",
                "test_missing_mean",
            ]
        },
        "evening_coverage": {
            key: evening_coverage.get(key)
            for key in [
                "n_feature_columns",
                "base_columns_available",
                "base_columns_missing",
                "train_missing_mean",
                "test_missing_mean",
            ]
        },
        "train_missing_mean": float(recovery_train[feature_cols].isna().mean().mean()),
        "test_missing_mean": float(recovery_test[feature_cols].isna().mean().mean()),
    }
    return recovery_train, recovery_test, coverage


def scan_s2_configs(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    references = {"id178": reference178, "id341": reference341}
    source_cache: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    target = "S2"
    target_idx = TARGETS.index(target)
    id341_losses = {
        split_name: target_loss_on_split(train, folds, target, reference341[:, target_idx])
        for split_name, folds in splits.items()
    }
    y = train[target].astype(int).to_numpy()
    for spec in MODEL_SPECS:
        if spec.name not in MODEL_NAMES:
            continue
        for split_name, folds in splits.items():
            source_cache[(split_name, target, spec.name)] = predict_target_for_split(
                features_train, feature_cols, y, folds, spec.name
            )
        for anchor_name in ANCHORS:
            anchor = references[anchor_name][:, target_idx]
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                for split_name, folds in splits.items():
                    source = np.where(
                        np.isnan(source_cache[(split_name, target, spec.name)]),
                        reference341[:, target_idx],
                        source_cache[(split_name, target, spec.name)],
                    )
                    values = np.clip((1.0 - weight) * anchor + weight * source, EPS, 1 - EPS)
                    loss = target_loss_on_split(train, folds, target, values)
                    split_losses[split_name] = loss
                    split_deltas[split_name] = float(loss - id341_losses[split_name])
                deltas = list(split_deltas.values())
                rows.append(
                    {
                        "target": target,
                        "model": spec.name,
                        "model_spec": asdict(spec),
                        "anchor": anchor_name,
                        "weight": weight,
                        "candidate": f"{target}_{anchor_name}_sleep_recovery_retry_{spec.name}_w{weight:g}",
                        "primary_wins_vs_id341": int(sum(delta < 0 for delta in deltas)),
                        "mean_delta_vs_id341": float(np.mean(deltas)),
                        "min_delta_vs_id341": float(np.min(deltas)),
                        "max_delta_vs_id341": float(np.max(deltas)),
                        "split_deltas_vs_id341": split_deltas,
                        "split_losses": split_losses,
                    }
                )
    rows = sorted(
        rows,
        key=lambda row: (
            -row["primary_wins_vs_id341"],
            row["mean_delta_vs_id341"],
            row["max_delta_vs_id341"],
            row["anchor"],
            row["weight"],
        ),
    )
    return rows, source_cache


def select_s2_config(scan_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eligible = [
        row
        for row in scan_rows
        if row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
        and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
        and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
    ]
    if not eligible:
        return {}
    return {
        "S2": sorted(
            eligible,
            key=lambda row: (-row["primary_wins_vs_id341"], row["mean_delta_vs_id341"], row["max_delta_vs_id341"]),
        )[0]
    }


def selected_s2_test_geometry(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    feature_cols: list[str],
    selected_configs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    geometry = selected_test_geometry(train, test, features_train, features_test, feature_cols, selected_configs)
    if geometry is None:
        return None
    target = "S2"
    config = selected_configs[target]
    source = predict_full_train_target(train, features_train, features_test, feature_cols, target, config["model"])
    geometry["s2_full_train_source_summary"] = {
        "model": config["model"],
        "source_mean": float(source.mean()),
        "source_std": float(source.std()),
        "source_min": float(source.min()),
        "source_max": float(source.max()),
    }
    return geometry


def build_report() -> dict[str, Any]:
    train, test = load_processed_frames()
    features_train, features_test, feature_coverage = build_s2_recovery_features(train, test)
    feature_cols = [col for col in features_train.columns if col.startswith(("sbt_", "sr_", "elr_"))]
    splits_all = load_split_manifest()
    splits = {name: splits_all[name] for name in PRIMARY_GATE_SPLITS if name in splits_all}
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows, source_cache = scan_s2_configs(train, features_train, feature_cols, splits, reference178, reference341)
    selected_configs = select_s2_config(scan_rows)
    assembled = evaluate_assembled_candidate(train, splits, reference178, reference341, source_cache, selected_configs)
    mean_delta = assembled["mean_primary_delta_vs_id341"]
    projected_proxy = float(ID341_TRANSFER_PROXY + mean_delta) if mean_delta is not None else ID341_TRANSFER_PROXY
    final_gate_pass = bool(
        selected_configs
        and assembled["primary_wins_vs_id341"] >= FINAL_PRIMARY_WINS_FLOOR
        and assembled["all_primary_splits_beat_id341"]
        and (assembled["max_primary_delta_vs_id341"] is not None and assembled["max_primary_delta_vs_id341"] <= 0.0)
        and projected_proxy < ID178_PUBLIC
    )
    public_058_reviewable = bool(final_gate_pass and projected_proxy <= PUBLIC_058_REVIEW_CUTOFF)
    geometry = selected_s2_test_geometry(train, test, features_train, features_test, feature_cols, selected_configs)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "phase2_block_b_s2_sleep_recovery_retry_post_id341",
        "purpose": (
            "Test the second phase-2 raw multimodal block: S2-only sleep/recovery retry using sleep-boundary, "
            "subject-residual, and evening recovery features under frozen primary validation gates."
        ),
        "phase2_plan": str(PHASE2_PLAN_PATH.relative_to(ROOT)),
        "phase2_plan_exists": PHASE2_PLAN_PATH.exists(),
        "block_a_gap_report": str(BLOCK_A_GAP_PATH.relative_to(ROOT)),
        "block_a_gap_report_exists": BLOCK_A_GAP_PATH.exists(),
        "feature_coverage": feature_coverage,
        "target_scan_focus": TARGET_SCAN_FOCUS,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS if spec.name in MODEL_NAMES],
        "blend_weights": BLEND_WEIGHTS,
        "selection_rule": {
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "final_primary_wins_floor": FINAL_PRIMARY_WINS_FLOOR,
            "final_projected_proxy_below_id178_public": ID178_PUBLIC,
        },
        "top_scan_rows": scan_rows[:16],
        "selected_configs": selected_configs,
        "assembled_candidate": assembled,
        "test_geometry": geometry,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "selected_target_count": int(len(selected_configs)),
            "selected_targets": list(selected_configs),
            "s2_has_config_signal": bool(selected_configs),
            "all_primary_splits_beat_id341": assembled["all_primary_splits_beat_id341"],
            "primary_wins_vs_id341": assembled["primary_wins_vs_id341"],
            "mean_primary_delta_vs_id341": mean_delta,
            "max_primary_delta_vs_id341": assembled["max_primary_delta_vs_id341"],
            "projected_public_proxy_from_id341": projected_proxy,
            "beats_id178_public_anchor_proxy": bool(projected_proxy < ID178_PUBLIC),
            "final_block_b_gate_pass": final_gate_pass,
            "public_058_reviewable": public_058_reviewable,
            "next_action": (
                "If final_block_b_gate_pass is false, close the local S2 sleep/recovery retry as a standalone "
                "artifact path and require a materially different raw source before creating a submission candidate."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    decision = report["decision"]
    lines = [
        "# Phase-2 block B S2 sleep/recovery retry — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Sleep-boundary feature columns: `{coverage['sleep_boundary_feature_columns']}`",
        f"- Subject-residual feature columns: `{coverage['subject_residual_feature_columns']}`",
        f"- Evening/recovery feature columns: `{coverage['evening_recovery_feature_columns']}`",
        f"- Combined S2 recovery feature columns: `{coverage['s2_recovery_feature_columns']}`",
        f"- Train/test missing mean: `{coverage['train_missing_mean']:.4f}` / `{coverage['test_missing_mean']:.4f}`",
        "",
        "## Selected config",
        "",
    ]
    if report["selected_configs"]:
        lines.extend(["| target | config | wins | mean Δ | max Δ |", "|---|---|---:|---:|---:|"])
        for target, row in report["selected_configs"].items():
            lines.append(
                f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
    else:
        lines.append("No S2 sleep/recovery config passed the fixed eligibility rule.")
    lines.extend(
        [
            "",
            "## Assembled candidate by primary split",
            "",
            "| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for split_name, split_report in report["assembled_candidate"]["split_reports"].items():
        candidate = split_report["candidate"]
        id341 = split_report["reference_id341"]
        id178 = split_report["reference_id178"]
        lines.append(
            f"| `{split_name}` | {candidate['mean_loss']:.6f} | {id341['mean_loss']:.6f} | "
            f"{split_report['delta_vs_id341']:+.6f} | {id178['mean_loss']:.6f} | "
            f"{split_report['delta_vs_id178']:+.6f} | `{split_report['beats_id341']}` |"
        )
    lines.extend(
        [
            "",
            "## Top scanned S2 rows",
            "",
            "| config | wins | mean Δ | max Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["top_scan_rows"][:10]:
        lines.append(
            f"| `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    if report.get("test_geometry"):
        geometry = report["test_geometry"]
        lines.extend(
            [
                "",
                "## In-memory test geometry",
                "",
                f"- Mean absolute delta vs ID341 base test: `{geometry['vs_base']['id341']['mean_abs_delta_all_targets']:.6f}`",
                f"- Mean absolute delta vs ID178 base test: `{geometry['vs_base']['id178']['mean_abs_delta_all_targets']:.6f}`",
                f"- S2 full-train source mean/std: `{geometry['s2_full_train_source_summary']['source_mean']:.6f}` / `{geometry['s2_full_train_source_summary']['source_std']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected targets: `{', '.join(decision['selected_targets']) if decision['selected_targets'] else 'none'}`",
            f"- S2 has config signal: `{decision['s2_has_config_signal']}`",
            f"- All primary splits beat ID341: `{decision['all_primary_splits_beat_id341']}`",
            f"- Primary wins vs ID341: `{decision['primary_wins_vs_id341']}`",
            f"- Mean primary Δ vs ID341: `{decision['mean_primary_delta_vs_id341']:+.6f}`",
            f"- Max primary Δ vs ID341: `{decision['max_primary_delta_vs_id341']:+.6f}`",
            f"- Projected public proxy from ID341: `{decision['projected_public_proxy_from_id341']:.6f}`",
            f"- Beats ID178 public anchor proxy: `{decision['beats_id178_public_anchor_proxy']}`",
            f"- Final block-B gate pass: `{decision['final_block_b_gate_pass']}`",
            f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
