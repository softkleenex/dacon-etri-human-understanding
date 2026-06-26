#!/usr/bin/env python3
"""ID338-preserving material-source Q3/S3 no-submit probe.

This report-only diagnostic consumes already registered label-free material
feature matrices and asks whether any small fold-internal Q3/S3 overlay can
improve over the confirmed ID338 public anchor without moving far from it.

It creates no test predictions, no submission CSV, no base prediction artifact,
no notebook/history artifact, no submission-log row, no Kaggle run, and no
DACON submission.
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

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    EPS,
    ModelSpec,
    find_oof,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_target_for_split,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id338_material_q3s3_probe_20260623.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TARGET_FOCUS = ["Q3", "S3"]
ANCHOR_ID = 338
CONTEXT_ANCHOR_IDS = [178, 287, 337, 338, 340]
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
MODEL_SPECS = (
    ModelSpec("lr_c0_01", "lr", c_value=0.01),
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_08", "lr", c_value=0.08),
    ModelSpec("lr_c0_20", "lr", c_value=0.20),
)
BLEND_WEIGHTS = [0.0025, 0.005, 0.01, 0.02, 0.035, 0.05]
CONFIG_WIN_FLOOR = 4
CONFIG_MEAN_DELTA_CEILING = -0.00005
CONFIG_MAX_DELTA_CEILING = 0.00050
CONFIG_MEAN_MOVEMENT_CEILING = 0.00085

SOURCE_MATRICES = {
    "phone_wearable_discordance": PROCESSED_DIR / "post_id341_phone_wearable_discordance_features_v1.parquet",
    "proximity_ambience": PROCESSED_DIR / "post_id341_proximity_ambience_cooccurrence_features_v1.parquet",
    "gps_path_topology": PROCESSED_DIR / "post_id341_gps_path_topology_features_v1.parquet",
    "circadian_hr_light": PROCESSED_DIR / "post_id341_circadian_hr_light_phase_features_v1.parquet",
    "usage_semantic": PROCESSED_DIR / "post_id341_usage_semantic_burst_markov_features_v1.parquet",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_source_matrix(source_name: str, path: Path, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "lifelog_date" not in frame.columns:
        raise ValueError(f"{path} missing lifelog_date")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["lifelog_date"]).dt.date
    train_part = frame[frame["split"].eq("train")].copy()
    feature_cols = [
        col
        for col in train_part.columns
        if col not in {"subject_id", "sleep_date", "lifelog_date", "date", "split"}
        and pd.api.types.is_numeric_dtype(train_part[col])
        and train_part[col].notna().any()
    ]
    renamed = train_part[["subject_id", "date", *feature_cols]].rename(
        columns={col: f"{source_name}__{col}" for col in feature_cols}
    )
    aligned = train[["subject_id", "date"]].merge(renamed, on=["subject_id", "date"], how="left", validate="one_to_one")
    prefixed_cols = [f"{source_name}__{col}" for col in feature_cols]
    metadata = {
        "path": rel(path),
        "rows": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "feature_columns": int(len(prefixed_cols)),
        "aligned_rows": int(len(aligned)),
        "mean_missing_rate": float(aligned[prefixed_cols].isna().mean().mean()) if prefixed_cols else None,
        "all_train_rows_aligned": bool(len(aligned) == len(train) and aligned[prefixed_cols].notna().any(axis=1).all()),
    }
    return aligned, metadata


def build_feature_groups(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    base = train[["subject_id", "date"]].copy()
    groups: dict[str, list[str]] = {}
    source_metadata: dict[str, Any] = {}
    for source_name, path in SOURCE_MATRICES.items():
        aligned, metadata = load_source_matrix(source_name, path, train)
        cols = [col for col in aligned.columns if col.startswith(f"{source_name}__")]
        base = base.merge(aligned[["subject_id", "date", *cols]], on=["subject_id", "date"], how="left", validate="one_to_one")
        groups[source_name] = cols
        source_metadata[source_name] = metadata
    groups["all_material_sources"] = [col for cols in groups.values() for col in cols]
    overall_metadata = {
        "sources": source_metadata,
        "total_feature_columns": int(len(groups["all_material_sources"])),
        "all_sources_train_aligned": bool(all(row["all_train_rows_aligned"] for row in source_metadata.values())),
    }
    return base, groups, overall_metadata


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def evaluate_candidate_matrix(
    target: str,
    candidate_values: np.ndarray,
    references: dict[int, np.ndarray],
) -> dict[str, float]:
    target_idx = TARGETS.index(target)
    delta = np.abs(candidate_values[:, target_idx] - references[ANCHOR_ID][:, target_idx])
    failed_direction = (references[287] - references[178]).reshape(-1)
    vec178 = (candidate_values - references[178]).reshape(-1)
    return {
        "mean_target_movement_vs_id338": float(delta.mean()),
        "max_target_movement_vs_id338": float(delta.max()),
        "mean_all_target_movement_vs_id338": float(np.mean(np.abs(candidate_values - references[ANCHOR_ID]))),
        "max_all_target_movement_vs_id338": float(np.max(np.abs(candidate_values - references[ANCHOR_ID]))),
        "oof_projection_failed_direction": projection(vec178, failed_direction),
    }


def scan_rows_for_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[int, np.ndarray],
    source_group: str,
    target: str,
    model_specs: tuple[ModelSpec, ...],
    split_names: list[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered = predict_target_for_split(features, feature_cols, y, folds, spec)
            anchor_loss = target_loss_on_split(train, folds, target, references[ANCHOR_ID])
            source_values = references[ANCHOR_ID].copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), references[ANCHOR_ID][:, target_idx], pred)
            source_loss = target_loss_on_split(train, folds, target, source_values)
            source_metrics = evaluate_candidate_matrix(target, source_values, references)
            rows.append(
                {
                    "source_group": source_group,
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"{source_group}_{spec.name}_source",
                    "model_spec": asdict(spec),
                    "weight": None,
                    "valid_rows": int(covered),
                    "loss": source_loss,
                    "id338_loss": anchor_loss,
                    "delta_vs_id338": float(source_loss - anchor_loss),
                    **source_metrics,
                }
            )
            for weight in BLEND_WEIGHTS:
                blended = references[ANCHOR_ID].copy()
                blended[:, target_idx] = np.clip(
                    (1.0 - weight) * references[ANCHOR_ID][:, target_idx] + weight * source_values[:, target_idx],
                    EPS,
                    1 - EPS,
                )
                loss = target_loss_on_split(train, folds, target, blended)
                blended_metrics = evaluate_candidate_matrix(target, blended, references)
                rows.append(
                    {
                        "source_group": source_group,
                        "target": target,
                        "split_name": split_name,
                        "candidate": f"blend_id338_{source_group}_{spec.name}_w{weight:g}",
                        "model_spec": asdict(spec),
                        "weight": weight,
                        "valid_rows": int(covered),
                        "loss": loss,
                        "id338_loss": anchor_loss,
                        "delta_vs_id338": float(loss - anchor_loss),
                        **blended_metrics,
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    frame = pd.DataFrame(rows)
    out: dict[str, list[dict[str, Any]]] = {}
    for target in TARGET_FOCUS:
        target_frame = frame[frame["target"].eq(target)].copy()
        summaries: list[dict[str, Any]] = []
        for candidate, group in target_frame.groupby("candidate", sort=False):
            row0 = group.iloc[0].to_dict()
            deltas = group.set_index("split_name")["delta_vs_id338"].to_dict()
            movement = group.set_index("split_name")["mean_target_movement_vs_id338"].to_dict()
            failed_projection = group.set_index("split_name")["oof_projection_failed_direction"].to_dict()
            delta_values = [float(value) for value in deltas.values()]
            movement_values = [float(value) for value in movement.values()]
            summaries.append(
                {
                    "target": target,
                    "source_group": row0["source_group"],
                    "candidate": str(candidate),
                    "model_spec": row0["model_spec"],
                    "weight": safe_float(row0.get("weight")),
                    "primary_wins_vs_id338": int(sum(delta < 0 for delta in delta_values)),
                    "mean_delta_vs_id338": float(np.mean(delta_values)),
                    "min_delta_vs_id338": float(np.min(delta_values)),
                    "max_delta_vs_id338": float(np.max(delta_values)),
                    "mean_target_movement_vs_id338": float(np.mean(movement_values)),
                    "max_target_movement_vs_id338": float(
                        group["max_target_movement_vs_id338"].astype(float).max()
                    ),
                    "mean_failed_projection": float(np.mean([float(v) for v in failed_projection.values()])),
                    "split_deltas_vs_id338": {str(key): float(value) for key, value in deltas.items()},
                    "split_mean_movements_vs_id338": {str(key): float(value) for key, value in movement.items()},
                }
            )
        out[target] = sorted(
            summaries,
            key=lambda row: (
                -row["primary_wins_vs_id338"],
                row["mean_delta_vs_id338"],
                row["max_delta_vs_id338"],
                row["mean_target_movement_vs_id338"],
                row["candidate"],
            ),
        )
    return out


def select_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in summaries.items():
        for row in rows:
            weight = row.get("weight")
            if weight is None:
                continue
            if (
                row["primary_wins_vs_id338"] >= CONFIG_WIN_FLOOR
                and row["mean_delta_vs_id338"] <= CONFIG_MEAN_DELTA_CEILING
                and row["max_delta_vs_id338"] <= CONFIG_MAX_DELTA_CEILING
                and row["mean_target_movement_vs_id338"] <= CONFIG_MEAN_MOVEMENT_CEILING
                and row["mean_failed_projection"] <= 0.02
            ):
                selected[target] = row
                break
    return selected


def anchor_context_rows(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[int, np.ndarray],
    split_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        for split_name in split_names:
            folds = splits[split_name]
            row = {"target": target, "split_name": split_name}
            for anchor_id in [178, 337, 338, 340]:
                row[f"id{anchor_id}_loss"] = target_loss_on_split(train, folds, target, references[anchor_id])
            row["id338_minus_id178"] = row["id338_loss"] - row["id178_loss"]
            row["id338_minus_id337"] = row["id338_loss"] - row["id337_loss"]
            row["id340_minus_id338"] = row["id340_loss"] - row["id338_loss"]
            rows.append(row)
    return rows


def build_report(split_names: list[str] | None = None) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    train = load_train_sorted()
    features, groups, feature_metadata = build_feature_groups(train)
    splits = load_split_manifest()
    references = {candidate_id: load_reference_values(train, candidate_id) for candidate_id in CONTEXT_ANCHOR_IDS}

    scan_rows: list[dict[str, Any]] = []
    for source_group, cols in groups.items():
        for target in TARGET_FOCUS:
            scan_rows.extend(
                scan_rows_for_target(
                    train=train,
                    features=features,
                    feature_cols=cols,
                    splits=splits,
                    references=references,
                    source_group=source_group,
                    target=target,
                    model_specs=MODEL_SPECS,
                    split_names=split_names,
                )
            )
    summaries = summarize_rows(scan_rows)
    selected = select_configs(summaries)
    selected_targets = sorted(selected)
    any_signal = bool(selected_targets)
    all_targets = selected_targets == TARGET_FOCUS
    top_rows_by_target = {target: rows[:12] for target, rows in summaries.items()}
    anchor_rows = anchor_context_rows(train, splits, references, split_names)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id338_material_q3s3_probe",
        "purpose": (
            "Run an ID338-preserving fold-internal no-submit probe over registered material raw-source "
            "feature matrices for Q3/S3 only."
        ),
        "target_focus": TARGET_FOCUS,
        "anchor_id": ANCHOR_ID,
        "inputs": {
            "source_matrices": {name: rel(path) for name, path in SOURCE_MATRICES.items()},
            "id338_oof": rel(find_oof(338)),
            "id287_failed_oof": rel(find_oof(287)),
            "split_manifest": rel(REPORTS_DIR / "id340_validation_split_manifest_20260618.json"),
        },
        "feature_metadata": feature_metadata,
        "config": {
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "config_mean_movement_ceiling": CONFIG_MEAN_MOVEMENT_CEILING,
        },
        "anchor_context_rows": anchor_rows,
        "selected_target_configs": selected,
        "top_rows_by_target": top_rows_by_target,
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "probe_completed": True,
            "selected_targets": selected_targets,
            "any_q3s3_signal_found": any_signal,
            "all_q3s3_targets_covered": all_targets,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": (
                "id338_material_q3s3_no_artifact_geometry_gate" if any_signal else "close_id338_material_q3s3_probe"
            ),
            "next_action": (
                "If selected targets exist, run only a no-artifact geometry gate against ID338/test-shape risk. "
                "Do not create a CSV/base/notebook/log artifact without a separate promotion contract."
            )
            if any_signal
            else "No target passed the ID338-preserving gate; keep waiting for DACON same-batch score visibility and design a different source.",
        },
        "no_full_test_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return "" if value is None else f"{value:.{digits}f}"


def table_rows(rows: list[dict[str, Any]], limit: int = 10) -> list[str]:
    lines = []
    for rank, row in enumerate(rows[:limit], start=1):
        lines.append(
            f"| {rank} | `{row['candidate']}` | `{row['source_group']}` | "
            f"{row['primary_wins_vs_id338']} | {fmt(row['mean_delta_vs_id338'])} | "
            f"{fmt(row['max_delta_vs_id338'])} | {fmt(row['mean_target_movement_vs_id338'])} | "
            f"{fmt(row['mean_failed_projection'], 4)} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# ID338 material Q3/S3 no-submit probe",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Feature Inputs",
        "",
        f"- Total feature columns: `{report['feature_metadata']['total_feature_columns']}`",
        f"- All sources train-aligned: `{report['feature_metadata']['all_sources_train_aligned']}`",
        "",
        "| source | features | mean missing | path |",
        "|---|---:|---:|---|",
    ]
    for name, row in report["feature_metadata"]["sources"].items():
        lines.append(
            f"| `{name}` | {row['feature_columns']} | {fmt(row['mean_missing_rate'], 4)} | `{row['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Selected Target Configs",
            "",
            "| target | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | `{row['source_group']}` | "
            f"{row['primary_wins_vs_id338']} | {fmt(row['mean_delta_vs_id338'])} | "
            f"{fmt(row['max_delta_vs_id338'])} | {fmt(row['mean_target_movement_vs_id338'])} | "
            f"{fmt(row['mean_failed_projection'], 4)} |"
        )
    lines.extend(["", "## Top Rows By Target", ""])
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
                *table_rows(rows, limit=10),
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- Probe completed: `{decision['probe_completed']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Any Q3/S3 signal found: `{decision['any_q3s3_signal_found']}`",
            f"- All Q3/S3 targets covered: `{decision['all_q3s3_targets_covered']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
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
