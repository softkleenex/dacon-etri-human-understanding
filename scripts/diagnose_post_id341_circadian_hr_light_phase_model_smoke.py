#!/usr/bin/env python3
"""No-submit fold-internal model smoke for post-ID341 circadian features.

This diagnostic is the first model-level check after materializing
`post_id341_circadian_hr_light_phase_features_v1.parquet`. It trains only tiny
fold-internal in-memory models for Q3/S3/S2 and compares them against ID178/ID341
OOF anchors across frozen validation lenses. It writes reports only: no full-test
predictions, submission CSV, base-prediction artifact, notebook/history artifact,
submission-log row, Kaggle run, persisted model, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
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
    BLEND_WEIGHTS,
    CONFIG_MAX_DELTA_CEILING,
    CONFIG_MEAN_DELTA_CEILING,
    CONFIG_WIN_FLOOR,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_target_for_split,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
MATRIX_PATH = ROOT / "data" / "processed" / "post_id341_circadian_hr_light_phase_features_v1.parquet"
SCHEMA_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_features_v1_schema.json"
MODEL_SMOKE_CONTRACT_REPORT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_model_smoke_contract_20260619.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_circadian_hr_light_phase_model_smoke_20260619.json"

SOURCE_FAMILY = "circadian_hr_light_phase_shape_v1"
TARGET_FOCUS = ["Q3", "S3", "S2"]
ANCHOR_IDS = [178, 341]
KEY_COLUMNS = ["subject_id", "lifelog_date", "split"]
FORBIDDEN_COLUMN_FRAGMENTS = ("target", "oof", "pred", "prediction", "submission", "public", "leaderboard")
ACTION_DECISION_KEYS = [
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {"path": rel(path), "exists": True, "parse_ok": False, "decision": {}, "error": str(exc)}
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
        "error": None,
    }


def no_action_flags(report: dict[str, Any]) -> bool:
    decision = report.get("decision", {}) if isinstance(report.get("decision"), dict) else {}
    return all(decision.get(key) is not True for key in ACTION_DECISION_KEYS)


def schema_contract() -> dict[str, Any]:
    schema = report_status(SCHEMA_REPORT)
    model_contract = report_status(MODEL_SMOKE_CONTRACT_REPORT)
    schema_dec = schema.get("decision", {}) if isinstance(schema.get("decision"), dict) else {}
    model_dec = model_contract.get("decision", {}) if isinstance(model_contract.get("decision"), dict) else {}
    model_contract_future_or_completed = bool(
        model_dec.get("model_smoke_run_allowed_next_under_contract") is True
        or model_dec.get("model_smoke_already_completed_under_contract") is True
    )
    checks = {
        "schema_exists_and_parse_ok": schema.get("exists") is True and schema.get("parse_ok") is True,
        "model_contract_exists_and_parse_ok": model_contract.get("exists") is True
        and model_contract.get("parse_ok") is True,
        "matrix_materialized": schema_dec.get("feature_matrix_materialized") is True,
        "schema_source_family_matches": schema_dec.get("source_family") == SOURCE_FAMILY,
        "schema_target_focus_matches": schema_dec.get("target_focus") == TARGET_FOCUS,
        "schema_boundary_passed": schema_dec.get("allowed_boundary_passed") is True,
        "schema_score_actions_closed": no_action_flags(schema),
        "model_contract_ready": model_dec.get("model_smoke_contract_ready") is True,
        "model_contract_allows_future_or_completed_smoke": model_contract_future_or_completed,
        "model_contract_does_not_train_now": model_dec.get("model_smoke_run_allowed_now") is False
        and model_dec.get("train_model_now") is False,
        "model_contract_actions_closed": no_action_flags(model_contract),
        "matrix_path_exists": MATRIX_PATH.exists(),
    }
    return {
        "schema_report": schema,
        "model_smoke_contract_report": model_contract,
        "checks": checks,
        "contract_ready_for_model_smoke": all(checks.values()),
    }


def existing_model_smoke_report_is_reusable(report: dict[str, Any]) -> bool:
    decision = report.get("decision", {}) if isinstance(report.get("decision"), dict) else {}
    safety = report.get("safety_contract", {}) if isinstance(report.get("safety_contract"), dict) else {}
    feature_metadata = report.get("feature_metadata", {}) if isinstance(report.get("feature_metadata"), dict) else {}
    return bool(
        report.get("diagnostic") == "post_id341_circadian_hr_light_phase_model_smoke"
        and decision.get("model_smoke_completed") is True
        and decision.get("source_family") == SOURCE_FAMILY
        and decision.get("target_focus") == TARGET_FOCUS
        and decision.get("public_058_evidence") is False
        and decision.get("score_route_opened") is False
        and decision.get("create_submission_artifact") is False
        and decision.get("submit_now") is False
        and decision.get("push_kaggle_now") is False
        and decision.get("run_kaggle_now") is False
        and safety.get("persisted_model_created") is False
        and safety.get("full_test_predictions_created") is False
        and safety.get("submission_csv_created") is False
        and safety.get("submission_log_modified") is False
        and safety.get("kaggle_run_executed") is False
        and safety.get("dacon_submission_executed") is False
        and feature_metadata.get("aligned_train_rows") == 450
        and feature_metadata.get("feature_columns") == 48
    )


def load_circadian_feature_frame(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_parquet(MATRIX_PATH)
    train_matrix = matrix[matrix["split"].eq("train")].copy()
    train_matrix["date"] = pd.to_datetime(train_matrix["lifelog_date"]).dt.date
    feature_cols = [
        column
        for column in train_matrix.columns
        if column not in (*KEY_COLUMNS, "date")
        and not any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
    ]
    aligned = train[["subject_id", "date"]].merge(
        train_matrix[["subject_id", "date", *feature_cols]],
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(train):
        raise ValueError("circadian feature matrix does not align with sorted train rows.")
    metadata = {
        "matrix_path": rel(MATRIX_PATH),
        "aligned_train_rows": int(len(aligned)),
        "feature_columns": int(len(feature_cols)),
        "feature_column_names": feature_cols,
        "missing_rate": float(aligned[feature_cols].isna().mean().mean()) if feature_cols else None,
        "forbidden_feature_columns": sorted(
            column
            for column in train_matrix.columns
            if any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
        ),
        "test_rows_present_but_unused_for_model_smoke": int(matrix["split"].eq("test").sum()),
    }
    return aligned, metadata


def scan_rows_for_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    target: str,
    split_names: Iterable[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered = predict_target_for_split(features, feature_cols, y, folds, spec)
            id178_loss = target_loss_on_split(train, folds, target, references["id178"])
            id341_loss = target_loss_on_split(train, folds, target, references["id341"])
            source_values = references["id341"].copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), references["id341"][:, target_idx], pred)
            source_loss = target_loss_on_split(train, folds, target, source_values)
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"circadian_phase_source_{spec.name}",
                    "model_spec": asdict(spec),
                    "anchor": None,
                    "weight": None,
                    "valid_rows": int(covered),
                    "loss": source_loss,
                    "id178_loss": id178_loss,
                    "id341_loss": id341_loss,
                    "delta_vs_id178": float(source_loss - id178_loss),
                    "delta_vs_id341": float(source_loss - id341_loss),
                }
            )
            for anchor_name, anchor_values in references.items():
                for weight in BLEND_WEIGHTS:
                    blended = anchor_values.copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                        1e-6,
                        1 - 1e-6,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    anchor_loss = id178_loss if anchor_name == "id178" else id341_loss
                    rows.append(
                        {
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"blend_{anchor_name}_circadian_phase_{spec.name}_w{weight:g}",
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "valid_rows": int(covered),
                            "loss": loss,
                            "id178_loss": id178_loss,
                            "id341_loss": id341_loss,
                            "delta_vs_anchor": float(loss - anchor_loss),
                            "delta_vs_id178": float(loss - id178_loss),
                            "delta_vs_id341": float(loss - id341_loss),
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
            deltas = group.set_index("split_name")["delta_vs_id341"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            delta_values = [float(value) for value in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "candidate": str(candidate),
                    "model_spec": row0.get("model_spec"),
                    "anchor": row0.get("anchor"),
                    "weight": row0.get("weight"),
                    "primary_wins_vs_id341": int(sum(delta < 0 for delta in delta_values)),
                    "mean_delta_vs_id341": float(np.mean(delta_values)),
                    "min_delta_vs_id341": float(np.min(delta_values)),
                    "max_delta_vs_id341": float(np.max(delta_values)),
                    "split_deltas_vs_id341": {str(key): float(value) for key, value in deltas.items()},
                    "split_losses": {str(key): float(value) for key, value in losses.items()},
                }
            )
        out[target] = sorted(
            summaries,
            key=lambda row: (
                -row["primary_wins_vs_id341"],
                row["mean_delta_vs_id341"],
                row["max_delta_vs_id341"],
                row["candidate"],
            ),
        )
    return out


def select_target_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in summaries.items():
        for row in rows:
            if (
                row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
                and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
                and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
            ):
                selected[target] = row
                break
    return selected


def anchor_table(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    split_names: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        for split_name in split_names:
            folds = splits[split_name]
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "id178_loss": target_loss_on_split(train, folds, target, references["id178"]),
                    "id341_loss": target_loss_on_split(train, folds, target, references["id341"]),
                }
            )
    return rows


def build_report(split_names: list[str] | None = None) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    contract = schema_contract()
    if not contract["contract_ready_for_model_smoke"]:
        raise RuntimeError(f"circadian model smoke contract is not ready: {contract['checks']}")
    model_contract_decision = contract["model_smoke_contract_report"]["decision"]
    if model_contract_decision.get("model_smoke_already_completed_under_contract") is True:
        existing_report = load_json(DEFAULT_OUTPUT) if DEFAULT_OUTPUT.exists() else {}
        if existing_model_smoke_report_is_reusable(existing_report):
            existing_report["contract"] = contract
            existing_report["idempotent_reused_existing_report"] = True
            existing_report["no_new_model_training_on_this_invocation"] = True
            return existing_report
        raise RuntimeError("circadian model smoke is marked completed, but the existing report is missing or unsafe.")
    train = load_train_sorted()
    features, feature_metadata = load_circadian_feature_frame(train)
    feature_cols = [
        column
        for column in feature_metadata["feature_column_names"]
        if pd.api.types.is_numeric_dtype(features[column]) and features[column].notna().any()
    ]
    splits = load_split_manifest()
    references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(scan_rows_for_target(train, features, feature_cols, splits, references, target, split_names))
    summaries = summarize_rows(scan_rows)
    selected = select_target_configs(summaries)
    selected_targets = sorted(selected.keys())
    any_signal = bool(selected_targets)
    all_targets_covered = selected_targets == TARGET_FOCUS
    top_rows_by_target = {target: rows[:8] for target, rows in summaries.items()}
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_circadian_hr_light_phase_model_smoke",
        "purpose": (
            "Run a lightweight fold-internal no-submit model smoke on the registered circadian HR/light feature matrix, "
            "comparing Q3/S3/S2 against ID178/ID341 OOF anchors."
        ),
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "feature_matrix": rel(MATRIX_PATH),
            "schema_report": rel(SCHEMA_REPORT),
            "model_smoke_contract": rel(MODEL_SMOKE_CONTRACT_REPORT),
        },
        "contract": contract,
        "feature_metadata": feature_metadata,
        "config": {
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "anchor_rows": anchor_table(train, splits, references, split_names),
        "selected_target_configs": selected,
        "top_rows_by_target": top_rows_by_target,
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "model_smoke_completed": True,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "any_circadian_phase_signal_found": any_signal,
            "all_focus_targets_covered": all_targets_covered,
            "public_058_evidence": False,
            "score_route_opened": False,
            "next_local_allowed_step": "circadian_phase_no_artifact_geometry_gate"
            if any_signal
            else "close_circadian_phase_model_smoke",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If selected targets exist, run a no-artifact geometry gate before any artifact contract. "
                "Otherwise close the circadian HR/light local model lane and return to the explicit-approved upstream route."
            ),
        },
        "safety_contract": {
            "in_memory_models_trained": True,
            "persisted_model_created": False,
            "full_test_predictions_created": False,
            "submission_csv_created": False,
            "base_prediction_artifact_created": False,
            "notebook_history_artifact_created": False,
            "submission_log_modified": False,
            "kaggle_run_executed": False,
            "dacon_submission_executed": False,
            "uses_target_labels_for_training_only_inside_cv": True,
            "uses_oof_predictions_as_baseline_only": True,
            "uses_public_scores": False,
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    feature_metadata = report["feature_metadata"]
    lines = [
        "# Circadian HR/light model smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is a no-submit, in-memory model smoke. It creates no full-test predictions, submission CSV, "
            "base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, persisted model, "
            "or DACON submission."
        ),
        "",
        "## Feature input",
        "",
        f"- Matrix: `{feature_metadata['matrix_path']}`",
        f"- Aligned train rows: `{feature_metadata['aligned_train_rows']}`",
        f"- Feature columns used: `{feature_metadata['feature_columns']}`",
        f"- Missing rate: `{feature_metadata['missing_rate']:.6f}`",
        f"- Test rows present but unused for model smoke: `{feature_metadata['test_rows_present_but_unused_for_model_smoke']}`",
        "",
        "## Selected target configs",
        "",
        "| target | candidate | wins | mean Δ341 | max Δ341 |",
        "|---|---|---:|---:|---:|",
    ]
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    lines.extend(["", "## Top rows by target", ""])
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | candidate | wins | mean Δ341 | max Δ341 |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows[:5], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Any circadian HR/light signal found: `{decision['any_circadian_phase_signal_found']}`",
            f"- Selected targets: `{', '.join(decision['selected_targets']) if decision['selected_targets'] else '(none)'}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Score route opened: `{decision['score_route_opened']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
            "## Boundary",
            "",
            (
                "This report can justify only another no-artifact geometry check. It is not a Public 0.58 claim and "
                "does not create ID342 or authorize DACON/Kaggle actions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    write_report(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        json.dumps(
            {
                "model_smoke_completed": decision["model_smoke_completed"],
                "selected_targets": decision["selected_targets"],
                "any_circadian_phase_signal_found": decision["any_circadian_phase_signal_found"],
                "public_058_evidence": decision["public_058_evidence"],
                "create_submission_artifact": decision["create_submission_artifact"],
                "submit_now": decision["submit_now"],
                "push_kaggle_now": decision["push_kaggle_now"],
                "run_kaggle_now": decision["run_kaggle_now"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
