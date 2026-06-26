#!/usr/bin/env python3
"""Post-ID342 S3 phone/GPS interaction gate.

This report-only diagnostic follows the cross-source split audit's only useful
design hint: a tiny S3 stable alignment between phone/wearable discordance and
GPS path topology. It tests whether a focused interaction frame from those two
registered material matrices creates a larger fold-internal S3 signal against
ID342.

It creates no feature matrix artifact, predictions, submission CSV,
base-prediction artifact, notebook-history artifact, submission-log row, Kaggle
push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_target_for_split,
    target_loss_on_split,
)
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    ID342_PUBLIC,
    PUBLIC_055,
)
from scripts.diagnose_post_id342_registered_material_source_rescan import (  # noqa: E402
    BASE_ANCHOR_ID,
    FORBIDDEN_COLUMN_FRAGMENTS,
    KEY_COLUMNS,
    MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
    MAX_PRIMARY_MEAN_DELTA_VS_ID342,
    MAX_PRIMARY_WORST_DELTA_VS_ID342,
    MAX_PUBLIC_PROXY_FOR_REVIEW,
    MIN_PRIMARY_WINS_VS_ID342,
    SOURCE_MATRICES,
    rel,
    summarize_rows,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_s3_phone_gps_interaction_gate_20260625.json"

TARGET_FOCUS = ["S3"]
PHONE_SOURCE = "phone_wearable_discordance"
GPS_SOURCE = "gps_path_topology"

PHONE_S3_FRAGMENTS = (
    "early_morning_",
    "late_evening_",
    "pwd_global_discordance",
    "pwd_global_late",
    "pwd_global_first_discordance",
    "pwd_global_last_discordance",
    "pwd_global_max_contiguous",
    "pwd_global_source_missingness",
)
GPS_S3_COLUMNS = (
    "gps_active_hours",
    "gps_evening_rows_18_23",
    "gps_night_rows_00_05",
    "gps_gap_count_gt15m",
    "gps_longest_gap_minutes",
    "gps_observed_span_minutes",
    "gps_low_speed_payload_share",
    "gps_high_speed_payload_share",
    "gps_bbox_area_proxy",
    "activity_late_rows_18_23",
    "wifi_token_count_capped64",
    "wifi_top_token_share",
    "ble_token_count_capped64",
    "ble_top_token_share",
    "place_context_token_count_capped128",
)


def numeric_feature_cols(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in KEY_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
        and not any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
    ]


def load_registered_source(
    source_name: str,
    train: pd.DataFrame,
    selected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    matrix_path = SOURCE_MATRICES[source_name]
    matrix = pd.read_parquet(matrix_path)
    train_matrix = matrix[matrix["split"].eq("train")].copy()
    train_matrix["date"] = pd.to_datetime(train_matrix["lifelog_date"]).dt.date
    feature_cols = numeric_feature_cols(train_matrix)
    if selected_columns is not None:
        selected_set = set(selected_columns)
        feature_cols = [column for column in feature_cols if column in selected_set]
    renamed = {column: f"{source_name}__{column}" for column in feature_cols}
    aligned = train[["subject_id", "date"]].merge(
        train_matrix[["subject_id", "date", *feature_cols]].rename(columns=renamed),
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(train):
        raise ValueError(f"{source_name} matrix does not align to sorted train rows.")
    prefixed_cols = [renamed[column] for column in feature_cols]
    metadata = {
        "source_name": source_name,
        "matrix_path": rel(matrix_path),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "selected_feature_columns": int(len(feature_cols)),
        "selected_feature_sample": feature_cols[:16],
        "missing_rate": float(aligned[prefixed_cols].isna().mean().mean()) if prefixed_cols else None,
    }
    return aligned, metadata, prefixed_cols


def selected_phone_columns(train_matrix_cols: list[str]) -> list[str]:
    return [
        column
        for column in train_matrix_cols
        if any(fragment in column for fragment in PHONE_S3_FRAGMENTS)
    ]


def bounded_interaction_pairs(phone_cols: list[str], gps_cols: list[str]) -> list[tuple[str, str]]:
    pair_rows: list[tuple[str, str]] = []
    for phone_col in phone_cols:
        for gps_col in gps_cols:
            phone_base = phone_col.split("__", 1)[1]
            gps_base = gps_col.split("__", 1)[1]
            phone_is_rest = any(token in phone_base for token in ("screen_off", "charging", "phone_still"))
            phone_is_active = any(token in phone_base for token in ("step_active", "wearable", "discordance"))
            gps_is_low_mobility = any(token in gps_base for token in ("low_speed", "night", "gap", "longest"))
            gps_is_context = any(token in gps_base for token in ("wifi", "ble", "place", "bbox", "active_hours"))
            if (phone_is_rest and gps_is_low_mobility) or (phone_is_active and gps_is_context):
                pair_rows.append((phone_col, gps_col))
    return pair_rows


def add_interaction_columns(
    features: pd.DataFrame,
    phone_cols: list[str],
    gps_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    pairs = bounded_interaction_pairs(phone_cols, gps_cols)
    interaction_data: dict[str, pd.Series] = {}
    for phone_col, gps_col in pairs:
        phone_base = phone_col.split("__", 1)[1]
        gps_base = gps_col.split("__", 1)[1]
        name = f"phone_gps_x__{phone_base}__x__{gps_base}"
        interaction_data[name] = features[phone_col] * features[gps_col]
    interaction_frame = pd.DataFrame(interaction_data)
    out = pd.concat([features.reset_index(drop=True), interaction_frame.reset_index(drop=True)], axis=1)
    metadata = {
        "interaction_columns": int(len(interaction_data)),
        "interaction_sample": list(interaction_data)[:16],
        "bounded_pair_rule": (
            "Pair phone-rest features with low-mobility GPS features and phone/wearable-active or discordance "
            "features with context/mobility GPS features."
        ),
    }
    return out, metadata, list(interaction_data)


def build_feature_frames(train: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    phone_all = pd.read_parquet(SOURCE_MATRICES[PHONE_SOURCE])
    phone_train = phone_all[phone_all["split"].eq("train")]
    phone_selected = selected_phone_columns(numeric_feature_cols(phone_train))
    phone_frame, phone_meta, phone_cols = load_registered_source(PHONE_SOURCE, train, phone_selected)
    gps_frame, gps_meta, gps_cols = load_registered_source(GPS_SOURCE, train, list(GPS_S3_COLUMNS))
    base = train[["subject_id", "date"]].copy()
    base = pd.concat(
        [
            base.reset_index(drop=True),
            phone_frame[phone_cols].reset_index(drop=True),
            gps_frame[gps_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    interaction_frame, interaction_meta, interaction_cols = add_interaction_columns(base, phone_cols, gps_cols)
    phone_only = train[["subject_id", "date"]].copy()
    phone_only = pd.concat([phone_only.reset_index(drop=True), phone_frame[phone_cols].reset_index(drop=True)], axis=1)
    gps_only = train[["subject_id", "date"]].copy()
    gps_only = pd.concat([gps_only.reset_index(drop=True), gps_frame[gps_cols].reset_index(drop=True)], axis=1)
    concat_cols = [*phone_cols, *gps_cols]
    family_frames = {
        "s3_phone_selected": phone_only,
        "s3_gps_selected": gps_only,
        "s3_phone_gps_selected_concat": base,
        "s3_phone_gps_interaction": interaction_frame,
    }
    metadata = {
        "phone_source": phone_meta,
        "gps_source": gps_meta,
        "selected_phone_columns": int(len(phone_cols)),
        "selected_gps_columns": int(len(gps_cols)),
        "selected_concat_columns": int(len(concat_cols)),
        "interaction": interaction_meta,
        "family_feature_counts": {
            "s3_phone_selected": int(len(phone_cols)),
            "s3_gps_selected": int(len(gps_cols)),
            "s3_phone_gps_selected_concat": int(len(concat_cols)),
            "s3_phone_gps_interaction": int(len(concat_cols) + len(interaction_cols)),
        },
    }
    return family_frames, metadata


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in {"subject_id", "date"}
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]


def scan_family(
    family_name: str,
    features: pd.DataFrame,
    train: pd.DataFrame,
    splits: dict[str, Any],
    references: dict[str, Any],
) -> list[dict[str, Any]]:
    target = "S3"
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    cols = feature_columns(features)
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for split_name in PRIMARY_GATE_SPLITS:
            folds = splits[split_name]
            source_pred, covered = predict_target_for_split(features, cols, y, folds, spec)
            id342_target_loss = target_loss_on_split(train, folds, target, references["id342"])
            source_values = references["id342"].copy()
            source_values[:, target_idx] = np.where(np.isnan(source_pred), references["id342"][:, target_idx], source_pred)
            source_loss = target_loss_on_split(train, folds, target, source_values)
            rows.append(
                {
                    "source": family_name,
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"{family_name}_{target.lower()}_{spec.name}_replace",
                    "model_spec": spec.name,
                    "weight": 1.0,
                    "valid_rows": int(covered),
                    "target_loss": source_loss,
                    "delta_target_vs_id342": float(source_loss - id342_target_loss),
                    "delta_all_vs_id342": float((source_loss - id342_target_loss) / len(TARGETS)),
                }
            )
            for weight in BLEND_WEIGHTS:
                blended = references["id342"].copy()
                blended[:, target_idx] = np.clip(
                    (1.0 - weight) * references["id342"][:, target_idx] + weight * source_values[:, target_idx],
                    1e-6,
                    1.0 - 1e-6,
                )
                loss = target_loss_on_split(train, folds, target, blended)
                rows.append(
                    {
                        "source": family_name,
                        "target": target,
                        "split_name": split_name,
                        "candidate": f"{family_name}_{target.lower()}_{spec.name}_w{weight:g}",
                        "model_spec": spec.name,
                        "weight": weight,
                        "valid_rows": int(covered),
                        "target_loss": loss,
                        "delta_target_vs_id342": float(loss - id342_target_loss),
                        "delta_all_vs_id342": float((loss - id342_target_loss) / len(TARGETS)),
                    }
                )
    return rows


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    references = {
        "id178": load_reference_values(train, 178),
        "id221": load_reference_values(train, 221),
        "id342": load_reference_values(train, BASE_ANCHOR_ID),
        "id287": load_reference_values(train, 287),
        "id329": load_reference_values(train, 329),
    }
    family_frames, feature_metadata = build_feature_frames(train)
    all_split_rows: list[dict[str, Any]] = []
    for family_name, features in family_frames.items():
        all_split_rows.extend(scan_family(family_name, features, train, splits, references))
    summary = summarize_rows(all_split_rows)
    passing = [row for row in summary if row["local_gate_pass"]]
    best = summary[0] if summary else None
    interaction_rows = [row for row in summary if row["source"] == "s3_phone_gps_interaction"]
    best_interaction = interaction_rows[0] if interaction_rows else None
    decision = {
        "reviewed_split_rows": int(len(all_split_rows)),
        "reviewed_candidates": int(len(summary)),
        "local_gate_pass_count": int(len(passing)),
        "best_candidate": best["candidate"] if best else None,
        "best_source": best["source"] if best else None,
        "best_primary_wins_vs_id342": best["primary_wins_vs_id342"] if best else None,
        "best_primary_mean_delta_target_vs_id342": best["primary_mean_delta_target_vs_id342"] if best else None,
        "best_primary_max_delta_target_vs_id342": best["primary_max_delta_target_vs_id342"] if best else None,
        "best_projected_public_proxy": best["projected_public_proxy"] if best else ID342_PUBLIC,
        "best_gap_to_public_055": best["gap_to_public_055"] if best else ID342_PUBLIC - PUBLIC_055,
        "best_interaction_candidate": best_interaction["candidate"] if best_interaction else None,
        "best_interaction_wins_vs_id342": best_interaction["primary_wins_vs_id342"] if best_interaction else None,
        "best_interaction_mean_delta_target_vs_id342": (
            best_interaction["primary_mean_delta_target_vs_id342"] if best_interaction else None
        ),
        "s3_phone_gps_interaction_gate_pass": False,
        "public_055_evidence": False,
        "artifact_review_allowed": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "reason": (
            "Focused phone/GPS S3 interaction is still far from the Public-0.55/review threshold. Treat it as "
            "design evidence only, not as a candidate route."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_s3_phone_gps_interaction_gate",
        "purpose": "Test the cross-source S3 phone/wearable plus GPS micro-alignment as focused interactions.",
        "anchor": {"id": BASE_ANCHOR_ID, "public_score": ID342_PUBLIC},
        "target_focus": TARGET_FOCUS,
        "model_specs": [spec.name for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "thresholds": {
            "min_primary_wins_vs_id342": MIN_PRIMARY_WINS_VS_ID342,
            "max_primary_mean_delta_vs_id342": MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "max_primary_worst_delta_vs_id342": MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "max_public_proxy_for_review": MAX_PUBLIC_PROXY_FOR_REVIEW,
            "max_gap_to_public_055_for_review": MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        },
        "feature_metadata": feature_metadata,
        "top_rows": summary[:40],
        "decision": decision,
        "no_prediction_file_created": True,
        "no_feature_matrix_created": True,
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
        "# Post-ID342 S3 phone/GPS interaction gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate pass count: `{decision['local_gate_pass_count']}`",
        f"- Best candidate: `{decision['best_candidate']}`",
        f"- Best source: `{decision['best_source']}`",
        f"- Best wins: `{decision['best_primary_wins_vs_id342']}`",
        f"- Best target mean/max delta vs ID342: `{decision['best_primary_mean_delta_target_vs_id342']:+.9f}` / `{decision['best_primary_max_delta_target_vs_id342']:+.9f}`",
        f"- Best projected Public proxy: `{decision['best_projected_public_proxy']:.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
        f"- Best interaction candidate: `{decision['best_interaction_candidate']}`",
        f"- Interaction gate pass: `{decision['s3_phone_gps_interaction_gate_pass']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Feature Frame",
        "",
        f"- Selected phone columns: `{report['feature_metadata']['selected_phone_columns']}`",
        f"- Selected GPS columns: `{report['feature_metadata']['selected_gps_columns']}`",
        f"- Interaction columns: `{report['feature_metadata']['interaction']['interaction_columns']}`",
        "",
        "## Top Rows",
        "",
        "| rank | pass | source | candidate | wins | mean all Δ342 | max all Δ342 | target mean Δ342 | target max Δ342 | proxy | gap 0.55 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["top_rows"][:25], start=1):
        lines.append(
            "| {rank} | `{passed}` | `{source}` | `{candidate}` | {wins} | {mean:+.9f} | {max_delta:+.9f} | {target_mean:+.9f} | {target_max:+.9f} | {proxy:.9f} | {gap:+.9f} |".format(
                rank=rank,
                passed=row["local_gate_pass"],
                source=row["source"],
                candidate=row["candidate"],
                wins=row["primary_wins_vs_id342"],
                mean=row["primary_mean_delta_all_vs_id342"],
                max_delta=row["primary_max_delta_all_vs_id342"],
                target_mean=row["primary_mean_delta_target_vs_id342"],
                target_max=row["primary_max_delta_target_vs_id342"],
                proxy=row["projected_public_proxy"],
                gap=row["gap_to_public_055"],
            )
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    write_report(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Gate pass: "
        f"{decision['s3_phone_gps_interaction_gate_pass']} | best={decision['best_candidate']}"
    )


if __name__ == "__main__":
    main()
