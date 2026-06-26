#!/usr/bin/env python3
"""Post-ID342 final-day independent signal sweep.

This report-only diagnostic implements the last-day plan after all registered
post-ID342 routes closed. It tests three small, independent-source hypotheses:

1. S3 phone/GPS temporal deviations from registered processed matrices.
2. S4 sleep-fragmentation x solar/calendar joint state.
3. Q3/S3/S4 compact joint-state features built from the above families.

It creates no feature matrix artifact, predictions, submission CSV,
base-prediction artifact, notebook-history artifact, submission-log row, Kaggle
push/run, or DACON submission.
"""

# ruff: noqa: E402

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

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import TARGETS  # noqa: E402
from scripts.diagnose_post_id342_processed_sleep_proxy_gate import (  # noqa: E402
    ANCHOR_ID,
    EPS,
    ID342_PUBLIC,
    MODEL_SPECS,
    PRIMARY_SPLITS,
    PUBLIC_055,
    TARGET_MAX_DELTA_CEILING,
    TARGET_MEAN_DELTA_CEILING,
    TARGET_WIN_FLOOR,
    load_processed_aligned,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_split,
    rel,
    weighted_loss,
)
from scripts.diagnose_post_id342_s3_phone_gps_interaction_gate import (  # noqa: E402
    build_feature_frames as build_phone_gps_frames,
)
from scripts.diagnose_post_id342_s3_phone_gps_interaction_gate import (
    feature_columns as s3_feature_columns,
)
from scripts.diagnose_post_id342_s4s3_sleep_fragmentation_gate import build_s4s3_families  # noqa: E402
from scripts.diagnose_post_id342_solar_seasonal_holiday_prior_gate import (  # noqa: E402
    build_feature_frame as build_solar_feature_frame,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_final_day_independent_signal_sweep_20260625.json"

FOCUS_TARGETS = ("Q3", "S3", "S4")
BLEND_WEIGHTS = (0.005, 0.01, 0.025, 0.05)
FAST_MODEL_SPECS = tuple(spec for spec in MODEL_SPECS if spec.kind == "lr")
REVIEW_PUBLIC_THRESHOLD = 0.58
DACON_PUBLIC_TARGET = 0.55
MAX_FAILED_DIRECTION_CORR = 0.20
FAILED_FRONTIER_IDS = (287, 329)

S4_SLEEP_SOLAR_COLUMNS = (
    "solar_daylight_hours",
    "solar_night_hours",
    "solar_daylight_change_1d",
    "solar_sunrise_minutes",
    "solar_sunset_minutes",
    "season_sin",
    "season_cos",
    "is_weekend",
    "near_korean_holiday_7d",
    "days_to_nearest_korean_holiday_abs",
)


def numeric_cols(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in {"subject_id", "date", "lifelog_date", "sleep_date", "split", "is_test"}
        and pd.api.types.is_numeric_dtype(frame[col])
        and frame[col].notna().any()
    ]


def add_past_temporal_features(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.sort_values(["subject_id", "date"]).reset_index(drop=True).copy()
    added: dict[str, pd.Series] = {}
    for col in feature_cols:
        lag = pd.Series(np.nan, index=out.index, dtype=float)
        past_mean = pd.Series(np.nan, index=out.index, dtype=float)
        for _subject_id, group in out.groupby("subject_id", sort=False):
            shifted = group[col].shift(1)
            lag.loc[group.index] = shifted
            past_mean.loc[group.index] = shifted.rolling(3, min_periods=1).mean()
        added[f"{col}__lag1"] = lag
        added[f"{col}__past3_mean"] = past_mean
        added[f"{col}__dev_past3"] = out[col] - past_mean
    temporal = pd.concat([out, pd.DataFrame(added)], axis=1)
    added_cols = list(added)
    metadata = {
        "base_columns": int(len(feature_cols)),
        "temporal_columns_added": int(len(added_cols)),
        "temporal_rule": "Per-subject lag1, previous-3-row mean, and current-minus-previous-3-row deviation.",
        "mean_missing_rate": float(temporal[feature_cols + added_cols].isna().mean().mean()),
    }
    return temporal, metadata


def selected_sleep_columns(processed: pd.DataFrame) -> list[str]:
    families, _metadata = build_s4s3_families(processed)
    selected = []
    for family in ("s4_waso_fragmentation_core", "sleep_hr_awake_light", "s3_onset_context"):
        selected.extend(families.get(family, [])[:18])
    seen = set()
    out = []
    for col in selected:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out[:48]


def build_sleep_solar_joint(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed, processed_metadata = load_processed_aligned(train)
    sleep_cols = selected_sleep_columns(processed)
    solar_frame, solar_metadata, _families = build_solar_feature_frame(train[["subject_id", "date"]])
    solar_cols = [col for col in S4_SLEEP_SOLAR_COLUMNS if col in solar_frame.columns]
    base = train[["subject_id", "date"]].copy()
    base = pd.concat(
        [
            base.reset_index(drop=True),
            processed[sleep_cols].reset_index(drop=True),
            solar_frame[solar_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    interaction_data: dict[str, pd.Series] = {}
    for sleep_col in sleep_cols[:16]:
        for solar_col in solar_cols:
            interaction_data[f"sleep_solar_x__{sleep_col}__x__{solar_col}"] = base[sleep_col] * base[solar_col]
    interaction_frame = pd.DataFrame(interaction_data)
    out = pd.concat([base.reset_index(drop=True), interaction_frame.reset_index(drop=True)], axis=1)
    metadata = {
        "processed_source": processed_metadata,
        "solar_source": solar_metadata,
        "sleep_columns": int(len(sleep_cols)),
        "solar_columns": int(len(solar_cols)),
        "interaction_columns": int(len(interaction_data)),
        "feature_count": int(len(numeric_cols(out))),
        "mean_missing_rate": float(out[numeric_cols(out)].isna().mean().mean()),
    }
    return out, metadata


def build_phone_gps_temporal(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    phone_gps_frames, source_metadata = build_phone_gps_frames(train)
    concat = phone_gps_frames["s3_phone_gps_selected_concat"]
    feature_cols = s3_feature_columns(concat)
    temporal, temporal_metadata = add_past_temporal_features(concat, feature_cols)
    metadata = {
        "source_metadata": source_metadata,
        "temporal_metadata": temporal_metadata,
        "feature_count": int(len(numeric_cols(temporal))),
    }
    return temporal, metadata


def build_joint_compact(
    train: pd.DataFrame,
    phone_gps_temporal: pd.DataFrame,
    sleep_solar_joint: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phone_cols = [col for col in numeric_cols(phone_gps_temporal) if "__dev_past3" in col or "__lag1" in col]
    phone_cols = phone_cols[:80]
    sleep_cols = [col for col in numeric_cols(sleep_solar_joint) if col.startswith("sleep_solar_x__")]
    sleep_cols = sleep_cols[:80]
    direct_sleep_solar = [
        col for col in numeric_cols(sleep_solar_joint) if not col.startswith("sleep_solar_x__")
    ][:80]
    out = train[["subject_id", "date"]].copy()
    out = pd.concat(
        [
            out.reset_index(drop=True),
            phone_gps_temporal[phone_cols].reset_index(drop=True),
            sleep_solar_joint[[*direct_sleep_solar, *sleep_cols]].reset_index(drop=True),
        ],
        axis=1,
    )
    metadata = {
        "phone_temporal_columns": int(len(phone_cols)),
        "sleep_solar_direct_columns": int(len(direct_sleep_solar)),
        "sleep_solar_interaction_columns": int(len(sleep_cols)),
        "feature_count": int(len(numeric_cols(out))),
        "mean_missing_rate": float(out[numeric_cols(out)].isna().mean().mean()),
    }
    return out, metadata


def movement_corr(candidate: np.ndarray, anchor: np.ndarray, failed: np.ndarray) -> float | None:
    movement = candidate - anchor
    failed_movement = failed - anchor
    if float(np.nanstd(movement)) <= 1e-12 or float(np.nanstd(failed_movement)) <= 1e-12:
        return None
    corr = np.corrcoef(movement, failed_movement)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def scan_family(
    family_name: str,
    features: pd.DataFrame,
    targets: tuple[str, ...],
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    failed_refs: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    feature_cols = numeric_cols(features)
    rows: list[dict[str, Any]] = []
    for target in targets:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in FAST_MODEL_SPECS:
            split_predictions: dict[str, np.ndarray] = {}
            split_coverage: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_split(features, feature_cols, y, splits[split_name], spec)
                split_predictions[split_name] = pred
                split_coverage[split_name] = int(covered)
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_failed_corr: dict[str, dict[str, float | None]] = {}
                wins = 0
                max_failed_corr = -1.0
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    blended = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * split_predictions[split_name],
                        EPS,
                        1.0 - EPS,
                    )
                    anchor_loss = weighted_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                    corr_by_id: dict[str, float | None] = {}
                    for failed_id, failed_values in failed_refs.items():
                        corr = movement_corr(blended, anchor_values[:, target_idx], failed_values[:, target_idx])
                        corr_by_id[f"id{failed_id}"] = corr
                        if corr is not None:
                            max_failed_corr = max(max_failed_corr, corr)
                    split_failed_corr[split_name] = corr_by_id
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                projected_public_proxy = float(ID342_PUBLIC + mean_delta / len(TARGETS))
                local_gate_pass = bool(
                    wins >= TARGET_WIN_FLOOR
                    and mean_delta <= TARGET_MEAN_DELTA_CEILING
                    and max_delta <= TARGET_MAX_DELTA_CEILING
                    and max_failed_corr <= MAX_FAILED_DIRECTION_CORR
                )
                review_gate_pass = bool(local_gate_pass and projected_public_proxy <= REVIEW_PUBLIC_THRESHOLD)
                dacon_gate_pass = bool(local_gate_pass and projected_public_proxy <= DACON_PUBLIC_TARGET)
                rows.append(
                    {
                        "family_name": family_name,
                        "target": target,
                        "config": f"{family_name}_{target.lower()}_{spec.name}_w{weight:g}",
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_rows_by_split": split_coverage,
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "projected_public_proxy": projected_public_proxy,
                        "gap_to_public_055": float(projected_public_proxy - PUBLIC_055),
                        "split_deltas": split_deltas,
                        "failed_direction_corr": split_failed_corr,
                        "max_failed_direction_corr": None if max_failed_corr < -0.5 else float(max_failed_corr),
                        "local_gate_pass": local_gate_pass,
                        "review_gate_pass": review_gate_pass,
                        "dacon_gate_pass": dacon_gate_pass,
                    }
                )
    return rows


def summarize_targets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in FOCUS_TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        best = target_rows[0] if target_rows else None
        out[target] = {
            "best_row": best,
            "local_gate_pass_count": int(sum(row["local_gate_pass"] for row in target_rows)),
            "review_gate_pass_count": int(sum(row["review_gate_pass"] for row in target_rows)),
            "dacon_gate_pass_count": int(sum(row["dacon_gate_pass"] for row in target_rows)),
        }
    return out


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    failed_refs = {failed_id: load_reference_values(train, failed_id) for failed_id in FAILED_FRONTIER_IDS}

    phone_gps_temporal, phone_gps_metadata = build_phone_gps_temporal(train)
    sleep_solar_joint, sleep_solar_metadata = build_sleep_solar_joint(train)
    joint_compact, joint_metadata = build_joint_compact(train, phone_gps_temporal, sleep_solar_joint)

    family_specs = {
        "s3_phone_gps_temporal_deviation": {
            "frame": phone_gps_temporal,
            "targets": ("S3",),
            "metadata": phone_gps_metadata,
        },
        "s4_sleep_solar_joint": {
            "frame": sleep_solar_joint,
            "targets": ("S4",),
            "metadata": sleep_solar_metadata,
        },
        "q3s3s4_compact_joint_state": {
            "frame": joint_compact,
            "targets": FOCUS_TARGETS,
            "metadata": joint_metadata,
        },
    }
    rows: list[dict[str, Any]] = []
    for family_name, spec in family_specs.items():
        rows.extend(
            scan_family(
                family_name=family_name,
                features=spec["frame"],
                targets=spec["targets"],
                train=train,
                splits=splits,
                anchor_values=anchor_values,
                failed_refs=failed_refs,
            )
        )
    rows = sorted(
        rows,
        key=lambda row: (
            not row["dacon_gate_pass"],
            not row["review_gate_pass"],
            not row["local_gate_pass"],
            row["projected_public_proxy"],
            row["mean_delta"],
            row["max_delta"],
        ),
    )
    local_pass = [row for row in rows if row["local_gate_pass"]]
    review_pass = [row for row in rows if row["review_gate_pass"]]
    dacon_pass = [row for row in rows if row["dacon_gate_pass"]]
    best = rows[0] if rows else None
    decision = {
        "reviewed_candidates": int(len(rows)),
        "local_gate_pass_count": int(len(local_pass)),
        "review_gate_pass_count": int(len(review_pass)),
        "dacon_gate_pass_count": int(len(dacon_pass)),
        "best_candidate": best["config"] if best else None,
        "best_family": best["family_name"] if best else None,
        "best_target": best["target"] if best else None,
        "best_wins": best["wins"] if best else None,
        "best_mean_delta": best["mean_delta"] if best else None,
        "best_max_delta": best["max_delta"] if best else None,
        "best_projected_public_proxy": best["projected_public_proxy"] if best else ID342_PUBLIC,
        "best_gap_to_public_055": best["gap_to_public_055"] if best else ID342_PUBLIC - PUBLIC_055,
        "target_summary": summarize_targets(rows),
        "final_day_independent_signal_gate_pass": bool(review_pass),
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "dacon_submit_meaningful_now": False,
        "next_action": (
            "Promote nothing unless a future row clears the Public-0.58 review proxy. Current final-day local "
            "signals remain too small for artifact creation, Kaggle escalation, or DACON submission."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_final_day_independent_signal_sweep",
        "purpose": "Implement the final-day independent-source sweep before any submission-spend decision.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "focus_targets": list(FOCUS_TARGETS),
            "blend_weights": list(BLEND_WEIGHTS),
            "model_specs": [asdict(spec) for spec in FAST_MODEL_SPECS],
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "max_failed_direction_corr": MAX_FAILED_DIRECTION_CORR,
            "review_public_threshold": REVIEW_PUBLIC_THRESHOLD,
            "dacon_public_target": DACON_PUBLIC_TARGET,
            "failed_frontier_ids": list(FAILED_FRONTIER_IDS),
        },
        "feature_families": {
            family_name: {
                "targets": list(spec["targets"]),
                "feature_count": int(len(numeric_cols(spec["frame"]))),
                "metadata": spec["metadata"],
            }
            for family_name, spec in family_specs.items()
        },
        "top_rows": rows[:40],
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
        "# Post-ID342 final-day independent signal sweep",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate passes: `{decision['local_gate_pass_count']}`",
        f"- Review gate passes: `{decision['review_gate_pass_count']}`",
        f"- DACON gate passes: `{decision['dacon_gate_pass_count']}`",
        f"- Best candidate: `{decision['best_candidate']}`",
        f"- Best target/family: `{decision['best_target']}` / `{decision['best_family']}`",
        f"- Best wins: `{decision['best_wins']}`",
        f"- Best mean/max delta: `{decision['best_mean_delta']:+.9f}` / `{decision['best_max_delta']:+.9f}`",
        f"- Best projected Public proxy: `{decision['best_projected_public_proxy']:.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
        f"- Gate pass: `{decision['final_day_independent_signal_gate_pass']}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Feature Families",
        "",
        "| family | targets | features | note |",
        "|---|---|---:|---|",
    ]
    for family_name, family in report["feature_families"].items():
        note = ""
        metadata = family["metadata"]
        if family_name == "s3_phone_gps_temporal_deviation":
            note = metadata["temporal_metadata"]["temporal_rule"]
        elif family_name == "s4_sleep_solar_joint":
            note = "Processed sleep fragmentation/onset features crossed with solar/calendar priors."
        else:
            note = "Compact union of phone/GPS temporal and sleep/solar state columns."
        lines.append(f"| `{family_name}` | `{family['targets']}` | {family['feature_count']} | {note} |")
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| rank | local | review | dacon | family | target | config | wins | mean delta | max delta | proxy | gap 0.55 | max failed corr |",
            "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(report["top_rows"][:25], start=1):
        max_failed_corr = row["max_failed_direction_corr"]
        corr_text = "" if max_failed_corr is None else f"{max_failed_corr:+.6f}"
        lines.append(
            f"| {rank} | `{row['local_gate_pass']}` | `{row['review_gate_pass']}` | `{row['dacon_gate_pass']}` | "
            f"`{row['family_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
            f"{row['mean_delta']:+.9f} | {row['max_delta']:+.9f} | {row['projected_public_proxy']:.9f} | "
            f"{row['gap_to_public_055']:+.9f} | {corr_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Review gate pass: "
        f"{decision['final_day_independent_signal_gate_pass']} | best={decision['best_candidate']}"
    )


if __name__ == "__main__":
    main()
