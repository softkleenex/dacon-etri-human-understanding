#!/usr/bin/env python3
"""Report-only Q3/S3 cross-modal disagreement smoke after local lane closure.

This diagnostic tests a narrow feature family that is intentionally different
from the already closed single-lane probes.  It asks whether phone/watch signals
disagree around the sleep boundary: screen/app/activity load versus darkness,
charging, stillness, and sleep physiology.

It never creates full-test predictions, base prediction artifacts, submission
CSVs, notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or
DACON submissions.
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
    ANCHOR_IDS,
    BLEND_WEIGHTS,
    CONFIG_MAX_DELTA_CEILING,
    CONFIG_MEAN_DELTA_CEILING,
    CONFIG_WIN_FLOOR,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_cross_modal_disagreement_smoke_20260618.json"
TARGET_FOCUS = ["Q3", "S3"]
EPS = 1e-6

RATIO_GROUPS = {
    "screen": [
        "screen_eve_ratio",
        "screen_night_ratio",
        "screen_presleep_ratio",
        "slp_screen_ratio",
    ],
    "dark": [
        "mlight_dark_ratio",
        "wlight_dark_ratio",
        "slp_wlight_dark",
    ],
    "active": [
        "act_active_ratio",
        "act_presleep_active",
        "slp_act_active_ratio",
        "slp_pedo_active",
    ],
    "still": [
        "act_still_ratio",
        "act_night_still",
        "slp_act_still_ratio",
    ],
    "charge": [
        "ac_night_charging",
        "ac_presleep_charging",
    ],
    "usage": [
        "mUsage_night_time_ratio",
        "mUsage_dopamine_ratio",
        "usage_late_ratio",
        "usage_eve_ratio",
    ],
    "hr_state": [
        "slp_hr_awake_ratio",
        "slp_hr_deep_ratio",
        "slp_hr_light_ratio",
    ],
}

CONTINUOUS_FEATURES = [
    "slp_hr_rmssd",
    "slp_hr_range",
    "slp_hr_spike_count",
    "slp_pedo_steps",
    "slp_pedo_mid_steps",
    "pedo_evening_steps",
    "usage_late_time",
    "usage_eve_time",
    "mUsage_total_time_mins",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_processed_frame(path: Path, is_test: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["lifelog_date"]).dt.date
    frame["is_test"] = is_test
    keep = ["subject_id", "date", "is_test"]
    for columns in RATIO_GROUPS.values():
        keep.extend(columns)
    keep.extend(CONTINUOUS_FEATURES)
    keep = [col for col in dict.fromkeys(keep) if col in frame.columns]
    return frame[keep].copy()


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").astype(float)


def mean_existing(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [numeric(frame, col) for col in cols if col in frame.columns]
    if not existing:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(existing, axis=1).mean(axis=1)


def build_cross_modal_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    train = read_processed_frame(PROCESSED_DIR / "df_train_raw.parquet", is_test=0)
    test = read_processed_frame(PROCESSED_DIR / "df_test_raw.parquet", is_test=1)
    frame = pd.concat([train, test], ignore_index=True, sort=False)

    group_means = {name: mean_existing(frame, cols) for name, cols in RATIO_GROUPS.items()}
    features = frame[["subject_id", "date", "is_test"]].copy()
    for name, values in group_means.items():
        features[f"xmod_{name}_mean"] = values

    screen = group_means["screen"]
    dark = group_means["dark"]
    active = group_means["active"]
    still = group_means["still"]
    charge = group_means["charge"]
    usage = group_means["usage"]
    hr_state = group_means["hr_state"]

    features["xmod_screen_minus_dark"] = screen - dark
    features["xmod_screen_minus_still"] = screen - still
    features["xmod_load_minus_rest"] = (screen + active + usage) / 3.0 - (dark + still + charge) / 3.0
    features["xmod_dark_screen_product"] = dark * screen
    features["xmod_dark_active_product"] = dark * active
    features["xmod_usage_active_product"] = usage * active
    features["xmod_quiet_dark_charge_product"] = dark * still * charge
    features["xmod_dark_without_charge"] = dark * (1.0 - charge)
    features["xmod_late_load_without_charge"] = (screen + active + usage) / 3.0 * (1.0 - charge)
    features["xmod_sleep_boundary_conflict"] = screen + active + usage + hr_state - dark - still - charge
    features["xmod_hr_rest_conflict"] = hr_state + active - still - charge

    pair_specs = [
        ("phone_watch_dark_absdiff", "mlight_dark_ratio", "wlight_dark_ratio"),
        ("phone_sleep_dark_absdiff", "mlight_dark_ratio", "slp_wlight_dark"),
        ("watch_sleep_dark_absdiff", "wlight_dark_ratio", "slp_wlight_dark"),
        ("screen_usage_absdiff", "screen_presleep_ratio", "usage_late_ratio"),
        ("screen_activity_absdiff", "screen_presleep_ratio", "act_presleep_active"),
        ("sleep_screen_activity_absdiff", "slp_screen_ratio", "slp_act_active_ratio"),
        ("sleep_screen_dark_absdiff", "slp_screen_ratio", "slp_wlight_dark"),
        ("sleep_active_dark_absdiff", "slp_act_active_ratio", "slp_wlight_dark"),
        ("charge_screen_absdiff", "ac_presleep_charging", "screen_presleep_ratio"),
        ("charge_dark_absdiff", "ac_presleep_charging", "slp_wlight_dark"),
        ("hr_awake_screen_absdiff", "slp_hr_awake_ratio", "slp_screen_ratio"),
        ("hr_awake_active_absdiff", "slp_hr_awake_ratio", "slp_act_active_ratio"),
    ]
    for out_name, left, right in pair_specs:
        features[f"xmod_{out_name}"] = (numeric(frame, left) - numeric(frame, right)).abs()

    for col in CONTINUOUS_FEATURES:
        if col in frame.columns:
            features[f"xmod_log1p_{col}"] = np.log1p(numeric(frame, col).clip(lower=0.0))

    ratio_cols = [col for cols in RATIO_GROUPS.values() for col in cols if col in frame.columns]
    cont_cols = [col for col in CONTINUOUS_FEATURES if col in frame.columns]
    observed = frame[ratio_cols + cont_cols].notna()
    features["xmod_observed_signal_count"] = observed.sum(axis=1).astype(float)
    features["xmod_missing_signal_count"] = (~observed).sum(axis=1).astype(float)
    features["xmod_missing_signal_ratio"] = features["xmod_missing_signal_count"] / max(len(ratio_cols + cont_cols), 1)

    raw_feature_cols = [col for col in features.columns if col.startswith("xmod_")]
    missing_indicators = features[raw_feature_cols].isna().astype(float).add_suffix("_missing")
    features = pd.concat([features, missing_indicators], axis=1).copy()
    feature_cols = [col for col in features.columns if col.startswith("xmod_")]

    coverage = {
        "processed_train": rel(PROCESSED_DIR / "df_train_raw.parquet"),
        "processed_test": rel(PROCESSED_DIR / "df_test_raw.parquet"),
        "n_rows": int(len(features)),
        "n_train_rows": int(features["is_test"].eq(0).sum()),
        "n_test_rows": int(features["is_test"].eq(1).sum()),
        "raw_feature_columns": int(len(raw_feature_cols)),
        "feature_columns": int(len(feature_cols)),
        "ratio_columns_used": ratio_cols,
        "continuous_columns_used": cont_cols,
        "train_any_nonnull_row_share": float(
            features.loc[features["is_test"].eq(0), raw_feature_cols].notna().any(axis=1).mean()
        ),
        "test_any_nonnull_row_share": float(
            features.loc[features["is_test"].eq(1), raw_feature_cols].notna().any(axis=1).mean()
        ),
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), raw_feature_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), raw_feature_cols].isna().mean().mean()),
    }
    return features, coverage


def align_train_features(train: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train_features = features[features["is_test"].eq(0)].copy()
    aligned = train[["subject_id", "date"]].merge(
        train_features.drop(columns=["is_test"]), on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    if len(aligned) != len(train):
        raise ValueError("Cross-modal disagreement features do not align with sorted train rows.")
    return aligned


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features.columns
        if col.startswith("xmod_") and pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()
    ]


def scan_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    target: str,
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for split_name in PRIMARY_GATE_SPLITS:
            folds = splits[split_name]
            pred, covered = predict_target_for_split(features, cols, y, folds, spec)
            source_values_by_anchor = {
                anchor_name: anchor_values.copy() for anchor_name, anchor_values in references.items()
            }
            for anchor_name, values in source_values_by_anchor.items():
                values[:, target_idx] = np.where(np.isnan(pred), references[anchor_name][:, target_idx], pred)
            id178_loss = target_loss_on_split(train, folds, target, references["id178"])
            id341_loss = target_loss_on_split(train, folds, target, references["id341"])
            source_loss = target_loss_on_split(train, folds, target, source_values_by_anchor["id341"])
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"cross_modal_disagreement_{spec.name}",
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
                source_values = source_values_by_anchor[anchor_name]
                for weight in BLEND_WEIGHTS:
                    blended = anchor_values.copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                        EPS,
                        1 - EPS,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    anchor_loss = id178_loss if anchor_name == "id178" else id341_loss
                    rows.append(
                        {
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"blend_{anchor_name}_cross_modal_disagreement_{spec.name}_w{weight:g}",
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
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: dict[str, list[dict[str, Any]]] = {}
    for target in TARGET_FOCUS:
        summaries = []
        target_frame = frame[frame["target"].eq(target)].copy()
        for candidate, group in target_frame.groupby("candidate", sort=False):
            deltas = group.set_index("split_name")["delta_vs_id341"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            values = [float(v) for v in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "candidate": str(candidate),
                    "model_spec": row0.get("model_spec"),
                    "anchor": row0.get("anchor"),
                    "weight": row0.get("weight"),
                    "primary_wins_vs_id341": int(sum(value < 0 for value in values)),
                    "mean_delta_vs_id341": float(np.mean(values)),
                    "min_delta_vs_id341": float(np.min(values)),
                    "max_delta_vs_id341": float(np.max(values)),
                    "split_deltas_vs_id341": {str(k): float(v) for k, v in deltas.items()},
                    "split_losses": {str(k): float(v) for k, v in losses.items()},
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


def select_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
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


def split_anchor_table(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        for split_name in PRIMARY_GATE_SPLITS:
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "id178_loss": target_loss_on_split(train, splits[split_name], target, references["id178"]),
                    "id341_loss": target_loss_on_split(train, splits[split_name], target, references["id341"]),
                }
            )
    return rows


def build_report() -> dict[str, Any]:
    train = load_train_sorted()
    raw_features, coverage = build_cross_modal_features()
    features = align_train_features(train, raw_features)
    cols = feature_columns(features)
    splits = load_split_manifest()
    references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(scan_target(train, features, cols, splits, references, target))
    summaries = summarize_rows(scan_rows)
    selected = select_configs(summaries)
    selected_targets = sorted(selected.keys())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_cross_modal_disagreement_smoke",
        "purpose": (
            "Test phone/watch cross-modal disagreement features for Q3/S3 after single-lane Q3/S3 local probes "
            "closed. This is a scout report and is not yet part of the authoritative negative-evidence map."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "processed_train": coverage["processed_train"],
            "processed_test": coverage["processed_test"],
            "split_manifest": rel(REPORTS_DIR / "id340_validation_split_manifest_20260618.json"),
            "id178_oof": rel(find_oof(178)),
            "id341_oof": rel(find_oof(341)),
        },
        "feature_coverage": coverage,
        "config": {
            "ratio_groups": RATIO_GROUPS,
            "continuous_features": CONTINUOUS_FEATURES,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "primary_gate_splits": PRIMARY_GATE_SPLITS,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "anchor_rows": split_anchor_table(train, splits, references),
        "top_rows_by_target": {target: rows[:8] for target, rows in summaries.items()},
        "selected_configs": selected,
        "decision": {
            "model_smoke_completed": True,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "q3_signal_found": "Q3" in selected,
            "s3_signal_found": "S3" in selected,
            "all_q3s3_targets_covered": selected_targets == TARGET_FOCUS,
            "public_058_evidence": False,
            "cross_modal_artifact_allowed": False,
            "negative_map_extension_recommended": not bool(selected_targets),
            "next_local_allowed_step": (
                "combined_no_artifact_geometry_gate" if selected_targets else "extend_negative_map_as_closed_scout"
            ),
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If selected targets exist, keep this as report-only context and run only a combined no-artifact "
                "geometry gate. If no selected targets exist, add it to the Q3/S3 negative map as a closed scout."
            ),
        },
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    coverage = report["feature_coverage"]
    lines = [
        "# Q3/S3 cross-modal disagreement smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It uses materialized processed features to test disagreement between phone/watch signals around the sleep boundary and creates no submission artifacts.",
        "",
        "## Decision",
        "",
        f"- Model smoke completed: `{decision['model_smoke_completed']}`",
        f"- Selected targets: `{decision['selected_targets']}`",
        f"- Q3 signal found: `{decision['q3_signal_found']}`",
        f"- S3 signal found: `{decision['s3_signal_found']}`",
        f"- All Q3/S3 targets covered: `{decision['all_q3s3_targets_covered']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Cross-modal artifact allowed: `{decision['cross_modal_artifact_allowed']}`",
        f"- Negative-map extension recommended: `{decision['negative_map_extension_recommended']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` train/test = `{coverage['n_train_rows']}/{coverage['n_test_rows']}`",
        f"- Raw feature columns: `{coverage['raw_feature_columns']}`",
        f"- Feature columns incl. missing indicators: `{coverage['feature_columns']}`",
        f"- Ratio columns used: `{len(coverage['ratio_columns_used'])}`",
        f"- Continuous columns used: `{len(coverage['continuous_columns_used'])}`",
        f"- Train any non-null row share: `{coverage['train_any_nonnull_row_share']:.3f}`",
        f"- Test any non-null row share: `{coverage['test_any_nonnull_row_share']:.3f}`",
        f"- Train missing mean: `{coverage['train_missing_mean']:.4f}`",
        f"- Test missing mean: `{coverage['test_missing_mean']:.4f}`",
        "",
        "## Top rows by target",
        "",
    ]
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [f"### {target}", "", "| candidate | wins | mean Δ vs ID341 | max Δ vs ID341 |", "|---|---:|---:|---:|"]
        )
        for row in rows[:6]:
            lines.append(
                f"| `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    lines.extend(["## Next action", "", decision["next_action"], ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    write_report(report, output)
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
