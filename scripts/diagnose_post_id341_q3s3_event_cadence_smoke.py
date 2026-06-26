#!/usr/bin/env python3
"""Report-only Q3/S3 raw event-cadence smoke after local lane closure.

This diagnostic tests a deliberately narrow source family that is different
from value/state transition features: per-day raw timestamp cadence, silence
windows, and inter-event gaps across phone/watch modalities. It is intended as
a no-submit probe for whether Q3 stress or S3 sleep-onset latency has any
remaining local signal before consuming Kaggle approval.

It never creates full-test predictions, base prediction artifacts, submission
CSVs, notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or
DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import math
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
RAW_DIR = ROOT / "data" / "raw"
ITEMS_DIR = RAW_DIR / "ch2025_data_items"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_event_cadence_smoke_20260618.json"
TARGET_FOCUS = ["Q3", "S3"]
ANCHOR_IDS = [178, 341]
TIME_BINS = {
    "afternoon_12_18": (12.0, 18.0),
    "evening_18_22": (18.0, 22.0),
    "late_22_24": (22.0, 24.0),
    "onset_24_27": (24.0, 27.0),
    "deep_27_31": (27.0, 31.0),
    "morning_31_36": (31.0, 36.0),
}
MODALITIES = {
    "screen": "ch2025_mScreenStatus.parquet",
    "activity": "ch2025_mActivity.parquet",
    "charge": "ch2025_mACStatus.parquet",
    "mlight": "ch2025_mLight.parquet",
    "wlight": "ch2025_wLight.parquet",
    "pedo": "ch2025_wPedo.parquet",
    "hr": "ch2025_wHr.parquet",
    "usage": "ch2025_mUsageStats.parquet",
    "wifi": "ch2025_mWifi.parquet",
    "ble": "ch2025_mBle.parquet",
    "gps": "ch2025_mGps.parquet",
    "ambience": "ch2025_mAmbience.parquet",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def time_bin(rel_hour: float) -> str:
    for name, (start, end) in TIME_BINS.items():
        if start <= rel_hour < end:
            return name
    return "outside"


def key_frame() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv", usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv", usecols=["subject_id", "lifelog_date"])
    train = train.rename(columns={"lifelog_date": "row_date"}).assign(is_test=0)
    test = test.rename(columns={"lifelog_date": "row_date"}).assign(is_test=1)
    keys = pd.concat([train, test], ignore_index=True)
    keys["subject_id"] = keys["subject_id"].astype(str)
    keys["date"] = pd.to_datetime(keys["row_date"]).dt.date
    return keys.drop(columns=["row_date"])


def anchored_window(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp"])
    raw["subject_id"] = raw["subject_id"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw[raw["timestamp"].notna()].copy()
    row_start = raw["timestamp"].dt.normalize() - pd.to_timedelta((raw["timestamp"].dt.hour < 12).astype(int), unit="D")
    raw["date"] = row_start.dt.date
    raw["rel_hour"] = (raw["timestamp"] - row_start).dt.total_seconds() / 3600.0
    return raw[raw["rel_hour"].between(12.0, 36.0, inclusive="left")][["subject_id", "date", "rel_hour"]]


def gap_stats(hours: pd.Series) -> dict[str, float]:
    values = np.asarray(sorted(hours.dropna().astype(float).tolist()), dtype=float)
    if len(values) == 0:
        return {
            "first": math.nan,
            "last": math.nan,
            "span": math.nan,
            "median_gap_min": math.nan,
            "p90_gap_min": math.nan,
            "max_gap_min": math.nan,
            "long_gap_30m": math.nan,
            "long_gap_60m": math.nan,
            "long_gap_120m": math.nan,
        }
    if len(values) == 1:
        gaps_min = np.asarray([], dtype=float)
    else:
        gaps_min = np.diff(values) * 60.0
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "span": float(values[-1] - values[0]) if len(values) > 1 else 0.0,
        "median_gap_min": float(np.median(gaps_min)) if len(gaps_min) else math.nan,
        "p90_gap_min": float(np.percentile(gaps_min, 90)) if len(gaps_min) else math.nan,
        "max_gap_min": float(np.max(gaps_min)) if len(gaps_min) else math.nan,
        "long_gap_30m": float(np.sum(gaps_min >= 30.0)) if len(gaps_min) else 0.0,
        "long_gap_60m": float(np.sum(gaps_min >= 60.0)) if len(gaps_min) else 0.0,
        "long_gap_120m": float(np.sum(gaps_min >= 120.0)) if len(gaps_min) else 0.0,
    }


def silence_features(group: pd.DataFrame) -> dict[str, float]:
    hours = np.asarray(sorted(group["rel_hour"].dropna().astype(float).tolist()), dtype=float)
    out: dict[str, float] = {}
    if len(hours) == 0:
        for anchor in [20.0, 22.0, 24.0, 27.0, 31.0]:
            out[f"gap_after_{int(anchor)}h_min"] = math.nan
            out[f"gap_before_{int(anchor)}h_min"] = math.nan
        return out
    for anchor in [20.0, 22.0, 24.0, 27.0, 31.0]:
        after = hours[hours >= anchor]
        before = hours[hours <= anchor]
        out[f"gap_after_{int(anchor)}h_min"] = float((after[0] - anchor) * 60.0) if len(after) else math.nan
        out[f"gap_before_{int(anchor)}h_min"] = float((anchor - before[-1]) * 60.0) if len(before) else math.nan
    return out


def aggregate_modality(name: str, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ITEMS_DIR / filename
    raw = anchored_window(path)
    raw["bin_name"] = raw["rel_hour"].map(time_bin)
    rows: list[dict[str, Any]] = []
    for (subject_id, date_value), group in raw.groupby(["subject_id", "date"], sort=False):
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "date": date_value,
            f"cad_{name}_record_count": float(len(group)),
        }
        for key, value in gap_stats(group["rel_hour"]).items():
            row[f"cad_{name}_{key}"] = value
        for key, value in silence_features(group).items():
            row[f"cad_{name}_{key}"] = value
        bin_counts = group["bin_name"].value_counts()
        for bin_name in TIME_BINS:
            row[f"cad_{name}_{bin_name}_count"] = float(bin_counts.get(bin_name, 0.0))
        row[f"cad_{name}_late_to_onset_count_delta"] = (
            row[f"cad_{name}_late_22_24_count"] - row[f"cad_{name}_onset_24_27_count"]
        )
        row[f"cad_{name}_onset_to_deep_count_delta"] = (
            row[f"cad_{name}_onset_24_27_count"] - row[f"cad_{name}_deep_27_31_count"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    feature_cols = [col for col in out.columns if col.startswith(f"cad_{name}_")] if len(out) else []
    return out, {
        "source": filename,
        "raw_rows_in_window": int(len(raw)),
        "daily_rows": int(len(out)),
        "feature_columns": int(len(feature_cols)),
    }


def build_event_cadence_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = key_frame()
    features = keys.copy()
    source_metadata: dict[str, Any] = {}
    for name, filename in MODALITIES.items():
        frame, metadata = aggregate_modality(name, filename)
        source_metadata[name] = metadata
        features = features.merge(frame, on=["subject_id", "date"], how="left", validate="one_to_one")
    raw_cols = [col for col in features.columns if col.startswith("cad_")]
    missing_indicators = features[raw_cols].isna().astype(float).add_suffix("_missing")
    features = pd.concat([features, missing_indicators], axis=1).copy()
    feature_cols = [col for col in features.columns if col.startswith("cad_")]
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int(features["is_test"].eq(0).sum()),
        "n_test_rows": int(features["is_test"].eq(1).sum()),
        "raw_feature_columns": int(len(raw_cols)),
        "feature_columns": int(len(feature_cols)),
        "train_any_nonnull_row_share": float(
            features.loc[features["is_test"].eq(0), raw_cols].notna().any(axis=1).mean()
        ),
        "test_any_nonnull_row_share": float(
            features.loc[features["is_test"].eq(1), raw_cols].notna().any(axis=1).mean()
        ),
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), raw_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), raw_cols].isna().mean().mean()),
        "source_metadata": source_metadata,
    }
    return features, coverage


def align_train_features(train: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train_features = features[features["is_test"].eq(0)].copy()
    aligned = train[["subject_id", "date"]].merge(
        train_features.drop(columns=["is_test"]), on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    if len(aligned) != len(train):
        raise ValueError("Event-cadence features do not align with sorted train rows.")
    return aligned


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features.columns
        if col.startswith("cad_") and pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()
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
                    "candidate": f"event_cadence_{spec.name}",
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
                        1e-6,
                        1 - 1e-6,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    anchor_loss = id178_loss if anchor_name == "id178" else id341_loss
                    rows.append(
                        {
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"blend_{anchor_name}_event_cadence_{spec.name}_w{weight:g}",
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
    raw_features, coverage = build_event_cadence_features()
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
        "diagnostic": "post_id341_q3s3_event_cadence_smoke",
        "purpose": (
            "Test raw timestamp cadence and silence-gap features as a materially different Q3/S3 source family "
            "after physio-light, semantic routine, calendar prior, subject drift, and row-level adherence lanes closed."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "split_manifest": rel(REPORTS_DIR / "id340_validation_split_manifest_20260618.json"),
            "id178_oof": rel(find_oof(178)),
            "id341_oof": rel(find_oof(341)),
        },
        "feature_coverage": coverage,
        "config": {
            "modalities": MODALITIES,
            "time_bins": TIME_BINS,
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
            "event_cadence_artifact_allowed": False,
            "next_local_allowed_step": (
                "combined_no_artifact_geometry_gate" if selected_targets else "close_event_cadence_lane"
            ),
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If selected targets exist, keep event-cadence as report-only context and run only a combined no-artifact geometry gate. "
                "If no selected targets exist, close this local lane and return to the explicit-approved upstream rebuild path."
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
        "# Q3/S3 event-cadence smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It uses raw timestamp cadence/silence-gap features and creates no submission artifacts.",
        "",
        "## Decision",
        "",
        f"- Model smoke completed: `{decision['model_smoke_completed']}`",
        f"- Selected targets: `{decision['selected_targets']}`",
        f"- Q3 signal found: `{decision['q3_signal_found']}`",
        f"- S3 signal found: `{decision['s3_signal_found']}`",
        f"- All Q3/S3 targets covered: `{decision['all_q3s3_targets_covered']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` train/test = `{coverage['n_train_rows']}/{coverage['n_test_rows']}`",
        f"- Raw feature columns: `{coverage['raw_feature_columns']}`",
        f"- Feature columns incl. missing indicators: `{coverage['feature_columns']}`",
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
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
