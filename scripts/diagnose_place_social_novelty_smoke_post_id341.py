#!/usr/bin/env python3
"""Report-only place/social novelty source smoke after ID341.

This diagnostic tests a stronger variant of the routine/place lane by using raw
Wi-Fi, BLE, and app-usage identifiers instead of only processed aggregate
columns. It creates label-free daily novelty features such as new-device share,
rolling-set Jaccard similarity, entropy, and top-identifier dominance, then
evaluates small fold-internal source models against ID178/ID341 on the frozen
post-ID341 split manifest.

It creates no submission CSV, base prediction artifact, notebook/history
artifact, submission-log row, Kaggle push, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
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
    ANCHOR_IDS,
    SCALES,
    evaluate_scale,
    full_test_sources_for_selected,
    source_predictions_for_selected_splits,
)
from scripts.diagnose_routine_stability_place_entropy_smoke_post_id341 import (  # noqa: E402
    CONFIG_MAX_DELTA_CEILING,
    CONFIG_MEAN_DELTA_CEILING,
    CONFIG_WIN_FLOOR,
    TARGET_FOCUS,
    scan_target_configs,
    select_target_configs,
)
from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    BLEND_WEIGHTS,
    MODEL_SPECS,
    REPORTS_DIR,
    load_reference_values,
    load_split_manifest,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import load_base_test_values  # noqa: E402
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"


def anchored_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    row_start = out["timestamp"].dt.normalize() - pd.to_timedelta((out["timestamp"].dt.hour < 12).astype(int), "D")
    out["row_date"] = row_start.dt.date
    out["rel_hour"] = (out["timestamp"] - row_start).dt.total_seconds() / 3600.0
    return out[out["rel_hour"].between(12.0, 36.0, inclusive="left")]


def entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return np.nan
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(max(p, 1e-12))
    return float(entropy)


def top_share_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return float(max(counter.values()) / total) if total > 0 and counter else np.nan


def novelty_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.sort_values(["subject_id", "row_date"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _subject, group in out.groupby("subject_id", sort=False):
        prev_set: set[str] = set()
        rolling_sets: deque[set[str]] = deque(maxlen=7)
        for idx, row in group.iterrows():
            current = set(row["_ids"])
            rolling_union = set().union(*rolling_sets) if rolling_sets else set()
            prev_union = prev_set

            def jaccard(current_set: set[str], reference: set[str]) -> float:
                if not current_set and not reference:
                    return np.nan
                union = current_set | reference
                if not union:
                    return np.nan
                return float(len(current_set & reference) / len(union))

            def new_share(current_set: set[str], reference: set[str]) -> float:
                if not current_set:
                    return np.nan
                return float(len(current_set - reference) / len(current_set))

            rows.append(
                {
                    "_row_index": idx,
                    f"{prefix}_jaccard_prev": jaccard(current, prev_union),
                    f"{prefix}_new_share_prev": new_share(current, prev_union),
                    f"{prefix}_jaccard_roll7": jaccard(current, rolling_union),
                    f"{prefix}_new_share_roll7": new_share(current, rolling_union),
                    f"{prefix}_repeat_share_roll7": np.nan
                    if not current
                    else float(len(current & rolling_union) / len(current)),
                }
            )
            prev_set = current
            rolling_sets.append(current)
    novelty = pd.DataFrame(rows).set_index("_row_index")
    return pd.concat([out, novelty.reindex(out.index)], axis=1)


def list_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return []


def daily_device_features(filename: str, list_col: str, id_key: str, prefix: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / filename, columns=["subject_id", "timestamp", list_col])
    df = anchored_window(df)
    accum: dict[tuple[str, Any], dict[str, Any]] = {}
    for subject_id, row_date, rel_hour, values in df[["subject_id", "row_date", "rel_hour", list_col]].itertuples(
        index=False
    ):
        key = (str(subject_id), row_date)
        rec = accum.setdefault(
            key,
            {
                "scan_count": 0,
                "item_count": 0,
                "evening_items": 0,
                "night_items": 0,
                "strong_items": 0,
                "rssi_sum": 0.0,
                "rssi_count": 0,
                "counter": Counter(),
            },
        )
        rec["scan_count"] += 1
        for item in list_items(values):
            if not isinstance(item, dict) or id_key not in item:
                continue
            ident = str(item[id_key])
            rec["counter"][ident] += 1
            rec["item_count"] += 1
            if 18 <= float(rel_hour) < 24:
                rec["evening_items"] += 1
            if 24 <= float(rel_hour) < 34:
                rec["night_items"] += 1
            try:
                rssi = float(item.get("rssi", np.nan))
            except (TypeError, ValueError):
                rssi = np.nan
            if np.isfinite(rssi):
                rec["rssi_sum"] += rssi
                rec["rssi_count"] += 1
                if rssi >= -60:
                    rec["strong_items"] += 1

    rows: list[dict[str, Any]] = []
    for (subject_id, row_date), rec in accum.items():
        counter: Counter[str] = rec["counter"]
        item_count = rec["item_count"]
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                "_ids": set(counter.keys()),
                f"{prefix}_scan_count": float(rec["scan_count"]),
                f"{prefix}_item_count": float(item_count),
                f"{prefix}_unique_count": float(len(counter)),
                f"{prefix}_items_per_scan": float(item_count / max(rec["scan_count"], 1)),
                f"{prefix}_entropy": entropy_from_counter(counter),
                f"{prefix}_top_share": top_share_from_counter(counter),
                f"{prefix}_evening_item_share": float(rec["evening_items"] / item_count) if item_count else np.nan,
                f"{prefix}_night_item_share": float(rec["night_items"] / item_count) if item_count else np.nan,
                f"{prefix}_strong_item_share": float(rec["strong_items"] / item_count) if item_count else np.nan,
                f"{prefix}_mean_rssi": float(rec["rssi_sum"] / rec["rssi_count"]) if rec["rssi_count"] else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["subject_id", "row_date"])
    features = novelty_columns(pd.DataFrame(rows), prefix)
    return features.drop(columns=["_ids"])


def daily_usage_features() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "ch2025_mUsageStats.parquet", columns=["subject_id", "timestamp", "m_usage_stats"])
    df = anchored_window(df)
    accum: dict[tuple[str, Any], dict[str, Any]] = {}
    for subject_id, row_date, rel_hour, values in df[
        ["subject_id", "row_date", "rel_hour", "m_usage_stats"]
    ].itertuples(index=False):
        key = (str(subject_id), row_date)
        rec = accum.setdefault(
            key,
            {
                "scan_count": 0,
                "app_counter": Counter(),
                "time_counter": Counter(),
                "total_time": 0.0,
                "late_time": 0.0,
                "evening_time": 0.0,
            },
        )
        rec["scan_count"] += 1
        for item in list_items(values):
            if not isinstance(item, dict):
                continue
            app = str(item.get("app_name", "")).strip()
            if not app:
                continue
            try:
                total_time = float(item.get("total_time", 0.0))
            except (TypeError, ValueError):
                total_time = 0.0
            total_time = max(total_time, 0.0)
            rec["app_counter"][app] += 1
            rec["time_counter"][app] += total_time
            rec["total_time"] += total_time
            if 18 <= float(rel_hour) < 24:
                rec["evening_time"] += total_time
            if 24 <= float(rel_hour) < 34:
                rec["late_time"] += total_time

    rows: list[dict[str, Any]] = []
    prefix = "psn_usage"
    for (subject_id, row_date), rec in accum.items():
        app_counter: Counter[str] = rec["app_counter"]
        time_counter: Counter[str] = rec["time_counter"]
        total_time = rec["total_time"]
        rows.append(
            {
                "subject_id": subject_id,
                "row_date": row_date,
                "_ids": set(app_counter.keys()),
                f"{prefix}_scan_count": float(rec["scan_count"]),
                f"{prefix}_unique_app_count": float(len(app_counter)),
                f"{prefix}_log_total_time": float(np.log1p(total_time)),
                f"{prefix}_evening_time_share": float(rec["evening_time"] / total_time) if total_time else np.nan,
                f"{prefix}_late_time_share": float(rec["late_time"] / total_time) if total_time else np.nan,
                f"{prefix}_app_entropy": entropy_from_counter(app_counter),
                f"{prefix}_time_entropy": entropy_from_counter(time_counter),
                f"{prefix}_top_app_share": top_share_from_counter(app_counter),
                f"{prefix}_top_time_share": top_share_from_counter(time_counter),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["subject_id", "row_date"])
    features = novelty_columns(pd.DataFrame(rows), prefix)
    return features.drop(columns=["_ids"])


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame.copy() if out is None else out.merge(frame, on=["subject_id", "row_date"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["subject_id", "row_date"])


def build_place_social_novelty_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    )
    rows["row_date"] = pd.to_datetime(rows["date"]).dt.date
    raw_features = merge_feature_frames(
        [
            daily_device_features("ch2025_mWifi.parquet", "m_wifi", "bssid", "psn_wifi"),
            daily_device_features("ch2025_mBle.parquet", "m_ble", "address", "psn_ble"),
            daily_usage_features(),
        ]
    )
    features = rows.merge(raw_features, on=["subject_id", "row_date"], how="left")
    feature_cols = [col for col in features.columns if col.startswith("psn_")]
    for col in list(feature_cols):
        features[f"{col}_missing"] = features[col].isna().astype(float)
    feature_cols = [col for col in features.columns if col.startswith("psn_")]
    coverage = {
        "n_rows": int(len(features)),
        "n_train_rows": int((features["is_test"] == 0).sum()),
        "n_test_rows": int((features["is_test"] == 1).sum()),
        "n_feature_columns": int(len(feature_cols)),
        "train_missing_mean": float(features.loc[features["is_test"].eq(0), feature_cols].isna().mean().mean()),
        "test_missing_mean": float(features.loc[features["is_test"].eq(1), feature_cols].isna().mean().mean()),
        "top_missing_features": {
            str(k): float(v)
            for k, v in features.loc[features["is_test"].eq(0), feature_cols]
            .isna()
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .items()
        },
        "raw_sources": ["wifi_bssid", "ble_address", "usage_app_name"],
    }
    return features, coverage


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    features, feature_coverage = build_place_social_novelty_features(train, test)
    feature_cols = [col for col in features.columns if col.startswith("psn_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)

    scan_rows = scan_target_configs(train, features_train, feature_cols, splits, reference178, reference341)
    selected_configs = select_target_configs(scan_rows)
    scale_rows: list[dict[str, Any]] = []
    if selected_configs:
        base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}
        split_sources = source_predictions_for_selected_splits(
            train, features_train, feature_cols, splits, selected_configs, reference341
        )
        test_sources = full_test_sources_for_selected(train, features, feature_cols, selected_configs)
        scale_rows = [
            evaluate_scale(
                scale,
                train,
                splits,
                selected_configs,
                split_sources,
                test_sources,
                reference178,
                reference341,
                base_tests,
            )
            for scale in SCALES
        ]
    reviewable_rows = [row for row in scale_rows if row["reviewable_low_movement_candidate"]]
    best_reviewable = (
        sorted(reviewable_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["scale"]))[0]
        if reviewable_rows
        else None
    )
    best_proxy = (
        sorted(scale_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"]))[
            0
        ]
        if scale_rows
        else None
    )
    top_scan_rows = {target: [row for row in scan_rows if row["target"] == target][:10] for target in TARGET_FOCUS}
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "place_social_novelty_smoke_post_id341",
        "purpose": "Evaluate raw WiFi/BLE/app identifier novelty as an independent post-ID341 source family.",
        "feature_coverage": feature_coverage,
        "config_rules": {
            "target_focus": TARGET_FOCUS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "scales": SCALES,
            "blend_weights": BLEND_WEIGHTS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        },
        "selected_target_configs": selected_configs,
        "top_scan_rows_by_target": top_scan_rows,
        "scale_rows": scale_rows,
        "best_reviewable_low_movement": best_reviewable,
        "best_by_proxy": best_proxy,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "selected_targets": sorted(selected_configs.keys()),
            "reviewable_low_movement_candidate_exists": bool(best_reviewable is not None),
            "best_reviewable_scale": None if best_reviewable is None else best_reviewable["scale"],
            "best_proxy": None if best_proxy is None else best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_scale": None if best_proxy is None else best_proxy["scale"],
            "public_058_plausible": False
            if best_proxy is None
            else bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "next_action": (
                "If no reviewable place/social novelty row appears, do not widen this source locally; prefer the "
                "report-only upstream rebuild or a materially different raw source."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    lines = [
        "# Place/social novelty smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push, or DACON submission.",
        "",
        "## Feature coverage",
        "",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Feature columns: `{coverage['n_feature_columns']}`",
        f"- Mean missing rate: train `{coverage['train_missing_mean']:.4f}`, test `{coverage['test_missing_mean']:.4f}`",
        f"- Raw sources: `{coverage['raw_sources']}`",
        "",
        "## Selected target configs",
        "",
        "| target | candidate | wins | mean Δ | max Δ |",
        "|---|---|---:|---:|---:|",
    ]
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    if report["scale_rows"]:
        lines.extend(
            [
                "",
                "## Scale scan",
                "",
                "| scale | primary wins | all primary | local | proxy | Δ341 mean/max | failed proj | reviewable |",
                "|---:|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in report["scale_rows"]:
            geom = row["test_geometry"]
            lines.append(
                f"| {row['scale']:.2f} | {row['primary_wins_vs_id341']} | `{row['all_primary_splits_beat_id341']}` | "
                f"{row['local_cv_subject_hole_5']:.6f} | {geom['proxy_public_risk']:.6f} | "
                f"{geom['mean_delta_vs_341']:.6f}/{geom['max_delta_vs_341']:.6f} | "
                f"{geom['projection_failed']:+.6f} | `{row['reviewable_low_movement_candidate']}` |"
            )
    lines.extend(["", "## Top scan rows by target", ""])
    for target, rows in report["top_scan_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | candidate | wins | mean Δ | max Δ |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows[:5], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    lines.extend(["## Decision", ""])
    best_reviewable = report["best_reviewable_low_movement"]
    if best_reviewable is None:
        lines.append("- Reviewable low-movement candidate: `False`")
    else:
        geom = best_reviewable["test_geometry"]
        lines.extend(
            [
                "- Reviewable low-movement candidate: `True`",
                f"- Best reviewable scale: `{best_reviewable['scale']:.2f}`",
                f"- Best reviewable proxy: `{geom['proxy_public_risk']:.6f}`",
            ]
        )
    if report["best_by_proxy"] is not None:
        lines.append(
            f"- Best proxy: `{report['best_by_proxy']['test_geometry']['proxy_public_risk']:.6f}` at scale `{report['best_by_proxy']['scale']:.2f}`"
        )
    lines.extend(
        [
            f"- Public `0.58` plausible: `{report['decision']['public_058_plausible']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
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
