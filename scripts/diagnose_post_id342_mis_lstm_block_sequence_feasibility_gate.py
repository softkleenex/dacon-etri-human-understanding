#!/usr/bin/env python3
"""Post-ID342 MIS-LSTM-style block-sequence feasibility gate.

This report-only diagnostic tests the newly prioritized source hypothesis from
the external research refresh: multi-hour continuous sensor blocks, separate
sparse-event blocks, and temporal aggregation before low-capacity downstream
heads. It is not a neural MIS-LSTM implementation; it is a bounded local
feasibility gate that asks whether the block-sequence representation has any
stable signal beyond ID342 before any Kaggle/no-submit runner is considered.

It creates no prediction CSV, feature matrix artifact, base prediction artifact,
submission CSV, submission-log row, Kaggle action, or DACON submission.
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

from scripts.diagnose_post_id342_evening_state_transition_gate import (  # noqa: E402
    ANCHOR_ID,
    BLEND_WEIGHTS,
    EPS,
    FAILED_FRONTIER_IDS,
    ID342_PUBLIC,
    ITEMS_DIR,
    MAX_FAILED_DIRECTION_CORR,
    MODEL_SPECS,
    PRIMARY_SPLITS,
    PUBLIC_055,
    TARGETS,
    add_calendar_features,
    add_previous_day_features,
    direction_metrics,
    load_reference_values,
    load_split_manifest,
    load_test_sorted,
    load_train_sorted,
    merge_feature_frames,
    predict_target_for_split,
    rank_feature_columns,
    rel,
    safe_array,
    weighted_split_loss,
)
from scripts.diagnose_post_id342_processed_sleep_proxy_gate import (  # noqa: E402
    TARGET_MAX_DELTA_CEILING,
    TARGET_MEAN_DELTA_CEILING,
    TARGET_WIN_FLOOR,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.json"
EXTERNAL_RESEARCH_REPORT = REPORTS_DIR / "post_id342_external_research_priority_20260625.json"

FOCUS_TARGETS = ["Q3", "S3", "S4"]
WINDOW_START = 12.0
WINDOW_END = 36.0
MAX_BLOCK_FEATURE_COLUMNS = 720

BLOCK_SCHEMES: dict[str, list[tuple[float, float]]] = {
    "block4_day_sleep": [(12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36)],
    "block3_evening_sleep": [(18, 21), (21, 24), (24, 27), (27, 30), (30, 33), (33, 36)],
}

TARGET_DOMAIN = {
    "Q3": "subjective pre-sleep stress vs personal average",
    "S3": "objective sleep onset latency guideline",
    "S4": "objective WASO / wakefulness after sleep onset guideline",
}

FEATURE_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all_block_sequence": (),
    "q3_day_evening_load": (
        "block4",
        "b12_16",
        "b16_20",
        "b20_24",
        "screen",
        "activity",
        "pedo",
        "hr",
        "diff",
        "early_late",
    ),
    "s3_sleep_onset_blocks": (
        "block3",
        "b18_21",
        "b21_24",
        "b24_27",
        "screen_off",
        "charging",
        "dark",
        "rest",
        "still",
        "early_late",
    ),
    "s4_sleep_fragment_blocks": (
        "block3",
        "b24_27",
        "b27_30",
        "b30_33",
        "b33_36",
        "screen_on",
        "bright",
        "moving",
        "hr_spike",
        "fragment",
        "std",
    ),
    "continuous_channels": ("hr", "wlight", "mlight", "pedo", "speed", "step", "light"),
    "sparse_event_channels": ("screen", "charge", "activity", "unlock", "still", "moving", "rest"),
}


def anchor_block_window(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour.astype(float)
    minute = out["timestamp"].dt.minute.astype(float)
    second = out["timestamp"].dt.second.astype(float)
    rel_hour = hour + minute / 60.0 + second / 3600.0
    anchor_date = out["timestamp"].dt.normalize()
    before_window_start = rel_hour < WINDOW_START
    anchor_date.loc[before_window_start] = anchor_date.loc[before_window_start] - pd.Timedelta(days=1)
    out["lifelog_date"] = anchor_date.dt.date
    out["rel_hour"] = rel_hour + before_window_start.astype(float) * 24.0
    return out[(out["rel_hour"] >= WINDOW_START) & (out["rel_hour"] < WINDOW_END)].copy()


def block_name(start: float, end: float) -> str:
    return f"b{int(start):02d}_{int(end):02d}"


def block_aggregates(
    df: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    aggs: tuple[str, ...],
    scheme_name: str,
    blocks: list[tuple[float, float]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if df.empty or not value_cols:
        return pd.DataFrame(columns=["subject_id", "lifelog_date"])
    for start, end in blocks:
        part = df[(df["rel_hour"] >= start) & (df["rel_hour"] < end)]
        name = block_name(start, end)
        if part.empty:
            continue
        for agg in aggs:
            grouped = part.groupby(["subject_id", "lifelog_date"], sort=False)[value_cols].agg(agg)
            grouped.columns = [f"mis_{scheme_name}_{prefix}_{name}_{col}_{agg}" for col in grouped.columns]
            frames.append(grouped.reset_index())
        counts = part.groupby(["subject_id", "lifelog_date"], sort=False).size()
        frames.append(counts.rename(f"mis_{scheme_name}_{prefix}_{name}_row_count").reset_index())
    return merge_feature_frames(frames)


def block_frames(
    df: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    aggs: tuple[str, ...],
) -> pd.DataFrame:
    frames = [
        block_aggregates(df, value_cols, prefix, aggs, scheme_name, blocks)
        for scheme_name, blocks in BLOCK_SCHEMES.items()
    ]
    return merge_feature_frames(frames)


def hr_row_stats(values: Any) -> tuple[float, float, float, float, float, float]:
    arr = safe_array(values)
    arr = arr[(arr > 0) & (arr < 250)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0.0
    return (
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float((arr <= 60).mean()),
        float((arr >= 90).mean()),
    )


def heart_rate_block_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wHr.parquet")
    df = anchor_block_window(raw)
    stats = pd.DataFrame(
        [hr_row_stats(value) for value in df["heart_rate"].to_numpy()],
        columns=["hr_mean", "hr_std", "hr_min", "hr_max", "hr_low60_rate", "hr_high90_rate"],
        index=df.index,
    )
    df = pd.concat([df, stats], axis=1)
    df["hr_spike"] = (df["hr_high90_rate"].fillna(0) > 0).astype(float)
    frame = block_frames(
        df,
        ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_low60_rate", "hr_high90_rate", "hr_spike"],
        "hr",
        ("mean", "min", "max"),
    )
    return frame, {"source": "wHr", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def light_block_features(filename: str, value_col: str, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / filename)
    df = anchor_block_window(raw)
    value = pd.to_numeric(df[value_col], errors="coerce")
    df[f"{prefix}_log1p"] = np.log1p(value.clip(lower=0))
    df[f"{prefix}_dark"] = (value.fillna(np.inf) < 10.0).astype(float)
    df[f"{prefix}_bright"] = (value.fillna(0.0) > 500.0).astype(float)
    frame = block_frames(
        df,
        [f"{prefix}_log1p", f"{prefix}_dark", f"{prefix}_bright"],
        prefix,
        ("mean", "max"),
    )
    return frame, {"source": prefix, "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def screen_block_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mScreenStatus.parquet")
    df = anchor_block_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    df["screen_on"] = (pd.to_numeric(df["m_screen_use"], errors="coerce").fillna(0) > 0).astype(float)
    df["screen_off"] = 1.0 - df["screen_on"]
    prev = df.groupby(["subject_id", "lifelog_date"], sort=False)["screen_on"].shift(1).fillna(0)
    df["screen_unlock"] = ((df["screen_on"] > 0) & (prev <= 0)).astype(float)
    frame = block_frames(df, ["screen_on", "screen_off", "screen_unlock"], "screen", ("mean", "sum"))
    return frame, {"source": "mScreenStatus", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def charging_block_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mACStatus.parquet")
    df = anchor_block_window(raw)
    df["charging"] = (pd.to_numeric(df["m_charging"], errors="coerce").fillna(0) > 0).astype(float)
    frame = block_frames(df, ["charging"], "charge", ("mean", "sum"))
    return frame, {"source": "mACStatus", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def activity_block_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_mActivity.parquet")
    df = anchor_block_window(raw).sort_values(["subject_id", "lifelog_date", "timestamp"])
    activity = pd.to_numeric(df["m_activity"], errors="coerce")
    df["activity_active"] = activity.isin([1, 2, 7, 8]).astype(float)
    df["activity_still"] = activity.eq(3).astype(float)
    df["activity_rest"] = activity.isin([3, 4]).astype(float)
    prev = activity.groupby([df["subject_id"], df["lifelog_date"]], sort=False).shift(1)
    df["activity_transition"] = activity.ne(prev).fillna(False).astype(float)
    frame = block_frames(
        df,
        ["activity_active", "activity_still", "activity_rest", "activity_transition"],
        "activity",
        ("mean", "sum"),
    )
    return frame, {"source": "mActivity", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def pedo_block_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(ITEMS_DIR / "ch2025_wPedo.parquet")
    df = anchor_block_window(raw)
    for col in ["step", "distance", "speed", "burned_calories", "walking_step", "running_step"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["moving"] = (df["step"].fillna(0) > 0).astype(float)
    df["rest"] = 1.0 - df["moving"]
    frame = block_frames(
        df,
        ["step", "distance", "speed", "burned_calories", "walking_step", "running_step", "moving", "rest"],
        "pedo",
        ("mean", "sum", "max"),
    )
    return frame, {"source": "wPedo", "raw_rows": int(len(raw)), "window_rows": int(len(df))}


def temporal_aggregation_features(features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col for col in features.columns if col.startswith("mis_") and pd.api.types.is_numeric_dtype(features[col])
    ]
    by_stem: dict[str, dict[str, str]] = {}
    for col in numeric_cols:
        parts = col.split("_")
        block_index = None
        for idx, part in enumerate(parts):
            if part.startswith("b") and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                block_index = idx
                break
        if block_index is None:
            continue
        block = f"{parts[block_index]}_{parts[block_index + 1]}"
        stem = "_".join(parts[:block_index] + parts[block_index + 2 :])
        by_stem.setdefault(stem, {})[block] = col

    added: dict[str, pd.Series] = {}
    for stem, block_cols in sorted(by_stem.items()):
        ordered_blocks = sorted(block_cols, key=lambda text: int(text[1:3]))
        if len(ordered_blocks) < 2:
            continue
        values = features[[block_cols[block] for block in ordered_blocks]].apply(pd.to_numeric, errors="coerce")
        added[f"mis_temporal_{stem}_block_mean"] = values.mean(axis=1)
        added[f"mis_temporal_{stem}_block_std"] = values.std(axis=1)
        added[f"mis_temporal_{stem}_block_range"] = values.max(axis=1) - values.min(axis=1)
        added[f"mis_temporal_{stem}_early_late_diff"] = values.iloc[:, -2:].mean(axis=1) - values.iloc[:, :2].mean(
            axis=1
        )
        for left, right in zip(ordered_blocks[:-1], ordered_blocks[1:], strict=False):
            added[f"mis_temporal_{stem}_diff_{left}_to_{right}"] = (
                features[block_cols[right]] - features[block_cols[left]]
            )
        if len(added) >= 260:
            break
    if not added:
        return features
    return pd.concat([features, pd.DataFrame(added, index=features.index)], axis=1)


def build_block_sequence_feature_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = pd.concat(
        [
            train[["subject_id", "date"]].assign(is_test=0),
            test[["subject_id", "date"]].assign(is_test=1),
        ],
        ignore_index=True,
    ).rename(columns={"date": "lifelog_date"})

    builders = [
        heart_rate_block_features,
        lambda: light_block_features("ch2025_wLight.parquet", "w_light", "wlight"),
        lambda: light_block_features("ch2025_mLight.parquet", "m_light", "mlight"),
        screen_block_features,
        charging_block_features,
        activity_block_features,
        pedo_block_features,
    ]
    frames: list[pd.DataFrame] = []
    source_metadata: list[dict[str, Any]] = []
    for builder in builders:
        frame, metadata = builder()
        metadata["feature_count"] = int(max(0, frame.shape[1] - 2))
        frames.append(frame)
        source_metadata.append(metadata)

    raw_features = merge_feature_frames(frames)
    features = keys.merge(raw_features, on=["subject_id", "lifelog_date"], how="left", validate="one_to_one")
    features = temporal_aggregation_features(features)
    features = add_calendar_features(features)
    features = add_previous_day_features(features)
    numeric_cols = [
        col
        for col in features.columns
        if col.startswith("mis_") or col.startswith("est_calendar_")
        if pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()
    ]
    ranked = rank_feature_columns(features, numeric_cols, min(MAX_BLOCK_FEATURE_COLUMNS, len(numeric_cols)))
    keep_cols = ["subject_id", "lifelog_date", "is_test", *ranked]
    features = features[keep_cols].copy()
    metadata = {
        "window": {"start_rel_hour": WINDOW_START, "end_rel_hour": WINDOW_END},
        "block_schemes": dict(BLOCK_SCHEMES),
        "source_metadata": source_metadata,
        "rows": int(len(features)),
        "train_rows": int((features["is_test"] == 0).sum()),
        "test_rows": int((features["is_test"] == 1).sum()),
        "numeric_feature_columns_before_rank": int(len(numeric_cols)),
        "numeric_feature_columns_used": int(len(ranked)),
        "mean_missing_rate": float(features[ranked].isna().mean().mean()) if ranked else None,
        "max_block_feature_columns": MAX_BLOCK_FEATURE_COLUMNS,
    }
    return features, metadata


def family_feature_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for family, keywords in FEATURE_FAMILY_KEYWORDS.items():
        if not keywords:
            families[family] = list(feature_cols)
        else:
            families[family] = [col for col in feature_cols if any(keyword in col for keyword in keywords)]
    return {family: cols for family, cols in families.items() if cols}


def scan_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_families: dict[str, list[str]],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    failed_refs: dict[int, np.ndarray],
    target: str,
) -> dict[str, Any]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for family, feature_cols in feature_families.items():
        for spec in MODEL_SPECS:
            split_source_predictions: dict[str, np.ndarray] = {}
            split_valid_counts: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_target_for_split(features, feature_cols, y, splits[split_name], spec)
                split_source_predictions[split_name] = np.where(np.isnan(pred), anchor_values[:, target_idx], pred)
                split_valid_counts[split_name] = int(covered)
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                split_direction_metrics: dict[str, Any] = {}
                wins = 0
                failed_corrs: list[float] = []
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    source_pred = split_source_predictions[split_name]
                    blended = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_pred, EPS, 1 - EPS
                    )
                    anchor_loss = weighted_split_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_split_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_anchor_losses[split_name] = anchor_loss
                    split_losses[split_name] = candidate_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                    metrics = direction_metrics(blended, anchor_values, failed_refs, target_idx)
                    split_direction_metrics[split_name] = metrics
                    corr = metrics.get("max_failed_direction_corr")
                    if corr is not None:
                        failed_corrs.append(float(corr))
                max_failed_corr = float(max(failed_corrs)) if failed_corrs else None
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                eligible = bool(
                    wins >= TARGET_WIN_FLOOR
                    and mean_delta <= TARGET_MEAN_DELTA_CEILING
                    and max_delta <= TARGET_MAX_DELTA_CEILING
                    and (max_failed_corr is None or max_failed_corr <= MAX_FAILED_DIRECTION_CORR)
                )
                rows.append(
                    {
                        "target": target,
                        "target_domain": TARGET_DOMAIN[target],
                        "config": f"{target.lower()}_id342_mis_blockseq_{family}_{spec.name}_w{weight:g}",
                        "feature_family": family,
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_counts": split_valid_counts,
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "split_deltas": split_deltas,
                        "split_losses": split_losses,
                        "split_anchor_losses": split_anchor_losses,
                        "max_failed_direction_corr": max_failed_corr,
                        "direction_metrics": split_direction_metrics,
                        "eligible": eligible,
                    }
                )
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    partial_rows = [row for row in rows if row["mean_delta"] < 0 and row["wins"] >= 3 and row["max_delta"] <= 0.003][:5]
    return {
        "target": target,
        "domain": TARGET_DOMAIN[target],
        "best": rows[0] if rows else None,
        "eligible_count": int(len(eligible_rows)),
        "selected": eligible_rows[0] if eligible_rows else None,
        "partial_signal_rows": partial_rows,
        "top_rows": rows[:10],
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["path"] = rel(path)
    payload["exists"] = True
    return payload


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    feature_frame, feature_metadata = build_block_sequence_feature_frame(train, test)
    features_train = feature_frame[feature_frame["is_test"].eq(0)].reset_index(drop=True)
    feature_cols = [
        col
        for col in features_train.columns
        if (col.startswith("mis_") or col.startswith("est_calendar_"))
        and pd.api.types.is_numeric_dtype(features_train[col])
    ]
    feature_families = family_feature_columns(feature_cols)
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    anchor_values = load_reference_values(train, ANCHOR_ID)
    failed_refs = {candidate_id: load_reference_values(train, candidate_id) for candidate_id in FAILED_FRONTIER_IDS}

    target_reports = {
        target: scan_target(
            train=train,
            features=features_train,
            feature_families=feature_families,
            splits=splits,
            anchor_values=anchor_values,
            failed_refs=failed_refs,
            target=target,
        )
        for target in FOCUS_TARGETS
    }
    selected_rows = {
        target: target_report["selected"]
        for target, target_report in target_reports.items()
        if target_report.get("selected") is not None
    }
    selected_targets = list(selected_rows)
    target_delta_sum = float(sum(float(row["mean_delta"]) for row in selected_rows.values()))
    estimated_all_target_delta = target_delta_sum / len(TARGETS)
    max_selected_delta = float(max([float(row["max_delta"]) for row in selected_rows.values()], default=0.0))
    projected_public_proxy = ID342_PUBLIC + estimated_all_target_delta
    final_gate_pass = bool(
        len(selected_targets) >= 2
        and estimated_all_target_delta <= -0.00050
        and max_selected_delta <= TARGET_MAX_DELTA_CEILING
        and projected_public_proxy <= 0.5920
    )
    partial_signal_targets = [
        target
        for target, target_report in target_reports.items()
        if target_report.get("partial_signal_rows") and target not in selected_targets
    ]
    external_priority = load_json(EXTERNAL_RESEARCH_REPORT)
    decision = {
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "s4_selected": "S4" in selected_targets,
        "target_delta_sum_vs_id342": target_delta_sum,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "max_selected_target_delta_vs_id342": max_selected_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "partial_signal_targets": partial_signal_targets,
        "mis_lstm_block_sequence_has_signal": bool(selected_targets),
        "final_mis_lstm_block_sequence_gate_pass": final_gate_pass,
        "public_055_evidence": False,
        "create_feature_matrix": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": bool(final_gate_pass),
        "run_kaggle_now": bool(final_gate_pass),
        "submit_now": False,
        "next_action": (
            "If this gate fails, do not launch a MIS-LSTM Kaggle runner. If it passes, run a separate no-submit "
            "artifact/runner contract review before any remote execution."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_mis_lstm_block_sequence_feasibility_gate",
        "purpose": (
            "Bounded local feasibility gate for MIS-LSTM-style multi-hour block sequences after 22 registered "
            "post-ID342 routes closed."
        ),
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "input_reports": {
            "external_research_priority": {
                "path": rel(EXTERNAL_RESEARCH_REPORT),
                "exists": external_priority.get("exists", False),
                "top_priority_route": external_priority.get("decision", {}).get("top_priority_route")
                if isinstance(external_priority.get("decision"), dict)
                else None,
            }
        },
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": list(BLEND_WEIGHTS),
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "max_failed_direction_corr": MAX_FAILED_DIRECTION_CORR,
            "failed_frontier_ids": FAILED_FRONTIER_IDS,
        },
        "feature_contract": {
            "domain_read": (
                "Q3 uses day/evening load and transition blocks. S3 uses pre-sleep/onset blocks with no-screen, "
                "dark, charging, stillness, and rest channels. S4 uses post-onset/morning-tail interruption blocks. "
                "This is a low-capacity feasibility proxy for MIS-LSTM, not a neural runner."
            ),
            "feature_metadata": feature_metadata,
            "feature_families": {
                family: {"feature_count": int(len(cols)), "example_features": cols[:12]}
                for family, cols in feature_families.items()
            },
        },
        "target_reports": target_reports,
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": not bool(final_gate_pass),
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    metadata = report["feature_contract"]["feature_metadata"]
    lines = [
        "# Post-ID342 MIS-LSTM block-sequence feasibility gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Domain Read",
        "",
        f"- {report['feature_contract']['domain_read']}",
        "",
        "## Feature Contract",
        "",
        f"- Window: `{metadata['window']['start_rel_hour']}` to `{metadata['window']['end_rel_hour']}` relative hours",
        f"- Rows: `{metadata['rows']}` (`{metadata['train_rows']}` train, `{metadata['test_rows']}` test)",
        f"- Numeric feature columns before rank: `{metadata['numeric_feature_columns_before_rank']}`",
        f"- Numeric feature columns used: `{metadata['numeric_feature_columns_used']}`",
        f"- Mean missing rate: `{metadata['mean_missing_rate']}`",
        "",
        "| source | raw rows | window rows | feature count |",
        "|---|---:|---:|---:|",
    ]
    for source in metadata["source_metadata"]:
        lines.append(
            f"| `{source['source']}` | `{source['raw_rows']}` | `{source['window_rows']}` | "
            f"`{source['feature_count']}` |"
        )
    lines.extend(["", "## Families", "", "| family | feature count | examples |", "|---|---:|---|"])
    for family, family_row in report["feature_contract"]["feature_families"].items():
        examples = ", ".join(f"`{col}`" for col in family_row["example_features"][:5])
        lines.append(f"| `{family}` | {family_row['feature_count']} | {examples} |")
    lines.extend(
        ["", "## Target Scan", "", "| target | selected | eligible | best config | wins | mean delta | max delta |"]
    )
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for target, target_report in report["target_reports"].items():
        best = target_report.get("best") or {}
        selected = target_report.get("selected")
        lines.append(
            f"| `{target}` | `{selected is not None}` | {target_report.get('eligible_count', 0)} | "
            f"`{best.get('config')}` | {best.get('wins')} | {float(best.get('mean_delta', 0.0)):.9f} | "
            f"{float(best.get('max_delta', 0.0)):.9f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Estimated all-target delta vs ID342: `{decision['estimated_all_target_delta_vs_id342']:.9f}`",
            f"- Projected Public proxy: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55: `{decision['gap_to_public_055_after_proxy']:.9f}`",
            f"- Final gate pass: `{decision['final_mis_lstm_block_sequence_gate_pass']}`",
            f"- Kaggle now: `{decision['run_kaggle_now']}`",
            f"- DACON submit now: `{decision['submit_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


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
    write_markdown(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Selected targets: "
        f"{decision['selected_targets']} | projected_public_proxy="
        f"{decision['projected_public_proxy_from_id342']:.9f} | final_gate_pass="
        f"{decision['final_mis_lstm_block_sequence_gate_pass']}"
    )


if __name__ == "__main__":
    main()
