#!/usr/bin/env python3
"""In-memory feature-generation dry run for proximity/ambience co-occurrence v1.

This report-only gate assembles the frozen 80-column proximity/ambience feature
schema in memory, verifies exact train/test row alignment, checks finite bounded
values and privacy guards, then stops. It does not persist a feature matrix,
train a model, create predictions or submission artifacts, edit the submission
log, run Kaggle, or submit to DACON.
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

from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_spec import (  # noqa: E402
    DEFAULT_OUTPUT as FEATURE_SPEC_OUTPUT,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_feature_spec import (  # noqa: E402
    FEATURE_GROUPS,
    GLOBAL_FEATURES,
    KEY_COLUMNS,
    SOURCE_FAMILY,
    TARGET_FOCUS,
    WINDOWS,
    decision,
    feature_names,
    load_json_report,
    rel,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_metadata_smoke import (  # noqa: E402  # noqa: E402
    AMBIENCE_PATH,
    BLE_PATH,
    METRICS_TRAIN,
    SAMPLE_SUBMISSION,
    SCREEN_PATH,
    WIFI_PATH,
    ambience_frame,
    proximity_frame,
    screen_frame,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun_20260619.json"
NEXT_ALLOWED_STEP = "report_only_proximity_ambience_cooccurrence_feature_matrix_smoke_contract"
FORBIDDEN_REPORT_TOKENS = ("bssid", "address:", "raw_token", "raw_identifier_value")


def expected_key_frame() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=KEY_COLUMNS)
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=KEY_COLUMNS)
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True).drop_duplicates(KEY_COLUMNS + ["split"])


def build_bin_frame() -> pd.DataFrame:
    wifi, _wifi_tokens = proximity_frame(WIFI_PATH, "m_wifi", "bssid", -65)
    ble, _ble_tokens = proximity_frame(BLE_PATH, "m_ble", "address", -70)
    ambience = ambience_frame()
    screen = screen_frame()

    key_cols = ["subject_id", "lifelog_date", "bin10", "hour"]
    wifi_bin = (
        wifi.groupby(key_cols)
        .agg(wifi_rows=("scan_rows", "sum"), wifi_devices=("device_count", "sum"), wifi_strong=("strong_count", "sum"))
        .reset_index()
    )
    ble_bin = (
        ble.groupby(key_cols)
        .agg(ble_rows=("scan_rows", "sum"), ble_devices=("device_count", "sum"), ble_strong=("strong_count", "sum"))
        .reset_index()
    )
    ambience_bin = (
        ambience.groupby(key_cols)
        .agg(
            ambience_rows=("top_score", "size"),
            speech=("speech", "max"),
            crowd=("crowd", "max"),
            inside=("inside", "max"),
            outside=("outside", "max"),
            vehicle=("vehicle", "max"),
            music=("music", "max"),
        )
        .reset_index()
    )
    screen_bin = (
        screen.groupby(key_cols)
        .agg(screen_rows=("m_screen_use", "size"), screen_on_mean=("m_screen_use", "mean"))
        .reset_index()
    )
    merged = (
        wifi_bin.merge(ble_bin, on=key_cols, how="outer")
        .merge(ambience_bin, on=key_cols, how="outer")
        .merge(screen_bin, on=key_cols, how="outer")
    )
    numeric_cols = [col for col in merged.columns if col not in key_cols]
    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
    merged["has_wifi"] = merged["wifi_rows"].gt(0)
    merged["has_ble"] = merged["ble_rows"].gt(0)
    merged["has_proximity"] = merged["has_wifi"] | merged["has_ble"]
    merged["has_ambience"] = merged["ambience_rows"].gt(0)
    merged["social_ambience"] = merged[["speech", "crowd"]].max(axis=1)
    merged["place_ambience"] = merged[["inside", "outside", "vehicle"]].max(axis=1)
    merged["vehicle_music"] = merged[["vehicle", "music"]].max(axis=1)
    merged["proximity_social_cooccur"] = merged["has_proximity"] & merged["social_ambience"].ge(0.10)
    merged["proximity_place_cooccur"] = merged["has_proximity"] & merged["place_ambience"].ge(0.10)
    merged["screen_proximity_cooccur"] = merged["has_proximity"] & merged["screen_on_mean"].gt(0)
    merged["social_screen_proximity_cooccur"] = merged["proximity_social_cooccur"] & merged["screen_on_mean"].gt(0)
    return merged


def aggregate_window(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    hours = set(WINDOWS[window]["hours"])
    part = frame[frame["hour"].isin(hours)].copy()
    prefix = f"proxamb_{window}"
    if part.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "lifelog_date",
                *[f"{prefix}_{group}_{metric}" for group, metrics in FEATURE_GROUPS.items() for metric in metrics],
            ]
        )
    grouped = (
        part.groupby(["subject_id", "lifelog_date"])
        .agg(
            has_wifi=("has_wifi", "sum"),
            wifi_devices=("wifi_devices", "sum"),
            wifi_strong=("wifi_strong", "sum"),
            has_ble=("has_ble", "sum"),
            ble_devices=("ble_devices", "sum"),
            ble_strong=("ble_strong", "sum"),
            speech_max=("speech", "max"),
            speech_mean=("speech", "mean"),
            crowd_max=("crowd", "max"),
            place_max=("place_ambience", "max"),
            vehicle_music_max=("vehicle_music", "max"),
            proximity_social=("proximity_social_cooccur", "sum"),
            proximity_place=("proximity_place_cooccur", "sum"),
            screen_proximity=("screen_proximity_cooccur", "sum"),
            social_screen_proximity=("social_screen_proximity_cooccur", "sum"),
        )
        .reset_index()
    )
    rename = {
        "has_wifi": f"{prefix}_wifi_proximity_richness_wifi_bins",
        "wifi_devices": f"{prefix}_wifi_proximity_richness_wifi_devices_sum",
        "wifi_strong": f"{prefix}_wifi_proximity_richness_wifi_strong_sum",
        "has_ble": f"{prefix}_ble_proximity_richness_ble_bins",
        "ble_devices": f"{prefix}_ble_proximity_richness_ble_devices_sum",
        "ble_strong": f"{prefix}_ble_proximity_richness_ble_strong_sum",
        "speech_max": f"{prefix}_ambience_social_place_speech_max",
        "speech_mean": f"{prefix}_ambience_social_place_speech_mean",
        "crowd_max": f"{prefix}_ambience_social_place_crowd_max",
        "place_max": f"{prefix}_ambience_social_place_place_max",
        "vehicle_music_max": f"{prefix}_ambience_social_place_vehicle_music_max",
        "proximity_social": f"{prefix}_cross_sensor_cooccurrence_proximity_social_bins",
        "proximity_place": f"{prefix}_cross_sensor_cooccurrence_proximity_place_bins",
        "screen_proximity": f"{prefix}_cross_sensor_cooccurrence_screen_proximity_bins",
        "social_screen_proximity": f"{prefix}_cross_sensor_cooccurrence_social_screen_proximity_bins",
    }
    return grouped.rename(columns=rename)


def global_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["subject_id", "lifelog_date"])
        .agg(
            wifi_bins=("has_wifi", "sum"),
            ble_bins=("has_ble", "sum"),
            ambience_bins=("has_ambience", "sum"),
            proximity_bins=("has_proximity", "sum"),
            proximity_social_bins=("proximity_social_cooccur", "sum"),
            evening_social_bins=(
                "proximity_social_cooccur",
                lambda values: int(values[frame.loc[values.index, "hour"].between(18, 23)].sum()),
            ),
            late_social_bins=(
                "proximity_social_cooccur",
                lambda values: int(
                    (values & ((frame.loc[values.index, "hour"] >= 22) | (frame.loc[values.index, "hour"] <= 3))).sum()
                ),
            ),
        )
        .reset_index()
    )
    grouped["proxamb_global_wifi_day_covered_flag"] = grouped["wifi_bins"].gt(0).astype(float)
    grouped["proxamb_global_ble_day_covered_flag"] = grouped["ble_bins"].gt(0).astype(float)
    grouped["proxamb_global_ambience_day_covered_flag"] = grouped["ambience_bins"].gt(0).astype(float)
    grouped["proxamb_global_proximity_social_bin_ratio"] = grouped["proximity_social_bins"] / grouped[
        "proximity_bins"
    ].clip(lower=1)
    grouped["proxamb_global_late_to_evening_social_cooccur_ratio"] = grouped["late_social_bins"] / grouped[
        "evening_social_bins"
    ].clip(lower=1)
    return grouped[["subject_id", "lifelog_date", *[f"proxamb_global_{name}" for name in GLOBAL_FEATURES]]]


def assemble_feature_frame() -> pd.DataFrame:
    expected = expected_key_frame()
    bins = build_bin_frame()
    features = expected.copy()
    for window in WINDOWS:
        features = features.merge(aggregate_window(bins, window), on=["subject_id", "lifelog_date"], how="left")
    features = features.merge(global_features(bins), on=["subject_id", "lifelog_date"], how="left")
    names = feature_names()
    features[names] = features[names].fillna(0.0).astype(float)
    return features[[*KEY_COLUMNS, "split", *names]]


def feature_summary(frame: pd.DataFrame) -> dict[str, Any]:
    names = feature_names()
    values = frame[names].to_numpy(dtype=float)
    zero_variance_count = int((frame[names].nunique(dropna=False) <= 1).sum())
    return {
        "row_count": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "column_count_including_keys": int(frame.shape[1]),
        "feature_column_count": int(len(names)),
        "expected_feature_column_count": 80,
        "feature_columns_match_spec": list(frame.columns[-len(names) :]) == names,
        "duplicate_key_rows": int(frame.duplicated(KEY_COLUMNS + ["split"]).sum()),
        "finite_all_values": bool(np.isfinite(values).all()),
        "nan_count": int(np.isnan(values).sum()),
        "feature_min": float(np.nanmin(values)) if values.size else None,
        "feature_max": float(np.nanmax(values)) if values.size else None,
        "zero_variance_feature_count": zero_variance_count,
        "nonzero_feature_count": int((frame[names].sum(axis=0).abs() > 0).sum()),
    }


def build_report() -> dict[str, Any]:
    spec = load_json_report(FEATURE_SPEC_OUTPUT.name)
    spec_decision = decision(spec)
    frame = assemble_feature_frame()
    summary = feature_summary(frame)
    names = feature_names()

    spec_contract_ready = bool(
        spec.get("exists")
        and spec.get("parse_ok")
        and spec_decision.get("feature_spec_ready") is True
        and spec_decision.get("source_family") == SOURCE_FAMILY
        and spec_decision.get("target_focus") == TARGET_FOCUS
        and spec_decision.get("feature_count") == len(names)
        and spec_decision.get("next_local_allowed_step")
        == "report_only_proximity_ambience_cooccurrence_feature_generation_dryrun_no_matrix"
        and spec_decision.get("feature_matrix_allowed_now") is False
        and spec_decision.get("train_model_now") is False
        and spec_decision.get("create_submission_artifact") is False
        and spec_decision.get("submit_now") is False
    )
    row_alignment_passed = bool(
        summary["row_count"] == 700
        and summary["train_rows"] == 450
        and summary["test_rows"] == 250
        and summary["duplicate_key_rows"] == 0
    )
    schema_passed = bool(
        summary["feature_column_count"] == 80
        and summary["feature_columns_match_spec"]
        and all(name.startswith("proxamb_") for name in names)
        and not any(token in name.lower() for token in ("bssid", "address", "token") for name in names)
    )
    values_passed = bool(
        summary["finite_all_values"]
        and summary["nan_count"] == 0
        and summary["nonzero_feature_count"] >= 60
        and summary["zero_variance_feature_count"] <= 20
    )
    report_text_probe = json.dumps(
        {
            "diagnostic": "post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun",
            "feature_columns": names,
        },
        ensure_ascii=False,
    ).lower()
    privacy_guard_passed = bool(not any(token in report_text_probe for token in FORBIDDEN_REPORT_TOKENS))
    dryrun_passed = bool(
        spec_contract_ready and row_alignment_passed and schema_passed and values_passed and privacy_guard_passed
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Assemble the frozen proximity/ambience co-occurrence feature schema in memory only, verify exact "
            "700-row alignment and bounded finite columns, then stop before any persisted feature matrix or model."
        ),
        "input_reports": {
            "feature_spec": {
                "path": rel(FEATURE_SPEC_OUTPUT),
                "exists": spec.get("exists"),
                "parse_ok": spec.get("parse_ok"),
            }
        },
        "raw_inputs": {
            "wifi": rel(WIFI_PATH),
            "ble": rel(BLE_PATH),
            "ambience": rel(AMBIENCE_PATH),
            "screen": rel(SCREEN_PATH),
        },
        "feature_frame_summary": summary,
        "feature_groups": [
            {"name": group, "metric_count": len(metrics), "metrics": metrics}
            for group, metrics in FEATURE_GROUPS.items()
        ],
        "windows": {
            name: {"hours": meta["hours"], "description": meta["description"]} for name, meta in WINDOWS.items()
        },
        "leakage_and_privacy_contract": {
            "labels_loaded": False,
            "oof_or_test_predictions_loaded": False,
            "public_scores_used": False,
            "raw_proximity_identifier_columns_created": False,
            "raw_proximity_identifier_values_persisted": False,
            "feature_frame_persisted": False,
            "feature_matrix_written": False,
            "all_missing_values_filled_with_zero_or_flags": summary["nan_count"] == 0,
        },
        "checks": {
            "spec_contract_ready": spec_contract_ready,
            "row_alignment_passed": row_alignment_passed,
            "schema_passed": schema_passed,
            "values_passed": values_passed,
            "privacy_guard_passed": privacy_guard_passed,
            "does_not_write_feature_matrix": True,
            "does_not_train_model": True,
            "does_not_create_submission_artifact": True,
        },
        "decision": {
            "feature_generation_dryrun_passed": dryrun_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "feature_count": len(names),
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": NEXT_ALLOWED_STEP if dryrun_passed else "fix_proximity_ambience_dryrun_inputs",
            "next_action": (
                "Dry run passed. A future no-approval step may only write a report-only feature-matrix smoke contract; "
                "do not write a feature matrix, train, create ID342, run Kaggle, or submit."
                if dryrun_passed
                else "Do not advance this lane until dry-run contract blockers are fixed."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    summary = report["feature_frame_summary"]
    lines = [
        "# Post-ID341 proximity/ambience co-occurrence feature-generation dry run",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Dry run passed: `{dec['feature_generation_dryrun_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Feature count: `{dec['feature_count']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Feature-frame summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Privacy/leakage contract", ""])
    for key, value in report["leakage_and_privacy_contract"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(md_path)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    if not report["decision"]["feature_generation_dryrun_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
