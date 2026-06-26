#!/usr/bin/env python3
"""Report-only metadata smoke for proximity/ambience co-occurrence work.

This script checks raw coverage, timestamp-bin co-occurrence feasibility, and
privacy-safe cardinality for WiFi/BLE proximity plus ambience context. It writes
only JSON/Markdown reports. It does not write a feature matrix, train a model,
create predictions or submission artifacts, edit the submission log, run Kaggle,
or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_design import (  # noqa: E402
    DEFAULT_OUTPUT as DESIGN_OUTPUT,
)
from scripts.diagnose_post_id341_proximity_ambience_cooccurrence_design import (  # noqa: E402
    SOURCE_FAMILY,
    TARGET_FOCUS,
    decision,
    load_json_report,
    rel,
)

REPORTS_DIR = ROOT / "reports"
RAW_ITEMS_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_proximity_ambience_cooccurrence_metadata_smoke_20260619.json"
WIFI_PATH = RAW_ITEMS_DIR / "ch2025_mWifi.parquet"
BLE_PATH = RAW_ITEMS_DIR / "ch2025_mBle.parquet"
AMBIENCE_PATH = RAW_ITEMS_DIR / "ch2025_mAmbience.parquet"
SCREEN_PATH = RAW_ITEMS_DIR / "ch2025_mScreenStatus.parquet"

AMBIENCE_GROUPS = {
    "speech": ["speech", "conversation", "narration", "monologue", "child speech"],
    "crowd": ["crowd", "cheering", "applause", "people"],
    "inside": ["inside", "room", "indoor", "public space"],
    "outside": ["outside", "urban", "rural", "natural"],
    "vehicle": ["vehicle", "car", "truck", "motor vehicle", "traffic"],
    "music": ["music", "song", "musical"],
}


def expected_days() -> pd.DataFrame:
    train = pd.read_csv(METRICS_TRAIN, usecols=["subject_id", "lifelog_date"])
    test = pd.read_csv(SAMPLE_SUBMISSION, usecols=["subject_id", "lifelog_date"])
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True).drop_duplicates(["subject_id", "lifelog_date", "split"])


def coverage_by_split(frame: pd.DataFrame) -> dict[str, Any]:
    expected = expected_days()
    days = frame[["subject_id", "lifelog_date"]].drop_duplicates()
    joined = expected.merge(days.assign(covered=True), on=["subject_id", "lifelog_date"], how="left")
    joined["covered"] = joined["covered"].fillna(False)
    out: dict[str, Any] = {}
    for split, split_df in joined.groupby("split"):
        out[str(split)] = {
            "expected_days": int(len(split_df)),
            "covered_days": int(split_df["covered"].sum()),
            "coverage_rate": float(split_df["covered"].mean()),
        }
    return out


def iter_payload(payload: Any) -> list[Any]:
    if isinstance(payload, (str, bytes, dict)) or not hasattr(payload, "__iter__"):
        return []
    return list(payload)


def proximity_frame(path: Path, payload_col: str, id_key: str, strong_threshold: int) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=["subject_id", "timestamp", payload_col])
    rows = []
    token_counter: Counter[str] = Counter()
    for subject_id, timestamp, payload in raw[["subject_id", "timestamp", payload_col]].itertuples(index=False):
        items = [item for item in iter_payload(payload) if isinstance(item, dict)]
        tokens: list[str] = []
        strong = 0
        rssi_values: list[float] = []
        classes: Counter[str] = Counter()
        for item in items:
            token = str(item.get(id_key, ""))
            if token:
                tokens.append(token)
                token_counter[token] += 1
            try:
                rssi = float(item.get("rssi"))
                rssi_values.append(rssi)
                if rssi >= strong_threshold:
                    strong += 1
            except (TypeError, ValueError):
                pass
            if "device_class" in item:
                classes[str(item.get("device_class", ""))] += 1
        ts = pd.to_datetime(timestamp)
        rows.append(
            {
                "subject_id": subject_id,
                "timestamp": ts,
                "lifelog_date": ts.strftime("%Y-%m-%d"),
                "bin10": ts.floor("10min"),
                "hour": int(ts.hour),
                "scan_rows": 1,
                "device_count": len(tokens),
                "strong_count": strong,
                "rssi_mean": float(pd.Series(rssi_values).mean()) if rssi_values else 0.0,
                "unique_tokens_in_row": len(set(tokens)),
                "class_count": len(classes),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "subject_id",
                "timestamp",
                "lifelog_date",
                "bin10",
                "hour",
                "scan_rows",
                "device_count",
                "strong_count",
                "rssi_mean",
                "unique_tokens_in_row",
                "class_count",
            ]
        )
    return frame, token_counter


def ambience_scores(payload: Any) -> dict[str, float]:
    scores = dict.fromkeys(AMBIENCE_GROUPS, 0.0)
    top_label = ""
    top_score = -1.0
    for item in iter_payload(payload):
        if not hasattr(item, "__len__") or len(item) < 2:
            continue
        label = str(item[0])
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            score = 0.0
        lowered = label.lower()
        if score > top_score:
            top_label = label
            top_score = score
        for group, tokens in AMBIENCE_GROUPS.items():
            if any(token in lowered for token in tokens):
                scores[group] = max(scores[group], score)
    scores["top_label_known"] = float(bool(top_label))
    scores["top_score"] = max(top_score, 0.0)
    return scores


def ambience_frame() -> pd.DataFrame:
    raw = pd.read_parquet(AMBIENCE_PATH, columns=["subject_id", "timestamp", "m_ambience"])
    score_frame = pd.DataFrame([ambience_scores(payload) for payload in raw["m_ambience"]])
    frame = pd.concat([raw[["subject_id", "timestamp"]].copy(), score_frame], axis=1)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["lifelog_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["bin10"] = frame["timestamp"].dt.floor("10min")
    frame["hour"] = frame["timestamp"].dt.hour
    return frame


def screen_frame() -> pd.DataFrame:
    raw = pd.read_parquet(SCREEN_PATH, columns=["subject_id", "timestamp", "m_screen_use"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["lifelog_date"] = raw["timestamp"].dt.strftime("%Y-%m-%d")
    raw["bin10"] = raw["timestamp"].dt.floor("10min")
    raw["hour"] = raw["timestamp"].dt.hour
    return raw


def bin_summary() -> tuple[dict[str, Any], dict[str, Any]]:
    wifi, wifi_tokens = proximity_frame(WIFI_PATH, "m_wifi", "bssid", -65)
    ble, ble_tokens = proximity_frame(BLE_PATH, "m_ble", "address", -70)
    ambience = ambience_frame()
    screen = screen_frame()

    wifi_bin = (
        wifi.groupby(["subject_id", "lifelog_date", "bin10", "hour"])
        .agg(wifi_rows=("scan_rows", "sum"), wifi_devices=("device_count", "sum"), wifi_strong=("strong_count", "sum"))
        .reset_index()
    )
    ble_bin = (
        ble.groupby(["subject_id", "lifelog_date", "bin10", "hour"])
        .agg(ble_rows=("scan_rows", "sum"), ble_devices=("device_count", "sum"), ble_strong=("strong_count", "sum"))
        .reset_index()
    )
    ambience_bin = (
        ambience.groupby(["subject_id", "lifelog_date", "bin10", "hour"])
        .agg(
            ambience_rows=("top_score", "size"),
            speech=("speech", "max"),
            crowd=("crowd", "max"),
            inside=("inside", "max"),
            outside=("outside", "max"),
            vehicle=("vehicle", "max"),
            music=("music", "max"),
            top_score=("top_score", "max"),
        )
        .reset_index()
    )
    screen_bin = (
        screen.groupby(["subject_id", "lifelog_date", "bin10", "hour"])
        .agg(screen_rows=("m_screen_use", "size"), screen_on_mean=("m_screen_use", "mean"))
        .reset_index()
    )
    keys = ["subject_id", "lifelog_date", "bin10", "hour"]
    merged = (
        wifi_bin.merge(ble_bin, on=keys, how="outer")
        .merge(ambience_bin, on=keys, how="outer")
        .merge(screen_bin, on=keys, how="outer")
    )
    numeric_cols = [c for c in merged.columns if c not in keys]
    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
    merged["has_wifi"] = merged["wifi_rows"] > 0
    merged["has_ble"] = merged["ble_rows"] > 0
    merged["has_proximity"] = merged["has_wifi"] | merged["has_ble"]
    merged["has_ambience"] = merged["ambience_rows"] > 0
    merged["social_ambience"] = merged[["speech", "crowd"]].max(axis=1)
    merged["place_ambience"] = merged[["inside", "outside", "vehicle"]].max(axis=1)
    merged["proximity_social_cooccur"] = merged["has_proximity"] & merged["social_ambience"].ge(0.10)
    merged["proximity_place_cooccur"] = merged["has_proximity"] & merged["place_ambience"].ge(0.10)
    merged["screen_proximity_cooccur"] = merged["has_proximity"] & merged["screen_on_mean"].gt(0)
    evening = merged[merged["hour"].between(18, 23)]
    late = merged[(merged["hour"] >= 22) | (merged["hour"] <= 3)]

    coverage = {
        "wifi": coverage_by_split(wifi),
        "ble": coverage_by_split(ble),
        "ambience": coverage_by_split(ambience),
        "screen": coverage_by_split(screen),
        "proximity_any_bins": coverage_by_split(merged[merged["has_proximity"]]),
        "proximity_social_cooccur_bins": coverage_by_split(merged[merged["proximity_social_cooccur"]]),
    }
    profile = {
        "bin_count": int(len(merged)),
        "subjects": int(merged["subject_id"].nunique()),
        "wifi_bins": int(merged["has_wifi"].sum()),
        "ble_bins": int(merged["has_ble"].sum()),
        "proximity_bins": int(merged["has_proximity"].sum()),
        "ambience_bins": int(merged["has_ambience"].sum()),
        "evening_bins": int(len(evening)),
        "late_bins": int(len(late)),
        "proximity_social_cooccur_bins": int(merged["proximity_social_cooccur"].sum()),
        "proximity_place_cooccur_bins": int(merged["proximity_place_cooccur"].sum()),
        "screen_proximity_cooccur_bins": int(merged["screen_proximity_cooccur"].sum()),
        "evening_proximity_social_cooccur_bins": int(evening["proximity_social_cooccur"].sum()),
        "late_proximity_social_cooccur_bins": int(late["proximity_social_cooccur"].sum()),
        "unique_wifi_tokens_count_only": int(len(wifi_tokens)),
        "unique_ble_tokens_count_only": int(len(ble_tokens)),
        "median_wifi_devices_per_wifi_bin": float(wifi_bin["wifi_devices"].median()) if len(wifi_bin) else 0.0,
        "median_ble_devices_per_ble_bin": float(ble_bin["ble_devices"].median()) if len(ble_bin) else 0.0,
        "speech_bins_ge_010": int(merged["speech"].ge(0.10).sum()),
        "crowd_bins_ge_010": int(merged["crowd"].ge(0.10).sum()),
        "inside_outside_vehicle_bins_ge_010": int(merged[["inside", "outside", "vehicle"]].max(axis=1).ge(0.10).sum()),
    }
    return profile, coverage


def build_report() -> dict[str, Any]:
    design_report = load_json_report(DESIGN_OUTPUT.name)
    design_decision = decision(design_report)
    profile, coverage = bin_summary()
    train_cov = coverage["proximity_social_cooccur_bins"]["train"]["coverage_rate"]
    test_cov = coverage["proximity_social_cooccur_bins"]["test"]["coverage_rate"]
    feasibility_gate = bool(
        design_decision.get("design_ready") is True
        and profile["proximity_social_cooccur_bins"] >= 500
        and profile["evening_proximity_social_cooccur_bins"] >= 100
        and train_cov >= 0.40
        and test_cov >= 0.40
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_proximity_ambience_cooccurrence_metadata_smoke",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "design_report": {
            "path": rel(DESIGN_OUTPUT),
            "exists": design_report.get("exists"),
            "parse_ok": design_report.get("parse_ok"),
        },
        "bin_profile": profile,
        "coverage_by_split": coverage,
        "feasibility_thresholds": {
            "min_proximity_social_cooccur_bins": 500,
            "min_evening_proximity_social_cooccur_bins": 100,
            "min_train_day_coverage": 0.40,
            "min_test_day_coverage": 0.40,
        },
        "privacy_note": "Raw WiFi/BLE identifiers were counted transiently only; no token values are written to this report.",
        "decision": {
            "metadata_smoke_passed": feasibility_gate,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "report_only_proximity_ambience_cooccurrence_feature_spec_no_matrix"
            if feasibility_gate
            else "close_proximity_ambience_cooccurrence_lane_no_artifact",
            "next_action": (
                "Metadata smoke passed. Next may only be a report-only feature spec; do not write a feature matrix, "
                "train, create ID342, run Kaggle, or submit."
                if feasibility_gate
                else "Close this lane as insufficiently covered; do not create artifacts."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 proximity/ambience co-occurrence metadata smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Metadata smoke passed: `{dec['metadata_smoke_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
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
        "## Bin profile",
        "",
    ]
    for key, value in report["bin_profile"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Coverage by split",
            "",
            "| source | split | expected days | covered days | coverage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for source, split_map in report["coverage_by_split"].items():
        for split, values in split_map.items():
            lines.append(
                f"| `{source}` | `{split}` | {values['expected_days']} | {values['covered_days']} | {values['coverage_rate']:.4f} |"
            )
    lines.extend(["", "## Privacy note", "", report["privacy_note"], ""])
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
    if not report["decision"]["metadata_smoke_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
