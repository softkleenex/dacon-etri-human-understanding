#!/usr/bin/env python3
"""Blueprint genuinely new post-ID341 source families.

This is a report-only diagnostic. It inspects raw/processed data coverage and
prior reports to choose source-family work that is not another local OOF
recombination. It trains no model, creates no submission artifact, and makes no
DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_SENSOR_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
PROCESSED_TRAIN = ROOT / "data" / "processed" / "df_train_raw.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "df_test_raw.parquet"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
EXTERNAL_DIR = ROOT / "data" / "external" / "etri_lifelog"
KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

PRIOR_REPORTS = [
    REPORTS_DIR / "post_id341_public_transfer_audit_20260618.json",
    REPORTS_DIR / "post_id341_family_priority_20260618.json",
    REPORTS_DIR / "external_etri_prior_target_reconstruction_20260616.json",
    REPORTS_DIR / "external_public05917_notebook_review_20260616.json",
]


SENSOR_ROLES = {
    "ch2025_mACStatus.parquet": "charge/bedside proxy",
    "ch2025_mActivity.parquet": "stillness/activity transitions",
    "ch2025_mAmbience.parquet": "ambient sound context",
    "ch2025_mBle.parquet": "near-device/social-place context",
    "ch2025_mGps.parquet": "mobility/place regularity",
    "ch2025_mLight.parquet": "phone ambient light/darkness",
    "ch2025_mScreenStatus.parquet": "screen on/off sleep-boundary proxy",
    "ch2025_mUsageStats.parquet": "app-use behavior and evening load",
    "ch2025_mWifi.parquet": "place/home-stability proxy",
    "ch2025_wHr.parquet": "night HR/recovery/fragmentation proxy",
    "ch2025_wLight.parquet": "watch light/darkness",
    "ch2025_wPedo.parquet": "steps/inactivity/exertion",
}


def parquet_metadata(path: Path) -> dict[str, Any]:
    meta = pq.ParquetFile(path).metadata
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(meta.num_rows),
        "columns": [meta.schema.column(i).name for i in range(meta.num_columns)],
        "role": SENSOR_ROLES.get(path.name, "unknown"),
    }


def read_basic_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def dataset_summary() -> dict[str, Any]:
    metrics = pd.read_csv(METRICS_TRAIN)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    processed_train = pd.read_parquet(PROCESSED_TRAIN)
    processed_test = pd.read_parquet(PROCESSED_TEST)
    feature_cols = [
        col for col in processed_train.columns if col not in {"subject_id", "sleep_date", "lifelog_date", *TARGETS}
    ]
    group_counts = {
        "screen": len([col for col in feature_cols if "screen" in col.lower()]),
        "activity": len([col for col in feature_cols if "act" in col.lower() or "activity" in col.lower()]),
        "pedo": len([col for col in feature_cols if "pedo" in col.lower() or "step" in col.lower()]),
        "heart_rate": len([col for col in feature_cols if "hr" in col.lower() or "whr" in col.lower()]),
        "light": len([col for col in feature_cols if "light" in col.lower()]),
        "place": len([col for col in feature_cols if any(token in col.lower() for token in ["gps", "wifi", "ble"])]),
        "usage": len([col for col in feature_cols if "usage" in col.lower() or "app" in col.lower()]),
        "demographic_calendar": len(
            [col for col in feature_cols if col.startswith("demo_") or col in {"dayofweek", "is_weekend"}]
        ),
    }
    target_prevalence = {target: float(metrics[target].mean()) for target in TARGETS}
    subject_rows = metrics.groupby("subject_id").size().to_dict()
    test_subject_rows = sample.groupby("subject_id").size().to_dict()
    return {
        "metrics_train_shape": list(metrics.shape),
        "sample_submission_shape": list(sample.shape),
        "processed_train_shape": list(processed_train.shape),
        "processed_test_shape": list(processed_test.shape),
        "n_processed_feature_columns": len(feature_cols),
        "processed_feature_group_counts": group_counts,
        "train_subject_rows": {str(k): int(v) for k, v in subject_rows.items()},
        "test_subject_rows": {str(k): int(v) for k, v in test_subject_rows.items()},
        "target_prevalence_train": target_prevalence,
        "date_ranges": {
            "train_sleep_date_min": str(metrics["sleep_date"].min()),
            "train_sleep_date_max": str(metrics["sleep_date"].max()),
            "test_sleep_date_min": str(sample["sleep_date"].min()),
            "test_sleep_date_max": str(sample["sleep_date"].max()),
        },
    }


def external_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EXTERNAL_DIR.glob("*.csv")):
        df = pd.read_csv(path, nrows=5)
        total_rows = sum(1 for _ in path.open(encoding="utf-8", errors="ignore")) - 1
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "rows": int(max(total_rows, 0)),
                "columns": list(df.columns),
            }
        )
    return rows


def prior_report_status() -> list[dict[str, Any]]:
    out = []
    for path in PRIOR_REPORTS:
        out.append({"path": str(path.relative_to(ROOT)), "exists": path.exists()})
    return out


def source_blueprints() -> list[dict[str, Any]]:
    return [
        {
            "name": "sleep_boundary_transition_v1",
            "priority": 1,
            "thesis": (
                "Infer sleep onset/wake boundary and sleep interruption structure from cross-modal transitions, "
                "rather than reusing daily aggregate OOF artifacts."
            ),
            "raw_inputs": [
                "mScreenStatus",
                "mActivity",
                "mACStatus",
                "mLight",
                "wLight",
                "wHr",
                "wPedo",
            ],
            "candidate_features": [
                "last screen-on before dark/still interval",
                "first screen-on after long still interval",
                "longest nightly still+dark+low-step segment",
                "charge-start/charge-end alignment with inferred sleep window",
                "HR dip timing and night HR recovery slope",
                "within-subject deviation of inferred bedtime/waketime/duration",
                "night interruption count from screen/activity/HR spikes",
            ],
            "target_focus": ["S1", "S2", "S3", "S4", "Q1"],
            "why_new": (
                "Processed features contain many daily aggregates and some slp_* summaries, but this lane explicitly "
                "models event boundaries and transitions with fold-internal per-subject baselines."
            ),
            "validation_contract": [
                "fit transition thresholds inside training folds only",
                "compare against ID178/ID341 on frozen ID340 split manifest",
                "no artifact unless it beats ID341 proxy and passes post-ID341 transfer audit",
            ],
            "compute": "Kaggle CPU recommended after local metadata smoke; no GPU required",
        },
        {
            "name": "evening_load_recovery_v1",
            "priority": 2,
            "thesis": (
                "Represent the day as load followed by recovery: daytime exertion/mobility/social/app load, "
                "then evening screen/light/HR recovery before sleep."
            ),
            "raw_inputs": ["wPedo", "mGps", "mUsageStats", "mAmbience", "mScreenStatus", "wHr", "mLight", "wLight"],
            "candidate_features": [
                "daytime step and movement intensity percentiles",
                "evening-to-day load ratio",
                "late app/session entropy and dopamine-app buckets",
                "ambient speech/music/silence evening mix",
                "night HR minus day HR recovery delta",
                "post-heavy-day next-sleep interaction terms",
            ],
            "target_focus": ["Q2", "Q3", "S2", "S4"],
            "why_new": (
                "External/paper guidance says subjective fatigue/stress are daily-experience outcomes; this lane tests "
                "a causal-ish load→recovery representation instead of target-wise prior shrinkage."
            ),
            "validation_contract": [
                "group-safe feature normalization by subject inside folds",
                "single-family model first, no stacking",
                "slice check on high-activity and high-screen evenings",
            ],
            "compute": "local light scan possible; Kaggle for repeated folds",
        },
        {
            "name": "routine_stability_place_entropy_v1",
            "priority": 3,
            "thesis": (
                "Capture circadian regularity and place/social entropy across adjacent days; public split may reward "
                "stable subject-specific routine rather than local OOF frontier moves."
            ),
            "raw_inputs": ["mWifi", "mBle", "mGps", "mUsageStats", "mActivity", "mScreenStatus"],
            "candidate_features": [
                "home-stay consistency from Wi-Fi/GPS proxies",
                "BLE/Wi-Fi novelty and transition entropy",
                "weekday-adjusted app/screen routine distance",
                "rolling 3/7-day deviations from subject routine",
                "previous-day and next-day smoothness features available at inference without labels",
            ],
            "target_focus": ["Q1", "Q2", "Q3", "S3"],
            "why_new": (
                "Uses unlabeled temporal continuity and raw place/context logs; avoids direct external-target prevalence transfer."
            ),
            "validation_contract": [
                "chronological and subject-hole lenses both required",
                "no use of test-label-derived anchors",
                "must improve transfer proxy without moving far from ID178/ID341 unless source is independently strong",
            ],
            "compute": "Kaggle CPU if raw event expansion becomes memory-heavy",
        },
        {
            "name": "external_prior_regularizer_v2",
            "priority": 4,
            "thesis": "Use older ETRI data only to regularize feature transformations and monotonic priors, not as labeled rows.",
            "raw_inputs": ["external etri_lifelog summary CSVs"],
            "candidate_features": [
                "sleep-score range sanity checks",
                "survey scale behavior priors",
                "feature bucket boundaries for duration/wake/activity only",
            ],
            "target_focus": ["Q2", "S2"],
            "why_new": (
                "Prior external attempts showed direct prevalence mismatch, especially S1/S3/S4; keep it as weak regularizer only."
            ),
            "validation_contract": [
                "never concatenate external rows into current train",
                "document every external-derived transform in submission notes",
                "drop if frozen split gate selects base for all targets again",
            ],
            "compute": "local only",
        },
    ]


def build_report() -> dict[str, Any]:
    sensor_meta = [parquet_metadata(path) for path in sorted(RAW_SENSOR_DIR.glob("*.parquet"))]
    data_summary = dataset_summary()
    blueprints = source_blueprints()
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_new_source_blueprint",
        "purpose": "Choose non-OOF-recombination source-family work after ID341 transfer audit blocked current candidates.",
        "prior_report_status": prior_report_status(),
        "raw_sensor_metadata": sensor_meta,
        "dataset_summary": data_summary,
        "external_summary": external_summary(),
        "source_blueprints": blueprints,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "top_new_source_family": blueprints[0]["name"],
            "next_action": (
                "Implement a report-only Kaggle/local smoke for sleep_boundary_transition_v1 feature generation and frozen-split "
                "validation. Do not create a submission artifact unless it beats ID341 proxy under the post-ID341 transfer audit."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    summary = report["dataset_summary"]
    lines = [
        "# Post-ID341 new source blueprint",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.",
        "",
        "## Why this exists",
        "",
        "The post-ID341 transfer audit blocked the current candidate set: local CV is a weak selector and no current row passes the Public `0.58` gate. The next useful work must be a genuinely new source family or a clean upstream rebuild, not more OOF recombination.",
        "",
        "## Data coverage read",
        "",
        f"- Train labels: `{summary['metrics_train_shape']}`; sample/test rows: `{summary['sample_submission_shape']}`",
        f"- Processed train/test: `{summary['processed_train_shape']}` / `{summary['processed_test_shape']}`",
        f"- Processed feature columns: `{summary['n_processed_feature_columns']}`",
        f"- Train sleep dates: `{summary['date_ranges']['train_sleep_date_min']}` to `{summary['date_ranges']['train_sleep_date_max']}`",
        f"- Test sleep dates: `{summary['date_ranges']['test_sleep_date_min']}` to `{summary['date_ranges']['test_sleep_date_max']}`",
        "",
        "Processed feature group counts:",
        "",
    ]
    for key, value in summary["processed_feature_group_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "Target prevalence in train:", ""])
    for target, value in summary["target_prevalence_train"].items():
        lines.append(f"- `{target}`: {value:.6f}")
    lines.extend(["", "Raw sensor metadata:", "", "| file | rows | role |", "|---|---:|---|"])
    for row in report["raw_sensor_metadata"]:
        lines.append(f"| `{row['file']}` | {row['rows']} | {row['role']} |")
    lines.extend(["", "## Prioritized new source families", ""])
    for item in report["source_blueprints"]:
        lines.extend(
            [
                f"### {item['priority']}. `{item['name']}`",
                "",
                f"Thesis: {item['thesis']}",
                "",
                f"Targets: `{', '.join(item['target_focus'])}`",
                "",
                f"Why new: {item['why_new']}",
                "",
                "Candidate features:",
            ]
        )
        lines.extend(f"- {feature}" for feature in item["candidate_features"])
        lines.extend(["", "Validation contract:"])
        lines.extend(f"- {rule}" for rule in item["validation_contract"])
        lines.extend(["", f"Compute: {item['compute']}", ""])
    lines.extend(
        [
            "## Decision",
            "",
            f"- Top new source family: `{report['decision']['top_new_source_family']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
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
        default=REPORTS_DIR / "post_id341_new_source_blueprint_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
