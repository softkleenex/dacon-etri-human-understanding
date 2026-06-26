#!/usr/bin/env python3
"""ID342-aware processed sleep temporal-deviation gate for Q3/S3.

This report-only diagnostic extends the processed sleep-proxy gate by asking
whether personal lag, past rolling baseline, and deviation-from-baseline
signals open Q3/S3 against ID342.

It creates no prediction CSV, feature parquet, base prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id342_processed_sleep_proxy_gate import (  # noqa: E402,I001
    ANCHOR_ID,
    BLEND_WEIGHTS,
    FOCUS_TARGETS,
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
    rel,
    scan_family,
    sleep_proxy_columns,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_processed_sleep_temporal_deviation_gate_20260624.json"


def build_temporal_features(
    features: pd.DataFrame,
    base_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    sorted_features = features.sort_values(["subject_id", "date"]).reset_index(drop=True)
    pieces: dict[str, pd.Series] = {}
    grouped = sorted_features.groupby("subject_id", sort=False)
    for col in base_cols:
        lag1 = grouped[col].shift(1)
        shifted = grouped[col].shift(1)
        rolling = shifted.groupby(sorted_features["subject_id"], sort=False).rolling(3, min_periods=1)
        roll3_mean = rolling.mean().reset_index(level=0, drop=True)
        roll3_std = rolling.std(ddof=0).reset_index(level=0, drop=True)
        pieces[f"{col}__lag1"] = lag1
        pieces[f"{col}__roll3_mean_past"] = roll3_mean
        pieces[f"{col}__roll3_std_past"] = roll3_std
        pieces[f"{col}__delta_lag1"] = sorted_features[col] - lag1
        pieces[f"{col}__dev_roll3_mean"] = sorted_features[col] - roll3_mean
    engineered = pd.concat(
        [sorted_features[["subject_id", "date"]].reset_index(drop=True), pd.DataFrame(pieces)],
        axis=1,
    )
    past_cols = [col for col in engineered.columns if col.endswith(("__lag1", "__roll3_mean_past", "__roll3_std_past"))]
    deviation_cols = [col for col in engineered.columns if col.endswith(("__delta_lag1", "__dev_roll3_mean"))]
    families = {
        "sleep_temporal_past_only": past_cols,
        "sleep_temporal_deviation": deviation_cols,
        "sleep_temporal_all": past_cols + deviation_cols,
    }
    metadata = {
        "base_feature_count": int(len(base_cols)),
        "families": {
            name: {
                "feature_count": int(len(cols)),
                "feature_sample": cols[:18],
                "mean_missing_rate": float(engineered[cols].isna().mean().mean()) if cols else None,
            }
            for name, cols in families.items()
        },
        "construction": {
            "lag1": "Previous available same-subject processed sleep-proxy value.",
            "roll3_mean_past": "Past-only same-subject rolling mean over up to 3 previous rows.",
            "roll3_std_past": "Past-only same-subject rolling standard deviation over up to 3 previous rows.",
            "delta_lag1": "Current processed sleep-proxy value minus lag1.",
            "dev_roll3_mean": "Current processed sleep-proxy value minus past rolling mean.",
        },
    }
    return engineered, families, metadata


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    features, processed_metadata = load_processed_aligned(train)
    proxy_families, proxy_metadata = sleep_proxy_columns(features)
    temporal_features, temporal_families, temporal_metadata = build_temporal_features(
        features,
        proxy_families["sleep_proxy"],
    )
    rows: list[dict[str, Any]] = []
    for family_name, feature_cols in temporal_families.items():
        rows.extend(scan_family(train, temporal_features, splits, anchor_values, family_name, feature_cols))
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    selected_targets = sorted({row["target"] for row in eligible_rows})
    estimated_all_target_delta = (
        float(sum(row["mean_delta"] for row in eligible_rows[: len(FOCUS_TARGETS)]) / 7) if eligible_rows else 0.0
    )
    projected_public_proxy = float(ID342_PUBLIC + estimated_all_target_delta)
    decision = {
        "eligible_row_count": int(len(eligible_rows)),
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "best_row": rows[0] if rows else None,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "processed_sleep_temporal_deviation_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Past-only and deviation-from-personal-sleep-baseline features do not open Q3/S3 against ID342. "
            "Keep this as negative domain evidence and do not escalate to Kaggle, artifacts, or DACON."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_processed_sleep_temporal_deviation_gate",
        "purpose": "Fold-internal Q3/S3 overlay scan over processed sleep lag and baseline-deviation features.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [spec.__dict__ for spec in MODEL_SPECS],
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "processed_metadata": processed_metadata,
        "proxy_family_metadata": proxy_metadata,
        "temporal_metadata": temporal_metadata,
        "top_rows": rows[:24],
        "eligible_rows": eligible_rows[:24],
        "decision": decision,
        "no_predictions_created": True,
        "no_new_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    family_metadata = report["temporal_metadata"]["families"]
    lines = [
        "# Post-ID342 processed sleep temporal-deviation gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Feature Families",
        "",
        "| family | features | mean missing rate | sample |",
        "|---|---:|---:|---|",
    ]
    for family_name, metadata in family_metadata.items():
        sample = ", ".join(f"`{col}`" for col in metadata["feature_sample"][:5])
        lines.append(
            f"| `{family_name}` | {metadata['feature_count']} | {metadata['mean_missing_rate']:.6f} | {sample} |"
        )
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| family | target | config | wins | mean delta | max delta | eligible |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["top_rows"][:12]:
        lines.append(
            f"| `{row['family_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
            f"{row['mean_delta']:.9f} | {row['max_delta']:.9f} | `{row['eligible']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Eligible rows: `{decision['eligible_row_count']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Projected Public proxy: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55: `{decision['gap_to_public_055_after_proxy']:.9f}`",
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
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Eligible rows: {decision['eligible_row_count']} | selected={decision['selected_targets']}")


if __name__ == "__main__":
    main()
