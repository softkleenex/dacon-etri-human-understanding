#!/usr/bin/env python3
"""Post-ID342 S4/S3 sleep fragmentation and onset gate.

This report-only diagnostic follows the target-reallocation finding that S4
WASO remains the largest loss and S3 SOL barely moves. It retests processed
sleep features with S4/S3-specific feature families: WASO/fragmentation,
sleep-onset context, sleep HR/awake ratios, and their union.

It creates no prediction CSV, feature parquet, base prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
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
    TARGETS,
    load_processed_aligned,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_split,
    rel,
    weighted_loss,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_s4s3_sleep_fragmentation_gate_20260625.json"

FOCUS_TARGETS = ["S4", "S3"]
BLEND_WEIGHTS = [0.005, 0.01, 0.02, 0.05, 0.08]

TARGET_DOMAIN = {
    "S4": "objective WASO / wakefulness after sleep onset guideline",
    "S3": "objective sleep onset latency guideline",
}


def unique_preserve_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def numeric_present(features: pd.DataFrame, cols: list[str]) -> list[str]:
    return [
        col
        for col in cols
        if col in features.columns and pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()
    ]


def build_s4s3_families(features: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, Any]]:
    cols = list(features.columns)
    lower = {col: col.lower() for col in cols}
    waso_tokens = ("awake", "fragment", "efficiency", "waso", "wake")
    onset_tokens = ("presleep", "bed", "screen", "charging", "active", "dark", "night", "eve")
    sleep_hr_tokens = ("sleep_hr", "slp_hr", "awake_ratio", "light_ratio", "rmssd", "spike", "slp_wlight")
    waso_cols = [col for col in cols if any(token in lower[col] for token in waso_tokens)]
    onset_cols = [col for col in cols if any(token in lower[col] for token in onset_tokens)]
    sleep_hr_cols = [col for col in cols if any(token in lower[col] for token in sleep_hr_tokens)]
    core_cols = [
        "wHr_sleep_duration_awake",
        "wHr_sleep_efficiency_proxy",
        "wHr_sleep_fragmentation_index",
        "wHr_deep_sleep_ratio",
        "wHr_slp_hr_awake_ratio",
        "slp_hr_awake_ratio",
        "slp_wlight_dark",
        "slp_wlight_light",
        "screen_presleep_on",
        "screen_presleep_ratio",
        "act_presleep_active",
        "ac_presleep_charging",
    ]
    families = {
        "s4_waso_fragmentation_core": numeric_present(features, unique_preserve_order(core_cols + waso_cols)),
        "s3_onset_context": numeric_present(features, onset_cols),
        "sleep_hr_awake_light": numeric_present(features, sleep_hr_cols),
    }
    families["s4s3_sleep_union"] = numeric_present(
        features,
        unique_preserve_order(
            families["s4_waso_fragmentation_core"] + families["s3_onset_context"] + families["sleep_hr_awake_light"]
        ),
    )
    metadata = {
        "families": {
            name: {
                "feature_count": int(len(feature_cols)),
                "feature_sample": feature_cols[:20],
                "mean_missing_rate": float(features[feature_cols].isna().mean().mean()) if feature_cols else None,
            }
            for name, feature_cols in families.items()
        },
        "domain_mapping": {
            "s4_waso_fragmentation_core": "awake duration, fragmentation, efficiency, and wake/awake ratios for S4/WASO.",
            "s3_onset_context": "pre-sleep screen/activity/charging/light context as a proxy for SOL.",
            "sleep_hr_awake_light": "sleep HR, awake/light ratios, RMSSD, and sleep-light features.",
            "s4s3_sleep_union": "union of all S4/S3 sleep-fragmentation and onset-context features.",
        },
    }
    return families, metadata


def scan_family(
    train: pd.DataFrame,
    features: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    family_name: str,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not feature_cols:
        return rows
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            split_predictions: dict[str, np.ndarray] = {}
            split_coverage: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_split(features, feature_cols, y, splits[split_name], spec)
                split_predictions[split_name] = pred
                split_coverage[split_name] = int(covered)
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                wins = 0
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
                    split_anchor_losses[split_name] = anchor_loss
                    split_losses[split_name] = candidate_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                rows.append(
                    {
                        "family_name": family_name,
                        "target": target,
                        "target_domain": TARGET_DOMAIN[target],
                        "config": f"{family_name}_{target.lower()}_{spec.name}_w{weight:g}",
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_rows_by_split": split_coverage,
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "split_deltas": split_deltas,
                        "split_losses": split_losses,
                        "split_anchor_losses": split_anchor_losses,
                        "eligible": bool(
                            wins >= TARGET_WIN_FLOOR
                            and mean_delta <= TARGET_MEAN_DELTA_CEILING
                            and max_delta <= TARGET_MAX_DELTA_CEILING
                        ),
                    }
                )
    return rows


def summarize_by_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for target in FOCUS_TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        best = target_rows[0] if target_rows else None
        eligible = [row for row in target_rows if row["eligible"]]
        summary[target] = {
            "domain": TARGET_DOMAIN[target],
            "best_row": best,
            "eligible_count": len(eligible),
            "best_mean_delta": None if best is None else best["mean_delta"],
            "best_max_delta": None if best is None else best["max_delta"],
            "best_wins": None if best is None else best["wins"],
        }
    return summary


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    features, processed_metadata = load_processed_aligned(train)
    families, family_metadata = build_s4s3_families(features)
    rows: list[dict[str, Any]] = []
    for family_name, feature_cols in families.items():
        rows.extend(scan_family(train, features, splits, anchor_values, family_name, feature_cols))
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    selected_targets = sorted({row["target"] for row in eligible_rows})
    estimated_all_target_delta = (
        float(sum(row["mean_delta"] for row in eligible_rows[: len(FOCUS_TARGETS)]) / len(TARGETS))
        if eligible_rows
        else 0.0
    )
    projected_public_proxy = float(ID342_PUBLIC + estimated_all_target_delta)
    decision = {
        "eligible_row_count": int(len(eligible_rows)),
        "selected_targets": selected_targets,
        "s4_selected": "S4" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "best_row": rows[0] if rows else None,
        "target_summary": summarize_by_target(rows),
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "s4s3_sleep_fragmentation_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "If S4/S3 remain unselected, processed WASO/SOL proxy features are closed and the next route needs "
            "raw sleep-stage event reconstruction or validation-gap redesign, not another processed-column overlay."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_s4s3_sleep_fragmentation_gate",
        "purpose": "Fold-internal S4/S3 overlay scan over processed WASO/SOL sleep-fragmentation feature families.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "processed_metadata": processed_metadata,
        "family_metadata": family_metadata,
        "top_rows": rows[:32],
        "eligible_rows": eligible_rows[:32],
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
    family_metadata = report["family_metadata"]["families"]
    lines = [
        "# Post-ID342 S4/S3 sleep fragmentation gate",
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
        sample = ", ".join(f"`{col}`" for col in metadata["feature_sample"][:6])
        missing = metadata["mean_missing_rate"]
        missing_text = "null" if missing is None else f"{missing:.6f}"
        lines.append(f"| `{family_name}` | {metadata['feature_count']} | {missing_text} | {sample} |")
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| family | target | config | wins | mean delta | max delta | eligible |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["top_rows"][:16]:
        lines.append(
            f"| `{row['family_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
            f"{row['mean_delta']:.9f} | {row['max_delta']:.9f} | `{row['eligible']}` |"
        )
    lines.extend(
        [
            "",
            "## Target Summary",
            "",
            "| target | domain | eligible | best wins | best mean delta | best max delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for target, summary in decision["target_summary"].items():
        lines.append(
            f"| `{target}` | {summary['domain']} | {summary['eligible_count']} | "
            f"{summary['best_wins']} | {summary['best_mean_delta']:.9f} | {summary['best_max_delta']:.9f} |"
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
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Eligible rows: {decision['eligible_row_count']} | selected={decision['selected_targets']}")


if __name__ == "__main__":
    main()
