#!/usr/bin/env python3
"""ID342-aware sleep-boundary Q3/S3 rescue smoke.

This report-only diagnostic reuses the existing sleep-boundary transition
features because S3 is the sleep-onset-latency target and Q3 can plausibly
respond to pre-sleep physiological strain. It evaluates only Q3/S3 against the
current ID342 OOF anchor under the frozen primary split lenses.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
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

from scripts.diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke import (  # noqa: E402
    EPS,
    FOCUS_TARGETS,
    ID342_PUBLIC,
    PRIMARY_SPLITS,
    PUBLIC_055,
    TARGETS,
    weighted_split_loss,
)
from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    MODEL_SPECS,
    build_sleep_boundary_features,
    load_reference_values,
    load_split_manifest,
    predict_target_for_split,
)
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_sleep_boundary_q3s3_rescue_smoke_20260624.json"
SEQUENCE_DISTANCE_SMOKE_PATH = REPORTS_DIR / "post_id342_sequence_summary_distance_kernel_smoke_20260624.json"

BLEND_WEIGHTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075

FEATURE_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all": (),
    "sol_light_screen_motion": ("screen", "charge", "light", "pedo", "low_step", "hr"),
    "stress_hr_activity_screen": ("hr", "activity", "screen", "pedo", "charge"),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def family_feature_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for family, keywords in FEATURE_FAMILY_KEYWORDS.items():
        if family == "all":
            families[family] = list(feature_cols)
        else:
            families[family] = [col for col in feature_cols if any(keyword in col for keyword in keywords)]
    return {family: cols for family, cols in families.items() if cols}


def scan_target(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_families: dict[str, list[str]],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    target: str,
) -> dict[str, Any]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for family, feature_cols in feature_families.items():
        for spec in MODEL_SPECS:
            split_source_predictions: dict[str, np.ndarray] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_target_for_split(
                    features_train,
                    feature_cols,
                    y,
                    splits[split_name],
                    spec,
                )
                if covered <= 0:
                    raise ValueError(f"No validation rows covered for {target} {family} {spec.name} {split_name}")
                split_source_predictions[split_name] = np.where(
                    np.isnan(pred),
                    anchor_values[:, target_idx],
                    pred,
                )
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                wins = 0
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    source_pred = split_source_predictions[split_name]
                    blended = np.clip((1.0 - weight) * anchor_values[:, target_idx] + weight * source_pred, EPS, 1 - EPS)
                    anchor_loss = weighted_split_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_split_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_anchor_losses[split_name] = anchor_loss
                    split_losses[split_name] = candidate_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                rows.append(
                    {
                        "target": target,
                        "config": f"{target}_id342_sleep_boundary_{family}_{spec.name}_w{weight:g}",
                        "feature_family": family,
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
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
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    return {
        "target": target,
        "best": rows[0] if rows else None,
        "eligible_count": int(len(eligible_rows)),
        "selected": eligible_rows[0] if eligible_rows else None,
        "top_rows": rows[:8],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    features, feature_coverage = build_sleep_boundary_features(train, test)
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    feature_cols = [col for col in features.columns if col.startswith("sbt_")]
    feature_families = family_feature_columns(feature_cols)
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    anchor_values = load_reference_values(train, 342)
    target_reports = {
        target: scan_target(train, features_train, feature_families, splits, anchor_values, target)
        for target in FOCUS_TARGETS
    }
    selected_targets = [
        target
        for target, target_report in target_reports.items()
        if target_report.get("selected") is not None
    ]
    selected_rows = {
        target: target_report["selected"]
        for target, target_report in target_reports.items()
        if target_report.get("selected") is not None
    }
    if selected_rows:
        mean_primary_delta = float(np.mean([row["mean_delta"] for row in selected_rows.values()]))
        max_primary_delta = float(np.max([row["max_delta"] for row in selected_rows.values()]))
    else:
        mean_primary_delta = 0.0
        max_primary_delta = 0.0
    partial_signal_rows = {
        target: [
            row
            for row in target_report["top_rows"]
            if row["mean_delta"] < 0 and row["wins"] >= 3 and row["max_delta"] <= 0.003
        ][:3]
        for target, target_report in target_reports.items()
    }
    partial_signal_targets = [target for target, rows in partial_signal_rows.items() if rows]
    projected_public_proxy = ID342_PUBLIC + mean_primary_delta
    final_gate_pass = bool(
        selected_targets
        and len(selected_targets) == len(FOCUS_TARGETS)
        and mean_primary_delta < -0.0005
        and max_primary_delta <= TARGET_MAX_DELTA_CEILING
        and projected_public_proxy < ID342_PUBLIC
    )
    decision = {
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "all_focus_targets_selected": len(selected_targets) == len(FOCUS_TARGETS),
        "mean_primary_delta_vs_id342": mean_primary_delta,
        "max_primary_delta_vs_id342": max_primary_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "sleep_boundary_has_q3s3_signal": bool(selected_targets),
        "partial_signal_targets": partial_signal_targets,
        "q3_partial_signal_unstable": "Q3" in partial_signal_targets and "Q3" not in selected_targets,
        "s3_partial_signal_unstable": "S3" in partial_signal_targets and "S3" not in selected_targets,
        "final_sleep_boundary_gate_pass": final_gate_pass,
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": bool(final_gate_pass),
        "run_kaggle_now": bool(final_gate_pass),
        "next_action": (
            "If this gate fails, keep sleep-boundary/SOL features as closed ID342-aware Q3/S3 evidence. "
            "If both targets pass, use Kaggle no-submit for parity only."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_sleep_boundary_q3s3_rescue_smoke",
        "purpose": "Re-test direct sleep-boundary/SOL features against ID342 for Q3/S3 only.",
        "sequence_distance_smoke_report": rel(SEQUENCE_DISTANCE_SMOKE_PATH),
        "anchor": "ID342",
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": list(BLEND_WEIGHTS),
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "feature_contract": {
            "domain_read": (
                "S3 is sleep-onset-latency guideline adherence; Q3 is pre-sleep stress. "
                "This smoke emphasizes screen-off, charging, low movement, light drop, and heart-rate boundary features."
            ),
            "coverage": feature_coverage,
            "feature_families": {
                family: {"feature_count": int(len(cols)), "example_features": cols[:12]}
                for family, cols in feature_families.items()
            },
        },
        "target_reports": target_reports,
        "partial_signal_rows": partial_signal_rows,
        "decision": decision,
        "in_memory_validation_predictions_only": True,
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    coverage = report["feature_contract"]["coverage"]
    lines = [
        "# Post-ID342 sleep-boundary Q3/S3 rescue smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Feature Contract",
        "",
        f"- Domain read: {report['feature_contract']['domain_read']}",
        f"- Rows: `{coverage['n_rows']}` total / `{coverage['n_train_rows']}` train / `{coverage['n_test_rows']}` test",
        f"- Feature columns: `{coverage['n_feature_columns']}`",
        f"- Train/test missing mean: `{coverage['train_missing_mean']:.4f}` / `{coverage['test_missing_mean']:.4f}`",
        "",
        "## Target Results",
        "",
        "| target | selected | best config | wins | mean delta | max delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for target, target_report in report["target_reports"].items():
        best = target_report["best"] or {}
        selected = target_report["selected"] is not None
        lines.append(
            "| {target} | `{selected}` | `{config}` | {wins} | {mean_delta:+.6f} | {max_delta:+.6f} |".format(
                target=target,
                selected=selected,
                config=best.get("config"),
                wins=int(best.get("wins", 0)),
                mean_delta=float(best.get("mean_delta", 0.0)),
                max_delta=float(best.get("max_delta", 0.0)),
            )
        )
    lines.extend(["", "## Top Rows", ""])
    for target, target_report in report["target_reports"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | config | wins | mean delta | max delta |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(target_report["top_rows"][:5], start=1):
            lines.append(
                "| {rank} | `{config}` | {wins} | {mean_delta:+.6f} | {max_delta:+.6f} |".format(
                    rank=rank,
                    config=row["config"],
                    wins=row["wins"],
                    mean_delta=row["mean_delta"],
                    max_delta=row["max_delta"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Sleep boundary has Q3/S3 signal: `{decision['sleep_boundary_has_q3s3_signal']}`",
            f"- Partial signal targets: `{decision['partial_signal_targets']}`",
            f"- Final gate pass: `{decision['final_sleep_boundary_gate_pass']}`",
            f"- Projected public proxy from ID342: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55 after proxy: `{decision['gap_to_public_055_after_proxy']:+.9f}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)} and {rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
