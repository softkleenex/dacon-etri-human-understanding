#!/usr/bin/env python3
"""Strict artifact gate for post-ID342 feature-similarity alpha rows.

This report-only gate reviews the locally eligible rows from
`post_id342_feature_similarity_source_decomposition`. It explicitly separates
"small local ID342-neighborhood improvement" from "leaderboard-worthy or
Kaggle-worthy evidence".

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
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

from scripts.diagnose_id179_vs_id335_sequence_probe_review import align, load_artifact  # noqa: E402
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    ANCHOR_ID,
    BASE_ID,
    ID342_PUBLIC,
    PUBLIC_055,
    SOURCE_ID,
    blend_values,
)
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    DEFAULT_OUTPUT as DECOMP_PATH,
)
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_feature_similarity_artifact_gate_20260624.json"
EPS = 1e-6

MIN_LOCAL_DELTA_FOR_ARTIFACT_REVIEW = -0.0010
MAX_PUBLIC_PROXY_FOR_ARTIFACT_REVIEW = 0.5920
MAX_PUBLIC_055_GAP_FOR_REMOTE_REVIEW = 0.0400
MAX_TEST_MOVEMENT_VS_ID342 = 0.0300
MAX_TARGET_MEAN_L1_INCREASE_VS_ID342 = 0.0010


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def prediction_distribution_stats(values: np.ndarray, train: pd.DataFrame, anchor_values: np.ndarray) -> dict[str, Any]:
    train_means = train[TARGETS].astype(float).mean().to_numpy()
    pred_means = values.mean(axis=0)
    target_mean_delta = pred_means - train_means
    anchor_delta = values - anchor_values
    return {
        "prediction_mean": float(np.mean(values)),
        "prediction_std": float(np.std(values)),
        "prediction_entropy": float(
            np.mean(-(values * np.log(values + EPS) + (1.0 - values) * np.log(1.0 - values + EPS)))
        ),
        "prediction_sharpness": float(np.mean(np.abs(values - 0.5))),
        "extreme_rate_05_95": float(np.mean((values < 0.05) | (values > 0.95))),
        "target_mean_l1_to_train": float(np.mean(np.abs(target_mean_delta))),
        "target_mean_max_abs_to_train": float(np.max(np.abs(target_mean_delta))),
        "test_mean_abs_delta_vs_id342": float(np.mean(np.abs(anchor_delta))),
        "test_max_abs_delta_vs_id342": float(np.max(np.abs(anchor_delta))),
        "per_target_mean_abs_delta_vs_id342": {
            target: float(np.mean(np.abs(anchor_delta[:, idx])))
            for idx, target in enumerate(TARGETS)
        },
    }


def build_candidate_review_rows(decomp: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    base_test = align(load_artifact("test", BASE_ID), test)[TARGETS].astype(float).to_numpy()
    source_test = align(load_artifact("test", SOURCE_ID), test)[TARGETS].astype(float).to_numpy()
    anchor_test = align(load_artifact("test", ANCHOR_ID), test)[TARGETS].astype(float).to_numpy()
    anchor_stats = prediction_distribution_stats(anchor_test, train, anchor_test)
    eligible_rows = [row for row in decomp["rows"] if row.get("eligible_local_gate")]
    review_rows: list[dict[str, Any]] = []
    for row in eligible_rows[:12]:
        values = blend_values(base_test, source_test, row["alpha_by_target"])
        stats = prediction_distribution_stats(values, train, anchor_test)
        public_proxy = ID342_PUBLIC + float(row["local_delta_vs_id342"])
        checks = {
            "local_delta_material_enough": row["local_delta_vs_id342"] <= MIN_LOCAL_DELTA_FOR_ARTIFACT_REVIEW,
            "public_proxy_low_enough": public_proxy <= MAX_PUBLIC_PROXY_FOR_ARTIFACT_REVIEW,
            "public_055_gap_low_enough_for_remote": (public_proxy - PUBLIC_055) <= MAX_PUBLIC_055_GAP_FOR_REMOTE_REVIEW,
            "test_movement_vs_id342_within_ceiling": stats["test_max_abs_delta_vs_id342"] <= MAX_TEST_MOVEMENT_VS_ID342,
            "target_mean_l1_not_worse_than_id342": (
                stats["target_mean_l1_to_train"] - anchor_stats["target_mean_l1_to_train"]
            )
            <= MAX_TARGET_MEAN_L1_INCREASE_VS_ID342,
            "not_post_public_neighborhood_scan": False,
        }
        gate_pass = bool(all(checks.values()))
        review_rows.append(
            {
                "name": row["name"],
                "changed_targets": row["changed_targets"],
                "alpha_by_target": row["alpha_by_target"],
                "local_delta_vs_id342": row["local_delta_vs_id342"],
                "mean_primary_delta_vs_id342": row["mean_primary_delta_vs_id342"],
                "max_primary_delta_vs_id342": row["max_primary_delta_vs_id342"],
                "primary_wins_vs_id342": row["primary_wins_vs_id342"],
                "projected_public_proxy": public_proxy,
                "gap_to_public_055": float(public_proxy - PUBLIC_055),
                "distribution_stats": stats,
                "distribution_delta_vs_id342": {
                    "target_mean_l1_to_train_delta": float(
                        stats["target_mean_l1_to_train"] - anchor_stats["target_mean_l1_to_train"]
                    ),
                    "prediction_entropy_delta": float(stats["prediction_entropy"] - anchor_stats["prediction_entropy"]),
                    "prediction_sharpness_delta": float(
                        stats["prediction_sharpness"] - anchor_stats["prediction_sharpness"]
                    ),
                },
                "checks": checks,
                "artifact_gate_pass": gate_pass,
            }
        )
    return review_rows, anchor_stats


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    decomp_path = args.decomposition if args.decomposition.is_absolute() else ROOT / args.decomposition
    decomp = json.loads(decomp_path.read_text(encoding="utf-8"))
    review_rows, anchor_stats = build_candidate_review_rows(decomp)
    passing_rows = [row for row in review_rows if row["artifact_gate_pass"]]
    best_row = review_rows[0] if review_rows else None
    decision = {
        "reviewed_rows": int(len(review_rows)),
        "artifact_gate_pass_count": int(len(passing_rows)),
        "best_reviewed_row": best_row["name"] if best_row else None,
        "best_projected_public_proxy": best_row["projected_public_proxy"] if best_row else ID342_PUBLIC,
        "best_gap_to_public_055": best_row["gap_to_public_055"] if best_row else ID342_PUBLIC - PUBLIC_055,
        "artifact_review_allowed": bool(passing_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "reason": (
            "No reviewed row passed the strict artifact gate. The local gains are too small, the public proxy remains "
            "near ID342 and far from 0.55, and the scan is explicitly post-public ID342-neighborhood tuning."
            if not passing_rows
            else "At least one row passed mechanical gates, but this report still creates no artifact; require separate exact approval and a full decision note."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_feature_similarity_artifact_gate",
        "purpose": "Strictly gate post-ID342 feature-similarity alpha rows before any artifact/Kaggle consideration.",
        "decomposition_report": rel(decomp_path),
        "thresholds": {
            "min_local_delta_for_artifact_review": MIN_LOCAL_DELTA_FOR_ARTIFACT_REVIEW,
            "max_public_proxy_for_artifact_review": MAX_PUBLIC_PROXY_FOR_ARTIFACT_REVIEW,
            "max_public_055_gap_for_remote_review": MAX_PUBLIC_055_GAP_FOR_REMOTE_REVIEW,
            "max_test_movement_vs_id342": MAX_TEST_MOVEMENT_VS_ID342,
            "max_target_mean_l1_increase_vs_id342": MAX_TARGET_MEAN_L1_INCREASE_VS_ID342,
        },
        "id342_distribution_stats": anchor_stats,
        "review_rows": review_rows,
        "decision": decision,
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID342 feature-similarity artifact gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Reviewed Rows",
        "",
        "| rank | row | pass | local Δ342 | mean Δ | max Δ | proxy | gap 0.55 | max test Δ342 | failed checks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(report["review_rows"], start=1):
        failed = [name for name, ok in row["checks"].items() if not ok]
        lines.append(
            "| {rank} | `{name}` | `{passed}` | {local_delta:+.6f} | {mean_delta:+.6f} | {max_delta:+.6f} | {proxy:.6f} | {gap:+.6f} | {max_test:.6f} | `{failed}` |".format(
                rank=rank,
                name=row["name"],
                passed=row["artifact_gate_pass"],
                local_delta=row["local_delta_vs_id342"],
                mean_delta=row["mean_primary_delta_vs_id342"],
                max_delta=row["max_primary_delta_vs_id342"],
                proxy=row["projected_public_proxy"],
                gap=row["gap_to_public_055"],
                max_test=row["distribution_stats"]["test_max_abs_delta_vs_id342"],
                failed=",".join(failed),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Artifact gate pass count: `{decision['artifact_gate_pass_count']}`",
            f"- Best reviewed row: `{decision['best_reviewed_row']}`",
            f"- Best projected public proxy: `{decision['best_projected_public_proxy']:.9f}`",
            f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
            f"- Artifact review allowed: `{decision['artifact_review_allowed']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Reason: {decision['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decomposition", type=Path, default=DECOMP_PATH)
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
