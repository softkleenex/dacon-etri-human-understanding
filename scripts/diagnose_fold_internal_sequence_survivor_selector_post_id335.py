#!/usr/bin/env python3
"""Fold-internal selector across sequence survivors after excluding ID242.

This report-only diagnostic selects among fixed sequence-survivor OOF artifacts
inside each outer fold and target. It tests whether lower-local sequence
descendants have a stable target-wise signal after selection is nested. Source
artifacts are still pre-existing OOF products, so this is not a true model-level
nested rebuild and must not create a submission artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    TARGETS,
    align,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.diagnose_strict_demographic_feature_audit import make_loso_folds  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
EPS = 1e-6
ANCHOR_ID = 178
FAILED_ID = 287
DEFAULT_CANDIDATE_IDS = [178, 179, 190, 191, 193, 177, 198, 200, 203, 206, 334, 335]
STRESS_CANDIDATE_IDS = [*DEFAULT_CANDIDATE_IDS, 180]


@dataclass(frozen=True)
class Scenario:
    name: str
    candidate_ids: tuple[int, ...]
    max_target_delta_vs_178: float | None
    max_target_failed_projection: float | None
    movement_penalty: float
    failed_projection_penalty: float


SCENARIOS = [
    Scenario("id179_anchor_only", (178, 179), 0.06, 0.02, 0.10, 0.80),
    Scenario("sequence_local_only", tuple(DEFAULT_CANDIDATE_IDS), None, None, 0.00, 0.00),
    Scenario("sequence_cap008_penalty", tuple(DEFAULT_CANDIDATE_IDS), 0.08, 0.015, 0.15, 1.00),
    Scenario("sequence_cap012_penalty", tuple(DEFAULT_CANDIDATE_IDS), 0.12, 0.020, 0.12, 0.80),
    Scenario("sequence_cap014_penalty", tuple(DEFAULT_CANDIDATE_IDS), 0.14, 0.025, 0.10, 0.65),
    Scenario("sequence_with_id180_stress", tuple(STRESS_CANDIDATE_IDS), 0.14, 0.030, 0.12, 1.20),
]


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def load_candidates(frame: pd.DataFrame, kind: str, candidate_ids: list[int]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact(kind, candidate_id), frame) for candidate_id in candidate_ids}


def target_geometry(
    *,
    candidate_test: pd.DataFrame,
    anchor_test: pd.DataFrame,
    failed_test: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    cand = candidate_test[target].astype(float).to_numpy()
    anchor = anchor_test[target].astype(float).to_numpy()
    failed = failed_test[target].astype(float).to_numpy()
    delta = cand - anchor
    failed_direction = failed - anchor
    denom = float(failed_direction @ failed_direction)
    return {
        "mean_delta_vs_178": float(np.mean(np.abs(delta))),
        "max_delta_vs_178": float(np.max(np.abs(delta))),
        "projection_failed": float(delta @ failed_direction / denom) if denom > 0 else 0.0,
    }


def candidate_target_metrics(test_candidates: dict[int, pd.DataFrame]) -> dict[int, dict[str, dict[str, float]]]:
    anchor_test = test_candidates[ANCHOR_ID]
    failed_test = test_candidates[FAILED_ID]
    metrics: dict[int, dict[str, dict[str, float]]] = {}
    for candidate_id, candidate_test in test_candidates.items():
        if candidate_id == FAILED_ID:
            continue
        metrics[candidate_id] = {
            target: target_geometry(
                candidate_test=candidate_test,
                anchor_test=anchor_test,
                failed_test=failed_test,
                target=target,
            )
            for target in TARGETS
        }
    return metrics


def arrays_for_target(candidates: dict[int, pd.DataFrame], target: str) -> dict[int, np.ndarray]:
    return {
        candidate_id: np.clip(frame[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        for candidate_id, frame in candidates.items()
    }


def build_pool(
    *,
    train: pd.DataFrame,
    target: str,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    oof_arrays: dict[int, np.ndarray],
    metrics: dict[int, dict[str, dict[str, float]]],
    scenario: Scenario,
) -> list[dict[str, Any]]:
    y_train = train[target].iloc[train_idx].astype(int).to_numpy()
    pool: list[dict[str, Any]] = []
    for candidate_id in scenario.candidate_ids:
        metric = metrics[candidate_id][target]
        if scenario.max_target_delta_vs_178 is not None:
            if metric["max_delta_vs_178"] > scenario.max_target_delta_vs_178 + 1e-12:
                continue
        if scenario.max_target_failed_projection is not None:
            if metric["projection_failed"] > scenario.max_target_failed_projection + 1e-12:
                continue
        pred_train = oof_arrays[candidate_id][train_idx]
        loss = target_loss(y_train, pred_train)
        objective = (
            loss
            + scenario.movement_penalty * metric["max_delta_vs_178"]
            + scenario.failed_projection_penalty * max(0.0, metric["projection_failed"])
        )
        pool.append(
            {
                "candidate_id": int(candidate_id),
                "train_loss": float(loss),
                "objective": float(objective),
                "target_max_delta_vs_178": metric["max_delta_vs_178"],
                "target_mean_delta_vs_178": metric["mean_delta_vs_178"],
                "target_projection_failed": metric["projection_failed"],
                "valid_pred": oof_arrays[candidate_id][valid_idx],
            }
        )
    if pool:
        return pool
    anchor_metric = metrics[ANCHOR_ID][target]
    return [
        {
            "candidate_id": ANCHOR_ID,
            "train_loss": target_loss(y_train, oof_arrays[ANCHOR_ID][train_idx]),
            "objective": target_loss(y_train, oof_arrays[ANCHOR_ID][train_idx]),
            "target_max_delta_vs_178": anchor_metric["max_delta_vs_178"],
            "target_mean_delta_vs_178": anchor_metric["mean_delta_vs_178"],
            "target_projection_failed": anchor_metric["projection_failed"],
            "valid_pred": oof_arrays[ANCHOR_ID][valid_idx],
        }
    ]


def run_scenario(
    *,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    oof_candidates: dict[int, pd.DataFrame],
    metrics: dict[int, dict[str, dict[str, float]]],
    scenario: Scenario,
) -> dict[str, Any]:
    selected = train[["subject_id", "date"]].copy()
    for target in TARGETS:
        selected[target] = np.nan
    choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    target_arrays = {target: arrays_for_target(oof_candidates, target) for target in TARGETS}

    for fold_idx, (raw_train_idx, raw_valid_idx) in enumerate(folds):
        train_idx = np.asarray(raw_train_idx, dtype=int)
        valid_idx = np.asarray(raw_valid_idx, dtype=int)
        for target in TARGETS:
            pool = build_pool(
                train=train,
                target=target,
                train_idx=train_idx,
                valid_idx=valid_idx,
                oof_arrays=target_arrays[target],
                metrics=metrics,
                scenario=scenario,
            )
            choice = min(
                pool,
                key=lambda row: (
                    row["objective"],
                    row["train_loss"],
                    row["target_max_delta_vs_178"],
                    max(0.0, row["target_projection_failed"]),
                ),
            )
            selected.loc[valid_idx, target] = choice["valid_pred"]
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "candidate_id": int(choice["candidate_id"]),
                    "train_loss": float(choice["train_loss"]),
                    "valid_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(), choice["valid_pred"]
                    ),
                    "objective": float(choice["objective"]),
                    "target_max_delta_vs_178": float(choice["target_max_delta_vs_178"]),
                    "target_projection_failed": float(choice["target_projection_failed"]),
                }
            )

    if selected[TARGETS].isna().any().any():
        raise ValueError("Fold-internal sequence selector failed to fill all predictions.")
    overall, target_scores = local_scores(train, selected)
    slices = slice_summary(train, selected)
    return {
        "scenario": {
            "name": scenario.name,
            "candidate_ids": list(scenario.candidate_ids),
            "max_target_delta_vs_178": scenario.max_target_delta_vs_178,
            "max_target_failed_projection": scenario.max_target_failed_projection,
            "movement_penalty": scenario.movement_penalty,
            "failed_projection_penalty": scenario.failed_projection_penalty,
        },
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "choice_counts": {
            target: dict(Counter(choice["candidate_id"] for choice in target_choices))
            for target, target_choices in choices.items()
        },
        "choices": choices,
    }


def best_rows(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: min(rows, key=lambda row: (row["overall"], row["robust_score"]))
        for split_name, rows in results.items()
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fold-internal sequence survivor selector — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It nests only the target-wise choice among existing OOF artifacts; source artifacts remain non-nested.",
        "",
        "## Results",
        "",
        "| split | scenario | overall | robust | worst target | worst subject |",
        "|---|---|---:|---:|---|---|",
    ]
    for split_name, rows in report["splits"].items():
        for row in rows:
            slices = row["slice_summary"]
            lines.append(
                f"| `{split_name}` | `{row['scenario']['name']}` | {row['overall']:.6f} | "
                f"{row['robust_score']:.6f} | {slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
                f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` |"
            )
    lines.extend(
        [
            "",
            "## Best rows and selected IDs",
            "",
            "| split | best scenario | overall | robust | selected IDs |",
            "|---|---|---:|---:|---|",
        ]
    )
    for split_name, row in report["best_rows"].items():
        selected_ids = sorted(
            {int(candidate_id) for target_counts in row["choice_counts"].values() for candidate_id in target_counts}
        )
        lines.append(
            f"| `{split_name}` | `{row['scenario']['name']}` | {row['overall']:.6f} | "
            f"{row['robust_score']:.6f} | `{selected_ids}` |"
        )
    lines.extend(
        [
            "",
            "## Choice counts for best rows",
            "",
        ]
    )
    for split_name, row in report["best_rows"].items():
        lines.extend([f"### {split_name}", ""])
        for target, counts in row["choice_counts"].items():
            lines.append(f"- `{target}`: `{counts}`")
        lines.append("")
    lines.extend(["## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, test = load_train_test()
    candidate_ids = sorted({FAILED_ID, *STRESS_CANDIDATE_IDS})
    oof_candidates = load_candidates(train, "oof", candidate_ids)
    test_candidates = load_candidates(test, "test", candidate_ids)
    metrics = candidate_target_metrics(test_candidates)
    splits = {
        "subject_hole_fold_internal": make_subject_hole_folds(train),
        "loso_fold_internal": make_loso_folds(train),
    }
    results = {
        split_name: [
            run_scenario(
                train=train,
                folds=folds,
                oof_candidates=oof_candidates,
                metrics=metrics,
                scenario=scenario,
            )
            for scenario in SCENARIOS
        ]
        for split_name, folds in splits.items()
    }
    best = best_rows(results)
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "fold_internal_sequence_survivor_selector_post_id335",
        "candidate_ids": sorted(set(STRESS_CANDIDATE_IDS)),
        "source_artifacts_are_preexisting": True,
        "target_geometry_uses_unlabeled_test_predictions": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "splits": results,
        "best_rows": best,
        "decision": {
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "reason": (
                "Fold-internal target selection can test whether sequence descendants have stable target-wise signal, "
                "but it still selects among pre-existing non-nested OOF artifacts and uses unlabeled test-geometry "
                "risk controls. Treat this as a map for future source generation, not as a submission gate."
            ),
        },
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "fold_internal_sequence_survivor_selector_post_id335_20260617.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    for split_name, row in report["best_rows"].items():
        print(f"{split_name}: {row['scenario']['name']} overall={row['overall']:.6f} robust={row['robust_score']:.6f}")
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
