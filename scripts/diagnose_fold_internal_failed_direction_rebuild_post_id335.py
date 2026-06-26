#!/usr/bin/env python3
"""Fold-internal failed-direction rebuild diagnostic after ID335.

This report-only diagnostic replays the ID310/ID311 aggregate-budget idea while
computing beta, projection penalties, and target choices inside each outer
training fold. It still uses pre-existing source OOF artifacts, so it is not a
true model-level nested rebuild, but it is stricter than full-OOF global scans.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from scripts.run_public_failed_direction_orthogonal_probe_experiment import (  # noqa: E402
    DEFAULT_GAMMAS,
    DEFAULT_SCALES,
    adjusted_values,
    projection_beta,
)
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
EPS = 1e-6
ANCHOR_ID = 178
FAILED_ID = 287
SOURCE_IDS = [220, 242, 243, 249, 252, 256, 268, 276, 278]
SETTINGS = [
    {"name": "id311_like_cap100_lam160", "cap": 0.10, "lambda_penalty": 1.60},
    {"name": "id310_like_cap120_lam160", "cap": 0.12, "lambda_penalty": 1.60},
    {"name": "tighter_cap060_lam120", "cap": 0.06, "lambda_penalty": 1.20},
]
NEGATIVE_REWARD_CAP = 0.02


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def load_candidates(frame: pd.DataFrame, kind: str, candidate_ids: list[int]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact(kind, candidate_id), frame) for candidate_id in candidate_ids}


def arrays_for_target(candidates: dict[int, pd.DataFrame], target: str) -> dict[int, np.ndarray]:
    return {
        candidate_id: np.clip(frame[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        for candidate_id, frame in candidates.items()
    }


def projection_global_contribution(delta: np.ndarray, failed_vector: np.ndarray, denominator: float) -> float:
    return float(delta @ failed_vector / denominator) if denominator > 0 else 0.0


def build_target_pool(
    *,
    train: pd.DataFrame,
    target: str,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    oof_arrays: dict[int, np.ndarray],
    total_denominator: float,
    scales: list[float],
    gammas: list[float],
) -> list[dict[str, Any]]:
    y_train = train[target].iloc[train_idx].astype(int).to_numpy()
    anchor = oof_arrays[ANCHOR_ID]
    failed = oof_arrays[FAILED_ID]
    failed_vector_train = failed[train_idx] - anchor[train_idx]
    pool: list[dict[str, Any]] = [
        {
            "kind": "anchor",
            "source_id": ANCHOR_ID,
            "scale": 0.0,
            "gamma": 0.0,
            "beta_to_failed": 0.0,
            "train_loss": target_loss(y_train, anchor[train_idx]),
            "train_max_abs_delta_vs_anchor": 0.0,
            "train_mean_abs_delta_vs_anchor": 0.0,
            "projection_global_contribution": 0.0,
            "valid_pred": anchor[valid_idx],
            "train_pred": anchor[train_idx],
        }
    ]
    for source_id in SOURCE_IDS:
        source = oof_arrays[source_id]
        beta = projection_beta(anchor[train_idx], source[train_idx], failed[train_idx])
        for scale in scales:
            for gamma in gammas:
                train_pred = adjusted_values(
                    anchor[train_idx], source[train_idx], failed[train_idx], scale, gamma, beta
                )
                valid_pred = adjusted_values(
                    anchor[valid_idx], source[valid_idx], failed[valid_idx], scale, gamma, beta
                )
                delta_train = train_pred - anchor[train_idx]
                pool.append(
                    {
                        "kind": "fold_internal_failed_direction",
                        "source_id": source_id,
                        "scale": float(scale),
                        "gamma": float(gamma),
                        "beta_to_failed": float(beta),
                        "train_loss": target_loss(y_train, train_pred),
                        "train_max_abs_delta_vs_anchor": float(np.max(np.abs(delta_train))),
                        "train_mean_abs_delta_vs_anchor": float(np.mean(np.abs(delta_train))),
                        "projection_global_contribution": projection_global_contribution(
                            delta_train, failed_vector_train, total_denominator
                        ),
                        "valid_pred": valid_pred,
                        "train_pred": train_pred,
                    }
                )
    return pool


def select_choice(
    pool: list[dict[str, Any]], cap: float, lambda_penalty: float, per_target_negative_cap: float
) -> dict[str, Any]:
    capped = [row for row in pool if row["train_max_abs_delta_vs_anchor"] <= cap + 1e-12]
    if not capped:
        capped = [pool[0]]
    return min(
        capped,
        key=lambda row: (
            row["train_loss"]
            + lambda_penalty * max(float(row["projection_global_contribution"]), -per_target_negative_cap),
            row["train_max_abs_delta_vs_anchor"],
        ),
    )


def run_setting(
    *,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    oof_candidates: dict[int, pd.DataFrame],
    setting: dict[str, Any],
    scales: list[float],
    gammas: list[float],
) -> dict[str, Any]:
    selected = train[["subject_id", "date"]].copy()
    for target in TARGETS:
        selected[target] = np.nan
    choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    per_target_negative_cap = NEGATIVE_REWARD_CAP / len(TARGETS)

    target_arrays = {target: arrays_for_target(oof_candidates, target) for target in TARGETS}
    for fold_idx, (raw_train_idx, raw_valid_idx) in enumerate(folds):
        train_idx = np.asarray(raw_train_idx, dtype=int)
        valid_idx = np.asarray(raw_valid_idx, dtype=int)
        total_denominator = 0.0
        for target in TARGETS:
            arr = target_arrays[target]
            failed_vector = arr[FAILED_ID][train_idx] - arr[ANCHOR_ID][train_idx]
            total_denominator += float(failed_vector @ failed_vector)
        total_denominator = total_denominator if total_denominator > 0 else 1.0

        for target in TARGETS:
            pool = build_target_pool(
                train=train,
                target=target,
                train_idx=train_idx,
                valid_idx=valid_idx,
                oof_arrays=target_arrays[target],
                total_denominator=total_denominator,
                scales=scales,
                gammas=gammas,
            )
            choice = select_choice(
                pool,
                cap=float(setting["cap"]),
                lambda_penalty=float(setting["lambda_penalty"]),
                per_target_negative_cap=per_target_negative_cap,
            )
            selected.loc[valid_idx, target] = choice["valid_pred"]
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "source_id": int(choice["source_id"]),
                    "scale": float(choice["scale"]),
                    "gamma": float(choice["gamma"]),
                    "beta_to_failed": float(choice["beta_to_failed"]),
                    "train_loss": float(choice["train_loss"]),
                    "valid_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(), choice["valid_pred"]
                    ),
                    "projection_global_contribution": float(choice["projection_global_contribution"]),
                    "train_max_abs_delta_vs_anchor": float(choice["train_max_abs_delta_vs_anchor"]),
                }
            )
    if selected[TARGETS].isna().any().any():
        raise ValueError("Fold-internal rebuild failed to fill all predictions.")
    overall, target_scores = local_scores(train, selected)
    slices = slice_summary(train, selected)
    return {
        "setting": setting,
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
            target: dict(Counter(choice["source_id"] for choice in target_choices))
            for target, target_choices in choices.items()
        },
        "choices": choices,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fold-internal failed-direction rebuild — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It recomputes beta/projection/choice inside each outer fold, but still uses existing source OOF artifacts.",
        "",
        "## Results",
        "",
        "| split | setting | overall | robust | worst target | worst subject | selected failed-source ids |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for split_name, rows in report["splits"].items():
        for row in rows:
            slices = row["slice_summary"]
            selected_ids = sorted(
                {
                    int(source_id)
                    for counts in row["choice_counts"].values()
                    for source_id, count in counts.items()
                    if int(source_id) != ANCHOR_ID and int(count) > 0
                }
            )
            lines.append(
                f"| `{split_name}` | `{row['setting']['name']}` | {row['overall']:.6f} | {row['robust_score']:.6f} | "
                f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
                f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` | `{selected_ids}` |"
            )
    lines.extend(["", "## Verdict", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, _test = load_train_test()
    candidate_ids = [ANCHOR_ID, FAILED_ID, *SOURCE_IDS]
    oof_candidates = load_candidates(train, "oof", candidate_ids)
    split_defs = {
        "subject_hole_fold_internal": make_subject_hole_folds(train, n_folds=5),
        "loso_fold_internal": make_loso_folds(train),
    }
    splits = {
        split_name: [
            run_setting(
                train=train,
                folds=folds,
                oof_candidates=oof_candidates,
                setting=setting,
                scales=DEFAULT_SCALES,
                gammas=DEFAULT_GAMMAS,
            )
            for setting in SETTINGS
        ]
        for split_name, folds in split_defs.items()
    }
    best_rows = {
        split_name: min(rows, key=lambda row: (row["overall"], row["robust_score"]))
        for split_name, rows in splits.items()
    }
    best_subject = best_rows["subject_hole_fold_internal"]
    best_loso = best_rows["loso_fold_internal"]
    promote = bool(best_subject["overall"] < 0.535 and best_loso["overall"] < 0.545)
    reason = (
        "Fold-internal recomputation preserves a meaningful part of the failed-direction signal. This is still not a promotion gate because source artifacts are pre-existing OOF products, but it justifies a true model-level fold-nested rebuild."
        if promote
        else "Fold-internal recomputation does not preserve enough robust signal to justify promotion. Keep the failed-direction family diagnostic-only."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "fold_internal_failed_direction_rebuild_post_id335",
        "anchor_id": ANCHOR_ID,
        "failed_id": FAILED_ID,
        "source_ids": SOURCE_IDS,
        "settings": SETTINGS,
        "splits": splits,
        "best_rows": best_rows,
        "decision": {
            "promote_generated_unsubmitted_artifact": False,
            "public_058_plausible": False,
            "true_model_level_rebuild_justified": promote,
            "reason": reason,
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "fold_internal_failed_direction_rebuild_post_id335_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
