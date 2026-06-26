#!/usr/bin/env python3
"""Selection-nested diagnostic across existing candidate OOF artifacts.

This is intentionally weaker than a true fold-nested modeling pipeline: the
candidate artifacts themselves may still be full-OOF/non-nested. It only nests
the *choice among fixed candidate artifacts* inside outer folds, so it is a
screening diagnostic, not promotion evidence.
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
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
EPS = 1e-6
DEFAULT_CANDIDATE_IDS = [178, 179, 334, 335, 295, 297, 304, 305, 306, 307, 310, 311]


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def load_oof_candidates(train: pd.DataFrame, candidate_ids: list[int]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact("oof", candidate_id), train) for candidate_id in candidate_ids}


def load_test_candidates(test: pd.DataFrame, candidate_ids: list[int]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in candidate_ids}


def candidate_target_loss(train: pd.DataFrame, pred: pd.DataFrame, target: str, idx: np.ndarray) -> float:
    return target_loss(train[target].iloc[idx].astype(int).to_numpy(), pred[target].iloc[idx].astype(float).to_numpy())


def run_outer_selector(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    selected = train[["subject_id", "date"]].copy()
    choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for target in TARGETS:
        selected[target] = np.nan

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx, dtype=int)
        valid_idx = np.asarray(valid_idx, dtype=int)
        for target in TARGETS:
            losses = {
                candidate_id: candidate_target_loss(train, pred, target, train_idx)
                for candidate_id, pred in candidates.items()
            }
            best_id, best_loss = min(losses.items(), key=lambda item: item[1])
            selected.loc[valid_idx, target] = candidates[best_id].loc[valid_idx, target].astype(float).to_numpy()
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "candidate_id": int(best_id),
                    "train_loss": float(best_loss),
                    "valid_loss": candidate_target_loss(train, candidates[best_id], target, valid_idx),
                }
            )

    if selected[TARGETS].isna().any().any():
        raise ValueError("Outer selector failed to fill all OOF predictions.")
    overall, target_scores = local_scores(train, selected)
    slices = slice_summary(train, selected)
    return {
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
        "choices": choices,
        "choice_counts": {
            target: dict(Counter(choice["candidate_id"] for choice in target_choices))
            for target, target_choices in choices.items()
        },
    }


def full_train_selector(
    train: pd.DataFrame,
    candidates: dict[int, pd.DataFrame],
    test_candidates: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    selected_oof = train[["subject_id", "date"]].copy()
    selected_test = next(iter(test_candidates.values()))[["subject_id", "date"]].copy()
    choices = {}
    for target in TARGETS:
        losses = {
            candidate_id: target_loss(train[target].astype(int).to_numpy(), pred[target].astype(float).to_numpy())
            for candidate_id, pred in candidates.items()
        }
        best_id, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_oof[target] = candidates[best_id][target].astype(float).to_numpy()
        selected_test[target] = test_candidates[best_id][target].astype(float).to_numpy()
        choices[target] = {"candidate_id": int(best_id), "train_loss": float(best_loss), "losses": losses}
    overall, target_scores = local_scores(train, selected_oof)
    return {
        "overall": overall,
        "target_scores": target_scores,
        "choices": choices,
        "selected_test_prediction_mean": {
            target: float(selected_test[target].astype(float).mean()) for target in TARGETS
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate outer-selector diagnostic — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It nests only the choice among existing OOF artifacts; candidate artifacts themselves may still be non-nested.",
        "",
        "## Split summary",
        "",
        "| split | overall | robust | worst target | worst subject | note |",
        "|---|---:|---:|---|---|---|",
    ]
    for split_name, split in report["splits"].items():
        slices = split["slice_summary"]
        lines.append(
            f"| `{split_name}` | {split['overall']:.6f} | {split['robust_score']:.6f} | "
            f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
            f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` | selection-nested only |"
        )
    lines.extend(
        [
            "",
            "## Full-train selected targets",
            "",
            "| target | selected id | train loss |",
            "|---|---:|---:|",
        ]
    )
    for target, choice in report["full_train_selector"]["choices"].items():
        lines.append(f"| {target} | {choice['candidate_id']} | {choice['train_loss']:.6f} |")
    lines.extend(["", "## Verdict", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path, candidate_ids: list[int]) -> dict[str, Any]:
    train, test = load_train_test()
    candidates = load_oof_candidates(train, candidate_ids)
    test_candidates = load_test_candidates(test, candidate_ids)
    splits = {
        "subject_hole_choice_nested": run_outer_selector(train, make_subject_hole_folds(train, n_folds=5), candidates),
        "loso_choice_nested": run_outer_selector(train, make_loso_folds(train), candidates),
    }
    full_selected = full_train_selector(train, candidates, test_candidates)
    failed_family_selected = sorted(
        {
            choice["candidate_id"]
            for split in splits.values()
            for choices in split["choices"].values()
            for choice in choices
            if choice["candidate_id"] in {304, 305, 306, 307, 310, 311, 295, 297}
        }
    )
    reason = (
        "Choice-nested selection still prefers non-nested failed-direction artifacts in several folds/targets, "
        "but this does not repair the underlying artifact-generation leakage. Treat it as a map for a future true "
        "fold-nested rebuild, not as a submission gate."
        if failed_family_selected
        else "Choice-nested selection does not select the failed-direction artifacts; keep them frozen."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "candidate_outer_selector_post_id335",
        "candidate_ids": candidate_ids,
        "splits": splits,
        "full_train_selector": full_selected,
        "failed_family_selected_ids": failed_family_selected,
        "decision": {
            "promote_generated_unsubmitted_artifact": False,
            "public_058_plausible": False,
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
        default=REPORTS_DIR / "candidate_outer_selector_post_id335_20260617.json",
    )
    parser.add_argument("--candidate-ids", default=",".join(str(value) for value in DEFAULT_CANDIDATE_IDS))
    args = parser.parse_args()
    report = run(args.output_json, [int(value) for value in args.candidate_ids.split(",")])
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(
        json.dumps(report["decision"] | {"failed_family_selected_ids": report["failed_family_selected_ids"]}, indent=2)
    )


if __name__ == "__main__":
    main()
