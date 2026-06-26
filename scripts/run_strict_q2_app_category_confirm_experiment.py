#!/usr/bin/env python3
"""Confirm a narrow Q2 semantic app-category LGBM overlay after broad diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import scripts.run_strict_q2_app_category_compact_experiment as base  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_BASE_ID = 117


def compact_confirm_model_pool() -> dict[str, object]:
    return {
        "lgbm_l7_lr025_n160_l20_confirm": LGBMClassifier(
            n_estimators=160,
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_lambda=20.0,
            random_state=42,
            verbosity=-1,
        )
    }


def compact_confirm_candidate_pairs() -> list[tuple[str, str]]:
    return [("appcat_abs", "lgbm_l7_lr025_n160_l20_confirm")]


def apply_confirm_patch() -> None:
    base.BLEND_WEIGHTS = [0.05, 0.10, 0.15]
    base.compact_model_pool = compact_confirm_model_pool
    base.compact_candidate_pairs = compact_confirm_candidate_pairs


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q2_app_category_confirm_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q2_app_category_confirm_diagnostic_id{base_id:03d}_{suffix}.json"
    report = {
        "base_id": base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "selector_info": selector_info,
        "confirm_note": (
            "Narrow confirmation after reports/strict_q2_app_category_compact_diagnostic_id116.json: "
            "only appcat_abs LGBM is retained and blend weights are capped at 0.15."
        ),
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q2_app_category_confirm_overlay")
    args = parser.parse_args()

    apply_confirm_patch()
    oof, test_pred, target_scores, selector_info, sample_original = base.predict_q2_app_category_overlay(
        base_id=args.base_id
    )
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "q2_candidate_scores": selector_info["q2_candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = base.promote_artifacts(
            args.experiment_id,
            args.slug,
            args.base_id,
            oof,
            test_pred,
            target_scores,
            selector_info,
            sample_original,
        )
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
