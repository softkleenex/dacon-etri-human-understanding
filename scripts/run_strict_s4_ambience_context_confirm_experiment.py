#!/usr/bin/env python3
"""Confirm a narrow S4 semantic ambience-context overlay after broad diagnostics."""

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

import scripts.run_strict_ambience_context_experiment as base  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "S4"
DEFAULT_BASE_ID = 118


def compact_confirm_model_pool() -> dict[str, object]:
    return {
        "lgbm_l15_lr015_n220_l50_confirm": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.65,
            reg_lambda=50.0,
            random_state=43,
            verbosity=-1,
        )
    }


def compact_confirm_candidate_pairs() -> list[tuple[str, str]]:
    return [("amb_focused_abs_rel", "lgbm_l15_lr015_n220_l50_confirm")]


def apply_confirm_patch() -> None:
    base.BLEND_WEIGHTS = [0.02, 0.03, 0.05, 0.07, 0.10]
    base.compact_model_pool = compact_confirm_model_pool
    base.compact_candidate_pairs = compact_confirm_candidate_pairs


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_s4_ambience_context_confirm_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_s4_ambience_context_confirm_diagnostic_id{base_id:03d}_{suffix}.json"
    report = {
        "base_id": base_id,
        "target": TARGET,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "selector_info": selector_info,
        "confirm_note": (
            "Narrow confirmation after reports/strict_s4_ambience_context_diagnostic_id118.json: "
            "only focused absolute+relative ambience LGBM is retained and blend weights are capped at 0.10."
        ),
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="s4_ambience_context_confirm_overlay")
    args = parser.parse_args()

    apply_confirm_patch()
    oof, test_pred, target_scores, selector_info, sample_original = base.predict_ambience_overlay(args.base_id, TARGET)
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "target": TARGET,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "candidate_scores": selector_info["candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = base.promote_artifacts(
            experiment_id=args.experiment_id,
            slug=args.slug,
            base_id=args.base_id,
            target=TARGET,
            oof=oof,
            test_pred=test_pred,
            target_scores=target_scores,
            selector_info=selector_info,
            sample_original=sample_original,
        )
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
