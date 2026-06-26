#!/usr/bin/env python3
"""Fold-internal second-stage calibration diagnostic after ID335.

This report-only diagnostic refits existing second-stage calibration recipes
inside each outer fold. It tests whether the calibration layer itself survives
without fitting and scoring on the same OOF rows. Source artifacts are still
pre-existing OOF products, so this is not a true model-level nested rebuild.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
from scripts.run_public_extreme_calibration_probe_experiment import (  # noqa: E402
    apply_quantile_calibrator,
    fit_quantile_calibrator,
)
from scripts.run_public_second_stage_calibration_probe_experiment import apply_temp_shift  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
EPS = 1e-6
DEFAULT_EXPERIMENT_IDS = [235, 239, 242, 243, 250, 251, 252]


@dataclass(frozen=True)
class Scenario:
    original_id: int
    name: str
    base_id: int
    target_methods: dict[str, dict[str, Any]]
    original_local_cv: float
    original_local_cv_name: str


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def experiment_json_path(experiment_id: int) -> Path:
    matches = sorted(REPORTS_DIR.glob(f"experiment_{experiment_id}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one experiment JSON for ID{experiment_id}, found {matches}")
    return matches[0]


def base_id_from_source(value: str) -> int:
    match = re.search(r"base_oof_id_(\d+)_", value)
    if not match:
        raise ValueError(f"Could not parse base id from source: {value}")
    return int(match.group(1))


def load_scenario(experiment_id: int) -> Scenario:
    metadata = json.loads(experiment_json_path(experiment_id).read_text(encoding="utf-8"))
    selector = metadata.get("selector_info", {})
    target_methods = selector.get("target_methods")
    if not isinstance(target_methods, dict):
        raise ValueError(f"ID{experiment_id} does not expose selector_info.target_methods")
    missing = sorted(set(TARGETS) - set(target_methods))
    if missing:
        raise ValueError(f"ID{experiment_id} target methods missing: {missing}")
    base_id = base_id_from_source(str(selector.get("base_source", "")))
    return Scenario(
        original_id=experiment_id,
        name=f"id{experiment_id}_like_on_id{base_id}",
        base_id=base_id,
        target_methods=target_methods,
        original_local_cv=float(metadata["local_cv"]),
        original_local_cv_name=str(metadata.get("local_cv_name", "")),
    )


def fit_apply_target_method(
    train_values: np.ndarray,
    valid_values: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    method = str(config["method"])
    train_values = np.clip(np.asarray(train_values, dtype=float), EPS, 1.0 - EPS)
    valid_values = np.clip(np.asarray(valid_values, dtype=float), EPS, 1.0 - EPS)
    if method == "base":
        return valid_values, {"method": "base"}
    if method == "temp_shift":
        temperature = float(config["temperature"])
        shift = float(config["shift"])
        return (
            apply_temp_shift(valid_values, temperature, shift),
            {"method": "temp_shift", "temperature": temperature, "shift": shift},
        )
    if method == "quantile":
        q = int(config["q"])
        calibrator = fit_quantile_calibrator(train_values, y_train, q=q)
        return apply_quantile_calibrator(valid_values, calibrator), {
            "method": "quantile",
            "q": q,
            "n_edges": len(calibrator.get("edges", [])),
        }
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
        model.fit(train_values, y_train)
        return np.clip(model.predict(valid_values), EPS, 1.0 - EPS), {
            "method": "isotonic",
            "n_thresholds": int(len(model.X_thresholds_)),
        }
    raise ValueError(f"Unknown calibration method: {method}")


def run_scenario(
    *,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_oof: pd.DataFrame,
    scenario: Scenario,
) -> dict[str, Any]:
    selected = train[["subject_id", "date"]].copy()
    for target in TARGETS:
        selected[target] = np.nan
    choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}

    for fold_idx, (raw_train_idx, raw_valid_idx) in enumerate(folds):
        train_idx = np.asarray(raw_train_idx, dtype=int)
        valid_idx = np.asarray(raw_valid_idx, dtype=int)
        for target in TARGETS:
            config = scenario.target_methods[target]
            y_train = train[target].iloc[train_idx].astype(int).to_numpy()
            values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            valid_pred, fitted = fit_apply_target_method(values[train_idx], values[valid_idx], y_train, config)
            selected.loc[valid_idx, target] = valid_pred
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "config": config,
                    "fitted": fitted,
                    "train_base_loss": target_loss(y_train, values[train_idx]),
                    "valid_base_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(), values[valid_idx]
                    ),
                    "valid_calibrated_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(), valid_pred
                    ),
                }
            )

    if selected[TARGETS].isna().any().any():
        raise ValueError("Fold-internal calibration failed to fill all predictions.")
    overall, target_scores = local_scores(train, selected)
    base_overall, base_target_scores = local_scores(train, base_oof)
    slices = slice_summary(train, selected)
    return {
        "scenario": {
            "name": scenario.name,
            "original_id": scenario.original_id,
            "base_id": scenario.base_id,
            "original_local_cv": scenario.original_local_cv,
            "original_local_cv_name": scenario.original_local_cv_name,
            "target_methods": scenario.target_methods,
        },
        "base_overall": base_overall,
        "base_target_scores": base_target_scores,
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "fold_internal_gap_vs_original": float(overall - scenario.original_local_cv),
        "improvement_vs_base": float(base_overall - overall),
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
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fold-internal second-stage calibration — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. Calibration is refit inside each outer fold, but the base artifacts are still pre-existing OOF products.",
        "",
        "## Results",
        "",
        "| split | scenario | base | original full-OOF | fold-internal | gap vs original | improvement vs base | robust | worst target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name, rows in report["splits"].items():
        for row in rows:
            slices = row["slice_summary"]
            lines.append(
                f"| `{split_name}` | `{row['scenario']['name']}` | {row['base_overall']:.6f} | "
                f"{row['scenario']['original_local_cv']:.6f} | {row['overall']:.6f} | "
                f"{row['fold_internal_gap_vs_original']:+.6f} | {row['improvement_vs_base']:+.6f} | "
                f"{row['robust_score']:.6f} | {slices['worst_target']} `{slices['worst_target_loss']:.6f}` |"
            )
    lines.extend(
        [
            "",
            "## Best rows by split",
            "",
            "| split | best scenario | fold-internal | robust | read |",
            "|---|---|---:|---:|---|",
        ]
    )
    for split_name, row in report["best_rows"].items():
        read = "calibration survives partially" if row["improvement_vs_base"] > 0 else "calibration fails vs base"
        lines.append(
            f"| `{split_name}` | `{row['scenario']['name']}` | {row['overall']:.6f} | "
            f"{row['robust_score']:.6f} | {read} |"
        )
    lines.extend(["", "## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def best_rows(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: min(rows, key=lambda row: (row["overall"], row["robust_score"]))
        for split_name, rows in results.items()
    }


def run(output_json: Path, experiment_ids: list[int]) -> dict[str, Any]:
    train, _test = load_train_test()
    scenarios = [load_scenario(experiment_id) for experiment_id in experiment_ids]
    base_ids = sorted({scenario.base_id for scenario in scenarios})
    base_artifacts = {base_id: align(load_artifact("oof", base_id), train) for base_id in base_ids}
    splits = {
        "subject_hole_fold_internal": make_subject_hole_folds(train),
        "loso_fold_internal": make_loso_folds(train),
    }
    results = {
        split_name: [
            run_scenario(
                train=train,
                folds=folds,
                base_oof=base_artifacts[scenario.base_id],
                scenario=scenario,
            )
            for scenario in scenarios
        ]
        for split_name, folds in splits.items()
    }
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "fold_internal_second_stage_calibration_post_id335",
        "experiment_ids": experiment_ids,
        "base_ids": base_ids,
        "source_artifacts_are_preexisting": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "splits": results,
        "best_rows": best_rows(results),
        "decision": {
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "model_level_nested_rebuild_still_required": True,
            "reason": (
                "Fold-internal calibration tests whether the second-stage layer survives outside full-OOF fitting, "
                "but the base artifacts are still pre-existing OOF products. Use this to choose rebuild priorities; "
                "do not promote any CSV or DACON submission from this diagnostic."
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
        default=REPORTS_DIR / "fold_internal_second_stage_calibration_post_id335_20260617.json",
    )
    parser.add_argument("--experiment-ids", type=int, nargs="+", default=DEFAULT_EXPERIMENT_IDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json, args.experiment_ids)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    for split_name, row in report["best_rows"].items():
        print(
            f"{split_name}: {row['scenario']['name']} overall={row['overall']:.6f} "
            f"robust={row['robust_score']:.6f} improvement_vs_base={row['improvement_vs_base']:+.6f}"
        )
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
