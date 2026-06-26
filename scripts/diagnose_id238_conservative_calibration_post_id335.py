#!/usr/bin/env python3
"""Conservative fold-internal calibration alternatives for the ID238 -> ID242 layer.

This report-only diagnostic asks whether the broken lower layer beneath ID252
can be salvaged without repeating ID242's full-OOF quantile/isotonic shortcut.
Calibration candidates are fit and selected inside each outer fold. The base
ID238 OOF artifact is still pre-existing, so this is not true model-level
nested evidence and must not create a submission artifact.
"""

from __future__ import annotations

import argparse
import json
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
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
EPS = 1e-6
BASE_ID = 238
FAILED_REFERENCE_ID = 242


@dataclass(frozen=True)
class Scenario:
    name: str
    families: tuple[str, ...]
    max_train_delta: float
    delta_penalty: float
    extreme_penalty: float


SCENARIOS = [
    Scenario("base_only", ("base",), 0.0, 0.0, 0.0),
    Scenario("temp_safe_cap012", ("base", "temp"), 0.12, 0.05, 0.02),
    Scenario("lowq_shrink_cap012", ("base", "quantile_shrink"), 0.12, 0.10, 0.03),
    Scenario("iso_shrink_cap012", ("base", "isotonic_shrink"), 0.12, 0.12, 0.04),
    Scenario("combined_safe_cap012", ("base", "temp", "quantile_shrink", "isotonic_shrink"), 0.12, 0.14, 0.05),
    Scenario("combined_moderate_cap020", ("base", "temp", "quantile_shrink", "isotonic_shrink"), 0.20, 0.18, 0.06),
]
TEMPERATURES = (0.80, 0.90, 1.00, 1.10, 1.25, 1.50)
SHIFTS = (-0.08, -0.04, 0.0, 0.04, 0.08)
QUANTILE_QS = (3, 4, 5, 8, 12, 20)
SHRINK_ALPHAS = (0.10, 0.20, 0.35, 0.50)


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def logit(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def apply_temp_shift(values: np.ndarray, temperature: float, shift: float) -> np.ndarray:
    return np.clip(sigmoid(logit(values) / temperature + shift), EPS, 1.0 - EPS)


def probability_shrink(base: np.ndarray, transformed: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * base + alpha * transformed, EPS, 1.0 - EPS)


def logit_shrink(base: np.ndarray, transformed: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(sigmoid((1.0 - alpha) * logit(base) + alpha * logit(transformed)), EPS, 1.0 - EPS)


def extreme_rate(pred: np.ndarray) -> float:
    arr = np.asarray(pred, dtype=float)
    return float(np.mean((arr <= 0.01) | (arr >= 0.99)))


def fit_isotonic(train_values: np.ndarray, y_train: np.ndarray, values: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
    model.fit(train_values, y_train)
    return np.clip(model.predict(values), EPS, 1.0 - EPS)


def candidate_rows_for_target(
    *,
    base_values: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    scenario: Scenario,
) -> list[dict[str, Any]]:
    train_base = np.clip(base_values[train_idx], EPS, 1.0 - EPS)
    valid_base = np.clip(base_values[valid_idx], EPS, 1.0 - EPS)
    rows: list[dict[str, Any]] = []

    def append_candidate(
        name: str, family: str, train_pred: np.ndarray, valid_pred: np.ndarray, params: dict[str, Any]
    ) -> None:
        train_delta = np.abs(train_pred - train_base)
        max_delta = float(np.max(train_delta))
        if max_delta > scenario.max_train_delta + 1e-12:
            return
        loss = target_loss(y_train, train_pred)
        rows.append(
            {
                "name": name,
                "family": family,
                "params": params,
                "train_loss": loss,
                "objective": float(
                    loss
                    + scenario.delta_penalty * float(np.mean(train_delta))
                    + scenario.extreme_penalty * extreme_rate(train_pred)
                ),
                "train_max_abs_delta_vs_base": max_delta,
                "train_mean_abs_delta_vs_base": float(np.mean(train_delta)),
                "train_extreme_rate": extreme_rate(train_pred),
                "valid_pred": valid_pred,
            }
        )

    if "base" in scenario.families:
        append_candidate("base", "base", train_base, valid_base, {})

    if "temp" in scenario.families:
        for temperature in TEMPERATURES:
            for shift in SHIFTS:
                train_pred = apply_temp_shift(train_base, temperature, shift)
                valid_pred = apply_temp_shift(valid_base, temperature, shift)
                append_candidate(
                    f"temp_t{temperature:.2f}_s{shift:+.2f}",
                    "temp",
                    train_pred,
                    valid_pred,
                    {"temperature": temperature, "shift": shift},
                )

    if "quantile_shrink" in scenario.families:
        for q in QUANTILE_QS:
            calibrator = fit_quantile_calibrator(train_base, y_train, q=q)
            train_quantile = apply_quantile_calibrator(train_base, calibrator)
            valid_quantile = apply_quantile_calibrator(valid_base, calibrator)
            for alpha in SHRINK_ALPHAS:
                for shrink_kind, shrink_fn in {
                    "prob": probability_shrink,
                    "logit": logit_shrink,
                }.items():
                    append_candidate(
                        f"q{q}_{shrink_kind}_a{alpha:.2f}",
                        "quantile_shrink",
                        shrink_fn(train_base, train_quantile, alpha),
                        shrink_fn(valid_base, valid_quantile, alpha),
                        {"q": q, "alpha": alpha, "shrink_kind": shrink_kind},
                    )

    if "isotonic_shrink" in scenario.families:
        train_iso = fit_isotonic(train_base, y_train, train_base)
        valid_iso = fit_isotonic(train_base, y_train, valid_base)
        for alpha in SHRINK_ALPHAS:
            for shrink_kind, shrink_fn in {
                "prob": probability_shrink,
                "logit": logit_shrink,
            }.items():
                append_candidate(
                    f"iso_{shrink_kind}_a{alpha:.2f}",
                    "isotonic_shrink",
                    shrink_fn(train_base, train_iso, alpha),
                    shrink_fn(valid_base, valid_iso, alpha),
                    {"alpha": alpha, "shrink_kind": shrink_kind},
                )

    if not rows:
        rows.append(
            {
                "name": "base_forced_fallback",
                "family": "base",
                "params": {},
                "train_loss": target_loss(y_train, train_base),
                "objective": target_loss(y_train, train_base),
                "train_max_abs_delta_vs_base": 0.0,
                "train_mean_abs_delta_vs_base": 0.0,
                "train_extreme_rate": extreme_rate(train_base),
                "valid_pred": valid_base,
            }
        )
    return rows


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
            y_train = train[target].iloc[train_idx].astype(int).to_numpy()
            y_valid = train[target].iloc[valid_idx].astype(int).to_numpy()
            base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            rows = candidate_rows_for_target(
                base_values=base_values,
                y_train=y_train,
                train_idx=train_idx,
                valid_idx=valid_idx,
                scenario=scenario,
            )
            choice = min(
                rows,
                key=lambda row: (
                    row["objective"],
                    row["train_loss"],
                    row["train_max_abs_delta_vs_base"],
                    row["train_extreme_rate"],
                ),
            )
            selected.loc[valid_idx, target] = choice["valid_pred"]
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "choice_name": choice["name"],
                    "family": choice["family"],
                    "params": choice["params"],
                    "n_candidates": len(rows),
                    "train_loss": float(choice["train_loss"]),
                    "valid_loss": target_loss(y_valid, choice["valid_pred"]),
                    "objective": float(choice["objective"]),
                    "train_max_abs_delta_vs_base": float(choice["train_max_abs_delta_vs_base"]),
                    "train_mean_abs_delta_vs_base": float(choice["train_mean_abs_delta_vs_base"]),
                    "train_extreme_rate": float(choice["train_extreme_rate"]),
                }
            )

    if selected[TARGETS].isna().any().any():
        raise ValueError("Conservative calibration failed to fill all predictions.")
    overall, target_scores = local_scores(train, selected)
    base_overall, base_target_scores = local_scores(train, base_oof)
    slices = slice_summary(train, selected)
    return {
        "scenario": {
            "name": scenario.name,
            "families": list(scenario.families),
            "max_train_delta": scenario.max_train_delta,
            "delta_penalty": scenario.delta_penalty,
            "extreme_penalty": scenario.extreme_penalty,
        },
        "base_id": BASE_ID,
        "failed_reference_id": FAILED_REFERENCE_ID,
        "base_overall": base_overall,
        "base_target_scores": base_target_scores,
        "overall": overall,
        "target_scores": target_scores,
        "improvement_vs_base": float(base_overall - overall),
        "gap_vs_failed_reference_full_oof": float(overall - 0.35765072486726873),
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
    }


def best_rows(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: min(rows, key=lambda row: (row["overall"], row["robust_score"]))
        for split_name, rows in results.items()
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID238 conservative calibration diagnostic — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No CSV, notebook, submission-log row, base artifact, or DACON submission was created.",
        "",
        "## Results",
        "",
        "| split | scenario | base | calibrated | improvement vs base | robust | worst target | read |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for split_name, rows in report["splits"].items():
        for row in rows:
            slices = row["slice_summary"]
            read = "beats base" if row["improvement_vs_base"] > 0 else "worse than base"
            lines.append(
                f"| `{split_name}` | `{row['scenario']['name']}` | {row['base_overall']:.6f} | "
                f"{row['overall']:.6f} | {row['improvement_vs_base']:+.6f} | "
                f"{row['robust_score']:.6f} | {slices['worst_target']} `{slices['worst_target_loss']:.6f}` | {read} |"
            )
    lines.extend(
        [
            "",
            "## Best rows",
            "",
            "| split | best scenario | calibrated | improvement vs base | robust |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split_name, row in report["best_rows"].items():
        lines.append(
            f"| `{split_name}` | `{row['scenario']['name']}` | {row['overall']:.6f} | "
            f"{row['improvement_vs_base']:+.6f} | {row['robust_score']:.6f} |"
        )
    lines.extend(["", "## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, _test = load_train_test()
    base_oof = align(load_artifact("oof", BASE_ID), train)
    splits = {
        "subject_hole_fold_internal": make_subject_hole_folds(train),
        "loso_fold_internal": make_loso_folds(train),
    }
    results = {
        split_name: [
            run_scenario(train=train, folds=folds, base_oof=base_oof, scenario=scenario) for scenario in SCENARIOS
        ]
        for split_name, folds in splits.items()
    }
    best = best_rows(results)
    improves_both_splits = all(row["improvement_vs_base"] > 0.0 for row in best.values())
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "id238_conservative_calibration_post_id335",
        "base_id": BASE_ID,
        "failed_reference_id": FAILED_REFERENCE_ID,
        "source_artifacts_are_preexisting": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "scenarios": [scenario.__dict__ | {"families": list(scenario.families)} for scenario in SCENARIOS],
        "splits": results,
        "best_rows": best,
        "decision": {
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "id242_layer_salvage_signal": bool(improves_both_splits),
            "model_level_nested_rebuild_still_required": True,
            "reason": (
                "This diagnostic only tests conservative fold-internal calibration on top of pre-existing ID238 OOF "
                "predictions. Any positive result can guide a rebuild, but it is not leaderboard evidence and cannot "
                "authorize a submission artifact."
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
        default=REPORTS_DIR / "id238_conservative_calibration_post_id335_20260617.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    for split_name, row in report["best_rows"].items():
        print(
            f"{split_name}: {row['scenario']['name']} overall={row['overall']:.6f} "
            f"improvement_vs_base={row['improvement_vs_base']:+.6f} robust={row['robust_score']:.6f}"
        )
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
