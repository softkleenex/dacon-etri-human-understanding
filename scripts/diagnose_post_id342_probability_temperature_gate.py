#!/usr/bin/env python3
"""ID342 probability-submission and temperature calibration gate.

This report-only diagnostic confirms that ID342's submitted CSV is a
probability file matching the base-test artifact, not a thresholded 0/1 file.
It then tests a small fold-internal temperature/shift calibration grid for
Q3/S3 to check whether simple probability calibration opens a safe route.

It creates no prediction CSV, feature parquet, base prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_clean_id221_inner_selection_probe import make_inner_folds  # noqa: E402
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402,I001
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
)

KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SUBMISSIONS_DIR = ROOT / "submissions"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_probability_temperature_gate_20260624.json"

ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
FOCUS_TARGETS = ["Q3", "S3"]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
EPS = 1e-6
TEMPERATURES = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
SHIFTS = [-0.25, -0.1, 0.0, 0.1, 0.25]
BLEND_WEIGHTS = [0.25, 0.5, 1.0]
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075


@dataclass(frozen=True)
class TempSpec:
    name: str
    temperature: float
    shift: float
    weight: float
    kind: str = "temp_shift"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def logit(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def transform(values: np.ndarray, spec: TempSpec) -> np.ndarray:
    if spec.kind == "base":
        return np.clip(values, EPS, 1.0 - EPS)
    raw = sigmoid(logit(values) / spec.temperature + spec.shift)
    return np.clip((1.0 - spec.weight) * values + spec.weight * raw, EPS, 1.0 - EPS)


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1.0 - EPS), labels=[0, 1]))


def weighted_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def candidate_specs() -> list[TempSpec]:
    specs = [TempSpec("base", temperature=1.0, shift=0.0, weight=0.0, kind="base")]
    for temperature in TEMPERATURES:
        for shift in SHIFTS:
            for weight in BLEND_WEIGHTS:
                if temperature == 1.0 and shift == 0.0:
                    continue
                specs.append(
                    TempSpec(
                        name=f"t{temperature:g}_s{shift:g}_w{weight:g}",
                        temperature=float(temperature),
                        shift=float(shift),
                        weight=float(weight),
                    )
                )
    return specs


def load_test_sorted() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def find_base_test(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base test ID{candidate_id}")
    return paths[0]


def find_submission(candidate_id: int) -> Path:
    paths = sorted(SUBMISSIONS_DIR.glob(f"submission_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing submission ID{candidate_id}")
    return paths[0]


def load_base_test_frame(test: pd.DataFrame, candidate_id: int) -> tuple[pd.DataFrame, Path]:
    path = find_base_test(candidate_id)
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merged = test[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} test predictions do not align with sorted test rows.")
    return merged, path


def submission_contract(test: pd.DataFrame, base_test: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    path = find_submission(candidate_id)
    submission = pd.read_csv(path)
    submission["date"] = pd.to_datetime(submission["lifelog_date"]).dt.date
    aligned = test[["subject_id", "date"]].merge(
        submission[["subject_id", "date", *TARGETS]],
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if aligned[TARGETS].isna().any().any():
        raise ValueError(f"Submission ID{candidate_id} does not align with sorted sample rows.")
    sub_values = aligned[TARGETS].astype(float).to_numpy()
    base_values = base_test[TARGETS].astype(float).to_numpy()
    abs_delta = np.abs(sub_values - base_values)
    unique_counts = {target: int(aligned[target].nunique(dropna=False)) for target in TARGETS}
    binary_targets = {
        target: bool(set(np.round(aligned[target].dropna().unique(), 12)).issubset({0.0, 1.0})) for target in TARGETS
    }
    return {
        "submission_path": rel(path),
        "base_test_path": rel(find_base_test(candidate_id)),
        "row_count": int(len(aligned)),
        "max_abs_delta_vs_base_test": float(abs_delta.max()),
        "mean_abs_delta_vs_base_test": float(abs_delta.mean()),
        "all_values_match_base_test": bool(abs_delta.max() <= 1e-12),
        "unique_counts_by_target": unique_counts,
        "binary_targets": binary_targets,
        "all_targets_binary": bool(all(binary_targets.values())),
        "target_means": {target: float(aligned[target].mean()) for target in TARGETS},
    }


def select_spec_for_outer_train(
    train: pd.DataFrame,
    anchor_values: np.ndarray,
    target: str,
    outer_train_idx: np.ndarray,
    specs: list[TempSpec],
) -> tuple[TempSpec, list[dict[str, Any]]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    base_values = anchor_values[:, target_idx]
    inner_folds = make_inner_folds(train, np.asarray(outer_train_idx, dtype=int))
    rows: list[dict[str, Any]] = []
    for spec in specs:
        weighted = 0.0
        total = 0
        for _inner_train_idx, inner_valid_idx in inner_folds:
            pred = transform(base_values[inner_valid_idx], spec)
            weighted += fold_loss(y[inner_valid_idx], pred) * len(inner_valid_idx)
            total += len(inner_valid_idx)
        loss = float(weighted / total)
        rows.append({"name": spec.name, "loss": loss, "spec": asdict(spec)})
    rows = sorted(rows, key=lambda row: (row["loss"], row["name"]))
    selected = TempSpec(**rows[0]["spec"])
    return selected, rows[:8]


def scan_temperature_gate(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
) -> list[dict[str, Any]]:
    train = train.copy()
    train["date_dt"] = pd.to_datetime(train["date"])
    specs = candidate_specs()
    rows: list[dict[str, Any]] = []
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        base_values = anchor_values[:, target_idx]
        split_deltas: dict[str, float] = {}
        split_losses: dict[str, float] = {}
        split_anchor_losses: dict[str, float] = {}
        selected_by_split: dict[str, list[str]] = {}
        wins = 0
        for split_name in PRIMARY_SPLITS:
            folds = splits[split_name]
            pred = np.zeros(len(train), dtype=float)
            selected_names: list[str] = []
            for fit_idx, valid_idx in folds:
                selected, _top_rows = select_spec_for_outer_train(train, anchor_values, target, fit_idx, specs)
                selected_names.append(selected.name)
                pred[valid_idx] = transform(base_values[valid_idx], selected)
            anchor_loss = weighted_loss(y, base_values, folds)
            candidate_loss = weighted_loss(y, pred, folds)
            delta = candidate_loss - anchor_loss
            split_anchor_losses[split_name] = anchor_loss
            split_losses[split_name] = candidate_loss
            split_deltas[split_name] = delta
            selected_by_split[split_name] = selected_names
            wins += int(delta < 0)
        mean_delta = float(np.mean(list(split_deltas.values())))
        max_delta = float(np.max(list(split_deltas.values())))
        rows.append(
            {
                "target": target,
                "config": f"{target}_fold_internal_temperature_shift",
                "wins": int(wins),
                "mean_delta": mean_delta,
                "max_delta": max_delta,
                "min_delta": float(np.min(list(split_deltas.values()))),
                "split_deltas": split_deltas,
                "split_losses": split_losses,
                "split_anchor_losses": split_anchor_losses,
                "selected_specs_by_split": selected_by_split,
                "eligible": bool(
                    wins >= TARGET_WIN_FLOOR
                    and mean_delta <= TARGET_MEAN_DELTA_CEILING
                    and max_delta <= TARGET_MAX_DELTA_CEILING
                ),
            }
        )
    return sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    base_test, base_test_path = load_base_test_frame(test, ANCHOR_ID)
    contract = submission_contract(test, base_test, ANCHOR_ID)
    rows = scan_temperature_gate(train, splits, anchor_values)
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
        "best_row": rows[0] if rows else None,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "submission_probability_contract_confirmed": bool(
            contract["all_values_match_base_test"] and not contract["all_targets_binary"]
        ),
        "temperature_shift_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "ID342 submission is a probability file matching base_test; thresholding is not a direct local lever. "
            "Fold-internal temperature/shift calibration does not open Q3/S3, so do not create artifacts or submit."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_probability_temperature_gate",
        "purpose": "Confirm probability-submission contract and test simple Q3/S3 temperature/shift calibration.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "base_test_path": rel(base_test_path),
        "submission_contract": contract,
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "temperatures": TEMPERATURES,
            "shifts": SHIFTS,
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "top_rows": rows[:12],
        "eligible_rows": eligible_rows,
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
    contract = report["submission_contract"]
    lines = [
        "# Post-ID342 probability and temperature gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Submission Contract",
        "",
        f"- Submission path: `{contract['submission_path']}`",
        f"- Base-test path: `{contract['base_test_path']}`",
        f"- Rows: `{contract['row_count']}`",
        f"- Max absolute delta vs base-test: `{contract['max_abs_delta_vs_base_test']:.12f}`",
        f"- All targets binary: `{contract['all_targets_binary']}`",
        "",
        "## Temperature Rows",
        "",
        "| target | wins | mean delta | max delta | min delta | eligible |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["top_rows"]:
        lines.append(
            f"| `{row['target']}` | {row['wins']} | {row['mean_delta']:.9f} | "
            f"{row['max_delta']:.9f} | {row['min_delta']:.9f} | `{row['eligible']}` |"
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
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Eligible rows: {decision['eligible_row_count']} | selected={decision['selected_targets']}")


if __name__ == "__main__":
    main()
