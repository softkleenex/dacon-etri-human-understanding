#!/usr/bin/env python3
"""ID342-aware subject-prior calibration gate for Q3/S3.

Q3 is defined relative to each participant's average pre-sleep stress, and S3
is a sleep-onset-latency guideline adherence target. This report-only
diagnostic checks whether ID342 should be nudged toward fold-internal
participant priors or global priors for Q3/S3.

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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_subject_prior_calibration_gate_20260624.json"

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
ALPHAS = [2.0, 5.0, 10.0]
BLEND_WEIGHTS = [0.1, 0.25, 0.5, 1.0]
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075

RESEARCH_SOURCES = [
    {
        "title": "Human Understanding AI Paper Challenge 2024 - Dataset Design",
        "url": "https://arxiv.org/html/2403.16509v1",
        "read": "The task predicts daily sleep quality, emotional state, stress, and sleep metrics from smartphone/watch lifelog data.",
    },
    {
        "title": "DACON Human Understanding AI data page",
        "url": "https://dacon.io/en/competitions/official/236468/data",
        "read": "DACON defines Q3 as pre-sleep stress and S3 as adherence to sleep-onset-latency guidance.",
    },
    {
        "title": "Human Understanding AI Paper Challenge 2024 - Dataset Design PDF",
        "url": "https://cfiles.dacon.co.kr/competitions/236690/Human%20Understanding%20AI%20Paper%20Challenge%202024%20-%20Dataset%20Design.pdf",
        "read": "Q3 is participant-relative stress, with lower-than-average stress mapped to 1; sleep metrics follow NSF-style guideline adherence.",
    },
]


@dataclass(frozen=True)
class CalibrationSpec:
    name: str
    method: str
    alpha: float
    weight: float


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


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1.0 - EPS), labels=[0, 1]))


def weighted_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def candidate_specs() -> list[CalibrationSpec]:
    specs: list[CalibrationSpec] = []
    for method in ["prob_subject_mean", "logit_subject_mean", "global_subject_mean"]:
        for alpha in ALPHAS:
            for weight in BLEND_WEIGHTS:
                specs.append(
                    CalibrationSpec(
                        name=f"{method}_a{alpha:g}_w{weight:g}",
                        method=method,
                        alpha=float(alpha),
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


def load_test_values(test: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_base_test(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merged = test[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} test predictions do not align with sorted test rows.")
    return merged[TARGETS].astype(float).to_numpy()


def subject_distribution_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        train_target = train[["subject_id", target]].copy()
        oof_frame = train[["subject_id"]].copy()
        oof_frame["prediction"] = anchor_oof[:, target_idx]
        test_frame = test[["subject_id"]].copy()
        test_frame["prediction"] = anchor_test[:, target_idx]
        rows[target] = {
            "train_label_mean": float(train[target].mean()),
            "oof_prediction_mean": float(anchor_oof[:, target_idx].mean()),
            "test_prediction_mean": float(anchor_test[:, target_idx].mean()),
            "train_label_mean_by_subject": train_target.groupby("subject_id")[target].mean().round(6).to_dict(),
            "oof_prediction_mean_by_subject": oof_frame.groupby("subject_id")["prediction"].mean().round(6).to_dict(),
            "test_prediction_mean_by_subject": test_frame.groupby("subject_id")["prediction"].mean().round(6).to_dict(),
        }
    return rows


def calibrated_split_predictions(
    train: pd.DataFrame,
    anchor_values: np.ndarray,
    target: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: CalibrationSpec,
) -> np.ndarray:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    anchor_target = anchor_values[:, target_idx]
    pred = np.zeros(len(train), dtype=float)
    for fit_idx, valid_idx in folds:
        fit = train.iloc[fit_idx][["subject_id"]].copy()
        fit["_y"] = y[fit_idx]
        fit["_p"] = anchor_target[fit_idx]
        valid_subjects = train.iloc[valid_idx][["subject_id"]].copy()
        global_y = float(fit["_y"].mean())
        global_p = float(fit["_p"].mean())
        stat_rows = []
        for subject_id, group in fit.groupby("subject_id", sort=False):
            n_rows = len(group)
            y_mean = float((group["_y"].sum() + spec.alpha * global_y) / (n_rows + spec.alpha))
            p_mean = float(group["_p"].mean())
            stat_rows.append({"subject_id": subject_id, "y_mean": y_mean, "p_mean": p_mean})
        stats = pd.DataFrame(stat_rows)
        merged = valid_subjects.merge(stats, on="subject_id", how="left")
        y_mean = merged["y_mean"].fillna(global_y).to_numpy(dtype=float)
        p_mean = merged["p_mean"].fillna(global_p).to_numpy(dtype=float)
        if spec.method == "prob_subject_mean":
            raw = anchor_target[valid_idx] + (y_mean - p_mean)
        elif spec.method == "logit_subject_mean":
            raw = sigmoid(logit(anchor_target[valid_idx]) + (logit(y_mean) - logit(p_mean)))
        elif spec.method == "global_subject_mean":
            raw = np.full(len(valid_idx), global_y, dtype=float)
        else:
            raise ValueError(f"Unknown calibration method: {spec.method}")
        pred[valid_idx] = np.clip((1.0 - spec.weight) * anchor_target[valid_idx] + spec.weight * raw, EPS, 1.0 - EPS)
    return pred


def scan_calibration(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in candidate_specs():
            split_deltas: dict[str, float] = {}
            split_losses: dict[str, float] = {}
            split_anchor_losses: dict[str, float] = {}
            wins = 0
            for split_name in PRIMARY_SPLITS:
                folds = splits[split_name]
                pred = calibrated_split_predictions(train, anchor_values, target, folds, spec)
                anchor_loss = weighted_loss(y, anchor_values[:, target_idx], folds)
                candidate_loss = weighted_loss(y, pred, folds)
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
                    "config": f"{target}_{spec.name}",
                    "calibration_spec": asdict(spec),
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
    return rows


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    splits = load_split_manifest()
    anchor_oof = load_reference_values(train, ANCHOR_ID)
    anchor_test = load_test_values(test, ANCHOR_ID)
    rows = scan_calibration(train, splits, anchor_oof)
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
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
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "best_row": rows[0] if rows else None,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "subject_prior_calibration_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Subject-prior calibration does not improve Q3/S3 against ID342 under primary lenses. "
            "Do not force participant-level priors into test predictions; use only as validation context."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_prior_calibration_gate",
        "purpose": "Check whether Q3/S3 should be nudged toward fold-internal subject or global priors.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "alphas": ALPHAS,
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "research_sources": RESEARCH_SOURCES,
        "subject_distribution_summary": subject_distribution_summary(train, test, anchor_oof, anchor_test),
        "top_rows": rows[:24],
        "eligible_rows": eligible_rows[:24],
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
    lines = [
        "# Post-ID342 subject-prior calibration gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Subject Distribution",
        "",
        "| target | train label mean | OOF prediction mean | test prediction mean |",
        "|---|---:|---:|---:|",
    ]
    for target, row in report["subject_distribution_summary"].items():
        lines.append(
            f"| `{target}` | {row['train_label_mean']:.6f} | "
            f"{row['oof_prediction_mean']:.6f} | {row['test_prediction_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| target | config | wins | mean delta | max delta | eligible |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["top_rows"][:12]:
        lines.append(
            f"| `{row['target']}` | `{row['config']}` | {row['wins']} | "
            f"{row['mean_delta']:.9f} | {row['max_delta']:.9f} | `{row['eligible']}` |"
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
