#!/usr/bin/env python3
"""Report-only strict audit for official demographic/static features.

This diagnostic does not create a DACON submission.  It compares static
demographic features against the current strict raw-feature pool under two
validation lenses:

- the repository's temporal subject-hole split, which simulates interleaved
  train/test dates for the same subjects;
- a leave-one-subject-out stress split, which checks whether demographics are
  only acting as subject identity memorization.

The purpose is to decide whether official paper demographics deserve a
target-specific strict modeling branch after ID329 failed to improve Public.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    TARGETS,
    build_feature_frames,
    numeric_feature_columns,
)
from src.features.demographics import DEMOGRAPHIC_FEATURE_COLUMNS, add_official_demographics  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_cols: list[str]
    model_name: str
    model: object


def make_loso_folds(train: pd.DataFrame, id_col: str = "subject_id") -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave one subject out per fold."""
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = train.index.to_numpy()
    for subject_id in sorted(train[id_col].astype(str).unique()):
        valid_idx = train.index[train[id_col].astype(str) == subject_id].to_numpy()
        train_idx = np.setdiff1d(all_indices, valid_idx, assume_unique=False)
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load strict feature frames and ensure demographic columns are present."""
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    train = add_official_demographics(train, id_col="subject_id")
    test = add_official_demographics(test, id_col="subject_id")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def model_pool(seed: int, n_estimators: int) -> dict[str, object]:
    return {
        "logreg_l2_c020": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=2000, solver="lbfgs", random_state=seed),
        ),
        "extratrees_d5_l8": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }


def predict_candidate(
    train: pd.DataFrame,
    target: str,
    candidate: Candidate,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    y = train[target].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train), dtype=float)
    has_prediction = np.zeros(len(train), dtype=bool)
    fold_details: list[dict[str, object]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(candidate.model)
        model.fit(train.iloc[train_idx][candidate.feature_cols], y.iloc[train_idx])
        pred = np.clip(model.predict_proba(train.iloc[valid_idx][candidate.feature_cols])[:, 1], 1e-6, 1.0 - 1e-6)
        oof[valid_idx] = pred
        has_prediction[valid_idx] = True
        fold_details.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_subjects": sorted(train.iloc[valid_idx]["subject_id"].astype(str).unique().tolist()),
                "valid_loss": float(log_loss(y.iloc[valid_idx], pred, labels=[0, 1])),
            }
        )

    if not has_prediction.all():
        raise ValueError(f"Candidate {candidate.name}/{target} did not predict every row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), fold_details


def evaluate_split(
    train: pd.DataFrame,
    split_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[Candidate],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    fold_details: dict[str, dict[str, list[dict[str, object]]]] = {}
    for candidate in candidates:
        target_losses: dict[str, float] = {}
        fold_details[candidate.name] = {}
        for target in TARGETS:
            pred, details = predict_candidate(train=train, target=target, candidate=candidate, folds=folds)
            target_loss = float(log_loss(train[target].astype(int), pred, labels=[0, 1]))
            target_losses[target] = target_loss
            fold_details[candidate.name][target] = details
        rows.append(
            {
                "split": split_name,
                "candidate": candidate.name,
                "model": candidate.model_name,
                "n_features": int(len(candidate.feature_cols)),
                "overall": float(np.mean(list(target_losses.values()))),
                "target_losses": target_losses,
            }
        )

    best_by_target: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        best = min(rows, key=lambda row: float(row["target_losses"][target]))
        best_by_target[target] = {
            "candidate": best["candidate"],
            "model": best["model"],
            "loss": float(best["target_losses"][target]),
            "n_features": int(best["n_features"]),
        }

    return {
        "split": split_name,
        "n_folds": int(len(folds)),
        "rows": sorted(rows, key=lambda row: float(row["overall"])),
        "best_by_target": best_by_target,
        "fold_details": fold_details,
    }


def build_candidates(train: pd.DataFrame, test: pd.DataFrame, seed: int, n_estimators: int) -> list[Candidate]:
    all_numeric = numeric_feature_columns(train, test)
    demo_cols = [col for col in DEMOGRAPHIC_FEATURE_COLUMNS if col in train.columns]
    if not demo_cols:
        raise ValueError("No demographic columns found after add_official_demographics().")
    base_cols = [col for col in all_numeric if col not in set(demo_cols)]
    compact_covariates = [
        col
        for col in [
            "date_ord",
            "dayofyear2",
            "weekofyear",
            "subject_code",
            "subject_date_frac_all",
            "days_since_subject_start_all",
            "days_to_subject_end_all",
        ]
        if col in train.columns
    ]
    feature_sets = {
        "demo_only": demo_cols,
        "compact_covariates": compact_covariates,
        "compact_plus_demo": sorted(set(compact_covariates + demo_cols)),
        "base_no_demo": base_cols,
        "base_plus_demo": sorted(set(base_cols + demo_cols)),
    }
    candidates: list[Candidate] = []
    for model_name, model in model_pool(seed=seed, n_estimators=n_estimators).items():
        for feature_name, feature_cols in feature_sets.items():
            if feature_cols:
                candidates.append(
                    Candidate(
                        name=f"{feature_name}__{model_name}",
                        feature_cols=feature_cols,
                        model_name=model_name,
                        model=model,
                    )
                )
    return candidates


def write_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Strict demographic/static-feature audit",
        "",
        f"Created: `{report['created_at_kst']}` KST",
        "",
        "This is a report-only diagnostic. It creates no submission file and performs no DACON submission.",
        "",
        "## Candidate feature sets",
        "",
        "| candidate | model | n features |",
        "|---|---|---:|",
    ]
    seen: set[str] = set()
    for split in report["splits"].values():
        for row in split["rows"]:
            key = str(row["candidate"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| `{row['candidate']}` | `{row['model']}` | {row['n_features']} |")

    for split_name, split in report["splits"].items():
        lines.extend(
            [
                "",
                f"## {split_name}",
                "",
                "| rank | candidate | overall | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(split["rows"], start=1):
            losses = row["target_losses"]
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['overall']:.6f} | "
                f"{losses['Q1']:.6f} | {losses['Q2']:.6f} | {losses['Q3']:.6f} | "
                f"{losses['S1']:.6f} | {losses['S2']:.6f} | {losses['S3']:.6f} | {losses['S4']:.6f} |"
            )
        lines.extend(["", "### Best by target", "", "| target | candidate | loss |", "|---|---|---:|"])
        for target, best in split["best_by_target"].items():
            lines.append(f"| {target} | `{best['candidate']}` | {best['loss']:.6f} |")

    lines.extend(["", "## Interpretation", ""])
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "reports/strict_demographic_feature_audit_20260617.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=180)
    args = parser.parse_args()

    train, test = prepare_frames()
    candidates = build_candidates(train=train, test=test, seed=args.seed, n_estimators=args.n_estimators)
    subject_hole = make_subject_hole_folds(train, n_folds=5)
    loso = make_loso_folds(train)
    splits = {
        "subject_hole_temporal": evaluate_split(train, "subject_hole_temporal", subject_hole, candidates),
        "leave_one_subject_out_stress": evaluate_split(train, "leave_one_subject_out_stress", loso, candidates),
    }

    subject_hole_rows = splits["subject_hole_temporal"]["rows"]
    loso_rows = splits["leave_one_subject_out_stress"]["rows"]
    best_subject_hole = subject_hole_rows[0]
    best_loso = loso_rows[0]
    demo_wins = {
        split_name: {
            target: best
            for target, best in split["best_by_target"].items()
            if str(best["candidate"]).startswith("demo_only") or "plus_demo" in str(best["candidate"])
        }
        for split_name, split in splits.items()
    }
    report = {
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "diagnostic": "strict_demographic_static_feature_audit",
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "targets": TARGETS,
        "demographic_columns": [col for col in DEMOGRAPHIC_FEATURE_COLUMNS if col in train.columns],
        "splits": splits,
        "demo_wins_by_target": demo_wins,
        "interpretation": [
            (
                f"Best subject-hole candidate is {best_subject_hole['candidate']} with overall "
                f"{best_subject_hole['overall']:.6f}; this split can reward subject-static signals because each "
                "subject appears in both train and validation holes."
            ),
            (
                f"Best LOSO stress candidate is {best_loso['candidate']} with overall {best_loso['overall']:.6f}; "
                "large degradation here means demographics are not a robust standalone cross-subject signal."
            ),
            "No submission artifact is promoted by this audit. Use target-level wins only as hypotheses for a future strict branch.",
        ],
        "no_dacon_submission_made": True,
    }

    output_json = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md = output_json.with_suffix(".md")
    write_markdown(report, output_md)
    print(f"Wrote {output_json.relative_to(ROOT)}")
    print(f"Wrote {output_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
