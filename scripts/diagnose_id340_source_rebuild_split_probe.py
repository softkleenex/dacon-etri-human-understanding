#!/usr/bin/env python3
"""Report-only ID340 source rebuild split probe.

This is the first small model-level probe that uses the frozen ID340 validation
split manifest. It trains lightweight source models inside each split and
reports whether any source deserves a future ID340 artifact discussion.

It intentionally creates no submission CSV, no base prediction artifact, no
notebook/history artifact, and no submission-log row.
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
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    TARGETS,
    augment_with_target_variant,
    build_feature_frames,
    numeric_feature_columns,
)

KST = timezone(timedelta(hours=9))
SPLIT_MANIFEST_PATH = ROOT / "reports" / "id340_validation_split_manifest_20260618.json"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
REFERENCE_IDS = [178, 287, 329]
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]


@dataclass(frozen=True)
class SourceCandidate:
    name: str
    variant: str
    model: object | None
    kind: str


def source_candidates() -> list[SourceCandidate]:
    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.35,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )
    tiny_et = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=96,
                    max_depth=5,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return [
        SourceCandidate("constant_prior", "none", None, "baseline"),
        SourceCandidate("subject_prior_only", "subject_prior_only", None, "baseline"),
        SourceCandidate("raw_logreg_base", "base", logreg, "model_source"),
        SourceCandidate("raw_logreg_subject_prior", "subject_prior", logreg, "model_source"),
        SourceCandidate("raw_logreg_past", "past", logreg, "model_source"),
        SourceCandidate("raw_tiny_et_subject_prior", "subject_prior", tiny_et, "model_source"),
        SourceCandidate("raw_tiny_et_past", "past", tiny_et, "model_source"),
    ]


def load_split_manifest(path: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest["folds"]:
        split_name = row["split_name"]
        splits.setdefault(split_name, []).append(
            (
                np.asarray(row["train_idx"], dtype=int),
                np.asarray(row["valid_idx"], dtype=int),
            )
        )
    return splits


def find_reference_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing reference OOF artifact for ID{candidate_id}")
    return paths[0]


def load_reference_predictions(train: pd.DataFrame, candidate_id: int) -> np.ndarray:
    path = find_reference_oof(candidate_id)
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.date
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference OOF ID{candidate_id} does not align with train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def fold_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, 1e-6, 1 - 1e-6), labels=[0, 1]))


def predict_baseline(
    candidate: SourceCandidate,
    train: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> np.ndarray:
    global_mean = float(y.iloc[train_idx].mean())
    if candidate.name == "constant_prior":
        return np.full(len(valid_idx), global_mean, dtype=float)
    if candidate.name == "subject_prior_only":
        subject_means = y.iloc[train_idx].groupby(train.iloc[train_idx]["subject_id"]).mean()
        return train.iloc[valid_idx]["subject_id"].map(subject_means).fillna(global_mean).to_numpy(dtype=float)
    raise ValueError(f"Unsupported baseline candidate: {candidate.name}")


def predict_model_source(
    candidate: SourceCandidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> np.ndarray:
    train_aug, _test_aug, extra_cols = augment_with_target_variant(
        train=train,
        test=test,
        y=y,
        train_idx=train_idx,
        target=target,
        variant=candidate.variant,
    )
    active_cols = feature_cols + extra_cols
    if y.iloc[train_idx].nunique() < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = clone(candidate.model)
    model.fit(train_aug.iloc[train_idx][active_cols], y.iloc[train_idx])
    return model.predict_proba(train_aug.iloc[valid_idx][active_cols])[:, 1]


def evaluate_source_candidate(
    candidate: SourceCandidate,
    split_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    target_rows: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        y = train[target].astype(int).reset_index(drop=True)
        weighted_loss = 0.0
        total_n = 0
        target_fold_losses: list[dict[str, Any]] = []
        for fold_id, (train_idx, valid_idx) in enumerate(folds):
            if candidate.kind == "baseline":
                pred = predict_baseline(candidate, train, y, train_idx, valid_idx)
            else:
                pred = predict_model_source(candidate, train, test, feature_cols, target, y, train_idx, valid_idx)
            loss = fold_log_loss(y.iloc[valid_idx].to_numpy(), pred)
            weighted_loss += loss * len(valid_idx)
            total_n += len(valid_idx)
            fold_loss = {
                "fold_id": int(fold_id),
                "target": target,
                "valid_n": int(len(valid_idx)),
                "loss": loss,
            }
            target_fold_losses.append(fold_loss)
            fold_rows.append(fold_loss)
        target_rows[target] = {
            "loss": float(weighted_loss / total_n),
            "valid_n_weighted": int(total_n),
            "folds": target_fold_losses,
        }
    mean_loss = float(np.mean([row["loss"] for row in target_rows.values()]))
    return {
        "candidate": candidate.name,
        "kind": candidate.kind,
        "split": split_name,
        "mean_loss": mean_loss,
        "target_losses": {target: row["loss"] for target, row in target_rows.items()},
        "fold_target_losses": fold_rows,
    }


def evaluate_reference(
    candidate_id: int,
    split_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train: pd.DataFrame,
    values: np.ndarray,
) -> dict[str, Any]:
    target_losses: dict[str, float] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        weighted_loss = 0.0
        total_n = 0
        for _fold_id, (_train_idx, valid_idx) in enumerate(folds):
            loss = fold_log_loss(y[valid_idx], values[valid_idx, target_idx])
            weighted_loss += loss * len(valid_idx)
            total_n += len(valid_idx)
        target_losses[target] = float(weighted_loss / total_n)
    return {
        "candidate": f"reference_id{candidate_id}",
        "kind": "existing_oof_reference",
        "split": split_name,
        "mean_loss": float(np.mean(list(target_losses.values()))),
        "target_losses": target_losses,
    }


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    feature_cols = numeric_feature_columns(train, test)
    splits = load_split_manifest(SPLIT_MANIFEST_PATH)
    candidates = source_candidates()
    reference_values = {candidate_id: load_reference_predictions(train, candidate_id) for candidate_id in REFERENCE_IDS}

    split_reports: dict[str, Any] = {}
    for split_name, folds in splits.items():
        source_results = [
            evaluate_source_candidate(candidate, split_name, folds, train, test, feature_cols)
            for candidate in candidates
        ]
        reference_results = [
            evaluate_reference(candidate_id, split_name, folds, train, reference_values[candidate_id])
            for candidate_id in REFERENCE_IDS
        ]
        all_results = sorted([*source_results, *reference_results], key=lambda row: row["mean_loss"])
        id178_loss = next(row["mean_loss"] for row in reference_results if row["candidate"] == "reference_id178")
        best_source = min(source_results, key=lambda row: row["mean_loss"])
        split_reports[split_name] = {
            "n_folds": len(folds),
            "results": all_results,
            "best_source": best_source,
            "reference_id178_loss": id178_loss,
            "best_source_delta_vs_id178": float(best_source["mean_loss"] - id178_loss),
        }

    best_by_split = {
        split_name: {
            "best_source": row["best_source"]["candidate"],
            "best_source_loss": row["best_source"]["mean_loss"],
            "reference_id178_loss": row["reference_id178_loss"],
            "delta_vs_id178": row["best_source_delta_vs_id178"],
        }
        for split_name, row in split_reports.items()
    }
    primary_passes = {
        split_name: best_by_split[split_name]["delta_vs_id178"] < 0
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in best_by_split
    }
    stable_source_counts: dict[str, int] = {}
    for split_name in PRIMARY_GATE_SPLITS:
        if split_name in split_reports:
            name = split_reports[split_name]["best_source"]["candidate"]
            stable_source_counts[name] = stable_source_counts.get(name, 0) + 1
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "id340_source_rebuild_split_probe",
        "purpose": (
            "Train lightweight raw-feature source candidates inside the frozen ID340 split manifest and decide "
            "whether any source is worth a future artifact discussion."
        ),
        "feature_count": len(feature_cols),
        "source_candidates": [
            {"name": candidate.name, "variant": candidate.variant, "kind": candidate.kind} for candidate in candidates
        ],
        "reference_ids": REFERENCE_IDS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "best_by_split": best_by_split,
        "primary_split_passes_vs_id178": primary_passes,
        "stable_source_counts_on_primary_splits": stable_source_counts,
        "split_reports": split_reports,
        "decision": {
            "promote_submission_candidate": False,
            "create_id340": False,
            "any_source_beats_id178_on_all_primary_splits": all(primary_passes.values()) if primary_passes else False,
            "why": (
                "This is a source-rebuild probe only. A future ID340 requires a stable source that improves the "
                "predeclared split lenses and then passes public-risk geometry; this report creates no artifact."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID340 source rebuild split probe",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains lightweight source models inside the frozen ID340 splits, but creates no submission CSV, no base-prediction artifact, no notebook/history artifact, and no DACON submission.",
        "",
        f"- Numeric feature count: `{report['feature_count']}`",
        f"- Source candidates: `{', '.join(row['name'] for row in report['source_candidates'])}`",
        "",
        "## Best source by split",
        "",
        "| split | best source | best loss | ID178 ref loss | Δ vs ID178 | pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for split_name, row in report["best_by_split"].items():
        pass_flag = row["delta_vs_id178"] < 0
        lines.append(
            f"| `{split_name}` | `{row['best_source']}` | {row['best_source_loss']:.6f} | "
            f"{row['reference_id178_loss']:.6f} | {row['delta_vs_id178']:+.6f} | `{pass_flag}` |"
        )
    lines.extend(
        [
            "",
            "## Primary gate",
            "",
        ]
    )
    for split_name, passed in report["primary_split_passes_vs_id178"].items():
        lines.append(f"- `{split_name}` beats ID178 reference: `{passed}`")
    lines.extend(
        [
            "",
            "Stable best-source counts on primary splits:",
            "",
        ]
    )
    for name, count in sorted(
        report["stable_source_counts_on_primary_splits"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## Top rows per split",
            "",
        ]
    )
    for split_name, split_report in report["split_reports"].items():
        lines.extend(
            [
                f"### `{split_name}`",
                "",
                "| rank | candidate | kind | mean loss | Δ vs ID178 |",
                "|---:|---|---|---:|---:|",
            ]
        )
        id178_loss = split_report["reference_id178_loss"]
        for rank, row in enumerate(split_report["results"][:8], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | `{row['kind']}` | "
                f"{row['mean_loss']:.6f} | {row['mean_loss'] - id178_loss:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "- Promote submission candidate: **no**",
            "- Create ID340: **no**",
            f"- Any source beats ID178 on all primary splits: `{report['decision']['any_source_beats_id178_on_all_primary_splits']}`",
            f"- Reason: {report['decision']['why']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "id340_source_rebuild_split_probe_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
