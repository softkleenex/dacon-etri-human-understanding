#!/usr/bin/env python3
"""Report-only ID340 sequence-summary split probe.

This evaluates the strongest surviving sequence-summary signal under the frozen
ID340 split manifest. Models are trained inside each split; fixed blends against
the ID178 reference are evaluated only as diagnostics. No artifact is promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    make_model,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
SPLIT_MANIFEST_PATH = ROOT / "reports" / "id340_validation_split_manifest_20260618.json"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
BLEND_WEIGHTS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
MODEL_SPECS = (
    ModelSpec("lr_c0_1", "lr", c_value=0.1),
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
)


def load_split_manifest(path: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest["folds"]:
        splits.setdefault(row["split_name"], []).append(
            (
                np.asarray(row["train_idx"], dtype=int),
                np.asarray(row["valid_idx"], dtype=int),
            )
        )
    return splits


def find_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base OOF ID{candidate_id}")
    return paths[0]


def load_reference_values(train: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_oof(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.date
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, 1e-6, 1 - 1e-6), labels=[0, 1]))


def fit_split_source_predictions(
    x_train: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = np.zeros(len(y), dtype=bool)
    for train_idx, valid_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(spec))
        model.fit(x_train[train_idx], y[train_idx])
        pred[valid_idx] = model.predict_proba(x_train[valid_idx])[:, 1]
        covered[valid_idx] = True
    return np.clip(pred, 1e-6, 1 - 1e-6), covered


def evaluate_values_on_split(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    values: np.ndarray,
) -> dict[str, Any]:
    target_losses: dict[str, float] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        total = 0
        weighted = 0.0
        for _train_idx, valid_idx in folds:
            loss = fold_loss(y[valid_idx], values[valid_idx, target_idx])
            weighted += loss * len(valid_idx)
            total += len(valid_idx)
        target_losses[target] = float(weighted / total)
    return {"mean_loss": float(np.mean(list(target_losses.values()))), "target_losses": target_losses}


def evaluate_sequence_source_split(
    split_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train: pd.DataFrame,
    x_train: np.ndarray,
    reference178: np.ndarray,
) -> dict[str, Any]:
    id178_eval = evaluate_values_on_split(train, folds, reference178)
    source_rows: list[dict[str, Any]] = []
    blend_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        source_values = np.zeros((len(train), len(TARGETS)), dtype=float)
        covered_by_target: dict[str, int] = {}
        for target_idx, target in enumerate(TARGETS):
            y = train[target].astype(int).to_numpy()
            pred, covered = fit_split_source_predictions(x_train, y, folds, spec)
            source_values[:, target_idx] = np.where(covered, pred, reference178[:, target_idx])
            covered_by_target[target] = int(covered.sum())
        source_eval = evaluate_values_on_split(train, folds, source_values)
        source_rows.append(
            {
                "candidate": f"raw_sequence_summary_{spec.name}",
                "model_spec": asdict(spec),
                "kind": "raw_source",
                "mean_loss": source_eval["mean_loss"],
                "target_losses": source_eval["target_losses"],
                "covered_by_target": covered_by_target,
            }
        )
        for weight in BLEND_WEIGHTS:
            blend_values = np.clip((1.0 - weight) * reference178 + weight * source_values, 1e-6, 1 - 1e-6)
            blend_eval = evaluate_values_on_split(train, folds, blend_values)
            blend_rows.append(
                {
                    "candidate": f"blend_id178_sequence_summary_{spec.name}_w{weight:g}",
                    "model_spec": asdict(spec),
                    "kind": "id178_fixed_blend",
                    "weight": weight,
                    "mean_loss": blend_eval["mean_loss"],
                    "target_losses": blend_eval["target_losses"],
                }
            )
    all_rows = sorted([*source_rows, *blend_rows], key=lambda row: (row["mean_loss"], row["candidate"]))
    best = all_rows[0]
    return {
        "split": split_name,
        "n_folds": len(folds),
        "reference_id178": id178_eval,
        "best": best,
        "best_delta_vs_id178": float(best["mean_loss"] - id178_eval["mean_loss"]),
        "top_rows": all_rows[:12],
    }


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    splits = load_split_manifest(SPLIT_MANIFEST_PATH)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, _test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    reference178 = load_reference_values(train, 178)

    split_reports = {
        split_name: evaluate_sequence_source_split(split_name, folds, train, x_train, reference178)
        for split_name, folds in splits.items()
    }
    primary_passes = {
        split_name: split_reports[split_name]["best_delta_vs_id178"] < 0
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_reports
    }
    best_counts: dict[str, int] = {}
    for split_name in PRIMARY_GATE_SPLITS:
        if split_name in split_reports:
            best_name = split_reports[split_name]["best"]["candidate"]
            best_counts[best_name] = best_counts.get(best_name, 0) + 1
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "id340_sequence_summary_split_probe",
        "purpose": "Evaluate sequence-summary source and fixed ID178 blends under the frozen ID340 split manifest.",
        "n_sequence_rows": int(len(seq_table)),
        "n_total_channels": int(len(channel_names)),
        "n_summary_features": int(x_train.shape[1]),
        "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "split_reports": split_reports,
        "primary_split_passes_vs_id178": primary_passes,
        "stable_best_counts_on_primary_splits": best_counts,
        "decision": {
            "promote_submission_candidate": False,
            "create_id340": False,
            "any_sequence_candidate_beats_id178_on_all_primary_splits": all(primary_passes.values())
            if primary_passes
            else False,
            "why": (
                "This is still a report-only split probe. Even if a fixed blend improves one split, ID340 requires "
                "stable gains across the frozen primary lenses plus public-risk geometry before artifact creation."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID340 sequence-summary split probe",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains sequence-summary source models inside the frozen ID340 splits and tests fixed blends against ID178; it creates no submission CSV, base artifact, or DACON submission.",
        "",
        f"- Sequence rows: `{report['n_sequence_rows']}`",
        f"- Total channels: `{report['n_total_channels']}`",
        f"- Summary features: `{report['n_summary_features']}`",
        "",
        "## Best by split",
        "",
        "| split | best | kind | loss | ID178 loss | Δ vs ID178 | pass |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for split_name, split_report in report["split_reports"].items():
        best = split_report["best"]
        id178_loss = split_report["reference_id178"]["mean_loss"]
        delta = split_report["best_delta_vs_id178"]
        lines.append(
            f"| `{split_name}` | `{best['candidate']}` | `{best['kind']}` | "
            f"{best['mean_loss']:.6f} | {id178_loss:.6f} | {delta:+.6f} | `{delta < 0}` |"
        )
    lines.extend(["", "## Primary gate", ""])
    for split_name, passed in report["primary_split_passes_vs_id178"].items():
        lines.append(f"- `{split_name}` beats ID178 reference: `{passed}`")
    lines.extend(["", "Stable best counts on primary splits:", ""])
    for name, count in sorted(
        report["stable_best_counts_on_primary_splits"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Top rows per split", ""])
    for split_name, split_report in report["split_reports"].items():
        id178_loss = split_report["reference_id178"]["mean_loss"]
        lines.extend(
            [
                f"### `{split_name}`",
                "",
                "| rank | candidate | kind | mean loss | Δ vs ID178 |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for rank, row in enumerate(split_report["top_rows"][:8], start=1):
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
            f"- Any sequence candidate beats ID178 on all primary splits: `{report['decision']['any_sequence_candidate_beats_id178_on_all_primary_splits']}`",
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
        default=ROOT / "reports" / "id340_sequence_summary_split_probe_20260618.json",
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
