#!/usr/bin/env python3
"""ID342-aware lightweight subject-state PCA gate for Q3/S3.

This report-only diagnostic is a small, non-heavy proxy for the deferred
subject-state representation route. It compresses existing label-free
env/social/routine and registered material feature matrices with fold-internal
PCA, then tests low-capacity LR overlays against ID342.

It creates no prediction CSV, feature matrix, base prediction artifact,
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402,I001
    feature_columns,
    aligned_feature_frame,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
)
from scripts.diagnose_post_id342_registered_material_fusion_gate import build_fusion_features  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_lightweight_subject_state_pca_gate_20260624.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FOCUS_TARGETS = ["Q3", "S3"]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
EPS = 1e-6
BLEND_WEIGHTS = [0.005, 0.01, 0.025, 0.05]
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075


@dataclass(frozen=True)
class PcaSpec:
    name: str
    n_components: int
    c_value: float


PCA_SPECS = (
    PcaSpec("pca004_lr_c0_03", 4, 0.03),
    PcaSpec("pca004_lr_c0_10", 4, 0.10),
    PcaSpec("pca008_lr_c0_03", 8, 0.03),
    PcaSpec("pca008_lr_c0_10", 8, 0.10),
    PcaSpec("pca016_lr_c0_03", 16, 0.03),
    PcaSpec("pca016_lr_c0_10", 16, 0.10),
    PcaSpec("pca032_lr_c0_03", 32, 0.03),
    PcaSpec("pca032_lr_c0_10", 32, 0.10),
    PcaSpec("pca064_lr_c0_03", 64, 0.03),
    PcaSpec("pca064_lr_c0_10", 64, 0.10),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def make_model(spec: PcaSpec, fit_rows: int, feature_count: int):
    n_components = max(1, min(spec.n_components, fit_rows - 1, feature_count))
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=n_components, random_state=53),
        LogisticRegression(C=spec.c_value, solver="liblinear", max_iter=500, random_state=53),
    )


def predict_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: PcaSpec,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            pred[valid_idx] = float(np.mean(y[fit_idx]))
            continue
        model = make_model(spec, len(fit_idx), len(feature_cols))
        model.fit(features.iloc[fit_idx][feature_cols], y[fit_idx])
        pred[valid_idx] = model.predict_proba(features.iloc[valid_idx][feature_cols])[:, 1]
    return np.clip(pred, EPS, 1 - EPS)


def build_feature_sources(train: pd.DataFrame) -> tuple[dict[str, tuple[pd.DataFrame, list[str]]], dict[str, Any]]:
    env_features, env_metadata = aligned_feature_frame(train)
    env_cols = feature_columns(env_features)

    material_features, material_metadata = build_fusion_features(train)
    material_cols = [
        column
        for column in material_features.columns
        if column not in {"subject_id", "date"}
        and pd.api.types.is_numeric_dtype(material_features[column])
        and material_features[column].notna().any()
    ]

    combo = pd.concat(
        [
            env_features[["subject_id", "date"]].reset_index(drop=True),
            env_features[env_cols].add_prefix("env__").reset_index(drop=True),
            material_features[material_cols].add_prefix("mat__").reset_index(drop=True),
        ],
        axis=1,
    )
    combo_cols = [column for column in combo.columns if column not in {"subject_id", "date"}]

    sources = {
        "env_social_routine": (env_features[["subject_id", "date", *env_cols]].copy(), env_cols),
        "registered_material": (
            material_features[["subject_id", "date", *material_cols]].copy(),
            material_cols,
        ),
        "env_plus_material": (combo, combo_cols),
    }
    metadata = {
        "env_social_routine": env_metadata,
        "registered_material": material_metadata,
        "env_plus_material": {
            "aligned_train_rows": int(len(combo)),
            "feature_columns": int(len(combo_cols)),
            "feature_column_sample": combo_cols[:16],
            "missing_rate": float(combo[combo_cols].isna().mean().mean()) if combo_cols else None,
        },
    }
    return sources, metadata


def scan_source(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    source_name: str,
    features: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in PCA_SPECS:
            split_predictions = {
                split_name: predict_split(features, feature_cols, y, splits[split_name], spec)
                for split_name in PRIMARY_SPLITS
            }
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                wins = 0
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    blended = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * split_predictions[split_name],
                        EPS,
                        1.0 - EPS,
                    )
                    anchor_loss = weighted_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_anchor_losses[split_name] = anchor_loss
                    split_losses[split_name] = candidate_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                rows.append(
                    {
                        "source_name": source_name,
                        "target": target,
                        "config": f"{source_name}_{target}_{spec.name}_w{weight:g}",
                        "feature_count": int(len(feature_cols)),
                        "pca_spec": asdict(spec),
                        "weight": float(weight),
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
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    sources, source_metadata = build_feature_sources(train)
    rows: list[dict[str, Any]] = []
    for source_name, (features, feature_cols) in sources.items():
        rows.extend(scan_source(train, splits, anchor_values, source_name, features, feature_cols))
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    selected_targets = sorted({row["target"] for row in eligible_rows})
    estimated_all_target_delta = float(sum(row["mean_delta"] for row in eligible_rows[: len(FOCUS_TARGETS)]) / len(TARGETS))
    projected_public_proxy = ID342_PUBLIC + estimated_all_target_delta
    decision = {
        "eligible_row_count": int(len(eligible_rows)),
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "lightweight_subject_state_pca_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Lightweight PCA over existing label-free matrices does not open Q3/S3. "
            "Do not escalate to Kaggle or artifacts; only a materially different subject-state model remains."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_lightweight_subject_state_pca_gate",
        "purpose": "Non-heavy subject-state representation scout using fold-internal PCA over existing features.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "pca_specs": [asdict(spec) for spec in PCA_SPECS],
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "source_metadata": source_metadata,
        "top_rows": rows[:24],
        "eligible_rows": eligible_rows[:24],
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
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
        "# Post-ID342 lightweight subject-state PCA gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Top Rows",
        "",
        "| source | target | config | wins | mean delta | max delta | eligible |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["top_rows"][:12]:
        lines.append(
            f"| `{row['source_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
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
