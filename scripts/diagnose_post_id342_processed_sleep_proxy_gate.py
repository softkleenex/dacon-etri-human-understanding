#!/usr/bin/env python3
"""ID342-aware processed sleep-proxy gate for Q3/S3.

This report-only diagnostic tests whether already-materialized processed
sleep-stage, sleep HR, activity, screen, light, and pre-sleep proxy columns
provide a stable independent overlay on top of ID342. It fits only
fold-internal low-capacity models and uses tiny blend weights.

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402,I001
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
PROCESSED_TRAIN_PATH = ROOT / "data" / "processed" / "df_train_raw.parquet"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_processed_sleep_proxy_gate_20260624.json"

FOCUS_TARGETS = ["Q3", "S3"]
PRIMARY_SPLITS = PRIMARY_GATE_SPLITS
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
EPS = 1e-6

SLEEP_KEYWORDS = ("sleep", "slp", "bed", "wake", "efficiency")
CORE_SLEEP_PREFIXES = ("wHr_sleep_", "wHr_slp_", "slp_")
BLEND_WEIGHTS = [0.005, 0.01, 0.02, 0.05]
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_leaf_nodes: int | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_10", "lr", c_value=0.10),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
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


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=61),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_leaf_nodes=int(spec.max_leaf_nodes),
                learning_rate=0.03,
                max_iter=80,
                l2_regularization=0.10,
                random_state=61,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def load_processed_aligned(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed = pd.read_parquet(PROCESSED_TRAIN_PATH)
    processed = processed.copy()
    processed["date"] = pd.to_datetime(processed["lifelog_date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(
        processed,
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(train) or aligned["lifelog_date"].isna().any():
        raise ValueError(f"Processed sleep-proxy frame does not align with train rows: {len(aligned)} != {len(train)}")
    aligned = aligned.sort_values(["subject_id", "date"]).reset_index(drop=True)
    metadata = {
        "processed_train_path": rel(PROCESSED_TRAIN_PATH),
        "processed_train_shape": [int(processed.shape[0]), int(processed.shape[1])],
        "aligned_train_rows": int(len(aligned)),
        "source_note": (
            "Uses existing processed df_train_raw columns derived from phone/watch sensors around sleep_date; "
            "no new feature matrix is materialized."
        ),
        "source_functions": [
            "src/data/make_dataset.py::extract_sleep_phase_features",
            "src/data/make_dataset.py::extract_sleep_hr_v9",
            "src/data/make_dataset.py::extract_sleep_pedo_v9",
            "src/data/make_dataset.py::extract_sleep_activity_v9",
            "src/data/make_dataset.py::extract_sleep_screen_v9",
            "src/data/make_dataset.py::extract_sleep_light_v9",
        ],
    }
    return aligned, metadata


def sleep_proxy_columns(features: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, Any]]:
    target_and_key_cols = {
        *TARGETS,
        "subject_id",
        "sleep_date",
        "lifelog_date",
        "date",
        "processed_date",
    }
    numeric_sleep_cols = [
        col
        for col in features.columns
        if col not in target_and_key_cols
        and any(keyword in col.lower() for keyword in SLEEP_KEYWORDS)
        and pd.api.types.is_numeric_dtype(features[col])
        and features[col].notna().any()
    ]
    families = {
        "sleep_proxy": numeric_sleep_cols,
        "sleep_proxy_no_ids": [col for col in numeric_sleep_cols if not col.startswith("ch2025_")],
        "core_sleep": [col for col in numeric_sleep_cols if col.startswith(CORE_SLEEP_PREFIXES)],
    }
    metadata = {
        "keyword_hits_numeric": int(len(numeric_sleep_cols)),
        "keywords": list(SLEEP_KEYWORDS),
        "families": {
            name: {
                "feature_count": int(len(cols)),
                "feature_sample": cols[:18],
                "mean_missing_rate": float(features[cols].isna().mean().mean()) if cols else None,
            }
            for name, cols in families.items()
        },
    }
    return families, metadata


def predict_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, int]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = 0
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            model = make_model(spec)
            model.fit(features.iloc[fit_idx][feature_cols], y[fit_idx])
            proba = model.predict_proba(features.iloc[valid_idx][feature_cols])[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
        covered += len(valid_idx)
    return pred, covered


def scan_family(
    train: pd.DataFrame,
    features: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    family_name: str,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            split_predictions: dict[str, np.ndarray] = {}
            split_coverage: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_split(features, feature_cols, y, splits[split_name], spec)
                split_predictions[split_name] = pred
                split_coverage[split_name] = int(covered)
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
                        "family_name": family_name,
                        "target": target,
                        "config": f"{family_name}_{target}_{spec.name}_w{weight:g}",
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_rows_by_split": split_coverage,
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
    features, processed_metadata = load_processed_aligned(train)
    families, family_metadata = sleep_proxy_columns(features)
    rows: list[dict[str, Any]] = []
    for family_name, feature_cols in families.items():
        rows.extend(scan_family(train, features, splits, anchor_values, family_name, feature_cols))
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    selected_targets = sorted({row["target"] for row in eligible_rows})
    estimated_all_target_delta = (
        float(sum(row["mean_delta"] for row in eligible_rows[: len(FOCUS_TARGETS)]) / len(TARGETS))
        if eligible_rows
        else 0.0
    )
    projected_public_proxy = float(ID342_PUBLIC + estimated_all_target_delta)
    best_row = rows[0] if rows else None
    decision = {
        "eligible_row_count": int(len(eligible_rows)),
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "best_row": best_row,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "processed_sleep_proxy_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Processed sleep-proxy columns are present and domain-relevant, but do not open Q3/S3 "
            "against ID342 under the stable primary lenses. Keep as negative evidence; do not create "
            "artifacts, Kaggle runs, or DACON submissions from this lane."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_processed_sleep_proxy_gate",
        "purpose": "Fold-internal Q3/S3 overlay scan over already-materialized processed sleep proxy columns.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "processed_metadata": processed_metadata,
        "family_metadata": family_metadata,
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
    family_metadata = report["family_metadata"]["families"]
    lines = [
        "# Post-ID342 processed sleep-proxy gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Feature Families",
        "",
        "| family | features | mean missing rate | sample |",
        "|---|---:|---:|---|",
    ]
    for family_name, metadata in family_metadata.items():
        sample = ", ".join(f"`{col}`" for col in metadata["feature_sample"][:6])
        lines.append(
            f"| `{family_name}` | {metadata['feature_count']} | {metadata['mean_missing_rate']:.6f} | {sample} |"
        )
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| family | target | config | wins | mean delta | max delta | eligible |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["top_rows"][:12]:
        lines.append(
            f"| `{row['family_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
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
