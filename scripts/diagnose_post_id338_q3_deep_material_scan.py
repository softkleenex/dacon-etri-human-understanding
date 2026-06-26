#!/usr/bin/env python3
"""Q3-focused deeper material-source scan against ID338.

Report-only diagnostic. It expands beyond the first logistic material scan with
additional low-capacity model families, but still creates no prediction artifact
or submission.
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id338_material_q3s3_probe import (  # noqa: E402
    ANCHOR_ID,
    BLEND_WEIGHTS,
    CONTEXT_ANCHOR_IDS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    build_feature_groups,
    evaluate_candidate_matrix,
    rel,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    EPS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id338_q3_deep_material_scan_20260623.json"
TARGET = "Q3"
CONFIG_WIN_FLOOR = 4
CONFIG_MEAN_DELTA_CEILING = -0.00004
CONFIG_MAX_DELTA_CEILING = 0.00060
CONFIG_MEAN_MOVEMENT_CEILING = 0.00100


@dataclass(frozen=True)
class DeepSpec:
    name: str
    kind: str
    c_value: float | None = None
    seed: int = 43


MODEL_SPECS = (
    DeepSpec("lr_c0_003", "lr", c_value=0.003),
    DeepSpec("lr_c0_01", "lr", c_value=0.01),
    DeepSpec("lr_c0_03", "lr", c_value=0.03),
    DeepSpec("lr_c0_10", "lr", c_value=0.10),
    DeepSpec("lr_c0_35", "lr", c_value=0.35),
    DeepSpec("hgb_l2_leaf7", "hgb"),
    DeepSpec("et_d3_l20", "et"),
    DeepSpec("rf_d3_l20", "rf"),
)


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def make_deep_model(spec: DeepSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=spec.seed),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.025,
                l2_regularization=2.0,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                random_state=spec.seed,
            ),
        )
    if spec.kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=20,
                max_features="sqrt",
                random_state=spec.seed,
                n_jobs=-1,
            ),
        )
    if spec.kind == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=20,
                max_features="sqrt",
                random_state=spec.seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown spec kind: {spec.kind}")


def fold_matrix(
    features: pd.DataFrame,
    feature_cols: list[str],
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = features.iloc[fit_idx]
    query = features.iloc[query_idx]
    usable_cols = [col for col in feature_cols if fit[col].notna().any()]
    fit_raw = fit[usable_cols].reset_index(drop=True)
    query_raw = query[usable_cols].reset_index(drop=True)
    fit_devs: dict[str, pd.Series] = {}
    query_devs: dict[str, pd.Series] = {}
    for col in usable_cols:
        global_median = fit[col].median()
        if pd.isna(global_median):
            global_median = 0.0
        subject_medians = fit.groupby("subject_id")[col].median()
        fit_subject_base = fit["subject_id"].map(subject_medians).fillna(global_median).reset_index(drop=True)
        query_subject_base = query["subject_id"].map(subject_medians).fillna(global_median).reset_index(drop=True)
        fit_devs[f"{col}_dev_subject"] = fit[col].reset_index(drop=True) - fit_subject_base
        query_devs[f"{col}_dev_subject"] = query[col].reset_index(drop=True) - query_subject_base
    return pd.concat([fit_raw, pd.DataFrame(fit_devs)], axis=1), pd.concat(
        [query_raw, pd.DataFrame(query_devs)], axis=1
    )


def predict_for_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: DeepSpec,
) -> tuple[np.ndarray, int]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = 0
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
            model = make_deep_model(spec)
            model.fit(fit_x, y[fit_idx])
            proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
        covered += len(valid_idx)
    return pred, covered


def logloss(values: np.ndarray, y: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(values, EPS, 1 - EPS), labels=[0, 1]))


def scan_source_group(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[int, np.ndarray],
    source_group: str,
    split_names: list[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(TARGET)
    y = train[TARGET].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered = predict_for_split(features, feature_cols, y, folds, spec)
            anchor_loss = target_loss_on_split(train, folds, TARGET, references[ANCHOR_ID])
            source_values = references[ANCHOR_ID].copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), references[ANCHOR_ID][:, target_idx], pred)
            source_loss = target_loss_on_split(train, folds, TARGET, source_values)
            source_metrics = evaluate_candidate_matrix(TARGET, source_values, references)
            rows.append(
                {
                    "source_group": source_group,
                    "target": TARGET,
                    "split_name": split_name,
                    "candidate": f"{source_group}_{spec.name}_source",
                    "model_spec": spec.__dict__,
                    "weight": None,
                    "valid_rows": int(covered),
                    "loss": source_loss,
                    "id338_loss": anchor_loss,
                    "delta_vs_id338": float(source_loss - anchor_loss),
                    **source_metrics,
                }
            )
            for weight in BLEND_WEIGHTS:
                blended = references[ANCHOR_ID].copy()
                blended[:, target_idx] = np.clip(
                    (1.0 - weight) * references[ANCHOR_ID][:, target_idx] + weight * source_values[:, target_idx],
                    EPS,
                    1 - EPS,
                )
                loss = target_loss_on_split(train, folds, TARGET, blended)
                blended_metrics = evaluate_candidate_matrix(TARGET, blended, references)
                rows.append(
                    {
                        "source_group": source_group,
                        "target": TARGET,
                        "split_name": split_name,
                        "candidate": f"blend_id338_{source_group}_{spec.name}_w{weight:g}",
                        "model_spec": spec.__dict__,
                        "weight": weight,
                        "valid_rows": int(covered),
                        "loss": loss,
                        "id338_loss": anchor_loss,
                        "delta_vs_id338": float(loss - anchor_loss),
                        **blended_metrics,
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for candidate, group in frame.groupby("candidate", sort=False):
        row0 = group.iloc[0].to_dict()
        deltas = group.set_index("split_name")["delta_vs_id338"].to_dict()
        movement = group.set_index("split_name")["mean_target_movement_vs_id338"].to_dict()
        projections = group.set_index("split_name")["oof_projection_failed_direction"].to_dict()
        delta_values = [float(value) for value in deltas.values()]
        movement_values = [float(value) for value in movement.values()]
        summaries.append(
            {
                "target": TARGET,
                "source_group": row0["source_group"],
                "candidate": str(candidate),
                "model_spec": row0["model_spec"],
                "weight": safe_float(row0.get("weight")),
                "primary_wins_vs_id338": int(sum(delta < 0 for delta in delta_values)),
                "mean_delta_vs_id338": float(np.mean(delta_values)),
                "min_delta_vs_id338": float(np.min(delta_values)),
                "max_delta_vs_id338": float(np.max(delta_values)),
                "mean_target_movement_vs_id338": float(np.mean(movement_values)),
                "max_target_movement_vs_id338": float(group["max_target_movement_vs_id338"].astype(float).max()),
                "mean_failed_projection": float(np.mean([float(value) for value in projections.values()])),
                "split_deltas_vs_id338": {str(key): float(value) for key, value in deltas.items()},
                "split_mean_movements_vs_id338": {str(key): float(value) for key, value in movement.items()},
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -row["primary_wins_vs_id338"],
            row["mean_delta_vs_id338"],
            row["max_delta_vs_id338"],
            row["mean_target_movement_vs_id338"],
            row["candidate"],
        ),
    )


def select_config(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row.get("weight") is None:
            continue
        if (
            row["primary_wins_vs_id338"] >= CONFIG_WIN_FLOOR
            and row["mean_delta_vs_id338"] <= CONFIG_MEAN_DELTA_CEILING
            and row["max_delta_vs_id338"] <= CONFIG_MAX_DELTA_CEILING
            and row["mean_target_movement_vs_id338"] <= CONFIG_MEAN_MOVEMENT_CEILING
            and row["mean_failed_projection"] <= 0.02
        ):
            return row
    return None


def build_report(split_names: list[str] | None = None) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    train = load_train_sorted()
    features, groups, feature_metadata = build_feature_groups(train)
    splits = load_split_manifest()
    references = {candidate_id: load_reference_values(train, candidate_id) for candidate_id in CONTEXT_ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    for source_group, cols in groups.items():
        scan_rows.extend(scan_source_group(train, features, cols, splits, references, source_group, split_names))
    summaries = summarize(scan_rows)
    selected = select_config(summaries)
    decision_next = (
        "run_q3_no_artifact_geometry_gate_before_any_promotion"
        if selected
        else "q3_remains_closed_after_deep_material_scan"
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id338_q3_deep_material_scan",
        "purpose": "Try a deeper but still low-capacity no-submit Q3 material-source scan against ID338.",
        "target_focus": [TARGET],
        "anchor_id": ANCHOR_ID,
        "feature_metadata": feature_metadata,
        "config": {
            "model_specs": [spec.__dict__ for spec in MODEL_SPECS],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "config_mean_movement_ceiling": CONFIG_MEAN_MOVEMENT_CEILING,
        },
        "selected_config": selected,
        "top_rows": summaries[:30],
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "scan_completed": True,
            "q3_signal_found": selected is not None,
            "selected_targets": [TARGET] if selected else [],
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": decision_next,
            "next_action": (
                "A selected Q3 row requires an in-memory geometry gate before any reserve artifact discussion."
                if selected
                else "Do not promote Q3 from material-source scans; continue with a different source or wait for public feedback."
            ),
        },
        "no_full_test_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return "" if value is None else f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    selected = report["selected_config"]
    lines = [
        "# ID338 Q3 deep material scan",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Selected Config",
        "",
        "| target | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if selected:
        lines.append(
            f"| `{TARGET}` | `{selected['candidate']}` | `{selected['source_group']}` | "
            f"{selected['primary_wins_vs_id338']} | {fmt(selected['mean_delta_vs_id338'])} | "
            f"{fmt(selected['max_delta_vs_id338'])} | {fmt(selected['mean_target_movement_vs_id338'])} | "
            f"{fmt(selected['mean_failed_projection'], 4)} |"
        )
    else:
        lines.append("| _(none)_ |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| rank | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(report["top_rows"][:20], start=1):
        lines.append(
            f"| {rank} | `{row['candidate']}` | `{row['source_group']}` | "
            f"{row['primary_wins_vs_id338']} | {fmt(row['mean_delta_vs_id338'])} | "
            f"{fmt(row['max_delta_vs_id338'])} | {fmt(row['mean_target_movement_vs_id338'])} | "
            f"{fmt(row['mean_failed_projection'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Scan completed: `{decision['scan_completed']}`",
            f"- Q3 signal found: `{decision['q3_signal_found']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
