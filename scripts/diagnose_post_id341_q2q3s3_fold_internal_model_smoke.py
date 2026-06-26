#!/usr/bin/env python3
"""Fold-internal Q2/Q3/S3 model smoke for the post-ID341 fallback lane.

This diagnostic consumes the Q2/Q3/S3 feature-generation metadata lane and runs
small fold-internal source models on frozen validation lenses. It compares only
against existing ID178/ID341 OOF anchors and remains report-only.

It does not create full-test predictions, base prediction artifacts, submission
CSVs, notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or
DACON submissions.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke import (  # noqa: E402
    DEFAULT_OUTPUT as FEATURE_METADATA_REPORT,
)
from scripts.diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke import (  # noqa: E402
    build_feature_metadata_frame,
    feature_columns,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q2q3s3_fold_internal_model_smoke_20260618.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TARGET_FOCUS = ["Q2", "Q3", "S3"]
ANCHOR_IDS = [178, 341]
EPS = 1e-6
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
BLEND_WEIGHTS = [0.005, 0.01, 0.025, 0.05]
CONFIG_WIN_FLOOR = 4
CONFIG_MEAN_DELTA_CEILING = -0.00005
CONFIG_MAX_DELTA_CEILING = 0.00075


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_02", "lr", c_value=0.02),
    ModelSpec("lr_c0_08", "lr", c_value=0.08),
    ModelSpec("lr_c0_20", "lr", c_value=0.20),
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
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
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with sorted train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def aligned_feature_frame(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, source_metadata = build_feature_metadata_frame()
    train_features = features[features["is_test"].eq(0)].copy()
    train_features["date"] = pd.to_datetime(train_features["row_date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(
        train_features.drop(columns=["is_test"]), on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    if len(aligned) != len(train):
        raise ValueError("Feature frame does not align with sorted train rows.")
    cols = [col for col in feature_columns(aligned) if pd.api.types.is_numeric_dtype(aligned[col])]
    cols = [col for col in cols if aligned[col].notna().any()]
    metadata = {
        "source_metadata": source_metadata,
        "aligned_train_rows": int(len(aligned)),
        "feature_columns": int(len(cols)),
        "raw_feature_columns": int(len([col for col in cols if not col.endswith("_missing")])),
        "missing_indicator_columns": int(len([col for col in cols if col.endswith("_missing")])),
        "mean_missing_rate": float(aligned[cols].isna().mean().mean()) if cols else None,
    }
    return aligned, metadata


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=43),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def target_loss_on_split(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    values: np.ndarray,
) -> float:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        loss = fold_loss(y[valid_idx], values[valid_idx, target_idx])
        weighted += loss * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


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


def predict_target_for_split(
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
            fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
            model = make_model(spec)
            model.fit(fit_x, y[fit_idx])
            proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
        covered += len(valid_idx)
    return pred, covered


def scan_rows_for_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    target: str,
    model_specs: tuple[ModelSpec, ...],
    split_names: list[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered = predict_target_for_split(features, feature_cols, y, folds, spec)
            source_values_by_anchor = {
                anchor_name: anchor_values.copy() for anchor_name, anchor_values in references.items()
            }
            for anchor_name, values in source_values_by_anchor.items():
                values[:, target_idx] = np.where(np.isnan(pred), references[anchor_name][:, target_idx], pred)
            id178_loss = target_loss_on_split(train, folds, target, references["id178"])
            id341_loss = target_loss_on_split(train, folds, target, references["id341"])
            source_loss = target_loss_on_split(train, folds, target, source_values_by_anchor["id341"])
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"q2q3s3_source_{spec.name}",
                    "model_spec": asdict(spec),
                    "anchor": None,
                    "weight": None,
                    "valid_rows": int(covered),
                    "loss": source_loss,
                    "id178_loss": id178_loss,
                    "id341_loss": id341_loss,
                    "delta_vs_id178": float(source_loss - id178_loss),
                    "delta_vs_id341": float(source_loss - id341_loss),
                }
            )
            for anchor_name, anchor_values in references.items():
                source_values = source_values_by_anchor[anchor_name]
                for weight in BLEND_WEIGHTS:
                    blended = anchor_values.copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                        EPS,
                        1 - EPS,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    anchor_loss = id178_loss if anchor_name == "id178" else id341_loss
                    rows.append(
                        {
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"blend_{anchor_name}_q2q3s3_{spec.name}_w{weight:g}",
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "valid_rows": int(covered),
                            "loss": loss,
                            "id178_loss": id178_loss,
                            "id341_loss": id341_loss,
                            "delta_vs_anchor": float(loss - anchor_loss),
                            "delta_vs_id178": float(loss - id178_loss),
                            "delta_vs_id341": float(loss - id341_loss),
                        }
                    )
    return rows


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not rows:
        return out
    frame = pd.DataFrame(rows)
    for target in TARGET_FOCUS:
        target_frame = frame[frame["target"].eq(target)].copy()
        summaries: list[dict[str, Any]] = []
        for candidate, group in target_frame.groupby("candidate", sort=False):
            deltas = group.set_index("split_name")["delta_vs_id341"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            delta_values = [float(v) for v in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "candidate": str(candidate),
                    "model_spec": row0.get("model_spec"),
                    "anchor": row0.get("anchor"),
                    "weight": row0.get("weight"),
                    "primary_wins_vs_id341": int(sum(delta < 0 for delta in delta_values)),
                    "mean_delta_vs_id341": float(np.mean(delta_values)),
                    "min_delta_vs_id341": float(np.min(delta_values)),
                    "max_delta_vs_id341": float(np.max(delta_values)),
                    "split_deltas_vs_id341": {str(k): float(v) for k, v in deltas.items()},
                    "split_losses": {str(k): float(v) for k, v in losses.items()},
                }
            )
        out[target] = sorted(
            summaries,
            key=lambda row: (
                -row["primary_wins_vs_id341"],
                row["mean_delta_vs_id341"],
                row["max_delta_vs_id341"],
                row["candidate"],
            ),
        )
    return out


def select_target_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in summaries.items():
        for row in rows:
            if (
                row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
                and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
                and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
            ):
                selected[target] = row
                break
    return selected


def split_anchor_table(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    split_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        for split_name in split_names:
            folds = splits[split_name]
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "id178_loss": target_loss_on_split(train, folds, target, references["id178"]),
                    "id341_loss": target_loss_on_split(train, folds, target, references["id341"]),
                }
            )
    return rows


def build_report(
    model_specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    split_names: list[str] | None = None,
) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    train = load_train_sorted()
    features, feature_metadata = aligned_feature_frame(train)
    cols = feature_columns(features)
    cols = [col for col in cols if pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()]
    splits = load_split_manifest()
    references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(
            scan_rows_for_target(train, features, cols, splits, references, target, model_specs, split_names)
        )
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    anchor_rows = split_anchor_table(train, splits, references, split_names)
    selected_targets = sorted(selected.keys())
    all_targets_covered = selected_targets == TARGET_FOCUS
    any_signal = bool(selected_targets)
    top_rows_by_target = {target: rows[:8] for target, rows in summaries.items()}

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_fold_internal_model_smoke",
        "purpose": (
            "Run a lightweight fold-internal model smoke on the Q2/Q3/S3 env-social-routine v2 features, "
            "comparing only against ID178/ID341 OOF anchors before any artifact promotion."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {
            "feature_metadata_report": rel(FEATURE_METADATA_REPORT),
            "split_manifest": rel(SPLIT_MANIFEST_PATH),
            "id178_oof": rel(find_oof(178)),
            "id341_oof": rel(find_oof(341)),
        },
        "feature_metadata": feature_metadata,
        "config": {
            "model_specs": [asdict(spec) for spec in model_specs],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "anchor_rows": anchor_rows,
        "selected_target_configs": selected,
        "top_rows_by_target": top_rows_by_target,
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "model_smoke_completed": True,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "any_q2q3s3_signal_found": any_signal,
            "all_q2q3s3_targets_covered": all_targets_covered,
            "public_058_evidence": False,
            "next_local_allowed_step": "combined_no_artifact_geometry_gate"
            if any_signal
            else "close_local_q2q3s3_model_smoke",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If selected targets exist, run a combined no-artifact geometry gate; otherwise close this local "
                "fallback and return to the explicit-approved upstream rebuild path."
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


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    feature_metadata = report["feature_metadata"]
    lines = [
        "# Q2/Q3/S3 fold-internal model smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Feature input",
        "",
        f"- Aligned train rows: `{feature_metadata['aligned_train_rows']}`",
        f"- Feature columns used: `{feature_metadata['feature_columns']}`",
        f"- Mean missing rate: `{feature_metadata['mean_missing_rate']:.4f}`",
        "",
        "## Selected target configs",
        "",
        "| target | candidate | wins | mean Δ341 | max Δ341 |",
        "|---|---|---:|---:|---:|",
    ]
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    lines.extend(["", "## Top rows by target", ""])
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | candidate | wins | mean Δ341 | max Δ341 |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows[:5], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Model smoke completed: `{decision['model_smoke_completed']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Any Q2/Q3/S3 signal found: `{decision['any_q2q3s3_signal_found']}`",
            f"- All Q2/Q3/S3 targets covered: `{decision['all_q2q3s3_targets_covered']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
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
