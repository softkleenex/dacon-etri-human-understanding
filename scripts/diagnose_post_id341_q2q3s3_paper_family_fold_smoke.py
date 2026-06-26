#!/usr/bin/env python3
"""Paper-family Q2/Q3/S3 fold-internal smoke for post-ID341.

This diagnostic tests narrow source families suggested by the reference-paper
feature schema rather than another broad local recombination pass:

- Q2: mUsageStats + day-of-week/rest-day context
- Q3: wPedo + mUsageStats + day-of-week/rest-day context
- S3: wPedo + mLight/wLight + day-of-week/rest-day context

It consumes already materialized processed train features and frozen validation
lenses, compares only against existing ID178/ID341 OOF anchors, and stays
report-only. It does not create full-test predictions, base prediction
artifacts, submission CSVs, notebook/history artifacts, submission-log rows,
Kaggle pushes/runs, or DACON submissions.
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

REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q2q3s3_paper_family_fold_smoke_20260619.json"

KST = timezone(timedelta(hours=9))
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
META_COLUMNS = {"subject_id", "sleep_date", "lifelog_date", "date", *TARGETS}

SOURCE_GROUP_PATTERNS = {
    "mUsageStats": ["mUsageStats", "mUsage", "usage_"],
    "wPedo": ["wPedo", "pedo_", "slp_pedo"],
    "mLight": ["mLight", "mlight_"],
    "wLight": ["wLight", "wlight_", "slp_wlight"],
    "DW": ["dayofweek", "is_weekend", "is_holiday", "is_rest_day"],
}

TARGET_FAMILIES = {
    "Q2": {
        "paper_family": "q2_mUsageStats_DW",
        "required_source_groups": ["mUsageStats", "DW"],
    },
    "Q3": {
        "paper_family": "q3_wPedo_mUsageStats_DW",
        "required_source_groups": ["wPedo", "mUsageStats", "DW"],
    },
    "S3": {
        "paper_family": "s3_wPedo_mLight_wLight_DW",
        "required_source_groups": ["wPedo", "mLight", "wLight", "DW"],
    },
}


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


def column_matches_source_group(column: str, source_group: str) -> bool:
    if column in META_COLUMNS:
        return False
    lowered = column.lower()
    return any(pattern.lower() in lowered for pattern in SOURCE_GROUP_PATTERNS[source_group])


def numeric_source_columns(frame: pd.DataFrame, source_group: str) -> list[str]:
    return [
        col
        for col in frame.columns
        if column_matches_source_group(col, source_group)
        and pd.api.types.is_numeric_dtype(frame[col])
        and frame[col].notna().any()
    ]


def load_aligned_processed_features(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed_train = pd.read_parquet(PROCESSED_DIR / "df_train_raw.parquet")
    processed_test = pd.read_parquet(PROCESSED_DIR / "df_test_raw.parquet")
    processed_train = processed_train.copy()
    processed_train["date"] = pd.to_datetime(processed_train["lifelog_date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(
        processed_train.drop(columns=["sleep_date", "lifelog_date", *TARGETS], errors="ignore"),
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(train):
        raise ValueError("Processed feature frame does not align with sorted train rows.")
    source_group_columns = {group: numeric_source_columns(aligned, group) for group in SOURCE_GROUP_PATTERNS}
    source_group_test_columns = {
        group: numeric_source_columns(processed_test, group) for group in SOURCE_GROUP_PATTERNS
    }
    family_columns = {
        target: feature_columns_for_target(aligned, target, source_group_columns=source_group_columns)
        for target in TARGET_FOCUS
    }
    metadata = {
        "processed_train": rel(PROCESSED_DIR / "df_train_raw.parquet"),
        "processed_test": rel(PROCESSED_DIR / "df_test_raw.parquet"),
        "processed_train_shape": [int(processed_train.shape[0]), int(processed_train.shape[1])],
        "processed_test_shape": [int(processed_test.shape[0]), int(processed_test.shape[1])],
        "aligned_train_rows": int(len(aligned)),
        "source_group_columns": {group: len(cols) for group, cols in source_group_columns.items()},
        "source_group_test_columns": {group: len(cols) for group, cols in source_group_test_columns.items()},
        "family_columns": {target: len(cols) for target, cols in family_columns.items()},
        "family_missing_rate": {
            target: float(aligned[cols].isna().mean().mean()) if cols else None
            for target, cols in family_columns.items()
        },
        "family_columns_by_target": family_columns,
    }
    return aligned, metadata


def feature_columns_for_target(
    frame: pd.DataFrame,
    target: str,
    source_group_columns: dict[str, list[str]] | None = None,
) -> list[str]:
    source_group_columns = source_group_columns or {
        group: numeric_source_columns(frame, group) for group in SOURCE_GROUP_PATTERNS
    }
    cols: list[str] = []
    for group in TARGET_FAMILIES[target]["required_source_groups"]:
        cols.extend(source_group_columns[group])
    # Preserve source-family order while removing duplicates caused by broad aliases.
    unique_cols = list(dict.fromkeys(cols))
    if not unique_cols:
        raise ValueError(f"No processed paper-family columns found for {target}.")
    return unique_cols


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
    family_name = TARGET_FAMILIES[target]["paper_family"]
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
                    "candidate": f"{family_name}_{spec.name}",
                    "paper_family": family_name,
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
                            "candidate": f"blend_{anchor_name}_{family_name}_{spec.name}_w{weight:g}",
                            "paper_family": family_name,
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
                    "paper_family": row0.get("paper_family"),
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
    features, feature_metadata = load_aligned_processed_features(train)
    splits = load_split_manifest()
    references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    family_feature_counts: dict[str, int] = {}
    for target in TARGET_FOCUS:
        cols = feature_columns_for_target(features, target)
        family_feature_counts[target] = len(cols)
        scan_rows.extend(
            scan_rows_for_target(train, features, cols, splits, references, target, model_specs, split_names)
        )
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    anchor_rows = split_anchor_table(train, splits, references, split_names)
    selected_targets = sorted(selected.keys())
    q3s3_selected = [target for target in selected_targets if target in {"Q3", "S3"}]
    all_targets_covered = selected_targets == TARGET_FOCUS
    top_rows_by_target = {target: rows[:8] for target, rows in summaries.items()}

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_paper_family_fold_smoke",
        "purpose": (
            "Run a narrow fold-internal smoke over processed paper-family source groups for Q2/Q3/S3, "
            "comparing only against ID178/ID341 OOF anchors before any artifact promotion."
        ),
        "target_focus": TARGET_FOCUS,
        "paper_families": TARGET_FAMILIES,
        "inputs": {
            "processed_train": rel(PROCESSED_DIR / "df_train_raw.parquet"),
            "processed_test": rel(PROCESSED_DIR / "df_test_raw.parquet"),
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
            "paper_family_smoke_completed": True,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "selected_q3s3_targets": q3s3_selected,
            "any_paper_family_signal_found": bool(selected_targets),
            "q3s3_covered_by_paper_family": bool(q3s3_selected),
            "all_q2q3s3_targets_covered": all_targets_covered,
            "public_058_evidence": False,
            "next_local_allowed_step": "paper_family_no_artifact_geometry_gate"
            if q3s3_selected
            else "close_paper_family_local_lane",
            "manual_probe_if_spending_submission": None,
            "manual_probe_family_status": "consumed_after_id341_id340_id337_public_micro_probe_slate",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If Q3/S3 is selected, run a no-artifact geometry gate; otherwise keep the score-relevant route "
                "at the explicit-approved clean ID221→ID226 upstream rebuild."
            ),
        },
        "family_feature_counts": family_feature_counts,
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
        "# Q2/Q3/S3 paper-family fold smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Paper-family source groups",
        "",
        "| target | paper family | source groups | feature columns | missing rate |",
        "|---|---|---|---:|---:|",
    ]
    for target in TARGET_FOCUS:
        family = report["paper_families"][target]
        missing_rate = feature_metadata["family_missing_rate"][target]
        lines.append(
            f"| `{target}` | `{family['paper_family']}` | "
            f"`{', '.join(family['required_source_groups'])}` | "
            f"{report['family_feature_counts'][target]} | {missing_rate:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected target configs",
            "",
            "| target | candidate | wins | mean Δ341 | max Δ341 |",
            "|---|---|---:|---:|---:|",
        ]
    )
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
            f"- Paper-family smoke completed: `{decision['paper_family_smoke_completed']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Selected Q3/S3 targets: `{decision['selected_q3s3_targets']}`",
            f"- Q3/S3 covered by paper family: `{decision['q3s3_covered_by_paper_family']}`",
            f"- All Q2/Q3/S3 targets covered: `{decision['all_q2q3s3_targets_covered']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Manual probe if spending a DACON submission: `{decision['manual_probe_if_spending_submission']}`",
            f"- Manual probe family status: `{decision['manual_probe_family_status']}`",
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
