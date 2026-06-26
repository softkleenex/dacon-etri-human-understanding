#!/usr/bin/env python3
"""Clean ID221-like feature-similarity smoke test after ID341.

This report-only diagnostic replays the historical ID221 feature-similarity
configs under frozen outer splits, fitting imputation/scaling on each outer
training fold only and using only outer-training labels for same-subject nearest
neighbor anchors.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, and makes no DACON submission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
TRAIN_PATH = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PATH = ROOT / "data" / "processed" / "df_test_raw.parquet"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
ID221_EXPERIMENT_PATH = (
    REPORTS_DIR / "experiment_221_id218_public_feature_similarity_overlay_cv0.532990_unsubmitted.json"
)
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
REFERENCE_ID = 178
BASE_ID = 218
HISTORICAL_ID221 = 221
CONTROL_ID = 341
FAILURE_ID = 287
GAP_ID = 329

FAMILY_TOKENS: dict[str, list[str]] = {
    "date_time": ["date", "day", "week", "rank", "frac", "subject_code", "weekend", "holiday"],
    "screen_app": ["screen", "usage", "app", "unlock", "charge", "charging"],
    "move_gps": ["gps", "pedo", "step", "distance", "calorie", "activity", "moving"],
    "light_hr": ["light", "hr", "sleep", "slp"],
}


def find_base_file(kind: str, candidate_id: int) -> Path:
    matches = sorted(BASE_DIR.glob(f"base_{kind}_id_{candidate_id}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing base_{kind} file for ID{candidate_id}")
    return matches[0]


def load_base(kind: str, candidate_id: int, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    path = find_base_file(kind, candidate_id)
    pred = pd.read_csv(path)
    pred["date"] = pred["date"].astype(str)
    missing = [column for column in ["subject_id", "date", *TARGETS] if column not in pred.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if labels is None:
        return pred[["subject_id", "date", *TARGETS]].copy()
    aligned = labels[["row_idx", "subject_id", "date"]].merge(
        pred[["subject_id", "date", *TARGETS]],
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if aligned[TARGETS].isna().any().any():
        raise ValueError(f"{path} does not align to train labels.")
    aligned = aligned.sort_values("row_idx")
    return aligned[["subject_id", "date", *TARGETS]].copy()


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
    test = pd.read_parquet(TEST_PATH).reset_index(drop=True)
    for frame in [train, test]:
        frame["date"] = frame["date"].astype(str)
        frame["date_dt"] = pd.to_datetime(frame["date"])
    return train, test


def label_frame(train: pd.DataFrame) -> pd.DataFrame:
    labels = train[["subject_id", "date", *TARGETS]].copy().reset_index(names="row_idx")
    labels["date"] = labels["date"].astype(str)
    return labels


def numeric_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    out = []
    for column in train.columns:
        if column in TARGETS or column not in test.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[column]) and pd.api.types.is_numeric_dtype(test[column]):
            if train[column].notna().sum() == 0 and test[column].notna().sum() == 0:
                continue
            out.append(column)
    return out


def feature_families(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(train, test)
    families = {}
    for family, tokens in FAMILY_TOKENS.items():
        cols = [column for column in numeric_cols if any(token in column.lower() for token in tokens)]
        if not cols:
            raise ValueError(f"Feature family {family!r} has no numeric columns.")
        families[family] = cols
    return families


def load_historical_configs() -> dict[str, dict[str, Any]]:
    data = json.loads(ID221_EXPERIMENT_PATH.read_text(encoding="utf-8"))
    configs = data.get("selector_info", {}).get("selected_configs", {})
    missing = [target for target in TARGETS if target not in configs]
    if missing:
        raise ValueError(f"ID221 metadata missing selected configs for {missing}")
    return configs


def scaled_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    fit_idx: np.ndarray,
    query_idx: np.ndarray | None = None,
    query_is_test: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    fit_values = imputer.fit_transform(train.iloc[fit_idx][cols])
    fit_scaled = scaler.fit_transform(fit_values)
    if query_is_test:
        query_values = imputer.transform(test[cols])
    else:
        if query_idx is None:
            raise ValueError("query_idx is required when query_is_test=False")
        query_values = imputer.transform(train.iloc[query_idx][cols])
    query_scaled = scaler.transform(query_values)
    return fit_scaled, query_scaled


def same_subject_anchor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_known: np.ndarray,
    x_query: np.ndarray,
    known_idx: np.ndarray,
    query_idx: np.ndarray | None,
    known_y: np.ndarray,
    config: dict[str, Any],
    query_is_test: bool = False,
) -> np.ndarray:
    k = int(config["k"])
    date_weight = float(config.get("date_weight", 0.0))
    known_df = train.iloc[known_idx].reset_index(drop=True)
    query_df = (test if query_is_test else train.iloc[query_idx]).reset_index(drop=True)  # type: ignore[arg-type]
    global_mean = float(np.mean(known_y)) if len(known_y) else 0.5
    out: list[float] = []
    known_subjects = known_df["subject_id"].to_numpy()
    known_dates = known_df["date_dt"].to_numpy()
    for query_pos, row in enumerate(query_df.itertuples(index=False)):
        subject_mask = known_subjects == row.subject_id
        if not subject_mask.any():
            out.append(global_mean)
            continue
        diff = x_known[subject_mask] - x_query[query_pos]
        distances = np.sqrt(np.mean(diff * diff, axis=1))
        if date_weight > 0:
            day_distances = np.abs(
                (known_dates[subject_mask] - np.datetime64(row.date_dt)) / np.timedelta64(1, "D")
            ).astype(float)
            distances = distances + date_weight * np.log1p(day_distances)
        nearest = np.argsort(distances)[: min(k, len(distances))]
        out.append(float(np.mean(known_y[subject_mask][nearest])))
    return np.clip(np.asarray(out, dtype=float), EPS, 1 - EPS)


def blend(base: np.ndarray, anchor: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    weight = float(config["blend_weight"])
    return np.clip((1.0 - weight) * base + weight * anchor, EPS, 1 - EPS)


def binary_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def average_loss(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([binary_loss(y_true[:, i], pred[:, i]) for i in range(len(TARGETS))]))


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def split_rows() -> list[dict[str, Any]]:
    data = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return data.get("folds", [])


def clean_id221_outer_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base218_oof: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    families: dict[str, list[str]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pred = np.zeros((len(train), len(TARGETS)), dtype=float)
    fold_details = []
    for fold in split_rows():
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        fold_target_losses = {}
        for target_idx, target in enumerate(TARGETS):
            config = configs[target]
            cols = families[str(config["family"])]
            x_known, x_valid = scaled_fold_features(train, test, cols, train_idx, valid_idx, query_is_test=False)
            anchor = same_subject_anchor(
                train=train,
                test=test,
                x_known=x_known,
                x_query=x_valid,
                known_idx=train_idx,
                query_idx=valid_idx,
                known_y=train.iloc[train_idx][target].astype(int).to_numpy(),
                config=config,
                query_is_test=False,
            )
            base_values = base218_oof.iloc[valid_idx][target].astype(float).to_numpy()
            pred[valid_idx, target_idx] = blend(base_values, anchor, config)
            fold_target_losses[target] = binary_loss(
                train.iloc[valid_idx][target].to_numpy(), pred[valid_idx, target_idx]
            )
        fold_details.append(
            {
                "split_name": fold["split_name"],
                "fold_id": int(fold["fold_id"]),
                "valid_n": int(len(valid_idx)),
                "loss": float(np.mean(list(fold_target_losses.values()))),
                "target_losses": fold_target_losses,
            }
        )
    return pred, fold_details


def full_train_test_preview(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base218_test: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    families: dict[str, list[str]],
) -> np.ndarray:
    pred = np.zeros((len(test), len(TARGETS)), dtype=float)
    fit_idx = np.arange(len(train))
    for target_idx, target in enumerate(TARGETS):
        config = configs[target]
        cols = families[str(config["family"])]
        x_known, x_test = scaled_fold_features(train, test, cols, fit_idx, query_is_test=True)
        anchor = same_subject_anchor(
            train=train,
            test=test,
            x_known=x_known,
            x_query=x_test,
            known_idx=fit_idx,
            query_idx=None,
            known_y=train[target].astype(int).to_numpy(),
            config=config,
            query_is_test=True,
        )
        base_values = base218_test[target].astype(float).to_numpy()
        pred[:, target_idx] = blend(base_values, anchor, config)
    return pred


def split_summary(
    fold_details: list[dict[str, Any]],
    y_true: np.ndarray,
    candidate: np.ndarray,
    references: dict[str, np.ndarray],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fold_rows = []
    for fold, detail in zip(split_rows(), fold_details, strict=True):
        valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        row = {
            "split_name": detail["split_name"],
            "fold_id": detail["fold_id"],
            "loss": detail["loss"],
            "valid_n": detail["valid_n"],
        }
        for name, ref in references.items():
            ref_loss = average_loss(y_true[valid_idx], ref[valid_idx])
            delta = float(detail["loss"] - ref_loss)
            row[f"delta_vs_{name}"] = delta
            by_split[detail["split_name"]][f"delta_vs_{name}"].append(delta)
        by_split[detail["split_name"]]["loss"].append(float(detail["loss"]))
        fold_rows.append(row)
    split_out = {
        split: {
            "mean_loss": float(np.mean(values["loss"])),
            **{key: float(np.mean(value)) for key, value in values.items() if key.startswith("delta_vs_")},
        }
        for split, values in by_split.items()
    }
    primary = [
        "subject_hole_5",
        "actual_test_band_holdout",
        "test_boundary_neighbor2_holdout",
        "test_boundary_neighbor5_holdout",
        "loso_10",
    ]
    primary_pass = {
        name.removeprefix("id"): all(split_out[split][f"delta_vs_{name}"] < 0 for split in primary)
        for name in references
    }
    return {
        "fold_rows": fold_rows,
        "split_summary": split_out,
        "primary_pass": {f"id{name}" if name.isdigit() else name: bool(value) for name, value in primary_pass.items()},
    }


def build_report() -> dict[str, Any]:
    train, test = load_train_test()
    labels = label_frame(train)
    y_true = train[TARGETS].astype(int).to_numpy()
    configs = load_historical_configs()
    families = feature_families(train, test)
    base218_oof = load_base("oof", BASE_ID, labels)
    base218_test = load_base("test", BASE_ID)
    reference_oof = {
        candidate_id: load_base("oof", candidate_id, labels)[TARGETS].astype(float).to_numpy()
        for candidate_id in [178, 218, 221, 341]
    }
    candidate_oof, fold_details = clean_id221_outer_predictions(train, test, base218_oof, configs, families)
    candidate_test = full_train_test_preview(train, test, base218_test, configs, families)
    base_test_arrays = {
        candidate_id: load_base("test", candidate_id)[TARGETS].astype(float).to_numpy()
        for candidate_id in [REFERENCE_ID, BASE_ID, HISTORICAL_ID221, CONTROL_ID, FAILURE_ID, GAP_ID]
    }
    vec = (candidate_test - base_test_arrays[REFERENCE_ID]).reshape(-1)
    failed_direction = (base_test_arrays[FAILURE_ID] - base_test_arrays[REFERENCE_ID]).reshape(-1)
    gap_direction = (base_test_arrays[GAP_ID] - base_test_arrays[REFERENCE_ID]).reshape(-1)
    full_loss = average_loss(y_true, candidate_oof)
    target_losses = {target: binary_loss(y_true[:, idx], candidate_oof[:, idx]) for idx, target in enumerate(TARGETS)}
    references = {
        "id178": reference_oof[178],
        "id218": reference_oof[218],
        "id221": reference_oof[221],
        "id341": reference_oof[341],
    }
    summary = split_summary(fold_details, y_true, candidate_oof, references)
    family_sizes = {family: len(cols) for family, cols in families.items()}
    used_families = Counter(str(config["family"]) for config in configs.values())
    geometry = {
        "mean_delta_vs_id178": float(np.mean(np.abs(candidate_test - base_test_arrays[REFERENCE_ID]))),
        "max_delta_vs_id178": float(np.max(np.abs(candidate_test - base_test_arrays[REFERENCE_ID]))),
        "mean_delta_vs_id341": float(np.mean(np.abs(candidate_test - base_test_arrays[CONTROL_ID]))),
        "max_delta_vs_id341": float(np.max(np.abs(candidate_test - base_test_arrays[CONTROL_ID]))),
        "mean_delta_vs_historical_id221": float(np.mean(np.abs(candidate_test - base_test_arrays[HISTORICAL_ID221]))),
        "max_delta_vs_historical_id221": float(np.max(np.abs(candidate_test - base_test_arrays[HISTORICAL_ID221]))),
        "projection_failed_id287": projection(vec, failed_direction),
        "projection_gap_id329": projection(vec, gap_direction),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_clean_id221_feature_similarity_smoke",
        "purpose": "Replay historical ID221 feature-similarity configs under frozen outer splits with fold-internal preprocessing and outer-train-only labels.",
        "base_id": BASE_ID,
        "historical_id221_metadata": str(ID221_EXPERIMENT_PATH.relative_to(ROOT)),
        "feature_family_sizes": family_sizes,
        "used_family_counts": dict(used_families),
        "selected_configs": configs,
        "full_oof": {
            "candidate_loss": full_loss,
            "target_losses": target_losses,
            "delta_vs_id178": float(full_loss - average_loss(y_true, references["id178"])),
            "delta_vs_id218": float(full_loss - average_loss(y_true, references["id218"])),
            "delta_vs_historical_id221": float(full_loss - average_loss(y_true, references["id221"])),
            "delta_vs_id341": float(full_loss - average_loss(y_true, references["id341"])),
        },
        "split_gate": summary,
        "test_geometry_preview_no_artifact": geometry,
        "decision": {
            "clean_id221_fixed_config_signal": bool(summary["primary_pass"]["id218"]),
            "beats_id178_primary": bool(summary["primary_pass"]["id178"]),
            "beats_id341_primary": bool(summary["primary_pass"]["id341"]),
            "matches_or_beats_historical_id221": bool(full_loss <= average_loss(y_true, references["id221"])),
            "promote_artifact": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "Use this as a smoke read only. If the fixed historical configs do not survive, the next clean ID221 rebuild "
                "must use inner-fold config selection rather than replaying the public-structure ID221 choices."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 clean ID221 feature-similarity smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Read",
        "",
        report["purpose"],
        "",
        "## OOF summary",
        "",
        f"- Candidate loss: `{report['full_oof']['candidate_loss']:.6f}`",
        f"- Δ vs ID178: `{report['full_oof']['delta_vs_id178']:+.6f}`",
        f"- Δ vs ID218 base: `{report['full_oof']['delta_vs_id218']:+.6f}`",
        f"- Δ vs historical ID221: `{report['full_oof']['delta_vs_historical_id221']:+.6f}`",
        f"- Δ vs ID341: `{report['full_oof']['delta_vs_id341']:+.6f}`",
        "",
        "## Split gate",
        "",
        "| split | loss | Δ178 | Δ218 | Δ221 | Δ341 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, row in report["split_gate"]["split_summary"].items():
        lines.append(
            f"| `{split}` | {row['mean_loss']:.6f} | {row['delta_vs_id178']:+.6f} | "
            f"{row['delta_vs_id218']:+.6f} | {row['delta_vs_id221']:+.6f} | {row['delta_vs_id341']:+.6f} |"
        )
    geom = report["test_geometry_preview_no_artifact"]
    lines.extend(
        [
            "",
            "## Test geometry preview",
            "",
            f"- Mean/max Δ vs ID178: `{geom['mean_delta_vs_id178']:.6f}` / `{geom['max_delta_vs_id178']:.6f}`",
            f"- Mean/max Δ vs ID341: `{geom['mean_delta_vs_id341']:.6f}` / `{geom['max_delta_vs_id341']:.6f}`",
            f"- Projection onto ID287 failed direction: `{geom['projection_failed_id287']:+.4f}`",
            f"- Projection onto ID329 gap direction: `{geom['projection_gap_id329']:+.4f}`",
            "",
            "## Decision",
            "",
            f"- Clean ID221 fixed-config signal: `{report['decision']['clean_id221_fixed_config_signal']}`",
            f"- Beats ID178 primary: `{report['decision']['beats_id178_primary']}`",
            f"- Beats ID341 primary: `{report['decision']['beats_id341_primary']}`",
            f"- Matches/beats historical ID221: `{report['decision']['matches_or_beats_historical_id221']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
            f"- Run heavy rebuild now: `{report['decision']['run_heavy_rebuild_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_clean_id221_feature_similarity_smoke_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
