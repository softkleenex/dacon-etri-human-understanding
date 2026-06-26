#!/usr/bin/env python3
"""No-artifact geometry gate for the post-ID341 Q2 lead.

This diagnostic takes the Q2-only lead from the Q2/Q3/S3 fold-internal model
smoke, fits the source model in memory, and checks local split movement plus
full-test geometry against ID178/287/329/341 anchors. It remains report-only.

It does not write full-test predictions, submission CSVs, base prediction
artifacts, notebook/history artifacts, submission-log rows, Kaggle pushes/runs,
or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke import (  # noqa: E402
    build_feature_metadata_frame,
    feature_columns,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BASE_PREDICTION_DIR,
    EPS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    ModelSpec,
    fold_matrix,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    make_model,
    predict_target_for_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
MODEL_SMOKE_REPORT = REPORTS_DIR / "post_id341_q2q3s3_fold_internal_model_smoke_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json"

PUBLIC_ID178 = 0.6014113009
PUBLIC_ID329 = 0.6017908577
ID341_PROXY = 0.6038297835
RECENT_GAP_FLOOR = min(PUBLIC_ID178 - 0.538638, PUBLIC_ID329 - 0.516718)
PUBLIC_058_GATE = 0.590
SCALES = [0.25, 0.50, 0.75, 1.00, 1.25]
REVIEW_PROXY_MARGIN = 0.00010
REVIEW_MAX_DELTA_VS_341 = 0.040
TARGET = "Q2"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_test_sorted() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def find_base_test(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base test ID{candidate_id}")
    return paths[0]


def load_base_test_values(test: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_base_test(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = test[["subject_id", "date"]].copy()
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Base test ID{candidate_id} does not align with sorted test rows.")
    return merged[TARGETS].astype(float).to_numpy()


def aligned_all_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    features, source_metadata = build_feature_metadata_frame()
    keyed = features.copy()
    keyed["date"] = pd.to_datetime(keyed["row_date"]).dt.date
    train_features = keyed[keyed["is_test"].eq(0)].drop(columns=["is_test"])
    test_features = keyed[keyed["is_test"].eq(1)].drop(columns=["is_test"])
    train_aligned = train[["subject_id", "date"]].merge(
        train_features, on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    test_aligned = test[["subject_id", "date"]].merge(
        test_features, on=["subject_id", "date"], how="left", validate="one_to_one"
    )
    all_features = pd.concat([train_aligned, test_aligned], ignore_index=True, sort=False)
    cols = [col for col in feature_columns(all_features) if pd.api.types.is_numeric_dtype(all_features[col])]
    cols = [col for col in cols if all_features.iloc[: len(train)][col].notna().any()]
    metadata = {
        "source_metadata": source_metadata,
        "train_rows": int(len(train_aligned)),
        "test_rows": int(len(test_aligned)),
        "feature_columns": int(len(cols)),
        "mean_train_missing_rate": float(train_aligned[cols].isna().mean().mean()) if cols else None,
        "mean_test_missing_rate": float(test_aligned[cols].isna().mean().mean()) if cols else None,
    }
    return all_features, cols, metadata


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    if denom <= 0:
        return 0.0
    return float(np.dot(vec, direction) / np.sqrt(denom))


def geometry_report(candidate_test: np.ndarray, base_tests: dict[int, np.ndarray], local_cv: float) -> dict[str, Any]:
    arr178 = base_tests[178]
    arr287 = base_tests[287]
    arr329 = base_tests[329]
    arr341 = base_tests[341]
    vec178 = (candidate_test - arr178).reshape(-1)
    failed_direction = (arr287 - arr178).reshape(-1)
    public_base = min(PUBLIC_ID178, PUBLIC_ID329)
    mean_delta_178 = float(np.mean(np.abs(candidate_test - arr178)))
    max_delta_178 = float(np.max(np.abs(candidate_test - arr178)))
    mean_delta_341 = float(np.mean(np.abs(candidate_test - arr341)))
    max_delta_341 = float(np.max(np.abs(candidate_test - arr341)))
    failed_projection = projection(vec178, failed_direction)
    geometry_proxy = float(
        public_base
        + 0.18 * mean_delta_178
        + 0.020 * max_delta_178
        + 0.12 * mean_delta_341
        + 0.012 * max_delta_341
        + 0.20 * max(0.0, failed_projection)
    )
    local_gap_proxy = float(local_cv + RECENT_GAP_FLOOR)
    proxy = float(max(geometry_proxy, local_gap_proxy))
    return {
        "mean_delta_vs_178": mean_delta_178,
        "max_delta_vs_178": max_delta_178,
        "mean_delta_vs_329": float(np.mean(np.abs(candidate_test - arr329))),
        "max_delta_vs_329": float(np.max(np.abs(candidate_test - arr329))),
        "mean_delta_vs_341": mean_delta_341,
        "max_delta_vs_341": max_delta_341,
        "projection_failed": failed_projection,
        "geometry_proxy": geometry_proxy,
        "local_plus_recent_gap_proxy": local_gap_proxy,
        "proxy_public_risk": proxy,
        "beats_id341_proxy": bool(proxy < ID341_PROXY),
        "public_058_plausible": bool(proxy <= PUBLIC_058_GATE),
    }


def evaluate_values_on_split(
    train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], values: np.ndarray
) -> dict[str, Any]:
    target_losses: dict[str, float] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        total = 0
        weighted = 0.0
        for _fit_idx, valid_idx in folds:
            pred = np.clip(values[valid_idx, target_idx], EPS, 1 - EPS)
            weighted += float(log_loss(y[valid_idx], pred, labels=[0, 1])) * len(valid_idx)
            total += len(valid_idx)
        target_losses[target] = float(weighted / total)
    return {"mean_loss": float(np.mean(list(target_losses.values()))), "target_losses": target_losses}


def load_selected_q2_config() -> dict[str, Any] | None:
    report = json.loads(MODEL_SMOKE_REPORT.read_text(encoding="utf-8"))
    selected = report.get("selected_target_configs", {})
    return selected.get(TARGET)


def split_sources_for_config(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    spec = ModelSpec(**config["model_spec"])
    y = train[TARGET].astype(int).to_numpy()
    return {
        split_name: predict_target_for_split(features_train, feature_cols, y, splits[split_name], spec)[0]
        for split_name in PRIMARY_GATE_SPLITS
    }


def full_test_source_for_config(
    train: pd.DataFrame,
    all_features: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    spec = ModelSpec(**config["model_spec"])
    y = train[TARGET].astype(int).to_numpy()
    n_train = len(train)
    fit_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, len(all_features), dtype=int)
    if len(np.unique(y)) < 2:
        return np.full(len(test_idx), float(np.mean(y)), dtype=float)
    fit_x, test_x = fold_matrix(all_features, feature_cols, fit_idx, test_idx)
    model = make_model(spec)
    model.fit(fit_x, y)
    return np.clip(model.predict_proba(test_x)[:, 1], EPS, 1 - EPS)


def assemble_split_values(
    split_name: str,
    scale: float,
    config: dict[str, Any],
    split_sources: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
) -> np.ndarray:
    values = references["id341"].copy()
    target_idx = TARGETS.index(TARGET)
    anchor_name = config.get("anchor") or "id341"
    anchor = references[anchor_name][:, target_idx]
    source = split_sources[split_name]
    effective_weight = float(config["weight"] * scale)
    values[:, target_idx] = np.where(
        np.isnan(source),
        values[:, target_idx],
        np.clip((1.0 - effective_weight) * anchor + effective_weight * source, EPS, 1 - EPS),
    )
    return values


def assemble_test_values(
    scale: float,
    config: dict[str, Any],
    test_source: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> np.ndarray:
    values = base_tests[341].copy()
    target_idx = TARGETS.index(TARGET)
    anchor_id = 178 if config.get("anchor") == "id178" else 341
    anchor = base_tests[anchor_id][:, target_idx]
    effective_weight = float(config["weight"] * scale)
    values[:, target_idx] = np.clip((1.0 - effective_weight) * anchor + effective_weight * test_source, EPS, 1 - EPS)
    return values


def evaluate_scale(
    scale: float,
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    config: dict[str, Any],
    split_sources: dict[str, np.ndarray],
    test_source: np.ndarray,
    references: dict[str, np.ndarray],
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    split_rows: dict[str, Any] = {}
    for split_name in PRIMARY_GATE_SPLITS:
        folds = splits[split_name]
        values = assemble_split_values(split_name, scale, config, split_sources, references)
        candidate_eval = evaluate_values_on_split(train, folds, values)
        id341_eval = evaluate_values_on_split(train, folds, references["id341"])
        id178_eval = evaluate_values_on_split(train, folds, references["id178"])
        split_rows[split_name] = {
            "candidate_loss": candidate_eval["mean_loss"],
            "id341_loss": id341_eval["mean_loss"],
            "id178_loss": id178_eval["mean_loss"],
            "delta_vs_id341": float(candidate_eval["mean_loss"] - id341_eval["mean_loss"]),
            "delta_vs_id178": float(candidate_eval["mean_loss"] - id178_eval["mean_loss"]),
            "target_losses": candidate_eval["target_losses"],
            "beats_id341": bool(candidate_eval["mean_loss"] < id341_eval["mean_loss"]),
        }
    primary_passes = {split_name: split_rows[split_name]["beats_id341"] for split_name in PRIMARY_GATE_SPLITS}
    local_cv = float(split_rows["subject_hole_5"]["candidate_loss"])
    test_values = assemble_test_values(scale, config, test_source, base_tests)
    geometry = geometry_report(test_values, base_tests, local_cv)
    reviewable = bool(
        primary_passes
        and all(primary_passes.values())
        and geometry["proxy_public_risk"] < ID341_PROXY - REVIEW_PROXY_MARGIN
        and geometry["max_delta_vs_341"] <= REVIEW_MAX_DELTA_VS_341
    )
    return {
        "scale": scale,
        "effective_weight": float(config["weight"] * scale),
        "split_rows": split_rows,
        "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
        "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
        "local_cv_subject_hole_5": local_cv,
        "test_geometry": geometry,
        "reviewable_low_movement_candidate": reviewable,
    }


def build_report() -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    config = load_selected_q2_config()
    scale_rows: list[dict[str, Any]] = []
    feature_metadata: dict[str, Any] = {}
    if config is not None:
        all_features, feature_cols, feature_metadata = aligned_all_features(train, test)
        features_train = all_features.iloc[: len(train)].reset_index(drop=True)
        splits = load_split_manifest()
        references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in [178, 341]}
        base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in [178, 287, 329, 341]}
        split_sources = split_sources_for_config(train, features_train, feature_cols, splits, config)
        test_source = full_test_source_for_config(train, all_features, feature_cols, config)
        scale_rows = [
            evaluate_scale(scale, train, splits, config, split_sources, test_source, references, base_tests)
            for scale in SCALES
        ]
    reviewable_rows = [row for row in scale_rows if row["reviewable_low_movement_candidate"]]
    best_reviewable = (
        sorted(reviewable_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["scale"]))[0]
        if reviewable_rows
        else None
    )
    best_proxy = (
        sorted(scale_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"]))[
            0
        ]
        if scale_rows
        else None
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_q2_lead_geometry_gate",
        "purpose": "No-artifact geometry gate for the Q2-only lead from the Q2/Q3/S3 model smoke.",
        "inputs": {
            "model_smoke_report": rel(MODEL_SMOKE_REPORT),
            "base_tests": {
                str(candidate_id): rel(find_base_test(candidate_id)) for candidate_id in [178, 287, 329, 341]
            },
        },
        "selected_q2_config": config,
        "feature_metadata": feature_metadata,
        "scale_rows": scale_rows,
        "best_reviewable_low_movement": best_reviewable,
        "best_by_proxy": best_proxy,
        "decision": {
            "q2_config_available": bool(config is not None),
            "reviewable_low_movement_candidate_exists": bool(best_reviewable is not None),
            "best_reviewable_scale": None if best_reviewable is None else best_reviewable["scale"],
            "best_proxy": None if best_proxy is None else best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_scale": None if best_proxy is None else best_proxy["scale"],
            "public_058_plausible": False
            if best_proxy is None
            else bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "q3_s3_still_uncovered": True,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Keep the Q2 lead as context only unless a later independent Q3/S3 source appears or the explicit-approved "
                "upstream rebuild confirms broader missing-target coverage."
            ),
        },
        "full_test_predictions_in_memory_only": bool(config is not None),
        "no_full_test_predictions_written": True,
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
    lines = [
        "# Q2 lead geometry gate — post ID341 Q2/Q3/S3 lane",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. Full-test predictions are computed in memory only for geometry inspection and are not written to disk.",
        "",
        "## Selected Q2 config",
        "",
        f"- Config available: `{decision['q2_config_available']}`",
    ]
    config = report.get("selected_q2_config")
    if config:
        lines.extend(
            [
                f"- Candidate: `{config['candidate']}`",
                f"- Weight: `{config['weight']}`",
                f"- Mean Δ341: `{config['mean_delta_vs_id341']:+.6f}`",
                f"- Max Δ341: `{config['max_delta_vs_id341']:+.6f}`",
            ]
        )
    if report["scale_rows"]:
        lines.extend(
            [
                "",
                "## Scale geometry",
                "",
                "| scale | eff weight | primary wins | all primary | local | proxy | max Δ341 | failed proj | reviewable |",
                "|---:|---:|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in report["scale_rows"]:
            geom = row["test_geometry"]
            lines.append(
                f"| {row['scale']:.2f} | {row['effective_weight']:.5f} | {row['primary_wins_vs_id341']} | "
                f"`{row['all_primary_splits_beat_id341']}` | {row['local_cv_subject_hole_5']:.6f} | "
                f"{geom['proxy_public_risk']:.6f} | {geom['max_delta_vs_341']:.6f} | "
                f"{geom['projection_failed']:+.6f} | `{row['reviewable_low_movement_candidate']}` |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Reviewable low-movement candidate exists: `{decision['reviewable_low_movement_candidate_exists']}`",
            f"- Best proxy: `{decision['best_proxy']}` at scale `{decision['best_proxy_scale']}`",
            f"- Public `0.58` plausible: `{decision['public_058_plausible']}`",
            f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
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
