#!/usr/bin/env python3
"""Target-restricted sleep-boundary v2 gate after the post-ID341 smoke.

This report-only diagnostic scans fixed target-level sleep-boundary configs on
frozen primary splits, assembles a conservative target-restricted candidate in
memory, trains full-train source models only to inspect test geometry, and runs a
public-risk proxy. It creates no submission CSV, base predictions, notebook, log
row, or DACON submission.
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
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    BASE_PREDICTION_DIR,
    BLEND_WEIGHTS,
    EPS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    TARGETS,
    build_sleep_boundary_features,
    evaluate_values_on_split,
    fold_matrix,
    load_reference_values,
    load_split_manifest,
    make_model,
    predict_target_for_split,
)
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
PUBLIC_ID178 = 0.6014113009
PUBLIC_ID329 = 0.6017908577
PUBLIC_ID287 = 0.6698787318
ID341_PROXY = 0.6038297835
RECENT_GAP_FLOOR = min(PUBLIC_ID178 - 0.538638, PUBLIC_ID329 - 0.516718)
TARGET_SCAN_FOCUS = ["Q1", "S1", "S2", "S4"]
ANCHOR_IDS = [178, 287, 329, 341]
CONFIG_WIN_FLOOR = 4
CONFIG_MAX_DELTA_CEILING = 0.0010
CONFIG_MEAN_DELTA_CEILING = -0.00020
PUBLIC_058_GATE = 0.590


def find_base_test(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base test ID{candidate_id}")
    return paths[0]


def load_base_test_values(test: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_base_test(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = test[["subject_id", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.date
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Base test ID{candidate_id} does not align with test rows.")
    return merged[TARGETS].astype(float).to_numpy()


def target_loss_on_split(
    train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], target: str, values: np.ndarray
) -> float:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    total = 0
    weighted = 0.0
    for _fit_idx, valid_idx in folds:
        pred = values[valid_idx, target_idx]
        weighted += float(log_loss(y[valid_idx], np.clip(pred, EPS, 1 - EPS), labels=[0, 1])) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def scan_target_configs(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    source_cache: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    references = {"id178": reference178, "id341": reference341}
    for target in TARGET_SCAN_FOCUS:
        target_idx = TARGETS.index(target)
        id341_losses = {
            split_name: target_loss_on_split(train, splits[split_name], target, reference341)
            for split_name in PRIMARY_GATE_SPLITS
        }
        for spec in MODEL_SPECS:
            for split_name in PRIMARY_GATE_SPLITS:
                pred, _covered = predict_target_for_split(
                    features_train,
                    feature_cols,
                    train[target].astype(int).to_numpy(),
                    splits[split_name],
                    spec,
                )
                source_values = reference341.copy()
                source_values[:, target_idx] = np.where(np.isnan(pred), reference341[:, target_idx], pred)
                source_cache[(split_name, target, spec.name)] = source_values
            for anchor_name, anchor_values in references.items():
                for weight in BLEND_WEIGHTS:
                    split_deltas: dict[str, float] = {}
                    split_losses: dict[str, float] = {}
                    for split_name in PRIMARY_GATE_SPLITS:
                        source_values = source_cache[(split_name, target, spec.name)]
                        values = reference341.copy()
                        values[:, target_idx] = np.clip(
                            (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                            EPS,
                            1 - EPS,
                        )
                        loss = target_loss_on_split(train, splits[split_name], target, values)
                        split_losses[split_name] = loss
                        split_deltas[split_name] = float(loss - id341_losses[split_name])
                    deltas = list(split_deltas.values())
                    wins = sum(delta < 0 for delta in deltas)
                    rows.append(
                        {
                            "target": target,
                            "model": spec.name,
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "candidate": f"{target}_{anchor_name}_{spec.name}_w{weight:g}",
                            "primary_wins_vs_id341": int(wins),
                            "mean_delta_vs_id341": float(np.mean(deltas)),
                            "min_delta_vs_id341": float(np.min(deltas)),
                            "max_delta_vs_id341": float(np.max(deltas)),
                            "split_deltas_vs_id341": split_deltas,
                            "split_losses": split_losses,
                        }
                    )
    rows = sorted(
        rows,
        key=lambda row: (
            row["target"],
            -row["primary_wins_vs_id341"],
            row["mean_delta_vs_id341"],
            row["max_delta_vs_id341"],
        ),
    )
    return rows, source_cache


def select_target_configs(scan_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target in TARGET_SCAN_FOCUS:
        target_rows = [row for row in scan_rows if row["target"] == target]
        eligible = [
            row
            for row in target_rows
            if row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
            and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
            and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
        ]
        if not eligible:
            continue
        selected[target] = sorted(eligible, key=lambda row: (row["mean_delta_vs_id341"], row["max_delta_vs_id341"]))[0]
    return selected


def assemble_values_for_split(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    selected_configs: dict[str, dict[str, Any]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> np.ndarray:
    values = reference341.copy()
    anchors = {"id178": reference178, "id341": reference341}
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        spec = next(spec for spec in MODEL_SPECS if spec.name == config["model"])
        pred, _covered = predict_target_for_split(
            features_train, feature_cols, train[target].astype(int).to_numpy(), folds, spec
        )
        source = np.where(np.isnan(pred), reference341[:, target_idx], pred)
        anchor = anchors[config["anchor"]][:, target_idx]
        values[:, target_idx] = np.clip((1.0 - config["weight"]) * anchor + config["weight"] * source, EPS, 1 - EPS)
    return values


def evaluate_assembled_candidate(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    selected_configs: dict[str, dict[str, Any]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> dict[str, Any]:
    split_reports = {}
    for split_name, folds in splits.items():
        values = assemble_values_for_split(
            train, features_train, feature_cols, folds, selected_configs, reference178, reference341
        )
        candidate_eval = evaluate_values_on_split(train, folds, values)
        id178_eval = evaluate_values_on_split(train, folds, reference178)
        id341_eval = evaluate_values_on_split(train, folds, reference341)
        split_reports[split_name] = {
            "split": split_name,
            "candidate": candidate_eval,
            "reference_id178": id178_eval,
            "reference_id341": id341_eval,
            "delta_vs_id178": float(candidate_eval["mean_loss"] - id178_eval["mean_loss"]),
            "delta_vs_id341": float(candidate_eval["mean_loss"] - id341_eval["mean_loss"]),
            "beats_id341": bool(candidate_eval["mean_loss"] < id341_eval["mean_loss"]),
        }
    return split_reports


def full_train_source_test_prediction(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    selected_configs: dict[str, dict[str, Any]],
    base_tests: dict[int, np.ndarray],
) -> np.ndarray:
    n_train = len(train)
    n_total = len(features)
    fit_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n_total, dtype=int)
    candidate = base_tests[341].copy()
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        spec = next(spec for spec in MODEL_SPECS if spec.name == config["model"])
        anchor_id = 178 if config["anchor"] == "id178" else 341
        y = train[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            source_pred = np.full(len(test_idx), float(np.mean(y)), dtype=float)
        else:
            fit_x, test_x = fold_matrix(features, feature_cols, fit_idx, test_idx)
            model = make_model(spec)
            model.fit(fit_x, y)
            source_pred = model.predict_proba(test_x)[:, 1]
        anchor = base_tests[anchor_id][:, target_idx]
        candidate[:, target_idx] = np.clip(
            (1.0 - config["weight"]) * anchor + config["weight"] * source_pred, EPS, 1 - EPS
        )
    return candidate


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


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


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    features, feature_coverage = build_sleep_boundary_features(train, test)
    feature_cols = [col for col in features.columns if col.startswith("sbt_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows, _source_cache = scan_target_configs(
        train, features_train, feature_cols, splits, reference178, reference341
    )
    selected_configs = select_target_configs(scan_rows)
    split_reports = evaluate_assembled_candidate(
        train, features_train, feature_cols, splits, selected_configs, reference178, reference341
    )
    primary_passes = {
        split_name: split_reports[split_name]["beats_id341"]
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_reports
    }
    subject_hole_local = float(split_reports["subject_hole_5"]["candidate"]["mean_loss"])
    base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}
    candidate_test = full_train_source_test_prediction(train, features, feature_cols, selected_configs, base_tests)
    geometry = geometry_report(candidate_test, base_tests, subject_hole_local)
    top_scan_rows = {target: [row for row in scan_rows if row["target"] == target][:10] for target in TARGET_SCAN_FOCUS}
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sleep_boundary_transition_v2_gate_post_id341",
        "purpose": "Target-restricted report-only gate for sleep_boundary_transition_v1 after smoke signal.",
        "feature_coverage": feature_coverage,
        "config_rules": {
            "target_scan_focus": TARGET_SCAN_FOCUS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "blend_weights": BLEND_WEIGHTS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        },
        "selected_target_configs": selected_configs,
        "top_scan_rows_by_target": top_scan_rows,
        "split_reports": split_reports,
        "primary_passes_vs_id341": primary_passes,
        "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
        "local_cv_subject_hole_5": subject_hole_local,
        "test_geometry": geometry,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "selected_targets": sorted(selected_configs.keys()),
            "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
            "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
            "beats_id341_proxy": geometry["beats_id341_proxy"],
            "public_058_plausible": geometry["public_058_plausible"],
            "promote_next_artifact": False,
            "next_action": (
                "If proxy remains above ID341, keep v2 as a feature-source lead but do not promote. "
                "If it beats ID341 proxy in a future v3, require full artifact contract with unique ID > 341."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    geometry = report["test_geometry"]
    lines = [
        "# Sleep-boundary transition v2 gate — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Selected target configs",
        "",
        "| target | candidate | wins | mean Δ | max Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
        )
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Assembled candidate by split",
            "",
            "| split | candidate loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for split_name, split_report in report["split_reports"].items():
        lines.append(
            f"| `{split_name}` | {split_report['candidate']['mean_loss']:.6f} | "
            f"{split_report['reference_id341']['mean_loss']:.6f} | {split_report['delta_vs_id341']:+.6f} | "
            f"{split_report['reference_id178']['mean_loss']:.6f} | {split_report['delta_vs_id178']:+.6f} | "
            f"`{split_report['beats_id341']}` |"
        )
    lines.extend(
        [
            "",
            "## Test geometry / public-risk proxy",
            "",
            f"- Local CV proxy (subject_hole_5): `{report['local_cv_subject_hole_5']:.6f}`",
            f"- Mean/max Δ vs ID178: `{geometry['mean_delta_vs_178']:.6f}` / `{geometry['max_delta_vs_178']:.6f}`",
            f"- Mean/max Δ vs ID341: `{geometry['mean_delta_vs_341']:.6f}` / `{geometry['max_delta_vs_341']:.6f}`",
            f"- Failed-direction projection: `{geometry['projection_failed']:+.6f}`",
            f"- Geometry proxy: `{geometry['geometry_proxy']:.6f}`",
            f"- Local+recent-gap proxy: `{geometry['local_plus_recent_gap_proxy']:.6f}`",
            f"- Conservative proxy: `{geometry['proxy_public_risk']:.6f}`",
            f"- Beats ID341 proxy: `{geometry['beats_id341_proxy']}`",
            f"- Public `0.58` plausible: `{geometry['public_058_plausible']}`",
            "",
            "## Decision",
            "",
            f"- Selected targets: `{', '.join(report['decision']['selected_targets'])}`",
            f"- All primary splits beat ID341: `{report['decision']['all_primary_splits_beat_id341']}`",
            f"- Primary wins vs ID341: `{report['decision']['primary_wins_vs_id341']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
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
        default=REPORTS_DIR / "sleep_boundary_transition_v2_gate_post_id341_20260618.json",
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
