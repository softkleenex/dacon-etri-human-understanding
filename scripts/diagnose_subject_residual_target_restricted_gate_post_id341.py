#!/usr/bin/env python3
"""Target-restricted subject-residual gate after the post-ID341 smoke.

This report-only diagnostic takes the weak target-level signal from
`subject_residual_sequence_drift_smoke_post_id341` and asks a stricter question:
can a small, fixed Q1/S2/S4-only residual overlay beat ID341 across the frozen
primary validation splits and become transfer-reviewable?

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, no Kaggle push/run, and no DACON submission.
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

from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import load_base_test_values  # noqa: E402
from scripts.diagnose_subject_residual_sequence_drift_smoke_post_id341 import (  # noqa: E402
    BLEND_WEIGHTS,
    EPS,
    ID178_PUBLIC,
    ID341_TRANSFER_PROXY,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    PUBLIC_058_REVIEW_CUTOFF,
    TARGETS,
    build_subject_residual_features,
    evaluate_values_on_split,
    geometry_vs_base_tests,
    load_processed_frames,
    load_reference_values,
    load_split_manifest,
    make_model,
    residual_matrix,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
SMOKE_PATH = REPORTS_DIR / "subject_residual_sequence_drift_smoke_post_id341_20260618.json"

TARGET_SCAN_FOCUS = ["Q1", "S2", "S4"]
ANCHORS = ["id178", "id341"]
CONFIG_WIN_FLOOR = 3
CONFIG_MEAN_DELTA_CEILING = -0.00005
CONFIG_MAX_DELTA_CEILING = 0.00050


def target_loss_on_split(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    pred_values: np.ndarray,
) -> float:
    y = train[target].astype(int).to_numpy()
    total = 0
    weighted = 0.0
    for _fit_idx, valid_idx in folds:
        loss = log_loss(y[valid_idx], np.clip(pred_values[valid_idx], EPS, 1 - EPS), labels=[0, 1])
        weighted += float(loss) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def predict_target_for_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec_name: str,
) -> np.ndarray:
    spec = next(spec for spec in MODEL_SPECS if spec.name == spec_name)
    pred = np.full(len(y), np.nan, dtype=float)
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            fit_x, valid_x = residual_matrix(features.iloc[fit_idx], features.iloc[valid_idx], feature_cols)
            model = make_model(spec)
            model.fit(fit_x, y[fit_idx])
            proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
    return pred


def scan_target_configs(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    references = {"id178": reference178, "id341": reference341}
    source_cache: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for target in TARGET_SCAN_FOCUS:
        target_idx = TARGETS.index(target)
        id341_losses = {
            split_name: target_loss_on_split(train, folds, target, reference341[:, target_idx])
            for split_name, folds in splits.items()
        }
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            for split_name, folds in splits.items():
                source_cache[(split_name, target, spec.name)] = predict_target_for_split(
                    features_train,
                    feature_cols,
                    y,
                    folds,
                    spec.name,
                )
            for anchor_name in ANCHORS:
                anchor_values = references[anchor_name][:, target_idx]
                for weight in BLEND_WEIGHTS:
                    split_deltas: dict[str, float] = {}
                    split_losses: dict[str, float] = {}
                    for split_name, folds in splits.items():
                        pred = source_cache[(split_name, target, spec.name)]
                        source = np.where(np.isnan(pred), reference341[:, target_idx], pred)
                        values = np.clip((1.0 - weight) * anchor_values + weight * source, EPS, 1 - EPS)
                        loss = target_loss_on_split(train, folds, target, values)
                        split_losses[split_name] = loss
                        split_deltas[split_name] = float(loss - id341_losses[split_name])
                    deltas = list(split_deltas.values())
                    rows.append(
                        {
                            "target": target,
                            "model": spec.name,
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "candidate": f"{target}_{anchor_name}_subject_residual_{spec.name}_w{weight:g}",
                            "primary_wins_vs_id341": int(sum(delta < 0 for delta in deltas)),
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
        rows = [row for row in scan_rows if row["target"] == target]
        eligible = [
            row
            for row in rows
            if row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
            and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
            and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
        ]
        if eligible:
            selected[target] = sorted(
                eligible,
                key=lambda row: (-row["primary_wins_vs_id341"], row["mean_delta_vs_id341"], row["max_delta_vs_id341"]),
            )[0]
    return selected


def assemble_values_for_split(
    reference178: np.ndarray,
    reference341: np.ndarray,
    source_cache: dict[tuple[str, str, str], np.ndarray],
    split_name: str,
    selected_configs: dict[str, dict[str, Any]],
) -> np.ndarray:
    references = {"id178": reference178, "id341": reference341}
    values = reference341.copy()
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        pred = source_cache[(split_name, target, config["model"])]
        source = np.where(np.isnan(pred), reference341[:, target_idx], pred)
        anchor = references[config["anchor"]][:, target_idx]
        values[:, target_idx] = np.clip((1.0 - config["weight"]) * anchor + config["weight"] * source, EPS, 1 - EPS)
    return values


def evaluate_assembled_candidate(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
    source_cache: dict[tuple[str, str, str], np.ndarray],
    selected_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split_reports = {}
    for split_name, folds in splits.items():
        values = assemble_values_for_split(reference178, reference341, source_cache, split_name, selected_configs)
        candidate_eval = evaluate_values_on_split(train, folds, values)
        id178_eval = evaluate_values_on_split(train, folds, reference178)
        id341_eval = evaluate_values_on_split(train, folds, reference341)
        split_reports[split_name] = {
            "candidate": candidate_eval,
            "reference_id178": id178_eval,
            "reference_id341": id341_eval,
            "delta_vs_id178": float(candidate_eval["mean_loss"] - id178_eval["mean_loss"]),
            "delta_vs_id341": float(candidate_eval["mean_loss"] - id341_eval["mean_loss"]),
            "beats_id341": bool(candidate_eval["mean_loss"] < id341_eval["mean_loss"]),
        }
    deltas = {split_name: row["delta_vs_id341"] for split_name, row in split_reports.items()}
    return {
        "selected_configs": selected_configs,
        "split_reports": split_reports,
        "primary_deltas_vs_id341": deltas,
        "primary_wins_vs_id341": int(sum(delta < 0 for delta in deltas.values())),
        "all_primary_splits_beat_id341": bool(deltas and all(delta < 0 for delta in deltas.values())),
        "mean_primary_delta_vs_id341": float(np.mean(list(deltas.values()))) if deltas else None,
        "max_primary_delta_vs_id341": float(np.max(list(deltas.values()))) if deltas else None,
    }


def predict_full_train_target(
    train: pd.DataFrame,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    spec_name: str,
) -> np.ndarray:
    spec = next(spec for spec in MODEL_SPECS if spec.name == spec_name)
    y = train[target].astype(int).to_numpy()
    fit_x, test_x = residual_matrix(features_train, features_test, feature_cols)
    if len(np.unique(y)) < 2:
        return np.full(len(features_test), float(np.mean(y)), dtype=float)
    model = make_model(spec)
    model.fit(fit_x, y)
    return np.clip(model.predict_proba(test_x)[:, 1], EPS, 1 - EPS)


def selected_test_geometry(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    feature_cols: list[str],
    selected_configs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not selected_configs:
        return None
    reference178_test = load_base_test_values(test, 178)
    reference341_test = load_base_test_values(test, 341)
    references = {"id178": reference178_test, "id341": reference341_test}
    values = reference341_test.copy()
    full_train_sources = {}
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        source = predict_full_train_target(
            train,
            features_train,
            features_test,
            feature_cols,
            target,
            config["model"],
        )
        full_train_sources[target] = {
            "model": config["model"],
            "source_mean": float(source.mean()),
            "source_std": float(source.std()),
        }
        anchor = references[config["anchor"]][:, target_idx]
        values[:, target_idx] = np.clip((1.0 - config["weight"]) * anchor + config["weight"] * source, EPS, 1 - EPS)
    return {
        "computed_in_memory_only": True,
        "full_train_sources": full_train_sources,
        **geometry_vs_base_tests(test, values),
    }


def build_report() -> dict[str, Any]:
    train, test = load_processed_frames()
    features_train, features_test, feature_coverage = build_subject_residual_features(train, test)
    feature_cols = [col for col in features_train.columns if col.startswith("sr_")]
    splits_all = load_split_manifest()
    splits = {name: splits_all[name] for name in PRIMARY_GATE_SPLITS if name in splits_all}
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows, source_cache = scan_target_configs(
        train, features_train, feature_cols, splits, reference178, reference341
    )
    selected_configs = select_target_configs(scan_rows)
    assembled = evaluate_assembled_candidate(
        train,
        splits,
        reference178,
        reference341,
        source_cache,
        selected_configs,
    )
    mean_delta = assembled["mean_primary_delta_vs_id341"]
    projected_proxy = float(ID341_TRANSFER_PROXY + mean_delta) if mean_delta is not None else ID341_TRANSFER_PROXY
    public_058_reviewable = bool(
        selected_configs
        and assembled["all_primary_splits_beat_id341"]
        and projected_proxy <= PUBLIC_058_REVIEW_CUTOFF
        and projected_proxy < ID178_PUBLIC
    )
    geometry = selected_test_geometry(train, test, features_train, features_test, feature_cols, selected_configs)
    top_scan_by_target = {
        target: [row for row in scan_rows if row["target"] == target][:10] for target in TARGET_SCAN_FOCUS
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "subject_residual_target_restricted_gate_post_id341",
        "purpose": "Apply a fixed target-restricted gate to Q1/S2/S4 subject-residual source signals after the smoke run.",
        "source_smoke_report": str(SMOKE_PATH.relative_to(ROOT)),
        "source_smoke_exists": SMOKE_PATH.exists(),
        "feature_coverage": feature_coverage,
        "target_scan_focus": TARGET_SCAN_FOCUS,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "config_selection_rule": {
            "primary_win_floor": CONFIG_WIN_FLOOR,
            "mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "top_scan_by_target": top_scan_by_target,
        "selected_configs": selected_configs,
        "assembled_candidate": assembled,
        "test_geometry": geometry,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "selected_target_count": int(len(selected_configs)),
            "selected_targets": list(selected_configs),
            "all_primary_splits_beat_id341": assembled["all_primary_splits_beat_id341"],
            "primary_wins_vs_id341": assembled["primary_wins_vs_id341"],
            "mean_primary_delta_vs_id341": mean_delta,
            "max_primary_delta_vs_id341": assembled["max_primary_delta_vs_id341"],
            "projected_public_proxy_from_id341": projected_proxy,
            "beats_id178_public_anchor_proxy": bool(projected_proxy < ID178_PUBLIC),
            "public_058_reviewable": public_058_reviewable,
            "next_action": (
                "Do not create ID342 from this gate unless all primary splits beat ID341 and the projected proxy falls "
                "below the actual ID178 public anchor. If the gate fails, treat Q1/S2/S4 residuals as design hints for "
                "the approved Kaggle phase-2 raw multimodal runner rather than local submission material."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    coverage = report["feature_coverage"]
    decision = report["decision"]
    lines = [
        "# Subject residual target-restricted gate — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Feature and scan setup",
        "",
        f"- Focus targets: `{', '.join(report['target_scan_focus'])}`",
        f"- Base numeric columns: `{coverage['selected_base_column_count']}`",
        f"- Residual feature columns: `{coverage['n_feature_columns']}`",
        f"- Train/test missing mean: `{coverage['train_missing_mean']:.4f}` / `{coverage['test_missing_mean']:.4f}`",
        f"- Selection rule: wins >= `{report['config_selection_rule']['primary_win_floor']}`, "
        f"mean Δ <= `{report['config_selection_rule']['mean_delta_ceiling']}`, "
        f"max Δ <= `{report['config_selection_rule']['max_delta_ceiling']}`",
        "",
        "## Selected configs",
        "",
    ]
    if report["selected_configs"]:
        lines.extend(["| target | config | wins | mean Δ | max Δ |", "|---|---|---:|---:|---:|"])
        for target, row in report["selected_configs"].items():
            lines.append(
                f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
    else:
        lines.append("No target config passed the fixed eligibility rule.")
    lines.extend(
        [
            "",
            "## Assembled candidate by primary split",
            "",
            "| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for split_name, split_report in report["assembled_candidate"]["split_reports"].items():
        candidate = split_report["candidate"]
        id341 = split_report["reference_id341"]
        id178 = split_report["reference_id178"]
        lines.append(
            f"| `{split_name}` | {candidate['mean_loss']:.6f} | {id341['mean_loss']:.6f} | "
            f"{split_report['delta_vs_id341']:+.6f} | {id178['mean_loss']:.6f} | "
            f"{split_report['delta_vs_id178']:+.6f} | `{split_report['beats_id341']}` |"
        )
    lines.extend(
        [
            "",
            "## Best scanned rows by target",
            "",
        ]
    )
    for target, rows in report["top_scan_by_target"].items():
        lines.extend([f"### `{target}`", "", "| config | wins | mean Δ | max Δ |", "|---|---:|---:|---:|"])
        for row in rows[:5]:
            lines.append(
                f"| `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    if report.get("test_geometry"):
        geometry = report["test_geometry"]
        lines.extend(
            [
                "## In-memory test geometry",
                "",
                f"- Mean absolute delta vs ID341 base test: `{geometry['vs_base']['id341']['mean_abs_delta_all_targets']:.6f}`",
                f"- Mean absolute delta vs ID178 base test: `{geometry['vs_base']['id178']['mean_abs_delta_all_targets']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- Selected targets: `{', '.join(decision['selected_targets']) if decision['selected_targets'] else 'none'}`",
            f"- All primary splits beat ID341: `{decision['all_primary_splits_beat_id341']}`",
            f"- Primary wins vs ID341: `{decision['primary_wins_vs_id341']}`",
            f"- Mean primary Δ vs ID341: `{decision['mean_primary_delta_vs_id341']:+.6f}`",
            f"- Max primary Δ vs ID341: `{decision['max_primary_delta_vs_id341']:+.6f}`",
            f"- Projected public proxy from ID341: `{decision['projected_public_proxy_from_id341']:.6f}`",
            f"- Beats ID178 public anchor proxy: `{decision['beats_id178_public_anchor_proxy']}`",
            f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "subject_residual_target_restricted_gate_post_id341_20260618.json",
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
