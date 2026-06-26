#!/usr/bin/env python3
"""Nested clean-ID221 feature-similarity selection probe after ID341.

This report-only diagnostic follows the clean-backbone contract more closely
than the fixed-config smoke: for each frozen outer split and target, it selects
feature-similarity configs using inner folds from the outer-training rows only,
then scores the held-out outer-valid rows.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, and makes no DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_clean_id221_feature_similarity_smoke import (  # noqa: E402
    BASE_ID,
    CONTROL_ID,
    FAILURE_ID,
    GAP_ID,
    HISTORICAL_ID221,
    REFERENCE_ID,
    TARGETS,
    average_loss,
    binary_loss,
    blend,
    feature_families,
    label_frame,
    load_base,
    load_historical_configs,
    load_train_test,
    projection,
    same_subject_anchor,
    scaled_fold_features,
    split_rows,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))


def candidate_grid(historical_config: dict[str, Any]) -> list[dict[str, Any]]:
    family = str(historical_config["family"])
    historical_k = int(historical_config["k"])
    historical_date_weight = float(historical_config.get("date_weight", 0.0))
    historical_blend_weight = float(historical_config["blend_weight"])
    k_values = [historical_k]
    date_weights = sorted({0.0, historical_date_weight})
    blend_weights = sorted({0.005, 0.01, min(0.03, historical_blend_weight), historical_blend_weight})
    configs = []
    for k in k_values:
        for date_weight in date_weights:
            for blend_weight in blend_weights:
                configs.append(
                    {
                        "name": f"{family}_k{k}_dw{date_weight:g}_w{blend_weight:g}",
                        "family": family,
                        "k": k,
                        "date_weight": date_weight,
                        "blend_weight": blend_weight,
                    }
                )
    return configs


def make_inner_folds(train, outer_train_idx: np.ndarray, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    inner_valid_by_fold: list[list[int]] = [[] for _ in range(n_splits)]
    outer_frame = train.iloc[outer_train_idx].copy()
    outer_frame["_row_idx"] = outer_train_idx
    for _, subject_frame in outer_frame.sort_values(["subject_id", "date_dt"]).groupby("subject_id", sort=False):
        chunks = np.array_split(subject_frame["_row_idx"].to_numpy(), n_splits)
        for fold_idx, chunk in enumerate(chunks):
            inner_valid_by_fold[fold_idx].extend(int(value) for value in chunk)
    folds = []
    outer_set = {int(value) for value in outer_train_idx}
    for valid_values in inner_valid_by_fold:
        valid_idx = np.asarray(sorted(valid_values), dtype=int)
        valid_set = {int(value) for value in valid_idx}
        train_idx = np.asarray(sorted(outer_set - valid_set), dtype=int)
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def predict_config(
    train,
    test,
    families: dict[str, list[str]],
    base218,
    target: str,
    known_idx: np.ndarray,
    query_idx: np.ndarray | None,
    config: dict[str, Any],
    query_is_test: bool = False,
) -> np.ndarray:
    cols = families[str(config["family"])]
    x_known, x_query = scaled_fold_features(
        train=train,
        test=test,
        cols=cols,
        fit_idx=known_idx,
        query_idx=query_idx,
        query_is_test=query_is_test,
    )
    anchor = same_subject_anchor(
        train=train,
        test=test,
        x_known=x_known,
        x_query=x_query,
        known_idx=known_idx,
        query_idx=query_idx,
        known_y=train.iloc[known_idx][target].astype(int).to_numpy(),
        config=config,
        query_is_test=query_is_test,
    )
    base_values = (
        base218[target].astype(float).to_numpy()
        if query_is_test
        else base218.iloc[query_idx][target].astype(float).to_numpy()
    )
    return blend(base_values, anchor, config)


def select_config_for_outer_train(
    train,
    test,
    families: dict[str, list[str]],
    base218_oof,
    target: str,
    outer_train_idx: np.ndarray,
    configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inner_folds = make_inner_folds(train, outer_train_idx)
    y_outer = train.iloc[outer_train_idx][target].astype(int).to_numpy()
    base_outer = base218_oof.iloc[outer_train_idx][target].astype(float).to_numpy()
    base_loss = binary_loss(y_outer, base_outer)
    candidate_rows: list[dict[str, Any]] = [
        {"name": "base", "loss": base_loss, "config": {"name": "base", "kind": "base"}}
    ]
    for config in configs:
        inner_pred = np.zeros(len(outer_train_idx), dtype=float)
        index_to_pos = {int(row_idx): pos for pos, row_idx in enumerate(outer_train_idx)}
        for inner_train_idx, inner_valid_idx in inner_folds:
            pred = predict_config(
                train=train,
                test=test,
                families=families,
                base218=base218_oof,
                target=target,
                known_idx=inner_train_idx,
                query_idx=inner_valid_idx,
                config=config,
                query_is_test=False,
            )
            for row_idx, value in zip(inner_valid_idx, pred, strict=True):
                inner_pred[index_to_pos[int(row_idx)]] = float(value)
        loss = binary_loss(y_outer, inner_pred)
        candidate_rows.append({"name": config["name"], "loss": loss, "config": config})
    candidate_rows = sorted(candidate_rows, key=lambda row: row["loss"])
    return candidate_rows[0]["config"], candidate_rows[:8]


def outer_probe_predictions(train, test, base218_oof, historical_configs, families):
    pred = np.zeros((len(train), len(TARGETS)), dtype=float)
    fold_details = []
    selected_counts = Counter()
    selected_by_target = defaultdict(Counter)
    for fold in split_rows():
        outer_train_idx = np.asarray(fold["train_idx"], dtype=int)
        outer_valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        target_rows = {}
        for target_idx, target in enumerate(TARGETS):
            configs = candidate_grid(historical_configs[target])
            selected, top_rows = select_config_for_outer_train(
                train=train,
                test=test,
                families=families,
                base218_oof=base218_oof,
                target=target,
                outer_train_idx=outer_train_idx,
                configs=configs,
            )
            if selected.get("kind") == "base":
                values = base218_oof.iloc[outer_valid_idx][target].astype(float).to_numpy()
            else:
                values = predict_config(
                    train=train,
                    test=test,
                    families=families,
                    base218=base218_oof,
                    target=target,
                    known_idx=outer_train_idx,
                    query_idx=outer_valid_idx,
                    config=selected,
                    query_is_test=False,
                )
            pred[outer_valid_idx, target_idx] = values
            selected_name = str(selected["name"])
            selected_counts[selected_name] += 1
            selected_by_target[target][selected_name] += 1
            target_rows[target] = {
                "selected": selected,
                "outer_valid_loss": binary_loss(train.iloc[outer_valid_idx][target].to_numpy(), values),
                "inner_top": top_rows,
            }
        fold_details.append(
            {
                "split_name": fold["split_name"],
                "fold_id": int(fold["fold_id"]),
                "valid_n": int(len(outer_valid_idx)),
                "loss": float(np.mean([row["outer_valid_loss"] for row in target_rows.values()])),
                "target_rows": target_rows,
            }
        )
    return pred, fold_details, selected_counts, selected_by_target


def full_train_test_preview(train, test, base218_oof, base218_test, historical_configs, families):
    pred = np.zeros((len(test), len(TARGETS)), dtype=float)
    selected_configs = {}
    fit_idx = np.arange(len(train))
    for target_idx, target in enumerate(TARGETS):
        selected, top_rows = select_config_for_outer_train(
            train=train,
            test=test,
            families=families,
            base218_oof=base218_oof,
            target=target,
            outer_train_idx=fit_idx,
            configs=candidate_grid(historical_configs[target]),
        )
        selected_configs[target] = {"selected": selected, "inner_top": top_rows}
        if selected.get("kind") == "base":
            pred[:, target_idx] = base218_test[target].astype(float).to_numpy()
        else:
            pred[:, target_idx] = predict_config(
                train=train,
                test=test,
                families=families,
                base218=base218_test,
                target=target,
                known_idx=fit_idx,
                query_idx=None,
                config=selected,
                query_is_test=True,
            )
    return pred, selected_configs


def split_summary(
    fold_details, y_true: np.ndarray, candidate: np.ndarray, references: dict[str, np.ndarray]
) -> dict[str, Any]:
    by_split: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fold, detail in zip(split_rows(), fold_details, strict=True):
        valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        by_split[detail["split_name"]]["loss"].append(float(detail["loss"]))
        for name, ref in references.items():
            delta = float(detail["loss"] - average_loss(y_true[valid_idx], ref[valid_idx]))
            by_split[detail["split_name"]][f"delta_vs_{name}"].append(delta)
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
    return {
        "split_summary": split_out,
        "primary_pass": {
            name: bool(all(split_out[split][f"delta_vs_{name}"] < 0 for split in primary)) for name in references
        },
    }


def build_report() -> dict[str, Any]:
    train, test = load_train_test()
    labels = label_frame(train)
    y_true = train[TARGETS].astype(int).to_numpy()
    historical_configs = load_historical_configs()
    families = feature_families(train, test)
    base218_oof = load_base("oof", BASE_ID, labels)
    base218_test = load_base("test", BASE_ID)
    references = {
        "id178": load_base("oof", REFERENCE_ID, labels)[TARGETS].astype(float).to_numpy(),
        "id218": base218_oof[TARGETS].astype(float).to_numpy(),
        "id221": load_base("oof", HISTORICAL_ID221, labels)[TARGETS].astype(float).to_numpy(),
        "id341": load_base("oof", CONTROL_ID, labels)[TARGETS].astype(float).to_numpy(),
    }
    candidate_oof, fold_details, selected_counts, selected_by_target = outer_probe_predictions(
        train=train,
        test=test,
        base218_oof=base218_oof,
        historical_configs=historical_configs,
        families=families,
    )
    candidate_test, full_train_selected = full_train_test_preview(
        train=train,
        test=test,
        base218_oof=base218_oof,
        base218_test=base218_test,
        historical_configs=historical_configs,
        families=families,
    )
    base_test_arrays = {
        candidate_id: load_base("test", candidate_id)[TARGETS].astype(float).to_numpy()
        for candidate_id in [REFERENCE_ID, BASE_ID, HISTORICAL_ID221, CONTROL_ID, FAILURE_ID, GAP_ID]
    }
    vec = (candidate_test - base_test_arrays[REFERENCE_ID]).reshape(-1)
    failed_direction = (base_test_arrays[FAILURE_ID] - base_test_arrays[REFERENCE_ID]).reshape(-1)
    gap_direction = (base_test_arrays[GAP_ID] - base_test_arrays[REFERENCE_ID]).reshape(-1)
    full_loss = average_loss(y_true, candidate_oof)
    summary = split_summary(fold_details, y_true, candidate_oof, references)
    active_selection_count = int(sum(count for name, count in selected_counts.items() if name != "base"))
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_clean_id221_inner_selection_probe",
        "purpose": "Select ID221-like feature-similarity configs inside frozen outer-training folds only, then score outer-valid rows.",
        "candidate_grid": {
            "per_target_family": "historical ID221 target family only",
            "k_values": [1, 3, 7],
            "date_weights": [0.0, 0.03, 0.1],
            "blend_weights": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07],
            "base_guard": True,
        },
        "full_oof": {
            "candidate_loss": full_loss,
            "delta_vs_id178": float(full_loss - average_loss(y_true, references["id178"])),
            "delta_vs_id218": float(full_loss - average_loss(y_true, references["id218"])),
            "delta_vs_historical_id221": float(full_loss - average_loss(y_true, references["id221"])),
            "delta_vs_id341": float(full_loss - average_loss(y_true, references["id341"])),
        },
        "split_gate": summary,
        "selection_summary": {
            "active_selection_count": active_selection_count,
            "total_selection_slots": int(sum(selected_counts.values())),
            "selected_counts_top": selected_counts.most_common(20),
            "selected_by_target_top": {
                target: counter.most_common(10) for target, counter in selected_by_target.items()
            },
            "full_train_selected": full_train_selected,
        },
        "test_geometry_preview_no_artifact": {
            "mean_delta_vs_id178": float(np.mean(np.abs(candidate_test - base_test_arrays[REFERENCE_ID]))),
            "max_delta_vs_id178": float(np.max(np.abs(candidate_test - base_test_arrays[REFERENCE_ID]))),
            "mean_delta_vs_id341": float(np.mean(np.abs(candidate_test - base_test_arrays[CONTROL_ID]))),
            "max_delta_vs_id341": float(np.max(np.abs(candidate_test - base_test_arrays[CONTROL_ID]))),
            "projection_failed_id287": projection(vec, failed_direction),
            "projection_gap_id329": projection(vec, gap_direction),
        },
        "decision": {
            "inner_selection_improves_id218": bool(summary["primary_pass"]["id218"]),
            "beats_id178_primary": bool(summary["primary_pass"]["id178"]),
            "beats_id341_primary": bool(summary["primary_pass"]["id341"]),
            "beats_historical_id221_primary": bool(summary["primary_pass"]["id221"]),
            "promote_artifact": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "If this probe improves ID218 and keeps low failed-direction exposure, implement the same inner-selection "
                "contract in the Kaggle clean-backbone runner. Otherwise treat clean ID221 as an anchor/control only."
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
        "# Post-ID341 clean ID221 inner-selection probe",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## OOF summary",
        "",
        f"- Candidate loss: `{report['full_oof']['candidate_loss']:.6f}`",
        f"- Δ vs ID178: `{report['full_oof']['delta_vs_id178']:+.6f}`",
        f"- Δ vs ID218: `{report['full_oof']['delta_vs_id218']:+.6f}`",
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
            "## Selection summary",
            "",
            f"- Active selections: `{report['selection_summary']['active_selection_count']}` / `{report['selection_summary']['total_selection_slots']}`",
            "",
            "Top selected configs:",
            "",
            "```json",
            json.dumps(report["selection_summary"]["selected_counts_top"], indent=2, ensure_ascii=False),
            "```",
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
            f"- Inner selection improves ID218: `{report['decision']['inner_selection_improves_id218']}`",
            f"- Beats ID178 primary: `{report['decision']['beats_id178_primary']}`",
            f"- Beats ID341 primary: `{report['decision']['beats_id341_primary']}`",
            f"- Beats historical ID221 primary: `{report['decision']['beats_historical_id221_primary']}`",
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
        default=REPORTS_DIR / "post_id341_clean_id221_inner_selection_probe_20260618.json",
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
