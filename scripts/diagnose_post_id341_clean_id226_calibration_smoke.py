#!/usr/bin/env python3
"""Clean ID226 fold-internal calibration/convex smoke after ID341.

This report-only diagnostic tests whether an ID226-like calibration layer can be
rebuilt without reusing the existing ID172 extreme full-OOF artifact. It starts
from the clean ID218 source, selects conservative calibration/shrink configs
inside each outer-training fold, and scores only the held-out outer-valid rows.

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
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_clean_id221_feature_similarity_smoke import (  # noqa: E402
    BASE_ID,
    CONTROL_ID,
    EPS,
    FAILURE_ID,
    GAP_ID,
    HISTORICAL_ID221,
    REFERENCE_ID,
    TARGETS,
    average_loss,
    binary_loss,
    label_frame,
    load_base,
    load_train_test,
    projection,
    split_rows,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
HISTORICAL_ID226 = 226


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, EPS, 1 - EPS)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def logit_shrink(base: np.ndarray, transformed: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(sigmoid((1 - alpha) * logit(base) + alpha * logit(transformed)), EPS, 1 - EPS)


def prob_shrink(base: np.ndarray, transformed: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1 - alpha) * base + alpha * transformed, EPS, 1 - EPS)


def fit_quantile(
    train_values: np.ndarray, y_train: np.ndarray, values: np.ndarray, q: int, prior: float = 8.0
) -> np.ndarray:
    train_values = np.clip(train_values.astype(float), EPS, 1 - EPS)
    values = np.clip(values.astype(float), EPS, 1 - EPS)
    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, q + 1)))
    if len(edges) <= 2:
        return np.repeat(float(np.mean(y_train)), len(values))
    bins = np.digitize(train_values, edges[1:-1], right=True)
    global_mean = float(np.mean(y_train))
    bin_values = []
    for bin_id in range(len(edges) - 1):
        mask = bins == bin_id
        count = int(mask.sum())
        if count == 0:
            bin_values.append(global_mean)
        else:
            bin_values.append(float((y_train[mask].sum() + prior * global_mean) / (count + prior)))
    value_bins = np.digitize(values, edges[1:-1], right=True)
    return np.clip(np.asarray([bin_values[bin_id] for bin_id in value_bins], dtype=float), EPS, 1 - EPS)


def fit_isotonic(train_values: np.ndarray, y_train: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(np.unique(train_values)) < 4 or len(np.unique(y_train)) < 2:
        return np.repeat(float(np.mean(y_train)), len(values))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(np.clip(train_values, EPS, 1 - EPS), y_train.astype(float))
    return np.clip(iso.predict(np.clip(values, EPS, 1 - EPS)), EPS, 1 - EPS)


def apply_config(
    train_values: np.ndarray,
    y_train: np.ndarray,
    values: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    base = np.clip(values.astype(float), EPS, 1 - EPS)
    kind = config["kind"]
    if kind == "base":
        return base
    if kind == "temp_shift":
        transformed = np.clip(
            sigmoid(logit(base) / float(config["temperature"]) + float(config["shift"])), EPS, 1 - EPS
        )
        return transformed
    if kind == "quantile_shrink":
        transformed = fit_quantile(train_values, y_train, values, q=int(config["q"]))
    elif kind == "isotonic_shrink":
        transformed = fit_isotonic(train_values, y_train, values)
    else:
        raise ValueError(f"Unknown config kind: {kind}")
    alpha = float(config["alpha"])
    if config.get("space") == "logit":
        return logit_shrink(base, transformed, alpha)
    return prob_shrink(base, transformed, alpha)


def config_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{"name": "base", "kind": "base"}]
    for temperature in [0.85, 1.0, 1.15]:
        for shift in [-0.12, 0.0, 0.12]:
            if temperature == 1.0 and shift == 0.0:
                continue
            configs.append(
                {
                    "name": f"temp_t{temperature:g}_s{shift:+.2f}",
                    "kind": "temp_shift",
                    "temperature": temperature,
                    "shift": shift,
                }
            )
    for q in [8, 16]:
        for alpha in [0.05, 0.1, 0.2]:
            for space in ["prob", "logit"]:
                configs.append(
                    {
                        "name": f"q{q}_{space}_a{alpha:g}",
                        "kind": "quantile_shrink",
                        "q": q,
                        "alpha": alpha,
                        "space": space,
                    }
                )
    for alpha in [0.05, 0.1]:
        for space in ["prob", "logit"]:
            configs.append(
                {
                    "name": f"iso_{space}_a{alpha:g}",
                    "kind": "isotonic_shrink",
                    "alpha": alpha,
                    "space": space,
                }
            )
    return configs


def make_inner_folds(train, outer_train_idx: np.ndarray, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    chunks_by_fold: list[list[int]] = [[] for _ in range(n_splits)]
    outer = train.iloc[outer_train_idx].copy()
    outer["_row_idx"] = outer_train_idx
    for _, subject_frame in outer.sort_values(["subject_id", "date_dt"]).groupby("subject_id", sort=False):
        chunks = np.array_split(subject_frame["_row_idx"].to_numpy(), n_splits)
        for fold_idx, chunk in enumerate(chunks):
            chunks_by_fold[fold_idx].extend(int(value) for value in chunk)
    outer_set = {int(value) for value in outer_train_idx}
    folds = []
    for valid_values in chunks_by_fold:
        valid_idx = np.asarray(sorted(valid_values), dtype=int)
        valid_set = {int(value) for value in valid_idx}
        train_idx = np.asarray(sorted(outer_set - valid_set), dtype=int)
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def select_config(
    train,
    source_oof,
    target: str,
    outer_train_idx: np.ndarray,
    configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inner_folds = make_inner_folds(train, outer_train_idx)
    y_outer = train.iloc[outer_train_idx][target].astype(int).to_numpy()
    outer_values = source_oof.iloc[outer_train_idx][target].astype(float).to_numpy()
    index_to_pos = {int(row_idx): pos for pos, row_idx in enumerate(outer_train_idx)}
    rows = []
    for config in configs:
        pred = np.zeros(len(outer_train_idx), dtype=float)
        for inner_train_idx, inner_valid_idx in inner_folds:
            train_values = source_oof.iloc[inner_train_idx][target].astype(float).to_numpy()
            y_train = train.iloc[inner_train_idx][target].astype(int).to_numpy()
            valid_values = source_oof.iloc[inner_valid_idx][target].astype(float).to_numpy()
            valid_pred = apply_config(train_values, y_train, valid_values, config)
            for row_idx, value in zip(inner_valid_idx, valid_pred, strict=True):
                pred[index_to_pos[int(row_idx)]] = float(value)
        loss = binary_loss(y_outer, pred if config["kind"] != "base" else outer_values)
        rows.append({"name": config["name"], "loss": loss, "config": config})
    rows = sorted(rows, key=lambda row: row["loss"])
    return rows[0]["config"], rows[:8]


def outer_predictions(train, source_oof, configs: list[dict[str, Any]]):
    pred = np.zeros((len(train), len(TARGETS)), dtype=float)
    fold_details = []
    selected_counts = Counter()
    selected_by_target: dict[str, Counter] = defaultdict(Counter)
    for fold in split_rows():
        outer_train_idx = np.asarray(fold["train_idx"], dtype=int)
        outer_valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        target_rows = {}
        for target_idx, target in enumerate(TARGETS):
            selected, top = select_config(train, source_oof, target, outer_train_idx, configs)
            train_values = source_oof.iloc[outer_train_idx][target].astype(float).to_numpy()
            y_train = train.iloc[outer_train_idx][target].astype(int).to_numpy()
            valid_values = source_oof.iloc[outer_valid_idx][target].astype(float).to_numpy()
            values = apply_config(train_values, y_train, valid_values, selected)
            pred[outer_valid_idx, target_idx] = values
            selected_counts[str(selected["name"])] += 1
            selected_by_target[target][str(selected["name"])] += 1
            target_rows[target] = {
                "selected": selected,
                "inner_top": top,
                "outer_valid_loss": binary_loss(train.iloc[outer_valid_idx][target].to_numpy(), values),
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


def full_train_test_preview(train, source_oof, source_test, configs: list[dict[str, Any]]):
    pred = np.zeros((len(source_test), len(TARGETS)), dtype=float)
    selected_by_target = {}
    fit_idx = np.arange(len(train))
    for target_idx, target in enumerate(TARGETS):
        selected, top = select_config(train, source_oof, target, fit_idx, configs)
        train_values = source_oof.iloc[fit_idx][target].astype(float).to_numpy()
        y_train = train.iloc[fit_idx][target].astype(int).to_numpy()
        test_values = source_test[target].astype(float).to_numpy()
        pred[:, target_idx] = apply_config(train_values, y_train, test_values, selected)
        selected_by_target[target] = {"selected": selected, "inner_top": top}
    return pred, selected_by_target


def split_summary(fold_details, y_true: np.ndarray, references: dict[str, np.ndarray]) -> dict[str, Any]:
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
    train, _test = load_train_test()
    labels = label_frame(train)
    y_true = train[TARGETS].astype(int).to_numpy()
    source_oof = load_base("oof", BASE_ID, labels)
    source_test = load_base("test", BASE_ID)
    references = {
        "id178": load_base("oof", REFERENCE_ID, labels)[TARGETS].astype(float).to_numpy(),
        "id218": source_oof[TARGETS].astype(float).to_numpy(),
        "id221": load_base("oof", HISTORICAL_ID221, labels)[TARGETS].astype(float).to_numpy(),
        "id226": load_base("oof", HISTORICAL_ID226, labels)[TARGETS].astype(float).to_numpy(),
        "id341": load_base("oof", CONTROL_ID, labels)[TARGETS].astype(float).to_numpy(),
    }
    configs = config_grid()
    candidate_oof, fold_details, selected_counts, selected_by_target = outer_predictions(train, source_oof, configs)
    candidate_test, full_train_selected = full_train_test_preview(train, source_oof, source_test, configs)
    base_test_arrays = {
        candidate_id: load_base("test", candidate_id)[TARGETS].astype(float).to_numpy()
        for candidate_id in [REFERENCE_ID, BASE_ID, HISTORICAL_ID221, HISTORICAL_ID226, CONTROL_ID, FAILURE_ID, GAP_ID]
    }
    vec = (candidate_test - base_test_arrays[REFERENCE_ID]).reshape(-1)
    failed_direction = (base_test_arrays[FAILURE_ID] - base_test_arrays[REFERENCE_ID]).reshape(-1)
    gap_direction = (base_test_arrays[GAP_ID] - base_test_arrays[REFERENCE_ID]).reshape(-1)
    full_loss = average_loss(y_true, candidate_oof)
    summary = split_summary(fold_details, y_true, references)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_clean_id226_calibration_smoke",
        "purpose": "Test conservative fold-internal ID226-like calibration on ID218 without reusing ID172 extreme full-OOF artifacts.",
        "source_id": BASE_ID,
        "blocked_extreme_source": 172,
        "candidate_grid": {
            "total_configs": len(configs),
            "families": sorted({config["kind"] for config in configs}),
            "note": "Config selection is inner-fold only; outer-valid rows are score-only.",
        },
        "full_oof": {
            "candidate_loss": full_loss,
            "delta_vs_id178": float(full_loss - average_loss(y_true, references["id178"])),
            "delta_vs_id218": float(full_loss - average_loss(y_true, references["id218"])),
            "delta_vs_historical_id221": float(full_loss - average_loss(y_true, references["id221"])),
            "delta_vs_historical_id226": float(full_loss - average_loss(y_true, references["id226"])),
            "delta_vs_id341": float(full_loss - average_loss(y_true, references["id341"])),
        },
        "split_gate": summary,
        "selection_summary": {
            "active_selection_count": int(sum(count for name, count in selected_counts.items() if name != "base")),
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
            "mean_delta_vs_historical_id226": float(
                np.mean(np.abs(candidate_test - base_test_arrays[HISTORICAL_ID226]))
            ),
            "max_delta_vs_historical_id226": float(np.max(np.abs(candidate_test - base_test_arrays[HISTORICAL_ID226]))),
            "projection_failed_id287": projection(vec, failed_direction),
            "projection_gap_id329": projection(vec, gap_direction),
        },
        "decision": {
            "clean_id226_calibration_signal": bool(summary["primary_pass"]["id218"]),
            "beats_id178_primary": bool(summary["primary_pass"]["id178"]),
            "beats_id341_primary": bool(summary["primary_pass"]["id341"]),
            "beats_historical_id226_primary": bool(summary["primary_pass"]["id226"]),
            "promote_artifact": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "Use this as a clean ID226 layer read only. If calibration does not beat ID218, do not replay ID227/228/237; "
                "move to a stronger raw/source model rather than reviving ID172-style full-OOF calibration."
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
        "# Post-ID341 clean ID226 calibration smoke",
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
        f"- Δ vs historical ID226: `{report['full_oof']['delta_vs_historical_id226']:+.6f}`",
        f"- Δ vs ID341: `{report['full_oof']['delta_vs_id341']:+.6f}`",
        "",
        "## Split gate",
        "",
        "| split | loss | Δ178 | Δ218 | Δ221 | Δ226 | Δ341 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in report["split_gate"]["split_summary"].items():
        lines.append(
            f"| `{split}` | {row['mean_loss']:.6f} | {row['delta_vs_id178']:+.6f} | "
            f"{row['delta_vs_id218']:+.6f} | {row['delta_vs_id221']:+.6f} | "
            f"{row['delta_vs_id226']:+.6f} | {row['delta_vs_id341']:+.6f} |"
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
            f"- Mean/max Δ vs historical ID226: `{geom['mean_delta_vs_historical_id226']:.6f}` / `{geom['max_delta_vs_historical_id226']:.6f}`",
            f"- Projection onto ID287 failed direction: `{geom['projection_failed_id287']:+.4f}`",
            f"- Projection onto ID329 gap direction: `{geom['projection_gap_id329']:+.4f}`",
            "",
            "## Decision",
            "",
            f"- Clean ID226 calibration signal: `{report['decision']['clean_id226_calibration_signal']}`",
            f"- Beats ID178 primary: `{report['decision']['beats_id178_primary']}`",
            f"- Beats ID341 primary: `{report['decision']['beats_id341_primary']}`",
            f"- Beats historical ID226 primary: `{report['decision']['beats_historical_id226_primary']}`",
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
        default=REPORTS_DIR / "post_id341_clean_id226_calibration_smoke_20260618.json",
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
