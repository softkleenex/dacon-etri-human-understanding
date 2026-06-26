#!/usr/bin/env python3
"""Fold-internal pair/logit blend diagnostic after the post-ID335 blueprint.

This report-only script rebuilds the pair-blend *selection* inside outer folds.
It still uses existing source OOF artifacts, so it is not a true model-level
nested rebuild, but it tests whether the target-wise pair/logit selection step
survives without looking at the outer validation labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    TARGETS,
    align,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.diagnose_strict_demographic_feature_audit import make_loso_folds  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
EPS = 1e-6
ANCHOR_ID = 178


@dataclass(frozen=True)
class Scenario:
    name: str
    source_ids: tuple[int, ...]
    cap: float
    penalty: float


SCENARIOS = [
    Scenario("direct_no_252_cap015", (178, 254, 255, 256, 257, 258, 266), 0.15, 0.00),
    Scenario("direct_no_252_cap022", (178, 254, 255, 256, 257, 258, 266), 0.22, 0.01),
    Scenario("with_ultra252_cap022", (178, 252, 255, 257, 258), 0.22, 0.01),
    Scenario("with_ultra252_cap026", (178, 252, 255, 257, 258), 0.26, 0.01),
    Scenario("full_upstream_cap026", (178, 252, 254, 255, 256, 257, 258, 266), 0.26, 0.01),
    Scenario("full_upstream_cap030", (178, 252, 254, 255, 256, 257, 258, 266), 0.30, 0.02),
]
WEIGHTS = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))
KINDS = ("single", "prob_pair", "logit_pair")


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def logit(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def blend_values(a_values: np.ndarray, b_values: np.ndarray, weight_b: float, kind: str) -> np.ndarray:
    a = np.clip(np.asarray(a_values, dtype=float), EPS, 1.0 - EPS)
    b = np.clip(np.asarray(b_values, dtype=float), EPS, 1.0 - EPS)
    if kind == "single":
        return a
    if kind == "prob_pair":
        return np.clip((1.0 - weight_b) * a + weight_b * b, EPS, 1.0 - EPS)
    if kind == "logit_pair":
        return np.clip(sigmoid((1.0 - weight_b) * logit(a) + weight_b * logit(b)), EPS, 1.0 - EPS)
    raise ValueError(f"Unknown blend kind: {kind}")


def load_candidates(train: pd.DataFrame, candidate_ids: list[int]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact("oof", candidate_id), train) for candidate_id in candidate_ids}


def arrays_for_target(candidates: dict[int, pd.DataFrame], target: str) -> dict[int, np.ndarray]:
    return {
        candidate_id: np.clip(frame[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        for candidate_id, frame in candidates.items()
    }


def source_risk(id_a: int, id_b: int, weight_b: float, kind: str) -> float:
    base_risk = {
        178: 0.0,
        252: 8.0,
        254: 1.0,
        255: 1.0,
        256: 2.0,
        257: 2.0,
        258: 1.0,
        266: 1.0,
    }
    if kind == "single":
        return float(base_risk.get(id_a, 3.0))
    return float((1.0 - weight_b) * base_risk.get(id_a, 3.0) + weight_b * base_risk.get(id_b, 3.0))


def build_pool_for_target(
    *,
    y_train: np.ndarray,
    arrays: dict[int, np.ndarray],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    source_ids: tuple[int, ...],
    cap: float,
    penalty: float,
) -> list[dict[str, Any]]:
    anchor = arrays[ANCHOR_ID]
    pool: list[dict[str, Any]] = []
    for id_a, id_b in combinations_with_replacement(source_ids, 2):
        for kind in KINDS:
            if kind == "single" and id_a != id_b:
                continue
            if kind != "single" and id_a == id_b:
                continue
            weights = (0.0,) if kind == "single" else WEIGHTS
            for weight_b in weights:
                train_pred = blend_values(arrays[id_a][train_idx], arrays[id_b][train_idx], weight_b, kind)
                delta = train_pred - anchor[train_idx]
                max_delta = float(np.max(np.abs(delta)))
                if max_delta > cap + 1e-12:
                    continue
                mean_delta = float(np.mean(np.abs(delta)))
                risk = source_risk(id_a, id_b, float(weight_b), kind)
                loss = target_loss(y_train, train_pred)
                pool.append(
                    {
                        "kind": kind,
                        "id_a": int(id_a),
                        "id_b": int(id_b),
                        "w_b": float(weight_b),
                        "train_loss": loss,
                        "objective": float(loss + penalty * risk + 0.05 * mean_delta),
                        "source_risk": risk,
                        "train_max_abs_delta_vs_anchor": max_delta,
                        "train_mean_abs_delta_vs_anchor": mean_delta,
                        "valid_pred": blend_values(arrays[id_a][valid_idx], arrays[id_b][valid_idx], weight_b, kind),
                    }
                )
    if pool:
        return pool
    fallback = anchor
    return [
        {
            "kind": "single",
            "id_a": ANCHOR_ID,
            "id_b": ANCHOR_ID,
            "w_b": 0.0,
            "train_loss": target_loss(y_train, fallback[train_idx]),
            "objective": target_loss(y_train, fallback[train_idx]),
            "source_risk": 0.0,
            "train_max_abs_delta_vs_anchor": 0.0,
            "train_mean_abs_delta_vs_anchor": 0.0,
            "valid_pred": fallback[valid_idx],
        }
    ]


def run_scenario(
    *,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: dict[int, pd.DataFrame],
    scenario: Scenario,
) -> dict[str, Any]:
    selected = train[["subject_id", "date"]].copy()
    for target in TARGETS:
        selected[target] = np.nan
    choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    target_arrays = {target: arrays_for_target(candidates, target) for target in TARGETS}

    for fold_idx, (raw_train_idx, raw_valid_idx) in enumerate(folds):
        train_idx = np.asarray(raw_train_idx, dtype=int)
        valid_idx = np.asarray(raw_valid_idx, dtype=int)
        for target in TARGETS:
            y_train = train[target].iloc[train_idx].astype(int).to_numpy()
            pool = build_pool_for_target(
                y_train=y_train,
                arrays=target_arrays[target],
                train_idx=train_idx,
                valid_idx=valid_idx,
                source_ids=scenario.source_ids,
                cap=scenario.cap,
                penalty=scenario.penalty,
            )
            choice = min(
                pool,
                key=lambda row: (
                    row["objective"],
                    row["train_loss"],
                    row["train_max_abs_delta_vs_anchor"],
                    row["source_risk"],
                ),
            )
            selected.loc[valid_idx, target] = choice["valid_pred"]
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "kind": choice["kind"],
                    "id_a": int(choice["id_a"]),
                    "id_b": int(choice["id_b"]),
                    "w_b": float(choice["w_b"]),
                    "train_loss": float(choice["train_loss"]),
                    "valid_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(), choice["valid_pred"]
                    ),
                    "objective": float(choice["objective"]),
                    "source_risk": float(choice["source_risk"]),
                    "train_max_abs_delta_vs_anchor": float(choice["train_max_abs_delta_vs_anchor"]),
                    "train_mean_abs_delta_vs_anchor": float(choice["train_mean_abs_delta_vs_anchor"]),
                }
            )

    if selected[TARGETS].isna().any().any():
        raise ValueError("Fold-internal pair blend failed to fill all predictions.")
    overall, target_scores = local_scores(train, selected)
    slices = slice_summary(train, selected)
    return {
        "scenario": {
            "name": scenario.name,
            "source_ids": list(scenario.source_ids),
            "cap": scenario.cap,
            "penalty": scenario.penalty,
        },
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "choice_counts": {
            target: dict(Counter(f"{choice['kind']}:{choice['id_a']}->{choice['id_b']}" for choice in target_choices))
            for target, target_choices in choices.items()
        },
        "choices": choices,
    }


def best_rows(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: min(rows, key=lambda row: (row["overall"], row["robust_score"]))
        for split_name, rows in results.items()
        if rows
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fold-internal pair/logit blend rebuild — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. Pair/logit blend choices are learned inside each outer fold, but source artifacts are still existing OOF products.",
        "",
        "## Results",
        "",
        "| split | scenario | overall | robust | worst target | worst subject | source ids |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for split_name, rows in report["splits"].items():
        for row in rows:
            scenario = row["scenario"]
            slices = row["slice_summary"]
            lines.append(
                f"| `{split_name}` | `{scenario['name']}` | {row['overall']:.6f} | "
                f"{row['robust_score']:.6f} | {slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
                f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` | `{scenario['source_ids']}` |"
            )
    lines.extend(
        [
            "",
            "## Best rows",
            "",
            "| split | best scenario | overall | robust | note |",
            "|---|---|---:|---:|---|",
        ]
    )
    for split_name, row in report["best_rows"].items():
        note = "uses direct ID252" if 252 in row["scenario"]["source_ids"] else "direct ID252 excluded"
        lines.append(
            f"| `{split_name}` | `{row['scenario']['name']}` | {row['overall']:.6f} | "
            f"{row['robust_score']:.6f} | {note} |"
        )
    lines.extend(["", "## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, _test = load_train_test()
    candidate_ids = sorted({candidate_id for scenario in SCENARIOS for candidate_id in scenario.source_ids})
    candidates = load_candidates(train, candidate_ids)
    splits = {
        "subject_hole_fold_internal": make_subject_hole_folds(train),
        "loso_fold_internal": make_loso_folds(train),
    }
    results = {
        split_name: [
            run_scenario(train=train, folds=folds, candidates=candidates, scenario=scenario) for scenario in SCENARIOS
        ]
        for split_name, folds in splits.items()
    }
    best = best_rows(results)
    direct_no_252_best = min(
        [row for rows in results.values() for row in rows if 252 not in row["scenario"]["source_ids"]],
        key=lambda row: (row["overall"], row["robust_score"]),
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "fold_internal_pair_blend_rebuild_post_id335",
        "source_artifacts_are_preexisting": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "scenarios": [scenario.__dict__ | {"source_ids": list(scenario.source_ids)} for scenario in SCENARIOS],
        "splits": results,
        "best_rows": best,
        "direct_no_252_best": direct_no_252_best,
        "decision": {
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "model_level_nested_rebuild_still_required": True,
            "reason": (
                "Fold-internal pair/logit selection is a useful stress test, but it still reuses existing OOF "
                "source artifacts. Even if it scores well locally, it cannot authorize a CSV or DACON submission "
                "without rebuilding the source artifacts and calibration choices inside true model-level outer folds."
            ),
        },
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md = output_json.with_suffix(".md")
    output_md.write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "fold_internal_pair_blend_rebuild_post_id335_20260617.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    for split_name, row in report["best_rows"].items():
        print(f"{split_name}: {row['scenario']['name']} overall={row['overall']:.6f} robust={row['robust_score']:.6f}")
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
