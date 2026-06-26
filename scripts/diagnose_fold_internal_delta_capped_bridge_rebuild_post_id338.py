#!/usr/bin/env python3
"""Fold-internal delta-capped bridge rebuild diagnostic after ID338.

This report-only diagnostic rebuilds target-wise bridge selection inside outer
folds. It still uses existing base/extreme OOF artifacts, so it is not a true
model-level rebuild yet; it tests whether the A1 bridge-choice layer survives
without selecting on the outer validation rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from scripts.run_public_confirmed_anchor_bridge_probe_experiment import blend_values  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
SCAN_PATH = REPORTS_DIR / "id235_frontier_targetwise_delta_capped_prob_logit_scan_20260616.json"
KST = timezone(timedelta(hours=9))
EPS = 1e-6


@dataclass(frozen=True)
class Scenario:
    name: str
    caps: tuple[float, ...]
    extreme_ids: tuple[int, ...]
    alphas: tuple[float, ...]
    method_penalty: float
    delta_penalty: float


def scan_alphas(scan: dict[str, Any]) -> tuple[float, ...]:
    alphas = {0.0}
    for row in scan["targetwise_cap_rows"]:
        for choice in row["choices"].values():
            alphas.add(round(float(choice["alpha"]), 3))
    # Add a small local grid so fold-internal selection is not tied only to
    # previously selected full-OOF alphas.
    alphas.update(round(value, 3) for value in np.linspace(0.0, 0.4, 17))
    return tuple(sorted(alphas))


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def load_candidates(train: pd.DataFrame, candidate_ids: list[int]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact("oof", candidate_id), train) for candidate_id in candidate_ids}


def arrays_for_target(candidates: dict[int, pd.DataFrame], target: str) -> dict[int, np.ndarray]:
    return {
        candidate_id: np.clip(frame[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        for candidate_id, frame in candidates.items()
    }


def build_pool_for_target(
    *,
    y_train: np.ndarray,
    arrays: dict[int, np.ndarray],
    base_id: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    scenario: Scenario,
    cap: float,
) -> list[dict[str, Any]]:
    base = arrays[base_id]
    pool = [
        {
            "kind": "base",
            "extreme_id": base_id,
            "method": "base",
            "alpha": 0.0,
            "train_loss": target_loss(y_train, base[train_idx]),
            "objective": target_loss(y_train, base[train_idx]),
            "train_max_abs_delta_vs_base": 0.0,
            "train_mean_abs_delta_vs_base": 0.0,
            "valid_pred": base[valid_idx],
        }
    ]
    for extreme_id in scenario.extreme_ids:
        extreme = arrays[extreme_id]
        for method in ("prob", "logit"):
            for alpha in scenario.alphas:
                if alpha <= 0:
                    continue
                train_pred = blend_values(base[train_idx], extreme[train_idx], alpha=alpha, method=method)
                delta_train = train_pred - base[train_idx]
                max_delta = float(np.max(np.abs(delta_train)))
                if max_delta > cap + 1e-12:
                    continue
                mean_delta = float(np.mean(np.abs(delta_train)))
                loss = target_loss(y_train, train_pred)
                objective = loss + scenario.delta_penalty * mean_delta + scenario.method_penalty * (method == "logit")
                pool.append(
                    {
                        "kind": "bridge",
                        "extreme_id": int(extreme_id),
                        "method": method,
                        "alpha": float(alpha),
                        "train_loss": loss,
                        "objective": float(objective),
                        "train_max_abs_delta_vs_base": max_delta,
                        "train_mean_abs_delta_vs_base": mean_delta,
                        "valid_pred": blend_values(base[valid_idx], extreme[valid_idx], alpha=alpha, method=method),
                    }
                )
    return pool


def run_cap(
    *,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: dict[int, pd.DataFrame],
    base_id: int,
    scenario: Scenario,
    cap: float,
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
                base_id=base_id,
                train_idx=train_idx,
                valid_idx=valid_idx,
                scenario=scenario,
                cap=cap,
            )
            choice = min(
                pool,
                key=lambda row: (
                    row["objective"],
                    row["train_loss"],
                    row["train_max_abs_delta_vs_base"],
                    row["extreme_id"],
                ),
            )
            selected.loc[valid_idx, target] = choice["valid_pred"]
            choices[target].append(
                {
                    "fold": int(fold_idx),
                    "kind": choice["kind"],
                    "extreme_id": int(choice["extreme_id"]),
                    "method": str(choice["method"]),
                    "alpha": float(choice["alpha"]),
                    "train_loss": float(choice["train_loss"]),
                    "valid_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(), choice["valid_pred"]
                    ),
                    "objective": float(choice["objective"]),
                    "train_max_abs_delta_vs_base": float(choice["train_max_abs_delta_vs_base"]),
                    "train_mean_abs_delta_vs_base": float(choice["train_mean_abs_delta_vs_base"]),
                    "n_pool": int(len(pool)),
                }
            )
    if selected[TARGETS].isna().any().any():
        raise ValueError("Fold-internal bridge selection failed to fill all predictions.")
    overall, target_scores = local_scores(train, selected)
    slices = slice_summary(train, selected)
    changed = sum(
        1
        for target_choices in choices.values()
        for choice in target_choices
        if choice["kind"] != "base" and choice["alpha"] > 0
    )
    total_choices = sum(len(target_choices) for target_choices in choices.values())
    return {
        "scenario": {
            "name": scenario.name,
            "cap": cap,
            "base_id": base_id,
            "extreme_ids": list(scenario.extreme_ids),
            "method_penalty": scenario.method_penalty,
            "delta_penalty": scenario.delta_penalty,
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
        "changed_choice_rate": float(changed / total_choices) if total_choices else 0.0,
        "choice_counts": {
            target: dict(
                Counter(
                    f"{choice['kind']}:{choice['extreme_id']}:{choice['method']}:a{choice['alpha']:.3f}"
                    for choice in target_choices
                )
            )
            for target, target_choices in choices.items()
        },
        "choices": choices,
    }


def build_scenarios(scan: dict[str, Any]) -> list[Scenario]:
    alphas = scan_alphas(scan)
    base_extremes = tuple(int(item) for item in scan["extreme_ids"])
    return [
        Scenario("safe_caps_all_extremes", (0.12, 0.15, 0.18, 0.20), base_extremes, alphas, 0.0, 0.02),
        Scenario("midrisk_caps_all_extremes", (0.20, 0.25, 0.30), base_extremes, alphas, 0.0, 0.01),
        Scenario("no_q240_extremes", (0.15, 0.20, 0.25, 0.30), (238, 239, 240, 241), alphas, 0.0, 0.02),
        Scenario("q240_only_stress", (0.12, 0.15, 0.20), (242, 243), alphas, 0.0, 0.02),
    ]


def best_rows(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: min(rows, key=lambda row: (row["overall"], row["robust_score"]))
        for split_name, rows in results.items()
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fold-internal delta-capped bridge rebuild — post ID338",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. Bridge choices are learned inside each outer fold, but base/extreme source artifacts are existing OOF products.",
        "",
        "## Results",
        "",
        "| split | scenario | cap | overall | robust | changed | worst target | worst subject |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for split_name, rows in report["splits"].items():
        for row in rows:
            scenario = row["scenario"]
            slices = row["slice_summary"]
            lines.append(
                f"| `{split_name}` | `{scenario['name']}` | {scenario['cap']:.2f} | "
                f"{row['overall']:.6f} | {row['robust_score']:.6f} | {row['changed_choice_rate']:.3f} | "
                f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
                f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` |"
            )
    lines.extend(
        [
            "",
            "## Best rows",
            "",
            "| split | scenario | cap | overall | robust | read |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for split_name, row in report["best_rows"].items():
        scenario = row["scenario"]
        read = "q240-free" if "no_q240" in scenario["name"] else "uses all/q240 candidates"
        lines.append(
            f"| `{split_name}` | `{scenario['name']}` | {scenario['cap']:.2f} | "
            f"{row['overall']:.6f} | {row['robust_score']:.6f} | {read} |"
        )
    lines.extend(["", "## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    scan = json.loads(SCAN_PATH.read_text(encoding="utf-8"))
    train, _test = load_train_test()
    base_id = int(scan["base_id"])
    scenarios = build_scenarios(scan)
    candidate_ids = sorted(
        {base_id, *[candidate_id for scenario in scenarios for candidate_id in scenario.extreme_ids]}
    )
    candidates = load_candidates(train, candidate_ids)
    splits = {
        "subject_hole_fold_internal": make_subject_hole_folds(train),
        "loso_fold_internal": make_loso_folds(train),
    }
    results: dict[str, list[dict[str, Any]]] = {}
    for split_name, folds in splits.items():
        split_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            for cap in scenario.caps:
                split_rows.append(
                    run_cap(
                        train=train,
                        folds=folds,
                        candidates=candidates,
                        base_id=base_id,
                        scenario=scenario,
                        cap=cap,
                    )
                )
        results[split_name] = split_rows
    best = best_rows(results)
    q240_free_rows = [
        row for split_rows in results.values() for row in split_rows if "no_q240" in row["scenario"]["name"]
    ]
    q240_free_best = min(q240_free_rows, key=lambda row: (row["overall"], row["robust_score"]))
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "fold_internal_delta_capped_bridge_rebuild_post_id338",
        "scan_path": str(SCAN_PATH.relative_to(ROOT)),
        "base_id": base_id,
        "source_artifacts_are_preexisting": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "scenarios": [
            scenario.__dict__ | {"extreme_ids": list(scenario.extreme_ids), "caps": list(scenario.caps)}
            for scenario in scenarios
        ],
        "splits": results,
        "best_rows": best,
        "q240_free_best": q240_free_best,
        "decision": {
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "model_level_nested_rebuild_still_required": True,
            "reason": (
                "Fold-internal target-wise bridge selection preserves a bridge signal, but it still relies on existing "
                "ID235/extreme OOF artifacts. Treat this as support for rebuilding the bridge/source layer inside true "
                "model-level outer folds, not as submission evidence."
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
