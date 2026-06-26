#!/usr/bin/env python3
"""Constrained target-combination scan under an aggregate failed-direction budget.

The aggregate-budget scan uses a Lagrangian penalty, which can miss the best
combination for a fixed total projection budget.  This diagnostic builds
target-wise residual candidate frontiers and combines them with a small Pareto
dynamic program.  The output row schema is intentionally compatible with
``run_public_failed_direction_aggregate_budget_probe_experiment.py`` promotion:
``lambda_penalty`` stores the projection budget for each promoted row.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_failed_direction_aggregate_budget_probe_experiment import (  # noqa: E402
    DEFAULT_CAPS,
    compact_choice,
    total_failed_direction_denominator,
)
from scripts.run_public_failed_direction_orthogonal_probe_experiment import (  # noqa: E402
    DEFAULT_GAMMAS,
    DEFAULT_SCALES,
    EPS,
    adjusted_values,
    artifact_exists,
    parse_float_list,
    parse_source_ids,
    projection_beta,
    relpath,
    summarize_prediction_delta,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

DEFAULT_BUDGETS = [-0.02, 0.0, 0.02, 0.05, 0.08, 0.12, 0.16]


@dataclass
class State:
    loss: float
    projection: float
    choices: dict[str, dict[str, Any]]


def projection_bin(value: float, width: float) -> int:
    return int(round(value / width))


def prune_states(states: list[State], bin_width: float, max_states: int) -> list[State]:
    best_by_bin: dict[int, State] = {}
    for state in states:
        key = projection_bin(state.projection, bin_width)
        previous = best_by_bin.get(key)
        if previous is None or state.loss < previous.loss:
            best_by_bin[key] = state
    compact = sorted(best_by_bin.values(), key=lambda item: (item.projection, item.loss))

    pareto: list[State] = []
    best_loss = float("inf")
    for state in compact:
        if state.loss < best_loss - 1e-12:
            pareto.append(state)
            best_loss = state.loss

    if len(pareto) <= max_states:
        return pareto

    # Preserve the best local states and a projection-spread sample.
    best_loss_states = sorted(pareto, key=lambda item: item.loss)[: max_states // 2]
    spread_indices = np.linspace(0, len(pareto) - 1, num=max_states - len(best_loss_states), dtype=int)
    selected = {id(state): state for state in best_loss_states}
    for index in spread_indices:
        selected[id(pareto[int(index)])] = pareto[int(index)]
    return sorted(selected.values(), key=lambda item: (item.projection, item.loss))


def build_target_pools(
    anchor_id: int,
    failed_id: int,
    source_ids: list[int],
    scales: list[float],
    gammas: list[float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    anchor_oof, anchor_test, anchor_source = load_artifact_pair(train, test, anchor_id)
    failed_oof, failed_test, failed_source = load_artifact_pair(train, test, failed_id)
    total_denominator = total_failed_direction_denominator(anchor_test, failed_test)
    usable_source_ids = [
        source_id for source_id in source_ids if source_id not in {anchor_id, failed_id} and artifact_exists(source_id)
    ]
    artifacts = {source_id: load_artifact_pair(train, test, source_id) for source_id in usable_source_ids}

    target_pools: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        anchor_train = np.clip(anchor_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        anchor_holdout = np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        failed_train = np.clip(failed_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        failed_holdout = np.clip(failed_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        failed_vector = failed_holdout - anchor_holdout
        pool: list[dict[str, Any]] = [
            {
                "kind": "anchor",
                "source_id": anchor_id,
                "scale": 0.0,
                "gamma": 0.0,
                "beta_to_failed": 0.0,
                "loss": float(log_loss(y, anchor_train, labels=[0, 1])),
                "max_abs_delta_vs_anchor": 0.0,
                "mean_abs_delta_vs_anchor": 0.0,
                "projection_to_failed_direction": 0.0,
                "projection_global_contribution": 0.0,
                "source": anchor_source,
            }
        ]
        for source_id, (source_oof, source_test, source_name) in artifacts.items():
            source_train = np.clip(source_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            source_holdout = np.clip(source_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            beta = projection_beta(anchor_holdout, source_holdout, failed_holdout)
            for scale in scales:
                for gamma in gammas:
                    prediction_train = adjusted_values(
                        anchor_train,
                        source_train,
                        failed_train,
                        scale=scale,
                        gamma=gamma,
                        beta=beta,
                    )
                    prediction_holdout = adjusted_values(
                        anchor_holdout,
                        source_holdout,
                        failed_holdout,
                        scale=scale,
                        gamma=gamma,
                        beta=beta,
                    )
                    delta = prediction_holdout - anchor_holdout
                    pool.append(
                        {
                            "kind": "failed_direction_constrained_combo",
                            "source_id": source_id,
                            "scale": scale,
                            "gamma": gamma,
                            "beta_to_failed": beta,
                            "loss": float(log_loss(y, prediction_train, labels=[0, 1])),
                            **summarize_prediction_delta(anchor_holdout, prediction_holdout, failed_holdout),
                            "projection_global_contribution": float(delta @ failed_vector / total_denominator),
                            "source": source_name,
                        }
                    )
        target_pools[target] = pool

    context = {
        "anchor_source": anchor_source,
        "failed_source": failed_source,
        "source_ids_used": usable_source_ids,
    }
    return target_pools, context


def target_frontier(candidates: list[dict[str, Any]], cap: float, per_target_limit: int) -> list[dict[str, Any]]:
    filtered = [candidate for candidate in candidates if float(candidate["max_abs_delta_vs_anchor"]) <= cap + 1e-12]
    filtered.sort(key=lambda item: (float(item["projection_global_contribution"]), float(item["loss"])))
    frontier: list[dict[str, Any]] = []
    best_loss = float("inf")
    for candidate in filtered:
        loss = float(candidate["loss"])
        if loss < best_loss - 1e-12:
            frontier.append(candidate)
            best_loss = loss

    by_loss = sorted(filtered, key=lambda item: float(item["loss"]))[: per_target_limit // 3]
    merged = {id(candidate): candidate for candidate in frontier}
    for candidate in by_loss:
        merged[id(candidate)] = candidate
    result = sorted(
        merged.values(), key=lambda item: (float(item["projection_global_contribution"]), float(item["loss"]))
    )
    if len(result) <= per_target_limit:
        return result

    keep_loss = sorted(result, key=lambda item: float(item["loss"]))[: per_target_limit // 2]
    spread_indices = np.linspace(0, len(result) - 1, num=per_target_limit - len(keep_loss), dtype=int)
    selected = {id(candidate): candidate for candidate in keep_loss}
    for index in spread_indices:
        selected[id(result[int(index)])] = result[int(index)]
    return sorted(
        selected.values(), key=lambda item: (float(item["projection_global_contribution"]), float(item["loss"]))
    )


def best_states_for_cap(
    target_pools: dict[str, list[dict[str, Any]]],
    cap: float,
    budgets: list[float],
    per_target_limit: int,
    state_limit: int,
    bin_width: float,
) -> list[dict[str, Any]]:
    frontiers = {
        target: target_frontier(target_pools[target], cap=cap, per_target_limit=per_target_limit) for target in TARGETS
    }
    max_budget = max(budgets)
    min_future_projection = {
        target: min(float(candidate["projection_global_contribution"]) for candidate in frontiers[target])
        for target in TARGETS
    }

    states = [State(loss=0.0, projection=0.0, choices={})]
    for target_index, target in enumerate(TARGETS):
        remaining_targets = TARGETS[target_index + 1 :]
        min_remaining = sum(min_future_projection[remaining] for remaining in remaining_targets)
        next_states: list[State] = []
        for state in states:
            for candidate in frontiers[target]:
                projection = state.projection + float(candidate["projection_global_contribution"])
                if projection + min_remaining > max_budget + 0.05:
                    continue
                choices = dict(state.choices)
                choices[target] = compact_choice(candidate)
                next_states.append(
                    State(
                        loss=state.loss + float(candidate["loss"]),
                        projection=projection,
                        choices=choices,
                    )
                )
        states = prune_states(next_states, bin_width=bin_width, max_states=state_limit)

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        feasible = [state for state in states if state.projection <= budget + 1e-12]
        if not feasible:
            continue
        best = min(feasible, key=lambda state: state.loss)
        target_losses = {target: float(best.choices[target]["loss"]) for target in TARGETS}
        rows.append(
            {
                "cap": cap,
                "lambda_penalty": budget,
                "projection_budget": budget,
                "overall": float(best.loss / len(TARGETS)),
                "target_losses": target_losses,
                "max_abs_delta_vs_anchor": float(
                    max(float(best.choices[target]["max_abs_delta_vs_anchor"]) for target in TARGETS)
                ),
                "mean_abs_delta_vs_anchor": float(
                    np.mean([float(best.choices[target]["mean_abs_delta_vs_anchor"]) for target in TARGETS])
                ),
                "aggregate_projection_to_failed_direction": float(best.projection),
                "n_changed_targets": int(sum(best.choices[target]["kind"] != "anchor" for target in TARGETS)),
                "choices": best.choices,
            }
        )
    return rows


def write_markdown(report: dict[str, Any], markdown_path: Path) -> None:
    best_rows = sorted(report["rows"], key=lambda row: float(row["overall"]))
    lines = [
        "# Failed-direction constrained-combo scan",
        "",
        "This diagnostic directly searches target combinations under fixed aggregate projection budgets. "
        "For promotion compatibility, each row's `lambda_penalty` equals its projection budget.",
        "",
        "## Best overall rows",
        "",
        "| rank | cap | budget | local | max Δ anchor | mean Δ anchor | aggregate projection | changed targets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(best_rows[:30], start=1):
        lines.append(
            f"| {rank} | {row['cap']:.3f} | {row['projection_budget']:+.3f} | {row['overall']:.6f} | "
            f"{row['max_abs_delta_vs_anchor']:.6f} | {row['mean_abs_delta_vs_anchor']:.6f} | "
            f"{row['aggregate_projection_to_failed_direction']:+.4f} | {row['n_changed_targets']} |"
        )
    lines.extend(
        [
            "",
            "## Best by budget",
            "",
            "| budget | cap | local | max Δ anchor | mean Δ anchor | aggregate projection |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    seen_budgets: set[float] = set()
    for row in sorted(best_rows, key=lambda item: (float(item["projection_budget"]), float(item["overall"]))):
        budget = float(row["projection_budget"])
        if budget in seen_budgets:
            continue
        seen_budgets.add(budget)
        lines.append(
            f"| {budget:+.3f} | {row['cap']:.3f} | {row['overall']:.6f} | "
            f"{row['max_abs_delta_vs_anchor']:.6f} | {row['mean_abs_delta_vs_anchor']:.6f} | "
            f"{row['aggregate_projection_to_failed_direction']:+.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-output", type=Path, required=True)
    parser.add_argument("--anchor-id", type=int, default=178)
    parser.add_argument("--failed-id", type=int, default=287)
    parser.add_argument("--source-ids", default="220-286,288-300,302-317")
    parser.add_argument("--caps")
    parser.add_argument("--budgets")
    parser.add_argument("--scales")
    parser.add_argument("--gammas")
    parser.add_argument("--per-target-limit", type=int, default=220)
    parser.add_argument("--state-limit", type=int, default=12000)
    parser.add_argument("--bin-width", type=float, default=0.0005)
    args = parser.parse_args()

    caps = parse_float_list(args.caps, DEFAULT_CAPS)
    budgets = parse_float_list(args.budgets, DEFAULT_BUDGETS)
    target_pools, context = build_target_pools(
        anchor_id=args.anchor_id,
        failed_id=args.failed_id,
        source_ids=parse_source_ids(args.source_ids),
        scales=parse_float_list(args.scales, DEFAULT_SCALES),
        gammas=parse_float_list(args.gammas, DEFAULT_GAMMAS),
    )
    rows: list[dict[str, Any]] = []
    for cap in caps:
        rows.extend(
            best_states_for_cap(
                target_pools,
                cap=cap,
                budgets=budgets,
                per_target_limit=args.per_target_limit,
                state_limit=args.state_limit,
                bin_width=args.bin_width,
            )
        )

    report = {
        "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
        "diagnostic": "failed_direction_constrained_combo_scan",
        "anchor_id": args.anchor_id,
        "failed_id": args.failed_id,
        "anchor_public_score": 0.6014113009 if args.anchor_id == 178 else None,
        "failed_public_score": 0.6698787318 if args.failed_id == 287 else None,
        "anchor_source": context["anchor_source"],
        "failed_source": context["failed_source"],
        "source_ids_requested": parse_source_ids(args.source_ids),
        "source_ids_used": context["source_ids_used"],
        "caps": caps,
        "projection_budgets": budgets,
        "lambda_penalties": budgets,
        "lambda_penalty_field_note": "For promotion compatibility, lambda_penalty equals projection_budget.",
        "scales": parse_float_list(args.scales, DEFAULT_SCALES),
        "gammas": parse_float_list(args.gammas, DEFAULT_GAMMAS),
        "validation_lens": "full_oof_failed_direction_constrained_combo_scan_not_nested",
        "caution": (
            "This scan directly optimizes target combinations under aggregate projection budgets. It is still "
            "full-OOF selected and not public proof."
        ),
        "rows": rows,
        "best_by_overall": sorted(rows, key=lambda row: float(row["overall"])),
        "best_under_projection_005": sorted(
            [row for row in rows if float(row["aggregate_projection_to_failed_direction"]) <= 0.05],
            key=lambda row: float(row["overall"]),
        ),
        "best_under_projection_002": sorted(
            [row for row in rows if float(row["aggregate_projection_to_failed_direction"]) <= 0.02],
            key=lambda row: float(row["overall"]),
        ),
    }
    scan_output = args.scan_output if args.scan_output.is_absolute() else ROOT / args.scan_output
    scan_output.parent.mkdir(parents=True, exist_ok=True)
    scan_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = scan_output.with_suffix(".md")
    write_markdown(report, markdown_path)
    print(f"Wrote {relpath(scan_output)}")
    print(f"Wrote {relpath(markdown_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
