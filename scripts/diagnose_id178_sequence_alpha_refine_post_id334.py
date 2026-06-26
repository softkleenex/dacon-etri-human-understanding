#!/usr/bin/env python3
"""Targeted report-only refinement around the ID334 rollback alpha pattern."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id178_sequence_alpha_scan_post_id334 import (  # noqa: E402
    build_conv_features,
    compact_combo,
    precompute_losses,
    stress_prediction_tables,
    summarize_combo,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))


TARGET_GRIDS: dict[str, list[float]] = {
    "Q1": [0.75, 0.875, 1.0],
    "Q2": [0.75, 0.875, 1.0],
    "Q3": [0.75, 0.875, 1.0],
    "S1": [0.0],
    "S2": [0.75, 0.875, 1.0],
    "S3": [0.0],
    "S4": [0.0, 0.125, 0.25, 0.375, 0.5],
}


def iter_target_grid() -> list[dict[str, float]]:
    targets = list(TARGET_GRIDS)
    rows = []
    for values in itertools.product(*[TARGET_GRIDS[target] for target in targets]):
        rows.append(dict(zip(targets, values, strict=True)))
    return rows


def select_rows(rows: list[dict[str, Any]], id334: dict[str, Any]) -> dict[str, Any]:
    stress_guarded = [
        row
        for row in rows
        if row["stress"]["min_improvement_vs_base"] >= id334["stress"]["min_improvement_vs_base"] - 1e-12
        and row["stress"]["mean_across_stress_strategies"] <= id334["stress"]["mean_across_stress_strategies"] + 0.00025
    ]
    geometry_guarded = [
        row
        for row in stress_guarded
        if row["geometry"]["max_abs_delta_vs_id178"] <= id334["geometry"]["max_abs_delta_vs_id178"] + 1e-12
        and row["geometry"]["projection_to_id287_failed_direction"]
        <= id334["geometry"]["projection_to_id287_failed_direction"] + 1e-12
    ]
    return {
        "best_stress": compact_combo(
            min(
                rows,
                key=lambda row: (
                    row["stress"]["mean_across_stress_strategies"],
                    row["stress"]["min_improvement_vs_base"] * -1.0,
                    row["local_oof_loss"],
                ),
            )
        ),
        "best_local_stress_guarded": compact_combo(
            min(
                stress_guarded,
                key=lambda row: (
                    row["local_oof_loss"],
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                ),
            )
        )
        if stress_guarded
        else None,
        "best_local_geometry_guarded": compact_combo(
            min(
                geometry_guarded,
                key=lambda row: (
                    row["local_oof_loss"],
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                ),
            )
        )
        if geometry_guarded
        else None,
        "top_12_geometry_guarded": [
            compact_combo(row)
            for row in sorted(
                geometry_guarded,
                key=lambda row: (
                    row["local_oof_loss"],
                    row["stress"]["mean_across_stress_strategies"],
                    row["geometry"]["max_abs_delta_vs_id178"],
                ),
            )[:12]
        ],
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID178 sequence alpha targeted refinement — post ID334",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It scans a small neighborhood around ID334 and does not create any submission artifact.",
        "",
        "## Anchor vs selected rows",
        "",
        "| row | local OOF | stress mean | min stress improvement | max Δ178 | failed projection | alphas |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    anchor = compact_combo(report["id334_anchor"])
    rows = {
        "id334_anchor": anchor,
        **{k: v for k, v in report["selected_rows"].items() if k != "top_12_geometry_guarded"},
    }
    for name, row in rows.items():
        if row is None:
            continue
        lines.append(
            f"| `{name}` | {row['local_oof_loss']:.6f} | {row['stress_mean']:.6f} | "
            f"{row['min_stress_improvement']:+.6f} | {row['max_delta_vs_id178']:.6f} | "
            f"{row['projection_to_id287_failed_direction']:+.4f} | `{row['alpha_by_target']}` |"
        )
    lines.extend(["", "## Verdict", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    test_sample = pd.read_csv(
        ROOT / "data" / "raw" / "ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"]
    )
    base_oof, base_test, base_source = load_artifact_pair(train, test, 173)
    id178_oof, id178_test, id178_source = load_artifact_pair(train, test, 178)
    _id287_oof, id287_test, id287_source = load_artifact_pair(train, test, 287)
    _id329_oof, id329_test, id329_source = load_artifact_pair(train, test, 329)

    alpha_values = sorted({value for values in TARGET_GRIDS.values() for value in values})
    conv_features = build_conv_features(train, test)
    strategies = stress_prediction_tables(train, test_sample, base_oof, conv_features)
    local_losses, stress_losses, _base_local_losses, stress_base_means = precompute_losses(
        train, base_oof, id178_oof, strategies, alpha_values
    )

    id334_alpha = {target: (1.0 if target in ["Q1", "Q2", "Q3", "S2"] else 0.0) for target in TARGETS}
    id334 = summarize_combo(
        id334_alpha,
        local_losses,
        stress_losses,
        stress_base_means,
        base_test,
        id178_test,
        id287_test,
        id329_test,
        label="id334_rollback_s1_s4",
    )
    rows = [
        summarize_combo(
            alpha_by_target,
            local_losses,
            stress_losses,
            stress_base_means,
            base_test,
            id178_test,
            id287_test,
            id329_test,
            label="targeted_grid_combo",
        )
        for alpha_by_target in iter_target_grid()
    ]
    selected_rows = select_rows(rows, id334)
    best_guarded = selected_rows["best_local_geometry_guarded"]
    promote = bool(
        best_guarded
        and best_guarded["local_oof_loss"] < id334["local_oof_loss"] - 0.0003
        and best_guarded["stress_mean"] <= id334["stress"]["mean_across_stress_strategies"] + 0.00025
    )
    reason = (
        "A tiny geometry-guarded alpha refinement improves ID334 local OOF while keeping stress essentially flat. "
        "It is still same-family and should only be promoted as a generated_unsubmitted artifact, never submitted directly."
        if promote
        else "No targeted alpha refinement is strong enough to promote. Keep ID334 as the cleaner controlled-probe artifact."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "id178_sequence_alpha_refine_post_id334",
        "target_grids": TARGET_GRIDS,
        "sources": {
            "id173": base_source,
            "id178": id178_source,
            "id287": id287_source,
            "id329": id329_source,
        },
        "id334_anchor": id334,
        "selected_rows": selected_rows,
        "decision": {
            "promote_generated_unsubmitted_artifact": promote,
            "public_058_plausible": False,
            "reason": reason,
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "id178_sequence_alpha_refine_post_id334_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"] | report["selected_rows"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
