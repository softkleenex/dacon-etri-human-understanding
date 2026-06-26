#!/usr/bin/env python3
"""Combined sleep/evening/external-prior gate after ID341.

Report-only diagnostic. This tests whether the surviving independent weak leads
around ID341 (sleep-boundary, evening-load/recovery, and older ETRI weak priors)
combine into a safer transfer candidate. It does not include same-family
sequence-summary deltas because the latest combined gate selected sequence scale
0.0 as the best proxy row.

No submission CSV, base prediction artifact, notebook/history artifact,
submission-log row, Kaggle run, or DACON submission is created.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_combined_new_source_gate_post_id341 import (  # noqa: E402
    assemble_elr_split_values,
    assemble_elr_test_values,
    assemble_sbt_split_values,
    assemble_sbt_test_values,
    build_evening_load_recovery_features,
    build_sleep_boundary_features,
    elr_full_test_sources_for_selected,
    elr_source_predictions_for_selected_splits,
    sbt_full_test_sources_for_selected,
    sbt_source_predictions_for_selected_splits,
)
from scripts.diagnose_external_etri_prior_smoke_post_id341 import (  # noqa: E402
    build_prior_tables,
    load_external_target_rows,
    make_prior_matrix,
)
from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    EPS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    TARGETS,
    evaluate_values_on_split,
    load_reference_values,
    load_split_manifest,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import (  # noqa: E402
    geometry_report,
    load_base_test_values,
)
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
SLEEP_V3_PATH = REPORTS_DIR / "sleep_boundary_transition_v3_scale_scan_post_id341_20260618.json"
EVENING_PATH = REPORTS_DIR / "evening_load_recovery_smoke_post_id341_20260618.json"
EXTERNAL_PRIOR_PATH = REPORTS_DIR / "external_etri_prior_smoke_post_id341_20260618.json"
PRIOR_TARGET_GROUPS = {
    "Q1": ["Q1"],
    "S2": ["S2"],
    "S4": ["S4"],
    "Q1_S2": ["Q1", "S2"],
    "Q1_S4": ["Q1", "S4"],
    "S2_S4": ["S2", "S4"],
    "Q1_S2_S4": ["Q1", "S2", "S4"],
}
ANCHOR_IDS = [178, 287, 329, 341]
PUBLIC_ID178 = 0.6014113009
PUBLIC_058_GATE = 0.590
SLEEP_SCALES = [0.0, 0.025, 0.05, 0.075]
EVENING_SCALES = [0.0, 0.025, 0.05, 0.075]
EXTERNAL_PRIOR_SCALES = [0.0, 0.0025, 0.005, 0.01, 0.02]
REVIEW_MAX_DELTA_VS_341 = 0.0075
PROMOTION_PROXY_CEILING = PUBLIC_ID178


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def combine_values(
    reference341: np.ndarray,
    sleep_scale: float,
    sleep_values: np.ndarray,
    evening_scale: float,
    evening_values: np.ndarray,
    external_scale: float,
    external_values: np.ndarray,
    external_targets: list[str],
) -> np.ndarray:
    out = np.clip(
        reference341 + sleep_scale * (sleep_values - reference341) + evening_scale * (evening_values - reference341),
        EPS,
        1 - EPS,
    )
    if external_scale > 0:
        for target in external_targets:
            target_idx = TARGETS.index(target)
            out[:, target_idx] = np.clip(
                out[:, target_idx] + external_scale * (external_values[:, target_idx] - reference341[:, target_idx]),
                EPS,
                1 - EPS,
            )
    return out


def evaluate_combo(
    sleep_scale: float,
    evening_scale: float,
    external_group: str,
    external_targets: list[str],
    external_scale: float,
    train,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
    sleep_split_base: dict[str, np.ndarray],
    evening_split_base: dict[str, np.ndarray],
    external_train_prior: np.ndarray,
    sleep_test_base: np.ndarray,
    evening_test_base: np.ndarray,
    external_test_prior: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    split_rows: dict[str, Any] = {}
    for split_name, folds in splits.items():
        values = combine_values(
            reference341,
            sleep_scale,
            sleep_split_base[split_name],
            evening_scale,
            evening_split_base[split_name],
            external_scale,
            external_train_prior,
            external_targets,
        )
        candidate_eval = evaluate_values_on_split(train, folds, values)
        id341_eval = evaluate_values_on_split(train, folds, reference341)
        id178_eval = evaluate_values_on_split(train, folds, reference178)
        split_rows[split_name] = {
            "candidate_loss": candidate_eval["mean_loss"],
            "id341_loss": id341_eval["mean_loss"],
            "id178_loss": id178_eval["mean_loss"],
            "delta_vs_id341": float(candidate_eval["mean_loss"] - id341_eval["mean_loss"]),
            "delta_vs_id178": float(candidate_eval["mean_loss"] - id178_eval["mean_loss"]),
            "beats_id341": bool(candidate_eval["mean_loss"] < id341_eval["mean_loss"]),
        }
    primary_passes = {
        split_name: split_rows[split_name]["beats_id341"]
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_rows
    }
    local_cv = float(split_rows["subject_hole_5"]["candidate_loss"])
    test_values = combine_values(
        base_tests[341],
        sleep_scale,
        sleep_test_base,
        evening_scale,
        evening_test_base,
        external_scale,
        external_test_prior,
        external_targets,
    )
    geometry = geometry_report(test_values, base_tests, local_cv)
    reviewable = bool(
        primary_passes
        and all(primary_passes.values())
        and geometry["max_delta_vs_341"] <= REVIEW_MAX_DELTA_VS_341
        and geometry["projection_failed"] <= 0
        and geometry["proxy_public_risk"] < geometry_report(base_tests[341], base_tests, local_cv)["proxy_public_risk"]
    )
    promotable = bool(
        reviewable and geometry["proxy_public_risk"] < PROMOTION_PROXY_CEILING and geometry["public_058_plausible"]
    )
    return {
        "sleep_scale": sleep_scale,
        "evening_scale": evening_scale,
        "external_group": external_group,
        "external_targets": external_targets,
        "external_scale": external_scale,
        "split_rows": split_rows,
        "primary_passes_vs_id341": primary_passes,
        "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
        "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
        "local_cv_subject_hole_5": local_cv,
        "test_geometry": geometry,
        "reviewable_low_movement_candidate": reviewable,
        "promotable_public_058_candidate": promotable,
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    geom = row["test_geometry"]
    return {
        "sleep_scale": row["sleep_scale"],
        "evening_scale": row["evening_scale"],
        "external_group": row["external_group"],
        "external_targets": row["external_targets"],
        "external_scale": row["external_scale"],
        "primary_wins_vs_id341": row["primary_wins_vs_id341"],
        "all_primary_splits_beat_id341": row["all_primary_splits_beat_id341"],
        "local_cv_subject_hole_5": row["local_cv_subject_hole_5"],
        "proxy_public_risk": geom["proxy_public_risk"],
        "beats_id341_proxy": geom["beats_id341_proxy"],
        "public_058_plausible": geom["public_058_plausible"],
        "mean_delta_vs_341": geom["mean_delta_vs_341"],
        "max_delta_vs_341": geom["max_delta_vs_341"],
        "projection_failed": geom["projection_failed"],
        "reviewable_low_movement_candidate": row["reviewable_low_movement_candidate"],
        "promotable_public_058_candidate": row["promotable_public_058_candidate"],
        "primary_deltas_vs_id341": {
            split_name: row["split_rows"][split_name]["delta_vs_id341"]
            for split_name in PRIMARY_GATE_SPLITS
            if split_name in row["split_rows"]
        },
    }


def build_report() -> dict[str, Any]:
    sleep_v3 = load_json(SLEEP_V3_PATH)
    evening = load_json(EVENING_PATH)
    external_prior_report = load_json(EXTERNAL_PRIOR_PATH)
    train, test, _metrics_sorted, _sample = build_feature_frames()
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}

    sleep_selected = sleep_v3["selected_target_configs"]
    evening_selected = evening["selected_target_configs"]
    sleep_features, sleep_coverage = build_sleep_boundary_features(train, test)
    evening_features, evening_coverage = build_evening_load_recovery_features(train, test)
    sleep_cols = [col for col in sleep_features.columns if col.startswith("sbt_")]
    evening_cols = [col for col in evening_features.columns if col.startswith("elr_")]
    sleep_train_features = sleep_features[sleep_features["is_test"].eq(0)].reset_index(drop=True)
    evening_train_features = evening_features[evening_features["is_test"].eq(0)].reset_index(drop=True)

    sleep_split_sources = sbt_source_predictions_for_selected_splits(
        train, sleep_train_features, sleep_cols, splits, sleep_selected, reference341
    )
    evening_split_sources = elr_source_predictions_for_selected_splits(
        train, evening_train_features, evening_cols, splits, evening_selected, reference341
    )
    sleep_test_sources = sbt_full_test_sources_for_selected(
        train, sleep_features, sleep_cols, sleep_selected, base_tests[341]
    )
    evening_test_sources = elr_full_test_sources_for_selected(train, evening_features, evening_cols, evening_selected)

    sleep_split_base = {
        split_name: assemble_sbt_split_values(
            1.0, split_name, sleep_selected, sleep_split_sources, reference178, reference341
        )
        for split_name in splits
    }
    evening_split_base = {
        split_name: assemble_elr_split_values(
            1.0, split_name, evening_selected, evening_split_sources, reference178, reference341
        )
        for split_name in splits
    }
    sleep_test_base = assemble_sbt_test_values(1.0, sleep_selected, sleep_test_sources, base_tests)
    evening_test_base = assemble_elr_test_values(1.0, evening_selected, evening_test_sources, base_tests)

    external_rows = load_external_target_rows()
    prior_tables = build_prior_tables(external_rows)
    external_train_prior, train_prior_usage = make_prior_matrix(train, prior_tables, reference341)
    external_test_prior, test_prior_usage = make_prior_matrix(test, prior_tables, base_tests[341])

    combo_rows = [
        evaluate_combo(
            sleep_scale,
            evening_scale,
            external_group,
            external_targets,
            external_scale,
            train,
            splits,
            reference178,
            reference341,
            sleep_split_base,
            evening_split_base,
            external_train_prior,
            sleep_test_base,
            evening_test_base,
            external_test_prior,
            base_tests,
        )
        for sleep_scale in SLEEP_SCALES
        for evening_scale in EVENING_SCALES
        for external_group, external_targets in PRIOR_TARGET_GROUPS.items()
        for external_scale in EXTERNAL_PRIOR_SCALES
        if not (sleep_scale == 0 and evening_scale == 0 and external_scale == 0)
    ]

    reviewable = [row for row in combo_rows if row["reviewable_low_movement_candidate"]]
    below_id178 = [row for row in reviewable if row["test_geometry"]["proxy_public_risk"] < PROMOTION_PROXY_CEILING]
    promotable = [row for row in combo_rows if row["promotable_public_058_candidate"]]
    best_proxy = min(
        combo_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"])
    )
    best_reviewable = (
        min(
            reviewable,
            key=lambda row: (row["test_geometry"]["proxy_public_risk"], row["test_geometry"]["max_delta_vs_341"]),
        )
        if reviewable
        else None
    )
    best_below_id178 = (
        min(
            below_id178,
            key=lambda row: (row["test_geometry"]["proxy_public_risk"], row["test_geometry"]["max_delta_vs_341"]),
        )
        if below_id178
        else None
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "combined_sleep_evening_external_prior_gate_post_id341",
        "purpose": (
            "Combine the surviving non-sequence weak leads around ID341: sleep-boundary, evening-load/recovery, "
            "and older-ETRI external weak priors."
        ),
        "inputs": {
            "sleep_v3": str(SLEEP_V3_PATH.relative_to(ROOT)),
            "evening_smoke": str(EVENING_PATH.relative_to(ROOT)),
            "external_prior_smoke": str(EXTERNAL_PRIOR_PATH.relative_to(ROOT)),
        },
        "input_reads": {
            "sleep_best_proxy": sleep_v3.get("decision", {}).get("best_proxy"),
            "evening_best_proxy": evening.get("decision", {}).get("best_proxy"),
            "external_best_proxy": external_prior_report.get("decision", {}).get("best_proxy"),
            "external_best_proxy_row_name": external_prior_report.get("decision", {}).get("best_proxy_row_name"),
        },
        "selected_configs": {
            "sleep_boundary": sleep_selected,
            "evening_load_recovery": evening_selected,
            "external_prior_groups": PRIOR_TARGET_GROUPS,
        },
        "feature_coverage": {
            "sleep_boundary": sleep_coverage,
            "evening_load_recovery": evening_coverage,
            "external_prior_train_usage": train_prior_usage,
            "external_prior_test_usage": test_prior_usage,
        },
        "scale_grid": {
            "sleep_scales": SLEEP_SCALES,
            "evening_scales": EVENING_SCALES,
            "external_prior_scales": EXTERNAL_PRIOR_SCALES,
        },
        "review_rules": {
            "max_delta_vs_341": REVIEW_MAX_DELTA_VS_341,
            "proxy_below_id178_required_for_promotion": PROMOTION_PROXY_CEILING,
            "public_058_gate": PUBLIC_058_GATE,
            "requires_non_positive_failed_projection": True,
            "requires_all_primary_splits_beat_id341_for_reviewable": True,
        },
        "top_rows": [
            compact_row(row)
            for row in sorted(
                combo_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"])
            )[:40]
        ],
        "best_by_proxy": compact_row(best_proxy),
        "best_reviewable_low_movement": compact_row(best_reviewable) if best_reviewable else None,
        "best_below_id178_proxy": compact_row(best_below_id178) if best_below_id178 else None,
        "reviewable_rows": [
            compact_row(row)
            for row in sorted(
                reviewable,
                key=lambda row: (row["test_geometry"]["proxy_public_risk"], row["test_geometry"]["max_delta_vs_341"]),
            )[:40]
        ],
        "promotable_rows": [compact_row(row) for row in promotable[:40]],
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "reviewable_low_movement_candidate_exists": bool(reviewable),
            "proxy_below_id178_candidate_exists": bool(best_below_id178 is not None),
            "public_058_plausible": bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "best_proxy": best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_sleep_scale": best_proxy["sleep_scale"],
            "best_proxy_evening_scale": best_proxy["evening_scale"],
            "best_proxy_external_group": best_proxy["external_group"],
            "best_proxy_external_scale": best_proxy["external_scale"],
            "next_action": (
                "If this combined non-sequence weak-source gate still does not drop below ID178 public, keep the local "
                "new-source stack closed and use the explicit-approved Kaggle upstream rebuild as the next score-relevant step."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(out) else f"{out:.{digits}f}"


def table_rows(rows: list[dict[str, Any]], limit: int = 40) -> list[str]:
    lines = []
    for rank, row in enumerate(rows[:limit], start=1):
        lines.append(
            f"| {rank} | {row['sleep_scale']:.3f} | {row['evening_scale']:.3f} | "
            f"`{row['external_group']}` | {row['external_scale']:.4f} | "
            f"{row['primary_wins_vs_id341']} | `{row['all_primary_splits_beat_id341']}` | "
            f"{fmt(row['local_cv_subject_hole_5'])} | {fmt(row['proxy_public_risk'])} | "
            f"{fmt(row['mean_delta_vs_341'])}/{fmt(row['max_delta_vs_341'])} | "
            f"{fmt(row['projection_failed'])} | `{row['reviewable_low_movement_candidate']}` | "
            f"`{row['promotable_public_058_candidate']}` |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Combined sleep + evening + external-prior gate — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Best rows",
        "",
    ]
    for label, row in [
        ("best_by_proxy", report["best_by_proxy"]),
        ("best_reviewable_low_movement", report["best_reviewable_low_movement"]),
        ("best_below_id178_proxy", report["best_below_id178_proxy"]),
    ]:
        if row is None:
            lines.append(f"- `{label}`: None")
            continue
        lines.append(
            f"- `{label}`: sleep `{row['sleep_scale']:.3f}`, evening `{row['evening_scale']:.3f}`, "
            f"external `{row['external_group']}` x `{row['external_scale']:.4f}`, wins `{row['primary_wins_vs_id341']}`, "
            f"local `{row['local_cv_subject_hole_5']:.6f}`, proxy `{row['proxy_public_risk']:.6f}`, "
            f"max Δ341 `{row['max_delta_vs_341']:.6f}`, failed proj `{row['projection_failed']:+.6f}`"
        )
    lines.extend(
        [
            "",
            "## Top proxy rows",
            "",
            "| rank | sleep | evening | external | ext scale | wins | all primary | local | proxy | Δ341 mean/max | failed proj | review | promote |",
            "|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|",
            *table_rows(report["top_rows"], limit=40),
            "",
            "## Reviewable rows",
            "",
            "| rank | sleep | evening | external | ext scale | wins | all primary | local | proxy | Δ341 mean/max | failed proj | review | promote |",
            "|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|",
            *table_rows(report["reviewable_rows"], limit=40),
            "",
            "## Decision",
            "",
            f"- Reviewable low-movement candidate exists: `{report['decision']['reviewable_low_movement_candidate_exists']}`",
            f"- Proxy-below-ID178 candidate exists: `{report['decision']['proxy_below_id178_candidate_exists']}`",
            f"- Public `0.58` plausible: `{report['decision']['public_058_plausible']}`",
            f"- Best proxy: `{report['decision']['best_proxy']:.6f}`",
            f"- Best proxy scales: sleep `{report['decision']['best_proxy_sleep_scale']}`, evening `{report['decision']['best_proxy_evening_scale']}`, external `{report['decision']['best_proxy_external_group']}` x `{report['decision']['best_proxy_external_scale']}`",
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
        default=REPORTS_DIR / "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
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
