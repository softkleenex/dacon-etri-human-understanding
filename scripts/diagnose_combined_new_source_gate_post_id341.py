#!/usr/bin/env python3
"""Combined no-artifact gate for post-ID341 new sources.

Combines the validated sleep-boundary scale-scan lead with the evening-load S4
lead by adding tiny deltas around ID341. This checks whether independent new
sources can push the conservative proxy below the actual ID178 public anchor
before any ID342 artifact is created.

No submission CSV, base prediction artifact, notebook/history artifact,
submission-log row, Kaggle push, or DACON submission is created.
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

from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (  # noqa: E402
    assemble_split_values as assemble_elr_split_values,
)
from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (
    assemble_test_values as assemble_elr_test_values,
)
from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (
    build_evening_load_recovery_features,
)
from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (
    full_test_sources_for_selected as elr_full_test_sources_for_selected,
)
from scripts.diagnose_evening_load_recovery_smoke_post_id341 import (
    source_predictions_for_selected_splits as elr_source_predictions_for_selected_splits,
)
from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    EPS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    build_sleep_boundary_features,  # noqa: E402
    evaluate_values_on_split,
    load_reference_values,
    load_split_manifest,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import (  # noqa: E402
    geometry_report,
    load_base_test_values,
)
from scripts.diagnose_sleep_boundary_transition_v3_scale_scan_post_id341 import (  # noqa: E402
    assemble_split_values as assemble_sbt_split_values,
)
from scripts.diagnose_sleep_boundary_transition_v3_scale_scan_post_id341 import (
    assemble_test_values as assemble_sbt_test_values,
)
from scripts.diagnose_sleep_boundary_transition_v3_scale_scan_post_id341 import (
    full_test_sources_for_selected as sbt_full_test_sources_for_selected,
)
from scripts.diagnose_sleep_boundary_transition_v3_scale_scan_post_id341 import (
    source_predictions_for_selected_splits as sbt_source_predictions_for_selected_splits,
)
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
SLEEP_V3_PATH = REPORTS_DIR / "sleep_boundary_transition_v3_scale_scan_post_id341_20260618.json"
EVENING_PATH = REPORTS_DIR / "evening_load_recovery_smoke_post_id341_20260618.json"
ANCHOR_IDS = [178, 287, 329, 341]
PUBLIC_ID178 = 0.6014113009
PUBLIC_058_GATE = 0.590
SLEEP_SCALES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
EVENING_SCALES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
REVIEW_MAX_DELTA_VS_341 = 0.006
PROMOTION_PROXY_CEILING = PUBLIC_ID178


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def combine_values(
    reference341: np.ndarray, sbt_scale: float, sbt_values: np.ndarray, elr_scale: float, elr_values: np.ndarray
) -> np.ndarray:
    return np.clip(
        reference341 + sbt_scale * (sbt_values - reference341) + elr_scale * (elr_values - reference341),
        EPS,
        1 - EPS,
    )


def evaluate_combo(
    sbt_scale: float,
    elr_scale: float,
    train,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
    sbt_split_base: dict[str, np.ndarray],
    elr_split_base: dict[str, np.ndarray],
    sbt_test_base: np.ndarray,
    elr_test_base: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    split_rows: dict[str, Any] = {}
    for split_name, folds in splits.items():
        values = combine_values(
            reference341, sbt_scale, sbt_split_base[split_name], elr_scale, elr_split_base[split_name]
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
    test_values = combine_values(base_tests[341], sbt_scale, sbt_test_base, elr_scale, elr_test_base)
    geometry = geometry_report(test_values, base_tests, local_cv)
    reviewable = bool(
        primary_passes
        and all(primary_passes.values())
        and geometry["max_delta_vs_341"] <= REVIEW_MAX_DELTA_VS_341
        and geometry["projection_failed"] <= 0
    )
    promotable = bool(
        reviewable and geometry["proxy_public_risk"] < PROMOTION_PROXY_CEILING and geometry["public_058_plausible"]
    )
    return {
        "sleep_scale": sbt_scale,
        "evening_scale": elr_scale,
        "split_rows": split_rows,
        "primary_passes_vs_id341": primary_passes,
        "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
        "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
        "local_cv_subject_hole_5": local_cv,
        "test_geometry": geometry,
        "reviewable_low_movement_candidate": reviewable,
        "promotable_public_058_candidate": promotable,
    }


def build_report() -> dict[str, Any]:
    sleep_v3 = load_json(SLEEP_V3_PATH)
    evening = load_json(EVENING_PATH)
    train, test, _metrics_sorted, _sample = build_feature_frames()
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}

    sbt_selected = sleep_v3["selected_target_configs"]
    elr_selected = evening["selected_target_configs"]
    sbt_features, sbt_coverage = build_sleep_boundary_features(train, test)
    elr_features, elr_coverage = build_evening_load_recovery_features(train, test)
    sbt_cols = [col for col in sbt_features.columns if col.startswith("sbt_")]
    elr_cols = [col for col in elr_features.columns if col.startswith("elr_")]
    sbt_train_features = sbt_features[sbt_features["is_test"].eq(0)].reset_index(drop=True)
    elr_train_features = elr_features[elr_features["is_test"].eq(0)].reset_index(drop=True)

    sbt_split_sources = sbt_source_predictions_for_selected_splits(
        train, sbt_train_features, sbt_cols, splits, sbt_selected, reference341
    )
    elr_split_sources = elr_source_predictions_for_selected_splits(
        train, elr_train_features, elr_cols, splits, elr_selected, reference341
    )
    sbt_test_sources = sbt_full_test_sources_for_selected(train, sbt_features, sbt_cols, sbt_selected, base_tests[341])
    elr_test_sources = elr_full_test_sources_for_selected(train, elr_features, elr_cols, elr_selected)

    sbt_split_base = {
        split_name: assemble_sbt_split_values(
            1.0, split_name, sbt_selected, sbt_split_sources, reference178, reference341
        )
        for split_name in splits
    }
    elr_split_base = {
        split_name: assemble_elr_split_values(
            1.0, split_name, elr_selected, elr_split_sources, reference178, reference341
        )
        for split_name in splits
    }
    sbt_test_base = assemble_sbt_test_values(1.0, sbt_selected, sbt_test_sources, base_tests)
    elr_test_base = assemble_elr_test_values(1.0, elr_selected, elr_test_sources, base_tests)

    combo_rows = [
        evaluate_combo(
            sbt_scale,
            elr_scale,
            train,
            splits,
            reference178,
            reference341,
            sbt_split_base,
            elr_split_base,
            sbt_test_base,
            elr_test_base,
            base_tests,
        )
        for sbt_scale in SLEEP_SCALES
        for elr_scale in EVENING_SCALES
        if not (sbt_scale == 0 and elr_scale == 0)
    ]
    reviewable = [row for row in combo_rows if row["reviewable_low_movement_candidate"]]
    below_id178 = [row for row in reviewable if row["test_geometry"]["proxy_public_risk"] < PROMOTION_PROXY_CEILING]
    best_reviewable = (
        sorted(
            reviewable,
            key=lambda row: (row["test_geometry"]["proxy_public_risk"], row["test_geometry"]["max_delta_vs_341"]),
        )[0]
        if reviewable
        else None
    )
    best_below_id178 = (
        sorted(
            below_id178,
            key=lambda row: (row["test_geometry"]["proxy_public_risk"], row["test_geometry"]["max_delta_vs_341"]),
        )[0]
        if below_id178
        else None
    )
    best_proxy = sorted(
        combo_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"])
    )[0]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "combined_new_source_gate_post_id341",
        "purpose": "Combine sleep-boundary and evening-load micro source deltas around ID341 before any artifact promotion.",
        "inputs": {
            "sleep_v3": str(SLEEP_V3_PATH.relative_to(ROOT)),
            "evening_smoke": str(EVENING_PATH.relative_to(ROOT)),
        },
        "selected_configs": {"sleep_boundary": sbt_selected, "evening_load_recovery": elr_selected},
        "feature_coverage": {"sleep_boundary": sbt_coverage, "evening_load_recovery": elr_coverage},
        "scale_grid": {"sleep_scales": SLEEP_SCALES, "evening_scales": EVENING_SCALES},
        "review_rules": {
            "max_delta_vs_341": REVIEW_MAX_DELTA_VS_341,
            "proxy_below_id178_required_for_promotion": PROMOTION_PROXY_CEILING,
            "public_058_gate": PUBLIC_058_GATE,
        },
        "combo_rows": combo_rows,
        "best_reviewable_low_movement": best_reviewable,
        "best_below_id178_proxy": best_below_id178,
        "best_by_proxy": best_proxy,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "reviewable_low_movement_candidate_exists": bool(best_reviewable is not None),
            "proxy_below_id178_candidate_exists": bool(best_below_id178 is not None),
            "public_058_plausible": bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "best_proxy": best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_sleep_scale": best_proxy["sleep_scale"],
            "best_proxy_evening_scale": best_proxy["evening_scale"],
            "next_action": (
                "If no combined row falls below ID178, keep both as source leads and move to routine-stability/place entropy. "
                "If a row falls below ID178 but not Public 0.58, require another independent source before artifact promotion."
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
        "# Combined new-source gate — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push, or DACON submission.",
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
        geom = row["test_geometry"]
        lines.append(
            f"- `{label}`: sleep scale `{row['sleep_scale']:.3f}`, evening scale `{row['evening_scale']:.3f}`, "
            f"wins `{row['primary_wins_vs_id341']}`, local `{row['local_cv_subject_hole_5']:.6f}`, "
            f"proxy `{geom['proxy_public_risk']:.6f}`, max Δ341 `{geom['max_delta_vs_341']:.6f}`, "
            f"failed proj `{geom['projection_failed']:+.6f}`"
        )
    lines.extend(
        [
            "",
            "## Top 20 proxy rows",
            "",
            "| rank | sleep scale | evening scale | wins | all primary | local | proxy | max Δ341 | failed proj | below ID178 | public 0.58 |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    top_rows = sorted(
        report["combo_rows"], key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"])
    )[:20]
    for rank, row in enumerate(top_rows, start=1):
        geom = row["test_geometry"]
        lines.append(
            f"| {rank} | {row['sleep_scale']:.3f} | {row['evening_scale']:.3f} | {row['primary_wins_vs_id341']} | "
            f"`{row['all_primary_splits_beat_id341']}` | {row['local_cv_subject_hole_5']:.6f} | "
            f"{geom['proxy_public_risk']:.6f} | {geom['max_delta_vs_341']:.6f} | {geom['projection_failed']:+.6f} | "
            f"`{geom['proxy_public_risk'] < PUBLIC_ID178}` | `{geom['public_058_plausible']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Reviewable low-movement candidate exists: `{report['decision']['reviewable_low_movement_candidate_exists']}`",
            f"- Proxy-below-ID178 candidate exists: `{report['decision']['proxy_below_id178_candidate_exists']}`",
            f"- Public `0.58` plausible: `{report['decision']['public_058_plausible']}`",
            f"- Best proxy: `{report['decision']['best_proxy']:.6f}`",
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
        default=REPORTS_DIR / "combined_new_source_gate_post_id341_20260618.json",
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
