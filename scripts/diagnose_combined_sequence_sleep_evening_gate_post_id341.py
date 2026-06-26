#!/usr/bin/env python3
"""Combined no-artifact gate for sequence + sleep + evening sources post ID341.

This extends `diagnose_combined_new_source_gate_post_id341.py` by adding the
surviving ID340 sequence-summary source as a third independent delta around the
ID341 anchor. The intent is not to create ID342, but to test whether combining
multiple small, fold-gated source families can lower the conservative public
proxy below the confirmed ID178 public anchor without increasing failed-direction
exposure.

No submission CSV, base prediction artifact, notebook/history artifact,
submission-log row, Kaggle push, or DACON submission is created.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

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
from scripts.diagnose_id340_sequence_summary_split_probe import (  # noqa: E402
    fit_split_source_predictions,
)
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    make_model,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    EPS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    TARGETS,
    build_sleep_boundary_features,
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
SEQUENCE_DECISION_PATH = REPORTS_DIR / "id340_sequence_summary_candidate_decision_20260618.json"
SEQUENCE_GEOMETRY_PATH = REPORTS_DIR / "id340_exact_sequence_full_geometry_gate_20260618.json"

ANCHOR_IDS = [178, 287, 329, 341]
PUBLIC_ID178 = 0.6014113009
PUBLIC_058_GATE = 0.590

SEQUENCE_SPEC = ModelSpec("lr_c0_3", "lr", c_value=0.3)
SEQUENCE_BASE_BLEND_WEIGHT = 0.03

SEQUENCE_SCALES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
SLEEP_SCALES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
EVENING_SCALES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]

REVIEW_MAX_DELTA_VS_341 = 0.010
PROMOTION_PROXY_CEILING = PUBLIC_ID178


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fit_full_sequence_source(x_train: np.ndarray, x_test: np.ndarray, train) -> np.ndarray:
    """Fit the fixed ID340 sequence model on all train rows and predict test."""

    source_test = np.zeros((x_test.shape[0], len(TARGETS)), dtype=float)
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(SEQUENCE_SPEC))
        model.fit(x_train, y)
        source_test[:, target_idx] = model.predict_proba(x_test)[:, 1]
    return np.clip(source_test, EPS, 1 - EPS)


def build_sequence_bases(
    train,
    test,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Build exact ID340 sequence candidate values for each split and test.

    The sequence candidate is the already-gated design:
    `0.97 * ID178 + 0.03 * sequence_summary_lr_c0_3`.
    """

    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)

    split_base: dict[str, np.ndarray] = {}
    for split_name, folds in splits.items():
        source_values = reference178.copy()
        for target_idx, target in enumerate(TARGETS):
            y = train[target].astype(int).to_numpy()
            pred, covered = fit_split_source_predictions(x_train, y, folds, SEQUENCE_SPEC)
            source_values[:, target_idx] = np.where(covered, pred, reference178[:, target_idx])
        split_base[split_name] = np.clip(
            (1.0 - SEQUENCE_BASE_BLEND_WEIGHT) * reference178 + SEQUENCE_BASE_BLEND_WEIGHT * source_values,
            EPS,
            1 - EPS,
        )

    source_test = fit_full_sequence_source(x_train, x_test, train)
    test_base = np.clip(
        (1.0 - SEQUENCE_BASE_BLEND_WEIGHT) * base_tests[178] + SEQUENCE_BASE_BLEND_WEIGHT * source_test,
        EPS,
        1 - EPS,
    )
    coverage = {
        "n_sequence_rows": int(len(seq_table)),
        "n_total_channels": int(len(channel_names)),
        "n_summary_features": int(x_train.shape[1]),
        "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        "sequence_spec": asdict(SEQUENCE_SPEC),
        "sequence_base_blend_weight": SEQUENCE_BASE_BLEND_WEIGHT,
    }
    return split_base, test_base, coverage


def combine_three_values(
    reference341: np.ndarray,
    seq_scale: float,
    seq_values: np.ndarray,
    sbt_scale: float,
    sbt_values: np.ndarray,
    elr_scale: float,
    elr_values: np.ndarray,
) -> np.ndarray:
    return np.clip(
        reference341
        + seq_scale * (seq_values - reference341)
        + sbt_scale * (sbt_values - reference341)
        + elr_scale * (elr_values - reference341),
        EPS,
        1 - EPS,
    )


def evaluate_combo(
    seq_scale: float,
    sbt_scale: float,
    elr_scale: float,
    train,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
    seq_split_base: dict[str, np.ndarray],
    sbt_split_base: dict[str, np.ndarray],
    elr_split_base: dict[str, np.ndarray],
    seq_test_base: np.ndarray,
    sbt_test_base: np.ndarray,
    elr_test_base: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    split_rows: dict[str, Any] = {}
    for split_name, folds in splits.items():
        values = combine_three_values(
            reference341,
            seq_scale,
            seq_split_base[split_name],
            sbt_scale,
            sbt_split_base[split_name],
            elr_scale,
            elr_split_base[split_name],
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
    test_values = combine_three_values(
        base_tests[341], seq_scale, seq_test_base, sbt_scale, sbt_test_base, elr_scale, elr_test_base
    )
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
        "sequence_scale": seq_scale,
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
    sequence_decision = load_json(SEQUENCE_DECISION_PATH)
    sequence_geometry = load_json(SEQUENCE_GEOMETRY_PATH)

    train, test, _metrics_sorted, _sample = build_feature_frames()
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}

    seq_split_base, seq_test_base, seq_coverage = build_sequence_bases(train, test, splits, reference178, base_tests)

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
            seq_scale,
            sbt_scale,
            elr_scale,
            train,
            splits,
            reference178,
            reference341,
            seq_split_base,
            sbt_split_base,
            elr_split_base,
            seq_test_base,
            sbt_test_base,
            elr_test_base,
            base_tests,
        )
        for seq_scale in SEQUENCE_SCALES
        for sbt_scale in SLEEP_SCALES
        for elr_scale in EVENING_SCALES
        if not (seq_scale == 0 and sbt_scale == 0 and elr_scale == 0)
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
    best_all_primary = sorted(
        [row for row in combo_rows if row["all_primary_splits_beat_id341"]],
        key=lambda row: (row["test_geometry"]["proxy_public_risk"], row["test_geometry"]["max_delta_vs_341"]),
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "combined_sequence_sleep_evening_gate_post_id341",
        "purpose": (
            "Combine the surviving sequence-summary, sleep-boundary, and evening-load source deltas around ID341 "
            "before any artifact promotion."
        ),
        "inputs": {
            "sequence_decision": str(SEQUENCE_DECISION_PATH.relative_to(ROOT)),
            "sequence_geometry": str(SEQUENCE_GEOMETRY_PATH.relative_to(ROOT)),
            "sleep_v3": str(SLEEP_V3_PATH.relative_to(ROOT)),
            "evening_smoke": str(EVENING_PATH.relative_to(ROOT)),
        },
        "input_reads": {
            "sequence_primary_gate_pass": sequence_decision.get("decision", {}).get("primary_gate_pass"),
            "sequence_exact_proxy": sequence_geometry.get("conservative_public_risk_proxy"),
            "sequence_exact_risk_tier": sequence_geometry.get("risk_tier"),
            "sleep_best_proxy": sleep_v3.get("decision", {}).get("best_proxy"),
            "evening_best_proxy": evening.get("decision", {}).get("best_proxy"),
        },
        "selected_configs": {
            "sequence_summary": {
                "candidate": "blend_id178_sequence_summary_lr_c0_3_w0.03",
                "base_id": 178,
                "source_model": asdict(SEQUENCE_SPEC),
                "base_blend_weight": SEQUENCE_BASE_BLEND_WEIGHT,
            },
            "sleep_boundary": sbt_selected,
            "evening_load_recovery": elr_selected,
        },
        "feature_coverage": {
            "sequence_summary": seq_coverage,
            "sleep_boundary": sbt_coverage,
            "evening_load_recovery": elr_coverage,
        },
        "scale_grid": {
            "sequence_scales": SEQUENCE_SCALES,
            "sleep_scales": SLEEP_SCALES,
            "evening_scales": EVENING_SCALES,
        },
        "review_rules": {
            "max_delta_vs_341": REVIEW_MAX_DELTA_VS_341,
            "proxy_below_id178_required_for_promotion": PROMOTION_PROXY_CEILING,
            "public_058_gate": PUBLIC_058_GATE,
            "requires_non_positive_failed_projection": True,
            "requires_all_primary_splits_beat_id341_for_reviewable": True,
        },
        "combo_rows": combo_rows,
        "best_reviewable_low_movement": best_reviewable,
        "best_below_id178_proxy": best_below_id178,
        "best_by_proxy": best_proxy,
        "best_all_primary_by_proxy": best_all_primary[0] if best_all_primary else None,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "reviewable_low_movement_candidate_exists": bool(best_reviewable is not None),
            "proxy_below_id178_candidate_exists": bool(best_below_id178 is not None),
            "public_058_plausible": bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "best_proxy": best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_sequence_scale": best_proxy["sequence_scale"],
            "best_proxy_sleep_scale": best_proxy["sleep_scale"],
            "best_proxy_evening_scale": best_proxy["evening_scale"],
            "next_action": (
                "If no row drops below ID178 public proxy, do not create ID342. Keep sequence/sleep as weak leads "
                "and move to a genuinely stronger raw source or validation redesign."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Combined sequence + sleep + evening gate — post ID341",
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
        ("best_all_primary_by_proxy", report["best_all_primary_by_proxy"]),
        ("best_reviewable_low_movement", report["best_reviewable_low_movement"]),
        ("best_below_id178_proxy", report["best_below_id178_proxy"]),
    ]:
        if row is None:
            lines.append(f"- `{label}`: None")
            continue
        geom = row["test_geometry"]
        lines.append(
            f"- `{label}`: sequence `{row['sequence_scale']:.3f}`, sleep `{row['sleep_scale']:.3f}`, "
            f"evening `{row['evening_scale']:.3f}`, wins `{row['primary_wins_vs_id341']}`, "
            f"local `{row['local_cv_subject_hole_5']:.6f}`, proxy `{geom['proxy_public_risk']:.6f}`, "
            f"max Δ341 `{geom['max_delta_vs_341']:.6f}`, failed proj `{geom['projection_failed']:+.6f}`"
        )
    lines.extend(
        [
            "",
            "## Top 25 proxy rows",
            "",
            "| rank | seq | sleep | evening | wins | all primary | local | proxy | max Δ341 | failed proj | below ID178 | public 0.58 |",
            "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    top_rows = sorted(
        report["combo_rows"], key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"])
    )[:25]
    for rank, row in enumerate(top_rows, start=1):
        geom = row["test_geometry"]
        lines.append(
            f"| {rank} | {row['sequence_scale']:.3f} | {row['sleep_scale']:.3f} | "
            f"{row['evening_scale']:.3f} | {row['primary_wins_vs_id341']} | "
            f"`{row['all_primary_splits_beat_id341']}` | {row['local_cv_subject_hole_5']:.6f} | "
            f"{geom['proxy_public_risk']:.6f} | {geom['max_delta_vs_341']:.6f} | "
            f"{geom['projection_failed']:+.6f} | `{geom['proxy_public_risk'] < PUBLIC_ID178}` | "
            f"`{geom['public_058_plausible']}` |"
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
        default=REPORTS_DIR / "combined_sequence_sleep_evening_gate_post_id341_20260618.json",
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
