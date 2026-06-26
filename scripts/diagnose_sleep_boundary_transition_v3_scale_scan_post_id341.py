#!/usr/bin/env python3
"""Geometry-capped scale scan for sleep-boundary v2 targets.

Report-only diagnostic. It reuses the v2-selected sleep-boundary target configs,
scans a global weight scale, evaluates frozen splits and test geometry, and
identifies whether a reviewable low-movement candidate exists. It creates no
submission CSV, base prediction artifact, notebook/history artifact, submission
log row, or DACON submission.
"""

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

from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    EPS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    TARGETS,
    build_sleep_boundary_features,
    evaluate_values_on_split,
    fold_matrix,
    load_reference_values,
    load_split_manifest,
    make_model,
    predict_target_for_split,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import (  # noqa: E402
    ID341_PROXY,
    geometry_report,
    load_base_test_values,
    scan_target_configs,
    select_target_configs,
)
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
V2_REPORT_PATH = REPORTS_DIR / "sleep_boundary_transition_v2_gate_post_id341_20260618.json"
SCALES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
ANCHOR_IDS = [178, 287, 329, 341]
REVIEW_MAX_DELTA_VS_341 = 0.040
REVIEW_PROXY_MARGIN = 0.00010


def source_predictions_for_selected_splits(
    train,
    features_train,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    selected_configs: dict[str, dict[str, Any]],
    reference341: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for split_name, folds in splits.items():
        for target, config in selected_configs.items():
            target_idx = TARGETS.index(target)
            spec = next(spec for spec in MODEL_SPECS if spec.name == config["model"])
            pred, _covered = predict_target_for_split(
                features_train, feature_cols, train[target].astype(int).to_numpy(), folds, spec
            )
            source_values = reference341.copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), reference341[:, target_idx], pred)
            out[(split_name, target)] = source_values[:, target_idx]
    return out


def full_test_sources_for_selected(
    train,
    features,
    feature_cols: list[str],
    selected_configs: dict[str, dict[str, Any]],
    reference341_test: np.ndarray,
) -> dict[str, np.ndarray]:
    n_train = len(train)
    n_total = len(features)
    fit_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n_total, dtype=int)
    out: dict[str, np.ndarray] = {}
    for target, config in selected_configs.items():
        spec = next(spec for spec in MODEL_SPECS if spec.name == config["model"])
        y = train[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            out[target] = np.full(len(test_idx), float(np.mean(y)), dtype=float)
            continue
        fit_x, test_x = fold_matrix(features, feature_cols, fit_idx, test_idx)
        model = make_model(spec)
        model.fit(fit_x, y)
        out[target] = np.clip(model.predict_proba(test_x)[:, 1], EPS, 1 - EPS)
    return out


def assemble_split_values(
    scale: float,
    split_name: str,
    selected_configs: dict[str, dict[str, Any]],
    split_sources: dict[tuple[str, str], np.ndarray],
    reference178: np.ndarray,
    reference341: np.ndarray,
) -> np.ndarray:
    values = reference341.copy()
    anchors = {"id178": reference178, "id341": reference341}
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        anchor = anchors[config["anchor"]][:, target_idx]
        source = split_sources[(split_name, target)]
        effective_weight = float(config["weight"] * scale)
        values[:, target_idx] = np.clip((1.0 - effective_weight) * anchor + effective_weight * source, EPS, 1 - EPS)
    return values


def assemble_test_values(
    scale: float,
    selected_configs: dict[str, dict[str, Any]],
    test_sources: dict[str, np.ndarray],
    base_tests: dict[int, np.ndarray],
) -> np.ndarray:
    values = base_tests[341].copy()
    for target, config in selected_configs.items():
        target_idx = TARGETS.index(target)
        anchor_id = 178 if config["anchor"] == "id178" else 341
        anchor = base_tests[anchor_id][:, target_idx]
        source = test_sources[target]
        effective_weight = float(config["weight"] * scale)
        values[:, target_idx] = np.clip((1.0 - effective_weight) * anchor + effective_weight * source, EPS, 1 - EPS)
    return values


def evaluate_scale(
    scale: float,
    train,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    selected_configs: dict[str, dict[str, Any]],
    split_sources: dict[tuple[str, str], np.ndarray],
    test_sources: dict[str, np.ndarray],
    reference178: np.ndarray,
    reference341: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    split_rows: dict[str, Any] = {}
    for split_name, folds in splits.items():
        values = assemble_split_values(scale, split_name, selected_configs, split_sources, reference178, reference341)
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
    test_values = assemble_test_values(scale, selected_configs, test_sources, base_tests)
    geometry = geometry_report(test_values, base_tests, local_cv)
    reviewable = bool(
        primary_passes
        and all(primary_passes.values())
        and geometry["proxy_public_risk"] < ID341_PROXY - REVIEW_PROXY_MARGIN
        and geometry["max_delta_vs_341"] <= REVIEW_MAX_DELTA_VS_341
    )
    return {
        "scale": scale,
        "effective_weights": {target: float(config["weight"] * scale) for target, config in selected_configs.items()},
        "split_rows": split_rows,
        "primary_passes_vs_id341": primary_passes,
        "primary_wins_vs_id341": int(sum(bool(value) for value in primary_passes.values())),
        "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
        "local_cv_subject_hole_5": local_cv,
        "test_geometry": geometry,
        "reviewable_low_movement_candidate": reviewable,
    }


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    features, feature_coverage = build_sleep_boundary_features(train, test)
    feature_cols = [col for col in features.columns if col.startswith("sbt_")]
    features_train = features[features["is_test"].eq(0)].reset_index(drop=True)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    scan_rows, _source_cache = scan_target_configs(
        train, features_train, feature_cols, splits, reference178, reference341
    )
    selected_configs = select_target_configs(scan_rows)
    base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}
    split_sources = source_predictions_for_selected_splits(
        train, features_train, feature_cols, splits, selected_configs, reference341
    )
    test_sources = full_test_sources_for_selected(train, features, feature_cols, selected_configs, base_tests[341])
    scale_rows = [
        evaluate_scale(
            scale,
            train,
            splits,
            selected_configs,
            split_sources,
            test_sources,
            reference178,
            reference341,
            base_tests,
        )
        for scale in SCALES
    ]
    reviewable_rows = [row for row in scale_rows if row["reviewable_low_movement_candidate"]]
    best_reviewable = (
        sorted(reviewable_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["scale"]))[0]
        if reviewable_rows
        else None
    )
    best_proxy = sorted(
        scale_rows, key=lambda row: (row["test_geometry"]["proxy_public_risk"], -row["primary_wins_vs_id341"])
    )[0]
    best_split = sorted(
        scale_rows,
        key=lambda row: (
            -row["primary_wins_vs_id341"],
            row["local_cv_subject_hole_5"],
            row["test_geometry"]["proxy_public_risk"],
        ),
    )[0]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sleep_boundary_transition_v3_scale_scan_post_id341",
        "purpose": "Scan lower-movement scale factors for the v2 selected sleep-boundary targets.",
        "v2_report_path": str(V2_REPORT_PATH.relative_to(ROOT)),
        "feature_coverage": feature_coverage,
        "selected_target_configs": selected_configs,
        "scales": SCALES,
        "review_rules": {
            "review_max_delta_vs_341": REVIEW_MAX_DELTA_VS_341,
            "review_proxy_margin_vs_id341": REVIEW_PROXY_MARGIN,
            "requires_all_primary_splits_beat_id341": True,
        },
        "scale_rows": scale_rows,
        "best_reviewable_low_movement": best_reviewable,
        "best_by_proxy": best_proxy,
        "best_by_split_then_proxy": best_split,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "reviewable_low_movement_candidate_exists": bool(best_reviewable is not None),
            "best_reviewable_scale": None if best_reviewable is None else best_reviewable["scale"],
            "best_proxy": best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_scale": best_proxy["scale"],
            "public_058_plausible": bool(best_proxy["test_geometry"]["public_058_plausible"]),
            "next_action": (
                "If a reviewable row exists, run a final no-artifact prepromotion gate with exact target weights and stricter geometry. "
                "Do not create ID342 or submit unless that prepromotion gate remains clean."
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
        "# Sleep-boundary transition v3 scale scan — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Selected v2 targets",
        "",
        "| target | base candidate | base weight |",
        "|---|---|---:|",
    ]
    for target, config in report["selected_target_configs"].items():
        lines.append(f"| `{target}` | `{config['candidate']}` | {config['weight']:.4f} |")
    lines.extend(
        [
            "",
            "## Scale scan",
            "",
            "| scale | primary wins | all primary | local | proxy | Δ341 mean/max | failed proj | reviewable |",
            "|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["scale_rows"]:
        geom = row["test_geometry"]
        lines.append(
            f"| {row['scale']:.2f} | {row['primary_wins_vs_id341']} | `{row['all_primary_splits_beat_id341']}` | "
            f"{row['local_cv_subject_hole_5']:.6f} | {geom['proxy_public_risk']:.6f} | "
            f"{geom['mean_delta_vs_341']:.6f}/{geom['max_delta_vs_341']:.6f} | "
            f"{geom['projection_failed']:+.6f} | `{row['reviewable_low_movement_candidate']}` |"
        )
    best_reviewable = report["best_reviewable_low_movement"]
    lines.extend(["", "## Decision", ""])
    if best_reviewable is None:
        lines.append("- Reviewable low-movement candidate: `False`")
    else:
        geom = best_reviewable["test_geometry"]
        lines.extend(
            [
                "- Reviewable low-movement candidate: `True`",
                f"- Best reviewable scale: `{best_reviewable['scale']:.2f}`",
                f"- Best reviewable proxy: `{geom['proxy_public_risk']:.6f}`",
                f"- Best reviewable max Δ vs ID341: `{geom['max_delta_vs_341']:.6f}`",
            ]
        )
    lines.extend(
        [
            f"- Best proxy scale: `{report['decision']['best_proxy_scale']:.2f}`",
            f"- Best proxy: `{report['decision']['best_proxy']:.6f}`",
            f"- Public `0.58` plausible: `{report['decision']['public_058_plausible']}`",
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
        default=REPORTS_DIR / "sleep_boundary_transition_v3_scale_scan_post_id341_20260618.json",
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
