#!/usr/bin/env python3
"""Diagnose target rollback variants of the ID178 sequence overlay.

This report-only diagnostic compares hypothetical variants that keep ID178's
sequence overlay for some targets and revert other targets to its base ID173.
It does not write submission CSVs, base prediction artifacts, notebooks, or
submission-log rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
EPS = 1e-6


VARIANTS: dict[str, dict[str, Any]] = {
    "id178_keep_all": {
        "keep_sequence_targets": ["Q1", "Q2", "Q3", "S1", "S2", "S4"],
        "rationale": "Original ID178 selected sequence overlay.",
    },
    "rollback_s1": {
        "keep_sequence_targets": ["Q1", "Q2", "Q3", "S2", "S4"],
        "rationale": "Revert S1 because LOSO and boundary stress dislike the S1 sequence overlay.",
    },
    "rollback_s4": {
        "keep_sequence_targets": ["Q1", "Q2", "Q3", "S1", "S2"],
        "rationale": "Revert S4 because boundary stress dislikes the S4 sequence overlay.",
    },
    "rollback_s1_s4": {
        "keep_sequence_targets": ["Q1", "Q2", "Q3", "S2"],
        "rationale": "Keep only the targets that are positive or neutral across stricter stress checks.",
    },
}


def find_base_file(kind: str, candidate_id: int) -> Path:
    paths = sorted((ROOT / "artifacts" / "base_predictions").glob(f"base_{kind}_id_{candidate_id:03d}_*.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one base_{kind} file for id {candidate_id}, found {paths}")
    return paths[0]


def candidate_slug(path: Path) -> str:
    match = re.search(r"id_\d+_(.+)\.csv$", path.name)
    return match.group(1) if match else path.stem


def load_prediction_pair(candidate_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    oof_path = find_base_file("oof", candidate_id)
    test_path = find_base_file("test", candidate_id)
    return (
        pd.read_csv(oof_path),
        pd.read_csv(test_path),
        {
            "oof": str(oof_path.relative_to(ROOT)),
            "test": str(test_path.relative_to(ROOT)),
            "slug": candidate_slug(oof_path),
        },
    )


def align_oof(pred: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    labels = train[["subject_id", "date"]].copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.strftime("%Y-%m-%d")
    out = labels.merge(pred, on=["subject_id", "date"], how="left")
    if out[TARGETS].isna().any().any():
        raise ValueError("OOF prediction failed to align with train rows.")
    return out


def align_test(pred: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    keys = test[["subject_id", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    out = keys.merge(pred, on=["subject_id", "date"], how="left")
    if out[TARGETS].isna().any().any():
        raise ValueError("Test prediction failed to align with test rows.")
    return out


def variant_frame(base173: pd.DataFrame, id178: pd.DataFrame, keep_sequence_targets: list[str]) -> pd.DataFrame:
    out = base173[["subject_id", "date", *TARGETS]].copy()
    for target in keep_sequence_targets:
        out[target] = id178[target].astype(float).to_numpy()
    return out


def mean_logloss(train: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    per_target = {
        target: float(
            log_loss(
                train[target].astype(int),
                np.clip(pred[target].astype(float), EPS, 1.0 - EPS),
                labels=[0, 1],
            )
        )
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def robust_strategy_score(
    robustness_report: dict[str, Any],
    keep_sequence_targets: set[str],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    means = []
    for strategy in robustness_report["strategies"]:
        losses = []
        per_target: dict[str, float] = {}
        for target in TARGETS:
            info = strategy["targets"][target]
            use_sequence = target in keep_sequence_targets
            loss = float(info["selected_blend_loss"] if use_sequence else info["base_loss"])
            per_target[target] = loss
            losses.append(loss)
        mean_loss = float(np.mean(losses))
        base_mean = float(strategy["base_mean"])
        rows[strategy["strategy"]] = {
            "mean_loss": mean_loss,
            "base_mean": base_mean,
            "improvement_vs_base": base_mean - mean_loss,
            "per_target": per_target,
        }
        means.append(mean_loss)
    return {
        "strategy_scores": rows,
        "mean_across_stress_strategies": float(np.mean(means)),
        "min_improvement_vs_base": float(min(row["improvement_vs_base"] for row in rows.values())),
    }


def geometry(
    test_pred: pd.DataFrame, id178_test: pd.DataFrame, id287_test: pd.DataFrame, id329_test: pd.DataFrame
) -> dict[str, float]:
    arr = test_pred[TARGETS].astype(float).to_numpy()
    a178 = id178_test[TARGETS].astype(float).to_numpy()
    a287 = id287_test[TARGETS].astype(float).to_numpy()
    a329 = id329_test[TARGETS].astype(float).to_numpy()
    failed_vec = (a287 - a178).reshape(-1)
    vec = (arr - a178).reshape(-1)
    denom = float(np.dot(failed_vec, failed_vec))
    projection_failed = float(np.dot(vec, failed_vec) / denom) if denom > 0 else 0.0
    return {
        "mean_abs_delta_vs_id178": float(np.mean(np.abs(arr - a178))),
        "max_abs_delta_vs_id178": float(np.max(np.abs(arr - a178))),
        "rmse_vs_id178": float(np.sqrt(np.mean((arr - a178) ** 2))),
        "mean_abs_delta_vs_id329": float(np.mean(np.abs(arr - a329))),
        "max_abs_delta_vs_id329": float(np.max(np.abs(arr - a329))),
        "projection_to_id287_failed_direction": projection_failed,
        "prediction_mean": float(np.mean(arr)),
        "prediction_std": float(np.std(arr)),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID178 sequence target rollback diagnostic",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Variant ranking",
        "",
        "| rank | variant | OOF loss | stress mean | min stress improvement | max Δ178 | failed projection | verdict |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(report["ranked_variants"], start=1):
        lines.append(
            f"| {rank} | `{row['variant']}` | {row['oof_loss']:.6f} | "
            f"{row['robustness']['mean_across_stress_strategies']:.6f} | "
            f"{row['robustness']['min_improvement_vs_base']:+.6f} | "
            f"{row['geometry']['max_abs_delta_vs_id178']:.6f} | "
            f"{row['geometry']['projection_to_id287_failed_direction']:+.4f} | {row['verdict']} |"
        )
    best = report["ranked_variants"][0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strongest diagnostic variant is `{best['variant']}`. It is more stress-consistent than the original ID178 overlay, but it is still a same-family sequence rollback, not a Public 0.58-proven candidate.",
            "",
            "If this is ever promoted, it must receive a new immutable submission ID with matching CSV, notebook, submission-log row, metadata JSON, and base OOF/test artifacts. Do not reuse the ID178 filename.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_json: Path, robustness_report_path: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    robustness = json.loads(robustness_report_path.read_text())
    id173_oof_raw, id173_test_raw, id173_files = load_prediction_pair(173)
    id178_oof_raw, id178_test_raw, id178_files = load_prediction_pair(178)
    _id287_oof_raw, id287_test_raw, id287_files = load_prediction_pair(287)
    _id329_oof_raw, id329_test_raw, id329_files = load_prediction_pair(329)

    id173_oof = align_oof(id173_oof_raw, train)
    id178_oof = align_oof(id178_oof_raw, train)
    id173_test = align_test(id173_test_raw, test)
    id178_test = align_test(id178_test_raw, test)
    id287_test = align_test(id287_test_raw, test)
    id329_test = align_test(id329_test_raw, test)

    rows = []
    for name, spec in VARIANTS.items():
        keep = set(spec["keep_sequence_targets"])
        oof_pred = variant_frame(id173_oof, id178_oof, list(keep))
        test_pred = variant_frame(id173_test, id178_test, list(keep))
        oof_loss, per_target = mean_logloss(train, oof_pred)
        robust = robust_strategy_score(robustness, keep)
        geom = geometry(test_pred, id178_test, id287_test, id329_test)
        verdict = "reviewable_hypothesis_only"
        if name == "id178_keep_all":
            verdict = "confirmed_anchor_reference"
        elif robust["min_improvement_vs_base"] <= 0:
            verdict = "reject_not_consistently_improving"
        rows.append(
            {
                "variant": name,
                "rationale": spec["rationale"],
                "keep_sequence_targets": sorted(keep),
                "reverted_to_id173_targets": [target for target in TARGETS if target not in keep],
                "oof_loss": oof_loss,
                "oof_per_target": per_target,
                "robustness": robust,
                "geometry": geom,
                "verdict": verdict,
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["verdict"] == "reject_not_consistently_improving",
            row["robustness"]["mean_across_stress_strategies"],
            row["geometry"]["max_abs_delta_vs_id178"],
        ),
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "id178_sequence_target_rollback_post_id329",
        "source_robustness_report": str(robustness_report_path),
        "source_files": {
            "id173": id173_files,
            "id178": id178_files,
            "id287": id287_files,
            "id329": id329_files,
        },
        "ranked_variants": ranked,
        "decision": {
            "promote_submission_candidate": False,
            "public_058_plausible": False,
            "public_060_probe_hypothesis": ranked[0]["variant"] if ranked else None,
            "reason": "Target rollback improves stress consistency, but it is same-family with ID178 and needs a full artifact contract plus explicit approval before any public probe.",
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    output_json.with_suffix(".md").write_text(build_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robustness-report",
        type=Path,
        default=REPORTS_DIR / "random_conv_sequence_robustness_post_id329_20260617.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "id178_sequence_target_rollback_post_id329_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.output_json, args.robustness_report)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(
        json.dumps(
            {
                "top_variant": report["ranked_variants"][0]["variant"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
