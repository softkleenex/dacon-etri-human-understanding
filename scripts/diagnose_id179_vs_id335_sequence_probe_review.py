#!/usr/bin/env python3
"""Report-only review of ID179 against ID335 after the post-ID335 rerank."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))
EPS = 1e-6


def load_artifact(kind: str, candidate_id: int) -> pd.DataFrame:
    paths = sorted((ROOT / "artifacts" / "base_predictions").glob(f"base_{kind}_id_{candidate_id:03d}_*.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one base_{kind} for id {candidate_id}, found {paths}")
    return pd.read_csv(paths[0])


def align(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.sort_values(["subject_id", "date"]).reset_index(drop=True)
    ref = reference[["subject_id", "date"]].copy()
    ref["date"] = pd.to_datetime(ref["date"]).dt.date
    if not out[["subject_id", "date"]].equals(ref.reset_index(drop=True)):
        raise ValueError("Artifact does not align with reference frame.")
    return out


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.run_strict_raw_feature_experiment import build_feature_frames

    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def local_scores(train: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    scores = {
        target: target_loss(train[target].astype(int).to_numpy(), pred[target].astype(float).to_numpy())
        for target in TARGETS
    }
    return float(np.mean(list(scores.values()))), scores


def slice_summary(train: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    tmp = train[["subject_id", "date", *TARGETS]].copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date"])
    tmp["date_bin"] = pd.qcut(tmp["date_dt"].map(lambda value: value.toordinal()), q=4, labels=["q1", "q2", "q3", "q4"])
    arr = pred[TARGETS].astype(float).to_numpy()
    target_idx = {target: idx for idx, target in enumerate(TARGETS)}
    subject_losses = {}
    for subject_id, group in tmp.groupby("subject_id", sort=True):
        idx = group.index.to_numpy()
        losses = [target_loss(group[target].astype(int).to_numpy(), arr[idx, target_idx[target]]) for target in TARGETS]
        subject_losses[str(subject_id)] = float(np.mean(losses))
    date_losses = {}
    for date_bin, group in tmp.groupby("date_bin", observed=True):
        idx = group.index.to_numpy()
        losses = [target_loss(group[target].astype(int).to_numpy(), arr[idx, target_idx[target]]) for target in TARGETS]
        date_losses[str(date_bin)] = float(np.mean(losses))
    _overall, targets = local_scores(train, pred)
    date_values = list(date_losses.values())
    late_early = float(date_losses.get("q4", date_values[-1]) - date_losses.get("q1", date_values[0]))
    return {
        "subject_losses": subject_losses,
        "date_bin_losses": date_losses,
        "target_losses": targets,
        "worst_subject": max(subject_losses, key=subject_losses.get),
        "worst_subject_loss": max(subject_losses.values()),
        "worst_date_bin": max(date_losses, key=date_losses.get),
        "worst_date_loss": max(date_losses.values()),
        "worst_target": max(targets, key=targets.get),
        "worst_target_loss": max(targets.values()),
        "late_minus_early": late_early,
    }


def robust_score(overall: float, slices: dict[str, Any]) -> float:
    return float(
        overall
        + 0.12 * max(0.0, slices["worst_subject_loss"] - overall)
        + 0.12 * max(0.0, slices["worst_date_loss"] - overall)
        + 0.12 * max(0.0, slices["worst_target_loss"] - overall)
        + 0.05 * max(0.0, slices["late_minus_early"])
    )


def geometry(test_pred: pd.DataFrame, anchors: dict[int, pd.DataFrame]) -> dict[str, float]:
    arr = test_pred[TARGETS].astype(float).to_numpy()
    arr178 = anchors[178][TARGETS].astype(float).to_numpy()
    arr287 = anchors[287][TARGETS].astype(float).to_numpy()
    arr329 = anchors[329][TARGETS].astype(float).to_numpy()
    arr335 = anchors[335][TARGETS].astype(float).to_numpy()
    failed = (arr287 - arr178).reshape(-1)
    vec = (arr - arr178).reshape(-1)
    denom = float(np.dot(failed, failed))
    return {
        "mean_delta_vs_178": float(np.mean(np.abs(arr - arr178))),
        "max_delta_vs_178": float(np.max(np.abs(arr - arr178))),
        "mean_delta_vs_329": float(np.mean(np.abs(arr - arr329))),
        "max_delta_vs_329": float(np.max(np.abs(arr - arr329))),
        "mean_delta_vs_335": float(np.mean(np.abs(arr - arr335))),
        "max_delta_vs_335": float(np.max(np.abs(arr - arr335))),
        "projection_failed": float(np.dot(vec, failed) / denom) if denom > 0 else 0.0,
        "prediction_mean": float(np.mean(arr)),
        "prediction_std": float(np.std(arr)),
    }


def review_candidate(
    candidate_id: int, train: pd.DataFrame, test: pd.DataFrame, anchors: dict[int, pd.DataFrame]
) -> dict[str, Any]:
    oof = align(load_artifact("oof", candidate_id), train)
    test_pred = align(load_artifact("test", candidate_id), test)
    overall, target_scores = local_scores(train, oof)
    slices = slice_summary(train, oof)
    return {
        "id": candidate_id,
        "overall": overall,
        "target_scores": target_scores,
        "slice_summary": slices,
        "robust_score": robust_score(overall, slices),
        "geometry": geometry(test_pred, anchors),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID179 vs ID335 sequence probe review",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Candidate comparison",
        "",
        "| id | overall | robust | worst subject | worst date | worst target | max Δ178 | max Δ335 | failed projection |",
        "|---:|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for row in report["candidates"]:
        slices = row["slice_summary"]
        geom = row["geometry"]
        lines.append(
            f"| {row['id']} | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` | "
            f"{slices['worst_date_bin']} `{slices['worst_date_loss']:.6f}` | "
            f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
            f"{geom['max_delta_vs_178']:.6f} | {geom['max_delta_vs_335']:.6f} | "
            f"{geom['projection_failed']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            report["decision"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, test = load_train_test()
    anchor_tests = {
        candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in [178, 287, 329, 335]
    }
    candidate_ids = [178, 179, 334, 335]
    candidates = [review_candidate(candidate_id, train, test, anchor_tests) for candidate_id in candidate_ids]
    by_id = {row["id"]: row for row in candidates}
    id179 = by_id[179]
    id335 = by_id[335]
    id179_reviewable = (
        id179["overall"] < id335["overall"]
        and id179["geometry"]["max_delta_vs_178"] < id335["geometry"]["max_delta_vs_178"]
        and id179["geometry"]["projection_failed"] <= id335["geometry"]["projection_failed"] + 0.001
    )
    reason = (
        "ID179 is locally stronger and much closer to ID178 than ID335, so it remains the cleanest same-family manual-review probe. However, it is still non-strict and OOF-selected; it is not Public 0.58-plausible and should not be submitted without explicit approval."
        if id179_reviewable
        else "ID179 does not clearly dominate ID335 under the post-ID335 review gate. Keep ID335 as the latest prepared reference and do not promote ID179."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "id179_vs_id335_sequence_probe_review",
        "candidates": candidates,
        "decision": {
            "id179_manual_reviewable": id179_reviewable,
            "public_058_plausible": False,
            "promote_generated_unsubmitted_artifact": False,
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
        default=REPORTS_DIR / "id179_vs_id335_sequence_probe_review_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
