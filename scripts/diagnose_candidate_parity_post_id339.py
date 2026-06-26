#!/usr/bin/env python3
"""Report-only parity check for ID337 and ID339 candidate construction.

This script recomputes the current review candidates and compares the rebuilt
OOF/test predictions against the immutable base prediction artifacts already in
the workspace. It creates only JSON/MD diagnostics.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.promote_id178_q123_seqsum_s080_id337_s1_candidate import build_candidate as build_id339  # noqa: E402
from scripts.promote_id178_s1_seed45502_candidate import build_s1_overlay as build_id337  # noqa: E402
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
EPS = 1e-12
REVIEW_MAX_ABS_DELTA = 1e-6
REVIEW_LOCAL_CV_DELTA = 1e-6
REVIEW_MIN_CORRELATION = 0.999999
CROSS_PLATFORM_MAX_ABS_DELTA = 3e-5
CROSS_PLATFORM_LOCAL_CV_DELTA = 1e-5
CROSS_PLATFORM_MIN_CORRELATION = 0.999999


def version_info() -> dict[str, str]:
    modules = {}
    for name in ["numpy", "pandas", "sklearn", "scipy"]:
        try:
            module = __import__(name)
            modules[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            modules[name] = f"unavailable: {exc!r}"
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        **modules,
    }


def flatten_targets(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGETS].astype(float).to_numpy().reshape(-1)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 0 or np.std(b) <= 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare_frames(rebuilt: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    rebuilt_values = rebuilt[TARGETS].astype(float).to_numpy()
    reference_values = reference[TARGETS].astype(float).to_numpy()
    delta = rebuilt_values - reference_values
    flat_rebuilt = rebuilt_values.reshape(-1)
    flat_reference = reference_values.reshape(-1)
    return {
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "rmse_delta": float(np.sqrt(np.mean(delta**2))),
        "correlation": corrcoef(flat_rebuilt, flat_reference),
        "max_abs_delta_by_target": {
            target: float(np.max(np.abs(rebuilt[target].astype(float) - reference[target].astype(float))))
            for target in TARGETS
        },
    }


def parity_status(oof_cmp: dict[str, Any], test_cmp: dict[str, Any], local_cv_delta: float) -> str:
    max_delta = max(oof_cmp["max_abs_delta"], test_cmp["max_abs_delta"])
    min_corr = min(oof_cmp["correlation"], test_cmp["correlation"])
    if max_delta <= EPS and abs(local_cv_delta) <= EPS:
        return "exact_pass"
    if (
        max_delta <= REVIEW_MAX_ABS_DELTA
        and abs(local_cv_delta) <= REVIEW_LOCAL_CV_DELTA
        and min_corr >= REVIEW_MIN_CORRELATION
    ):
        return "review_pass"
    return "fail"


def cross_platform_status(oof_cmp: dict[str, Any], test_cmp: dict[str, Any], local_cv_delta: float) -> str:
    max_delta = max(oof_cmp["max_abs_delta"], test_cmp["max_abs_delta"])
    min_corr = min(oof_cmp["correlation"], test_cmp["correlation"])
    if (
        max_delta <= CROSS_PLATFORM_MAX_ABS_DELTA
        and abs(local_cv_delta) <= CROSS_PLATFORM_LOCAL_CV_DELTA
        and min_corr >= CROSS_PLATFORM_MIN_CORRELATION
    ):
        return "cross_platform_review_pass"
    return "cross_platform_review_fail"


def evaluate_candidate(
    *,
    candidate_id: int,
    name: str,
    rebuilt_oof: pd.DataFrame,
    rebuilt_test: pd.DataFrame,
    rebuilt_local_cv: float,
    rebuilt_target_scores: dict[str, float],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    reference_oof, reference_test, artifact_source = load_artifact_pair(train, test, candidate_id)
    log_path = ROOT / "submissions" / "submission_log.csv"
    log = pd.read_csv(log_path)
    log_row = log[log["id"].astype(str).eq(str(candidate_id))].iloc[-1].to_dict()
    reference_local_cv = float(pd.to_numeric(log_row["local_cv_score_num"]))

    oof_cmp = compare_frames(rebuilt_oof, reference_oof)
    test_cmp = compare_frames(rebuilt_test, reference_test)
    local_cv_delta = float(rebuilt_local_cv - reference_local_cv)
    status = parity_status(oof_cmp, test_cmp, local_cv_delta)
    platform_status = cross_platform_status(oof_cmp, test_cmp, local_cv_delta)
    return {
        "candidate_id": candidate_id,
        "name": name,
        "submission_file": log_row.get("submission_file", ""),
        "artifact_source": artifact_source,
        "reference_local_cv": reference_local_cv,
        "rebuilt_local_cv": rebuilt_local_cv,
        "local_cv_delta": local_cv_delta,
        "rebuilt_target_scores": rebuilt_target_scores,
        "oof_comparison": oof_cmp,
        "test_comparison": test_cmp,
        "parity_status": status,
        "passes_review_gate": status in {"exact_pass", "review_pass"},
        "cross_platform_status": platform_status,
        "passes_cross_platform_gate": platform_status == "cross_platform_review_pass",
    }


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()

    id337_oof, id337_test, id337_target_scores, _id337_selector, _sample = build_id337(
        base_id=178,
        target="S1",
        n_kernels=512,
        seed=45502,
        model_name="lr_c0_1",
        weight=0.05,
    )
    id337_local_cv = float(np.mean(list(id337_target_scores.values())))

    id339_oof, id339_test, id339_local_cv, id339_target_scores, _id339_selector, _sample = build_id339(
        base_id=178,
        s1_source_id=337,
        scale=0.80,
    )

    candidates = [
        evaluate_candidate(
            candidate_id=337,
            name="ID178 S1 seed45502 random-conv weight0.05 safety",
            rebuilt_oof=id337_oof,
            rebuilt_test=id337_test,
            rebuilt_local_cv=id337_local_cv,
            rebuilt_target_scores=id337_target_scores,
            train=train,
            test=test,
        ),
        evaluate_candidate(
            candidate_id=339,
            name="ID178 Q123 sequence-summary scale0.80 + ID337 S1 safety",
            rebuilt_oof=id339_oof,
            rebuilt_test=id339_test,
            rebuilt_local_cv=id339_local_cv,
            rebuilt_target_scores=id339_target_scores,
            train=train,
            test=test,
        ),
    ]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "candidate_parity_post_id339",
        "purpose": "Recompute ID337/ID339 and compare rebuilt predictions against stored base prediction artifacts.",
        "environment": version_info(),
        "thresholds": {
            "exact_max_abs_delta": EPS,
            "review_max_abs_delta": REVIEW_MAX_ABS_DELTA,
            "review_local_cv_delta": REVIEW_LOCAL_CV_DELTA,
            "review_min_correlation": REVIEW_MIN_CORRELATION,
            "cross_platform_max_abs_delta": CROSS_PLATFORM_MAX_ABS_DELTA,
            "cross_platform_local_cv_delta": CROSS_PLATFORM_LOCAL_CV_DELTA,
            "cross_platform_min_correlation": CROSS_PLATFORM_MIN_CORRELATION,
        },
        "candidates": candidates,
        "decision": {
            "all_candidates_pass_review_gate": all(row["passes_review_gate"] for row in candidates),
            "all_candidates_pass_cross_platform_gate": all(row["passes_cross_platform_gate"] for row in candidates),
            "no_submission_created": True,
            "no_dacon_submission_made": True,
            "next_action": (
                "Use the strict review gate for same-environment rebuilds and the cross-platform gate for Kaggle CPU "
                "or other architecture/package-version parity checks. Passing parity does not imply public-LB gain."
            ),
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate parity check — post ID339",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, no submission-log row, no base prediction artifact, and no DACON submission.",
        "",
        "## Environment",
        "",
    ]
    for key, value in report["environment"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Candidate parity",
            "",
            "| id | status | ref local | rebuilt local | Δ local | OOF max Δ | test max Δ | min corr | file |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["candidates"]:
        min_corr = min(row["oof_comparison"]["correlation"], row["test_comparison"]["correlation"])
        lines.append(
            f"| {row['candidate_id']} | `{row['parity_status']}` / `{row['cross_platform_status']}` | "
            f"{row['reference_local_cv']:.6f} | "
            f"{row['rebuilt_local_cv']:.6f} | {row['local_cv_delta']:+.3e} | "
            f"{row['oof_comparison']['max_abs_delta']:.3e} | {row['test_comparison']['max_abs_delta']:.3e} | "
            f"{min_corr:.9f} | `{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- All candidates pass strict review gate: `{report['decision']['all_candidates_pass_review_gate']}`",
            f"- All candidates pass cross-platform gate: `{report['decision']['all_candidates_pass_cross_platform_gate']}`",
            "- DACON submission: **no**",
            "",
            report["decision"]["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "candidate_parity_post_id339_20260618.json",
    )
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
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
