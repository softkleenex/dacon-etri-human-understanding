#!/usr/bin/env python3
"""Scan/promote delta-capped bridges from a confirmed public anchor.

The current local frontier is much lower than the best confirmed public result,
but the ID178 probe showed a large local-to-public gap.  This script treats
ID178 as a conservative public-known anchor and moves toward stronger local
frontiers only under target-wise test-delta caps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_ALPHAS = [0.03, 0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
DEFAULT_CAPS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def blend_values(
    anchor_values: np.ndarray | pd.Series,
    frontier_values: np.ndarray | pd.Series,
    alpha: float,
    method: str,
) -> np.ndarray:
    anchor = np.clip(np.asarray(anchor_values, dtype=float), EPS, 1.0 - EPS)
    frontier = np.clip(np.asarray(frontier_values, dtype=float), EPS, 1.0 - EPS)
    if method == "prob":
        blended = (1.0 - alpha) * anchor + alpha * frontier
    elif method == "logit":
        blended = sigmoid((1.0 - alpha) * logit(anchor) + alpha * logit(frontier))
    else:
        raise ValueError(f"Unsupported method: {method}")
    return np.clip(blended, EPS, 1.0 - EPS)


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def scan_anchor_bridge(
    anchor_id: int,
    frontier_ids: list[int],
    caps: list[float],
    alphas: list[float],
) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    anchor_oof, anchor_test, anchor_source = load_artifact_pair(train, test, anchor_id)
    artifacts = {frontier_id: load_artifact_pair(train, test, frontier_id) for frontier_id in frontier_ids}
    rows: list[dict[str, Any]] = []
    target_candidate_pool: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}

    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        anchor_train = np.clip(anchor_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        anchor_holdout = np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        target_candidate_pool[target].append(
            {
                "kind": "anchor",
                "frontier_id": anchor_id,
                "method": "anchor",
                "alpha": 0.0,
                "loss": float(log_loss(y, anchor_train, labels=[0, 1])),
                "test_max_abs_delta_vs_anchor": 0.0,
                "test_mean": float(np.mean(anchor_holdout)),
            }
        )
        for frontier_id, (frontier_oof, frontier_test, frontier_source) in artifacts.items():
            frontier_train = np.clip(frontier_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            frontier_holdout = np.clip(frontier_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            for method in ["prob", "logit"]:
                for alpha in alphas:
                    blended_train = blend_values(anchor_train, frontier_train, alpha=alpha, method=method)
                    blended_test = blend_values(anchor_holdout, frontier_holdout, alpha=alpha, method=method)
                    target_candidate_pool[target].append(
                        {
                            "kind": "bridge",
                            "frontier_id": frontier_id,
                            "method": method,
                            "alpha": alpha,
                            "loss": float(log_loss(y, blended_train, labels=[0, 1])),
                            "test_max_abs_delta_vs_anchor": float(np.max(np.abs(blended_test - anchor_holdout))),
                            "test_mean": float(np.mean(blended_test)),
                            "frontier_source": frontier_source,
                        }
                    )

    for cap in caps:
        choices: dict[str, Any] = {}
        target_losses: dict[str, float] = {}
        max_delta_all = 0.0
        mean_alpha = 0.0
        n_changed = 0
        for target in TARGETS:
            eligible = [
                row
                for row in target_candidate_pool[target]
                if float(row["test_max_abs_delta_vs_anchor"]) <= cap + 1e-12
            ]
            if not eligible:
                raise ValueError(f"No eligible candidates for {target} at cap={cap}.")
            best = min(eligible, key=lambda row: float(row["loss"]))
            choices[target] = best
            target_losses[target] = float(best["loss"])
            max_delta_all = max(max_delta_all, float(best["test_max_abs_delta_vs_anchor"]))
            if best["kind"] != "anchor":
                mean_alpha += float(best["alpha"])
                n_changed += 1
        rows.append(
            {
                "cap": cap,
                "overall": float(np.mean(list(target_losses.values()))),
                "target_losses": target_losses,
                "max_delta_all": max_delta_all,
                "n_changed_targets": n_changed,
                "mean_alpha_changed": float(mean_alpha / n_changed) if n_changed else 0.0,
                "choices": choices,
            }
        )

    best_by_cap = sorted(rows, key=lambda row: float(row["overall"]))
    return {
        "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
        "diagnostic": "confirmed_public_anchor_delta_capped_bridge",
        "anchor_id": anchor_id,
        "anchor_source": anchor_source,
        "anchor_public_score": 0.6014113009 if anchor_id == 178 else None,
        "frontier_ids": frontier_ids,
        "caps": caps,
        "alphas": alphas,
        "validation_lens": "full_oof_targetwise_bridge_scan_not_nested",
        "caution": (
            "This scan uses full OOF labels and is not fold-nested. The delta cap controls movement from a "
            "confirmed public anchor but does not prove leaderboard improvement."
        ),
        "target_candidate_pool_top5": {
            target: sorted(pool, key=lambda row: float(row["loss"]))[:5]
            for target, pool in target_candidate_pool.items()
        },
        "cap_rows": rows,
        "best_by_cap": best_by_cap,
    }


def predict_from_scan(
    scan: dict[str, Any],
    cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    anchor_id = int(scan["anchor_id"])
    anchor_oof, anchor_test, anchor_source = load_artifact_pair(train, test, anchor_id)
    artifact_ids = sorted(
        {int(choice["frontier_id"]) for row in scan["cap_rows"] for choice in row["choices"].values()}
    )
    artifacts = {
        artifact_id: load_artifact_pair(train, test, artifact_id)
        for artifact_id in artifact_ids
        if artifact_id != anchor_id
    }
    matches = [row for row in scan["cap_rows"] if abs(float(row["cap"]) - cap) < 1e-12]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cap row for cap={cap}, found {len(matches)}.")
    row = matches[0]
    oof = anchor_oof.copy()
    test_pred = anchor_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        choice = row["choices"][target]
        frontier_id = int(choice["frontier_id"])
        if choice["kind"] == "anchor" or frontier_id == anchor_id:
            selected_oof = np.clip(anchor_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            selected_test = np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            frontier_source = anchor_source
        else:
            frontier_oof, frontier_test, frontier_source = artifacts[frontier_id]
            selected_oof = blend_values(
                anchor_oof[target],
                frontier_oof[target],
                alpha=float(choice["alpha"]),
                method=str(choice["method"]),
            )
            selected_test = blend_values(
                anchor_test[target],
                frontier_test[target],
                alpha=float(choice["alpha"]),
                method=str(choice["method"]),
            )
        oof[target] = selected_oof
        test_pred[target] = selected_test
        y = train[target].astype(int).to_numpy()
        target_details[target] = {
            "choice": choice,
            "anchor_source": anchor_source,
            "frontier_source": frontier_source,
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "test_mean": float(np.mean(selected_test)),
            "test_max_abs_delta_vs_anchor": float(
                np.max(np.abs(selected_test - np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)))
            ),
        }
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "scan_diagnostic": scan["diagnostic"],
        "anchor_id": anchor_id,
        "anchor_public_score": scan.get("anchor_public_score"),
        "frontier_ids": scan["frontier_ids"],
        "cap": cap,
        "scan_overall": float(row["overall"]),
        "scan_max_delta_all": float(row["max_delta_all"]),
        "choices": row["choices"],
        "target_details": target_details,
        "selection_contract": (
            f"Move target-wise from confirmed public anchor ID{anchor_id:03d} toward stronger local artifacts "
            f"while enforcing max test delta <= {cap:.3f} versus the anchor."
        ),
        "caution": scan["caution"],
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    scan_path: Path,
    cap: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: confirmed public anchor bridge\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Scan report: `{relpath(scan_path)}`\n"
            f"- Cap versus anchor: `{cap:.3f}`\n"
            f"- Full-OOF local CV: `{local_cv:.6f}`\n\n"
            "This artifact moves from a confirmed public anchor toward stronger local candidates under a "
            "target-wise test-delta cap. It is not fold-nested and no DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_confirmed_anchor_bridge_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --scan-path {relpath(scan_path)} --cap {cap}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\nReview the metadata and candidate risk before any leaderboard-facing submission."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def write_artifacts(
    experiment_id: int,
    slug: str,
    scan_path: Path,
    cap: float,
) -> dict[str, Any]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    oof, test_pred, target_scores, selector_info, sample_original = predict_from_scan(scan=scan, cap=cap)
    local_cv = float(np.mean(list(target_scores.values())))
    status = "unsubmitted"
    submission_path = SUBMISSIONS_DIR / f"submission_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_path = NOTEBOOK_DIR / f"submission_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_path = REPORTS_DIR / f"experiment_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_path = BASE_PREDICTION_DIR / f"base_oof_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    base_test_path = BASE_PREDICTION_DIR / f"base_test_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {relpath(path)}")

    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)
    metadata = {
        "experiment_id": experiment_id,
        "slug": slug,
        "method": "Confirmed-public-anchor target-wise delta-capped bridge",
        "submission_file": submission_path.name,
        "notebook_file": relpath(notebook_path),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "confirmed_public_anchor_delta_capped_bridge_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": relpath(base_oof_path),
        "base_test_file": relpath(base_test_path),
        "validation": (
            f"Target-wise bridge selected from {relpath(scan_path)} at cap={cap:.3f}. "
            "This is a public-anchor risk-control probe, not strict proof."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, scan_path, cap)
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Confirmed-public-anchor bridge from scan {relpath(scan_path)} with cap={cap:.3f}; "
            f"local CV {local_cv:.6f}. Not strict Subject-Hole CV, not fold-nested; no DACON submission was made."
        ),
        model_description="Confirmed-public-anchor delta-capped bridge probe",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-id", type=int, default=178)
    parser.add_argument("--frontier-ids", default="247,248,252")
    parser.add_argument("--caps", default=None)
    parser.add_argument("--alphas", default=None)
    parser.add_argument(
        "--output", type=Path, default=REPORTS_DIR / "id178_confirmed_public_anchor_bridge_scan_20260616.json"
    )
    parser.add_argument("--scan-path", type=Path, default=None)
    parser.add_argument("--cap", type=float, default=None)
    parser.add_argument("--experiment-id", type=int, default=None)
    parser.add_argument("--slug", default="id178_confirmed_anchor_bridge")
    args = parser.parse_args()

    if args.experiment_id is not None:
        if args.scan_path is None or args.cap is None:
            raise ValueError("--experiment-id promotion requires --scan-path and --cap.")
        metadata = write_artifacts(
            experiment_id=args.experiment_id,
            slug=args.slug,
            scan_path=args.scan_path if args.scan_path.is_absolute() else ROOT / args.scan_path,
            cap=args.cap,
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    scan = scan_anchor_bridge(
        anchor_id=args.anchor_id,
        frontier_ids=parse_int_list(args.frontier_ids),
        caps=parse_float_list(args.caps, DEFAULT_CAPS),
        alphas=parse_float_list(args.alphas, DEFAULT_ALPHAS),
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": relpath(output), "best_by_cap": scan["best_by_cap"][:5]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
