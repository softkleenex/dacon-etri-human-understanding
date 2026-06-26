#!/usr/bin/env python3
"""Scan/promote recovery probes with an aggregate failed-direction budget.

This is a less conservative sibling of
``run_public_failed_direction_orthogonal_probe_experiment.py``.  Instead of
requiring every target to have near-zero projection onto the ID178→ID287 failed
direction, it allows targets to trade projection under one aggregate budget.
The intent is to push local score lower while still measuring how much the
candidate follows the observed public-failure vector.
"""

from __future__ import annotations

import argparse
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
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

DEFAULT_CAPS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
DEFAULT_LAMBDAS = [
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.12,
    0.16,
    0.22,
    0.30,
    0.45,
    0.65,
    0.90,
    1.20,
    1.60,
    2.20,
    3.00,
    4.50,
    6.50,
    9.00,
    12.00,
]


def compact_choice(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in choice.items() if key not in {"pred_oof", "pred_test"} and not key.startswith("_")
    }


def total_failed_direction_denominator(anchor_test: pd.DataFrame, failed_test: pd.DataFrame) -> float:
    total = 0.0
    for target in TARGETS:
        failed_vector = failed_test[target].astype(float).to_numpy() - anchor_test[target].astype(float).to_numpy()
        total += float(failed_vector @ failed_vector)
    return total if total > 0.0 else 1.0


def scan_failed_direction_aggregate_budget(
    anchor_id: int,
    failed_id: int,
    source_ids: list[int],
    caps: list[float],
    lambda_penalties: list[float],
    scales: list[float],
    gammas: list[float],
    negative_reward_cap: float,
) -> dict[str, Any]:
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
                "pred_oof": anchor_train,
                "pred_test": anchor_holdout,
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
                    delta_summary = summarize_prediction_delta(anchor_holdout, prediction_holdout, failed_holdout)
                    pool.append(
                        {
                            "kind": "failed_direction_aggregate_budget",
                            "source_id": source_id,
                            "scale": scale,
                            "gamma": gamma,
                            "beta_to_failed": beta,
                            "loss": float(log_loss(y, prediction_train, labels=[0, 1])),
                            **delta_summary,
                            "projection_global_contribution": float(delta @ failed_vector / total_denominator),
                            "source": source_name,
                            "pred_oof": prediction_train,
                            "pred_test": prediction_holdout,
                        }
                    )
        target_pools[target] = pool

    rows: list[dict[str, Any]] = []
    per_target_negative_cap = negative_reward_cap / max(len(TARGETS), 1)
    for cap in caps:
        cap_pools = {
            target: [
                candidate
                for candidate in target_pools[target]
                if float(candidate["max_abs_delta_vs_anchor"]) <= cap + 1e-12
            ]
            for target in TARGETS
        }
        for lambda_penalty in lambda_penalties:
            choices = {
                target: min(
                    cap_pools[target],
                    key=lambda candidate: (
                        float(candidate["loss"])
                        + lambda_penalty
                        * max(float(candidate["projection_global_contribution"]), -per_target_negative_cap)
                    ),
                )
                for target in TARGETS
            }
            all_delta = []
            all_failed_vector = []
            for target in TARGETS:
                anchor_holdout = np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
                failed_holdout = np.clip(failed_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
                all_delta.append(choices[target]["pred_test"] - anchor_holdout)
                all_failed_vector.append(failed_holdout - anchor_holdout)
            delta_vector = np.concatenate(all_delta)
            failed_vector = np.concatenate(all_failed_vector)
            denominator = float(failed_vector @ failed_vector)
            aggregate_projection = 0.0 if denominator <= 0.0 else float(delta_vector @ failed_vector / denominator)
            rows.append(
                {
                    "cap": cap,
                    "lambda_penalty": lambda_penalty,
                    "overall": float(np.mean([choices[target]["loss"] for target in TARGETS])),
                    "target_losses": {target: float(choices[target]["loss"]) for target in TARGETS},
                    "max_abs_delta_vs_anchor": float(
                        max(float(choices[target]["max_abs_delta_vs_anchor"]) for target in TARGETS)
                    ),
                    "mean_abs_delta_vs_anchor": float(np.mean(np.abs(delta_vector))),
                    "aggregate_projection_to_failed_direction": aggregate_projection,
                    "n_changed_targets": int(sum(choices[target]["kind"] != "anchor" for target in TARGETS)),
                    "choices": {target: compact_choice(choices[target]) for target in TARGETS},
                }
            )

    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row["cap"],
            tuple(
                (
                    target,
                    row["choices"][target]["source_id"],
                    row["choices"][target]["scale"],
                    row["choices"][target]["gamma"],
                )
                for target in TARGETS
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    return {
        "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
        "diagnostic": "failed_direction_aggregate_budget_scan",
        "anchor_id": anchor_id,
        "failed_id": failed_id,
        "anchor_public_score": 0.6014113009 if anchor_id == 178 else None,
        "failed_public_score": 0.6698787318 if failed_id == 287 else None,
        "anchor_source": anchor_source,
        "failed_source": failed_source,
        "source_ids_requested": source_ids,
        "source_ids_used": usable_source_ids,
        "caps": caps,
        "lambda_penalties": lambda_penalties,
        "scales": scales,
        "gammas": gammas,
        "negative_reward_cap": negative_reward_cap,
        "validation_lens": "full_oof_failed_direction_aggregate_budget_scan_not_nested",
        "caution": (
            "This scan optimizes target choices with an aggregate public-failure-direction budget. It is "
            "non-nested and uses public feedback from ID287 for risk control, so it is a probe, not proof."
        ),
        "rows": unique_rows,
        "best_by_overall": sorted(unique_rows, key=lambda row: float(row["overall"])),
        "best_under_projection_005": sorted(
            [row for row in unique_rows if float(row["aggregate_projection_to_failed_direction"]) <= 0.05],
            key=lambda row: float(row["overall"]),
        ),
        "best_under_projection_002": sorted(
            [row for row in unique_rows if float(row["aggregate_projection_to_failed_direction"]) <= 0.02],
            key=lambda row: float(row["overall"]),
        ),
    }


def write_scan_markdown(report: dict[str, Any], markdown_path: Path) -> None:
    lines = [
        "# Failed-direction aggregate-budget scan",
        "",
        (
            f"Anchor ID{report['anchor_id']} public={report['anchor_public_score']} and failed "
            f"ID{report['failed_id']} public={report['failed_public_score']} define the risk direction. "
            "Rows below trade target-level residual choices under an aggregate projection budget."
        ),
        "",
        "## Best overall rows",
        "",
        "| rank | cap | lambda | local | max Δ anchor | mean Δ anchor | aggregate failed projection | changed targets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["best_by_overall"][:20], start=1):
        lines.append(
            f"| {rank} | {row['cap']:.3f} | {row['lambda_penalty']:.2f} | {row['overall']:.6f} | "
            f"{row['max_abs_delta_vs_anchor']:.6f} | {row['mean_abs_delta_vs_anchor']:.6f} | "
            f"{row['aggregate_projection_to_failed_direction']:+.4f} | {row['n_changed_targets']} |"
        )
    lines.extend(
        [
            "",
            "## Best rows with aggregate projection <= +0.05",
            "",
            "| rank | cap | lambda | local | max Δ anchor | mean Δ anchor | aggregate failed projection | changed targets |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(report["best_under_projection_005"][:20], start=1):
        lines.append(
            f"| {rank} | {row['cap']:.3f} | {row['lambda_penalty']:.2f} | {row['overall']:.6f} | "
            f"{row['max_abs_delta_vs_anchor']:.6f} | {row['mean_abs_delta_vs_anchor']:.6f} | "
            f"{row['aggregate_projection_to_failed_direction']:+.4f} | {row['n_changed_targets']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this scan is intentionally more ambitious than strict orthogonal candidates. "
            "Projection near zero is safer; lower local rows with positive projection are score-pressure probes.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_scan_row(scan: dict[str, Any], cap: float, lambda_penalty: float) -> dict[str, Any]:
    matches = [
        row
        for row in scan["rows"]
        if abs(float(row["cap"]) - cap) < 1e-12 and abs(float(row["lambda_penalty"]) - lambda_penalty) < 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for cap={cap}, lambda={lambda_penalty}; found {len(matches)}.")
    return matches[0]


def predict_from_scan(
    scan: dict[str, Any],
    cap: float,
    lambda_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    anchor_id = int(scan["anchor_id"])
    failed_id = int(scan["failed_id"])
    anchor_oof, anchor_test, anchor_source = load_artifact_pair(train, test, anchor_id)
    failed_oof, failed_test, failed_source = load_artifact_pair(train, test, failed_id)
    row = find_scan_row(scan, cap, lambda_penalty)
    source_ids = sorted(
        {
            int(choice["source_id"])
            for choice in row["choices"].values()
            if int(choice["source_id"]) not in {anchor_id, failed_id}
        }
    )
    artifacts = {source_id: load_artifact_pair(train, test, source_id) for source_id in source_ids}

    oof = anchor_oof.copy()
    test_pred = anchor_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        choice = row["choices"][target]
        source_id = int(choice["source_id"])
        anchor_train = np.clip(anchor_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        anchor_holdout = np.clip(anchor_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        failed_train = np.clip(failed_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        failed_holdout = np.clip(failed_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if choice["kind"] == "anchor" or source_id == anchor_id:
            selected_train = anchor_train
            selected_holdout = anchor_holdout
            source_name = anchor_source
        else:
            source_oof, source_test, source_name = artifacts[source_id]
            selected_train = adjusted_values(
                anchor_train,
                np.clip(source_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS),
                failed_train,
                scale=float(choice["scale"]),
                gamma=float(choice["gamma"]),
                beta=float(choice["beta_to_failed"]),
            )
            selected_holdout = adjusted_values(
                anchor_holdout,
                np.clip(source_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS),
                failed_holdout,
                scale=float(choice["scale"]),
                gamma=float(choice["gamma"]),
                beta=float(choice["beta_to_failed"]),
            )
        oof[target] = selected_train
        test_pred[target] = selected_holdout
        y = train[target].astype(int).to_numpy()
        target_details[target] = {
            "choice": choice,
            "source": source_name,
            "selected_loss": float(log_loss(y, selected_train, labels=[0, 1])),
            **summarize_prediction_delta(anchor_holdout, selected_holdout, failed_holdout),
            "test_prediction_mean": float(np.mean(selected_holdout)),
            "test_prediction_std": float(np.std(selected_holdout)),
            "test_prediction_min": float(np.min(selected_holdout)),
            "test_prediction_max": float(np.max(selected_holdout)),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "scan_path": scan.get("_scan_path"),
        "scan_row": {key: value for key, value in row.items() if key != "choices"},
        "anchor_id": anchor_id,
        "failed_id": failed_id,
        "anchor_source": anchor_source,
        "failed_source": failed_source,
        "source_ids": source_ids,
        "target_details": target_details,
        "selection_contract": (
            f"Select per-target residual artifacts from aggregate failed-direction budget scan at "
            f"cap={cap:.3f}, lambda_penalty={lambda_penalty:.2f}."
        ),
        "caution": (
            "Full-OOF aggregate-budget recombination. Lower local score is not public proof; use the "
            "aggregate projection and movement metrics for risk ranking."
        ),
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
    lambda_penalty: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: aggregate failed-direction budget probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Scan report: `{relpath(scan_path)}`\n"
            f"- Cap vs ID178: `{cap:.3f}`\n"
            f"- Lambda penalty: `{lambda_penalty:.2f}`\n"
            f"- Full-OOF local CV: `{local_cv:.6f}`\n\n"
            "This artifact trades target-level residual choices under an aggregate ID287-failure-direction budget."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_failed_direction_aggregate_budget_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --scan-path {relpath(scan_path)} "
            f"--cap {cap} --lambda-penalty {lambda_penalty}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nNo DACON submission was made."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def promote_from_scan(
    experiment_id: int,
    slug: str,
    scan_path: Path,
    cap: float,
    lambda_penalty: float,
) -> dict[str, Any]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan["_scan_path"] = relpath(scan_path)
    oof, test_pred, target_scores, selector_info, sample_original = predict_from_scan(scan, cap, lambda_penalty)
    local_cv = float(np.mean(list(target_scores.values())))

    status = "unsubmitted"
    filename = f"submission_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name
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
        "method": "Aggregate failed-direction budget public recovery probe",
        "submission_file": submission_path.name,
        "notebook_file": relpath(notebook_path),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "full_oof_failed_direction_aggregate_budget_scan_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": relpath(base_oof_path),
        "base_test_file": relpath(base_test_path),
        "validation": (
            f"Aggregate failed-direction budget scan from {relpath(scan_path)} at cap={cap:.3f}, "
            f"lambda_penalty={lambda_penalty:.2f}. High-risk full-OOF recombination; no DACON submission was made."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path,
        submission_path,
        metadata_path,
        experiment_id,
        slug,
        local_cv,
        scan_path,
        cap,
        lambda_penalty,
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Aggregate failed-direction budget recovery selected from {relpath(scan_path)} at cap={cap:.3f}, "
            f"lambda_penalty={lambda_penalty:.2f}. Full-OOF residual recombination; not strict Subject-Hole CV; "
            "no DACON submission was made."
        ),
        model_description="Aggregate failed-direction budget recovery probe",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-output", type=Path)
    parser.add_argument("--scan-path", type=Path)
    parser.add_argument("--anchor-id", type=int, default=178)
    parser.add_argument("--failed-id", type=int, default=287)
    parser.add_argument("--source-ids", default="220-286,288-300,302-307")
    parser.add_argument("--caps")
    parser.add_argument("--lambda-penalties")
    parser.add_argument("--scales")
    parser.add_argument("--gammas")
    parser.add_argument("--negative-reward-cap", type=float, default=0.02)
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug")
    parser.add_argument("--cap", type=float)
    parser.add_argument("--lambda-penalty", type=float)
    args = parser.parse_args()

    if args.scan_output is not None:
        scan_output = args.scan_output if args.scan_output.is_absolute() else ROOT / args.scan_output
        report = scan_failed_direction_aggregate_budget(
            anchor_id=args.anchor_id,
            failed_id=args.failed_id,
            source_ids=parse_source_ids(args.source_ids),
            caps=parse_float_list(args.caps, DEFAULT_CAPS),
            lambda_penalties=parse_float_list(args.lambda_penalties, DEFAULT_LAMBDAS),
            scales=parse_float_list(args.scales, DEFAULT_SCALES),
            gammas=parse_float_list(args.gammas, DEFAULT_GAMMAS),
            negative_reward_cap=args.negative_reward_cap,
        )
        scan_output.parent.mkdir(parents=True, exist_ok=True)
        scan_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path = scan_output.with_suffix(".md")
        write_scan_markdown(report, markdown_path)
        print(f"Wrote {relpath(scan_output)}")
        print(f"Wrote {relpath(markdown_path)}")

    if args.experiment_id is not None:
        if args.scan_path is None or args.slug is None or args.cap is None or args.lambda_penalty is None:
            raise ValueError("--experiment-id requires --scan-path, --slug, --cap, and --lambda-penalty.")
        scan_path = args.scan_path if args.scan_path.is_absolute() else ROOT / args.scan_path
        metadata = promote_from_scan(
            experiment_id=args.experiment_id,
            slug=args.slug,
            scan_path=scan_path,
            cap=args.cap,
            lambda_penalty=args.lambda_penalty,
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    if args.scan_output is None and args.experiment_id is None:
        raise ValueError("Provide --scan-output and/or --experiment-id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
