#!/usr/bin/env python3
"""Promote a target-wise delta-capped bridge from an existing scan report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def blend_predictions(
    base_values: pd.Series | np.ndarray,
    extreme_values: pd.Series | np.ndarray,
    alpha: float,
    method: str,
) -> np.ndarray:
    base_arr = np.clip(np.asarray(base_values, dtype=float), EPS, 1.0 - EPS)
    extreme_arr = np.clip(np.asarray(extreme_values, dtype=float), EPS, 1.0 - EPS)
    if method == "prob":
        blended = (1.0 - alpha) * base_arr + alpha * extreme_arr
    elif method == "logit":
        blended = sigmoid((1.0 - alpha) * logit(base_arr) + alpha * logit(extreme_arr))
    else:
        raise ValueError(f"Unsupported blend method: {method}")
    return np.clip(blended, EPS, 1.0 - EPS)


def load_artifact_pair(train: pd.DataFrame, test: pd.DataFrame, artifact_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{artifact_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{artifact_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{artifact_id:03d} base OOF/test artifact.")
    return (
        align_candidate(pd.read_csv(oof_paths[0]), train),
        align_candidate(pd.read_csv(test_paths[0]), test),
        f"{oof_paths[0].name}; {test_paths[0].name}",
    )


def load_scan_row(scan_path: Path, cap: float) -> dict:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    rows = scan["targetwise_cap_rows"]
    matches = [row for row in rows if abs(float(row["cap"]) - cap) < 1e-12]
    if len(matches) != 1:
        raise ValueError(f"Expected one scan row for cap={cap}, found {len(matches)}.")
    return matches[0]


def predict_targetwise_bridge(
    scan_path: Path,
    cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    scan_row = load_scan_row(scan_path, cap)
    base_id = int(json.loads(scan_path.read_text(encoding="utf-8"))["base_id"])
    choices = scan_row["choices"]

    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    artifact_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame, str]] = {base_id: (base_oof, base_test, base_source)}

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}

    for target in TARGETS:
        choice = choices[target]
        extreme_id = int(choice["extreme_id"])
        method = str(choice["method"])
        alpha = float(choice["alpha"])
        if extreme_id not in artifact_cache:
            artifact_cache[extreme_id] = load_artifact_pair(train, test, extreme_id)
        extreme_oof, extreme_test, extreme_source = artifact_cache[extreme_id]

        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        extreme_values = np.clip(extreme_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        extreme_test_values = np.clip(extreme_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)

        blended_oof = blend_predictions(base_values, extreme_values, alpha=alpha, method=method)
        blended_test = blend_predictions(base_test_values, extreme_test_values, alpha=alpha, method=method)
        oof[target] = blended_oof
        test_pred[target] = blended_test

        target_details[target] = {
            "base_id": base_id,
            "extreme_id": extreme_id,
            "method": method,
            "alpha_extreme": alpha,
            "base_source": base_source,
            "extreme_source": extreme_source,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "extreme_loss": float(log_loss(y, extreme_values, labels=[0, 1])),
            "bridge_loss": float(log_loss(y, blended_oof, labels=[0, 1])),
            "test_prediction_mean_base": float(np.mean(base_test_values)),
            "test_prediction_mean_extreme": float(np.mean(extreme_test_values)),
            "test_prediction_mean_bridge": float(np.mean(blended_test)),
            "test_prediction_max_abs_delta_vs_base": float(np.max(np.abs(blended_test - base_test_values))),
            "scan_loss": float(choice["loss"]),
            "scan_max_delta": float(choice["max_delta"]),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "scan_path": str(scan_path.relative_to(ROOT)),
        "scan_cap": cap,
        "scan_overall": float(scan_row["overall"]),
        "scan_max_delta_all": float(scan_row["max_delta_all"]),
        "base_id": base_id,
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "target_details": target_details,
        "selection_contract": (
            f"For each target, select the best prob/logit bridge from {scan_path.name} subject to "
            f"max test delta <= {cap:.3f} versus ID{base_id:03d}. This is full-OOF target-wise selection, "
            "not fold-nested."
        ),
        "caution": (
            "Target-wise delta-capped leaderboard calibration probe. The delta cap controls movement versus "
            "the ID235 anchor, but selection still uses full OOF labels and is not public proof."
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
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: target-wise delta-capped bridge probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Scan report: `{scan_path.relative_to(ROOT)}`\n"
            f"- Delta cap versus scan base: `{cap:.3f}`\n"
            f"- Full-OOF public-structure probe CV: `{local_cv:.6f}`\n\n"
            "This artifact selects target-wise prob/logit bridge settings from a fixed scan report. "
            "It is non-strict and not fold-nested."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_targetwise_delta_capped_bridge_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --scan-path {scan_path} --cap {cap}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before explicitly accepting the target-wise full-OOF selection risk."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--cap", type=float, required=True)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    scan_path = args.scan_path if args.scan_path.is_absolute() else ROOT / args.scan_path
    oof, test_pred, target_scores, selector_info, sample_original = predict_targetwise_bridge(
        scan_path=scan_path,
        cap=args.cap,
    )
    local_cv = float(np.mean(list(target_scores.values())))

    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name
    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

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
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "Target-wise delta-capped prob/logit bridge public probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_targetwise_delta_capped_bridge_not_nested",
        "strict_subject_hole_comparable": False,
        "fold_nested_calibration": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            f"Target-wise bridge selected from {scan_path.relative_to(ROOT)} with cap={args.cap:.3f}. "
            "This is non-strict, not fold-nested, and should only be used as a leaderboard calibration probe."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        scan_path=scan_path,
        cap=args.cap,
    )

    notes = (
        f"Target-wise delta-capped prob/logit bridge selected from {scan_path.relative_to(ROOT)} "
        f"with cap={args.cap:.3f}; local CV {local_cv:.6f}. Not strict Subject-Hole CV, not fold-nested; "
        "no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description="Target-wise delta-capped prob/logit bridge public probe",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
