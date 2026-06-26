#!/usr/bin/env python3
"""Promote target-wise pair probability/logit blend probes from a diagnostic report."""

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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def blend_values(a_values: np.ndarray | pd.Series, b_values: np.ndarray | pd.Series, weight_b: float, kind: str) -> np.ndarray:
    a = np.clip(np.asarray(a_values, dtype=float), EPS, 1.0 - EPS)
    b = np.clip(np.asarray(b_values, dtype=float), EPS, 1.0 - EPS)
    if kind in {"single", "base"}:
        return a
    if kind == "prob_pair":
        return np.clip((1.0 - weight_b) * a + weight_b * b, EPS, 1.0 - EPS)
    if kind == "logit_pair":
        return np.clip(sigmoid((1.0 - weight_b) * logit(a) + weight_b * logit(b)), EPS, 1.0 - EPS)
    raise ValueError(f"Unknown blend kind: {kind}")


def selected_configs_from_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        if target not in report["targets"]:
            raise ValueError(f"Diagnostic report missing target: {target}")
        configs[target] = report["targets"][target]["best"]
    return configs


def predict_pair_logit_blend_probe(
    report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    configs = selected_configs_from_report(report)
    artifact_ids = sorted(
        {
            int(config["id_a"])
            for config in configs.values()
        }
        | {
            int(config["id_b"])
            for config in configs.values()
        }
    )
    artifacts = {artifact_id: load_artifact_pair(train, test, artifact_id) for artifact_id in artifact_ids}
    oof = train[["subject_id", "date"]].copy().reset_index(drop=True)
    test_pred = test[["subject_id", "date"]].copy().reset_index(drop=True)
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        config = configs[target]
        kind = str(config["kind"])
        id_a = int(config["id_a"])
        id_b = int(config["id_b"])
        weight_b = float(config.get("w_b", 0.0))
        a_oof, a_test, a_source = artifacts[id_a]
        b_oof, b_test, b_source = artifacts[id_b]
        selected_oof = blend_values(a_oof[target], b_oof[target], weight_b, kind)
        selected_test = blend_values(a_test[target], b_test[target], weight_b, kind)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        y = train[target].astype(int).to_numpy()
        target_details[target] = {
            "config": config,
            "source_a": a_source,
            "source_b": b_source,
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "test_prediction_mean": float(np.mean(selected_test)),
            "test_prediction_std": float(np.std(selected_test)),
            "test_prediction_min": float(np.min(selected_test)),
            "test_prediction_max": float(np.max(selected_test)),
        }
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "diagnostic_report": str(relpath(report_path)),
        "artifact_ids": artifact_ids,
        "selected_configs": configs,
        "target_details": target_details,
        "validation_lens": "public_structure_full_oof_targetwise_pair_prob_logit_blend_oracle",
        "selection_contract": (
            "Target-wise pair blend configs are selected from a full-OOF diagnostic report, then applied to "
            "the matching OOF/test artifacts."
        ),
        "caution": (
            "High-risk OOF-selected artifact recombination probe. This is not strict Subject-Hole CV, "
            "not nested, and not public proof."
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
    diagnostic_report: Path,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: pair logit blend probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Full-OOF local CV: `{local_cv:.6f}`\n\n"
            "This artifact recombines existing artifacts target-wise using probability/logit pair blends."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_pair_logit_blend_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --diagnostic-report {relpath(diagnostic_report)}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nNo DACON submission was made."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="targetwise_pair_logit_blend_probe")
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    diagnostic_report = args.diagnostic_report.resolve()
    oof, test_pred, target_scores, selector_info, sample_original = predict_pair_logit_blend_probe(diagnostic_report)
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
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "Target-wise full-OOF pair probability/logit blend probe",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_full_oof_targetwise_pair_prob_logit_blend_oracle",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": (
            f"Target-wise pair probability/logit blend selected from {relpath(diagnostic_report)}. "
            "High-risk, non-strict, and not nested."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, diagnostic_report)
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Target-wise pair probability/logit blend selected from {relpath(diagnostic_report)}. "
            "High-risk full-OOF recombination; not strict Subject-Hole CV; no DACON submission was made."
        ),
        model_description="Target-wise pair logit blend probe",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
