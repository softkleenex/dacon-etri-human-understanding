#!/usr/bin/env python3
"""Promote a public-structure feature-similarity overlay from a grid report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    load_base_artifact,
    make_interleaved_subject_folds,
)
from scripts.run_strict_feature_similarity_anchor_overlay_experiment import (  # noqa: E402
    blocked_feature_similarity_candidate_oof,
    candidate_test_prediction,
)
from scripts.run_strict_fixed_feature_similarity_overlay_experiment import feature_family_tokens  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    numeric_feature_columns,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 218


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def build_feature_families(train: pd.DataFrame, test: pd.DataFrame, env_family: str = "env24") -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(train, test)
    families = {"all": numeric_cols}
    for family, tokens in feature_family_tokens(env_family).items():
        cols = [col for col in numeric_cols if any(token in col.lower() for token in tokens)]
        if cols:
            families[family] = cols
    return families


def scale_selected_families(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_configs: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    families = build_feature_families(train, test)
    needed_families = sorted({str(config["family"]) for config in selected_configs.values()})
    missing = sorted(set(needed_families) - set(families))
    if missing:
        raise ValueError(f"Unknown feature families required by report: {missing}")
    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    family_sizes: dict[str, int] = {}
    for family in needed_families:
        cols = families[family]
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        combined = pd.concat([train[cols], test[cols]], ignore_index=True)
        values = scaler.fit_transform(imputer.fit_transform(combined))
        scaled[family] = (values[: len(train)], values[len(train) :])
        family_sizes[family] = len(cols)
    return scaled, family_sizes


def selected_configs_from_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        target_report = report["targets"][target]
        best = target_report["best"]
        if float(best["loss"]) < float(target_report["base_loss"]):
            config = best["config"].copy()
            config["name"] = best["name"]
            configs[target] = config
        else:
            configs[target] = {"kind": "base", "name": "base"}
    return configs


def predict_from_report(
    base_id: int,
    diagnostic_report: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    diagnostic_report = diagnostic_report.resolve()
    report = json.loads(diagnostic_report.read_text(encoding="utf-8"))
    if int(report["base_id"]) != base_id:
        raise ValueError(f"Report base_id={report['base_id']} does not match requested base_id={base_id}.")
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)
    selected_configs = selected_configs_from_report(report)
    active_configs = {target: config for target, config in selected_configs.items() if config.get("kind") != "base"}
    scaled, family_sizes = scale_selected_families(train_raw, test_raw, active_configs) if active_configs else ({}, {})
    fold_ids = make_interleaved_subject_folds(train)
    folds = [(np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]) for fold in sorted(np.unique(fold_ids))]

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        y = train_raw[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        config = selected_configs[target]
        if config.get("kind") == "base":
            selected_oof = base_values
            selected_test = base_test_values
            fold_details: list[dict[str, Any]] = []
        else:
            selected_oof, fold_details = blocked_feature_similarity_candidate_oof(
                train=train,
                test=test,
                scaled=scaled,
                y=y,
                base_values=base_values,
                folds=folds,
                config=config,
            )
            selected_test = candidate_test_prediction(
                train=train,
                test=test,
                scaled=scaled,
                y=y,
                base_test_values=base_test_values,
                config=config,
            )
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "improvement_vs_base": float(log_loss(y, base_values, labels=[0, 1]) - log_loss(y, selected_oof, labels=[0, 1])),
            "fold_details": fold_details,
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
        }
    target_scores = {
        target: float(log_loss(train_raw[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "diagnostic_report": str(relpath(diagnostic_report)),
        "diagnostic_kind": report.get("diagnostic"),
        "validation_lens": report.get("validation_lens"),
        "selected_configs": selected_configs,
        "modified_targets": [target for target, config in selected_configs.items() if config.get("kind") != "base"],
        "unchanged_targets": [target for target, config in selected_configs.items() if config.get("kind") == "base"],
        "family_sizes": family_sizes,
        "grid": report.get("grid", {}),
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are blended target-wise with same-subject feature-similarity "
            "anchors selected from the listed public-structure grid report."
        ),
        "caution": "Public-structure local full-grid selection; not nested and not public proof.",
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
    diagnostic_report: Path,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: public feature-similarity overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses same-subject feature-neighbor labels blended into the base predictions."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_feature_similarity_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--diagnostic-report {relpath(diagnostic_report)}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id218_public_feature_similarity_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_from_report(
        args.base_id, args.diagnostic_report
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
        "method": f"Target-wise public feature-similarity overlay on ID{args.base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_feature_similarity_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise same-subject feature-similarity overlay under public-structure interleaved CV.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id, args.diagnostic_report
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Public-structure feature-similarity overlay on ID{args.base_id:03d}; target-wise configs selected "
            f"from {relpath(args.diagnostic_report)}. Not strict/nested; no DACON submission was made."
        ),
        model_description=f"public feature-similarity overlay on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
