#!/usr/bin/env python3
"""Promote a target-wise clean multi-signal overlay.

This script combines several small non-ID172 diagnostics into one artifact
instead of creating a chain of tiny one-target submissions. Each target chooses
between the base prediction and a small set of clean side-signal overlays:

- window-pair sequence features inspired by the external Public 0.5917 notebook;
- compact smartphone/light features;
- one report-only random-conv sequence diagnostic.

The selection is still a public-structure local diagnostic, not public proof.
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

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    EPS,
    MODEL_SPECS,
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
    make_kernels,
    random_conv_features,
    table_to_tensor,
)
from scripts.run_public_smartphone_light_compact_overlay_experiment import (  # noqa: E402
    predict_from_report as predict_smartphone_light_from_report,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
from scripts.run_public_window_pair_sequence_overlay_experiment import (  # noqa: E402
    predict_from_report as predict_window_pair_from_report,
)
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

DEFAULT_BASE_ID = 215
DEFAULT_WINDOW_PAIR_REPORT = REPORTS_DIR / "id215_window_pair_sequence_diagnostic_20260616.json"
DEFAULT_SMARTPHONE_LIGHT_REPORT = REPORTS_DIR / "id215_smartphone_light_compact_diagnostic_20260616.json"
DEFAULT_RANDOM_CONV_REPORT = REPORTS_DIR / "id215_random_conv_sequence_diagnostic_nk384_seed43030_20260616.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def parse_random_conv_best(best_name: str) -> dict[str, Any]:
    if best_name == "base":
        return {"kind": "base"}
    if not best_name.startswith("blend_"):
        raise ValueError(f"Unsupported random-conv candidate: {best_name}")
    body = best_name.removeprefix("blend_")
    model_name, weight_raw = body.rsplit("_", maxsplit=1)
    return {"kind": "blend", "model": model_name, "weight": float(weight_raw)}


def predict_random_conv_from_report(
    base_id: int,
    report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report["base_id"]) != base_id:
        raise ValueError(f"Report base_id={report['base_id']} does not match requested base_id={base_id}.")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    spec_lookup = {spec.name: spec for spec in MODEL_SPECS}

    seed = int(report["seed"])
    n_kernels = int(report["n_kernels"])
    kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
    x_train = random_conv_features(train_tensor, kernels)
    x_test = random_conv_features(test_tensor, kernels)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    selected_configs: dict[str, Any] = {}

    for target in TARGETS:
        config = parse_random_conv_best(str(report["targets"][target]["best"]))
        selected_configs[target] = config
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if config["kind"] == "base":
            selected_oof = base_values
            selected_test = base_test_values
            raw_loss = None
        else:
            model_name = str(config["model"])
            weight = float(config["weight"])
            raw_oof, raw_test = fit_oof_model(x_train, x_test, y, fold_ids, spec_lookup[model_name])
            raw_loss = float(log_loss(y, raw_oof, labels=[0, 1]))
            selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
            selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "raw_sequence_loss": raw_loss,
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "diagnostic_report": str(relpath(report_path)),
        "selected_configs": selected_configs,
        "modified_targets": [target for target, config in selected_configs.items() if config["kind"] != "base"],
        "unchanged_targets": [target for target, config in selected_configs.items() if config["kind"] == "base"],
        "seed": seed,
        "n_kernels": n_kernels,
        "n_sequence_rows": int(len(seq_table)),
        "n_total_channels": int(len(channel_names)),
        "target_details": target_details,
        "caution": "Single-report random-conv side signal; local public-structure evidence only.",
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def predict_multi_signal_overlay(
    base_id: int,
    window_pair_report: Path,
    smartphone_light_report: Path,
    random_conv_report: Path,
    min_improvement: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {
        "base": (base_oof, base_test, {"base_source": base_source})
    }
    window_oof, window_test, _window_scores, window_info, _ = predict_window_pair_from_report(base_id, window_pair_report)
    candidates["window_pair"] = (window_oof, window_test, window_info)
    phone_oof, phone_test, _phone_scores, phone_info, _ = predict_smartphone_light_from_report(
        base_id, smartphone_light_report
    )
    candidates["smartphone_light"] = (phone_oof, phone_test, phone_info)
    random_oof, random_test, _random_scores, random_info, _ = predict_random_conv_from_report(base_id, random_conv_report)
    candidates["random_conv"] = (random_oof, random_test, random_info)

    oof = base_oof.copy()
    test_pred = base_test.copy()
    selected_sources: dict[str, str] = {}
    target_details: dict[str, Any] = {}
    target_scores: dict[str, float] = {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_loss = float(log_loss(y, base_values, labels=[0, 1]))
        losses: dict[str, float] = {}
        for source_name, (source_oof, _source_test, _source_info) in candidates.items():
            values = np.clip(source_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            losses[source_name] = float(log_loss(y, values, labels=[0, 1]))
        best_source = min(losses, key=lambda source: (losses[source], source))
        if base_loss - losses[best_source] < min_improvement:
            best_source = "base"
        selected_oof = np.clip(candidates[best_source][0][target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        selected_test = np.clip(candidates[best_source][1][target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        selected_sources[target] = best_source
        target_scores[target] = float(log_loss(y, selected_oof, labels=[0, 1]))
        target_details[target] = {
            "selected_source": best_source,
            "base_loss": base_loss,
            "selected_loss": target_scores[target],
            "candidate_losses": losses,
            "improvement_vs_base": float(base_loss - target_scores[target]),
            "test_prediction_mean_before": float(np.mean(base_test[target].astype(float))),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(
                np.max(np.abs(selected_test - base_test[target].astype(float).to_numpy()))
            ),
        }

    selector_info = {
        "base_source": base_source,
        "supporting_reports": {
            "window_pair": str(relpath(window_pair_report)),
            "smartphone_light": str(relpath(smartphone_light_report)),
            "random_conv": str(relpath(random_conv_report)),
        },
        "selected_sources": selected_sources,
        "modified_targets": [target for target, source in selected_sources.items() if source != "base"],
        "unchanged_targets": [target for target, source in selected_sources.items() if source == "base"],
        "source_selector_info": {
            "window_pair": window_info,
            "smartphone_light": phone_info,
            "random_conv": random_info,
        },
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved unless one clean side-signal candidate improves that target "
            "under the local public-structure diagnostic by at least min_improvement."
        ),
        "min_improvement": min_improvement,
        "caution": "Composite local public-structure clean overlay; no ID172 calibration source, but still not public proof.",
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
    window_pair_report: Path,
    smartphone_light_report: Path,
    random_conv_report: Path,
    min_improvement: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: clean multi-signal overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact combines small clean side signals target-wise instead of chaining tiny overlays."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_multi_signal_clean_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--window-pair-report {relpath(window_pair_report)} "
            f"--smartphone-light-report {relpath(smartphone_light_report)} "
            f"--random-conv-report {relpath(random_conv_report)} "
            f"--min-improvement {min_improvement:g}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="id215_multi_signal_clean_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--window-pair-report", type=Path, default=DEFAULT_WINDOW_PAIR_REPORT)
    parser.add_argument("--smartphone-light-report", type=Path, default=DEFAULT_SMARTPHONE_LIGHT_REPORT)
    parser.add_argument("--random-conv-report", type=Path, default=DEFAULT_RANDOM_CONV_REPORT)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    args = parser.parse_args()
    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_multi_signal_overlay(
        base_id=args.base_id,
        window_pair_report=args.window_pair_report,
        smartphone_light_report=args.smartphone_light_report,
        random_conv_report=args.random_conv_report,
        min_improvement=args.min_improvement,
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
        "method": f"Target-wise clean multi-signal overlay on ID{args.base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_multi_signal_clean_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise clean side-signal selection under public-structure interleaved CV.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path,
        submission_path,
        metadata_path,
        args.experiment_id,
        args.slug,
        local_cv,
        args.base_id,
        args.window_pair_report,
        args.smartphone_light_report,
        args.random_conv_report,
        args.min_improvement,
    )

    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Public-structure clean multi-signal overlay on ID{args.base_id:03d}; combines window-pair, "
            "smartphone/light, and random-conv diagnostics target-wise. Not strict Subject-Hole CV; "
            "no DACON submission was made."
        ),
        model_description=f"clean multi-signal overlay on ID{args.base_id:03d}",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
