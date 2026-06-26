#!/usr/bin/env python3
"""Diagnose/promote recurrence/GAF sequence-image summary overlays.

This deterministic representation summarizes each aligned 44-bin daily sensor
sequence as recurrence-rate and Gramian Angular Field style image statistics.
It is intentionally different from the random-convolution, summary-statistic,
and SAX/PAA branches while keeping the local memory footprint small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    N_BINS,
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    table_to_tensor,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
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
DEFAULT_BASE_ID = 200
BLEND_WEIGHTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
RECURRENCE_EPS = [0.35, 0.70, 1.20]
GAF_LAGS = [1, 2, 4, 8]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_depth: int | None = None
    max_leaf_nodes: int | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_1", "lr", c_value=0.1),
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
    ModelSpec("et_d4", "et", max_depth=4),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", random_state=43, max_iter=300),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.035,
                max_leaf_nodes=int(spec.max_leaf_nodes),
                l2_regularization=2.0,
                min_samples_leaf=16,
                random_state=43,
            ),
        )
    if spec.kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=int(spec.max_depth),
                min_samples_leaf=12,
                max_features=0.55,
                random_state=43,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def recurrence_gaf_features(tensor: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(tensor, dtype=np.float32)
    n_rows, n_channels, n_bins = values.shape
    if n_bins != N_BINS:
        raise ValueError(f"Expected {N_BINS} bins, got {n_bins}.")

    # Tensor values are already robust-scaled by table_to_tensor. Tanh keeps the
    # GAF projection bounded without using any labels or test-only transform.
    bounded = np.tanh(values / 3.0).astype(np.float32)
    complement = np.sqrt(np.clip(1.0 - bounded**2, 0.0, 1.0)).astype(np.float32)
    diff_abs = np.abs(values[:, :, :, None] - values[:, :, None, :])

    blocks: list[np.ndarray] = []
    names: list[str] = []

    def add_channel_block(block: np.ndarray, suffix: str) -> None:
        blocks.append(np.asarray(block, dtype=np.float32).reshape(n_rows, n_channels))
        names.extend(f"{name}__{suffix}" for name in channel_names)

    for eps in RECURRENCE_EPS:
        recurrence = diff_abs <= eps
        add_channel_block(recurrence.mean(axis=(2, 3)), f"recurrence_rate_eps{eps:g}")
        early = recurrence[:, :, : n_bins // 2, : n_bins // 2].mean(axis=(2, 3))
        late = recurrence[:, :, n_bins // 2 :, n_bins // 2 :].mean(axis=(2, 3))
        cross = recurrence[:, :, : n_bins // 2, n_bins // 2 :].mean(axis=(2, 3))
        add_channel_block(early, f"recurrence_early_eps{eps:g}")
        add_channel_block(late, f"recurrence_late_eps{eps:g}")
        add_channel_block(cross, f"recurrence_cross_eps{eps:g}")
        add_channel_block(late - early, f"recurrence_late_minus_early_eps{eps:g}")
        for lag in GAF_LAGS:
            lag_rec = np.abs(values[:, :, :-lag] - values[:, :, lag:]) <= eps
            add_channel_block(lag_rec.mean(axis=2), f"recurrence_lag{lag}_eps{eps:g}")

    gasf = bounded[:, :, :, None] * bounded[:, :, None, :] - complement[:, :, :, None] * complement[:, :, None, :]
    gadf = complement[:, :, :, None] * bounded[:, :, None, :] - bounded[:, :, :, None] * complement[:, :, None, :]
    for matrix_name, matrix in [("gasf", gasf), ("gadf", gadf)]:
        add_channel_block(matrix.mean(axis=(2, 3)), f"{matrix_name}_mean")
        add_channel_block(matrix.std(axis=(2, 3)), f"{matrix_name}_std")
        add_channel_block(np.diagonal(matrix, axis1=2, axis2=3).mean(axis=2), f"{matrix_name}_diag_mean")
        add_channel_block(matrix[:, :, : n_bins // 2, : n_bins // 2].mean(axis=(2, 3)), f"{matrix_name}_early_mean")
        add_channel_block(matrix[:, :, n_bins // 2 :, n_bins // 2 :].mean(axis=(2, 3)), f"{matrix_name}_late_mean")
        add_channel_block(matrix[:, :, : n_bins // 2, n_bins // 2 :].mean(axis=(2, 3)), f"{matrix_name}_cross_mean")
        for lag in GAF_LAGS:
            diag_values = np.diagonal(matrix[:, :, :-lag, lag:], axis1=2, axis2=3)
            add_channel_block(diag_values.mean(axis=2), f"{matrix_name}_diag_lag{lag}_mean")

    # Cross-channel recurrence summaries are compact global descriptors of the
    # day-level sensor image. They add only a few columns and often catch
    # synchronized phone/watch behavior that per-channel summaries miss.
    channel_means = bounded.mean(axis=2)
    channel_std = bounded.std(axis=2)
    global_blocks = [
        channel_means.mean(axis=1),
        channel_means.std(axis=1),
        channel_std.mean(axis=1),
        channel_std.std(axis=1),
    ]
    global_names = [
        "global_channel_mean_mean",
        "global_channel_mean_std",
        "global_channel_std_mean",
        "global_channel_std_std",
    ]
    corr_features = []
    for row in bounded:
        corr = np.corrcoef(row)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = corr[np.triu_indices_from(corr, k=1)]
        corr_features.append([float(upper.mean()), float(upper.std()), float(np.quantile(upper, 0.25)), float(np.quantile(upper, 0.75))])
    corr_arr = np.asarray(corr_features, dtype=np.float32)
    global_blocks.extend([corr_arr[:, idx] for idx in range(corr_arr.shape[1])])
    global_names.extend(["global_channel_corr_mean", "global_channel_corr_std", "global_channel_corr_q25", "global_channel_corr_q75"])

    features = np.concatenate([*blocks, *[block.reshape(n_rows, 1) for block in global_blocks]], axis=1)
    names.extend(global_names)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if features.shape[1] != len(names):
        raise ValueError(f"Feature/name mismatch: {features.shape[1]} vs {len(names)}")
    return features, names

def fit_oof_model(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(x_test.shape[0], dtype=float)
    n_folds = 0
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(spec))
        model.fit(x_train[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x_train[valid_idx])[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        n_folds += 1
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred / n_folds, EPS, 1.0 - EPS)


def candidate_key(model_name: str, weight: float) -> str:
    return f"blend__recurrence_gaf__{model_name}__w{weight:g}"


def parse_candidate_key(key: str) -> dict[str, Any]:
    if key == "base":
        return {"kind": "base"}
    if key.startswith("raw__"):
        return {"kind": "raw", "model": key.split("__", 1)[1]}
    kind, feature_set, model_name, weight_raw = key.split("__", 3)
    if kind != "blend" or feature_set != "recurrence_gaf" or not weight_raw.startswith("w"):
        raise ValueError(f"Unexpected candidate key: {key}")
    return {"kind": "blend", "feature_set": feature_set, "model": model_name, "weight": float(weight_raw[1:])}


def build_recurrence_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = recurrence_gaf_features(train_tensor, channel_names)
    x_test, _ = recurrence_gaf_features(test_tensor, channel_names)
    info = {
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_bins": N_BINS,
        "n_recurrence_gaf_features": int(x_train.shape[1]),
        "feature_name_sample": feature_names[:40],
        "representation": "Recurrence-rate and Gramian Angular Field style image summaries over the aligned raw tensor.",
    }
    return train, test, sample_original, x_train, x_test, channel_names, info


def compute_model_predictions(
    train: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    fold_ids: np.ndarray,
    needed_models: set[str] | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    spec_lookup = {spec.name: spec for spec in MODEL_SPECS}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    specs_used = []
    for model_name, spec in spec_lookup.items():
        if needed_models is not None and model_name not in needed_models:
            continue
        specs_used.append(asdict(spec))
        predictions[model_name] = {}
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            oof_pred, test_pred = fit_oof_model(x_train, x_test, y, fold_ids, spec)
            predictions[model_name][target] = {"oof": oof_pred, "test": test_pred}
    return predictions, specs_used


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train, test, _sample_original, x_train, x_test, _channel_names, feature_info = build_recurrence_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    model_predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report: dict[str, Any] = {
        "diagnostic": "public_recurrence_gaf_sequence_overlay",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "feature_info": feature_info,
        "model_specs": specs_used,
        "blend_weights": BLEND_WEIGHTS,
        "base_scores": base_scores,
        "base_mean": float(np.mean(list(base_scores.values()))),
        "targets": {},
    }
    target_best_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidates: dict[str, dict[str, Any]] = {
            "base": {"kind": "base", "loss": base_scores[target], "test_mean": float(np.mean(base_test_values))}
        }
        for model_name, by_target in model_predictions.items():
            raw_oof = by_target[target]["oof"]
            raw_test = by_target[target]["test"]
            candidates[f"raw__{model_name}"] = {
                "kind": "raw",
                "model": model_name,
                "loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                "test_mean": float(np.mean(raw_test)),
            }
            for weight in BLEND_WEIGHTS:
                pred = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                candidates[candidate_key(model_name, weight)] = {
                    "kind": "blend",
                    "feature_set": "recurrence_gaf",
                    "model": model_name,
                    "weight": weight,
                    "loss": float(log_loss(y, pred, labels=[0, 1])),
                    "test_mean": float(np.mean(blended_test)),
                    "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                }
        best_name, best_info = min(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "best": best_name,
            "best_config": parse_candidate_key(best_name),
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top10": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
        }
    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict_from_report(base_id: int, report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if int(report["base_id"]) != base_id:
        raise ValueError(f"Report base_id={report['base_id']} does not match requested base_id={base_id}.")
    train, test, sample_original, x_train, x_test, _channel_names, feature_info = build_recurrence_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    selected_configs = {target: report["targets"][target]["best_config"] for target in TARGETS}
    needed_models = {config["model"] for config in selected_configs.values() if config.get("kind") in {"blend", "raw"}}
    model_predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids, needed_models)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        config = selected_configs[target]
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if config["kind"] == "base":
            selected_oof = base_values
            selected_test = base_test_values
        else:
            side = model_predictions[config["model"]][target]
            raw_oof = side["oof"]
            raw_test = side["test"]
            if config["kind"] == "raw":
                selected_oof = raw_oof
                selected_test = raw_test
            else:
                weight = float(config["weight"])
                selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
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
        "feature_info": feature_info,
        "model_specs_used": specs_used,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are blended target-wise with deterministic recurrence/GAF sequence-image "
            "summary models selected from the listed local public-structure diagnostic."
        ),
        "caution": "Public-structure/non-strict recurrence/GAF sequence overlay; not strict Subject-Hole evidence.",
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
            f"# Submission {experiment_id:03d}: recurrence/GAF sequence-image overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses deterministic recurrence-rate and Gramian Angular Field style summaries."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_recurrence_gaf_sequence_overlay_experiment.py "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--diagnostic-report {relpath(diagnostic_report)}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def promote_from_report(experiment_id: int, slug: str, base_id: int, diagnostic_report: Path) -> dict[str, Any]:
    diagnostic_report = diagnostic_report.resolve()
    if experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    oof, test_pred, target_scores, selector_info, sample_original = predict_from_report(base_id, diagnostic_report)
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
        "method": f"Target-wise recurrence/GAF sequence-image overlay on ID{base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_recurrence_gaf_sequence_image",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise recurrence/GAF sequence-image overlay under public-structure interleaved CV.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id, diagnostic_report)
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Public-structure recurrence/GAF sequence-image overlay on ID{base_id:03d}; target-wise selected from "
            f"{relpath(diagnostic_report)}. Not strict Subject-Hole CV; no DACON submission was made."
        ),
        model_description=f"recurrence/GAF sequence-image overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "id200_recurrence_gaf_sequence_diagnostic.json")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="id200_recurrence_gaf_sequence_overlay")
    parser.add_argument("--diagnostic-report", type=Path)
    args = parser.parse_args()
    if args.promote:
        if args.experiment_id is None or args.diagnostic_report is None:
            raise ValueError("--promote requires --experiment-id and --diagnostic-report.")
        metadata = promote_from_report(args.experiment_id, args.slug, args.base_id, args.diagnostic_report)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    report = run_diagnostic(args.base_id, args.output)
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "feature_info": report["feature_info"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
