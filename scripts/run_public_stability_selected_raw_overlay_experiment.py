#!/usr/bin/env python3
"""Diagnose/promote stability-selected raw-feature overlays under the public split lens.

This script tests a conservative version of the external-notebook stability hint:
for each target/fold it selects a small number of raw numeric features using only
that fold's training labels, penalizes train/test distribution drift, fits low
variance side models, and only allows light blends into an existing anchor.
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
    numeric_feature_columns,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 203
TOP_K_VALUES = [8, 16, 32, 64]
BLEND_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
MAX_REPORT_FEATURES = 20


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
    ModelSpec("hgb_l5", "hgb", max_leaf_nodes=5),
    ModelSpec("et_d3", "et", max_depth=3),
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
                max_iter=70,
                learning_rate=0.03,
                max_leaf_nodes=int(spec.max_leaf_nodes),
                l2_regularization=3.0,
                min_samples_leaf=18,
                random_state=43,
            ),
        )
    if spec.kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=260,
                max_depth=int(spec.max_depth),
                min_samples_leaf=14,
                max_features=0.65,
                random_state=43,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def feature_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arr = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    arr = arr.replace([np.inf, -np.inf], np.nan)
    return arr.to_numpy(dtype=np.float32)


def nan_iqr(values: np.ndarray) -> np.ndarray:
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    return q75 - q25


def select_stable_columns(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    columns: list[str],
    top_k: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select label-correlated, distribution-stable columns using fold-train labels only."""

    fold_x = x_train[train_idx]
    fold_y = y[train_idx].astype(int)
    if len(np.unique(fold_y)) < 2:
        variances = np.nanvar(fold_x, axis=0)
        order = np.argsort(np.nan_to_num(variances, nan=-1.0))[::-1]
        selected = order[:top_k]
        details = [
            {"feature": columns[int(idx)], "score": float(np.nan_to_num(variances[int(idx)])), "fallback": True}
            for idx in selected[:MAX_REPORT_FEATURES]
        ]
        return selected.astype(int), details

    pos = fold_x[fold_y == 1]
    neg = fold_x[fold_y == 0]
    pos_mean = np.nanmean(pos, axis=0)
    neg_mean = np.nanmean(neg, axis=0)
    pos_var = np.nanvar(pos, axis=0)
    neg_var = np.nanvar(neg, axis=0)
    pooled = np.sqrt(0.5 * (pos_var + neg_var)) + EPS
    effect = np.abs(pos_mean - neg_mean) / pooled

    fold_median = np.nanmedian(fold_x, axis=0)
    test_median = np.nanmedian(x_test, axis=0)
    scale = nan_iqr(np.vstack([fold_x, x_test])) + EPS
    location_shift = np.abs(fold_median - test_median) / scale

    fold_missing = np.mean(~np.isfinite(fold_x), axis=0)
    test_missing = np.mean(~np.isfinite(x_test), axis=0)
    missing_shift = np.abs(fold_missing - test_missing)
    finite_rate = 1.0 - 0.5 * (fold_missing + test_missing)

    score = effect * finite_rate / (1.0 + 0.35 * location_shift + 1.25 * missing_shift)
    score = np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)
    order = np.argsort(score)[::-1]
    selected = order[: min(top_k, len(order))]
    details = []
    for idx in selected[:MAX_REPORT_FEATURES]:
        idx_int = int(idx)
        details.append(
            {
                "feature": columns[idx_int],
                "score": float(score[idx_int]),
                "effect": float(np.nan_to_num(effect[idx_int])),
                "location_shift": float(np.nan_to_num(location_shift[idx_int])),
                "missing_shift": float(np.nan_to_num(missing_shift[idx_int])),
            }
        )
    return selected.astype(int), details


def fit_stability_oof_model(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    columns: list[str],
    top_k: int,
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(x_test.shape[0], dtype=float)
    n_folds = 0
    selection_details: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        selected_idx, top_features = select_stable_columns(x_train, x_test, y, train_idx, columns, top_k)
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(spec))
        model.fit(x_train[train_idx][:, selected_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x_train[valid_idx][:, selected_idx])[:, 1]
        test_pred += model.predict_proba(x_test[:, selected_idx])[:, 1]
        selection_details.append(
            {
                "fold": int(fold),
                "n_selected": int(len(selected_idx)),
                "top_features": top_features,
            }
        )
        n_folds += 1
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred / n_folds, EPS, 1.0 - EPS), selection_details


def candidate_key(top_k: int, model_name: str, weight: float) -> str:
    return f"blend__stable_top{top_k}__{model_name}__w{weight:g}"


def parse_candidate_key(key: str) -> dict[str, Any]:
    if key == "base":
        return {"kind": "base"}
    if key.startswith("raw__"):
        _raw, feature_set, model_name = key.split("__", 2)
        return {"kind": "raw", "feature_set": feature_set, "model": model_name, "top_k": int(feature_set[10:])}
    kind, feature_set, model_name, weight_raw = key.split("__", 3)
    if kind != "blend" or not feature_set.startswith("stable_top") or not weight_raw.startswith("w"):
        raise ValueError(f"Unexpected candidate key: {key}")
    return {
        "kind": "blend",
        "feature_set": feature_set,
        "top_k": int(feature_set[10:]),
        "model": model_name,
        "weight": float(weight_raw[1:]),
    }


def compute_side_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold_ids: np.ndarray,
    needed_pairs: set[tuple[int, str]] | None = None,
) -> tuple[dict[int, dict[str, dict[str, dict[str, np.ndarray]]]], dict[str, Any]]:
    columns = numeric_feature_columns(train, test)
    x_train = feature_matrix(train, columns)
    x_test = feature_matrix(test, columns)
    predictions: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    selection_details: dict[str, Any] = {}
    spec_lookup = {spec.name: spec for spec in MODEL_SPECS}
    for top_k in TOP_K_VALUES:
        predictions[top_k] = {}
        for model_name, spec in spec_lookup.items():
            if needed_pairs is not None and (top_k, model_name) not in needed_pairs:
                continue
            predictions[top_k][model_name] = {}
            for target in TARGETS:
                y = train[target].astype(int).to_numpy()
                oof_pred, test_pred, details = fit_stability_oof_model(
                    x_train=x_train,
                    x_test=x_test,
                    y=y,
                    fold_ids=fold_ids,
                    columns=columns,
                    top_k=top_k,
                    spec=spec,
                )
                predictions[top_k][model_name][target] = {"oof": oof_pred, "test": test_pred}
                detail_key = f"stable_top{top_k}__{model_name}__{target}"
                selection_details[detail_key] = details
    info = {
        "n_input_features": len(columns),
        "top_k_values": TOP_K_VALUES,
        "input_feature_sample": columns[:40],
        "selection_details": selection_details,
        "selection_rule": (
            "For each target/fold, rank numeric features by fold-train standardized class separation, "
            "penalized by label-free train/test median and missing-rate drift; fit only the top-k columns."
        ),
    }
    return predictions, info


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    side_predictions, feature_info = compute_side_predictions(train, test, fold_ids)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report: dict[str, Any] = {
        "diagnostic": "public_stability_selected_raw_overlay",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "feature_info": feature_info,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
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
        for top_k, by_model in side_predictions.items():
            for model_name, by_target in by_model.items():
                raw_oof = by_target[target]["oof"]
                raw_test = by_target[target]["test"]
                raw_key = f"raw__stable_top{top_k}__{model_name}"
                candidates[raw_key] = {
                    "kind": "raw",
                    "feature_set": f"stable_top{top_k}",
                    "model": model_name,
                    "top_k": top_k,
                    "loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                    "test_mean": float(np.mean(raw_test)),
                }
                for weight in BLEND_WEIGHTS:
                    pred = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                    blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                    candidates[candidate_key(top_k, model_name, weight)] = {
                        "kind": "blend",
                        "feature_set": f"stable_top{top_k}",
                        "model": model_name,
                        "top_k": top_k,
                        "weight": weight,
                        "loss": float(log_loss(y, pred, labels=[0, 1])),
                        "test_mean": float(np.mean(blended_test)),
                        "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                    }
        # Raw side models are tracked for diagnostics but not allowed as selected artifacts;
        # the promotion path remains an anchor-preserving light blend or base.
        selectable = {name: info for name, info in candidates.items() if name == "base" or info.get("kind") == "blend"}
        best_name, best_info = min(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "best": best_name,
            "best_config": parse_candidate_key(best_name),
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top10_selectable": sorted(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
            "top10_all": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
        }
    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict_from_report(
    base_id: int, report_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if int(report["base_id"]) != base_id:
        raise ValueError(f"Report base_id={report['base_id']} does not match requested base_id={base_id}.")
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    selected_configs = {target: report["targets"][target]["best_config"] for target in TARGETS}
    needed_pairs = {
        (int(config["top_k"]), config["model"])
        for config in selected_configs.values()
        if config.get("kind") in {"blend", "raw"}
    }
    side_predictions, feature_info = compute_side_predictions(train, test, fold_ids, needed_pairs=needed_pairs)
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
            side = side_predictions[int(config["top_k"])][config["model"]][target]
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
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are blended target-wise with fold-train stability-selected raw-feature "
            "side models chosen from the listed local public-structure diagnostic. Raw side models are diagnostic-only."
        ),
        "caution": "Public-structure/non-strict stability-selected raw overlay; not strict Subject-Hole evidence.",
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
            f"# Submission {experiment_id:03d}: stability-selected raw-feature overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses fold-train stability-selected raw numeric features as low-variance side models."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_stability_selected_raw_overlay_experiment.py "
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
        "method": f"Target-wise stability-selected raw-feature overlay on ID{base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_stability_selected_raw_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise stability-selected raw-feature overlay under public-structure interleaved CV.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id, diagnostic_report
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Public-structure stability-selected raw-feature overlay on ID{base_id:03d}; target-wise selected from "
            f"{relpath(diagnostic_report)}. Raw side models were diagnostic-only; not strict Subject-Hole CV; "
            "no DACON submission was made."
        ),
        model_description=f"Stability-selected raw-feature overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id203_stability_selected_raw_overlay_diagnostic.json",
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="id203_stability_selected_raw_overlay")
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
        "n_input_features": report["feature_info"]["n_input_features"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
