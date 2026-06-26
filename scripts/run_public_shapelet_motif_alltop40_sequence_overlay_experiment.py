#!/usr/bin/env python3
"""Diagnose/promote all-top40 shapelet/motif sequence overlays.

This representation samples a compact, label-free motif bank from the aligned
44-bin daily sensor tensor. Each row is transformed by z-normalized subsequence
matching distances, motif positions, and late-vs-early distance contrasts. It is
intended as a different clean signal family from SAX/PAA, recurrence/GAF, and
wavelet/spectral summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
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
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
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

DEFAULT_BASE_ID = 212
SHAPELET_LENGTHS = [3, 5, 7, 11, 15, 21]
MOTIFS_PER_CHANNEL_LENGTH = 5
N_SELECTED_CHANNELS = 40
BLEND_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_01", "lr", c_value=0.01),
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_1", "lr", c_value=0.1),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
    ModelSpec("et_d4", "et", max_depth=4),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def z_normalize(segment: np.ndarray) -> np.ndarray:
    values = np.asarray(segment, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-6)


def select_channels(train_tensor: np.ndarray, n_channels: int = N_SELECTED_CHANNELS) -> np.ndarray:
    variance = train_tensor.var(axis=(0, 2))
    mean_abs = np.abs(train_tensor).mean(axis=(0, 2))
    score = variance + 0.05 * mean_abs
    order = np.argsort(score)[::-1]
    return order[: min(n_channels, len(order))].astype(int)


def build_motif_bank(train_tensor: np.ndarray, channel_names: list[str]) -> list[dict[str, Any]]:
    motifs: list[dict[str, Any]] = []
    selected_channels = select_channels(train_tensor)
    quantiles = np.linspace(0.10, 0.90, MOTIFS_PER_CHANNEL_LENGTH)
    for channel_idx in selected_channels:
        channel_series = train_tensor[:, channel_idx, :]
        for length in SHAPELET_LENGTHS:
            subsequences: list[np.ndarray] = []
            keys: list[tuple[int, int]] = []
            scores: list[float] = []
            for row_idx in range(channel_series.shape[0]):
                for start_idx in range(channel_series.shape[1] - length + 1):
                    segment = channel_series[row_idx, start_idx : start_idx + length]
                    normalized = z_normalize(segment)
                    slope = float(segment[-1] - segment[0])
                    score = float(segment.std() + 0.15 * abs(segment.mean()) + 0.10 * abs(slope))
                    subsequences.append(normalized.astype(np.float32))
                    keys.append((row_idx, start_idx))
                    scores.append(score)
            if not scores:
                continue
            score_values = np.asarray(scores, dtype=np.float32)
            ranks = np.argsort(score_values)
            picks = [int(ranks[min(len(ranks) - 1, max(0, round(q * (len(ranks) - 1))))]) for q in quantiles]
            picks.append(int(ranks[-1]))
            seen: set[tuple[int, int, int, int]] = set()
            for pick_idx in picks:
                row_idx, start_idx = keys[pick_idx]
                key = (int(channel_idx), int(length), int(row_idx), int(start_idx))
                if key in seen:
                    continue
                seen.add(key)
                motifs.append(
                    {
                        "channel": int(channel_idx),
                        "channel_name": channel_names[int(channel_idx)],
                        "length": int(length),
                        "start": int(start_idx),
                        "source_row": int(row_idx),
                        "score": float(score_values[pick_idx]),
                        "prototype": subsequences[pick_idx].astype(np.float32).tolist(),
                    }
                )
    return motifs


def apply_shapelets(tensor: np.ndarray, motifs: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(tensor, dtype=np.float32)
    n_rows = values.shape[0]
    features = np.zeros((n_rows, len(motifs) * 4), dtype=np.float32)
    feature_names: list[str] = []
    for motif_idx, motif in enumerate(motifs):
        channel_idx = int(motif["channel"])
        length = int(motif["length"])
        prototype = np.asarray(motif["prototype"], dtype=np.float32)
        series = values[:, channel_idx, :]
        n_positions = series.shape[1] - length + 1
        distances = np.zeros((n_rows, n_positions), dtype=np.float32)
        for start_idx in range(n_positions):
            segment = series[:, start_idx : start_idx + length]
            means = segment.mean(axis=1, keepdims=True)
            stds = segment.std(axis=1, keepdims=True) + 1e-6
            normalized = (segment - means) / stds
            distances[:, start_idx] = ((normalized - prototype.reshape(1, -1)) ** 2).mean(axis=1)
        min_idx = np.argmin(distances, axis=1)
        min_dist = distances[np.arange(n_rows), min_idx]
        mean_dist = distances.mean(axis=1)
        early_min = distances[:, : max(1, n_positions // 2)].min(axis=1)
        late_min = distances[:, n_positions // 2 :].min(axis=1)
        base_idx = motif_idx * 4
        features[:, base_idx] = min_dist
        features[:, base_idx + 1] = mean_dist
        features[:, base_idx + 2] = min_idx / max(1, n_positions - 1)
        features[:, base_idx + 3] = late_min - early_min
        stem = f"{motif['channel_name']}__L{length}__s{motif['start']}__r{motif['source_row']}"
        feature_names.extend(
            [
                f"{stem}__min_dist",
                f"{stem}__mean_dist",
                f"{stem}__argmin_norm",
                f"{stem}__late_minus_early_min",
            ]
        )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), feature_names


def build_shapelet_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]
]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    motifs = build_motif_bank(train_tensor, channel_names)
    x_train, feature_names = apply_shapelets(train_tensor, motifs)
    x_test, _ = apply_shapelets(test_tensor, motifs)
    serializable_motifs = [{key: value for key, value in motif.items() if key != "prototype"} for motif in motifs[:80]]
    info = {
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_bins": int(train_tensor.shape[2]),
        "n_shapelet_motifs": int(len(motifs)),
        "n_shapelet_features": int(x_train.shape[1]),
        "shapelet_lengths": SHAPELET_LENGTHS,
        "motifs_per_channel_length": MOTIFS_PER_CHANNEL_LENGTH,
        "n_selected_channels": N_SELECTED_CHANNELS,
        "selected_channel_names": [channel_names[int(idx)] for idx in select_channels(train_tensor)],
        "motif_sample": serializable_motifs,
        "feature_name_sample": feature_names[:40],
        "representation": "Label-free train-tensor motif bank with z-normalized subsequence distance, position, and late-vs-early contrast features.",
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


def candidate_key(model_name: str, weight: float) -> str:
    return f"blend__shapelet_motif__{model_name}__w{weight:g}"


def parse_candidate_key(key: str) -> dict[str, Any]:
    if key == "base":
        return {"kind": "base"}
    if key.startswith("raw__"):
        return {"kind": "raw", "feature_set": "shapelet_motif", "model": key.split("__", 1)[1]}
    kind, feature_set, model_name, weight_raw = key.split("__", 3)
    if kind != "blend" or feature_set != "shapelet_motif" or not weight_raw.startswith("w"):
        raise ValueError(f"Unexpected candidate key: {key}")
    return {"kind": "blend", "feature_set": feature_set, "model": model_name, "weight": float(weight_raw[1:])}


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train, test, _sample_original, x_train, x_test, _channel_names, feature_info = build_shapelet_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report: dict[str, Any] = {
        "diagnostic": "public_shapelet_motif_alltop40_sequence_overlay",
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
        for model_name, by_target in predictions.items():
            raw_oof = by_target[target]["oof"]
            raw_test = by_target[target]["test"]
            candidates[f"raw__{model_name}"] = {
                "kind": "raw",
                "feature_set": "shapelet_motif",
                "model": model_name,
                "loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                "test_mean": float(np.mean(raw_test)),
            }
            for weight in BLEND_WEIGHTS:
                pred = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                candidates[candidate_key(model_name, weight)] = {
                    "kind": "blend",
                    "feature_set": "shapelet_motif",
                    "model": model_name,
                    "weight": weight,
                    "loss": float(log_loss(y, pred, labels=[0, 1])),
                    "test_mean": float(np.mean(blended_test)),
                    "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                }
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
    train, test, sample_original, x_train, x_test, _channel_names, feature_info = build_shapelet_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    selected_configs = {target: report["targets"][target]["best_config"] for target in TARGETS}
    needed_models = {config["model"] for config in selected_configs.values() if config.get("kind") in {"blend", "raw"}}
    predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids, needed_models)
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
            side = predictions[config["model"]][target]
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
            f"ID{base_id:03d} predictions are lightly blended target-wise with deterministic shapelet/motif "
            "sequence models selected from the listed local public-structure diagnostic. Raw side models are diagnostic-only."
        ),
        "caution": "Public-structure/non-strict shapelet/motif sequence overlay; motif bank is label-free but not strict Subject-Hole evidence.",
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
            f"# Submission {experiment_id:03d}: all-top40 shapelet/motif sequence overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses deterministic shapelet/motif distance features over the aligned raw tensor."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_shapelet_motif_alltop40_sequence_overlay_experiment.py "
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
        "method": f"Target-wise all-top40 shapelet/motif sequence overlay on ID{base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_shapelet_motif_alltop40_sequence",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise all-top40 shapelet/motif sequence overlay under public-structure interleaved CV.",
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
            f"Public-structure all-top40 shapelet/motif sequence overlay on ID{base_id:03d}; target-wise selected from "
            f"{relpath(diagnostic_report)}. Raw side models were diagnostic-only; not strict Subject-Hole CV; "
            "no DACON submission was made."
        ),
        model_description=f"all-top40 shapelet/motif sequence overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id212_shapelet_motif_alltop40_sequence_diagnostic.json",
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="id212_shapelet_motif_alltop40_sequence_overlay")
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
        "feature_info": {
            key: value
            for key, value in report["feature_info"].items()
            if key not in {"motif_sample", "feature_name_sample"}
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
