#!/usr/bin/env python3
"""Run a strict Subject-Hole diagnostic for Chronos embedding overlays.

This script is intended for Kaggle/offloaded execution. It lazily imports
Chronos/Torch only when embeddings must be extracted, writes the embeddings as a
small CSV cache, and then evaluates whether target-wise blends improve a strict
Subject-Hole base artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

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

from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    REPORTS_DIR,
    TARGETS,
    build_feature_frames,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EPS = 1e-6
DEFAULT_BASE_ID = 129
DEFAULT_MODEL_ID = "amazon/chronos-t5-mini"
DEFAULT_EMBEDDING_DIMS = 8
DEFAULT_TARGETS = ["Q2", "Q3", "S3", "S4"]
DEFAULT_MODELS = ["lr_c05", "hgb_l7_l2", "et_d6_l12"]
DEFAULT_BLEND_WEIGHTS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_SERIES_COLUMNS = [
    "ch2025_wHr_hr_mean_mean",
    "wHr_sleep_hr_mean",
    "ch2025_mActivity_m_activity_mean",
    "ch2025_mLight_light_mean_mean",
    "ch2025_wLight_light_mean_mean",
    "pedo_total_steps",
    "screen_on_total",
    "mUsage_total_time_mins",
    "usage_total_time",
]


def safe_name(value: str) -> str:
    """Return a compact column-safe feature source name."""

    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")


def robust_series(values: pd.Series) -> np.ndarray:
    """Build a finite, robustly standardized context series for Chronos."""

    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if np.isfinite(arr).any():
        median = float(np.nanmedian(arr))
    else:
        median = 0.0
    arr = np.where(np.isfinite(arr), arr, median)
    q25, q75 = np.percentile(arr, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - median) / scale, -10.0, 10.0).astype(np.float32)


def reduce_embedding_matrix(embedding: np.ndarray, n_dims: int) -> np.ndarray:
    """Chunk-average Chronos hidden states to keep the downstream table small."""

    if embedding.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embedding.shape}.")
    hidden_dim = embedding.shape[1]
    chunks = np.array_split(np.arange(hidden_dim), n_dims)
    reduced = np.zeros((embedding.shape[0], n_dims), dtype=np.float32)
    for chunk_idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        reduced[:, chunk_idx] = embedding[:, chunk].mean(axis=1)
    return reduced


def choose_device(device: str) -> str:
    """Resolve the requested Torch device without importing Torch at module load."""

    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_chronos_pipeline(model_id: str, device: str) -> Any:
    """Load Chronos lazily so normal lint/tests do not download or allocate models."""

    import torch
    from chronos import ChronosPipeline

    resolved_device = choose_device(device)
    dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
    print(f"Loading {model_id} on {resolved_device} with dtype={dtype} ...", flush=True)
    return ChronosPipeline.from_pretrained(model_id, device_map=resolved_device, torch_dtype=dtype)


def build_chronos_embedding_cache(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_path: Path,
    source_columns: list[str],
    model_id: str,
    device: str,
    n_dims: int,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract or load row/date-level Chronos embeddings."""

    if output_path.exists() and not force:
        embeddings = pd.read_csv(output_path)
        embeddings["date"] = pd.to_datetime(embeddings["date"]).dt.date
        feature_cols = [col for col in embeddings.columns if col.startswith("chronos_")]
        return embeddings, {
            "embedding_cache": str(output_path.relative_to(ROOT)),
            "cache_reused": True,
            "n_embedding_rows": int(len(embeddings)),
            "n_embedding_features": int(len(feature_cols)),
            "source_columns": source_columns,
            "model_id": model_id,
            "embedding_dims_per_source": n_dims,
        }

    available_columns = [col for col in source_columns if col in train.columns and col in test.columns]
    if not available_columns:
        raise ValueError("None of the requested Chronos source columns exist in train/test frames.")

    all_rows = pd.concat(
        [
            train[["subject_id", "date", *available_columns]].assign(row_kind="train"),
            test[["subject_id", "date", *available_columns]].assign(row_kind="test"),
        ],
        ignore_index=True,
    )
    all_rows["date"] = pd.to_datetime(all_rows["date"]).dt.date
    all_rows = all_rows.sort_values(["subject_id", "date", "row_kind"]).reset_index(drop=True)

    pipeline = load_chronos_pipeline(model_id=model_id, device=device)
    import torch

    embedding_frames: list[pd.DataFrame] = []
    for source_col in available_columns:
        source_safe = safe_name(source_col)
        rows_for_source: list[pd.DataFrame] = []
        for subject_id, group in all_rows.groupby("subject_id", sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            context_values = robust_series(group[source_col])
            context = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedded = pipeline.embed(context)
            embedding_tensor = embedded[0] if isinstance(embedded, (tuple, list)) else embedded
            embedding_array = embedding_tensor[0, : len(group), :].detach().cpu().float().numpy()
            reduced = reduce_embedding_matrix(embedding_array, n_dims=n_dims)
            reduced_df = pd.DataFrame(
                reduced,
                columns=[f"chronos_{source_safe}_e{idx:02d}" for idx in range(n_dims)],
            )
            reduced_df.insert(0, "date", group["date"].to_numpy())
            reduced_df.insert(0, "subject_id", subject_id)
            rows_for_source.append(reduced_df)
        embedding_frames.append(pd.concat(rows_for_source, ignore_index=True))
        print(f"Embedded {source_col}: {len(embedding_frames[-1])} rows", flush=True)

    embeddings = embedding_frames[0]
    for frame in embedding_frames[1:]:
        embeddings = embeddings.merge(frame, on=["subject_id", "date"], how="outer", validate="one_to_one")
    embeddings = embeddings.sort_values(["subject_id", "date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(output_path, index=False)
    feature_cols = [col for col in embeddings.columns if col.startswith("chronos_")]
    info = {
        "embedding_cache": str(output_path.relative_to(ROOT)),
        "cache_reused": False,
        "n_embedding_rows": int(len(embeddings)),
        "n_embedding_features": int(len(feature_cols)),
        "source_columns": available_columns,
        "model_id": model_id,
        "embedding_dims_per_source": n_dims,
        "context_note": (
            "Chronos contexts are robust-standardized per subject/source using known covariates only; "
            "no target labels are used in embedding extraction."
        ),
    }
    return embeddings, info


def align_embedding_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align embedding rows to the canonical train/test order."""

    embeddings = embeddings.copy()
    embeddings["date"] = pd.to_datetime(embeddings["date"]).dt.date
    if embeddings.duplicated(["subject_id", "date"]).any():
        duplicated = embeddings.loc[embeddings.duplicated(["subject_id", "date"], keep=False), ["subject_id", "date"]]
        raise ValueError(f"Chronos embeddings are not unique by subject/date:\n{duplicated.head()}")

    train_key = train[["subject_id", "date"]].copy()
    test_key = test[["subject_id", "date"]].copy()
    train_key["date"] = pd.to_datetime(train_key["date"]).dt.date
    test_key["date"] = pd.to_datetime(test_key["date"]).dt.date
    train_emb = train_key.merge(embeddings, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_emb = test_key.merge(embeddings, on=["subject_id", "date"], how="left", validate="one_to_one")
    feature_cols = [col for col in train_emb.columns if col.startswith("chronos_")]
    train_emb["chronos_present_frac"] = train_emb[feature_cols].notna().mean(axis=1) if feature_cols else 0.0
    test_emb["chronos_present_frac"] = test_emb[feature_cols].notna().mean(axis=1) if feature_cols else 0.0
    feature_cols.append("chronos_present_frac")
    return train_emb, test_emb, feature_cols


def load_base_oof(train: pd.DataFrame, base_id: int) -> tuple[pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF artifact.")
    return align_candidate(pd.read_csv(oof_paths[0]), train), oof_paths[0].name


def make_embedding_model(model_name: str) -> object:
    if model_name == "lr_c05":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, solver="liblinear", random_state=61),
        )
    if model_name == "hgb_l7_l2":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.03,
                max_leaf_nodes=7,
                l2_regularization=2.0,
                random_state=62,
            ),
        )
    if model_name == "et_d6_l12":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=6,
                min_samples_leaf=12,
                max_features=0.6,
                random_state=63,
                n_jobs=2,
            ),
        )
    raise ValueError(f"Unknown embedding model: {model_name}")


def fit_embedding_oof(
    train: pd.DataFrame,
    train_emb: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    model_name: str,
) -> np.ndarray:
    y = train[target].astype(int).to_numpy()
    oof = np.zeros(len(train), dtype=float)
    has_prediction = np.zeros(len(train), dtype=bool)
    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_embedding_model(model_name))
        model.fit(train_emb.iloc[train_idx][feature_cols], y[train_idx])
        oof[valid_idx] = model.predict_proba(train_emb.iloc[valid_idx][feature_cols])[:, 1]
        has_prediction[valid_idx] = True
    if not has_prediction.all():
        raise ValueError(f"Folds did not produce predictions for every {target} row.")
    return np.clip(oof, EPS, 1.0 - EPS)


def evaluate_target(
    train: pd.DataFrame,
    train_emb: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_oof: pd.DataFrame,
    target: str,
    model_names: list[str],
    blend_weights: list[float],
) -> dict[str, Any]:
    y = train[target].astype(int).to_numpy()
    base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    candidates = {"base": base_values}
    candidate_details: dict[str, dict[str, Any]] = {"base": {"kind": "base"}}

    for model_name in model_names:
        raw_oof = fit_embedding_oof(train, train_emb, feature_cols, folds, target, model_name)
        raw_name = f"raw_{model_name}"
        candidates[raw_name] = raw_oof
        candidate_details[raw_name] = {"kind": "raw_chronos_embedding", "model": model_name}
        for weight in blend_weights:
            blend_name = f"blend_{model_name}_{weight:.2f}"
            candidates[blend_name] = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
            candidate_details[blend_name] = {"kind": "blend", "model": model_name, "weight": weight}

    candidate_losses = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in candidates.items()}
    selected = np.zeros(len(train), dtype=float)
    nested_choices = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidates.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected[valid_idx] = candidates[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_best_name, full_best_loss = min(candidate_losses.items(), key=lambda item: item[1])
    nested_loss = float(log_loss(y, selected, labels=[0, 1]))
    return {
        "base_loss": candidate_losses["base"],
        "full_best_candidate": full_best_name,
        "full_best_loss": float(full_best_loss),
        "nested_loss": nested_loss,
        "nested_improves": bool(nested_loss < candidate_losses["base"]),
        "candidate_losses": candidate_losses,
        "candidate_details": candidate_details,
        "nested_choices": nested_choices,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--blend-weights", nargs="+", type=float, default=DEFAULT_BLEND_WEIGHTS)
    parser.add_argument("--source-columns", nargs="+", default=DEFAULT_SERIES_COLUMNS)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-dims", type=int, default=DEFAULT_EMBEDDING_DIMS)
    parser.add_argument("--embedding-output", default="data/processed/hf_chronos_embeddings.csv")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, base_source = load_base_oof(train, args.base_id)
    embedding_output = ROOT / args.embedding_output
    embeddings, embedding_info = build_chronos_embedding_cache(
        train=train,
        test=test,
        output_path=embedding_output,
        source_columns=args.source_columns,
        model_id=args.model_id,
        device=args.device,
        n_dims=args.embedding_dims,
        force=args.force_extract,
    )
    train_emb, _test_emb, feature_cols = align_embedding_features(train, test, embeddings)
    folds = make_subject_hole_folds(train, n_folds=5)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }

    report: dict[str, Any] = {
        "base_id": args.base_id,
        "base_source": base_source,
        "diagnostic": f"strict_chronos_embedding_overlay_id{args.base_id:03d}",
        "validation": "Subject-Hole nested candidate selection diagnostic only; no artifact is promoted here.",
        "targets_checked": args.targets,
        "models_checked": args.models,
        "blend_weights": args.blend_weights,
        "fold_sizes": [[int(len(train_idx)), int(len(valid_idx))] for train_idx, valid_idx in folds],
        "embedding_info": embedding_info,
        "n_aligned_embedding_features": int(len(feature_cols)),
        "base_target_scores": base_scores,
        "base_mean_score": float(np.mean(list(base_scores.values()))),
        "targets": {},
    }
    for target in args.targets:
        result = evaluate_target(
            train=train,
            train_emb=train_emb,
            feature_cols=feature_cols,
            folds=folds,
            base_oof=base_oof,
            target=target,
            model_names=args.models,
            blend_weights=args.blend_weights,
        )
        result["mean_if_promoted_by_nested"] = float(
            (sum(base_scores.values()) - base_scores[target] + result["nested_loss"]) / len(TARGETS)
        )
        report["targets"][target] = result
        print(
            target,
            "base",
            f"{result['base_loss']:.6f}",
            "full_best",
            result["full_best_candidate"],
            f"{result['full_best_loss']:.6f}",
            "nested",
            f"{result['nested_loss']:.6f}",
            "improves",
            result["nested_improves"],
        )

    output_path = (
        Path(args.output)
        if args.output
        else REPORTS_DIR / f"strict_chronos_embedding_overlay_diagnostic_id{args.base_id:03d}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
