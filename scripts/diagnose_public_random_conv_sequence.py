#!/usr/bin/env python3
"""Diagnose ROCKET-style random convolution features over daily sensor sequences.

This is a report-only diagnostic. It builds multichannel half-hour sequences from
raw sensors, extracts deterministic random convolution summaries, and scans small
classifiers/blends against a base artifact under the public-structure
interleaved-within-subject validation lens.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from functools import reduce
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

from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402

EPS = 1e-6
N_BINS = 44
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
BLEND_WEIGHTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]


@dataclass(frozen=True)
class KernelSpec:
    channels: tuple[int, ...]
    length: int
    dilation: int
    weights: list[list[float]]
    bias: float


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
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
    ModelSpec("et_d5", "et", max_depth=5),
)


def anchored_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.normalize()
    df.loc[hour < 12, "date"] = df.loc[hour < 12, "date"] - pd.Timedelta(days=1)
    rel_hour = hour.astype(float) + (hour < 12).astype(float) * 24.0 + df["timestamp"].dt.minute / 60.0
    df["bin"] = np.floor((rel_hour - 12.0) * 2.0).astype(int)
    return df[(rel_hour >= 12.0) & (rel_hour < 34.0) & df["bin"].between(0, N_BINS - 1)].copy()


def mean_positive_array(value: object) -> float:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.nan
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(arr.mean()) if arr.size else np.nan


def list_count(value: object) -> float:
    return float(len(value)) if isinstance(value, list) else np.nan


def list_rssi_stat(value: object, stat: str) -> float:
    if not isinstance(value, list) or not value:
        return np.nan
    vals = []
    for item in value:
        if isinstance(item, dict) and "rssi" in item:
            try:
                vals.append(float(item["rssi"]))
            except (TypeError, ValueError):
                pass
    if not vals:
        return np.nan
    arr = np.asarray(vals, dtype=float)
    if stat == "max":
        return float(np.max(arr))
    if stat == "mean":
        return float(np.mean(arr))
    raise ValueError(stat)


def usage_total(value: object) -> float:
    if not isinstance(value, list):
        return np.nan
    total = 0.0
    for item in value:
        if isinstance(item, dict):
            try:
                total += float(item.get("total_time", 0.0))
            except (TypeError, ValueError):
                pass
    return total


def gps_stat(value: object, stat: str) -> float:
    if not isinstance(value, list) or not value:
        return np.nan
    speeds: list[float] = []
    lats: list[float] = []
    lons: list[float] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            speeds.append(float(item.get("speed", np.nan)))
            lats.append(float(item.get("latitude", np.nan)))
            lons.append(float(item.get("longitude", np.nan)))
        except (TypeError, ValueError):
            continue
    speeds_arr = np.asarray(speeds, dtype=float)
    speeds_arr = speeds_arr[np.isfinite(speeds_arr)]
    if stat == "speed_mean":
        return float(np.mean(speeds_arr)) if speeds_arr.size else np.nan
    if stat == "fix_count":
        return float(len(value))
    lat_arr = np.asarray(lats, dtype=float)
    lon_arr = np.asarray(lons, dtype=float)
    lat_arr = lat_arr[np.isfinite(lat_arr)]
    lon_arr = lon_arr[np.isfinite(lon_arr)]
    if stat == "lat_std":
        return float(np.std(lat_arr)) if lat_arr.size else np.nan
    if stat == "lon_std":
        return float(np.std(lon_arr)) if lon_arr.size else np.nan
    raise ValueError(stat)


def ambience_keyword_score(value: object, keyword: str) -> float:
    if not isinstance(value, list):
        return np.nan
    best = 0.0
    keyword_low = keyword.lower()
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label = str(item[0]).lower()
            if keyword_low in label:
                try:
                    best = max(best, float(item[1]))
                except (TypeError, ValueError):
                    pass
    return best


def pivot_channel(df: pd.DataFrame, value_col: str, channel_name: str, agg: str = "mean") -> pd.DataFrame:
    grouped = df.groupby(["subject_id", "date", "bin"])[value_col].agg(agg).reset_index()
    pivot = grouped.pivot_table(index=["subject_id", "date"], columns="bin", values=value_col, aggfunc="first")
    pivot = pivot.reindex(columns=range(N_BINS))
    pivot.columns = [f"{channel_name}__b{int(col):02d}" for col in pivot.columns]
    return pivot.reset_index()


def build_channel_frames(raw_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    df = anchored_frame(raw_dir / "ch2025_mScreenStatus.parquet")
    df["screen"] = pd.to_numeric(df["m_screen_use"], errors="coerce")
    frames.append(pivot_channel(df, "screen", "screen_mean", "mean"))
    frames.append(pivot_channel(df, "screen", "screen_sum", "sum"))

    df = anchored_frame(raw_dir / "ch2025_mACStatus.parquet")
    df["charging"] = pd.to_numeric(df["m_charging"], errors="coerce")
    frames.append(pivot_channel(df, "charging", "charging_mean", "mean"))

    df = anchored_frame(raw_dir / "ch2025_mActivity.parquet")
    df["activity"] = pd.to_numeric(df["m_activity"], errors="coerce")
    df["still"] = (df["activity"] == 0).astype(float)
    df["active"] = df["activity"].isin([3, 7, 8]).astype(float)
    frames.append(pivot_channel(df, "activity", "activity_mean", "mean"))
    frames.append(pivot_channel(df, "still", "still_mean", "mean"))
    frames.append(pivot_channel(df, "active", "active_mean", "mean"))

    for filename, raw_col, name in [
        ("ch2025_mLight.parquet", "m_light", "phone_light"),
        ("ch2025_wLight.parquet", "w_light", "watch_light"),
    ]:
        df = anchored_frame(raw_dir / filename)
        df[name] = np.log1p(pd.to_numeric(df[raw_col], errors="coerce"))
        frames.append(pivot_channel(df, name, f"{name}_log_mean", "mean"))
        frames.append(pivot_channel(df, name, f"{name}_log_max", "max"))

    df = anchored_frame(raw_dir / "ch2025_wHr.parquet")
    df["hr"] = df["heart_rate"].map(mean_positive_array)
    frames.append(pivot_channel(df, "hr", "hr_mean", "mean"))
    frames.append(pivot_channel(df, "hr", "hr_min", "min"))
    frames.append(pivot_channel(df, "hr", "hr_max", "max"))

    df = anchored_frame(raw_dir / "ch2025_wPedo.parquet")
    for raw_col, name, agg in [
        ("step", "step_sum", "sum"),
        ("distance", "distance_sum", "sum"),
        ("speed", "pedo_speed_mean", "mean"),
        ("step_frequency", "step_freq_mean", "mean"),
    ]:
        df[name] = pd.to_numeric(df[raw_col], errors="coerce")
        frames.append(pivot_channel(df, name, name, agg))

    df = anchored_frame(raw_dir / "ch2025_mGps.parquet")
    for stat in ["speed_mean", "fix_count", "lat_std", "lon_std"]:
        df[f"gps_{stat}"] = df["m_gps"].map(lambda value, stat=stat: gps_stat(value, stat))
        frames.append(pivot_channel(df, f"gps_{stat}", f"gps_{stat}", "mean"))

    for filename, col, prefix in [
        ("ch2025_mWifi.parquet", "m_wifi", "wifi"),
        ("ch2025_mBle.parquet", "m_ble", "ble"),
    ]:
        df = anchored_frame(raw_dir / filename)
        df[f"{prefix}_count"] = df[col].map(list_count)
        df[f"{prefix}_rssi_max"] = df[col].map(lambda value: list_rssi_stat(value, "max"))
        df[f"{prefix}_rssi_mean"] = df[col].map(lambda value: list_rssi_stat(value, "mean"))
        frames.append(pivot_channel(df, f"{prefix}_count", f"{prefix}_count", "mean"))
        frames.append(pivot_channel(df, f"{prefix}_rssi_max", f"{prefix}_rssi_max", "mean"))
        frames.append(pivot_channel(df, f"{prefix}_rssi_mean", f"{prefix}_rssi_mean", "mean"))

    df = anchored_frame(raw_dir / "ch2025_mUsageStats.parquet")
    df["usage_total_log"] = np.log1p(df["m_usage_stats"].map(usage_total))
    df["usage_app_count"] = df["m_usage_stats"].map(list_count)
    frames.append(pivot_channel(df, "usage_total_log", "usage_total_log", "sum"))
    frames.append(pivot_channel(df, "usage_app_count", "usage_app_count", "mean"))

    df = anchored_frame(raw_dir / "ch2025_mAmbience.parquet")
    for keyword in ["music", "speech", "vehicle", "outside", "inside"]:
        df[f"amb_{keyword}"] = df["m_ambience"].map(
            lambda value, keyword=keyword: ambience_keyword_score(value, keyword)
        )
        frames.append(pivot_channel(df, f"amb_{keyword}", f"amb_{keyword}", "mean"))

    return frames


def build_daily_sequence_table(raw_dir: Path) -> pd.DataFrame:
    frames = build_channel_frames(raw_dir)
    out = reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["subject_id", "date"]).reset_index(drop=True)


def align_table(
    train: pd.DataFrame, test: pd.DataFrame, seq_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_key = train[["subject_id", "date"]].copy()
    test_key = test[["subject_id", "date"]].copy()
    train_key["date"] = pd.to_datetime(train_key["date"]).dt.date
    test_key["date"] = pd.to_datetime(test_key["date"]).dt.date
    train_seq = train_key.merge(seq_table, on=["subject_id", "date"], how="left")
    test_seq = test_key.merge(seq_table, on=["subject_id", "date"], how="left")
    feature_cols = [col for col in train_seq.columns if col not in ["subject_id", "date"]]
    return train_seq, test_seq, feature_cols


def table_to_tensor(
    train_seq: pd.DataFrame, test_seq: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    channel_names = sorted({col.split("__b", 1)[0] for col in feature_cols})
    train_tensor = np.zeros((len(train_seq), len(channel_names) * 2, N_BINS), dtype=np.float32)
    test_tensor = np.zeros((len(test_seq), len(channel_names) * 2, N_BINS), dtype=np.float32)
    for channel_idx, channel in enumerate(channel_names):
        cols = [f"{channel}__b{bin_idx:02d}" for bin_idx in range(N_BINS)]
        train_values = train_seq.reindex(columns=cols).to_numpy(dtype=float)
        test_values = test_seq.reindex(columns=cols).to_numpy(dtype=float)
        observed = train_values[np.isfinite(train_values)]
        median = float(np.nanmedian(observed)) if observed.size else 0.0
        q25 = float(np.nanpercentile(observed, 25)) if observed.size else 0.0
        q75 = float(np.nanpercentile(observed, 75)) if observed.size else 1.0
        scale = max(q75 - q25, 1e-3)
        train_present = np.isfinite(train_values).astype(np.float32)
        test_present = np.isfinite(test_values).astype(np.float32)
        train_values = np.nan_to_num((train_values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        test_values = np.nan_to_num((test_values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        train_tensor[:, channel_idx, :] = np.clip(train_values, -8.0, 8.0).astype(np.float32)
        test_tensor[:, channel_idx, :] = np.clip(test_values, -8.0, 8.0).astype(np.float32)
        train_tensor[:, len(channel_names) + channel_idx, :] = train_present
        test_tensor[:, len(channel_names) + channel_idx, :] = test_present
    names = channel_names + [f"present_{name}" for name in channel_names]
    return train_tensor, test_tensor, names


def make_kernels(n_channels: int, n_kernels: int, seed: int) -> list[KernelSpec]:
    rng = np.random.default_rng(seed)
    kernels: list[KernelSpec] = []
    lengths = [3, 5, 7, 9, 11]
    dilations = [1, 2, 3, 4]
    for _ in range(n_kernels):
        length = int(rng.choice(lengths))
        dilation = int(rng.choice([d for d in dilations if (length - 1) * d < N_BINS]))
        n_selected = int(rng.choice([1, 1, 2, 3, 4]))
        channels = tuple(
            sorted(rng.choice(n_channels, size=min(n_selected, n_channels), replace=False).astype(int).tolist())
        )
        weights = rng.normal(0.0, 1.0, size=(len(channels), length))
        weights = weights - weights.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.sum(weights**2))
        weights = weights / max(norm, 1e-6)
        bias = float(rng.normal(0.0, 0.25))
        kernels.append(
            KernelSpec(channels=channels, length=length, dilation=dilation, weights=weights.tolist(), bias=bias)
        )
    return kernels


def apply_kernel(x: np.ndarray, kernel: KernelSpec) -> np.ndarray:
    weights = np.asarray(kernel.weights, dtype=np.float32)
    n_rows = x.shape[0]
    n_positions = N_BINS - (kernel.length - 1) * kernel.dilation
    conv = np.zeros((n_rows, n_positions), dtype=np.float32)
    for channel_pos, channel_idx in enumerate(kernel.channels):
        for offset in range(kernel.length):
            conv += (
                weights[channel_pos, offset]
                * x[:, channel_idx, offset * kernel.dilation : offset * kernel.dilation + n_positions]
            )
    conv += np.float32(kernel.bias)
    return conv


def random_conv_features(x: np.ndarray, kernels: list[KernelSpec]) -> np.ndarray:
    features = np.zeros((x.shape[0], len(kernels) * 4), dtype=np.float32)
    for kernel_idx, kernel in enumerate(kernels):
        conv = apply_kernel(x, kernel)
        base = kernel_idx * 4
        features[:, base] = conv.max(axis=1)
        features[:, base + 1] = conv.min(axis=1)
        features[:, base + 2] = conv.mean(axis=1)
        features[:, base + 3] = (conv > 0).mean(axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


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
                learning_rate=0.04,
                max_leaf_nodes=int(spec.max_leaf_nodes),
                l2_regularization=1.0,
                min_samples_leaf=12,
                random_state=43,
            ),
        )
    if spec.kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=int(spec.max_depth),
                min_samples_leaf=10,
                max_features=0.55,
                random_state=43,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def fit_oof_model(
    x_train: np.ndarray, x_test: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, spec: ModelSpec
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


def run_diagnostic(base_id: int, n_kernels: int, seed: int, output_path: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
    x_train = random_conv_features(train_tensor, kernels)
    x_test = random_conv_features(test_tensor, kernels)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))

    report: dict[str, Any] = {
        "diagnostic": "public_random_conv_sequence",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_bins": N_BINS,
        "n_kernels": n_kernels,
        "n_random_conv_features": int(x_train.shape[1]),
        "seed": seed,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "targets": {},
    }
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report["base_scores"] = base_scores
    report["base_mean"] = float(np.mean(list(base_scores.values())))
    model_oof: dict[str, pd.DataFrame] = {
        spec.name: pd.DataFrame(index=train.index, columns=TARGETS, dtype=float) for spec in MODEL_SPECS
    }
    model_test: dict[str, pd.DataFrame] = {
        spec.name: pd.DataFrame(index=test.index, columns=TARGETS, dtype=float) for spec in MODEL_SPECS
    }
    for spec in MODEL_SPECS:
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            oof_pred, test_pred = fit_oof_model(x_train, x_test, y, fold_ids, spec)
            model_oof[spec.name][target] = oof_pred
            model_test[spec.name][target] = test_pred

    target_best_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = base_oof[target].astype(float).to_numpy()
        candidates: dict[str, dict[str, Any]] = {
            "base": {"kind": "base", "loss": base_scores[target], "test_mean": float(base_test[target].mean())}
        }
        for spec in MODEL_SPECS:
            raw = model_oof[spec.name][target].astype(float).to_numpy()
            raw_test = model_test[spec.name][target].astype(float).to_numpy()
            raw_loss = float(log_loss(y, raw, labels=[0, 1]))
            candidates[f"raw_{spec.name}"] = {
                "kind": "raw",
                "model": spec.name,
                "loss": raw_loss,
                "test_mean": float(np.mean(raw_test)),
            }
            for weight in BLEND_WEIGHTS:
                pred = np.clip((1.0 - weight) * base_values + weight * raw, EPS, 1.0 - EPS)
                test_pred = np.clip(
                    (1.0 - weight) * base_test[target].astype(float).to_numpy() + weight * raw_test, EPS, 1.0 - EPS
                )
                candidates[f"blend_{spec.name}_{weight:g}"] = {
                    "kind": "blend",
                    "model": spec.name,
                    "weight": weight,
                    "loss": float(log_loss(y, pred, labels=[0, 1])),
                    "test_mean": float(np.mean(test_pred)),
                    "test_max_abs_delta": float(np.max(np.abs(test_pred - base_test[target].astype(float).to_numpy()))),
                }
        best_name, best_info = min(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "best": best_name,
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top10": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
        }
    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=173)
    parser.add_argument("--n-kernels", type=int, default=192)
    parser.add_argument("--seed", type=int, default=43017)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "id173_random_conv_sequence_diagnostic.json")
    args = parser.parse_args()
    report = run_diagnostic(args.base_id, args.n_kernels, args.seed, args.output)
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "n_total_channels": report["n_total_channels"],
        "n_random_conv_features": report["n_random_conv_features"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
