#!/usr/bin/env python3
"""Diagnose/promote a compact Q2 WiFi/BLE environment-token overlay.

This is intended for memory-safe remote execution on Kaggle. It uses only a
small label-free vocabulary of hashed WiFi/BLE environment tokens, then trains a
compact Q2-only LGBM overlay on top of an existing strict base artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_booster_compact_experiment import build_q2_booster_frames, load_base_artifact  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "Q2"
DEFAULT_BASE_ID = 116
RAW_CONTEXT_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
TOP_WIFI_TOKENS = 32
TOP_BLE_TOKENS = 24
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
WINDOWS = {
    "all12_24": (12.0, 24.0),
    "eve18_24": (18.0, 24.0),
    "late22_24": (22.0, 24.0),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_token_hash(token: str) -> str:
    return hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:10]


def safe_feature_token(token: str) -> str:
    return stable_token_hash(token)


def iter_wifi_records(records: object):
    if not isinstance(records, (list, np.ndarray)):
        return
    for item in records:
        if not isinstance(item, dict):
            continue
        token = str(item.get("bssid", "")).strip().lower()
        if not token:
            continue
        try:
            rssi = float(item.get("rssi", np.nan))
        except Exception:
            rssi = np.nan
        yield token, rssi


def iter_ble_records(records: object):
    if not isinstance(records, (list, np.ndarray)):
        return
    for item in records:
        if not isinstance(item, dict):
            continue
        address = str(item.get("address", "")).strip().lower()
        device_class = str(item.get("device_class", "unknown")).strip().lower()
        if not address:
            continue
        token = f"{address}|class={device_class}"
        try:
            rssi = float(item.get("rssi", np.nan))
        except Exception:
            rssi = np.nan
        yield token, rssi


def load_context_frame(name: str, raw_dir: Path = RAW_CONTEXT_DIR) -> pd.DataFrame:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw context file: {path.relative_to(ROOT)}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour_float"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    return df


def select_top_tokens(df: pd.DataFrame, value_col: str, kind: str, top_n: int) -> list[str]:
    counter: Counter[str] = Counter()
    iterator = iter_wifi_records if kind == "wifi" else iter_ble_records
    for records in df[value_col]:
        for token, _rssi in iterator(records) or []:
            counter[token] += 1
    return [token for token, _count in counter.most_common(top_n)]


def add_window_token_features(
    row: dict[str, float | object],
    group: pd.DataFrame,
    value_col: str,
    kind: str,
    top_tokens: list[str],
) -> None:
    token_to_idx = {token: idx for idx, token in enumerate(top_tokens)}
    iterator = iter_wifi_records if kind == "wifi" else iter_ble_records
    strong_threshold = -65.0 if kind == "wifi" else -70.0

    for window_name, (lo, hi) in WINDOWS.items():
        subset = group[(group["hour_float"] >= lo) & (group["hour_float"] < hi)]
        counts = np.zeros(len(top_tokens), dtype=float)
        strong_counts = np.zeros(len(top_tokens), dtype=float)
        max_rssi = np.full(len(top_tokens), np.nan, dtype=float)
        total_seen = 0.0
        total_strong = 0.0
        total_records = float(len(subset))
        other_seen = 0.0

        for records in subset[value_col]:
            for token, rssi in iterator(records) or []:
                total_seen += 1.0
                if np.isfinite(rssi) and rssi >= strong_threshold:
                    total_strong += 1.0
                idx = token_to_idx.get(token)
                if idx is None:
                    other_seen += 1.0
                    continue
                counts[idx] += 1.0
                if np.isfinite(rssi) and rssi >= strong_threshold:
                    strong_counts[idx] += 1.0
                if np.isfinite(rssi):
                    max_rssi[idx] = rssi if np.isnan(max_rssi[idx]) else max(max_rssi[idx], rssi)

        denom = total_seen + 1e-6
        positive = counts[counts > 0] / denom
        row[f"env_{kind}_{window_name}_scan_records"] = total_records
        row[f"env_{kind}_{window_name}_seen_total"] = np.log1p(total_seen)
        row[f"env_{kind}_{window_name}_strong_total"] = np.log1p(total_strong)
        row[f"env_{kind}_{window_name}_other_share"] = other_seen / denom
        row[f"env_{kind}_{window_name}_top_unique"] = float((counts > 0).sum())
        row[f"env_{kind}_{window_name}_top_entropy"] = float(-(positive * np.log(positive + 1e-9)).sum()) if len(positive) else 0.0

        for token, idx in token_to_idx.items():
            token_hash = safe_feature_token(token)
            prefix = f"env_{kind}_{window_name}_{idx:02d}_{token_hash}"
            row[f"{prefix}_seen"] = np.log1p(counts[idx])
            row[f"{prefix}_strong"] = np.log1p(strong_counts[idx])
            row[f"{prefix}_share"] = counts[idx] / denom
            row[f"{prefix}_rssi"] = 0.0 if np.isnan(max_rssi[idx]) else (max_rssi[idx] + 100.0) / 70.0


def build_environment_token_features(raw_dir: Path = RAW_CONTEXT_DIR) -> tuple[pd.DataFrame, dict]:
    wifi = load_context_frame("ch2025_mWifi.parquet", raw_dir=raw_dir)
    ble = load_context_frame("ch2025_mBle.parquet", raw_dir=raw_dir)
    top_wifi = select_top_tokens(wifi, "m_wifi", "wifi", TOP_WIFI_TOKENS)
    top_ble = select_top_tokens(ble, "m_ble", "ble", TOP_BLE_TOKENS)

    row_map: dict[tuple[str, object], dict[str, float | object]] = {}
    for (subject_id, date), group in wifi.groupby(["subject_id", "date"], sort=False):
        row = row_map.setdefault((str(subject_id), date), {"subject_id": subject_id, "date": date})
        add_window_token_features(row, group, "m_wifi", "wifi", top_wifi)
    for (subject_id, date), group in ble.groupby(["subject_id", "date"], sort=False):
        row = row_map.setdefault((str(subject_id), date), {"subject_id": subject_id, "date": date})
        add_window_token_features(row, group, "m_ble", "ble", top_ble)

    features = pd.DataFrame(row_map.values())
    top_token_report = {
        "wifi_hashes": [stable_token_hash(token) for token in top_wifi],
        "ble_hashes": [stable_token_hash(token) for token in top_ble],
        "top_wifi_count": len(top_wifi),
        "top_ble_count": len(top_ble),
        "privacy_note": "Raw BSSID/BLE addresses are used only transiently for feature construction; reports store stable hashes only.",
    }
    return features, top_token_report


def build_q2_env_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, test, sample_original = build_q2_booster_frames()
    env_daily, token_report = build_environment_token_features()
    env_daily = env_daily.copy()
    env_daily["date"] = pd.to_datetime(env_daily["date"]).dt.date
    train_env = train.copy()
    test_env = test.copy()
    train_env["date"] = pd.to_datetime(train_env["date"]).dt.date
    test_env["date"] = pd.to_datetime(test_env["date"]).dt.date
    train_env = train_env.merge(env_daily, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_env = test_env.merge(env_daily, on=["subject_id", "date"], how="left", validate="one_to_one")
    return train_env, test_env, sample_original, token_report


def env_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [col for col in train.columns if col.startswith("env_") and col in test.columns]


def compact_env_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=180,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=18,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=30.0,
        random_state=42,
        verbosity=-1,
    )


def fit_predict_env_model(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train, n_folds=5)
    y = train[TARGET].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train), dtype=float)
    has_prediction = np.zeros(len(train), dtype=bool)
    test_pred = np.zeros(len(test), dtype=float)

    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train.iloc[train_idx][feature_cols])
        x_valid = imputer.transform(train.iloc[valid_idx][feature_cols])
        x_test = imputer.transform(test[feature_cols])
        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(compact_env_model())
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q2 row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q2_env_token_overlay(base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original, token_report = build_q2_env_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    feature_cols = env_feature_columns(train, test)
    if not feature_cols:
        raise ValueError("No environment token feature columns were generated.")

    y = train[TARGET].astype(int).values
    base_candidate_name = f"id{base_id:03d}_base"
    raw_candidate_name = "env_wifi_ble_hash_tokens__lgbm_l7_lr025_n180_l30"
    raw_oof, raw_test = fit_predict_env_model(train, test, feature_cols)

    candidate_oof = {
        base_candidate_name: base_oof[TARGET].astype(float).to_numpy(),
        raw_candidate_name: raw_oof,
    }
    candidate_test = {
        base_candidate_name: base_test[TARGET].astype(float).to_numpy(),
        raw_candidate_name: raw_test,
    }
    candidate_details = {
        base_candidate_name: {"source": "base"},
        raw_candidate_name: {
            "feature_set": "env_wifi_ble_hash_tokens",
            "model": "lgbm_l7_lr025_n180_l30",
            "n_features": len(feature_cols),
            "blend": False,
        },
    }
    for weight in BLEND_WEIGHTS:
        blend_name = f"blend_base{1.0 - weight:.2f}_{raw_candidate_name}_{weight:.2f}"
        candidate_oof[blend_name] = np.clip((1.0 - weight) * candidate_oof[base_candidate_name] + weight * raw_oof, 1e-6, 1.0 - 1e-6)
        candidate_test[blend_name] = np.clip((1.0 - weight) * candidate_test[base_candidate_name] + weight * raw_test, 1e-6, 1.0 - 1e-6)
        candidate_details[blend_name] = {"blend_candidate": raw_candidate_name, "base_weight": 1.0 - weight, "candidate_weight": weight}

    candidate_scores = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())}
    folds = make_subject_hole_folds(train, n_folds=5)
    selected_q2_oof = np.zeros(len(train), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q2_oof[valid_idx] = candidate_oof[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_fold_log_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_losses = {name: log_loss(y, values, labels=[0, 1]) for name, values in candidate_oof.items()}
    full_best_name, full_best_loss = min(full_losses.items(), key=lambda item: item[1])

    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[TARGET] = np.clip(selected_q2_oof, 1e-6, 1.0 - 1e-6)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], 1e-6, 1.0 - 1e-6)

    target_scores = {target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1])) for target in TARGETS}
    selector_info = {
        "base_source": base_source,
        "feature_family": "wifi_ble_environment_hash_tokens",
        "token_report": token_report,
        "n_features": len(feature_cols),
        "candidate_pool_size": len(candidate_oof),
        "q2_candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(full_losses.items())},
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q2. Q2 validation rows receive "
            "the compact env-token candidate selected using only each outer fold's training rows. Final test Q2 "
            "uses the candidate with the best full-training OOF log loss."
        ),
        "contamination_note": (
            "Top WiFi/BLE token vocabularies are selected from unlabeled train/test context rows. This is "
            "label-free transductive feature construction and uses no target labels."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q2_env_token_compact_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q2_env_token_compact_diagnostic_id{base_id:03d}_{suffix}.json"
    path.write_text(
        json.dumps({"base_id": base_id, "local_cv": local_cv, "target_scores": target_scores, "selector_info": selector_info}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_repro_notebook(notebook_path: Path, submission_path: Path, metadata_path: Path, experiment_id: int, slug: str, local_cv: float, base_id: int) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: Q2 environment-token compact overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q2 target and replaces Q2 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q2_env_token_compact_experiment.py --promote "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nDo not submit before reviewing metadata and audit reports."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(experiment_id: int, submission_path: Path, notebook_path: Path, local_cv: float, notes: str, base_id: int) -> None:
    log_path = SUBMISSIONS_DIR / "submission_log.csv"
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    if (log_df["id"].astype(str) == str(experiment_id)).any():
        raise ValueError(f"submission_log.csv already has id={experiment_id}")
    row = dict.fromkeys(log_df.columns, "")
    row.update(
        {
            "id": str(experiment_id),
            "date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "submission_file": submission_path.name,
            "model_description": f"Strict Q2 env-token compact overlay on ID{base_id:03d}",
            "local_cv_score": f"{local_cv:.6f}",
            "public_lb_score": "",
            "is_submitted": "False",
            "notes": notes,
            "is_submitted_bool": "False",
            "local_cv_score_num": f"{local_cv:.6f}",
            "public_lb_score_num": "",
            "cv_lb_gap": "",
            "artifact_exists": "True",
            "status_clean": "generated_unsubmitted",
            "risk_reason": "Generated locally but not submitted/scored.",
            "safe_for_default_ensemble": "False",
            "artifact_role": "submission",
            "duplicate_file_count": "1",
            "notebook_file": str(notebook_path.relative_to(ROOT)),
        }
    )
    pd.concat([log_df, pd.DataFrame([row])], ignore_index=True).to_csv(log_path, index=False)


def promote_artifacts(
    experiment_id: int,
    slug: str,
    base_id: int,
    oof: pd.DataFrame,
    test_pred: pd.DataFrame,
    target_scores: dict[str, float],
    selector_info: dict,
    sample_original: pd.DataFrame,
) -> dict:
    if experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")
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
        "experiment_id": experiment_id,
        "slug": slug,
        "method": "Q2 compact WiFi/BLE environment-token overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with a Q2-only compact LGBM blend over hashed WiFi/BLE environment tokens.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id)
    notes = (
        f"Strict Q2-only compact WiFi/BLE environment-token overlay on ID{base_id:03d}. Uses a small label-free "
        "hashed token vocabulary from raw mWifi/mBle context; no raw addresses are stored in metadata and no DACON submission was made."
    )
    append_submission_log(experiment_id, submission_path, notebook_path, local_cv, notes, base_id)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q2_env_token_compact_overlay")
    args = parser.parse_args()

    oof, test_pred, target_scores, selector_info, sample_original = predict_q2_env_token_overlay(base_id=args.base_id)
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "q2_candidate_scores": selector_info["q2_candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = promote_artifacts(args.experiment_id, args.slug, args.base_id, oof, test_pred, target_scores, selector_info, sample_original)
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
