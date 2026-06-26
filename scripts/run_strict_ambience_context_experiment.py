#!/usr/bin/env python3
"""Diagnose/promote target-specific semantic ambience-context overlays."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_booster_compact_experiment import load_base_artifact  # noqa: E402
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
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_BASE_ID = 118
RAW_CONTEXT_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
TARGET_CHOICES = ["Q2", "Q3", "S4"]
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]
WINDOWS = {
    "day12_18": (12.0, 18.0, 0),
    "eve18_22": (18.0, 22.0, 0),
    "late22_24": (22.0, 24.0, 0),
    "sleep00_03": (0.0, 3.0, -1),
    "sleep03_06": (3.0, 6.0, -1),
    "wake06_09": (6.0, 9.0, -1),
}
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "quiet": ("silence",),
    "speech": (
        "speech",
        "conversation",
        "narration",
        "monologue",
        "babbling",
        "shout",
        "laughter",
        "crying",
        "sobbing",
        "whimper",
        "child speech",
    ),
    "music_media": (
        "music",
        "singing",
        "television",
        "radio",
        "musical instrument",
        "guitar",
        "keyboard (musical)",
        "lullaby",
    ),
    "traffic_vehicle": (
        "vehicle",
        "motor vehicle",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "train",
        "rail",
        "traffic",
        "aircraft",
        "bicycle",
        "boat",
    ),
    "snore_breath": (
        "snoring",
        "breathing",
        "snort",
        "gasp",
        "cough",
        "sneeze",
        "heart sounds",
        "heartbeat",
    ),
    "mechanical_noise": (
        "mechanical",
        "mechanisms",
        "fan",
        "air conditioning",
        "hiss",
        "hum",
        "buzz",
        "mains hum",
        "printer",
        "microwave",
        "electric shaver",
        "engine",
        "tools",
    ),
    "household_contact": (
        "door",
        "typing",
        "keyboard",
        "typewriter",
        "dishes",
        "cutlery",
        "glass",
        "keys",
        "chink",
        "clink",
        "tick",
        "clock",
        "walk",
        "footsteps",
        "shuffle",
        "scissors",
        "camera",
    ),
    "alarm_bell": ("alarm", "bell", "beep", "bleep", "chime", "telephone", "ringtone", "ping"),
    "outdoor_nature": (
        "outside",
        "rural",
        "urban",
        "wind",
        "rain",
        "water",
        "ocean",
        "bird",
        "cricket",
        "insect",
        "animal",
        "dog",
        "cat",
        "frog",
    ),
    "impact_burst": (
        "explosion",
        "fireworks",
        "firecracker",
        "gunshot",
        "burst",
        "pop",
        "rattle",
        "crackle",
        "hammer",
        "breaking",
    ),
    "indoor_room": ("inside", "room", "public space", "large room", "small room"),
}
CATEGORY_NAMES = [*CATEGORY_KEYWORDS.keys(), "other"]


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def ambience_category(label: object) -> str:
    text = normalize_label(label)
    if not text:
        return "other"
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def iter_ambience_items(value: object):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return
    for item in value:
        if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2:
            try:
                yield str(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue


def scan_category_scores(value: object) -> tuple[dict[str, float], float]:
    scores = dict.fromkeys(CATEGORY_NAMES, 0.0)
    raw_probs: list[float] = []
    for label, probability in iter_ambience_items(value):
        category = ambience_category(label)
        probability = float(np.clip(probability, 0.0, 1.0))
        scores[category] = max(scores[category], probability)
        raw_probs.append(probability)
    if not raw_probs:
        return scores, 0.0
    arr = np.asarray(raw_probs, dtype=float)
    probs = arr / (arr.sum() + 1e-9)
    entropy = float(-(probs * np.log(probs + 1e-9)).sum())
    return scores, entropy


def build_ambience_context_features(raw_dir: Path = RAW_CONTEXT_DIR) -> tuple[pd.DataFrame, dict]:
    path = raw_dir / "ch2025_mAmbience.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing ambience file: {path.relative_to(ROOT)}")
    ambience = pd.read_parquet(path)
    ambience = ambience.copy()
    ambience["timestamp"] = pd.to_datetime(ambience["timestamp"])
    ambience["date"] = ambience["timestamp"].dt.date
    ambience["hour_float"] = ambience["timestamp"].dt.hour + ambience["timestamp"].dt.minute / 60.0

    accum: dict[tuple[str, object, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    top_examples: dict[str, dict[str, float]] = {category: {} for category in CATEGORY_NAMES}
    label_counts: dict[str, int] = defaultdict(int)
    for record in ambience.itertuples(index=False):
        hour = float(record.hour_float)
        matched_windows = [
            (window_name, day_offset)
            for window_name, (lo, hi, day_offset) in WINDOWS.items()
            if lo <= hour < hi
        ]
        if not matched_windows:
            continue
        scores, entropy = scan_category_scores(record.m_ambience)
        for label, probability in iter_ambience_items(record.m_ambience):
            category = ambience_category(label)
            label_counts[label] += 1
            current = top_examples[category].get(label, 0.0)
            top_examples[category][label] = max(current, probability)

        for window_name, day_offset in matched_windows:
            lifelog_date = pd.Timestamp(record.date) + pd.Timedelta(days=day_offset)
            key = (record.subject_id, lifelog_date.date(), window_name)
            bucket = accum[key]
            bucket["n_scans"] += 1.0
            bucket["entropy_sum"] += entropy
            bucket["nonquiet_sum"] += max(0.0, 1.0 - scores.get("quiet", 0.0))
            bucket["arousal_sum"] += max(
                scores.get("speech", 0.0),
                scores.get("music_media", 0.0),
                scores.get("traffic_vehicle", 0.0),
                scores.get("mechanical_noise", 0.0),
                scores.get("alarm_bell", 0.0),
                scores.get("impact_burst", 0.0),
            )
            for category, score in scores.items():
                bucket[f"{category}_sum"] += score
                bucket[f"{category}_max"] = max(bucket[f"{category}_max"], score)
                bucket[f"{category}_gt20"] += float(score >= 0.20)
                bucket[f"{category}_gt50"] += float(score >= 0.50)

    by_day: dict[tuple[str, object], dict[str, float | object]] = defaultdict(dict)
    for (subject_id, date, window_name), bucket in accum.items():
        row = by_day[(subject_id, date)]
        row["subject_id"] = subject_id
        row["date"] = date
        n_scans = max(bucket.get("n_scans", 0.0), 1.0)
        row[f"amb_{window_name}_n_scans"] = bucket.get("n_scans", 0.0)
        row[f"amb_{window_name}_entropy_mean"] = bucket.get("entropy_sum", 0.0) / n_scans
        row[f"amb_{window_name}_nonquiet_mean"] = bucket.get("nonquiet_sum", 0.0) / n_scans
        row[f"amb_{window_name}_arousal_mean"] = bucket.get("arousal_sum", 0.0) / n_scans
        for category in CATEGORY_NAMES:
            row[f"amb_{window_name}_{category}_mean"] = bucket.get(f"{category}_sum", 0.0) / n_scans
            row[f"amb_{window_name}_{category}_max"] = bucket.get(f"{category}_max", 0.0)
            row[f"amb_{window_name}_{category}_gt20_ratio"] = bucket.get(f"{category}_gt20", 0.0) / n_scans
            row[f"amb_{window_name}_{category}_gt50_ratio"] = bucket.get(f"{category}_gt50", 0.0) / n_scans

    features = pd.DataFrame(list(by_day.values()))
    if features.empty:
        raise ValueError("No ambience context features were generated.")

    feature_cols = [col for col in features.columns if col.startswith("amb_")]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    report = {
        "raw_file": str(path.relative_to(ROOT)),
        "raw_shape": list(ambience.shape),
        "feature_shape": list(features.shape),
        "n_feature_columns": len(feature_cols),
        "windows": WINDOWS,
        "category_keywords": CATEGORY_KEYWORDS,
        "top_examples_by_category": {
            category: [
                label for label, _score in sorted(examples.items(), key=lambda item: item[1], reverse=True)[:10]
            ]
            for category, examples in top_examples.items()
        },
        "top_raw_labels": [label for label, _count in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[:30]],
    }
    return features, report


def merge_ambience_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    amb, report = build_ambience_context_features()
    amb = amb.copy()
    amb["date"] = pd.to_datetime(amb["date"]).dt.date
    train_aug = train.copy()
    test_aug = test.copy()
    train_aug["date"] = pd.to_datetime(train_aug["date"]).dt.date
    test_aug["date"] = pd.to_datetime(test_aug["date"]).dt.date
    train_aug = train_aug.merge(amb, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_aug = test_aug.merge(amb, on=["subject_id", "date"], how="left", validate="one_to_one")
    amb_cols = [col for col in train_aug.columns if col.startswith("amb_") and col in test_aug.columns]
    train_aug[amb_cols] = train_aug[amb_cols].fillna(0.0)
    test_aug[amb_cols] = test_aug[amb_cols].fillna(0.0)
    train_aug, test_aug, rel_cols = add_subject_relative_ambience_features(train_aug, test_aug, amb_cols)
    report["n_abs_features"] = len(amb_cols)
    report["n_relative_features"] = len(rel_cols)
    return train_aug, test_aug, sample_original, report


def add_subject_relative_ambience_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    compact_cols = [
        col
        for col in source_cols
        if col.endswith(("_mean", "_max", "_ratio")) and not col.endswith("_n_scans")
    ]
    combined = pd.concat(
        [
            train[["subject_id", *compact_cols]].assign(_part="train"),
            test[["subject_id", *compact_cols]].assign(_part="test"),
        ],
        ignore_index=True,
    )
    relative = pd.DataFrame(index=combined.index)
    relative["_part"] = combined["_part"]
    for col in compact_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        grouped = values.groupby(combined["subject_id"])
        median = grouped.transform("median")
        q25 = grouped.transform(lambda series: series.quantile(0.25))
        q75 = grouped.transform(lambda series: series.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        rank = values.groupby(combined["subject_id"]).rank(pct=True, method="average")
        safe_col = col.replace("amb_", "", 1)
        relative[f"rel_amb_iqr_{safe_col}"] = ((values - median) / (iqr + 1e-6)).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)
        relative[f"rel_amb_rank_{safe_col}"] = rank.fillna(0.5)

    rel_train = relative[relative["_part"] == "train"].drop(columns=["_part"]).reset_index(drop=True)
    rel_test = relative[relative["_part"] == "test"].drop(columns=["_part"]).reset_index(drop=True)
    rel_cols = list(rel_train.columns)
    train_out = pd.concat([train.reset_index(drop=True), rel_train], axis=1)
    test_out = pd.concat([test.reset_index(drop=True), rel_test], axis=1)
    return train_out, test_out, rel_cols


def ambience_feature_sets(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict[str, list[str]]:
    abs_cols = [col for col in train.columns if col.startswith("amb_") and col in test.columns]
    rel_cols = [col for col in train.columns if col.startswith("rel_amb_") and col in test.columns]
    sleep_terms = ("sleep00_03", "sleep03_06", "wake06_09", "late22_24")
    q_terms = ("day12_18", "eve18_22", "late22_24")
    if target == "S4":
        focused_abs = [col for col in abs_cols if any(term in col for term in sleep_terms)]
        focused_rel = [col for col in rel_cols if any(term in col for term in sleep_terms)]
    else:
        focused_abs = [col for col in abs_cols if any(term in col for term in q_terms)]
        focused_rel = [col for col in rel_cols if any(term in col for term in q_terms)]
    return {
        "amb_focused_abs": focused_abs,
        "amb_focused_abs_rel": [*focused_abs, *focused_rel],
        "amb_all_abs": abs_cols,
    }


def compact_model_pool() -> dict[str, object]:
    return {
        "lgbm_l7_lr02_n180_l30": LGBMClassifier(
            n_estimators=180,
            learning_rate=0.02,
            num_leaves=7,
            max_depth=3,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_lambda=30.0,
            random_state=42,
            verbosity=-1,
        ),
        "lgbm_l15_lr015_n220_l50": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.65,
            reg_lambda=50.0,
            random_state=43,
            verbosity=-1,
        ),
        "lr_l2_c02": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=1000, solver="liblinear", random_state=42),
        ),
    }


def compact_candidate_pairs() -> list[tuple[str, str]]:
    return [
        ("amb_focused_abs", "lgbm_l7_lr02_n180_l30"),
        ("amb_focused_abs_rel", "lgbm_l15_lr015_n220_l50"),
        ("amb_all_abs", "lr_l2_c02"),
    ]


def fit_predict_one_candidate(
    model_proto: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train, n_folds=5)
    y = train[target].astype(int).reset_index(drop=True)
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
            model = clone(model_proto)
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True
    if not has_prediction.all():
        raise ValueError(f"Folds did not produce predictions for every {target} row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_ambience_overlay(
    base_id: int,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original, ambience_report = merge_ambience_features()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    feature_sets = ambience_feature_sets(train, test, target)
    model_pool = compact_model_pool()

    y = train[target].astype(int).to_numpy()
    base_candidate_name = f"id{base_id:03d}_base"
    candidate_oof = {base_candidate_name: base_oof[target].astype(float).to_numpy()}
    candidate_test = {base_candidate_name: base_test[target].astype(float).to_numpy()}
    candidate_details = {base_candidate_name: {"source": "base"}}
    for feature_set_name, model_name in compact_candidate_pairs():
        feature_cols = feature_sets[feature_set_name]
        if not feature_cols:
            raise ValueError(f"No ambience features generated for {feature_set_name}.")
        raw_oof, raw_test = fit_predict_one_candidate(model_pool[model_name], train, test, feature_cols, target)
        raw_name = f"{feature_set_name}__{model_name}"
        candidate_oof[raw_name] = raw_oof
        candidate_test[raw_name] = raw_test
        candidate_details[raw_name] = {
            "feature_set": feature_set_name,
            "model": model_name,
            "n_features": len(feature_cols),
            "blend": False,
        }
        for weight in BLEND_WEIGHTS:
            blend_name = f"blend_base{1.0 - weight:.2f}_{raw_name}_{weight:.2f}"
            candidate_oof[blend_name] = np.clip(
                (1.0 - weight) * candidate_oof[base_candidate_name] + weight * raw_oof,
                1e-6,
                1.0 - 1e-6,
            )
            candidate_test[blend_name] = np.clip(
                (1.0 - weight) * candidate_test[base_candidate_name] + weight * raw_test,
                1e-6,
                1.0 - 1e-6,
            )
            candidate_details[blend_name] = {
                "blend_candidate": raw_name,
                "base_weight": 1.0 - weight,
                "candidate_weight": weight,
            }

    candidate_scores = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())}
    folds = make_subject_hole_folds(train, n_folds=5)
    selected_target_oof = np.zeros(len(train), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_target_oof[valid_idx] = candidate_oof[best_name][valid_idx]
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
    oof[target] = np.clip(selected_target_oof, 1e-6, 1.0 - 1e-6)
    test_pred[target] = np.clip(candidate_test[full_best_name], 1e-6, 1.0 - 1e-6)
    target_scores = {
        column: float(log_loss(train[column].astype(int), oof[column].astype(float), labels=[0, 1]))
        for column in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "semantic_ambience_context",
        "ambience_report": ambience_report,
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "candidate_pool_size": len(candidate_oof),
        "candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {target: nested_choices},
        "full_train_test_choices": {
            target: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(full_losses.items())},
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except {target}. "
            f"{target} validation rows receive the semantic ambience candidate selected using only each "
            f"outer fold's training rows. Final test {target} uses the candidate with the best full-training "
            "OOF log loss."
        ),
        "contamination_note": (
            "Ambience labels are derived from unlabeled raw mAmbience sensor rows and compressed by a "
            "hand-written semantic dictionary before target fitting. Subject-relative features use no labels."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_diagnostic_report(base_id: int, target: str, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_{target.lower()}_ambience_context_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_{target.lower()}_ambience_context_diagnostic_id{base_id:03d}_{suffix}.json"
    path.write_text(
        json.dumps(
            {
                "base_id": base_id,
                "target": target,
                "local_cv": local_cv,
                "target_scores": target_scores,
                "selector_info": selector_info,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
    target: str,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: {target} semantic ambience-context overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base ID: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This artifact changes one target only by using semantic AudioSet ambience context features. "
            "OOF candidate choice is fold-nested; no DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/run_strict_ambience_context_experiment.py --target {target} "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata and audit reports before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def promote_artifacts(
    experiment_id: int,
    slug: str,
    base_id: int,
    target: str,
    oof: pd.DataFrame,
    test_pred: pd.DataFrame,
    target_scores: dict[str, float],
    selector_info: dict,
    sample_original: pd.DataFrame,
) -> dict:
    if experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")
    base_target_score = float(selector_info["candidate_scores"][f"id{base_id:03d}_base"])
    if target_scores[target] >= base_target_score:
        raise ValueError(
            f"Refusing to promote non-improving {target} ambience overlay: "
            f"{target_scores[target]} >= {base_target_score}"
        )
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
        "method": f"{target} semantic ambience-context overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": f"Subject-Hole CV with {target}-only fold-nested semantic ambience candidate selection.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id, target)
    notes = (
        f"Strict {target}-only semantic ambience-context overlay on ID{base_id:03d}. "
        "Compresses raw mAmbience AudioSet labels into auditable semantic categories and time windows; "
        "no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict {target} semantic ambience-context overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGET_CHOICES, required=True)
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug")
    args = parser.parse_args()

    slug = args.slug or f"{args.target.lower()}_ambience_context_overlay"
    oof, test_pred, target_scores, selector_info, sample_original = predict_ambience_overlay(args.base_id, args.target)
    local_cv = float(np.mean(list(target_scores.values())))
    diagnostic_path = write_diagnostic_report(args.base_id, args.target, local_cv, target_scores, selector_info)
    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "target": args.target,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "candidate_scores": selector_info["candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = promote_artifacts(
            experiment_id=args.experiment_id,
            slug=slug,
            base_id=args.base_id,
            target=args.target,
            oof=oof,
            test_pred=test_pred,
            target_scores=target_scores,
            selector_info=selector_info,
            sample_original=sample_original,
        )
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
