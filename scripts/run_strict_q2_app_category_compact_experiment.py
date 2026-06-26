#!/usr/bin/env python3
"""Diagnose/promote a compact Q2 semantic app-category overlay.

The top-app-token overlay improved Q2, while broader token and relative variants
overfit. This experiment compresses raw app names into hand-auditable semantic
categories and time-window routine features before fitting a compact Q2-only
candidate on top of an existing strict base artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
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
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "Q2"
DEFAULT_BASE_ID = 116
RAW_CONTEXT_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
WINDOWS = {
    "all12_24": (12.0, 24.0),
    "aft12_18": (12.0, 18.0),
    "eve18_22": (18.0, 22.0),
    "late22_24": (22.0, 24.0),
}
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "messaging": (
        "카카오톡",
        "kakaotalk",
        "kakao talk",
        "telegram",
        "텔레그램",
        "message",
        "메시지",
        "문자",
        "line",
        "라인",
        "discord",
        "messenger",
    ),
    "video": (
        "youtube",
        "유튜브",
        "netflix",
        "넷플릭스",
        "tving",
        "티빙",
        "wavve",
        "웨이브",
        "coupang play",
        "쿠팡플레이",
        "twitch",
        "afreeca",
        "아프리카",
        "치지직",
    ),
    "browser_search_news": (
        "naver",
        "네이버",
        "chrome",
        "크롬",
        "google",
        "구글",
        "daum",
        "다음",
        "samsung internet",
        "인터넷",
        "browser",
        "뉴스",
        "news",
    ),
    "social": (
        "instagram",
        "인스타",
        "facebook",
        "페이스북",
        "twitter",
        "트위터",
        "threads",
        "tiktok",
        "틱톡",
        "x ",
    ),
    "game": ("game", "게임", "roblox", "로블록스", "원신", "붕괴", "play 게임", "play games"),
    "commerce_food": (
        "coupang",
        "쿠팡",
        "11번가",
        "gmarket",
        "옥션",
        "쇼핑",
        "배민",
        "배달",
        "요기요",
        "컬리",
        "당근",
        "마켓",
    ),
    "finance_reward": (
        "cashwalk",
        "캐시워크",
        "toss",
        "토스",
        "bank",
        "은행",
        "pay",
        "페이",
        "증권",
        "카드",
        "kb",
        "신한",
        "우리",
        "하나",
    ),
    "study_work": (
        "classroom",
        "zoom",
        "teams",
        "slack",
        "notion",
        "docs",
        "word",
        "excel",
        "powerpoint",
        "gmail",
        "mail",
        "메일",
        "calendar",
        "캘린더",
        "school",
        "학교",
    ),
    "music_audio": (
        "music",
        "음악",
        "spotify",
        "melon",
        "멜론",
        "flo",
        "genie",
        "지니",
        "podcast",
        "팟캐스트",
        "팟빵",
    ),
    "religion": ("성경", "bible", "찬송", "교회", "qt", "말씀"),
    "health_fitness": ("health", "헬스", "fit", "fitness", "건강", "운동", "만보기", "걸음"),
    "map_transport": ("map", "maps", "지도", "카카오맵", "티맵", "tmap", "지하철", "버스", "taxi", "택시", "내비"),
    "photo_camera": ("camera", "카메라", "gallery", "갤러리", "photo", "사진", "album", "앨범"),
    "system_home": (
        "one ui",
        "홈",
        "launcher",
        "system",
        "설정",
        "settings",
        "play services",
        "google play",
        "samsung",
    ),
}
CATEGORY_NAMES = [*CATEGORY_KEYWORDS.keys(), "other"]
DOPAMINE_CATEGORIES = {"video", "social", "game", "music_audio"}
UTILITY_CATEGORIES = {"messaging", "browser_search_news", "study_work", "map_transport", "finance_reward"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_app_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def app_category(app_name: object) -> str:
    text = normalize_app_name(app_name)
    if not text:
        return "other"
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            return category
    return "other"


def load_usage_frame(raw_dir: Path = RAW_CONTEXT_DIR) -> pd.DataFrame:
    path = raw_dir / "ch2025_mUsageStats.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing usage stats file: {path.relative_to(ROOT)}")
    usage = pd.read_parquet(path)
    usage = usage.copy()
    usage["timestamp"] = pd.to_datetime(usage["timestamp"])
    usage["date"] = usage["timestamp"].dt.date
    usage["hour_float"] = usage["timestamp"].dt.hour + usage["timestamp"].dt.minute / 60.0
    return usage


def build_app_category_features(raw_dir: Path = RAW_CONTEXT_DIR) -> tuple[pd.DataFrame, dict]:
    usage = load_usage_frame(raw_dir=raw_dir)
    category_totals = dict.fromkeys(CATEGORY_NAMES, 0.0)
    top_examples: dict[str, dict[str, float]] = {category: {} for category in CATEGORY_NAMES}

    rows: list[dict[str, float | object]] = []
    for (subject_id, date), group in usage.groupby(["subject_id", "date"], sort=False):
        row: dict[str, float | object] = {"subject_id": subject_id, "date": date}
        for window_name, (lo, hi) in WINDOWS.items():
            subset = group[(group["hour_float"] >= lo) & (group["hour_float"] < hi)]
            minutes_by_category = dict.fromkeys(CATEGORY_NAMES, 0.0)
            events_by_category = dict.fromkeys(CATEGORY_NAMES, 0.0)
            first_hour = dict.fromkeys(CATEGORY_NAMES, np.nan)
            last_hour = dict.fromkeys(CATEGORY_NAMES, np.nan)
            total_minutes = 0.0
            total_events = 0.0
            active_scan_rows = 0.0

            for _, record in subset.iterrows():
                apps = record["m_usage_stats"]
                if not isinstance(apps, (list, np.ndarray)):
                    continue
                row_total = 0.0
                hour = float(record["hour_float"])
                for app in apps:
                    if not isinstance(app, dict):
                        continue
                    minutes = float(app.get("total_time", 0) or 0) / 60000.0
                    if minutes <= 0:
                        continue
                    category = app_category(app.get("app_name", ""))
                    app_name = normalize_app_name(app.get("app_name", "unknown"))[:60]
                    minutes_by_category[category] += minutes
                    events_by_category[category] += 1.0
                    category_totals[category] += minutes
                    if app_name:
                        top_examples[category][app_name] = top_examples[category].get(app_name, 0.0) + minutes
                    first_hour[category] = hour if np.isnan(first_hour[category]) else min(first_hour[category], hour)
                    last_hour[category] = hour if np.isnan(last_hour[category]) else max(last_hour[category], hour)
                    total_minutes += minutes
                    total_events += 1.0
                    row_total += minutes
                if row_total > 0:
                    active_scan_rows += 1.0

            denom = total_minutes + 1e-6
            shares = np.array([minutes_by_category[category] / denom for category in CATEGORY_NAMES])
            positive = shares[shares > 0]
            dopamine_minutes = sum(minutes_by_category[category] for category in DOPAMINE_CATEGORIES)
            utility_minutes = sum(minutes_by_category[category] for category in UTILITY_CATEGORIES)
            row[f"appcat_{window_name}_total_min"] = np.log1p(total_minutes)
            row[f"appcat_{window_name}_n_events"] = np.log1p(total_events)
            row[f"appcat_{window_name}_active_scans"] = np.log1p(active_scan_rows)
            row[f"appcat_{window_name}_entropy"] = float(-(positive * np.log(positive + 1e-9)).sum()) if positive.size else 0.0
            row[f"appcat_{window_name}_top_share"] = float(shares.max()) if shares.size else 0.0
            row[f"appcat_{window_name}_dopamine_share"] = dopamine_minutes / denom
            row[f"appcat_{window_name}_utility_share"] = utility_minutes / denom
            row[f"appcat_{window_name}_dopamine_minus_utility"] = (dopamine_minutes - utility_minutes) / denom
            for category in CATEGORY_NAMES:
                prefix = f"appcat_{window_name}_{category}"
                row[f"{prefix}_min"] = np.log1p(minutes_by_category[category])
                row[f"{prefix}_share"] = minutes_by_category[category] / denom
                row[f"{prefix}_events"] = np.log1p(events_by_category[category])
                row[f"{prefix}_first_hour"] = 0.0 if np.isnan(first_hour[category]) else first_hour[category] / 24.0
                row[f"{prefix}_last_hour"] = 0.0 if np.isnan(last_hour[category]) else last_hour[category] / 24.0
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    category_report = {
        "category_totals_min": {category: float(value) for category, value in sorted(category_totals.items())},
        "top_examples_by_category": {
            category: [name for name, _value in sorted(examples.items(), key=lambda item: -item[1])[:8]]
            for category, examples in sorted(top_examples.items())
        },
        "category_keywords": {category: list(keywords) for category, keywords in CATEGORY_KEYWORDS.items()},
        "feature_note": "Raw app names are mapped to auditable semantic categories; no app-name one-hot columns are used.",
    }
    return feature_frame, category_report


def add_subject_relative_category_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    combined = pd.concat(
        [
            train[["subject_id", *source_cols]].assign(_part="train"),
            test[["subject_id", *source_cols]].assign(_part="test"),
        ],
        ignore_index=True,
    )
    relative = pd.DataFrame(index=combined.index)
    relative["_part"] = combined["_part"]
    compact_cols = [col for col in source_cols if col.endswith(("_share", "_total_min", "_dopamine_share", "_utility_share", "_entropy", "_top_share"))]
    for col in compact_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        grouped = values.groupby(combined["subject_id"])
        median = grouped.transform("median")
        q25 = grouped.transform(lambda series: series.quantile(0.25))
        q75 = grouped.transform(lambda series: series.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        rank = values.groupby(combined["subject_id"]).rank(pct=True, method="average")
        safe_col = col.replace("appcat_", "", 1)
        relative[f"rel_appcat_iqr_{safe_col}"] = ((values - median) / (iqr + 1e-6)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        relative[f"rel_appcat_rank_{safe_col}"] = rank.fillna(0.5)

    rel_train = relative[relative["_part"] == "train"].drop(columns=["_part"]).reset_index(drop=True)
    rel_test = relative[relative["_part"] == "test"].drop(columns=["_part"]).reset_index(drop=True)
    rel_cols = list(rel_train.columns)
    train_out = pd.concat([train.reset_index(drop=True), rel_train], axis=1)
    test_out = pd.concat([test.reset_index(drop=True), rel_test], axis=1)
    return train_out, test_out, rel_cols


def build_q2_app_category_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    appcat_daily, category_report = build_app_category_features()
    appcat_daily = appcat_daily.copy()
    appcat_daily["date"] = pd.to_datetime(appcat_daily["date"]).dt.date
    train_aug = train.copy()
    test_aug = test.copy()
    train_aug["date"] = pd.to_datetime(train_aug["date"]).dt.date
    test_aug["date"] = pd.to_datetime(test_aug["date"]).dt.date
    train_aug = train_aug.merge(appcat_daily, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_aug = test_aug.merge(appcat_daily, on=["subject_id", "date"], how="left", validate="one_to_one")
    appcat_cols = [col for col in train_aug.columns if col.startswith("appcat_") and col in test_aug.columns]
    train_aug, test_aug, rel_cols = add_subject_relative_category_features(train_aug, test_aug, appcat_cols)
    category_report["n_appcat_features"] = len(appcat_cols)
    category_report["n_relative_features"] = len(rel_cols)
    return train_aug, test_aug, sample_original, category_report


def app_category_feature_sets(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    appcat_cols = [col for col in train.columns if col.startswith("appcat_") and col in test.columns]
    rel_cols = [col for col in train.columns if col.startswith("rel_appcat_") and col in test.columns]
    return {
        "appcat_abs": appcat_cols,
        "appcat_rel": rel_cols,
        "appcat_abs_rel": [*appcat_cols, *rel_cols],
    }


def compact_model_pool() -> dict[str, object]:
    return {
        "lgbm_l7_lr025_n160_l20": LGBMClassifier(
            n_estimators=160,
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_lambda=20.0,
            random_state=42,
            verbosity=-1,
        ),
        "lgbm_l7_lr015_n220_l40": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.65,
            reg_lambda=40.0,
            random_state=43,
            verbosity=-1,
        ),
        "lr_l2_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=1000, solver="liblinear", random_state=42),
        ),
    }


def compact_candidate_pairs() -> list[tuple[str, str]]:
    return [
        ("appcat_abs", "lgbm_l7_lr025_n160_l20"),
        ("appcat_abs_rel", "lgbm_l7_lr015_n220_l40"),
        ("appcat_abs", "lr_l2_c03"),
    ]


def fit_predict_one_candidate(
    model_proto: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
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
            model = clone(model_proto)
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q2 row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q2_app_category_overlay(base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original, category_report = build_q2_app_category_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    feature_sets = app_category_feature_sets(train, test)
    model_pool = compact_model_pool()

    y = train[TARGET].astype(int).values
    base_candidate_name = f"id{base_id:03d}_base"
    candidate_oof = {base_candidate_name: base_oof[TARGET].astype(float).to_numpy()}
    candidate_test = {base_candidate_name: base_test[TARGET].astype(float).to_numpy()}
    candidate_details = {base_candidate_name: {"source": "base"}}

    for feature_set_name, model_name in compact_candidate_pairs():
        feature_cols = feature_sets[feature_set_name]
        if not feature_cols:
            raise ValueError(f"No features generated for {feature_set_name}.")
        raw_oof, raw_test = fit_predict_one_candidate(model_pool[model_name], train, test, feature_cols)
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

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "semantic_app_category_usage",
        "category_report": category_report,
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
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
            "the compact semantic app-category candidate selected using only each outer fold's training rows. "
            "Final test Q2 uses the candidate with the best full-training OOF log loss."
        ),
        "contamination_note": (
            "App category features are constructed from unlabeled train/test mUsageStats app names and durations. "
            "The category dictionary is hand-written before target fitting and uses no target labels."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q2_app_category_compact_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q2_app_category_compact_diagnostic_id{base_id:03d}_{suffix}.json"
    path.write_text(
        json.dumps({"base_id": base_id, "local_cv": local_cv, "target_scores": target_scores, "selector_info": selector_info}, indent=2, ensure_ascii=False),
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
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: Q2 semantic app-category compact overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q2 target and replaces Q2 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q2_app_category_compact_experiment.py --promote "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nDo not submit before reviewing metadata and audit reports."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
    base_id: int,
) -> None:
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
            "model_description": f"Strict Q2 semantic app-category compact overlay on ID{base_id:03d}",
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
        "method": "Q2 compact semantic app-category overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with a Q2-only compact blend over semantic app-category features.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id)
    notes = (
        f"Strict Q2-only compact semantic app-category overlay on ID{base_id:03d}. Compresses raw mUsageStats "
        "app names into auditable categories and time windows; no DACON submission was made."
    )
    append_submission_log(experiment_id, submission_path, notebook_path, local_cv, notes, base_id)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q2_app_category_compact_overlay")
    args = parser.parse_args()

    oof, test_pred, target_scores, selector_info, sample_original = predict_q2_app_category_overlay(base_id=args.base_id)
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
        metadata = promote_artifacts(
            args.experiment_id,
            args.slug,
            args.base_id,
            oof,
            test_pred,
            target_scores,
            selector_info,
            sample_original,
        )
        payload = {"mode": "promote", "diagnostic_report": payload["diagnostic_report"], **metadata}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
