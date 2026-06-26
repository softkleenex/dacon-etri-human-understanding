#!/usr/bin/env python3
"""External ETRI prior smoke after ID341.

Report-only diagnostic. It converts the older 2019/2020 same-institution ETRI
summary CSVs into weak target-like priors, then checks whether tiny demographic
and weekend prior overlays around ID341 improve frozen local gates or the
post-ID341 public-transfer proxy.

This intentionally does not concatenate external rows into current training,
does not train on public LB feedback, and creates no submission CSV, base
prediction artifact, notebook/history artifact, submission-log row, Kaggle run,
or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_sleep_boundary_transition_smoke_post_id341 import (  # noqa: E402
    EPS,
    PRIMARY_GATE_SPLITS,
    REPORTS_DIR,
    TARGETS,
    evaluate_values_on_split,
    load_reference_values,
    load_split_manifest,
)
from scripts.diagnose_sleep_boundary_transition_v2_gate_post_id341 import (  # noqa: E402
    geometry_report,
    load_base_test_values,
)
from scripts.run_strict_raw_feature_experiment import build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
EXTERNAL_DIR = ROOT / "data" / "external" / "etri_lifelog"
ANCHOR_IDS = [178, 287, 329, 341]
PUBLIC_ID178 = 0.6014113009
PUBLIC_058_GATE = 0.590
PRIOR_MIN_COUNT = 3
PRIOR_SHRINKAGE = 20.0
BLEND_WEIGHTS = [0.005, 0.01, 0.02, 0.05, 0.10]
TARGET_GROUPS = {
    "Q_all": ["Q1", "Q2", "Q3"],
    "S_sleep_external": ["S1", "S2", "S3", "S4"],
    "all_external": TARGETS,
}
GROUP_LEVELS = [
    ["gender", "age_decade", "is_weekend"],
    ["gender", "age_decade"],
    ["age_decade", "is_weekend"],
    ["gender", "is_weekend"],
    ["gender"],
    ["age_decade"],
    ["is_weekend"],
]


def safe_ratio(num: float, den: float) -> float | None:
    if den <= 0 or not np.isfinite(den):
        return None
    return float(num / den)


def normalize_user_id(value: Any) -> str:
    return str(value).strip()


def normalize_gender(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.startswith("F"):
        return "F"
    if text.startswith("M"):
        return "M"
    return None


def age_decade(value: Any) -> float | None:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(age):
        return None
    return float(int(age // 10 * 10))


def load_external_user_info() -> pd.DataFrame:
    frames = []
    for filename in ["user_info_2019_2018_updated.csv", "user_info_2020.csv"]:
        path = EXTERNAL_DIR / filename
        df = pd.read_csv(path)
        df["source_file"] = filename
        frames.append(df)
    info = pd.concat(frames, ignore_index=True, sort=False)
    info["user_id"] = info["userId"].map(normalize_user_id)
    info["gender_norm"] = info["gender"].map(normalize_gender)
    info["age_decade"] = info["age"].map(age_decade)
    info = info[["user_id", "gender_norm", "age_decade", "source_file"]].drop_duplicates("user_id")
    return info


def survey_target_rows(info: pd.DataFrame) -> pd.DataFrame:
    rows = []

    survey_2019 = pd.read_csv(EXTERNAL_DIR / "user_survey_2019_2018.csv")
    survey_2019["cohort"] = "survey_2019_2018"
    survey_2019["user_id"] = survey_2019["userId"].map(normalize_user_id)
    survey_2019["date"] = pd.to_datetime(survey_2019["inputDt"], errors="coerce").dt.date
    for column, target, positive_high in [
        ("sleep", "Q1", True),
        ("pmStress", "Q3", False),
    ]:
        if column not in survey_2019:
            continue
        means = survey_2019.groupby("user_id")[column].transform("mean")
        label = np.where(
            survey_2019[column].notna() & means.notna(),
            (survey_2019[column] >= means if positive_high else survey_2019[column] <= means).astype(float),
            np.nan,
        )
        tmp = survey_2019[["user_id", "date", "cohort"]].copy()
        tmp["target"] = target
        tmp["label"] = label
        rows.append(tmp)

    survey_2020 = pd.read_csv(EXTERNAL_DIR / "user_survey_2020.csv")
    survey_2020["cohort"] = "survey_2020"
    survey_2020["user_id"] = survey_2020["userId"].map(normalize_user_id)
    survey_2020["date"] = pd.to_datetime(survey_2020["date"], errors="coerce").dt.date
    for column, target, positive_high in [
        ("sleep", "Q1", True),
        ("pmFatigue", "Q2", False),
        ("pmStress", "Q3", False),
    ]:
        if column not in survey_2020:
            continue
        means = survey_2020.groupby("user_id")[column].transform("mean")
        label = np.where(
            survey_2020[column].notna() & means.notna(),
            (survey_2020[column] >= means if positive_high else survey_2020[column] <= means).astype(float),
            np.nan,
        )
        tmp = survey_2020[["user_id", "date", "cohort"]].copy()
        tmp["target"] = target
        tmp["label"] = label
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["user_id", "date", "cohort", "target", "label"])
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["date", "target", "label"])
    out = out.merge(info, on="user_id", how="left")
    out["is_weekend"] = pd.to_datetime(out["date"]).dt.dayofweek.ge(5).astype(int)
    return out


def sleep_target_rows(info: pd.DataFrame) -> pd.DataFrame:
    rows = []

    sleep_2019 = pd.read_csv(EXTERNAL_DIR / "user_sleep_2019_2018.csv")
    sleep_2019["cohort"] = "sleep_2019_2018"
    sleep_2019["user_id"] = sleep_2019["userId"].map(normalize_user_id)
    sleep_2019["date"] = pd.to_datetime(sleep_2019["endDt"], errors="coerce").dt.date
    sleep_2019["total_sleep_sec"] = pd.to_numeric(sleep_2019["total_sleep_time"], errors="coerce")
    sleep_2019["time_in_bed_sec"] = pd.to_numeric(sleep_2019["time_in_bed"], errors="coerce")
    sleep_2019["waso_sec"] = pd.to_numeric(sleep_2019["waso"], errors="coerce")
    sleep_2019["S1"] = sleep_2019["total_sleep_sec"].between(7 * 3600, 9 * 3600, inclusive="both")
    sleep_2019["S2"] = (sleep_2019["total_sleep_sec"] / sleep_2019["time_in_bed_sec"].replace(0, np.nan)).ge(0.85)
    sleep_2019["S4"] = sleep_2019["waso_sec"].le(20 * 60)
    for target in ["S1", "S2", "S4"]:
        tmp = sleep_2019[["user_id", "date", "cohort", target]].copy()
        tmp["target"] = target
        tmp["label"] = tmp[target].astype(float)
        rows.append(tmp[["user_id", "date", "cohort", "target", "label"]])

    sleep_2020 = pd.read_csv(EXTERNAL_DIR / "user_sleep_2020.csv")
    sleep_2020["cohort"] = "sleep_2020"
    sleep_2020["user_id"] = sleep_2020["userId"].map(normalize_user_id)
    sleep_2020["date"] = pd.to_datetime(sleep_2020["date"], errors="coerce").dt.date
    for col in ["wakeupduration", "lightsleepduration", "deepsleepduration", "remsleepduration", "durationtosleep"]:
        sleep_2020[col] = pd.to_numeric(sleep_2020[col], errors="coerce")
    sleep_2020["total_sleep_sec"] = (
        sleep_2020["lightsleepduration"] + sleep_2020["deepsleepduration"] + sleep_2020["remsleepduration"]
    )
    sleep_2020["time_in_bed_sec"] = sleep_2020["total_sleep_sec"] + sleep_2020["wakeupduration"]
    sleep_2020["S1"] = sleep_2020["total_sleep_sec"].between(7 * 3600, 9 * 3600, inclusive="both")
    sleep_2020["S2"] = (sleep_2020["total_sleep_sec"] / sleep_2020["time_in_bed_sec"].replace(0, np.nan)).ge(0.85)
    sleep_2020["S3"] = sleep_2020["durationtosleep"].between(5 * 60, 30 * 60, inclusive="both")
    sleep_2020["S4"] = sleep_2020["wakeupduration"].le(20 * 60)
    for target in ["S1", "S2", "S3", "S4"]:
        tmp = sleep_2020[["user_id", "date", "cohort", target]].copy()
        tmp["target"] = target
        tmp["label"] = tmp[target].astype(float)
        rows.append(tmp[["user_id", "date", "cohort", "target", "label"]])

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["date", "target", "label"])
    out = out.merge(info, on="user_id", how="left")
    out["is_weekend"] = pd.to_datetime(out["date"]).dt.dayofweek.ge(5).astype(int)
    return out


def load_external_target_rows() -> pd.DataFrame:
    info = load_external_user_info()
    external = pd.concat([survey_target_rows(info), sleep_target_rows(info)], ignore_index=True, sort=False)
    external = external.rename(columns={"gender_norm": "gender"})
    external = external.dropna(subset=["gender", "age_decade", "is_weekend", "label"])
    external["label"] = external["label"].astype(float)
    external["age_decade"] = external["age_decade"].astype(float)
    external["is_weekend"] = external["is_weekend"].astype(int)
    return external


def row_signature(row: pd.Series, cols: list[str]) -> tuple[Any, ...]:
    return tuple(row[col] for col in cols)


def current_prior_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["gender"] = np.where(df["demo_gender_female"].fillna(0).astype(float).ge(0.5), "F", "M")
    out["age_decade"] = pd.to_numeric(df["demo_age_decade"], errors="coerce").fillna(-1).astype(float)
    out["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
    return out


def build_prior_tables(external: pd.DataFrame) -> dict[str, Any]:
    tables: dict[str, Any] = {"levels": GROUP_LEVELS, "targets": {}}
    for target in TARGETS:
        target_rows = external[external["target"].eq(target)].copy()
        if target_rows.empty:
            continue
        global_mean = float(target_rows["label"].mean())
        target_tables = []
        for level_cols in GROUP_LEVELS:
            grouped = (
                target_rows.groupby(level_cols, dropna=False)["label"]
                .agg(["mean", "count"])
                .reset_index()
                .query("count >= @PRIOR_MIN_COUNT")
            )
            mapping: dict[tuple[Any, ...], dict[str, float]] = {}
            for _idx, row in grouped.iterrows():
                key = tuple(row[col] for col in level_cols)
                count = float(row["count"])
                mean = float(row["mean"])
                shrunk = float((count * mean + PRIOR_SHRINKAGE * global_mean) / (count + PRIOR_SHRINKAGE))
                mapping[key] = {"mean": mean, "count": count, "shrunk": shrunk}
            target_tables.append({"cols": level_cols, "mapping": mapping})
        tables["targets"][target] = {
            "global_mean": global_mean,
            "n": int(len(target_rows)),
            "cohort_counts": dict(Counter(target_rows["cohort"])),
            "tables": target_tables,
        }
    return tables


def prior_for_row(row: pd.Series, target_table: dict[str, Any]) -> tuple[float, str, float]:
    for table in target_table["tables"]:
        cols = table["cols"]
        key = row_signature(row, cols)
        hit = table["mapping"].get(key)
        if hit is not None:
            return float(hit["shrunk"]), "+".join(cols), float(hit["count"])
    return float(target_table["global_mean"]), "global", float(target_table["n"])


def make_prior_matrix(
    df: pd.DataFrame, prior_tables: dict[str, Any], fallback: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    keys = current_prior_keys(df)
    out = fallback.copy()
    usage: dict[str, Counter] = {target: Counter() for target in TARGETS}
    support_counts: dict[str, list[float]] = {target: [] for target in TARGETS}
    for target_idx, target in enumerate(TARGETS):
        target_table = prior_tables["targets"].get(target)
        if target_table is None:
            continue
        values = []
        for row_idx in keys.index:
            value, level, support = prior_for_row(keys.loc[row_idx], target_table)
            values.append(value)
            usage[target][level] += 1
            support_counts[target].append(support)
        out[:, target_idx] = np.clip(np.asarray(values, dtype=float), EPS, 1 - EPS)
    usage_summary = {
        target: {
            "levels": dict(counter),
            "mean_support": float(np.mean(support_counts[target])) if support_counts[target] else None,
        }
        for target, counter in usage.items()
        if counter
    }
    return out, usage_summary


def target_prevalence(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for target in TARGETS:
        target_rows = rows[rows["target"].eq(target)]
        if target_rows.empty:
            out[target] = {"n": 0, "mean": None}
        else:
            out[target] = {"n": int(len(target_rows)), "mean": float(target_rows["label"].mean())}
    return out


def current_train_prevalence(train: pd.DataFrame) -> dict[str, float]:
    return {target: float(train[target].mean()) for target in TARGETS}


def make_candidate_values(base: np.ndarray, prior: np.ndarray, targets: list[str], weight: float) -> np.ndarray:
    out = base.copy()
    for target in targets:
        target_idx = TARGETS.index(target)
        out[:, target_idx] = np.clip((1.0 - weight) * base[:, target_idx] + weight * prior[:, target_idx], EPS, 1 - EPS)
    return out


def evaluate_candidate(
    name: str,
    targets: list[str],
    weight: float,
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reference178: np.ndarray,
    reference341: np.ndarray,
    prior_train: np.ndarray,
    prior_test: np.ndarray,
    base_tests: dict[int, np.ndarray],
) -> dict[str, Any]:
    candidate_train = make_candidate_values(reference341, prior_train, targets, weight)
    candidate_test = make_candidate_values(base_tests[341], prior_test, targets, weight)
    split_rows: dict[str, Any] = {}
    for split_name, folds in splits.items():
        candidate_eval = evaluate_values_on_split(train, folds, candidate_train)
        id341_eval = evaluate_values_on_split(train, folds, reference341)
        id178_eval = evaluate_values_on_split(train, folds, reference178)
        split_rows[split_name] = {
            "candidate_loss": candidate_eval["mean_loss"],
            "id341_loss": id341_eval["mean_loss"],
            "id178_loss": id178_eval["mean_loss"],
            "delta_vs_id341": float(candidate_eval["mean_loss"] - id341_eval["mean_loss"]),
            "delta_vs_id178": float(candidate_eval["mean_loss"] - id178_eval["mean_loss"]),
            "beats_id341": bool(candidate_eval["mean_loss"] < id341_eval["mean_loss"]),
            "target_losses": candidate_eval["target_losses"],
        }
    primary_passes = {
        split_name: split_rows[split_name]["beats_id341"]
        for split_name in PRIMARY_GATE_SPLITS
        if split_name in split_rows
    }
    local_cv = float(split_rows["subject_hole_5"]["candidate_loss"])
    geometry = geometry_report(candidate_test, base_tests, local_cv)
    primary_wins = int(sum(bool(value) for value in primary_passes.values()))
    reviewable = bool(
        primary_passes
        and all(primary_passes.values())
        and geometry["projection_failed"] <= 0
        and geometry["proxy_public_risk"] < geometry_report(base_tests[341], base_tests, local_cv)["proxy_public_risk"]
    )
    promotable = bool(reviewable and geometry["proxy_public_risk"] < PUBLIC_ID178 and geometry["public_058_plausible"])
    return {
        "name": name,
        "targets": targets,
        "weight": weight,
        "split_rows": split_rows,
        "primary_passes_vs_id341": primary_passes,
        "primary_wins_vs_id341": primary_wins,
        "all_primary_splits_beat_id341": bool(primary_passes and all(primary_passes.values())),
        "local_cv_subject_hole_5": local_cv,
        "test_geometry": geometry,
        "reviewable_low_movement_candidate": reviewable,
        "promotable_public_058_candidate": promotable,
    }


def candidate_specs(available_targets: list[str]) -> list[tuple[str, list[str], float]]:
    groups: dict[str, list[str]] = {target: [target] for target in available_targets}
    for name, targets in TARGET_GROUPS.items():
        filtered = [target for target in targets if target in available_targets]
        if filtered:
            groups[name] = filtered
    specs = []
    for name, targets in groups.items():
        for weight in BLEND_WEIGHTS:
            specs.append((name, targets, weight))
    return specs


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    geometry = row["test_geometry"]
    return {
        "name": row["name"],
        "targets": row["targets"],
        "weight": row["weight"],
        "primary_wins_vs_id341": row["primary_wins_vs_id341"],
        "all_primary_splits_beat_id341": row["all_primary_splits_beat_id341"],
        "local_cv_subject_hole_5": row["local_cv_subject_hole_5"],
        "proxy_public_risk": geometry["proxy_public_risk"],
        "beats_id341_proxy": geometry["beats_id341_proxy"],
        "public_058_plausible": geometry["public_058_plausible"],
        "mean_delta_vs_341": geometry["mean_delta_vs_341"],
        "max_delta_vs_341": geometry["max_delta_vs_341"],
        "projection_failed": geometry["projection_failed"],
        "reviewable_low_movement_candidate": row["reviewable_low_movement_candidate"],
        "promotable_public_058_candidate": row["promotable_public_058_candidate"],
        "primary_deltas_vs_id341": {
            split_name: row["split_rows"][split_name]["delta_vs_id341"]
            for split_name in PRIMARY_GATE_SPLITS
            if split_name in row["split_rows"]
        },
    }


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    external = load_external_target_rows()
    prior_tables = build_prior_tables(external)
    splits = load_split_manifest()
    reference178 = load_reference_values(train, 178)
    reference341 = load_reference_values(train, 341)
    base_tests = {candidate_id: load_base_test_values(test, candidate_id) for candidate_id in ANCHOR_IDS}
    prior_train, train_prior_usage = make_prior_matrix(train, prior_tables, reference341)
    prior_test, test_prior_usage = make_prior_matrix(test, prior_tables, base_tests[341])
    available_targets = [target for target in TARGETS if target in prior_tables["targets"]]
    rows = [
        evaluate_candidate(
            name,
            targets,
            weight,
            train,
            splits,
            reference178,
            reference341,
            prior_train,
            prior_test,
            base_tests,
        )
        for name, targets, weight in candidate_specs(available_targets)
    ]
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            not row["reviewable_low_movement_candidate"],
            -row["primary_wins_vs_id341"],
            row["test_geometry"]["proxy_public_risk"],
            row["local_cv_subject_hole_5"],
        ),
    )
    reviewable = [row for row in rows if row["reviewable_low_movement_candidate"]]
    promotable = [row for row in rows if row["promotable_public_058_candidate"]]
    best_proxy = min(rows, key=lambda row: row["test_geometry"]["proxy_public_risk"]) if rows else None
    best_local = min(rows, key=lambda row: row["local_cv_subject_hole_5"]) if rows else None
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "external_etri_prior_smoke_post_id341",
        "purpose": (
            "Estimate whether older ETRI 2019/2020 summary CSVs are useful as weak external priors around ID341 "
            "without concatenating external rows into current training."
        ),
        "external_inputs": sorted(str(path.relative_to(ROOT)) for path in EXTERNAL_DIR.glob("*.csv")),
        "external_label_mapping": {
            "Q1": "survey sleep >= per-user mean",
            "Q2": "2020 pmFatigue <= per-user mean",
            "Q3": "pmStress <= per-user mean",
            "S1": "total sleep duration in [7h, 9h]",
            "S2": "sleep efficiency >= 0.85",
            "S3": "2020 duration-to-sleep in [5m, 30m]; lower confidence because 2019 lacks SOL",
            "S4": "wake duration/WASO <= 20m",
        },
        "prior_strategy": {
            "group_levels": GROUP_LEVELS,
            "min_count": PRIOR_MIN_COUNT,
            "shrinkage_to_global": PRIOR_SHRINKAGE,
            "blend_weights": BLEND_WEIGHTS,
            "candidate_groups": TARGET_GROUPS,
        },
        "coverage": {
            "external_long_rows": int(len(external)),
            "external_users": int(external["user_id"].nunique()),
            "external_targets_available": available_targets,
            "candidate_rows_scanned": int(len(rows)),
        },
        "external_target_prevalence": target_prevalence(external),
        "current_train_target_prevalence": current_train_prevalence(train),
        "prior_target_metadata": {
            target: {
                "n": data["n"],
                "global_mean": data["global_mean"],
                "cohort_counts": data["cohort_counts"],
            }
            for target, data in prior_tables["targets"].items()
        },
        "train_prior_usage": train_prior_usage,
        "test_prior_usage": test_prior_usage,
        "top_rows": [compact_row(row) for row in rows_sorted[:30]],
        "best_proxy_row": compact_row(best_proxy) if best_proxy else None,
        "best_local_row": compact_row(best_local) if best_local else None,
        "reviewable_rows": [compact_row(row) for row in reviewable[:30]],
        "promotable_rows": [compact_row(row) for row in promotable[:30]],
        "decision": {
            "external_prior_reviewable_low_movement_candidate_exists": bool(reviewable),
            "external_prior_promotable_public_058_candidate_exists": bool(promotable),
            "best_proxy": None if best_proxy is None else best_proxy["test_geometry"]["proxy_public_risk"],
            "best_proxy_row_name": None if best_proxy is None else best_proxy["name"],
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "next_action": (
                "Use external ETRI data only as weak-prior/reference evidence unless a future source passes the combined "
                "transfer gate. Do not concatenate older external rows into current training without a separate leak-free proof."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(out) else f"{out:.{digits}f}"


def row_table(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"| `{row['name']}` | `{','.join(row['targets'])}` | {row['weight']:.3f} | "
            f"{row['primary_wins_vs_id341']} | {fmt(row['local_cv_subject_hole_5'])} | "
            f"{fmt(row['proxy_public_risk'])} | {fmt(row['mean_delta_vs_341'])}/"
            f"{fmt(row['max_delta_vs_341'])} | {fmt(row['projection_failed'])} | "
            f"`{row['reviewable_low_movement_candidate']}` | `{row['promotable_public_058_candidate']}` |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External ETRI prior smoke — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Purpose",
        "",
        report["purpose"],
        "",
        "## Coverage",
        "",
        f"- External long target rows: `{report['coverage']['external_long_rows']}`",
        f"- External users: `{report['coverage']['external_users']}`",
        f"- Targets available: `{report['coverage']['external_targets_available']}`",
        f"- Candidate overlays scanned: `{report['coverage']['candidate_rows_scanned']}`",
        "",
        "## Target prevalence comparison",
        "",
        "| target | external n | external mean | current train mean |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        ext = report["external_target_prevalence"].get(target, {})
        lines.append(
            f"| `{target}` | {ext.get('n', 0)} | {fmt(ext.get('mean'))} | "
            f"{fmt(report['current_train_target_prevalence'].get(target))} |"
        )
    lines.extend(
        [
            "",
            "## Best scanned rows",
            "",
            "| name | targets | weight | primary wins | local | proxy | Δ341 mean/max | failed proj | review | promote |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
            *row_table(report["top_rows"], limit=30),
            "",
            "## Reviewable rows",
            "",
            "| name | targets | weight | primary wins | local | proxy | Δ341 mean/max | failed proj | review | promote |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
            *row_table(report["reviewable_rows"], limit=30),
            "",
            "## Decision",
            "",
            f"- External-prior reviewable low-movement candidate exists: `{report['decision']['external_prior_reviewable_low_movement_candidate_exists']}`",
            f"- External-prior Public 0.58 promotable candidate exists: `{report['decision']['external_prior_promotable_public_058_candidate_exists']}`",
            f"- Best proxy: `{fmt(report['decision']['best_proxy'])}`",
            f"- Best proxy row: `{report['decision']['best_proxy_row_name']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "external_etri_prior_smoke_post_id341_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
