#!/usr/bin/env python3
"""Create a public-structure Q1/Q2/Q3 temporal-anchor probe artifact.

This script intentionally uses a validation lens different from the strict
Subject-Hole CV used by IDs 95-118. It is meant to test the DACON split structure
where train/test share subjects across dates. It must not be treated as a strict
Subject-Hole improvement and must not be submitted without explicit approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

EPS = 1e-6
DEFAULT_BASE_ID = 118
PROBE_TARGETS = ["Q1", "Q2", "Q3"]
FIXED_ANCHOR_PARAMS: dict[str, dict[str, float]] = {
    "Q1": {"alpha_base": 0.75, "half_life": 2.0},
    "Q2": {"alpha_base": 0.60, "half_life": 21.0},
    "Q3": {"alpha_base": 0.55, "half_life": 21.0},
}


def load_base_artifact(train: pd.DataFrame, test: pd.DataFrame, base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF/test artifact.")
    base_oof = align_candidate(pd.read_csv(oof_paths[0]), train)
    base_test = align_candidate(pd.read_csv(test_paths[0]), test)
    return base_oof, base_test, f"{oof_paths[0].name}; {test_paths[0].name}"


def add_date_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["date_dt"] = pd.to_datetime(out["date"])
    return out


def make_interleaved_subject_folds(train: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    fold_ids = np.zeros(len(train), dtype=int)
    ordered = train.sort_values(["subject_id", "date_dt"])
    for _subject_id, group in ordered.groupby("subject_id", sort=False):
        for rank, idx in enumerate(group.index.to_numpy()):
            fold_ids[idx] = rank % n_folds
    return fold_ids


def temporal_anchor_prediction(
    known_df: pd.DataFrame,
    known_y: np.ndarray,
    query_df: pd.DataFrame,
    half_life: float,
) -> np.ndarray:
    known = known_df[["subject_id", "date_dt"]].copy().reset_index(drop=True)
    known["y"] = np.asarray(known_y, dtype=float)
    query = query_df[["subject_id", "date_dt"]].copy().reset_index(drop=True)
    by_subject = {
        subject_id: group.sort_values("date_dt").reset_index(drop=True)
        for subject_id, group in known.groupby("subject_id", sort=False)
    }
    global_mean = float(np.mean(known_y)) if len(known_y) else 0.5
    values: list[float] = []
    for row in query.itertuples(index=False):
        group = by_subject.get(row.subject_id)
        if group is None or len(group) == 0:
            values.append(global_mean)
            continue
        subject_mean = float(group["y"].mean())
        before = group[group["date_dt"] < row.date_dt]
        after = group[group["date_dt"] > row.date_dt]
        components: list[tuple[float, float]] = []
        if len(before):
            delta_days = max(float((row.date_dt - before.iloc[-1]["date_dt"]).days), 0.0)
            components.append((float(before.iloc[-1]["y"]), float(np.exp(-delta_days / half_life))))
        if len(after):
            delta_days = max(float((after.iloc[0]["date_dt"] - row.date_dt).days), 0.0)
            components.append((float(after.iloc[0]["y"]), float(np.exp(-delta_days / half_life))))
        if components:
            numerator = sum(value * weight for value, weight in components)
            denominator = sum(weight for _value, weight in components)
            values.append(numerator / denominator)
        else:
            values.append(subject_mean if np.isfinite(subject_mean) else global_mean)
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def blocked_anchor_oof(
    train: pd.DataFrame,
    y: np.ndarray,
    base_values: np.ndarray,
    alpha_base: float,
    half_life: float,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    selected = np.zeros(len(train), dtype=float)
    fold_details: list[dict] = []
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        anchor = temporal_anchor_prediction(
            known_df=train.iloc[train_idx],
            known_y=y[train_idx],
            query_df=train.iloc[valid_idx],
            half_life=half_life,
        )
        selected[valid_idx] = np.clip(alpha_base * base_values[valid_idx] + (1.0 - alpha_base) * anchor, EPS, 1.0 - EPS)
        fold_details.append(
            {
                "fold": int(fold),
                "n_valid": int(len(valid_idx)),
                "loss": float(log_loss(y[valid_idx], selected[valid_idx], labels=[0, 1])),
            }
        )
    return selected, fold_details


def predict_temporal_anchor_probe(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train_raw, test_raw, _metrics_sorted, sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    base_oof, base_test, base_source = load_base_artifact(train_raw, test_raw, base_id)

    oof = base_oof.copy()
    test_pred = base_test.copy()
    fold_ids = make_interleaved_subject_folds(train)
    target_details: dict[str, dict] = {}

    for target in PROBE_TARGETS:
        params = FIXED_ANCHOR_PARAMS[target]
        alpha_base = float(params["alpha_base"])
        half_life = float(params["half_life"])
        y = train[target].astype(int).to_numpy()
        base_train = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test = np.clip(test_pred[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        selected_oof, fold_details = blocked_anchor_oof(
            train=train,
            y=y,
            base_values=base_train,
            alpha_base=alpha_base,
            half_life=half_life,
            fold_ids=fold_ids,
        )
        test_anchor = temporal_anchor_prediction(
            known_df=train,
            known_y=y,
            query_df=test,
            half_life=half_life,
        )
        selected_test = np.clip(alpha_base * base_test + (1.0 - alpha_base) * test_anchor, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "alpha_base": alpha_base,
            "half_life": half_life,
            "base_loss": float(log_loss(y, base_train, labels=[0, 1])),
            "blocked_anchor_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "fold_details": fold_details,
            "test_anchor_mean": float(np.mean(test_anchor)),
            "test_anchor_std": float(np.std(test_anchor)),
            "test_prediction_mean_before": float(np.mean(base_test)),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test))),
        }

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1])) for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "fold_rule": (
            "Within each subject, sorted train dates are assigned to fold = date_rank % 5. "
            "For each validation fold, Q-target temporal anchors use only non-validation labels from the same subject."
        ),
        "modified_targets": PROBE_TARGETS,
        "unchanged_targets": [target for target in TARGETS if target not in PROBE_TARGETS],
        "fixed_anchor_params": FIXED_ANCHOR_PARAMS,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for S1/S2/S3/S4. Q1/Q2/Q3 validation rows use fixed "
            "same-subject temporal-anchor blends evaluated under an interleaved within-subject blocked CV lens. "
            "Final test Q targets use anchors computed from all labeled training dates for the same subject."
        ),
        "caution": (
            "This is not a strict Subject-Hole CV improvement. It is a public-structure probe for the observed "
            "same-subject train/test date split. Do not compare the score directly with IDs 95-118 and do not submit "
            "without explicit approval."
        ),
    }
    return oof.drop(columns=["date_dt"], errors="ignore"), test_pred.drop(columns=["date_dt"], errors="ignore"), target_scores, selector_info, sample_original


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
            f"# Submission {experiment_id:03d}: Q1/Q2/Q3 temporal-anchor public-structure probe\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure blocked CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable probe artifact. "
            "The score is not strict Subject-Hole CV; it is an interleaved within-subject blocked diagnostic. "
            "Do not submit before explicit approval."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_temporal_anchor_probe_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The artifact changes Q1/Q2/Q3 only and keeps S1/S2/S3/S4 from the base ID."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="q123_temporal_anchor_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_temporal_anchor_probe(args.base_id)
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
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "Q1/Q2/Q3 temporal-anchor public-structure probe",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "Interleaved within-subject blocked CV for Q1/Q2/Q3 temporal anchors; S targets remain from ID base. "
            "This probe is not directly comparable to strict Subject-Hole CV."
        ),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id)

    notes = (
        f"Public-structure probe on ID{args.base_id:03d}: Q1/Q2/Q3 use same-subject temporal-anchor blends "
        "validated by interleaved within-subject blocked CV; S1/S2/S3/S4 unchanged. Not strict Subject-Hole CV, "
        "not directly comparable with IDs 95-118, and no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Public-structure Q123 temporal-anchor probe on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
