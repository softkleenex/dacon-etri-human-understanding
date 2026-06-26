#!/usr/bin/env python3
"""Fold-nested selector over strict artifact OOF/test predictions."""

from __future__ import annotations

import argparse
import hashlib
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
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

DEFAULT_CANDIDATE_IDS = [98, 99, 100, 101, 102]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_ids: list[int],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    candidate_oof: dict[str, pd.DataFrame] = {}
    candidate_test: dict[str, pd.DataFrame] = {}
    candidate_files: dict[str, dict[str, str]] = {}
    for candidate_id in candidate_ids:
        oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id:03d}_*.csv"))
        test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{candidate_id:03d}_*.csv"))
        if len(oof_paths) != 1 or len(test_paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one OOF/test pair for candidate id {candidate_id}: "
                f"oof={oof_paths}, test={test_paths}"
            )
        name = oof_paths[0].stem.replace("base_oof_id_", "")
        candidate_oof[name] = align_candidate(pd.read_csv(oof_paths[0]), train)
        candidate_test[name] = align_candidate(pd.read_csv(test_paths[0]), test)
        candidate_files[name] = {
            "oof": str(oof_paths[0].relative_to(ROOT)),
            "test": str(test_paths[0].relative_to(ROOT)),
        }
    return candidate_oof, candidate_test, candidate_files


def predict_artifact_selector(
    candidate_ids: list[int],
    tie_margin: float = 0.0,
    prefer_full_oof_best: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    if tie_margin < 0:
        raise ValueError("tie_margin must be non-negative.")
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    candidate_oof, candidate_test, candidate_files = load_candidate_predictions(train, test, candidate_ids)
    candidate_oof_arrays = {name: frame[TARGETS].astype(float).to_numpy() for name, frame in candidate_oof.items()}
    candidate_test_arrays = {name: frame[TARGETS].astype(float).to_numpy() for name, frame in candidate_test.items()}

    full_oof_losses: dict[str, dict[str, float]] = {}
    full_oof_best: dict[str, str] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).values
        losses = {
            name: log_loss(y, values[:, target_idx], labels=[0, 1]) for name, values in candidate_oof_arrays.items()
        }
        full_oof_losses[target] = {name: float(value) for name, value in sorted(losses.items())}
        full_oof_best[target] = min(losses.items(), key=lambda item: item[1])[0]

    folds = make_subject_hole_folds(train, n_folds=5)
    selected_oof = np.zeros((len(train), len(TARGETS)), dtype=float)
    nested_choices: dict[str, list[dict]] = {target: [] for target in TARGETS}

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        for target_idx, target in enumerate(TARGETS):
            y = train[target].astype(int).values
            losses = {
                name: log_loss(y[train_idx], values[train_idx, target_idx], labels=[0, 1])
                for name, values in candidate_oof_arrays.items()
            }
            best_name, best_loss = min(losses.items(), key=lambda item: item[1])
            selected_name = best_name
            if prefer_full_oof_best:
                preferred_name = full_oof_best[target]
                if losses[preferred_name] <= best_loss + tie_margin:
                    selected_name = preferred_name
            selected_oof[valid_idx, target_idx] = candidate_oof_arrays[selected_name][valid_idx, target_idx]
            nested_choices[target].append(
                {
                    "fold": fold_idx,
                    "candidate": selected_name,
                    "train_best_candidate": best_name,
                    "train_fold_log_loss": float(best_loss),
                    "selected_train_fold_log_loss": float(losses[selected_name]),
                    "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
                }
            )

    target_scores = {
        target: float(log_loss(train[target].astype(int), selected_oof[:, target_idx], labels=[0, 1]))
        for target_idx, target in enumerate(TARGETS)
    }

    selected_test = np.zeros((len(test), len(TARGETS)), dtype=float)
    full_train_test_choices: dict[str, dict] = {}
    for target_idx, target in enumerate(TARGETS):
        losses = full_oof_losses[target]
        best_name = full_oof_best[target]
        best_loss = losses[best_name]
        selected_test[:, target_idx] = candidate_test_arrays[best_name][:, target_idx]
        full_train_test_choices[target] = {
            "candidate": best_name,
            "full_oof_log_loss": float(best_loss),
            "candidate_losses": losses,
        }

    oof = pd.DataFrame({"subject_id": train["subject_id"].values, "date": pd.to_datetime(train["date"]).dt.date})
    test_pred = pd.DataFrame({"subject_id": test["subject_id"].values, "date": pd.to_datetime(test["date"]).dt.date})
    for target_idx, target in enumerate(TARGETS):
        oof[target] = np.clip(selected_oof[:, target_idx], 1e-6, 1.0 - 1e-6)
        test_pred[target] = np.clip(selected_test[:, target_idx], 1e-6, 1.0 - 1e-6)

    selector_info = {
        "candidate_ids": candidate_ids,
        "candidate_files": candidate_files,
        "nested_choices": nested_choices,
        "full_train_test_choices": full_train_test_choices,
        "tie_margin": tie_margin,
        "prefer_full_oof_best": prefer_full_oof_best,
        "selection_contract": (
            "OOF artifact selection is fold-nested: each outer validation hole receives predictions from the "
            "strict artifact candidate selected using only that outer fold's training rows. Final test "
            "predictions use the candidate with the best full-training OOF log loss per target."
        ),
        "contamination_note": (
            "This selector reuses existing strict OOF/test artifacts and does not train new base models. "
            "It inherits each candidate's feature contract, including the label-free transductive WiFi "
            "home-anchor note from id102."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    candidate_ids: list[int],
    tie_margin: float,
    prefer_full_oof_best: bool,
) -> None:
    candidates_arg = ",".join(str(value) for value in candidate_ids)
    tie_args = ""
    if tie_margin:
        tie_args += f" --tie-margin {tie_margin:g}"
    if prefer_full_oof_best:
        tie_args += " --prefer-full-oof-best"
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: strict artifact selector\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Candidate strict IDs: `{candidates_arg}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            "The generating script selects among existing strict artifact OOF/test predictions with "
            "fold-nested target-wise selection."
        ),
        nbf.v4.new_code_cell(
            f".venv/bin/python scripts/run_strict_artifact_selector_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --candidate-ids {candidates_arg}{tie_args}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
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
            "model_description": "Strict fold-nested selector over strict artifact predictions",
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


def parse_candidate_ids(raw: str) -> list[int]:
    candidate_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not candidate_ids:
        raise ValueError("At least one candidate id is required.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"Candidate ids must be unique: {candidate_ids}")
    return candidate_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="strict_artifact_selector")
    parser.add_argument(
        "--candidate-ids",
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_IDS),
        help="Comma-separated strict experiment ids with base_oof/base_test artifacts.",
    )
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.0,
        help="If --prefer-full-oof-best is set, keep the full-OOF best candidate when it is within this loss margin.",
    )
    parser.add_argument(
        "--prefer-full-oof-best",
        action="store_true",
        help="Use full-OOF best target candidate as a stability tie-break when it is near the fold-training best.",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")
    candidate_ids = parse_candidate_ids(args.candidate_ids)

    oof, test_pred, target_scores, selector_info, sample_original = predict_artifact_selector(
        candidate_ids=candidate_ids,
        tie_margin=args.tie_margin,
        prefer_full_oof_best=args.prefer_full_oof_best,
    )
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
        "method": "strict artifact selector",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "feature_contract": (
            "Fold-nested target-wise selection over existing strict artifact OOF/test predictions. "
            "No new base-model training, future target interpolation, Optuna, same-OOF clipping, or DACON submission. "
            f"Stability tie-break prefer_full_oof_best={args.prefer_full_oof_best}, "
            f"tie_margin={args.tie_margin:g}."
        ),
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        candidate_ids=candidate_ids,
        tie_margin=args.tie_margin,
        prefer_full_oof_best=args.prefer_full_oof_best,
    )

    tie_note = ""
    if args.prefer_full_oof_best:
        tie_note = (
            f" Uses full-OOF best candidate as a stability tie-break when fold-train loss is within "
            f"{args.tie_margin:g}; this is less pure than the id103 zero-margin nested selector."
        )
    notes = (
        f"Strict fold-nested target-wise selector over strict artifacts {candidate_ids}. "
        "No new base-model training, future target interpolation, Optuna, same-OOF clipping, or DACON submission. "
        "Inherits candidate feature contracts and records full target choices in metadata."
        f"{tie_note}"
    )
    append_submission_log(
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
