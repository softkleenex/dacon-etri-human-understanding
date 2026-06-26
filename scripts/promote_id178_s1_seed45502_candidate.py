#!/usr/bin/env python3
"""Promote an ID178 + S1-only random-conv seed45502 overlay as no-submit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    MODEL_SPECS,
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
    make_kernels,
    random_conv_features,
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

KST = timezone(timedelta(hours=9))
EPS = 1e-6
DEFAULT_BASE_ID = 178
DEFAULT_TARGET = "S1"
DEFAULT_N_KERNELS = 512
DEFAULT_SEED = 45502
DEFAULT_MODEL = "lr_c0_1"
DEFAULT_WEIGHT = 0.15
ID178_PUBLIC_SCORE = 0.6014113009
ID178_LOCAL_REFERENCE = 0.5386382667903545
ID178_PUBLIC_GAP = ID178_PUBLIC_SCORE - ID178_LOCAL_REFERENCE


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_lookup() -> dict[str, Any]:
    return {spec.name: spec for spec in MODEL_SPECS}


def mean_logloss(train: Any, pred: Any) -> tuple[float, dict[str, float]]:
    scores = {
        target: float(
            log_loss(
                train[target].astype(int),
                np.clip(pred[target].astype(float), EPS, 1.0 - EPS),
                labels=[0, 1],
            )
        )
        for target in TARGETS
    }
    return float(np.mean(list(scores.values()))), scores


def build_s1_overlay(
    *,
    base_id: int,
    target: str,
    n_kernels: int,
    seed: int,
    model_name: str,
    weight: float,
) -> tuple[Any, Any, dict[str, float], dict[str, Any], Any]:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1].")
    specs = model_lookup()
    if model_name not in specs:
        raise ValueError(f"Unknown model spec: {model_name}")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    kernels = make_kernels(train_tensor.shape[1], n_kernels=n_kernels, seed=seed)
    x_train = random_conv_features(train_tensor, kernels)
    x_test = random_conv_features(test_tensor, kernels)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=np.asarray(train["date"], dtype="datetime64[ns]")))

    y = train[target].astype(int).to_numpy()
    raw_oof, raw_test = fit_oof_model(x_train, x_test, y, fold_ids, specs[model_name])

    oof = base_oof.copy()
    test_pred = base_test.copy()
    base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
    selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
    oof[target] = selected_oof
    test_pred[target] = selected_test

    local_cv, target_scores = mean_logloss(train, oof)
    base_cv, base_target_scores = mean_logloss(train, base_oof)
    base_loss = float(log_loss(y, base_values, labels=[0, 1]))
    selected_loss = float(log_loss(y, selected_oof, labels=[0, 1]))
    raw_loss = float(log_loss(y, raw_oof, labels=[0, 1]))
    selector_info = {
        "base_id": base_id,
        "base_source": base_source,
        "modified_targets": [target],
        "unchanged_targets": [name for name in TARGETS if name != target],
        "target": target,
        "n_kernels": n_kernels,
        "seed": seed,
        "model": model_name,
        "weight": weight,
        "n_sequence_rows": int(len(seq_table)),
        "n_value_channels": int(len(channel_names) // 2),
        "n_total_channels": int(len(channel_names)),
        "n_random_conv_features": int(x_train.shape[1]),
        "base_cv": base_cv,
        "base_target_scores": base_target_scores,
        "target_details": {
            target: {
                "base_loss": base_loss,
                "raw_sequence_loss": raw_loss,
                "selected_loss": selected_loss,
                "target_improvement_vs_base": base_loss - selected_loss,
                "test_prediction_mean_before": float(np.mean(base_test_values)),
                "test_prediction_mean_after": float(np.mean(selected_test)),
                "test_prediction_max_abs_delta_vs_base": float(np.max(np.abs(selected_test - base_test_values))),
                "test_prediction_mean_abs_delta_vs_base": float(np.mean(np.abs(selected_test - base_test_values))),
            }
        },
        "public_gap_context": {
            "id178_public_score": ID178_PUBLIC_SCORE,
            "id178_local_reference": ID178_LOCAL_REFERENCE,
            "id178_public_minus_local_gap": ID178_PUBLIC_GAP,
            "naive_gap_projected_public": float(local_cv + ID178_PUBLIC_GAP),
            "interpretation": (
                f"Projection is only a rough context anchor. This {target}-only candidate keeps the other targets identical to "
                f"ID{base_id:03d}, "
                "but ID178/ID287/ID329 prove local-to-public transfer is fragile."
            ),
        },
        "source_diagnostics": [
            "reports/kaggle_sequence_source_remote_summary_post_id335_20260617.json",
            "reports/kaggle_sequence_anchor_probe_remote_summary_post_id335_20260618.json",
            "reports/sequence_anchor_probe_decision_post_id335_20260618.json",
        ],
        "selection_contract": (
            f"{target}-only replay of the post-ID335 nk{n_kernels} seed{seed} random-conv source on top of "
            f"ID{base_id:03d}. All non-{target} targets are exactly ID{base_id:03d}. "
            "No DACON submission was made."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_repro_notebook(
    *,
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
    target: str,
    n_kernels: int,
    seed: int,
    model_name: str,
    weight: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID{base_id:03d} {target}-only random-conv seed overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n"
            f"- Local OOF CV: `{local_cv:.6f}`\n"
            f"- Overlay: `{target}` only, seed `{seed}`, kernels `{n_kernels}`, model `{model_name}`, weight `{weight}`\n\n"
            "This notebook is a reproduction pointer for an immutable generated_unsubmitted artifact. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/promote_id178_s1_seed45502_candidate.py --experiment-id {experiment_id} "
            f"--slug {slug} --base-id {base_id} --target {target} --n-kernels {n_kernels} "
            f"--seed {seed} --model {model_name} --weight {weight:g}"
        ),
        nbf.v4.new_markdown_cell(
            "### Decision note\n\n"
            "This artifact changes only S1 from the confirmed ID178 public anchor. "
            "It must pass movement/projection review and still requires explicit user approval before DACON submission."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=336)
    parser.add_argument("--slug", default="id178_s1_seed45502_randomconv_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--n-kernels", type=int, default=DEFAULT_N_KERNELS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = build_s1_overlay(
        base_id=args.base_id,
        target=args.target,
        n_kernels=args.n_kernels,
        seed=args.seed,
        model_name=args.model,
        weight=args.weight,
    )
    local_cv = float(np.mean(list(target_scores.values())))

    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.json"
    decision_name = f"decision_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.md"
    base_oof_name = f"base_oof_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    decision_path = REPORTS_DIR / decision_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name

    for path in [submission_path, notebook_path, metadata_path, decision_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS, eps=EPS)
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
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": f"ID{args.base_id:03d} + {args.target}-only random-conv nk{args.n_kernels} seed{args.seed} overlay",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "selector_info": selector_info,
        "risk_decision": {
            "public_058_plausible": False,
            "public_060_controlled_probe_hypothesis": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": (
                f"Candidate modifies only {args.target} from ID{args.base_id:03d} and is supported by a Kaggle anchor replay. "
                "It is still selected under a public-structure diagnostic lens and is not a robust Public 0.58 claim."
            ),
        },
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "notebook_file": str(notebook_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision_md = (
        f"# Decision note — submission {args.experiment_id:03d} {args.slug}\n\n"
        f"Created: `{metadata['created_at_kst']}`\n\n"
        "## Decision\n\n"
        "- DACON submission: **no**\n"
        "- Public 0.58 plausible: **no**\n"
        "- Controlled public-anchor probe hypothesis: **yes, still unsubmitted**\n\n"
        "## Rationale\n\n"
        f"This artifact keeps ID{args.base_id:03d} unchanged for six targets and replaces only `{args.target}` with "
        f"a random-conv sequence blend from seed `{args.seed}`, `{args.n_kernels}` kernels, model `{args.model}`, "
        f"weight `{args.weight}`. The same seed survived replay on ID178/179/190/191 anchors, but all evidence remains "
        "public-structure-local rather than true public validation.\n\n"
        "## Artifact contract\n\n"
        f"- Submission CSV: `{submission_path.relative_to(ROOT)}`\n"
        f"- Notebook: `{notebook_path.relative_to(ROOT)}`\n"
        f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
        f"- Base OOF: `{base_oof_path.relative_to(ROOT)}`\n"
        f"- Base test: `{base_test_path.relative_to(ROOT)}`\n"
        f"- Local OOF CV: `{local_cv:.6f}`\n"
    )
    decision_path.write_text(decision_md, encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        base_id=args.base_id,
        target=args.target,
        n_kernels=args.n_kernels,
        seed=args.seed,
        model_name=args.model,
        weight=args.weight,
    )

    notes = (
        f"Generated ID{args.base_id:03d} {args.target}-only random-conv seed{args.seed} overlay artifact. "
        f"All non-{args.target} targets stay at ID{args.base_id:03d}; "
        f"{args.target} uses nk{args.n_kernels}/{args.model}/weight{args.weight:g} "
        "from the post-ID335 Kaggle anchor replay. "
        "Not Public 0.58-plausible; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"ID{args.base_id:03d} {args.target}-only random-conv seed{args.seed} overlay",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
