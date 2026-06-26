#!/usr/bin/env python3
"""Report-only test-geometry OOF stress after remote parity.

This slices existing OOF artifacts by train rows that resemble the hidden test
row geometry. It intentionally does not create a submission candidate: prior
public results show that full-OOF stress gains can be transfer traps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_test_split_fold_alignment import (  # noqa: E402
    _actual_test_band_holdout,
    _build_timeline,
    _test_neighbor_holdout,
)

KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
BASE_ID = 178
ANCHOR_IDS = [178, 287, 329]
WATCH_IDS = [337, 339, 338, 336, 179, 307, 304, 295, 297, 334, 335]
RERANK_PATH = ROOT / "reports" / "post_remote_parity_public_anchor_rerank_post_id339_20260618.json"
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
BASE_DIR = ROOT / "artifacts" / "base_predictions"


def find_oof_file(candidate_id: int) -> Path | None:
    paths = sorted(BASE_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    return paths[0] if paths else None


def candidate_ids_from_rerank(limit: int = 20) -> list[int]:
    ids: list[int] = []
    if RERANK_PATH.exists():
        report = json.loads(RERANK_PATH.read_text(encoding="utf-8"))
        for row in report.get("top_reviewable_low_movement", [])[:limit]:
            ids.append(int(row["id"]))
    for candidate_id in [*ANCHOR_IDS, *WATCH_IDS]:
        if candidate_id not in ids:
            ids.append(candidate_id)
    return ids


def load_truth() -> pd.DataFrame:
    train = pd.read_csv(ROOT / "data/raw/ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    truth = train[["subject_id", "lifelog_date", *TARGETS]].copy()
    truth["date"] = truth["lifelog_date"].dt.date.astype(str)
    return truth[["subject_id", "date", *TARGETS]]


def load_log_map() -> dict[int, dict[str, Any]]:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    out: dict[int, dict[str, Any]] = {}
    for _, row in log.dropna(subset=["id_num"]).iterrows():
        candidate_id = int(row["id_num"])
        public = pd.to_numeric(row.get("public_lb_score_num"), errors="coerce")
        local = pd.to_numeric(row.get("local_cv_score_num"), errors="coerce")
        out[candidate_id] = {
            "submission_file": row.get("submission_file", ""),
            "local_cv": None if pd.isna(local) else float(local),
            "public_lb": None if pd.isna(public) or float(public) <= 0.1 else float(public),
            "status_clean": row.get("status_clean", ""),
        }
    return out


def build_masks() -> dict[str, np.ndarray]:
    train = pd.read_csv(ROOT / "data/raw/ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    test = pd.read_csv(ROOT / "data/raw/ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"])
    timeline = _build_timeline(train, test)
    masks = {
        "all_oof": np.arange(len(train), dtype=int),
        "actual_test_band": _actual_test_band_holdout(train, timeline, bins=10)[0][1],
        "test_boundary_neighbor2": _test_neighbor_holdout(train, timeline, neighbor_positions=2)[0][1],
        "test_boundary_neighbor5": _test_neighbor_holdout(train, timeline, neighbor_positions=5)[0][1],
        "fragmented_subjects": train.index[train["subject_id"].isin(["id04", "id05", "id06", "id08"])].to_numpy(),
        "simple_subjects": train.index[~train["subject_id"].isin(["id04", "id05", "id06", "id08"])].to_numpy(),
    }
    return {name: np.asarray(sorted(set(values)), dtype=int) for name, values in masks.items()}


def score_predictions(merged: pd.DataFrame, indices: np.ndarray) -> dict[str, Any]:
    target_losses: dict[str, float] = {}
    for target in TARGETS:
        y_true = merged.loc[indices, f"{target}_true"].astype(int).to_numpy()
        y_pred = np.clip(merged.loc[indices, f"{target}_pred"].astype(float).to_numpy(), 1e-15, 1 - 1e-15)
        target_losses[target] = float(log_loss(y_true, y_pred, labels=[0, 1]))
    return {
        "mean_loss": float(np.mean(list(target_losses.values()))),
        "target_losses": target_losses,
        "n_rows": int(len(indices)),
    }


def candidate_family(candidate_id: int, submission_file: str) -> str:
    text = submission_file.lower()
    if candidate_id in {173, 177, 178, 179, 334, 335, 336, 337, 338} or "sequence" in text or "randomconv" in text:
        return "sequence_temporal_family"
    if candidate_id in {287, 295, 296, 297, 304, 305, 306, 307} or "failed_direction" in text or "orthogonal" in text:
        return "failed_direction_combo_family"
    if candidate_id == 339 or "seqsum" in text or "seq-summary" in text:
        return "sequence_summary_combo_family"
    return "other_family"


def build_report() -> dict[str, Any]:
    truth = load_truth()
    masks = build_masks()
    log_map = load_log_map()
    candidate_ids = candidate_ids_from_rerank()

    rows: list[dict[str, Any]] = []
    base_scores: dict[str, float] = {}
    base_file = find_oof_file(BASE_ID)
    if base_file is None:
        raise FileNotFoundError(f"Missing base OOF artifact for ID{BASE_ID}.")

    for candidate_id in candidate_ids:
        path = find_oof_file(candidate_id)
        if path is None:
            continue
        pred = pd.read_csv(path)
        pred["date"] = pd.to_datetime(pred["date"]).dt.date.astype(str)
        merged = truth.merge(pred, on=["subject_id", "date"], suffixes=("_true", "_pred"), how="inner")
        if len(merged) != len(truth):
            raise ValueError(f"OOF artifact ID{candidate_id} does not align with truth rows: {len(merged)}")
        slice_scores = {name: score_predictions(merged, idx) for name, idx in masks.items()}
        if candidate_id == BASE_ID:
            base_scores = {name: value["mean_loss"] for name, value in slice_scores.items()}
        info = log_map.get(candidate_id, {})
        rows.append(
            {
                "id": candidate_id,
                "family": candidate_family(candidate_id, str(info.get("submission_file", ""))),
                "submission_file": info.get("submission_file", ""),
                "local_cv": info.get("local_cv"),
                "public_lb": info.get("public_lb"),
                "status_clean": info.get("status_clean", ""),
                "oof_file": str(path.relative_to(ROOT)),
                "slice_scores": slice_scores,
            }
        )

    if not base_scores:
        raise ValueError("Base ID178 score row was not computed.")

    public178 = log_map.get(178, {}).get("public_lb")
    enriched = []
    for row in rows:
        deltas = {name: float(row["slice_scores"][name]["mean_loss"] - base_scores[name]) for name in masks}
        known_public_delta = (
            None if row["public_lb"] is None or public178 is None else float(row["public_lb"] - public178)
        )
        transfer_trap_known = bool(
            known_public_delta is not None and known_public_delta > 0 and deltas["test_boundary_neighbor2"] < 0
        )
        enriched.append(
            {
                **row,
                "delta_vs_id178": deltas,
                "known_public_delta_vs_id178": known_public_delta,
                "transfer_trap_known": transfer_trap_known,
                "geometry_mean_delta": float(
                    np.mean(
                        [
                            deltas["actual_test_band"],
                            deltas["test_boundary_neighbor2"],
                            deltas["test_boundary_neighbor5"],
                        ]
                    )
                ),
            }
        )

    enriched = sorted(enriched, key=lambda row: row["geometry_mean_delta"])
    known_traps = [row for row in enriched if row["transfer_trap_known"]]
    watch = [row for row in enriched if row["id"] in WATCH_IDS]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "test_geometry_oof_stress_post_remote_parity",
        "purpose": "Slice existing OOF artifacts by train rows matching hidden-test geometry; report-only anti-overfit gate.",
        "masks": {name: int(len(idx)) for name, idx in masks.items()},
        "base_id": BASE_ID,
        "base_scores": base_scores,
        "rows_by_geometry_mean_delta": enriched,
        "known_transfer_traps": known_traps,
        "watch_candidates": watch,
        "decision": {
            "promote_submission_candidate": False,
            "create_id340": False,
            "why": (
                "OOF geometry stress is useful as a warning lens, but known public failures ID287/ID329 also look "
                "strong on these slices. Therefore no candidate can be promoted from this full-OOF stress alone."
            ),
            "next_action": (
                "If pursuing ID340, rebuild the source under a true nested/test-geometry-aware split rather than "
                "recombining existing OOF artifacts."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Test-geometry OOF stress — post remote parity",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It slices existing OOF artifacts; it does not train a model, create a submission CSV, update the submission log, or submit to DACON.",
        "",
        "## Slice sizes",
        "",
    ]
    for name, count in report["masks"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## Known transfer traps",
            "",
            "These rows prove why this diagnostic cannot promote a candidate by itself: they improve OOF geometry slices but did not improve public LB.",
            "",
            "| id | public Δ vs 178 | all Δ | test-band Δ | neighbor2 Δ | geometry mean Δ | file |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["known_transfer_traps"]:
        lines.append(
            f"| {row['id']} | {row['known_public_delta_vs_id178']:+.6f} | "
            f"{row['delta_vs_id178']['all_oof']:+.6f} | {row['delta_vs_id178']['actual_test_band']:+.6f} | "
            f"{row['delta_vs_id178']['test_boundary_neighbor2']:+.6f} | {row['geometry_mean_delta']:+.6f} | "
            f"`{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Watch candidates",
            "",
            "| id | family | local | public | all Δ | test-band Δ | neighbor2 Δ | neighbor5 Δ | geometry mean Δ | read |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["watch_candidates"]:
        read = "watch only"
        if row["id"] == 337:
            read = "controlled probe only"
        elif row["id"] == 339:
            read = "OOF geometry improves, but proxy/movement still blocks promotion"
        elif row["family"] == "failed_direction_combo_family":
            read = "quarantined family"
        local_cv = "" if row["local_cv"] is None else f"{row['local_cv']:.6f}"
        public_lb = "" if row["public_lb"] is None else f"{row['public_lb']:.6f}"
        lines.append(
            f"| {row['id']} | `{row['family']}` | "
            f"{local_cv} | "
            f"{public_lb} | "
            f"{row['delta_vs_id178']['all_oof']:+.6f} | "
            f"{row['delta_vs_id178']['actual_test_band']:+.6f} | "
            f"{row['delta_vs_id178']['test_boundary_neighbor2']:+.6f} | "
            f"{row['delta_vs_id178']['test_boundary_neighbor5']:+.6f} | "
            f"{row['geometry_mean_delta']:+.6f} | {read} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Promote submission candidate: **no**",
            "- Create ID340: **no**",
            f"- Reason: {report['decision']['why']}",
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
        default=ROOT / "reports" / "test_geometry_oof_stress_post_remote_parity_20260618.json",
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
