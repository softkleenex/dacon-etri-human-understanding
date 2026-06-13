#!/usr/bin/env python3
"""Create a simple mean ensemble from scored submission CSV files.

The script is intentionally conservative: it only reads files that look like
submission CSVs with a `score...` token in the filename, excludes OOF/embedding
artifacts, averages the top-K lowest filename scores, clips probabilities, and
writes `submissions/submission_ensemble_latest.csv`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "date"]
SCORE_RE = re.compile(r"score([0-9]+(?:\.[0-9]+)?)")


def parse_score(path: Path) -> float | None:
    match = SCORE_RE.search(path.name)
    return float(match.group(1)) if match else None


def load_submission_log(input_dir: Path) -> dict[str, dict[str, str]]:
    log_path = input_dir / "submission_log.csv"
    if not log_path.exists():
        return {}
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    rows = {}
    for _, row in log_df.iterrows():
        filename = str(row.get("submission_file", "")).strip()
        if filename:
            rows[Path(filename).name] = row.to_dict()
    return rows


def safe_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except ValueError:
        return None


def load_blacklist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text())
    return {Path(name).name for name in payload.get("files", [])}


def candidate_files(
    input_dir: Path, include_unscored: bool, max_public_gap: float, blacklist_path: Path | None
) -> list[tuple[float, Path, str]]:
    candidates: list[tuple[float, Path, str]] = []
    log_rows = load_submission_log(input_dir)
    blacklisted = load_blacklist(blacklist_path)
    excluded_tokens = ("oof", "embedding", "log", "meta", "ensemble_latest")
    for path in input_dir.glob("*.csv"):
        lower = path.name.lower()
        if any(token in lower for token in excluded_tokens):
            continue
        if path.name in blacklisted and not include_unscored:
            continue

        row = log_rows.get(path.name, {})
        local_score = safe_float(row.get("local_cv_score")) or parse_score(path)
        public_score = safe_float(row.get("public_lb_score"))
        if local_score is None:
            continue

        if public_score and public_score > 0:
            if public_score >= 10 or abs(public_score - local_score) > max_public_gap:
                continue
            reason = f"local={local_score:.6f}, public={public_score:.6f}"
        elif include_unscored:
            reason = f"local={local_score:.6f}, unscored"
        else:
            continue

        candidates.append((local_score, path, reason))
    return sorted(candidates, key=lambda x: (x[0], x[1].name))


def validate_frame(path: Path, df: pd.DataFrame) -> None:
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing target columns: {missing}")
    if df[TARGET_COLS].isna().any().any():
        raise ValueError(f"{path} contains NaN probabilities")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="submissions", type=Path)
    parser.add_argument("--output", default="submissions/submission_ensemble_latest.csv", type=Path)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--eps", default=1e-15, type=float)
    parser.add_argument(
        "--include-unscored", action="store_true", help="Allow candidates without a real public LB sanity check"
    )
    parser.add_argument("--max-public-gap", default=0.03, type=float, help="Maximum allowed |public LB - local CV| gap")
    parser.add_argument("--blacklist", default=Path("reports/submission_blacklist.json"), type=Path)
    args = parser.parse_args()

    ranked = candidate_files(
        args.input_dir,
        include_unscored=args.include_unscored,
        max_public_gap=args.max_public_gap,
        blacklist_path=args.blacklist,
    )
    if not ranked:
        raise SystemExit(
            f"No safe scored submission candidates found in {args.input_dir}. "
            "Use --include-unscored only for exploratory local ensembling."
        )

    selected = ranked[: args.top_k]
    frames = []
    base_ids = None
    for _score, path, reason in selected:
        df = pd.read_csv(path)
        validate_frame(path, df)
        ids = df[[c for c in ID_COLS if c in df.columns]].copy()
        if base_ids is None:
            base_ids = ids
        elif not ids.equals(base_ids):
            raise ValueError(f"ID columns/order differ in {path}; refusing to ensemble")
        frames.append(df[TARGET_COLS].astype(float))
        print(f"Using {path} ({reason})")

    out = base_ids.copy() if base_ids is not None and not base_ids.empty else pd.DataFrame(index=frames[0].index)
    mean_probs = sum(frames) / len(frames)
    out[TARGET_COLS] = np.clip(mean_probs, args.eps, 1 - args.eps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} from {len(selected)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
