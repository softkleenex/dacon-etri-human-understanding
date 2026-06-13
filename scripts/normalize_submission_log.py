#!/usr/bin/env python3
"""Normalize DACON submission history and derive risk labels.

This script is intentionally non-destructive: it reads submissions/submission_log.csv
and writes derived files under reports/ and submissions/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "submissions" / "submission_log.csv"
DEFAULT_OUT = ROOT / "reports" / "submission_log_normalized.csv"
DEFAULT_BLACKLIST = ROOT / "reports" / "submission_blacklist.json"
DEFAULT_CANONICAL = ROOT / "submissions" / "submission_log.csv"
DEFAULT_BACKUP = ROOT / "reports" / "submission_log_original_backup.csv"


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().replace({"": np.nan, "None": np.nan}), errors="coerce")


def artifact_role(name: object) -> str:
    filename = Path(str(name).strip()).name
    if not filename:
        return "none"
    if filename == "submission_log.csv":
        return "log"
    if filename.startswith("submission_") and filename.endswith(".csv"):
        return "submission"
    if filename.startswith("oof_") and filename.endswith(".csv"):
        return "oof"
    if filename.startswith("user_embeddings_") and filename.endswith(".csv"):
        return "embedding"
    if filename.endswith("_meta.json"):
        return "metadata"
    return "other"


def classify(row: pd.Series, max_gap: float) -> tuple[str, str]:
    submitted = bool(row["is_submitted_bool"])
    local = row["local_cv_score_num"]
    public = row["public_lb_score_num"]
    exists = bool(row["artifact_exists"])
    filename = str(row["submission_file"]).strip()
    duplicate_count = int(row["duplicate_file_count"])

    if duplicate_count > 1:
        return "duplicate_artifact_mapping", "Multiple log rows point to the same artifact filename."
    if not exists and filename:
        return "missing_artifact", "Logged file is absent from submissions/."
    if submitted and pd.isna(public):
        return "submitted_unscored", "Marked submitted but has no public LB score."
    if submitted and public == 0:
        return "submitted_placeholder_score", "Marked submitted with public LB score 0.0 placeholder."
    if pd.notna(public) and public >= 10:
        return "invalid_public_score", "Public score is penalty-scale or malformed."
    if pd.notna(local) and pd.notna(public) and public > 0 and abs(public - local) >= max_gap:
        return "cv_lb_gap_risk", f"Absolute CV-LB gap is >= {max_gap}."
    if submitted and pd.notna(public) and public > 0:
        return "scored", "Submitted and has a plausible public LB score."
    if not submitted and pd.notna(local):
        return "generated_unsubmitted", "Generated locally but not submitted/scored."
    return "unknown", "Insufficient metadata."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--max-gap", type=float, default=0.03)
    parser.add_argument(
        "--update-canonical",
        action="store_true",
        help="Write normalized audit columns back to submissions/submission_log.csv",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Missing submission log: {args.log}")

    df = pd.read_csv(args.log, dtype=str).fillna("")
    for col in [
        "id",
        "date",
        "submission_file",
        "model_description",
        "local_cv_score",
        "public_lb_score",
        "is_submitted",
        "notes",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["submission_file"] = df["submission_file"].map(lambda x: Path(str(x).strip()).name if str(x).strip() else "")
    df["artifact_role"] = df["submission_file"].map(artifact_role)
    file_counts = df.loc[df["submission_file"].astype(bool), "submission_file"].value_counts()
    df["duplicate_file_count"] = df["submission_file"].map(lambda name: int(file_counts.get(name, 0)) if name else 0)
    df["is_submitted_bool"] = df["is_submitted"].map(to_bool)
    df["local_cv_score_num"] = to_float(df["local_cv_score"])
    df["public_lb_score_num"] = to_float(df["public_lb_score"])
    df["cv_lb_gap"] = df["public_lb_score_num"] - df["local_cv_score_num"]
    df["artifact_exists"] = df["submission_file"].map(
        lambda name: bool(str(name).strip()) and (ROOT / "submissions" / Path(str(name)).name).exists()
    )

    classifications = df.apply(lambda row: classify(row, args.max_gap), axis=1)
    df["status_clean"] = [x[0] for x in classifications]
    df["risk_reason"] = [x[1] for x in classifications]
    df["safe_for_default_ensemble"] = (
        df["status_clean"].eq("scored")
        & df["cv_lb_gap"].abs().lt(args.max_gap)
        & df["artifact_role"].eq("submission")
        & df["duplicate_file_count"].eq(1)
    )

    blacklist_statuses = {
        "missing_artifact",
        "submitted_unscored",
        "submitted_placeholder_score",
        "invalid_public_score",
        "cv_lb_gap_risk",
        "duplicate_artifact_mapping",
    }
    blacklisted = df[df["status_clean"].isin(blacklist_statuses) & df["submission_file"].astype(bool)]
    duplicate_files = sorted(
        {Path(x).name for x in df.loc[df["duplicate_file_count"].gt(1), "submission_file"].tolist() if str(x).strip()}
    )
    blacklist = {
        "max_gap": args.max_gap,
        "files": sorted({Path(x).name for x in blacklisted["submission_file"].tolist() if str(x).strip()}),
        "duplicate_artifact_files": duplicate_files,
        "by_status": df["status_clean"].value_counts().to_dict(),
        "safe_default_candidates": sorted(
            {
                Path(x).name
                for x in df.loc[df["safe_for_default_ensemble"], "submission_file"].tolist()
                if str(x).strip()
            }
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.blacklist.write_text(json.dumps(blacklist, ensure_ascii=False, indent=2))

    if args.update_canonical:
        if not DEFAULT_BACKUP.exists():
            DEFAULT_BACKUP.write_text(args.log.read_text())
        df.to_csv(DEFAULT_CANONICAL, index=False)
        print(f"Updated {DEFAULT_CANONICAL.relative_to(ROOT)} with audit columns")
        print(f"Backup at {DEFAULT_BACKUP.relative_to(ROOT)}")

    summary = df["status_clean"].value_counts().to_string()
    print(f"Wrote {args.out.relative_to(ROOT)}")
    print(f"Wrote {args.blacklist.relative_to(ROOT)}")
    print(summary)
    print(f"Safe default ensemble candidates: {int(df['safe_for_default_ensemble'].sum())}")
    print(f"Blacklisted files: {len(blacklist['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
