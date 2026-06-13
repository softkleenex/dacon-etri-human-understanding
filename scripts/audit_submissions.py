#!/usr/bin/env python3
"""Audit DACON submission artifacts and the canonical submission log."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
REPORT_MD = ROOT / "reports" / "submissions_audit.md"
REPORT_JSON = ROOT / "reports" / "submissions_audit.json"
SAMPLE = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
ALLOWED_NAMES = {".gitkeep", "README.md", "submission_log.csv"}
ALLOWED_PREFIXES = ("submission_", "oof_", "user_embeddings_")
CRITICAL_TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


@dataclass
class Finding:
    severity: str
    area: str
    title: str
    detail: str
    recommendation: str


def read_log() -> pd.DataFrame:
    path = SUBMISSIONS / "submission_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def audit_layout() -> list[Finding]:
    findings: list[Finding] = []
    if not SUBMISSIONS.exists():
        return [
            Finding(
                "critical",
                "layout",
                "Missing submissions directory",
                "submissions/ is absent.",
                "Restore submissions/ with submission_log.csv and README.md.",
            )
        ]

    nested = [str(p.relative_to(ROOT)) for p in SUBMISSIONS.rglob("*") if p.is_dir()]
    if nested:
        findings.append(
            Finding(
                "high",
                "layout",
                "Nested folders inside submissions",
                str(nested),
                "Keep submissions/ flat; archive bundles under artifacts/.",
            )
        )

    unexpected = []
    generic = []
    for path in SUBMISSIONS.iterdir():
        if not path.is_file():
            continue
        if path.name == "submission.csv":
            generic.append(path.name)
        if path.name in ALLOWED_NAMES or path.name.endswith("_meta.json") or path.name.startswith(ALLOWED_PREFIXES):
            continue
        unexpected.append(path.name)

    if generic:
        findings.append(
            Finding(
                "high",
                "layout",
                "Generic submission filename present",
                str(generic),
                "Rename permanent artifacts to submission_id_<id>_<model>_cv<score>_<status>.csv.",
            )
        )
    if unexpected:
        findings.append(
            Finding(
                "medium",
                "layout",
                "Unexpected file names in submissions",
                str(unexpected),
                "Move non-submission assets to reports/, models/, data/, or artifacts/.",
            )
        )
    return findings


def audit_log(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if df.empty:
        return [
            Finding(
                "critical",
                "log",
                "Missing or empty submission log",
                "submissions/submission_log.csv could not be read or has no rows.",
                "Restore the canonical submission history before ensembling.",
            )
        ]

    files = df.get("submission_file", pd.Series(dtype=str)).astype(str).str.strip()
    nonblank = files[files.astype(bool)]
    duplicates = nonblank.value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        findings.append(
            Finding(
                "high",
                "log",
                "Duplicate artifact mapping in submission log",
                duplicates.to_dict().__repr__(),
                "Treat these files as historically preserved but unsafe for default ensemble until each row has a unique immutable artifact.",
            )
        )

    if "artifact_exists" in df.columns:
        missing = df[df["artifact_exists"].astype(str).str.lower().eq("false") & files.astype(bool)]
        if not missing.empty:
            findings.append(
                Finding(
                    "high",
                    "log",
                    "Logged artifacts are missing on disk",
                    missing[["id", "submission_file"]].head(20).to_dict("records").__repr__(),
                    "Restore from archive or mark rows as missing_artifact before using logs for decisions.",
                )
            )

    if "status_clean" in df.columns:
        risky = df[~df["status_clean"].isin(["scored", "generated_unsubmitted"])]["status_clean"].value_counts()
        if not risky.empty:
            findings.append(
                Finding(
                    "medium",
                    "log",
                    "Risk-labelled rows exist",
                    risky.to_dict().__repr__(),
                    "Keep rows for history, but exclude them from default ensembling and score claims.",
                )
            )

    return findings


def audit_orphans(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    logged = {Path(x).name for x in df.get("submission_file", pd.Series(dtype=str)).astype(str) if str(x).strip()}
    actual = {p.name for p in SUBMISSIONS.glob("submission_*.csv") if p.name != "submission_log.csv"}
    historical_exempt = {"submission_ensemble_latest.csv"}
    orphans = sorted(actual - logged - historical_exempt)
    if orphans:
        findings.append(
            Finding(
                "medium",
                "artifacts",
                "Submission CSV files are not referenced by submission_log.csv",
                str(orphans[:40]),
                "Add log rows for important artifacts, or quarantine obsolete files under artifacts/.",
            )
        )
    return findings


def audit_submission_format() -> list[Finding]:
    findings: list[Finding] = []
    if not SAMPLE.exists():
        return findings
    sample = pd.read_csv(SAMPLE)
    sample_cols = list(sample.columns)
    sample_rows = len(sample)
    checked = 0
    bad_shape = []
    bad_values = []
    for path in sorted(SUBMISSIONS.glob("submission_*.csv")):
        if path.name == "submission_log.csv":
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            bad_shape.append({"file": path.name, "error": str(exc)})
            continue
        checked += 1
        if list(df.columns) != sample_cols or len(df) != sample_rows:
            bad_shape.append({"file": path.name, "rows": len(df), "columns": list(df.columns)})
            continue
        targets = [col for col in CRITICAL_TARGETS if col in df.columns]
        if targets:
            numeric = df[targets].apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any():
                bad_values.append({"file": path.name, "issue": "non-numeric target values"})
            elif ((numeric < 0) | (numeric > 1)).any().any():
                bad_values.append({"file": path.name, "issue": "target values outside [0, 1]"})
    if bad_shape:
        findings.append(
            Finding(
                "critical",
                "format",
                "Submission files do not match sample format",
                str(bad_shape[:20]),
                "Fix or quarantine malformed submissions; never submit them to DACON.",
            )
        )
    if bad_values:
        findings.append(
            Finding(
                "high",
                "format",
                "Submission files contain invalid target values",
                str(bad_values[:20]),
                "Clip/calibrate intentionally in code, then regenerate submission files.",
            )
        )
    if checked == 0:
        findings.append(
            Finding(
                "medium",
                "format",
                "No submission CSV files found",
                "No files matching submissions/submission_*.csv were checked.",
                "Generate at least one immutable submission artifact before submitting.",
            )
        )
    return findings


def write_report(findings: list[Finding], df: pd.DataFrame) -> None:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings = sorted(findings, key=lambda item: (order.get(item.severity, 9), item.area, item.title))
    status_counts = df["status_clean"].value_counts().to_dict() if "status_clean" in df.columns else {}
    safe_count = int(df["safe_for_default_ensemble"].astype(str).str.lower().eq("true").sum()) if "safe_for_default_ensemble" in df.columns else 0
    report = ["# Submissions Integrity Audit\n"]
    report.append(f"Found **{len(findings)} findings**. Safe default ensemble rows: **{safe_count}**.\n")
    if status_counts:
        report.append("## Status Counts\n")
        for status, count in status_counts.items():
            report.append(f"- `{status}`: {count}")
        report.append("")
    report.append("## Findings\n")
    for idx, finding in enumerate(findings, 1):
        report.append(f"### {idx}. [{finding.severity.upper()}] {finding.title}\n")
        report.append(f"- Area: `{finding.area}`")
        report.append(f"- Detail: {finding.detail}")
        report.append(f"- Recommendation: {finding.recommendation}\n")
    REPORT_MD.write_text("\n".join(report))
    REPORT_JSON.write_text(
        json.dumps(
            {
                "findings": [asdict(finding) for finding in findings],
                "status_counts": status_counts,
                "safe_default_ensemble_rows": safe_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")
    print(f"Wrote {REPORT_JSON.relative_to(ROOT)}")
    print(f"Findings: {len(findings)}")


def main() -> int:
    df = read_log()
    findings: list[Finding] = []
    findings.extend(audit_layout())
    findings.extend(audit_log(df))
    findings.extend(audit_orphans(df))
    findings.extend(audit_submission_format())
    write_report(findings, df)
    return 1 if any(f.severity in {"critical", "high"} for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
