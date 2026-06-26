#!/usr/bin/env python3
"""Audit notebook/submission/log artifact matching for DACON experiments.

Rows before ``STRICT_FROM_ID`` are treated as historical records: they should be
checked and reported, but legacy gaps must not block new work. Rows from that ID
onward are strict experiment contracts:

1. one submission_log row,
2. one immutable submission CSV,
3. one matching notebook under notebooks/history,
4. the notebook must either embed the exact CSV or reference the CSV filename.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
NOTEBOOK_DIR = ROOT / "notebooks" / "history"
REPORT_MD = ROOT / "reports" / "experiment_artifacts_audit.md"
REPORT_JSON = ROOT / "reports" / "experiment_artifacts_audit.json"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
STRICT_FROM_ID = 94


@dataclass
class Finding:
    severity: str
    area: str
    title: str
    detail: str
    recommendation: str


def severity_for(row_id: int, legacy_severity: str = "low", strict_severity: str = "high") -> str:
    return strict_severity if row_id >= STRICT_FROM_ID else legacy_severity


def notebook_candidates(row: pd.Series) -> list[Path]:
    explicit = str(row.get("notebook_file", "")).strip()
    if explicit and explicit.lower() != "nan":
        path = ROOT / explicit
        return [path]

    try:
        row_id = int(row["id"])
    except (TypeError, ValueError):
        return []

    return sorted(NOTEBOOK_DIR.glob(f"submission_{row_id:03d}_*.ipynb"))


def read_notebook_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.append("".join(source))
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


def embedded_csv_frame(notebook_text: str) -> pd.DataFrame | None:
    match = re.search(r'csv_data\s*=\s*"""(.*?)"""', notebook_text, flags=re.DOTALL)
    if not match:
        return None
    csv_text = match.group(1).strip()
    if not csv_text:
        return None
    return pd.read_csv(io.StringIO(csv_text))


def frames_match(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str]:
    if list(left.columns) != list(right.columns):
        return False, "column mismatch"
    if len(left) != len(right):
        return False, f"row mismatch: notebook={len(left)} csv={len(right)}"

    id_cols = [c for c in ["subject_id", "sleep_date", "lifelog_date"] if c in left.columns and c in right.columns]
    for col in id_cols:
        if not left[col].astype(str).equals(right[col].astype(str)):
            return False, f"{col} values/order mismatch"

    common_targets = [c for c in TARGETS if c in left.columns and c in right.columns]
    if common_targets:
        left_vals = left[common_targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        right_vals = right[common_targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isnan(left_vals).any() or np.isnan(right_vals).any():
            return False, "non-numeric target value"
        max_abs_diff = float(np.max(np.abs(left_vals - right_vals)))
        if max_abs_diff > 1e-12:
            return False, f"target values differ; max_abs_diff={max_abs_diff:.3g}"

    return True, "embedded CSV matches submission file"


def audit_row(row: pd.Series, submission_counts: dict[str, int]) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    row_id = int(row["id"])
    submission_file = str(row.get("submission_file", "")).strip()
    submission_name = Path(submission_file).name if submission_file and submission_file.lower() != "nan" else ""
    submission_path = ROOT / "submissions" / submission_name if submission_name else None
    notebooks = notebook_candidates(row)

    row_stats = {
        "id": row_id,
        "submission_file": submission_name,
        "notebooks": [str(path.relative_to(ROOT)) for path in notebooks if path.exists()],
        "contract_status": "ok",
    }

    if not submission_name:
        findings.append(
            Finding(
                severity_for(row_id),
                "contract",
                "Submission log row lacks submission_file",
                f"row id={row_id}",
                "From id 94 onward, every experiment row must point to one immutable submission CSV.",
            )
        )
        row_stats["contract_status"] = "missing_submission_file"
    elif submission_path is None or not submission_path.exists():
        findings.append(
            Finding(
                severity_for(row_id, legacy_severity="medium", strict_severity="critical"),
                "contract",
                "Logged submission CSV is missing",
                f"row id={row_id}, file={submission_name}",
                "Restore the CSV or quarantine/fix the log row before using this experiment.",
            )
        )
        row_stats["contract_status"] = "missing_submission_csv"
    else:
        if row_id >= STRICT_FROM_ID:
            expected_prefix = f"submission_id_{row_id:03d}_"
            if not submission_name.startswith(expected_prefix):
                findings.append(
                    Finding(
                        "high",
                        "contract",
                        "Strict submission filename does not include row id",
                        f"row id={row_id}, file={submission_name}, expected_prefix={expected_prefix}",
                        "Use unique immutable filenames like submission_id_094_<short-model>_cv0.xxxxxx_unsubmitted.csv.",
                    )
                )
                row_stats["contract_status"] = "bad_submission_filename"
            if submission_counts.get(submission_name, 0) > 1:
                findings.append(
                    Finding(
                        "critical",
                        "contract",
                        "Strict submission filename is duplicated in log",
                        f"row id={row_id}, file={submission_name}",
                        "Every new experiment row must have a globally unique submission_file value.",
                    )
                )
                row_stats["contract_status"] = "duplicate_submission_filename"

    existing_notebooks = [path for path in notebooks if path.exists()]
    if len(existing_notebooks) != 1:
        findings.append(
            Finding(
                severity_for(row_id, legacy_severity="medium", strict_severity="critical"),
                "contract",
                "Notebook mapping is not exactly one",
                f"row id={row_id}, notebooks={[str(path.relative_to(ROOT)) for path in notebooks]}",
                "Keep exactly one notebooks/history/submission_<id>_*.ipynb artifact per experiment row.",
            )
        )
        row_stats["contract_status"] = "bad_notebook_mapping"
        return findings, row_stats

    notebook_path = existing_notebooks[0]
    if row_id >= STRICT_FROM_ID and not notebook_path.name.startswith(f"submission_{row_id:03d}_"):
        findings.append(
            Finding(
                "high",
                "contract",
                "Strict notebook filename does not include row id",
                f"row id={row_id}, notebook={notebook_path.relative_to(ROOT)}",
                "Use notebooks/history/submission_<id>_<short-model>_cv<score>.ipynb for new experiments.",
            )
        )
        row_stats["contract_status"] = "bad_notebook_filename"

    notebook_text = read_notebook_text(notebook_path)

    if submission_name and submission_name not in notebook_text:
        csv_frame = embedded_csv_frame(notebook_text)
        if csv_frame is None:
            findings.append(
                Finding(
                    severity_for(row_id, legacy_severity="low", strict_severity="high"),
                    "contract",
                    "Notebook does not reference submission CSV",
                    f"row id={row_id}, notebook={notebook_path.relative_to(ROOT)}, file={submission_name}",
                    "Reference the immutable CSV filename in the notebook or embed the generated CSV data.",
                )
            )
            row_stats["contract_status"] = "notebook_does_not_reference_submission"
        elif submission_path is not None and submission_path.exists():
            submission_df = pd.read_csv(submission_path)
            match_ok, reason = frames_match(csv_frame, submission_df)
            row_stats["embedded_csv_check"] = reason
            if not match_ok:
                findings.append(
                    Finding(
                        severity_for(row_id, legacy_severity="medium", strict_severity="critical"),
                        "contract",
                        "Notebook embedded CSV differs from submission CSV",
                        f"row id={row_id}, notebook={notebook_path.relative_to(ROOT)}, file={submission_name}: {reason}",
                        "Regenerate either the notebook or CSV so the row is reproducible.",
                    )
                )
                row_stats["contract_status"] = "embedded_csv_mismatch"
    else:
        row_stats["reference_check"] = "submission filename referenced by notebook"

    return findings, row_stats


def main() -> int:
    if not LOG_PATH.exists():
        raise SystemExit(f"Missing {LOG_PATH.relative_to(ROOT)}")

    df = pd.read_csv(LOG_PATH, dtype=str).fillna("")
    if "id" not in df.columns:
        raise SystemExit("submission_log.csv is missing id column")

    findings: list[Finding] = []
    stats: list[dict] = []
    submission_counts = (
        df["submission_file"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
        .dropna()
        .map(lambda name: Path(name).name)
        .value_counts()
        .to_dict()
    )
    for _, row in df.iterrows():
        row_id_text = str(row.get("id", "")).strip()
        if not row_id_text:
            continue
        try:
            int(row_id_text)
        except ValueError:
            findings.append(
                Finding(
                    "critical",
                    "contract",
                    "Submission row id is not an integer",
                    f"id={row_id_text}",
                    "Use stable integer experiment IDs.",
                )
            )
            continue
        row_findings, row_stats = audit_row(row, submission_counts)
        findings.extend(row_findings)
        stats.append(row_stats)

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings = sorted(findings, key=lambda item: (order.get(item.severity, 9), item.area, item.title))
    critical = sum(f.severity == "critical" for f in findings)
    high = sum(f.severity == "high" for f in findings)
    medium = sum(f.severity == "medium" for f in findings)
    low = sum(f.severity == "low" for f in findings)
    strict_findings = [f for f in findings if f"row id={STRICT_FROM_ID}" in f.detail or re.search(r"row id=([0-9]+)", f.detail)]
    strict_blockers = [
        f
        for f in strict_findings
        if (match := re.search(r"row id=([0-9]+)", f.detail)) and int(match.group(1)) >= STRICT_FROM_ID
    ]

    report = ["# Experiment Artifact Contract Audit\n"]
    report.append(
        f"Found **{len(findings)} findings**: {critical} critical, {high} high, {medium} medium, {low} low."
    )
    report.append(f"Strict contract starts at submission id **{STRICT_FROM_ID}**.\n")
    report.append("## Findings\n")
    for idx, finding in enumerate(findings, 1):
        report.append(f"### {idx}. [{finding.severity.upper()}] {finding.title}\n")
        report.append(f"- Area: `{finding.area}`")
        report.append(f"- Detail: {finding.detail}")
        report.append(f"- Recommendation: {finding.recommendation}\n")
    report.append("## Stats\n")
    report.append(
        "```json\n"
        + json.dumps(
            {
                "strict_from_id": STRICT_FROM_ID,
                "rows_checked": len(stats),
                "strict_blockers": [asdict(finding) for finding in strict_blockers],
                "sample_rows": stats[-10:],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n"
    )

    REPORT_MD.write_text("\n".join(report), encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps(
            {
                "findings": [asdict(finding) for finding in findings],
                "strict_from_id": STRICT_FROM_ID,
                "strict_blockers": [asdict(finding) for finding in strict_blockers],
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")
    print(f"Wrote {REPORT_JSON.relative_to(ROOT)}")
    print(f"Findings: {len(findings)} ({critical} critical, {high} high, {medium} medium, {low} low)")
    print(f"Strict blockers: {len(strict_blockers)}")
    return 1 if strict_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
