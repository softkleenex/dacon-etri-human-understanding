#!/usr/bin/env python3
"""Audit documentation, command, and submission-record consistency."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOTS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "AGENTS.md", ROOT / "docs"]
STALE_PATTERNS = {
    "references/": "Use docs/competition/ after the documentation migration.",
    "docs/SUBMISSION_HISTORY.md": "Use submissions/submission_log.csv and reports/submission_log_normalized.csv.",
    "auto_wait_submit.py": "Use make kaggle-wait-and-pull + normalize + deliberate dacon-submit.",
    "notebooks/generate_notebooks.py": "Use notebooks/kaggle_runner/ and make kaggle-push-notebook.",
}
REQUIRED_SUBMISSION_COLUMNS = {
    "id",
    "date",
    "submission_file",
    "model_description",
    "local_cv_score",
    "public_lb_score",
    "is_submitted",
    "notes",
    "status_clean",
    "risk_reason",
    "safe_for_default_ensemble",
}


@dataclass
class Finding:
    severity: str
    area: str
    title: str
    detail: str
    recommendation: str


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in DOC_ROOTS:
        if root.is_file() and root.exists():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.md"))
    return sorted(set(files))


def make_targets() -> set[str]:
    makefile = ROOT / "Makefile"
    targets: set[str] = set()
    for line in makefile.read_text(errors="ignore").splitlines():
        if line and not line.startswith(("\t", "#", ".")) and ":" in line:
            name = line.split(":", 1)[0].strip()
            if re.match(r"^[A-Za-z0-9_.-]+$", name):
                targets.add(name)
    return targets


def shell_make_refs(text: str) -> set[str]:
    refs = set(re.findall(r"`make\s+([A-Za-z0-9_.-]+)`", text))
    refs.update(re.findall(r"^\s*make\s+([A-Za-z0-9_.-]+)\b", text, flags=re.MULTILINE))
    return refs


def markdown_link_targets(path: Path) -> list[str]:
    text = path.read_text(errors="ignore")
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


def audit_docs() -> list[Finding]:
    findings: list[Finding] = []
    targets = make_targets()
    for path in iter_markdown_files():
        text = path.read_text(errors="ignore")
        rel = path.relative_to(ROOT)
        for stale, recommendation in STALE_PATTERNS.items():
            if stale in text:
                findings.append(
                    Finding(
                        "high", "docs", "Stale documentation reference", f"{rel} contains `{stale}`.", recommendation
                    )
                )
        for ref in shell_make_refs(text):
            if ref not in targets:
                findings.append(
                    Finding(
                        "high",
                        "docs",
                        "Documented Make target is missing",
                        f"{rel} references `make {ref}`.",
                        "Update the doc or add the Make target.",
                    )
                )
        for link in markdown_link_targets(path):
            if "://" in link or link.startswith(("#", "mailto:")):
                continue
            target = link.split("#", 1)[0]
            if target and not (path.parent / target).resolve().exists():
                findings.append(
                    Finding(
                        "medium",
                        "docs",
                        "Broken Markdown link",
                        f"{rel} links to `{link}`.",
                        "Update or remove the broken local link.",
                    )
                )
    return findings


def audit_records() -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    stats: dict = {}
    log_path = ROOT / "submissions" / "submission_log.csv"
    blacklist_path = ROOT / "reports" / "submission_blacklist.json"
    if not log_path.exists():
        return [
            Finding(
                "critical",
                "records",
                "Missing submission log",
                "submissions/submission_log.csv is absent.",
                "Restore the canonical submission log.",
            )
        ], stats

    rows = list(csv.DictReader(log_path.open(newline="")))
    columns = set(rows[0].keys()) if rows else set()
    missing_cols = sorted(REQUIRED_SUBMISSION_COLUMNS - columns)
    stats["submission_rows"] = len(rows)
    stats["submission_columns"] = sorted(columns)
    if missing_cols:
        findings.append(
            Finding(
                "critical",
                "records",
                "Submission log missing audit columns",
                f"Missing: {missing_cols}",
                "Run make normalize-submissions.",
            )
        )

    ids = [r.get("id", "") for r in rows if r.get("id", "")]
    duplicate_ids = sorted({x for x in ids if ids.count(x) > 1})
    stats["duplicate_ids"] = duplicate_ids
    if duplicate_ids:
        findings.append(
            Finding(
                "medium",
                "records",
                "Duplicate submission IDs",
                f"Duplicate ids: {duplicate_ids[:10]}",
                "Prefer immutable run IDs for future rows.",
            )
        )

    safe_rows = [r for r in rows if str(r.get("safe_for_default_ensemble", "")).lower() == "true"]
    stats["safe_for_default_ensemble_rows"] = len(safe_rows)
    if not safe_rows:
        findings.append(
            Finding(
                "high",
                "records",
                "No safe ensemble candidates",
                "No rows are marked safe_for_default_ensemble=True.",
                "Refresh scores or relax criteria only after review.",
            )
        )

    if blacklist_path.exists():
        blacklist = json.loads(blacklist_path.read_text())
        blacklisted = set(blacklist.get("files", []))
        unsafe_safe = [
            Path(r.get("submission_file", "")).name
            for r in safe_rows
            if Path(r.get("submission_file", "")).name in blacklisted
        ]
        stats["blacklisted_safe_conflicts"] = unsafe_safe
        if unsafe_safe:
            findings.append(
                Finding(
                    "critical",
                    "records",
                    "Safe candidates conflict with blacklist",
                    f"Conflicts: {unsafe_safe}",
                    "Regenerate blacklist and submission normalization.",
                )
            )
    else:
        findings.append(
            Finding(
                "high",
                "records",
                "Missing submission blacklist",
                "reports/submission_blacklist.json is absent.",
                "Run make normalize-submissions.",
            )
        )

    return findings, stats


def audit_scripts() -> list[Finding]:
    findings: list[Finding] = []
    script_refs: set[str] = set()
    for path in iter_markdown_files() + [ROOT / "Makefile"]:
        if not path.exists():
            continue
        script_refs.update(re.findall(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)", path.read_text(errors="ignore")))
    for ref in sorted(script_refs):
        if not (ROOT / ref).exists():
            findings.append(
                Finding(
                    "high",
                    "workflow",
                    "Referenced script is missing",
                    ref,
                    "Update documentation/Makefile or restore the script.",
                )
            )
    return findings


def main() -> int:
    findings = []
    findings.extend(audit_docs())
    record_findings, stats = audit_records()
    findings.extend(record_findings)
    findings.extend(audit_scripts())

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings = sorted(findings, key=lambda f: (order.get(f.severity, 9), f.area, f.title))
    critical = sum(f.severity == "critical" for f in findings)
    high = sum(f.severity == "high" for f in findings)
    medium = sum(f.severity == "medium" for f in findings)

    report = ["# Documentation & Records Integrity Audit\n"]
    report.append(f"Found **{len(findings)} findings**: {critical} critical, {high} high, {medium} medium.\n")
    report.append("## Findings\n")
    for idx, finding in enumerate(findings, 1):
        report.append(f"### {idx}. [{finding.severity.upper()}] {finding.title}\n")
        report.append(f"- Area: `{finding.area}`\n")
        report.append(f"- Detail: {finding.detail}\n")
        report.append(f"- Recommendation: {finding.recommendation}\n")
    report.append("## Record Stats\n")
    report.append(f"```json\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n```\n")

    out_md = ROOT / "reports" / "docs_records_audit.md"
    out_json = ROOT / "reports" / "docs_records_audit.json"
    out_md.write_text("\n".join(report))
    out_json.write_text(
        json.dumps({"findings": [asdict(f) for f in findings], "stats": stats}, ensure_ascii=False, indent=2)
    )
    print(f"Wrote {out_md.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Findings: {len(findings)} ({critical} critical, {high} high, {medium} medium)")
    return 1 if critical or high else 0


if __name__ == "__main__":
    raise SystemExit(main())
