#!/usr/bin/env python3
"""Recursive repository structure audit for competition workspaces."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "data", "models", "artifacts"}
EXPECTED_ROOT_MD = {"README.md", "AGENTS.md", "CONTRIBUTING.md", "CHANGELOG.md"}
ALLOWED_ROOT = {
    ".env.example",
    ".geminiignore",
    ".gitignore",
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "Makefile",
    "README.md",
    "dataset-metadata.json",
    "pyproject.toml",
}
ALLOWED_ROOT_DIRS = {
    ".git",
    ".venv",
    "_inbox",
    "artifacts",
    "checks",
    "conf",
    "data",
    "docs",
    "models",
    "notebooks",
    "reports",
    "scripts",
    "src",
    "submissions",
    "tests",
}
GENERATED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "lightning_logs",
    "catboost_info",
}


@dataclass
class Finding:
    severity: str
    area: str
    title: str
    detail: str
    recommendation: str


def human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for unit in units:
        if v < 1024 or unit == units[-1]:
            return f"{v:.1f}{unit}" if unit != "B" else f"{int(v)}B"
        v /= 1024
    return f"{n}B"


def iter_files(include_artifacts: bool = False) -> list[Path]:
    files: list[Path] = []
    skip = SKIP_DIRS - ({"artifacts"} if include_artifacts else set())
    for path in ROOT.rglob("*"):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in skip for part in rel_parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def git_ls_files() -> set[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return set(out.splitlines())


def audit_root() -> list[Finding]:
    findings: list[Finding] = []
    root_items = {p.name for p in ROOT.iterdir()}
    root_files = {p.name for p in ROOT.iterdir() if p.is_file()}
    root_dirs = {p.name for p in ROOT.iterdir() if p.is_dir()}

    extra_md = sorted([name for name in root_files if name.endswith(".md") and name not in EXPECTED_ROOT_MD])
    if extra_md:
        findings.append(
            Finding(
                "high",
                "root",
                "Unexpected root Markdown files",
                str(extra_md),
                "Keep root docs limited to README, AGENTS, CONTRIBUTING, and CHANGELOG.",
            )
        )

    unexpected_files = sorted(root_files - ALLOWED_ROOT - {".env"})
    if unexpected_files:
        findings.append(
            Finding(
                "medium",
                "root",
                "Unexpected root files",
                str(unexpected_files[:30]),
                "Move ad-hoc files into docs/, scripts/, reports/, or artifacts/.",
            )
        )

    unexpected_dirs = sorted(root_dirs - ALLOWED_ROOT_DIRS)
    if unexpected_dirs:
        findings.append(
            Finding(
                "medium",
                "root",
                "Unexpected root directories",
                str(unexpected_dirs),
                "Move generated/tool-specific directories to ignored artifacts/ or delete them.",
            )
        )

    if ".env" in root_items:
        findings.append(
            Finding(
                "low",
                "root",
                "Local .env exists",
                ".env is present locally and ignored.",
                "OK locally; never commit secrets.",
            )
        )
    return findings


def audit_recursive_files() -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    stats: dict = {}
    files = iter_files()
    ext_counts = Counter(p.suffix.lower() or "<none>" for p in files)
    stats["file_count_scanned"] = len(files)
    stats["extension_counts"] = dict(ext_counts.most_common(30))

    pdfs = [str(p.relative_to(ROOT)) for p in files if p.suffix.lower() == ".pdf"]
    if pdfs:
        findings.append(
            Finding(
                "high",
                "docs",
                "Readable PDF remains in tracked workspace",
                str(pdfs),
                "Convert to Markdown, verify, then remove the PDF.",
            )
        )

    generated = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.name in GENERATED_NAMES and not any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts)
    ]
    if generated:
        findings.append(
            Finding(
                "medium",
                "generated",
                "Generated/cache directories present",
                str(generated[:30]),
                "Delete cache/generated directories before commit.",
            )
        )

    top_sizes = {}
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in {".git", ".venv"}):
        top_sizes[d.name] = human(dir_size(d))
    stats["top_level_sizes"] = top_sizes
    return findings, stats


def audit_submissions() -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    stats: dict = {}
    sub = ROOT / "submissions"
    if not sub.exists():
        findings.append(
            Finding(
                "critical",
                "submissions",
                "Missing submissions directory",
                "submissions/ does not exist.",
                "Restore canonical submissions directory.",
            )
        )
        return findings, stats

    dirs = [p for p in sub.rglob("*") if p.is_dir()]
    if dirs:
        findings.append(
            Finding(
                "high",
                "submissions",
                "Submissions directory has nested folders",
                str([str(p.relative_to(ROOT)) for p in dirs[:30]]),
                "Keep submissions/ flat; move bundles/checkpoints/data to artifacts/.",
            )
        )

    bad_names = [p.name for p in sub.glob("submission.csv")]
    if bad_names:
        findings.append(
            Finding(
                "high",
                "submissions",
                "Generic submission.csv present",
                str(bad_names),
                "Rename preserved artifacts to immutable descriptive names.",
            )
        )

    suspicious = []
    allowed_prefixes = ("submission_", "oof_", "user_embeddings_")
    allowed_names = {"submission_log.csv", "README.md", ".gitkeep"}
    for p in sub.iterdir():
        if (
            p.is_file()
            and p.name not in allowed_names
            and not p.name.startswith(allowed_prefixes)
            and not p.name.endswith("_meta.json")
        ):
            suspicious.append(p.name)
    if suspicious:
        findings.append(
            Finding(
                "medium",
                "submissions",
                "Suspicious top-level submission files",
                str(suspicious),
                "Move logs/db/misc files to artifacts/ or reports/.",
            )
        )

    files = [p for p in sub.rglob("*") if p.is_file()]
    stats["submissions_files"] = len(files)
    stats["submissions_size"] = human(sum(p.stat().st_size for p in files))
    stats["submissions_categories"] = dict(
        Counter(
            "log"
            if p.name == "submission_log.csv"
            else "readme"
            if p.name == "README.md"
            else "submission"
            if p.name.startswith("submission_")
            else "oof"
            if p.name.startswith("oof_")
            else "embedding"
            if p.name.startswith("user_embeddings_")
            else "meta"
            if p.name.endswith("_meta.json")
            else "other"
            for p in files
        )
    )
    return findings, stats


def audit_nested_project_copies() -> list[Finding]:
    findings: list[Finding] = []
    candidates = []
    for d in ROOT.rglob("*"):
        if not d.is_dir():
            continue
        rel_parts = d.relative_to(ROOT).parts
        if any(part in {".git", ".venv", "artifacts"} for part in rel_parts):
            continue
        child_names = {p.name for p in d.iterdir()}
        if {"src", "scripts", "conf"}.issubset(child_names) and d != ROOT:
            candidates.append(str(d.relative_to(ROOT)))
    if candidates:
        findings.append(
            Finding(
                "high",
                "structure",
                "Nested project copies found",
                str(candidates),
                "Move copied project bundles to ignored artifacts/ or delete them.",
            )
        )
    return findings


def audit_tracked_ignored_patterns() -> list[Finding]:
    findings: list[Finding] = []
    tracked = git_ls_files()
    tracked_generated = sorted(
        p
        for p in tracked
        if p not in {"data/raw/.gitkeep", "data/processed/.gitkeep", "data/interim/.gitkeep", "models/.gitkeep"}
        and p.startswith((
            "data/raw/",
            "data/processed/",
            "data/interim/",
            "models/",
            "artifacts/",
            "lightning_logs/",
            "catboost_info/",
        ))
    )
    if tracked_generated:
        findings.append(
            Finding(
                "high",
                "git",
                "Generated or heavy paths are tracked",
                str(tracked_generated[:40]),
                "Remove from git with git rm --cached or move summaries to reports/.",
            )
        )
    return findings


def main() -> int:
    findings: list[Finding] = []
    stats: dict = {}
    findings.extend(audit_root())
    rec_findings, rec_stats = audit_recursive_files()
    findings.extend(rec_findings)
    stats.update(rec_stats)
    sub_findings, sub_stats = audit_submissions()
    findings.extend(sub_findings)
    stats.update(sub_stats)
    findings.extend(audit_nested_project_copies())
    findings.extend(audit_tracked_ignored_patterns())

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings = sorted(findings, key=lambda f: (order.get(f.severity, 9), f.area, f.title))
    critical = sum(f.severity == "critical" for f in findings)
    high = sum(f.severity == "high" for f in findings)
    medium = sum(f.severity == "medium" for f in findings)
    low = sum(f.severity == "low" for f in findings)

    report = ["# Recursive Repository Structure Audit\n"]
    report.append(
        f"Found **{len(findings)} findings**: {critical} critical, {high} high, {medium} medium, {low} low.\n"
    )
    report.append("## Findings\n")
    for idx, finding in enumerate(findings, 1):
        report.append(f"### {idx}. [{finding.severity.upper()}] {finding.title}\n")
        report.append(f"- Area: `{finding.area}`\n")
        report.append(f"- Detail: {finding.detail}\n")
        report.append(f"- Recommendation: {finding.recommendation}\n")
    report.append("## Stats\n")
    report.append(f"```json\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n```\n")

    out_md = ROOT / "reports" / "repository_structure_audit.md"
    out_json = ROOT / "reports" / "repository_structure_audit.json"
    out_md.write_text("\n".join(report))
    out_json.write_text(
        json.dumps({"findings": [asdict(f) for f in findings], "stats": stats}, ensure_ascii=False, indent=2)
    )
    print(f"Wrote {out_md.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Findings: {len(findings)} ({critical} critical, {high} high, {medium} medium, {low} low)")
    return 1 if critical or high else 0


if __name__ == "__main__":
    raise SystemExit(main())
