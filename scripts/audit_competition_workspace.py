#!/usr/bin/env python3
"""Non-destructive repository audit for DACON/Kaggle-style ML workspaces."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


@dataclass
class Issue:
    severity: str
    area: str
    title: str
    detail: str
    recommendation: str


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f}{u}" if u != "B" else f"{int(v)}B"
        v /= 1024
    return f"{n}B"


def make_targets() -> set[str]:
    makefile = ROOT / "Makefile"
    if not makefile.exists():
        return set()
    targets = set()
    for line in makefile.read_text(errors="ignore").splitlines():
        if line and not line.startswith(("\t", "#", ".")) and ":" in line:
            name = line.split(":", 1)[0].strip()
            if name and re.match(r"^[A-Za-z0-9_.-]+$", name):
                targets.add(name)
    return targets


def referenced_make_targets() -> set[str]:
    refs = set()
    # Only count shell-like references, not ordinary prose such as "make it available".
    patterns = [
        re.compile(r"`make\s+([A-Za-z0-9_.-]+)`"),
        re.compile(r"^\s*make\s+([A-Za-z0-9_.-]+)\b", re.MULTILINE),
    ]
    for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "AGENTS.md"] + list(
        (ROOT / "docs").rglob("*.md")
    ):
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for pattern in patterns:
            for match in pattern.findall(text):
                if "*" not in match:
                    refs.add(match)
    return refs


def find_missing_markdown_links() -> list[tuple[str, str]]:
    missing = []
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "AGENTS.md"] + list(
        (ROOT / "docs").rglob("*.md")
    ):
        if not path.exists():
            continue
        for link in link_re.findall(path.read_text(errors="ignore")):
            if "://" in link or link.startswith("#") or link.startswith("mailto:"):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            if not (path.parent / target).resolve().exists():
                missing.append((str(path.relative_to(ROOT)), link))
    return missing


def audit_submission_log() -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    stats: dict = {}
    log_path = ROOT / "submissions" / "submission_log.csv"
    if not log_path.exists():
        issues.append(
            Issue(
                "critical",
                "submissions",
                "submission_log.csv is missing",
                "No canonical submission history found.",
                "Restore or recreate the submission log before further leaderboard submissions.",
            )
        )
        return issues, stats

    rows = list(csv.DictReader(log_path.open(newline="")))
    stats["submission_rows"] = len(rows)
    ids = [r.get("id", "") for r in rows]
    duplicate_ids = [k for k, v in Counter(ids).items() if k and v > 1]
    if duplicate_ids:
        issues.append(
            Issue(
                "high",
                "submissions",
                "Duplicate submission IDs",
                f"Duplicate ids: {duplicate_ids[:10]}",
                "Make ids unique or replace id with immutable run_id.",
            )
        )

    submitted = [r for r in rows if str(r.get("is_submitted", "")).lower() == "true"]
    missing_public = [
        r
        for r in submitted
        if not str(r.get("public_lb_score", "")).strip() or str(r.get("public_lb_score", "")).strip() == "0.0"
    ]
    stats["submitted_rows"] = len(submitted)
    stats["submitted_missing_public_or_zero"] = len(missing_public)
    has_status_clean = bool(rows and "status_clean" in rows[0])
    if missing_public and not has_status_clean:
        issues.append(
            Issue(
                "critical",
                "submissions",
                "Submitted rows lack real public LB scores",
                f"{len(missing_public)} submitted rows have blank/0.0 public_lb_score.",
                "Run `make normalize-submissions` to add explicit scored/risk status columns.",
            )
        )
    elif missing_public:
        issues.append(
            Issue(
                "high",
                "submissions",
                "Historical submitted rows lack real public LB scores",
                f"{len(missing_public)} rows are now labeled via status_clean/risk_reason.",
                "Use safe_for_default_ensemble and status_clean instead of raw is_submitted for candidate selection.",
            )
        )

    missing_files = []
    for r in rows:
        f = str(r.get("submission_file", "")).strip()
        if f and not (ROOT / "submissions" / f).exists() and not (ROOT / f).exists():
            missing_files.append(f)
    stats["logged_files_missing"] = len(missing_files)
    if missing_files:
        issues.append(
            Issue(
                "high",
                "submissions",
                "Logged submission files are missing",
                f"{len(missing_files)} logged files not found, e.g. {missing_files[:5]}",
                "Archive only top candidates, but keep or mark missing artifacts explicitly.",
            )
        )

    gaps = []
    for r in rows:
        try:
            local = float(r.get("local_cv_score", ""))
            public = float(r.get("public_lb_score", ""))
        except ValueError:
            continue
        if local > 0 and public > 0 and public < 10:
            gaps.append((public - local, r))
    if gaps:
        abs_gaps = sorted(gaps, key=lambda x: abs(x[0]), reverse=True)
        stats["largest_cv_lb_gaps"] = [
            {
                "id": r.get("id"),
                "file": r.get("submission_file"),
                "local": r.get("local_cv_score"),
                "public": r.get("public_lb_score"),
                "gap": round(g, 6),
            }
            for g, r in abs_gaps[:8]
        ]
        bad = [x for x in gaps if abs(x[0]) >= 0.03]
        if bad:
            issues.append(
                Issue(
                    "high",
                    "validation",
                    "Large CV-LB gaps exist",
                    f"{len(bad)} scored rows have |public-local| >= 0.03.",
                    "Treat these as validation failures; compare their feature sets and fold strategy before ensembling them.",
                )
            )

    return issues, stats


def main() -> int:
    issues: list[Issue] = []
    stats: dict = {}

    status = run(["git", "status", "--short"]).splitlines()
    stats["git_status_entries"] = len(status)
    by_code = Counter(line[:2].strip() or "??" for line in status)
    stats["git_status_by_code"] = dict(by_code)
    if len(status) > 100:
        issues.append(
            Issue(
                "critical",
                "repo hygiene",
                "Working tree is too dirty for reliable experiments",
                f"git status has {len(status)} entries.",
                "Create a checkpoint branch, commit/restash intentional code, and keep generated artifacts ignored.",
            )
        )

    for d in [
        "data",
        "models",
        "submissions",
        "lightning_logs",
        "checkpoints",
        "scratch",
        "notebooks",
        "src",
        "scripts",
        "docs",
    ]:
        stats[f"size_{d}"] = human(size_bytes(ROOT / d))
    if size_bytes(ROOT / "models") > 2 * 1024**3:
        issues.append(
            Issue(
                "medium",
                "artifacts",
                "Model directory is very large",
                f"models/ is {stats['size_models']}.",
                "Keep only champion artifacts locally; move old runs to external storage or ignored archives.",
            )
        )
    if size_bytes(ROOT / "submissions") > 500 * 1024**2:
        issues.append(
            Issue(
                "medium",
                "artifacts",
                "Submissions directory is bloated",
                f"submissions/ is {stats['size_submissions']}.",
                "Retain top submissions plus metadata; move embeddings/OOFs to an experiments artifact directory.",
            )
        )

    tracked = run(["git", "ls-files"]).splitlines()
    tracked_generated = [p for p in tracked if p.startswith(("lightning_logs/", "catboost_info/"))]
    stats["tracked_generated_files"] = len(tracked_generated)
    if tracked_generated:
        issues.append(
            Issue(
                "high",
                "repo hygiene",
                "Generated training logs are tracked by Git",
                f"{len(tracked_generated)} lightning/catboost files are tracked.",
                "Stop tracking generated logs (`git rm --cached`) and keep summaries in docs/reports.",
            )
        )

    actual_targets = make_targets()
    referenced_targets = referenced_make_targets()
    missing_targets = sorted(referenced_targets - actual_targets)
    stats["make_targets"] = sorted(actual_targets)
    stats["referenced_missing_make_targets"] = missing_targets
    if missing_targets:
        issues.append(
            Issue(
                "high",
                "commands",
                "Docs reference missing Make targets",
                f"Missing targets: {missing_targets}",
                "Either add aliases or update docs so every advertised command runs.",
            )
        )

    missing_links = find_missing_markdown_links()
    stats["missing_markdown_links"] = missing_links[:50]
    if missing_links:
        issues.append(
            Issue(
                "medium",
                "docs",
                "Markdown links point to missing paths",
                f"{len(missing_links)} missing local links found.",
                "Update links after the references -> docs/competition migration and restore submission history references.",
            )
        )

    sub_issues, sub_stats = audit_submission_log()
    issues.extend(sub_issues)
    stats.update(sub_stats)

    # Script references in Makefile.
    makefile = (ROOT / "Makefile").read_text(errors="ignore") if (ROOT / "Makefile").exists() else ""
    missing_scripts = []
    for script in sorted(set(re.findall(r"scripts/[A-Za-z0-9_./-]+\.py", makefile))):
        if not (ROOT / script).exists():
            missing_scripts.append(script)
    stats["missing_scripts_referenced_by_makefile"] = missing_scripts
    if missing_scripts:
        issues.append(
            Issue(
                "critical",
                "commands",
                "Makefile references missing scripts",
                f"Missing scripts: {missing_scripts}",
                "Restore scripts or change Make targets to existing entrypoints.",
            )
        )

    report = []
    report.append("# Competition Workspace Audit\n")
    report.append("## Executive Summary\n")
    critical = sum(i.severity == "critical" for i in issues)
    high = sum(i.severity == "high" for i in issues)
    medium = sum(i.severity == "medium" for i in issues)
    report.append(f"Found **{len(issues)} issues**: {critical} critical, {high} high, {medium} medium.\n")
    report.append("## Issues\n")
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for i, issue in enumerate(sorted(issues, key=lambda x: (order.get(x.severity, 9), x.area, x.title)), 1):
        report.append(f"### {i}. [{issue.severity.upper()}] {issue.title}\n")
        report.append(f"- Area: `{issue.area}`\n")
        report.append(f"- Detail: {issue.detail}\n")
        report.append(f"- Recommendation: {issue.recommendation}\n")
    report.append("## Key Stats\n")
    for k in sorted(stats):
        v = stats[k]
        if isinstance(v, (list, dict)):
            report.append(f"- `{k}`: `{json.dumps(v, ensure_ascii=False)[:1000]}`\n")
        else:
            report.append(f"- `{k}`: `{v}`\n")

    out_md = ROOT / "reports" / "workspace_audit.md"
    out_json = ROOT / "reports" / "workspace_audit.json"
    out_md.write_text("\n".join(report))
    out_json.write_text(
        json.dumps({"issues": [asdict(i) for i in issues], "stats": stats}, ensure_ascii=False, indent=2)
    )
    print(f"Wrote {out_md.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Issues: {len(issues)} ({critical} critical, {high} high, {medium} medium)")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
