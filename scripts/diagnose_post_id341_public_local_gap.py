#!/usr/bin/env python3
"""Diagnose why local CV is not transferring to DACON public LB after ID341.

This is intentionally local-only. It reads the submission log and existing
parity/closure reports, then writes a public-local gap diagnosis without
creating a submission artifact or touching Kaggle/DACON.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"

TARGET_PUBLIC = 0.58
ID178_PUBLIC_ANCHOR = 0.6014113009
RECENT_REFERENCE_IDS = (178, 329, 337, 340)
WATCH_START_ID = 334


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def round_float(value: Any, digits: int = 9) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def finite_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def correlation_block(rows: pd.DataFrame) -> dict[str, Any]:
    pair = rows[["local_cv_score_num", "public_lb_score_num"]].dropna()
    if len(pair) < 3:
        return {"n": int(len(pair)), "pearson": None, "spearman": None, "kendall": None}
    return {
        "n": int(len(pair)),
        "pearson": round_float(pair.corr(method="pearson").iloc[0, 1]),
        "spearman": round_float(pair.corr(method="spearman").iloc[0, 1]),
        "kendall": round_float(pair.corr(method="kendall").iloc[0, 1]),
    }


def actual_public_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with real-looking public LB scores, excluding placeholders."""
    actual_statuses = {"scored", "cv_lb_gap_risk"}
    rows = df[df["status_clean"].isin(actual_statuses)].copy()
    rows = rows[rows["local_cv_score_num"].notna() & rows["public_lb_score_num"].notna()].copy()
    rows = rows[(rows["public_lb_score_num"] > 0) & (rows["public_lb_score_num"] < 2)].copy()
    rows["public_minus_local"] = rows["public_lb_score_num"] - rows["local_cv_score_num"]
    rows["abs_gap"] = rows["public_minus_local"].abs()
    return rows.sort_values("id")


def gap_stats(rows: pd.DataFrame) -> dict[str, Any]:
    gaps = rows["public_minus_local"].dropna()
    if gaps.empty:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "optimistic_gap_count": 0,
            "severe_gap_count_ge_003": 0,
        }
    return {
        "n": int(len(gaps)),
        "min": round_float(gaps.min()),
        "median": round_float(gaps.median()),
        "p75": round_float(gaps.quantile(0.75)),
        "p90": round_float(gaps.quantile(0.90)),
        "max": round_float(gaps.max()),
        "optimistic_gap_count": int((gaps > 0).sum()),
        "severe_gap_count_ge_003": int((gaps.abs() >= 0.03).sum()),
    }


def compact_rows(rows: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    view = rows
    if limit is not None:
        view = view.head(limit)
    output: list[dict[str, Any]] = []
    for _, row in view.iterrows():
        output.append(
            {
                "id": int(row["id"]),
                "submission_file": row["submission_file"],
                "local_cv": round_float(row["local_cv_score_num"], 6),
                "public_lb": round_float(row["public_lb_score_num"], 9),
                "public_minus_local": round_float(row.get("public_minus_local"), 9),
                "status_clean": row["status_clean"],
                "model_description": row.get("model_description"),
            }
        )
    return output


def watch_candidates(df: pd.DataFrame, recent_gap_floor: float, recent_gap_stress: float) -> list[dict[str, Any]]:
    rows = df[
        (df["id"] >= WATCH_START_ID)
        & (df["status_clean"] == "generated_unsubmitted")
        & df["local_cv_score_num"].notna()
    ].copy()
    rows = rows.sort_values("local_cv_score_num")
    output: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        local_cv = float(row["local_cv_score_num"])
        floor_proxy = local_cv + recent_gap_floor
        stress_proxy = local_cv + recent_gap_stress
        output.append(
            {
                "id": int(row["id"]),
                "submission_file": row["submission_file"],
                "local_cv": round(local_cv, 6),
                "recent_floor_proxy": round(floor_proxy, 6),
                "recent_stress_proxy": round(stress_proxy, 6),
                "passes_public_058_floor_proxy": bool(floor_proxy <= TARGET_PUBLIC),
                "passes_public_058_stress_proxy": bool(stress_proxy <= TARGET_PUBLIC),
            }
        )
    return output


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    decision = report["decision"]
    references = report["stress_references"]
    watch = report["watch_candidates"][:8]
    lines = [
        "# Post-ID341 Public/Local Gap Diagnosis",
        "",
        "## Decision",
        "",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Public 0.58 local CV target under recent floor: `{decision['local_cv_needed_for_public_058_recent_floor']}`",
        f"- Public 0.58 local CV target under recent stress: `{decision['local_cv_needed_for_public_058_recent_stress']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Stress references",
        "",
        f"- Reference gaps: `{references['reference_gaps']}`",
        f"- Recent floor gap: `{references['recent_gap_floor']}`",
        f"- Recent stress gap: `{references['recent_gap_stress']}`",
        "",
        "## Watch candidates by recent proxy",
        "",
        "| id | local_cv | floor_proxy | stress_proxy | pass_floor | pass_stress |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in watch:
        lines.append(
            "| {id} | {local_cv:.6f} | {recent_floor_proxy:.6f} | {recent_stress_proxy:.6f} | {floor} | {stress} |".format(
                id=row["id"],
                local_cv=row["local_cv"],
                recent_floor_proxy=row["recent_floor_proxy"],
                recent_stress_proxy=row["recent_stress_proxy"],
                floor="yes" if row["passes_public_058_floor_proxy"] else "no",
                stress="yes" if row["passes_public_058_stress_proxy"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Recent public evidence says local CV alone is not a promotion signal. "
            "The current watch candidates only look promising under older optimistic gaps, "
            "not under the recent confirmed-control transfer floor. Therefore this report does not authorize "
            "a DACON submission or a new local ID342 artifact.",
            "",
            "The Mac/M4 suspicion is tracked separately, but the existing parity evidence and the current "
            "dependency snapshot do not make it the primary explanation for the public-local gap.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report() -> dict[str, Any]:
    df = pd.read_csv(SUBMISSION_LOG)
    actual = actual_public_rows(df)
    recent_actual = actual[actual["id"] >= 85].copy()

    reference_rows = actual[actual["id"].isin(RECENT_REFERENCE_IDS)].copy()
    missing_reference_ids = sorted(set(RECENT_REFERENCE_IDS) - set(reference_rows["id"].astype(int).tolist()))
    if 178 in missing_reference_ids or 329 in missing_reference_ids:
        raise RuntimeError("Required public anchor rows ID178 and ID329 are missing from the submission log.")

    reference_gaps = {
        f"id{int(row['id'])}_gap": round_float(row["public_minus_local"])
        for _, row in reference_rows.sort_values("id").iterrows()
    }
    recent_gap_floor = min(float(row["public_minus_local"]) for _, row in reference_rows.iterrows())
    recent_gap_stress = max(float(row["public_minus_local"]) for _, row in reference_rows.iterrows())

    watch = watch_candidates(df, recent_gap_floor=recent_gap_floor, recent_gap_stress=recent_gap_stress)
    any_watch_floor_pass = any(row["passes_public_058_floor_proxy"] for row in watch)
    any_watch_stress_pass = any(row["passes_public_058_stress_proxy"] for row in watch)

    parity_report = read_json(REPORTS_DIR / "candidate_parity_crossplatform_gate_post_id339_20260618.json")
    local_closure = read_json(REPORTS_DIR / "post_id341_local_source_closure_synthesis_20260618.json")

    environment_snapshot = {
        "tensorflow": package_version("tensorflow"),
        "tensorflow_macos": package_version("tensorflow-macos"),
        "tensorflow_metal": package_version("tensorflow-metal"),
        "torch": package_version("torch"),
        "lightgbm": package_version("lightgbm"),
        "xgboost": package_version("xgboost"),
        "catboost": package_version("catboost"),
        "scikit_learn": package_version("scikit-learn"),
    }

    report = {
        "inputs": {
            "submission_log": relpath(SUBMISSION_LOG),
            "candidate_parity_crossplatform_gate": "reports/candidate_parity_crossplatform_gate_post_id339_20260618.json",
            "local_source_closure": "reports/post_id341_local_source_closure_synthesis_20260618.json",
        },
        "summary": {
            "actual_public_rows": int(len(actual)),
            "recent_actual_public_rows_id_ge_85": int(len(recent_actual)),
            "target_public": TARGET_PUBLIC,
            "best_actual_public": compact_rows(actual.sort_values("public_lb_score_num"), limit=5),
            "worst_transfer_gaps": compact_rows(actual.sort_values("public_minus_local", ascending=False), limit=5),
        },
        "correlations": {
            "all_actual_public": correlation_block(actual),
            "recent_actual_id_ge_85": correlation_block(recent_actual),
        },
        "gap_stats": {
            "all_actual_public": gap_stats(actual),
            "recent_actual_id_ge_85": gap_stats(recent_actual),
        },
        "stress_references": {
            "reference_ids": list(RECENT_REFERENCE_IDS),
            "available_reference_ids": [int(row["id"]) for _, row in reference_rows.sort_values("id").iterrows()],
            "missing_reference_ids": missing_reference_ids,
            "reference_gaps": reference_gaps,
            "recent_gap_floor": round_float(recent_gap_floor),
            "recent_gap_stress": round_float(recent_gap_stress),
            "id178_public_anchor": ID178_PUBLIC_ANCHOR,
            "local_cv_needed_for_public_058_recent_floor": round_float(TARGET_PUBLIC - recent_gap_floor, 6),
            "local_cv_needed_for_public_058_recent_stress": round_float(TARGET_PUBLIC - recent_gap_stress, 6),
        },
        "watch_candidates": watch,
        "environment_snapshot": environment_snapshot,
        "supporting_reports": {
            "candidate_parity_decision": (parity_report or {}).get("decision"),
            "local_source_closure_decision": (local_closure or {}).get("decision"),
        },
        "decision": {
            "public_local_gap_diagnosis_ready": True,
            "mac_m4_environment_primary_explanation": False,
            "any_watch_candidate_passes_public_058_recent_floor_proxy": any_watch_floor_pass,
            "any_watch_candidate_passes_public_058_recent_stress_proxy": any_watch_stress_pass,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "local_cv_needed_for_public_058_recent_floor": round_float(TARGET_PUBLIC - recent_gap_floor, 6),
            "local_cv_needed_for_public_058_recent_stress": round_float(TARGET_PUBLIC - recent_gap_stress, 6),
            "next_action": (
                "Do not promote local CV-only candidates. The next score-relevant move remains the "
                "explicit-approved no-submit Kaggle/Linux upstream rebuild, followed by a separate "
                "Public-0.58 artifact-contract review only if pulled reports are safe, beat the current "
                "confirmed public anchor band, "
                "and show min_proxy <= 0.58; ID178-only improvement above 0.58 is a lead, not ID342."
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    report = build_report()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = output.with_suffix(".md")
    write_markdown(md_path, report)
    print(f"Wrote {relpath(output)}")
    print(f"Wrote {relpath(md_path)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
