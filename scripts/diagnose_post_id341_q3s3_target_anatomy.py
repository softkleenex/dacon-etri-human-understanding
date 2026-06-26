#!/usr/bin/env python3
"""Diagnose Q3/S3 target anatomy after post-ID341 local source closures.

This report-only diagnostic inspects the official training labels for the two
still-uncovered targets, Q3 and S3. It does not train a model, produce
predictions, create submission artifacts, edit the submission log, run Kaggle,
or submit to DACON. The goal is to ground the remaining hypothesis space in the
label structure before any further local source-family work.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TRAIN_PATH = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
NEGATIVE_MAP = REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json"
FALLBACK_CHAIN = REPORTS_DIR / "post_id341_q2q3s3_local_fallback_chain_contract_20260618.json"
FRONTIER = REPORTS_DIR / "post_id341_no_approval_frontier_20260619.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_target_anatomy_20260619.json"
KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FOCUS_TARGETS = ["Q3", "S3"]
ID178_PUBLIC = 0.6014113009
PUBLIC_058 = 0.58


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "parse_ok": False, "decision": {}, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "parse_ok": False, "decision": {}, "path": rel(path), "error": str(exc)}
    payload.setdefault("path", rel(path))
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def constant_logloss(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return binary_entropy(p)


def lag1_stats(df: pd.DataFrame, target: str) -> dict[str, Any]:
    pairs: list[tuple[int, int]] = []
    for _, group in df.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        values = group[target].astype(int).tolist()
        pairs.extend(zip(values[:-1], values[1:], strict=False))
    if not pairs:
        return {"pairs": 0, "agreement_rate": None, "positive_after_positive": None, "positive_after_negative": None}
    prev = pd.Series([p for p, _ in pairs], dtype=float)
    cur = pd.Series([c for _, c in pairs], dtype=float)
    prev_pos = cur[prev == 1]
    prev_neg = cur[prev == 0]
    return {
        "pairs": int(len(pairs)),
        "agreement_rate": float((prev == cur).mean()),
        "positive_after_positive": None if prev_pos.empty else float(prev_pos.mean()),
        "positive_after_negative": None if prev_neg.empty else float(prev_neg.mean()),
    }


def target_summary(df: pd.DataFrame, target: str) -> dict[str, Any]:
    y = df[target].astype(float)
    prevalence = float(y.mean())
    per_subject = df.groupby("subject_id")[target].mean().sort_index()
    return {
        "target": target,
        "positive_rate": prevalence,
        "negative_rate": float(1.0 - prevalence),
        "constant_logloss_floor": constant_logloss(prevalence),
        "class_balance_gap": float(abs(prevalence - 0.5)),
        "subject_positive_rate_std": float(per_subject.std(ddof=0)),
        "subject_positive_rate_min": float(per_subject.min()),
        "subject_positive_rate_max": float(per_subject.max()),
        "subject_positive_rate_range": float(per_subject.max() - per_subject.min()),
        "lag1": lag1_stats(df, target),
    }


def same_day_correlations(df: pd.DataFrame, focus: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        if target == focus:
            continue
        corr = df[[focus, target]].corr().iloc[0, 1]
        joint_positive = float(((df[focus] == 1) & (df[target] == 1)).mean())
        focus_pos = df[df[focus] == 1][target]
        focus_neg = df[df[focus] == 0][target]
        rows.append(
            {
                "target": target,
                "pearson_corr": None if pd.isna(corr) else float(corr),
                "joint_positive_rate": joint_positive,
                f"{target}_positive_given_{focus}_positive": None if focus_pos.empty else float(focus_pos.mean()),
                f"{target}_positive_given_{focus}_negative": None if focus_neg.empty else float(focus_neg.mean()),
            }
        )
    return sorted(rows, key=lambda row: abs(row["pearson_corr"] or 0.0), reverse=True)


def build_report() -> dict[str, Any]:
    df = pd.read_csv(TRAIN_PATH)
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    required_cols = {"subject_id", "lifelog_date", *TARGETS}
    missing_cols = sorted(required_cols - set(df.columns))
    target_summaries = {target: target_summary(df, target) for target in FOCUS_TARGETS}
    correlations = {target: same_day_correlations(df, target) for target in FOCUS_TARGETS}
    negative_map = load_json(NEGATIVE_MAP)
    fallback_chain = load_json(FALLBACK_CHAIN)
    frontier = load_json(FRONTIER)
    negative_decision = negative_map.get("decision", {})
    fallback_decision = fallback_chain.get("decision", {})
    frontier_decision = frontier.get("decision", {})
    evidence_checks = {
        "train_file_exists": TRAIN_PATH.exists(),
        "required_columns_present": not missing_cols,
        "row_count_is_450": len(df) == 450,
        "subject_count_is_10": df["subject_id"].nunique() == 10,
        "q3s3_negative_map_closed": negative_decision.get("all_local_q3s3_lanes_closed") is True,
        "fallback_keeps_q2_context_only": fallback_decision.get("local_q2_lead_context_only") is True,
        "frontier_has_no_no_approval_score_route": frontier_decision.get("no_approval_score_route_exists") is False,
    }
    anatomy_read = {
        "q3_is_subjective_stress": True,
        "s3_is_sleep_onset_latency_guideline": True,
        "q3_positive_rate": target_summaries["Q3"]["positive_rate"],
        "s3_positive_rate": target_summaries["S3"]["positive_rate"],
        "q3_lag1_agreement_rate": target_summaries["Q3"]["lag1"]["agreement_rate"],
        "s3_lag1_agreement_rate": target_summaries["S3"]["lag1"]["agreement_rate"],
        "q3_top_same_day_correlation": correlations["Q3"][0],
        "s3_top_same_day_correlation": correlations["S3"][0],
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_target_anatomy",
        "purpose": "Inspect Q3/S3 label structure before reopening any post-ID341 local source-family work.",
        "inputs": {
            "train_metrics": rel(TRAIN_PATH),
            "negative_evidence_map": negative_map.get("path", rel(NEGATIVE_MAP)),
            "fallback_chain": fallback_chain.get("path", rel(FALLBACK_CHAIN)),
            "no_approval_frontier": frontier.get("path", rel(FRONTIER)),
        },
        "dataset": {
            "rows": int(len(df)),
            "subjects": int(df["subject_id"].nunique()),
            "date_min": str(df["lifelog_date"].min().date()),
            "date_max": str(df["lifelog_date"].max().date()),
            "missing_required_columns": missing_cols,
        },
        "target_focus": FOCUS_TARGETS,
        "target_summaries": target_summaries,
        "same_day_correlations": correlations,
        "anatomy_read": anatomy_read,
        "evidence_checks": evidence_checks,
        "implications": [
            {
                "target": "Q3",
                "read": "Subjective stress is a relative-to-self evening questionnaire label; local metadata/routine lanes have not supplied enough Q3/S3 transfer evidence.",
                "allowed_next_work": "Only a materially new stress-specific source design or the approved upstream rebuild; no artifact from current local sources.",
            },
            {
                "target": "S3",
                "read": "S3 is objective sleep-onset-latency guideline adherence; closed lanes already tested light/onset, calendar, cadence, routine, place/social, dropout, and cross-modal disagreement views.",
                "allowed_next_work": "A new S3 hypothesis must explain SOL directly and clear negative-map novelty before any model smoke.",
            },
        ],
        "decision": {
            "q3s3_target_anatomy_ready": all(evidence_checks.values()),
            "target_focus": FOCUS_TARGETS,
            "label_only_signal_is_artifact_candidate": False,
            "current_local_sources_still_closed": negative_decision.get("all_local_q3s3_lanes_closed") is True,
            "q3_s3_still_uncovered": negative_decision.get("q3_s3_still_uncovered") is True,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Use this anatomy only to screen future Q3/S3 hypotheses. Do not create ID342 from label-only or "
                "closed-lane evidence; the score-relevant route remains the explicit-approved upstream rebuild."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    anatomy = report["anatomy_read"]
    lines = [
        "# Post-ID341 Q3/S3 target anatomy",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model and creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Q3/S3 target anatomy ready: `{decision['q3s3_target_anatomy_ready']}`",
        f"- Current local sources still closed: `{decision['current_local_sources_still_closed']}`",
        f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
        f"- Public 0.58 evidence: `{decision['public_058_evidence']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Label read",
        "",
        f"- Rows: `{report['dataset']['rows']}`, subjects: `{report['dataset']['subjects']}`",
        f"- Q3 positive rate: `{anatomy['q3_positive_rate']:.6f}`; lag-1 agreement: `{anatomy['q3_lag1_agreement_rate']:.6f}`",
        f"- S3 positive rate: `{anatomy['s3_positive_rate']:.6f}`; lag-1 agreement: `{anatomy['s3_lag1_agreement_rate']:.6f}`",
        f"- Q3 top same-day correlation: `{anatomy['q3_top_same_day_correlation']['target']}` corr `{anatomy['q3_top_same_day_correlation']['pearson_corr']:.6f}`",
        f"- S3 top same-day correlation: `{anatomy['s3_top_same_day_correlation']['target']}` corr `{anatomy['s3_top_same_day_correlation']['pearson_corr']:.6f}`",
        "",
        "## Target summaries",
        "",
    ]
    for target, row in report["target_summaries"].items():
        lines.extend(
            [
                f"### {target}",
                f"- Positive rate: `{row['positive_rate']:.6f}`",
                f"- Constant logloss floor: `{row['constant_logloss_floor']:.6f}`",
                f"- Subject positive-rate range: `{row['subject_positive_rate_range']:.6f}`",
                f"- Lag-1 agreement: `{row['lag1']['agreement_rate']:.6f}`",
                "",
            ]
        )
    lines.extend(["## Evidence checks", ""])
    for key, value in report["evidence_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0 if report["decision"]["q3s3_target_anatomy_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
