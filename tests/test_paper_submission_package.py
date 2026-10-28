import json
import math
from pathlib import Path

import pytest

from scripts import audit_paper_submission_package as package_audit
from scripts.extract_paper_submission_metadata import build_metadata, load_edas_receipt


def write_tex(path: Path) -> None:
    path.write_text(
        r"""
\newcommand{\paperauthorname}{Shangjae Lee}
\newcommand{\paperaffiliation}{Kyungpook National University}
\newcommand{\paperemail}{dltkdwo31@knu.ac.kr}
\title{Test Paper}
\begin{abstract}A short abstract for metadata extraction.\end{abstract}
\begin{IEEEkeywords}lifelog, ensemble learning\end{IEEEkeywords}
""",
        encoding="utf-8",
    )


def write_receipt(path: Path, submitted_pdf: Path, *, include_status: bool = True) -> None:
    receipt = {
        "observed_at_kst": "2026-07-17 17:56:39 UTC+09:00",
        "paper_id": "1571325954",
        "registered_title": "Test Paper",
        "review_manuscript_uploaded": True,
        "source": "Author-confirmed EDAS paper record",
        "submitted_pdf": str(submitted_pdf),
        "submitted_pdf_sha256": "placeholder",
    }
    if include_status:
        receipt["status"] = "Active (has manuscript)"
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_metadata_uses_edas_receipt(tmp_path: Path) -> None:
    tex = tmp_path / "paper.tex"
    receipt = tmp_path / "edas_receipt.json"
    write_tex(tex)
    write_receipt(receipt, tmp_path / "submitted.pdf")

    metadata = build_metadata(tex, receipt)

    assert metadata["edas_paper_id"] == "1571325954"
    assert metadata["edas_status"] == "Active (has manuscript)"
    assert metadata["review_manuscript_uploaded"] is True
    assert metadata["registered_title_matches_source"] is True


def test_receipt_rejects_missing_required_field(tmp_path: Path) -> None:
    receipt = tmp_path / "edas_receipt.json"
    write_receipt(receipt, tmp_path / "submitted.pdf", include_status=False)

    with pytest.raises(ValueError, match="missing fields"):
        load_edas_receipt(receipt)


def test_pdf_status_requires_current_build_and_receipt_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "paper.pdf"
    compiled = tmp_path / "compiled.pdf"
    submitted = tmp_path / "submitted.pdf"
    receipt = tmp_path / "edas_receipt.json"
    current.write_bytes(b"current")
    compiled.write_bytes(b"current")
    submitted.write_bytes(b"submitted")
    submitted_sha = package_audit.sha256_file(submitted)
    receipt.write_text(
        json.dumps(
            {
                "submitted_pdf": "submitted.pdf",
                "submitted_pdf_sha256": submitted_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(package_audit, "CURRENT_PDF", current)
    monkeypatch.setattr(package_audit, "COMPILED_PDF", compiled)
    monkeypatch.setattr(package_audit, "SUBMITTED_PDF", submitted)
    monkeypatch.setattr(package_audit, "EDAS_RECEIPT", receipt)
    monkeypatch.setattr(package_audit, "ROOT", tmp_path)

    assert package_audit.pdf_status()["matches_expected"] is True

    compiled.write_bytes(b"stale")
    assert package_audit.pdf_status()["matches_expected"] is False


def test_mean_log_loss_and_midpoint_use_all_targets() -> None:
    key = ("subject", "2026-01-01")
    labels = {key: dict.fromkeys(package_audit.TARGETS, 1.0)}
    left = {key: dict.fromkeys(package_audit.TARGETS, 0.6)}
    right = {key: dict.fromkeys(package_audit.TARGETS, 0.8)}

    midpoint = package_audit.midpoint_predictions(left, right)

    assert midpoint[key] == {target: pytest.approx(0.7) for target in package_audit.TARGETS}
    assert package_audit.mean_log_loss(labels, midpoint) == pytest.approx(-math.log(0.7))


def test_lineage_attribution_uses_tracked_evidence_without_private_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_text = (package_audit.PAPER_DIR / "paper.tex").read_text(encoding="utf-8")
    clean_root = tmp_path / "clean-clone"
    evidence_path = clean_root / "docs" / "paper_submission_20260626" / "lineage_attribution_evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(package_audit.LINEAGE_ATTRIBUTION_EVIDENCE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(package_audit, "ROOT", clean_root)
    monkeypatch.setattr(package_audit, "LINEAGE_ATTRIBUTION_EVIDENCE", evidence_path)
    monkeypatch.setattr(package_audit, "METRICS_TRAIN", clean_root / "data" / "raw" / "ch2026_metrics_train.csv")
    monkeypatch.setattr(
        package_audit,
        "FEATURE_SIMILARITY_REPORT",
        clean_root / "reports" / "id218_public_feature_similarity_narrow_grid_20260616.json",
    )
    monkeypatch.setattr(
        package_audit,
        "OOF_FILES",
        {
            "178": clean_root
            / "artifacts/base_predictions/base_oof_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638.csv",
            "218": clean_root
            / "artifacts/base_predictions/base_oof_id_218_id215_multi_signal_clean_overlay_cv0.534103.csv",
            "221": clean_root
            / "artifacts/base_predictions/base_oof_id_221_id218_public_feature_similarity_overlay_cv0.532990.csv",
            "342": clean_root
            / "artifacts/base_predictions/base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
        },
    )

    status = package_audit.lineage_attribution_status(paper_text)

    assert status["matches_expected"] is True
    assert status["local_recomputation"]["available"] is False
    assert status["local_recomputation"]["matches_evidence"] is None
    assert all(status["method_checks"].values())
    assert all(status["metric_checks"].values())
    assert all(status["alpha_sensitivity_checks"].values())
    assert all(status["metric_consistency_checks"].values())


def test_lineage_attribution_rejects_changed_source_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paper_text = (package_audit.PAPER_DIR / "paper.tex").read_text(encoding="utf-8")
    evidence = json.loads(package_audit.LINEAGE_ATTRIBUTION_EVIDENCE.read_text(encoding="utf-8"))
    evidence["source_artifacts"]["id342_oof"]["sha256"] = "0" * 64
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(package_audit, "LINEAGE_ATTRIBUTION_EVIDENCE", evidence_path)

    status = package_audit.lineage_attribution_status(paper_text)

    assert status["matches_expected"] is False
    assert status["source_records_valid"]["id342_oof"] is False


def test_lineage_attribution_rejects_invalid_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text('{"method": [], "metrics": {}, "source_artifacts": {}}', encoding="utf-8")
    monkeypatch.setattr(package_audit, "LINEAGE_ATTRIBUTION_EVIDENCE", evidence_path)

    status = package_audit.lineage_attribution_status("paper claims")

    assert status["schema_valid"] is False
    assert status["matches_expected"] is False
