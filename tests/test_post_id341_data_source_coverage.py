import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_data_source_coverage.py"


def load_module():
    spec = importlib.util.spec_from_file_location("data_source_coverage", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_data_source_coverage_is_ready_but_no_submit():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["data_source_coverage_audit_ready"] is True
    assert decision["all_major_sources_present"] is True
    assert decision["all_required_evidence_reports_present"] is True
    assert decision["upstream_runner_references_required_source_checks"] is True
    assert decision["data_source_coverage_ready"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["run_kaggle_now"] is False


def test_external_data_is_present_but_direct_concat_is_blocked():
    module = load_module()
    report = module.build_report()

    external_items = report["coverage"]["external_etri"]["items"]
    assert set(external_items) == set(module.EXTERNAL_ETRI_FILES)
    assert all(item["exists"] for item in external_items.values())
    assert report["decision"]["external_direct_concat_allowed"] is False
    assert report["decision"]["external_prior_reviewable"] is False


def test_raw_modalities_and_runner_checks_are_complete():
    module = load_module()
    report = module.build_report()

    raw_items = report["coverage"]["raw_modalities"]["items"]
    assert set(raw_items) == set(module.RAW_MODALITY_FILES)
    assert all(item["exists"] for item in raw_items.values())
    assert all((item.get("rows") or 0) > 0 for item in raw_items.values())
    assert all(report["coverage"]["kaggle_runner_patterns"]["patterns"].values())


def test_markdown_records_no_id342_and_no_external_concat(tmp_path):
    module = load_module()
    report = module.build_report()
    output = tmp_path / "coverage.md"

    module.write_markdown(output, json.loads(json.dumps(report)))
    text = output.read_text(encoding="utf-8")

    assert "Data/source coverage ready: `True`" in text
    assert "External direct concat allowed: `False`" in text
    assert "Do not concatenate external rows or create ID342" in text
