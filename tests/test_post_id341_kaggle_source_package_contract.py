import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
SELFTEST_SCRIPT = ROOT / "scripts" / "diagnose_kaggle_source_zip_selftest_post_id341.py"
PACKET_SCRIPT = ROOT / "scripts" / "diagnose_kaggle_upstream_execution_packet_post_id341.py"

CRITICAL_PACKAGE_ENTRIES = [
    "Makefile",
    "dataset-metadata.json",
    "scripts/diagnose_post_id341_target_gap_budget.py",
    "scripts/diagnose_post_id341_missing_target_signal_inventory.py",
    "scripts/diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py",
    "scripts/diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py",
    "scripts/diagnose_post_id341_q3s3_remaining_route_inventory.py",
    "scripts/diagnose_post_id341_q3s3_physio_light_onset_smoke.py",
    "scripts/diagnose_post_id341_q3s3_local_lane_closure.py",
    "scripts/diagnose_post_id341_q3s3_event_cadence_smoke.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_v2_design.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_metadata_smoke.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_resource_gate.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_model_smoke.py",
    "scripts/diagnose_post_id341_q3s3_semantic_routine_lane_closure.py",
    "scripts/diagnose_post_id341_q3s3_negative_evidence_map.py",
    "scripts/diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_design.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_metadata_smoke.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_spec.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_generation_dryrun.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract.py",
    "scripts/write_post_id341_usage_semantic_burst_markov_feature_matrix.py",
    "scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop.py",
    "scripts/diagnose_post_id341_self_supervised_subject_state_embedding_design.py",
    "reports/post_id341_target_gap_budget_20260618.json",
    "reports/post_id341_missing_target_signal_inventory_20260618.json",
    "reports/post_id341_q2q3s3_source_hypothesis_matrix_20260618.json",
    "reports/post_id341_q2q3s3_local_fallback_chain_contract_20260618.json",
    "reports/post_id341_q3s3_remaining_route_inventory_20260618.json",
    "reports/post_id341_q3s3_physio_light_onset_smoke_20260618.json",
    "reports/post_id341_q3s3_local_lane_closure_20260618.json",
    "reports/post_id341_q3s3_event_cadence_smoke_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_v2_design_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_metadata_smoke_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_resource_gate_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_model_smoke_20260618.json",
    "reports/post_id341_q3s3_semantic_routine_lane_closure_20260618.json",
    "reports/post_id341_q3s3_negative_evidence_map_20260618.json",
    "reports/post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_design_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_spec_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.json",
    "reports/post_id341_usage_semantic_burst_markov_features_v1_schema.json",
    "reports/post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "reports/post_id341_self_supervised_subject_state_embedding_design_20260619.json",
    "reports/place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "reports/routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "reports/sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet",
    "data/external/etri_lifelog/",
    "artifacts/base_predictions/",
    "submissions/",
    "submissions/submission_log.csv",
    "submissions/submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv",
    "submissions/submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv",
    "submissions/submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv",
    "submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_kaggle_dataset_body() -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^build-kaggle-dataset:\n(?P<body>(?:\t.*\n)+)", text, flags=re.MULTILINE)
    assert match, "missing build-kaggle-dataset target"
    return match.group("body")


def test_makefile_source_zip_packages_required_payload_roots():
    body = build_kaggle_dataset_body()
    assert "zip -r $(KAGGLE_DATASET_DIR)/source.zip" in body
    for token in [
        "src",
        "conf",
        "scripts",
        "pyproject.toml",
        "README.md",
        "Makefile",
        "dataset-metadata.json",
        "reports",
        "data/raw",
        "data/processed",
        "data/external",
        "artifacts/base_predictions",
        "submissions",
    ]:
        assert f" {token} " in f" {body} ", token
    assert "submissions/submission_log.csv" not in body


def test_kaggle_dataset_version_commands_upload_directories():
    text = MAKEFILE.read_text(encoding="utf-8")

    assert 'datasets version -p $(KAGGLE_DATASET_DIR) -m "Update source code" -r zip' in text
    assert (
        'datasets version -p $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR) -m "Refresh post-ID341 upstream patch payload" -r zip'
        in text
    )


def test_source_zip_required_entries_include_runner_dependencies():
    selftest = load_module(SELFTEST_SCRIPT, "source_zip_selftest_contract")
    packet = load_module(PACKET_SCRIPT, "upstream_execution_packet_contract")

    selftest_entries = set(selftest.REQUIRED_ENTRIES)
    packet_entries = set(packet.REQUIRED_SOURCE_ZIP_ENTRIES)
    for entry in CRITICAL_PACKAGE_ENTRIES:
        assert entry in selftest_entries
        assert entry in packet_entries
    assert packet_entries.issubset(
        selftest_entries | {"scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py"}
    )


def test_packaged_readiness_checks_all_anchor_and_blueprint_files():
    selftest = load_module(SELFTEST_SCRIPT, "source_zip_selftest_readiness_contract")
    source = SELFTEST_SCRIPT.read_text(encoding="utf-8")

    for key in [
        "makefile",
        "dataset_metadata",
        "submissions_dir",
        "submission_log",
        "anchor_submission_id178",
        "anchor_submission_id287",
        "anchor_submission_id329",
        "anchor_submission_id341",
    ]:
        assert key in source
    for entry in CRITICAL_PACKAGE_ENTRIES:
        assert entry in selftest.REQUIRED_ENTRIES


def test_source_zip_selftest_does_not_delete_unrelated_build_outputs():
    source = SELFTEST_SCRIPT.read_text(encoding="utf-8")

    assert "SOURCE_ZIP.parent" in source
    assert 'shutil.rmtree(ROOT / "build")' not in source
    assert "transient_package_dir" in source
