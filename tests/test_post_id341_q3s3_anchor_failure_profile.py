import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_anchor_failure_profile.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_anchor_failure_profile", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_q3s3_anchor_failure_profile_is_report_only_and_blocks_submission_actions():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert report["diagnostic"] == "post_id341_q3s3_anchor_failure_profile"
    assert decision["anchor_failure_profile_ready"] is True
    assert decision["q3s3_oof_gain_alone_is_submission_evidence"] is False
    assert decision["public_058_evidence"] is False
    assert decision["manual_probe_if_spending_submission"] is None
    assert decision["manual_probe_family_status"] == "consumed_after_id341_id340_id337_public_micro_probe_slate"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["no_model_trained"] is True
    assert report["no_new_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_dacon_submission_made"] is True


def test_q3s3_anchor_failure_profile_records_known_public_traps_and_upstream_signal():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert report["target_focus"] == ["Q3", "S3"]
    assert set(report["candidate_ids"]) == {178, 221, 226, 287, 329, 341}
    assert set(report["known_public_ids"]) == {178, 287, 329}
    assert 287 in decision["q3s3_known_oof_public_trap_ids"]
    assert 329 in decision["q3s3_known_oof_public_trap_ids"]
    assert decision["unsubmitted_upstream_has_q3s3_oof_signal"] is True
    assert decision["unsubmitted_upstream_requires_rebuild_not_direct_submit"] is True

    traps = {row["candidate_id"]: row for row in report["known_public_traps"]}
    assert traps[287]["public_delta_vs_id178"] > 0
    assert traps[287]["q3s3_mean_oof_delta_vs_id178"] < 0
    assert traps[329]["public_delta_vs_id178"] > 0
    assert traps[329]["q3s3_mean_oof_delta_vs_id178"] < 0


def test_q3s3_anchor_failure_profile_has_slice_and_split_evidence():
    module = load_module()
    report = module.build_report()

    rows = report["target_profile_rows"]
    assert any(row["candidate_id"] == 226 and row["target"] == "Q3" for row in rows)
    assert any(row["candidate_id"] == 287 and row["target"] == "S3" for row in rows)
    for row in rows:
        if row["target"] in {"Q3", "S3"}:
            assert set(row["split_deltas_vs_id178"]) == set(module.PRIMARY_GATE_SPLITS)

    assert set(report["persistence_slices"]) == {"Q3", "S3"}
    for target, slices in report["persistence_slices"].items():
        assert sum(row["rows"] for row in slices) == 450, target
    assert report["slice_loss_rows"]
