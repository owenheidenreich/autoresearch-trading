from __future__ import annotations

import json
from pathlib import Path

import pytest

from v4.scripts import run_protocol101_full_trader_stage1_entry_campaign as campaign


def test_owner_authorization_is_exact() -> None:
    authorization, observed_hash = campaign.verify_owner_root()
    assert authorization["authorized"] is True
    assert authorization["seed_45_or_G9_authorized"] is False
    assert observed_hash == campaign.graph.owner_authorization_sha256(
        authorization
    )


def test_accepted_plans_are_unblocked_and_exclude_seed45() -> None:
    for hypothesis in campaign.HYPOTHESES:
        _args, plan = campaign.accepted_plan(hypothesis)
        assert plan["blockers"] == []
        assert plan["seeds"] == [42, 43, 44]
        assert plan["seed_45_status"] == "protected_not_executable"


def test_accepted_plan_binds_durable_wrapper_and_scientific_runner() -> None:
    _args, plan = campaign.accepted_plan("H0")
    provenance = plan["provenance"]
    hashes = plan["code_hashes"]
    assert provenance["runner_source_sha256"] == campaign.sha256_path(
        campaign.durable.RUNNER_PATH
    )
    assert provenance["scientific_runner_source_sha256"] == campaign.sha256_path(
        campaign.durable.ORIGINAL_RUNNER_PATH
    )
    assert str(campaign.durable.RUNNER_PATH) in hashes
    assert str(campaign.durable.ORIGINAL_RUNNER_PATH) in hashes
    serializer_path = str(Path(campaign.model_artifact.__file__))
    assert serializer_path in hashes
    assert (
        provenance["fresh_model_artifact_serializer_source_sha256"]
        == campaign.sha256_path(Path(serializer_path))
    )
    binding = plan["identity_receipt"]["fresh_model_artifact_binding"]
    assert binding["status"] == "PASS"
    assert binding["prediction_semantics_changed"] is False


def test_progress_is_atomic_and_preserves_forbidden_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(campaign, "FITTED_ROOT", tmp_path / campaign.CAMPAIGN_NAMESPACE)
    campaign.update_progress(
        status="test",
        current_hypothesis="H0",
        training=False,
    )
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["status"] == "test"
    assert payload["completed_units"] == 0
    assert payload["forbidden_action_flags"]["seed_45_or_G9_executed"] is False
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_progress_preserves_resolved_blocker_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "resolved_blocker_history": [
                    {"exception": "ValueError: exact blocker", "status": "resolved"}
                ]
            }
        )
    )
    monkeypatch.setattr(campaign, "PROGRESS_PATH", progress)
    monkeypatch.setattr(campaign, "FITTED_ROOT", tmp_path / campaign.CAMPAIGN_NAMESPACE)

    campaign.update_progress(
        status="training_or_resuming",
        current_hypothesis="H0",
        training=True,
    )

    payload = json.loads(progress.read_text())
    assert payload["resolved_blocker_history"] == [
        {"exception": "ValueError: exact blocker", "status": "resolved"}
    ]


def test_old_model_namespace_is_not_fresh_namespace() -> None:
    assert "entry_fresh_attempt001" in str(campaign.FITTED_ROOT)
    assert "scoped_canonical_stage1_h" not in str(campaign.FITTED_ROOT)


def _write_test_unit_summary(path: Path) -> str:
    payload = {
        "schema_version": "Protocol101FreshEntryUnitSummaryV1",
        "unit": {
            "config": {"fee": 3.0},
            "validation": {
                "primary_noise_scale": 1.0,
                "primary_noise_seed": 42,
                "decision_count": 1,
                "candidate_count": 1,
                "metrics": {},
                "expected_calibration_error": 0.0,
                "calibration_observations": [],
                "fee_sensitivity": {},
                "fill_edge_band": {},
                "noise_diagnostics": {},
                "simulator_semantics": {},
                "diagnostics": [
                    {
                        "calibrated_confidence": 0.75,
                        "selected_label_before_fee": 4.0,
                    }
                ],
                "entry_intents": [],
                "trades": [],
                "skipped_events": [],
            },
        },
    }
    payload["summary_hash"] = campaign.unit_summary.stable_hash(payload)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return campaign.sha256_path(path)


def test_unit_summary_reference_supports_mixed_compact_and_full_units(
    tmp_path: Path,
) -> None:
    compact_path = tmp_path / "compact" / "summary.json"
    full_path = tmp_path / "full" / "summary.json"
    compact_original_sha256 = _write_test_unit_summary(compact_path)
    full_sha256 = _write_test_unit_summary(full_path)
    campaign.unit_summary.compact_summary(compact_path)

    observed = [
        campaign.load_verified_unit_summary_reference(path)[1]
        for path in (compact_path, full_path)
    ]

    compact_sha256 = campaign.sha256_path(compact_path)
    assert observed == [
        {compact_original_sha256, compact_sha256},
        {full_sha256},
    ]
    assert compact_sha256 != compact_original_sha256
