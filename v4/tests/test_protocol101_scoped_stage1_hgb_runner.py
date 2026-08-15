from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as runner


def _args(*, mode: str, smoke_approved: bool, training_approved: bool) -> Namespace:
    return Namespace(
        mode=mode,
        out_dir=(
            runner.DEFAULT_FRESH_OUT
            if mode in {"dry-run", "train-hypothesis"}
            else Path("/tmp/protocol101-test-runner")
        ),
        readiness=runner.DEFAULT_READINESS,
        hypothesis="H0",
        owner_approved_plumbing_smoke=smoke_approved,
        owner_approved_offline_training=training_approved,
        force=False,
        smoke_rows_per_session=20,
    )


def test_dry_run_is_ready_after_owner_signatures() -> None:
    plan = runner.runner_plan(
        _args(mode="dry-run", smoke_approved=False, training_approved=False)
    )
    assert plan["status"] == "v5_core_ready_pending_independent_acceptance"
    assert plan["blockers"] == []
    assert plan["contract_id"] == "protocol101-scoped-canonical-stage1-v1"
    assert len(plan["feature_names"]) == 17
    assert plan["fold_count"] == 5
    assert plan["fold_eligible_sessions"] == 271
    assert plan["simulator_version"].startswith(
        "protocol101_serial_simulator_v5"
    )
    assert plan["identity_receipt"]["status"] == "PASS"
    assert plan["campaign_ready"] is False
    assert plan["side_effects"]["research_model_training_executed"] is False


def test_smoke_and_training_modes_are_separately_owner_locked() -> None:
    smoke = runner.runner_plan(
        _args(mode="plumbing-smoke", smoke_approved=False, training_approved=False)
    )
    training = runner.runner_plan(
        _args(mode="train-hypothesis", smoke_approved=True, training_approved=False)
    )
    assert "owner_approval_missing:plumbing_smoke" in smoke["blockers"]
    assert "owner_approval_missing:offline_training" in training["blockers"]
    assert (
        "fresh_campaign_execution_authorization_not_present"
        in training["blockers"]
    )


def test_smoke_approval_does_not_authorize_research_training() -> None:
    plan = runner.runner_plan(
        _args(mode="train-hypothesis", smoke_approved=True, training_approved=False)
    )
    assert plan["status"] == "blocked"
    assert "owner_approval_missing:offline_training" in plan["blockers"]
