from __future__ import annotations

import pandas as pd
import pytest

from v5.ops import causal_day_simulator as sim
from v5.ops.export_causal_day_replay import ReplayExportError, export_replay
from v5.tests.test_causal_day_simulator import SESSION, _quotes


def _result() -> sim.SimulationResult:
    def policy(state: sim.DecisionState) -> sim.Action:
        if state.minute == "09:35":
            ids = state.entry_candidates["contract_id"].astype(str).tolist()
            return sim.Action(
                "BUY",
                "call",
                reason="fixture entry",
                probability=0.7,
                candidate_probabilities={value: 0.7 / len(ids) for value in ids},
                diagnostics={"pattern": "known_answer"},
            )
        if state.minute == "09:36":
            return sim.Action("SELL", reason="fixture exit", probability=0.8)
        return sim.Action("HOLD" if state.position else "ABSTAIN")

    return sim.simulate_session(
        _quotes(), SESSION, policy, trade_cap=1, capture_replay_state=True
    )


def test_exports_minute_events_trades_and_considered_ladder(tmp_path) -> None:
    out = tmp_path / "replay"
    receipt = export_replay(_result(), out, policy_source="known_answer_fixture")
    assert receipt["trades"] == 1
    assert receipt["considered_ladder_rows"] > 0
    report = (out / "replay.md").read_text()
    assert "Minute-by-minute decisions" in report
    assert "not evidence of model skill" in report
    assert "fixture entry" in report
    ladder = pd.read_csv(out / "considered_ladder.csv")
    assert ladder["selected"].sum() == 1


def test_out_of_fold_claim_requires_bound_artifact_hash(tmp_path) -> None:
    with pytest.raises(ReplayExportError, match="artifact SHA-256"):
        export_replay(
            _result(),
            tmp_path / "replay",
            policy_source="model",
            out_of_fold=True,
        )
