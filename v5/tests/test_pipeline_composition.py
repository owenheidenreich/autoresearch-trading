"""End-to-end plumbing: admission -> entry freeze -> packet -> session index -> gate.

Everything here is synthetic — a seeded RNG, no market data, no model.  The
point is that the rung 5-7 pipeline *composes*: each seam's artifact is
produced by one module and consumed by the next, and the known silent-failure
modes (entry re-optimization, no-trade-day dropout, caller-claimed gate
release) are refused by machinery rather than convention.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from v5.research import entry_stream as es
from v5.research import feature_admission as fa
from v5.research import gate_receipts as gr
from v5.research import training_twin as tt
from v5.research.validation import candidate_packet as cp
from v5.research.validation import replay_gate as rg
from v5.research.validation import session_index as si


SESSIONS = 60
TRADED = 50  # ten declared sessions deliberately have no trade
AS_OF = datetime(2026, 8, 10, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def calendar() -> list[str]:
    return [
        day.strftime("%Y-%m-%d")
        for day in pd.bdate_range("2026-01-05", periods=SESSIONS)
    ]


@pytest.fixture(scope="module")
def folds(calendar) -> dict[str, str]:
    return {
        session: f"fold_{index // (SESSIONS // 5)}"
        for index, session in enumerate(calendar)
    }


def _synthetic_ledger(tmp_path) -> tuple:
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    features = [
        {
            "name": name,
            "contract_id": "entry.synthetic.v1",
            "status": "ADMITTED",
            "receipts": [{"path": str(receipt_file), "sha256": fa.sha256_file(receipt_file)}],
            "availability_clock_ms": 320,
            "valid_until": "2026-12-31T00:00:00+00:00",
        }
        for name in ("syn_alpha", "syn_beta")
    ]
    ledger = {"schema_version": "v5.feature-admission-ledger.v1", "features": features}
    ledger["ledger_sha256"] = fa.ledger_sha256(ledger)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path, ledger["ledger_sha256"]


def test_a_windowed_v5_ledger_admits_features_today(tmp_path) -> None:
    """Validity windows are a ledger-content gap, not a code gap."""

    path, _ = _synthetic_ledger(tmp_path)
    frame = pd.DataFrame({"syn_alpha": [1.0, 2.0], "syn_beta": [0.1, 0.2]})
    matrix = fa.admitted_feature_matrix(
        frame, ["syn_alpha", "syn_beta"], ledger_path=path, as_of=AS_OF
    )
    assert matrix.shape == (2, 2)


def test_latency_receipt_and_arrival_injection_compose() -> None:
    receipt = tt.make_latency_receipt(
        source_family="DATABENTO_OPRA_CBBO_1M",
        p50_ms=227.0, p99_ms=319.5, max_ms=600.0,
        session_count=2, measured_on="2026-08-07", valid_until="2026-11-07",
        evidence_path="synthetic://composition-test",
    )
    frame = pd.DataFrame(
        {"event_time": pd.date_range("2026-01-05 14:30", periods=5, freq="1min", tz="UTC")}
    )
    with pytest.raises(tt.TrainingTwinError, match="no arrival column"):
        tt.assert_no_zero_lag(frame)
    injected = tt.simulate_historical_arrival(
        frame, receipt=receipt, source_family="DATABENTO_OPRA_CBBO_1M", as_of="2026-08-10"
    )
    tt.assert_no_zero_lag(injected)


def _entry_stream(calendar) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for index, session in enumerate(calendar):
        abstain = index >= TRADED
        rows.append(
            {
                "session": session,
                "decision_time": f"{session}T14:34:30+00:00",
                "side": "WAIT" if abstain else ("CALL" if index % 2 else "PUT"),
                "abstain": bool(abstain),
                "score": float(rng.uniform(0.0, 1.0)),
                "fold": f"fold_{index // (SESSIONS // 5)}",
            }
        )
    return pd.DataFrame(rows)


def _trades_from_stream(stream: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = []
    for index, entry in enumerate(stream.loc[~stream["abstain"]].itertuples(index=False)):
        gross = float(rng.normal(80.0, 200.0))
        rows.append(
            {
                "trade_id": f"T{index:03d}",
                "session": entry.session,
                "fold": entry.fold,
                "evaluation_role": "out_of_fold",
                "feature_time": f"{entry.session}T14:34:00+00:00",
                "decision_time": entry.decision_time,
                "entry_time": f"{entry.session}T14:35:00+00:00",
                "exit_time": f"{entry.session}T15:00:00+00:00",
                "side": entry.side,
                "quantity": 1,
                "entry_price": 100.0,
                "exit_price": 100.0 + gross / 100.0,
                "gross_pnl": gross,
            }
        )
    return pd.DataFrame(rows)


def test_the_full_pipeline_composes_and_refuses_its_silent_failures(
    tmp_path, calendar, folds
) -> None:
    ledger_path, ledger_sha = _synthetic_ledger(tmp_path)

    # Rung 5 output: a frozen, content-addressed entry stream.
    stream = _entry_stream(calendar)
    lock = es.freeze_entry_stream(
        stream,
        model_spec_sha256="d" * 64,
        feature_ledger_sha256=ledger_sha,
        frozen_on="2026-08-05",
    )
    assert lock.non_abstained_count == TRADED

    # Rung 6 output: a trade ledger that must be exactly the frozen entries.
    trades = _trades_from_stream(stream)
    es.assert_trades_match_entry_stream(trades, stream, lock=lock)

    # Entry re-optimization through the back door is refused: dropping one
    # "bad" entry after the freeze is exactly what the lock exists to catch.
    with pytest.raises(es.EntryStreamError, match="added, dropped"):
        es.assert_trades_match_entry_stream(trades.iloc[1:], stream, lock=lock)
    altered = stream.copy()
    altered.loc[altered.index[0], "score"] = 0.999
    with pytest.raises(es.EntryStreamError, match="altered after freezing"):
        es.assert_trades_match_entry_stream(trades, altered, lock=lock)

    # The packet names the frozen stream and exports the owner's artifacts.
    spx = pd.DataFrame(
        [
            {"event_time": f"{session}T{clock}:00+00:00", "close": 6000.0 + index}
            for index, session in enumerate(calendar)
            for clock in ("14:00", "16:00")
        ]
    )
    bounds = {}
    for fold in sorted(set(folds.values())):
        members = sorted(s for s, f in folds.items() if f == fold)
        bounds[fold] = {
            "start": f"{members[0]}T00:00:00+00:00",
            "end": (pd.Timestamp(members[-1]) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%dT00:00:00+00:00"
            ),
        }
    packet = cp.export_candidate_packet(
        trades=trades,
        spx=spx,
        candidate_manifest={
            "candidate_id": "COMPOSITION-001",
            "candidate_sha256": "e" * 64,
            "feature_contract_sha256": ledger_sha,
            "entry_stream_sha256": lock.entry_stream_sha256,
            "fill_law": "synthetic",
            "label_law": "synthetic",
            "fold_policy": "chronological_5",
            "fold_boundaries": bounds,
        },
        output_dir=tmp_path / "packet",
    )
    ledger_csv = pd.read_csv(packet.trades_csv)

    # The aggregator zero-fills the ten declared no-trade sessions.
    sessions = si.session_frame_from_trades(
        ledger_csv, session_calendar=calendar, fold_by_session=folds
    )
    assert len(sessions) == SESSIONS
    assert int((sessions["trade_count"] == 0).sum()) == SESSIONS - TRADED
    assert (sessions.loc[sessions["trade_count"] == 0, "net_pnl"] == 0.0).all()
    with pytest.raises(si.SessionIndexError, match="outside the declared calendar"):
        si.session_frame_from_trades(
            ledger_csv, session_calendar=calendar[:TRADED - 1], fold_by_session=folds
        )

    # The gate consumes the aggregated frame under receipts that describe it.
    rng = np.random.default_rng(31)
    frame = sessions.rename(columns={"net_pnl": "policy_net"})
    frame["comparator_net"] = rng.normal(5.0, 30.0, SESSIONS)
    for name in rg.CONTROL_NAMES:
        frame[f"control_{name}"] = rng.normal(-25.0, 20.0, SESSIONS)
    power = rg.make_power_receipt(
        declared_effect=70.0, session_sd=150.0, index_sessions=SESSIONS,
        frozen_on="2026-08-01",
    )
    lock_receipt = rg.make_comparator_lock(
        comparator_name="constant_side",
        selected_on="2026-08-01",
        selection_basis="frozen before evaluation",
    )
    result = rg.evaluate(
        frame,
        cells=[rg.Cell("fee1.54_lat0", 1.54, 0, "policy_net", "comparator_net")],
        controls={name: f"control_{name}" for name in rg.CONTROL_NAMES},
        power_receipt=power,
        comparator_lock=lock_receipt,
        evaluation_date="2026-08-05",
        skill_folds_positive=4,
        skill_null_passed_known_answer_gate=True,
    )
    assert result.verdict in {"SUPPORTED", "NOT_SUPPORTED"}
    assert set(result.controls) == set(rg.CONTROL_NAMES)

    # The dropout frame that sailed through before the repair is now refused.
    dropped = frame.head(TRADED)
    with pytest.raises(rg.ReplayGateError, match="never be dropped"):
        rg.evaluate(
            dropped,
            cells=[rg.Cell("fee1.54_lat0", 1.54, 0, "policy_net", "comparator_net")],
            controls={name: f"control_{name}" for name in rg.CONTROL_NAMES},
            power_receipt=power,
            comparator_lock=lock_receipt,
            evaluation_date="2026-08-05",
            skill_folds_positive=4,
            skill_null_passed_known_answer_gate=True,
        )
