from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from v5.research.validation.candidate_packet import (
    CandidatePacketError,
    REQUIRED_TRADE_COLUMNS,
    export_candidate_packet,
)


def _manifest() -> dict:
    return {
        "candidate_id": "candidate-001",
        "candidate_sha256": "a" * 64,
        "feature_contract_sha256": "b" * 64,
        "entry_stream_sha256": "c" * 64,
        "fill_law": "locked-option-fill-v1",
        "label_law": "arrival-plus-horizon-v1",
        "fold_policy": "chronological-oof-v1",
        "fold_boundaries": {
            "fold-1": {
                "start": "2026-08-03T14:00:00Z",
                "end": "2026-08-03T14:42:00Z",
            },
            "fold-2": {
                "start": "2026-08-03T14:42:00Z",
                "end": "2026-08-03T15:00:00Z",
            },
        },
    }


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "session": "2026-08-03",
                "fold": "fold-1",
                "evaluation_role": "out_of_fold",
                "feature_time": "2026-08-03T14:29:00Z",
                "decision_time": "2026-08-03T14:30:00Z",
                "entry_time": "2026-08-03T14:30:02Z",
                "exit_time": "2026-08-03T14:40:00Z",
                "side": "CALL",
                "quantity": 1,
                "entry_price": 5.0,
                "exit_price": 6.0,
                "gross_pnl": 100.0,
            },
            {
                "trade_id": "t2",
                "session": "2026-08-03",
                "fold": "fold-2",
                "evaluation_role": "confirmation",
                "feature_time": "2026-08-03T14:44:00Z",
                "decision_time": "2026-08-03T14:45:00Z",
                "entry_time": "2026-08-03T14:45:02Z",
                "exit_time": "2026-08-03T14:55:00Z",
                "side": "PUT",
                "quantity": 1,
                "entry_price": 4.0,
                "exit_price": 3.5,
                "gross_pnl": -50.0,
            },
        ]
    )


def _spx() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": pd.date_range(
                "2026-08-03T14:25:00Z", periods=36, freq="min"
            ),
            "close": [6300.0 + index for index in range(36)],
        }
    )


def test_packet_exports_authoritative_csv_receipt_and_two_visuals(tmp_path) -> None:
    trades = _trades()
    original = trades.copy(deep=True)
    result = export_candidate_packet(
        trades=trades,
        spx=_spx(),
        candidate_manifest=_manifest(),
        output_dir=tmp_path,
        slippage_per_side_points=0.10,
    )

    pd.testing.assert_frame_equal(trades, original)
    assert result.trade_count == 2
    output = pd.read_csv(result.trades_csv)
    assert output["commission_usd"].tolist() == pytest.approx([3.08, 3.08])
    assert output["slippage_usd"].tolist() == pytest.approx([20.0, 20.0])
    assert output["net_pnl"].tolist() == pytest.approx([76.92, -73.08])
    assert output["equity"].tolist() == pytest.approx([10076.92, 10003.84])

    trades_html = result.trades_html.read_text()
    assert trades_html.count('class="entry-marker"') == 2
    assert trades_html.count('class="exit-marker"') == 2
    assert 'data-trade-id="t1"' in trades_html
    assert 'aria-label="Net equity curve"' in result.equity_html.read_text()

    summary = json.loads(result.summary_json.read_text())
    assert summary["authoritative_outputs"] == ["trades.csv", "manifest.json"]
    assert summary["diagnostic_outputs"] == ["trades.html", "equity.html"]
    assert summary["net_pnl"] == pytest.approx(3.84)

    manifest = json.loads(result.manifest_json.read_text())
    unsigned = dict(manifest)
    receipt_hash = unsigned.pop("receipt_sha256")
    expected = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert receipt_hash == expected
    for name, expected_hash in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected_hash


def test_the_packet_says_which_cost_components_it_charged(tmp_path) -> None:
    """A fee-only export must not be readable as full round-trip friction.

    The default charges the measured $3.08 fee and no spread, while the measured
    aggressive option round trip is $26.48. Nothing refuses the default -- a
    fee-only diagnostic is legitimate -- so the packet has to say so in writing.
    See v5/research/findings/FRICTION_DECOMPOSITION_2026_08_12.md.
    """

    fees_only = json.loads(
        export_candidate_packet(
            trades=_trades(),
            spx=_spx(),
            candidate_manifest=_manifest(),
            output_dir=tmp_path / "fees_only",
        ).summary_json.read_text()
    )["cost_model"]

    assert fees_only["excludes_spread_crossing"] is True
    assert fees_only["components_omitted"] == ["spread_crossing"]
    assert fees_only["charged_round_trip_usd"] == pytest.approx(3.08)
    assert fees_only["reference_measured_aggressive_round_trip_usd"] == pytest.approx(26.48)

    with_spread = json.loads(
        export_candidate_packet(
            trades=_trades(),
            spx=_spx(),
            candidate_manifest=_manifest(),
            output_dir=tmp_path / "with_spread",
            slippage_per_side_points=0.10,
        ).summary_json.read_text()
    )["cost_model"]

    assert with_spread["excludes_spread_crossing"] is False
    assert with_spread["components_omitted"] == []
    assert with_spread["components_charged"] == ["commission", "spread_or_slippage"]
    # 2 x $1.54 fee + 2 x 0.10 points x 100 multiplier
    assert with_spread["charged_round_trip_usd"] == pytest.approx(23.08)


def test_packet_is_deterministic_across_output_directories(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_candidate_packet(
        trades=_trades(), spx=_spx(), candidate_manifest=_manifest(), output_dir=first
    )
    export_candidate_packet(
        trades=_trades(), spx=_spx(), candidate_manifest=_manifest(), output_dir=second
    )
    for name in ("trades.csv", "trades.html", "equity.html", "summary.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_empty_trade_ledger_produces_an_explicit_empty_packet(tmp_path) -> None:
    result = export_candidate_packet(
        trades=pd.DataFrame(columns=REQUIRED_TRADE_COLUMNS),
        spx=_spx(),
        candidate_manifest=_manifest(),
        output_dir=tmp_path,
    )
    assert result.trade_count == 0
    summary = json.loads(result.summary_json.read_text())
    assert summary["ending_equity"] == 10_000.0
    assert summary["first_entry"] is None
    assert 'id="trade-count">Trades: 0' in result.trades_html.read_text()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.__setitem__("evaluation_role", "train"), "training_rows"),
        (
            lambda frame: frame.__setitem__(
                "feature_time", ["2026-08-03T14:31:00Z", frame.iloc[1]["feature_time"]]
            ),
            "not_causal",
        ),
        (
            lambda frame: frame.__setitem__(
                "exit_time", ["2026-08-03T14:50:00Z", frame.iloc[1]["exit_time"]]
            ),
            "overlapping",
        ),
        (
            lambda frame: frame.__setitem__(
                "exit_time", ["2026-08-04T14:40:00Z", frame.iloc[1]["exit_time"]]
            ),
            "crosses_or_mismatches_session",
        ),
    ],
)
def test_packet_rejects_leakage_overlap_and_cross_session_trades(
    tmp_path, mutation, message
) -> None:
    trades = _trades()
    mutation(trades)
    with pytest.raises(CandidatePacketError, match=message):
        export_candidate_packet(
            trades=trades,
            spx=_spx(),
            candidate_manifest=_manifest(),
            output_dir=tmp_path,
        )


def test_packet_refuses_to_overwrite_existing_outputs(tmp_path) -> None:
    export_candidate_packet(
        trades=_trades(), spx=_spx(), candidate_manifest=_manifest(), output_dir=tmp_path
    )
    with pytest.raises(CandidatePacketError, match="refusing_to_overwrite"):
        export_candidate_packet(
            trades=_trades(),
            spx=_spx(),
            candidate_manifest=_manifest(),
            output_dir=tmp_path,
        )


def test_packet_enforces_accounting_identity_and_declared_fold_window(tmp_path) -> None:
    trades = _trades()
    trades.loc[0, "gross_pnl"] = 99.0
    with pytest.raises(CandidatePacketError, match="accounting_identity"):
        export_candidate_packet(
            trades=trades,
            spx=_spx(),
            candidate_manifest=_manifest(),
            output_dir=tmp_path,
        )

    trades = _trades()
    manifest = _manifest()
    manifest["fold_boundaries"]["fold-1"]["end"] = "2026-08-03T14:20:00Z"
    with pytest.raises(CandidatePacketError, match="outside_declared_fold"):
        export_candidate_packet(
            trades=trades,
            spx=_spx(),
            candidate_manifest=manifest,
            output_dir=tmp_path,
        )
