from __future__ import annotations

import pytest

from v5.ops import causal_day_simulator as sim
from v5.research.causal_day_reporting import ReportingError, summarize_results
from v5.tests.test_causal_day_simulator import SESSION, _quotes


def _trade_result(session: str = SESSION) -> sim.SimulationResult:
    ledger = sim.ActionLedger(
        {
            ("09:35", "morning_entry"): sim.Action("BUY", "call"),
            ("09:36", "morning_exit"): sim.Action("SELL"),
        }
    )
    return sim.simulate_session(_quotes(), session, ledger, trade_cap=1)


def test_reports_fixed_account_economics_and_risk_metrics() -> None:
    result = _trade_result()
    report = summarize_results([result])
    assert report["sessions"] == 1
    assert report["trades"] == 1
    assert report["days_traded"] == 1
    assert report["premium_at_risk_max_usd"] == pytest.approx(124.0)
    assert report["spread_paid_mean_usd"] == pytest.approx(20.0)
    assert report["fees_total_usd"] == pytest.approx(3.08)
    assert report["net_usd_total"] == pytest.approx(result.realised_pnl_usd)
    assert report["entry_otm_depth_mean_points"] == pytest.approx(5.0)
    assert report["maximum_itm_depth_mean_points"] == pytest.approx(-5.0)
    assert report["final_itm_depth_mean_points"] == pytest.approx(-5.0)
    assert report["otm_to_itm_conversion_rate"] == 0.0
    assert report["time_to_cross_mean_minutes"] is None
    assert report["underlying_mfe_mean_points"] == 0.0
    assert report["underlying_mae_mean_points"] == 0.0
    interval = report["net_usd_per_session_corrected_interval"]
    assert interval["session_unit"]
    assert interval["family_size"] == 1
    assert interval["low"] == pytest.approx(result.realised_pnl_usd)
    assert interval["high"] == pytest.approx(result.realised_pnl_usd)


def test_refuses_blocked_terminal_population() -> None:
    quotes = _quotes(missing_bid_minutes={"16:00"})
    ledger = sim.ActionLedger({("15:00", "afternoon_entry"): sim.Action("BUY", "call")})
    result = sim.simulate_session(quotes, SESSION, ledger, trade_cap=1)
    with pytest.raises(ReportingError, match="blocked terminal"):
        summarize_results([result])
