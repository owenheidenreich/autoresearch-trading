"""Chain-internal state must be causal, informational, and honestly missing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
    ChainFeatureError,
    chain_state,
    contract_chain_features,
)

SESSION = "2024-03-15"


def _ladder(
    minutes: list[str],
    *,
    session: str = SESSION,
    call_iv: float = 0.20,
    put_iv: float = 0.20,
    call_bid_size: float = 10.0,
    strikes: tuple[float, ...] = (4980.0, 4990.0, 5000.0, 5010.0, 5020.0),
    spot: float = 5000.0,
) -> pd.DataFrame:
    rows = []
    for minute in minutes:
        for strike in strikes:
            for is_call in (True, False):
                intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
                # A mild quadratic smile so the fit is well posed.
                skew = 0.00002 * (strike - spot) ** 2
                rows.append(
                    {
                        "session": session,
                        "minute": minute,
                        "strike": strike,
                        "is_call": is_call,
                        "mid": intrinsic + 10.0,
                        "self_iv": (call_iv if is_call else put_iv) + skew,
                        "moneyness_itm_points": (spot - strike) if is_call else (strike - spot),
                        "bid_size": call_bid_size if is_call else 10.0,
                        "ask_size": 10.0,
                        "spread": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_chain_state_emits_every_declared_field() -> None:
    state = chain_state(_ladder(["09:31", "09:32"]))
    assert list(state.columns) == ["session", "minute", *CHAIN_STATE_FEATURES]
    assert len(state) == 2


def test_risk_reversal_carries_the_sign_of_demand_skew() -> None:
    """Calls bid up relative to puts must read positive, and vice versa."""

    calls_rich = chain_state(_ladder(["09:31"], call_iv=0.25, put_iv=0.20))
    puts_rich = chain_state(_ladder(["09:31"], call_iv=0.20, put_iv=0.25))
    assert calls_rich["risk_reversal"].iloc[0] > 0.0
    assert puts_rich["risk_reversal"].iloc[0] < 0.0


def test_depth_imbalance_reads_the_side_that_is_leaning() -> None:
    heavy_bid = chain_state(_ladder(["09:31"], call_bid_size=40.0))
    assert heavy_bid["chain_depth_imbalance"].iloc[0] > 0.0
    balanced = chain_state(_ladder(["09:31"], call_bid_size=10.0))
    assert balanced["chain_depth_imbalance"].iloc[0] == pytest.approx(0.0)


def test_change_terms_are_nan_until_their_lag_is_available() -> None:
    """A lag must not silently reach further back than it claims."""

    minutes = [f"09:{m:02d}" for m in range(31, 60)]
    state = chain_state(_ladder(minutes)).set_index("minute")
    assert np.isnan(state.loc["09:31", "risk_reversal_change_15m"])
    assert np.isnan(state.loc["09:35", "atm_iv_change_5m"])
    assert not np.isnan(state.loc["09:36", "atm_iv_change_5m"])
    assert not np.isnan(state.loc["09:46", "risk_reversal_change_15m"])


def test_a_lag_across_a_missing_minute_is_nan_not_a_further_reach() -> None:
    minutes = [f"09:{m:02d}" for m in range(31, 50) if m != 40]
    state = chain_state(_ladder(minutes)).set_index("minute")
    # 09:45 would need 09:40, which the session does not have.
    assert np.isnan(state.loc["09:45", "atm_iv_change_5m"])


def test_chain_state_is_causal_under_a_mutated_future() -> None:
    """The project's standing control: perturbing later minutes changes nothing."""

    minutes = [f"09:{m:02d}" for m in range(31, 50)]
    base = _ladder(minutes)
    mutated = base.copy()
    later = mutated["minute"] > "09:40"
    mutated.loc[later, "self_iv"] *= 3.0
    mutated.loc[later, "bid_size"] *= 7.0

    before = chain_state(base)
    after = chain_state(mutated)
    early = before["minute"] <= "09:40"
    pd.testing.assert_frame_equal(
        before[early].reset_index(drop=True),
        after[early.to_numpy()].reset_index(drop=True),
    )


def test_smile_residual_is_zero_on_a_surface_that_fits_and_nonzero_on_a_kink() -> None:
    ladder = _ladder(["09:31"])
    clean = contract_chain_features(ladder)
    assert np.nanmax(np.abs(clean["smile_residual"])) < 1e-6

    kinked = ladder.copy()
    target = kinked.index[(kinked["strike"] == 5000.0) & kinked["is_call"]][0]
    kinked.loc[target, "self_iv"] += 0.05
    dirty = contract_chain_features(kinked)
    assert abs(dirty.loc[target, "smile_residual"]) > 0.01


def test_contract_features_align_to_the_ladder_index() -> None:
    ladder = _ladder(["09:31", "09:32"])
    features = contract_chain_features(ladder)
    assert list(features.columns) == list(CONTRACT_CHAIN_FEATURES)
    assert features.index.equals(ladder.index)


def test_a_ladder_missing_columns_is_refused() -> None:
    with pytest.raises(ChainFeatureError, match="missing columns"):
        chain_state(pd.DataFrame({"session": [SESSION], "minute": ["09:31"]}))


def test_dispersion_is_nan_rather_than_invented_when_strikes_are_unpaired() -> None:
    ladder = _ladder(["09:31"])
    calls_only = ladder[ladder["is_call"]].copy()
    state = chain_state(calls_only)
    assert np.isnan(state["implied_spot_dispersion_ratio"].iloc[0])
