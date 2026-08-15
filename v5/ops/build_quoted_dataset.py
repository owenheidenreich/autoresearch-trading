"""The same candidate trades, priced at the touch instead of at the tape.

`build_decision_dataset` scores a trade from last-trade minute bars. That was
enough to build a policy and not enough to score one: a print lands sometimes on
the bid and sometimes on the ask, the put/call-parity residual has a standard
deviation of **$102.60** against a **$19.79** round trip, and a model free to
choose its own minute learns to buy the prints that landed low. On 2026-08-14
that produced a policy beating a composition-matched control by $68/trade, in
all four out-of-sample years, whose profit turned out to equal the price
distortion it had selected to within a dollar fifty.

This module removes the possibility rather than arguing about it.

**Entry is charged at the ask. Exit is paid at the bid.** Both come from the
owned quote corpus, so the spread is *charged* rather than inferred, and no
amount of selection can conjure a spread that is not there. Only the exchange
and regulatory fees are added on top; charging a modelled round trip as well
would bill the spread twice.

**Features are read from the mid.** That is the honest split: a trader observes
the market at the midpoint and transacts at the touch.

**The feature set is identical to the trade-corpus build**, so the only thing
that changes between the two runs is the price source. Volume is joined from the
trade corpus for the same session, and a contract with no print in a minute
traded zero — that is the true value, not a gap to impute.

The cost is 251 sessions instead of 1,045. That is the whole trade: four times
less data, and a price that cannot be gamed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.build_decision_dataset import (
    DECISION_EVERY,
    FIRST_DECISION,
    HORIZONS,
    LABEL_HORIZON,
    LAST_DECISION,
    LOOKBACKS,
    MONEYNESS_BAND_POINTS,
    path_features,
)
from v5.ops.measure_fill_quality import QUOTE_CORPUS
from v5.ops.measure_hold_occupancy import LAST_INDEX, _index, _label
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER, TRADE_CORPUS
from v5.research import greeks as gk

# Measured from a real IBKR paper fill. Fees only: crossing the spread is paid
# in the prices themselves here, so charging a round trip too would double-bill.
FEES_PER_ROUND_TRIP_USD = 3.08

MIN_PAIRED_STRIKES = 5
PARITY_WINDOW_POINTS = 30.0


def quote_frames(path: Path):
    """Bid, ask and mid per minute per contract, plus the parity spot."""

    try:
        d = pd.read_parquet(
            path, columns=["event_time", "strike", "right", "bid", "ask", "mid"]
        )
    except Exception:
        return None
    if d.empty:
        return None
    for column in ("strike", "bid", "ask", "mid"):
        d[column] = pd.to_numeric(d[column], errors="coerce").astype(float)
    d = d.dropna(subset=["strike", "bid", "ask", "mid"])
    # A one-sided or crossed quote is not a market anyone can trade against.
    d = d[(d["bid"] > 0.0) & (d["ask"] > d["bid"])]
    if d.empty:
        return None
    d["minute"] = (
        pd.to_datetime(d["event_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    frames = {
        name: d.pivot_table(
            index="minute", columns=["strike", "right"], values=name, aggfunc="last"
        ).sort_index()
        for name in ("bid", "ask", "mid")
    }
    mid = frames["mid"]
    if mid.empty:
        return None

    calls = mid.loc[:, mid.columns.get_level_values("right") == "C"]
    puts = mid.loc[:, mid.columns.get_level_values("right") == "P"]
    calls.columns = calls.columns.get_level_values("strike")
    puts.columns = puts.columns.get_level_values("strike")
    shared = calls.columns.intersection(puts.columns)
    if len(shared) < MIN_PAIRED_STRIKES:
        return None
    # Same parity estimator as the trade corpus, on mids rather than prints.
    implied = (calls[shared] - puts[shared]).add(pd.Series(shared, index=shared))
    coarse = implied.median(axis=1)
    offsets = np.abs(implied.columns.to_numpy(float) - coarse.to_numpy()[:, None])
    near = implied.where(offsets <= PARITY_WINDOW_POINTS)
    enough = near.notna().sum(axis=1) >= MIN_PAIRED_STRIKES
    if not enough.any():
        return None
    return frames, near.mean(axis=1).where(enough)


def traded_volume(session: str, template: pd.DataFrame, corpus: Path) -> pd.DataFrame:
    """Per-contract traded volume for the same session, aligned to the quotes.

    A contract that printed nothing in a minute traded zero. That is the value,
    not a missing observation, so it is filled rather than dropped.
    """

    path = corpus / f"{session}.spxw_0dte.ohlcv-1m.parquet"
    if not path.exists():
        return pd.DataFrame(0.0, index=template.index, columns=template.columns)
    try:
        frame = pd.read_parquet(path, columns=["ts_event", "volume", "strike", "right"])
    except Exception:
        return pd.DataFrame(0.0, index=template.index, columns=template.columns)
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce").astype(float)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "volume"])
    frame["minute"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    got = frame.pivot_table(
        index="minute", columns=["strike", "right"], values="volume", aggfunc="sum"
    )
    return got.reindex(index=template.index, columns=template.columns).fillna(0.0)


def session_rows(path: Path, corpus: Path, *, adaptive: bool = False) -> pd.DataFrame | None:
    got = quote_frames(path)
    if got is None:
        return None
    frames, spot = got
    bid, ask, mid = frames["bid"], frames["ask"], frames["mid"]
    session = path.name.replace("databento_spxw_0dte_", "")[:10]

    volume = traded_volume(session, mid, corpus)
    bid_f, ask_f, mid_f = bid.ffill(), ask.ffill(), mid.ffill()
    spot_f = spot.ffill()
    strikes = mid.columns.get_level_values("strike").to_numpy(float)
    is_call = mid.columns.get_level_values("right").to_numpy() == "C"

    rolling_5 = volume.rolling(5, min_periods=1).sum()
    rolling_15 = volume.rolling(15, min_periods=1).sum()
    chain_15 = rolling_15.sum(axis=1)
    call_15 = rolling_15.loc[:, is_call].sum(axis=1)

    rows = []
    for index in range(FIRST_DECISION, LAST_DECISION + 1, DECISION_EVERY):
        minute = _label(index)
        if minute not in mid.index:
            continue
        here = float(spot_f.get(minute, np.nan))
        if not np.isfinite(here):
            continue
        feats = path_features(spot_f, minute, adaptive=adaptive)
        if feats is None:
            continue

        entry_ask = ask.loc[minute].to_numpy(float)
        entry_mid = mid.loc[minute].to_numpy(float)
        entry_bid = bid.loc[minute].to_numpy(float)
        moneyness = np.where(is_call, here - strikes, strikes - here)
        # A candidate needs a live two-sided quote at the decision minute. That
        # is a tradeability fact known then, not an outcome.
        eligible = np.flatnonzero(
            np.isfinite(entry_ask)
            & np.isfinite(entry_bid)
            & (entry_bid > 0.0)
            & (entry_ask > entry_bid)
            & (np.abs(moneyness) <= MONEYNESS_BAND_POINTS)
        )
        if not eligible.size:
            continue

        chain_here = float(chain_15.get(minute, 0.0))
        call_here = float(call_15.get(minute, 0.0))
        paid = entry_ask[eligible]
        fair = entry_mid[eligible]
        block = {
            "session": session,
            "entry_minute": minute,
            "spot": here,
            "strike": strikes[eligible],
            "is_call": is_call[eligible],
            "moneyness": moneyness[eligible],
            "moneyness_rel": moneyness[eligible] / here,
            # Features describe the market at mid; the trade is charged at the touch.
            "entry_premium": fair * CONTRACT_MULTIPLIER,
            "entry_premium_rel": fair * CONTRACT_MULTIPLIER / (here * CONTRACT_MULTIPLIER),
            "entry_ask_usd": paid * CONTRACT_MULTIPLIER,
            "entry_bid_usd": entry_bid[eligible] * CONTRACT_MULTIPLIER,
            "spread_usd": (paid - entry_bid[eligible]) * CONTRACT_MULTIPLIER,
            "minutes_to_expiry": float(LAST_INDEX - index),
            "contract_volume_5m": rolling_5.loc[minute].to_numpy(float)[eligible],
            "contract_volume_15m": rolling_15.loc[minute].to_numpy(float)[eligible],
            "contract_share_of_chain": (
                rolling_15.loc[minute].to_numpy(float)[eligible] / chain_here
                if chain_here > 0
                else np.zeros(eligible.size)
            ),
            "chain_volume_15m": chain_here,
            "chain_call_share_15m": (call_here / chain_here) if chain_here > 0 else 0.5,
            "round_trip_usd": FEES_PER_ROUND_TRIP_USD,
            **feats,
        }
        for horizon in HORIZONS:
            exit_label = _label(min(index + horizon, LAST_INDEX))
            if exit_label in bid_f.index:
                out_bid = bid_f.loc[exit_label].to_numpy(float)[eligible]
                out_mid = mid_f.loc[exit_label].to_numpy(float)[eligible]
            else:
                out_bid = np.full(eligible.size, np.nan)
                out_mid = np.full(eligible.size, np.nan)
            # What a taker actually keeps: bought the offer, sold the bid.
            block[f"gross_{horizon}m"] = (out_bid - paid) * CONTRACT_MULTIPLIER
            # The same trade mid-to-mid, which is what a spread-free world pays.
            block[f"gross_mid_{horizon}m"] = (out_mid - fair) * CONTRACT_MULTIPLIER
        rows.append(pd.DataFrame(block))

    if not rows:
        return None
    table = pd.concat(rows, ignore_index=True)
    # Greeks recomputed from the mid, never taken from the vendor: a vendor greek
    # is a train/live divergence the live path cannot reproduce.
    solved = gk.greeks_batch(
        table["entry_premium"].to_numpy(float) / CONTRACT_MULTIPLIER,
        table["spot"].to_numpy(float),
        table["strike"].to_numpy(float),
        table["minutes_to_expiry"].to_numpy(float),
        table["is_call"].to_numpy(bool),
    )
    table["iv"] = solved["iv"]
    table["delta"] = solved["delta"]
    table["gamma_dollars"] = solved["gamma"] * table["spot"] ** 2 / 100.0
    table["theta_share"] = solved["theta_per_minute"] / (
        table["entry_premium"] / CONTRACT_MULTIPLIER
    )
    table["vega_rel"] = solved["vega"] / table["entry_premium"]
    return table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    p.add_argument("--trades", type=Path, default=TRADE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--adaptive", action="store_true",
                   help="use whatever session history exists, so the 09:30-10:30 "
                        "window is scoreable instead of silently absent")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = [
        f for f in sorted(args.quotes.glob("*.parquet"))
        if "official_context" not in f.name
    ]
    if args.limit:
        files = files[: args.limit]

    parts = []
    for i, path in enumerate(files, 1):
        got = session_rows(path, args.trades, adaptive=args.adaptive)
        if got is not None:
            parts.append(got)
        if i % 50 == 0:
            print(f"  {i}/{len(files)} sessions, "
                  f"{sum(len(x) for x in parts):,} candidates", flush=True)

    table = pd.concat(parts, ignore_index=True)
    table["net_label"] = table[f"gross_{LABEL_HORIZON}m"] - table["round_trip_usd"]
    table["profitable"] = (table["net_label"] > 0.0).astype(int)
    # Mid-to-mid, fees only. The gap between this and `net_label` is the spread,
    # and it is what a policy would need to be worth to survive paying it.
    table["net_label_fair"] = (
        table[f"gross_mid_{LABEL_HORIZON}m"] - table["round_trip_usd"]
    )
    table["profitable_fair"] = (table["net_label_fair"] > 0.0).astype(int)
    table["parity_residual"] = 0.0  # priced at the touch; nothing to infer
    table = table.sort_values(["session", "entry_minute"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)
    print(
        f"\n{len(table):,} candidate trades, {table['session'].nunique()} sessions, "
        f"{len(table) / table['session'].nunique():.0f} per session"
    )
    print(f"  profitable paying the spread: {100 * table['profitable'].mean():.1f}%")
    print(f"  profitable mid-to-mid:        {100 * table['profitable_fair'].mean():.1f}%")
    print(
        f"  mean premium ${table['entry_premium'].mean():,.0f}, "
        f"mean spread ${table['spread_usd'].mean():.2f} "
        f"({100 * table['spread_usd'].mean() / table['entry_premium'].mean():.2f}% of premium)"
    )
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
