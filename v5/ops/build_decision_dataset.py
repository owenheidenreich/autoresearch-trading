"""Every trade the bot could have taken, with what it knew when it decided.

Prior tables in this project answered "what does the average slot pay?" They
picked one contract per slot — the nearest to at-the-money — and scored every
slot. That is the right shape for measuring an instrument and the wrong shape
for **learning a policy**, because a policy's whole job is to be choosy: which
minute, which contract, and whether to trade at all.

So this table is one row per *candidate trade*: a decision minute, a contract
the bot could actually have bought at that minute, everything knowable at that
instant, and what the trade would have paid. Roughly forty candidates per
decision point across the chain, and the model picks among them or picks none.

**Three things here are deliberately different from every earlier table.**

*The chain is open.* Contracts are offered out to `MONEYNESS_BAND_POINTS` on
both sides rather than the charter's near-ATM band, because that band was a
declared choice and the measured break-even is not flat across it — 42.6% on
the cheapest quartile against 45.5% on the dearest.

*Cost scales with the contract.* A flat round trip across an open chain teaches
a model to buy the cheapest thing on the board, which is how the percentage-
excursion entry failed in the opposite direction. The cost here is interpolated
through the six **measured** points of the moneyness study, so a $35 contract
is charged $9 and a $19,865 one is charged $369.

*Volume is in the features.* The corpus has carried per-contract per-minute
volume since it was downloaded and no screen in this project has ever read it.
Chain-wide call/put volume balance is the closest thing the owned data has to
order flow.

**Causality.** Every feature reads the decision minute or earlier; the outcome
reads later minutes and nothing else does. `test_decision_dataset.py` enforces
this by mutating the future and asserting no feature moves — the control that
would have caught the one-minute leak of 2026-08-13, which a shuffled-label
null structurally cannot see.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import FIRST_INDEX, LAST_INDEX, _index, _label, session_pivot
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER, TRADE_CORPUS
from v5.research import greeks as gk

# How often the bot is allowed to think. Five minutes keeps time-of-day
# resolution fine enough to learn "early afternoon" without making the table
# four times larger than the greek solve can carry.
DECISION_EVERY = 5
FIRST_DECISION = _index("09:35")
LAST_DECISION = _index("15:30")

# How far across the chain the bot may look. The charter's band was 25 points.
MONEYNESS_BAND_POINTS = 60.0

# Horizons scored for every candidate. The first is the label's horizon; the
# others are recorded so a later exit study needs no rebuild.
HORIZONS = (15, 30, 60)
LABEL_HORIZON = 30

# Lookbacks for the causal path features, in minutes.
LOOKBACKS = (5, 15, 30, 60)

# The measured round trip against the premium it was measured at, from the
# moneyness study. Interpolated, never extrapolated flat: charging one number
# across an open chain would let a model dodge the cost by buying cheap paper.
COST_PREMIUM_USD = (35.0, 333.0, 1945.0, 6778.0, 19865.0, 98809.0)
COST_ROUND_TRIP_USD = (9.0, 12.0, 25.0, 84.0, 369.0, 822.0)


def round_trip_cost(premium_usd: np.ndarray) -> np.ndarray:
    """Measured round trip for a contract at this premium."""

    return np.interp(
        np.asarray(premium_usd, float), COST_PREMIUM_USD, COST_ROUND_TRIP_USD
    )


def volume_pivot(path: Path) -> pd.DataFrame | None:
    """Contracts by minute, carrying traded volume rather than price."""

    try:
        frame = pd.read_parquet(
            path, columns=["ts_event", "volume", "strike", "right"]
        )
    except Exception:
        return None
    if frame.empty:
        return None
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce").astype(float)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "volume"])
    if frame.empty:
        return None
    frame["minute"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    got = frame.pivot_table(
        index="minute", columns=["strike", "right"], values="volume", aggfunc="sum"
    ).sort_index()
    return got if not got.empty else None


def parity_prices(
    prices: pd.DataFrame, spot: pd.Series, strikes: np.ndarray, is_call: np.ndarray
) -> pd.DataFrame:
    """Each contract's value implied by the other side of its own strike.

    ``C = P + S - K`` and ``P = C - S + K``. Returns NaN for a contract whose
    twin never printed. Reads only the minute it is given, so it is as causal as
    the price it is derived from.
    """

    columns = prices.columns
    twin = {}
    for position, (strike, right) in enumerate(zip(strikes, columns.get_level_values("right"))):
        other = "P" if right == "C" else "C"
        key = (strike, other)
        twin[position] = columns.get_loc(key) if key in columns else -1

    values = prices.to_numpy(float)
    out = np.full(values.shape, np.nan)
    shift = spot.to_numpy(float)[:, None] - strikes[None, :]
    for position, source in twin.items():
        if source < 0:
            continue
        sign = 1.0 if is_call[position] else -1.0
        out[:, position] = values[:, source] + sign * shift[:, position]
    return pd.DataFrame(out, index=prices.index, columns=columns)


MIN_HISTORY_MINUTES = 5


def path_features(
    spot: pd.Series, minute: str, *, adaptive: bool = False
) -> dict | None:
    """What the underlying had done by ``minute``, and nothing after it.

    **``adaptive`` decides whether the first hour of the session exists at all.**

    The fixed form demands a full sixty minutes of prior history for the longest
    lookback. The session opens at 09:30, so the earliest minute it can score is
    **10:30** on a perfectly dense series, and **10:35** in the real corpus once
    sparse minutes drop out — which silently removed the entire 09:30-10:30
    window from every dataset built before 2026-08-14, including the
    conditional-drift census, the selective policy and the factorial. The blind
    spot was invisible because the tables simply started later and nothing
    asserted otherwise.

    Adaptive uses whatever history exists, capped at the requested window, and
    reports ``history_minutes`` so a model can tell a truncated sixty-minute
    range from a complete one. That is not a weaker feature, it is the honest
    one: at 09:40 a live bot has ten minutes of session history and no more, and
    a policy that refuses to trade until 10:31 is a policy nobody chose.

    The default stays fixed so every table built before this change reproduces.
    """

    history = spot.loc[:minute].dropna()
    need = MIN_HISTORY_MINUTES if adaptive else max(LOOKBACKS) + 1
    if len(history) < need:
        return None
    now = float(history.iloc[-1])
    out: dict[str, float] = {}
    for back in LOOKBACKS:
        window = history.loc[_label(_index(minute) - back) :]
        if adaptive:
            # Two points is the least that can express a move or a range.
            if len(window) < 2:
                window = history.iloc[-2:]
        elif len(window) < back // 3:
            return None
        out[f"move_{back}m_rel"] = (now - float(window.iloc[0])) / now
        out[f"range_{back}m_rel"] = (float(window.max()) - float(window.min())) / now
    steps = history.to_numpy(float)
    returns = np.diff(steps[-31:]) / steps[-31:-1] if len(steps) > 31 else np.array([0.0])
    out["realised_vol_30m"] = float(np.std(returns)) if returns.size else 0.0
    low, high = float(history.min()), float(history.max())
    out["range_position"] = (now - low) / (high - low) if high > low else 0.5
    out["session_range_rel"] = (high - low) / now
    # How much session history actually informed the windows above. Without it
    # a truncated sixty-minute range is indistinguishable from a complete one.
    out["history_minutes"] = float(len(history))
    out["minutes_since_open"] = float(_index(minute) - FIRST_INDEX)
    out["minutes_to_close"] = float(LAST_INDEX - _index(minute))
    return out


def session_rows(path: Path) -> pd.DataFrame | None:
    """Every candidate trade in one session."""

    got = session_pivot(path)
    if got is None:
        return None
    pivot, spot = got
    volume = volume_pivot(path)
    if volume is None:
        return None
    volume = volume.reindex(index=pivot.index, columns=pivot.columns).fillna(0.0)
    # The outcome uses the last known price for a contract that stops printing,
    # which Phase 0 settled: dropping those contracts removes a population that
    # is 91.9% winners and is a look-ahead filter.
    filled = pivot.ffill()
    spot_f = spot.ffill()
    strikes = pivot.columns.get_level_values("strike").to_numpy(float)
    is_call = pivot.columns.get_level_values("right").to_numpy() == "C"

    rolling_5 = volume.rolling(5, min_periods=1).sum()
    rolling_15 = volume.rolling(15, min_periods=1).sum()
    chain_15 = rolling_15.sum(axis=1)
    call_15 = rolling_15.loc[:, is_call].sum(axis=1)

    # A second, independent estimate of every contract's value, from the other
    # side of its own strike: put/call parity says C - P = S - K.
    #
    # This exists to catch a specific accidental mechanism. These are last-trade
    # prints, not quotes, and a contract deep in the money trades rarely across
    # a wide spread. A model free to choose its moment can learn to buy the
    # prints that landed on the bid and sell the ones that landed on the ask,
    # which books the spread as profit and is not available to anyone who has to
    # pay it. Averaging each print with its parity twin halves that bounce, and
    # the residual between the two says how far a print sat from fair value.
    parity = parity_prices(filled, spot_f, strikes, is_call)
    fair = pd.DataFrame(
        np.where(np.isfinite(parity.to_numpy()),
                 0.5 * (filled.to_numpy() + parity.to_numpy()),
                 filled.to_numpy()),
        index=filled.index, columns=filled.columns,
    )

    rows = []
    for index in range(FIRST_DECISION, LAST_DECISION + 1, DECISION_EVERY):
        minute = _label(index)
        if minute not in pivot.index:
            continue
        here = float(spot_f.get(minute, np.nan))
        if not np.isfinite(here):
            continue
        feats = path_features(spot_f, minute)
        if feats is None:
            continue

        price = pivot.loc[minute].to_numpy(float)
        moneyness = np.where(is_call, here - strikes, strikes - here)
        # A contract is a candidate only if it actually printed this minute.
        # That is a tradeability fact known at the decision minute, not an
        # outcome, so it is a legitimate reason to exclude.
        eligible = np.flatnonzero(
            np.isfinite(price)
            & (price > 0.0)
            & (np.abs(moneyness) <= MONEYNESS_BAND_POINTS)
        )
        if not eligible.size:
            continue

        chain_here = float(chain_15.get(minute, 0.0))
        call_here = float(call_15.get(minute, 0.0))
        vol_5 = rolling_5.loc[minute].to_numpy(float)[eligible]
        vol_15 = rolling_15.loc[minute].to_numpy(float)[eligible]
        entry = price[eligible]
        premium = entry * CONTRACT_MULTIPLIER
        minutes_left = float(LAST_INDEX - index)

        block = {
            "session": path.name[:10],
            "entry_minute": minute,
            "spot": here,
            "strike": strikes[eligible],
            "is_call": is_call[eligible],
            "moneyness": moneyness[eligible],
            "moneyness_rel": moneyness[eligible] / here,
            "entry_premium": premium,
            "entry_premium_rel": premium / (here * CONTRACT_MULTIPLIER),
            "minutes_to_expiry": minutes_left,
            "contract_volume_5m": vol_5,
            "contract_volume_15m": vol_15,
            "contract_share_of_chain": np.where(chain_here > 0, vol_15 / chain_here, 0.0),
            "chain_volume_15m": chain_here,
            "chain_call_share_15m": (call_here / chain_here) if chain_here > 0 else 0.5,
            "round_trip_usd": round_trip_cost(premium),
            **{k: v for k, v in feats.items()},
        }
        entry_fair = fair.loc[minute].to_numpy(float)[eligible]
        block["entry_fair_premium"] = entry_fair * CONTRACT_MULTIPLIER
        # How far this print sat from its parity twin. Negative means the print
        # was below fair value, which is what buying the bid looks like.
        block["parity_residual"] = (entry - entry_fair) * CONTRACT_MULTIPLIER
        for horizon in HORIZONS:
            exit_label = _label(min(index + horizon, LAST_INDEX))
            if exit_label in filled.index:
                out_price = filled.loc[exit_label].to_numpy(float)[eligible]
                out_fair = fair.loc[exit_label].to_numpy(float)[eligible]
            else:
                out_price = np.full(eligible.size, np.nan)
                out_fair = np.full(eligible.size, np.nan)
            block[f"gross_{horizon}m"] = (out_price - entry) * CONTRACT_MULTIPLIER
            # The same trade scored on both sides at parity-averaged value, so
            # bid/ask bounce cancels instead of accruing to whoever timed it.
            block[f"gross_fair_{horizon}m"] = (
                out_fair - entry_fair
            ) * CONTRACT_MULTIPLIER
        # Best and worst the trade ever showed, for the later exit study only.
        window = filled.loc[minute : _label(min(index + LABEL_HORIZON, LAST_INDEX))]
        taken = window.to_numpy(float)[:, eligible]
        block["path_max"] = (np.nanmax(taken, axis=0) - entry) * CONTRACT_MULTIPLIER
        block["path_min"] = (np.nanmin(taken, axis=0) - entry) * CONTRACT_MULTIPLIER
        rows.append(pd.DataFrame(block))

    if not rows:
        return None
    table = pd.concat(rows, ignore_index=True)
    got = gk.greeks_batch(
        table["entry_premium"].to_numpy(float) / CONTRACT_MULTIPLIER,
        table["spot"].to_numpy(float),
        table["strike"].to_numpy(float),
        table["minutes_to_expiry"].to_numpy(float),
        table["is_call"].to_numpy(bool),
    )
    table["iv"] = got["iv"]
    table["delta"] = got["delta"]
    table["gamma_dollars"] = got["gamma"] * table["spot"] ** 2 / 100.0
    table["theta_share"] = got["theta_per_minute"] / (
        table["entry_premium"] / CONTRACT_MULTIPLIER
    )
    table["vega_rel"] = got["vega"] / table["entry_premium"]
    return table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=TRADE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.corpus.glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]

    parts = []
    for i, path in enumerate(files, 1):
        got = session_rows(path)
        if got is not None:
            parts.append(got)
        if i % 100 == 0:
            rows = sum(len(x) for x in parts)
            print(f"  {i}/{len(files)} sessions, {rows:,} candidates", flush=True)

    table = pd.concat(parts, ignore_index=True)
    table["net_label"] = (
        table[f"gross_{LABEL_HORIZON}m"] - table["round_trip_usd"]
    )
    table["profitable"] = (table["net_label"] > 0.0).astype(int)
    # The same label with bid/ask bounce cancelled. If a policy is profitable on
    # `net_label` and not on this, it was timing prints, not the market.
    table["net_label_fair"] = (
        table[f"gross_fair_{LABEL_HORIZON}m"] - table["round_trip_usd"]
    )
    table["profitable_fair"] = (table["net_label_fair"] > 0.0).astype(int)
    table = table.sort_values(["session", "entry_minute"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)
    print(
        f"\n{len(table):,} candidate trades, {table['session'].nunique()} sessions, "
        f"{len(table) / table['session'].nunique():.0f} per session"
    )
    print(
        f"  profitable net of measured cost: {100 * table['profitable'].mean():.1f}%"
    )
    print(
        f"  mean premium ${table['entry_premium'].mean():,.0f}, "
        f"mean round trip ${table['round_trip_usd'].mean():.2f}"
    )
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
