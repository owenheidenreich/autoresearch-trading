"""One population where both halves of the decision apply.

The entry harness and the exit harness were built on different tables: the entry
one carries many candidates per minute and no forward path, the exit one carries
a forward path and one contract per minute. Neither can host a factorial,
because an entry arm needs contracts to choose between and an exit arm needs a
path to act on, and the attribution needs both on the *same* trades.

So this emits two tables that share a `trade_id`:

* **candidates** — one row per trade the bot could open, with every entry-time
  feature and what opening it costs at the ask;
* **paths** — one row per trade-minute, with the exit-time state and what closing
  it pays at the bid.

Splitting them rather than denormalising keeps the paths table from repeating
thirty entry features down every minute of every trade.

**Priced at the touch throughout.** Entry at the ask, exit at the bid, features
from the mid. Last-trade prints are not used anywhere: their put/call-parity
residual has a standard deviation of $102.60 against a $19.79 round trip, and
both an entry model and an exit model have already been shown to harvest that
before finding anything about the market.

The cadence is coarser than the entry-only study — a decision every fifteen
minutes rather than five — because the factorial needs a forward path for every
candidate rather than for one contract per slot, and that is the term that grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.build_decision_dataset import LOOKBACKS, path_features
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD, quote_frames
from v5.ops.measure_fill_quality import QUOTE_CORPUS
from v5.ops.measure_hold_occupancy import LAST_INDEX, _index, _label
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER, TRADE_CORPUS
from v5.research import greeks as gk

DECISION_EVERY = 15
FIRST_DECISION = _index("09:35")
LAST_DECISION = _index("15:00")
MONEYNESS_BAND_POINTS = 40.0
MAX_HOLD_MINUTES = 30


def _volume_pivot(session: str, template: pd.DataFrame, corpus: Path) -> pd.DataFrame:
    """Traded volume aligned to the quote grid; no print in a minute means zero."""

    path = corpus / f"{session}.spxw_0dte.ohlcv-1m.parquet"
    empty = pd.DataFrame(0.0, index=template.index, columns=template.columns)
    if not path.exists():
        return empty
    try:
        frame = pd.read_parquet(path, columns=["ts_event", "volume", "strike", "right"])
    except Exception:
        return empty
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



def attach_greeks(frame: pd.DataFrame, price_usd_column: str, names: dict) -> pd.DataFrame:
    """Greeks recomputed from the mid, never taken from the vendor.

    A vendor greek is a train/live divergence the live path cannot reproduce, so
    the same function must run offline and live. Attached per session rather than
    once at the end so a session's table is complete on its own and can be tested.
    """

    price = frame[price_usd_column].to_numpy(float)
    if price_usd_column.endswith("_usd"):
        price = price / CONTRACT_MULTIPLIER
    solved = gk.greeks_batch(
        price, frame["spot"].to_numpy(float), frame["strike"].to_numpy(float),
        frame["minutes_to_expiry"].to_numpy(float), frame["is_call"].to_numpy(bool),
    )
    frame[names["iv"]] = solved["iv"]
    frame[names["delta"]] = solved["delta"]
    frame[names["gamma"]] = solved["gamma"] * frame["spot"] ** 2 / 100.0
    frame[names["theta"]] = solved["theta_per_minute"] / price
    frame[names["vega"]] = solved["vega"] / (
        price * CONTRACT_MULTIPLIER if names["vega"] == "vega_rel" else 1.0
    )
    return frame


def session_tables(path: Path, corpus: Path):
    got = quote_frames(path)
    if got is None:
        return None
    frames, spot = got
    bid, ask, mid = frames["bid"], frames["ask"], frames["mid"]
    session = path.name.replace("databento_spxw_0dte_", "")[:10]

    volume = _volume_pivot(session, mid, corpus)
    bid_f, mid_f = bid.ffill(), mid.ffill()
    spot_f = spot.ffill()
    strikes = mid.columns.get_level_values("strike").to_numpy(float)
    is_call = mid.columns.get_level_values("right").to_numpy() == "C"
    minutes = list(mid.index)
    where = {m: i for i, m in enumerate(minutes)}

    rolling_5 = volume.rolling(5, min_periods=1).sum()
    rolling_15 = volume.rolling(15, min_periods=1).sum()
    chain_15 = rolling_15.sum(axis=1)
    call_15 = rolling_15.loc[:, is_call].sum(axis=1)

    candidates, paths = [], []
    for index in range(FIRST_DECISION, LAST_DECISION + 1, DECISION_EVERY):
        minute = _label(index)
        if minute not in where:
            continue
        here = float(spot_f.get(minute, np.nan))
        if not np.isfinite(here):
            continue
        feats = path_features(spot_f, minute)
        if feats is None:
            continue

        entry_ask = ask.loc[minute].to_numpy(float)
        entry_bid = bid.loc[minute].to_numpy(float)
        entry_mid = mid.loc[minute].to_numpy(float)
        moneyness = np.where(is_call, here - strikes, strikes - here)
        eligible = np.flatnonzero(
            np.isfinite(entry_ask) & np.isfinite(entry_bid)
            & (entry_bid > 0.0) & (entry_ask > entry_bid)
            & (np.abs(moneyness) <= MONEYNESS_BAND_POINTS)
        )
        if not eligible.size:
            continue

        chain_here = float(chain_15.get(minute, 0.0))
        call_here = float(call_15.get(minute, 0.0))
        vol5 = rolling_5.loc[minute].to_numpy(float)[eligible]
        vol15 = rolling_15.loc[minute].to_numpy(float)[eligible]
        fair = entry_mid[eligible]
        paid = entry_ask[eligible]
        ids = [
            f"{session}|{minute}|{'C' if is_call[c] else 'P'}{strikes[c]:.0f}"
            for c in eligible
        ]

        candidates.append(pd.DataFrame({
            "trade_id": ids,
            "session": session,
            "entry_minute": minute,
            "minute_index": float(index),
            "spot": here,
            "strike": strikes[eligible],
            "is_call": is_call[eligible],
            "moneyness": moneyness[eligible],
            "moneyness_rel": moneyness[eligible] / here,
            "entry_premium": fair * CONTRACT_MULTIPLIER,
            "entry_premium_rel": fair / here,
            "entry_ask_usd": paid * CONTRACT_MULTIPLIER,
            "entry_mid_usd": fair * CONTRACT_MULTIPLIER,
            "spread_usd": (paid - entry_bid[eligible]) * CONTRACT_MULTIPLIER,
            "minutes_to_expiry": float(LAST_INDEX - index),
            "contract_volume_5m": vol5,
            "contract_volume_15m": vol15,
            "contract_share_of_chain": (vol15 / chain_here) if chain_here > 0
                                        else np.zeros(eligible.size),
            "chain_volume_15m": chain_here,
            "chain_call_share_15m": (call_here / chain_here) if chain_here > 0 else 0.5,
            "round_trip_usd": FEES_PER_ROUND_TRIP_USD,
            **feats,
        }))

        # Forward path for each candidate, from the entry minute onward.
        window = minutes[where[minute] : where[minute] + MAX_HOLD_MINUTES + 1]
        if len(window) < 2:
            continue
        columns = mid.columns[eligible]
        path_bid = bid_f.loc[window, columns].to_numpy(float)
        path_mid = mid_f.loc[window, columns].to_numpy(float)
        path_spot = spot_f.reindex(window).to_numpy(float)
        good = np.isfinite(path_bid) & np.isfinite(path_mid)
        keep = np.isfinite(path_spot)[:, None] & good

        steps = len(window)
        running_max = np.maximum.accumulate(np.where(good, path_mid, -np.inf), axis=0)
        running_min = np.minimum.accumulate(np.where(good, path_mid, np.inf), axis=0)
        back1 = np.vstack([np.full((1, len(columns)), np.nan), path_mid[:-1]])
        back3 = np.vstack([np.full((min(3, steps), len(columns)), np.nan), path_mid[:-3]])[:steps]
        spot3 = np.concatenate([[np.nan] * min(3, steps), path_spot[:-3]])[:steps]
        left = np.array([float(LAST_INDEX - _index(m)) for m in window])
        entry_row = path_mid[0]

        block = pd.DataFrame({
            "trade_id": np.tile(np.asarray(ids), steps),
            "session": session,
            "minute_in_trade": np.repeat(np.arange(steps, dtype=float), len(columns)),
            "minutes_left": np.repeat(float(steps - 1) - np.arange(steps), len(columns)),
            "minutes_to_expiry": np.repeat(left, len(columns)),
            "bid": path_bid.ravel(),
            "mid": path_mid.ravel(),
            "spot": np.repeat(path_spot, len(columns)),
            "strike": np.tile(strikes[eligible], steps),
            "is_call": np.tile(is_call[eligible], steps),
            "return_from_entry": (path_mid / entry_row - 1.0).ravel(),
            "peak_so_far": (running_max / entry_row - 1.0).ravel(),
            "trough_so_far": (running_min / entry_row - 1.0).ravel(),
            "drawdown_from_peak": (
                path_mid / np.where(np.isfinite(running_max) & (running_max > 0),
                                    running_max, np.nan) - 1.0
            ).ravel(),
            "return_1m": (path_mid / back1 - 1.0).ravel(),
            "return_3m": (path_mid / back3 - 1.0).ravel(),
            "underlying_return_from_entry": np.repeat(path_spot / here - 1.0, len(columns)),
            "underlying_return_3m": np.repeat(path_spot / spot3 - 1.0, len(columns)),
            "moneyness_now": (
                np.where(is_call[eligible][None, :],
                         path_spot[:, None] - strikes[eligible][None, :],
                         strikes[eligible][None, :] - path_spot[:, None])
            ).ravel(),
        })[keep.ravel()]
        paths.append(block)

    if not candidates or not paths:
        return None
    entry = attach_greeks(
        pd.concat(candidates, ignore_index=True), "entry_mid_usd",
        {"iv": "iv", "delta": "delta", "gamma": "gamma_dollars",
         "theta": "theta_share", "vega": "vega_rel"},
    )
    forward = attach_greeks(
        pd.concat(paths, ignore_index=True), "mid",
        {"iv": "iv", "delta": "delta", "gamma": "gamma_dollars",
         "theta": "theta_share_of_price", "vega": "vega"},
    )
    return entry, forward


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    p.add_argument("--trades", type=Path, default=TRADE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-candidates", type=Path, required=True)
    p.add_argument("--out-paths", type=Path, required=True)
    args = p.parse_args()

    files = [f for f in sorted(args.quotes.glob("*.parquet"))
             if "official_context" not in f.name]
    if args.limit:
        files = files[: args.limit]

    cand_parts, path_parts = [], []
    for i, path in enumerate(files, 1):
        got = session_tables(path, args.trades)
        if got is not None:
            cand_parts.append(got[0])
            path_parts.append(got[1])
        if i % 50 == 0:
            print(f"  {i}/{len(files)} sessions, "
                  f"{sum(len(x) for x in cand_parts):,} candidates, "
                  f"{sum(len(x) for x in path_parts):,} path rows", flush=True)

    candidates = pd.concat(cand_parts, ignore_index=True)
    paths = pd.concat(path_parts, ignore_index=True)

    for frame, out in ((candidates, args.out_candidates), (paths, args.out_paths)):
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out)

    trades = candidates["trade_id"].nunique()
    print(
        f"\n{len(candidates):,} candidates ({trades:,} unique), "
        f"{len(paths):,} path rows, {candidates['session'].nunique()} sessions"
    )
    print(f"  {len(candidates) / candidates['session'].nunique():.0f} candidates/session, "
          f"{candidates.groupby(['session', 'entry_minute']).ngroups / candidates['session'].nunique():.0f} "
          f"decision points/session")
    print(f"  mean entry ask ${candidates['entry_ask_usd'].mean():,.0f}, "
          f"mean spread ${candidates['spread_usd'].mean():.2f}")
    print(f"\ncandidates: {args.out_candidates}\npaths: {args.out_paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
