"""Outcome-blind semantic gate on consolidated top-of-book (`cmbp-1`) OPRA data.

What this answers, and what it deliberately does not
---------------------------------------------------
One question only: **can a trade print be signed against a quote that had already
arrived at that instant, and can the book's causal response then be observed?**

That is the mechanism test the 2026-08-22 direction decision named as the single
remaining action with decision value under the standing STOP. It is the question
`ohlcv-1s` fails by construction -- an OHLCV record carries open/high/low/close/
volume and no bid, ask, side or action, so it cannot sign anything.

**This gate reads no outcome.** No label, no P&L, no entry or exit value, no
forward return, no corpus outcome column. It touches raw vendor market data and
nothing else, so it spends no alpha. That restriction is the point: a passing
gate establishes *parser possibility*, never an edge.

**A pass is not evidence of profitability.** These sessions were selected around a
prior route -- 31 trades per session, seven symbols -- so event prevalence on an
unbiased full band remains UNKNOWN. Read the retained share as "can the semantics
be built", never as "how often this happens in the market".

The signing law
---------------
A trade is signed only against the touch standing **strictly before** it: the
previous event's `bid_px_00`/`ask_px_00` for that same instrument, never the
trade row's own book. A print at the prior ask is buyer-initiated, at the prior
bid seller-initiated, and anything strictly inside the touch is **ambiguous and
is retained as ambiguous rather than guessed**. Locked (`bid == ask`) and crossed
(`bid > ask`) prior books are excluded from signing and counted, because a sign
taken from a degenerate book is not a sign.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "v5.cmbp-touch-semantics.v1"
TRADE_ACTION = "T"

#: Minimum share of trades that must sign cleanly for the semantics to be usable
#: at all. Not an economic bar -- a parser bar. Below this the "strict touch"
#: construction is discarding so much of the tape that it cannot describe it.
MIN_SIGNED_SHARE = 0.20


class CmbpSemanticError(RuntimeError):
    """The slice could not be verified, so its semantics must not be trusted."""


@dataclass(frozen=True)
class SessionSemantics:
    session: str
    rows: int
    symbols: int
    trades: int
    no_prior_quote: int
    locked_prior: int
    crossed_prior: int
    clean_prior: int
    at_ask_buyer_initiated: int
    at_bid_seller_initiated: int
    inside_ambiguous: int
    outside_ambiguous: int
    touch_moved_after_trade: int
    ask_rose_after_buy: int
    bid_fell_after_sell: int

    @property
    def signed(self) -> int:
        return self.at_ask_buyer_initiated + self.at_bid_seller_initiated

    @property
    def signed_share(self) -> float:
        return self.signed / self.trades if self.trades else 0.0

    def payload(self) -> dict[str, Any]:
        d = asdict(self)
        d["signed"] = self.signed
        d["signed_share"] = self.signed_share
        return d


REQUIRED = ("ts_event", "action", "price", "bid_px_00", "ask_px_00",
            "instrument_id", "symbol", "publisher_id", "size")


def classify_session(frame: pd.DataFrame, session: str) -> SessionSemantics:
    """Strict prior-touch classification for one session. Fail-closed on shape."""

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise CmbpSemanticError(f"{session}: missing required fields {missing}")
    if frame.empty:
        raise CmbpSemanticError(f"{session}: no rows")

    order = ["instrument_id"] + [c for c in ("ts_recv", "ts_event") if c in frame.columns]
    d = frame.sort_values(order, kind="stable")
    g = d.groupby("instrument_id", sort=False)
    prev_bid, prev_ask = g["bid_px_00"].shift(1), g["ask_px_00"].shift(1)
    post_bid, post_ask = g["bid_px_00"].shift(-1), g["ask_px_00"].shift(-1)

    is_trade = d["action"].astype(str) == TRADE_ACTION
    t = d[is_trade]
    pb, pa = prev_bid[is_trade], prev_ask[is_trade]
    qb, qa = post_bid[is_trade], post_ask[is_trade]

    has_prior = pb.notna() & pa.notna()
    locked = has_prior & (pb == pa)
    crossed = has_prior & (pb > pa)
    clean = has_prior & ~locked & ~crossed

    at_ask = clean & (t["price"] == pa)
    at_bid = clean & (t["price"] == pb)
    inside = clean & (t["price"] > pb) & (t["price"] < pa)
    outside = clean & ((t["price"] < pb) | (t["price"] > pa))
    moved = clean & ((qb != pb) | (qa != pa))

    return SessionSemantics(
        session=session, rows=int(len(d)), symbols=int(d["symbol"].nunique()),
        trades=int(is_trade.sum()), no_prior_quote=int((~has_prior).sum()),
        locked_prior=int(locked.sum()), crossed_prior=int(crossed.sum()),
        clean_prior=int(clean.sum()),
        at_ask_buyer_initiated=int(at_ask.sum()), at_bid_seller_initiated=int(at_bid.sum()),
        inside_ambiguous=int(inside.sum()), outside_ambiguous=int(outside.sum()),
        touch_moved_after_trade=int(moved.sum()),
        ask_rose_after_buy=int((at_ask & (qa > pa)).sum()),
        bid_fell_after_sell=int((at_bid & (qb < pb)).sum()),
    )


def verify_parquet(path: Path, session: str) -> SessionSemantics:
    p = Path(path)
    if not p.is_file():
        raise CmbpSemanticError(f"{session}: no parquet at {p}")
    try:
        frame = pd.read_parquet(p).reset_index()
    except Exception as exc:  # noqa: BLE001 - unreadable input is a FAIL, not a skip
        raise CmbpSemanticError(f"{session}: parquet unreadable: {exc}") from exc
    return classify_session(frame, session)


def build_receipt(results: list[SessionSemantics], *, manifest_sha256: str,
                  decode_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    trades = sum(r.trades for r in results)
    signed = sum(r.signed for r in results)
    share = signed / trades if trades else 0.0
    verdict = "SEMANTICS_PASS_ONLY" if share >= MIN_SIGNED_SHARE else "SEMANTIC_STOP_UNSIGNABLE"
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "verdict": verdict,
        "reads_no_outcome": True,
        "alpha_charged": False,
        "min_signed_share_required": MIN_SIGNED_SHARE,
        "sessions": len(results),
        "rows": sum(r.rows for r in results),
        "trades": trades,
        "signed": signed,
        "signed_share": share,
        "inside_ambiguous": sum(r.inside_ambiguous for r in results),
        "outside_ambiguous": sum(r.outside_ambiguous for r in results),
        "locked_prior": sum(r.locked_prior for r in results),
        "crossed_prior": sum(r.crossed_prior for r in results),
        "no_prior_quote": sum(r.no_prior_quote for r in results),
        "touch_moved_after_trade": sum(r.touch_moved_after_trade for r in results),
        "ask_rose_after_buy": sum(r.ask_rose_after_buy for r in results),
        "bid_fell_after_sell": sum(r.bid_fell_after_sell for r in results),
        "decode_identity": decode_identity,
        "caveat": ("selected-symbol slice chosen around a prior route; a pass establishes parser "
                   "possibility only and event prevalence on an unbiased band is UNKNOWN"),
        "per_session": [r.payload() for r in results],
    }
