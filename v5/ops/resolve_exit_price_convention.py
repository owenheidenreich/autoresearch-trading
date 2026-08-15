"""What price does a 0DTE contract get when it stops trading before the exit?

Phase 0 of job 24.  Two measurements of the same quantity — the pooled near-ATM
60-minute break-even accuracy — disagree by 3.37 points on the same corpus and
the same moneyness band:

* **54.24%** (``option_payoff_by_era``) inner-joins the entry minute to the exit
  minute, so a contract with no trade print at the exit minute is **dropped**.
* **50.87%** (the exit-rule study) keeps it and values it at its **last print**.

The difference matters far beyond bookkeeping.  Dropping a contract that stopped
trading is a **look-ahead filter**: at the entry minute nothing tells the bot
which contracts will still be printing an hour later, so a population defined by
that fact is not one the bot could have selected.  Valuing it at its last print
is causal but assumes a stale price is still obtainable, which is exactly the
assumption that fails if the contract went to zero.

Neither convention is chosen here.  Both are computed, together with two bounds
(mark to zero, mark to intrinsic), and then the question is **settled by
measurement**: the owned quote corpus overlaps this trade corpus, and it carries
a bid and an ask for every contract at every minute whether or not it traded.
For the contracts that vanish from the trade corpus we therefore read the price
that actually existed at the exit minute, and compare it with what each
convention assumed.

The module fits nothing, searches nothing and proposes no policy.  Entry minute,
exit minute, moneyness band and round-trip cost are all fixed declared numbers
carried from the receipts that produced the disagreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENTRY_MINUTE = "09:35"
EXIT_MINUTE = "10:35"
CONTRACT_MULTIPLIER = 100.0

# Near ATM, carried from the moneyness study: |moneyness| <= 25 index points,
# where moneyness is signed so that positive means in the money.
NEAR_ATM_POINTS = 25.0

# The $3.08 measured fee plus the near-ATM spread crossing measured on the owned
# quote corpus 2026-08-13.  Carried, not measured here.
ROUND_TRIP_USD = 25.0

TRADE_CORPUS = Path("/Volumes/AR_TRADING_DATA/spxw_0dte_2022-06-01_2026-07-31/raw")
QUOTE_CORPUS = (
    Path.home() / ".autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized"
)


# --------------------------------------------------------------------------
# The trade corpus: build one row per contract the bot could have bought
# --------------------------------------------------------------------------


def _spot(frame: pd.DataFrame, price: str) -> float | None:
    """Put/call parity proxy: the strike where call and put prices are closest."""

    piv = frame.pivot_table(
        index="strike", columns="right", values=price, aggfunc="last"
    ).dropna()
    if piv.empty or not {"C", "P"}.issubset(piv.columns):
        return None
    return float((piv["C"] - piv["P"]).abs().idxmin())


def session_entries(path: Path) -> pd.DataFrame | None:
    """Near-ATM contracts trading at the entry minute, with their exit treatment.

    One row per contract.  ``exit_close`` is the trade print at the exit minute
    and is NaN when the contract did not trade in that minute; ``last_close``
    and ``last_minute`` describe its final print at or before the exit minute,
    which is what the last-print convention would use instead.
    """

    try:
        frame = pd.read_parquet(
            path, columns=["ts_event", "close", "symbol", "strike", "right"]
        )
    except Exception:
        # A non-trading day banks an empty file that carries no schema at all.
        return None
    if frame.empty:
        return None
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce").astype(float)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None

    frame["minute"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    entry = frame[frame["minute"] == ENTRY_MINUTE]
    exit_ = frame[frame["minute"] == EXIT_MINUTE]
    if entry.empty or exit_.empty:
        return None
    s0, s1 = _spot(entry, "close"), _spot(exit_, "close")
    if s0 is None or s1 is None or s0 == s1:
        return None

    entry = entry.drop_duplicates(subset=["strike", "right"], keep="last").copy()
    is_call = entry["right"].eq("C").to_numpy()
    entry["moneyness"] = np.where(
        is_call, s0 - entry["strike"], entry["strike"] - s0
    )
    picked = entry[entry["moneyness"].abs() <= NEAR_ATM_POINTS].copy()
    if picked.empty:
        return None

    # The exit print, where one exists.
    at_exit = (
        exit_.drop_duplicates(subset=["strike", "right"], keep="last")
        .set_index(["strike", "right"])["close"]
    )
    # The final print at or before the exit minute, which is what the last-print
    # convention falls back to.  Includes the entry bar itself, so a contract
    # that never traded again is valued at what it was bought for.
    window = frame[(frame["minute"] >= ENTRY_MINUTE) & (frame["minute"] <= EXIT_MINUTE)]
    window = window.sort_values("minute")
    last = window.groupby(["strike", "right"], sort=False).agg(
        last_close=("close", "last"), last_minute=("minute", "last")
    )

    keys = pd.MultiIndex.from_arrays([picked["strike"], picked["right"]])
    picked["entry_close"] = picked["close"].to_numpy(float)
    picked["exit_close"] = at_exit.reindex(keys).to_numpy(float)
    picked["last_close"] = last["last_close"].reindex(keys).to_numpy(float)
    picked["last_minute"] = last["last_minute"].reindex(keys).to_numpy(object)
    is_call = picked["right"].eq("C").to_numpy()
    picked["intrinsic_exit"] = np.where(
        is_call,
        np.maximum(0.0, s1 - picked["strike"].to_numpy(float)),
        np.maximum(0.0, picked["strike"].to_numpy(float) - s1),
    )
    picked["premium"] = picked["entry_close"] * CONTRACT_MULTIPLIER
    picked["correct"] = is_call == bool(s1 > s0)
    picked["session"] = path.name[:10]
    picked["spot_entry"] = s0
    picked["spot_exit"] = s1
    return picked[
        [
            "session",
            "symbol",
            "strike",
            "right",
            "moneyness",
            "premium",
            "correct",
            "entry_close",
            "exit_close",
            "last_close",
            "last_minute",
            "intrinsic_exit",
            "spot_entry",
            "spot_exit",
        ]
    ]


# --------------------------------------------------------------------------
# Break-even under a stated convention
# --------------------------------------------------------------------------


def breakeven(entry_close: np.ndarray, exit_price: np.ndarray, correct: np.ndarray) -> dict:
    """Accuracy a policy must reach for these payoffs to break even.

    ``L / (W + L)`` where ``W`` is the mean net dollars earned when the side was
    right and ``L`` the mean net dollars lost when it was wrong.
    """

    net = (exit_price - entry_close) * CONTRACT_MULTIPLIER - ROUND_TRIP_USD
    ok = correct.astype(bool)
    if ok.sum() < 50 or (~ok).sum() < 50:
        return {"observations": int(len(net)), "breakeven": None}
    win, loss = float(net[ok].mean()), float(-net[~ok].mean())
    denom = win + loss
    return {
        "observations": int(len(net)),
        "correct_n": int(ok.sum()),
        "wrong_n": int((~ok).sum()),
        "mean_net_when_correct_usd": round(win, 2),
        "mean_net_when_wrong_usd": round(-loss, 2),
        "breakeven": round(loss / denom, 6) if denom > 0 and win > 0 else None,
    }


def conventions(table: pd.DataFrame) -> dict:
    """The four ways the disagreement could be resolved, all on one population."""

    entry = table["entry_close"].to_numpy(float)
    exit_close = table["exit_close"].to_numpy(float)
    correct = table["correct"].to_numpy(bool)
    present = np.isfinite(exit_close)

    out = {
        "drop_vanished": {
            **breakeven(entry[present], exit_close[present], correct[present]),
            "meaning": (
                "inner-join entry to exit; a contract with no print at the exit "
                "minute is removed. Not causal: the bot cannot know at entry "
                "which contracts will still be printing an hour later."
            ),
        },
        "last_print": {
            **breakeven(entry, np.where(present, exit_close, table["last_close"]), correct),
            "meaning": (
                "value a vanished contract at its final print at or before the "
                "exit minute. Causal, but assumes a stale price is obtainable."
            ),
        },
        "mark_to_zero": {
            **breakeven(entry, np.where(present, exit_close, 0.0), correct),
            "meaning": "worst case: every vanished contract is worthless.",
        },
        "mark_to_intrinsic": {
            **breakeven(
                entry,
                np.where(present, exit_close, table["intrinsic_exit"]),
                correct,
            ),
            "meaning": (
                "value a vanished contract at exercise value against the exit "
                "spot. A lower bound: the contract still holds time value at "
                f"{EXIT_MINUTE}."
            ),
        },
    }
    return out


def vanished_profile(table: pd.DataFrame) -> dict:
    """Who the vanishing contracts are, which is what decides the question."""

    present = np.isfinite(table["exit_close"].to_numpy(float))
    gone = table[~present]
    if gone.empty:
        return {"vanished_n": 0}
    stale = gone["last_close"].to_numpy(float) / gone["entry_close"].to_numpy(float) - 1.0
    never = (gone["last_minute"] == ENTRY_MINUTE).to_numpy(bool)
    return {
        "vanished_n": int(len(gone)),
        "vanished_share": round(float((~present).mean()), 4),
        "share_of_vanished_that_never_traded_again": round(float(never.mean()), 4),
        "share_of_vanished_on_the_correct_side": round(
            float(gone["correct"].to_numpy(bool).mean()), 4
        ),
        "share_of_present_on_the_correct_side": round(
            float(table[present]["correct"].to_numpy(bool).mean()), 4
        ),
        "mean_last_print_vs_entry_pct": round(100 * float(np.mean(stale)), 2),
        "median_last_print_vs_entry_pct": round(100 * float(np.median(stale)), 2),
        "mean_entry_premium_usd": round(float(gone["premium"].mean()), 2),
        "mean_entry_premium_usd_present": round(float(table[present]["premium"].mean()), 2),
        "mean_abs_moneyness_vanished": round(float(gone["moneyness"].abs().mean()), 2),
        "mean_abs_moneyness_present": round(
            float(table[present]["moneyness"].abs().mean()), 2
        ),
    }


def by_year(table: pd.DataFrame) -> dict:
    """The same split year by year.

    The quote settlement can only reach the sessions the owned quote corpus
    covers, which are the most recent and the most liquid. If the vanishing
    contracts look the same in every year, the settlement carries; if the early
    years differ, it does not, and this is where that shows.
    """

    out = {}
    for year, part in table.groupby(table["session"].str[:4], sort=True):
        conv = conventions(part)
        profile = vanished_profile(part)
        out[str(year)] = {
            "sessions": int(part["session"].nunique()),
            "entries": int(len(part)),
            "vanished_share": profile.get("vanished_share", 0.0),
            "share_of_vanished_on_the_correct_side": profile.get(
                "share_of_vanished_on_the_correct_side"
            ),
            "mean_last_print_vs_entry_pct": profile.get("mean_last_print_vs_entry_pct"),
            "breakeven_drop_vanished": conv["drop_vanished"]["breakeven"],
            "breakeven_last_print": conv["last_print"]["breakeven"],
            "breakeven_mark_to_zero": conv["mark_to_zero"]["breakeven"],
        }
    return out


# --------------------------------------------------------------------------
# The settlement: read the price that actually existed
# --------------------------------------------------------------------------


def quote_exit_prices(path: Path) -> pd.DataFrame | None:
    """Bid, ask and mid at the exit minute for every contract in the chain.

    The quote corpus records a row per contract per minute whether or not it
    traded, so it can price exactly the contracts the trade corpus loses.
    """

    d = pd.read_parquet(path, columns=["event_time", "raw_symbol", "bid", "ask", "mid"])
    if d.empty:
        return None
    minute = (
        pd.to_datetime(d["event_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    d = d[minute == EXIT_MINUTE]
    if d.empty:
        return None
    d = d.drop_duplicates(subset=["raw_symbol"], keep="last")
    return d.rename(
        columns={"bid": "quote_bid", "ask": "quote_ask", "mid": "quote_mid"}
    )[["raw_symbol", "quote_bid", "quote_ask", "quote_mid"]]


def settle(table: pd.DataFrame, quotes: pd.DataFrame) -> dict:
    """Compare each convention with the price that actually existed at the exit."""

    joined = table.merge(quotes, left_on="symbol", right_on="raw_symbol", how="left")
    quoted = np.isfinite(joined["quote_mid"].to_numpy(float))
    present = np.isfinite(joined["exit_close"].to_numpy(float))
    entry = joined["entry_close"].to_numpy(float)
    correct = joined["correct"].to_numpy(bool)
    mid = joined["quote_mid"].to_numpy(float)
    bid = joined["quote_bid"].to_numpy(float)
    last = joined["last_close"].to_numpy(float)

    # Restrict every comparison to contracts the quote corpus can price, so the
    # conventions are compared on one population rather than four.
    keep = quoted
    sub = joined[keep]
    gone = keep & ~present

    exit_close = joined["exit_close"].to_numpy(float)
    truth_mid = breakeven(entry[keep], mid[keep], correct[keep])
    truth_bid = breakeven(entry[keep], bid[keep], correct[keep])
    as_dropped = breakeven(
        entry[keep & present], exit_close[keep & present], correct[keep & present]
    )
    as_last = breakeven(
        entry[keep], np.where(present[keep], exit_close[keep], last[keep]), correct[keep]
    )
    # The isolation that answers the question. Identical to ``last_print`` in
    # every respect except that the contracts which stopped printing are valued
    # at the quote that actually existed instead of at a stale trade. Every
    # other contract keeps its trade price, so trade-versus-mid level effects
    # cannot leak into the comparison.
    corrected = breakeven(
        entry[keep], np.where(present[keep], exit_close[keep], mid[keep]), correct[keep]
    )

    error = mid[gone] - last[gone]
    priced = None
    if gone.any():
        priced = {
            "mean_actual_mid_usd": round(float(np.mean(mid[gone]) * CONTRACT_MULTIPLIER), 2),
            "mean_assumed_last_print_usd": round(
                float(np.mean(last[gone]) * CONTRACT_MULTIPLIER), 2
            ),
            "mean_error_usd": round(float(np.mean(error) * CONTRACT_MULTIPLIER), 2),
            "median_error_usd": round(float(np.median(error) * CONTRACT_MULTIPLIER), 2),
            "share_where_last_print_overstates": round(float((error < 0).mean()), 4),
            "share_actually_worthless_under_5c": round(float((mid[gone] < 0.05).mean()), 4),
            "mean_actual_mid_vs_entry_pct": round(
                100 * float(np.mean(mid[gone] / entry[gone] - 1.0)), 2
            ),
        }
    return {
        "sessions": int(sub["session"].nunique()),
        "observations": int(keep.sum()),
        "unquoted_dropped": int((~quoted).sum()),
        "vanished_and_quoted_n": int(gone.sum()),
        "truth_quote_mid": truth_mid,
        "truth_quote_bid": truth_bid,
        "same_population_drop_vanished": as_dropped,
        "same_population_last_print": as_last,
        "corrected_quote_for_vanished_only": corrected,
        "vanished_contracts_priced_by_quotes": priced,
    }


# --------------------------------------------------------------------------


def build(corpus: Path, limit: int | None = None) -> tuple[pd.DataFrame, int, int]:
    files = sorted(corpus.glob("*.parquet"))
    if limit:
        files = files[:limit]
    parts, skipped = [], 0
    for i, path in enumerate(files, 1):
        got = session_entries(path)
        if got is None:
            skipped += 1
        else:
            parts.append(got)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    if not parts:
        raise SystemExit("no usable sessions")
    return pd.concat(parts, ignore_index=True), len(files), skipped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=TRADE_CORPUS)
    p.add_argument("--quote-corpus", type=Path, default=QUOTE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    table, available, skipped = build(args.corpus, args.limit)
    conv = conventions(table)
    profile = vanished_profile(table)
    years = by_year(table)

    print(
        f"\n{table['session'].nunique()} sessions, {len(table):,} near-ATM entries, "
        f"{profile['vanished_n']:,} vanish before {EXIT_MINUTE} "
        f"({100 * profile['vanished_share']:.1f}%)\n"
    )
    head = f"{'convention':<20} {'n':>8} {'correct':>9} {'wrong':>9} {'break-even':>11}"
    print(head)
    print("-" * len(head))
    for name, row in conv.items():
        be = row.get("breakeven")
        print(
            f"{name:<20} {row['observations']:>8,} "
            f"{row.get('mean_net_when_correct_usd', 0):>9,.0f} "
            f"{row.get('mean_net_when_wrong_usd', 0):>9,.0f} "
            f"{(f'{100 * be:.2f}%' if be else 'n/a'):>11}"
        )

    head = (
        f"\n{'year':<6} {'sess':>5} {'entries':>8} {'vanished':>9} {'of those right':>15} "
        f"{'last vs entry':>14} {'drop':>7} {'last':>7}"
    )
    print(head)
    print("-" * (len(head) - 1))
    for year, row in years.items():
        right = row["share_of_vanished_on_the_correct_side"]
        drift = row["mean_last_print_vs_entry_pct"]
        right_txt = f"{100 * right:.1f}%" if right is not None else "-"
        drift_txt = f"{drift:+.0f}%" if drift is not None else "-"
        print(
            f"{year:<6} {row['sessions']:>5} {row['entries']:>8,} "
            f"{100 * row['vanished_share']:>8.1f}% {right_txt:>15} {drift_txt:>14} "
            f"{100 * row['breakeven_drop_vanished']:>6.2f}% "
            f"{100 * row['breakeven_last_print']:>6.2f}%"
        )

    settlement = None
    quote_files = sorted(args.quote_corpus.glob("databento_spxw_0dte_*.parquet"))
    quote_files = [f for f in quote_files if "official_context" not in f.name]
    if quote_files:
        overlap = set(table["session"].unique())
        parts = []
        for i, path in enumerate(quote_files, 1):
            session = path.name.replace("databento_spxw_0dte_", "")[:10]
            if session not in overlap:
                continue
            got = quote_exit_prices(path)
            if got is not None:
                got["session"] = session
                parts.append(got)
            if i % 50 == 0:
                print(f"  quotes {i}/{len(quote_files)}", flush=True)
        if parts:
            quotes = pd.concat(parts, ignore_index=True)
            overlapping = table[table["session"].isin(set(quotes["session"]))]
            settlement = settle(overlapping, quotes.drop(columns=["session"]))

    payload = {
        "schema_version": "v5.exit-price-convention.v1",
        "question": (
            "Two measurements of the pooled near-ATM 60-minute break-even "
            "disagree (54.24% vs 50.87%). Which handling of a contract that "
            "stops trading before the exit minute is right?"
        ),
        "entry_minute": ENTRY_MINUTE,
        "exit_minute": EXIT_MINUTE,
        "moneyness_band_points": NEAR_ATM_POINTS,
        "round_trip_cost_usd": ROUND_TRIP_USD,
        "round_trip_cost_provenance": "CARRIED from the owned quote corpus, not measured here",
        "trade_corpus": str(args.corpus),
        "quote_corpus": str(args.quote_corpus),
        "sessions_available": available,
        "sessions_used": int(table["session"].nunique()),
        "sessions_skipped": skipped,
        "entries": int(len(table)),
        "conventions": conv,
        "vanished_profile": profile,
        "by_year": years,
        "settlement_against_quotes": settlement,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if settlement:
        v = settlement["vanished_contracts_priced_by_quotes"]
        print(
            f"\nsettlement on {settlement['sessions']} overlapping sessions, "
            f"{settlement['vanished_and_quoted_n']:,} vanished contracts priced by quotes:"
        )
        print(
            f"  last print assumed ${v['mean_assumed_last_print_usd']:,.0f}, "
            f"actual mid ${v['mean_actual_mid_usd']:,.0f}, "
            f"error ${v['mean_error_usd']:+,.0f}"
        )
        for key in (
            "same_population_drop_vanished",
            "same_population_last_print",
            "corrected_quote_for_vanished_only",
            "truth_quote_mid",
            "truth_quote_bid",
        ):
            be = settlement[key].get("breakeven")
            print(f"  {key:<32} {(f'{100 * be:.2f}%' if be else 'n/a'):>8}")
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
