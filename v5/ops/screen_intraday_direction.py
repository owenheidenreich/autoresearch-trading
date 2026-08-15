"""The test this project has never run: does any declared rule beat the bar?

Six research campaigns produced one measured negative and five stops for lack of
statistical power. **No signal has ever been tested for directional edge on the
option corpus.** This module does that, once, on a family declared before it runs.

What makes it a screen rather than a search:

* the entry rules are **closed-form, named and enumerated here**, so the
  multiplicity is exactly countable before any number exists;
* nothing is fitted and no threshold is searched — every rule is a sign test on
  a causal feature;
* the family is scored **whole**. Every member is reported, the best member is
  not selected, and the confidence bound carries the Bonferroni correction for
  the declared family size;
* abstention is a first-class outcome: a rule returning zero stands down, and
  those slots are excluded rather than counted wrong.

A positive here is **not** a result. It is permission to spend a known-answer
campaign on the specific member, which is the gate that stopped both previous
attempts and which no number below bypasses.

## The spot estimator, which had to be fixed first

Every earlier measurement located the underlying by finding the strike where the
call and put prices are closest. That is quantised to the five-point strike grid,
which is coarser than the moves being measured — it is why 39 of 77 five-minute
slots a session showed "no move" at all.

Put/call parity gives a continuous estimate instead: for any strike ``K`` with
both sides quoted, ``S = K + C - P`` up to a discount factor negligible over
hours. Averaging that across near-the-money strikes gives a spot series fine
enough to compute a five-minute return from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import FIRST_INDEX, LAST_INDEX, _index, _label
from v5.ops.resolve_exit_price_convention import (
    CONTRACT_MULTIPLIER,
    NEAR_ATM_POINTS,
    TRADE_CORPUS,
)
from v5.research import statistics as st

# --- the declaration ---------------------------------------------------------
# Everything in this block is frozen before the run and hashed into the receipt.

HOLDS_MINUTES = (15, 60)
LOOKBACKS_MINUTES = (5, 15, 30)
COSTS_USD = (25.0, 14.0)
PRIMARY_COST_USD = 25.0
# Strikes within this distance of spot are averaged for the parity estimate.
PARITY_WINDOW_POINTS = 30.0
MIN_PARITY_STRIKES = 3
# Session-block bootstrap: sessions are the independent unit, not trades.
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260813


def _sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def momentum_with(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Trade with the last fifteen minutes."""

    return _sign(f["return_15m"])


def momentum_against(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fade the last fifteen minutes."""

    return -_sign(f["return_15m"])


def fast_momentum_with(f: Mapping[str, np.ndarray]) -> np.ndarray:
    return _sign(f["return_5m"])


def fast_momentum_against(f: Mapping[str, np.ndarray]) -> np.ndarray:
    return -_sign(f["return_5m"])


def confirmed_momentum(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Trade only when the five- and thirty-minute returns agree.

    Abstains on disagreement. Selectivity bought at the cost of occupancy, which
    is the same shape as the declared ``confirmed_gap`` rule for the open.
    """

    fast, slow = _sign(f["return_5m"]), _sign(f["return_30m"])
    return np.where((fast == slow) & (fast != 0), fast, 0.0)


def contrarian_stretch(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fade a move that has carried price to the edge of the session range."""

    at_high = f["range_position"] >= 0.9
    at_low = f["range_position"] <= 0.1
    return np.where(at_high, -1.0, np.where(at_low, 1.0, 0.0))


def breakout_with(f: Mapping[str, np.ndarray]) -> np.ndarray:
    """Trade with a move that has carried price to the edge of the session range."""

    at_high = f["range_position"] >= 0.9
    at_low = f["range_position"] <= 0.1
    return np.where(at_high, 1.0, np.where(at_low, -1.0, 0.0))


ENTRY_RULES: Mapping[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    "momentum_with": momentum_with,
    "momentum_against": momentum_against,
    "fast_momentum_with": fast_momentum_with,
    "fast_momentum_against": fast_momentum_against,
    "confirmed_momentum": confirmed_momentum,
    "contrarian_stretch": contrarian_stretch,
    "breakout_with": breakout_with,
}

# Declared family size: every rule at every hold is one member.
FAMILY_SIZE = len(ENTRY_RULES) * len(HOLDS_MINUTES)


def declaration() -> dict:
    """The frozen statement of what will be scored, hashed into the receipt."""

    return {
        "entry_rules": sorted(ENTRY_RULES),
        "holds_minutes": list(HOLDS_MINUTES),
        "lookbacks_minutes": list(LOOKBACKS_MINUTES),
        "moneyness_band_points": NEAR_ATM_POINTS,
        "costs_usd": list(COSTS_USD),
        "primary_cost_usd": PRIMARY_COST_USD,
        "family_size": FAMILY_SIZE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scoring": (
            "mean net dollars per trade, with a session-block bootstrap lower "
            "bound at a Bonferroni-corrected one-sided level for the declared "
            "family size. Every member reported; the best is not selected."
        ),
        "abstention": "a rule returning zero stands down; those slots are excluded",
    }


def declaration_hash() -> str:
    payload = json.dumps(declaration(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


# --- the corpus --------------------------------------------------------------


def parity_spot(pivot: pd.DataFrame) -> pd.Series | None:
    """Continuous underlying estimate per minute, from ``S = K + C - P``.

    Averaged over strikes near the money, which is where parity is tightest and
    where both sides actually trade.
    """

    calls = pivot.loc[:, pivot.columns.get_level_values("right") == "C"]
    puts = pivot.loc[:, pivot.columns.get_level_values("right") == "P"]
    calls.columns = calls.columns.get_level_values("strike")
    puts.columns = puts.columns.get_level_values("strike")
    shared = calls.columns.intersection(puts.columns)
    if len(shared) < MIN_PARITY_STRIKES:
        return None
    implied = (calls[shared] - puts[shared]).add(pd.Series(shared, index=shared))
    coarse = implied.median(axis=1)
    # Keep only strikes near that first estimate, then average again: a far
    # in-the-money strike whose put barely trades carries a stale parity.
    near = implied.where((implied.columns.to_numpy() - coarse.to_numpy()[:, None])
                         .__abs__() <= PARITY_WINDOW_POINTS)
    spot = near.mean(axis=1)
    enough = near.notna().sum(axis=1) >= MIN_PARITY_STRIKES
    return spot.where(enough)


def session_slots(path: Path, hold: int) -> pd.DataFrame | None:
    """Every serial slot, with causal features and the payoff of each side."""

    try:
        frame = pd.read_parquet(path, columns=["ts_event", "close", "strike", "right"])
    except Exception:
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
    pivot = frame.pivot_table(
        index="minute", columns=["strike", "right"], values="close", aggfunc="last"
    ).sort_index()
    if pivot.empty:
        return None
    spot = parity_spot(pivot)
    if spot is None or spot.notna().sum() < 60:
        return None
    # A continuous series for feature use: the underlying exists every minute
    # even when no contract printed.
    ffilled_spot = spot.ffill()
    filled = pivot.ffill()
    strikes = pivot.columns.get_level_values("strike").to_numpy(float)
    is_call = pivot.columns.get_level_values("right").to_numpy() == "C"

    rows = []
    entry_index = FIRST_INDEX
    while entry_index + hold <= LAST_INDEX:
        entry_label, exit_label = _label(entry_index), _label(entry_index + hold)
        entry_index += hold
        if entry_label not in pivot.index or exit_label not in pivot.index:
            continue
        s0, s1 = ffilled_spot.get(entry_label), ffilled_spot.get(exit_label)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1):
            continue

        # Causal features: everything strictly at or before the entry minute.
        history = ffilled_spot.loc[:entry_label].dropna()
        if len(history) < max(LOOKBACKS_MINUTES) + 1:
            continue
        feats = {}
        for back in LOOKBACKS_MINUTES:
            past = ffilled_spot.get(_label(_index(entry_label) - back))
            if past is None or not np.isfinite(past) or past <= 0:
                feats = {}
                break
            feats[f"return_{back}m"] = s0 / past - 1.0
        if not feats:
            continue
        low, high = float(history.min()), float(history.max())
        feats["range_position"] = (s0 - low) / (high - low) if high > low else 0.5

        entry_price = pivot.loc[entry_label].to_numpy(float)
        exit_present = pivot.loc[exit_label].to_numpy(float)
        exit_price = np.where(
            np.isfinite(exit_present), exit_present, filled.loc[exit_label].to_numpy(float)
        )
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        eligible = (
            np.isfinite(entry_price)
            & np.isfinite(exit_price)
            & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        )
        picked = {}
        for side, want_call in (("call", True), ("put", False)):
            on_side = np.flatnonzero(eligible & (is_call == want_call))
            if not on_side.size:
                break
            i = int(on_side[np.argmin(np.abs(moneyness[on_side]))])
            picked[f"{side}_gross"] = (
                exit_price[i] - entry_price[i]
            ) * CONTRACT_MULTIPLIER
            picked[f"{side}_premium"] = entry_price[i] * CONTRACT_MULTIPLIER
        if len(picked) < 4:
            continue

        rows.append(
            {
                "session": path.name[:10],
                "entry_minute": entry_label,
                "hour": entry_label[:2],
                "spot_move": s1 - s0,
                **feats,
                **picked,
            }
        )
    return pd.DataFrame(rows) if rows else None


# --- scoring -----------------------------------------------------------------


def score(table: pd.DataFrame, rule_name: str, cost: float, z_alpha: float) -> dict:
    """One family member, scored whole."""

    features = {
        name: table[name].to_numpy(float)
        for name in table.columns
        if name.startswith("return_") or name == "range_position"
    }
    if not features:
        return {"rule": rule_name, "trades": 0, "verdict": "no features"}
    call = ENTRY_RULES[rule_name](features)
    taken = call != 0
    if taken.sum() < 100:
        return {"rule": rule_name, "trades": int(taken.sum()), "verdict": "too few trades"}

    gross = np.where(
        call > 0, table["call_gross"].to_numpy(float), table["put_gross"].to_numpy(float)
    )
    net = (gross - cost)[taken]
    sessions = table["session"].to_numpy()[taken]
    # Directional accuracy is about the underlying, not about which contract
    # happened to pay: the rule called a side and the spot either went that way
    # or it did not.
    correct = np.sign(table["spot_move"].to_numpy(float))[taken] == call[taken]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = net[np.concatenate([index[s] for s in chosen])].mean()
    lower = float(np.quantile(draws, _tail_level(z_alpha)))

    return {
        "rule": rule_name,
        "trades": int(taken.sum()),
        "sessions": int(len(unique)),
        "abstention_rate": round(float(1.0 - taken.mean()), 4),
        "accuracy": round(float(correct.mean()), 6),
        "mean_net_usd": round(float(net.mean()), 2),
        "mean_net_usd_per_session": round(
            float(net.sum() / len(unique)), 2
        ),
        "bootstrap_lower_usd": round(lower, 2),
        "clears_zero": bool(lower > 0.0),
        "bootstrap_level": round(_tail_level(z_alpha), 6),
    }


def _tail_level(z_alpha: float) -> float:
    """One-sided tail probability for a z value, via the shared quantile."""

    lo, hi = 1e-9, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if st.normal_quantile(1.0 - mid) > z_alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=TRADE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.corpus.glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]

    z_family = st.bonferroni_quantile(FAMILY_SIZE)
    results = {}
    for hold in HOLDS_MINUTES:
        parts = []
        for i, path in enumerate(files, 1):
            got = session_slots(path, hold)
            if got is not None:
                parts.append(got)
            if i % 200 == 0:
                print(f"  {hold}m: {i}/{len(files)}", flush=True)
        if not parts:
            continue
        table = pd.concat(parts, ignore_index=True)
        block = {"slots": int(len(table)), "sessions": int(table["session"].nunique())}
        for cost in COSTS_USD:
            block[f"cost_{cost}"] = [
                score(table, name, cost, z_family) for name in sorted(ENTRY_RULES)
            ]
        # Secondary, and labelled as such: the last two hours only.
        late = table[table["hour"].isin({"14", "15"})]
        if len(late) > 500:
            block["late_session_only"] = {
                "slots": int(len(late)),
                "cost_25.0": [
                    score(late, name, PRIMARY_COST_USD, z_family)
                    for name in sorted(ENTRY_RULES)
                ],
            }
        results[f"{hold}m"] = block

    payload = {
        "schema_version": "v5.intraday-direction-screen.v1",
        "declaration": declaration(),
        "declaration_sha256": declaration_hash(),
        "bonferroni_z": round(z_family, 6),
        "corpus": str(args.corpus),
        "files_scanned": len(files),
        "by_hold": results,
        "what_a_positive_means": (
            "permission to spend a known-answer campaign on that member, and "
            "nothing else. It is not a validated result and no economics here "
            "have passed a gate."
        ),
        "known_limitations": [
            "the underlying is a put/call parity estimate from traded option "
            "prices, not an SPX print",
            "prices are last trades, so they carry bid/ask bounce",
            "the round trip is carried from the owned quote corpus",
            "the late-session block is secondary and its multiplicity is not "
            "separately charged; it may not be read as a declared result",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\ndeclaration {declaration_hash()[:16]}, family {FAMILY_SIZE}, "
          f"Bonferroni z {z_family:.3f}")
    for hold, block in results.items():
        print(f"\n{hold} hold — {block['sessions']} sessions, {block['slots']:,} slots")
        head = (
            f"  {'rule':>22} {'trades':>8} {'stood down':>11} {'accuracy':>9} "
            f"{'net/trade':>10} {'lower bound':>12} {'clears 0':>9}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for row in block[f"cost_{PRIMARY_COST_USD}"]:
            if "accuracy" not in row:
                print(f"  {row['rule']:>22} {row['trades']:>8,}   {row['verdict']}")
                continue
            print(
                f"  {row['rule']:>22} {row['trades']:>8,} "
                f"{100 * row['abstention_rate']:>10.1f}% {100 * row['accuracy']:>8.2f}% "
                f"{row['mean_net_usd']:>10,.1f} {row['bootstrap_lower_usd']:>12,.1f} "
                f"{('YES' if row['clears_zero'] else 'no'):>9}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
