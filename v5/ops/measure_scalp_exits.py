"""What is a 0DTE contract worth to someone who exits well, rather than on a clock?

Every option-layer measurement this project has made buys at one minute and sells
at a fixed minute later. That is not how the strategy is meant to work. A scalper
buys a call, watches it, and sells into the move — taking a profit when the
contract has made its largest premium change, cutting quickly when it has not.
The fixed-horizon assumption was load-bearing in every conclusion so far, and it
was never tested.

It can be tested, because the corpus carries **high and low for every contract
every minute** and nothing in this project has ever read them. Only closes were
used. The high is what a resting take-profit order would have been filled at; the
low is what a stop would have been filled at. That is enough to price a real exit
policy rather than a clock.

## What is measured

For every entry slot and every near-the-money contract:

* **Maximum favourable excursion** — the best price the contract reached before
  the window ended. The ceiling on any exit rule.
* **Maximum adverse excursion** — the worst it reached. The floor any stop has
  to survive.
* **Time to the peak**, which is what decides whether a fast exit has anything
  to capture.
* The payoff under declared exit rules, simulated minute by minute.

## The rules of the simulation, stated so they cannot flatter

**A stop is checked before a target inside the same minute.** When both are
touched in one bar, the order of events is unknowable from a minute bar, so the
loss is taken. This is the conservative choice and it costs the good rules more
than it costs the bad ones.

**A target fills at the target, not at the high.** Reaching the high proves the
limit would have been hit; it does not entitle the trade to the extreme.

**A stop fills at the stop, not at the low**, except when the bar opened beyond
it, in which case it fills at the open — a gap through the level is not a fill at
the level.

**Nothing is fitted.** Every level is a declared constant. The exits are the ones
a trader would name, not ones searched for on this data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import FIRST_INDEX, LAST_INDEX, _label
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER, TRADE_CORPUS
from v5.ops.screen_intraday_direction import parity_spot

# Entries every quarter hour; each trade may run this many minutes before a
# time stop closes it.
ENTRY_EVERY_MINUTES = 15
WINDOW_MINUTES = 30
NEAR_ATM_POINTS = 25.0
COST_PER_LEG_USD = 23.0  # measured aggressive round trip, 2026-08-13

# Declared exit rules. ``(take_profit, stop_loss, trail)`` as fractions of the
# entry premium; ``None`` means the leg is not used.
EXIT_RULES: dict[str, tuple[float | None, float | None, float | None]] = {
    "hold_to_horizon": (None, None, None),
    "tp20_sl20": (0.20, -0.20, None),
    "tp30_sl20": (0.30, -0.20, None),
    "tp50_sl30": (0.50, -0.30, None),
    "tp100_sl30": (1.00, -0.30, None),
    "sl30_let_run": (None, -0.30, None),
    "trail30": (None, None, 0.30),
    "trail50": (None, None, 0.50),
}


def simulate_exit(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    entry: float,
    rule: tuple[float | None, float | None, float | None],
) -> float:
    """Exit price under one declared rule, walking the window minute by minute."""

    take, stop, trail = rule
    peak = entry
    for i in range(len(highs)):
        o, h, l = opens[i], highs[i], lows[i]
        if not np.isfinite(h) or not np.isfinite(l):
            continue
        level = None
        if stop is not None:
            level = entry * (1.0 + stop)
        if trail is not None:
            trailing = peak * (1.0 - trail)
            level = trailing if level is None else max(level, trailing)
        # Stop first: when a bar touches both, a minute bar cannot say which
        # came first, so the trade takes the loss.
        if level is not None and l <= level:
            return float(min(o, level)) if np.isfinite(o) else float(level)
        if take is not None and h >= entry * (1.0 + take):
            return float(entry * (1.0 + take))
        peak = max(peak, h)
    last = closes[np.isfinite(closes)]
    return float(last[-1]) if last.size else float(entry)


def session_trades(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_parquet(
            path, columns=["ts_event", "open", "high", "low", "close", "strike", "right"]
        )
    except Exception:
        return None
    if frame.empty:
        return None
    for col in ("strike", "open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None
    frame["minute"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    grids = {
        field: frame.pivot_table(
            index="minute", columns=["strike", "right"], values=field, aggfunc="last"
        ).sort_index()
        for field in ("open", "high", "low", "close")
    }
    close_grid = grids["close"]
    if close_grid.empty:
        return None
    spot = parity_spot(close_grid)
    if spot is None:
        return None
    spot = spot.ffill()
    strikes = close_grid.columns.get_level_values("strike").to_numpy(float)
    is_call = close_grid.columns.get_level_values("right").to_numpy() == "C"
    minutes = list(close_grid.index)
    position = {m: i for i, m in enumerate(minutes)}
    arrays = {k: v.reindex(columns=close_grid.columns).to_numpy(float) for k, v in grids.items()}

    rows = []
    entry_index = FIRST_INDEX
    while entry_index + WINDOW_MINUTES <= LAST_INDEX:
        entry_label = _label(entry_index)
        exit_label = _label(entry_index + WINDOW_MINUTES)
        entry_index += ENTRY_EVERY_MINUTES
        if entry_label not in position or exit_label not in position:
            continue
        s0, s1 = spot.get(entry_label), spot.get(exit_label)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1):
            continue
        i0, i1 = position[entry_label], position[exit_label]
        entry_prices = arrays["close"][i0]
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        eligible = np.isfinite(entry_prices) & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        if not eligible.any():
            continue
        up = bool(s1 > s0)

        for want_call in (True, False):
            on_side = np.flatnonzero(eligible & (is_call == want_call))
            if not on_side.size:
                continue
            j = int(on_side[np.argmin(np.abs(moneyness[on_side]))])
            entry = float(entry_prices[j])
            hi = arrays["high"][i0 + 1 : i1 + 1, j]
            lo = arrays["low"][i0 + 1 : i1 + 1, j]
            op = arrays["open"][i0 + 1 : i1 + 1, j]
            cl = arrays["close"][i0 + 1 : i1 + 1, j]
            if not np.isfinite(hi).any():
                continue
            finite_hi = hi[np.isfinite(hi)]
            finite_lo = lo[np.isfinite(lo)]
            peak_at = int(np.nanargmax(hi)) + 1 if np.isfinite(hi).any() else 0
            row = {
                "session": path.name[:10],
                "entry_minute": entry_label,
                "side": "call" if want_call else "put",
                "correct": want_call == up,
                "entry_premium": entry * CONTRACT_MULTIPLIER,
                "mfe": float(finite_hi.max()) / entry - 1.0,
                "mae": float(finite_lo.min()) / entry - 1.0 if finite_lo.size else np.nan,
                "minutes_to_peak": peak_at,
            }
            for name, rule in EXIT_RULES.items():
                exit_price = simulate_exit(hi, lo, op, cl, entry, rule)
                row[f"net_{name}"] = (
                    exit_price - entry
                ) * CONTRACT_MULTIPLIER - COST_PER_LEG_USD
            # The ceiling on any exit model at any speed: sell at the best price
            # the contract ever showed. Unattainable by construction — it needs
            # perfect foresight — but if even this does not clear the cost by a
            # wide margin then no exit model can, however fast it runs.
            row["net_oracle_peak"] = (
                float(finite_hi.max()) - entry
            ) * CONTRACT_MULTIPLIER - COST_PER_LEG_USD
            # The high can be a single aggressive print at the far side of a wide
            # spread, which a seller could only capture by already resting there.
            # The best *close* is the conservative ceiling: a price the contract
            # held for a whole minute.
            finite_cl = cl[np.isfinite(cl)]
            row["net_oracle_close"] = (
                (float(finite_cl.max()) - entry) * CONTRACT_MULTIPLIER - COST_PER_LEG_USD
                if finite_cl.size
                else np.nan
            )
            # And the same foresight restricted to the first five minutes, which
            # is the horizon a one-second exit model would actually be working on.
            early = hi[:5]
            early = early[np.isfinite(early)]
            row["net_oracle_first5"] = (
                (float(early.max()) - entry) * CONTRACT_MULTIPLIER - COST_PER_LEG_USD
                if early.size
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def summarise(table: pd.DataFrame) -> dict:
    ok = table["correct"].to_numpy(bool)
    out = {
        "trades": int(len(table)),
        "sessions": int(table["session"].nunique()),
        "mean_entry_premium_usd": round(float(table["entry_premium"].mean()), 2),
        "excursions": {
            "mean_mfe_pct": round(100 * float(table["mfe"].mean()), 2),
            "median_mfe_pct": round(100 * float(table["mfe"].median()), 2),
            "mean_mae_pct": round(100 * float(table["mae"].mean()), 2),
            "median_minutes_to_peak": float(table["minutes_to_peak"].median()),
            "share_reaching_plus_20pct": round(float((table["mfe"] >= 0.20).mean()), 4),
            "share_reaching_plus_50pct": round(float((table["mfe"] >= 0.50).mean()), 4),
            "share_reaching_plus_100pct": round(float((table["mfe"] >= 1.00).mean()), 4),
        },
        "by_exit_rule": {},
    }
    for name in [*EXIT_RULES, "oracle_first5", "oracle_close", "oracle_peak"]:
        net = table[f"net_{name}"].to_numpy(float)
        keep = np.isfinite(net)
        net, ok_ = net[keep], ok[keep]
        win, loss = float(net[ok_].mean()), float(-net[~ok_].mean())
        out["by_exit_rule"][name] = {
            "mean_net_usd": round(float(net.mean()), 2),
            "mean_net_when_correct_usd": round(win, 2),
            "mean_net_when_wrong_usd": round(-loss, 2),
            "breakeven_accuracy": round(loss / (win + loss), 6) if win > 0 else None,
            "share_of_trades_profitable": round(float((net > 0).mean()), 4),
        }
    return out


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
        got = session_trades(path)
        if got is not None:
            parts.append(got)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    table = pd.concat(parts, ignore_index=True)

    overall = summarise(table)
    by_hour = {
        hour: summarise(part)
        for hour, part in table.groupby(table["entry_minute"].str[:2])
        if len(part) > 2_000
    }

    payload = {
        "schema_version": "v5.scalp-exits.v1",
        "question": (
            "Every prior measurement sold on a clock. What is a contract worth "
            "to someone who exits on the move instead?"
        ),
        "entry_every_minutes": ENTRY_EVERY_MINUTES,
        "window_minutes": WINDOW_MINUTES,
        "cost_per_leg_usd": COST_PER_LEG_USD,
        "exit_rules": {k: list(v) for k, v in EXIT_RULES.items()},
        "oracle_rules_are_not_policies": (
            "oracle_peak sells at the best price the contract ever showed, "
            "oracle_close at the best price it held for a whole minute, and "
            "oracle_first5 at the best of the first five minutes. All three "
            "require perfect foresight and none is attainable. They are the "
            "ceiling any exit model is competing against, and oracle_close is "
            "the one to quote, since a single print at the high may be "
            "uncapturable."
        ),
        "simulation_rules": [
            "a stop is checked before a target inside the same minute",
            "a target fills at the target, not at the high",
            "a stop fills at the stop, or at the open when the bar gapped through it",
            "every level is declared, none is fitted",
        ],
        "corpus": str(args.corpus),
        "overall": overall,
        "by_entry_hour": by_hour,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    e = overall["excursions"]
    print(
        f"\n{overall['trades']:,} trades over {overall['sessions']} sessions, "
        f"mean premium ${overall['mean_entry_premium_usd']:,.0f}, "
        f"{WINDOW_MINUTES}-minute window\n"
    )
    print(
        f"  best price reached: mean {e['mean_mfe_pct']:+.1f}%, median "
        f"{e['median_mfe_pct']:+.1f}%; worst {e['mean_mae_pct']:+.1f}%; "
        f"median {e['median_minutes_to_peak']:.0f} min to the peak"
    )
    print(
        f"  reached +20%: {100 * e['share_reaching_plus_20pct']:.1f}%   "
        f"+50%: {100 * e['share_reaching_plus_50pct']:.1f}%   "
        f"+100%: {100 * e['share_reaching_plus_100pct']:.1f}%\n"
    )
    head = (
        f"  {'exit rule':>16} {'net/trade':>10} {'when right':>11} {'when wrong':>11} "
        f"{'break-even':>11} {'% winners':>10}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, row in overall["by_exit_rule"].items():
        be = row["breakeven_accuracy"]
        print(
            f"  {name:>16} {row['mean_net_usd']:>10,.1f} "
            f"{row['mean_net_when_correct_usd']:>11,.1f} "
            f"{row['mean_net_when_wrong_usd']:>11,.1f} "
            f"{(f'{100 * be:.2f}%' if be else 'unwinnable'):>11} "
            f"{100 * row['share_of_trades_profitable']:>9.1f}%"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
