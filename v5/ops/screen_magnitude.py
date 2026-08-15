"""Can a causal signal find the hours where the move outruns the option price?

The directional screen closed with a mechanism rather than just a negative: at
sixty minutes the median hour moves 6.79 points and loses $39, while the top
eighth of hours by movement breaks even at 35.53% accuracy and pays $453. If a
signal could identify those hours in advance, direction would barely matter.

**The trap this module is built around.** Realised volatility clusters, so
predicting that the next hour will be busy is easy and nearly worthless — the
option price already knows. The only question worth asking is whether the move
beats **what the option charged for it**, so every rule here is scored on the
profit and loss of an actual position, never on whether the range was large.

Scoring uses the **side-averaged leg**: the mean of the near-ATM call and the
near-ATM put, each charged its own round trip. That is exactly half the straddle
and exactly what a one-contract policy earns if its direction call is a coin
flip, so a positive number means the window pays **without any directional skill
at all**. It is the cleanest possible statement of a magnitude edge.

**Thresholds are calibrated on prior sessions only.** A full-sample tercile is
look-ahead, which is the defect that contaminated this project's earlier work.
Each session's thresholds come from an expanding window over the sessions before
it, and a warm-up period is never traded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from v5.research import statistics as st

# --- the declaration ---------------------------------------------------------

COST_PER_LEG_USD = 25.0
# Sessions used only to calibrate thresholds, never traded.
WARMUP_SESSIONS = 150
# Quantiles the expanding calibration tracks.
CALIBRATION_QUANTILES = (0.33, 0.5, 0.67)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260813

# Features a rule may read. Each is causal at the entry minute and each is
# compared against its own expanding-window quantile rather than a constant.
CALIBRATED_FEATURES = (
    "range_30m",
    "range_60m",
    "expansion",
    "cheapness",
    "straddle_share_of_spot",
)


def derived_features(table: pd.DataFrame) -> pd.DataFrame:
    """Ratios that make the raw ranges comparable across sessions and eras."""

    out = table.copy()
    spot = out["spot"].to_numpy(float)
    # How much the last fifteen minutes moved relative to the last hour: above
    # 0.25 means activity is picking up.
    out["expansion"] = out["range_15m"] / out["range_60m"].replace(0.0, np.nan)
    # What the option costs, as a share of the underlying.
    out["straddle_share_of_spot"] = out["straddle_premium"] / (spot * 100.0)
    # Recent realised movement per dollar of option cost. High means the option
    # looks cheap against what the underlying has actually been doing.
    out["cheapness"] = out["range_30m"] / (
        out["straddle_premium"].replace(0.0, np.nan) / 100.0
    )
    return out


def rule_high_recent_range(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """Volatility clustering, taken at face value: buy when it has been busy."""

    return (f["range_30m"] > q["range_30m_0.67"]).to_numpy()


def rule_low_recent_range(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """The opposite: buy the quiet, betting on mean reversion in volatility."""

    return (f["range_30m"] < q["range_30m_0.33"]).to_numpy()


def rule_expansion(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """Buy when the last fifteen minutes are busy relative to the last hour."""

    return (f["expansion"] > q["expansion_0.67"]).to_numpy()


def rule_cheap_vs_recent(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """Buy when the option is cheap against recent realised movement.

    This is the one rule that compares the forecast with the price rather than
    reading either alone, so it is the only member that could in principle
    survive the fact that the market also knows volatility clusters.
    """

    return (f["cheapness"] > q["cheapness_0.67"]).to_numpy()


def rule_dear_vs_recent(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """The mirror. As a *long* rule it should lose; it is the control."""

    return (f["cheapness"] < q["cheapness_0.33"]).to_numpy()


def rule_cheap_option(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """Buy when the option is simply cheap relative to the underlying."""

    return (f["straddle_share_of_spot"] < q["straddle_share_of_spot_0.33"]).to_numpy()


def rule_late_session(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """The last ninety minutes, where gamma is largest. A declared clock rule."""

    return (f["minutes_to_close"] <= 90).to_numpy()


def rule_range_edge(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """Buy at the edge of the session range, where a breakout may follow."""

    return ((f["range_position"] >= 0.9) | (f["range_position"] <= 0.1)).to_numpy()


def rule_always(q: Mapping[str, float], f: pd.DataFrame) -> np.ndarray:
    """Trade every slot. The baseline every other member must beat."""

    return np.ones(len(f), dtype=bool)


RULES: Mapping[str, Callable[[Mapping[str, float], pd.DataFrame], np.ndarray]] = {
    "always": rule_always,
    "high_recent_range": rule_high_recent_range,
    "low_recent_range": rule_low_recent_range,
    "expansion": rule_expansion,
    "cheap_vs_recent": rule_cheap_vs_recent,
    "dear_vs_recent": rule_dear_vs_recent,
    "cheap_option": rule_cheap_option,
    "late_session": rule_late_session,
    "range_edge": rule_range_edge,
}

# ``always`` is a baseline, not a hypothesis, so it is not counted in the family.
FAMILY_SIZE = (len(RULES) - 1) * 2  # every rule at each of two holds


def declaration() -> dict:
    return {
        "rules": sorted(RULES),
        "family_size": FAMILY_SIZE,
        "baseline_excluded_from_family": "always",
        "cost_per_leg_usd": COST_PER_LEG_USD,
        "warmup_sessions": WARMUP_SESSIONS,
        "calibration_quantiles": list(CALIBRATION_QUANTILES),
        "calibrated_features": list(CALIBRATED_FEATURES),
        "scoring": (
            "mean net dollars of the side-averaged near-ATM leg, each leg charged "
            "its own round trip. Equals half the straddle, and equals what a "
            "one-contract policy earns with a coin-flip direction call, so a "
            "positive number is a magnitude edge requiring no directional skill."
        ),
        "thresholds": (
            "expanding window over prior sessions only; a session is scored "
            "against quantiles computed from the sessions before it"
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def declaration_hash() -> str:
    return hashlib.sha256(
        json.dumps(declaration(), sort_keys=True).encode()
    ).hexdigest()


# --- causal calibration ------------------------------------------------------


def expanding_thresholds(table: pd.DataFrame) -> pd.DataFrame:
    """Per-session quantiles computed from strictly earlier sessions.

    Shifting by one session is what makes this causal: a session never
    contributes to the thresholds it is judged against.
    """

    # Accumulate values session by session, recording each quantile *before*
    # the session that will be judged against it is added.
    sessions = sorted(table["session"].unique())
    pools = {name: [] for name in CALIBRATED_FEATURES}
    rows = []
    for session in sessions:
        row = {"session": session}
        for name in CALIBRATED_FEATURES:
            pool = pools[name]
            if pool:
                values = np.concatenate(pool)
                values = values[np.isfinite(values)]
            else:
                values = np.array([])
            for q in CALIBRATION_QUANTILES:
                row[f"{name}_{q}"] = (
                    float(np.quantile(values, q)) if values.size >= 200 else np.nan
                )
            row[f"{name}_n"] = int(values.size)
        rows.append(row)
        part = table[table["session"] == session]
        for name in CALIBRATED_FEATURES:
            pools[name].append(part[name].to_numpy(float))
    return pd.DataFrame(rows).set_index("session")


def score(
    table: pd.DataFrame,
    thresholds: pd.DataFrame,
    rule_name: str,
    z_alpha: float,
) -> dict:
    """One member, scored on the side-averaged leg."""

    fired = np.zeros(len(table), dtype=bool)
    for session, part in table.groupby("session", sort=False):
        q = thresholds.loc[session].to_dict()
        if any(
            not np.isfinite(q.get(f"{name}_{quant}", np.nan))
            for name in CALIBRATED_FEATURES
            for quant in CALIBRATION_QUANTILES
        ):
            continue
        fired[table.index.get_indexer(part.index)] = RULES[rule_name](q, part)

    if fired.sum() < 200:
        return {"rule": rule_name, "trades": int(fired.sum()), "verdict": "too few trades"}

    # Side-averaged leg: half the straddle gross, less one round trip.
    net = table["straddle_gross"].to_numpy(float)[fired] / 2.0 - COST_PER_LEG_USD
    sessions = table["session"].to_numpy()[fired]
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = net[np.concatenate([index[s] for s in chosen])].mean()
    lower = float(np.quantile(draws, _tail_level(z_alpha)))

    abs_move = table["abs_move"].to_numpy(float)[fired]
    return {
        "rule": rule_name,
        "trades": int(fired.sum()),
        "share_of_slots": round(float(fired.mean()), 4),
        "sessions": int(len(unique)),
        "mean_abs_move_points": round(float(abs_move.mean()), 3),
        "mean_straddle_premium_usd": round(
            float(table["straddle_premium"].to_numpy(float)[fired].mean()), 2
        ),
        "mean_net_usd": round(float(net.mean()), 2),
        "bootstrap_lower_usd": round(lower, 2),
        "clears_zero": bool(lower > 0.0),
    }


def _tail_level(z_alpha: float) -> float:
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
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw = derived_features(pd.read_parquet(args.table))
    z_family = st.bonferroni_quantile(FAMILY_SIZE)
    results = {}
    for hold, part in raw.groupby("hold"):
        part = part.reset_index(drop=True)
        sessions = sorted(part["session"].unique())
        traded = part[part["session"].isin(sessions[WARMUP_SESSIONS:])].reset_index(
            drop=True
        )
        thresholds = expanding_thresholds(part)
        results[f"{hold}m"] = {
            "slots_after_warmup": int(len(traded)),
            "sessions_after_warmup": int(traded["session"].nunique()),
            "rules": [
                score(traded, thresholds, name, z_family) for name in sorted(RULES)
            ],
        }

    payload = {
        "schema_version": "v5.magnitude-screen.v1",
        "declaration": declaration(),
        "declaration_sha256": declaration_hash(),
        "bonferroni_z": round(z_family, 6),
        "source_table": str(args.table),
        "by_hold": results,
        "what_a_positive_means": (
            "a window that pays without any directional skill. It would still "
            "need a known-answer campaign before anything is believed."
        ),
        "known_limitations": [
            "the underlying is a put/call parity estimate from traded prices",
            "the round trip is carried from the owned quote corpus and charged "
            "per leg",
            "thresholds are expanding-window but the rule set itself was chosen "
            "with the earlier findings in view, which is a multiplicity the "
            "Bonferroni correction does not capture",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\ndeclaration {declaration_hash()[:16]}, family {FAMILY_SIZE}, "
          f"Bonferroni z {z_family:.3f}")
    for hold, block in results.items():
        print(
            f"\n{hold} hold — {block['sessions_after_warmup']} sessions after warm-up, "
            f"{block['slots_after_warmup']:,} slots"
        )
        head = (
            f"  {'rule':>20} {'trades':>8} {'share':>7} {'|move|':>8} "
            f"{'straddle':>9} {'net/trade':>10} {'lower':>9} {'clears 0':>9}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for row in block["rules"]:
            if "mean_net_usd" not in row:
                print(f"  {row['rule']:>20} {row['trades']:>8,}   {row['verdict']}")
                continue
            print(
                f"  {row['rule']:>20} {row['trades']:>8,} "
                f"{100 * row['share_of_slots']:>6.1f}% {row['mean_abs_move_points']:>8.2f} "
                f"{row['mean_straddle_premium_usd']:>9,.0f} {row['mean_net_usd']:>10,.1f} "
                f"{row['bootstrap_lower_usd']:>9,.1f} "
                f"{('YES' if row['clears_zero'] else 'no'):>9}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
