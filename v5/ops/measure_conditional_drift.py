"""Is there any observable state in which long 0DTE premium drifts up?

Every screen this project has run tested a **rule**: pick a signal, act on it,
score the profit and loss. Rules are tested one at a time and each negative
closes only itself, which is why six campaigns have closed and the space still
feels open.

This module asks the question one level up. A trading rule is a *function of
observable state*. If the expected gross payoff of holding a long near-ATM
option is non-positive in **every** observable state, then no function of those
states can have positive expected gross either — entry rule, exit rule, or the
two combined. One pass closes the whole rule space; forty do not.

That is the same argument that caps the optimal-stopping exit at zero. For any
two stopping times `t_in <= t_out` on a supermartingale, `E[P_out - P_in] <= 0`.
A causal entry is a stopping time too, so the cap applies to entries exactly as
it applies to exits. The only escape is a state in which the price process is
*not* a supermartingale. This measures whether such a state exists.

**What is measured.** For a declared grid of causal states, the mean gross
payoff of a long call, a long put, and the straddle, over a fixed hold. No exit
model, no entry model, no selection: a fixed-horizon hold on every slot, so
nothing downstream can attenuate or flatter the number.

**Two answers come out, and they are different answers.**

* The **census**: every declared cell with its session-block interval, before
  and after a Bonferroni correction for the whole grid. Does any cell's lower
  bound clear zero?
* The **dispersion test**: is the spread of drift *across* the cells of a state
  larger than chance produces? This is the properly-powered question, because a
  state can carry information without any single cell clearing a profit bar.
  A negative census with a negative dispersion test says the observable carries
  no information at all. A negative census with a *positive* dispersion test
  says there is information here that this sample cannot monetise — a different
  finding with a different remedy.

**Causality.** Cut points come from strictly prior sessions on an expanding
window, never from the whole sample, so every cell assignment is one the bot
could have made at the decision minute. Clock and calendar states use fixed
declared buckets. The grid is hashed before it runs.

No rule, model, threshold search or selection appears here. It is a measurement
of the instrument, in the same class as `measure_variance_premium`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.research.statistics import bonferroni_quantile, normal_quantile

BOOTSTRAP_DRAWS = 2_000
PERMUTATION_DRAWS = 1_000
SEED = 20260813
# Cut points are read from prior sessions only; the first sessions have no prior
# window and are dropped by the rule, never by their outcome.
WARMUP_SESSIONS = 60
# ...and from a *trailing* window rather than everything seen so far. The corpus
# spans a 7.08x range of 60-minute move dispersion across its years, so a window
# that never forgets would still be cutting 2026 slots on mostly-2022 quantiles:
# late slots would pile into the top cells and a "cell" would quietly become a
# proxy for the calendar. One year of trailing sessions re-adapts within a year
# of any regime shift and keeps the cells populated throughout.
TRAILING_SESSIONS = 250
MIN_CELL_SLOTS = 200
QUARTILES = (0.25, 0.50, 0.75)

SIDES = ("call", "put", "straddle")

# Every state below is a function of the parity spot at or before the entry
# minute, of the option prices quoted at the entry minute, or of the calendar.
# `kind` says how the cell boundaries are drawn and which permutation null the
# dispersion test uses:
#   "expanding" — quartiles from strictly prior sessions; varies within a session
#   "fixed"     — declared absolute buckets; varies within a session
#   "session"   — constant within a session, so the null shuffles whole sessions
DECLARED_STATES: tuple[dict, ...] = (
    {
        "name": "minutes_to_close",
        "kind": "fixed",
        "edges": (60.0, 180.0, 300.0),
        "labels": ("last hour", "1-3h left", "3-5h left", "over 5h left"),
        "why": "the clock; theta and the spread both move with it",
    },
    {
        "name": "realised_range_60m",
        "kind": "expanding",
        "why": "how far the underlying has just travelled — the magnitude axis",
    },
    {
        "name": "implied_share_of_spot",
        "kind": "expanding",
        "why": "what the straddle costs as a share of spot — the price of vol",
    },
    {
        "name": "implied_over_realised",
        "kind": "expanding",
        "why": "the option's price against what movement has actually delivered;"
        " the most direct mispricing observable the corpus supports",
    },
    {
        "name": "range_position",
        "kind": "fixed",
        "edges": (0.25, 0.50, 0.75),
        "labels": ("bottom quarter", "lower middle", "upper middle", "top quarter"),
        "why": "where spot sits inside the session range",
    },
    {
        "name": "signed_move_60m",
        "kind": "expanding",
        "why": "the hour's move, signed — the direction axis",
    },
    {
        "name": "signed_move_15m",
        "kind": "expanding",
        "why": "the last quarter hour, signed — short-horizon momentum",
    },
    {
        "name": "session_range",
        "kind": "expanding",
        "why": "how wide the session has been — the regime axis",
    },
    {
        "name": "day_of_week",
        "kind": "session",
        "why": "the 0DTE expiry calendar; free, and constant within a session",
    },
    {
        "name": "clock_x_richness",
        "kind": "cross",
        "of": ("minutes_to_close", "implied_over_realised"),
        "why": "the two axes most likely to interact: when, and how dear",
    },
    {
        "name": "magnitude_x_direction",
        "kind": "cross",
        "of": ("realised_range_60m", "signed_move_60m"),
        "why": "the two terms that actually move a long option's value",
    },
)


def grid_hash() -> str:
    """Content hash of the declared grid, so the family is fixed before it runs."""

    payload = json.dumps(
        {
            "states": DECLARED_STATES,
            "sides": SIDES,
            "quartiles": QUARTILES,
            "warmup_sessions": WARMUP_SESSIONS,
            "trailing_sessions": TRAILING_SESSIONS,
            "min_cell_slots": MIN_CELL_SLOTS,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def derive_state_columns(table: pd.DataFrame) -> pd.DataFrame:
    """The raw value behind each declared state, all from entry-minute inputs."""

    spot = table["spot"].to_numpy(float)
    out = table.copy()
    out["realised_range_60m"] = table["range_60m"].to_numpy(float) / spot
    out["implied_share_of_spot"] = table["straddle_premium"].to_numpy(float) / (
        spot * 100.0
    )
    # Both terms are shares of spot, so the ratio is scale free. A session whose
    # prior hour never moved cannot form the ratio and is left out of this one
    # state only; it keeps its place in every other state.
    realised = out["realised_range_60m"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["implied_over_realised"] = np.where(
            realised > 0.0, out["implied_share_of_spot"].to_numpy(float) / realised, np.nan
        )
    out["signed_move_60m"] = table["move_60m"].to_numpy(float) / spot
    out["signed_move_15m"] = table["move_15m"].to_numpy(float) / spot
    out["session_range"] = table["session_range"].to_numpy(float) / spot
    out["day_of_week"] = pd.to_datetime(table["session"]).dt.dayofweek.to_numpy()
    return out


def expanding_cells(
    values: np.ndarray, sessions: np.ndarray, order: list[str]
) -> np.ndarray:
    """Quartile cell per slot, cut on strictly prior sessions only.

    Returns -1 where no assignment is possible: inside the warm-up, or where the
    value itself is missing. Both are refusals to assign, not outcome filters.
    """

    cells = np.full(len(values), -1, dtype=int)
    rank = {s: i for i, s in enumerate(order)}
    position = np.array([rank[s] for s in sessions])
    seen: list[np.ndarray] = []
    for i, session in enumerate(order):
        rows = np.flatnonzero(position == i)
        if i >= WARMUP_SESSIONS and seen:
            window = seen[-TRAILING_SESSIONS:]
            prior = np.concatenate(window)
            prior = prior[np.isfinite(prior)]
            if prior.size >= MIN_CELL_SLOTS:
                edges = np.quantile(prior, QUARTILES)
                here = values[rows]
                assigned = np.searchsorted(edges, here, side="right")
                cells[rows] = np.where(np.isfinite(here), assigned, -1)
        seen.append(values[rows])
    return cells


def fixed_cells(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    assigned = np.searchsorted(np.asarray(edges, float), values, side="right")
    return np.where(np.isfinite(values), assigned, -1)


def build_cells(table: pd.DataFrame, order: list[str]) -> dict[str, np.ndarray]:
    """Cell index per slot for every declared state. -1 means unassigned."""

    sessions = table["session"].to_numpy()
    cells: dict[str, np.ndarray] = {}
    for spec in DECLARED_STATES:
        name = spec["name"]
        if spec["kind"] == "expanding":
            cells[name] = expanding_cells(
                table[name].to_numpy(float), sessions, order
            )
        elif spec["kind"] == "fixed":
            cells[name] = fixed_cells(table[name].to_numpy(float), spec["edges"])
        elif spec["kind"] == "session":
            raw = table[name].to_numpy()
            cells[name] = raw.astype(int)
        elif spec["kind"] == "cross":
            first, second = (cells[n] for n in spec["of"])
            width = int(max(second.max(), 0)) + 1
            both = first * width + second
            cells[name] = np.where((first < 0) | (second < 0), -1, both)
    return cells


def cell_labels(spec: dict, index: int, cells: dict[str, np.ndarray]) -> str:
    if spec["kind"] == "fixed":
        return spec["labels"][index]
    if spec["kind"] == "session":
        return ("Mon", "Tue", "Wed", "Thu", "Fri")[index]
    if spec["kind"] == "expanding":
        return ("Q1 lowest", "Q2", "Q3", "Q4 highest")[index]
    outer, inner = (next(s for s in DECLARED_STATES if s["name"] == n) for n in spec["of"])
    width = int(max(cells[spec["of"][1]].max(), 0)) + 1
    return (
        f"{cell_labels(outer, index // width, cells)}"
        f" / {cell_labels(inner, index % width, cells)}"
    )


def session_totals(
    values: np.ndarray, session_index: np.ndarray, n_sessions: int, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-session sum and count, which is all a session-block bootstrap needs."""

    sums = np.bincount(
        session_index[mask], weights=values[mask], minlength=n_sessions
    )
    counts = np.bincount(session_index[mask], minlength=n_sessions).astype(float)
    return sums, counts


def bootstrap_matrix(rng: np.random.Generator, n_sessions: int) -> np.ndarray:
    """One shared session resample for every cell, so cells stay comparable."""

    return rng.multinomial(
        n_sessions, np.full(n_sessions, 1.0 / n_sessions), size=BOOTSTRAP_DRAWS
    ).astype(float)


def dispersion(
    values: np.ndarray, cells: np.ndarray, n_cells: int
) -> float:
    """Slot-weighted variance of the cell means — how much the state separates."""

    # Non-finite payoffs must be dropped, not summed. `swapped_sessions` leaves
    # a gap wherever the donor session was shorter, and a single NaN weight
    # turns a cell sum into NaN, the statistic into NaN, and `null >= observed`
    # into False — which silently counts a failed draw as evidence *for* the
    # state. Unmasked, that pinned six states at p=0.000.
    keep = (cells >= 0) & np.isfinite(values)
    if not keep.any():
        return 0.0
    counts = np.bincount(cells[keep], minlength=n_cells).astype(float)
    sums = np.bincount(cells[keep], weights=values[keep], minlength=n_cells)
    live = counts >= MIN_CELL_SLOTS
    if live.sum() < 2:
        return 0.0
    means = sums[live] / counts[live]
    weights = counts[live] / counts[live].sum()
    return float(np.sum(weights * (means - np.sum(weights * means)) ** 2))


def permuted_values(
    values: np.ndarray, keys: np.ndarray, session_index: np.ndarray
) -> np.ndarray:
    """Shuffle payoffs inside each session.

    This preserves session identity, slot counts and each session's own payoff
    distribution, and destroys only the link between a slot's within-session
    state and what it paid. It is the null for "does this observable separate
    payoffs", not for "is the mean negative".

    **It has no power over a state that barely moves within a session.**
    `session_range` only ever grows through a session and `signed_move_60m` is
    strongly persistent, so nearly every slot of a session shares a cell, the
    shuffle cannot move a slot to a different cell, and the null collapses onto
    the observed value — reporting `p = 1.000` for "no information" when it has
    actually measured nothing. Those states need `swapped_sessions` instead.
    """

    order = np.lexsort((keys, session_index))
    return values[order]


def session_layout(session_index: np.ndarray, n_sessions: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column address of every slot in a sessions x slots rectangle."""

    within = np.zeros(len(session_index), dtype=int)
    seen = np.zeros(n_sessions, dtype=int)
    for i, s in enumerate(session_index):
        within[i] = seen[s]
        seen[s] += 1
    return session_index, within


def swapped_sessions(
    values: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    n_sessions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Give each session another session's payoffs, slot position for slot position.

    The complement of `permuted_values`. Within-session shape is preserved on
    both sides — the states keep their own within-session pattern and the
    payoffs keep theirs — and only the pairing *between* sessions is broken. So
    it has full power over a state that varies session to session, and none over
    a state that is a pure function of slot position, which `minutes_to_close`
    is. Between them the two nulls cover every declared state.
    """

    width = int(columns.max()) + 1
    grid = np.full((n_sessions, width), np.nan)
    grid[rows, columns] = values
    donor = grid[rng.permutation(n_sessions)]
    return donor[rows, columns]


def within_session_share(values: np.ndarray, session_index: np.ndarray) -> float:
    """How much of a state's variance lives inside sessions rather than across.

    Decides which null speaks for the state. It is a property of the observable
    alone, so it is fixed before any payoff is read.
    """

    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0
    v, s = values[finite], session_index[finite]
    counts = np.bincount(s).astype(float)
    sums = np.bincount(s, weights=v)
    live = counts > 0
    means = np.zeros_like(counts)
    means[live] = sums[live] / counts[live]
    total = float(np.var(v))
    if total <= 0.0:
        return 0.0
    between = float(np.sum(counts[live] * (means[live] - v.mean()) ** 2) / len(v))
    return float(max(0.0, min(1.0, 1.0 - between / total)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw = derive_state_columns(pd.read_parquet(args.table))
    rng = np.random.default_rng(SEED)

    census: list[dict] = []
    tests: list[dict] = []
    for hold, part in raw.groupby("hold"):
        part = part.sort_values(["session", "entry_minute"]).reset_index(drop=True)
        order = sorted(part["session"].unique())
        rank = {s: i for i, s in enumerate(order)}
        session_index = part["session"].map(rank).to_numpy()
        n_sessions = len(order)
        cells = build_cells(part, order)
        counts_matrix = bootstrap_matrix(rng, n_sessions)
        rows, columns = session_layout(session_index, n_sessions)
        # Where each state's variance lives, computed from the state alone and
        # before any payoff is touched, so the choice of null cannot be steered
        # by the answer it produces.
        within_shares = {
            spec["name"]: (
                within_session_share(
                    cells[spec["name"]].astype(float), session_index
                )
            )
            for spec in DECLARED_STATES
        }
        # A cell that lives mostly in one year is a calendar bucket wearing a
        # state's name, and its drift would say more about that year than about
        # the observable. Recorded per cell so a reader can see it.
        years = part["session"].str[:4].to_numpy()

        for side in SIDES:
            gross = part[f"{side}_gross"].to_numpy(float)
            premium = (
                part["straddle_premium"].to_numpy(float)
                if side == "straddle"
                else part[f"{side}_premium"].to_numpy(float)
            )
            share = gross / premium

            for spec in DECLARED_STATES:
                name = spec["name"]
                assigned = cells[name]
                n_cells = int(assigned.max()) + 1 if assigned.max() >= 0 else 0
                sums, counts, keep_cells = [], [], []
                for c in range(n_cells):
                    mask = assigned == c
                    if mask.sum() < MIN_CELL_SLOTS:
                        continue
                    s, n = session_totals(gross, session_index, n_sessions, mask)
                    sums.append(s)
                    counts.append(n)
                    keep_cells.append(c)
                if not keep_cells:
                    continue
                sum_matrix = np.vstack(sums)
                count_matrix = np.vstack(counts)
                # mean over a resample = (drawn session sums) / (drawn counts)
                draws = (counts_matrix @ sum_matrix.T) / (counts_matrix @ count_matrix.T)

                for j, c in enumerate(keep_cells):
                    mask = assigned == c
                    column = draws[:, j]
                    lo, hi = np.quantile(column, [0.025, 0.975])
                    spread = np.std(column, ddof=1)
                    census.append(
                        {
                            "hold_minutes": int(hold),
                            "side": side,
                            "state": name,
                            "cell": cell_labels(spec, c, cells),
                            "slots": int(mask.sum()),
                            "sessions": int(np.unique(session_index[mask]).size),
                            "mean_premium_usd": round(float(premium[mask].mean()), 2),
                            "share_in_busiest_year": round(
                                float(
                                    pd.Series(years[mask]).value_counts(normalize=True).iloc[0]
                                ),
                                3,
                            ),
                            "mean_gross_usd": round(float(gross[mask].mean()), 3),
                            "gross_share_of_premium": round(
                                float(share[mask].mean()), 6
                            ),
                            "ci95_usd": [round(float(lo), 2), round(float(hi), 2)],
                            "clears_zero": bool(lo > 0.0),
                            "bootstrap_sd_usd": round(float(spread), 3),
                        }
                    )

                # Does the state separate payoffs at all? Two permutation nulls,
                # because neither one alone has power over every state.
                observed = dispersion(share, assigned, n_cells)
                block = (
                    np.zeros(len(part), dtype=int)
                    if spec["kind"] == "session"
                    else session_index
                )
                within = np.empty(PERMUTATION_DRAWS)
                across = np.empty(PERMUTATION_DRAWS)
                for b in range(PERMUTATION_DRAWS):
                    within[b] = dispersion(
                        permuted_values(share, rng.random(len(part)), block),
                        assigned,
                        n_cells,
                    )
                    across[b] = dispersion(
                        swapped_sessions(share, rows, columns, n_sessions, rng),
                        assigned,
                        n_cells,
                    )
                p_within = float((within >= observed).mean())
                p_across = float((across >= observed).mean())
                # Whichever null can actually move this state's cells is the one
                # that speaks for it. Decided from the observable, not the result.
                share_within = within_shares[name]
                primary = "within_session" if share_within > 0.5 else "session_swap"
                p_primary = p_within if primary == "within_session" else p_across
                tests.append(
                    {
                        "hold_minutes": int(hold),
                        "side": side,
                        "state": name,
                        "observed_dispersion": round(observed, 10),
                        # A null with no spread has measured nothing, whatever
                        # p-value it prints. Recorded so that is checkable.
                        "null_sd_within_session": round(float(np.std(within)), 12),
                        "null_sd_session_swap": round(float(np.std(across)), 12),
                        "null_p95_within_session": round(float(np.quantile(within, 0.95)), 12),
                        "null_p95_session_swap": round(float(np.quantile(across, 0.95)), 12),
                        "within_session_variance_share": round(share_within, 4),
                        "primary_null": primary,
                        "p_within_session": round(p_within, 4),
                        "p_session_swap": round(p_across, 4),
                        "p_value": round(p_primary, 4),
                        "separates_payoffs": bool(p_primary < 0.05),
                    }
                )

    family = len(census)
    z_plain, z_corrected = normal_quantile(0.975), bonferroni_quantile(family)
    for row in census:
        half = z_corrected * row["bootstrap_sd_usd"]
        row["bonferroni_ci_usd"] = [
            round(row["mean_gross_usd"] - half, 2),
            round(row["mean_gross_usd"] + half, 2),
        ]
        row["clears_zero_corrected"] = bool(row["mean_gross_usd"] - half > 0.0)

    clearing = [r for r in census if r["clears_zero"]]
    corrected = [r for r in census if r["clears_zero_corrected"]]
    separating = [t for t in tests if t["separates_payoffs"]]

    payload = {
        "schema_version": "v5.conditional-drift.v1",
        "question": (
            "Is there any observable state, knowable at the decision minute, in "
            "which the expected gross payoff of a long near-ATM SPXW 0DTE option "
            "is positive? A trading rule is a function of state, so a state with "
            "no positive cell admits no profitable rule built on it."
        ),
        "grid_sha256": grid_hash(),
        "declared_states": list(DECLARED_STATES),
        "family_size": family,
        "bonferroni_z": round(z_corrected, 4),
        "plain_z": round(z_plain, 4),
        "cells_clearing_zero_uncorrected": len(clearing),
        "cells_clearing_zero_corrected": len(corrected),
        "states_separating_payoffs": len(separating),
        "states_tested": len(tests),
        "computes_no_policy": True,
        "no_model_fitted": True,
        "cut_points": "expanding window, strictly prior sessions only",
        "source_table": str(args.table),
        "census": census,
        "dispersion_tests": tests,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"grid {payload['grid_sha256'][:12]}  {family} cells  z={z_corrected:.2f}")
    for hold in sorted({r["hold_minutes"] for r in census}):
        print(f"\n{hold}-minute hold")
        head = (
            f"  {'state':>22} {'cell':>28} {'side':>9} {'slots':>7} "
            f"{'gross$':>9} {'as %':>8} {'ci95 lo':>9} {'ci95 hi':>9}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for row in census:
            if row["hold_minutes"] != hold or row["side"] == "straddle":
                continue
            print(
                f"  {row['state']:>22} {row['cell']:>28} {row['side']:>9} "
                f"{row['slots']:>7,} {row['mean_gross_usd']:>9,.1f} "
                f"{100 * row['gross_share_of_premium']:>7.2f}% "
                f"{row['ci95_usd'][0]:>9,.1f} {row['ci95_usd'][1]:>9,.1f}"
            )

    print("\ndoes any observable separate payoffs at all?")
    print(
        f"  {'state':>22} {'within-var':>11} {'null':>15} "
        + " ".join(f"{h}m/{s[:4]:<5}" for h in (15, 60) for s in SIDES)
    )
    for spec in DECLARED_STATES:
        hits = [t for t in tests if t["state"] == spec["name"]]
        if not hits:
            continue
        line = (
            f"  {spec['name']:>22} {hits[0]['within_session_variance_share']:>11.3f} "
            f"{hits[0]['primary_null']:>15} "
        )
        for hold in (15, 60):
            for side in SIDES:
                one = [t for t in hits if t["hold_minutes"] == hold and t["side"] == side]
                mark = "*" if one and one[0]["separates_payoffs"] else " "
                line += f"{one[0]['p_value']:>8.3f}{mark}" if one else f"{'-':>9}"
        print(line)
    print("  * separates payoffs at p<0.05 under the null that has power over it")

    print(
        f"\ncells clearing zero: {len(clearing)}/{family} uncorrected, "
        f"{len(corrected)}/{family} after Bonferroni"
    )
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
