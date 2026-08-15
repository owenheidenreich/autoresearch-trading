# What the cost bars are actually made of

**Dated 2026-08-12. Phase 0.1 of the [project plan](../../STATUS.md). No data was purchased and no vendor
was contacted; every figure below was recomputed from receipts already on disk.**

**One sentence for the owner: the ES cost bar is sound and mostly measured, but three documents described
it wrongly, and the option cost used by the validation packet is 8.6x smaller than the option cost the
project actually measured.**

## Why this was done first

Every negative result this project has produced was judged against these two numbers. Before spending
anything on more data, they needed auditing — a wrong bar invalidates the comparison in either direction.
It also settles the `slippage_accounting` divergence axis, whose exit condition is precisely this
decomposition.

## The ES bar: 0.358 points / $17.92

| Component | Points | USD | Status |
|---|---:|---:|---|
| Spread crossing, 1.0734 ticks x $12.50 | 0.26835 | $13.4175 | **MEASURED** — time-weighted over 1,095,818 quote states, 20 sessions, elevated-volatility quartile |
| Commission, round turn | 0.09000 | $4.5000 | **ASSUMED** — a hard-coded constant from a published retail fee schedule |
| Slippage | 0.00000 | $0.0000 | **ABSENT** — no slippage term was ever computed |
| **Total** | **0.35835** | **$17.9175** | |

Source: `v4/scripts/measure_pathd_es_bbo1s_spread.py:29-31` (`COMMISSIONS_DOLLARS = 4.50`) and
`v4/audit/autoresearch/pathd_es_spread_measurement_2026_08_04/spread_measurement_result.json`, whose
`friction_formula` field states it verbatim.

### The assumption is real but the bar is robust

There is a precedent for taking this seriously. The **option** fee was also once an assumed figure —
$0.65/side — and when it was finally measured from a real IBKR paper fill it came back at **$1.54/side**,
wrong by **2.4x**, after propagating into four documents
(`v4/audit/autoresearch/pathd_phase0b_trackc_paper_transitions_2026_08_04/trackc_transition_evidence.json`).
The ES $4.50 has exactly the status the $0.65 had: plausible, published, never checked against a fill.

Applying that same 2.4x error in both directions:

| Commission per round turn | Total bar | vs frozen 0.358 |
|---|---:|---:|
| $1.88 (2.4x low) | 0.30585 pts | **−14.7%** |
| $4.50 (frozen) | 0.35835 pts | — |
| $10.80 (2.4x high) | 0.48435 pts | **+35.2%** |

**The bar is robust because 75% of it is measured.** Even the worst precedent moves it by about a third,
which changes no verdict this project has reached — G1's floor was 22–88x the bar, and job 15's required
edge was set by measurement, not friction. **Phase 1 is not at risk from this**, and re-measuring the
commission is worth doing but is not a blocker.

### The missing slippage term is defensible, and should be stated rather than repaired

The formula charges exactly one full spread crossing per round trip — half at entry, half at exit — at the
*elevated-volatility* spread rather than the unconditional one. For the one-contract ES game this project
declares, that is a reasonable model of an aggressive fill: ES is deep, and queue-position or
market-impact effects on a single lot are small. The omission is a modelling choice, not an error. What was
wrong is describing it as though slippage had been measured.

## The option bars: two different numbers, repeatedly conflated

| Quantity | Value | Contains | Share of the $565 average premium |
|---|---:|---|---:|
| `option_round_trip_fee_dollars` | **$3.08** | fees only | **0.545%** |
| Measured aggressive round trip (ledger row 181) | **$26.48** | fees **and** spread crossing | **4.687%** |

These are 8.6x apart and describe different things. Both are correct; neither is a substitute for the
other. Three consequences follow.

**1. `STATUS.md` §5 mixes them in one table.** It lists "$3.08" as the option round-trip friction and then
"4.68%" as that friction's share of the $565 premium. But 4.68% of $565 is $26.44 — the aggressive figure.
$3.08 is 0.545%. The third row does not derive from the second. Corrected in this change.

**2. The validation packet charges the smaller one by default.**
[`candidate_packet.py`](../validation/candidate_packet.py) applies
`2 x commission_per_side_usd x quantity` with `commission_per_side_usd` defaulting to the measured $1.54,
and a separate `slippage_per_side_points` defaulting to **0.0**. So a packet exported at its defaults
charges **$3.08 per contract and no spread at all**. There is no `spread` parameter; the slippage argument
is the only way to charge spread crossing.

A candidate judged at the default would be credited a cost 8.6x below what the project measured it would
actually pay. That is the single most consequential finding here, because it sits directly on the G5 path.

**Repaired by disclosure rather than refusal.** The packet's `summary.json` now carries a `cost_model`
block naming `components_charged`, `components_omitted`, `excludes_spread_crossing`, and the $26.48
reference. Refusing a zero default outright would break legitimate fee-only diagnostics and would guess at
the caller's intent; stating the model cannot be misread and cannot be forgotten.

**3. The evidence index called the ES bar a "measured spread, fees, and slippage" study.** Corrected.

## The `slippage_accounting` axis is now settled

The axis asked "whether slippage is counted once, twice or not at all." The answer is **not at all,
anywhere** — no frozen cost constant in this project contains a slippage term, and the packet's slippage
input defaults to zero. There was never a double count. There was a documentation error in the opposite
direction: three places claimed a component that does not exist.

Its `settles_when` required the decomposition plus the packet either consuming it or documenting its
argument. Both halves are done, so the axis moves `UNKNOWN` → `MEASURED_DIFFERENT` with a named repair.

## What still needs the owner

**Measure the ES commission**, the same way the option fee was measured: one guarded round trip on the
paper account, reading `avgCost` and `RealizedPnL`. This is Tier 1 (broker contact) and is **not urgent** —
the sensitivity above shows it cannot overturn a verdict. It should be done before any candidate is judged
in ES dollars, not before Phase 1.

## What this finding did not do

Purchase data, contact a vendor or broker, change any frozen value, alter any verdict, or touch reserved
sessions. The only behavioural code change is an additive `cost_model` block in the packet's summary.
