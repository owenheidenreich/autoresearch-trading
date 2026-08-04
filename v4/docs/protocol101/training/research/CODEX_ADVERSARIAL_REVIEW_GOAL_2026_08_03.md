# Codex Adversarial Review Goal — Claude's Phase-1 Close-Out, Fix Research, and Feasibility Study (2026-08-03)

**Your job is to BREAK these findings, not to confirm them.** Claude produced them; Claude has already been
wrong twice in this programme in ways that cost real time (a clock leak signed off as clean, and a
"condition-5 satisfied" claim read off the wrong population). Assume the same standard of error is present
here and go looking for it. **A refutation is the most valuable output you can return.** "Everything checks
out" is the least valuable, and should only be said after genuine attempts to falsify have failed.

Do not read Claude's result files and agree with them. **Recompute independently from the source artifacts.**

---

## The claims under review

| # | Claim | Number | Source |
|---|---|---|---|
| C1 | 0DTE long premium is negative-EV **before any cost** | **−$13.00**/trade gross, median −$90, win 35.7%, negative 5/5 folds | close-out |
| C2 | Round-trip friction dominates the loss | **−$26.48** (67%) of −$39.48; 4.68% of $565.39 premium | fix research |
| C3 | Passive entry recovers 55% of the loss | −$41.32 → **−$18.56**/trade, +$22.76, 83.3% fill, adv.sel −$0.72, stable 5/5 folds | fix research |
| C4 | The entry model has zero ranking power | decile curve flat/non-monotonic: top −$16.37, bottom −$17.75, middle best −$11.48 | fix research |
| C5 | The learned exit model is not a policy | `learned_exit_index==0` in 980/1031; identical to `exit_immediate` in 982/1031 | fix research |
| C6 | Path-D's cadence was mathematically unclearable | **116.2%** required win rate, SPXW 0DTE @1m aggressive | feasibility |
| C7 | Horizon dominates instrument choice | same option @60m needs 57.2% (58.7% drift-adjusted) vs 116.2% @1m | feasibility |
| C8 | ES futures @15–60m is the only reachable cell | **52.1–54.2%** required win rate | feasibility |

Documents: `PATHD_PHASE1_CLOSEOUT_2026_08_03.md`, `PATHD_PHASE1_FIX_RESEARCH_FINDINGS_2026_08_03.md`,
`PATHD_GROSS_EXPECTANCY_FEASIBILITY_STUDY_2026_08_03.md`. Script: `v4/research/feasibility_gross_expectancy.py`.
Commits `d05f0137`, `e4f56942`, `ac485174`, `d551b594`.

---

## Attack surfaces Claude already believes are weak — go here first

**A1 — ES/VX friction is ASSUMED, not measured (C8).** OHLCV carries no quotes. Claude used $17.00 round
trip for ES (1 tick crossing + ~$4.50 commissions) and $55.00 for VX from published tick structure. **This
single unmeasured constant is load-bearing for the only attractive result in the entire study.** Check
whether GLBX quote/MBP data is available within existing entitlements. If ES is wider than 1 tick at the
times a strategy would actually trade, 52.1% degrades — by how much? **Do not purchase data to answer this.**

**A2 — `ES.c.0` is a CONTINUOUS front-month contract (C7, C8).** Claude computed close-to-close moves as
`c[h:] - c[:-h]` **without any roll handling**. A quarterly roll produces a large artificial jump. Quantify
the contamination: identify roll dates in the 261 sessions and recompute median |move| with roll days
excluded. If median |move| is inflated by rolls, the ES hurdle is **understated** and C8 weakens.

**A3 — the fill model may be the wrong event entirely (C3).** Claude counts a passive buy limit at `L` as
filled when `ask[k] <= L`, i.e. the whole offer trades down to the limit. But a resting buy limit actually
fills when a **seller crosses down and hits it** — which is a different event, and `ask <= L` may be
capturing "the market already moved through me" rather than "I got filled." Is this condition too strict,
too loose, or measuring the wrong thing? There is also **no queue priority** — an 83.3% fill rate inside
60 s assumes favourable position. Construct a defensible alternative fill model and report how much of the
+$22.76 survives.

**A4 — the symmetric-move metric is a first-order screen (C6, C7, C8).**
`required_win = 50% + F/(2·M)` assumes a two-outcome ±M payoff. Real distributions are fat-tailed and
**options are convex by construction**, so a symmetric model may systematically misprice option cells
relative to futures cells. Claude used **median** |move| for M; **mean** |move| would give different (and
for fat tails, materially different) answers. Recompute with mean and with a distribution-aware breakeven.
Does the ES-vs-options ranking survive, or is it an artefact of the estimator?

**A5 — the +$22.76 was extrapolated across populations (C4).** It was measured on **878** control
trajectories, then applied as a **uniform constant shift** to all **156,950** candidates for the decile
analysis. Those are different populations with different spread distributions. Does C4 hold if the shift
is estimated per-candidate from its own quoted spread instead of applied as a constant?

**A6 — option move sizes are conditioned on selection (C6, C7).** M for options was measured from
**entered** trajectories (`option_bid` paths), not from all options at all times. Entered contracts may
systematically differ in spread and volatility. Also, windows are taken only where fully inside the
trajectory, which **biases toward earlier-in-day windows** and excludes late-day 60-minute holds that would
hit forced-flat. Quantify both biases.

**A7 — bid-path vs mid-path (C1, C6).** Option moves were measured on `option_bid`, and C1's "gross"
is bid-to-bid. Bid-to-bid understates or overstates true economic move relative to mid-to-mid depending on
spread dynamics. Does C1's −$13.00 survive a mid-to-mid measurement?

**A8 — overlapping windows inflate n (C6–C8).** Rolling `h`-minute differences are heavily autocorrelated;
reported `n` is not an independent sample count. This mostly affects any confidence statement, not the
medians — confirm that, or show where it bites.

---

## Attack surfaces Claude has NOT flagged — find your own

Do not limit yourself to A1–A8. Specifically consider: whether `DETERMINISTIC_CONTROL_OOF` entries are
genuinely unbiased for execution measurement or systematically selected; whether the 5/5 fold stability of
C3 is real or an artefact of a mechanical effect that cannot vary; whether C5's 95% immediate-exit rate
indicates a **training or serialization defect** rather than a learned behaviour (this matters — if the
exit model is broken rather than degenerate, several downstream numbers change); and whether the
close-out's "no subpopulation is positive" survives slicing Claude did not try.

---

## Method requirements

- **Recompute from source artifacts** under `/Volumes/AR_TRADING_DATA/`. Do not import Claude's result
  JSON/CSV and re-summarise it.
- Where you reproduce a number, state it to the same precision and say **UPHELD / WEAKENED / REFUTED**.
- Where you refute, give the corrected number and the minimal evidence that establishes it.
- Distinguish **"the number is wrong"** from **"the number is right but the conclusion doesn't follow."**
  Both are valuable; they are different findings.

## Hard constraints

- **Protected 36-session firewall is SPENT — never reopen it.** `holdout_open_count` must remain 0.
- **Causal clock is t−60s.** Never source SPX context from the bar stamped `t`.
- **Do not modify** `FILL_LAW`, the causal clock, the label law, or the OOF firewall. Any counterfactual
  fill model must be separate and explicitly labelled.
- **No paid Databento/Polygon downloads**, no broker/order/live/paper-submit, no training, no promotion,
  no launchd/plist/runtime-flag edits, without explicit owner authorization (CLAUDE.md hard stops).
- **No reward-hacking in reverse either** — do not manufacture a refutation. If a claim survives a genuine
  attack, say so plainly.

## Deliverable

A written adversarial review:

1. A verdict table: C1–C8, each **UPHELD / WEAKENED / REFUTED**, with your independently computed number.
2. For every WEAKENED/REFUTED: the corrected figure and what conclusion changes as a result.
3. Findings of your own that Claude did not anticipate.
4. A single bottom-line: **does the recommendation still stand?** Claude's recommendation is *do not pivot
   to ES yet; first test 0DTE options at 30–60 minute holds with passive entry on the corpus already owned,
   and separately confirm the ES friction assumption.* Say whether your review supports, modifies, or
   overturns that.
5. Anything you could not check, and what it would take.

End with `STOP_FOR_CLAUDE_VERIFICATION`.

*Prepared by Claude Opus 5 — 2026-08-03. Adversarial review requested in both directions per the standing
division of labor.*
