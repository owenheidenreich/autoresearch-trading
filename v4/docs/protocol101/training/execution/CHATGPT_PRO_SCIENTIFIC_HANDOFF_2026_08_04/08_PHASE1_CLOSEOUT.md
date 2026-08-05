# Path-D Phase-1 — Close-Out (2026-08-03)

**Status: `CLOSED — NO_INCREMENTAL_EDGE`.** Stage 2 is not authorized. No further training is recommended
on this strategy class. This document is the terminal record for Phase-1; the roadmap status board points
here.

> ## ⚠ Amended 2026-08-03 after Codex adversarial review (`321b3bbd`)
>
> **The aggregate Phase-1 negative result is UPHELD.** Codex independently reproduced the core economics
> from raw partitions: gross −$13.00, friction −$26.48 (67.06% of the loss, 4.683% of $565.39 premium),
> realized −$39.48. A mid-to-mid recomputation off raw OPRA quotes (156,948/156,950 rows joined) gives
> −$12.12 gross, still negative 5/5 folds. **Three claims below were weakened and must be read with these
> corrections:**
>
> **1. "No subpopulation is positive" is FALSE — withdrawn.** The **10:30–10:59 ET** half-hour has
> **+$5.99 mean GROSS** over 10,063 candidates across 120 sessions, positive in **4/5 folds**
> (+9.8/+7.6/−17.3/+30.8/+6.4). Claude independently re-verified this. It is exploratory and
> multiplicity-exposed (1 of 11 half-hours), and **net is still −$20.08** because friction is ~4× the
> gross drift — so it reinforces the friction conclusion while refuting the universal-closure language.
> Correct statement: *no subpopulation is positive after costs.*
>
> **2. The +$22.76 passive figure is an upper bound, not a point estimate.** It holds only under an
> optimistic offer-touch fill. Codex's stress ladder: touch +$22.76 → two-second persistence +$21.54 →
> touch-plus-one-tick +$20.11 → **one-tick penetration +$12.57** → **two-tick penetration +$2.31**.
> The improvement stays positive in all five folds at one-tick penetration, so the mechanical spread saving
> is real, but **its magnitude is unidentified without queue data.** Quote the range, not the headline.
>
> **3. "The learned exit model is not a policy" is imprecise — withdrawn.** Codex verified it is **not** a
> serialization defect: sealed models reproduce stored predictions with zero error and utility
> recomputation is exact. It is a **valid, degenerate, highly conservative learned policy** whose
> risk-lower-bound calibration (fold `mean_lcb90` offsets −$813.64 to −$895.13) overwhelms nearly every
> point prediction — only 17.46% of first-state `a_ref` labels are positive, and 171/180 of even those exit
> at index zero. That is a better diagnosis than Claude's and is what should be debugged if the exit model
> is ever revisited.
>
> Two feasibility-study claims were **refuted** — see that document's correction banner. The Phase-1
> closure itself stands.

## What was asked, and what was answered

**The question.** Does a maximally-honest learned SPXW 0DTE long-options trader — minute-cadence entry,
1-second exit, trained on a causal t−60s clock — have a real edge?

**The answer.** No, and the reason is structural rather than a modelling shortfall. Buying SPXW 0DTE
premium at minute cadence is a **negative-expectancy position before any transaction cost is paid**.

## The verdict chain

**1. GATE 1 — `UNDERPOWERED`, STOP.** The four-box replay returned `UNDERPOWERED` with 2 of 7 acceptance
conditions passing. Learned/learned pooled PnL −$7,024 against best comparator `matched_random_3` at
−$4,474; bootstrap LCB −$150.20; 1 of 3 fold deltas positive; 0 of 8 fee/latency cells positive.
Independently verified by Claude — the replay semantic SHA `b07784a0…` was recomputed and matched, so the
report is unedited. Negative controls were all rejected and the 36-session firewall stayed closed.

**2. Failure-mode diagnostic — the entry model is not the defect.** Across **156,950** causal OOF
candidates the mean trade loses **−$39.48** at a ~33% win rate, and the loss is **flat across all five
chronological folds** (−45.91 / −40.81 / −38.02 / −35.54 / −37.03). A flat profile is a constant
structural cost, not a decaying edge. No subpopulation is positive — not any hour, either side, any
moneyness bucket, any quoted-spread bucket, or any premium quintile.

**3. Fix research — execution is recoverable, profitability is not.** Posting passively at the bid with a
60-second window instead of crossing at ask + one tick improves **−$41.32 → −$18.56/trade (+$22.76, 55%)**
at an 83.3% fill rate, with adverse selection of only −$0.72, and it is **stable in 5/5 folds**
($20.06–$24.50). It is nonetheless **0/5 folds profitable**: not trading ($0.00) still dominates.

**4. The terminal finding — the position is negative-EV before costs.** Strip out *all* friction (buy at
bid, sell at bid, zero fees) and the average trade still loses **−$13.00**, median −$90, win rate 35.7%,
**negative in 5/5 folds** (−18.59 / −14.00 / −11.63 / −9.26 / −11.33). That is theta. No model, feature,
or execution improvement repairs a position whose gross expectancy is negative.

Supporting: the entry model's **decile ranking is flat** — top decile −$16.37 vs bottom −$17.75
(passive-adjusted), with the *middle* decile best at −$11.48. Non-monotonic. Gifting the model the entire
+$22.76 execution improvement still leaves the top decile 0/4 folds positive.

## Three durable assets

1. **The causal t−60s pipeline.** Audited PASS and demonstrated to work end-to-end: clock, conservative
   fill/label law, entry→exit OOF firewall, storage contract, four-box replay. It correctly produced a
   negative answer, which is exactly what a trustworthy pipeline should do. Reusable for any future
   hypothesis.
2. **The passive-execution costing correction.** +$22.76/trade, stable 5/5 folds. Future Path-D economics
   should be quoted against a realistic passive fill (−$18.56), not the aggressive law (−$41.32). This is
   a *costing* correction, not a strategy.
3. **The knowledge that this class is negative-EV before costs.** −$13.00 gross, 5/5 folds. This is the
   most valuable output: it closes a direction permanently rather than leaving it to be re-litigated.

## What was disproven along the way

- The autoresearch_v2 "confirmed edge" (`signed18`, +$540/session) was a **60-second SPX look-ahead**,
  caught by the runtime decision-parity gate at 18.44% decision match. Quarantined `INVALID_EXPERIMENT`.
- The protected 36-session holdout was **spent** proving that invalidation. There is no clean historical
  out-of-sample data left; forward confirmation can only come from fresh live paper.
- The learned exit model is **not a policy**. `learned_exit_index == 0` in 980/1031 (95.1%) trajectories
  and `learned_exit_value` is identical to `exit_immediate` in 982/1031 (95.2%). Its entire benefit is one
  bit of information — *do not hold 0DTE premium* — which is true and useful but constant.

## The honest next move, if Path-D continues

**A feasibility study, not a training run.** The question to answer with data *before* committing to any
model is: **does any instrument/horizon combination reachable through IBKR have non-negative gross
expectancy?** Gross expectancy is measurable directly from quotes and requires no fitting. If nothing
clears zero gross, no amount of modelling will help, and that is answerable cheaply.

Preconditions for any future training round — these are gates, not suggestions:

1. **Change the position.** Negative gross expectancy cannot be fixed by predicting it better. Either a
   structurally different position, or a horizon with materially less theta per unit time. Selling premium
   is only favoured *passively* — that is market-making (queue priority, inventory risk), a different
   business requiring its own governance packet.
2. **Change the features.** The current 18-feature contract has zero ranking power. Retraining the same
   features with a different learner or hyperparameters is wasted compute. New features must be causal at
   t−60s, live-twin available on Databento OPRA, and pre-registered before fitting.
3. **Re-pose the target.** The label currently bundles a large, deterministic, decision-time-observable
   cost (the quoted spread) with a small, noisy directional signal, which destroys signal-to-noise.
   Predict the gross move and subtract known cost separately. Second-order — it does not fix −$13 gross.
4. **Governance.** The holdout is spent, so any new search requires pre-registration with a hard budget
   under family maxT correction, and forward validation on fresh live paper — never on these 215 sessions.

## Governance state at close

| Item | State |
|---|---|
| Protected 36-session firewall | **CLOSED** — `holdout_open_count = 0`, no firewall session decoded |
| Broker / paper order / promotion / paid download | **None executed** |
| Frozen causal clock, fill/label law, OOF firewall | **Unmodified** — verified by `git diff` |
| Paper default / runtime posture | **Unchanged** |
| Development sessions used | Exactly the 215 pre-firewall sessions |

## Evidence index

- Roadmap + GATE 1 detail: [`PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md`](PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md)
- Fix-research findings: [`PATHD_PHASE1_FIX_RESEARCH_FINDINGS_2026_08_03.md`](PATHD_PHASE1_FIX_RESEARCH_FINDINGS_2026_08_03.md) (`d05f0137`)
- Cold-session research prompt: [`PATHD_PHASE1_FIX_RESEARCH_GOAL_PROMPT_2026_08_03.md`](PATHD_PHASE1_FIX_RESEARCH_GOAL_PROMPT_2026_08_03.md) (`2e14cb61`)
- Replay: `/Volumes/AR_TRADING_DATA/reports/phase1_four_box/replay.json`, semantic SHA `b07784a0…`
- Entry campaign SHA `c7a9ae05…`; exit campaign SHA `69540b75…`
- Commits: `86b1b156` (Stage 1 stop) · `b2e0fdee` (condition-7 fee-axis degeneracy) · `2e14cb61` ·
  `d05f0137`

## Meta-lesson, recorded deliberately

The heavy governed Path-D apparatus reached six frozen generations and ~30k lines **without once running a
real fit**. The lean causal path answered the question in a day. Then the first "confirmed edge" it
produced turned out to be a clock leak that a rubber-stamp verification missed, and cost the protected
holdout to disprove. The rule that follows: **run the cheap real experiment before the elaborate
fake-tested pipeline, and verify the feature-availability clock, not just the future-outcome guard.**

`NO_INCREMENTAL_EDGE` was the predicted honest outcome at the start of Phase-1. It is the outcome. That is
a successful research programme, not a failed one.

*Signed: Claude Opus 5 — 2026-08-03 — status: PHASE-1 CLOSED.*
