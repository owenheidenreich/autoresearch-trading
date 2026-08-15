# V5 was a magnitude sensor, not a trader

**Outcome-blind attribution of the closed Job 39 V5 policy, 2026-08-14.** No new fit, selector,
threshold, economic subgroup, P&L column, target column, reserved session, paper order or live order was
opened.

## What changed for the bot

The corrected V5 run established that the network learned later ITM depth. This attribution establishes
why that did not become a stable trading policy:

> V5 learned a static option-geometry ranking plus a chart/clock-dependent common score offset. It did
> not learn a chart-conditioned contract choice, a buy-now-versus-wait action, or an exit action.

That distinction closes every nearby V5 cutoff or sequence-encoder retry. The next admissible entry model
must contain an explicit state-by-contract interaction and must train WAIT in the same action objective as
the contracts.

## The architectural identity

Every Job 39 entry head is one linear map over the concatenation of the shared state and one contract
node:

`score(t, i) = bias + w_state · state(t) + w_node · node(t, i)`

Within one routed minute, the state term is identical for every contract. Therefore:

`score(t, i) - score(t, j) = w_node · [node(t, i) - node(t, j)]`

Candles, clock, account state and the global ladder summary can move all scores up or down, but cannot
change which call, put or strike ranks first. Four heads provide two static entry rankings by regime; they
do not make contract preference conditional on the tape.

The 423,053 stored OOF scores reconstruct as state offset plus contract-node contribution to within
`7.63e-6` points, and the checkpoint computation itself reconstructs to `2.38e-7`. State offset is
constant within every minute. Full-score and node-only argmax agree on 47,602 of 47,622 minutes; the 20
near-tie differences are floating-point ordering effects at reconstruction-scale differences.

The predeclared numerical rule required literal argmax identity on every minute, so
`architectural_separability_verified` is correctly recorded as false in the receipt. The source algebra,
mutation test and reconstruction nevertheless establish the narrower architectural fact without using
outcomes. The failed literal condition is not silently relabelled as a pass.

## WAIT and EXIT were never trained

The Job 39 fitting loss reads only `model(batch).contract_logits`. Neither `abstain_logits` nor
`exit_logits` enters the loss, so their parameters receive no gradients. A new regression cutoff was then
used outside the model to decide whether to trade.

A mutation/gradient test now proves this property directly. Calling the artifact a four-role trader would
be inaccurate: the fit was a contract-level 120-minute magnitude regressor with an external activation
law.

## What the fitted score actually used

Contract-node contribution was dominated by contemporaneous option geometry:

| Contract proxy | Within-minute correlation with node contribution |
|---|---:|
| self vega | **+0.851** |
| absolute delta | **+0.788** |
| moneyness | +0.718 |
| theta per minute | -0.700 |
| gamma | +0.671 |
| entry ask | +0.640 |

The total score surface was 29.95% within-minute and 70.05% between-minute variance. Neutralizing the
candle state produced the largest common-offset change (mean absolute 2.00 points), followed by explicit
clock (1.46) and global ladder summary (0.44). These channels changed the level of every contract score,
not the within-minute ordering.

## Why V5 fired at the open in one fold

The five frozen training-prefix cutoffs were not on a stable score scale:

- folds 1, 4 and 5: cutoff above every scored minute, so no activation;
- fold 2: 94 crossing minutes across 22 sessions;
- fold 3: three crossing minutes across two sessions; and
- all 24 activated sessions first crossed during 09:35--09:39.

This is an early-entry and calibration collapse, not evidence that 09:35 is the best trade time. The
opening had the day's highest common score surface in most folds, so a threshold intended to admit about
two trades per training session fired as soon as the day became eligible whenever the fold scale happened
to cross it.

The named 10:00 and 13:30 areas remain causal clock/context features, not forced entries. V5 could lift or
lower every contract around those times, but its head could not use the observed candle structure to flip
the preferred direction or strike.

## Consequence

Do not fit another Job 39 family member, cutoff, rank target, seed or horizon. The evidence-compatible
successor is a compact action-value policy with:

1. an explicit tape-state × option-side interaction;
2. an explicit tape-state × moneyness interaction;
3. WAIT in the trained action vector; and
4. executable value of entering now relative to preserving the slot for later.

The implemented design-only successor has 48 computed parameters and passes its mutation/gradient tests.
Fitting remains refused because both its architecture and serial action-advantage label are outside the
signed gate scope.

## Evidence

- declaration: `v5/work/entry-exit-attribution/ATTRIBUTION_DECLARATION_V2.json`
- receipt: `v4/audit/autoresearch/causal_day_v5_instability_attribution_2026_08_14_attempt002/receipt.json`
- generated metrics: `/Volumes/AR_TRADING_DATA/derived/causal_day_v5_instability_attribution_v2/attribution_metrics.json`
- compact design: `v5/work/entry-exit-attribution/COMPACT_INTERACTION_SUCCESSOR_V1.md`

The owner-controlled status, policy gate, knobs, statistics and do-not-retest ledger were not edited.
