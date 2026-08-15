# The executable target is buy this contract now versus preserve the slot

**Pre-fit label result, 2026-08-14. No model, selector, threshold, paper order, live order or reserved
session was used.**

## What changed for the bot

The project now has the exact supervised target the prior magnitude fit lacked. Every flat minute contains
one WAIT value, and every eligible affordable contract contains one ENTER value under the existing
$10,000 ask-entry, first-bid-at/after-120-minutes, fee and validated-settlement laws.

> The learning problem is no longer “which option later goes deep ITM?” It is “does this exact executable
> contract pay more than preserving the only trade slot for every later opportunity?”

This is a label substrate, not evidence that the choice can be predicted.

## Frozen law

The V2 declaration was sealed before target diagnostics with self-hash
`2abdf192bcac3eba5f3fd2e38cc3f45c499eee92af22724ac6cb043638c31e39`.

For contract `i` at minute `t`:

`Q_enter(t,i) = net_bid_120m_usd(t,i)`

The existing value buys at the ask, exits at the first executable bid at or after the 120-minute request
or validated PM cash settlement, and subtracts the measured $3.08 round-trip fee.

With exactly one trade available:

`Q_wait(t) = max(0, every Q_enter(s,j) at a strictly later minute s)`

`A_enter(t,i) = Q_enter(t,i) - Q_wait(t)`

The future values are supervised labels only. Scored features remain causal. WAIT wins by its predicted
value at inference; there is no prediction cutoff, current-day rank or forced key time.

## One structural defect repaired before output

V1 assumed every decision minute had at least one eligible contract row. That is false: 1,964 of 79,218
minute states have no affordable eligible contract. The first full build refused on the first session
before computing a diagnostic or writing an output.

V2 preserves those states as WAIT-only minutes. V1 remains immutable history. This changes no payoff,
horizon, outcome or selection law and prevents the dataset from silently dropping moments when the bot
cannot act.

## Full population and integrity

- sessions: **243** (`2025-08-01` through `2026-07-30`)
- decision minutes: **79,218** (326 per session)
- eligible contract actions: **698,231**, unchanged from the settlement-complete dataset
- WAIT-only minutes: **1,964**
- primary one-trade oracle actions: **243**, one per session
- model fits: **0**
- reserved sessions: **0**

Every Q(wait) excludes the current minute and is non-increasing as the remaining day shrinks. Every
artifact and receipt hash re-verifies.

## Declared scale diagnostics

| Quantity | 10th percentile | Median | 90th percentile | 99th percentile |
|---|---:|---:|---:|---:|
| executable Q(enter) | -$678 | **-$173** | +$757 | +$3,357 |
| Q(wait) | +$237 | **+$1,437** | +$3,857 | +$9,827 |
| A(enter) | -$3,540 | **-$1,315** | -$310 | -$20 |
| session hindsight ceiling | +$1,219 | **+$2,567** | +$5,125 | +$11,945 |

Only **5,026 / 698,231 = 0.72%** of contract actions have positive executable advantage over waiting.
There are 3,932 right-to-left record minutes with positive best advantage, **4.96%** of all decision
minutes. Every session has at least one positive full-hindsight opportunity, but that is an oracle ceiling,
not a tradable result.

The primary oracle is direction-balanced: 118 calls and 125 puts. Its predeclared broad time-band counts
are 39 opening, 33 around the 10:00 band, 98 in the rest of the morning, 18 pre-13:30 afternoon, 14 around
the 13:30 band and 41 later. These are descriptive ceiling locations and cannot create a time filter.

## Consequence

The target is numerically attainable and economically aligned, but highly selective. A magnitude ranker
cannot pass it by buying the highest-vega option at the open. The next fit must estimate WAIT and ENTER in
common dollar units and succeed by action argmax across chronological folds.

No label, horizon, trade cap, time band, side or loss was changed after reading these diagnostics. If the
single compact fit cannot clear spread-free existence and 4/5 chronology, the long-selector branch closes
instead of receiving a nearby retry.

## Evidence

- V2 declaration: `v5/work/entry-exit-attribution/ACTION_ADVANTAGE_DECLARATION_V2.json`
- receipt: `v4/audit/autoresearch/causal_day_action_advantage_2026_08_14_attempt002/receipt.json`
- generated root: `/Volumes/AR_TRADING_DATA/derived/causal_day_action_advantage_v2`

The owner-controlled status, gate, knobs, statistics and do-not-retest ledger were not edited.
