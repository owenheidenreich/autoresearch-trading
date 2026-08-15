# The model learned ITM depth, but the causal rank policy did not generalize through time

**Corrected Job 39 result, 2026-08-14. No promotion, paper order, live order, external contact or reserved
session was involved.**

## What changed for the bot

The corrected width-3 neural four-head model can rank later ITM depth, and its causal V5 selector actually
took trades. The pooled spread-free result looked profitable and beat both controls by point estimate.
It nevertheless **failed the chronological evidence requirement**: nearly every trade came from one fold,
three of five later blocks produced no trades, and none of the corrected confidence bounds cleared zero.

The result is negative for this corrected policy, but for a different reason from the void V4 result:

> **The model learned a real magnitude ranking. The ranking did not become a stable out-of-sample trading
> policy across time.**

No bid economics or exit model was opened after that failure.

## V4 remains void

V4 is not reinterpreted as negative. Its selector required a 30-point prediction at the exact upper
support boundary of a target clipped to 30, selected zero trades and never measured economics. The
immutable correction receipt classifies it as `INCONCLUSIVE_SPECIFICATION_DEFECT` and leaves its valid fit
and predictions intact.

V5 repaired only activation. It retained the same `neural_four_head` architecture, width 3, 341 computed
parameters, 120-minute horizon, label, five chronological folds, two-trade cap, ticket-only risk mode and
all seven kill conditions.

## The causal rank law

For each fold, V5 scored only that fold's strictly earlier training sessions. It collapsed the eligible
contract surface to the maximum model score in each minute and froze the k-th largest value where
`k = 2 × training sessions`. It then walked each later scored session minute by minute with that cutoff,
the 120-minute occupancy clock and the two-trade cap.

This is an average top-two training-prefix operating point. It is not a whole-day top-N rule: no future
score from the day being traded can affect an earlier action.

| Fold | Earlier sessions | Real cutoff | Training maximum | Scored trades |
|---:|---:|---:|---:|---:|
| 1 | 93 | 20.8167 | 21.9742 | 0 |
| 2 | 123 | 19.7153 | 20.1699 | 22 |
| 3 | 153 | 24.6958 | 25.4802 | 2 |
| 4 | 183 | 34.5171 | 34.9514 | 0 |
| 5 | 213 | 28.0800 | 29.2167 | 0 |

Every cutoff was finite, interior to its training score distribution and frozen before its score block.
The selector-specific gate now refuses any absolute prediction threshold at or above the label clip
ceiling, covering the V4 defect class without modifying the owner-controlled policy-fit gate.

## The model did learn

Before economics were read, the 423,053 real out-of-fold predictions showed:

- Pearson correlation with actual 120-minute maximum ITM depth: **+0.3300**;
- monotonically rising actual depth through all ten prediction deciles: **−10.28 to +6.50 points**;
- predictions at or above the diagnostic 25-point mark: **14,103 real versus 0 shuffled**; and
- prediction standard deviation: **11.44 real versus 5.11 shuffled**.

The 25-point count is a diagnostic, not a trading threshold. These results establish learned ordering,
not economic value.

## Economic result

Kill condition #1 passed once V5 selected real trades:

| Policy | Trades | Days | Mean gross mid-to-mid/trade | Median |
|---|---:|---:|---:|---:|
| Real causal rank policy | 24 | 24 | **+$137.81** | −$117.50 |
| Exact composition-matched control | 24 | 24 | −$133.23 | −$477.50 |
| Shuffled-label rank policy | 37 | 37 | +$68.51 | −$217.50 |

The matched control reproduced scored session, entry regime, side, absolute-delta quintile,
entry-ask-premium quintile and trade count; it excluded model trades, sampled without replacement and
obeyed the same 120-minute non-overlap law. All 24 matches were complete. Thus the real policy beat both
controls by pooled point estimate.

But the evidence was concentrated:

- fold 2: 22 trades, **+$183.30/trade**;
- fold 3: 2 trades, **−$362.50/trade**; and
- folds 1, 4 and 5: **no trades**.

Counting all scored sessions, including abstentions, absolute gross was positive in only **1/5 folds**.
Real-minus-matched and real-minus-shuffled were each positive in only **2/5 folds**, below the frozen 4/5
standard. With V4's full 648-member pre-exit family retained rather than narrowed after the fit, the
one-sided session-block lower bounds were:

| Quantity | Corrected lower bound per session |
|---|---:|
| Real gross | **−$108.75** |
| Real minus composition-matched | **−$44.78** |
| Real minus shuffled | **−$172.59** |

Therefore kill condition 6, `chronological_out_of_sample`, failed. The pooled profit is not stable enough
to credit as a policy result.

## Seven signed conditions

| Condition | Result |
|---|---|
| Mid-to-mid gross positive | Pass |
| Beats composition-matched control | Pass by pooled point estimate |
| Beats shuffled-label null | Pass by pooled point estimate |
| Per-feature timestamp audit | Pass |
| No post-entry slot filter | Pass |
| Chronological out of sample with corrected confidence | **Fail** |
| No reserved sessions | Pass |

The stop is consequential: ask-to-bid economics, broader cells and the conditional exit fit remain unread.
No alternative cutoff, target rank, seed, architecture or horizon follows this result.

## Evidence-budget limitation

The run used the re-ruling's generous budget: 7,557 effective observations / 20 = **377 parameters**,
admitting the 341-parameter primary model. The conservative design-effect route permits only **29–50
parameters**, so it admits none of the declared architectures. This negative result does not erase that
limitation, and no outcome here promotes anything.

## Evidence

- selector-defect correction: `v4/audit/autoresearch/causal_day_magnitude_v4_specification_correction_2026_08_14_attempt001/receipt.json`
- V5 declaration: `v5/work/entry-exit-attribution/DECLARATION_V5.json`
- pre-economics rank calibration and diagnostics: `v4/audit/autoresearch/causal_day_rank_selector_calibration_2026_08_14_attempt001/receipt.json`
- corrected economics: `v4/audit/autoresearch/causal_day_magnitude_corrected_primary_economics_2026_08_14_attempt001/receipt.json`

The owner-controlled `STATUS.md`, policy-fit gate, frozen knobs, statistics and do-not-retest ledger were
not edited.
