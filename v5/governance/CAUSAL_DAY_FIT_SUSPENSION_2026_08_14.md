# Suspension: §4 of the fit reopening is withdrawn pending a re-ruling

**Owner instruction, 2026-08-14, after signing. In force until explicitly lifted.**

The reopening at [`CAUSAL_DAY_FIT_REOPENING_2026_08_14.md`](CAUSAL_DAY_FIT_REOPENING_2026_08_14.md) was
signed the same day. **§4 of that document contains a table of parameter counts, and every number in it is
wrong.** No model may be fitted — simple or complex — until §4 is re-ruled in full.

## What was wrong

§4 authorised the neural comparison by changing the sessions-per-parameter rule's unit from sessions to
causal decision states, giving a budget of 93,798 / 20 = **4,689 parameters**. It then tabulated each
architecture against that budget. The counts were transcribed from
[`work/entry-exit-attribution/LOG.md`](../work/entry-exit-attribution/LOG.md) and the
[atlas finding](../research/findings/CAUSAL_DAY_TRADER_ATLAS_2026_08_14.md), not computed.

Built at the real tensorizer dimensions — candle 18, ladder 23, account 5, position 10, clock 5, hidden 16:

| Architecture | §4 claims | Actual | Share of the 4,689 budget |
|---|---:|---:|---:|
| `shallow_joint` | 676 | **1,604** | 34.2% |
| `shallow_four_head` | 720 | **1,688** | 36.0% |
| `neural_joint` | 1,252 | **3,780** | 80.6% |
| `neural_four_head` | 1,296 | **3,864** | **82.4%** — §4's table implies 27.6% |
| `four_independent` | 4,920 | **14,952** | 318.9% — excluded either way |

Every row is wrong by roughly 2.4x to 3.0x.

**The conclusion §4 reached survives; the margin it was granted on does not.** Four of five architectures
still fall under the budget, so the table's bottom line is accidentally correct. But §4 presented
`neural_four_head` as using about a quarter of the available budget when it uses **more than four
fifths**. A ruling that would plausibly have been made differently at 82% was made at an implied 28%.

That is the reason for this suspension. The error is not that the answer changed — it is that the owner
ruled on a margin that did not exist, and a governance document is worthless if its load-bearing numbers
are transcribed rather than computed.

## Two weaknesses that let it pass

1. **`fit_blockers` trusted a caller-supplied integer.** `trainable_parameter_count` and
   `build_architecture` already existed in
   [`causal_day_architectures.py`](../research/causal_day_architectures.py); the gate simply never called
   them. A runner passing §4's `1,296` would have cleared a gate that the real 3,864-parameter model
   fails.
2. **The existing tests asserted the same wrong literals**, so the suite *encoded* the false counts rather
   than catching them. A test that repeats a number from a document proves the transcription is
   self-consistent, not that it is true.

Both are repaired alongside this suspension: the widths are now named constants in the tensorizer, counts
are computed from built models, and the gate verifies a declared count against the computed one.

## What this suspension blocks

**Every architecture, through every route.** The check lives in `fit_blockers`, not only in
`load_reopening`, so a hand-constructed `Reopening` object cannot bypass it. While this document is in
force the gate refuses all five architectures regardless of label, horizon, corpus or declared kill
conditions.

Nothing else in the reopening is withdrawn. §1's release of the G1 and ledger-row-340 blockers, §2's
evidence statement, §3's seven kill conditions and §5's bars all stand. Only §4 is suspended, and because
§4 is what admits the sequence architectures, the practical effect is a full stop.

## What lifts it

A signed re-ruling of §4 that:

1. states every architecture's **computed** parameter count, verified against a built model; and
2. addresses the strongest objection §4 never confronted — that **93,798 causal minute states are not
   93,798 independent observations.** They are 243 sessions x ~386 minutes, and consecutive states share
   almost all of their history, ladder and label. §4 argued the unit should be "the decision the model is
   asked to make", which is true of what a model *consumes* but says nothing about *independent
   information*. The effective sample size lies somewhere between 243 and 93,798, and §4 silently assumed
   the top of that range.

The re-ruling package must present the argument against the unit change at least as strongly as the
argument for it, and must rest on a measurement of effective sample size rather than on rhetoric.

Deleting this file does not lift the suspension; the gate refuses when the document is absent as well as
when it is present, so the release is the new signed ruling, never the removal of this one.
