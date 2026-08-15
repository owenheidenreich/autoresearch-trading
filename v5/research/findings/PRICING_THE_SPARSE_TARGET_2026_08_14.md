# The data was never the constraint on the owner's strategy — the selectivity is

**Finding, 2026-08-14. Model-free. No policy or threshold was fitted. Arithmetic on banked prevalence.**

## What changed for the bot

The owner's stated strategy — buy a near-OTM 0DTE contract, hold it through a 10–30 point move, a few
trades a day — has been priced. The answer is not the one the project has given for every other screen.

**More sessions would not settle it.** If the owner's thesis were true at the size claimed, it would be
provable on **27 sessions**, and the project owns 243. What blocks it is not evidence volume but the
**precision lift** a policy must achieve: from a 1.57% base rate to 17.51% just to break even on the
30-point target. That is an **11.1x lift**, and nothing this project has measured has ever produced one.

This reverses the standing recommendation. Buying quote history was the obvious next move under the
"n is the only lever" reading in the old STATUS header. For *this* target it would buy nothing.

## The ladder

From the settlement-validated dataset receipt: 698,231 affordable near-OTM candidates over 243 sessions,
mean entry ask **$519.20**. A win closes when the underlying first reaches the declared depth and pays
intrinsic less the entry ask, the measured **$17.54** exit spread and **$3.08** fees. A loss gives up the
premium. Family of nine cells, one-sided 95%, 2 trades per session.

| Target | Base rate | Break-even precision | Lift needed | Sessions to prove 2x break-even |
|---|---:|---:|---:|---:|
| 10 pt / 60m | 11.33% | 53.16% | 4.7x | not achievable — 2x exceeds 100% |
| 10 pt / 120m | 19.60% | 53.16% | 2.7x | not achievable |
| 20 pt / 60m | 4.01% | 26.34% | 6.6x | **16** |
| 20 pt / 120m | 9.02% | 26.34% | 2.9x | **16** |
| 30 pt / 60m | 1.57% | 17.51% | **11.1x** | **27** |
| 30 pt / 120m | 4.21% | 17.51% | 4.2x | **27** |

The session counts are small precisely *because* the hypothesised edge is enormous. A policy hitting twice
break-even on a 4.7:1 payoff is extraordinarily profitable, and extraordinary things are easy to see. The
measurement problem that has dominated this project — `detectable_sharpe`, the 1/sqrt(n) wall, the
underpowered G1 — applies to **marginal** edges. It does not bind here.

## The owner's instinct about big moves is arithmetically correct

The project has repeatedly measured short holds and small targets. The ladder says that was the harder
game:

- The **10-point** target pays about **0.9 : 1** and breaks even at **53.16%** — a coin flip on a payoff
  that barely covers its own costs. This is the regime every prior negative result lived in.
- The **30-point** target pays **4.7 : 1** and breaks even at **17.51%**.

**Holding for the larger move lowers the accuracy bar by 35 percentage points.** "Long defined 10–30 point
moves, healthy risk reward" is not a stylistic preference; it is the only end of this ladder where the
required accuracy is not the binding problem. Pinned by
`test_the_deep_target_asks_less_accuracy_than_the_shallow_one`.

## What this does and does not say

It does **not** say a policy can achieve 17.51%. It says that is the number to beat, that 243 sessions can
resolve whether it does, and that the honest experiment is therefore available today rather than after a
purchase.

It also does not lift either fit gate. The causal day trader remains refused by G1 and by
[`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md) row 340, which now stands in direct conflict with row
41's own stated next step. See [STATUS §16](../../STATUS.md).

Two declared assumptions any screen freezing a number from this must restate rather than inherit:

1. **The exit.** A target is scored as though the position closes when the underlying *first* reaches the
   declared depth. A policy exiting earlier earns less; one holding past the depth may give it back.
2. **The loss.** The headline column assumes a miss loses the whole premium. The receipt also carries 25%
   and 50% recovery columns, which lower the break-even and the lift.

## Evidence

- receipt: `v4/audit/autoresearch/sparse_target_price_2026_08_14/receipt.json`
- tool: [`ops/price_sparse_target.py`](../../ops/price_sparse_target.py)
- new inverse-power function: `sessions_for_accuracy_edge` in
  [`research/statistics.py`](../statistics.py), with five regression tests
- prevalence source: `v4/audit/autoresearch/causal_day_dataset_settlement_validated_2026_08_14/receipt.json`

Nothing here contacted a vendor, broker or runtime; fit a model; tuned a threshold; used post-cutoff data;
or authorized promotion.
