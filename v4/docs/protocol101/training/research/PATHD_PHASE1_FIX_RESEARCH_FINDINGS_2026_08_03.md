# Path-D Phase-1 — Fix-Research Findings (2026-08-03)

Executed against the goal prompt in
[`PATHD_PHASE1_FIX_RESEARCH_GOAL_PROMPT_2026_08_03.md`](PATHD_PHASE1_FIX_RESEARCH_GOAL_PROMPT_2026_08_03.md).
Read-only analysis. No training, no broker, no paid data, firewall untouched.

## Verdict

**`FIX_CANDIDATE` for execution — `NO_FIX` for profitability.**

Passive entry recovers **$22.76/trade (55% of the loss)** and is the first effect in this project that is
**stable in all 5 folds**. It still leaves **−$18.56/trade**, so **not trading ($0.00) continues to
dominate**. Passive entry narrows the gap; nothing in the data closes it.

---

## 1. Two corrections to the prior record

**(a) The learned exit model does not have a learned policy.** It was previously described (by Claude) as
having "real skill — preserve this asset." That was overstated. `learned_exit_index == 0` for **980/1031
(95.1%)** of trajectories, and `learned_exit_value` is **literally identical** to `exit_immediate` in
**982/1031 (95.2%)** rows. Its entire measured benefit (−$284 → −$54/trade) is one bit of information:
*do not hold 0DTE premium.* That is a real and useful fact, but it is a constant, not a policy.

**(b) The cost decomposition is refined.** The earlier split (execution −$16.74 / decay −$19.74 /
fees −$3.00) conflated the exit half-spread into "decay." Measured directly on all **156,950** candidates:

| Component | Per trade | Share |
|---|---|---|
| **Instant round-trip friction** (buy at fill, sell at bid, incl. fees) | **−$26.48** | **67%** |
| Directional/decay drift beyond the instant mark | −$13.00 | 33% |
| **Total realized** | **−$39.48** | |

Average premium paid is **$565.39**, so the round trip costs **4.68% of notional before the position
moves at all**. Quoted spread averages $13.48 (median $10.00) and the frozen law pays a further $10.00
tick above the ask.

## 2. Lever B — holding period (answered first; it is decisive and cheap)

Mean $/trade by exit policy over all 1,031 trajectories:

| Policy | $/trade |
|---|---|
| exit_60s | −53.44 |
| exit_immediate | −53.85 |
| learned_exit | −54.15 |
| exit_300s | −57.39 |
| exit_900s | −66.06 |
| **hold to 15:55** | **−284.07** |

Shortening the hold is worth an enormous amount (−$284 → −$54) but **cannot reach positive**. Across the
full 8 policies × 5 folds grid, the **best of all 40 cells is −$27.79**, and that cell
(`stop50_target100`, fold 4) is −$97.79 in fold 1 — noise, not an effect. `exit_immediate ≈ −$53.85` is
effectively the floor, and it *is* the round-trip friction. **There is no exit timing that beats
"get out now," and getting out now is what costs the money.**

## 3. Lever A — passive entry (the counterfactual)

**Design.** At the exact moment the frozen law would cross (`ask + one tick`), instead post a passive buy
limit `L` and wait up to `W` seconds. Fill at the first second `k ≤ W` where `ask[k] ≤ L`, then hold `H`
seconds. Population: the **878 DETERMINISTIC_CONTROL_OOF** entries — deliberately the *non*-signal-selected
set, so the measurement is about execution and not about the entry model.

> **A v1 of this analysis was discarded as invalid.** It required the fill index `k < learned_exit_index`;
> since that index is 0 for 95.1% of rows, passive fills were structurally impossible and the resulting
> 2–5% fill rates were an artifact of the test, not a market measurement. v2 removes the coupling.

**Result (best configuration: limit = bid, W = 60 s, H = 0):**

| | Value |
|---|---|
| Aggressive baseline (cross at ask+tick) | −$41.32/trade |
| **Passive (filled)** | **−$18.56/trade** |
| **Improvement** | **+$22.76/trade (55%)** |
| Fill rate | 83.3% |
| Adverse selection `E[agg|fill] − E[agg|no-fill]` | **−$0.72** |
| Not trading | **$0.00** |

**Adverse-selection control.** The trap is that passive fills are selected — you get filled when the
market is moving against you. Measured, that effect is **−$0.72**, i.e. essentially absent at the
bid/60 s configuration: the would-fill and would-not-fill subsets have near-identical aggressive
expectancy. The adverse component that *does* exist shows up inside the PnL — a no-drift round trip at
bid→bid would cost only fees (−$3.00), and the measured −$18.56 means the bid falls ~$15.56 between fill
and exit. That is the true cost of passive filling, and it is still far smaller than the $26.48 friction
it replaces. Wider limits are *worse*: `mid` −$22.72, `ask_minus_tick` −$19.69, with larger adverse
selection (−$3.89, −$6.69).

**Fold stability — independently recomputed:**

| Fold | n | Fill % | Aggressive | Passive | Improvement |
|---|---|---|---|---|---|
| 0 | 177 | 80.2 | −42.32 | −19.76 | **+22.56** |
| 1 | 184 | 83.2 | −42.40 | −18.03 | **+24.37** |
| 2 | 176 | 80.1 | −43.00 | −18.50 | **+24.50** |
| 3 | 152 | 87.5 | −39.18 | −19.13 | **+20.06** |
| 4 | 189 | 85.7 | −39.48 | −17.60 | **+21.88** |

**5/5 folds positive, range $20.06–$24.50. 0/5 folds profitable.** The stability is expected and is *not*
evidence of alpha: this is a mechanical effect (not paying spread + tick), and mechanical effects do not
decay the way predictive signal does. That is precisely why it survived when everything else in this
project decayed.

**Honest limitation.** The fill model counts a fill when the offer trades down to the limit
(`ask[k] ≤ L`). Real passive fills also require queue priority; an 83.3% fill rate inside 60 s assumes
favourable queue position. **Real-world fill rate would be lower and adverse selection higher**, so
+$22.76 is an *upper* bound on the recoverable amount.

## 4. Lever C — direction (not pursued to conclusion)

The mirror position would collect the +$13.00 drift but pays the same ~$26.48 friction, so a *crossing*
premium seller is also negative. Only a *passive* seller — collecting both the drift and the spread — is
structurally favoured, and that is market-making: it requires queue priority and inventory risk
management, is a different business from a retail IBKR execution path, and would need its own governance
packet. Not recommended on this evidence.

## 5. What would have to be true for this to work

Breakeven after the passive fix requires directional edge > **$18.56 on $565 premium = 3.3% per trade in
expectation**. The entry model shows ~zero ranking skill in every fold (Spearman +0.072 / +0.032 / +0.040 /
NaN / −0.009), its predictions turn wholly negative by folds 3–4, and it correctly declines to trade at
all in the last 66 sessions. **There is no evidence in this corpus that such an edge exists.**

## 6. Recommendation

1. **Adopt the passive-entry finding as a costing correction, not a strategy.** Any future Path-D economics
   should be quoted against a realistic passive fill (−$18.56) rather than the aggressive law (−$41.32).
   The frozen `FILL_LAW` should NOT be edited — this belongs in a new, explicitly-labelled counterfactual
   fill model so the frozen law and its receipts stay intact.
2. **Do not resume entry-signal search on this corpus.** The holdout is spent; multiplicity would
   manufacture a false positive with nothing left to catch it.
3. **If Path-D continues**, the only honest framing is: find a directional edge worth >3.3%/trade in
   expectation, and validate it forward on fresh live paper — not on these 215 sessions.

## 7. Falsifiers

This recommendation should be discarded if any of these hold:

- A queue-aware fill simulation (position in book, not just `ask ≤ L`) drops the fill rate materially
  below 83% or pushes adverse selection well past −$5/trade — the +$22.76 would then be largely illusory.
- The passive improvement fails to reproduce on a differently-constructed control population.
- An entry signal is demonstrated with >3.3%/trade expected edge that is stable in the most recent folds
  (not folds 0–1 only) and survives family maxT correction under a pre-registered budget.

## Reproduction

Diagnostics are in the session scratchpad (`decay_diagnostic.py`, `lever_a_v2.py`); all inputs are the
committed artifacts under `/Volumes/AR_TRADING_DATA/`. Every headline number here was recomputed a second
time with independent code before being reported; the passive headline (−$41.32 → −$18.56, 83.3% fill)
reproduced exactly.

*Signed: Claude Opus 5 — 2026-08-03.*
