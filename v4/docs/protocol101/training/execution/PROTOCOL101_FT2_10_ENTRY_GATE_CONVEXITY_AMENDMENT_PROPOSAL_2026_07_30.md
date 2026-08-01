# FT2-10 Entry-Gate Convexity Amendment — PROPOSAL

**STATUS: PROPOSAL (Claude-written for Codex review + implementation).** NOT
YET APPLIED. A governed FT2-10 amendment; requires Codex critical review, then
implementation via the standard reseal (new authority hash, re-pin,
consistency checker green, graph unchanged), then owner sign-off. Roles-flip:
**Claude writes, Codex reviews.** Prepared 2026-07-30 from the walking-skeleton
Option-D finding.

## The defect (confirmed empirically, not inferred)

The frozen composer entry gate
`composer_spec.json:/stage_2_conservative_upside_rank/positive_after_fee_rule`
requires the equal-horizon **mean of the calibrated q10 (10th-percentile) MFE
and profit-area** to be strictly > 0 in both dollars and return.

Walking-skeleton Option-D evidence (45-session clean dev slice, real heads +
real conformal calibration, frozen composer, 0 threshold changes):

- **Realized q10 MFE return is structurally NEGATIVE at every horizon**:
  h3 −16%, h5 −15%, h10 −13%, h20 −11%, h45 −9%, h90 −8%, session −7%.
  Meanwhile median/mean/‑win-rate climb strongly with horizon
  (session median +51%, mean +212%, 83% winners).
- Therefore the gate demands the model predict something that **does not exist
  in the labels**: a positive pessimistic-decile favorable excursion for a
  decaying long option. **0 of 4,191 eligible contracts pass**, at every
  guardrail setting AND the no-screen (alpha=0) control. No model, on any data,
  can satisfy it.
- The realized distribution IS the Pickles profile: a **high win rate** (83%
  of session-horizon outcomes positive here) — mostly small wins/scratches
  with a minority of large wins. It is NOT a loss-seeking strategy. But **every
  real trade carries a losing tail** — that is inherent, unavoidable risk. The
  gate demands the pessimistic decile (that loss tail) *also* be profitable —
  i.e., a **risk-free entry**, which no tradeable position offers. Requiring
  zero downside even in the bad case means **never entering**. The correct
  stance is to accept the normal loss tail at entry and *minimize* it
  downstream with the exit model.
- It is also **misaligned with the RLAC training target**, which calls a
  majority of minutes enter-worthy (u_label mean +0.47, fee_cleared 54%,
  ENTER-justified 99.9% of minutes). The model is trained toward one notion of
  "worth entering" and gated on a stricter, unsatisfiable one.

Three cold design reviews could not catch this: the spec is internally
consistent; the defect only appears when a fitted model meets the real label
distribution. This is exactly the class of bug the ~$10 dry-run exists to find.

## The fix

Move the entry **GATE** from the pessimistic q10 of the outcome to a
**central-tendency expected-upside** criterion; keep the conservative q10 for
**RANKING** among gate-passers (preserving the D51 loose-screen +
conservative-rank intent). Downside risk stays where the Charter puts it — the
**exit model + per-trade cap (D48) + daily breaker (D49)** — not the entry gate.

This matches the two models' actual jobs. The **entry** model scans
**market-wide across many contracts** to choose a good entry — a broad
selection decision made once, from forecasts. The **exit** model is **locked
onto the single live position** — that exact contract's greeks, tick-by-tick
at 1-second resolution — and is the only component that can actually see and
act on a developing loss. Loss control belongs to the model that holds the
position, not to a market-wide entry filter that demands the loss never happen.

**Proposed new `positive_after_fee_rule` (GATE):**
> A contract is entry-eligible only if the equal-horizon mean of the
> **calibrated conservative lower bound on the EXPECTED (mean) MFE and profit
> area** is strictly greater than zero in both dollars and return, after fees.
> "Conservative lower bound on the expected upside" = the conformal lower
> confidence bound on the *mean* MFE forecast — a principled "don't trade on
> noise" guard — NOT the q10 of the outcome distribution.

**RANK (unchanged in spirit):** among gate-passers, keep ranking by the
existing conservative (q10) upside — robust selection is fine once the gate is
satisfiable.

**Diagnostic support for the operating point** (evaluated on realized labels;
the model gates on calibrated forecasts of them):

| Candidate gate | pass rate | passer median session-MFE (Δ vs fail) |
|---|---|---|
| current: q10 MFE > 0 | **0.000** | — (unsatisfiable) |
| expected (mean-horizon) MFE > 0 | 0.75 | +0.85 (+0.90) |
| expected upside > 5% (clears fee) | 0.67 | +0.99 (+1.00) |
| expected upside (h45–session avg) > 5% | 0.76 | +0.92 (+0.97) |

An expected-upside gate is **satisfiable (~67–76%)**, strongly **discriminating
(+0.90 to +1.00 median session-MFE gap)**, and aligned with the RLAC target.

**Design points to finalize in review (recommended defaults in bold):**
1. Central statistic: **expected (mean)** MFE — the convex strategy monetizes
   the tail, which the mean captures; median is the conservative alternative.
2. Conservative margin: **conformal lower bound on the expected MFE** (keeps a
   principled noise guard without demanding the bad decile win).
3. Horizon set: **consider excluding the structurally-dead ultra-short horizons
   (h3/h5)** from the gate, or down-weighting them; the equal-horizon average
   currently drags in always-negative short horizons.
4. Operating threshold: start at **expected upside > fee-equivalent (~5%)**;
   the other frozen downstream gates (cluster-uncertainty WAIT, q90 regret,
   action-conditioned) remain unchanged and further trim.

## Scope — what does NOT change

Only the entry **gate criterion** in `stage_2_conservative_upside_rank`. The
RLAC targets, path-label definitions, folds, fees, serial simulator, D48 cap,
D49 budget/soft-close, the exit/lifecycle design, the guardrail bottom-tail
screen (D51), and all other frozen contracts are UNCHANGED. Downside protection
is unchanged — it lives in exit + cap + breaker, as designed.

## Implementation + verification path (for Codex)

1. Codex critically reviews this design (is the convexity diagnosis right? is
   the expected-upside gate the correct fix? any second-order effect?).
2. If sound: implement via governed reseal — edit `composer_spec.json`
   (`positive_after_fee_rule`), fold this amendment into the consolidated
   authority (new hash, re-pin across specs/receipts/checker), keep the graph
   unchanged, consistency checker must stay green, add a checker rule that the
   entry gate references a *satisfiable* central-tendency statistic (guards
   against a future q10-gate regression).
3. Re-run walking-skeleton Stage-1 (Option-D faithful capped) to confirm the
   entry now trades on real opportunity. First real exercise of the downstream
   uncertainty/regret/action-conditioned gates — watch for secondary
   over-abstention there.
4. Claude verifies Codex's implementation (cross-model check), then owner
   sign-off on the amendment.
