# Protocol101 Stage-1 G4 + Holdout Revision — DRAFT FOR OWNER SIGNATURE

Drafted: 2026-07-19. Supersedes, once signed, the G4 row and the holdout
drawdown sentence in `PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`.
This is a repair of a measured-infeasible gate, executed before any
Stage-1 training exists — not a response to disappointing results.

## Why the current G4 cannot stand (evidence chain)

1. **It has never been passed** — 0/42 gated attempts in the Group 2
   program, 0 anywhere else.
2. **It is below the no-skill noise floor.** Fold-native measurement
   (271 sessions, 5 folds): random selection draws down **$10.7k–$38.2k
   per fold** (mean $20.5k) against the current cap of ~$2,500
   (25% of $10k peak). At 2 trades/day, no-skill drawdown is still
   $4.7k–$19.3k per fold. `protocol101_stage1_g4_feasibility_training_scope_v2`.
3. **No oracle can anchor a cap.** Every hindsight selector — including
   one FORCED to trade daily, and top-q% skill sweeps down to mediocre —
   shows ~$0 drawdown, because every session contains large winners;
   drawdown in this game is produced entirely by imperfect selection.
   `protocol101_stage1_g4_forced_oracle_feasibility` (2026-07-19):
   forced-daily oracle DD $0 at k=1/2/4; slot-skill frontier at 2/day:
   top-10% DD ≤ $38, top-25% ≤ $143, top-50% ≤ $1,702, random $19,327.
4. **Real strategies sit near the no-skill floor.** The best April-era
   result and the best current heuristic (+$15,953 pooled, z=2.15) both
   carry drawdowns comparable to or exceeding their total profit —
   because real errors are regime-correlated, unlike synthetic pickers.
5. **Conclusion:** any small fixed or equity-relative cap either rejects
   every real convex 0DTE strategy or is meaningless. Drawdown must be
   judged (a) relative to profit earned and (b) against account ruin —
   exactly matching the owner's trader spec ("convexity welcome",
   single-contract, risk-scaled).

## Proposed replacement text — G4 (v2)

> **G4 Drawdown (v2, owner-signed 2026-MM-DD):** a candidate passes G4
> iff BOTH hold on the strict one-account serial simulator, fees applied:
>
> **(a) Profitability-proportional drawdown (Calmar floor):** pooled
> fee-adjusted net PnL divided by pooled maximum drawdown >= 1.0.
> (Pooled = across all CV test folds, chronological concatenation.)
>
> **(b) Ruin backstop:** on every individual fold, strict-serial equity
> never falls below $5,000 (50% of the $10,000 starting cash).
>
> **Report-only diagnostic (not gating):** pooled max drawdown as a
> fraction of the matched-policy random-null median drawdown from the
> current null-canary artifact, to visualize risk-axis skill.
>
> Rationale: hindsight measurement proved no absolute or equity-relative
> cap can separate skill from noise in this game (oracle DD = $0 at any
> mandated frequency; no-skill DD floor $4.7k-$38k >> any sane cap).
> Clause (a) demands that every dollar of drawdown is purchased with at
> least a dollar of realized profit; clause (b) caps account destruction
> regardless of eventual recovery. Neither clause penalizes convex
> concentration, per the owner's trader specification.

## Proposed replacement text — holdout sentence

Replace:

> fee-adjusted PnL > 0, max DD <= $1,500, and result within the 90%
> bootstrap CI implied by CV

with:

> fee-adjusted PnL > 0; drawdown judged by the same owner-signed G4 rule
> (v2: Calmar >= 1.0 and equity floor $5,000) active for this candidate
> generation, under identical fee/stress and one-account semantics; and
> the result within the 90% bootstrap CI implied by CV. A holdout result
> wildly ABOVE expectations remains an audit trigger, not a celebration.

## Sanity checks of the proposed rule against known results

| Strategy | PnL / DD | G4 v2 verdict | Correct? |
|---|---|---|---|
| Random selection (any cadence) | negative PnL | fail (a) | ✅ no skill fails |
| Current best heuristic (skew) | +$15,953 / ~$20k+ | fail (a), Calmar ~0.75 | ✅ consistent with its z=2.15 G2 failure |
| Synthetic top-50% selector | +$16k min / $1.7k max | pass (a) Calmar ≥ 19, pass (b) | ✅ modest real skill passes |
| Hypothetical convex winner: +$60k pooled, $45k DD, fold min equity $6.1k | Calmar 1.33 | pass | ✅ convexity not punished |
| Hypothetical grinder: +$8k pooled, $12k DD | Calmar 0.67 | fail | ✅ risk-inefficient edge correctly gated |

## Owner sign-off checklist

- [x] Calmar floor at `1.0` (alternatives: 0.75 lenient / 1.5 strict)
- [x] Ruin floor at `$5,000` = 50% of starting cash (alternatives:
      $4,000 / $6,000)
- [x] Null-relative drawdown kept report-only (not gating), to avoid
      duplicating G2's job
- [x] Holdout sentence replacement approved
- [x] Effective for all Stage-1 candidate generations from signature date

Signature: OWEN HEIDENREICH Date: 07-19-2026
