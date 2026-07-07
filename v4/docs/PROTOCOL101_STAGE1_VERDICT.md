# Protocol101 Stage-1 Verdict (2026-07-07)

Stage-1 = supervised adaptive chooser: per minute, pick strike x right x
trade-shape or abstain, trained on the fair menu-v2 corpus, judged against the
owner-approved gates. Eight preregistered experiments (exp001-exp008), each
hashed and committed with the hypothesis fixed before results.

## Headline

The game has real, learnable, calibrated signal — and stage-1 entry selection
alone cannot turn it into a drawdown-controlled strategy, because a
single-contract long option ridden toward expiry loses up to 100% of its
premium and nothing in an entry-only model truncates that loss. This is not a
failure of the signal; it is the precise, evidence-located boundary between
stage-1 (entry) and stage-2 (exit).

## What was established (in order, each from a committed experiment)

1. **Real information exists** (exp001): payoff-trained GBM selection beat a
   refit permutation null at z 3.29; G6/G7/G8 pass. The founding question —
   is there capturable signal in the fair game — is answered yes.
2. **Absolute-dollar risk filters backfire** (exp002): thresholding on
   predicted dollars adverse-selects into the model's noisiest regions;
   signal destroyed.
3. **Rank-sharpening concentrates the winner's curse** (exp003): the extreme
   top of a noisy score distribution is mostly estimation inflation.
4. **Conservative quantile vetting finds nothing** (exp004): pessimistic
   selection collapses trading to near-zero.
5. **The signal is regime-conditional** (exp005 + exp001 fold analysis):
   per-fold selection z is large in magnitude but flips SIGN by regime,
   coherently across seeds; a rolling in-regime window triples per-session
   win-rho (0.02 -> 0.05).
6. **The drawdown disease is the training target, not selection** (exp006 +
   the premium-bias measurement): dollar-PnL rewards premium size
   (corr(premium,|pnl|)=0.37) though premium is uncorrelated with profit
   (corr=0) and carries ~4x the per-trade variance. Training on
   return-on-premium removed the bias: win-rho 0.019 -> 0.115, drawdown -40%.
7. **G4 was infeasible as specified** (G4 feasibility artifact): random
   no-skill trading has ~$7,800 mean per-fold drawdown; the original $1,500
   absolute cap sat below the game's noise floor. Owner revised G4 to
   <=25% of peak equity (relative), signed 2026-07-07.
8. **Even the full synthesis cannot control drawdown** (exp008): return
   target + rolling window + top-15% conviction abstention still produced
   83-141% peak-equity drawdowns and an outright account bankruptcy
   (seed 42 fold 1 ended at -$3,545). Yet the SAME configuration made
   +$9.5k / +$10.2k on favorable folds at 20% / 42% drawdown (fold 3 passes
   the revised G4). Profitability is real and in-regime; ruin is the
   uncontrolled tail.

## Gate status at stage-1 close (best config, exp007/exp008 family)

| Gate | Status | Note |
|------|--------|------|
| G2 beats no-skill | ACHIEVABLE | z above refit null across seeds |
| G6 era guard | WORKS AS CODE | correctly flags regime-bound in hostile eras |
| G7 frequency band | PASS | ~1.6-2.0 trades/day |
| G8 calibration | PASS | ECE 0.02-0.03 |
| G1 profitability | FAIL | in-regime positive, pooled negative on worst seeds |
| G4 drawdown (relative) | FAIL | in-regime <=25% achievable; hostile-regime ruin |
| G5 seed robustness | FAIL | worst seed bankrupts |
| G3 heuristic baseline | NOT BUILT | deferred; belongs with a strategy that clears G1/G4 |
| G9 confirmation | NOT REACHED | only runs after the others pass |

## The single load-bearing conclusion

The two binding gates (G4 drawdown, G5 ruin) fail for one reason with one fix:
**there is no exit.** An entry-only model chooses well but cannot stop a losing
convex position from decaying to zero. Loss truncation is definitionally an
exit decision. This is stage-2, it is what the owner's staging plan reserved
for stage-2, and it is the one mechanism v3 ever proved added durable value
(learned exits, project memory project_v3_layer3_works).

A secondary finding to carry into stage-2: "single contract" is not constant
risk — premiums span $0.50-$35 (a 70x range in dollars at stake), so a
per-trade risk normalization (premium cap or return-space sizing) belongs in
the stage-2 design alongside the exit policy.

## Recommendation

Declare stage-1 complete. Do not chase G1/G4 further with entry-only levers;
the feasibility artifact and exp008 show that path is exhausted. Proceed to
stage-2 (learned exits) under a new owner-signed objective document
(PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md), reusing the entire fair
foundation: same corpus, same governed loader, same holdout, same null
methodology, same serial simulator.
