# Protocol101 Stage-2 Objective & Gates — PROPOSAL (awaiting owner sign-off)

Status: DRAFT for owner review. Per the staging plan, stage-2 work does not
begin until the owner signs this document. It reuses the entire fair
foundation unchanged: menu-v2 corpus (304 pass sessions), governed loader,
protected holdout (2025-05-16..2025-06-30), refit-null methodology, and the
strict serial simulator.

## Why stage-2 exists (established, not assumed)

The stage-1 verdict (PROTOCOL101_STAGE1_VERDICT.md) proved that entry selection
alone contains real signal (win-rho ~0.11, selection z above null) but cannot
control drawdown or ruin: a single-contract long option ridden toward expiry
loses up to 100% of premium, and an entry-only model has no mechanism to stop
it (exp008: an outright account bankruptcy in a hostile-regime fold, alongside
+$9.5k/+$10.2k profitable folds at <=42% drawdown). Loss truncation is an exit
decision by definition. Stage-2 adds the exit.

## What stage-2 is

Given a position already opened (by a fixed stage-1 entry policy or a simple
entry rule), a model observes the position's evolving state each minute and
decides HOLD or EXIT-NOW, scored against the actual raw quote path we already
own and verify. Holds become genuinely adaptive (2 minutes or 2 hours),
decided in-flight — the behavior the owner described. This is the v3 mechanism
that demonstrably added value (project_v3_layer3_works).

## Objective

Maximize fee-adjusted net PnL of the ENTRY+EXIT system through the strict
serial simulator, subject to the constraints below. The exit policy is scored
by the improvement it produces over a fixed hold-to-policy-deadline baseline on
the SAME entries — so the experiment isolates the exit's contribution.

## Data / methodology law (unchanged from stage-1)

- Governed loader only; accepted non-holdout sessions; holdout locked.
- Chronological expanding-window folds, 1-session embargo.
- Exit labels/paths computed from the raw per-minute quote path already in the
  processed rows (no new data; no lookahead — the exit at minute t may use only
  the position's state through minute t).
- Nulls: a permutation/refit exit-null (random exit timing at matched exit
  rate) is the no-skill reference; the exit must beat it.
- >=3 seeds, worst-seed floor.

## Proposed gates (a stage-2 exit policy is promotion-worthy only if ALL hold)

| Gate | PROPOSED threshold |
|------|--------------------|
| S1 Beats hold-baseline | Entry+exit fee-adjusted PnL > fixed-deadline-exit PnL on the same entries, pooled and on >= 4/5 folds |
| S2 Drawdown | Max drawdown <= 25% of peak equity on every fold (same relative gate as revised G4) |
| S3 No ruin | Account equity never goes negative on any fold, any seed (hard) |
| S4 Beats random exit | PnL improvement over the hold-baseline exceeds a matched-rate random-exit null by z >= 3 (pooled) and z >= 2 (worst seed) |
| S5 Seed robustness | Worst seed satisfies S1, S2, S3 |
| S6 Frequency preserved | Entry frequency stays in the G7 band 0.3-6.0 trades/day (exits must not gut participation) |
| S7 Loss truncation works | Mean loss on losing trades is materially smaller than the hold-baseline's (the exit's mechanistic job; report the number) |
| S8 Calibration | If the exit emits a probability, ECE <= 0.10 |
| S9 Confirmation | A fresh unused seed independently satisfies S1/S2/S3 |

## Holdout protocol (unchanged)

One shot per promoted entry+exit system, owner override token, test role only;
burns on failure. Same as stage-1.

## Open questions for the owner

1. **Position sizing / premium normalization.** Stage-1 showed "single
   contract" spans a 70x dollar-risk range ($0.50-$35 premiums). Options:
   (a) cap entry premium (e.g. <=$5.00 = $500 at risk); (b) keep single
   contract and let the exit handle risk; (c) size by predicted return.
   Recommendation: (a) premium cap as a simple hard risk rail in stage-2, with
   (c) deferred to a later stage. Your call.
2. **Entry policy for stage-2.** Score exits on top of (a) the best stage-1
   entry model frozen, or (b) a simple always-enter-ATM baseline to isolate the
   exit's contribution cleanly? Recommendation: (b) first for a clean exit
   signal, then (a). 
3. **Model family.** Start with gradient-boosted trees on position-state
   features (v3 used HistGB exits successfully); escalate to a sequence model
   only if learning curves demand it. GPU spend still gated on that evidence.
4. Any gate above you want stricter? S3 (no ruin) is proposed as an absolute
   hard gate — confirm.

## Sign-off checklist

- [ ] Approve stage-2 objective (exit policy improving entry+exit system)?
- [ ] Position sizing: premium cap $5.00, or alternative?
- [ ] Entry policy for isolation: always-ATM baseline first?
- [ ] S3 no-ruin as a hard absolute gate — confirm.
- [ ] Any gate stricter?
