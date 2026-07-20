# Protocol101 — Current Status (one page, plain language)

**This is the single source of truth for "where are we."** Updated every
working session. If it conflicts with memory or older docs, this wins.
Last updated: 2026-07-19.

## The project in two questions

1. **Does the model see the same game in training data as it will live?**
   ← everything since June. Nearly done.
2. **Can it make money in that game?** ← not started. Begins after Q1 closes.

## Question 1 scoreboard

| Check | Status |
|---|---|
| Vendor data ≡ IBKR recordings (translated) | ✅ Rehearsal PASSED on fresh days incl. CPI (07-18) |
| Replay determinism | ✅ Proven automatically daily |
| Live real-time ≡ recording | 📏 Measuring this week; first result Friday 07-24 |
| FINAL EXAM (sealed days, one shot) | ⏳ ~Aug 5, after collection ends 08-04 |

## Running automatically right now (no attention needed)

- Recorder: captures every session through Aug 4.
- Shadow logger: model's decisions written every 15 min (no orders, ever).
- Boundary ledger: per-minute record of what data had arrived (for the
  live-vs-recording comparison).
- Sealing: sessions from 07-13 on (except dev days 07-13/14/20) lock into
  the vault at 13:30 daily, untouched until the final exam.

## Needs YOU (the complete list)

1. **Friday 07-24 after close**: run the end-of-week batch with Fable —
   live-vs-recording result for 07-20, rehearsal add-on (completes the
   practice-exam requirement), sealed-week review, vendor downloads.
2. **G4 gate revision**: measurement DONE (forced oracle + skill
   frontier); signable draft ready —
   PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md (Calmar >= 1.0 +
   $5,000 equity floor). Needs only your signature.
3. **Sealed-day report fix**: one instruction to Codex — its 13:30 report
   must show counts only for sealed days, never decision numbers.
4. **ThetaData**: month is active; batch remaining downloads before it
   expires (Friday works).

## Day classes (why some days are "special")

- **Burned** (Jun 30–Jul 2): used to design everything. Can't prove
  anything anymore. Repair/diagnosis surface.
- **Validation/Dev** (Jul 10, 13, 14, 20): fresh days we're allowed to
  look at. Used for the rehearsal and this week's live comparison.
- **Sealed** (Jul 21–24, 27–31, Aug 3–4 expected): the locked final exam.
  Nobody looks until it runs. Expected ~11 days incl. FOMC (Jul 28/29).
  (Jul 15–17 were lost to an internet outage — already replaced by the
  Aug extension.)

## Prior-campaign knowledge (mined 07-18, ready for Stage-1)

PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md holds the April backtest
campaign's full edge ledger. Headlines: the durable April edge was
OPENING STRUCTURE (gap + first-15 acceptance/range; PF 1.32 → 1.54
gated), NOT VWAP-reclaim (dropping vwap_reclaim_state improved results);
honest directional baseline PF 1.132 over 780 days; a 28-item
do-not-retest list; and the protocol-number decoder (051/054/066/081/
101/113/155/160). Best April edges map to H0 — the hypothesis we
expected to be a mere control. The lost protocol-farm history (Apr 28 –
May 20) was RECOVERED from a single 58k-line Codex transcript
(PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md): all 160 protocol
numbers decoded, key correction — Protocol051's "edge" is a model score
margin, not dollars. Founding transcript hash-preserved in
_history_backup/ (untracked).

## The Trader Charter (the root goal, in plain language)

PROTOCOL101_TRADER_CHARTER.md — drafted 07-19 from the owner's answers:
a convex hunter with a survival guarantee. Droughts must be FLAT (abstain,
don't force), never deep holes. Year-one floor: beat SPY or index-and-quit;
dream band 3x-10x, no profit caps in the final system (Stage-2's job).
Daily circuit breaker: 5% of current equity ends the session. Survival
outranks everything. Awaiting owner markup/signature.

## The execution plan (exam day -> live paper, mechanics + freezes)

PROTOCOL101_STAGE1_TO_LIVE_EXECUTION_PLAN.md — the full playbook:
what one training attempt IS at base level, batch structure (Codex
executes, Fable verifies, owner signs), five phases with a freeze map
(Phase 0 prereqs -> Stage-1 batches -> candidate hardening -> Stage-2
learned exits -> live shadow -> guarded paper). NEW pre-exam build item:
adapt the Stage-1 runner to the canonical contract + plumbing smoke, so
exam day = training day.

## After the final exam (if it passes)

Stage-1 training on 15 months of history: five feature-set hypotheses,
gates against recalibrated luck-bands, everything already built (nulls,
baselines, noise injection, pessimistic guards; training design doc
amended with April priors + G4 v2, awaiting signature alongside it). First hopeful clue already logged: a simple put/call-skew rule
showed the strongest signal in project history (+$15,953, z=2.15 — real
but not yet significant). If training finds a candidate: confirmation
seed → holdout → replay validation → live shadow → guarded paper trading.

## Standing rules (the short version)

Never read the sealed vault. Never modify frozen contracts/preregistrations
(hashes in the arc handoff). No paper orders until the post-exam phase with
a validated candidate and the owner present. All tests batch at week-end;
weekdays are collection only.

Deep detail: PROTOCOL101_CANONICAL_GAME_ARC_HANDOFF_2026_07_11.md (design
history) and PROTOCOL101_SHADOW_ASOF_WEEK_PLAN_2026_07_18.md (this week).
