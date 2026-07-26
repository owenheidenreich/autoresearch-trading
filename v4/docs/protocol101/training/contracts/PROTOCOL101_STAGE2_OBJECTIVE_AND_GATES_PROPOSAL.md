# Protocol101 Stage-2 Objective & Gates — PROPOSAL (awaiting owner sign-off)

Status: OWNER REVIEW IN PROGRESS (2026-07-07). Resolved so far: learned exit
approved; premium not capped, prioritize OTM, model discovers (Q1); gate
thresholds agreed incl. S3 no-ruin hard (Q4); model family is a tree-first
default, no owner input required (Q3). One item awaiting final confirmation:
the entry-isolation order for scoring exits (Q2, recommended (b) always-simple
entries first). Stage-2 experiments do not begin until the owner gives the
final go. Reuses the entire fair foundation unchanged: menu-v2 corpus (304
pass sessions), governed loader, protected holdout (2025-05-16..2025-06-30),
refit-null methodology, and the strict serial simulator.

## Why stage-2 exists (established, not assumed)

The stage-1 verdict (PROTOCOL101_STAGE1_VERDICT.md) proved that entry selection
alone contains real signal (win-rho ~0.11, selection z above null) but cannot
control drawdown or ruin: a single-contract long option ridden toward expiry
loses up to 100% of premium, and an entry-only model has no mechanism to stop
it (exp008: an outright account bankruptcy in a hostile-regime fold, alongside
+$9.5k/+$10.2k profitable folds at <=42% drawdown). Loss truncation is an exit
decision by definition. Stage-2 adds the exit.

This matches trading reality (owner, 2026-07-07): 0DTE traders almost never
hold to expiry — theta decay is too powerful and the risk/reward collapses as
expiry approaches; holding to expiry is considered foolish. The one exception,
a market-on-close "lotto" in the final ~10 minutes, is a different game and is
explicitly OUT OF SCOPE for this project. So the learned exit is not an
optional refinement; it is the normal, correct behavior of the trader we are
building, and its absence is exactly why stage-1 could not control risk.

## What stage-2 is (and how it differs from stage-1's fixed exits)

Stage-1 already had exits — but FIXED, parametric ones. Every trade-shape in
the menu encodes an exit rule: a stop (exit if the bid falls X%), a target
(exit if it rises Y%), and a time cap (exit at max-hold or the 15:55 forced
flat). The convex shapes 4-6 have stop=1.00, i.e. NO stop — "ride it to the
forced flat." Those are precisely the "hold too long" trades the owner calls
foolish, and they are what bankrupted the hostile-regime fold in exp008.

Stage-2 replaces the fixed stop/target/time rule with a LEARNED, per-minute
decision. Given a position already open, a model observes its evolving state
each minute (current bid vs entry, elapsed time, unrealized return, live market
context) and outputs HOLD or EXIT-NOW, scored against the actual raw quote path
we already own and verify. Holds become genuinely adaptive — 2 minutes or 2
hours, decided in-flight on the position's behavior rather than a preset number.
This is the v3 mechanism that demonstrably added durable value
(project_v3_layer3_works), and it is the direct realization of the "disciplined
trader who exits when the thesis plays out or breaks" that the owner described.

## Objective

Maximize fee-adjusted net PnL of the ENTRY+EXIT system through the strict
serial simulator, subject to the constraints below. The exit policy is scored
by the improvement it produces over the stage-1 FIXED-EXIT baseline on the SAME
entries — i.e. against the best fixed stop/target/time rule, not against
holding to the deadline (which we already know is foolish). Isolating the exit
this way means any gain is attributable to the learned exit decision alone.

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

## Open questions and their resolutions

### 1. Premium / capital efficiency — RESOLVED (owner, 2026-07-07): no hard cap

The owner's guidance: do NOT cap entry premium to a fixed dollar amount.
Instead, prioritize OTM options as the more prudent use of capital (you pay
less premium for the same directional exposure), noting OTM 0DTE options
typically cost under ~$1,300/contract ($13.00) unless VIX is elevated — and
let the model DISCOVER the best moneyness/premium trade-off on its own rather
than being told.

This is well-supported by stage-1: the return-on-premium objective already
steers selection toward cheaper, higher-return contracts on its own (measured:
the cheapest premium quartile had ~9.4% mean return vs ~1.4% for the priciest).
So stage-2 keeps the return-on-premium objective and adds NO strategy-level
premium cap. The only premium-related control is a plain affordability/sanity
rail (already in the simulator: a trade can't cost more than available cash),
which is risk hygiene, not a strategy constraint. The model is free to learn
that OTM is usually the right call.

### 2. Which positions does the exit model practice on? (experimental control)

To study an exit rule you must first decide WHAT positions it is exiting from —
because the final P&L blends two things: how good the ENTRY was and how good the
EXIT was. If we let the real stage-1 entry model choose the positions, a good
result could come from good entries, good exits, or luck, and we can't tell
which. So there are two ways to run it:

- **(a) Real entries:** the exit model manages the positions the stage-1 entry
  model actually picked. Realistic, but entry-quality and exit-quality are
  tangled together — you can't cleanly credit the exit.
- **(b) Fixed simple entries (recommended first):** hold the entry constant and
  dumb — e.g. always buy a standard ATM (or fixed-OTM) option on a regular
  cadence — so EVERY position is the same kind. Then any improvement in P&L is
  caused ONLY by the exit decision, because the entry never varied. This is a
  controlled experiment: it isolates the exit's contribution.

The analogy: to test a new brake, you bolt it onto a standard car, not onto a
race car that also has a new engine — otherwise you can't tell whether the lap
time improved because of the brake or the engine. Recommendation: run (b) first
to prove the exit adds value in isolation, then (a) to measure the full
entry+exit system. Your call on whether to accept that order.

### 3. Model family — this is a default I will use, not a question for you

Clarification (it was written ambiguously before): this is my recommended
technical default, not a decision I need from you. I will start the exit model
as a gradient-boosted tree over the position's current-state features (the same
tree family v3 used successfully for exits, and appropriate for this data
scale). I will only escalate to a heavier "sequence model" — one that reads the
full minute-by-minute path of the position rather than a snapshot — if the
simple model plateaus AND the evidence (learning curves) shows that modeling the
sequence is the binding constraint. Renting GPUs stays gated on that same
evidence. You are welcome to weigh in, but no input is required; absent an
objection I proceed with the tree-first ladder.

### 4. Gate thresholds — RESOLVED (owner, 2026-07-07): agreed as proposed

Including S3 (no ruin) as an absolute hard gate.

## Sign-off checklist

- [x] Approve stage-2 objective (learned exit improving the entry+exit system) — owner: yes.
- [x] Premium: no hard cap; prioritize OTM; model discovers (Q1) — owner-resolved.
- [ ] Entry isolation order (Q2): confirm (b) simple-fixed entries first, then (a) real entries. (Harness will support both via a flag either way.)
- [x] S3 no-ruin as a hard absolute gate, and all proposed thresholds (Q4) — owner: agreed.
- [x] Model family (Q3): tree-first default, escalate only on evidence — no owner input required.

Final go to build + run stage-2 experiments: pending the Q2 confirmation above.
