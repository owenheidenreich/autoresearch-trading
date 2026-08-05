# Protocol101 — Stage-1 to Live: The Execution Plan

Written 2026-07-19 and reconciled 2026-07-25 after scoped clean-window
synchronization evidence. This is the mechanical playbook: what training
actually is at the base level, who
does what, and where every freeze and gate sits, from the first training
batch to live paper trading. Root reference: PROTOCOL101_TRADER_CHARTER.md.
Design reference: PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md.

Goal-sizing control:
`PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md`. That document
controls how this playbook is divided into separate Codex goals. Where this
document describes several actions inside one batch, the goal-sized system's
RUN -> GATE -> AUDIT separation is binding.

Roles (fixed): **Codex** executes (one narrow goal per batch, later scheduled
only after the manual path is reliable). An **independent read-only checker**
verifies every packet before its verdict enters the ledger and keeps STATUS
current; this role may be Fable or a separate Codex task that did not produce
the batch. **Owner** signs charter, objectives, promotions, and activations.
The maker/checker separation is law: no batch verdict is accepted on the word
of the agent that produced it.

---

## PHASE 0 — Prerequisites

Complete: charter signed; G4 v2 signed; governed five-fold scaffold;
noise-injection module; intersection-guard audit; fees grounded; scoped
synchronization evidence and owner decision passed for the 17-feature initial
contract; exact-contract preflight passed; canonical HGB executor, disposable
smoke, independent validation, deterministic G1-G8 aggregator, and runner
freeze completed.

**Current gate:** separate owner authorization for the real H0 offline batch.
The legacy masked-v2 runner is historical evidence and must not launch new
training.

**Gate out of Phase 0:** scoped synchronization decision, charter, and training
design signed; exact-contract preflight green; plumbing smoke independently
verified. Freeze at this boundary: exact 17-feature allowlist, quarantine list,
selection mechanics, training-only epsilon rule, gates, exact-contract nulls,
fold boundaries, four-hypothesis set, and runner code hash.

---

## PHASE 1 — Stage-1 training (the entry hunt)

### What one attempt actually is, at the base level

1. The governed loader yields the 301 accepted training-scope sessions.
2. The canonical feature builder produces, for every minute of every
   session and every one of the 42 ladder slots, only the authorized 17
   features —
   with measured cross-feed noise INJECTED into the quote-derived dials
   (so the model can never profit from precision that does not survive
   the feed difference).
3. For CV fold k (5 chronological expanding windows, 1-session embargo):
   a bounded gradient-boosted tree (depth/trees capped) learns, from the
   TRAIN sessions only, to score "how good is entering this contract at
   this minute under exit shape j" — the label comes from replaying that
   candidate's actual price path to the shape's exit, at the pessimistic
   fill rung, fees included.
4. The trained model then scores the TEST sessions it has never seen.
   Scores become decisions only through the frozen v1.4 semantics:
   trade only when conviction clears the noise dead-band; name a
   contract only when it clearly beats the runner-up; ties fall back to
   nearest ATM; intersection guards decide tradability. Each fold's
   score-noise epsilon is calibrated on training-only noise replicas and
   frozen before unseen sessions are scored.
5. The strict serial simulator replays those decisions chronologically —
   one account, $10k, single contract, premium-plus-fee affordability,
   5% of session-starting-equity daily circuit breaker, forced flat 15:55 —
   producing an equity curve,
   a trade list, and per-trade price paths.
6. The gate battery scores the result: G1 profit, G2 beats-luck (vs the
   exact-contract recalibrated null bands), G3 beats the exact-contract
   fixed heuristic,
   G4 v2 Calmar+floor, G5 worst-seed, G6 era, G7 frequency, G8
   calibration. Plus charter metrics, report-only: four-bucket outcome
   distribution, harvest ratio, underwater duration.

That is one attempt: fit, score, decide, replay, gate. Nothing in it is
tunable at run time; everything variable was frozen at Phase-0 exit.

### What a batch is

One hypothesis (e.g. H1 = context + skew composites), 21 policy/seed
configurations evaluated over all five folds: 7 exit shapes x 3 seeds x
5 folds = 105 fitted units. The primary batch is mandatory. A conservative
batch is conditional, separately preregistered, and never automatic.

Each hypothesis uses three separate Codex goals defined by the goal-sized
system:

1. `RUN`: preregister and execute all 105 units, then stop.
2. `GATE`: mechanically aggregate G1-G8 without fitting, then stop.
3. `AUDIT`: independently accept, void, or block the packet and freeze its
   routing verdict, then stop.

Only an accepted AUDIT freeze may authorize the next hypothesis.

### Batch order and cadence

Batch 1 = H0, run manually with owner and an independent checker watching.
Then H1, H2, H3 — one at a time, scheduled only after the manual path is
reliable. All four run regardless of interim results.

### Freezes inside Phase 1

- No threshold, feature, epsilon, fold, or gate may change. Ever.
- A discovered BUG aborts the batch; the fix is committed; the batch
  reruns under a new attempt id with the old one marked void — bugs are
  never patched mid-flight.
- The do-not-retest list is enforced at preregistration time.
- Harness improvements discovered during Stage-1 are PROPOSALS for the
  next candidate generation, owner-signed — never applied mid-stage.

### Gate out of Phase 1 (per hypothesis routing)

- `eligible_offline_candidate`: passed G1–G8 → proceed to Phase 2 with
  THIS model, whose weights are hashed and frozen at this moment.
- `stage2_learned_exits_candidate`: entry signal real (G2 z>=3.0 pooled,
  G5 worst-seed z>=2.0) but full gates fail with the salvageable-MFE
  signature → unlocks Phase 3 planning (Stage-2).
- `rejected_no_real_signal`: recorded; next hypothesis.
- All four reject → review the separately parity-tested opening-structure
  proposal, same-vendor data plane / Path B, or a different strategy class.
  Gates do not move.

---

## PHASE 2 — Candidate hardening (per candidate, ~1 week)

Runs for any `eligible_offline_candidate` (from Stage-1 fixed-shape
training, or later from Stage-2 composed training).

1. **G9 confirmation seed**: one fresh never-used seed must independently
   satisfy G1/G2/G4. Fails → candidate dead, no retry.
2. **Holdout, one shot**: owner override token opens the locked
   2025-05-16→06-30 window. Charter/G4-v2 criteria. Result outside the
   CV-implied confidence band in EITHER direction triggers audit. A
   failed holdout burns the candidate permanently.
3. **Candidate freeze**: model weights + config + contract hashes bound
   into a candidate manifest. From here, any change of any kind = a new
   candidate that restarts Phase 2.
4. **Recorder-day / no-order shadow decision validation**: the frozen candidate runs on all
   inspectable recorder days (dev days + spent sealed days) against
   vendor same-days — decision agreement >= the battery bars (0.98
   action / 0.99 slot). This is the per-candidate parity exam; the
   canonical machinery makes passing likely, but it is verified, not
   assumed.

**Gate out of Phase 2:** all four steps green + owner signature on the
candidate manifest.

---

## PHASE 3 — Stage-2, learned exits (if/when evidence unlocks it)

Trigger: a `stage2_learned_exits_candidate` packet, or an owner decision
to improve a passing candidate's harvest ratio.

1. **Owner-signed Stage-2 objective doc first** (project law). Charter
   binds it: the exit model's reward = fee-adjusted PnL with explicit
   dual mandate — loss truncation (the scratch engine) AND tail capture
   (the big-win engine, harvest ratio as report card). No profit caps.
2. Mechanics at base level: the ENTRY model is frozen and never
   retrained. Its trades' minute-by-minute causal paths (PnL velocity,
   realized vol since entry, MFE decay — the validated H3a feature
   lineage) become the dataset. A second bounded model learns
   hold-vs-exit each minute. Baselines: the 7 fixed shapes. Same
   simulator, same gates, same nulls (re-run for the composed policy),
   walk-forward only, no overlap between exit-training days and
   evaluation days (the April overlap lesson, now law).
3. Composed candidate (frozen entry + learned exit) → Phase 2 hardening,
   identically.

Freeze rule: entry/exit entanglement is forbidden — if Stage-2 evidence
suggests the entry model should change, that is a NEW Stage-1 candidate
generation, owner-signed, back through Phase 1.

---

## PHASE 4 — Live shadow (the last unpaid exam, ~2 weeks)

1. **Build the true event-driven shadow daemon** (one Codex goal): the
   frozen candidate consumes the recorder's live stream in real time,
   makes minute-boundary decisions, submits NOTHING. Reuses the
   boundary-ledger fidelity work; preregistered before it runs.
2. Run >= 10 sessions. Pass bars, preregistered: settled live-vs-replay
   decision agreement >= 0.995; zero guard violations; recorder capture
   quality green throughout.
3. The shadow period doubles as fill-model groundwork: live quote stream
   at decision moments is archived for the paper-fill reconciliation.

**Gate out of Phase 4:** shadow packet green + owner signature to arm
paper trading. This signature is the single most consequential one in
the plan; nothing before it has touched an order path.

---

## PHASE 5 — Guarded paper trading (~4+ weeks)

1. Runtime posture change (owner-executed): paper orders enabled inside
   the guard stack — single contract, intersection-guard tradability,
   5%-of-equity daily breaker, forced flat, kill switch documented.
2. Run >= 20 sessions. Every fill reconciled against the simulator's
   predicted fill (the standing simulator-audit loop begins); every
   session's decisions reconciled against same-day replay (the parity
   smoke alarm keeps running).
3. Reports carry the charter lines: SPY benchmark comparison,
   four-bucket outcome distribution, harvest ratio, underwater duration,
   daily-breaker events.
4. Preregistered paper verdict: fills within modeled bounds; decision
   parity holds; PnL consistent with the CV-implied band (above OR below
   band = audit). Paper proves execution + continued parity — profit
   expectations remain owned by the charter's tiers.

**Gate out of Phase 5:** paper packet green → the real-money decision,
which belongs to the owner alone, with Fable's honest brief and no
recommendation pressure either way. Out of scope of this document.

---

## The freeze map (one table)

| Boundary | What freezes there | Who signs |
|---|---|---|
| Phase 0 exit | Scoped feature contract, selection mechanics, gates, nulls, folds, hypothesis set, runner hash | Owner (sync decision, charter, design) + independent smoke verification |
| Each batch start | Its preregistered config + ledger entry | Validator (mechanical) |
| Phase 1 exit | Nothing new — hypotheses simply routed | — |
| Phase 2 step 3 | Candidate manifest (weights+config+contracts) | Owner |
| Phase 3 start | Stage-2 objective doc; entry model permanently | Owner |
| Phase 4 exit | Shadow-validated candidate; paper activation | Owner |
| Phase 5 exit | Paper-validated candidate | Owner (real-money decision) |

Standing loops that never stop across all phases: recorder + sealing,
daily replay-determinism check, parity drift alarm, STATUS page, weekly
cost-per-accepted-change review, morning owner paragraph.

## Timeline

No calendar promise is authoritative. The signatures, preflight, smoke, and
machinery freeze are complete, so separately owner-authorized H0 may begin
immediately. Every later date remains
downstream of evidence. Candidate hardening, shadow validation, and guarded
paper operation still take place in sequence; synchronization readiness does
not collapse those stages.
