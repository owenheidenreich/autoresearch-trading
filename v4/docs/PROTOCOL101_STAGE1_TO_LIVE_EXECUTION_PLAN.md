# Protocol101 — Stage-1 to Live: The Execution Plan

Written 2026-07-19, before the sealed exam ("Stage 0") has run. This is
the mechanical playbook: what training actually is at the base level, who
does what, and where every freeze and gate sits, from the first training
batch to live paper trading. Root reference: PROTOCOL101_TRADER_CHARTER.md.
Design reference: PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md.

Roles (fixed): **Codex** executes (goal mode, one narrow goal per batch,
later nightly automation). **Fable** verifies (independent checker on
every packet before its verdict enters the ledger), keeps STATUS current,
drafts proposals. **Owner** signs (charter, objectives, promotions,
activations), reads the plain-language paragraph each morning, reviews
loop economics weekly. The maker/checker separation is law: no batch
verdict is accepted on the word of the agent that produced it.

---

## PHASE 0 — Prerequisites (now → sealed exam, ~Aug 5)

Already done: charter drafted; G4 v2 signed; nulls recalibrated; heuristic
baselines run; noise injection + intersection guards built; rehearsal
passed 3/3-pending-07-20; fees grounded.

**Remaining build item (new, do before the exam):** adapt the Stage-1
runner (`run_protocol101_stage1_bounded_hgb_search.py`) to the canonical
contract — canonical features via the certified builders, frozen v1.4
selection semantics, noise injection, intersection guards, 5% daily
breaker, charter metrics (four-bucket outcome distribution, harvest
ratio, underwater duration), packet validator, owner paragraph. Then
SMOKE it on burned days (tiny attempt count) purely to prove the plumbing
runs end-to-end — explicitly not evidence, labeled `plumbing_smoke_only`.
One Codex goal. Without this, exam-day cannot be training-day.

**Gate out of Phase 0:** sealed exam routes `canonical_v1_4_sealed_confirmed`
AND charter signed AND training design signed. Freeze at this boundary:
canonical transform, v1.4 selection contract, gate thresholds, null
bands, fold boundaries, the five-hypothesis set, the runner code hash.

---

## PHASE 1 — Stage-1 training (the entry hunt)

### What one attempt actually is, at the base level

1. The governed loader yields the 271 accepted sessions (fifteen months).
2. The canonical feature builder produces, for every minute of every
   session and every one of the 42 ladder slots, the 34 allowed dials —
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
   contract only when it clearly beats the runner-up; ties to nearest
   ATM; intersection guards decide tradability.
5. The strict serial simulator replays those decisions chronologically —
   one account, $10k, single contract, affordability enforced, 5%
   daily circuit breaker, forced flat 15:55 — producing an equity curve,
   a trade list, and per-trade price paths.
6. The gate battery scores the result: G1 profit, G2 beats-luck (vs the
   recalibrated null bands), G3 beats the skew heuristic (+$15,953),
   G4 v2 Calmar+floor, G5 worst-seed, G6 era, G7 frequency, G8
   calibration. Plus charter metrics, report-only: four-bucket outcome
   distribution, harvest ratio, underwater duration.

That is one attempt: fit, score, decide, replay, gate. Nothing in it is
tunable at run time; everything variable was frozen at Phase-0 exit.

### What a batch is

One hypothesis (e.g. H2 = context + skew composites), 21 attempts:
7 exit shapes x 3 seeds, run primary and conservative. A batch is one
Codex goal with this fixed skeleton:

1. Preregister: config hashed, ledger entry written BEFORE any result.
2. Run all 21 attempts to completion (no early stopping on peeking).
3. Emit the packet: per-attempt gate table, pooled results, charter
   metrics, path diagnostics, cost accounting (compute/tokens/wall),
   and the OWNER PARAGRAPH — five plain-language sentences: what was
   tried, what happened, what it means, what it does not mean, what
   happens next.
4. Mechanical packet validator (dumb gate: files present, row counts,
   hashes, side-effect flags false) — the batch is not "done" because
   Codex says so; it is done when the validator exits 0.
5. Independent verification (Fable, or a read-only Codex goal Fable
   spot-audits): claims vs artifacts. Only then does the routing verdict
   enter the ledger.

### Batch order and cadence

Batch 1 = H0, run manually with owner + Fable watching (Article-2 rule:
never schedule what has not been reliable by hand). Then H1, H2, H3, H4 —
one at a time, nightly cadence once the process is boring. All five run
regardless of interim results. Expected wall time: days per batch;
the whole of Stage-1 in roughly two to three weeks.

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
- All five reject → preregistered fork: same-vendor data plane / Path B /
  owner review. Gates do not move.

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
4. **Recorder-day decision validation**: the frozen candidate runs on all
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
| Phase 0 exit | Transform, v1.4 contract, gates, nulls, folds, hypothesis set, runner hash | Sealed exam (mechanical) + owner (charter, design) |
| Each batch start | Its preregistered config + ledger entry | Validator (mechanical) |
| Phase 1 exit | Nothing new — hypotheses simply routed | — |
| Phase 2 step 3 | Candidate manifest (weights+config+contracts) | Owner |
| Phase 3 start | Stage-2 objective doc; entry model permanently | Owner |
| Phase 4 exit | Shadow-validated candidate; paper activation | Owner |
| Phase 5 exit | Paper-validated candidate | Owner (real-money decision) |

Standing loops that never stop across all phases: recorder + sealing,
daily replay-determinism check, parity drift alarm, STATUS page, weekly
cost-per-accepted-change review, morning owner paragraph.

## Honest timeline (calendar, not promises)

Exam ~Aug 5 → Stage-1 batches ~Aug 5–25 → (if candidate) hardening ~1
week → shadow build+run ~2 weeks → paper ~4+ weeks → real-money decision
plausibly late October / November. If Stage-1 routes to Stage-2 instead:
add ~3–4 weeks for the objective doc + exit training before hardening.
Every date here is downstream of evidence; none is a commitment.
