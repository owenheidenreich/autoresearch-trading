# Protocol101 Full Trader Graph V2 — Revised Plan (Rev C)

Status: **REVISED NON-EXECUTABLE PLAN — PLAN OF RECORD PENDING CONSOLIDATION
AND OWNER SIGNATURE (see §10: a consolidated single-authority document is now
a required deliverable before signature)**

Prepared: 2026-07-28 by Fable (Claude Opus), at owner request.
Rev B: 2026-07-28, after Codex adversarial review of Rev A. Codex's four
blocking findings are dispositioned in §0. This revision loop (brief → Fable
revision → Codex review → Rev B) is itself the planning-graph discipline
working; it is recorded here deliberately.
Rev C: 2026-07-28, after a trader-identity review of the owner's settled
decisions against the signed Trader Charter. Four owner questions were asked
and answered (§0.5). Rev C adds two hard-safety product amendments, a
profile tripwire, a guardrail-philosophy commitment, floor-ablation evidence,
census additions, and a documented-limitations register.

Predecessor documents (both preserved unchanged as planning history):

- Codex planning brief:
  `PROTOCOL101_FULL_TRADER_GRAPH_V2_PLANNING_AND_FABLE_REVIEW_BRIEF_2026_07_28.md`
  (SHA-256 `765591b130ccc6d1e9a12d65252808027a08c02118d9be2ace22bd2834ee2e3f`,
  independently recomputed and verified 2026-07-28).
- Rev A of this document (superseded by this Rev B in place; Rev A content is
  reconstructable from git history).

This document is not a training authorization, not a Goal prompt, not an
owner-signed contract, and not evidence that a profitable model exists.

---

## 0. Disposition Of Codex Review Findings (Rev A → Rev B)

| Codex finding | Disposition | Where |
|---|---|---|
| B1: Census circular (uses entry-quality families before they're defined; oracle ceiling undefined without exit/serial semantics; outer-train firewall leaky across folds) | **ACCEPTED.** Path-label definitions now freeze BEFORE the census (new FT2-04); census is threshold-independent (distributions + Pareto frontier + power inputs); hard-stop only on genuine impossibility; census sessions must appear in NO outer-test slice of any fold (proved by manifest intersection) | §4 Phase 0 |
| B2: AUC ≤ 0.53 arbitrary | **ACCEPTED WITH MODIFICATION.** The signed 0.55 ceiling stands. Point-estimate passing is replaced by evidence-aware passing: session-block bootstrap 95% CI upper bound must sit below 0.55; a CI straddling the ceiling returns `insufficient_paired_evidence`, not pass or fail. Primary gates are decision transfer, label/opportunity bias, and boundary behavior; AUC is one component. Positive and null control discriminators validate the harness itself | §5.2 |
| B3: Horizon trim discards short-trade resolution without evidence | **ACCEPTED.** Approved horizons {3, 5, 10, 20, 45, 90, session} restored. FT2-10 may prune only with pre-validation redundancy evidence. Multiplicity note corrected: heads of one jointly trained model are not independent hypotheses; the multiplicity family counts candidates and composer/guardrail variants | §6.1 |
| B4: Blanket fp16 unsafe | **ACCEPTED.** Integer/tick storage for identities, timestamps, prices; fp32 for precision-sensitive context; fp16 only for normalized tensors after a quantified prediction-and-decision equivalence test | §6.2 |
| Q1: Core-lane failure must not be read as product failure | **ACCEPTED.** Preregistered bounded rich-lane feasibility route on core-lane failure | §5.3 |
| Q2: Recorder claim inconsistent; sealing governance unverified | **ACCEPTED AND CONFIRMED BY NEW VERIFICATION.** Fresh check (2026-07-28): collection gates exist for 07-10/13/14/22/28 only (gaps on other sessions); sealed-day assignment is active — 06-30/07-01/07-02 `burned`, 07-10 `validation`, 07-13/14/20 `development`, 07-28 `sealed`. Rev A's "collecting through the present" claim was wrong as stated. A manifest-only paired-evidence inventory node (FT2-24) now precedes feature admission | §2 V12, §5.2 |
| Q3: Delta-over-brief two-document authority problem | **ACCEPTED.** A consolidated single narrative authority + machine-readable Graph V2 JSON is now a required pre-signature deliverable (FT2-01). The catch-all signature box is deleted | §4, §10 |
| Q4: Compute minimum-detectable-improvement before GPU spend | **ACCEPTED.** MDE computation is part of FT2-11 and a prerequisite for the first GPU tranche | §6.4, §8 |

---

## 0.5 Trader-Identity Review (Rev C) — Owner Questions Asked And Answered

A dedicated review tested the settled decisions against the signed Trader
Charter from a 0DTE trader's perspective, hunting for specs that would
produce a *different trader* than the one the Charter defines. Four findings
required owner decisions; all four were put to the owner on 2026-07-28 and
answered:

| # | Finding (Charter conflict) | Owner decision (2026-07-28) |
|---|---|---|
| T1 | No per-trade size limit: affordability allows up to $35.00 mid ($3,500 premium at risk on $10k); a single open trade can lose 8-35% of the account, and the realized-only breaker cannot stop an open position. Violates Charter commitment 3 in spirit ("worst single day 3-5%") | **5% per-trade premium cap** — premium at risk plus round-trip fee ≤ 5% of session-start equity ($500 now; scales). Becomes hard-coded safety (§6.0) |
| T2 | Daily breaker is realized-only (verified: simulator v5 `DAILY_LOSS_BASIS = raw_realized_net_pnl_at_occupancy_exit`); after -$400 realized, a new $500-premium trade makes the worst day ≈ -9% | **Budget-aware entries** — a new entry is legal only if realized session loss + new premium at risk ≤ the 5% daily budget. Sizes down when losing; worst day bounded ≈ -5%. Breaker itself unchanged (§6.0) |
| T3 | Four-bucket Pickles profile is report-only: a candidate could pass every PnL/risk gate while being a different animal (e.g., 40% losers) | **Profile tripwire** — grossly-off distribution routes to `owner_decision_required`; never auto-pass, never auto-fail, never an optimizable target (§6.4) |
| T4 | Quality-first screening tuned tight builds a scratch-mill scalper (the Charter's explicit anti-identity) by filtering slow-starting home-run paths | **Loose screen + upside ranks** — guardrails exclude only clearly-bad bottom-tail paths; conservative upside does the ranking; census must report the excluded-winners rate to verify the screen is not eating the big-win column (§6.1) |

Additional Rev C changes not requiring owner questions: floor-ablation
comparator and harvest tripwire (§6.3); census additions (§4 Phase 0);
documented-limitations register (§6.5).

---

## 1. What Changed And Why (Executive Delta vs Codex Brief)

The Codex brief was faithful to the owner's product vision and its governance
skeleton was sound. Its defects were: **no arithmetic, no feasibility
diagnostic, a mega-node design contract, an uncapped feature-admission
search, an uncapped compute commitment, and one real data-contract gap.**
This plan:

1. **Adds a label-side feasibility diagnostic (FT2-05 Opportunity Census)**
   after path-label freeze (FT2-04) and before design ceremony. Days of CPU,
   no model. Hard-stop only for genuine impossibility; otherwise it informs
   the owner decision and calibrates statistical power.
2. **Adds the open-state decision-row extension (15:31→15:55)** as a governed
   machinery deliverable. Verified: decision rows currently end 15:30 ET
   (359/session) while normalized quotes reach 16:00 ET and forced-flat label
   coverage already passes acceptance.
3. **Splits the FT2-10 mega-contract into three separately reviewable
   contracts** (data/tensor/label; entry science; evidence/statistics).
4. **Caps feature admission at 3 preregistered transforms** with
   evidence-aware discriminator passing (§5.2) and a paired-evidence
   manifest inventory first (FT2-24).
5. **Cuts campaign 1 to a single feature lane** with a preregistered
   rich-lane feasibility route if the core lane fails.
6. **Keeps the approved 7 forecast horizons** (Rev A's trim withdrawn).
7. **Amends D29** with compute tranches, an MDE prerequisite, and a 14-day
   owner check-in. The 12-trial plateau survives; unbounded spend does not.
8. **Decides the hardware split with numbers** (§3, §8): M5 for census,
   label builds, machinery pilots, and all inference; leased GPU for the two
   inner autoresearch loops; checkpoint/resume proof before any paid loop.
9. **Adds the daily-stop × premium interaction and floor-slippage realism**
   to census and lifecycle evidence.
10. **Extends the signed D1 V2 exposure-matching law to the full-ladder
    game** rather than inventing a parallel control standard.
11. **Expresses Graph V2 in the existing Graph V1 controller schema** and
    requires one consolidated authority document before signature (FT2-01).

Where this document is silent, the Codex brief stands — but per Q3, silence
is resolved by the consolidated FT2-01 document before anything executes,
so no executing node ever interprets two documents.

---

## 2. Verification Record (Claims Tested Against The Repo)

All checks run 2026-07-28 on this machine. Nothing below is from memory.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| V1 | Trader Charter signed; Pickles profile; 5% daily stop; SPY floor; four-bucket distribution | **CONFIRMED** | `training/contracts/PROTOCOL101_TRADER_CHARTER.md` (signed 07/25/2026) |
| V2 | G4 v2 = pooled Calmar ≥ 1.0 + $5,000 per-fold equity floor; old caps measured infeasible | **CONFIRMED** | `PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md` (signed) |
| V3 | G8 report-only until an action-conditioned calibration gate is preregistered and owner-signed | **CONFIRMED** | `PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md` (signed) |
| V4 | Sync decision authorizes exactly 17 features; Family C quarantined at AUC 0.550524 vs 0.55 ceiling; D-family 0.502582, E-family 0.500492; evidence = 864 option-eligible minutes over 5 days | **CONFIRMED** | `synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md` (signed) |
| V5 | D1 V2 incremental-edge law with 8-dimension exposure matching | **CONFIRMED** | `PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md` |
| V6 | Graph V1 controller state exists (loop budgets, receipts, terminals, gate dominance), parked before FT-10 | **CONFIRMED** | `PROTOCOL101_FULL_TRADER_GRAPH_V1.json`; runner `v4/scripts/run_protocol101_full_trader_graph.py` |
| V7 | Governed corpus = 301 pass-only sessions, 2025-01-02 → 2026-03-31, decision rows 09:32→15:30 ET (359/session) | **CONFIRMED** | `..._15mo_training_scope_acceptance/summary.json` |
| V8 | Lifecycle gap: decision rows stop 15:30; manage window runs to 15:55 | **CONFIRMED (nuanced)** | Normalized quotes reach 16:00 ET; `label_spot_forced_flat_coverage: true`. Gap is the open-state decision-row contract, not raw data |
| V9 | 21×2 ladder snapshot tensors, masks, contract IDs, per-policy labels already exist per decision row | **CONFIRMED** | `v4/dataset/spxw_0dte_neural.py` (~L1014-1120). Missing: identity-keyed 90-min per-contract history + path labels |
| V10 | Simulator v5: $10k start, $3 round-trip fee, 5% session-start-equity daily stop, forced flat 15:55 | **CONFIRMED** | `v4/model/protocol101_serial_simulator_v5.py` (L84-98) |
| V11 | Hardware: Apple M5, 16 GB RAM, 10 cores | **CONFIRMED** | `sysctl` |
| V12 | Recorder collection state | **CORRECTED (Rev B)** | Collection gates pass for 2026-07-10, 07-13, 07-14, 07-22, 07-28; **no gates for other July sessions in this root**. Sealed-day assignment active (`protocol101_sealed_day_assignment/last_assignment_run.json`, run 2026-07-28T20:32Z): 06-30/07-01/07-02 `burned`; 07-10 `validation`; 07-13/14/20 `development`; 07-28 `sealed`; others `unassigned_pre_rule`. **Paired days are class-governed, not freely usable.** Rev A overstated availability |
| V13 | H0-H3 / D1 / M0-M1 negative-history narrative | **CORROBORATED** | Graph V1 `historical_negative_evidence`; D1 amendment; G8 revision audit (420 model units verified; 7 G1-profitable rows; none surviving) |

**Correction to the brief (retained from Rev A):** G4 v2, the charter's
four-bucket reporting, the 5% daily stop, and D1 V2's matching law are
denominator-independent and carry into the Full Trader unchanged. Only
G1/G2/G3/G5/G6/G7/G8 need re-derivation for the new action space.

---

## 3. The Arithmetic The Brief Was Missing

Planning estimates. FT2-30 must measure actuals; deviation > 2x is a
mechanical finding.

**Decision surface.**
- Flat-state decisions: 301 × 359 = **108,059 minutes**.
- Open-state extension: ≤ 25 rows/session (15:31→15:55), ≤ 7,525 minutes.
- Ladder: 21 strikes × 2 rights = **42 slots + WAIT**.
- Label build: ~15-30 eligible contracts/minute → **~1.6-3.2M
  contract-minute path labels**. One-time, CPU-parallel, checkpoint per
  session. M5-feasible (hours).

**Trades are the statistical bottleneck, not minutes.**
- At 1-3 trades/day: **300-900 trades corpus-wide**; an outer-fold test
  slice (~60 sessions) holds **60-180 trades**.
- Lifecycle specialization sees a few hundred frozen-entry OOF trajectories;
  broad pretraining on all eligible contract paths (brief §12.1) is what
  makes that survivable.
- Consequences (now law in §6.4): session-level effective sample size,
  session-block bootstrap, `insufficient_evidence` as a distinct terminal,
  and an MDE computation before GPU spend.

**Tensor and memory.**
- Per flat decision: market history 90 × ~24 ≈ 2.2k values; per-contract
  causal paths 42 × 90 × ~8 ≈ 30.2k; cross-sectional snapshot ≈ 0.6k →
  **~33k values/decision**.
- Naively materialized fp32: **~14 GB** — exceeds 16 GB RAM headroom. The
  tensor contract therefore mandates identity-keyed contract-path storage
  with per-batch assembly (paths shared across adjacent rows; ~10-20x
  dedup) and the §6.2 precision rules. Storage precision per §6.2 (Rev B):
  ticks/ints exact, fp32 context, fp16 only after equivalence proof.

**Training time.**
- Primary model: shared contract encoder + market encoder + 1-2 cross-ladder
  attention layers + multi-horizon heads ≈ **1-5M parameters**.
- M5 (MPS): est. 5-15 min/epoch → **3-15 h/run**. One pilot: fine.
  Campaign: not fine.
- Campaign: ≥ 12 serious trials × refits × seeds × two loops →
  **order 100-300 GPU-hours** leased (10-30x M5).
- Operational constraint (verified): recorder occupies ~05:30-13:15 local on
  collection days; no concurrent local training (brief §19.1). Local
  training = evenings/weekends only. **Campaign runs on leased GPU; M5 runs
  everything else** (§8).

---

## 4. Revised Graph V2 Topology

Graph V1's controller schema (receipts, `loop_budgets`, terminal states,
`gate_dominance_requirements`, drift tripwires) is retained verbatim. V2 is
a topology amendment in the same JSON schema. **(NEW)** = not in the Codex
brief; **(SPLIT)** = split from a brief mega-node; **(REV B)** = changed by
the Codex review.

### Phase 0 — Consolidation, label freeze, feasibility

- **FT2-01-CONSOLIDATED-AUTHORITY (NEW, REV B).** Merge the Codex brief,
  this Rev B, and the Codex review disposition into ONE narrative authority
  document plus ONE machine-readable Graph V2 JSON. The owner signs that
  single document. No node executes off a delta chain.
- **FT2-04-PATH-LABEL-FREEZE (NEW, REV B).** Freeze, before any census
  statistic is computed: exact path-label definitions for all six families
  (dollar + return-on-premium), the oracle-selector semantics the census
  will use (entry at executable ask; exits evaluated at executable bids
  under a small preregistered set of transparent oracle exit rules — e.g.,
  best-achievable bid by horizon, hold-to-flat — with serial one-account
  replay and the daily stop applied through simulator v5), and the census
  session set. **Census sessions must appear in NO outer-test slice of any
  fold; the FT2-04 packet must prove the intersection is empty from the
  fold manifests.**
- **FT2-05-OPPORTUNITY-CENSUS (REV B).** Label-side only; no model. On the
  frozen census sessions, report **threshold-independent** evidence:
  distributions of each path-quality family by phase, premium band, and
  moneyness; the Pareto frontier of quality-vs-upside; qualifying-minute
  share as a **curve across candidate guardrail levels** (not one number);
  ceiling PnL of the preregistered oracle rules under serial replay with
  fees, the daily stop, **and the Rev C hard-safety rules (5% premium cap +
  budget-aware entries, §6.0)**; P5's share of each ceiling; measured
  label-build compute; and the statistical-power inputs FT2-11 needs
  (session-level variance components, MDE curves vs trade count).
  Rev C additions (from the trader-identity review):
  - **Excluded-winners rate:** of all oracle big-win paths (fee-adjusted
    return ≥ +40%, the Charter's big-win column), the fraction excluded at
    each candidate guardrail level — the direct measurement of "is the
    comfort screen eating the home runs."
  - **Friction by premium band:** round-trip cost (spread crossing + fees)
    as a fraction of premium, per band — showing where the game is even
    playable (cheap wings can carry ~25%+ friction inside current
    tradability guards).
  - **Expected trade-rate curves** at each guardrail level under the §6.0
    rules, feeding §6.1's joint-conservatism control.
  - Hard stop (`STOP-OWNER-DECISION`) **only** for genuine impossibility:
    unusable labels, no executable post-fee opportunities, or ceilings
    indistinguishable from fee drag under session-block CIs.
  - Otherwise the census cannot kill the program; it informs FT2-21 owner
    approval and calibrates FT2-11.

### Phase A — Scientific contracts (SPLIT)

- **FT2-08-DATA-TENSOR-LABEL-CONTRACT.** Corpus roles, fold manifests,
  identity-keyed contract-path storage, tensor schema and masking, storage
  precision rules (§6.2), open-state row contract, hash discipline.
  Inherits FT2-04's frozen labels unchanged.
- **FT2-10-ENTRY-SCIENCE-CONTRACT.** Targets on the approved horizons
  {3, 5, 10, 20, 45, 90, session} (pruning only with pre-validation
  redundancy evidence), composer semantics, deterministic training-side
  guardrail calibration (reusing census code paths and distributions),
  uncertainty-aware WAIT, and the action-conditioned calibration gate
  required by signed G8 v2 (WAIT-rate reliability, selected-contract regret
  vs forecast, exit-decision reliability).
- **FT2-11-EVIDENCE-STATISTICS-CONTRACT.** Session-level ESS; session-block
  bootstrap; preregistered multiplicity family (candidates × composer and
  guardrail variants × admission attempts — **not** per-head, per B3);
  full-ladder D1 V2 extension (§6.4); the `insufficient_evidence` boundary;
  and the **minimum detectable improvement** computation that gates the
  first GPU tranche.
- **FT2-20-PARALLEL-DESIGN-REVIEW / FT2-21-OWNER-DESIGN-APPROVAL.** Three
  contracts reviewed as three documents; a defect in one does not reopen
  the other two (bounded `design_repair` budget per contract).

### Phase B — Feature admission and machinery

- **FT2-24-PAIRED-EVIDENCE-MANIFEST-INVENTORY (NEW, REV B).** Manifest-only
  inventory of every paired recorder/historical day: sealed-day class
  (burned / validation / development / sealed / unassigned), completeness,
  gate status, and **governance availability for admission testing**.
  Verified reality this node must formalize: only 07-13, 07-14, 07-20 are
  currently `development`; 07-10 is `validation`; the original parity days
  are `burned`; 07-28 is `sealed`. If available development days are too
  few for CI-based admission, the honest output is a collection plan, not a
  weaker test.
- **FT2-25-FULL-LADDER-FEATURE-ADMISSION (REV B).** Per §5.2.
- **FT2-26-INDEPENDENT-FEATURE-ADMISSION-AUDIT.** Unchanged.
- **FT2-28-LIFECYCLE-ROW-BUILD (NEW).** Build + governed acceptance of the
  15:31→15:55 open-state rows.
- **FT2-30-ENTRY-HARNESS-PILOT.** As in the brief, plus: publish actuals
  for every §3 estimate; run the fp16 equivalence test (§6.2); and if GPU
  is authorized, **prove checkpoint/resume on the leased GPU with a ≤ $10
  dry run before any paid loop** (this project has lost GPU work to lease
  expiry, workspace wipes, and duplicate processes before).
- **FT2-31-INDEPENDENT-ENTRY-MACHINERY-ACCEPTANCE.** Unchanged.

### Phases C-F

Unchanged from the brief (entry autoresearch → entry freeze → lifecycle
design/autoresearch → four-box → freeze → confirmation → holdout → shadow →
paper), with these amendments:

1. **Compute tranches + MDE prerequisite + 14-day owner check-in** (§8).
2. **Lifecycle evidence adds floor-slippage realism** (§6.3).
3. **Campaign 1 runs one feature lane, with the preregistered rich-lane
   feasibility route on core-lane failure** (§5.3).

---

## 5. Feature Admission Law (Rev B)

### 5.1 The conflict, restated precisely

The signed 17-feature contract authorizes near-ATM **aggregate** composites
(D-family) and internal delta/gamma (E-family) only — nothing per-contract.
Family C (per-slot option-price paths) failed its discriminator at
**0.550524 vs the frozen 0.55 ceiling**. The Full Trader's mandatory feature
floor requires per-contract premium history. The brief correctly refuses to
silently amend the signed contract; this section governs how the amendment
may be earned.

### 5.2 Admission test (Rev B — replaces Rev A's 0.53 rule)

- **At most 3 canonical-price transforms**, preregistered together with
  rationale before any is tested (e.g., quantized log-premium changes;
  premium normalized by straddle mid; rank-normalized within-ladder
  premium). Every attempt, pass or fail, enters the multiplicity family.
- **Discriminator harness validation first:** a positive control (known
  vendor-identifying field) must be caught and a null control (known-clean
  feature) must pass, or the harness itself is rejected.
- **Evidence-aware ceiling test:** the signed 0.55 ceiling stands. A
  transform passes the discriminator component only if its
  **session-block bootstrap 95% CI upper bound is below 0.55** on
  leave-one-session-out evaluation. A CI straddling 0.55 returns
  `insufficient_paired_evidence` — route to FT2-24's collection plan, never
  to pass or fail. (This is what Rev A's 0.53 was groping for: passing must
  be demonstrable, not point-estimate luck. The arbitrary second constant
  is withdrawn.)
- **Primary gates** (all required, AUC is one component, not a lone veto):
  decision-transfer agreement ≥ 0.99 (the signed decision's stable-probe
  standard); no label/opportunity-correlated drift; sane boundary/stress
  behavior; economic materiality.
- **Governance:** only days FT2-24 classes as available (currently
  `development`) may be used. Sealed and burned days are untouchable.
- If all 3 transforms fail → `STOP-OWNER-DECISION` with the honest options:
  product minimum unmet, or collect more development-class paired days and
  retest under the same preregistration.

### 5.3 One lane for campaign 1 (amends D18, Rev B adds the escape route)

Campaign 1 trains **parity-core + admitted canonical price history** only.
The ladder-rich challenger is deferred. **Rev B addition (Codex Q1):** if
the core lane terminates `no_genuine_entry_signal`, the routing packet must
present the owner a preregistered, bounded **rich-lane feasibility option**
(admission-tested rich features, one bounded feasibility run, no silent
expansion) before any conclusion about the Full Trader product is recorded.
Core-lane failure is evidence about the core lane, not about the product.

---

## 6. Science Defaults (For The Split Contracts)

Defaults for FT2-08/10/11 to adopt or overturn with recorded reasons.

### 6.0 Hard-Safety Product Amendments (Rev C, owner-approved 2026-07-28)

These amend the brief's §5.3 hard-coded-safety list. They are product
contract changes, owner-decided in the §0.5 review, and must be implemented
**identically** in the historical serial simulator and every live/shadow
guard (same-game law — a cap that exists offline but not live, or vice
versa, is a synchronization defect):

1. **Per-trade premium cap (T1):** premium at risk plus the round-trip fee
   overlay ≤ **5% of session-start equity** ($500 at $10,000; scales with
   the account). Contracts above the cap are masked from the eligible
   action set exactly like unaffordable contracts — a safety mask, never an
   alpha feature.
2. **Budget-aware entries (T2):** a new entry is legal only if
   `realized_session_loss + (new premium at risk + fee) ≤ 5% daily budget`.
   Consequences to document, not hide: the effective ladder narrows as
   realized losses accumulate (natural size-down when losing); a day can
   reach forced-WAIT before the breaker fires (soft landing); the breaker
   itself keeps its signed realized-only basis.
3. **Interaction notes:** the census (FT2-05) must apply both rules to its
   opportunity accounting; the entry model's masks reflect them at every
   minute; worst-day arithmetic under both rules is bounded ≈ -5% realized
   with no open position able to add more than the remaining budget.

### 6.1 Entry (FT2-10) — Rev B/C

- Horizons: **{3, 5, 10, 20, 45, 90, remaining-session} — the approved set,
  unchanged.** Rev A's trim is withdrawn (Codex B3: short-trade resolution
  at 3-8 minutes is an owner emphasis; pruning requires pre-validation
  redundancy evidence, recorded in the FT2-10 packet, before any pruned
  variant is registered).
- All six path-property families from the brief (§9.3) retained.
- **Guardrail philosophy (T4, owner-decided):** guardrails are a
  **bottom-tail exclusion**, not a top-tier selection. They exclude clearly
  bad paths (calibration anchored near the worst ~quartile of training-role
  path quality, exact quantiles FT2-10's to set); the conservative-upside
  stage does the actual ranking. Rationale: 0DTE long entries start
  underwater by construction (spread crossing), and many Charter-profile
  big wins chop red before the move; a tight comfort screen structurally
  builds the scratch-mill scalper the Charter forbids. FT2-10 must justify
  its chosen operating point against BOTH census curves: expected trade
  rate AND excluded-winners rate (§4 Phase 0).
- Guardrail calibration: quantile-based thresholds from training-role
  distributions per market phase, frozen per fold before outer access,
  emitted with hashes. The census (FT2-05) supplies the distributions and
  the code path; FT2-10 must reuse, not reimplement.
- **Joint-conservatism control (Rev C):** three abstention mechanisms stack
  (guardrails, conservative-upside-after-fees, uncertainty-margin WAIT).
  FT2-10 must set them jointly against a census-derived design trade-rate
  band — a calibration target, not a gate — so independently-paranoid
  settings cannot silently starve the trader into `insufficient_evidence`.

### 6.2 Data/tensor (FT2-08) — Rev B storage precision law

- Identity-keyed contract-path store with per-batch tensor assembly.
- **Exact types for exact things:** contract identities, strikes, session
  dates, timestamps as integers; option prices as integer ticks
  (SPXW 0.05/0.10 increments) or fixed-point.
- **fp32** for precision-sensitive market/context fields (SPX levels,
  Greeks, path differences).
- **fp16 only** for normalized model-input tensors, and only after FT2-30
  demonstrates quantified prediction-and-decision equivalence (identical
  actions on a reference slice; score drift below a preregistered bound)
  vs the fp32 reference.
- Contract-identity continuity exactly as brief §6.2 (recentring must never
  silently re-map a slot's history).

### 6.3 Lifecycle (FT2-60 series) — Rev C additions

- Open-state rows 15:31→15:55 mandatory (FT2-28).
- **Floor-slippage realism:** every lifecycle packet reports realized exit
  price vs committed floor (gap-loss distribution). No intraminute
  protection is claimed; the evidence shows what crossing actually cost.
- **Floor ablation (Rev C):** the exit comparator set must include
  **learned-exit-without-floor**. The floor must earn its keep on evidence;
  if floor-on materially degrades tail capture vs floor-off, that is a
  finding, not an implementation detail.
- **Floor breathing room (Rev C):** the floor equation may be explicitly
  time- and profit-conditional (loose early in a trade, tighter late /
  deep in profit). An upward-only ratchet on noisy minute bids otherwise
  converts every post-MFE retrace into a forced exit and chokes the
  Charter's big-win column, which requires enduring -20/-30% retraces from
  peak mid-trade.
- **Harvest tripwire (Rev C):** if the harvest ratio (realized / peak
  available PnL) collapses under floor-on relative to floor-off, route
  `owner_decision_required` — protection is being bought with the dream.
- Broad causal pretraining precedes specialization on frozen-entry OOF
  trajectories (brief §12.1 retained).

### 6.4 Evidence/statistics (FT2-11) — Rev B

- Effective sample size and all bootstraps at **session level**.
- Multiplicity family preregistered before Phase C: admission attempts,
  inner-loop shortlist candidates, both lifecycle families, registered
  composer/guardrail variants. **Not per-head** (Codex B3 accepted: heads
  of one jointly trained model are shared-parameter outputs, not
  independent hypotheses; what multiplies is selectable variants).
- **Minimum detectable improvement before spend:** using census variance
  components, compute the smallest incremental edge over P5/matched-random
  detectable at the required confidence given projected trade counts. If
  the MDE exceeds any plausible edge, that is a `resource_owner_decision_
  required` finding **before** the first GPU tranche, not after.
- `insufficient_evidence` is a distinct terminal from `no_genuine_signal`.
- **Four-bucket profile tripwire (T3, owner-decided, Rev C):** every
  candidate packet reports the Charter's four-bucket outcome distribution
  (big win / scratch / small loss / big loss, fee-aware thresholds). A
  grossly-off profile — big-loss share above 2x the Charter's ~2%, or a
  big-win column near zero — routes `owner_decision_required` at the
  freeze gate. Never auto-pass, never auto-fail, and **never an optimizable
  target** (a gated profile invites scratch-harvesting; the Charter itself
  warns against optimizing the profile directly).
- Full-ladder incremental-edge controls extend the signed D1 V2 matching
  table unchanged (entry-intent exact; executed count ≤ 5%; call/put TV
  ≤ 5%; moneyness TV ≤ 6% on ATM/NEAR/WING from slot 10; premium ≤ 10%;
  holding ≤ 10%; occupancy ≤ 10%; outcome-blind; post-replay verification
  fails closed; timing AND contract choice randomized feature-free).

### 6.5 Documented-Limitations Register (Rev C)

Known, accepted limitations of campaign 1 — written down so nobody
rediscovers them as surprises or silently "fixes" them mid-campaign:

1. **Minute cadence:** decisions fire on completed minutes; the trader is
   structurally ~1 minute late to breakouts. This biases learnable edge
   toward structural moves (VWAP, trend, phase behavior) and away from
   scalps — consistent with the Charter's anti-scalper identity, but it
   means breakout-scalp alpha is out of scope by construction.
2. **Myopic entry:** WAIT triggers when nothing qualifies *now*; there is
   no "a better setup is likely at 10:30" forecast. Phase-calibrated
   selectivity is the patience proxy for campaign 1. A look-ahead
   opportunity head is Stage-3 (sequential agent) territory; adding it now
   would balloon scope (established project discipline: no wave-2 features
   during wave-1 implementation).
3. **No intraminute protection:** the floor and all exits evaluate at
   completed-minute bids; intraminute gaps are absorbed as slippage and
   measured (§6.3), not prevented.
4. **Entry model does not see own recent PnL:** flat-state features are
   market/ladder state; the trader has no tilt and no hot-hand — by
   design. Account state affects only masks (affordability, cap, budget).

---

## 7. Self-Improvement Loop, Bounded

Retained from Rev A, unchanged by review:

1. **Trial ledger discipline:** a "serious trial" is declared in the
   registry with its hypothesis before results are seen; the plateau
   counter is computed by the controller from the ledger, never
   self-reported.
2. **The loop cannot touch its own stopping rule.** Plateau constant,
   tranche sizes, and the evidence contract are graph state, not loop
   state. Improvement authority covers weights, architecture, loss,
   optimizer, regularization, calibration implementation — nothing else.

---

## 8. Hardware And Compute Plan

| Workload | Where | Why |
|---|---|---|
| Opportunity census, label builds | **M5 (CPU-parallel)** | Hours-scale, one-time, checkpointable |
| Machinery pilots, smoke trainings, fp16 equivalence test | **M5 (MPS)** | One 3-15 h run overnight/weekend is fine |
| Entry + lifecycle inner autoresearch loops | **Leased GPU (Akash or equivalent)** | 100-300 GPU-hour campaign; M5 would take months of nights around the recorder window |
| All inference, shadow, live | **M5** | 1-5M-param forward pass is milliseconds; the live risk is feature-build latency — FT2-30 measures it |

Rules:

- **Tranche authorization (amends D29):** GPU spend in owner-authorized
  tranches (proposed **$150/tranche**). Prerequisites for the FIRST
  tranche: the ≤ $10 checkpoint/resume dry-run proof **and** the FT2-11 MDE
  computation showing the campaign can detect a plausible edge. Tranche
  exhaustion → `resource_owner_decision_required`.
- **14-day owner check-in** regardless of tranche state.
- **No local training during recorder hours** (recorder occupies
  ~05:30-13:15 local on collection days — verified).
- Every GPU run checkpoints durably such that a killed lease loses ≤ 1
  trial.

---

## 9. Settled-Decision Deltas (vs Brief §27) — Rev B

| ID | Status | Change |
|---|---|---|
| D07 | **RETAINED (Rev B reverts Rev A)** | Approved horizons {3, 5, 10, 20, 45, 90, session} stand; FT2-10 may prune only with pre-validation redundancy evidence |
| D18 | **AMENDED** | Campaign 1 = parity-core lane only; preregistered bounded rich-lane feasibility route on core-lane failure |
| D29 | **AMENDED** | 12-trial plateau retained; plus tranches, MDE prerequisite, 14-day check-in |
| D36 | **CLARIFIED** | No early M5 size caps; §3 estimates published now; FT2-30 measures actuals |
| D39 | **REVISED (Rev B)** | Census is a feasibility/power diagnostic after label freeze; hard-stop only on genuine impossibility; otherwise informs owner approval |
| D40 | **NEW** | Open-state rows 15:31→15:55 as governed deliverable |
| D41 | **REVISED (Rev B)** | Admission: 3 preregistered transforms; harness validated by positive/null controls; session-block 95% CI upper bound < 0.55; transfer ≥ 0.99; bias/boundary gates primary; `insufficient_paired_evidence` is a legal outcome |
| D42 | **REVISED (Rev B)** | Census sessions must appear in NO outer-test slice of any fold (manifest-proved), replacing Rev A's leaky "outer-train only" rule |
| D43 | **NEW** | Floor-slippage is first-class lifecycle evidence |
| D44 | **NEW** | Full-ladder controls extend signed D1 V2 matching law unchanged |
| D45 | **NEW (Rev B)** | One consolidated authority document + machine-readable Graph V2 JSON before owner signature (FT2-01) |
| D46 | **NEW (Rev B)** | Paired-evidence manifest inventory (FT2-24) precedes feature admission; sealed-day governance is binding |
| D47 | **NEW (Rev B)** | Storage precision law: ints/ticks exact, fp32 sensitive, fp16 only after equivalence proof |
| D48 | **NEW (Rev C, owner-answered)** | Hard per-trade premium cap: premium + round-trip fee ≤ 5% of session-start equity; enforced as a safety mask identically offline and live |
| D49 | **NEW (Rev C, owner-answered)** | Budget-aware entries: realized session loss + new premium at risk ≤ 5% daily budget; breaker keeps signed realized-only basis |
| D50 | **NEW (Rev C, owner-answered)** | Four-bucket profile tripwire at freeze gates: grossly-off profile → `owner_decision_required`; never a gate, never an optimizable target |
| D51 | **NEW (Rev C, owner-answered)** | Guardrails are bottom-tail exclusion; conservative upside ranks; FT2-10 justifies its operating point against census trade-rate AND excluded-winners curves |
| D52 | **NEW (Rev C)** | Floor ablation (learned-exit-without-floor comparator), time/profit-conditional floor allowance, harvest-ratio tripwire |
| D53 | **NEW (Rev C)** | Documented-limitations register (§6.5) is part of the signed authority; removing a limitation mid-campaign is a scientific change requiring owner review |
| All others | **RETAINED** | As in the brief |

---

## 10. Owner Sign-Off (Rev B)

Per Codex Q3, the catch-all "everything not amended" box is **deleted**. The
signature target is the FT2-01 consolidated document, which must contain the
complete merged authority. Until FT2-01 exists, nothing executes. The
substantive decisions the consolidated document will carry:

- [ ] Census as feasibility/power diagnostic after label freeze (D39 Rev B).
- [ ] One-lane campaign 1 with rich-lane escape route (D18).
- [ ] Approved horizons retained (D07).
- [ ] Admission law: caps + CI-based ceiling + transfer/bias primary (D41).
- [ ] Compute tranches at $150 + MDE prerequisite + 14-day check-in (D29).
- [ ] Open-state row extension (D40).
- [ ] Sealed-day governance binding on admission evidence (D46).
- [ ] Storage precision law (D47).
- [x] Per-trade premium cap 5% (D48) — **owner-answered 2026-07-28** (§0.5 T1).
- [x] Budget-aware entries (D49) — **owner-answered 2026-07-28** (§0.5 T2).
- [x] Profile tripwire (D50) — **owner-answered 2026-07-28** (§0.5 T3).
- [x] Loose screen + upside ranking (D51) — **owner-answered 2026-07-28** (§0.5 T4).

The four answered items still appear in the consolidated FT2-01 document for
formal signature; the answers above are their recorded provenance.

**Plain-language decision memos:** every owner-gate packet in Graph V2 must
open with a one-page memo in plain English: the question being decided, the
options, the evidence in trader terms, the recommendation, and what happens
next under each choice. The owner is the judge of the product, not the
machinery; no owner decision may require reading model internals to make.

---

## 11. Recommended First Goal (One Task, One Decision)

**FT2-01 Consolidated Authority** is now the first Goal: merge brief +
Rev B + review dispositions into one narrative document and one Graph V2
JSON in the V1 controller schema, with zero new science invented during
consolidation (any discovered contradiction is surfaced as an owner
question, not resolved silently). It ends with the single signable document.

FT2-04 (label freeze) and FT2-05 (census) follow as the second and third
Goals, in that order.

---

## 12. Highest Allowed Claim

> Protocol101 Full Trader Graph V2 has a review-hardened, trader-identity-
> checked revised plan (Rev C) awaiting consolidation into a single signable
> authority.

No model is designed, trained, selected, or eligible for anything by this
document. No protected resource, broker path, paid download, or runtime
change is authorized by this document.
