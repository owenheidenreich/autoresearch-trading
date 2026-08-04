# Walking-Skeleton Learnings → Real-Campaign Ledger

**STATUS: LIVING METHODOLOGY LEDGER, NOT AUTHORITY.** The single place every
Walking-Skeleton learning is recorded with an explicit *disposition* — so each
one either propagates into the real campaign (via a governed amendment or a
process change), is a tracked watch-item, or is deliberately accepted. Nothing
is allowed to just evaporate. Update this as each stage runs.

Disposition types:
- **DESIGN CHANGE** — alters the real model/contracts → governed amendment
  folded into the consolidated authority (new hash, checker, review).
- **PROCESS CHANGE** — alters how the real campaign is run (data, orchestration).
- **WATCH-ITEM** — a risk to verify at a known point in the real campaign.
- **VALIDATION** — confirms a design/process choice; keep doing it.

| # | Learning | Evidence | Disposition | Status |
|---|---|---|---|---|
| L1 | **Entry gate was structurally unsatisfiable.** The composer required positive *pessimistic-decile* (q10) upside — a risk-free entry, which no real trade offers → 0 trades for any model. | Option-D Stage-1: realized q10 MFE negative at every horizon; 0/4191 contracts pass; even the no-screen control abstains 100%. | **DESIGN CHANGE** → FT2-10 entry-gate convexity amendment: gate on *expected* upside after fees; keep q10 for ranking; downside → exit + cap + breaker. Codex added a required statistical correction: gate on the **calibrated session-clustered lower-confidence bound on the conditional MEAN** (not an individual-outcome conformal quantile, which would recreate the q10 defect). | **OWNER-SIGNED (2026-07-31, Owen Heidenreich).** Authority `edcbee06`→`82d9573e` folded into binding authority; sign-off receipt in `protocol101_ft2_10_entry_gate_convexity_amendment/owner_approval_receipt.json`. Claude verified independently: authority + graph hashes reproduced; provenance owner-signed→convexity clean; anti-regression rule `entry_gate_uses_satisfiable_central_tendency_statistic` real (forbids "q10" in the gate) & passing; checker 32/32 re-run green; gate now passes **552 proposals** (was 0/4191). L1 CLOSED. |
| L2 | **Loss control belongs to the exit model, not the entry gate.** Entry scans market-wide for a setup (broad, once, from forecasts); exit is locked on one position — its greeks, that contract, tick-by-tick — the only component that can see and cut a developing loss. | Census: dumb-entry + good-exit captured ~45% of the ceiling; L1 mechanism. | **DESIGN CHANGE / carry-forward** — embedded in L1; a binding design principle for the FT2-60 lifecycle contract when written (the exit must be strong; it owns loss control). | Captured in L1 amendment; flagged for FT2-60 design. |
| L3 | **Conformal calibration is very conservative on thin data.** 45 sessions → the calibrated q10 upside was ~5× more pessimistic than raw (0.265 → 0.051), which compounds over-abstention. | Option-D replay predictions (raw vs calibrated). | **WATCH-ITEM** — more development data tightens calibration; strengthens the case for the 2022-2023+ backfill (D57) and must be reflected in the D54 entry-feasibility gate and MDE-before-spend. | Flagged; feeds D54/D57. |
| L4 | **The downstream composer gates are still untested.** Uncertainty-WAIT, q90-regret, and the action-conditioned gate never ran — nothing survived the entry gate to reach them. | Option-D: 0 contracts past stage-2. | **WATCH-ITEM** — after the L1 fix, the Stage-1 re-run exercises them for the first time; watch for a second-order over-abstention (a repeat of L1 one layer down). | **RESOLVED (2026-07-31): second-order over-abstention CONFIRMED, but it is NOT an L1-class defect — see L9.** Post-fix Stage-1 rerun: 552 proposals cleared the new entry gate, then **every** downstream numeric gate individually rejected all 552; action-conditioned gate was *unavailable*. Still 0 BUY. |
| L9 | **The remaining 0-BUY is a weak/thin model correctly failing conservative confidence gates — NOT a structural defect.** After the L1 fix the block moved one layer down: the uncertainty-margin gate requires the predicted directional cluster-gap (≤0.08, median 0.015) to exceed the model's own 90% conformal prediction-error bar (`wait_margin`=0.7038 ≈ model_gap_error) — the dumb GBT's error is ~10× its signal, so it can't clear ANY confidence gate. Additionally the action-conditioned gate is unavailable (19 realized WAIT outcomes < 50 minimum). Unlike L1 (unsatisfiable for *any* model on *any* data), these shrink with a stronger model + more data. | Stage-1 headline_decisions.csv: wait_margin 0.7038 constant vs cluster_gap max 0.078; diagnostic_packet.json individual downstream pass counts all 0; action gate WAIT-outcomes 19<50. | **WATCH-ITEM / PROCESS CHANGE.** Owner selected fork (b): an explicitly quarantined forced-BUY harness that cannot alter or bypass the real composer in runtime. That harness exercised Stages 2–3; the real signal-to-error feasibility issue remains open for the governed data-scaling curve before GPU spend. | **PLUMBING FORK RESOLVED (2026-07-31); REAL-MODEL WATCH-ITEM OPEN.** Real composer remains byte-frozen and `ABSTAINS`; 552 real A6 proposals yielded 78 deterministic quarantined intents solely downstream. |
| L10 | **Branch-coverage canaries prove control-flow seams, not policy quality or alpha.** The deterministic forced-BUY selection and lifecycle canaries successfully fired learned exit, floor, and forced-flat paths, but they also shape trade mix and P&L. | Stage 2–3 replay: 9 accepted trades, 4 learned exits, 4 floor triggers, 1 forced-flat; descriptive P&L −$552; four buckets 0/4/2/3. | **PROCESS CHANGE** — keep branch canaries in a separately labeled plumbing suite. Never tune entry, lifecycle, floor, or promotion policy from canary-shaped results; the real campaign must report governed performance separately. | Applied to Stage 2–3 packet and audit; carry forward to every later skeleton/parity stage. |
| L11 | **ROC AUC can look strong while rare-event usefulness remains weak.** The throwaway lifecycle classifier's calibration ROC AUC was 0.8372, but average precision was only 0.0664. | `lifecycle_model_manifest.json` in the Stage 2–3 evidence packet. | **WATCH-ITEM** — the real lifecycle campaign must lead with prevalence-aware precision-recall, calibration, and session-clustered evidence; ROC AUC alone is not sufficient evidence of a usable exit model. | Recorded; verify at the governed FT2-60 lifecycle-model gate. |
| L12 | **The historical simulator needs two native exit clocks.** Frozen simulator v5 requires threshold-exit source time to equal realized occupancy time with zero label age, while official 1-second rows have a distinct exchange-event timestamp and nonzero market-quote age. | Independent Stage 2–3 review reproduced the simulator rejection when the raw event clock was supplied directly. The quarantined adapter now preserves actual `exit_source_event_time_ns` and `exit_market_quote_age_ms` in trade metadata and the audit checks exact clock arithmetic/freshness. | **DESIGN WATCH-ITEM** — do not mutate frozen simulator v5 for the skeleton. Before real-campaign acceptance/parity, make source-event time, decision/occupancy time, and quote age first-class schema fields with exact invariant tests. | Skeleton adapter verified; native real-campaign schema requirement remains open. |
| L13 | **Static chart checks do not replace rendered visual QA.** The first 35/35 structural audit passed while the multi-session SPX chart mapped global bar IDs as dense array indexes, causing every displayed date to collapse to 2025-03-10. | Independent local-browser review of `trades-on-spx.html`; fixed exporter now densely reindexes the filtered SPX rows and trade markers. Final audit 38/38 plus rendered span 2025-03-06 through 2025-03-10. | **PROCESS CHANGE** — every owner-facing D60/parity visualization requires both machine checks and an actual local render check covering session/date span and marker alignment. | Applied; regression test and audit rule added. |
| L14 | **Safety cutoffs need exact-boundary tests.** A strict `>` comparison admitted decisions at exactly 15:30 even though the binding last legal decision is 15:29. | Independent Stage 2–3 review of the forced-BUY prefilter; exact 15:29-pass / 15:30-reject regression test now passes. | **VALIDATION / PROCESS CHANGE** — historical and live adapters must test both sides of every time and risk boundary, including equality. | Applied to the quarantined harness; carry into runtime-parity tests. |
| L5 | **Sub-minute data must be sourced by governed role, not recency.** Only 3 of 30 owned 1s sessions were firewall-safe (23 outer-test, 4 protected holdout). | Stage-0 firewall proof; the $10 clean-download correction. | **PROCESS CHANGE** — the real exit-calibration acquisition (cmbp-1) selects sessions by governed train/dev role with a firewall-intersection proof, never "recent N." | Applied at Stage-0; generalize to the real acquisition. |
| L6 | **Bulk-batch acquisition, not serial streaming, at scale.** 13 sessions took 66 min but only 37 CPU-sec — the cost was vendor latency + retries + batch-queue, not data volume. | Stage-0 clean-1s run; process was ~99% idle. | **PROCESS CHANGE** — the full backfill submits bulk batch jobs (parallel server-side prep), under the same cost caps. | Documented in the parallelism/diamond doc; apply at backfill. |
| L7 | **Stage-by-stage review cadence works — do not batch stages.** The "build 1-2-3 in one shot" attempt correctly stopped at Stage 1 on the active-gate, surfacing L1 at the first fit instead of after three stages of throwaway build. | Stage-1 stop record. | **VALIDATION** — keep the one-stage-at-a-time gate for the real campaign. | Confirmed. |
| L8 | **The dry-run-before-spend thesis is validated.** Three cold design reviews could not catch L1 (spec was internally consistent; the defect only appears when a fitted model meets the real label distribution). A ~$10 dry-run did. | Review history vs Option-D finding. | **VALIDATION** — always run the Walking Skeleton before committing to the full data download + GPU campaign. | Confirmed. |

## How this ledger enforces propagation

1. **DESIGN CHANGES don't count as "handled" until they're in the binding
   authority** (new hash, checker, owner sign-off) — L1 is not done until then.
2. **PROCESS CHANGES are referenced from the real acquisition/campaign goals**
   (L5, L6) so the operators inherit them, not rediscover them.
3. **WATCH-ITEMS name the exact real-campaign checkpoint** where they get
   verified (L3/L9 → D54/D57; L11/L12 → lifecycle and parity acceptance).
4. Every future Walking-Skeleton stage (2-6) appends its learnings here with a
   disposition before that stage is called "reviewed."

## Open items to fold into the FT2-21 bundle / authority when convenient
- L1 as a formal owner-adopted amendment (D-number) once Codex implements it.
- L5 (firewall-role acquisition) and L6 (bulk batch) as process addenda.
- L2 as a binding note for the FT2-60 lifecycle contract.

---

## Standing process (owner directive 2026-07-31)

This ledger is updated after EVERY Walking-Skeleton phase — the moment Codex
gets snagged on something — in THIS file, not a separate tracker. Purpose:
learn from each snag, don't tunnel-vision on the skeleton, and carry every
lesson forward into the real large-scale model's design.

## Emerging synthesis — what Stage 1 is teaching us about the REAL entry model

Stage 1 (the entry model) has hit TWO structural blocks before a single
downstream stage ran, and **both are entry-side**. That is itself the headline:
**the entry model is the hard part of this whole system.** Anticipate this for
the beefy model now.

- **L1 (fixed): a gate can be internally consistent yet empirically
  unsatisfiable.** Real-model carry-forward: before the real campaign, every
  gate gets a "can ANY model clear this against the real label distribution?"
  satisfiability test — not just a consistency review. The three cold reviews
  missed L1 precisely because they only checked consistency.

- **L9 (plumbing fork resolved; real-model watch open): the entry model's
  confidence (signal-to-error) on 45 sessions is
  ~10× too weak to clear the downstream uncertainty/regret gates.** These gates
  are demanding *by design* (they enforce "don't trade on noise") — that is
  correct and protective. Real-model carry-forward, the key anticipation:
  **the beefy model's feasibility hinges on signal > conformal error**, which
  needs (a) more data and (b) more model capability. **Before the GPU campaign,
  measure a data-scaling curve: how does signal-to-error shrink as dev data
  grows 45 → ~300 sessions? If the 10× gap doesn't close, the entry edge may be
  too thin for ANY model — which is exactly what we must learn cheaply before
  spending.** This is the D54 entry-feasibility question made concrete, and it
  quantifies the value of the D57 backfill.

- **Cross-cutting:** a demanding confidence gate means the real entry model must
  be genuinely strong OR the trader is mostly WAIT (abstention-as-success, which
  the Charter permits). Per the component-freeze logic, the real test is whether
  the COMBINED system (a selective/weak-ish entry + a strong exit) still profits
  — so a mostly-abstaining entry is not automatically a failure. Track this.

## Stage 2–3 synthesis — what the downstream skeleton adds

- The forced-BUY fork did its narrow job: real A6 proposals flowed through a
  quarantined intent wrapper into official 1-second lifecycle rows, learned
  HOLD/EXIT decisions, the upward-only floor, serial replay, and D60 outputs.
  That proves the seams can execute; it says nothing about alpha.
- The lifecycle baseline demonstrates why rare-event metrics must lead the real
  exit-model review: ROC AUC `0.8372` coexists with average precision `0.0664`.
- The independent replay found two parity-sensitive boundary issues that static
  success counts hid: a 15:30 equality error and an implicit two-clock adapter
  contract. Both now have explicit checks; the latter remains a native-schema
  requirement for the real simulator/runtime-parity campaign.
- Rendered QA found a multi-session coordinate defect after the structural HTML
  checks passed. Visualization is part of verification, not presentation polish.

---

## Phase update 2026-07-31 — Codex V1 retrospective + V2 rerun packet review

Source: `PROTOCOL101_WALKING_SKELETON_V2_RERUN_PACKET_2026_07_31.md` (Codex),
Claude-verified against authority + artifacts. New learnings:

- **L10 — Cadence: the V1 skeleton was NON-COMPLIANT, not the authority ambiguous.**
  V1 made 1-second HOLD/EXIT/floor *decisions* (confirmed in stage2_3/report.md).
  Authority §2.4 + A1 unambiguously specify **completed-minute** decisions and
  fills, and explicitly REJECTED sub-minute decisions because the normalized
  corpus has one CBBO snapshot per contract-minute; sub-minute acquisition is
  "deferred to a separate campaign-2 owner decision" (L223). D58/D59 govern
  1-second *labels/calibration*, a different axis Codex conflated with decision
  cadence. **Disposition: fidelity DEFECT in the skeleton (fix to comply);
  Codex's "blocking Q1 ambiguity" framing is corrected.** Real-model carry:
  decisions stay completed-minute unless the owner deliberately opens the
  campaign-2 sub-minute-decision amendment (which carries data-cost + IBKR
  live-parity burden and cannot backfill minute history before 2023-03-28).

- **L11 — The real lifecycle objective is not yet specified.** The action-advantage
  foundation (`..._HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md`) defines
  `A_hold = Q(hold) - Q(exit@bid)` but EXPLICITLY BLOCKS training until slot
  opportunity cost, switching cost, calibrated fill/latency/quote-age
  uncertainty, and distributional targets are added. V1's harness lifecycle used
  a convenient HGB HOLD/EXIT proxy — the forbidden shortcut. **Disposition:
  DESIGN work (real FT2-60), Claude writes / Codex reviews. You cannot faithfully
  skeleton a model that isn't specified.** Confirms RSD-06.

- **L12 — Adopt Codex's V2 discipline as the permanent operating standard.**
  Two-lane A/B separation (autonomous model-fidelity vs deterministic canary),
  `fidelity_manifest.json`, the two-axis stage card (engineering coverage vs
  policy acceptance), immutable attempt dirs, rendered visual QA, and the RSD
  register. These prevent the exact "forced activity = pass" trap that damaged
  V1. **Disposition: PROCESS CHANGE, adopt now.**

- **META (owner steer 2026-07-31): the skeleton is a discovery instrument, not a
  destination.** It has surfaced its findings (L1 fixed; entry signal-to-error =
  the D54 question; lifecycle objective undesigned; cadence defect; process
  rules). Route findings to the REAL spec — do NOT iterate skeletons for their
  own sake. The next quantitative gate is the D54 simple-model entry-feasibility
  + scaling study on the REAL entry path (CPU, likely no new download), which is
  what decides the ~$1,100 spend. Codex's full "miniature-but-real neural V2"
  over-rotates toward rebuilding the real campaign under a skeleton label; prefer
  fold-findings + design-the-real-gaps + a LEAN completed-minute confirm run.

---

## Phase update 2026-07-31 (b) — 1-second-exit pivot + parity finding

Owner decisions: (1) amend toward **1-second EXIT decisions** (pivot on new
information; floor=stop=exit is already sub-minute); (2) V2 = **full
miniature-neural, from scratch, quarantined** (build it exactly like the real
thing, miniature); (3) confirm decision parity via recorder days.

- **L13 — Parity is the binding constraint on 1-second exit, and it is the
  HARDEST version of the parity gate (Claude-verified against the synchronization
  contracts).** The recorder captures event-driven sub-minute data (good), BUT
  the existing parity certification is **minute-cadence and microstructure-MASKED**:
  cross-vendor (IBKR-live vs Databento-historical) parity was only achievable by
  masking `bid/ask/mid/spread/sizes/iv/greeks` before model scoring, because the
  raw fields caused action/selected-contract mismatches; even then features are
  "not byte-identical." Those masked microstructure fields are exactly what a
  tick-by-tick exit model decides on, and cross-vendor divergence is WORSE at
  1-second than at minute. **The binding constraint is not cadence — it is which
  vendor trains the exit model.** Three paths: (A) same-vendor IBKR-train/IBKR-live
  (parity-clean; needs a recorder-capture campaign — only ~4 recorder days exist);
  (B) cross-vendor Databento-train/IBKR-live (fast data, real feasibility risk —
  the microstructure that failed parity is the exit's own input); (C) hybrid —
  1-second floor/stop (robust price-threshold trigger, parity-tolerant) + minute
  learned exit (certified). **Disposition: OPEN — run a cheap 1-second
  cross-vendor parity probe on the existing 4 recorder days BEFORE amending the
  authority or committing to a data campaign.** Pairs with the standing
  parity-gate hard rule (no training until train/live feature parity certified).
  Real-model carry: the 1-second-exit amendment must specify the vendor/training
  path and its parity certification, not just the cadence.

---

## Phase update 2026-07-31 (c) — recorder audit, cbbo-1s cost, staged-D59 (A7)

- **L14 — Recorder-day inventory (Claude-verified).** 13 IBKR capture dirs exist
  (~3 GB event-driven JSONL each). **6 COMPLETE**: 2026-06-30, 07-01, 07-02,
  07-10, 07-13, 07-14 (~436–440 one-per-minute health files; 06-30 & 07-02
  recovery=pass). **INCOMPLETE/UNUSABLE**: 06-29 (dev partial, 32 files),
  07-20 (recovery=FAIL, died ~2h early), 07-29 (aborted, 21 files), and
  07-06/07/08/09 (empty, no capture). For a parity probe, incomplete days only
  reduce coverage — use the 6 complete days. 06-30/07-01/07-02 already have
  Databento paired builds (minute cadence, microstructure-masked).
- **Databento coverage (verified via metadata range):** `cbbo-1s`
  **2025-02-20→present**; `cmbp-1` 2023-03-28→present; `cbbo-1m` (entry minute
  substrate) back to 2013-04-01.
- **Cost (free get_cost, exact 0DTE ladders 58–86 syms/day from the captures):**
  6 complete days — `cbbo-1s` **$1.34** total; `cmbp-1` **$9.88** total.
- **DESIGN DECISION — staged data ladder + A7 amendment (owner-approved
  2026-07-31).** Replace the "cmbp-1 for all later acquisitions" mandate with a
  risk ladder: (1) $1.34 cbbo-1s parity check → (2) Walking Skeleton V2 → (3)
  cbbo-1s backfill to 2025-02-20 (~360 sessions) = first prototype → (4) cmbp-1
  only if an edge is shown. **Governed via amendment A7 (staged D59):** cbbo-1s
  ACCEPTED for parity/skeleton/prototype + 1-second decision cadence; cmbp-1-
  derived-1s REQUIRED for the trusted floor/stop label corpus before promotion.
  Rationale: cbbo-1s = 1 snapshot/sec and can miss an intra-second
  cross-then-recover that a stop/floor cares about; cmbp-1 sees every quote.
  Proposal: `PROTOCOL101_D59_STAGED_SUBMINUTE_REPRESENTATION_AMENDMENT_PROPOSAL_2026_07_31.md`
  (Claude writes → Codex reviews → owner signs). New checker rule
  `sub_minute_corpus_tier_tag_required`. **No download executed** — awaiting A7
  sign-off + explicit green-light. Real-model carry: the trusted exit corpus is
  cmbp-1; combined entry+exit replay only valid in the 1s-overlap window (≥2025-02-20).

---

## Phase update 2026-07-31 (d) — A7 Codex review and governed reseal

- **L15 — Trusted floor/stop labels must inspect raw events before
  downsampling.** Codex endorsed the Tier-S/Tier-T cut with a required technical
  clarification. Official `cbbo-1s` is the last consolidated BBO on an interval
  (and may omit an interval with no qualifying update/trade), while `cmbp-1`
  provides every consolidated top-of-book update event. Merely retaining
  `cmbp-1` and then labeling only the downsampled last state would still miss a
  cross-then-recover. Tier-T floor/stop crossing labels therefore inspect the
  raw CMBP event path; only the derived 1-second representation is a model
  input. **Disposition: DESIGN CLARIFICATION, folded into A7.** Tier S remains
  `1-second-approximate`, quarantined, and forbidden from trusted promotion,
  paper-readiness, paper, or real-money floor/stop-label paths.
- **A7 enforcement is machine-readable, not prose-only.** The governed policy
  and canonical corpus registry classify the two current `cbbo-1s` raw roots
  plus the V1 downstream derivative. Forty-one discovered current sub-minute
  manifests are covered through the registry sidecar without rewriting their
  immutable pre-A7 provenance. Every post-A7 corpus must embed its tier fields
  and register before use. Checker rule
  `sub_minute_corpus_tier_tag_required` includes negative fixtures proving it
  rejects Tier-S promotion marking, a trusted consumer bound to Tier S, and an
  unregistered current corpus.
- **Reseal result:** authority
  `82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2`
  → `1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc`;
  Graph V2 remains
  `9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08`;
  cross-contract checker 33/33 green. **Status: awaiting Claude independent
  verification and final owner signature; no training, download, broker,
  recorder, protected-resource, runtime/default, or graph action occurred.**

---

## Phase update 2026-07-31 (d) — A7 signed; cbbo-1s parity download authorized

- **A7 CLOSED — owner-signed** (Owen Heidenreich, 2026-07-31). Claude-verified:
  authority `82d9573e`→`1d215845` (reproduced as plain sha256), graph unchanged
  `9955085a`, checker 33 checks/0 failures + new rule
  `sub_minute_corpus_tier_tag_required` passing, Tier-T guarantee verbatim, both
  Codex clarifications (raw-event-path-before-downsampling; cadence boundary =
  no 1s-decision activation, FT2-08 minute stays authoritative) folded into the
  authority. Sign-off receipt: `protocol101_d59_staged_subminute_representation_amendment/owner_approval_receipt.json`.
- **Rung 1 authorized:** owner green-lit the $1.34 cbbo-1s parity download
  (Tier-S). All four A7 conditions met (free estimate, hard cap $15, owner
  green-light, quarantine `v4/raw/opra_1s_parity_probe/`). Download in progress;
  then the cross-vendor 1-second parity probe (Phase 1 same-vendor viability +
  Phase 2 cross-vendor comparison on the masked microstructure fields) decides
  the cadence path.

- **Rung-1 download COMPLETE + Claude-verified (2026-07-31):** cbbo-1s for the 6
  complete recorder days pulled to `v4/raw/opra_1s_parity_probe/` (DBN+parquet,
  hashed manifest, `tier_tag: probe`). Actual cost $1.34 = estimate, under cap.
  ~9.0M rows total. Schema carries bid/ask/sizes (the masked microstructure).
  Next: the cross-vendor 1-second parity probe decides the cadence path.

---

## Phase update 2026-07-31 (e) — WS2 1-second parity probe

- **L16 — One-second price levels transfer; the full exit feature vector and
  first-cross event do not transfer cleanly.** The six-day Tier-S probe aligned
  every IBKR ladder contract to its exact padded OSI symbol with zero unmatched
  symbols and measured 5,570,116 paired symbol-seconds. IBKR same-vendor
  reconstruction yielded 5,569,565 clean active symbol-seconds, 90.66%–98.29%
  active-ladder coverage, and 95.21%–99.14% one-second continuity. Bid differed
  from its minute checkpoint on 71.94%–85.80% of rows, so the sub-minute path
  carries real state variation (not a predictive-signal or alpha claim).
- Cross-vendor bid/ask/mid level parity was strong (daily correlation at least
  0.999963; weighted bid MAE $0.0218; daily bid p95 $0.10–$0.20), but displayed
  sizes remained vendor-sensitive (22.2/22.7-contract weighted bid/ask-size
  MAE, 23.31%–38.39% exact agreement, 51.24%–55.43% nonzero direction
  agreement). The frozen common-history 80% trailing-bid floor flipped on
  0.1479% of states, yet first-cross decisions differed on 17.59% of triggered
  quote segments; 92.25% of dual-vendor triggers were within one second. The
  old minute certificate cannot waive this result: it masked these fields,
  had zero exact rows, and accepted only zero action/contract flips.
- **Cadence-path disposition: recommend C now; preserve A as the full learned
  destination; do not authorize B from this probe.** A future governed design
  should use an IBKR-live one-second protective floor/stop plus the existing
  completed-minute learned exit. A fully learned one-second policy should use
  same-vendor IBKR-train/IBKR-live after FT2-60 freezes its objective and sample
  law and a real recorder campaign supplies sufficient independent sessions and
  action-diverse held trajectories. The present six days are only 6/45 of the
  frozen session hard-bound minimum; their 125 completed-minute open-state rows
  collapse to five trajectories on three sessions (120 HOLD, five forced-flat,
  no learned EXIT, no native one-second labels). **Tier-S feasibility only:**
  trusted historical floor/stop labels remain Tier T; no authority or cadence
  changed. Evidence:
  `v4/audit/autoresearch/protocol101_ws2_parity_probe/`.

---

## Phase update 2026-07-31 (e) — WS2 parity probe COMPLETE + Claude-verified

- **L15 — Cross-vendor 1-second exit parity result (Claude-verified).** On 6
  recorder-paired days (5.57M paired symbol-seconds): price LEVELS transfer well
  (bid/ask/mid corr ≥0.99996); SIZES/SPREAD do NOT (size corr 0.88–0.98, exact
  agreement 23–38%; spread agreement ~40%); floor STATE agrees 99.85% but exact
  first-cross TIMING disagrees 17.59% (269/~1500 triggered segments). Receipt
  hash 44eaa3aa… reproduced independently; validation 18/18.
  **Conclusion:** a learned cross-vendor 1-second exit (Databento-train/IBKR-live,
  path B) is NOT reliable and is rejected. A price-threshold floor/stop is robust.
- **DECISION INPUT — cadence path — [SUPERSEDED by phase (f)/L16; decision REOPENED].**
  Original (retained for the record): recommended C now (minute entry + minute
  learned exit + IBKR-live 1s floor), A as destination via a recorder campaign.
  **Corrections (phase (f)):** (i) a 1-second floor CHECK is a governed change vs
  authority D24 (completed-minute floor), so C is NOT "buildable now" without
  governance + ablation — the "NO cadence amendment needed" claim above is WRONG;
  (ii) the ~65–90-session figure was misapplied (Phase-F shadow bound, not a
  training minimum); (iii) "Databento is not a live-exit source" is WRONG —
  Databento offers live OPRA (Path D). Path A/B/C trichotomy dissolved.

---

## Phase update 2026-07-31 (f) — adversarial review corrected the record (Codex)

Codex adversarial review of the cadence brief; Claude re-verified every point
against source — all CONFIRMED. Corrections to prior claims:

- **L15 spread claim was WRONG.** The "~40% spread agreement" cited the
  same-vendor (IBKR-second vs its own minute checkpoint) table, not cross-vendor.
  Real cross-vendor: spread exact-level ~69–83%, within $0.10 on 97–99%; only
  spread-CHANGE direction is noisy (~46–58%). Price levels ≥0.99996 (5th pct
  ~0.997). **Parity is more favorable than L15 implied.** Sizes genuinely do NOT
  transfer (exact 23–38%, direction 51–55%).
- **Trigger/first-cross metric is not decision-grade:** the floor was computed
  from `(bid_ibkr+bid_dbn)/2` (`run_parity_probe.py:574`) — info neither live
  system has. State-flip (~0.145%) is robust; the 17.59% exact-cross stat is
  observation/segmentation-dependent (session bootstrap ~12.9–22.8%) and must NOT
  gate architecture.
- **"18/18 validation" overstated:** ≥3 checks just assert the report recommends
  C / preserves A / rejects B (circular).
- **65–90 sessions misapplied:** 45 is the Phase-F live-shadow hard-bound minimum
  (`bootstrap_spec.json:239`), not a 1s-lifecycle training minimum. Data need
  must be power-derived. "Only 5 trajectories" = 5 shadow trajectories, not all
  constructible from recorded full-ladder paths (minute foundation already built
  198,261 state rows / 709 trades).
- **DOC CONFLICT (fix):** ledger "no cadence amendment needed" (L320) vs authority
  D24 completed-minute floor check (L502) — a 1s floor check IS a governed change.
- **Recorder confound:** no gross drop (2026-07-14: 6.48M callbacks, ~104ms BBO
  inter-arrival ≈ IBKR's ~100ms aggregation, not Claude's 250ms). BUT recorder
  `event_timestamp_utc` = local TCP packet time (batched; ~85% shared), so exact
  cross-vendor timing is confounded; the probe cannot attribute size/spread
  differences among IBKR aggregation / SMART-NBBO / entitlements / recorder, and
  cannot prove a price/path model is unreliable.

- **L16 — PATH D (the missed architecture; now the lead candidate).** "Same-vendor"
  need not mean IBKR-recorded history. **Databento offers live OPRA**, so train +
  decide live on Databento (same vendor → no model-input parity problem), IBKR =
  execution/guard only. Multirate design: (1) slow minute decision plane (deep
  history, regime/ranking/entry intent); (2) fast Databento cmbp-1 historical +
  Databento live sub-minute execution/position plane; (3) deterministic risk
  governor (forced-flat, stale-data, loss limits, catastrophic boundary); floor
  updated at slow cadence, monitored continuously; (4) separate MODEL cadence from
  ORDER cadence (intent → bounded quote-fresh marketable-limit window; needs A1
  change). Open costs: Databento live OPRA entitlement + cost, outage handling,
  symbol sync, IBKR execution reconciliation — needs an owner-approved estimate.
  Fallback **Path B-prime:** price/path-only features (relative bid/PnL/MFE/
  giveback/time/underlying-path; exclude displayed size + vendor greeks; spread/
  IBKR state as execution/abstention guards only) — falsifiable on the 6 owned
  days. **A7 sequencing fix:** buy a small trusted cmbp-1 raw-event slice BEFORE
  any large cbbo-1s backfill (else we optimize against approximate stop labels).
  Path C is NOT "buildable now" in the trusted sense (floor-cadence governance +
  ablation required). Decision REOPENED; no path settled.

---

## Phase update 2026-07-31 (g) — Path B-prime falsification COMPLETE, Claude verification pending

- **L17 — B-prime deterministic representation-transfer falsification.** The
  six already-owned recorder-paired days were tested with 399 entry-only-frozen
  counterfactual trajectories. No model was trained. Both feeds ran the same
  preregistered deterministic state machine: an upward-only 65%-of-entry / 80%-of-
  running-peak bid floor, fixed adverse-path and giveback exits, and a 45-minute
  or 15:55 ET time stop. IBKR used `received_timestamp_utc`; Databento `ts_recv`
  was mapped to the same canonical `received_boundary_second_utc`. Neither
  `event_timestamp_utc` nor `ts_event` was used. Each feed used its own entry ask
  and prior state; no cross-vendor average was available to either decision path.
- **Preregistered verdict: `INSUFFICIENT`, not YES and not NO.** At both the 1s
  and 5s quote-age caps, price/path-only exit decisions agreed within 5 seconds
  on 88.97% of 399 resolved trajectories (six-session cluster 95% CI
  86.95%–91.14%), below the 90% YES point gate. State-level decisions agreed
  99.856% (CI 99.804%–99.893%), while 29.82% of trajectories had at least one
  action flip (CI 24.81%–34.24%), missing the 20% point / 30% CI-high YES gates.
  Economic exit-price disagreement was much smaller: median $0 and p95 $20 per
  contract (p95 cluster CI $10–$40). No preregistered cluster-CI severe-failure
  rule fired, so a NO claim would also overstate the evidence.
- **The simplification was directionally right, but not established as a live
  contract.** Adding displayed size plus symmetrically recomputed IV/delta made
  transfer worse: price/path-only improved ≤5s exit agreement by 5.76 percentage
  points (CI 3.40–8.75), reduced any-flip trajectories by 16.04 points (CI
  12.89–18.84), and improved p95 exit-price disagreement by $10 (CI $0–$19.50).
  This supports excluding vendor-sensitive size/Greek features from a
  cross-vendor representation; it does not authorize B-prime.
- **Architecture consequence:** another inherited multi-day IBKR campaign is
  not the automatic next step. The 1s and 5s headline exit-agreement,
  exit-price, and trajectory-flip results were identical; eligibility-denominator
  rates changed only negligibly. The YES miss includes point-estimate gates, so
  more sessions merely narrowing these intervals would not make this exact
  preregistered rule pass. Any later recorder sample count needs a new
  decision/power contract. Path D remains a candidate, B-prime remains unproven,
  Path C remains governed, and no architecture is committed by this Tier-S probe.
- Evidence: `v4/audit/autoresearch/protocol101_ws2_bprime_falsification/`.
  Highest claim: measured whether a price/path-only exit representation
  transfers cross-vendor on 6 recorder-paired days; Tier-S; no promotion/alpha
  claim; no architecture committed. Terminal: `STOP_FOR_CLAUDE_VERIFICATION`.

---

## Phase update 2026-07-31 (h) — WS2 latency/fill sensitivity COMPLETE, Claude verification pending

- **L18 — Received-clock latency and marketable-limit sensitivity.** The 399
  frozen B-prime trajectories supplied 798 fixed Databento decisions (399 entry
  asks and 399 price/path-only exit bids). No policy was fitted or retuned. For
  each decision, IBKR BBO and last/last-size state was reconstructed only by
  `received_timestamp_utc`; Databento `ts_recv` supplied T. The preregistered
  delays were 0/1/2/5/10/30/60 seconds and buffers were $0/$0.05/$0.10/$0.20.
  No cross-vendor average or `event_timestamp_utc` decision field was used.
- **Executable-price sensitivity rises quickly in the tail.** Median adverse
  slippage stayed approximately $0 per contract, but entry/exit p95 adverse
  slippage was $1/$0 at 0s, $20/$20 at 1s, $30/$30 at 2s, $50/$40 at 5s,
  $60/$50 at 10s, $110/$70 at 30s, and $142.50/$90 at 60s. Fresh-touch
  eligibility remained 98.25%–100% under the existing 1.5-second paper-guard
  quote-age limit. These are assumed-delay sensitivities, not measured API or
  end-to-end live latency.
- **Tier-S buffer recommendation: $0.00 at the frozen five-second window
  (`PASS`).** At the Databento decided price, IBKR-order-guard eligibility was
  100% for entry and 99.75% for exit. Conditional immediate-touch-or-later-
  trade-through fill proxies were 99.50% for entry (session-cluster 95% CI low
  98.94%) and 99.75% for exit (CI low 99.22%); strict observed trade-through by
  five seconds was 92.98%/96.73%. The realized adverse p95 proxy was $0 because
  immediate crosses executed at the observed better-or-equal touch and delayed
  trade-throughs were conservatively assigned the unbuffered limit. Every
  larger buffer also passed, so the preregistered smallest-passing rule selected
  zero rather than paying for no required fill-proxy improvement.
- **Interpretation and hard limit:** a Path-D design should separate model
  cadence from order cadence and submit/cancel a bounded quote-fresh limit
  promptly; it should not intentionally wait for a later executable quote when
  tail slippage grows this fast. But no historical order existed. Queue
  position, depth, routing, partial fills, cancels, and API latency are unknown,
  so this packet does **not** measure true fill probability and does not commit
  an execution architecture or amend A1.
- Evidence: `v4/audit/autoresearch/protocol101_ws2_latency_sweep/`. Highest
  claim: measured historical slippage-vs-assumed-delay and marketable-limit fill
  behavior on 6 recorder-paired days; Tier-S; sensitivity not true live latency;
  no architecture committed. Terminal: `STOP_FOR_CLAUDE_VERIFICATION`.

---

## Phase update 2026-07-31 (g) — B-prime falsification (Claude-verified)

- **L17 — Price/path-only exit representation transfers cross-vendor only
  BORDERLINE (INSUFFICIENT); price/path >> size+greek.** 399 frozen trajectories,
  6 paired days. Discrete "same exit ≤5s" agreement **0.890 [0.869, 0.911]** —
  under the preregistered 0.90 YES gate, above the 0.80 severe-failure floor →
  neither YES nor NO. BUT per-second HOLD/EXIT **state agreement 0.999** and
  economic exit-price diff **median $0, p95 $20/contract**. Size+greek variant
  materially worse (0.832, ~2x decision-flips). Claude-verified: receipt
  `fba996a5…` reproduced; preregistration frozen-before-results; validation
  non-circular (verdict independently re-derived, 84 checks). Codex applied its
  own earlier anti-circularity critique.
  **Disposition:** (1) **DESIGN lesson (durable):** the real exit model should
  lean on price/path features (relative bid/PnL/MFE/MAE/giveback/time/SPX-path),
  NOT displayed size or vendor greeks. (2) **Path B-prime NOT a cheap escape** —
  a cross-vendor learned exit (Databento-train/IBKR-live, no live subscription)
  is only borderline, unproven at 6 days. (3) **Reframe — reinforces Path D:**
  the 89% is cross-vendor MODEL-decision agreement, which only bites B-prime;
  Path D trains+decides on Databento (model never sees IBKR data), so its only
  cross-vendor exposure is EXECUTION slippage (the small $0/$20 number, still a
  proxy — true round-trip needs live latency measurement). Decision now hinges on
  the remaining de-risk item: **Databento-live OPRA cost/feasibility.**

---

## Phase update 2026-07-31 (i) — latency sweep + Path D review → project reshape

- **L19 — Latency/fill sweep (Claude-verified; receipt 5483d6ff, validation 12/12,
  non-circular).** $0 extra buffer, submit immediately → 99.5% entry / 99.75% exit
  immediate touch; tail cost is from WAITING (p95 adverse slippage $50→$142 entry,
  $40→$90 exit as delay 5s→60s). Execute fast; never rest passively.
- **L20 — CORRECTION: ThetaData stays on the critical path.** Prior claim (it drops
  for Path D) was WRONG. Feature floor requires official SPX price action; Databento
  sells SPX options, NOT the cash index. A matched historical+live SPX source
  (ThetaData ~$50/mo, or equivalent) is mandatory. Path D is NOT single-vendor.
- **L21 — CORRECTION: Databento $199 subscription bundling UNCONFIRMED.** Public
  pages suggest Standard bundles live + 10yr historical, but caps/fair-use unknown
  and live OPRA entitlement was ABSENT on 2026-07-24. Do not assume "$200 = all data."
  Owner-facing confirmation required.
- **L22 — Path D adversarial verdict: conditional GO as lead architecture; NO-GO for
  paper/A1-amendment today.** KEY REFRAME: execution is a SECOND CAUSAL POLICY, not a
  thin adapter. Project is now THREE components: entry + exit/lifecycle + execution
  policy. Additions/changes: (a) execution = broker-quote-anchored bounded
  marketable-limit (fresh IBKR quote at submit, one tick through, IOC/short timer,
  bounded requotes, reconcile-before-next; exit no-fill ESCALATES); (b) first-class
  broker-facing deterministic risk governor (Databento feed-loss while holding →
  forced-flat via IBKR); (c) A1 rewrite = full order state machine (all clocks,
  no-fill/partial/late-fill/disconnect/no-bid/15:55/fees/D48-49/collar) used
  IDENTICALLY across training labels, replay, PnL, floor, and live; trusted
  sub-minute exit labels need cmbp-1 raw event-path, not cbbo-1s; (d) Path C dropped
  (its 1s floor is an unauthorized authority change). Comparative: D lead (conditional
  on execution-bridge + entitlement), A cleanest-but-delayed benchmark, B/B-prime
  inferior. **Next step: free offline Execution-Bridge Skeleton (6 days, latency
  rungs 100/250/500/1000ms) before any live rung / A1 amendment / subscription.**

---

## Phase update 2026-07-31 (j) — COST CONSTRAINT: Path D shelved, revert to affordable cross-vendor

- **L23 — Owner cost constraint: Databento live ($199/mo) is unaffordable → Path D
  SHELVED.** Revert to the certified "old way": train Databento/ThetaData, decide +
  execute on IBKR live. This is the `protocol101-live-v2-microstructure-masked`
  contract (cross-vendor model handled by masking size/greeks/spread, keeping
  price/path, minute cadence). **Upside:** deciding AND executing on IBKR is
  same-vendor, so the Path D decide-Databento/fill-IBKR reconciliation risk
  VANISHES — reverting for cost accidentally removes a real risk. Path D analysis
  NOT wasted (execution-as-policy, latency/slippage data, ThetaData/SPX necessity
  all transfer).
  **Affordable architecture (lead):** minute learned entry + minute learned exit
  (Databento-train / IBKR-decide, microstructure-masked) + 1-second price-based
  floor/stop on IBKR live price (robust; price transfers 99.996%) + IBKR execution
  (marketable-limit, same-vendor). = essentially Path C. Governance to-do: the 1s
  floor check is an authority change vs D24 (needs amendment).
  **Tradeoff accepted:** model can't use displayed size/greeks/spread live (masked);
  the LEARNED tick-by-tick 1s exit is DEFERRED (B-prime cross-vendor only 89%).
  **Affordable route to the tick-by-tick learned exit = Path A** (train on IBKR
  recordings, run on IBKR, same-vendor, no subscription) — cost is TIME; recorder is
  built and runs ~free; bank sub-minute IBKR history over months.
  **Costs kept:** Databento historical pay-as-you-go (cheap) + ThetaData $50/mo (SPX,
  mandatory) + IBKR (cheap/owned). NOT the $200 Databento live.

---

## Phase update 2026-07-31 (k) — CORRECTION: greeks ARE used (self-computed), synchronized

- **L24 — CORRECTION to L23 phrasing.** "Microstructure-masked → model can't use
  greeks live" was MISLEADING. Authority §3.1 feature floor: the model uses
  **internally recomputed IV/delta/gamma** (from canonical price, spot, strike,
  time-to-expiry, frozen constants); only **raw VENDOR greeks are prohibited**
  (they differ IBKR vs Databento). So greeks ARE model features and ARE
  synchronized train↔live — exactly the owner's "same game / synchronized"
  principle. What the mask removes is raw vendor microstructure (vendor bid/ask/
  spread/sizes/vendor-greeks), not the self-computed features. B-prime's
  "size+greek worse" result CONFIRMS this existing design choice (recomputed >
  vendor). The 17 signed model-facing features are all synchronized-computable
  (SPX context, premium path, moneyness, recomputed greeks) — no raw vendor fields.

---

## Phase update 2026-08-01 — Path D REVIVED with deployment-cost financing; single fast exit

- **L25 — Two-layer exit (minute-smart + 1s-stop) is a cross-vendor COMPROMISE, not a
  design ideal.** Stop and smart-exit aren't redundant (stop = reactive downside,
  always sells late; smart = proactive profit-taking near peaks, key for convex
  0DTE). But a once-a-minute smart exit is crippled. The clean design is ONE fast
  (1-second) exit that is both smart and protective — which requires same-vendor
  sub-minute (Path A or D).
- **L26 — Path D revived as target, financed as deployment-not-research.** Owner
  reconsidered: Path D is the best idea; the $199/mo Databento live is a DEPLOYMENT
  cost, not a research cost — build+prove the whole model on HISTORICAL Databento
  (cheap, pay-as-you-go), subscribe to live ONLY once it provably makes money
  (cancel anytime). Path D's wins: (a) kills the recurring cross-vendor parity
  treadmill (every feature change → IBKR re-check) — PROVIDED SPX is also
  same-vendor (ThetaData train+live); (b) avoids Path A's months-long IBKR-recording
  clock; (c) same-vendor lets the 1s exit use FULL microstructure (sizes/spread/tick)
  that cross-vendor had to mask → richer exit. Remaining hard part = the execution
  bridge (decide Databento/fill IBKR), but only needed at Phase 2 (after edge proven).
- **Databento $199 Standard coverage (owner research):** L0 (definitions/ohlcv-1m/
  statistics) fully included all history; L1 (cbbo-1m/cbbo-1s/cmbp-1/tcbbo) only
  trailing 12 months, older pay-as-you-go; ES(CME)/VX(CFE) separate. So subscription
  = live access + recent 12mo, NOT a cheap deep backfill. Deep sub-minute training
  backfill stays a priced-first pay-as-you-go buy.
- **PLAN (collapses the sprawl):** Phase 1 (now, cheap, no subscription, no parity
  treadmill) — design the 1-second smart+protective exit objective (Q2) + architecture
  (Q3), then build+backtest the Path D trader on historical Databento+ThetaData;
  gate = does it show provable edge? Phase 2 (only if edge proven) — solve execution
  bridge, start $199/mo live, paper-trade. Moots: banking IBKR recordings (Path A =
  fallback), the two-layer exit. Next step: design the exit objective + architecture
  (Claude writes / Codex reviews) + price the sub-minute backfill.

---

## Phase update 2026-08-01 (b) — Path D COMMITTED; $199/mo accepted

- **CORRECTION to L26:** "$199 = deployment-not-research cost" was WRONG (owner
  caught it). The real proof is forward PAPER-TRADING, which Path D cannot do
  without the live Databento feed (backtest is only the first gate). So the
  subscription is a VALIDATION cost incurred through the paper-validation period,
  not just at final deployment.
- **DECISION (owner, 2026-08-01): Path D committed; pay the $199/mo; single fast
  1-second exit; stop designing around avoiding the subscription.** Rationale:
  better design, kills the cross-vendor parity treadmill, and a 1-minute smart
  exit is structurally crippled (can only sell at peak if the peak lands within
  ~5s of the minute mark).
- **Sequencing discipline (kept):** backtest edge FIRST (don't run paid
  paper-trading on a model that failed backtest); the $199 clock runs through
  paper-validation after backtest edge is shown. Subscription may start earlier
  IF the 12-month L1 backfill discount justifies it.
- **Discount study (owner requested):** produce a pay-as-you-go vs $199-subscription
  cost table across the full 9-dataset footprint (OPRA definition/cbbo-1m/ohlcv-1m/
  statistics/cbbo-1s/cmbp-1/tcbbo + ES GLBX.MDP3 ohlcv-1m + VX XCBF.PITCH ohlcv-1m).
  Subscription covers L0 fully + L1 trailing-12mo; older L1 + ES + VX pay-as-you-go.
  Free get_cost sampling estimate; also answers subscribe-now-vs-later.
- **NEXT:** (1) exit objective + architecture design (Q2/Q3, Claude writes/Codex
  reviews) — Phase-1 long pole; (2) discount table in parallel.

---

## Phase update 2026-08-01 (c) — Databento subscription-vs-payg cost table

- **L27 — Cost table (free get_cost, representative day 2026-06-30 / 70 syms,
  extrapolated by trading days).** Full backfill EXCLUDING tcbbo: pay-as-you-go
  ~$1,560 vs one-month $199 sub ~$1,005 (residual ~$806 + $199) → **sub saves ~$555
  on the backfill AND unlocks live.** Dominant real cost = cmbp-1 older history
  (2023-03-28→2025-08-01) ~$767. Sub makes all L0 free (ohlcv-1m $373, defs, stats)
  + trailing-12mo L1 free.
- **tcbbo ANOMALY (skepticism):** get_cost priced tcbbo at $15.98/day = 12× cmbp-1,
  which is backwards (tcbbo = BBO@trades should be ≤ cmbp-1 = BBO@every event).
  Do NOT trust the $13k figure. cmbp-1 covers the sub-minute need → **DROP tcbbo**
  unless a specific use appears (verify the number first if so).
- **Buy strategy:** subscribe ONE month → download all-L0-free + 12mo-L1-free +
  pay-as-you-go older-L1 (cmbp-1) → cancel (keep data) → resubscribe for live at
  paper-trade time. Verify data retained after cancellation.
- **Data chronology (owner rule):** sub-minute substrate = cmbp-1 contiguous
  2023-03-28→now (derive 1s from it pre-2025-02-20; cbbo-1s canonical-cross-check
  from 2025-02-20); minute substrate = ohlcv-1m/cbbo-1m 2022→now; ES 2022→now;
  VX only 2026-04→now (VIX-proxy gap before — flag, don't patch). HARD RULE: every
  pay-as-you-go purchase STOPS at 2025-08-01 (newer is free under sub).
- **NEXT:** the 1-second exit objective + architecture design (Q2/Q3).

---

## Phase update 2026-08-01 (d) — tcbbo dropped; IBKR trimmed; Path D transition plan

- **tcbbo DROPPED — not used.** Every repo mention is an optional audit-slice
  alternative to cbbo-1s; project prefers cmbp-1/cbbo-1s. Never in a tensor/feature
  contract. The $9-13k is irrelevant. Real backfill ≈ $1,005.
- **IBKR subscriptions:** KEEP OPRA L1 $1.50/mo (execution quote; waived at ≥$20
  commissions); CANCEL CBOE Streaming Market Indexes $3.50/mo (SPX/VIX now from
  ThetaData under Path D; execution doesn't use it).
- **Path D Transition Plan written:** PROTOCOL101_PATH_D_TRANSITION_PLAN_2026_08_01.md
  — download manifest (subscribe-first; cmbp-1 2023-03-28→now ~$767 is the one big
  line; L0 + trailing-12mo-L1 free; stop payg at 2025-08-01; ThetaData=SPX;
  ES/VX=Databento futures; VX only from 2026-04); data-use plan (normalize+align →
  exit objective design → exit tensor/labels from frozen-entry OOF trajectories →
  backtest); blast-radius of changed assumptions (mask no longer needed; parity
  pivots to execution reconciliation; A1→order state machine; minute-entry+1s-exit;
  IBKR index dropped; Path A recording dropped; execution = 2nd causal policy);
  what survives/deprecated/new.
- **NEXT:** Claude drafts the 1-second exit objective + architecture; Codex runs a
  repo-wide Path D assumption blast-radius audit (read-only).

---

## Phase update 2026-08-01 (e) — 12mo staged cost; ToS research; exit design drafted

- **12-months-only backfill ≈ $208** ($199 sub gives the entire trailing 12mo of ALL
  schemas free incl. ~$325 cmbp-1; only ES/VX futures ~$9 pay-as-you-go). Deferred
  deep backfill ~$797 (older cmbp-1) = pay-as-you-go ANYTIME, no active sub needed.
  Added to the transition plan (§2a). Recommended: 12mo-first for research, buy the
  deep history only if warranted.
- **ThinkorSwim/Schwab execution research** prompt written (owner runs in ChatGPT) —
  evaluate Schwab Trader API as execution-only replacement for IBKR (free ~$514),
  covering programmatic SPXW 0DTE orders, marketable-limit/IOC/cancel-confirm/partial
  fills, fresh-quote-at-submit, paper API, algo-trading policy, costs, go/no-go.
- **1-second exit objective + architecture DESIGN drafted:**
  PROTOCOL101_ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_2026_08_01.md (Claude
  writes → Codex reviews → owner signs). Learned 1s exit = sole decision-maker; floor
  demoted to catastrophic backstop in the risk governor. Unblocks the action-advantage
  objective by specifying slot-opportunity-cost, switching cost, fill/latency
  uncertainty, and distributional targets; convexity (cut losers/run winners); causal
  1s features (self-computed greeks; Path-D microstructure now admissible); OOF
  trajectories from frozen entry; architecture = transparent GBT baseline then
  miniature-neural; lead eval with PR/AP not ROC AUC; floor-on/off ablation; 4-bucket.
- **NEXT PHASE (after: Codex audit returns + owner runs ToS research + downloads):**
  enter the next planning phase.

---

## Phase update 2026-08-01 (f) — coverage-table web verification

- **ChatGPT OPRA-coverage table: directionally right, UNVERIFIED on the costly part**
  (Claude web-check). Databento per-schema/per-tier included-history is behind a JS
  portal, not externally readable. Confirmed: ES/VX separate from OPRA Standard;
  "L0/L1" = pricing tier ≠ depth level. Likely WRONG-in-our-favor: cbbo-1m appears to
  include ~10yr (Databento's "10 more years" list names trades/cbbo-1m/ohlcv/
  statistics/definition), not "trailing 12mo." UNCONFIRMED (the ~$767 swing):
  cmbp-1 + cbbo-1s history depth under Standard — NOT in the 10-yr list (consistent
  with a 12mo cap but unproven). DEFINITIVE check = run get_cost on an older cmbp-1
  range AFTER subscribing ($0=included, price=payg). No impact on the ~$208 12-month
  research start; only the deferred deep backfill cost is affected. Transition plan §2b
  updated.

---

## Phase update 2026-08-01 (g) — coverage CONFIRMED; IBKR stays; blast-radius → phasing

- **Coverage table CONFIRMED** via Databento source chain (pricing: Standard = 16+yr
  L0 / 1yr L1 / 1mo L2-L3; schema doc: cmbp-1/cbbo-1s/cbbo-1m/tcbbo all L1). Earlier
  "cbbo-1m under-counted" guess was WRONG (that "10 more years" = catalog availability,
  not Standard inclusion). So ~$797 deep cmbp-1 backfill IS pay-as-you-go; ~$208 12mo
  plan stands. Availability: cmbp-1/cbbo-1s/tcbbo don't exist before ~2023-02-28
  (cbbo-1m finest before that). Transition plan §2b corrected.
- **IBKR STAYS** (Schwab has no paperMoney API → can't paper-validate). Keep OPRA
  $1.50; defer CBOE-index $3.50 cancellation until the Path D runtime replaces the
  current IBKR-SPX/VIX usage.
- **Codex blast-radius audit received:** Path D = governed redesign of decision-plane
  + exit + fill law + FT2 chain (7 amendment areas; 39 files pin old fill-law hash;
  graph + synchronization gates). BUT large "safe to keep" set (minute entry + 17
  features, identity/ladder/firewalls/self-greeks, label families conceptually,
  D48/D49/serial/15:55, simulator core, IBKR execution guards, order-state scaffold),
  and MOST blocking items are Phase-2 (live).
- **RECOMMENDATION — phase the amendments to match prove-before-pay:**
  Phase 1 (backtest edge, cheap) = buy 12mo data + sign FT2-60 + minimal model-plane
  amendment (unmask for same-vendor model; additive 1s exit contract on frozen minute
  entry; conservative latency-calibrated backtest fill model). Gate on edge.
  Phase 2 (only if edge proven) = execution-policy contract + risk governor + full A1
  rewrite + graph reissue (FT2-24/25/26/92) + supersede synchronization gates +
  runtime/promotion packet + 39-file fill-law resupersession.
- **NEXT:** owner buys 12mo data; Codex reviews FT2-60; Claude drafts the Phase-1/
  Phase-2 amendment sequencing plan (turns the audit into a small now-set + deferred
  backlog).

---

## Phase update 2026-08-01 (h) — subscription purchased; storage sizing; Phase-1 sequencing

- **Subscription PURCHASED** (OPRA Standard, Equity-options = correct plan; covers
  SPX/SPXW index options — confirmed via Databento coverage docs + empirical SPXW pulls).
- **Storage sizing (billable-anchored to actual on-disk files):** `cmbp-1` is huge —
  ~8.7GB raw/day, ~1.2GB/day on disk → **~300+ GB for 12 months.** Everything else tiny
  (`cbbo-1s` ~4GB/12mo; minute/defs negligible).
- **KEY: Phase 1 does NOT need `cmbp-1`.** Per A7, Tier-S `cbbo-1s` (1-second) suffices
  for backtest/prototype; `cmbp-1` (Tier-T raw-event-path) is only for trusted
  floor/stop labels before promotion (Phase 2). So **Phase-1 download = `cbbo-1s` +
  minute + defs + ES/VX ≈ ~5 GB** (trivial). `cmbp-1` (~300GB) deferred to Phase 2 —
  then use derive-to-1s-and-discard-raw or monthly chunks; free under sub anytime.
  Keep DBN.zst OR parquet, not both. **DO NOT pull cmbp-1 now.**
- **Phase-1/Phase-2 amendment sequencing plan WRITTEN** (owner-agreed phasing):
  PROTOCOL101_PATH_D_AMENDMENT_SEQUENCING_PLAN_2026_08_01.md. Phase 1 (prove edge,
  cheap): new Path-D model-plane contract (supersede mask's governing role for Path D;
  same-vendor unmasked; single clock; self-greeks) + sign FT2-60 + additive FT2-08 1s
  exit tensor/labels + provisional latency-calibrated backtest fill. Unchanged:
  minute entry+17 features, identity/ladder/firewalls, FT2-04 entry labels, D48/D49/
  serial/15:55, simulator core, IBKR guards. GATE: backtest edge. Phase 2 (only if
  edge): execution-policy+risk-governor, full A1 rewrite, graph reissue, supersede
  sync gates, runtime/promotion packet, 39-file fill-law resupersession; GATE:
  paper-readiness.
- **NEXT:** owner downloads Phase-1 data (cbbo-1s, NOT cmbp-1); Codex reviews FT2-60;
  then execute the Phase-1 reseal set.

---

## Phase update 2026-08-01 (i) — PHASE 1 STARTED

- **Path-D model-plane contract PROPOSAL written** (Claude writes → Codex reviews →
  owner signs): PROTOCOL101_PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_2026_08_01.md.
  Databento OPRA + ThetaData SPX same-vendor model plane; IBKR execution-only;
  microstructure mask lifted (same-vendor) with causal/leakage discipline + self-
  computed greeks retained; single clock; entry minute / exit 1s (cbbo-1s Tier-S for
  Phase-1). Supersedes mask cert + scoped-sync decision GOVERNING ROLE for Path-D only
  (old artifacts immutable). New checker rule path_d_model_plane_source_purity.
  Phase-1 (backtest) scope only — no live/runtime/paper.
- **Phase-1 kickoff handoffs:** (1) Codex design-review goal covering BOTH the
  model-plane contract + FT2-60 exit design (review-only checkpoint before reseal;
  BLOCKING defect → STOP/roles-flip). (2) Owner downloads Phase-1 data = cbbo-1s (NOT
  cmbp-1) + minute/defs/ES/VX + ThetaData SPX (~5GB).
- **Sequence:** Codex review → Claude verify → owner sign → Codex implements Phase-1
  reseal set (model-plane contract + additive FT2-08 1s exit tensor/labels +
  provisional latency-calibrated backtest fill) → build+run Path-D backtest → EDGE GATE.

---

## Phase update 2026-08-01 (j) — Codex Phase-1 governance review → repair package written

- **Codex governance review returned NEEDS-CHANGES** (Claude-verified correct against
  source). Blocking: Q(hold) undefined/oracle (foundation q_hold = hindsight upper bound);
  Phase-1 fill law absent; tier contradiction (cbbo-1s vs cmbp-1); source-purity rule
  breaks position-state parity; governance identity unresolved (parent authority
  hard-requires minute exits/floor/feature-restrictions + checker pins hash); two docs
  insufficient (need FT2-08 exit contract + fill law).
- **Repair package written** (Claude writes → Codex re-review → owner signs), plan-approved:
  1. `..._PATH_D_MODEL_PLANE_CONTRACT_PROPOSAL_V2_2026_08_01.md` — 3-way source split
     (market-alpha Databento/ThetaData ONLY / permitted broker position-state / forbidden
     IBKR-alpha+vendor-greeks); frozen per-namespace allowlists (entry=17 unchanged, new
     exit allowlist); fail-closed feature-lineage manifest + negative fixtures; full clock
     spec; Tier-S cbbo-1s.
  2. `..._ONE_SECOND_EXIT_OBJECTIVE_ARCHITECTURE_DESIGN_V2_2026_08_01.md` — DEPLOYABLE
     Q_hold/Q_exit (finite-horizon continuation + flat-slot law, NO oracle); each economic
     term once; slot-cost in $ (minute-forecast honest); switching no-double-count; distrib
     heads exact contract; OOF fold-specific; economic acceptance ≥4/5 folds (§5.4) +
     action-conditioned gate (L449); PR/AP diagnostic-only; floor = deterministic backstop;
     Tier-S claim boundary.
  3. `..._FT2_08_PATHD_1S_EXIT_TENSOR_LABEL_CONTRACT_2026_08_01.md` — additive 1s exit
     tensor/labels on frozen minute entry; occupancy-release semantics.
  4. `..._PATHD_PHASE1_PROVISIONAL_FILL_LAW_2026_08_01.md` — frozen marketable-limit
     backtest fill/no-fill/latency law, identical across labels/replay/PnL; Δ-sensitivity
     band; conservative; Phase-1-provisional (superseded by Phase-2 A1).
  5. `..._PATHD_PHASE1_AUTHORITY_OVERLAY_2026_08_01.md` — immutable overlay pinning parent
     1d215845 + graph 9955085a + A7 + the 4 constituents; resolves governance identity
     WITHOUT resealing parent/legacy; owner signs the overlay.
- v1 model-plane + FT2-60 docs retained as the Codex-review-rejected versions (audit trail).
- **NEXT:** Codex re-reviews the full 5-artifact Phase-1 set (A–E rubric) → Claude verifies →
  owner signs overlay → implement (build FT2-08 1s exit tensor/labels on cbbo-1s, wire fill
  law, train GBT baseline exit on OOF-from-frozen-entry, run serial-replay Tier-S backtest)
  → Tier-S feasibility gate.

---

## Phase update 2026-08-01 (k) — Codex Phase-1 re-review = NEEDS-CHANGES (round 2)

- **Codex re-review: NEEDS-CHANGES again** (Claude accepts — findings correct). Closed:
  Tier-S consistency, broker-state-in-principle, additive FT2-08/fill existence, OOF,
  PR/AP-diagnostic. STILL OPEN (blocking): (1) Q_hold still CIRCULAR (defined via "the same
  learned exit policy" we're training; no frozen reference/Bellman/H/terminal value);
  (2) fill law non-executable + CONTRADICTS FT2-08 (entry ask+tick vs legacy A1) + fee
  double-count vs frozen $3/$4 round-trip reserve + Δ unfrozen + 15:55 minute-language;
  (3) clock concepts listed but no frozen fields/endpoints/quote-age numerics/watermark-fail;
  (4) floor is an EXAMPLE not a preregistered rule (formula/params/priority absent);
  (5) economic terms double-count risk (flat-slot vs switching overlap, no equation).
  High: allowlist asserted-not-frozen; historical(simulated-fill)/live(broker-fill) lineage
  law missing; EXIT calibration only named; overlay hashes placeholder + "no training" vs
  learned-backtest authorization conflict.
- **META (Claude own-error pattern):** two rounds bounced because I wrote design-level PROSE,
  not FROZEN EXECUTABLE contracts. Fix requires a different KIND of doc (equations/params/
  tables). Genuine correctness issues (circular Q_hold, contradictory fills, undefined floor)
  must be fixed regardless; production edge-case formalism is an owner rigor decision for a
  Tier-S throwaway. Concrete Q_hold fix available: advantage vs a FROZEN hold-to-flat
  reference policy (non-circular, deployable, path-computable).
- **DECISION PENDING (owner):** rigor level for the 3rd revision — Tier-S bar (fix
  correctness, defer production edge-cases w/ sign-off) vs full production bar vs step-back.

---

## Phase update 2026-08-01 (l) — SEQUENCING RESET: architecture transition BEFORE model work

- **Owner correction (accepted):** stop jumping to model-training/backtest questions —
  we are still TRANSITIONING the architecture to Databento-live-OPRA-decide. Sub-minute
  structure question is CLOSED (already measured: 93% of minutes move; median within-minute
  bid range ~$80-165/contract; latency sweep confirms seconds-scale movement). The
  descriptive spike + the 5 governance contracts + the model backtest were all premature.
- **Correct order:** (1) make the Path-D architecture real — Databento-live decide -> IBKR
  execute — with a FUNCTIONAL, TESTABLE decide->execute boundary + latency measurement, and
  a codebase reorg other AIs can build on; THEN (2) model training/backtest; THEN (3) the
  governance contracts (Codex's Phase-1 findings become the checklist for that later
  formalization). The 5 Phase-1 contracts (model-plane v2, FT2-60 v2, FT2-08 exit, fill law,
  overlay) are SHELVED pending the architecture transition + evidence.
- **Dispatched:** Codex transition-architecture audit + reorg plan (3 planes: decision
  service / execution adapter / decide->execute boundary + risk governor; reorg map;
  boundary testability; latency-test harness; incremental migration + orientation). Audit/
  design only — no refactor/live/training/contracts yet.
- **CLAUDE self-note:** repeatedly over-ran the owner's sequencing (proposed model work
  while still mid-transition). Hold the line: transition architecture first.

---

## Phase update 2026-08-01 (m) — Path-D exit research pipeline BUILT + smoke-validated

- **Built (Claude, research-grade, quarantined):** the full Path-D exit pipeline on real
  cbbo-1s — non-circular label builder (a_hold = advantage vs FROZEN hold-to-flat; proven
  NON-ORACLE: a_hold <= oracle_adv on 100% of rows, mean -$0.81 vs oracle +$4.38), GBT
  trainer with leave-one-day-out OOF, conservative-fill serial replay, baseline panel.
  Scripts: run_pathd_exit_label_smoke.py, run_pathd_exit_backtest_smoke.py; artifacts in
  protocol101_pathd_exit_research/.
- **Smoke result (6 days, deterministic entry fixture — machinery validation, NOT
  feasibility):** OOF a_hold R^2 = **-0.54 (negative)** = NO genuine out-of-sample skill at
  6-day scale (expected; thin data, echoes WS1). PnL table LOOKS like learned wins (-$9.67
  vs hold-to-flat -$95) but this is an EARLY-EXIT CONFOUND, not skill: exit-immediate also
  beats hold-to-flat (-$36), and R^2 is negative. **Not claimed as signal** (no-reward-hack
  discipline). The guard baselines (exit-immediate + OOF R^2) that expose the confound carry
  to the real run.
- **Status:** pipeline machinery COMPLETE + validated end-to-end on real data. Real
  feasibility answer requires 12-month cbbo-1s (download pending) + frozen-entry OOF (not the
  fixture). Runs in parallel with Codex's transition-architecture audit.

---

## Phase update 2026-08-01 (n) — Codex transition-architecture audit VERIFIED + endorsed

- **Codex transition audit received + Claude-verified** (5 foundational findings confirmed:
  NullSimulator reject-all; PaperOrderIntent broker-shaped-only; executor LMT/DAY+cancel-
  without-proof; paper-default dispatches Protocol160; Protocol160 monolithic). Verdict
  ENDORSED: additive `v4/path_d/` overlay, DON'T touch Protocol160 until the boundary passes
  offline+latency+fake-gateway+authorized-paper evidence.
- **Architecture (sound + safe):** clean planes — contracts (stdlib-only) / decision (no IBKR
  imports) / risk governor (sole submit-authorizer, forced-flat on holding-feed-loss) /
  execution (isolated; full order state machine w/ no-fill escalation + reconciliation) /
  observability / runtime orchestrator. Decide->execute boundary = ExecutionIntentV1 strict
  JSON + ExecutorPort (same port for SimulatedExecutor and IbkrGatewayExecutor -> no `if live`
  branch) -> functional + latency-testable OFFLINE (the owner's requirement). Closes the
  adversarial review's critical findings. Migration steps 1-6 offline (buildable now), 7-11
  live/cutover (gated). Respects CLAUDE.md cleanup rules.
- **Integration:** Claude's exit-model pipeline = the "frozen model / deterministic exit
  policy" node (decision brain); Codex's architecture = the plumbing (nervous system).
  Complementary, parallel, non-blocking.
- **THREE PARALLEL TRACKS:** (1) Codex implements the offline Path-D foundation (steps 1-6 +
  harness; goal drafted); (2) Claude's exit-model research (blocked on 12mo data + entry
  trajectories); (3) owner downloads 12mo cbbo-1s (unblocks track 2). None touch live/broker.

---

## Phase update 2026-08-01 (o) — Path-D data-acquisition plan written (Codex download goal)

- **Planning doc written:** PROTOCOL101_PATH_D_DATA_ACQUISITION_PLAN_2026_08_01.md (owner-
  approved plan). Contains the copy-paste Codex download goal.
- **Confirmed tooling exists (owner was right):** download_thetadata_index_bars.py (ThetaData
  SPX/VIX, --auth-check, paid-data guard, out data/vendor/thetadata/index) — 429 SPX + 439 VIX
  days already on disk from 2024-10-01 (gap-fill only). Plus Databento cbbo-1s/context-proxy
  scripts + build_databento_neural_dataset (official context) + paid_data_guard.
- **Plan (5 phases):** A subscription+auth safety (get_cost=$0 assert; ThetaData --auth-check);
  B acquire trailing-12mo free-under-sub OPRA (definition/ohlcv-1m/statistics/cbbo-1m/cbbo-1s;
  NOT cmbp-1/tcbbo) + auto-approved ~$9 ES/VX futures + ThetaData gap-fill; C process via
  build_databento_neural_dataset --context-mode official -> aligned corpus (minute entry + 1s
  exit substrates); D analyze+test (data-quality report + spend ledger + test_paid_data_guard
  + re-run run_pathd_exit_label_smoke.py at 12mo scale); E starting-line handoff.
- **Spend policy (owner-set):** free-under-sub default; auto-approve ES/VX ~$9; STOP for any
  other paid. Storage outside iCloud, parquet only, ~4.2 GB.
- **NEXT:** owner hands Codex the download goal; Claude verifies on completion -> then the
  entry+exit model plan.

---

## Phase update 2026-08-01 (p) — 12mo corpus acquired + Claude-VERIFIED

- **Codex acquisition COMPLETE + Claude-verified.** Window 2025-08-01→2026-07-31, 251 aligned
  OPRA sessions × 5 schemas; official SPX/VIX pairing 251/251; ES/VX futures context present.
  Exit-label pipeline re-run at scale: 37,047,600 rows, non-oracle (a_hold<=oracle 100%,
  is_oracle=False) + no-leakage PASS. Integrity 3,708 files/21.48 GB hashed; 38 tests pass.
- **Spend VERIFIED within authorization:** $8.55 of $9 — OPRA subscription $0 (free),
  ThetaData $0 (existing sub), ES/VX $8.55 (authorized). No cmbp-1/tcbbo/older-L1 (129 on-disk
  cmbp files are 2024 legacy; 0 in-window). Evidence: protocol101_pathd_data_acquisition/
  {handoff, data_quality_report, spend_ledger, pathd_exit_label_validation_12m}.
- **HONEST NOTE:** corr(pnl,a_hold) collapsed -0.42 (6-day smoke) -> +0.019 (12mo). The smoke
  "hint" was small-sample noise; real feasibility needs the full model (all features + proper
  OOF), not naive correlations. No false expectation into modeling.
- **STARTING LINE REACHED:** corpus ready for the entry+exit model plan. Parallel: Codex offline
  execution-architecture implementation (steps 1-6). NEXT: plan the entry + exit models.

---

## Phase update 2026-08-01 (q) — offline execution foundation VERIFIED; model-plan goal drafted

- **Codex offline execution foundation (steps 1-6) Claude-VERIFIED + endorsed.** v4/path_d/
  package (contracts/decision/risk/execution/features/compat) matches the audit. Import-boundary
  invariant TESTED + passing (test_path_d_import_boundaries_forbid_ibkr...; no ib-lib imports
  anywhere — "IBKR" hits are enum strings only). Feature-shim byte/hash equivalence TESTED
  (legacy preserved). ExecutionIntentV1 semantic-hash, governor forced-flat-on-feed-loss,
  disconnect->UNKNOWN->reconcile, received-clock offline E2E all TESTED + passing. Artifacts:
  offline_e2e_transcript.jsonl + latency_bounds.md (24 rows, 6 sessions x 4 rungs). 24 dedicated
  path_d tests pass; Codex's "48" = path_d + shim-affected legacy (reconciled, legitimate). No
  Protocol158/160/runtime/registry edits. Step 7 (live IBKR adapter) + cutover deferred.
- **Entry+exit MODEL-PLAN Codex goal drafted** (owner-approved plan): a Codex goal that PRODUCES
  the research-grade entry+exit model plan (design), binding to the built v4/path_d/ boundary,
  reusing the verified corpus + validated exit pipeline + shelved FT2-60/model-plane/A6 designs,
  with self-fooling guards (exit-immediate baseline, OOF R², non-circular target) baked in; build
  deferred to a later goal after Claude-verify + owner-review.
- **State:** two parallel tracks healthy — (execution plumbing) offline foundation done+verified;
  (decision brain) model plan next. NEXT: owner hands Codex the model-plan goal; Claude verifies
  its output; then owner review -> build goal -> feasibility gate.

---

## Phase update 2026-08-01 (r) — entry+exit model PLAN Claude-VERIFIED (review-ready)

- **Codex entry+exit model plan** (PROTOCOL101_PATHD_ENTRY_EXIT_MODEL_PLAN_2026_08_01.md)
  Claude-verified: PASS, review-ready. Fixes ALL prior-round defects + bakes in guards:
  non-circular A_ref = H_t - E_t (frozen hold-to-flat vs exit-until-filled; oracle audit-only/
  forbidden); A6 gate on conditional MEAN (q10 rank-only, drops h3/h5); leak-free fold-specific
  OOF-from-frozen-entry (never full-fit); economic acceptance Box D beats best honest comparator
  pooled + >=4/5 outer folds + bootstrap LB + action-conditioned calibration gate (PR/AP
  diagnostic-only); deterministic floor floor_bid=max(0,0.5*P_entry+3/100); one shared fill-law
  hash ($1.50/side once, entry fee cancels — no double-count); binds to ExecutionIntentV1 +
  DeterministicGovernor + SimulatedExecutor; self-fooling guards (must beat exit-immediate +
  matched-random, OOF-R2 tripwire, negative controls, mutate-future, large-win skepticism,
  scale-lesson). Firewall arithmetic checks (70+5+140+36=251; 30 protected holdout, opened once).
- **Honesty catch:** 30-holdout is protected-from-model/economic-inspection but NOT pristine raw
  (aggregate label stats already leaked) — caveat must appear in every result.
- **5 OPEN OWNER DECISIONS (Claude recommends CONFIRM all):** (1) entry=signed 17, enrichment
  exit-only; (2) H=300s, 0.25 penalty, 90% levels; (3) fixed net -50% floor (vs governor moving
  80%); (4) VIX/ES/VX = diagnostics/strata not alpha; (5) accept holdout-not-pristine caveat.
- **NEXT:** owner confirms the 5 -> BUILD goal (execute the plan: entry HGB->neural, freeze,
  exit OOF, four-box replay through the boundary) -> Tier-S feasibility gate. Nothing built yet.

---

## Phase update 2026-08-01 (s) — 5 decisions confirmed; BUILD goal drafted + handed off

- **Owner confirmed all 5 open decisions:** (1) entry=signed 17, enrichment exit-only; (2) H=300s,
  0.25 penalty, 90% levels; (3) fixed net -50% deterministic floor; (4) VIX/ES/VX diagnostics/
  strata not alpha; (5) holdout-not-pristine caveat accepted. (Owner asked + understood the
  "deterministic backstop" rationale: a catastrophic floor must be fixed/mechanical/knowable, not
  trailing/forecast-based — a seatbelt/airbag, not a second decision-maker.)
- **BUILD goal drafted + handed to Codex:** execute PROTOCOL101_PATHD_ENTRY_EXIT_MODEL_PLAN as the
  binding spec, per §8 sequence with stops (freeze prereg -> entry HGB->neural + gate [stop if entry
  fails] -> freeze entry -> exit trajectories/labels from frozen OOF -> exit HGB->neural + §7 guards
  -> four-box replay through the v4/path_d/ boundary + §6.3 economic acceptance -> open 30-holdout
  ONCE). Tier-S/quarantined; no live/broker/paper/promotion/governance-freeze/runtime/cmbp-1.
  Verdict = PASS / no_genuine_signal / insufficient_evidence / owner_decision_required.
- **THIS IS THE FEASIBILITY-GATE RUN** — the honest "does a Path-D entry+exit show edge?" answer
  the whole pivot built toward. Nothing built yet; Codex executes; Claude verifies the result
  (reproduce guards, confirm holdout opened once, no self-fooling) -> owner decides next phase.

---

## Phase update 2026-08-01 (t) — background feature-signal probe (entry microstructure)

- **Model-free directional probe** (while Codex builds signed-17): do the Path-D re-admitted
  microstructure fields carry incremental ENTRY-ranking signal? Training-only (14 sampled
  fit-history sessions, 57 hourly decision-minutes, 406 band candidates), within-minute Spearman
  vs forward-30min upside; incremental = vs moneyness-residual. NO holdout, NO model.
  Artifacts: protocol101_pathd_feature_research/{entry_feature_probe.json, FINDING_...md}.
- **FINDING: order-book SIZE IMBALANCE is the standout** — raw within-minute rho +0.167,
  incremental-over-moneyness +0.129, t~2.59 (58% of minutes positive) — STRONGER than the
  moneyness geometry the signed-17 relies on, and genuinely additive to it. Open interest weak
  secondary (+0.089 incr). Spread/volume/quote-age negligible. bid_size/ask_size are exactly the
  fields the parity mask REMOVED -> the mask was discarding a real (moderate) entry signal Path D
  re-admits. Supports the owner's "widen the entry" hypothesis.
- **CAVEATS (honest):** moderate magnitude, directional probe only, moneyness-only control
  (delta/gamma null in base view), rank-signal != tradeable-edge-after-costs (imbalance decays
  fast on 0DTE). A hypothesis to TEST, not proven edge.
- **DISPOSITION:** do NOT add to the current run (would break decision #1's clean signed-17
  baseline / change two things at once). IF the signed-17 build underwhelms (or as a follow-up),
  test widened entry = signed-17 + order-book imbalance (+OI) under the SAME fair firewall +
  guards. Extra-features direction now has concrete, prioritized support.

---

## Phase update 2026-08-01 (u) — Databento historical->live FIELD PARITY (owner catch)

- **Owner catch (correct + important):** "same vendor" != "same fields"; the live OPRA feed isn't
  automatically the training download. Ran a field-parity inventory (FINDING_databento_live_field_
  parity_2026_08_01.md).
- **Findings:** (a) live OPRA = CONSOLIDATED schemas cbbo-1s/cmbp-1/cbbo-1m; MBP-1/TBBO RETIRED for
  OPRA May 2025 -> our training (cbbo-1s/cbbo-1m) matches the live family; do NOT build on mbp-1/
  tbbo. (b) bid/ask + bid_size/ask_size ARE live (consolidated CBBO carries sizes; confirmed in our
  own cbbo-1s parquet bid_sz_00/ask_sz_00). (c) **open_interest / stat_open_interest are STATISTICS
  schema = DAILY/EOD, NOT real-time** -> barred as intraday features (prior-day static only).
  (d) greeks self-computed (live-reproducible); SPX via ThetaData live.
- **Impact on the feature finding:** size_imbalance (standout, +0.129 incr) is LIVE-USABLE ->
  widen-entry direction is deployable. open_interest (weak secondary) DISQUALIFIED as intraday
  (EOD-only). So the widen-entry candidate narrows to order-book imbalance (+ always-safe
  price/spread), NOT OI.
- **Governance hook:** the model plan §3 already requires a per-feature "historical/live twin"
  fail-closed; THIS inventory is its ground truth. Enforce: only live-usable fields as model
  inputs; EOD-only fields barred intraday; live adapter (step 8) must reproduce the exact cbbo-1s
  schema/sampling (or cmbp-1->1s with the same rule) — verify on the first live-shadow day.

---

## Phase update 2026-08-01 (v) — build-correction proposal (research vs frozen build)

- **Checked Codex's build state (filesystem, read-only):** prereg FROZEN (feature_lineage.json,
  preregistration.json, session_assignments.json); building decoder/dataset/model machinery
  (v4/research/pathd_entry_*.py, path_d/execution/research_*.py); PRE-FIT, PRE-EVIDENCE
  (holdout_open_count=0). Build dir: protocol101_pathd_entry_exit_model_research_2026_08_01/.
- **Findings vs build:** (1) ENTRY clean — signed-17 only; OI/sizes/volume FORBIDDEN as entry alpha.
  (2) EXIT uses live-safe microstructure (bid/ask sizes, imbalance, spread) BUT also
  last_causal_open_interest w/ 90s carry — OI is EOD-only (no real-time twin) -> live-parity gap.
  (3) BURNED outer-fold-1 primary: reason FROZEN_FOUNDATION_DRIFT_BEFORE_DECODE, access_count=0,
  no-reopen -> weakens >=4/5 gate + signals foundation instability.
- **Correction proposal written** (PROTOCOL101_PATHD_BUILD_CORRECTION_PROPOSAL_2026_08_01.md,
  evidence->build->fix): P1 root-cause the drift + re-freeze/restore-5-folds-if-benign (or restate
  gate if real) + add byte-identical foundation-stability gate before fold decode; P2 drop/redefine
  last_causal_open_interest (EOD) + verify minute_volume live + enforce field-parity live-twin ground
  truth across all 17+49 features; P3 record size_imbalance (+sizes) as pre-vetted live-safe
  widen-entry follow-up (entry stays signed-17 this run). Codex goal drafted; applies at pre-fit
  pause. NEXT: Codex applies -> Claude verifies -> then the fit.

---

## Phase update 2026-08-01 (w) — Codex corrections APPLIED + Claude-VERIFIED (PASS)

Codex applied the 3-correction proposal and returned `STOP_FOR_CLAUDE_VERIFICATION` at the pre-fit
boundary. Corrected build root: `protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/`
(supersedes the original frozen root). Claude verified independently (reproduced hashes, not
rubber-stamped) — **PASS**:

- **P1 (drift/folds):** corrected prereg SHA-256 `4a01f145…` reproduced byte-identically (matches
  sidecar + restoration receipt); `foundation_generation_sha256 caf3e086…` matches. Root cause =
  a synthetic TEST authorization reached the real validator before capability auth → classified
  `BENIGN_TEST_CONTAMINATION`. All 5 outer folds restored (`PRISTINE_ABSENT`), acceptance still
  pooled AND ≥4/5. Burn preserved immutably (`access_count:0`, all receipts null, `BURNED_NO_REOPEN`).
  Stability gate is real: `_rehash_dataset_source_files` rehashes sealed source bytes, blocks
  symlink/traversal/malformed receipts, FAILS LOUD on mismatch (55 = 47+5+1+2 frozen hashes).
- **P2 (OI/live-twin):** entry = exactly signed-17 (byte match, `entry_microstructure_enrichment:false`);
  exit 49→48, delta = precisely `last_causal_open_interest` removed (now `NO_INTRADAY_LIVE_TWIN`/EOD);
  `last_causal_minute_volume` retained but `minute_volume_adapter_receipt_required_before_exit_fit:True`
  (exit fit blocked); `implementation_receipt_required_before_fit:True`; `feature_without_live_twin`
  fixture rejects OI-as-intraday.
- **P3 (lead):** `size_imbalance` recorded as `p3_forward_only_entry_enrichment` (fresh separate
  prereg required; forbidden as current-run alpha), enforced by `test_p3_widen_entry_lead_is_forward_only`.
- **No fit / no holdout:** `model_fit_executed:false`, `holdout_open_count:0`, folds pristine. Ran
  `test_pathd_foundation_correction.py` + `test_pathd_evidence_gate.py` independently → **20/20 pass**
  (safety-scanned: no broker/data/network).
- **Honest caveat carried:** 30-session firewall protected from model/economic inspection but not
  pristine from already-published full-corpus aggregate label statistics. Highest claim = Tier-S
  feasibility, not-promotable.

**TWO GATES REMAIN before any fit** (verdict `owner_decision_required`): (1) owner/Claude decision on
`COMPOSITE_CALIBRATION_TERMINAL_RULE` (before machinery/stability sealing); (2) the `minute_volume`
live adapter receipt (before exit fit). NEXT: owner resolves the terminal-rule decision.

GitHub health: 3 curated commits (gitignore hardening / path_d + Path-D package / worktree sync) were
already made + pushed after plan approval; this round adds the verified correction unit
(`v4/research/` code, Path-D scripts + tests, corrected audit dir, gitignore corrected-dir exception).

---

## Phase update 2026-08-01 (x) — corrected v2 -> v3 + governance decisions VERIFIED (PASS)

Two more correction rounds landed and were Claude-verified independently (hashes reproduced,
tests run, no rubber-stamp).

**Corrected-v2** (`..._corrected_v2_2026_08_01`, prereg `f5e8ed8b`): dropped `last_causal_minute_volume`
too (no proven live adapter), so exit 48->47; corrected entry-D `D.near_atm` lineage to completed
OPRA `cbbo-1m` (live-derivable). Entry still exactly signed-17. Drift re-confirmed
BENIGN_TEST_CONTAMINATION via timeline (burn 19:45:37 predates freeze 19:51:02 by ~325s; txn id =
test fixture) + semantic session-assignment identity (canonical hash match; raw bytes differ only by
JSON formatting). Committed 9d8df1d8.

**Terminal-rule investigation:** Codex's read-only analysis reconciled the "<60" discrepancy exactly
(`known_exit_weight_session_counts_by_fold = [0,0,19,48,56]`, min 60 -> ALL five learned-exit folds
abstain, not just fold 1) and found a hole in my Candidate-A lean (fold-scoped abstain has
survivor-selection risk: dropping a hard-era fold can inflate pooled economics). It recommended
**B-R** (status-preserving whole-run hard stop) and surfaced a NEW blocker: the A_ref q50/q90
role contradiction (q90 both EXIT-gate-required and diagnostic-isolated). Both independently verified
against the prereg.

**Corrected-v3** (`..._corrected_v3_2026_08_01`, prereg `8ed2b729`, foundation_generation `da65e572`):
- **B-R adopted** at `/calibration_and_statistics/composite_calibration_terminal_rule`:
  `policy=B_R_STATUS_PRESERVING_WHOLE_RUN_HARD_STOP`; all-five-valid before the >=4/5 economic gate;
  `four_as_five_or_survivor_pooling=False`; status_precedence [invalid_result, insufficient_evidence,
  owner_decision_required, no_genuine_signal, PASS]; failure_mapping keeps INVALID_TARGET_COVERAGE->
  invalid_result vs INSUFFICIENT_EVIDENCE->insufficient_evidence; `forbidden_rescue` bans fold/target
  deletion, survivor pooling, status relabeling, same-generation reseed. 11 nested_structural_skip
  blocks match the [0,0,19,48,56] geometry.
- **A_ref contradiction resolved** (Codex Option 1): A_ref q10/q50/q90 = one atomic decision-critical
  node; only the 4 local-path families (downside/recovery/giveback/remaining_tail_300) stay
  diagnostic-only; q90 = binding EXIT upper-coverage support; guarded against the naive
  `q10(-A_ref)=-q90(A_ref)` tie claim.
- `resolved_owner_decisions=[AREF_ACTION_CALIBRATION_ROLE, COMPOSITE_CALIBRATION_TERMINAL_RULE]`,
  `unresolved_owner_decision=None`. entry-17 / exit-47; folds PRISTINE_ABSENT; model_fit/seal/holdout
  all False/0; burn preserved; v2 preserved byte-for-byte. Independent tests: 172 passed across the
  Path-D suite.

Remaining before any fit (per v3 prefit_pause, all authorizations still False): Claude verification
(this, PASS) -> a SEPARATE post-verification release, then machinery/stability seal. Known geometry:
exit abstains this data era (all folds <60 exit-weight sessions); entry feasibility is what this run
can produce. Tier-S, quarantined, not-promotable.

---

## Phase update 2026-08-01 (y) — seal goal failed-closed (my error) -> corrected-v3.1 release generation VERIFIED (PASS)

**Seal-only goal was impossible by design (my mistake, not Codex's).** Codex stopped fail-closed and
wrote nothing. Verified the frozen contract: `assert_correction_prefit_release` requires
machinery_seal + foundation_stability_seal + **model_fit** all authorized together (plus
claude_verification_pending=False, separate_post_verification_release_required=False), and
`seal_foundation_stability_receipt` internally calls that release gate AND requires a corpus-integrity
receipt. So seal-without-fit and seal-without-corpus are both contradictory: seal+corpus+fit are ONE
atomic owner-authorized, Claude-verified release. My separate-seal-gate framing was wrong.

**Decision (owner):** release-generation path (NOT Codex's contract-repair-to-separate). Chosen for
leverage: the contract bundles them by design, v3 is already verified, and the fit has an internal
fail-loud stability gate — separating seal from fit would be unnecessary surgery.

**corrected-v3.1** (`..._corrected_v3_1_2026_08_01`, prereg `7fa621d6`, foundation_generation
`1b6eb620`): authorization-only superseding generation. Claude-verified independently:
- feature_lineage.json + session_assignments.json **byte-identical** to v3.
- Full recursive prereg diff = **exactly 8 leaves**: 7 authorization flag flips
  (machinery_seal/foundation_stability_seal/model_fit/corpus_decode/nested_or_outer_evidence_open
  ->True; claude_verification_pending/separate_post_verification_release_required->False) + the
  `supersedes` pointer (relationship AUTHORIZATION_ONLY_SCIENCE_BYTE_IDENTICAL). No science changed.
- protected_holdout_open_authorized, foundation_stability_receipt_sealed, live_or_broker_action all
  stay False. `assert_correction_prefit_release(v3.1)` PASS. Folds PRISTINE_ABSENT;
  model_fit_executed=False; holdout 0. v2/v3/burn byte-identical; no seal receipts written.

**Next gate = the entry feasibility fit (Goal 2)** fires against v3.1: seal machinery/lineage ->
corpus-integrity -> foundation-stability -> fit entry (HGB->neural) OOF on 5 outer folds under B-R ->
economics vs P5 + matched-random under the shared transparent exit -> negative controls must fail ->
mutate-future audit -> typed verdict. Exit abstains this era; protected holdout stays sealed.

---

## Phase update 2026-08-01 (z) — build gaps + corrected-v3.2 gate-hardening VERIFIED (PASS after reconciliation)

The fit-only goal was premature: the EXECUTABLE pipeline wasn't built (runner bound to v3, no
end-to-end campaign entrypoint, P5/matched-random/negative-control/SPX-ref producers uninstalled).
Codex fail-closed correctly. During the build it found a real governance hole: the immutable
`pathd_evidence_gate.py` could increment access_count / open evidence BEFORE validating the B-R
calibration-scope-gate receipt (0 references to it in the gate) — a bypassable fail-closed rail.

**Decision (owner):** corrected-v3.2 — fix the immutable gate + refreeze (not a bypassable wrapper),
per the "safety rails must be hard" standard.

**corrected-v3.2** (prereg `c69e763a`, foundation_generation `5d0d5728`, source_policy `618525ce`):
- GATE FIX verified at code level: `_validated_calibration_scope_gate_before_open` requires a VALID
  `pathd.calibration_scope_gate_receipt.v1` (evidence_access_count==0) and is called BEFORE lock,
  capability, and any access-count mutation (fail-before-open at the immutable layer).
- Science byte-identical to v3.1 (feature_lineage + session_assignments identical; A_ref `b30e57ef`
  + B-R `84b3467f` unchanged; entry-17 / exit-47). Only gate source + source_hash_policy + supersedes
  changed. Executable-generation enforcement added (fixes runner-bound-to-v3).
- Real deterministic producers built: P5, shared control-exit selection, HGB/NEURAL, 22 negative
  controls, 8 matched-random schedules, common replay, causal SPX-reference; caller-authored PnL/
  verdicts/pass-flags rejected.
- Correctly UNRELEASED (model_fit_authorized=False, claude_verification_pending=True); folds
  PRISTINE_ABSENT; nothing fit/decoded/sealed/holdout-opened; v1/v2/v3/v3.1 + burn byte-identical.

**Verify-don't-rubber-stamp caught a real gap:** Codex reported "all passed" but my independent full
11-file suite showed **4 registered governance tests RED** — a new `_assert_corrected_v32_release_
if_present` guard fail-closing (correctly) because `claude_verification_release.json` is absent, plus
one SPX-ref source-inspection mismatch; Codex's checkpoint subsets missed them. Reconciliation goal
issued. Codex reconciled honestly: tests now assert the guard fails closed when unreleased AND assert
original tamper/stale/missing-lineage behavior via an in-process monkeypatch fixture (NO on-disk
release file faked, no guard weakened, no science changed). My independent re-run: **175 passed, 0
failed**; release file still absent; hashes unchanged.

**Next gate:** the v3.2 RELEASE generation (owner post-verification release, like v3.1 was for v3) ->
then the entry feasibility fit against the released v3.2 executable. Tier-S, quarantined, not-promotable.

---

## Phase update 2026-08-01 (aa) — corrected-v3.2 RELEASE issued + VERIFIED (PASS)

Owner post-verification release for the gate-hardened v3.2 executable, via a distinct superseding
release document (NOT a prereg mutation): `claude_verification_release.json` (self-hash `17c028d5`,
bound to prereg `c69e763a` / foundation_generation `5d0d5728`). Claude-verified independently:
- Release flags authorize machinery_seal/foundation_stability_seal/model_fit/corpus_decode/
  nested_or_outer_evidence_open; protected_holdout_open=False; live/broker not granted.
- FROZEN prereg still reads UNRELEASED (model_fit_authorized=False, claude_verification_pending=True)
  — authorization lives only in the release doc; foundation byte-identical.
- ADVERSARIAL check: temporarily removed the release artifact -> production guard
  `_assert_corrected_v32_release_if_present` FAILS CLOSED ("release chain is incomplete"); restored
  byte-identical. So the release is load-bearing and the state-aware tests still enforce fail-closed.
- Full 11-file suite: 175 passed (release present). Science (feature_lineage/session_assignments/
  A_ref/B-R/entry-17/exit-47), v1-v3.1, and fold-1 burn all byte-identical. Folds PRISTINE_ABSENT,
  holdout 0, nothing fit/decoded/sealed/evidence-opened.

**VX backfill decision: NO** (owner-facing analysis). VX futures gap (only ~2026-04->07, ~$16 to
backfill ~167 sessions at ~$0.095/session) is NOT worth it: VX is non-alpha context-diagnostics-only
(absent from entry-17/exit-47; vix_es_vx_model_alpha=False), the gap is already frozen into the
contract (VX availability_boundary_session=2026-04-01, pre_boundary NOT_REQUESTED, dedicated vx_era
bin), the primary vol context (VIX index) is 251/251, and using backfilled VX would require breaking
the verified foundation. VIX index full-year; VX futures intentionally partial.

**Next gate: entry feasibility fit against released v3.2** — now UNBLOCKED (release present, gate
green). Retarget the fit goal from v3.1 to the released v3.2 executable.

---

## Phase update 2026-08-02 (bb) — LEAN fit RAN; signed-17 entry = weak/decaying signal; owner accepts + reconsiders scope

**v3.3 verification FAILED** (real fit faked in 100% of tests: `_install_fixture` stubs
`_fit_outer_scope` + `opportunities_from_authorized_dataset`; the "real fit orchestrator" test
monkeypatches every `fit_*`; the regression guard is `inspect.getsource`, not runtime). Same
consumer-only gap as v3.2, one layer deeper. Owner chose the LEAN path.

**LEAN feasibility fit executed on real data (first real fit in 6 generations).** Direct script
(scratchpad/lean_entry_feasibility_v2.py) reusing the frozen decisions (signed-17, HGB params,
$3-8 band, OOF, holdout NEVER loaded) with processed net-PnL labels under one fixed shared exit.
5-fold OOF Spearman(pred, realized net PnL):
  fold1 +0.146 | fold2 +0.061 | fold3 +0.061 | fold4 +0.003 | fold5 **-0.110**
Signal **decays monotonically and reverses** by the most recent/largest fold. Mean rho +0.03
(negligible); econ +$32/session (3/5 positive). Negative controls behaved (shuffled ~0,
sign-reversed = -real); in folds 4-5 shuffled ties/beats real -> no signal left in recent data.
Classic illusory/early-regime signature (cf. attempt_131 PF 1.87->1.03). **Leans no_genuine_signal.**
Caveats: faithful-not-exact signed-17, one exit policy (not forward-MFE composer), single seed,
per-session (not per-minute) economics, no B-R/P5. Research-grade indication, not the frozen verdict.

**Owner decision: ACCEPT the signal is weak; reconsider scope.** signed-17 entry-as-is has no
robust standalone edge in this 12-month window. The science CODE is sound (causal/no-leak/calibrated
per the deep audit), so the finding is trustworthy. Meta-lesson reaffirmed: governance apparatus
(~30k lines, 6 generations) massively outran the research question; the lean fit answered it in ~1hr.

**Open leads for scope reconsideration:** (1) widen-entry with `size_imbalance`/order-book (the
FINDING_entry_microstructure_signal probe found it the STRONGEST incremental signal, +0.129 > the
moneyness geometry signed-17 relies on; substrate has bid_size/ask_size) — cheapest evidence-backed
next probe; (2) exit/lifecycle focus (data-limited this era); (3) different strategy class/question;
(4) bank infrastructure + pause. v3.3 working tree is uncommitted/unused pending owner decision.

---

## Phase update 2026-08-02 (cc) — honest autoresearch loop BUILT + RAN; entry-feature hypothesis FALSIFIED; holdout SEALED

Built the honest autoresearch loop (`v4/research/lean_autoresearch/`): pre-registered 144-config grid
(4 feature sets S0-S3 x depth x leaf x threshold x 3 exits), 5 disqualifier guards (negative controls
fail, forward-stability incl. fold-5, min-power, economic bar, leakage tripwire), fixed budget (no
"iterate until pass"), one-shot sealed-holdout protocol. Preregistration frozen + hashed BEFORE running
(prereg_sha256 9db41e6d). Self-proven it can say NO (pure noise fails the guards).

**Full 144-trial run: 27 survivors, but the result FALSIFIES the entry hypothesis.** Survival is FLAT
across feature sets (S0=7, S1=6, S2=7, S3=7) and 100% concentrated in the SHORT 25-min exit (0/48 mid,
0/48 long). So: (1) the widen-entry/microstructure lead is DEAD — signed-17 alone does as well as
+size_imbalance/depth/spread/iv/theta; the earlier "+0.129 incremental" probe did NOT become a
model-level edge. (2) The only forward-stable effect is an EXIT-POLICY effect (quick 25-min hold), not
entry selection, and it's weak (OOF rho +0.05-0.09, worst-fold econ +$9-25/session).

**Owner decision: do NOT fire the one-shot holdout** — declined to spend it on an exit artifact where
the entry features don't discriminate. **Holdout remains SEALED (0 opens), preserved for a future
genuine entry hypothesis.** This is the disciplined outcome: the loop found forward-stable survivors,
we read them honestly (features irrelevant), and we did not manufacture a "pass."

**Durable takeaways:** (a) no robust signed-17 OR widen-entry ENTRY edge in this 12-month window;
(b) FORWARD LEAD: the short 25-min hold is the only forward-stable exit -> points at the EXIT/lifecycle
direction (data-limited this era), not entry; (c) the lean autoresearch loop is reusable honest
infra for the next hypothesis. Meta: lean path answered in ~1 day what 6 governed generations couldn't.

---

## Phase update 2026-08-03 (dd) — CORRECTION: lean-loop leakage bug; autoresearch_v2 CONFIRMED a modest entry edge

**Correction to (cc):** Codex's audit found my committed lean loop had a NON-CAUSAL bug — `gt_session_q60`
computed the entry threshold from the whole test session's scores (future minutes), tainting 16/27
S0-S3 survivors. Plus the sign-reversed guard never fired and matched-random used process-randomized
`hash(session)`. So my confident "signed-17 entry FALSIFIED" was PREMATURE — made under a buggy,
crude, non-serial evaluator with the wrong (120-min) exit. Owning this: verify-my-own-work failed here;
I committed a leaky evaluator and over-claimed a negative.

**autoresearch_v2 (Codex-built, ~5,420 LOC) replaced the lean loop + the governed Path-D orchestration.**
Experiment-compiler design: agents submit TYPED JSON hypotheses; the ENGINE owns causality lint
(`assert_mutate_future_invariant` — mutate future outcomes, assert decision-time scores unchanged; my
q60 bug would FAIL this), threshold-free OOF caching, paired component screens, session-blocked maxT
correction, strict one-account serial replay (simulator-v5), power routing, semantic-hypothesis dedup,
and 7 canonical terminal statuses (dev runs cannot emit CONFIRMED_EDGE).

**Result: CONFIRMED_EDGE (Claude-verified, calibrated).** Winner `signed18_model_side_nearest`: HGB on
the 18-col signed-17 (microstructure/vol-time REMOVED — didn't help; learned contract-ranking REMOVED —
didn't beat nearest-ATM), contributing TIMING (first score>0 per block) + SIDE (call/put); deterministic
nearest-ATM entry; frozen 25-min exit. Dev: +$542/session vs momentum-side control, 5/5 folds. One-shot
holdout (29 sessions, opened ONCE, model frozen before access, verified holdout_open_count=1):
**+$540.79/session, 95% CI +$90-$991, p=0.0092, 21/29 positive**; candidate $153/trade vs control
$67/trade; model avoids the daily-loss-stop 15x more (3 vs 18). Verified: causal (mutate-future guards),
one-shot discipline clean, 12 adversarial engine tests pass. CALIBRATION: real but MODEST + UNCERTAIN
(CI low end +$90 ~ the $100 practical bar; regression-to-mean after selection likely), OFFLINE only
(not live), holdout now SPENT, one model family. First genuinely credible positive signal in the project.

**Next gate = runtime DECISION parity via Databento LIVE OPRA (+ ThetaData live), NOT old IBKR** (Path-D
made decide-parity same-vendor; IBKR is execution-only = fill parity). Then a separate paper-promotion
review. Model SHA c5d0115b; confirmation prereg a579adf4.

---

## Phase update 2026-08-03 (ee) — signed18 edge INVALIDATED by parity gate; causal Phase-1 rebuild; purchase-ready

The autoresearch_v2 "confirmed edge" (signed18, +$540/session) was **INVALIDATED** by the runtime
decision-parity gate: training used a **60-second SPX look-ahead** (ThetaData bar stamped t; its close
is only causally available at t+60s). Under the correct t−60s clock only 18% of decisions reproduce, so
the edge was largely the leak. My verification missed it — I checked the future-outcome guard + one-shot
discipline but NOT feature-availability-clock parity (the exact train↔live parity Path-D exists for);
the parity gate caught it before any live/paper. signed18 quarantined INVALID_EXPERIMENT; holdout SPENT.

**Causal Phase-1 rebuild committed (cba15b1a):** pathd_phase1_entry/exit/replay/storage.py rebuild the
signed-18 from raw at the causal t−60s clock (bar stamped t structurally excluded); conservative
one-tick-through fill; 5 OOF folds + prequential initial-history training-only receipts; full-dev shadow
can't make exit trajectories; 52 causal 1-second exit features; four-box + matched-random. Claude-audited
PASS (clock, fill/label, entry→exit OOF firewall). Live-OPRA parity proven (914/914 exact CBBO-1m);
emission lag 2336ms measured. Runtime boundary = read-only databento-no-order + ibkr-paper-dry-run.

**Purchase-ready (Phase A verified):** footprint 89.69 GB < 150 GB cap (corrected my earlier ~5-15GB
under-estimate; trajectories run to 15:55 not 25min, 9 label copies); suite 193 passed / 4 failed (the 4
= legacy v3.2 release-reconciliation drift, NOT Phase-1); SSD runbook fail-closed (5-guard destructive
erase). Owner buying a 2TB (workspace, not data — 12mo corpus already local). Next: Phase B
owner-supervised setup -> Phase C causal training. NO_INCREMENTAL_EDGE is the honest likely outcome;
holdout spent so even a promising causal result needs fresh months to confirm.

Open housekeeping: retire the superseded v3.2 release-reconciliation gate so the 4 governance tests run
their real assertions (do NOT xfail — that would suppress genuine invariant coverage; do NOT rewrite the
immutable receipt). Codex task.

---

## Phase update 2026-08-03 (ff) — Stage 0 SSD workspace COMPLETE; corpus independently verified

Codex retired the invalidated v3.2 release-reconciliation gate without xfail, receipt rewrite, or
invariant weakening (`fa1d9b08`; governance test file 7/7 passed). The verified external whole disk was
`/dev/disk4`, 2,000,365,371,904 bytes. It was owner-confirmed, erased, and encrypted as ordinary APFS
`AR_TRADING_DATA`; Time Machine is stopped/unconfigured and the volume has no Backup role.

The real Mac exposed two fail-closed storage-preflight portability defects: current `diskutil -plist`
uses boolean `Encryption` rather than `Encrypted`, and accepts a mountpoint but not arbitrary child
directories. Codex added strict equivalent-key handling (missing/false/contradictory still fail) and
same-device mountpoint resolution (`b0c5c73d`); the Phase-1 fixture set passes 19/19 and authoritative
preflight now returns `PASS`.

The first local-copy attempt was interrupted by an accidental owner eject after 637 files / 3.84 GiB.
The incomplete marker worked: no manifest was issued and the partial tree was preserved, not trusted,
under a manifest-backed quarantine at
`/Volumes/AR_TRADING_DATA/reports/quarantine/relocation_interrupted_20260804T002438Z/`. `fsck_apfs -n`
then passed. The clean retry returned `COPIED_AND_VERIFIED`, `source_preserved:true`, 3,708 files and
21,484,792,678 bytes. Independent full re-hashes of source and destination matched manifest SHA-256
`7929a43e6b3e3398991b78ba9e937e006531b76b1b0cd1e5480b35b12cb550d6`. Post-copy preflight reports
98.704% free and 25,609,636,917 bytes allocated under the 150 GB cap.

**Next gate:** independent Claude verification of Stage 0, then explicit owner authorization for Stage 1
causal model training. No model fit, firewall opening, broker/paper action, or paid data request occurred.

---

## Phase update 2026-08-03 (gg) — causal Phase-1 ran; authoritative verdict `UNDERPOWERED`; STOP

The owner waived independent Claude verification of Stage 0 and explicitly authorized Stage 1. Codex
ran `materialize-entry → train-entry → build-trajectories → train-exit → replay` over exactly the first
215 development sessions. The 36-session firewall remained closed (`holdout_open_count=0` in the frozen
assignment; every generated campaign/replay says `protected_holdout_opened:false`). No broker, paper
order, paid download, promotion, default, runtime flag, or launch scheduling path ran.

Entry campaign `c7a9ae05...71c` produced 1,167 OOF receipts, including 153 learned evaluation trades and
878 deterministic-control evaluation trades. Exit construction produced 1,167 feature partitions,
1,167 baseline-label partitions, and all 9,336 fee/latency partitions (13,730,079 baseline rows); a full
identity/Parquet audit found zero errors. Exit campaign `69540b75...a36` produced exactly 1,031 OOF
prediction partitions and positive target-skill correlation in 4/5 folds.

The frozen replay returned **`UNDERPOWERED`**, semantic hash `b07784a0...5807`. The learned subset has
60 sessions but only 3 outer folds; the gate checks that subset, not the full 166-session/all-five-fold
evaluation index. Negative controls were all rejected, so underpowered takes precedence. Independently,
the economics were adverse: learned integrated `-$7,024` versus best comparator `matched_random_3` at
`-$4,474`; bootstrap LCB `-$150.20`; fold deltas `+$1,955 / -$4,195 / -$310`; zero of eight sensitivity
cells directionally positive. Only 2/7 exact Gate-1 conditions passed (negative controls rejected and
positive exit target skill). Byte-independent reproduction matched both `replay.json`
(`b115b00a...4cf9`) and `trajectory_outcomes.parquet` (`9fa99aa6...691b`). Phase-1 tests pass 33/33.

Real corpus execution exposed two narrow implementation contradictions. Eighteen exact raw symbols used
schema-local Databento IDs shifted by `2^25` between CBBO-1m and CBBO-1s; unique-symbol authentication
restored the same contract without substitution. Forty-seven held paths lacked a fresh 15:55 BBO; the
loader had incorrectly aborted instead of applying the preregistered conservative zero terminal
write-down. Both were repaired without dropping receipts, inventing prices, changing economic labels, or
opening the firewall (`dd4b21bc`). Sensitivity labels were also repriced from the already-built causal
feature matrix with all eight pre-optimization hashes preserved (`0b2da32b`).

**Decision:** stop. Do not proceed to runtime parity, live shadow, paper promotion, or holdout reuse.
This is honest no-edge/insufficient-coverage evidence, not a candidate for rescue by tuning.

---

## 2026-08-03 — PHASE-1 CLOSE-OUT: `NO_INCREMENTAL_EDGE` (Claude Opus 5)

Terminal record: `v4/docs/protocol101/training/research/PATHD_PHASE1_CLOSEOUT_2026_08_03.md`.

GATE 1's `UNDERPOWERED` verdict raised the obvious follow-up — was the stop a coverage artefact or a real
no-edge? A failure-mode diagnostic and a fix-research pass answered it: **real, and structural.**

**The entry model is not the defect.** Across 156,950 causal OOF candidates the mean trade loses
**−$39.48** at ~33% win rate, and the loss is **flat across all five chronological folds**
(−45.91 / −40.81 / −38.02 / −35.54 / −37.03) — a constant structural cost, not a decaying edge. No
subpopulation is positive: not any hour, side, moneyness bucket, quoted-spread bucket, or premium quintile.

**The terminal finding.** With **all** friction removed (buy at bid, sell at bid, zero fees) the average
trade still loses **−$13.00**, median −$90, win 35.7%, **negative in 5/5 folds**. Buying SPXW 0DTE premium
at minute cadence is **negative-expectancy before any cost is paid**. That is theta. No model, feature, or
execution improvement repairs it.

**Execution is recoverable; profitability is not.** Posting passively at the bid (60 s window) instead of
crossing at ask+tick improves **−$41.32 → −$18.56/trade (+$22.76, 55%)** at 83.3% fill with adverse
selection of only −$0.72, and is **stable 5/5 folds** ($20.06–$24.50) — the only effect in this programme
that does not decay, because it is *mechanical* rather than predictive. Still **0/5 folds profitable**:
not trading ($0) dominates. Upper bound — the fill model has no queue priority. The frozen `FILL_LAW` was
**not** modified; this is a costing correction (`d05f0137`).

**Two corrections to Claude's own earlier record.** (a) The learned exit model is **not a policy** —
`learned_exit_index == 0` in 980/1031 (95.1%) and `learned_exit_value` is identical to `exit_immediate` in
982/1031 (95.2%); its whole benefit is one bit, *don't hold 0DTE premium*. An earlier note calling it
"real skill, preserve this asset" was overstated. (b) A first version of the passive-entry analysis was
**discarded as invalid** — it required the fill index to precede `learned_exit_index`, which is 0 for 95%
of rows, making passive fills structurally impossible; the 2–5% fill rates it produced were an artefact of
the test. Also, Claude's earlier "condition 5 power floor already satisfied" note was wrong (it read the
full evaluation index instead of the `LEARNED_OOF` subset the gate tests) and was retracted.

**Feature contract has zero ranking power.** Decile curve by model score is flat and non-monotonic
(top −$16.37, bottom −$17.75, middle best at −$11.48); top decile is 0/4 folds positive even after being
gifted the entire +$22.76 execution improvement. Same-feature retrains are wasted compute.

**Decision:** **Phase-1 CLOSED.** No further training on this strategy class. The next honest step is a
**feasibility study, not a training run** — does any instrument/horizon reachable through IBKR have
non-negative *gross* expectancy? That is measurable from quotes and requires no fitting; if nothing clears
zero gross, no modelling helps. Governance at close: firewall CLOSED (`holdout_open_count = 0`), no
broker/paper/promotion/paid-download activity, frozen clock and fill/label law unmodified (verified by
`git diff`), paper default unchanged.

**Meta-lesson (reaffirmed).** The governed apparatus reached six frozen generations and ~30k lines without
once running a real fit; the lean causal path answered the question in a day; the first "confirmed edge"
it produced was a clock leak that a rubber-stamp verification missed and cost the protected holdout to
disprove. **Run the cheap real experiment before the elaborate fake-tested pipeline, and verify the
feature-availability clock, not just the future-outcome guard.** `NO_INCREMENTAL_EDGE` was the predicted
honest outcome at the start of Phase-1, and it is the outcome — a successful research programme, not a
failed one.
