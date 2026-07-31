# FT2 Next-Steps Focus — Working Handoff (2026-07-30)

**STATUS: WORKING HANDOFF. NOT AUTHORITY, NOT A CONTRACT, NOT SIGNED, NOT A
GRAPH ARTIFACT.** Delete or ignore once consumed. The binding documents are
the consolidated authority and the Graph V2 JSON; this file only orients the
next conversation.

## Where the program actually is (verified)

- The scoped 5-fix round (the STOP-REDESIGN-REQUIRED residuals from rerun002)
  is **DONE and independently Fable-verified 2026-07-30**. Packet:
  `v4/audit/autoresearch/protocol101_ft2_scoped_final_round_5_fixes_attempt001/`.
- Verification reproduced every hash and reran the test battery green:
  Graph V2 now `35859a40…` (new FT2-92 `insufficient_evidence` →
  `STOP-OWNER-DECISION` edge; 47 nodes / 104 edges), simulator v5 `7296a437…`
  and canonical intent/fill law `5c117d71…` unchanged, census rebuilt to **v4**
  (460,937 governed rows; 37,759 removed / 7,205 restored; both denominators
  exact), and the RLAC circularity fix is sound at the spec + synthetic-T1
  level (full-population targets, freeze-before-composer, provenance checker
  bans composer/model ancestors, planted-circularity negative control caught).
- **Consolidated authority SHA-256 CHANGED this round: `2363d3f9…` (pre-round)
  → `3c7a0aaf…` (now).** The earlier "unchanged at 2363d3f9" note was stale
  (it read the pre-round memory line). The A1/A2/A3 owner amendments edited the
  text. ⚠️ Landmine #1 is now RESOLVED as a **confirmed drift** — see below.
- **Scoped round is COMMITTED: `b77dbdb7` on `v4/phase-0`** (2026-07-30),
  scoped precisely to the round's 145-file diff manifest (92 tracked; gitignored
  parquet/csv data dumps excluded, hashes preserved via the committed manifest).
  ~1809 UNRELATED pre-existing working-tree changes were deliberately left
  untouched — a separate matter, not this round. This handoff doc itself is
  still uncommitted.

## Immediate next steps, in order

### Step 1 — COMMIT ✅ DONE (`b77dbdb7`, 2026-07-30)
The scoped round is committed on `v4/phase-0`, scoped to its diff manifest.
Remaining: this handoff doc and ~1809 unrelated pre-existing changes are still
uncommitted — decide separately whether/what to commit there; the owner said
"commit the round and pause," so nothing further was swept in.

### Step 2 — Owner authorizes the delta-scoped 3-seat review (the gate)
This is an owner decision, already made in principle (2026-07-30): the final
gate is a **delta-scoped** fresh-seat review, NOT a full re-review. Dispatch
one Goal that runs three fresh, isolated seats (trading-realism /
ML-statistics / live-parity) whose mandate is strictly:
- **A. Verify the 5 fixes** actually landed (inspect the repairs, not the
  crosswalk claims).
- **B. Fresh-scan ONLY the changed regions** — the Goal must emit/consume a
  diff manifest of what changed since rerun002; the already-triple-reviewed
  ~95% is OUT OF SCOPE. A new BLOCKING counts only inside changed regions or
  as a direct consequence.
- Isolation upgrade: each seat writes to a private path the others don't know;
  controller collects after all complete (fixes the rerun001/002 shared-file
  imperfection).
- Routing: clean / documentable-only → **FT2-21 owner design approval**
  (ends Phase A). Any in-scope BLOCKING → **roles flip: Fable writes the
  fix, Codex reviews** (owner-decided fallback; do not spend another Codex
  repair budget on the same class of miss).

(The full review-Goal shape was drafted earlier this session under the
rerun002 discussion; reuse it, just retarget it at the scoped-round packet
and the `35859a40` graph.)

### Step 3 — FT2-21 owner design approval
If the delta review passes, the packet goes to the owner. Approval ends
Phase A (design) and unlocks **Phase B: machinery construction** — the first
Goal that builds real tensors from real data. That is the "start cutting
metal" milestone.

## Landmines for the next conversation

1. **✅ FIXED & re-sealed 2026-07-30 (commit `a7602fdc`).** Was a confirmed
   in-scope BLOCKING-class drift: the authority misstated the hash of the graph
   it governs. Repair (roles-flip, Fable-written, **Codex review still owed**):
   corrected both graph-hash references + added amendment A4; re-pinned the
   authority hash `3c7a0aaf`→`d115b953` across 21 active files; recomputed
   FT2-04/05 receipts; re-ran finalize + checker + diff manifest + aggregate so
   the chain reproduces at `d115b953`; added checker Guard A (authority recorded
   graph hash == live) and Guard B (every spec `product_contract_hash` ==
   authority) — **checker now 31/31**. New aggregate self-hash
   `06b1aec845da9a2062bd72a4c381f2fa0554f136247f6fb8607248e765015050`.
   Original diagnosis retained below for the reviewer:
   - The authority doc's own hash changed (`2363d3f9` → `3c7a0aaf`) via the
     A1/A2/A3 amendments, and the doc↔its-own-hash is self-consistent
     (on-disk = `3c7a0aaf` = the round's recorded `authority_sha256`). So the
     landmine's original premise ("hash unchanged") was wrong.
   - **The real defect:** the authority text records "Graph V2 JSON: SHA-256"
     at **line 15 and line 1101** as `b06a26be…` — the **pre-amendment** graph
     hash. Fix 3 changed the graph to `35859a40…` (added the FT2-92
     `insufficient_evidence` → `STOP-OWNER-DECISION` edge) but did **not**
     update these two references. The governing authority now asserts the graph
     is `b06a26be` while the actual graph is `35859a40`. This is branch (b):
     a real doc↔hash drift — the exact class that caused the rerun002 stop.
   - **Checker blind spot (compounding):** the round's upgraded v3 consistency
     checker (T2, 29/29 PASS) does NOT catch it. Its `authority_hash` check
     only confirms the authority *file* hashes to `3c7a0aaf` (byte-stable); it
     never parses the authority *text* to compare the recorded graph hash
     against the live graph. Same class of gap as last round's "checker lacked
     this check." Any fix must also add that text↔live-graph check.
   - **Cascade cost — NOT a one-liner.** Editing lines 15 & 1101 changes the
     authority text → new authority hash ≠ `3c7a0aaf` → invalidates every round
     artifact that pins `3c7a0aaf` (crosswalk `authority_sha256`,
     `aggregate_receipt` `product_contract_hash`, all 5 packet receipts, the
     checker's `AUTHORITY_HASH` constant, diff-manifest baseline). It re-seals
     the round.
   - **Routing:** in-scope BLOCKING → per the owner-decided fallback, roles
     flip (Fable writes the fix, Codex reviews). Feed this to the delta review
     as a **pre-registered, already-confirmed** finding so it verifies the fix
     rather than re-discovering it — OR fix it now under owner authorization
     before the review runs. Owner decision; program is paused.
2. **Census is v4 now.** Any spec still citing v2/v3 census hashes is stale —
   the consistency checker's census-hash rule should already catch this;
   verify it ran against v4.
3. **Don't let the review re-open the frozen 95%.** Scope discipline is the
   whole point; a full re-review re-introduces the infinite-loop risk.
4. **Verification was spec + synthetic only.** Nothing has been fit on real
   data yet. The parity gate (train-vs-live feature parity) still stands and
   is a separate future hard gate — do not let Phase B skip it.

## Key context the next conversation may lack

- Owner is building the Charter trader (patient, structural, minutes-to-hours,
  NOT a scalper). t+1 next-minute fills confirmed (minute-only data; tick-data
  purchase deferred to campaign 2). No-bid = full-loss. 5% per-trade premium
  cap + budget-aware entries + $1.00 soft-close floor. Component entry freeze
  (promotable dollars deferred to the combined system). Loose screen + upside
  ranking with a mandatory 0.0 no-screen control arm.
- The RLAC (Realized-Label Audit Composer) is the circularity fix: a
  model-free hindsight judge computed purely from frozen labels that produces
  the action-head training targets over ALL eligible contract-minutes, so the
  confidence heads no longer depend on the composer that depends on them.
- Full journey so far: reviews went 28 → 17 → 6 findings, then the scoped
  round closed the last 5. Trend is convergence, not churn.
