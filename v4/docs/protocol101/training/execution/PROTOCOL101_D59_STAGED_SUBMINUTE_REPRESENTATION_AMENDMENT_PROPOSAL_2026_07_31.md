# D59 Staged Sub-Minute Representation Amendment (A7) — PROPOSAL

**STATUS: PROPOSAL (Claude-written for Codex review + implementation).** NOT YET
APPLIED. A governed amendment to the consolidated authority (D59). Requires
Codex critical review, then implementation via the standard reseal (new
authority self-hash, re-pin, consistency checker green + new checker rule, graph
unchanged `9955085a`), then owner sign-off. Roles: **Claude writes, Codex
reviews.** Prepared 2026-07-31 from the walking-skeleton parity/cost findings.

Binding authority at proposal time: `PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
(self-hash `82d9573e…`, post-convexity A6). Graph V2 `9955085a` (unchanged).

## Why (owner-directed staging of the data spend)

D59 currently mandates that **any** later sub-minute acquisition use raw
`cmbp-1` (tick), retain the raw ticks, and derive 1-second BBO via a certified
downsampler — "only derived 1-second data may be trusted. Raw ticks are not
model inputs." That is the correct *end state*, but it forces the most expensive
data (`cmbp-1`) before we have earned it.

The owner adopts a **risk-laddered** acquisition instead: prove each rung
cheaply before funding the next.

1. `cbbo-1s`, 6 recorder-paired days (~$1.34) → 1-second cross-vendor parity check.
2. Pass → Walking Skeleton V2 (miniature-neural).
3. Pass → `cbbo-1s` backfill to its earliest coverage (**2025-02-20 → present**,
   ~360 sessions) → larger skeleton = first prototype (more data, plausible edge).
4. Edge shown → *then* fund `cmbp-1` for the real/bigger model.

This amendment enables rungs 1–3 without abandoning D59's safety intent.

**The safety reason D59 preferred `cmbp-1`, stated plainly:** `cbbo-1s` carries
**one snapshot per second**. A stop-loss / protective-floor decision depends on
whether the executable **bid crossed** a level, and a cross-then-recover can
happen entirely **within one second** — invisible to `cbbo-1s`, visible to
`cmbp-1` (every quote). So `cbbo-1s` is adequate for the 1-second *decision*
cadence and for iteration, but **not** for a *trusted* floor/stop **label**.
The fix is to tier the representation by use, not to repeal D59.

## The amendment (A7): tiered sub-minute representation

D59's canonical-representation rule is amended to a two-tier policy.

**Tier S — skeleton / prototype / probe (`cbbo-1s` ACCEPTED).**
`cbbo-1s` (vendor-consolidated 1-second BBO; OPRA.PILLAR coverage 2025-02-20 →
present) is an accepted sub-minute representation for:
- cross-vendor (IBKR-recorder vs Databento) 1-second parity testing;
- Walking-Skeleton and prototype iteration; and
- the 1-second exit **decision** cadence.

Any corpus built on `cbbo-1s` MUST be tagged `skeleton|prototype|probe`, and its
floor/stop labels are explicitly recorded as **1-second-approximate** (they may
miss intra-second cross-then-recover events). Tier-S evidence may inform
iteration and feasibility but MAY NOT be cited as a promotion-grade loss-control
claim.

**Tier T — trusted / promotion (`cmbp-1`-derived REQUIRED).**
The final trusted exit-label corpus — specifically the **floor-crossing /
stop-loss labels** — MUST be `cmbp-1`-derived 1-second data with raw ticks
retained and the in-house downsampler certified against the owned 30-session
official `cbbo-1s` pilot, before any promotion, paper-readiness, or real-money
path. This preserves D59's original guarantee exactly, for the exact place it
matters.

**D58 interaction:** unchanged in intent. A7 clarifies that D58's *trusted*
intra-minute floor-protection calibration is a Tier-T (`cmbp-1`) obligation;
Tier-S (`cbbo-1s`) floor/stop calibration is provisional/report-only.

**Combined-evaluation window:** the entry model retains the deep minute
substrate (`cbbo-1m` to 2013). A *combined* entry+exit replay is valid only
where both minute-entry and 1-second-exit data exist — i.e., ≥ 2025-02-20 for a
Tier-S run. The intersection must be selected up front (the WS2-1 lesson), not
discovered after training.

**Acquisition sanction (scoped):** A7 authorizes `cbbo-1s` parity/skeleton
acquisitions under a per-acquisition free cost estimate + hard cost cap and
quarantined output. It does **not** authorize any `cmbp-1` bulk purchase (rung 4
remains a separate owner decision), and **D57** (2022–2024 minute backfill)
stays deferred. No specific download is executed by this amendment; each
acquisition still needs an explicit owner green-light.

## Scope — what does NOT change

D59 Tier-T guarantee (cmbp-1-derived, downsampler certified) for the trusted
corpus; the entry minute substrate; D48 cap; D49 budget/soft-close; A1
next-minute entry fills; the exit/lifecycle design; the parity gate (no training
until train/live feature parity certified); the graph (`9955085a`); and all
other frozen contracts. Downside protection still lives in exit + cap + breaker.

## Anti-regression checker rule (new)

Add `sub_minute_corpus_tier_tag_required`:
- every sub-minute corpus/artifact must carry a tier tag (`skeleton|prototype|probe`
  for `cbbo-1s`; `trusted` only for `cmbp-1`-derived + downsampler-certified);
- no promotion / paper-readiness / real-money path may consume a Tier-S
  (`cbbo-1s`) corpus for floor/stop labels;
- any corpus feeding a promotion/paper-readiness gate must be `cmbp-1`-derived
  and downsampler-certified.
This permanently blocks a future silent use of `cbbo-1s` labels on a trusted path.

## Implementation + verification path (for Codex)

1. Critically review this design: is the intra-second floor-crossing rationale
   correct? is the Tier-S/Tier-T split the right cut? any second-order effect
   (e.g., a place that already assumes cmbp-1-only)? is the checker rule
   enforceable as specified?
2. If sound: governed reseal — amend the D59 block in the authority to add A7
   (tiered policy), record A7 in the post-signature amendment log and the
   deliverable-hashes/version section, compute the new authority self-hash
   (`82d9573e…` → new), keep the graph unchanged (`9955085a`), keep the
   consistency checker green and ADD `sub_minute_corpus_tier_tag_required`,
   and re-pin the new hash across specs/receipts that cite the authority.
3. Claude verifies independently (reproduce the new hash, confirm graph
   unchanged, confirm the checker rule is real and passing, confirm D59 Tier-T
   guarantee is preserved verbatim).
4. Owner sign-off (A7).

No model training, fitting, download, broker contact, recorder activation, or
runtime/default change is authorized by this amendment.
