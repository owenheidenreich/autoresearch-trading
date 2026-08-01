# Cadence/Parity Decision — Adversarial Review Brief

**Purpose.** The owner wants an independent, adversarial second opinion on a
pivotal architecture decision for the Protocol101 SPXW 0DTE long-options trader:
at what time resolution ("cadence") should the trader ENTER, and EXIT, and how
should the stop-loss work — and whether the evidence we used to decide is sound.
**Do not trust this brief. Verify every load-bearing claim against the cited
files and re-derive the numbers.** Poke holes. If the recommendation is wrong,
say so and say why.

Roles for this task: **Codex adversarial reviewer.** No authority edit, no
training, no download, no broker/recorder action — this is a review only.

---

## 1. Plain-English situation

- **Entry model** = scanner that picks which 0DTE option contract to buy and when.
- **Exit model** = manager of the single open position (when to sell).
- **Cadence** = how often the trader is allowed to act (completed-minute = acts on
  1-min bar closes; 1-second = acts tick-by-tick).
- **The floor** = an upward-only trailing stop-loss (a price line under the
  position; only ratchets up).
- **Two data feeds:** Databento = historical vendor we train/backtest on; IBKR =
  the live broker feed we actually trade on. **Parity** = do they agree enough
  that a strategy trained on Databento survives live on IBKR?

**The proposed decision (Path C):** minute-cadence entry + minute-cadence
*learned* exit + a **1-second trailing stop** checked on the live IBKR feed.
Paths considered:
- **Path A** — full *learned* 1-second exit, trained on IBKR's own recordings and
  run live on IBKR (same-vendor). Treated as the long-term destination.
- **Path B** — learned 1-second exit trained on Databento, run live on IBKR
  (cross-vendor). **Currently rejected.**
- **Path C** — the interim above. **Currently recommended.**

**Why minute cadence at all:** the deep training history (needed for the entry
scanner across regimes/events) exists mostly at **1-minute** resolution; 1-second
option data only exists recently (see §3). Minute cadence is therefore a
data-availability compromise, not a claim that minute is optimal.

---

## 2. The owner's two challenges (treat as first-class review questions)

1. **"Couldn't the parity failure be a problem with our recorder?"** IBKR option
   prices visibly change every second. The probe found sizes/spread/exact-timing
   disagree cross-vendor — but did it prove the *market* disagrees, or just that
   *our recorder's reconstruction of IBKR* disagrees with Databento? Claude's
   current position: the probe did **not** isolate (a) recorder-capture fidelity
   vs (b) IBKR's inherently coarse snapshot feed (Claude asserts IBKR streams
   ~250 ms top-of-book snapshots, sizes approximate — **verify this against the
   recorder event timings, do not take it on faith**) vs (c) a true market
   difference. Claude claims this ambiguity does not change the decision because
   Path A is same-vendor (parity irrelevant) and Path B is not being relied on.
   **Challenge this.**
2. **"The architecture seems over-engineered / weirdly built."** Why train
   entry AND exit on a minute cadence but bolt on a 1-second stop? Is Path C a
   principled design or an awkward stopgap? Is minute-cadence decision-making
   defensible for a 0DTE option that moves every second?

---

## 3. Evidence and exact filepaths (verify these)

**Parity probe packet** — `v4/audit/autoresearch/protocol101_ws2_parity_probe/`
- `parity_probe_report.md` — narrative + tables.
- `run_parity_probe.py` — the analysis code (audit the method: identifier
  alignment, 1-second consolidation rule, the trigger/first-cross metric).
- `methodology.json`, `validation.json` (claims 18/18 pass), `receipt.json`
  (self-hash `44eaa3aae629aeeea6a1bbc1f6d7cfef56d169f46a251b7b9c1b206f306ffe0d`;
  Claude reproduced it as sha256 of the sorted-compact JSON with `receipt_hash`
  removed — re-verify).
- `analysis_results.json`, `cross_vendor_field_parity_by_day.csv`,
  `cross_vendor_trigger_flip_by_day.csv`, `cross_vendor_lag_sensitivity.csv`,
  `same_vendor_variation_by_field_day.csv`, `identifier_alignment.csv`,
  `minute_certificate_context.csv`.

**Headline numbers Claude verified (re-derive independently):**
- 6 complete recorder days: 2026-06-30, 07-01, 07-02, 07-10, 07-13, 07-14.
- 5,570,116 paired symbol-seconds; ~5,569,565 clean IBKR; IBKR clean coverage
  90.7–98.3%/day.
- Price (bid/ask/mid) cross-vendor correlation ≥ 0.99996.
- Sizes correlation ~0.88–0.98; exact size agreement ~23–38%.
- Spread agreement ~40%.
- Floor-STATE disagreement 0.1479%; first-cross TIMING disagreement 17.59%
  (**269 disagreements over ~1,500 triggered segments — small sample**).
- Data sizing: 6/45 sessions; only **5 open-state trajectories on 3 sessions**;
  frozen minimum 100 open-state decisions / 45-session hard bound / 65–90 planning
  range; `exact_frozen_lifecycle_held_trajectory_minimum = UNKNOWN`.

**Existing parity certification (the prior art)** —
`v4/docs/protocol101/synchronization/contracts/PROTOCOL101_PARITY_CERTIFICATION_V2_MICROSTRUCTURE_MASKED_2026_07_07.md`
- Minute-cadence (360 decision-rows/session); cross-vendor parity achieved only
  after MASKING bid/ask/mid/spread/sizes/iv/greeks before model scoring.
  (This is why a 1-second exit that decides on those fields is suspect
  cross-vendor.)

**Recorder design + raw captures** —
`v4/docs/protocol101/synchronization/design/PROTOCOL101_RECORDER_FIRST_PARITY.md`;
captures at `~/.autoresearch-trading/live_runtime/ibkr_capture/<day>/*/market_events.jsonl`
(event types incl. `option_update`, `option_delta`, `index_update`,
`ladder_definition`; "one full ladder checkpoint per completed minute"). **To
test the recorder-confound: measure IBKR option-update inter-arrival times and
per-second update counts here, and compare against Databento event density.**

**Databento probe data** — `v4/raw/opra_1s_parity_probe/` (cbbo-1s parquet +
`acquisition_manifest.json`, tier_tag `probe`). Coverage facts (verified via
Databento metadata): cbbo-1s from 2025-02-20; cmbp-1 (tick) from 2023-03-28;
cbbo-1m (minute) from 2013.

**Authority (post-A7, self-hash `1d215845`)** —
`v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- Decision timing / cadence: §2.4 (~L194–229; completed-minute, A1 next-minute
  fills, and the note that the normalized corpus has one CBBO snapshot per
  contract-minute).
- Protective floor: D24 (~L898) and ~L502 (completed-minute floor — Path C
  proposes moving the *check* to 1-second).
- A7 staged sub-minute tiers: D59/A7 block (~L1065–1112).

**A7 amendment packet** —
`v4/audit/autoresearch/protocol101_d59_staged_subminute_representation_amendment/`
(`owner_approval_receipt.json`, `implementation_report.md`, `codex_review.md`);
proposal at `.../execution/PROTOCOL101_D59_STAGED_SUBMINUTE_REPRESENTATION_AMENDMENT_PROPOSAL_2026_07_31.md`.

**Exit objective (not yet designed — relevant to Path A)** —
`v4/docs/protocol101/training/research/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md`
(training explicitly blocked pending slot opportunity cost, switching cost,
fill/latency/quote-age uncertainty, distributional targets).

**Living ledger + V2 plan** —
`.../execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md`
(L1–L15); `.../execution/PROTOCOL101_WALKING_SKELETON_V2_RERUN_PACKET_2026_07_31.md`.

---

## 4. Known weak points Claude already concedes (attack further)

- The probe conflates recorder fidelity / IBKR coarseness / true difference
  (§2.1). It measured "our-IBKR vs Databento," not "IBKR-truth vs Databento."
- The trigger-timing result rests on a small event count (269 segments).
- The 1-second data is only 6 days (well under any training minimum); no native
  1-second labels and only 5 open-state trajectories, so all *training-feasibility*
  statements are weak.
- Greeks are absent from cbbo-1s (derived), so cross-vendor greek parity was not
  directly tested.
- Claude's "IBKR ~250 ms snapshot" claim is an assertion, not measured here.

---

## 5. Questions for the adversarial reviewer

1. Is `run_parity_probe.py` methodologically sound (identifier alignment, the
   last-BBO-per-second consolidation, the first-cross/flip metric, the
   lag-sensitivity handling)? Any bug or bias that would flip a conclusion?
2. Does the recorder/IBKR/true-market confound undermine the "sizes/spread/timing
   don't transfer" finding — and therefore the rejection of Path B? Can you
   isolate the cause from the recorder captures (update frequency, inter-arrival
   times, size behavior) vs Databento?
3. Is rejecting Path B justified from this evidence, or premature?
4. Is Path C (minute entry + minute learned exit + 1-second trailing stop) a
   sound design for a 0DTE long-options trader, or an awkward compromise? Is
   minute-cadence *decision*-making defensible when the option reprices every
   second? Is a faster *entry* also warranted, or does entry genuinely need the
   deep minute history?
5. Is Path A correctly scoped, including the ~65–90 IBKR-session requirement and
   the same-vendor parity claim? Is that data requirement right, too high, or too
   low?
6. Is there a better/simpler architecture we're missing — e.g., building the
   exit on cmbp-1 (tick, from 2023-03-28) with proper event-path labeling; a
   different cross-vendor reconciliation; or an entirely different cadence design?
7. Is amendment A7 (staged D59 Tier-S/Tier-T) sound, or does it create a hidden
   trap? (It is already owner-signed; adversarial critique still welcome.)

## 6. Constraints
Governance is sacred: no authority edit, no training, no paid download, no
broker/recorder action as part of this review. Protected holdout / outer-test /
embargo data are untouchable. The trader is guarded paper infrastructure only;
nothing here is real-money authorization.

**A strong review returns:** a verdict on the probe's validity, a verdict on the
recorder confound, a verdict on C-vs-A-vs-B, any architecture we should consider
instead, and the single most important thing we got wrong.
