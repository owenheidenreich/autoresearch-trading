# Lifecycle-training log

## 2026-08-15 — Phase 0: charter and registration

- Recorded the owner-approved development charter in
  [`DEVELOPMENT_CHARTER_2026_08.md`](../../governance/DEVELOPMENT_CHARTER_2026_08.md), citing the current
  conversation’s attached-plan approval as the signature event and the contemporaneous prior-packet log.
- Registered job 46 in `STATUS.md` before creating this work packet, satisfying the v5 file policy.
- Re-scoped the 2026-08-15 program STOP only for model construction and diagnostic development on
  pre-cutoff owned data. Closed measurements remain closed as confirmatory claims; no trading or model
  promotion is authorized.
- Adopted amendments A1–A13 revision 2 as stricter job-46 controls. Reason: the owner-approved plan
  explicitly adopts them, and they prevent the known-answer, exposure, capacity, and reproducibility
  defects the review identified.
- Confirmed the sole external authority: an exact Databento preflight followed by the fixed SPXW
  definitions + CBBO-1m backfill only if its total is at or below $75. No vendor call, download, spend,
  data mutation, real target, model fit, or order occurred in this phase.

## 2026-08-15 — Phase 1: pre-acquisition freeze correction

- Replaced the free-OHLCV-only downloader with a tested declaration-verified backfill runner. It derives
  the exact 794 nonempty source sessions from immutable SSD Parquet metadata, prices both schemas for
  every member before requesting data, and filters saved rows by each row's own OSI expiry.
- The first cost-only preflight was stopped before it wrote a receipt or made a data request. Reason:
  adopted amendment A8 requires the dataset, target, and QC semantics to be frozen before *vendor
  contact*, while the first-touch corpus extension was not yet hashed. The metadata queries themselves
  cannot purchase data; nevertheless the next preflight will be made only after that complete freeze.

## 2026-08-15 — Phase 1: semantic freeze sealed

- Sealed `PREACQUISITION_SEMANTIC_FREEZE_V1.json` with the A1–A13 chronology, target law, data/QC
  semantics, and SHA-256 hashes for every declared source. Revalidated all 12 source hashes immediately
  before the vendor preflight.
- Bound that freeze and the acquisition-runner hash into `BACKFILL_DECLARATION_V1.json`; the runner now
  rejects any declaration, source-inventory, runner, or frozen-semantic drift before it can make a vendor
  request. The focused guard, normalization, and first-touch tests passed (15 passed).

## 2026-08-15 — Phase 1: exact cost gate STOP

- The complete declaration-verified Databento preflight priced all 794 nonempty sessions and both declared
  schemas (1,588 requests) at **$671.898150**, above the owner-authorized hard cap of **$75.00**.
  Definitions cost $0.00; CBBO-1m cost $671.898150.
- The immutable receipt is self-hash valid and records `STOP_OVER_HARD_CAP`, with `download_performed`,
  `money_spent`, `broker_contacted`, and `reserved_sessions_used` all false. No acquisition was run.
- Per the owner-approved plan’s cap gate, the full-ownable-corpus construction, fit, and evaluation path is
  stopped. Changing the corpus or increasing the cap requires a new owner decision; no narrower substitute
  was selected.

### Request-scope diagnosis — the STOP priced 40 expirations, 2026-08-16.

- Reopened the $671.90 STOP as a suspected defect rather than a price. The 08-15 preflight requested
  `cbbo-1m` with `stype_in="parent"` (`SPXW.OPT`), which covers **every SPXW expiration listed that
  day**; symbology resolution measures **12,828 / 15,836 / 15,980** instruments on three sampled
  dates. The declaration applied the same-day rule only *after saving*, so the request paid for ~40
  expirations to keep one.
- Charter §2 authorizes only "SPXW contracts whose own OSI expiry equals each session date", so the
  priced request was **broader than the owner approved**. The STOP stands as correct on the request
  it priced; its receipt is untouched.
- Why it was invisible: parent-scope price is nearly flat by year ($0.7695 in 2022 → $0.9421 in
  2024) because it tracks the whole SPXW universe, which barely changes; the recorded 0DTE-scoped
  acquisition costs varied 5x ($0.0128–$0.0646) with actual 0DTE activity.
- Built `ops/probe_backfill_request_scope.py` (5 tests) — a diagnostic that prices both scopes over
  identical windows, cost-metadata only. Every 100th session of the frozen inventory: parent
  **$0.8483**/session vs traded-symbol **$0.0134** (**63.4x**); projected parent **$673.56** against
  the measured $671.90, i.e. it reproduces the STOP to 0.2%.
- Priced the **full quoted ladder** (symbology-resolved, free, no download): 2022-06-01 362 symbols
  $0.0215; 2024-03-15 812 $0.0483; 2025-07-31 988 $0.0476. `definition` is **$0.00** at all 794
  sessions per the original receipt — the $23.03 definitions estimate was too high.
- Issued `BACKFILL_DECLARATION_V2.json` (self-hash `4d8f01a7…`, supersedes V1 without editing it)
  and `ops/acquire_lifecycle_backfill.py` (9 tests): resolve the session ladder by symbology, keep
  only own-OSI-expiry symbols, price and request exactly those. The ceiling is enforced twice — the
  preflight gate and a running total in the acquisition loop — and acquisition refuses if a ladder
  changed size since its quote. One test fails if `cbbo-1m` is ever priced at parent scope again.
- Corrected exact preflight launched over all 794 sessions (cost queries only). No spend, download,
  vendor data request, fit or promotion occurred. **V1's semantic-freeze rule makes a request-shape
  change a fresh owner decision, so acquisition is held pending owner approval of V2.**
- Finding: `research/findings/BACKFILL_REQUEST_SCOPE_2026_08_16.md`.

### Phase 4 machinery — the two-phase lifecycle trainer, 2026-08-16. No real fit.

- Audited what Phase 4 actually needs and found the gap: the 120-parameter `compact_shared_lifecycle`
  model, its exit-target semantics (`build_exit_action_targets`, `exit_action_value_loss`) and the
  simulator all existed, but **no trainer orchestrated them** — no module referenced
  `CompactSharedLifecyclePolicy` outside its own tests, a design recorder and the capacity harness.
- Built `research/lifecycle_trainer.py` implementing the frozen protocol's two structural
  requirements rather than leaving them to a runner's discipline:
  - **Nested out-of-fold trajectories.** `ChronologySplit` takes the earliest 40% as the training
    prefix and `array_split`s the rest into five contiguous score blocks; `inner_folds` yields
    strictly-earlier-train/later-holdout pairs inside the prefix. Every `Trajectory` carries its
    generator's training sessions, so disjointness is auditable after the fact, and
    `assert_trajectories_are_out_of_fold` refuses a leak. This is ledger row 341's finding made
    mechanical: every prior exit result was measured on random entries, the one regime where a
    conditional exit's value cancels.
  - **Frozen entry before exit.** `train_exit_head` freezes the shared representation and all entry
    parameters, then verifies bitwise equality afterwards rather than trusting `requires_grad`;
    `train_entry_phase` symmetrically verifies it did not touch the exit head.
- Production-mirrored law throughout: AdamW, gradient clipping, plateau stop with **best-checkpoint
  restore**, SHA-256 seeds, WAIT trained toward its structural $0 floor (the first capacity harness
  left WAIT gradient-free). Inference laws implemented as declared: enter only when the best
  contract strictly exceeds a non-negative WAIT; SELL fires when its value is at least HOLD's, with
  forced liquidation left simulator-owned.
- The trainer consumes `SessionEpisode` records rather than reaching into a dataset, so it is
  exercised on known-answer synthetic episodes today and binds to the real builder when the corpus
  exists. 15 focused tests, including one that fails if exit training moves any entry weight and one
  that fails if a trajectory's generator trained on its own session.
- **907 v5 tests pass; project checker green.** No real fit, no economics, no spend, no vendor data.

### Owner approves declaration V2; acquisition armed behind the cap, 2026-08-16.

- Owen Heidenreich approved in conversation: **"approve declaration V2 and let the acquisition run
  if its total is at or below $75."** Recorded as
  `governance/BACKFILL_V2_APPROVAL_2026_08_16.md` — the fresh owner decision V1's semantic-freeze
  rule required after a post-preflight request-shape change.
- The $75 charter ceiling is unchanged and unraised. Scope is unchanged from what was already
  authorized (same dataset, schemas, dates, destination); only the request shape narrows from every
  listed SPXW expiration to the charter's same-day expiry.
- The corrected exact preflight is still running (~2,400 vendor cost/symbology calls over 794
  sessions). Acquisition executes only on a `PASS` receipt at or below the ceiling; otherwise it
  stops and the result is recorded as before.

### Destination moved to the SSD; declaration V3, 2026-08-16.

- Pre-download disk check found the V2 destination sits on the **internal disk at 99% capacity,
  5.5 GiB free**. The backfill is ~1 GB scaled from the owned corpus (483 MB over 251 sessions), so
  it would fit but leave the machine at ~4.5 GB — and Phase 2's derived data is far larger. The
  external SSD has **1.8 TiB free** and already holds the OHLCV source for these exact sessions and
  every derived tree.
- Owner decided the SSD. Issued `BACKFILL_DECLARATION_V3.json`, self-hash `36da20cb…`, changing
  **destination only**: a field-by-field diff shows no difference from V2 outside
  destination/provenance, and the `request` block is byte-identical. Cap unchanged at $75.
- The V2 preflight was killed 20 minutes in rather than completed, because a preflight receipt binds
  to its declaration's hash and would have been unusable under V3. Restarted under V3. No receipt
  was written for V2, so nothing was superseded after the fact.
- Approval on file: `governance/BACKFILL_V2_APPROVAL_2026_08_16.md` (covers V2 and its
  destination-only successor).

### Preflight PASS at $19.24; acquisition running, 2026-08-16.

- The corrected exact preflight under declaration V3 (`36da20cb…`) completed over all 794 sessions:
  **$19.2450 against the $75 ceiling — PASS.** Definitions **$0.0000**; `cbbo-1m` **$19.2450**
  (min $0.0117, mean $0.0242, max $0.0593 per session). Resolved same-day ladders run **222 / 384 /
  1,048** symbols at min/median/max. Receipt records no download, spend, broker contact or reserved
  session at preflight time.
- **The measured correction is 34.9x**: $671.90 at parent scope against $19.24 for the identical
  charter-authorized data. The V1 STOP was right about its request and wrong about the price of the
  approved scope; both facts are now on the record with receipts.
- Acquisition started under the owner's conditional approval, writing to the SSD root. The runner
  re-resolves each session's ladder and refuses any session whose ladder size moved since its quote,
  enforces the ceiling as a running total, and skips already-present files so an interrupted run
  resumes rather than re-buying.
- Receipt path `…/acquisition_receipt.json`; a watcher will report completion or a silent exit.

### Acquisition interrupted by a vendor streaming error and resumed, 2026-08-16.

- The first acquisition run died at **112 of 794 sessions** (244 MB) on
  `BentoError: Error streaming response: Response ended prematurely` — a transient vendor-side
  failure, not a refusal by any gate.
- **A reporting trap worth recording:** the run was launched with its output piped through `tail`,
  so the shell reported **exit code 0** for a crashed Python process. The completion notification
  therefore said "completed". Only the receipt-or-death watcher caught it, exiting
  `ACQUISITION_EXITED_WITHOUT_RECEIPT`. Never read an exit status through a pipe; the watcher is
  what made the failure visible.
- Damage assessment: none. `_download_one` writes its parquet only after the frame materialises, so
  the interrupted session left no partial file — 113 definitions and 112 cbbo files, and the three
  most recent cbbo files reparse cleanly with `all_same_day=True` and 288–480 contracts each.
- Resumed by **process-level retry** rather than by adding retry code: the runner already skips
  existing outputs, so re-invoking it continues where it stopped, and this keeps the
  `implementation_sha256` that declaration V3 pins valid. Adding a retry loop inside the module
  would have invalidated the declaration and forced another hour-long preflight.
- The retry wrapper stops on a receipt and surfaces a non-blip refusal (ladder-size mismatch or cap
  breach) rather than looping on it. Progress and terminal states are monitored.

### Risk-law amendment drafted for signature; a serial-account defect found and fixed, 2026-08-16.

- Owner requested an amendment to the position-sizing risk law before Phase 2 builds action masks.
  Drafted `governance/CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md` — **UNSIGNED**, superseding
  (not editing) the 2026-08-13 amendment's 13%-of-equity ticket and 5% breaker via that document's own
  procedure. Nothing may be fitted or masked under it until signed.
- **A simulator defect was found while verifying the owner's serial-account question, and it was the
  most consequential thing in this cycle.** `simulate_session` opened every session with
  `cash = STARTING_EQUITY_USD` and measured all three daily-breaker sites against that same constant:
  the account re-seeded $10,000 daily and the breaker never tracked real equity. No cross-session walk
  existed. Repaired: session-starting equity is now a parameter, the breaker measures against it, and
  `simulate_serial_account` carries ending cash forward, refuses a replay that re-seeds, stops at the
  survival floor, and carries realised P&L through a blocked terminal. Eight regression tests.
  **915 v5 tests pass.**
- Measured before drafting, from the owned ladder (933,198 two-sided OTM-band candidates) and 32,976
  quote-priced trades: eligibility by cap **52.52 / 80.55 / 93.08 / 97.90%** at $500/$1k/$1.5k/$2k;
  median round trip **10.77%** of premium under $200 against **1.37%** at $1k–2k; drawdown reaching
  ≥99% of premium in **0.34%** of trades and ≥90% in **4.32%**; mean realised loss on losers
  **47.5% of premium**, not 100%.
- **Three cited figures did not reproduce** (8.89/49.55/79.75/92.99% eligibility and a 43.63% band
  share). Measured values are reported instead and the discrepancy is stated in the amendment rather
  than resolved silently; the argument's direction survives on either population.
- Survival, on the existing 20,000-simulated-year machinery with **random-entry (edgeless)** outcomes:
  ruin **0.0 / 0.0 / 1.9 / 84.6%** at $500/$1k/$1.5k/$2k. The draft carries that warning prominently —
  the $2,000 cap is a bet that the model has an edge — and makes the review condition fire on an
  edgeless first evaluation. Breaker firing at $2k: **10.06%** at 5% (fails the declared tolerance)
  against **0.41%** at 20%, which is the arithmetic resolving STATUS row 21.
- A declared stop is measured to slip: with −50% declared, the worst realised loss at a $2,000 cap was
  **−$1,473**, so the amendment requires the stop be simulator-enforced and the realised worst loss
  reported and checked after the fact.
- The backfill download continued uninterrupted throughout (147/794 at the time of writing); no rail,
  ceiling or declaration was touched.

### Risk amendment SIGNED; alpha charge recorded as an obligation, 2026-08-16.

- Owner signed in conversation: *"please sign and authorize. we need to allow the model to have this
  freedom to choose more contracts so it can have more room for its edge discovery. yes its a bet it
  has an edge. but we will fine tune an entry and exit model so it doesnt take massive whopping 50%
  losses."* The §6 ruin warning was accepted explicitly, not by omission, and the signature block
  records that.
- **In force now:** $2,000 absolute per-trade premium ceiling (dollars, not a share of equity, so it
  cannot drift into deep-ITM tickets as the account grows); `moneyness_band` deep-ITM bar retained;
  20% daily breaker on session-starting equity; max loss charged as a declared stop rather than
  premium paid. STATUS row 21's long-standing ticket/breaker conflict is closed by measurement.
- **The owner's stated mitigation is binding through §4, not taken on trust.** "We will fine tune an
  entry and exit model so it doesn't take massive losses" is exactly the claim §4 requires be proven:
  the declared stop must fire in the simulator, and the realised worst loss must be reported and
  checked against the declared maximum. A policy that only looked safe because it got lucky fails the
  risk check regardless of P&L. The stop level is set per fit declaration; the −50% in §4/§6 is the
  measurement's illustration, not the policy stop.
- §9's three review conditions were not waived by the signature: the amendment expires at $25,000
  equity, on any realised loss exceeding the declared maximum, or on a first full evaluation showing
  no edge — the last because §6 makes the $2,000 cap conditional on an edge existing.
- **Alpha charge recorded as an obligation rather than posted.** Procedure item 4 requires this
  widening (52.52% → 97.90% of the tradeable band) to spend alpha from the job-46 experiment ledger.
  `AlphaLedger` is constructed against `option_sessions`, and the final corpus size is still being
  acquired, so opening it now would fix the bar at a denominator known to be wrong. **The amendment
  counts as declared experiment #1 and must be charged when the ledger opens at Phase 5, before any
  fit.** Recorded here so it cannot be quietly skipped.
- Phase 2 may now build action masks against the signed law. The download continued uninterrupted
  throughout this cycle.

### Stop-floor and underpower addendum SIGNED after independent verification, 2026-08-16.

- Owner instructed signature of `governance/ADDENDUM_STOP_LEVEL_AND_UNDERPOWER_2026_08_16.md`, which
  binds a **-40%-or-wider floor** on any declared stop in job-46 fit declarations and reads the
  ticket/breaker amendment's §9.3 as requiring a **powered** negative before the cap reverts.
- **The addendum was drafted by a different session, so nothing in it was signed on trust.** Every
  quantitative claim was recomputed from `quoted_exit_paths.parquet` first: median path drawdown
  -44.2%, quartiles -65.5%/-21.6%, catastrophe rates 4.32/2.27/0.34%, and **all fifteen cells** of the
  stop-versus-winner table reproduce **exactly**.
- Its own stated limitation was tested too. Under the stricter "stop fires *before* the gain" reading
  the harmed shares at +30% are **29.3/16.7/12.0/8.4/3.8%** for stops of -20 through -50%, against the
  **58.7/42.4/35.6/29.2/19.1%** tabled. The table is a genuine upper bound, ~2-3.5x conservative as it
  says, and the -40% floor holds either way (-20% harms 2.0x as many winners by the table, 3.5x by the
  stricter measure).
- **A tension worth stating plainly:** the owner's signing rationale for the $2,000 cap was to avoid
  "massive whopping 50% losses", and this addendum forbids stops *tighter* than -40%. These are
  compatible because the floor binds the **declared risk-law stop**, not the learned exit. The exit
  head may sell whenever predicted SELL >= HOLD, including at -15%; what it may not do is rely on a
  mandatory hard stop tighter than -40%, which the project has already measured to cut winners and
  raise the break-even bar by 15 points (ledger row 331). Loss control is expected to come from the
  learned exit, with the declared stop as the backstop that §4 requires be enforced and checked.
- §9.1 (equity) and §9.2 (breached maximum loss) are untouched and unconditional; §3.4 caps the
  underpower branch at two consecutive evaluations before a fresh owner decision, so it is not an
  open-ended licence to continue.
- Download unaffected: **408/794** and running.

### Retry wrapper defect — my own, found and fixed, 2026-08-16.

- The acquisition stopped at **444/794** (899 MB) on a vendor **504 gateway timeout** — transient and
  retryable, like the earlier premature-stream-end.
- **The retry wrapper did not retry.** `status` is a **read-only special variable in zsh**, so
  `status=$?` aborted the script during attempt 1 and the loop never ran a second attempt. The first
  attempt had already carried the corpus 112 -> 444 before the 504 landed, which is why progress
  looked healthy right up until the wrapper died.
- Two lessons recorded rather than rediscovered: (1) a retry wrapper that has never been observed to
  retry is not a retry wrapper — this one was written, launched, and trusted without a single
  observed second attempt; (2) combined with the earlier `| tail` masking of a crashed process's exit
  code, both failures in this acquisition have been in the *reporting* layer, not the downloader.
- Replacement wrapper: `rc=$?` instead of the reserved name, syntax-checked with `zsh -n` before
  launch, per-attempt logs, progress counted from disk each attempt, a **refusal guard** that stops
  rather than retries on a ceiling breach, ladder drift, or declaration mismatch, and a
  **three-stall** guard so a permanently failing state cannot spin for 60 attempts.
- No data was lost or re-bought: `_download_one` skips existing outputs, so each restart resumes.
  Spend is still governed by the passing $19.2450 preflight and the running-total ceiling check.

### Runner hardened after a degraded-vendor stall; declaration V4, 2026-08-16.

- After the 504 at session 444, the corrected retry wrapper made **zero** progress: the 504 had moved
  to `symbology.resolve`, and `acquire` resolved **every** declared session's ladder before checking
  whether its outputs already existed. Each restart therefore needed ~794 successful vendor calls to
  reach ~350 sessions of real work, and any single timeout in that pass discarded the attempt.
  **Retrying could not win**, so the loop was stopped rather than left to burn calls.
- Vendor health probed directly before deciding: 3 of 3 resolves succeeded but latency ran **3.1 s,
  3.0 s and 29.3 s**. The service is degraded, not down — which is exactly the regime the old design
  handled worst.
- Three fixes, all in the pinned module rather than routed around it:
  1. **Skip complete sessions before any vendor call.** A session whose definition and cbbo outputs
     both exist is recorded from disk and never resolved or re-requested. Restart cost is now
     proportional to work remaining, not to the whole corpus.
  2. **Bounded retry with backoff** (5 attempts; 5/15/45/120 s) around symbology, cost and download
     calls — and **a refusal is never retried**: ceiling breaches, ladder drift and declaration
     mismatches raise on the first attempt, because they are decisions rather than blips.
  3. **Checkpointed preflight.** Each priced session appends to a `.partial.jsonl` and a resumed pass
     skips what it already priced, so a late failure can no longer discard ~2,400 free cost calls.
- Three new tests pin the behaviour: a complete session triggers **no** symbology and **no**
  timeseries call; a transient failure retries then surfaces; a refusal surfaces on attempt 1.
  **918 v5 tests pass**, checker green.
- **Declaration V4** (`30ad430b…`) pins the hardened runner. A field diff shows no difference from V3
  outside the runner hash and provenance: dataset, schemas, request shape, date range, inventory,
  destination and the **$75 ceiling** are byte-identical. The 444 sessions already acquired under V3
  are retained, and the running-total ceiling check still counts every declared session, so the job
  as a whole remains governed.
- Changing the module invalidated the V3 preflight by design — the implementation hash is what
  guarantees the code that spends is the code that was priced — so the preflight is being re-run
  under V4 rather than the control being bypassed. Costing is free; only time was spent.

### Owner instruction — the exit is scored on two skills, not one average, 2026-08-16.

- Owner, for the Phase 4/5 fit declaration: *"the exit's success metric shouldn't be 'did it beat
  holding.' It should be the two things you just named separately — how much of the loss it avoided
  on bad entries, and how much of the excursion it captured on good ones. Those are different skills,
  they can fail independently, and averaging them into one number is precisely how the earlier work
  hid the fact that its exits had neither."*
- **This is confirmed by the project's own receipts, not merely accepted.** Round 3 measured hold-60
  at **+$675** on the 36.7% of entries that became big movers and **−$598** on the 26.6% that never
  moved. Those cancel almost exactly under a random entry — and a random entry is the single regime
  every prior exit study used, which is why row 341 could conclude the exit "is a stopwatch" while
  row 41 showed it was conditionally powerful. One averaged number is what made those two true at
  once.
- Recorded as binding on the declaration in `PLAN.md` phase 5: loss averted on entries that did not
  develop, and excursion captured on entries that did, each reported separately and each against a
  **duration-matched** control (row 341: comparing a fitted rule to a long clock credits holding time
  rather than deciding). "Beat holding" is explicitly refused as a metric.
- Two guards recorded with it: a degenerate exit must read as degenerate — always-cut posts high loss
  averted with near-zero capture, always-hold the reverse, and **neither counts as skill** — and both
  populations are defined by realised outcome, so the decomposition is **attribution, never a
  selection rule** (round 3's own caveat). Serial executable P&L stays the decision criterion.
- This complements rather than replaces drafted amendment **A5** (duration-matched exit control),
  which the development charter already adopted; A5 fixes the comparator, this fixes the statistic.

### Scale-sensitivity study — cap holds, breaker sound, ruin is the throttle, 2026-08-16.

- Owner-requested read-only study before Phase 2 hardcodes action masks. Declaration
  `SCALE_SENSITIVITY_DECLARATION_V1.json` (`b6b36b28...`) froze a 28-cell family (4 accounts x 7 win
  rates, 20,000 paths x 252 sessions) **before any outcome was computed**. Finding:
  `research/findings/SCALE_SENSITIVITY_2026_08_16.md`; receipt
  `v4/audit/autoresearch/scale_sensitivity_2026_08_16/receipt.json`.
- **Q1 cap: holds in simulation, but the mask code does not implement it.** Max premium bought was
  **$1,990 at every account level including $100,000**. However `audit_causal_day_coverage.py:33`
  derives the action-mask ceiling as `SESSION_START_EQUITY_USD * TICKET_CEILING_SHARE`, and
  `check_occupancy_risk.simulate` takes a share too. Both read a frozen $10,000 today, so neither
  drifts yet - but the serial-account repair made equity compound for the first time, so wiring live
  equity into either turns $1,300 into **$13,000** at a $100k account, which buys deep ITM. **Phase 2
  must restate `MAX_ENTRY_ASK_USD` as a fixed dollar constant.** Separately, the dollar cap alone does
  **not** bar ITM: 37.3% of cap-eligible contracts are in the money (99th pct +18.4 points), so the
  `moneyness_band` bar is load-bearing. Limit stated: the owned ladder is bounded to +/-25 points, so
  nothing deeper is observable.
- **Q2 breaker: NOT mis-sized; the premise does not hold, and no amendment is drafted.** The concern
  assumed max loss = premium paid, which the signed amendment replaced. Under the -40% stop a maximum
  ticket loses ~$800: median loss **$373**, 90th pct **$683**, worst observed **$1,293**, so two
  median losses are $746 against a $2,000 breaker. Simulated breaker firing is **0.11%** of sessions
  at worst and 0.00% elsewhere; affordability blocks **0.00%** of offers. The trades/session drop to
  0.447 at $10k/p=0.30 is **death, not throttling** - 95.5% of those paths are ruined.
- **Q3 required edge (declared threshold: ruin <= 5% of breaching the 50% floor in 252 sessions,
  reusing the project's existing 5% tolerance convention):** break-even ~37% at $10k but survival
  needs **45-50%**; $25k needs 35-40%; $50k 35-40%; $100k 30-35%. **The ten-point gap between
  breaking even and surviving at $10k is the number Phase 5 should be judged against.**
- **Q4 per-trade economics is not size-coupled; the apparent movement is survivorship censoring.**
  Affordability never binds and max premium is identical everywhere, and where ruin vanishes the
  values converge exactly ($193.22/$193.58/$193.34 at $25k/$50k/$100k, p=0.50). Recorded requirement:
  any Phase 5 packet reporting per-trade P&L across account sizes must report it on surviving paths
  and say so.
- **Q5 constraints stop binding between $25k and $50k** - ruin 29.6%/5.9% at p=0.35 and 3.0%/0.1% at
  p=0.40. The signed amendment's $25,000 expiry review lands almost exactly on the measured
  transition, which is luck rather than design but gives the review a measured basis.
- Study assumes a hypothetical edge and says so in the declaration, the receipt and the finding: win
  rate is a swept parameter, the population's measured random-entry rate is **31.65%**, and no row is
  evidence an edge exists. The running backfill acquisition/preflight was not touched.

### Scale-study corrections applied, 2026-08-16. No fit, no purchase, acquisition undisturbed.

- **1. Entry ceiling is now a literal dollar constant, and the code cannot contradict the law.**
  `MAX_ENTRY_ASK_USD = SESSION_START_EQUITY_USD * TICKET_CEILING_SHARE` is **deleted**, not left
  computed-but-unused; `audit_causal_day_coverage` now declares `MAX_ENTRY_TICKET_USD = 2_000.0` and
  charges premium **plus fees**, matching the signed amendment's wording exactly. Both equity
  constants are gone from the module. The same defect in the risk simulator is corrected the same
  way: `check_occupancy_risk.simulate` takes `premium_ceiling_usd` (dollars) instead of a share, and
  affordability is now `min(ceiling, session-starting equity)`. `CHARTER_PREMIUM_SHARE` is retained
  **only** so prior receipts remain reproducible, and is no longer read by the simulation path.
- **Consequence recorded in PLAN phase 3:** the ceiling moved from an equity-derived **$1,300** to
  **$2,000**, so `entry_eligible` admits a wider action space and **every mask/candidate artifact
  built before today is stale**. Rebuild; never mix pre- and post-correction candidate tables.
- **2. `moneyness_band` is declared independently load-bearing**, with tests proving a **cheap ITM**
  contract is still refused and that each guard alone is insufficient — a $200 five-point-ITM ticket
  is refused on moneyness, a $3,000 OTM ticket on price. This blocks the plausible refactor that
  collapses them believing a price cap implies OTM; measured, 37.3% of cap-eligible contracts are ITM.
- **The equity-reference test is structural, not textual.** It parses the eligibility path with `ast`
  and asserts no *name reference* to equity or share terms, and that the ceiling constant is an
  `ast.Constant` rather than any computed expression. A first textual version failed on my own
  docstring forbidding equity, which is exactly the false positive a word-scan produces.
- **3. Survivorship added to the do-not-retest ledger as a defect class**, alongside the
  vanishing-contract look-ahead, since both are populations silently selected by outcome. Carries the
  owner's presentation ruling: surviving-path per-trade economics **and** ruin probability with
  **time-to-ruin** as first-class headline numbers, never a footnote and never a composite.
- **4. Owner rulings encoded in PLAN phase 5** for the Phase 4/5 declaration: $2,000 fixed with no
  automatic growth until the $25,000 review (under-deployment at a large account is the accepted cost
  of never drifting into deep ITM); a **three-state** outcome grid where break-even-to-survival is a
  *real but insufficient edge* routing to declared iteration, not a pass and not a failure; no paper
  trading until survival clears at the owner's actual account size; and the target stated **before**
  the fit — 31.65% baseline, 45-50% needed, **+14 points, about twice the best entry effect this
  project has ever measured (+7.1pp)**.
- **929 v5 tests pass** (11 new), project checker green. The preflight ran undisturbed throughout and
  is at 640/794 sessions priced.

### Design brief issued to Fable; settlement-source assumption tested and FAILS for the backfill era, 2026-08-16.

- Wrote `DESIGN_BRIEF_FOR_FABLE.md` — the learning-content brief for a separate high-effort planning
  pass with full repository access. It cites primary sources rather than transcribing them, states the
  four facts Fable cannot infer (the ~1,045-session two-era corpus, the budget that must be
  re-measured rather than inherited, today's $1,300 -> $2,000 ceiling change that stales every prior
  mask, and the pre-stated +14-point target), and asks for five deliverables: the feature contract
  field by field, the label at corpus scale, the curriculum, a diagnostic suite that separates
  "learned geometry" from "learned timing" **before** economics are read, and the missing
  `SessionEpisode` adapter. Boundaries: no reopening of signed law, no governance writing. The brief
  states the poor measured prior explicitly and invites "the features cannot close this gap" as a
  legitimate answer.
- **The settlement-source assumption was flagged untested and is now measured: it FAILS for the new
  eras.** The owned official, non-derived SPX 16:00 source
  (`raw/index/spx_1m/*.official_spx.parquet`) exists for **251 sessions, 2025-08-01 to 2026-07-31,
  and for nothing else** — there are zero official files anywhere on either root covering
  2022-06 to 2025-07. So **about 76% of the ~1,045-session corpus has no official settlement source**,
  which is a majority-of-corpus condition, not an edge case. The Phase-2 fallback in the approved plan
  (parity spot plus the validated-cash-settlement law with a paired zero-recovery sensitivity, source
  labelled per session) therefore governs the bulk of the corpus rather than a remainder.
- A second constraint measured on the acquired files, relevant to that fallback: backfill-era sessions
  **end at 15:58, not 16:00**, and only **42-53%** of rows in the closing minutes are two-sided
  (47-58 distinct contracts still quoting). Terminal accounting for these eras cannot assume a 16:00
  two-sided chain, and the eras are **not** interchangeable with the owned year for label
  construction. Recorded for the corpus builder and named in the brief as an era difference Fable must
  address rather than average over.
- No fit, purchase, vendor contact or reserved-session use. The preflight ran undisturbed to 716/794.

### Learning-content design delivered — the chain's own internals are the one unfitted family, 2026-08-16.

- Answered the brief in [`LEARNING_CONTENT_DESIGN_2026_08_16.md`](LEARNING_CONTENT_DESIGN_2026_08_16.md):
  the feature contract field by field, the label at corpus scale across both eras, the curriculum,
  the diagnostic suite, and the `SessionEpisode` wiring spec. Primary sources were read rather than
  the brief's summaries — the V5 attribution receipt, entry-exit LOG cycles 1 and 4, ledger rows
  326–346, the issued G3 ledger, and the built trainer/tensorizer/builder modules.
- **The design's one substantive claim:** of row 338's four named unexplored sources, cross-asset is
  barred by the owner's SPXW/SPX-only ruling, term structure cannot exist in a 0DTE-only corpus, and
  the event calendar is outside charter §2 — leaving **the option chain's own internals** as the
  only untested information family this corpus already owns. Signed skew and its 15-minute change,
  displayed depth imbalance across and within strikes, and per-contract smile residual enter as
  **state**; every fitted model to date has read the chain only as geometry. `bid_size`/`ask_size`
  have been in the ladder tensor since it was built and **no fitted model has ever read them**.
- **Capacity-neutral by measurement, not by argument.** Four census-dead chart channels and the two
  geometry context aggregates come out; seven chain-internal state fields and two per-contract
  fields go in. Counts derived by building the variants, never transcribed (the 2026-08-14 ruling):
  current **120** (96 entry / 24 exit, reproducing the external review's split), proposed **118**
  (94 entry). A pre-declared shrink ladder runs 118/109/100/91/82/73, whose last rung's entry phase
  is **49** — exactly the smallest per-fit budget the review projected. The shrink drops features,
  never hidden width, because `causal_day_hidden_size` is frozen at 3.
- **Two era-shortcut bars found while writing the contract.** The ladder-context `max` of
  `moneyness_itm_points` is the ladder's edge, i.e. **ladder width**, and 2022 ladders are ~⅓ the
  width of 2025's (222/384/1,048 symbols at min/median/max), so it is an era detector under a
  chronology where era and time are confounded; per-contract volume/open-interest is era-asymmetric
  in coverage *and* sits in a G3-**barred** family. Both barred, and diagnostic D7 verifies the bar
  with an era probe rather than trusting it.
- **Every proposed field was checked against the issued G3 ledger, not assumed.** Five of the seven
  chain-internal fields are exact ADMITTED rows (`chain_depth_imbalance`, `put_call_depth_ratio`,
  `smile_curvature`, `opra_atm_iv_change_5m`, `opra_implied_spot_dispersion_bps`); the rest derive
  from admitted parents and are flagged as needing a v5 ledger row before any parity/paper claim.
  This also settled the tape source: the whole `entry.opra_implied_spot.v1` family is admitted 8/8
  while ES-derived chart channels appear nowhere in the option-feature ledger, which — with the
  owner's "SPXW and SPX only" instruction and the era-asymmetric official SPX file — puts the tape
  on parity spot in both eras. **A gap recorded rather than smoothed:** the account-state family is
  G3-**BARRED**, so those two fields are development-legal but must be re-certified before parity.
- **The settlement finding is made binding on the build**, not noted: settlement source becomes a
  first-class per-session column, `LAST_QUOTE_MINUTE` becomes per-session (backfill sessions end
  15:58 with 42–53% two-sided closing rows), the zero-recovery twin propagates into member Q's
  targets rather than a footnote, terminal-resolution shares are a **per-era pre-fit QC gate**, and
  a result whose sign differs between settlement twins is declared not bankable.
- **Phase-4a added to the curriculum — the decision point moves before the fit.** A ≤25-parameter
  probe of the proposed features against member P's label, session-clustered, with a known-answer
  twin carrying a planted ≈+20pp lift to calibrate power at this geometry. Verdict rule declared in
  advance: plant recovered and real-feature Wilson-upper lift **< +4.0pp** ⇒ **stop and publish the
  negative with its receipt**; plant not recovered ⇒ advisory-underpowered, owner decides. The bar
  sits between member P's +2.4pp break-even and the +7.1pp best effect ever measured here. This is
  the drawdown preflight's lesson applied before rather than after: an uncalibrated null at this
  scale means nothing.
- **The diagnostic suite makes V5's autopsy a standing gate.** D1 measures the interaction share of
  within-minute score variance (V5 reconstructed additively at `2.38e-7`); **D2's tape-counterfactual
  flip rate is the pass/fail** — a model that cannot reorder side or depth when the state changes
  fails before economics are read; D3 permutes the *inputs* being claimed rather than the label,
  since row 337 established a shuffled-label null structurally cannot see this; D8 requires the
  suite to classify its own planted positive and negative controls before it is trusted.
- **One real defect found in the built trainer contract:** `SessionEpisode.sell_paths` is keyed by
  decision-minute only, but a sell path depends on **which contract** the entry model chose. The key
  must become `(decision_index, ladder_column)`. Mechanical, no semantic change to the frozen law,
  flagged in the design rather than fixed here.
- The design states plainly that "these features cannot close the gap" is a legitimate and
  publishable outcome, and structures Phase-4a so that answer is cheap and receipted rather than
  discovered after a third expensive fit.
- No code written, no fit, no purchase, no vendor contact, no gate/knob/ledger edit; the multi-root
  corpus builder and the adapter remain the parallel session's work and this design is their spec.
  The preflight ran undisturbed throughout.

### Concurrent-session collision on the design document — recorded, not smoothed, 2026-08-16.

- **Two sessions wrote the design at the same time, and one overwrote the other.** A second session
  answered the brief concurrently and created
  [`LEARNING_CONTENT_DESIGN_2026_08_16.md`](LEARNING_CONTENT_DESIGN_2026_08_16.md); this session then
  wrote its own answer to the same path with `Write`, which **replaced that file's bytes**. The file
  was untracked, so git could not recover the superseded version. The other session has since
  re-integrated its material — notably the complete per-field G3 ledger mapping — into the merged
  document, so no content is known to remain lost; but the overwrite happened and is recorded here
  rather than left to be inferred from two near-identical log entries.
- **The duplicate log entry that this cycle added has been removed** in favour of the entry above,
  which is the fuller record of the same delivery. Nothing in the entry above is disputed: both
  sessions independently reached the same design — the chain's own internals as the one unfitted
  family, capacity-neutral at 118 parameters, Phase-4a as the pre-fit decision point, D1–D8, and the
  `sell_paths` keying defect.
- **Independent verification performed this cycle, and it agrees with the entry above:** parameter
  counts derived by building the variants rather than transcribing them — current **120** (96 entry /
  24 exit, reproducing the external review's split), proposed **118** (94 entry), shrink ladder
  109/100/91/82/73; and the G3 mapping checked directly against the issued ledger — five of the seven
  chain-internal fields are exact ADMITTED rows, `entry.opra_implied_spot.v1` is admitted 8/8, and
  the account-state family is BARRED, so those two fields are development-legal but need
  re-certification before any parity or paper claim.
- **Process lesson worth keeping:** `Write` to a path a parallel session may own is destructive and
  silent — it reports success identically whether it created or replaced. When two sessions share a
  work packet, the safe primitive is to read first and edit, or to write to a distinct path and
  reconcile. This is the same class as the two earlier reporting-layer failures in this job (the
  `| tail` exit-code mask and the `status=$?` retry wrapper): the tool said fine while something was
  lost.
- No code was changed, no fit, no purchase, no vendor contact, no gate/knob/ledger edit. Project
  checker green; structure tests pass.

### Backfill acquisition COMPLETE — 794/794 sessions at $19.2450 of the $75 ceiling, 2026-08-16.

- **Owner authorized the resume in conversation ("resume."), and the fresh safety read required by
  the hard rules was performed first:** the V2 approval (which covers the destination-only and
  runner-hardening successors — scope, dataset, schemas, dates and destination are unchanged), the
  charter §2 ceiling, and declaration V4. The runner's own `implementation_sha256` was verified to
  match the V4 preflight's pin **byte-for-byte before launch**, because a mismatch is a refusal by
  design rather than a warning.
- **Complete: 794 cbbo + 794 definition files, 1,588 recorded, 1.7 GB.** 700 files downloaded this
  run and 888 already-present files skipped without a vendor call, which is the hardened runner's
  restart-proportional-to-remaining-work behaviour doing exactly what it was built for. Receipt
  `acquisition_receipt.json`, self-hash **valid**, `quoted_total_usd` **$19.2450** against the
  **$75.00** cap, declaration/preflight/implementation hashes all pinned and matching.
- **A stale V3 wrapper would now be refused, not merely outdated** — worth recording before someone
  reuses it. `cost_preflight_v3.json` pins the pre-hardening runner hash, and the runner checks its
  own hash against the preflight receipt, so any V3-targeted resume fails the guard. The resume ran
  under a V4 wrapper pointing at `BACKFILL_DECLARATION_V4.json` and `cost_preflight_v4.json`.
- **Two launch defects, both mine, both in the reporting layer again.** (1) Invoking the runner by
  script path fails at import — `ModuleNotFoundError: No module named 'v5'`; it must be run as
  `python -m v5.ops.acquire_lifecycle_backfill`. It died before any vendor call, so nothing was spent.
  (2) The first wrapper was written as `... > log 2>&1; echo "EXIT: $?"`, which returns *echo's*
  status, so **a crashed process was reported as exit code 0** — the fourth instance of this job's
  reporting-layer failure class, produced one cycle after documenting the pattern. The replacement
  captures `rc=$?` and was syntax-checked with `zsh -n` before launch. **Standing rule earned the
  hard way: in this job, an exit code is not evidence — the receipt and the file count are.**
- **Post-acquisition QC on the delivered bytes** (8 sampled sessions spanning 2022-06-01 to
  2025-07-31): **zero violations of the same-day expiry rule** — every OSI symbol's own YYMMDD
  equals its session — with 324–988 distinct symbols per session, matching the preflight's resolved
  ladder range (222/384/1,048 at min/median/max). Closing-minute two-sided share measures
  **50.5–51.4%**, inside the 42–53% band recorded when the era difference was found.
- **One conflict reported rather than resolved.** The 2026-08-16 settlement finding records
  backfill-era sessions ending at **15:58**; the delivered raw files carry rows stamped through
  **15:59** in `ts_recv` (America/New_York). Both can be true under different conventions — a CBBO-1m
  bar stamped 15:59 covers the 15:58 minute, which is this project's own `knowable_at = bar_minute + 1`
  discipline — but **which convention the corpus builder adopts changes terminal accounting**, so it
  must be settled from the normalized minute labels rather than assumed. The design's instruction to
  read the session grid per session from the data rather than assume it holds under either reading.
- No fit, no promotion, no broker contact, no reserved sessions, and no spend beyond the receipted
  $19.2450. **The corpus is now the full ~1,045 sessions** the design was written against: 794
  backfill plus the 251 owned quote sessions.

### ACQUISITION DEFECT — the request window drops the final minute of every session, 2026-08-16.

**Every one of the 794 acquired sessions is missing its closing minute bar, and as a result not one of
them can enter the episode build.** Found by post-acquisition QC before any corpus build or fit. No
data is lost or corrupted; a bar was never requested.

- **Mechanism, verified rather than inferred.** `_bounds` in
  [`download_spxw_history.py`](../../ops/download_spxw_history.py) builds the `cbbo-1m` window as
  `09:30 → 16:00` ET, and vendor time ranges are **half-open `[start, end)`**. CBBO-1m bars are
  **end-stamped** in `ts_recv`. So the bar stamped `16:00` — the one covering event minute
  **15:59** — sits exactly on the exclusive bound and was never delivered.
- **The signature is exact and holds across the whole range.** Predicted before measuring: if bars
  are end-stamped under a half-open window, delivered bars run `ts_recv` 09:31→15:59 while `ts_event`
  runs 09:30→15:58. Measured on 2022-06-01, 2023-06-21, 2024-07-11 and 2025-07-31: **389 bars,
  `ts_recv` 09:31→15:59, `ts_event` 09:30→15:58 in every one.** Sampled 62 sessions for a 16:00 row:
  **0 of 62 have one.**
- **Why it blocks everything downstream.** Two independent gates reject the era for this one bar:
  `included_for_episode_build` requires `missing_rth_quote_minutes == 0` over the declared 390-minute
  09:31–16:00 clock, so **all 794 sessions are excluded**; and `attach_candidate_outcomes` raises
  `DatasetError: missing underlying snapshot in full clock` because it requires a finite underlying at
  every one of those 390 minutes. The missing minute is also precisely the one terminal accounting
  reads — `terminal = rth[minute == LAST_QUOTE_MINUTE]`.
- **The owned era is unaffected and the reason is now known.** Its normalized inputs carry a 16:00 row
  in 16/16 sampled sessions (grid 08:01→16:01) because that corpus was acquired by the earlier
  pathd pipeline under a wider window. **Both eras share the same clock semantics** —
  `timestamp_source = databento_cbbo_1m_ts_recv`, seconds binned to zero — so this is a window defect,
  not a cross-era timestamp divergence. An earlier reading in this cycle suspected a one-minute
  era-dependent clock shift from the normalizer's column preference; that suspicion was **wrong and is
  withdrawn**, because the owned corpus is itself ts_recv-based.
- **This also settles the 15:58-versus-15:59 conflict recorded earlier**: both statements were right
  about different columns. `ts_event` ends 15:58, `ts_recv` ends 15:59, and the normalizer keys on
  `ts_recv`, so the normalized backfill grid ends **15:59**.
- **Same family as the parent-scope defect that caused the $671.90 STOP:** a request shape that was
  wrong in a way no receipt could reveal, because the receipt faithfully describes what was asked for.
  The preflight, ceiling, hashes and same-day rule were all correct and all passed. **What no gate
  checked is whether the delivered clock matches the clock the dataset law requires** — worth a
  standing post-acquisition QC assertion rather than a one-off catch.
- **Two routes, and the choice is the owner's because one spends and the other edits a frozen law.**
  **Route A — buy the missing bar:** re-request with an end of 16:01 (or a narrow `[15:59, 16:01)`
  top-up). Arithmetic estimate from the receipted figure, **not a vendor quote**: $19.2450 / 389
  minutes ≈ **$0.05** for a one-minute top-up, or ≈ **$19.29** to re-request the full widened window,
  either well inside the unspent $55.75 of the $75 ceiling. Makes both eras structurally identical at
  390 minutes and touches no frozen semantics. **Route B — adopt a per-era clock:** no spend, but it
  changes the frozen causal clock and *encodes* an era asymmetry into the corpus, which is exactly the
  era-shortcut risk the design bars features for. **Route A is recommended**; it removes an asymmetry
  instead of recording one. Either way this is a request-shape change, so V1's semantic-freeze rule
  makes it a fresh owner decision.
- No spend, no vendor request, no fit, no gate/knob/ledger edit, and no corpus build was attempted
  under the defect. `check_project.py` green.

### Closing-bar repair — owner chose the full re-request; declaration V5, 2026-08-17.

- **The cheap route was priced and refused, which is why the owner had a real choice.** Built
  `ops/topup_lifecycle_closing_bar.py` (12 tests) to buy only the missing bar, and its exact preflight
  over all 794 sessions returned **$0.414442 against the $0.05 authorized — `STOP_OVER_HARD_CAP`,
  nothing downloaded.** My **$0.05 estimate was 8.4x too low**: I had divided the receipted $19.2450
  by 389 bars, assuming cost scales with bar count. Measured, it does not — the bulk window cost
  **$0.0242/session for 389 bars** while a single bar costs **$0.000522/session**, i.e. 2.2% of the
  price for 0.26% of the data, because the per-request floor dominates when buying one minute.
  Receipt: `closing_bar_topup_preflight.json`, self-hash valid, `money_spent: false`.
- **Owner decision, 2026-08-17: re-request the full corrected window (~$19.29)** rather than the
  $0.41 top-up, for one uniform receipt covering the whole corpus instead of a base plus a patch.
  Recorded here as the fresh owner decision V1's semantic-freeze rule requires for a request-shape
  change.
- **The window is now declared data rather than a constant.** `acquire_lifecycle_backfill` gains
  `_window(session, schema, cbbo_close=...)`; `load_declaration` **requires** `cbbo_close_minute` as
  `HH:MM` and refuses a declaration without it, so the close can never again be an implicit constant
  that silently drops a bar. `definition` keeps its full-UTC-day window; only `cbbo-1m` is affected.
- **A pinning gap was found and closed while making that change.** `implementation_sha256` is
  `file_sha256(Path(__file__))` — it covers the runner **only**, while the request window came from
  `download_spxw_history._bounds` in a *different* file. So the window could have been changed
  without invalidating the pin, and the guarantee that "the code that spends is the code that was
  priced" had a hole exactly where this defect lived. Declarations may now carry
  `implementation_dependencies`, whose bytes are verified before any vendor call; V5 pins
  `download_spxw_history.py`. A test fails if those bytes drift.
- **Declaration V5** (`65607a71…`) supersedes V4 without editing it. Field diff against V4 is exactly:
  `request` (close 16:00 → **16:01**), `destination` (a **new root**, since the runner refuses to
  overwrite and a clean 390-bar corpus is the point), `hard_cap_usd` (**$22.00**, deliberately not the
  charter's $75, so the owner's ~$19.29 authorization binds mechanically rather than being
  remembered), the two implementation hashes, `spend_context`, and `supersedes`. **Dataset, schemas,
  date range, source inventory and the same-day rule are byte-identical.** Cumulative spend stays at
  or below **$41.25** of the $75 charter ceiling.
- **V4's receipt and 794 files are untouched** and remain the immutable record of what V4 requested
  and paid for. V4 can no longer be re-run, correctly: its pin no longer matches the changed runner.
- Regression tests pin the defect so it cannot return: a 16:00 close and a 16:01 close produce
  different exclusive ends; the window is DST-aware across July and December; `definition` ignores the
  cbbo close; a missing or malformed `cbbo_close_minute` is refused; and a drifted window dependency
  is refused. **949 v5 tests pass** (+20), `check_project.py` green.
- The V5 preflight is running (cost and symbology calls only). Acquisition executes only on a `PASS`
  at or below the $22.00 cap; otherwise it stops and is recorded as before. No spend yet under V5.

### A delivered-clock gate, and the V5 chain left running attended-by-receipt, 2026-08-17.

- **The lesson is now a tool rather than a paragraph.** `ops/verify_backfill_clock.py` (5 tests)
  asserts that the *delivered* quote clock supports the dataset law: every session must carry all 390
  minutes of the 09:31-16:00 grid, including the terminal minute. Run against the real V4 corpus it
  returns **`FAIL_CLOCK_SHORTFALL`, 0/3 sessions, 3 lacking 16:00** — it reproduces the defect
  mechanically instead of relying on someone thinking to look. **This is the control the job was
  missing:** every prior gate asked whether the request was authorized and affordable; none asked
  whether the bytes that came back carry the clock the builder needs.
- **Owner is away until 16:00 and asked for the authorized work to complete unattended.** Two signed
  rails were honoured rather than bent: charter §4 bars **unattended or scheduled jobs**, so **no cron
  or launchctl entry was installed** — the chain is a background process of this session, the same
  pattern the V3 and V4 acquisitions used. And only already-authorized work runs: preflight gate ->
  acquisition (capped) -> free local clock verification. **The corpus build and any fit deliberately
  do not auto-run**, because both need decisions that should not be taken while the owner is absent —
  in particular the settlement-source law for the parity-settled 76% of the corpus.
- The chain stops rather than continues on every refusal class (ceiling breach, ladder drift,
  declaration or dependency hash mismatch, a non-`PASS` preflight), carries a three-stall guard, and
  confirms each stage **against its receipt rather than its exit code** — this job has now produced
  four separate reporting-layer failures, so a green exit status is not accepted as evidence anywhere
  in it.
- Preflight health at hand-off: 168 sessions checkpointed, **$3.5061 at 150 sessions**, projecting
  **~$18.6** against the $22.00 cap and the owner's ~$19.29 authorization. A transient
  `BentoServerError` was retried automatically and the checkpoint resumed, both as designed.
- **954 v5 tests pass** (+25 this cycle), `check_project.py` green. No fit, no promotion, no broker
  contact, no reserved sessions, and no spend beyond the receipted V4 total at the time of writing.

### The V5 chain died mid-acquisition; diagnosed and resumed on owner approval, 2026-08-17.

- **Found on owner return, by inspection rather than by report.** The chain left running at hand-off
  is **not alive** — no acquisition process exists, and the only surviving Python processes are
  editor language servers. It is the fourth time this job's *reporting* layer, not its downloader,
  has been the thing that failed: the status file's last line is `stage 2 attempt 1: 0/794` written
  **10:28**, while the acquisition's own output files continue to **17:12**. The wrapper was killed
  along with its child, so it never wrote the `attempt 1 rc=` line that would have announced the
  stop.
- **The stop was a process death, not a decision.** No refusal string appears in the attempt log, no
  receipt was written, the stall guard never fired, and the cap was never approached. The attempt log
  contains only transient `BentoServerError` retries, each handled as designed. Diagnosis: the
  process group was terminated externally (session end, sleep, or terminal close) between 17:12 and
  the owner's return.
- **State measured from disk and priced from the receipt, not estimated.** 406 of 794 cbbo sessions
  present (2022-06-01 → 2024-01-11), 813 parquet files, 820 MB. Summing those sessions' own prices in
  the V5 preflight receipt: **$9.0900 incurred, $10.1550 remaining, $19.2450 declared total against
  the $22.00 cap.** Cumulative across V4 and V5 to date: **$28.3350 of the $75 charter ceiling**;
  **$38.4900** if V5 completes, leaving $36.51.
  *(An earlier proportional guess would have been close but wrong; the per-session prices vary five-fold
  with actual 0DTE activity, which is the same property that made the parent-scope defect invisible.)*
- **Owner approved the resume in conversation on 2026-08-17** — *"i approve the re-download of the half
  way installed data"* — and instructed that the v5 tracking documents be brought current first.
  **Resumption re-buys nothing:** `_download_one` skips existing outputs, so the remaining spend is
  the $10.16 of sessions not yet on disk, and the running-total ceiling check still counts every
  declared session.
- **Tracking documents updated before the restart, in that order:** STATUS row 46 (it still described
  the V3 acquisition as running at 915 tests, and is now the current V4-complete → clock-defect →
  V5-resumed state with the spend arithmetic and the settlement fact); PLAN phase 2 (rewritten around
  the two request-shape defects and the delivered-clock QC gate); and this entry. **STATUS was behind
  the packet log rather than ahead of it** — recorded because STATUS is the page that is supposed to
  win on conflicts, so its being stale is itself the reportable condition.
- **The same wrapper is reused rather than rewritten.** It already encodes this job's hard-won
  lessons — `rc` instead of zsh's read-only `status`, never reading an exit code through a pipe,
  confirming each stage against its **receipt** rather than its exit code, treating a refusal
  (ceiling, ladder drift, hash or dependency mismatch) as a decision that stops instead of retrying,
  and a three-stall guard. Rewriting it would have risked reintroducing exactly those defects.
  Stage 1 passes immediately on the existing `PASS` preflight receipt; stage 2 resumes from 406;
  stage 3 runs the free local clock verification.
- No gate, knob, ledger row or declaration was edited to permit the resume, and no new authorization
  was assumed: V5's $22.00 cap and the owner's ~$19.29 decision both bind unchanged.

### V5 acquisition COMPLETE; the clock repair worked, and its residue exposed a worse defect, 2026-08-17.

- **Acquisition finished 21:32 on attempt 1 after the resume: 794/794 sessions, 1,588 files,
  `$19.2450` of the `$22.00` cap.** Cumulative spend **$38.4900 of the $75 charter ceiling**, leaving
  $36.51. The receipt is `acquisition_receipt_v5.json`; stage 3 ran the free local clock verification
  at 21:37 (`clock_verification_v5.json`).
- **The repair worked: 788 of 794 sessions now carry the terminal 16:00 bar, against 0 of 794 under
  V4.** That is the defect this re-request existed to fix, and it is fixed.
- **The gate still returns `FAIL_CLOCK_SHORTFALL` on 7 sessions, and the arithmetic in the summary
  line is not a discrepancy:** 787 fully OK + 7 failed = 794; the "6 lacking 16:00" is a *subset* of
  the 7, because the seventh has its terminal bar but a hole elsewhere.
  - **6 are canonical US early closes** — 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29 and
    2024-12-24 deliver 224 minutes ending **13:14**; 2025-07-03 delivers 210 ending **13:00**. **No
    16:00 bar exists on those days**, so this is not a defect and **no further spend can repair it**.
  - **1 is a genuine interior gap:** 2025-07-30 carries 387 minutes and the terminal bar, missing
    **11:20–11:22**.
- **Following that residue found a defect that PASSES every gate, which is the more important
  result.** `2022-11-25` — Friday after Thanksgiving, and the *only* early close in the 2022 portion
  of the range — reports a full 390-minute clock ending 16:00. It should have stopped at 13:00 like
  its 2023 and 2024 counterparts. **Measured rather than assumed:** its two-sided contract count is
  *frozen at exactly 237* for every minute from 13:00 through 16:00, whereas **every** normal session
  sampled across all four years drifts minute to minute (2022-06-01 191→182, 2022-08-15 176→171,
  2022-12-23 222→217, 2023-06-21 171→166, 2024-03-15 416→407, 2025-07-31 514→496). A frozen count
  over eleven consecutive minutes is a **stale book carried forward**, not trading.
  - **So vendor behaviour changed between 2022 and 2023:** pad the post-close minutes with the last
    book, then later truncate at the close. 2022-12-23 was checked and is **not** an early close
    (Christmas Eve 2022 fell on a Saturday), so its full clock is correct — the pattern is specific to
    the early close, not to the 2022 era.
  - **Why it is worse than the 7 failures:** those fail loudly and get excluded. This one is *silently
    admitted* — full clock, terminal bar present — while roughly three hours of it are fabricated flat
    prices. A first-touch label computed there can never touch, and terminal accounting would read a
    13:00 quote as a 16:00 settlement. It is the same family as the vanishing-contract look-ahead and
    the survivorship rows: **a population that looks complete because the defect is invisible to the
    check that admits it.**
- **Consequences recorded, no code changed and no corpus built.** Three things are now known to be
  missing from the build law, and all three are cheap: an **early-close calendar** (the fixed
  390-minute clock in `included_for_episode_build` is wrong for those days in both directions — it
  wrongly rejects six real sessions and wrongly accepts one fabricated one); a **staleness
  assertion** in `verify_backfill_clock` (a frozen two-sided count across consecutive minutes is
  mechanically detectable, and would have caught 2022-11-25 without anyone thinking to look); and the
  **same verification run against the owned 2025-08→2026-07 root**, which has never been checked and
  contains its own early closes (2025-11-28, 2025-12-24, 2026-07-03).
- **Recommendation, for the owner's decision rather than taken here:** exclude 2022-11-25 and the
  seven failing sessions from the episode build — 8 of 794 sessions, ~1% of the era, against a
  corpus whose whole purpose is honest labels. Excluding a fabricated session is not the
  outcome-shaped dropping that ledger row 331 bars: the exclusion is decided by a *delivery* property
  knowable before any outcome, not by the session's later path.
- No fit, no promotion, no broker contact, no reserved sessions, and no spend beyond the receipted
  V5 total. `check_project.py` green.

### Preparing the fit: a liveness gate, the frozen-baseline boundary, and normalization, 2026-08-18.

Owner instruction: *"i want everything prepared to begin training the model. do anything you
recommend to get us there."* Work is ordered by what the fit inherits — session eligibility first,
because everything downstream is computed from whichever sessions the corpus admits.

- **The clock gate now measures liveness, not presence (`v2`), and the staleness finding generalised
  under measurement.** Rather than a hardcoded holiday calendar — which the vendor's own
  inconsistency across years would break — each session's **last live minute** is derived from the
  data: a per-minute aggregate of top-of-book that is *identical* to its predecessor across every
  contract is a frozen book, not trading. One measurement separates all three populations.
- **The result is more coherent than the presence-only reading suggested. Every early close stops at
  13:00, and the vendor pads a frozen book afterwards; only the amount of padding changed by year:**

  | session | delivered to | last live | stale minutes |
  |---|---|---|---|
  | 2022-11-25 | 16:00 | 13:00 | **180** |
  | 2023-07-03, 2023-11-24, 2024-11-29, 2024-12-24 | 13:14 | 13:00 | 14 |
  | 2024-07-03 | 13:14 | 13:01 | 13 |
  | 2025-07-03 | 13:00 | 13:00 | 0 |

  So the six "early closes" are one market fact (a 13:00 close) plus a vendor artifact that shrank
  from three hours to nothing. **2022-11-25 remains the dangerous member** — it alone pads to a full
  390-minute clock and therefore passes a presence check.
- **Verdict on the backfill era: 786 of 794 build-eligible.** Six `STALE_PADDED`, one
  `EARLY_CLOSE_TRUNCATED` (2025-07-03), one `INTERIOR_GAP` (2025-07-30, terminal bar present,
  11:20–11:22 missing). The receipt now publishes `build_eligible_sessions` directly, so the corpus
  build consumes an eligibility list rather than re-deriving one — presence alone would have admitted
  2022-11-25.
- **Two defects in my own gate, found by running it on the owned era rather than assuming symmetry.**
  (1) The owned corpus stores `ts_recv` as the **pandas index**, so `to_pandas()` promotes it out of
  the columns and the liveness read raised `KeyError` on that era alone; fixed by reading arrow
  columns explicitly. (2) The owned era is delivered on a **wider 08:01–16:01 grid**, so anchoring
  contiguity and the liveness walk on the *delivered* first/last minute mislabelled a genuine early
  close as an interior gap and would have read a post-close repeat as a frozen book — both now
  reason strictly within the required 09:31–16:00 window. Regression tests pin the wider grid and the
  padded-but-complete session. **I also read a piped exit status as success while diagnosing this**
  — the same `| tail` trap already recorded twice in this job; the receipt is what caught it.
- **The semantic freeze was checked rather than assumed, and it bounds the design.** All **12** pinned
  sources match `PREACQUISITION_SEMANTIC_FREEZE_V1` exactly, zero drift. That has a consequence worth
  stating plainly: `causal_day_compact_shared_lifecycle.py` is pinned, so the learning-content
  design's 118-parameter contract **may not be implemented as an edit to it**. The freeze names its
  own sanctioned route — *"development iterations must be generated by the declared alpha ledger
  rather than silently editing this baseline"* — so the revised contract becomes a **new declared
  member alongside an untouched baseline**, charged to the ledger. Recorded now so it is not
  discovered as a governance problem mid-fit.
- **Normalization needed a new driver, and the reason is not a preference.** The pinned
  `normalize_lifecycle_quote_backfill.run()` refuses the V5 acquisition receipt outright — it
  requires `completion.expected_files`, a key the current acquisition runner no longer writes — and
  it is **not resumable**: it raises on an existing output, so a re-run after an interruption marks
  every already-normalized session `DEGRADED`. Against a ~6–8 hour pass, in a job that has already
  lost two multi-hour processes to external death, that is a hazard.
  `ops/normalize_lifecycle_corpus.py` (9 tests) therefore re-uses `normalize_session` **unchanged**
  and replaces only the orchestration: it skips completed sessions, stages each write and renames so
  a crash cannot leave a half-file a resume would trust, records structural failures as data instead
  of crashing, and — a guarantee the original never offered — **verifies the frozen hash before doing
  any work**, so the semantics applied are provably the declared ones.
- **Normalization is running** over the 794 acquired sessions to
  `lifecycle_normalized_2022-06-01_2025-07-31`. Smoke-tested first on two real sessions: correct
  builder schema, parity spot present on **99.2%** of rows — which is also the underlying source the
  design specifies for this era, so the parity-settled majority is served by machinery that already
  exists. Measured cost is ~14 s on the smallest session (141k rows), projecting several hours.
  **The row-wise loop inside the frozen normalizer is the reason and it was left alone deliberately:**
  optimising it would edit a pinned source for speed, which the post-contact rule forbids.
- **969 v5 tests pass** (+15 this cycle), `check_project.py` green. No fit, no spend, no vendor
  contact, no gate/knob/ledger edit, and no corpus built yet.

### The backfill era builds end to end; two more assumptions failed first, 2026-08-18.

Owner approved excluding the eight flagged sessions and asked for the remaining work planned and
executed. The decisive move was to stop building *around* the pipeline and run a real session
*through* it early — which cost minutes and found two structural blockers that would each have
surfaced hours into a full build.

- **Eligibility is now one artifact and the liveness verdict is ANDed into it.** `audit_causal_day_coverage`
  takes `--clock-receipt` (repeatable, one per data root) and intersects `included_for_episode_build`
  with the verifier's `build_eligible_sessions`. This is load-bearing rather than tidy: the coverage
  rule tests `missing_rth_quote_minutes == 0`, which **2022-11-25 satisfies** while three hours of it
  are a frozen book. A v1 presence-only receipt is refused by name so it cannot be mistaken for a
  liveness verdict. **Owned era measured: 247 of 251 eligible** (1 truncated early close, 3 interior
  gaps, and — consistent with the vendor having stopped padding by 2025 — no stale sessions). With
  786 backfill sessions that is **1,033 build-eligible**, against the ~1,045 projection.
- **Blocker 1 — vendor greeks are absent from the backfill era and the coverage audit required them.**
  The owned era carries vendor `iv/delta/gamma/theta/vega`; the backfill is normalized from CBBO
  top-of-book and carries none, so the audit raised on every session of it. They are **diagnostics
  only** — the module's own receipt says the builder must recompute greeks causally, and it does — so
  they are now optional, with `vendor_greeks_present` recorded per session. Requiring a reporting
  column would have refused three quarters of the corpus.
- **Blocker 2 — the parity spot is unsolvable at exactly the minute settlement is read.** The V5
  repair restored the 16:00 *bar*, but the strict solver still leaves the last 2-5 minutes NaN
  (measured: last solvable minute 15:55-15:58 across sampled sessions, 16:00 never), and
  `attach_candidate_outcomes` requires a finite underlying at **every** minute of the 390-minute clock
  and refuses the whole session otherwise. **So the approved plan's fallback — "use the documented
  parity spot" — does not work as written for this era.** Measured cause: paired two-sided strikes
  decay 10 (15:50) -> 6 (15:55) -> 3 (15:59) -> **1 (16:00)**, below the frozen normalizer's
  three-strike minimum.
- **The fix is a relaxed solve, and it is principled rather than a fudge.** SPXW options are
  **European**, so `S = K + C - P` holds *exactly* at any single paired strike; averaging strikes is
  noise reduction, not a mathematical requirement. `ops/repair_parity_spot.py` (8 tests) recomputes
  **only** the minutes the strict rule left empty, using the paired strikes nearest the money, and
  labels every value: `underlying_price_source` (`strict_parity` / `relaxed_parity` / `unavailable`)
  and `underlying_parity_strikes`, so a one-strike estimate is visibly thinner than a ten-strike one.
  **Nothing is forward-filled and nothing is carried between minutes** — an unsolvable minute stays
  NaN and is reported, which is the discipline the stale-book defect teaches. Measured on 2022-06-01:
  387 strict minutes, 3 repaired (15:58 with 4 strikes, 15:59 with 3, 16:00 with 1), clock complete.
- **End-to-end proof on a real backfill session — the pipeline works.** 390 candles, 7,755 ladder
  rows, **3,260 candidates**, exits resolving as 3,224 `executable_bid` + 36
  `validated_cash_settlement` at 60m (2,809 / 451 at 120m), and the declared first-touch label at
  **100% coverage** with a **36.3%** base rate. That last number is itself a result: the owned year's
  is **31.8%**, so the eras differ as the design predicted, and per-era baselines must be measured
  rather than inherited.
- **Normalization is running** (70/794 at the time of writing) and is the gate on everything
  downstream. `ops/normalize_lifecycle_corpus.py` reuses the frozen `normalize_session` unchanged and
  verifies its pinned hash before doing any work.
- **Chain-internal features built, outside the pinned builder.** `research/chain_internal_features.py`
  (10 tests) derives the seven minute-common state fields and the two per-contract fields from the
  ladder the pinned builder emits — risk reversal and its 15-minute change, ATM IV change, chain and
  per-contract depth imbalance, put/call depth ratio, smile curvature, smile residual, and parity
  dispersion. Tests pin the properties that matter: **causality under a mutated future**, lags that
  return NaN across a missing minute rather than reaching further back, per-side smile fits, and
  dispersion that is NaN rather than invented when strikes are unpaired.
- **991 v5 tests pass** (+22 this cycle), `check_project.py` green. No fit, no spend, no vendor
  contact, no gate/knob/ledger edit.

### The superseded 389-bar corpus is named and mechanically refused, 2026-08-18.

- **Owner raised the hazard:** two 1.7 GB corpora now sit on the SSD covering the *same* 794
  sessions, and they differ by exactly one bar per session. Nothing that globs a directory can tell
  them apart, and the differing bar is the one terminal accounting reads. Confirmed by measurement
  before acting — the old root delivers **389 minutes ending 15:59** with no terminal bar on every
  sampled session; the fixed root delivers 390 ending 16:00.
- **A guard first, because naming only helps a reader.** `assert_source_clock()` in
  `ops/normalize_lifecycle_corpus.py` samples a source root and refuses one whose sessions lack the
  terminal minute, before any work. It is deliberately a **majority** test over a spread sample
  rather than a unanimity, because six genuine early closes lack 16:00 legitimately — and the
  separation is not marginal: **fixed root 12/12, superseded root 0/12.** Three tests pin it,
  including one proving a handful of real early closes cannot refuse a good root.
- **Then the rename.** `lifecycle_spxw_quote_backfill_2022-06-01_2025-07-31` ->
  `..._SUPERSEDED_389bar_DO_NOT_BUILD`, with a `SUPERSEDED.md` inside recording the defect, its
  cause, the replacement root, and the verification status. Checked first that **no code references
  the old root** (only declarations V1–V4 and V4's receipt, which are immutable historical records),
  that **nothing had files open** under it, and that the running normalizer reads the *fixed* root —
  it continued uninterrupted through the rename (108/794 after).
- **Kept rather than deleted, deliberately.** V4's acquisition receipt records **1,588 file paths and
  hashes** and is the immutable record of what was requested and paid for ($19.2450); deleting the
  files would leave that receipt unresolvable. Space is not the constraint — 1.8 TB free — so the
  evidence is retained and merely made unmistakable. **Deletion remains available as an owner call
  and would reclaim 1.7 GB;** the four orchestration tests that the new guard initially broke were
  repaired by giving them a valid source root rather than by adding a skip flag, because an escape
  hatch is exactly how a guard like this gets bypassed later.
- **994 v5 tests pass** (+3), `check_project.py` green. Normalization undisturbed.

### Normalization complete; parity repair running; the build driver written, 2026-08-18.

- **Normalization finished: `PASS`, 794/794, zero degraded**, on the first attempt, with the frozen
  normalizer's pinned hash verified before any work. Receipt
  `normalization_receipt_v5_attempt1.json`. Parity-spot coverage per session runs **min 180 / median
  385 / max 390** of 390 minutes — the median shortfall of five is exactly the terminal gap the
  repair pass exists to close, and the 180 is an early-close session already excluded by the clock
  gate.
- **Parity repair is running** over all 794 normalized sessions into
  `lifecycle_repaired_2022-06-01_2025-07-31`, filling only the minutes the strict solver left empty
  and labelling each with its source and backing strike count.
- **Built `ops/build_lifecycle_corpus.py` (9 tests), the last blocker before a corpus.** The pinned
  `build_causal_day_dataset.run()` takes settlement only from `load_validated_settlements`, which
  demands a receipt marked `VALIDATED_FOR_TERMINAL_ACCOUNTING` — and that receipt exists for **243
  owned sessions and nothing else**. Since roughly three quarters of this corpus is parity-settled,
  the pinned orchestration structurally cannot express what most of it needs. The driver therefore
  reuses `build_session` and its label law **unchanged** and replaces only the orchestration:
  - **Settlement is resolved by era, never by fallback.** `official_1600` for the owned era from the
    validated receipt; `parity_close` for the backfill era, read through `prepare_quotes` so the
    value is the one the builder itself will see rather than a separately-derived number that could
    drift from it. An owned session missing a validated settlement is **refused**, not quietly
    downgraded to a derived one — pinned by a test.
  - **Every emitted table carries `era` and `settlement_source`**, so no later reader has to infer
    which law produced a number. The paired zero-recovery sensitivity needs no second pass: the
    pinned builder already emits `net_*_zero_recovery_*` beside its settled columns.
  - **Base rates are reported per era and never pooled**, also pinned by a test. This is not
    bookkeeping — the first real backfill session measured **36.3%** against the owned year's
    **31.8%**, so a pooled figure would average over a genuine era difference.
  - Resumable per session across all five tables, staged-write-then-rename throughout.
- **1,003 v5 tests pass** (+9), `check_project.py` green. No fit, no spend, no vendor contact.
