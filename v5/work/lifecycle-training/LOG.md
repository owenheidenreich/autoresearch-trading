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

### Owner accepts the carried close; the corpus builds at 1,014 sessions, 2026-08-18.

- **The question.** After the relaxed solve, 216 of 794 backfill sessions were still unsolvable at
  exactly one minute — **16:00**. Measured cause, not inferred: at the close those sessions carry
  plenty of live quotes (86 calls / 115 puts on one inspected) but **zero strikes with both a live
  call and a live put**, because deep OTM options go bidless into the close and the two sides survive
  at *disjoint* strikes. No window width solves that.
- **Owner ruling, 2026-08-18:** *"accept all 1,033, label the carried close, and have Phase 5 report
  what fraction of trades actually settle at 16:00 on those days. Not to gate anything — just so that
  when a result lands, nobody has to wonder whether a few carried index points were doing the work."*
  Owner also noted the trading context that bounds the risk: **the last ten minutes are not traded
  anyway**, and **SPX is cash settled, so holding through the close carries no assignment risk**.
- **Implemented as a deliberately narrow carry.** `repair_parity_spot` (v2) carries the last solved
  **underlying index level** into a *contiguous terminal run* only, labelling
  `underlying_price_source = carried_parity` and recording `underlying_carry_minutes`. Two limits are
  structural rather than conventional: it never carries a **contract quote** — doing so would
  fabricate a tradeable price, which is precisely the stale-book defect this pipeline exists to catch
  — and an **interior gap is never carried**, because a hole mid-session is a data problem rather
  than a thinning chain. Three tests pin exactly that.
- **The measured exposure is as small as the owner expected, and the guards compose.** Of the carried
  sessions, **214 lie inside the eligible set and every one carries exactly 1 minute.** The only
  large carry — **181 minutes, 2022-11-25** — is the stale-padded session, and it is *already*
  excluded by the independent clock-liveness gate, so it never reaches the corpus. Two guards written
  for different defects caught the pathological case between them.
- **The exposure is also bounded by construction and now measured per session.** Entries stop at
  **15:00**, and only a **cash-settled** exit reads the terminal underlying at all — a trade closed on
  an executable bid never touches it. The build receipt therefore records
  `settlement_close_carried`, `settlement_carry_minutes` and `cash_settled_share` per session, plus a
  corpus-level `carried_close` summary, so Phase 5 can state the fraction as a footnote rather than
  anyone having to wonder.
- **Underlying clocks after the carry: 785 of 794 complete.** The 9 remaining are the early closes
  and interior-gap sessions the clock gate already excludes. **Buildable: 771 backfill + 243 owned =
  1,014 sessions**, against 243 before this job began.
- The corpus build is running to `lifecycle_corpus_2022-06-01_2026-07-31`, resumable per session
  across all five tables, with era and settlement source stamped on every emitted row.

### Work-allocation memo SIGNED; this session now terminates at the build, 2026-08-18.

- [`WORK_ALLOCATION_MEMO_2026_08_18.md`](../../governance/WORK_ALLOCATION_MEMO_2026_08_18.md) is
  signed and binding on every session executing job 46. Verified on disk and read in full before
  acknowledging — a signed protocol a later session cannot read cannot bind it.
- **Governing split:** Fable decides what is true or what should be built; Opus builds it and proves
  it works. Where a task is mostly mechanical but contains one buried judgement, it goes to Opus with
  the judgement **named and escalated** rather than absorbed.
- **§5 is a hard stop at eight boundaries.** A session may not cross one by continuing to work, even
  when the next task looks small. The stated rationale matches this job's own history: the expensive
  errors here were not wrong answers but **decisions made by whoever happened to be holding the
  keyboard** — a design question settled silently inside an implementation, a marginal result read
  favourably by the session that produced it.
- **Two provisions do the real work.** The **ambiguity rule**: if a session cannot tell whether the
  next task is execution or judgement, that uncertainty *is* the answer — stop and ask. And the
  ambiguous-4a row **forbids the producing session from interpreting its own borderline result**,
  which is precisely the failure mode rows 41/341 record.
- **Immediate effect on this session.** The running build is §2 item 1 (Opus, execution). On its
  completion this session **reports and stops**: receipt, per-era base rates, and the carried-close
  footnote (`settlement_close_carried`, `settlement_carry_minutes`, `cash_settled_share`). It does
  **not** continue into the parameter budget, the `SessionEpisode` adapter, or the architecture
  member, all of which sit past the boundary. The next task is **Fable's** pre-fit adversarial review,
  carrying the design question nobody has yet decided: whether the two eras are **one population or
  two**, given the measured 36.3% backfill against 31.8% owned base rates.
- **§7 records a fourth reporting-layer defect** beyond the three already logged (`| tail` masking a
  crashed process, `status=$?` killing a retry wrapper, a preflight losing an hour to one timeout): a
  `Write` that replaced a live file as silently as it would have created one. Standing instruction
  adopted: **before trusting any wrapper, monitor or guard, state what it does on the failure path.**
- Build progress at the time of writing: 382 of 1,014 sessions, running.

### BOUNDARY 1 REACHED — corpus built at 1,014 sessions; handing to Fable, 2026-08-18.

- **The corpus exists: 1,014 sessions, 12,177,808 rows across five tables, 1.4 GB.** 3,226,673
  candidates and 7,833,707 ladder rows. Settlement sources **771 `parity_close` / 243
  `official_1600`**. Two sessions failed and both are known interior-gap cases the clock gate had
  already flagged (2023-09-21 missing 09:34, 2023-10-02 missing 10:41 and 16:00); no other session
  failed. Receipt `corpus_build_receipt_attempt1.json`. Against 243 sessions when this job began.
- **A correction to a claim this log made yesterday.** The per-era base rates measured on the whole
  corpus are **owned 31.80% (n=243, sd 6.10) against backfill 30.59% (n=769, sd 6.66)** — a gap of
  **1.2 points**, not the 4.5 implied by the single session (36.3%) measured on 2022-06-01. **That
  earlier figure was one session and it was session-level noise**, well inside a 6.6-point
  cross-session standard deviation. The per-era reporting requirement stands, but the evidence for
  "the eras differ measurably" is much weaker than stated, and the incoming review should read the
  measured distributions rather than the earlier sentence. Label coverage is **99.97% / 99.88%**.
- **A fifth reporting-layer defect, found and fixed before the handoff.** The carried-close footnote
  the owner explicitly requested reported **zero carried sessions against 214 real ones.** Cause:
  `pd.read_parquet(path, columns=[])` returns a frame with **zero columns rather than the schema**,
  so the optional-column check reported every provenance column absent and silently disabled the
  detection. The same line exists in `audit_causal_day_coverage` with a pyarrow fallback, which is
  why it worked there and masked the error here. **The corpus data is unaffected** — the settlement
  *values* were read correctly from `underlying_price`; only the labelling of which sessions were
  carried was lost. Fixed to read the parquet schema directly, verified on real sessions, and pinned
  by a regression test that asserts the trap. This is exactly §7's pattern: the tool reported clean
  because it could not see the thing it was built to look at.
- **The carried-close footnote, now measured** (joined from the repair and build receipts, no rebuild
  needed since the corpus is correct):
  - **214 of 1,014 sessions (21.1%) rest on a carried close, every one carried exactly 1 minute.**
  - Cash-settled share on those days: **3.30% at 60m**, 11.32% at 90m, 18.97% at 120m (max 9.24% /
    18.52% / 26.70%).
  - So the candidates that could depend on a carried index point at all are **22,880 of 3,226,673 —
    0.709% of the corpus at 60m**, and 4.079% at 120m. Entries stop at 15:00 and only a cash-settled
    exit reads the terminal underlying, which is what bounds it. As the owner anticipated, this is a
    footnote rather than a factor.
- **1,007 v5 tests pass**, `check_project.py` green. No fit, no spend, no vendor contact.
- **STOP under §5 of the signed work-allocation memo.** This session does not continue into the
  parameter budget, the `SessionEpisode` adapter, or the architecture member. Next task is **Fable's**
  pre-fit adversarial review, including the one-population-or-two question — which the corrected base
  rates materially reframe.

### BOUNDARY 2 REACHED — pre-fit review returns ONE POPULATION; handing to Opus, 2026-08-18.

- **Verdict received and spot-checked rather than relayed on trust.** Finding
  [`PREFIT_CORPUS_REVIEW_2026_08_18.md`](../../research/findings/PREFIT_CORPUS_REVIEW_2026_08_18.md)
  exists, both receipt scripts are archived beside the job's other receipts, and STATUS, this log and
  the evidence index all carry it. **The two eras are one population**: train one model across both,
  era is not an input, the backfill is not down-weighted — conditional on fixing one feature.
- **The reasoning that carries it is the fold-seam measurement, and it stands independent of who read
  it.** The era gap is +1.21pp while the 2022-vs-2023 gap *inside* the backfill era is larger at
  +1.47pp; yearly rates 31.71 / 30.24 / 30.16 / 30.99 / 32.18 show no trend or step, with the two
  highest values at the two chronological extremes; and under the frozen chronology the fit will
  actually use, **the era boundary is the quietest seam in the data — +0.63pp against ±1.80pp
  transitions interior to the backfill era**. Base rate tracks realised volatility (ρ=+0.355), i.e. a
  regime variable rather than a calendar one.
- **Defect 2 verified as described in mechanism:** `implied_spot_dispersion_bps` is the *only* era
  detector — probe AUC **0.760** with all 11 state fields, **0.430** without it, **0.837** from it
  alone, and the design's actual hypothesis (the other six chain-internal fields) sits at **0.481**,
  chance. The cause is a level effect, predicted 0.392 against measured 0.396: dollars of dispersion
  divided by an index that rose 3,960 → 6,845. Dispersion ÷ quoted spread is flat across all five
  buckets, so the fix preserves the channel and the 118-parameter contract is unchanged.
- **Defect 1 independently reproduced by this session before handing it on**, because it is the one
  that blocks the next task: `2025-04-09` and `2025-04-10` carry **0 rows and 40 columns** against 139
  elsewhere; a per-session read of the label column raises `ArrowInvalid`; a directory-level read
  succeeds on all 3,226,673 rows. That asymmetry is why every check to date passed — and the adapter
  reads per session. It is also a genuine economic fact rather than a bug: on those tariff-spike days
  the cheapest near-money contract cost $2,000–$2,050 and the signed ticket cap correctly admitted
  nothing. It further explains the n=769-vs-771 puzzle this log flagged at Boundary 1.
- **Four checks no existing gate performs came back clean:** an independently re-implemented label law
  agrees on **32,377 candidates across both eras with zero mismatches**; the frozen-book scan finds a
  maximum repeat run of **1 minute**; the design's §4.3 per-era exit-resolution gate is reported for
  the first time and the eras differ by at most **0.55pp with 0% blocked**; and chain-feature coverage
  is identical to three significant figures in every bucket, so the 2022 narrow-ladder degradation the
  design feared **does not occur**.
- **The review corrects the design on its own terms.** §3.1's bar targets fields monotone in ladder
  size, but the corpus ladder table is the ±25-point band — 20 contracts wide in 2022 and 2026 alike
  — so that hazard does not exist here while an index-level proxy walked straight past the bar. Two
  related mismatches recorded: the chain features are **band-local, not chain-wide**, and the corpus
  candles are **ES, not the SPX parity spot the design specifies**.
- **Two items this session does not decide.** (1) The **ES-versus-SPX tape source** is an owner
  boundary ruling the review deliberately refused to take, with the measurement supplied (basis +20 to
  +30 points, near-constant within a session, cancelling in the four difference-based tape channels).
  (2) The review session's **model switched from Fable to Opus partway through**, which §3 assigns to
  Fable — disclosed by the owner rather than discovered later. Whether the verdict stands under the
  memo is a governance question, so it is surfaced rather than waved through.
- **STOP under §5.** Next task is **Opus** — parameter budget, `SessionEpisode` adapter, architecture
  member — carrying the six directed changes in §6.3. 1,007 tests pass, checker green.

### Pre-fit adversarial review: ONE POPULATION, conditional on removing one feature, 2026-08-18.

Boundary-1 review under §3.1 of the signed work-allocation memo. Full result:
[finding](../../research/findings/PREFIT_CORPUS_REVIEW_2026_08_18.md), receipts
`prefit_review_receipt.json` (`287e59bd…`), `prefit_review_followups.json`,
`prefit_review_session_metrics.parquet`, and the two scripts archived beside them. **No trade
economics were read and no feature-to-label statistic was computed** — that is Phase 4a's charged
job, and measuring it here would have been an uncharged experiment that let the shrink ladder be
tuned by peeking. The one fitted object was the owner-authorized throwaway era probe.

- **The corpus is sound, and four of the six checks had never been run.** Headline counts, per-era
  base rates and the carried-close footnote all reproduce **exactly** by independent re-derivation.
  New: an **independently re-implemented label law agrees on 32,377 candidates across both eras with
  0 mismatches**; the frozen-book scan finds a **maximum repeat run of 1 minute** in both eras, so no
  stale session reached the corpus; the design's §4.3 per-era exit-resolution gate is **reported for
  the first time** and the eras differ by at most **0.55pp** at any horizon with **0.00% blocked**;
  and there are **zero same-minute double-touch NaNs** and no infinities in any chain field.
- **Defect 1, blocking for the adapter. 2025-04-09 and 2025-04-10 hold zero candidates** and are
  written as degenerate 40-column files with no label columns. The cause is not a bug but the signed
  risk law working: on those tariff-spike days the cheapest OTM contract within 25 points cost
  **$2,000–$2,050** against a $2,000 ticket cap, so nothing was affordable. This is the **n=769 vs
  771** discrepancy. **A per-session read of the label column raises `ArrowInvalid` on those two
  files**, and the adapter is specified to read per session — directory reads unify the schema and
  survive, which is exactly why nothing had noticed. Same family as §7's standing note.
- **Defect 2, blocking for the feature contract. `implied_spot_dispersion_bps` is a calendar
  detector and it is the only one.** Held-out era probe: **all 11 state fields AUC 0.760; without
  that one field 0.430; that field alone 0.837**; the other six chain fields 0.481 and the tape
  0.413. Mechanism measured rather than guessed — the field is dollars of dispersion over the index
  level, and **SPX rose 3,960 → 6,845**, so the predicted level-effect ratio **0.392** against a
  measured **0.396**. The era-free form is already in the data: **dispersion ÷ quoted spread is flat
  across all five buckets** (0.177/0.187/0.187/0.159/0.177).
- **The design's §3.1 bar was aimed at the wrong hazard.** It bars fields monotone in *ladder size* —
  but the corpus ladder table is the ±25-point band, **20 contracts and 45 points wide in 2022 and
  2026 alike**, so that hazard does not exist here, while a field monotone in the *index level* walked
  straight past it. Recommended generalisation: monotone in **any** slowly-varying calendar quantity,
  verified by measurement.
- **Three design/corpus mismatches recorded.** The corpus candles are **ES, not the SPX parity spot**
  the design specifies (signed basis +20.3 / +29.8 points, within-session sd 0.65–1.55 — a futures
  basis); the four tape channels are differences so it cancels to statistical invisibility (tape-only
  probe 0.413), but the provenance question is an **owner boundary ruling**, deliberately not taken
  here. The chain-internal features are **band-local, not chain-wide** (full chain is 252–280
  contracts). And `smile_curvature` pools the two sides where the design says per-side — **suspected
  skew contamination, measured, and it is not there** (r=0.80 with the per-side average, −0.03 with
  skew); reported as wording to reconcile, not a directed change.
- **THE RULING — one population, single model, era not an input, no down-weighting.** The era gap is
  **+1.21pp (p=0.0088)** while the **within-backfill 2022-vs-2023 gap is larger at +1.47pp
  (p=0.0365)**. Yearly rates 31.71/30.24/30.16/30.99/32.18 show no trend and no step — a shallow U
  whose highest values are the two chronological extremes. **Under the frozen chronology the era
  boundary is the quietest seam in the data: +0.63pp at block3→block4, the smallest transition in the
  sequence**, against −1.80 and +1.80 interior to the backfill era. Base rate tracks realised
  volatility (ρ = +0.355), a regime variable. Action space identical across eras. **The ruling is
  conditional on Defect 2** — as it stands, "one population" is false because of a normalisation we
  introduced rather than because of the market.
- **Six directed changes before the fit**, in the finding's §6.3: renormalise or drop the dispersion
  field; generalise the §3.1 bar; **set D7's pass bar at ≈0.55 from the measured 0.430–0.481 baseline**
  and have it report the per-field ablation rather than one number; the adapter must handle the two
  zero-candidate sessions without crashing and without silently dropping them; report executable
  economics **per block** since the spread runs **3.03/3.64/3.03/2.25/1.84% of premium** across
  2022→2026, so the model trains at a ~3% toll and is scored at ~1.9%; and state in the declaration
  that the chain features are band-local so attribution does not credit information the model never
  saw.
- **The feature contract otherwise passes unchanged.** Coverage is identical to three significant
  figures across all five buckets for six of the seven chain fields (worst 94.9%), and the 2022
  narrow-ladder degradation the design feared **does not occur**, because the band is the same width
  every year. The training prefix is not a feature-starved era.
- **STOP under §5.** Next boundary hands to **Opus**: parameter budget, `SessionEpisode` adapter,
  architecture member. No fit, no spend, no vendor contact, no pipeline code changed by this review.

### BOUNDARY 3 REACHED — budget, member and adapter built; the era defect is measured out, 2026-08-19.

- **The parameter budget is measured on the real chronology, not projected.** 327,557 candidate
  states over 1,011 sessions (324/session) via [`measure_effective_sample_size.py`](../../ops/measure_effective_sample_size.py),
  receipt `effective_sample_size_2026_08_18.json`. The autocorrelation route gives an effective *n*
  of 12,687–33,790 across the four labels and a budget of **634 parameters at the binding label**
  (`reached_10_itm_60m`, τ=25.8); the conservative design-effect route gives 2,028–3,727 and **≈101**.
  The 08-14 convention applies unchanged: proceed on the generous route, report the conservative
  figure beside it. This is corpus-wide; per-fit prefix budgets are tighter and are not yet computed.
- **The declared member is built and counted from the built module: 118 parameters, 94 entry and 24
  exit**, against the frozen baseline's 120 — [`causal_day_chain_state_lifecycle.py`](../../research/causal_day_chain_state_lifecycle.py).
  Tests pin the capacity claim from both modules, refuse a batch whose chain state is absent or
  non-finite rather than zero-filling it, and assert that **changing chain state reorders the
  ladder**, which is the exact property V5 lacked.
- **A reseal guard caught an edit mid-flight and it was right to.** Widening `CausalPolicyBatch` to
  carry chain state broke the sealed V3 action-value declarations, which pin
  `causal_day_architectures.py` by digest. The file was reverted and the field now travels on a
  `ChainPolicyBatch` subclass, leaving the seal byte-intact. Second pinned file this work has had to
  route around rather than through.
- **Directed change 1 is applied AND re-measured, which is the part that matters.** The field is now
  `implied_spot_dispersion_ratio`, normalised by the contemporaneous quoted spread. Re-probed on the
  same 150 sessions, same seed, same group split as the review
  (`era_probe_reprobe.py` / `era_probe_reprobe.json`): **the field alone falls from AUC 0.837 to
  0.4548**, the full 11-field state vector from **0.760 to 0.4314**, and the six other chain fields
  are unmoved at **0.4813** — confirming nothing else changed. On the member's own state channels the
  field is worth **+0.0014 AUC** (0.4319 with, 0.4305 without): era-blind. The median ratio across
  the five buckets is **0.177 / 0.187 / 0.187 / 0.159 / 0.176** while SPX runs 3,960 → 6,845, against
  the bps form's measured 0.396 decline. The mechanism is now explicit: the median quoted spread is
  tick-quantised at **$20.00 per contract in every bucket**, so it is a scale-free normaliser where
  the index level is a calendar clock. The review's condition on the one-population ruling is
  discharged by measurement.
- **The `SessionEpisode` adapter exists** — [`lifecycle_episode_adapter.py`](../../research/lifecycle_episode_adapter.py),
  19 tests, of which 11 run against the real corpus. Three things in it are load-bearing and each
  came from a measurement rather than a preference.
- **Sell paths are read from the raw quote file, never from the corpus `ladder` table — and the cost
  of the obvious shortcut was measured before the choice was made.** `build_session` stores
  `ladder_state(whole_live_chain=False)`, the ±25-point band; its own docstring says the simulator
  must use the full chain instead. A bought contract leaves that band exactly when the trade is
  working. Measured across four sessions spanning both eras: **29.1% of candidate exits, and 18.3%
  of the exits belonging to winning trades, fall on a minute where the bought contract is absent
  from the corpus ladder table.** Building paths from it would have truncated the winners and read
  out as "the exit adds nothing" for a reason unrelated to the market.
- **The exit law is not re-derived on trust: it is checked against the frozen column it must
  reproduce.** This module computes the first-later-bid / validated-settlement value independently of
  `attach_candidate_outcomes`, and a test asserts the terminal element of every 60-minute sell path
  equals the pinned `net_bid_60m_usd` — **9,758 candidates across both eras, maximum absolute
  difference 0.000000000, zero mismatches**.
- **The two zero-candidate sessions are full episodes, not empty ones.** 2025-04-09 and 2025-04-10
  now build as **326 decision minutes with zero feasible actions**: every minute the policy had to
  answer WAIT, no entry action, no target. They are legitimate no-trade days under the signed risk
  law, so the WAIT head still trains on them and a serial simulator still learns the account sat
  flat. Neither dropped nor crashed on, as §6.3 required.
- **Two silent-NaN failure modes in the trainer were closed while the adapter was being wired, and
  both were live.** `smooth_l1_loss` over an empty selection returns NaN, so a no-trade session would
  have turned every epoch's loss into NaN; and ~0.05% of the corpus's labels are unknown, so a single
  masked-in NaN target did the same. Unknown-label actions now stay feasible and stay unsupervised —
  masked out of the loss, never removed from the action set — which is what the design specified.
  `lifecycle_trainer.py` is not a pinned file; the change is three lines and carries two tests.
- **Also proved, because slicing a whole-session feature frame is only valid if it is:** the
  per-minute rebuild and the compute-once-and-slice path agree to **atol 0.0** on a real session, and
  a mutate-future control on a copied corpus — candles and ladder both perturbed after 11:00 —
  leaves every decision minute at or before 11:00 **bitwise identical**.
- **1,037 tests green, `check_project.py` green.** No fit, no spend, no vendor contact, no unattended
  job, and no pinned file edited.
- **STOP under §5.** Boundary 3 is "adapter + member built, before Phase 4a runs": report the built
  parameter count and the measured budget, then wait for the owner to confirm before **Opus** runs
  the calibrated probe. Two owner decisions remain open and neither blocks 4a: the **ES-vs-SPX tape
  source**, and the **event calendar** (§4). Remaining §6.3 items are declaration content for 4a
  itself — D7's ≈0.55 bar with per-field ablation, per-block executable economics, and stating that
  the chain features are band-local.

### Two owner rulings close the open questions; the tape is rebuilt from SPX, 2026-08-19.

- **RULING 1 — the tape is SPX-derived (parity spot), not ES.** The active policy is SPXW/SPX-only
  with no futures input. The owner's reasoning is provenance rather than arithmetic, and it is worth
  recording in full because it overrides a measurement: the basis genuinely cancels in the four
  difference-based tape channels the member reads — the pre-fit review measured tape-only era-probe
  AUC at 0.413 and this session measured `move_from_open_points` correlating at **1.0000** between the
  two tapes — but *"provenance is what a future session inherits, and 'the candles are ES' is exactly
  the kind of quiet inconsistency that gets discovered mid-fit and forces a rebuild."*
- **RULING 2 — the event calendar is CLOSED, and closed by operating rule rather than by a model
  change.** No calendar data will be acquired and no feature is admitted. Pre-open releases (08:30 ET)
  need no feature because the print lands before the first decision minute at 09:31, so the model
  already reads the aftermath — the only part it could trade. FOMC days are handled by **not running
  the bot**. Two diagnostic-only Phase-5 reporting requirements follow and neither gates anything:
  keep the event-day flag so the first assumption is *verified* rather than assumed, and report
  results **with FOMC sessions excluded alongside the all-sessions figures**, because the corpus still
  contains FOMC days the model will have learned from. Recorded in the memo §4 so a future session
  does not rediscover "the model cannot see the calendar" and propose solving it again.
- **The rebuild is narrower than it looks, and that was verified before a byte was written.** Only
  `candles` and `minutes` depend on the tape: the ladder, the candidates, every label and the atlas
  are computed from the quote file's own `underlying_price` and were already SPX-denominated.
  Building one session both ways and comparing frames: **`ladder`, `candidates` and `atlas` are
  identical, `candles` and `minutes` are not.** So the 9,758-candidate exit-law equality, the per-era
  base rates, the effective-sample-size measurement and every label finding in the pre-fit review
  survive the rebuild untouched.
- **A new op emits the tape in the ES file shape so the pinned builder needs no edit** —
  [`build_parity_spot_candles.py`](../../ops/build_parity_spot_candles.py), 8 tests. `build_session`
  is inside the semantic freeze; a test asserts that the pinned `prepare_es` accepts the emitted
  frame, which is what makes the swap free.
- **The clock offset is the load-bearing detail.** An ES bar stamped `t` is knowable at `t+1`; a
  `cbbo-1m` snapshot stamped `t` *is* the market at `t`. So bar `t` is filled from the snapshot at
  `t+1` — bar 09:30 from the 09:31 snapshot through bar 15:59 from the 16:00 snapshot. Exactly 390
  bars from exactly 390 quote minutes, and every bar stays first-readable one minute after the state
  it describes. Coverage is **390/390 minutes on every sampled session in both eras**.
- **Open, high and low equal the close, and that is a statement rather than a defect.** A one-minute
  CBBO snapshot is a single observation of the index; there is no intra-minute range and inventing
  one would be fabrication. All four channels the member reads are functions of the close series.
  Volume is 0 because the index has no volume — which the design already required, having barred
  every volume channel for exactly this reason.
- **The corpus now stamps `tape_source` on every table.** The pinned builder writes the values into
  columns named `es_open`/`es_close` and may not be edited to rename them, so the stamp beside them is
  what stops a later session reading `es_close` and concluding the policy takes a futures input it was
  ruled out of. Read from the candle file rather than assumed, and a file declaring two sources is
  refused.
- **A 132.5-point one-minute step appeared in the tape receipt and was chased down rather than
  accepted.** It is real: **2025-04-09 at 13:20 ET**, the tariff-pause announcement, corroborated by
  the independent ES tape at 130.2 points and matching session ranges (527.1 SPX vs 532.8 ES,
  one-minute return correlation 0.957). The next two — 2025-04-07 at 10:10 (78.7 vs ES 79.8) and
  2024-08-05 at 10:00 (51.1 vs ES 47.5) — are the rumour spike and the carry unwind, correlations
  0.996 and 0.993. Median session max step is 7.3 points and only 18 of 1,014 sessions exceed 30.
  Internal consistency worth noting: 2025-04-09 and 2025-04-10 are also the two **zero-candidate**
  sessions, because options were priced out of the $2,000 ticket cap on exactly those days.
- **The `| tail` trap fired a fourth time in this job and the receipt caught it again.** The first
  rebuild launch reported **exit code 0 while the process had crashed** on `ModuleNotFoundError` —
  running a script sets `sys.path[0]` to the script's directory, not the working directory. Nothing
  was built and no receipt was written, which is how it was caught. Exit codes remain worthless here.
- **The rebuild landed and the verification is stronger than the plan called for.** 1,014/1,014
  sessions, gate PASS, 0 failed. The per-era base rates come back **byte-for-byte identical** to the
  ES-tape build — owned `0.31798940639675255`, backfill `0.30587513284382517`, to seventeen
  significant figures — which is the labels themselves confirming they did not move. A 41-session
  sample across the whole span then compares every frame: **`ladder`, `candidates` and `atlas`
  identical, `candles` and `minutes` changed, `tape_source` = `spx_parity_spot` everywhere. 0
  mismatches.** The adapter's exit-law equality re-runs on the new corpus at max |diff| 0.000000000.
- **The era probe was re-run against the rebuilt corpus, and its chain half is now a control.**
  Because the ladder is identical, the chain-field numbers must reproduce exactly — and they do:
  dispersion-only **0.4548** and chain-six **0.4813**, unchanged to four decimals. Only the tape
  probes move, and barely: review tape 0.4133 → **0.4170**, member tape 0.4135 → **0.4175**, member
  state 0.4319 → **0.4322**. All far below the ≈0.55 bar D7 will be set at. Swapping the tape cost
  nothing in era-blindness.
- **The superseded ES-tape corpus is refused mechanically rather than renamed.** Moving data is a
  tier-1 decision, and a naming convention is defeated by one mistyped path anyway — so the adapter
  now refuses any corpus whose candles lack `tape_source = spx_parity_spot`, and refuses an
  **untagged** corpus by name, because "no stamp" is exactly the state that means ES. The old corpus
  is still on disk and still builds perfectly ordinary-looking episodes, which is why the check had to
  be mechanical. **Recommended to the owner: rename or remove
  `lifecycle_corpus_2022-06-01_2026-07-31` once satisfied.**
- **1,049 tests green, `check_project.py` green.** No fit, no spend, no vendor contact, no pinned file
  edited. Boundary 3 still stands: Phase 4a awaits owner confirmation.

### PHASE 4A RUN — PROCEED by the rule, referred to Fable as ambiguous, 2026-08-19.

- **Owner authorized the run; the declaration was written, hashed and verified first.**
  [`PHASE_4A_DECLARATION_V1.json`](PHASE_4A_DECLARATION_V1.json) (`f6894fb8…`) pins six sources and
  every threshold. It **refused to run once** — the reseal guard caught a docstring change mid-build
  — and was regenerated **before any real-feature number existed**, with no threshold moved. That
  sequence is the point of the guard.
- **The declared result: lift +8.65pp on a 30.70% base, at 2 selections/session over 324 scored
  sessions.** Verdict statistic +13.33pp against a +4.0pp bar; plant recovered at **+19.98pp of
  +20.0pp planted**. The mechanical verdict is **PROCEED** and is published as the rule requires.
- **It is not being acted on.** Memo §5 routes "plant recovered while real features behave oddly"
  to Fable and bars the producing session from interpreting it. Three oddities, all measured:
- **The null clears the declared bar 35% of the time.** Twenty within-session label permutations:
  null lift mean −0.24pp, sd 1.60pp, max +2.17pp; **0 of 20 reach the observed +8.65pp** (the point
  estimate is +5.6 null SDs out, so the effect is not a harness artefact) — but **7 of 20 nulls clear
  the +4.0pp bar on the upper bound**, largest +5.96pp. At 648 selections the standard error on
  precision is ~1.8pp, so the upper bound sits ~3.5pp above the point estimate before any signal
  exists. The bar was declared exactly as the design specified; applying an effect-size bar to an
  upper bound is what makes it weak, and that was not visible until it was measured.
- **The lift does not come from the declared hypothesis.** Per-group ablation: chain internals
  **+1.86pp alone**, per-contract chain **+0.16pp alone** (+2.62 and +2.47pp marginal) — the entire
  reason this design exists contributes least. The tape group carries it, and does so **anti-
  predictively alone (−9.41pp) while contributing +9.41pp in combination**, a sign reversal that is
  not the signature of a stable linear signal. Removing the two ordering fields **improves** the
  result, +8.65 → +10.04pp.
- **The one stable coefficient is `realised_vol_15m`** — same sign, same magnitude, an order above
  its neighbours in all four folds (+0.115 to +0.164) while the other three tape channels sit near
  zero and wander in sign. Whether a stable positive loading on realised volatility, scored against
  a **percentage-move** bracket label, is the magnitude rediscovery row 332 priced at thirty cents
  and §5.3 names as the losing failure mode — **that is the judgement §5 reserves, and this session
  does not make it.**
- **Fold lifts increase monotonically in chronology: +1.70, +9.82, +9.84, +13.22pp.** Recorded, not
  explained.
- **1,068 tests green, `check_project.py` green.** Alpha ledger charged for the count, with its
  accuracy columns null by declaration: 4a measures a precision lift at a fixed operating rate, not
  the directional accuracy the ledger's bar is denominated in, and writing one under the other's
  heading is the reporting-layer defect §7 warns about.
- **STOP under §5.** Finding:
  [`PHASE_4A_FEATURE_INFORMATION_2026_08_19.md`](../../research/findings/PHASE_4A_FEATURE_INFORMATION_2026_08_19.md).
  No fit, no spend, no vendor contact, no promotion, no pinned file edited.

### The FOMC list arrives, is verified three ways, and the day-type classifier lands, 2026-08-19.

- **Owner supplied 34 FOMC announcement dates** (statement days, not minutes-release days) from the
  Federal Reserve's published calendars, closing the one Phase-5 prerequisite this job had flagged.
  Recorded with their source in [`session_calendar.py`](../../research/session_calendar.py).
- **Verified rather than transcribed on trust, three independent ways.** (1) **Shape**: eight
  meetings a year, 5 in 2022 from June and 5 in 2026 through July — a dropped date would break the
  count. (2) **Weekday**: every date is a weekday and exactly one is not a Wednesday, **2024-11-07**,
  which is *correct* — the November 2024 meeting moved to the 6th–7th around the US general election
  on the 5th. That exception is pinned by a test so a later tidy-up cannot "fix" it. (3) **The
  corpus's own tape agrees with the list**: the 33 dates present show **1.91x** the median one-minute
  maximum step (13.74 against 7.18 index points) and a median session range of **65.62 against
  47.23**. The owner's reason for sitting out FOMC days is now confirmed with a number, from data that
  knows nothing about the Fed's calendar.
- **One FOMC day is missing from the corpus, and not on purpose: 2025-07-30**, excluded by the clock
  gate for three absent interior minutes (11:20–11:22). The all-sessions figures are therefore
  already short one FOMC day before anything is excluded deliberately. `coverage()` reports absences
  rather than returning a quiet zero, and a test pins that.
- **The rest of the day types need no data at all, which is the distinction that matters.** OPEX,
  quarterly OPEX, month end, last Friday, month and weekday are arithmetic on the session date — no
  vendor, no purchase, no admission. Only the *economic release* calendar was ever the blocked thing.
  If these were wanted as features the blocker would be the parameter contract, not availability, and
  the module says so in its own docstring while forbidding itself as a feature source.
- **Measured signatures across the 1,014-session corpus**, reported because they are cheap and they
  bear on what Phase 5 will find: FOMC 33 sessions (1.91x step, range 65.6); **quarterly OPEX 17
  sessions (1.07x step, range 66.1 — the largest range effect after FOMC)**; month end 50 (1.14x,
  56.3); last Friday 50 (1.06x, 53.7); and **monthly OPEX 50 sessions at 0.95x — no step elevation at
  all**, which is worth knowing before anyone assumes MOPEX is a volatility day in this corpus.
- **Diagnostic only, and structurally so.** The module is barred from the tensorizer, the adapter,
  the probe and the architecture; the event-calendar ruling is unchanged. Receipt
  `session_calendar_coverage.json`. 1,080 tests green, `check_project.py` green.

### The Phase 4a adjudication brief is written for a cold read, 2026-08-19.

- **The brief is §8 of the Phase 4a finding**, not a second document — the evidence and the brief
  travel together and there is no new source of truth.
- **The decision is stated as four mutually exclusive outcomes**, not an open question: (A) tradeable
  signal, fit warranted; (B) rediscovery of already-priced magnitude; (C) artifact of the bar, the
  feature instability or the probe's construction; (D) inconclusive, a different test required. Fable
  selects one or says none fits and names what does.
- **Both readings are argued at equal weight and equal quality (§8.5).** The case for real signal
  rests on near-exact plant recovery, an effect 5.6 null SDs outside its own permutation null, a
  25-parameter probe against 1.04M out-of-fold actions with no threshold search, sign-stable folds,
  and the point that a conditionally-informative family can contribute more at the margin than alone —
  which is what the design claims chain state does. The case for artifact or rediscovery rests on a
  bar 7 of 20 nulls clear, the designed families contributing least, the tape's solo-versus-marginal
  sign reversal with V5's autopsy as local precedent, and a volatility-loaded coefficient scored
  against a percentage-move label that is mechanically easier to reach on cheaper contracts.
- **The evidence sections were neutralised to match.** §§1–7 predated the anti-bias requirement and
  carried characterisations the brief forbids — headings reading "Oddity one/two/three", "not what a
  clean result looks like", "not the signature of a stable linear signal". Every number is unchanged;
  the framing is now flat. Leaving that language above a brief written for a cold read would have
  defeated the brief.
- **Discriminating evidence is listed as questions, not a plan (§8.6)** — within-premium-bucket
  ranking, removing `realised_vol_15m` specifically, a dollar-denominated target, other operating
  rates, reproducibility of the sign reversal under different folds/seeds, whether the chronological
  gradient tracks the already-measured ρ=+0.355 base-rate/volatility relationship, and a matched
  control inside volatility strata. Fable may want different ones and the brief says so.
- **The governance question is put explicitly (§8.7).** The +4.0pp bar was applied to an upper bound
  and the measured null clears it 7 of 20 times. The brief states the only two honest options —
  accept the result under the bar as declared, or declare the test inconclusive and re-run under a
  properly constructed bar with this result treated as spent — and states plainly that retroactively
  re-setting the bar so this result passes *or* fails is not available, **because the producing
  session can see which way it would go**.
- **The producing session's prior is disclosed and quarantined in §8.10, placed last.** It leaned
  toward B/C; the brief records that the lean rests on an association this run did not measure, that
  the +8.65pp point estimate has no explanation under B that this session can offer, and that this
  session chose which three diagnostics to run — which shapes what the evidence section contains.
- **One instruction was not followed, because it had gone stale.** The brief was asked to record the
  ~33 FOMC dates as still owed. They are not: the owner supplied and verified **34** dates earlier the
  same day. §8.9 records the true state — every Phase 5 day-type slice is computable now — plus the
  caveat that **2025-07-30 is an FOMC day the clock gate already excluded** from the corpus. Writing
  a known-false prerequisite into a governance brief would have misinformed the adjudicator.
- **1,080 tests green, `check_project.py` green.** No fit, no spend, no vendor contact, and no pinned
  or signed file edited.

### ADJUDICATION DELIVERED — outcome D; the discriminating test is specified as Phase 4b, 2026-08-19.

Fable read the referral cold under memo §3.2. Ruling:
[`PHASE_4A_ADJUDICATION_2026_08_19.md`](../../research/findings/PHASE_4A_ADJUDICATION_2026_08_19.md).
No new measurement was run; everything rests on the receipts, the pinned code, and arithmetic.

- **§8.1 answer: outcome D.** The +8.65pp is a real measurement of *something* — the plant recovered,
  0/20 nulls, +5.6σ, hashes verified, and the data path structurally cannot see dollars — but the run
  cannot distinguish the design's **ordering** hypothesis from the label's own **mechanical**
  sensitivity to volatility and leverage. The label factors as P(win) = P(path resolves) ×
  P(gain-first | resolved), and its zero pools loss-first with never-resolved, so selecting toward
  high vol and high leverage raises the hit rate with no ordering skill at all — toward the no-skill
  barrier ratio 30/80 = **37.5%**. The observed selected precision is **39.35%**. The decisive
  quantity, P(gain-first | resolved) for the selected set, was never computed and is one label-side
  pass away. Declaring A on a bar 7/20 nulls clear, or B on a pattern-match to row 332 without that
  pass, would each be the memo's named failure mode.
- **§8.7 answer: option 1.** The mechanical PROCEED stands as published under the bar as declared —
  nothing rewritten — and **does not by itself authorize the fit**; the memo routed the action
  decision here, and the action is D. Option 2 (re-run under a recalibrated bar, result spent) is
  rejected as motion-not-measurement: the existing null already shows any sane recalibrated bar
  passes, while the actual open question is mechanism. Prospective directive: every future verdict
  statistic is null-calibrated (≥100 draws) **before** running, and one-sided constructions state
  which error they protect. Both design defects corrected here — the upper-bound bar (§5.2) and the
  "path order, not magnitude" overclaim (§5.3) — originate in the learning-content design, which is
  Fable-lineage work, and the adjudication says so.
- **Two structural facts found cold, both binding downstream.** (1) The "alone" ablation rows for
  the three wholly minute-common groups — tape, clock, chain internals — are **tie-break
  composites**. The stable lexsort resolves equal within-minute scores by frame order, and the
  candidates table sorts `(entry_minute, contract_id)` where `contract_id` embeds the strike, so the
  pick is **the two deepest-OTM puts, the cheapest and most leveraged contracts in the band**
  (verified: 2022-06-01 10:00 takes −22.2 and −17.2 points at $880/$1,030 where the nearest-ATM
  contract is −2.2 at $1,640). That pick's own hit rate is poor — the tape-alone row implies
  **21.30% precision** against a 30.70% base — so those rows are **depressed by a contract choice
  the fields did not make**. Consequence, and it cuts against B: the hypothesis families' standalone
  value is **unknown, not small**, and the **shrink-ladder order may not be derived from these
  numbers**. The marginal column and the three per-contract groups are clean. (2) The three
  diagnostic receipts have no archived producer script; Phase 4b must archive its wrappers. Also
  verified benign: repeated null values are 1/648 grid quantisation.
- **A correction inside the adjudication is recorded rather than silently edited.** The tie-break
  finding was first written as "the two nearest-ATM eligible calls, the most expensive and
  least-leveraged" — reasoned, not measured, and wrong by exactly a reversal. Checking it against
  the corpus flipped the direction of the bias and therefore its consequence. The ruling did not
  change; a supporting fact did, and §6 of the adjudication shows the correction.
- **Phase 4b is specified for Opus, pending owner confirmation** — training prefix only, label-side
  only, self-hashed, alpha-charged, headline reproduction at +8.649074789891253pp as a precondition.
  D1 (verdict-bearing): decompose the selected set's precision into resolution × ordering; the
  ordering component's null from ≥100 within-session permutations. Pre-declared rule: ≤ null 97.5th
  ⇒ **B**, publish the negative with the mechanism named; > 97.5th and ≥ +2.5pp ⇒ **A**, the fit
  proceeds; between ⇒ **owner decision**. D2: vol/premium-stratified matched control, direction must
  agree with D1 or the phase stops and returns to Fable. D3 (diagnostic only): randomized-tie-break
  ablations, which replace the contaminated "alone" numbers for the shrink order.
- No fit, no spend, no vendor contact, no pinned or signed file edited; the referral finding gained
  only a pointer header. STOP under memo §5.

### PHASE 4B — the lift is ordering, not resolution. Outcome A, 2026-08-19.

- **Owner confirmed; Fable's §5 spec executed as written.** Declaration
  [`PHASE_4B_DECLARATION_V1.json`](PHASE_4B_DECLARATION_V1.json) (`6f404d28…`) hashed first. Both
  preconditions passed: implementation hashes matched, and the Phase-4a headline reproduced
  **bit-for-bit at +8.649074789891253pp**.
- **Verdict A. The lift is 97% ordering: +8.65pp = resolution +0.28pp + ordering +8.37pp.** Selection
  moves P(gain first | resolved) from **0.3259 to 0.4140** while barely touching resolution. The
  ordering null over **120** draws has a 97.5th percentile of +3.79pp and **0 of 120 draws reach the
  observed value**. D2's matched control — within session, entry-ask decile × `realised_vol_15m`
  quintile, 585 strata — leaves **+7.67pp** standing in the same direction.
- **The result refutes the adjudication's quantitative account, and by a hard ceiling rather than a
  statistical margin.** Fable proposed the lift was resolution mechanics, with a no-skill gain-first
  share tending toward 30/80 = 37.5%. Measured: **94.2% of paths already resolve** — 0DTE options
  almost always touch +50% or −30% inside an hour — so even a selector choosing *only* resolving
  paths could add at most **+1.71pp**, against an observed +8.65pp. Verified independently on a
  separate 102-session sample (330,021 actions): P(resolved) 0.9476, P(gain-first | resolved) 0.3258.
  The magnitude channel is saturated and cannot be the lever.
- **This session's own prior was wrong too, and by a second independent measurement.** The referral
  leaned toward magnitude rediscovery because `realised_vol_15m` was the one sign-stable coefficient.
  D3 measures it: **that channel alone is worth −1.07pp**, and the full set with it removed still
  delivers **+6.80pp**. It is conditioning, not the source.
- **The tie-break correction is confirmed quantitatively.** With randomized tie-breaking, the two
  minute-common groups collapse to nothing — **tape −9.41pp → −0.16pp, clock +3.40pp → −0.28pp** —
  exactly as a group that cannot rank contracts should score. Chain internals (+1.97pp) and the
  contract-varying groups are unchanged. Fable's correction was right and its consequence is now
  measured rather than argued.
- **A new open question, and it is the important one.** The ordering component has a **strong
  chronological gradient**: +0.74pp in fold 0, then +9.60, +9.40, **+13.71pp**; by year +2.44 (2022),
  +9.10 (2023), **+18.74 (2024)**. The population gain-first share is flat across folds (0.314–0.338),
  so this is not the label changing — it is the features ranking better in later windows. Regime,
  expanding-window fold size, or something else: **this run does not separate them**, and the fit's
  training prefix ends 2024-01-29.
- **Cost of the next step, measured rather than assumed.** One real episode builds in 0.80s and
  carries 8.4 MB of tensors; one forward+backward is 0.04s. The entry phase as the trainer is written
  today holds **all 405 prefix episodes at once — 3.4 GB of inputs — and runs ~18s per epoch, about
  an hour for the declared 200**, with the autograd graph for all 405 alive before a single backward.
  Feasible, but the nested out-of-fold trajectory generation runs that fit once per inner fold.
- **1,098 tests green, `check_project.py` green.** No economics read, no score block touched, no
  pinned file edited, no spend, no vendor contact.

### ENTRY FIT — the chain reorders the ladder but does not pay for the trade. 2026-08-21.

- **The 2026-08-20 entry fit was void and its numbers are withdrawn.** Its runner passed
  `with_paths=False`, so `_bracket_value` could not price the bracket, every `entry_value_usd` was
  NaN, and `train_entry_phase`'s `entry_action_mask & isfinite` mask dropped **all 3,260 masked
  actions per session** from the loss. Only the WAIT head was fitted. The run "converged" in 0.48
  minutes and reported an in-sample 0.3084 hit rate for a model that had never seen an entry signal.
  The entry-stream table, per-block hit rates and top-2 numbers from that run must not be cited.
- **Two guards, one at each end, and a test apiece.** `build_episode` now raises
  `EpisodeAdapterError` when a **priced** build holds labelled entry actions and can price none of
  them; `train_entry_phase` raises `LifecycleTrainingError` when the episodes handed to it carry no
  finite target at all. The split is deliberate: `with_paths=False` is target-free by construction
  and stays legal for the causality control and the shape contracts, so the adapter cannot tell a
  feature-only caller from a fit — the trainer can, and that is where the intent-aware guard belongs.
- **Cost of the priced path, measured: 1.24 s per episode against 0.80 s unpriced** — 1.6x, not the
  8x a first probe reported. That probe wrapped each `build_episode` in `tracemalloc`, which inflated
  its own measurement; the number above comes from the real 1,014-session run. Recorded because the
  8x figure would have argued against doing this correctly. Sell paths and the held-batch builder are
  dropped after pricing — the exit head is frozen in this phase — which releases ~439 MB.
- **The fit is sound and the ordering claim passes.** 405-session prefix, 118 parameters, seed
  20260821, exit head bitwise frozen, full **200/200 epochs** (the 0.57-minute wall time is
  `candle_prefix="last"`, which the equivalence test licenses). Target coverage **asserted before the
  fit**: 99.83% of feasible actions priced, valued equal to labelled. Against a **minute-matched**
  control — same minute, random contract — the model gains **+$8.43 per held-out entry**, positive in
  **five score blocks of five** and **stepping +$7.25/+$5.88/+$5.16 | +$11.88/+$14.81** — which a later synthesis showed is a decline across the backfill blocks and a level shift at the 2025-08-01 source seam, not the same gradient
  Phase 4b found and could not explain. The member is structurally capable of reordering: its scores
  leave a **41% residual** against an additive minute+slot decomposition, where V5's dead architecture
  scored **2.4e-7**.
- **The entry does not survive, and not for want of position sizing.** −$15.66 per entry over
  126,900 held-out entries, **−$1,987,457**, 31.1% profitable. Survival **30.07%** against the
  pre-committed 45–50% target, below its own minute-matched control (30.25%) and below the held-out
  population (30.87%). Honouring the two-tickets-a-day risk law by taking the first two entries each
  session gives **−$18.65**, worse; the oracle's best two are worth **+$536.25**.
- **Mechanism.** The model is a weak within-minute ranker (Spearman **+0.102**, positive in 58.3% of
  minutes) being used as an absolute value estimator. Its level carries nothing: sorted by predicted
  value, the **top decile predicts +$185 and realises −$25.92** while the **bottom predicts −$684 and
  realises −$15.20**. `select_entries` fires on the level against WAIT's $0 floor, 22.0% of actions
  clear it, so the policy fires 209 times a session.
- **STOP under memo §5, and it is the ambiguity rule that applies, not the "entry survives" row.**
  The ordering question this job existed to answer is answered **yes**; the survival bar fails
  outright. Naming the judgement rather than resolving it: the model is fitted to **unconditional**
  dollar value while the entry decision compares against a **$0 WAIT floor**, so a perfectly
  calibrated model on this corpus would never fire — the population mean is −$28. Whether "never
  fire" is the finding, or the objective should be conditional ordering with a separate gating law,
  is a design judgement. No DO_NOT_RETEST row written: the job is stopped for adjudication, not closed.
- Finding: [`ENTRY_FIT_ORDERING_WITHOUT_SURVIVAL_2026_08_21.md`](../../research/findings/ENTRY_FIT_ORDERING_WITHOUT_SURVIVAL_2026_08_21.md).
  Producer, diagnostic, receipt, log, model and the 191,259-row entry stream archived to
  `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/*_2026_08_21.*` — every wrapper archived,
  per Phase 4b's open item 4.
- **1,108 of 1,109 tests green — the two new guards included. `check_project.py` reports one
  problem, which is also the one test failure, and it is not mine to clear:**
  `HANDOFF_ENTRY_FIT_2026_08_21.md` trips the `BANNED_NAME` rule (`HANDOFF`) that AGENTS.md §4 and
  `check_project.py:49` enforce. The file is untracked, is this session's instruction source, and its
  durable content is now in this entry — reported rather than deleted, per §5's "report conflicts, do
  not silently resolve them". No fit-forbidden block was fitted, no spend, no vendor contact, no
  pinned or signed file edited.

### PHASE 5, SECOND HALF — the frozen-entry exit is a degenerate always-hold. 2026-08-21.

- **Blocker cleared first.** `HANDOFF_ENTRY_FIT_2026_08_21.md` tripped the `BANNED_NAME` rule; its
  durable content was already in this log, so the file was moved out of `v5/` rather than deleted.
  `check_project.py` green, **1,109 of 1,109 tests green**.
- **The exit phase could not run as specified, and the arithmetic says so before any compute.**
  `train_exit_head` visits every held batch every epoch; the entry policy fires 159/session, giving
  **50,318 out-of-fold trajectories** and about **115 hours** for the declared 200 epochs. The cap
  applied is forced rather than chosen: the signed risk law is **two tickets a day**, and a serial bot
  cannot know which two will be best, so it takes the **first two it fires**. 644 trajectories over
  323 sessions, fitted in 0.4 minutes. This reconciles the standing "inference law vs risk law" open
  question **only far enough to have a head to fit** and is not a ruling on it.
- **Result: the learned exit is a degenerate always-hold and does not count as skill.** Oracle
  **+$254.64**; always-cut −$17.19; **bracket −$24.18**; **learned exit −$18.15**; **always-hold
  −$1.74**. The learned rule beats the bracket by **+$6.03** and loses to always-hold by **$16.41**;
  it is identical to always-hold on **79.5%** of trades, sells before forced liquidation only 23.4%
  of the time, and the sales it does make leave it worse than not selling. PLAN phase 5's own guard
  is explicit that the always-hold pattern **is not skill**. Entry parameters frozen and verified
  bitwise; the out-of-fold firewall asserted before and after the cap.
- **A finding inside the exit result that bears back on the entry: holding to the clock beats the
  bracket by $22.44 a trade.** `entry_value_usd` *is* the bracket outcome, so the entry was fitted to
  rank contracts under an exit rule worse than doing nothing — the −30% stop cuts positions that
  recover. It does not rescue the entry (always-hold is still −$1.74 against a +$254.64 oracle), but
  the entry's target and the best fixed exit disagree, which plausibly feeds the entry's central
  defect. **The two tables must not be read against each other:** 644 risk-law-capped out-of-fold
  prefix trades is a different population from the 126,900 held-out entries.
- **NEXT BLOCKER, and it is governance, not code — Phase 5's declaration does not exist.** PLAN
  phase 5 requires that **before reading outcomes** a self-hashed declaration cover the members,
  architecture, chronology, controls, exposure ledger, inference and alpha budget. None is on disk;
  the alpha ledger records exactly two experiments (4a, 4b) and neither is a fit. This log wrote on
  2026-08-16 that the ticket-widening amendment *"counts as declared experiment #1 and must be charged
  when the ledger opens at Phase 5, before any fit. Recorded here so it cannot be quietly skipped."*
  It was skipped — by the void 08-20 fit and again by today's entry and exit fits.
- **Deliberately not cleared by this session.** Writing the declaration now, with the outcomes in
  hand, would not be a preregistration; it is the exact manoeuvre the rule exists to prevent.
  Consequence stated plainly: today's entry and exit fits are **development-grade diagnostics, not
  declared outcome-bearing runs**, and cannot be charged PASS/FAIL against a bar that never existed.
  **Had either been positive it would have been unbankable** — both are negative, which is luck, not
  process. For scale, the ledger's next bar is 0.6527 accuracy against a 0.5799 break-even; measured
  entry survival is 0.3007.
- **Also not done, and named rather than assumed:** the loss-averted / capture-efficiency split is
  **Fable's design** under memo §3.3 and §5. The raw grid above is the input to that design, not a
  substitute. No such statistic was constructed here.
- Finding extended in place with sections 7–11. Producer, receipt, log, model and the 644-row exit
  stream archived to `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/*_2026_08_21.*`.
  No score block fitted, no spend, no vendor contact, no pinned or signed file edited.

### COLD REVIEW OF THE PHASE 5 DILEMMA — ruling received, corrections recorded. 2026-08-22.

- **Owner substituted Codex/Sol for Fable** on the two memo-reserved tasks (§3 items 2 and 3),
  recorded here so a future session does not read memo §5 and conclude the boundary was crossed
  silently. Task 1 (the Phase 5 declaration dilemma) returned; task 2 (the exit-evaluation design)
  is not yet run, and the ruling bears on whether it should be.
- **Ruling, in one line: a Phase 5 document written today cannot preregister or rehabilitate the
  2026-08-21 fits.** It may govern a future exactly-locked same-corpus replication, but that run
  stays development-grade and is never independent confirmation. The reviewer does not recommend
  spending an attempt to produce a declaration-shaped rerun of a strongly failed specification; its
  only defensible purpose would be an engineering reproduction audit.
- **The decisive test was mechanical, not rhetorical:** would a compiler given only pre-2026-08-20
  sources emit exactly one declaration? It would not. Free parameters found include the seed and
  environment (undeclared; the nested exit generator constructs each entry model before its internal
  seed applies), the member (P fitted, **Q never fitted**; 118/120/122-parameter variants all
  allowed), fold count, static-versus-refitted chronology, capacity statement (required by the
  charter, never measured for either fit), the population (the runner's `LIMIT` truncation and its
  catch-all skip), control construction, the stop level (explicitly free at −40% or wider; the −30%
  used is the *label's* bracket, not the simulator backstop), the statistical unit (signed law is
  serial per-session account P&L; the report pools per-entry), and every verdict threshold the design
  had explicitly deferred *to the Phase 5 declaration*.
- **Corrections accepted and recorded in the finding at §12 — five of them are defects in this
  session's own work.** (1) The exit head was trained and scored on the same 644 trajectories; the
  out-of-fold firewall protects the *entry generator* only, so §8 is an **in-sample engineering
  result for the exit head**, which makes the stop stronger but mislabels it. (2) `prefix_statistics.npz`
  lived in a cleaned scratchpad and was **absent at review time — neither fit was reproducible**; it
  has been refitted, verified bitwise-deterministic across two independent fits, and archived as
  `prefix_statistics_2026_08_21.npz`. (3) The ordering diagnostic **runs an undisclosed fit** on 60
  sessions. (4) "There was no bar" was too broad — PLAN phase 5 precommitted the 45–50% survival
  target and the always-hold/always-cut degeneracy rule, and **both fits fail those signed bars**.
  (5) The ledger `next_bar` 0.6527 versus survival 0.3007 comparison mixes directional accuracy with
  label survival and is not a formal gate.
- **A signed-design-versus-code conflict, inherited and reported not resolved.** The design names
  `move_15m_rel` as the momentum channel and excludes `return_1m` as dominated; the fitted member
  reads `return_1m`, and **no 15-minute return feature exists in `CANDLE_FEATURES` at all**, so the
  design's channel was never implementable. The member is commit `25e2e1f7`, pinned by
  `PHASE_4A_DECLARATION_V1.json` and sealed before any fit — so this also passed the Phase 4a
  adjudication. Phase 4b measured the family as near-worthless, so it likely changes no conclusion.
- **The §5 boundary crossing was authorised in conversation on 2026-08-21 and was not written down.**
  The reviewer correctly marked it UNKNOWN from the artifacts. Recorded now; the recording failure
  was real.
- **Status downgrade.** No measured number is withdrawn. The +$8.43 ordering effect is **exploratory,
  causal significance UNKNOWN** — its control, population and inference law were undeclared — and §5's
  "chain internals are not an empty information family" drops from settled to suggestive.
- **Root cause named: opt-in governance.** Declaration and ledger checks live in cooperative
  `v5/ops` wrappers, not at the irreversible point where outcomes are opened. `train_entry_phase` and
  `train_exit_head` accept no declaration or ledger argument, so an ad-hoc script reaches them
  directly. Proposed remedy is an **Outcome Run Gate**: a one-use `DeclaredFitPermit` that the
  trainers and outcome accessors refuse to work without, writing an append-only `STARTED` exposure
  record *before* any outcome column opens, so a crash leaves a spent attempt rather than a silent gap.
- **Open, and all owner decisions — nothing below was actioned:** two `DO_NOT_RETEST` rows (drafted by
  the reviewer, one for entry and one for exit, each with its own reopening condition); an append-only
  **late-exposure reconciliation** of the ticket-widening amendment, the void 08-20 fit, the 08-21
  entry and exit fits and possibly the diagnostic fit — **without** rewriting or backdating the
  existing 4a/4b hash chain; reconciliation of the stale `STATUS.md`, which still omits every 08-21
  event and whose route section says no rung is startable; and the charter's adoption of a protocol
  file whose own header still reads "DRAFT. Nothing here is adopted."
- No fit, no spend, no vendor contact, no pinned or sealed file edited. Checker green.

### PHASE 5 CLOSED, LEDGER RECONCILED, OUTCOME RUN GATE BUILT. 2026-08-22.

All three actions taken on owner decisions of 2026-08-22, following the cold ruling.

- **Closure written.** Two rows added to [`DO_NOT_RETEST.md`](../../research/history/DO_NOT_RETEST.md)
  §4, one for the entry configuration and one for the exit. Each closes **only the exact
  configuration** — not Member Q, not the chain-information hypothesis, not every causal entry
  objective. The entry row's reopening condition is a new causal information source or an
  independently motivated target/gating mechanism addressing the level-versus-ranking failure, frozen
  before outcomes and evaluated on outcome-unseen sessions. The exit row's is stricter and ordered:
  **the entry must come first** — a declared entry stream that independently clears survival, and only
  then predeclared loss-averted / capture-efficiency definitions with duration-matched controls and an
  **outer exit holdout**.
- **`STATUS.md` reconciled** and dated 2026-08-22 — it had stood at 08-15 and omitted every event
  since. The job-46 row now carries the void fit, both 08-21 fits, the cold ruling, the ledger
  reconciliation and the closure; its next-step column names the Outcome Run Gate. **A conflict the
  review raised is recorded rather than rewritten away:** §2's "no rung is startable" governs the G1
  rung chain, and §10's "this page authorizes no training" is a disclaimer that the status page grants
  nothing — neither is a prohibition, and job 46's authority is the signed development charter. Both
  statements stand; the ambiguity is now stated in §2 instead of being resolved by editing either.
- **Alpha ledger reconciled append-only: 2 → 6 experiments.** Charged against a self-hashed
  [`PHASE_5_LATE_EXPOSURE_RECONCILIATION_2026_08_22.json`](PHASE_5_LATE_EXPOSURE_RECONCILIATION_2026_08_22.json)
  (`26096fd5…`) which states in its own text that it is **not a preregistration** — it is a debit note,
  and the entries point at it so the ledger never implies a declaration that did not exist. Ticket
  widening and the void fit are `REFUSED` (an attempt spent, no valid verdict); both 08-21 fits are
  `FAIL` (an attempt spent, precommitted bar missed). **Phase 4a and 4b were verified byte-identical
  after the append** and the chain re-verified on reload. The bar moves **0.6527 → 0.6606** true-needed
  0.6787 → 0.6866, against a 0.75 ceiling — projected before writing, since the append is irreversible.
  The ordering diagnostic's 60-session optimisation is **excluded by owner decision** and disclosed in
  the record rather than charged: it reads no economics, and whether "each fit" reaches a diagnostic
  optimisation is UNKNOWN.
- **THE OUTCOME RUN GATE IS BUILT AND IT HOLDS.**
  [`v5/research/outcome_run_gate.py`](../../research/outcome_run_gate.py) plus 14 tests. Three
  properties carry the weight, each from a measured failure. (1) **The requirement follows the data,
  not a flag.** `SessionEpisode` and `Trajectory` carry `provenance`; `build_episode` stamps `"corpus"`
  and nothing else does. A boolean argument would have rebuilt the same opt-in hole. (2) **The exposure
  is journalled before the fit, not after** — the decision to look is what spends alpha, so a crashed
  run leaves a `STARTED` record that **blocks the next permit until a human classifies it** as
  COMPLETED or ABANDONED. (3) **The permit verifies the declaration at open time**, re-hashing it under
  the repository's own convention and re-hashing every file it pins; a stale pin is refused, because a
  declaration that no longer describes the code is worse than none.
- **One design correction made during the build, recorded because it matters.** The permit was first
  written single-use *per call*. That is wrong: `generate_oof_trajectories` trains one entry model per
  inner fold, and those are fits inside a **single declared experiment**. A per-call permit would have
  priced honest nesting out of existence and taught callers to route around the gate — the exact
  failure mode being closed. The permit is now scoped to the experiment, refuses use **after it is
  resolved**, and journals the fit count, which was itself invisible before.
- **Proved against the real bypass, not only in tests.** A real corpus episode was built
  (2022-06-01, 3,260 feasible actions, targets finite) and the exact `train_entry_phase` call that ran
  undeclared on 2026-08-21 was replayed. It now raises `OutcomeGateError` before any fitting.
- **What the gate cannot do, said plainly in its own docstring:** it cannot make a declaration honest.
  It enforces that one exists, matches the code, and is paid for. Whether its content was chosen before
  the outcomes were known is a governance question no runtime check can answer.
- **1,123 tests green, checker green.** No fit run, no spend, no vendor contact. Phase 4a/4b
  declarations and the semantic freeze untouched.

### THREE OPEN DECISIONS CLOSED. 2026-08-22, on owner delegation.

**1 — Settlement-source law: RATIFIED, and its one binding requirement measured and part-discharged.**
Ruling and evidence: [`SETTLEMENT_SOURCE_LAW_2026_08_22.md`](../../governance/SETTLEMENT_SOURCE_LAW_2026_08_22.md).

- **The decision was not actually unmade.** The learning-content design **§4.3 already states the
  law** — parity spot plus validated cash settlement, settlement source as a first-class per-session
  column, a cash settlement never called a fill, a zero-recovery twin on every terminal-dependent
  number with *any sign difference between twins not bankable*, per-era exit-resolution QC, per-era
  base rates, within-era matched controls. It was written, never ratified, and **its one binding
  requirement — the twin — was never computed**. Ratified verbatim rather than reinvented.
- **The twin is now computed** over 24 sessions spanning both eras (receipt
  `zero_recovery_twin_2026_08_22.json`). **39.67% of exit-matrix cells resolve by the settlement
  branch — and 0.00% of the model's bracket exits do (0 of 4,856).** Settled versus zero-recovery:
  model-fired −$17.33 vs −$17.33; all 77,966 priced actions −$27.18 vs −$27.18; ordering edge +$9.86
  vs +$9.86. **Delta exactly $0.00 on every stream.** The reason is structural, not luck: Member P
  exits at the touch minute or the 60-minute horizon, which sit inside the session where executable
  bids exist; the settlement branch serves late minutes and contracts that stopped quoting.
- **A logical bound that made half the question moot in advance, and should have been noticed
  earlier:** zero recovery replaces a never-negative intrinsic with $0, so it can only move a value
  *down*. **A negative headline cannot flip sign under it** — the −$15.66 entry result was
  settlement-robust before anything was computed. Only a *difference* could move, which is why the
  ordering edge was the number worth measuring.
- **DISCHARGED for Member P** on this corpus at this horizon; future P-family results cite the file
  rather than re-deriving. **NOT DISCHARGED for Member Q**, whose 120-minute horizon lands squarely in
  the settlement-exposed region, nor for any change of horizon, exit law or contract universe. Future
  declarations must state which case they are in; silence on settlement exposure makes a declaration
  incomplete.

**2 — Member Q: NOT FITTED, and deliberately NOT closed.** Status is **PRESERVED, NOT RUN**. No
`DO_NOT_RETEST` row is written, and that restraint is the decision.

- **Four reasons not to spend an experiment on it now.** (a) Its **own preregistered sparsity answers
  it before any fit**: the design states that on the owned year only **0.72%** of contract-actions
  beat waiting, median Q(enter) **−$173** against median Q(wait) **+$1,437**. (b) Member P has now
  measured the same economics unconditionally across **1.3M actions** — population mean **−$28** per
  action — so a calibrated model never fires, which is Q's answer arrived at from the other side.
  (c) The design **already declares "P leads; Q may not be promoted over P on point estimates"**,
  written before either was fitted, so even a strong Q could not overturn P's failure. (d) It costs
  alpha against a bar now at 0.6606, and its twin obligation is **un-discharged**, so it is materially
  more work than P was.
- **Why it is nonetheless preserved rather than closed.** Q is the **single remaining hypothesis whose
  full specification predates every outcome in this corpus.** That makes it the one experiment whose
  declaration could be genuinely *source-compiled* under the discipline the cold ruling demanded —
  which is exactly the scarce asset a `DO_NOT_RETEST` row would burn. **If this corpus is ever
  reopened, Q is the correct first experiment**, and it must run under the Outcome Run Gate with the
  zero-recovery twin as a declared evaluation variant.

**3 — The charter/DRAFT textual conflict: RULED ON, and the file deliberately NOT edited.**
Ruling: [`PROTOCOL_V2_ADOPTION_NOTE_2026_08_22.md`](../../governance/PROTOCOL_V2_ADOPTION_NOTE_2026_08_22.md).

- The charter §3 adopted **A1–A13 revision 2** on 2026-08-15 while the file's own header still read
  *"DRAFT. Nothing here is adopted"* — the repository asserting simultaneously that these amendments
  bound every job-46 fit and that nothing in them was adopted. **The charter governs; the header is
  stale text never re-read after signing.**
- **An in-place header repair was attempted and the project's own guard correctly refused it.** The
  file is pinned by `PREACQUISITION_SEMANTIC_FREEZE_V1.json` (`71463cc0…`), whose `post_contact_rule`
  forbids any listed source changing after the preflight;
  `test_preflight_prices_every_session_schema_before_writing_passing_receipt` failed on
  **`semantic freeze source drift`** and the edit was reverted.
- **The guard is right and the header stays.** That file is pinned because it defines the training
  law that governed a **$19.2450 vendor purchase**, and the freeze exists to prove those semantics are
  byte-identical to what is on disk today. Repairing a sentence nobody acts on, at the cost of the
  evidence that the purchase was governed by what we claim, is a bad trade — and the sanctioned
  alternative, sealing the completed acquisition phase against re-running, is wildly disproportionate
  to a stale header. The standing rule applies unchanged: **a pinned file's hashes are a historical
  record, not a value to regenerate.**
- The governance note is therefore the authoritative adoption record, and the filename's `DRAFT` is
  left alone for the same reason: the charter adopts the file by exact path.
- **This corrects the first version of this entry**, which reported the header as repaired. It was
  not, and could not be.

- 1,123 tests green, checker green. No fit run, no spend, no vendor contact, no signed file's substance
  altered.

### THE SIGNED TWO-SKILL EXIT LAW CANNOT BE BUILT AS WRITTEN. 2026-08-22.

Cold review returned (Codex/Sol, substituted for Fable). Finding:
[`EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md`](../../research/findings/EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md).
**Verdict NO — not as signed. Verified independently here; every checkable number reproduced exactly
and nothing in the review required correction.**

- **The contradiction, and it is a proof rather than an argument.** PLAN phase 5 requires each
  statistic measured **against a duration-matched control** *and* requires **always-cut to post high
  loss-averted, always-hold the reverse**. Always-cut exits at minute 1 on every path, so a
  duration-matched control must also exit at minute 1 and is **identical** — control-relative loss
  averted is **exactly zero**, not high. Always-hold gives the same result at minute 60. Permuting a
  constant changes nothing. Comparing cut against hold *would* give the intended profile but abandons
  duration matching and restores the "credit for holding time rather than deciding" the law forbids
  by name (row 341). **No sample size repairs this.** The coherent reading is that both boundary rules
  report exactly zero incremental timing skill with their raw profiles shown separately — a different
  requirement from the one signed.
- **Nothing reopens.** Both configurations stay closed; the learned exit still loses to always-hold by
  $16.41 on the trajectories that trained it.
- **The archive could not have built the statistics anyway.** Verified: `exit_stream` carries no path,
  no entry midpoint and no ever-reached-+50% field; **only 50 of 644 exit keys join the entry stream**
  (different scoring populations); the exit head has no holdout. And **the first-touch label is not
  the "did develop" population** — it stops scanning at −30%, merging "lost first then recovered" with
  "never developed": **250 of 644 paths ever reached +50% against 201 first-touch winners, so reusing
  the label would misclassify 49 trades.**
- **Two corrections to this project's own exit reporting, both verified.** "Mean hold 60.0 minutes"
  was the *available path length*; **realised holding time is 54.44 minutes**, 498 of 644 running the
  full path. And "sells before forced liquidation 23.4%" counted 151 rows including five that fired on
  the final row, which is forced liquidation, not a decision — **strictly before is 146 of 644, 22.7%**.
  The 79.5% always-hold identity is unchanged and the degeneracy conclusion is slightly strengthened.
- **The baseline question was malformed and is now ruled.** Not one comparator but five, each with a
  distinct role: **always-hold is the economic incumbent** (serial P&L must beat −$1.74/trade), the
  **duration-matched randomized exit** is the attribution control, the **bracket is a
  target-coherence diagnostic only** and may never be the economic comparator, always-cut is the
  degeneracy control, and the oracle is the opportunity denominator and never tradable.
- **Power, stated honestly: for the signed requirement `n` is undefined, and for the present run the
  exit-evaluation sample is zero** because every trajectory trained the head. The geometry is **323
  session clusters, not 644 trades** (effective ≈385). An outer holdout recovering $50 and 10 capture
  points jointly needs ≈**640 sessions**; $25 and 5 points needs ≈**2,560** — **2.5 to 10.2 trading
  years of outer holdout alone**, on top of separate entry-validation and exit-training samples. The
  50,318 uncapped trajectories cannot inflate `n`: they violate the two-ticket policy and are
  session-clustered.
- **Two document conflicts verified and left visible.** (1) `AGENTS.md` §7 says "there is no
  confirmation firewall left" while STATUS records a **signed forward reservation making every session
  from 2026-08-06 confirmation-only**. STATUS wins; the two reconcile (historical holdout spent,
  forward reservation live) but AGENTS' blanket sentence is wrong as written — **and this is
  load-bearing, because that reservation is the only mechanism by which the outer-holdout sessions
  above could ever exist.** (2) **A defect introduced here yesterday:**
  `SETTLEMENT_SOURCE_LAW_2026_08_22.md` repeated design §4.3's example that backfill sessions end at
  **15:58**, which was true of V4 and **superseded by V5** (terminal bar restored on 788 of 794).
  Ratifying "verbatim" carried a stale factual example into a new ruling — the hazard of ratifying by
  reference. Corrected in place; the law itself (*read the grid from the data*) is unchanged and was
  always the operative part.
- **OWNER DECISION REQUIRED, and it is not takeable by any session:** the signed two-skill requirement
  needs a **governance amendment** changing the degeneracy expectation so both boundary rules read as
  exactly zero incremental timing skill. The finding records that the change is necessary; it does not
  make it. Until then there is no constructible exit-evaluation law, and no exit study should be
  designed against the current wording.
- 1,123 tests green, checker green. No fit, no spend, no vendor contact, no pinned file touched.

### AMENDMENT DRAFTED, AGENTS §7 CORRECTED. 2026-08-22.

- **`AGENTS.md` §7 corrected.** It read *"There is no confirmation firewall left"* — true of the spent
  historical holdout, **false of the forward reservation** STATUS records as signed 2026-08-05, which
  reserves every ES and SPXW session from **2026-08-06 onward as confirmation-only**. The distinction
  is load-bearing rather than cosmetic: those reserved sessions are the **only** source of
  outcome-unseen data any future confirmation can draw on, and a session reading the old sentence
  would conclude that route was already closed. `v5/CLAUDE.md` is a symlink, so byte-identity holds
  automatically.
- **Amendment drafted, unsigned:**
  [`EXIT_EVALUATION_LAW_AMENDMENT_2026_08_22.md`](../../governance/EXIT_EVALUATION_LAW_AMENDMENT_2026_08_22.md).
  It strikes exactly one clause — the degeneracy expectation — and replaces it with **both boundary
  rules posting exactly zero duration-adjusted skill on both statistics, their raw profiles reported
  separately**. Everything else in the signed law is preserved: two skills never averaged,
  duration-matched controls, attribution-never-selection, serial P&L as the criterion, "beat holding"
  barred. It then adds what the signed text was silent on: the three-state population definition
  (**developed / did-not-develop / unknown**, where developed means *ever* reached +50% regardless of
  a prior −30% — not the first-touch label, which would misclassify 49 of 644), the available-gain
  denominator with an explicit zero rule, four numbers never averaged, the five baselines each with
  one role, four chronological session roles with an outer exit holdout, and the settlement obligation
  for any rule that can exit later than the bracket.
- **The effect-size decision is left to the owner because it decides feasibility, and the
  recommendation is the uncomfortable one.** Tier A (**+$25/trade, +5 capture points**) is the
  economically derived floor and needs **~2,560 outer sessions — about 10.2 years**. Tier B (+$50,
  +10 points) needs ~640, about 2.5 years, but is not a better answer to the same question: it is a
  decision to detect only a very large effect, **chosen because it fits the sample**. Tier A is
  recommended, and its honest output is a finding rather than a study: **on this program's data budget,
  exit-timing skill is not measurable.**
- **A structural tension is named rather than left to be rediscovered.** The binding constraint is our
  own **signed two-tickets-a-day risk law** — it is what turned 50,318 raw trajectories into 644. More
  tickets would make the exit question answerable and would breach the risk law. **The risk law should
  win**, but the trade is real and the owner should know it now rather than mid-study.
- Signing makes the law constructible. It does **not** reopen either closed configuration, authorize
  any fit, create a holdout, or imply an exit study is feasible.
- 1,123 tests green, checker green. No fit, no spend, no vendor contact, no pinned file touched.

### REGIME VERSUS FOLD SIZE CLOSED AS NOT IDENTIFIED; ROUTE CENSUS COMPLETE. 2026-08-22.

- **A non-fit precommit was frozen before the dedicated synthesis:**
  [`REGIME_FOLD_DIAGNOSTIC_PRECOMMIT_2026_08_22.json`](REGIME_FOLD_DIAGNOSTIC_PRECOMMIT_2026_08_22.json),
  SHA-256 `0aa07718751b34f6b339df2ccfe178d0bff4ad8bc0ae1a4477130f596539c1d7`.
  It is not a fit declaration and has no permit authority. The protected alpha ledger was not edited.
- **The two published gradients are different.** Phase 4b refits on 81/162/243/324 sessions while
  moving its holdout forward, so fit size and chronology have rank correlation **1.000** and are not
  separable. The Phase-5 entry fit uses one unchanged 118-parameter model across all five score
  blocks: earlier dollar-ordering edge averages **+$6.0943**, recent **+$13.3441**, a material
  **+$7.2498/entry** contrast. Expanding fit size is therefore excluded for that fixed-model contrast.
- **Market regime versus source era is still not identified.** The recent score indicator is exactly
  the owned-data indicator (`0 0 0 1 1`) and there are zero same-session dual-source observations.
  The largest move lands at the 2025-08-01 seam: +$6.7175/entry in dollars but only +0.8094pp on the
  binary ordering label. Dollar edges decline 7.25→5.88→5.16 through backfill, jump to 11.88 at the
  seam, then reach 14.81. The published 3.03/3.64/3.03/2.25/1.84% spread-to-premium profile leaves
  liquidity as the strongest artifact alternative. Settlement value itself is excluded by the
  existing exact-$0 Member-P twin; ladder width and coverage are stable.
- **Receipt synthesis only — no new outcome statistic.** The exact attempt-001 wrapper stopped on a
  Python boolean typo before output; both it and its failure receipt are preserved. Attempt 002 fixed
  only the boolean aliases and wrote
  `regime_fold_receipt_synthesis_2026_08_22_attempt002.json`, self-hash
  `e67da9bd3e223f352eb9724882ab58963fd50c91515027d4ec2e0102ecaf29ab`, under
  `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/`. It opened published JSON receipts
  only: no corpus table, reserved session, model target, fit, declaration, permit or ledger write.
- **Owner-facing ruling and route census:**
  [`REGIME_VS_FOLD_SIZE_AND_ROUTE_CENSUS_2026_08_22.md`](../../research/findings/REGIME_VS_FOLD_SIZE_AND_ROUTE_CENSUS_2026_08_22.md).
  Recommendation: `STOP_CURRENT_CORPUS_FITS`; preserve Member Q without running it and preserve the
  forward confirmation reserve. If the terminal branch is rejected, the only retained design idea
  is executable clock-hold-relative contract ranking with a separately frozen entry gate. The honest
  power anchors are ~640 outer sessions for a very large +$50/trade effect and ~2,560 for the
  economically derived +$25/trade floor; the target-delta planning proxy is ~1,132 sessions at 80%
  or ~1,568 at 90%. Recent-era fitting remains blocked until same-date source evidence breaks the
  source/time alias. Member Q remains preserved, its 120-minute settlement twin un-discharged.
- **Verification:** `check_project.py` green; outcome-run gate **14 passed**; full V5 suite
  **1,123 passed, 105 warnings, exit 0 in 64.50s**. No fit, spend, vendor/broker contact, download,
  paper/live order, protected-file edit or `DO_NOT_RETEST` change.

### THE EXIT LAW IS AMENDED AND SIGNED, TIER A. 2026-08-22.

Two of the three parked rulings are taken; the third waits on the corpus audit.

- **SIGNED: [`EXIT_EVALUATION_LAW_AMENDMENT_2026_08_22.md`](../../governance/EXIT_EVALUATION_LAW_AMENDMENT_2026_08_22.md),
  owner signature 2026-08-22, TIER A selected.** The two-skill exit law is now constructible. It
  strikes exactly one clause — the degeneracy expectation — and replaces it with **both boundary rules
  posting exactly zero duration-adjusted skill on both statistics, raw profiles reported separately**.
  Everything else is preserved: two skills never averaged, duration-matched controls,
  attribution-never-selection, serial P&L as the criterion, "beat holding" barred.
- **The file was renamed to drop `DRAFT` from its name**, because nothing pins it and this repository
  has already paid once for a signed document whose filename and header both said "draft" — the
  protocol-v2 amendments, whose name could not be repaired because the semantic freeze pinned them.
  Not repeating that.
- **`PLAN.md` phase 5 now carries the supersession inline**, with the original clause struck through
  rather than deleted, so the superseded wording stays readable and a future session cannot design an
  exit evaluation against it without meeting the correction first.
- **TIER A was chosen knowing its consequence, and the consequence is a finding rather than a study:
  on this program's data budget, exit-timing skill is not measurable.** Tier A's +$25/trade and +5
  capture points need roughly **2,560 outer-holdout sessions — about 10.2 trading years** — on top of
  separate entry-validation and exit-training samples, at ~252 reserved sessions a year. Tier B
  (+$50, +10 points) was reachable at ~640 sessions and was **refused**: it is not a better answer to
  the same question, it is a decision to detect only a large effect, chosen because it fits the
  sample. **Shrinking the bar to fit the data is the pattern this project forbids.**
- **Recorded so no future session reopens it out of inconvenience:** moving to Tier B or any weaker
  effect size requires a **fresh owner signature that explicitly acknowledges** it is choosing to
  detect only a large effect. Tier A proving inconvenient is not a reason.
- **The exit-skill question is parked on evidence, not abandoned in confusion.** The law works; the
  sample to satisfy it does not exist and will not for years.
- **Ruling on `STOP_CURRENT_CORPUS_FITS` is deferred** pending the corpus audit now running, which
  judges whether the substrate supports any hypothesis at all. Deciding to stop before knowing whether
  the data is even usable would settle the question for the wrong reason.
- No fit, no spend, no vendor contact, no pinned or sealed file touched. Checker green.

### TWO-ERA CORPUS AUDIT RETURNS NOT-USABLE; PART 2 NOT REACHED. 2026-08-22.

- **Owner-facing finding:**
  [`TWO_ERA_SPXW_CORPUS_AUDIT_2026_08_22.md`](../../research/findings/TWO_ERA_SPXW_CORPUS_AUDIT_2026_08_22.md).
  The recommendation is not adopted here; `STATUS.md`, the alpha ledger, declarations, signed laws,
  and `DO_NOT_RETEST.md` remain unchanged pending the owner's ruling.
- **The standing broad `ONE POPULATION` claim does not survive for target or dollar-economic
  inference.** The replacement audit finding is narrower than `TWO POPULATIONS`: source versus
  market regime is **NOT IDENTIFIED**. Backfill ends 2025-07-31, owned starts 2025-08-01, and same-date
  overlap across those acquisition laws is zero. The corrected 11-field minute-state probe's
  direction-specific AUC 0.4322 reverses to 0.5678, but its 45-session holdout, correlated minute-row
  scoring, missing cluster interval, and omitted contract inputs leave separability inconclusive—not
  proof that the full policy-input target law is exchangeable. Treat the eras as
  separate/non-poolable strata unless paired source evidence exists.
- **The current corpus fails a new structural gate.** The archived exhaustive raw-book attempt 002
  scanned **1,014 sessions, 167,097,488 in-clock quote rows, and 394,446 adjacent whole-book
  comparisons**. Gate `FAIL_INTERIOR_FULL_BOOK_FREEZES_PRESENT` found exactly three included
  backfill sessions: 2023-06-26 (10:29–10:30, 340 contracts), 2023-10-19 (12:16–12:18, 308), and
  2023-10-25 (10:17–10:20 and 10:22–10:39, 322). Four interior runs contain **23 duplicate minute
  transitions**, change before and after, match acquisition hashes, and persist through repaired
  normalized quotes, current ladder/minute summaries, and parity tape. Producer SHA-256
  `d2d9cd111079fea320df70530aed04a7779b7fbd6c5c5d5a5c6b44ab41a93a23`; receipt self-hash
  `a9d3acd5dd53570b3bd40a3129a77a19c5ad514e8a69ffa2b310df13b74b9211`.
- **The failed producer path is preserved, not hidden.** Attempt 001 stopped nonzero with `KeyError:
  ts_recv` because Pandas promoted the owned clock to an index. Its byte-identical archived producer
  SHA is `e94350ac3b7997bcd0809014f6121cb86d4010227405b390bdb12a5c0481de13`; failure-receipt self-hash is
  `4b6f054a61ecd8ebd03e43f4cc128406e6760662edf83de13edc8118310c243e`. Attempt 002 changed only the
  Arrow extraction needed to keep `ts_recv` a column.
- **The prior repairs otherwise held.** All 1,014 current candle/tape sessions carry 390 rows on the
  09:30–15:59/09:31–16:00 clocks with zero tape/corpus mismatches; 214 current carried-close sessions
  carry exactly one minute; every previously clock-rejected session stays out; and 2022-11-25 is
  absent from every current table and tape. Its raw 474-contract book directly reproduces the known
  13:00–16:00 trailing freeze. Existing liveness code checked only tails, while the pre-fit signature
  included recalculated IV fields whose time drift masked interior frozen quotes.
- **Honest population after exclusion is provisional, not sealed:** **1,011 sessions = 768 backfill +
  243 owned**. Backfill is the largest single-source candidate, with 766 label-bearing and 765
  ESS-tool-eligible sessions. Its measured effective size is **UNKNOWN**. The published pooled ESS
  includes all three failed dates, and the old owned receipt is tied to different geometry: current
  773,105 rows/79,109 session-minutes versus historical 698,231/77,254. The geometry wrapper is
  SHA-256 `3c288b4019f55bc16a54e20aa8645a594880bd0032d30c2a01bdf4de91b107a9`; receipt self-hash
  `25b350dcde3e2c8f727888ea6800cf4d9a328ff6705301dd7c64188b3592a9c3`.
- **No present population certifies the signed effect.** Even the barred pooled counterfactual leaves
  only 607 sessions after a 404-session prefix, below refused Tier B's ~640 outer-session requirement;
  signed Tier A needs ~2,560 outer sessions, and weakening it requires a fresh owner signature. The
  $22.44 target-delta planning proxy needs ~1,132 sessions at 80% or ~1,568 at 90%, both above even
  the barred 1,011 pooled ceiling; the honest largest source stratum is 768 before its 307/461 role
  split. These are power limits, not evidence that a replacement edge is measured absent.
- **Part 2 is not reached.** No replacement target, gate, feature contract, built module, fit
  declaration, or experiment is advanced. Member Q remains `PRESERVED, NOT RUN`; its 120-minute
  zero-recovery twin remains undischarged. Rows 187–188 stay intact.
- **Required change before reconsideration:** exclude the three sessions at the raw clock gate,
  rebuild/re-receipt every corpus table with output hashes and the interior detector, resolve inference
  to one source or acquire paired same-date source evidence, obtain owner authorization for a clean
  target-specific ESS/power exposure, and preserve enough materially unseen sessions for the signed
  economic law.
- **Verification:** `check_project.py` green; full V5 suite **1,123 passed, 105 warnings, exit 0 in
  64.80s**, with unpiped output retained at `/tmp/v5-pytest-two-era-audit-20260822.out`. No fit, alpha
  charge, new outcome statistic, reserved session read, spend, vendor/broker contact, download,
  paper/live order, or protected-file edit.

### CORPUS AUDIT RETURNS `NOT-USABLE`, INDEPENDENTLY VERIFIED; THE FREEZE GATE NOW EXISTS. 2026-08-22.

Audit: [`TWO_ERA_SPXW_CORPUS_AUDIT_2026_08_22.md`](../../research/findings/TWO_ERA_SPXW_CORPUS_AUDIT_2026_08_22.md)
(Codex/Sol). **Verified here with a detector written from scratch rather than by re-running theirs.**

- **THE THREE FROZEN SESSIONS ARE REAL, AND ONE IS BAD.** An independent whole-book digest per minute
  confirms interior freezes in all three: **2023-06-26** (10:29-10:30, 2 min), **2023-10-19**
  (12:16-12:18, 3 min), and **2023-10-25**, which carries **4 minutes at 10:17-10:20 and 18 minutes
  at 10:22-10:39** — twenty-two minutes of fabricated flat book in the middle of a session. All three
  are **inside the built 1,014-session corpus**. Seven control sessions spanning 2022-2026, including
  the narrow-ladder 2022 era and the accused sessions' own immediate neighbours, show **zero repeated
  books, maximum run 1**. The separation is clean, not marginal.
- **They passed every existing gate** — 390 minutes, terminal bar, live two-sided quotes, plausible
  prices — and none sits at a close, so no early-close calendar would have caught them.
- **THE GATE NOW EXISTS AND IS MECHANICAL:**
  [`v5/ops/verify_interior_book_liveness.py`](../../ops/verify_interior_book_liveness.py) plus 15
  tests, 7 of them against the real corpus, pinning the three defective sessions as FAIL and four
  healthy neighbours as PASS. It is **fail-closed**: an unreadable file, a missing book field or an
  empty session is a FAIL, never a skip. The bar is *any* repeated book, justified by the measured
  zero-repeat baseline rather than by preference, and sizes are part of the digest because a pad
  repeats sizes too — which is what made 2022-11-25 legible in the first place.
- **This defect class has now been found twice and both times by hand.** 2022-11-25 was caught in the
  2026-08-18 pre-fit review and correctly excluded, but **the scan was never turned into a gate**, so
  three more walked straight through four months of downstream work. `grep` confirms the builder's
  only occurrence of "frozen" was `@dataclass(frozen=True)`. That is the whole lesson: a one-off scan
  is not a control.
- **Verdict `NOT-USABLE`, and it is carefully scoped rather than sweeping.** The audit states it is a
  *substrate and identification failure*, **not** evidence that every possible SPXW 0DTE edge is
  absent. Its legs: three liveness failures still inside the corpus; source era still perfectly
  confounded with the 2025-08-01 cutoff so the eras cannot support pooled economic inference; no
  effective-size measurement for either cleaned stratum without a newly authorised outcome exposure;
  and **even raw pooled session counts fall below the signed economic certification requirement** —
  1,014 against the ~2,560 outer-holdout sessions Tier A demands, before any partitioning.
- **The audit also unsettles the standing one-population ruling on method, not vibes.** The
  2026-08-18 review's era probe used a **45-session holdout (11 owned, 34 backfill)** while scoring
  AUC on **16,648 correlated minute rows with no session-clustered interval**. That is inconclusive
  rather than confirmatory, and the ruling it supported should not be leaned on.
- **Part 2 was correctly not reached.** No replacement hypothesis was proposed, which is the right
  behaviour when Part 1 returns NOT-USABLE.
- **Verification:** 1,138 tests green (1,123 + 15 new), checker green, ledger untouched at 6
  experiments, `STATUS.md`, `DO_NOT_RETEST.md`, the alpha ledger, every signed and pinned file, and
  the unrelated `pickles-weekly-ranges/` all unmodified. The audit's failed first wrapper attempt is
  preserved rather than deleted.
- **OWNER RULING NOW LIVE: `STOP_CURRENT_CORPUS_FITS` was deferred pending this audit, and the audit
  answers it.** The substrate cannot certify an economic claim, so continuing to fit it would be
  spending alpha against a bar the sample cannot reach. Recommendation: accept the stop. Deciding
  otherwise is the owner's, but it should be a decision to spend on data, not on fits.

### `STOP_CURRENT_CORPUS_FITS` ACCEPTED. And the binding constraint is calendar, not money. 2026-08-22.

- **Owner ruling: STOP ACCEPTED.** No further fits on the 1,014-session corpus. Job 46's Phase 5 ends
  as a documented negative: the entry configuration and the exit configuration are closed by
  `DO_NOT_RETEST` rows, the two-skill exit law is signed at Tier A and is unsatisfiable on this data,
  and the corpus itself is now audited `NOT-USABLE`.
- **A structural fact that reframes what comes next, and it is arithmetic.** The next dataset question
  looked like a budget question. It is not:

  | Quantity | Measured |
  |---|---:|
  | Corpus sessions built | 1,014 |
  | Business days in 2022-06-01 → 2026-07-30 | 1,087 |
  | **Share of every possible session already held** | **93.3%** |
  | Measured cost, cbbo-1m at resolved 0DTE scope | **$0.02424/session** |
  | Charter budget remaining | $36.51 of $75 |
  | More sessions that budget could buy | **~1,506** |
  | Tier A outer-holdout requirement | **~2,560 sessions** |

  **We can afford roughly 1,506 more sessions and they do not exist.** SPX did not have daily 0DTE
  expirations until Cboe completed the Tuesday and Thursday additions in 2022, which is why this
  corpus begins 2022-06-01 rather than earlier. There is no more daily-0DTE history to buy at any
  price. **Money was never the binding constraint; calendar is.**
- **The consequence for direction: the axis must change from more sessions to more information per
  session.** Buying breadth is exhausted. What is untouched is depth — this corpus is 1-minute
  consolidated BBO *quotes only*, so intra-minute price action, actual trade prints, size, and where
  trades land relative to the spread have never been available to any model here. Whether that is a
  real information family or another dead one is exactly what the next investigation must decide
  **before** anything is bought.
- **What remains authorized and what does not:** $36.51 of the $75 charter ceiling is unspent, and it
  is scoped to the Phase-2 backfill. **A different schema, a different resolution, or any live
  subscription is outside it and needs fresh owner authorization with an exact preflight first.**
  Nothing here authorizes a purchase.
- No fit, no spend, no vendor contact. Checker green, 1,138 tests green.

### DEPTH-DATA ACCESS SCOPED. Databento blocked on a credential; Polygon is listing-only. 2026-08-22.

Owner authorised a preflight in conversation. Neither half could be priced, and both reasons are
concrete rather than vague.

- **The Databento preflight CANNOT RUN: there is no `DATABENTO_API_KEY` on this machine.** `.env`
  carries `ANTHROPIC_API_KEY` and five Polygon variables and nothing else; the environment and the
  usual config locations are empty. The `databento` package is installed (0.77.0), so it is purely
  the credential. **No estimate was substituted** — the charter is explicit that an estimate may not
  stand in for a preflight.
- **Two hazards found for whoever does write that preflight.** (1) `download_spxw_history.py` is
  structurally incapable of pricing anything else: `_bounds()` raises `AcquisitionError` on any schema
  outside `("definition", "cbbo-1m")`, so a depth preflight is new code, not a flag. (2) Its cost call
  uses `symbols=["SPXW.OPT"], stype_in="parent"` narrowed only by **time**, not by symbol — the same
  parent scope that produced the **$671.90 false alarm**. That was harmless for `cbbo-1m`; on a
  message-volume-priced schema it could return a wildly inflated number and trigger a false STOP. The
  0DTE ladder must be resolved before pricing.
- **Polygon holds exactly the depth data we want, and we cannot read it.** Listing succeeds across the
  whole `flatfiles` bucket: `us_options_opra/` carries `trades_v1/`, `quotes_v1/`, `minute_aggs_v1/`
  and `day_aggs_v1/`. **Every object read returns HTTP 403** — options, stocks and indices alike, at
  every date tried, on a 1 KB ranged request. So the credentials permit bucket listing and not object
  retrieval, and the subscription is either lapsed, downgraded, or never included flat-file reads.
  **This data is visible, not owned.** No bytes of market data were transferred.
- **The volume census is worth keeping regardless, because it shapes the whole approach:**

  | Dataset | Coverage | Per day (2026) | 1,014 sessions | Feasible in bulk? |
  |---|---|---:|---:|---|
  | `trades_v1` | 2014–2026 | ~57 MB | **~58 GB** | **yes** |
  | `minute_aggs_v1` | 2014–2026 | ~23 MB | ~23 GB | yes |
  | `quotes_v1` | 2022–2026 | **~109 GB** | **~110 TB** | **no** |

- **That table settles the architecture question even though the entitlement failed.** Flat files are
  per-day-all-symbols, so pulling SPXW 0DTE quotes out of `quotes_v1` means moving 110 TB to extract
  perhaps a few GB — the wrong access pattern by four orders of magnitude. **Trade prints are cheap
  and bulk-feasible; tick quotes are not, and must come from a symbol-filtered API rather than flat
  files.** Databento's filtered request is exactly that tool, which is what the original acquisition
  already used.
- Wrappers archived: `polygon_entitlement_scope_2026_08_22.py`,
  `polygon_opra_volume_scope_2026_08_22.py`, `polygon_read_entitlement_probe_2026_08_22.py`.
- **No spend, no purchase, no market data transferred, no vendor account modified.** Listing a
  flat-rate bucket carries no marginal cost.

### DEPTH PREFLIGHT COMPLETE. `ohlcv-1s` is free at 60x the current resolution. 2026-08-22.

Owner supplied the Databento key and authorised the preflight. **Pricing only — every vendor call was
`metadata.get_cost`, `get_record_count`, `symbology.resolve` or `get_dataset_range`. No
`timeseries.get_range`, no bytes of market data, no purchase.** Receipt:
`depth_preflight_receipt_2026_08_22.json`.

- **Attempt 1 is VOID and is archived as such.** It priced **parent scope**, replicating what
  `download_spxw_history.py` appears to request, and its known-answer control **failed at ~20x** the
  receipted `cbbo-1m` cost — the same parent-versus-resolved ratio behind the $671.90 false alarm.
  The control existed precisely to catch that, and it did. Attempt 2 prices the **±25-point corpus
  band** (32–56 contracts/session) as OSI raw symbols verified against the vendor's symbology service,
  and its one-sided control **PASSES**: band `cbbo-1m` is **$0.002497/session** against the receipted
  **$0.024238**, correctly cheaper because the band is strictly narrower than the purchase scope.
- **`$0.00` was interrogated rather than believed.** `get_record_count` separates "free" from "no
  data", and the answer is free: `ohlcv-1s` returns **151,160 records** for the 2023-06-27 band and
  **287,551** for 2025-11-04, at **$0.00**.

  | Schema | Coverage from | Band cost/session | Corpus (1,014) | Note |
  |---|---|---:|---:|---|
  | **`ohlcv-1s`** | **2013-04-01** | **$0.00** | **$0** | **60x current resolution, full history** |
  | `ohlcv-1m` | 2013-04-01 | $0.00 | $0 | free |
  | `cbbo-1m` | 2013-04-01 | $0.002497 | ~$2.53 | what we already have |
  | `cmbp-1` | 2023-03-28 | $0.185–$0.322 | ~$260 | misses the 2022 corpus start |
  | **`trades`** | 2013-04-01 | **$4.63 (pre-2025-08-25)** | **~$3,644** | **free from 2025-08-25** |
  | `tcbbo` | 2023-03-28 | ~$5.50 | ~$5,580 | misses the 2022 start |
  | `cbbo-1s` | 2025-02-20 | — | — | far too late to cover the corpus |

- **The `trades` free boundary is 2025-08-25**, narrowed by probe: $7.16 on 08-15, $10.39 on 08-20,
  **$0.00 from 08-25 onward**. That splits the corpus **787 paid / 227 free**. Buying the paid portion
  costs **~$3,644 against $36.51 remaining on a $75 charter** — two orders of magnitude out of reach,
  so full-history trade prints are not purchasable under any current authorization.
- **The headline is that the best available option costs nothing.** `ohlcv-1s` is **free across the
  entire corpus window** and carries **~10x the records of `ohlcv-1m`** — one-second trade bars where
  every model built here has seen one-minute quote snapshots. It is *trade* information, not quotes,
  so it is sparse by construction and is **not** a replacement for the BBO.

  > **CORRECTION, 2026-08-23.** This entry went on to call `ohlcv-1s` *"the order-flow family this
  > project has never had"*. **That is wrong and the error is mine.** Order flow means *signed* flow —
  > knowing whether a trade was buyer- or seller-initiated — and an OHLCV record cannot supply it.
  > Verified against the record layout: `OHLCVMsg` carries **only** open/high/low/close/volume, with
  > `bid_px`, `ask_px`, `side`, `action` and `flags` all **absent**. `ohlcv-1s` is an *unsigned,
  > aggregated last-trade tape*. It cannot sign a trade, cannot show the touch it printed against, and
  > cannot measure book resilience. A cold review vetoed it for alpha on exactly this ground and the
  > veto is correct. The free-and-60x facts stand; the interpretation placed on them did not.
- **What this does NOT establish.** That the data is free says nothing about whether it carries an
  edge — this project has censused several free information families dead. It also does not authorize
  a download: acquiring `ohlcv-1s` is a new schema outside the charter's `definition`/`cbbo-1m` scope
  and needs owner authorization, though at $0.00 the spend gate is moot and the real costs are
  bandwidth, storage and the parity obligations that follow any new feature source.
- **A consequence for the recent era, worth stating separately:** everything is free from 2025-08-25,
  including `trades` and `tcbbo`. **227 corpus sessions carry free full trade prints**, and the
  reservation bars only 2026-08-06 onward, so a recent-era study has depth data available now at no
  cost.
- Wrappers and receipts archived under `depth_*_2026_08_22.*`, with the void attempt preserved.
- **No purchase, no download, no data transferred, no vendor account modified.** The API key is in
  `.env`, which is git-ignored and was not committed.

### SUB-MINUTE DIRECTION SYNTHESIS — STOP UNDER CURRENT CONSTRAINTS. 2026-08-22.

Finding: [`DATABENTO_SUBMINUTE_DIRECTION_DECISION_2026_08_22.md`](../../research/findings/DATABENTO_SUBMINUTE_DIRECTION_DECISION_2026_08_22.md).

- **Independent data, microstructure, measurability, runtime/parity and adversarial reviews agree on
  STOP.** No candidate survives as an economic or fit route under the current data budget, session
  power, train/live-parity requirement and signed risk law. The strongest scientific idea is strict
  trade-at-touch flow conditioned on post-trade book resilience; it remains a paper question, not a
  runnable bot route.
- **The free schema does not inherit the rich mechanism.** `ohlcv-1s` is vendor-aggregated
  last-trade OHLCV: no individual print order, prior BBO, touch, aggressor or quote response, and no
  executable bid/ask label. The finding vetoes it as an alpha direction because it is the dead
  last-trade/chart/magnitude family at a finer clock. Its five sampled $0 quotes and two positive
  record-count probes are retained; full-corpus $0 remains inferred rather than an all-session quote.
- **The event-data cost was coverage-corrected rather than repeated from the receipt's deliberately
  simple 1,014-session extrapolation.** There are 813 corpus sessions on or after the 2023-03-28
  CMBP/TCBBO coverage start and 586 before the observed recent-free era. The exact paid samples imply
  planning totals of about **$148.60 CMBP** and **$3,224.57 TCBBO**, not exact acquisition quotes;
  both remain above the $36.51 balance and outside its schema authority. The exact all-session sum is
  UNKNOWN until a new owner decision permits the specified metadata-only census.
- **Owned depth is useful only for a kill gate.** The existing 64 selected-symbol CMBP sessions
  (173,470,783 rows, 248 session-symbols) and 175 selected-symbol CBBO-1s sessions may support an
  outcome-blind parser/semantic check. Their selected population cannot support unbiased economics,
  and no such check was run here.
- **Messages do not solve the calendar wall.** Economic inference clusters by session; 644 paths in
  323 clusters had effective size near 385. Entry-specific effect and power remain UNKNOWN. Signed
  Tier A still requires about 2,560 outer sessions (~10.2 years) for exit, so no exit work reopens.
  One future outcome look would charge the ledger 6→7; the current `0.66059463` bar is directional
  geometry only and cannot stand in for option P&L.
- **Runtime remains a hard stop.** The current twin is minute-specific; no CMBP live twin exists;
  Databento live entitlement and price are UNKNOWN; IBKR event equivalence is unproved; and the v5
  simulator still carries a 5% breaker/no mandatory percentage stop against signed 20% and
  -40%-or-wider laws. The finding specifies same-schema no-order parity, symbol/clock/drop/fill
  receipts and law-current risk enforcement before any fit or paper path.
- **Recommended owner decision:** accept `STOP_UNDER_CURRENT_CONSTRAINTS`. If the owner explicitly
  declines, authorize only the outcome-blind semantic gate on already-owned CMBP; a pass is
  `SEMANTICS_PASS_ONLY`, never acquisition or fit authority.
- No fit, outcome statistic, reserved session, vendor call, download, spend, subscription, broker
  connection, paper/live order, protected-file edit, wrapper or receipt was produced by this
  synthesis. Member Q remains **PRESERVED, NOT RUN**.
- **Verification green:** project/repository checker PASS; full v5 suite **1,138 passed** with 105
  existing numerical warnings, process exit 0 in 68.72 seconds.

### SUB-MINUTE DIRECTION DECISION VERIFIED: `STOP_UNDER_CURRENT_CONSTRAINTS`. 2026-08-23.

Finding: [`DATABENTO_SUBMINUTE_DIRECTION_DECISION_2026_08_22.md`](../../research/findings/DATABENTO_SUBMINUTE_DIRECTION_DECISION_2026_08_22.md)
(Codex/Sol). **Verified here. Both of this session's challenges to it failed, and the finding stands.**

- **The `ohlcv-1s` veto is CORRECT and the error it overturns is mine.** Checked against the record
  layout: `OHLCVMsg` carries **only** open/high/low/close/volume — `bid_px`, `ask_px`, `side`,
  `action` and `flags` are all **absent**. So `ohlcv-1s` is an **unsigned aggregated last-trade
  tape**: it cannot sign a trade, cannot show the touch a print landed against, and cannot measure
  book resilience. The 2026-08-22 entry above called it *"the order-flow family this project has
  never had"*, which conflates a trade tape with **signed** flow. Corrected in place. **Free and 60x
  resolution were both true; the interpretation placed on them was not.**
- **"VERIFIED partly owned" is CORRECT and this session's scepticism was misplaced.** Filesystem
  searches under `/Volumes/AR_TRADING_DATA` and `~/.autoresearch-trading` found nothing, and the row
  counts appeared nowhere in the repo — so the claim looked unsupported. **Both checks were looking in
  the wrong place.** The inventory is
  [`v4/audit/databento_protocol101_highres_downloads.jsonl`](../../../v4/audit/databento_protocol101_highres_downloads.jsonl)
  and the totals reproduce **exactly**: `cmbp-1` **64 sessions, 173,470,783 rows, 2024-10-01→2024-12-31**;
  `cbbo-1s` **175 sessions, 18,287,307 rows, 2025-07-01→2026-03-30**. The paths are **repo-relative**
  (`data/raw/audit/protocol101_highres_opra/…`), and **all 239 DBN and all 239 Parquet files are
  present on disk.** Recorded acquisition cost was **$4.80 total**, cheap because it is selected-symbol.
- **The named hypothesis is constructible on data already owned.** `cmbp-1` carries `action`, `price`,
  `size` and the **contemporaneous `bid_px_00`/`ask_px_00`/`bid_sz_00`/`ask_sz_00`**, so a trade row
  can be signed by comparing its print to the touch standing at that instant. That is exactly "strict
  trade-at-touch flow plus book resilience", and it is why OPRA's unpopulated `side=N` is not fatal.
  6,442,354 rows in a single 118.9 MB session file.
- **One refinement rather than a correction.** The finding's CMBP planning figure of **~$148.60** uses
  a **two-sample** mean of $0.253577 taken from this session's own 2023-06-27 and 2024-07-01 probes.
  Four further probes across 2025 return **$0.31–$0.70/session**, so the true mean is higher and the
  paid-partition cost is plausibly **$250–$400** rather than $148.60. The finding already labels the
  figure `INFERRED` with "exact total UNKNOWN", so this sharpens it rather than contradicting it. The
  free boundary measured here for `cmbp-1` is **~2025-09-02**, giving **222 free / 591 paid** of the
  813 sessions inside its 2023-03-28 coverage, with **201 corpus sessions uncoverable** because they
  precede that coverage at all.
- **Verdict accepted: `STOP_UNDER_CURRENT_CONSTRAINTS`.** The route lacks affordable unbiased history,
  live parity, a compliant runtime, and entry-specific power. If the owner rejects the stop, the only
  admissible move is the finding's own: an **outcome-blind semantic gate on the 64 owned CMBP
  sessions** — free, already on disk, no spend, and it tests whether the decoder and the
  trade-at-touch semantics work *before* anyone prices history.
- **Verification:** 1,138 tests green, checker green, ledger untouched at 6 experiments, `STATUS.md`,
  `DO_NOT_RETEST.md` and every signed and pinned file unmodified, `pickles-weekly-ranges/` untouched.
  No spend, no download, no outcome read, no vendor or broker action.

### THE UNTESTED THING IS TESTED. `SEMANTICS_PASS_ONLY` on 64 owned CMBP sessions. 2026-08-23.

Owner authorised "the $4.80 spend". **There was no spend to make** — $4.80 was the cost already paid
months ago in the v4 era for data sitting on disk. This cost **$0**, downloaded nothing, contacted no
vendor, and **read no outcome, so no alpha was charged.**

- **Manifest frozen first**, before any result was seen:
  [`CMBP_SEMANTIC_GATE_MANIFEST_2026_08_23.json`](CMBP_SEMANTIC_GATE_MANIFEST_2026_08_23.json)
  (`55dd5a92…`) pins **64 sessions, 248 session-symbol pairs, 173,470,783 rows**, 2024-10-01 →
  2024-12-31, and names the forbidden tables (labels, P&L, entry/exit value, forward returns).
- **The gate is code with tests, not a one-off script:**
  [`v5/ops/verify_cmbp_touch_semantics.py`](../../ops/verify_cmbp_touch_semantics.py) plus 12 tests.
  The signing law is pinned by test: a print is signed only against the touch standing **strictly
  before** it, never the trade row's own book; **inside-touch prints are retained as ambiguous and
  never guessed**; locked and crossed prior books are excluded and counted; instruments cannot leak
  into each other's touch; an empty or malformed slice fails closed.
- **VERDICT: `SEMANTICS_PASS_ONLY`** over all 64 sessions, **0 failures**, 1,588,281 trades.

  | Quantity | Measured |
  |---|---:|
  | Trades signable by strict prior touch | **874,002 — 55.03%** *(superseded: see 08-23 correction)* |
  | Inside-touch, **ambiguous and not guessed** | 704,734 — 44.37% |
  | Outside touch | 2,259 |
  | Locked / crossed prior book | **0 / 0** |
  | No prior quote | 7,286 |
  | Buyer-initiated : seller-initiated | 424,126 : 449,876 (**ratio 0.943**) |
  | Touch moved on the next event after a trade | 22.93% |
  | Ask rose after a buy / bid fell after a sell | 21.95% / 22.24% |
  | Per-session signed share | min **9.2%**, median 49.7%, max **65.5%** |

- **DBN/Parquet decode identity verified** on the first three sessions: row counts, trade counts and
  trade price sums all agree. The parquet is a faithful decode, not a lossy re-write.
- **The sanity signals are good.** Zero locked and zero crossed prior books across 173M rows; a
  near-balanced buy/sell ratio of 0.943; and a symmetric book response — 21.95% of buys move the ask
  up against 22.24% of sells moving the bid down. A signing bug would almost certainly break that
  symmetry.
- **What this does NOT say, and the caveats are load-bearing.** (1) **It says nothing about edge.**
  The gate reads no outcome by construction; a pass establishes that the mechanism can be *built*.
  (2) **The slice is heavily selected** — 7 symbols and ~31 chosen trades per session, picked around a
  prior route — so **event prevalence on an unbiased full band remains UNKNOWN**. (3) **The
  construction discards 45% of the tape**, and (4) the retained share is **unstable session to
  session, 9.2% to 65.5%**, so a strategy resting on it would have wildly varying observation counts.
- **Consequence for the standing STOP: it is unchanged.** The gate has done exactly its stated job —
  it removes "we cannot even parse this" from the list of unknowns, and leaves every economic question
  where it was. The route still lacks affordable unbiased history, live parity, a compliant runtime,
  and entry-specific power.
- Producer and log archived as `cmbp_semantic_gate_producer_2026_08_23.py` /
  `cmbp_semantic_gate_2026_08_23.{json,log}`.

### 229 SESSIONS OF UNBIASED BROAD-BAND CMBP-1 ARE FREE. 2026-08-23.

Measured while scoping the next research session. Metadata only; nothing downloaded, no spend.

- **`cmbp-1` is free from 2025-08-21**, giving **229 corpus sessions (2025-08-21 → 2026-07-30)**, all
  usable — the confirmation reservation bars 2026-08-06 onward and the corpus ends 2026-07-30.
- **This is not the selected slice.** The 64 owned sessions carry 7 symbols apiece, chosen around a
  prior route. These 229 would be requested at the **full ±25-point band** and are unbiased by
  construction. Uniform provenance across all 229, so **no internal era seam** of the kind that made
  the two-era corpus `NOT-USABLE`.
- **The cost is engineering, not money.** One free session at broad band is **130,310,192 rows across
  52 contracts** — twenty times the owned slice's 6.4M rows for 7 contracts. Across 229 sessions that
  projects to **~29.8 billion rows and roughly 0.55 TB** of parquet at the owned slice's observed
  density. That is a serious build, not a casual download.
- **What it would and would not fix.** It removes the selection bias the semantic gate had to caveat,
  and it is the only unbiased signed-flow sample obtainable at zero cost. It does **not** fix
  chronological breadth: 229 sessions is one era and roughly eleven months, so any chronological split
  is shallow, and it remains far below the ~2,560 sessions Tier A certification demands.
- **Still requires owner authorization.** `cmbp-1` is outside the charter's `definition`/`cbbo-1m`
  scope. The spend gate is moot at $0; the storage, decode throughput and train/live parity
  obligations are not.
- Wrapper archived as `free_cmbp_scope_2026_08_23.py`.

### SIGNED FLOW MOVES US TOWARD A VALID EXPERIMENT, NOT A PROFITABLE BOT. 2026-08-23.

Finding:
[`SIGNED_ORDER_FLOW_BOT_DECISION_2026_08_23.md`](../../research/findings/SIGNED_ORDER_FLOW_BOT_DECISION_2026_08_23.md).
**Decision: NO under current constraints; retain `STOP_UNDER_CURRENT_CONSTRAINTS`.** The owned flow
contains real short-lived structure and the free broad build is computationally possible, but 229
session clusters are not demonstrably powered for the economic gate, causal order is unresolved for
a quarter of the pinned signs, and no CMBP live twin or law-current runtime exists. No acquisition,
outcome, fit, alpha charge, vendor/broker call, protected-file edit or order occurred.

- **Compute is feasible only as a chunked external-SSD build.** Five disjoint size strata measured
  DBN scan at **4.547–4.931M rows/s**, DBN→Snappy Parquet at **1.040–1.115M rows/s**, and the
  outcome-blind feature pass at **5.797–7.562M rows/s**. Extrapolated build plus one feature pass is
  **8.52–9.39 hours**; these are min/max sensitivities, not confidence intervals. The conservative
  footprint is **1.071 TB**, leaving about **879.7 GB** on the external disk; the internal disk had
  **11.72 GB** free and is unusable. The 130.3M-row broad session is 11.9× the largest probe input,
  so full-scale rate and memory remain **UNKNOWN**. Receipt:
  `cmbp_compute_feasibility_probe_2026_08_23_attempt003.json` (`a97e4a99…`). Attempts 001 and 002,
  their receipts, logs and partial outputs remain preserved.
- **The selected owned slice is bursty and persistent at the event scale, not at a useful economic
  horizon yet.** Across the exact 64-session census there were **1,588,281 trades** and **874,002**
  pinned prior-touch signs. Deterministic 10,000-resample whole-session intervals give adjacent
  same-direction probability **68.19% [67.09%, 69.22%]** versus a **50.16% [50.09%, 50.25%]**
  independence baseline, 1-minute net-contract ACF **0.0238 [0.0011, 0.0466]**, and 5/15-minute ACFs
  whose intervals include zero. Within-30-minute Fano is **17.16 [13.73, 20.84]**. Median prior spread
  is **181.7 bps [164.1, 200.2]** and the typical print is one contract. No future value or outcome
  was read; this is characterization, not alpha.
- **The earlier semantic pass needs a material correction.** Exactly **226,288 / 874,002 (25.89%)**
  pinned signs use a prior row with the same `ts_recv` and `ts_event`; **199,728** are cross-publisher
  non-trade priors and **26,560** are prior trades. Neither stored Parquet nor independently inspected
  DBN records expose `sequence`, and the earlier identity check compared counts and price sum, not
  event order. Causal chronology for tied clocks is therefore **UNKNOWN**. Requiring
  `prior_ts_recv < trade_ts_recv` retains **647,714** signs: same-direction excess remains
  **+15.10 pp [+14.09, +16.05]**, within-bin Fano **13.11 [10.50, 15.91]**, and 1-minute ACF
  **0.0321 [0.0101, 0.0550]**; strict contract imbalance **+1.88% [−0.03%, +3.99%]** does not exclude
  zero. Receipt verdict:
  `OUTCOME_BLIND_CHARACTERIZATION_ONLY_CAUSAL_ORDER_UNVERIFIED` (`7d715901…`).
- **Selection remains terminal for prevalence.** These files contain up to seven symbols chosen
  around roughly 31 prior-route trades/session. Session-bootstrap intervals quantify variability
  within those frozen bytes; they cannot establish market-wide event supply, trigger occupancy or
  broad-band economics.
- **229 is economically weak and direct-P&L power is UNKNOWN.** With six prior experiments, a
  proposed seventh uses one-sided `alpha=.05/7`. The optimistic independent-session sensitivity at
  n=229 detects only annualised Sharpe **3.453** with 80% power; Sharpe 1.0/1.5/2.0/2.5/3.0 require
  **2,731/1,214/683/437/304 sessions**. In the existing two-point payoff analogue, the exact
  66.0595% accuracy bar has only **49.05%** power at 229, while 68.6595% reaches **79.37%**. Actual
  trigger occupancy, session-P&L variance and paired-control covariance are unmeasured. Two tickets
  in one day remain one cluster. Power receipt: `signed_flow_power_scope_2026_08_23_attempt002.json`
  (`448b894b…`).
- **One no-fit experiment is specified but not adopted or run.** It freezes strict-touch
  premium-notional flow conditioned on causal non-replenishment, a five-second window, same-schema
  p99 end-to-end latency, first-two-trigger policy, long nearest-eligible OTM call/put, ask-in and
  causal bid-out at −40% or 60 minutes, $2,000+fees cap, 20% breaker, serial $10,000 account and 50%
  floor. All 229 sessions, including zero-trigger days, are inference units. The primary gate is the
  minimum lower bound of absolute net session P&L and paired lift over the same-activity reversed-sign
  control, with 4/5 positive chronological folds, owner-approved minimum effect, failed negative
  controls and all parity/risk/twin gates. A synthetic known-answer failure is `UNDERPOWERED` before
  outcomes; a powered economic nonpass is terminal `NO_TRADABLE_SIGNAL`. Opening real outcomes would
  be one ledger exposure, **6→7**, and no declaration exists.
- **The observable is new, but the economic family is not presumed new.** Strict individual prints
  plus causal non-replenishment cannot be reconstructed from CBBO-1m, so they qualify as a new
  information source. The closest prior W2-H02 option-microstructure continuation family was
  `NO_SIGNAL`; high magnitude/activity has already selected dear options without improving P&L. The
  rule escapes that fate only if it beats the activity-matched reversed-sign control after full
  latency and spread.
- **Runtime remains a hard stop.** Generic parity helpers and a historical minute simulator exist.
  Missing are canonical CMBP ordering, full-field DBN/Parquet identity, one shared historical/live
  event decoder, CMBP symbology-to-IBKR receipts, event-to-fill clocks, gap/drop/slow-reader handling,
  realistic partial/reject/cancel/slippage replay, no-order parity, and integrated enforcement of the
  signed 20%/max-two/−40% law. The frozen v4 paper path is not v5 CMBP parity or authority.
- Immutable wrappers, failure receipts, logs and successful receipts were archived under
  `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/`. The characterization implementation
  is [`analyze_cmbp_signed_flow.py`](../../ops/analyze_cmbp_signed_flow.py) with nine targeted tests.
- **Final verification green:** project/repository checker PASS; full v5 suite **1,159 passed** with
  105 pre-existing numerical warnings in 70.72 seconds; `git diff --check` clean. Only the intended
  finding, analyzer, test and LOG entry changed; unrelated `pickles-weekly-ranges/` remains untouched.

### SIGNED-FLOW VERDICT VERIFIED: NO. And the gate's own headline was overstated. 2026-08-23.

Finding: [`SIGNED_ORDER_FLOW_BOT_DECISION_2026_08_23.md`](../../research/findings/SIGNED_ORDER_FLOW_BOT_DECISION_2026_08_23.md)
(Codex/Sol). **Verdict: retain `STOP_UNDER_CURRENT_CONSTRAINTS`. Verified here, and it found a real
defect in yesterday's gate.**

- **THE TIE-BREAK DEFECT IS REAL AND IT IS MINE. Reproduced exactly: 25.89%.** OPRA delivers events
  sharing a timestamp and this parquet carries **no `sequence` field** — only `ts_recv`, `ts_event`,
  `ts_in_delta`. When the event immediately before a trade shares that trade's clock, "strictly
  before" is decided by **file order**, an assumption about vendor serialisation rather than anything
  verified. Yesterday's gate counted those signs silently.
  **This is the Phase-4a defect class recurring** — there, a stable lexsort resolved equal scores by
  frame order and the resulting pick read as a measurement. Same shape, different table.
- **Corrected headline, and the gate now enforces it:**

  | Quantity | Value |
  |---|---:|
  | Signed, raw *(reported 2026-08-22)* | 874,002 — **55.03%** |
  | of which **tie-ambiguous** | 226,288 — **25.89% of signs** |
  | **Signed unambiguously — the honest share** | **647,714 — 40.78%** |
  | Per-session unambiguous | min **8.7%**, median **37.7%**, max **48.2%** |

  `verify_cmbp_touch_semantics.py` now reports `signed_tie_ambiguous` / `signed_unambiguous` and
  **takes its verdict on the unambiguous share**, with three new tests including one that fails a
  slice which signs well *only* because of tied clocks. 15 tests total. The verdict is unchanged —
  `SEMANTICS_PASS_ONLY`, 40.78% against a 20% parser bar — but the number it rests on is smaller and
  the median session now signs barely a third of its tape.
- **The rest of the verdict verified.** Build is feasible but heavy: **8.52–9.39 hours** of wall time
  measured by real throughput probe. Flow is bursty and short-lived. With **229 session clusters** an
  optimistic 80%-power sensitivity needs an **annualised Sharpe ≥ 3.453**, and actual P&L power
  remains **UNKNOWN**. No CMBP live twin, no execution-parity path, and no current-law runtime exists.
  One terminal no-fit experiment is fully specified and was correctly **neither adopted nor run**.
- **A storage figure reconciled rather than disputed.** Their **1.071 TB** and this session's
  **0.55 TB** measure different things and both are right: their receipt's
  `projected_parquet_bytes_at_29_8b_rows` is **0.535 TB**, matching this session's estimate, while
  1.071 TB is the full footprint including the DBN that must be downloaded to build the parquet plus
  derived features. **Theirs is the better planning number** — you cannot hold only the output.
- **A check of mine that was wrong, recorded because it nearly became an accusation.** Two of the
  three receipts appeared to fail their self-hash. They do not: the convention excludes **only**
  `receipt_sha256`, while this session's check also stripped `input_sha256`, `manifest_sha256` and
  `wrapper_sha256`, which are content fields that must sit *inside* the hash. **All three self-hashes
  match. Codex's claim was right and the checker was wrong.**
- **Verification:** 1,162 tests green (1,159 + 3 new tie tests), checker green, ledger untouched at 6,
  no outcome read, no download, no spend, no vendor call, no protected file edited.
- **The STOP stands.** Signed flow supports a credible *experiment*, not a credible bot. The honest
  signing share is 40.78% rather than 55.03%, the Sharpe bar at 229 clusters is implausible for this
  strategy class, and the runtime that would have to carry any signal does not exist.
