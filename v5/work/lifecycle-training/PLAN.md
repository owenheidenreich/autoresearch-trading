# Plan — full-ownable-corpus SPXW lifecycle development

**Registered as job 46 on 2026-08-15.** The controlling state is
[`v5/STATUS.md`](../../STATUS.md); authorization and rails are in
[`DEVELOPMENT_CHARTER_2026_08.md`](../../governance/DEVELOPMENT_CHARTER_2026_08.md).

This packet implements the owner-approved plan at
`/Users/och/.claude/plans/develop-a-plan-for-smooth-hare.md`. It serves the project question by testing
whether a causal, serially executable SPXW 0DTE long-call/long-put lifecycle can discover value beyond
the measured cost of entry and exit, without mistaking development evidence for permission to trade.

## Phases

1. **Governance — complete.** Record the charter, register this job, and establish this log.
2. **Acquisition and QC — running under declaration V5 (2026-08-17).** Two request-shape defects were
   found and corrected in this phase, both of a kind no receipt could reveal because each receipt
   faithfully described what was *asked for*.
   - **Scope.** The V1 preflight priced `cbbo-1m` at parent scope (every listed SPXW expiration,
     12,828-15,980 instruments/session), returned **$671.90** against the $75 ceiling, and correctly
     STOPPED. Charter §2 authorizes only same-day-expiry contracts, so that was never the price of the
     authorized data: the full resolved 0DTE ladder is **362-988 contracts at $0.0215-$0.0483/session**
     with `definition` **$0.00**. Declaration V2 fixed the request shape; V3 moved the destination to
     the SSD; V4 hardened the runner against a degraded vendor. **The V4 acquisition completed 794/794
     at $19.2450.** See [scope finding](../../research/findings/BACKFILL_REQUEST_SCOPE_2026_08_16.md).
   - **Clock.** Post-acquisition QC then found **every one of those 794 sessions missing its closing
     minute bar**: vendor windows are half-open and `cbbo-1m` bars are end-stamped, so the bar covering
     15:59 sat on the exclusive `16:00` bound and was never delivered. Both dataset gates reject a
     session for that one bar, so none could enter the episode build. The $0.05-estimated top-up
     actually priced at **$0.414442** and was refused; the owner chose the **full corrected
     re-request** on 08-17. **Declaration V5** (`65607a71…`) sets the close to **16:01**, writes to a
     new root, caps at **$22.00** so the owner's ~$19.29 authorization binds mechanically, and pins
     `download_spxw_history.py` as an implementation dependency — the file where the window lived and
     which the old runner-only hash did not cover.
   - **State:** V5 preflight **PASS at $19.2450 of $22.00**; acquisition interrupted at 406/794 by
     process death (no refusal, no ceiling breach, no drift) and **resumed 08-17 on owner approval**.
     Resumption is idempotent — existing outputs are skipped, so nothing is re-bought. Cumulative
     spend **$28.34 → $38.49 of the $75 charter ceiling**.
   - **QC on completion:** loader cleanliness, completeness, per-session 0DTE membership, no
     future-expiry quoted contracts, **and the delivered-clock gate**
     ([`verify_backfill_clock.py`](../../ops/verify_backfill_clock.py)), which asserts every session
     carries all 390 minutes including the terminal one. That gate is the control this phase was
     missing: every earlier gate asked whether the request was authorized and affordable, none asked
     whether the bytes that came back carry the clock the builder requires.
3. **Corpus construction.** Build all complete causal episodes with source-labelled settlement, first-touch
   labels, and target arrays. **The entry ceiling changed on 2026-08-16** from an equity-derived
   $1,300 to the signed fixed **$2,000 premium-plus-fees**, so `entry_eligible` now admits a wider
   action space and **every mask/candidate artifact built before that date is stale**. Rebuild rather
   than reuse; do not mix pre- and post-correction candidate tables. Re-measure effective sample size and bind a per-fit parameter statement.
4. **Preflight.** Execute the amended production-law capacity and full-gate known-answer preflights at the
   acquired chronology. Publish their sensitivity statement; development shortfall informs interpretation
   rather than auto-stopping this job.
5. **Declared lifecycle members.** Before reading outcomes, self-hash and verify a declaration covering
   the ITM/action-value member and the first-touch member, their shared architecture, chronology,
   controls, exposure ledger, inference, and alpha-ledger budget. Train entry, then frozen-entry exit,
   and evaluate midpoint before executable touch economics.

   **The exit is scored on two separate skills, never on one averaged number** (owner instruction,
   2026-08-16 — binding on the declaration):

   - **Loss averted on entries that did not develop** — mean net under the learned exit minus the
     duration-matched control, on the population whose forward path never reached the declared move.
   - **Excursion captured on entries that did develop** — realised gain as a fraction of the gain
     that was actually available, on the population that did reach it.

   Both are reported separately and each against a **duration-matched** control, because comparing a
   fitted rule to a long fixed clock credits it for holding time rather than for deciding (ledger row
   341). **"Beat holding" is not an acceptable exit metric.** These are different skills that fail
   independently, and averaging them is how the earlier work concealed exits that had neither: round
   3 measured hold-60 at **+$675** on big movers and **−$598** on non-movers, figures that cancel
   almost exactly under a random entry, which is the only regime every prior exit study measured.

   Two guards the declaration must carry with them: a degenerate exit must read as degenerate — an
   always-cut rule should post high loss-averted with near-zero capture, an always-hold rule the
   reverse, and **neither pattern counts as skill** — and both populations are defined by realised
   outcome, so this decomposition is **attribution, never a selection rule**. Serial executable P&L
   remains the decision criterion; these two numbers explain it rather than replace it.

   **Owner rulings of 2026-08-16, binding on this declaration:**

   - **Ticket sizing stays fixed at $2,000 in dollars, with no automatic growth**, until the $25,000
     equity review re-asks it against real results. No step-up ladder and no
     percentage-with-ceiling formulation. **The model being under-deployed at a large account is the
     accepted cost of never drifting into deep ITM.**
   - **Report both, as headline numbers.** Surviving-path per-trade economics *and* ruin probability
     with **time-to-ruin**. Never a footnote, never folded into one risk-adjusted composite — this
     project's history is that composites hide the failure they average over.
   - **`moneyness_band` is a separate, independently necessary guard**, named as such. 37.3% of
     contracts cheap enough to clear the $2,000 ticket are in the money (99th percentile +18.4
     points), so the price cap does not imply an OTM contract. Pinned by
     `v5/tests/test_entry_ceiling_is_dollars.py`.
   - **The outcome grid has three states, not two.** A result between break-even (~37%) and survival
     (~45–50%) at $10,000 is **a real but insufficient edge** — neither a failure nor a pass. It
     routes to replay diagnosis and declared alpha-ledger iteration under the development charter.
     **No paper trading until the model clears survival at the owner's actual account size**; the
     ~10-point gap between making money and surviving drawdowns is the binding bar, not break-even.
   - **State the target before the fit, not after.** Random-entry baseline **31.65%**; survival at
     $10,000 needs **45–50%**, about **+14 points**. That is roughly **twice the best entry effect
     this project has ever measured (+7.1pp)**. The declaration must say so plainly, so the bar is
     known to be demanding in advance rather than rationalised once a result is in hand.
   **Trainer built 2026-08-16, ahead of the data:** [`research/lifecycle_trainer.py`](../../research/lifecycle_trainer.py)
   implements the 40%-prefix/five-block chronology, nested out-of-fold trajectory generation (a
   trajectory's generator may never have trained on its session), and the frozen-entry exit phase
   verified bitwise. 15 tests. It consumes `SessionEpisode` records, so binding it to the corpus
   builder is the remaining wiring.
6. **Declared iteration.** Spend at most the declared roughly 20 experiments. Inspect replays before each
   single-change hypothesis; charge and log every member. End with an honest development classification.
7. **Dormant runtime artifacts.** Build and test recorded-session offline/runtime parity, an attended
   no-order shadow kit, and an unexecuted attended paper-launch script/checklist.
8. **Handoff.** Update durable findings, status, evidence index, and this log; run structural and test
   verification.

## Rails

- Only pre-2026-08-06 sessions are eligible for research data or evaluation.
- The sole permissible spend is the Phase-2 backfill, after exact preflight and at or below $75.
- No paper/live orders, broker contact, live subscription, real money, or unattended execution.
- No gate, knob, receipt, or ledger manipulation to obtain a pass.
- The amended protocol’s declaration, exposure, divergence, deterministic seed, checkpoint, capacity,
  session-inference, control, and semantic-freeze requirements bind every outcome-bearing run.
