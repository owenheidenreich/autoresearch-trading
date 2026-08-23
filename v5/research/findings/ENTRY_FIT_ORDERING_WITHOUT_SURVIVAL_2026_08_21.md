# The chain can reorder the ladder. It cannot pay for the trade.

**Job 46, entry fit. 2026-08-21. Opus. Result: the ordering claim passes and the survival bar
fails, so this is a stop-and-flag under memo §5, not a verdict.**

## 1. What changed, in one sentence

The option chain's own internals **do** carry usable ranking information — the fitted model beats a
matched random pick by **+$8.43 per entry** on held-out sessions, in the same direction in all five
score blocks — but the effect is roughly a third of what the trade needs, so **every held-out entry
stream still loses money** and nothing here can be traded.

## 2. First, the fit that was reported on 2026-08-20 was void

It trained on nothing. Its runner built episodes with `with_paths=False`, which leaves the adapter
unable to price the entry bracket; every `entry_value_usd` came back NaN; and `train_entry_phase`
masks the supervised term with `entry_action_mask & isfinite`, so **all 3,260 masked actions per
session were dropped from the loss and only the WAIT head received gradient.** The run converged in
0.48 minutes and reported a plausible hit rate for a model that had never seen an entry signal.

That failure mode — a silent empty objective — is the most expensive one available here, because it
answers nothing while looking like an answer. Two guards now close it from both ends:

- **`build_episode`** raises `EpisodeAdapterError` when a priced build has labelled entry actions
  and cannot price a single one. Test: `test_a_priced_build_refuses_a_session_it_could_not_price`.
- **`train_entry_phase`** raises `LifecycleTrainingError` when the episodes it is handed carry no
  finite target at all, before it spends anything. Test:
  `test_the_entry_phase_refuses_an_entirely_unsupervised_corpus`.

The split is deliberate. `with_paths=False` is target-free *by construction* and stays legal — the
causality control and the shape contracts use it and want no quote file — so the adapter cannot tell
a feature-only caller from a fit. The trainer can, and that is where the intent-aware guard belongs.

**Cost of doing it properly, measured: 1.24 s per episode against 0.80 s for the unpriced path** —
about 1.6x, not the 8x a first probe suggested. That probe wrapped each build in `tracemalloc`,
which inflated its own measurement several-fold; the number above is from the real run. Building all
1,014 sessions priced takes about 21 minutes.

## 3. The result

Fitted on the 405-session chronological prefix (2022-06-01 → 2024-01-29), scored on the five
contiguous fit-forbidden blocks. 118 parameters, seed 20260821, exit head frozen and verified
bitwise. Target coverage was asserted **before** the fit: 99.83% of feasible actions carried a
finite target, and valued matched labelled exactly.

Dollars are per entry, executable: ask in, first-later-bid out, one round trip of fees.

| Stream | Entries/session | Model | Minute-matched control | Ordering edge | Survival | Control |
|---|---:|---:|---:|---:|---:|---:|
| Training prefix (in sample) | 159.5 | −$17.68 | −$25.67 | **+$7.98** | 30.20% | 30.72% |
| Score block 0 (2024-01-30→07-29) | 238.1 | −$14.06 | −$21.31 | **+$7.25** | 28.54% | 28.70% |
| Score block 1 (2024-07-30→2025-01-29) | 256.3 | −$16.05 | −$21.93 | **+$5.88** | 29.98% | 30.83% |
| Score block 2 (2025-01-30→07-31) | 213.5 | −$19.29 | −$24.44 | **+$5.16** | 30.14% | 30.68% |
| Score block 3 (2025-08-01→2026-02-02) | 172.2 | −$13.70 | −$25.57 | **+$11.88** | 30.74% | 30.47% |
| Score block 4 (2026-02-03→07-30) | 164.4 | −$14.70 | −$29.51 | **+$14.81** | 31.61% | 30.76% |
| **All held out** | **209.0** | **−$15.66** | **−$24.09** | **+$8.43** | **30.07%** | **30.25%** |

**The ordering effect is real.** Positive in five blocks of five, out of sample, forward in time,
against a control that holds the minute fixed and randomises only the contract — so it is contract
selection, not luck about when to trade. It also **grows chronologically**, +$5.16 → +$14.81, which
is the same gradient Phase 4b found in the ordering component and could not explain.

> **CORRECTION, 2026-08-22 — this sentence is wrong and the error is in the presentation, not the
> arithmetic.** "+$5.16 → +$14.81" quotes the minimum and the maximum, not the first and last blocks.
> **In true chronological order the series is +$7.25, +$5.88, +$5.16, +$11.88, +$14.81** — it
> *declines* monotonically across the three backfill blocks and then **steps** at block 3. Block 3
> begins **2025-08-01, which is exactly the boundary between the parity-settled backfill quote source
> and the officially-settled owned source**, so "recent" and "owned data era" are perfectly collinear
> here (`0 0 0 1 1` against `0 0 0 1 1`) and no statistic on these five blocks can tell a market
> regime from a data-source change. The model's own selected population also shifts across the seam
> (213.5 → 172.2 entries/session). **This is a level shift at a provenance seam, not a chronological
> trend, and it is `NOT_IDENTIFIABLE` on current data.** See
> [`REGIME_VS_FOLD_SIZE_AND_ROUTE_CENSUS_2026_08_22.md`](REGIME_VS_FOLD_SIZE_AND_ROUTE_CENSUS_2026_08_22.md).

**The entry does not survive.** −$15.66 per entry over 126,900 held-out entries is **−$1,987,457**.
Only 31.1% of entries are profitable. Survival is **30.07% against a pre-committed 45–50% target**,
and it does not beat its own control: the minute-matched random pick survives at 30.25%, and the
whole held-out population of feasible actions at 30.87%. On the label the model is **flat to
slightly negative**, having gained nothing against the ≈+14 point target.

Both columns are worth reading together, because they disagree about what the model learned: it
buys contracts that **lose less money** without buying contracts that **hit +50% more often**. Those
are different objectives, and the model was fitted to the first.

**It is not a position-sizing problem.** The model fires 209 times a session against a signed risk
law of two tickets a day. Taking only the **first two entries it fires each session** — the
executable reading of that law — gives **−$18.65 per entry**, which is worse, not better. Against
that, the oracle's best two entries per session are worth **+$536.25**. The opportunity is present
and the model captures none of it.

## 4. Why it fails, mechanically

The member is **structurally capable of reordering**, which is the thing V5 could not do. Decomposing
its scores into an additive minute-effect plus contract-slot-effect leaves a residual of **41% of
score variance** (range 18–74% across 40 sampled sessions). V5's autopsy scored **2.4e-7** on the
same decomposition — it could not rank contracts even in principle. That barrier is gone.

The capability is there; the accuracy is thin, and the calibration is wrong in a way that matters:

- **Within a minute, the ranking is weakly right.** Predicted-vs-realised Spearman is **+0.102**,
  positive in 58.3% of minutes. Real, small.
- **Across minutes, the level is uninformative.** Sorted into deciles of predicted value, realised
  value is flat: the **top** decile predicts +$185 and realises **−$25.92**, while the **bottom**
  decile predicts −$684 and realises **−$15.20**. The ordering the model has is local to a minute;
  its absolute number carries nothing.
- **The fire/wait decision uses the level, not the ranking.** `select_entries` enters whenever the
  best contract's predicted value exceeds WAIT's $0 floor. 22.0% of all actions clear that floor, so
  the policy fires on most minutes — 209 a session — while the realised mean of what it buys is
  −$15.66.

So the model is a weak within-minute ranker being used as an absolute value estimator, and the
$0 floor is doing the deciding.

The fit itself is sound: it ran its **full 200 epochs** rather than plateauing early (verified by
re-running the same optimisation with the epoch count kept). The 0.57-minute wall time is real and
comes from `candle_prefix="last"`, which the equivalence test licenses for both declared members.

## 5. What this settles and what it does not

**Settled: chain internals are not an empty information family.** They order the ladder, out of
sample, forward in time, consistently. That was the open question this job existed to answer and the
answer is yes.

**Settled: this entry stream cannot be traded.** On its own objective it loses money in every
held-out block, under the risk law as well as without it, and it is below the base rate on the
label. No amount of threshold-picking rescues a stream whose top predicted decile is its worst
realised one.

**Not settled, and not for this session to settle.** The ordering edge is +$8.43 against a $24.09
gap to break-even — 35% of the way. Whether "a real but sub-scale ordering effect" counts as the
honest negative the design brief invited, or as a calibration problem worth one more pass, is a
design judgement. The specific question underneath it: the model is fitted to **unconditional**
dollar value while the entry decision compares against a **$0 WAIT floor**, so a perfectly
calibrated model on this corpus would simply never fire — the population mean is −$28. Whether the
objective should be conditional ordering with a separate gating law, or whether "never fire" *is*
the finding, is exactly the kind of question this project has previously settled by accident inside
an implementation.

**No DO_NOT_RETEST row is written yet.** The job is stopped at a boundary for adjudication, not
closed.

## 6. Provenance

- Producer: `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/entry_fit_producer_2026_08_21.py`
- Receipt: `entry_fit_receipt_2026_08_21.json` · Log: `entry_fit_2026_08_21.log`
- Diagnostic: `entry_fit_ordering_diagnostic_2026_08_21.{py,json}`
- Entry stream, 191,259 rows: `entry_stream_2026_08_21.parquet` · Model: `entry_model_2026_08_21.pt`
- Statistics reused from the void run's `prefix_statistics.npz` — fitted on the prefix and nothing
  later, and unaffected by the pricing defect that voided the fit consuming them.
- Terminal-dependent numbers inherit the **unmade settlement-source decision** for the parity-settled
  76% of the corpus. No zero-recovery twin was run; the sign here is not close, but the requirement
  stands.

---

# Addendum — the frozen-entry exit. Phase 5's second half, same day.

## 7. The exit phase could not run as specified, and why

`train_exit_head` visits every held batch on every epoch. The fitted entry policy fires **159 entries
a session**, which across the four inner folds is **50,318 out-of-fold trajectories** — at the
measured 0.04 s forward pass, about **115 hours** for the declared 200 epochs. The exit phase cannot
run on the unconstrained entry stream.

The cap is forced rather than chosen. The signed risk law is **two tickets a day**, and a bot walking
the session serially cannot know which two of its entries will turn out best — it takes the **first
two it fires**. That is the only causal reading, it is what the runtime would hold, and it brings the
phase to 644 trajectories over 323 sessions and under a minute of fitting.

This is the standing "inference law vs risk law" open question in concrete form. It is reconciled
here **only far enough to have an exit head to fit**, and that reconciliation is not a ruling.

## 8. The learned exit is a degenerate always-hold

644 out-of-fold trajectories, entry parameters frozen and verified bitwise, mean hold 60.0 minutes.

| Rule | Mean | Median | Total | Profitable |
|---|---:|---:|---:|---:|
| Oracle (per-path best) | **+$254.64** | +$86.92 | +$163,986 | 78.4% |
| Always cut immediately | −$17.19 | −$13.08 | −$11,074 | 28.6% |
| **Bracket** (+50% / −30% / 60m) | −$24.18 | −$73.08 | −$15,574 | 32.5% |
| **Learned exit** | −$18.15 | −$88.08 | −$11,689 | 32.0% |
| Always hold to the clock | **−$1.74** | −$93.08 | −$1,119 | 31.4% |

**The learned exit beats the bracket by +$6.03 a trade and loses to always-hold by $16.41.** It is
identical to always-hold on **79.5%** of trades and to always-cut on 0.8%: it sells before forced
liquidation only 23.4% of the time, and the times it does sell make it worse than not having sold.

The plan's own degeneracy guard settles how to read this: *"an always-cut rule should post high
loss-averted with near-zero capture, an always-hold rule the reverse, and **neither pattern counts as
skill**."* The learned exit is the always-hold pattern. **It is not an exit skill.**

(The oracle share is not quoted as a capture ratio here. A negative total over a positive oracle
total produces a number — −7.1% — that looks like a statistic and means nothing. The two-skill split
that would measure this properly is Fable's to design.)

## 9. The finding inside the exit result that bears on the entry

**Holding to the clock beats the bracket by $22.44 a trade on this stream.** The entry model was
fitted to `entry_value_usd`, which *is* the bracket outcome — so the entry was trained to rank
contracts under an exit rule that is worse than doing nothing. The −30% stop is cutting positions
that recover.

This does not rescue the entry: always-hold is −$1.74, still a loss, still against a +$254.64 oracle.
But it does mean the entry's target and the best available fixed exit disagree, and that is a
plausible contributor to §4's central defect — a model whose within-minute ranking is weakly right
while its level is uninformative.

Stated as a caution, not a claim: these 644 trades are the risk-law-capped out-of-fold prefix stream,
a different and much smaller population than §3's 126,900 held-out entries. The two tables must not
be read against each other.

## 10. The blocker this phase ends on, which is governance and not code

**Phase 5 requires a declaration that does not exist.** `PLAN.md` phase 5: *"**Before reading
outcomes**, self-hash and verify a declaration covering the ITM/action-value member and the
first-touch member, their shared architecture, chronology, controls, exposure ledger, inference, and
alpha-ledger budget."* No `PHASE_5_DECLARATION` is on disk. Phases 4a and 4b each had one, each
self-hashed by a dedicated `v5/ops/run_*.py` runner that charged the alpha ledger; the ledger
accordingly records exactly two experiments and neither is a fit.

The work log anticipated this in writing on 2026-08-16: *"The amendment counts as declared experiment
#1 and **must be charged when the ledger opens at Phase 5, before any fit**. Recorded here so it
cannot be quietly skipped."* It was skipped — by the void 2026-08-20 fit and again by this session's
entry and exit fits.

**This blocker must not be cleared by writing the declaration now.** A declaration authored after its
outcomes are known is not a preregistration, and producing one would be precisely the manoeuvre the
rule exists to prevent. The honest consequence:

- The entry fit and the exit fit are **development-grade diagnostics, not declared outcome-bearing
  runs.** They cannot be charged PASS or FAIL against a pre-declared bar, because there was no bar.
- **Had either result been positive it would have been unbankable.** Both are negative, so the gap
  costs little this time — a session does not talk itself into a loss. That is luck, not process.
- The declaration must be written and self-hashed **before any re-fit**, and the ticket-widening
  amendment charged as declared experiment #1, as the log required.

For scale: the ledger's next bar is **0.6527** accuracy (true accuracy needed 0.6787) against a
break-even of 0.5799. The measured entry survival is **0.3007**. Nothing here is close to the bar the
ledger would have applied.

## 11. Provenance, addendum

- Producer: `entry_exit_fit_producer_2026_08_21.py` · Receipt: `exit_fit_receipt_2026_08_21.json`
- Log: `exit_fit_2026_08_21.log` · Stream, 644 rows: `exit_stream_2026_08_21.parquet`
- Fitted entry+exit model: `lifecycle_model_2026_08_21.pt`
- All under `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/`.
- Out-of-fold firewall enforced by `assert_trajectories_are_out_of_fold`, before and after the risk-law cap.

---

# 12. Corrections, after a cold adversarial review. 2026-08-22.

An external cold review (Codex/Sol, commissioned by the owner in place of Fable) ruled on §10 and
audited the producers. It was right on every material point below. These are recorded here rather
than edited into the body above, because the body is what was published and the corrections are part
of the record.

**12.1 — The exit head has no holdout, and §8 did not say so.** `train_exit_head` was given the 644
risk-law-capped trajectories and the learned exit was then scored **on those same 644**. The
out-of-fold firewall that §11 cites protects the *entry generator* — no trajectory came from an entry
model that trained on its own session — and it does **not** create a holdout for the exit head. §8's
numbers are therefore an **in-sample engineering result for the exit head**, and calling it a Phase 5
exit verdict would be wrong. The reviewer's own reading of the consequence is the right one and makes
the stop *stronger*, not weaker: the learned rule loses to always-hold by $16.41 a trade **even where
it was trained**. But the claim must carry its correct label.

**12.2 — The results were not reproducible when published.** Both producers load
`prefix_statistics.npz` from a prior session's scratchpad, which has since been cleaned. The file was
absent at review time, so neither fit could be reproduced from a clean checkout. It has now been
**refitted and archived** as `prefix_statistics_2026_08_21.npz` beside the receipts, and verified
deterministic: two independent `FeatureStatistics.fit` passes over the same 405-session prefix are
bitwise identical, and the artifact is byte-for-byte the same size as the original. Reusing a
scratchpad file as a fit dependency instead of archiving it into evidence was the error.

**12.3 — The ordering diagnostic ran an undisclosed fit.** `entry_fit_ordering_diagnostic_2026_08_21.py`
constructs a fresh model and runs `_plateau_fit` on 60 prefix sessions to recover the epoch count.
That is a fit. §4 presented it as a convergence check and did not disclose it as an additional
optimisation. Whether the charter's "each fit" wording is meant to charge a diagnostic optimisation
is **UNKNOWN** and is an owner question, but it must be on the list either way.

**12.4 — §10's "there was no bar" was too broad.** No Phase 5 *declaration-specific inference law*
existed. But `PLAN.md` phase 5 had already precommitted the **45–50% survival target against a 31.65%
baseline** and the rule that **always-cut / always-hold behaviour is not skill**. Both fits fail those
precommitted bars: entry survival 30.07%, and the learned exit is the always-hold pattern on 79.5% of
trades. The results are not unmeasurable against signed text — they fail it.

**12.5 — §10's ledger comparison mixed quantities.** The alpha ledger's `next_bar` of 0.6527 is a
**directional-accuracy** bar; 0.3007 is label survival. Phase 4a's own declaration records
precision-like statistics as null precisely because they are not comparable. §10's "nothing here is
close to the bar the ledger would have applied" overstates a loose scale comparison as if it were a
formal gate. It is not one.

**12.6 — A signed-design-versus-code conflict in the fitted member, inherited rather than introduced.**
`LEARNING_CONTENT_DESIGN_2026_08_16.md` names `move_from_open_points` and `move_15m_rel` as the tape
channels and **excludes `return_1m` as dominated by the 15-minute term**. The fitted member reads
`close_from_session_open_points`, `range_position`, **`return_1m`** and `realised_vol_15m`. The first
is the design's name for the same quantity; the conflict is real for the other two — and it is worse
than a substitution, because **no 15-minute return feature exists in `CANDLE_FEATURES` at all**, so
the design's momentum channel was never implementable as written and the excluded feature was used in
its place.

This is **not** a defect of this session: the member was built in commit `25e2e1f7`, is pinned by
`PHASE_4A_DECLARATION_V1.json`, and was sealed before any fit — so the conflict passed the Phase 4a
adjudication as well. Phase 4b independently measured this channel family as near-worthless
(randomized tie-breaking collapses the tape group to −0.16pp; `realised_vol_15m` alone is −1.07pp), so
it probably does not change any conclusion. It is still a signed-text-versus-code conflict and is
reported rather than resolved.

**12.7 — The §5 boundary crossing was authorised, and the authorisation was not written down.** The
work log shows an entry-phase STOP for adjudication followed by an exit fit, with no on-disk evidence
of authority to cross. The review correctly marked this **UNKNOWN** from the artifacts. The
authorisation existed — the owner instructed this session, in conversation on 2026-08-21, to clear the
blocker and proceed through the next phase of the plan. Failing to record an in-conversation
authorisation in the log is the defect, and this paragraph is the record.

**12.9 — Two exit-reporting mislabels, found by the 2026-08-22 cold review of the exit law.**
§8 reported **"mean hold 60.0 minutes"**; that field is the *available path length*, not the learned
rule's realised holding time, which is **54.44 minutes** — 498 of 644 trades run the full path. §8
also reported the rule selling before forced liquidation **23.4%** of the time; that counted 151 rows
where a SELL fired *anywhere*, including five on the final row, which is forced liquidation rather
than a decision. **Strictly before the final row is 146 of 644, 22.7%.** The 79.5% always-hold
identity is unchanged, and the degeneracy conclusion is unchanged and slightly strengthened. Full
treatment: [`EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md`](EXIT_TWO_SKILL_REQUIREMENT_COLD_REVIEW_2026_08_22.md).

**12.8 — What the review did not change.** The entry and exit measurements themselves stand as
computed; no number in §3 or §8 is withdrawn. What changes is their *status* and their *labels*: the
exit figure is in-sample for the exit head, and the "+$8.43 ordering effect is real" language of §3
must be read as **exploratory** — its control, population and inference law were undeclared, so its
causal significance is **UNKNOWN**, not settled. §5's claim that chain internals "are not an empty
information family" is downgraded accordingly, from settled to suggestive.
