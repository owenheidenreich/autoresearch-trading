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
