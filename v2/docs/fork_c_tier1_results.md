# Fork C Phase 1 results — Tier-1 day-gate classifier (null)

> **Scope-of-claim header.** This is a test of whether Pickles' SPX 0DTE
> directional-long-day call is predictable from features known by 10:00 ET
> using a strictly labeled subset of his 167 journal days, under the label
> rule defined in [label_rules.md](../fork_c/label_rules.md). A positive
> result would have shown the day-level call is learnable, not that his
> trades on positive days have edge for us. This negative result falsifies
> this specific label-and-feature combination at this sample size; it does
> not falsify behavioral cloning in general, nor Row-2 (MAGIC TIME) as an
> alternate target, nor the broader 5-class taxonomy tested in R3.

## Verdict

**Phase 2 three-way gate: `null` on every non-majority model.** At a 25-row
test fold with 3 positives, no model clears the advance bar
(base rate + 5 %) on PR-AUC or precision@top-30%, and both logistic
regressions have worse Brier than the majority baseline. HGBT was skipped
per the pre-registered conditional (lr_all val PR-AUC did not exceed
base rate + 5%).

Research-tier only. These classifiers are not deployable as a gate.

## What was measured

**Dataset:** [tier1_dataset.csv](../fork_c/tier1_dataset.csv) — 125
trading-session rows × 91 columns (7 label context + 79 base features from
`X_sim` at `bar_of_day == 30` + 5 aggregates from bars 0-30 only). 42 of
the 167 canonical label rows are weekends/holidays (all label=0) and were
correctly dropped at extraction.

**Labels:** [tier1_labels.csv](../fork_c/tier1_labels.csv) — 167 rows, 25
label=1 and 142 label=0, 162 High-confidence and 5 Medium, 0 Low (shadow
file empty after review). The parser emitted 85 candidate hits on 38
candidate days; the deterministic proposer ([propose_tier1_labels.py](../fork_c/propose_tier1_labels.py))
drafted first-pass labels; 12 human overrides were applied in
[apply_human_overrides.py](../fork_c/apply_human_overrides.py).

**Split (frozen by acknowledgment):** chronological 60/20/20, no shuffling.

| split | n | positives | negatives | pos rate | date range |
|---|---|---|---|---|---|
| train | 75 | 19 | 56 | 0.253 | 2023-10-23 … 2024-04-03 |
| val | 25 | 3 | 22 | 0.120 | 2024-04-04 … 2024-05-15 |
| test | 25 | 3 | 22 | 0.120 | 2024-05-16 … 2024-07-03 |

Train base rate is materially higher than val/test (regime shift, not a
split artifact).

**Preflight halted.** Both val and test have only 3 positives (threshold=5).
Late-positives fraction is 56.0% — just below the 60 % halt threshold but
material enough that the task is roughly half detection / half forecast.
[preflight_acknowledged.json](../artifacts/fork_c_tier1/preflight_acknowledged.json)
records the decision: accept inflated CIs, keep the pre-registered
60/20/20, freeze calendar boundaries, report the three-way outcome
honestly. The acknowledgment also fixed one proposer mislabel
(2023-10-30 → hedge-only per §N5 of label_rules.md) and flagged two rows
(2023-11-30 09:24, 2024-05-14 09:13) with pre-open
`first_qualifying_time_et` that make positive detection trivially easy at
the cutoff.

**Feature extraction discipline.** `X_sim` point-in-time at bar 30.
5 aggregates computed from bars 0-30 only (`am_session_range_bps`,
`am_session_end_vs_open_bps`, `ovn_proxy_direction`,
`prior_session_end_ret`, `vwap_sigma_session_frac`). The σ-band scalar
reuses Fork A1's `compute_daily_vwap_sigma` with unit volumes for
continuity. [test_extract_scope.py](../fork_c/tests/test_extract_scope.py)
blanks `X_sim[day_start+31:]` on a fixture day and asserts the extracted
base row and the aggregates are byte-identical; a companion test perturbs
a prior session's spot to confirm the σ-band correctly depends on prior
sessions. Three tests pass.

## Test-fold numbers (applied once with val-selected threshold)

Test n = 25 (pos = 3, base rate = 0.120).  Advance bar = 0.170.
Majority-baseline Brier on test = 0.123.

Bootstrap 95 % CIs from 1000 resamples. CIs marked ⚠️ cross the advance bar.

| model | PR-AUC (CI) | Brier (CI) | P@30 (CI) | R@30 (CI) | verdict |
|---|---|---|---|---|---|
| majority | 0.120 [0.040, 0.240] | **0.123** [0.064, 0.183] | 0.120 [0.000, 0.240] | 1.000 [1.000, 1.000] | null |
| lr_all (84 feat) | 0.128 [0.040, 0.362] | 0.301 [0.198, 0.421] | 0.143 [0.000, 0.333] | 0.667 [0.000, 1.000] | null |
| lr_curated (15 feat) | 0.166 [0.040, 0.500] | 0.240 [0.183, 0.300] | 0.100 [0.000, 0.333] | 0.333 [0.000, 1.000] | null |
| hgbt_all | — SKIPPED — | — | — | — | — |

**Lifts at top-K (test fold):**

| model | top-10% | top-20% | top-30% |
|---|---|---|---|
| lr_all | 0.00 | 0.00 | 0.00 |
| lr_curated | 0.00 | **1.67** | 1.04 |

The one positive signal worth noting: `lr_curated` puts 1 of 3 test
positives inside its top 20 % (5 predictions), which is a 1.67× lift over
base rate. But this reduces to chance at top-10 %, the PR-AUC CI crosses
the advance bar, and the Brier is worse than majority — so this does not
rescue the gate.

**Scaler discipline.** Scaler fit on train only. Max `|train_mean −
(train+val)_mean|` across 84 features = 1.003, well above noise.

**HGBT trigger.** Train base rate = 0.253, trigger = 0.303. lr_all val
PR-AUC = 0.161, below trigger. Skipped per plan — not a post-hoc
decision.

## Why it failed

The two leading candidates are not mutually exclusive:

1. **Regime shift in label rate.** Train positive rate is 25.3 %; val/test
   are both 12 %. Whatever lexical or behavioral regularity produced
   Pickles' earlier-period positives got weaker or changed in the later
   period covered by val + test. A classifier that generalized the
   train-regime signal to val/test's lower base rate over-assigned
   probability mass and took a Brier penalty for its trouble.
2. **Signal may not live at the 10:00 ET cutoff.** 56 % of positives have
   a first-qualifying event after 10:00 ET. For those days the cutoff
   features can only forecast, and 30-minute early-session features may
   not carry enough information about whether Pickles will decide to take
   a directional SPX 0DTE long at 11:50 or 12:57 vs sitting out or
   trading spreads only.

The small sample (25 positives total; 3 per minority fold) does not let us
distinguish these two explanations from plain "no signal under this
framing."

## Sensitivity re-run

**SKIPPED.** The pre-registered sensitivity rerun was defined as
`primary (High + Medium) vs primary + Low`. After round-1 review rescued
all Low candidates, the shadow file is empty, so the sensitivity dataset
would be identical to the primary. Mutating the sensitivity rerun into
a different experiment (e.g., "High vs High + Medium") would not be the
pre-registered comparison and is not done here. If we later want to test
label-confidence sensitivity, that is a separate follow-up diagnostic.

## What this proves — and what it doesn't

**Proves:**
- The current 125-row labeled dataset with 79 base + 5 aggregate features
  at bar 30 does not separate label-1 from label-0 days at 5 % margin
  over base rate under bootstrap 95 % CI.
- Logistic regression (all-feature or curated) is worse-calibrated than
  the majority baseline on this test fold (higher Brier).
- The three-way gate's `null` branch triggered cleanly; this is not an
  `underpowered` outcome camouflaged as null.

**Does not prove:**
- That Pickles' day-call is unlearnable in general. A different label
  definition, different cutoff, or different feature set could clear.
- That behavioral cloning as an approach has failed. This is one label,
  one cutoff, one feature basis, one sample size.
- That Row-2 (MAGIC TIME) is unlearnable. Phase 2 never ran.

## Decision

Fork C Phase 1 does not clear to Phase 2. Per the plan, the escalation
options for a `null` verdict are:

1. **Row-3 as alternate target.** A finer-grained label (e.g., directional
   bias per-15m session block) might carry more signal at 10:00 ET than
   the whole-day binary.
2. **ES/NQ/AD data acquisition.** Pickles' decision process explicitly
   references ES/NQ/AD volume and volatility. SPX-only features may be
   structurally incapable of reproducing his call.
3. **Reconsider project frame.** If neither of the above has a cheap path
   to evidence, falling back to mechanical-strategy design (not behavioral
   cloning) deserves honest reconsideration.

No default selection is made here. The choice is for the researcher.

## Provenance

- Plan: `~/.claude/plans/read-this-context-and-snappy-tide.md`
- Branch: `codex/fork-c-phase1` (safety anchor: `codex/pre-fork-c` at
  commit `41be6a5`).
- Data SHA-256: `f71ff9ef56cd3564c870599ac56777a98b22bf6e63f95e93f4e5559b3b1ad74f`.
- Labels SHA-256: see `tier1_dataset_meta.json` (updated after
  2023-10-30 override in round 1.1).
- Label rules SHA-256: see `tier1_dataset_meta.json`.
- Artifacts: [v2/artifacts/fork_c_tier1/](../artifacts/fork_c_tier1/)
  contains `preflight.json`, `preflight_acknowledged.json`,
  `train_report.json`, `test_report.json`, `fitted_models.pkl`.
