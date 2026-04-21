# Experiment Redesign — 2026-04-20

## Purpose

Reconcile the v3 research trail after the late-session `NR10` dry-run
falsified the original "B+C stack is ready to ship as a teacher" reading.

This note does **not** say the prior research was worthless. It says the
findings fall into different classes of signal, and we mixed those classes
together:

- **tradable directional edge**
- **regime-localization edge**
- **anti-edge / failure-mode edge**
- **risk/exit calibration**

The redesign below separates those categories and gives each one the right
experiment.

## What Survives

### 1. ORC has a real directional discriminator

Source:
- `v3/reference/side_error_deep_dive_2026_04_20.md`
- `v3/analysis/side_error_dive.py`
- `v3/analysis/test1_orc_sigma_filter_dryrun.py`

What survived:
- `sigma_pos` is the strongest direct fix found so far.
- It is measured on the correct denominator: **bars where ORC actually fires**.
- This is the cleanest example of a finding that translates from cohort
  insight into trigger-level improvement.

Interpretation:
- The real ORC edge is not "break + VWAP trend."
- The real ORC edge is "break + correct side of VWAP."

### 2. The late-session abstention cohort is a real regime

Source:
- `v3/reference/stage1_research_findings_2026_04_20.md`
- `v3/reference/omar_findings_2026_04_20.md`
- `v3/reference/volume_profile_2026_04_20.md`
- `v3/reference/combined_confluence_2026_04_20.md`

What survived:
- Abstention bars are late, inside first15, and structurally different from
  ORC wins.
- OMAR / developing VP / squeeze features are locating something real about
  that cohort.

Interpretation:
- This is **regime-localization edge**, not yet a standalone directional
  trigger.
- The research found **where** missed opportunities live more clearly than it
  found **how to enter them**.

### 3. The breakout-follow trigger is probably the wrong template

Source:
- `v3/reference/time_of_day_cap_2026_04_20.md`
- `v3/analysis/test2_nr10_teacher_dryrun.py`
- `v3/analysis/late_session_fakeout_split.py`

What survived:
- Raw and filtered late-session breaks have high fake-out behavior.
- When the setup is split by fake-out outcome, the population is not
  homogeneous. Clean breaks and failed breaks behave differently.

Interpretation:
- The earlier research did not prove "no late-session edge."
- It proved "do not force a one-bar breakout-follow teacher onto this regime."

### 4. Negative results are still valuable

Source:
- `v3/reference/stage1_research_findings_2026_04_20.md`
- `v3/reference/hvn_lvn_2026_04_20.md`
- `v3/reference/sr_confluence_2026_04_20.md`
- `v3/reference/decay_analysis_2026_04_20.md`

What survived:
- Fixed `-35/+60` exits are harmful.
- `breakout_confirmation` looks like a chased-break anti-signal for ORC.
- HVN/LVN and basic S/R do not add useful abstention discrimination.

Interpretation:
- "What not to do" is part of the edge map.

## What Broke In The Old Experiment Design

### 1. Cohort enrichment was treated like trigger precision

Examples:
- `v3/analysis/omar_retest.py`
- `v3/analysis/combined_confluence.py`

Problem:
- These scripts mostly answer "does this feature occur more often on oracle
  bars than on random bars?"
- That is useful for localization.
- It is **not** the same as "if I fire a trigger when this feature is present,
  do I get a directional edge?"

### 2. The late-session trigger family was fixed too early

Examples:
- `v3/analysis/time_of_day_cap.py`
- `v3/analysis/test2_nr10_teacher_dryrun.py`

Problem:
- The project locked onto `NR10 breakout-follow` before proving that
  breakout-follow was the correct trigger family for the abstention regime.
- The dry-run falsified the trigger family, not the regime itself.

### 3. Controls were too weak for final claims

Examples:
- `v3/analysis/combined_confluence.py`

Problem:
- Random-bar controls are good for first-pass discovery.
- They are not strong enough for final design decisions when the question is
  trigger precision.
- In `combined_confluence.py`, the control cohort also hard-codes
  `oracle_direction="call"` for every sampled bar, which makes filter `B`
  direction handling less trustworthy than it should be.

### 4. Repeated threshold tuning happened on the same 986-day cache

Problem:
- The sequence of docs is strong as exploratory research.
- It is weak as confirmation research because the same history is reused for
  threshold selection, filter stacking, and final claims.

## Redesigned Experiment Set

### Experiment 1 — ORC Direction Gate, Walk-Forward

Universe:
- Bars where baseline `ORC` fires.

Question:
- Does `sigma_pos` reduce side-error out of sample?

Compare:
- baseline ORC
- ORC + `sigma_pos <= 0` for calls / `>= 0` for puts
- ORC + `sigma_pos` + `breakout_confirmation` as a negative veto

Metrics:
- side_error count
- entered_right count
- retention of correct ORC fires
- realized contract PnL under the fixed mechanical exit and under hold-to-end

Acceptance rule:
- Keep only variants that reduce side_error materially without collapsing
  entered_right.

### Experiment 2 — Late-Session Localization Model

Universe:
- All bars in minutes `[40, 120]`
- `close` inside first15 range

Positive class:
- abstention oracle bars

Negative class:
- all other bars in the same universe, sampled with matching minute buckets
  and day counts

Features:
- OMAR distance
- developing-VP distance
- `sigma_pos`
- `abs_sigma_pos`
- `last10_range / omar_range`
- `breakout_confirmation`
- `volume_ratio`
- `bars_since_break_*`

Model:
- start with logistic regression and gradient-boosted trees

Question:
- Can the feature stack rank the oracle bar near the top **within the correct
  candidate universe**?

Success metric:
- top-k recall within day
- AUROC / PR-AUC inside the late-session candidate pool

### Experiment 3 — Trigger-Family Search, Not Filter Search

Universe:
- Bars that score highly in Experiment 2

Trigger families to compare:
- breakout-follow
- failed-break reversal
- break then reclaim
- second-test continuation
- re-entry into pre-break range after failed expansion

Rule:
- Keep the localization features fixed.
- Change only the trigger/event definition.

Question:
- Which trigger family turns the late-session regime into a point-in-time
  directional edge?

Success metric:
- direction match on oracle coincidences
- forward-move quality
- trigger density

### Experiment 4 — Two-Stage Fakeout Experiment

Universe:
- Filtered late-session breakout attempts

Stage 1:
- detect breakout attempt

Stage 2:
- detect whether it remains outside the pre-break range or fails back inside

Branches:
- clean-break branch: continue with breakout direction
- fakeout branch: test reversal entries only after the failure is observed

Question:
- Is the real late-session edge a **routing rule** rather than a single
  immediate entry rule?

Success metric:
- forward move after the stage-2 decision point
- realized PnL under mechanical fills

### Experiment 5 — Matched Confluence Retest

Repeat OMAR / VP / VWAP tests with stricter controls:
- match by minute bucket
- match by inside-first15 status
- match by direction where relevant
- report both enrichment and trigger-level performance

Question:
- Which confluence features still matter after the denominator is fixed?

Expected outcome:
- OMAR and VP may survive as localization features.
- They should no longer be described as ready-made directional triggers
  unless they prove that separately.

### Experiment 6 — Push Surviving Features Into `v2`

Do not wait for a perfect teacher library.

Add to the active `v2` feature pipeline:
- `sigma_pos`
- `abs_sigma_pos`
- distance to nearest OMAR level
- OMAR range
- `last10_range / omar_range`
- late-session-inside-first15 indicator
- optional developing-VP distance if compute cost is acceptable

Question:
- Do these features improve supervised ranking / competence signals even when
  hand-built teachers remain imperfect?

Success metric:
- lift in replay metrics
- lift in competence / opportunity ordering quality
- better calibration on abstention-vs-side-error slices

## Bottom Line

The prior research did not show "no edge."

It showed:
- one **real ORC directional edge**
- one **real late-session regime edge**
- several **real anti-edges**
- and one important failure: **we picked the late-session trigger family too
  early**

The next phase should stop asking "does B+C make NR10 work?" and start asking:

**"What is the correct trigger family once B+C tells us we are in the
late-session missed-opportunity regime?"**
