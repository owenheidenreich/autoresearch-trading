# Autonomous Self-Improvement Handoff

This document is for an autonomous AI agent that will continue the `v2/` research loop with minimal human supervision.

It is not permission to improvise the mission.
It is a bounded operating manual for recursive self-improvement inside the current exact-chain research phase.

Read this file after:

1. [v2/HANDOFF.md](/Users/gduby/Documents/autoresearch-trading/v2/HANDOFF.md)
2. [v2/docs/founder_intent.md](/Users/gduby/Documents/autoresearch-trading/v2/docs/founder_intent.md)
3. [v2/program.md](/Users/gduby/Documents/autoresearch-trading/v2/program.md)
4. [v2/COMMANDS.md](/Users/gduby/Documents/autoresearch-trading/v2/COMMANDS.md)

## Purpose

Your job is to improve the research system without corrupting the evidence base.

Self-improvement means:

- choosing better hypotheses from current traces and logs
- updating your own written operating context after each experiment
- retiring weak idea families when evidence says they are weak
- preserving a clean chain of custody from data to model to replay to decision

Self-improvement does **not** mean:

- inventing new success criteria
- silently broadening the mission
- treating screening results as official evidence
- mutating the dataset without the explicit trigger
- optimizing for activity instead of truth

## Mission Lock

The current mission is not live trading.
The current mission is a trustworthy exact-chain research system for SPX 0DTE long-options training and replay.

If any local optimization conflicts with that mission, stop optimizing and return to the mission.

## Trust Hierarchy

When sources disagree, trust them in this order:

1. [v2/docs/founder_intent.md](/Users/gduby/Documents/autoresearch-trading/v2/docs/founder_intent.md)
2. [v2/program.md](/Users/gduby/Documents/autoresearch-trading/v2/program.md)
3. [v2/results.tsv](/Users/gduby/Documents/autoresearch-trading/v2/results.tsv)
4. [v2/lab_notebook.md](/Users/gduby/Documents/autoresearch-trading/v2/lab_notebook.md)
5. [v2/HANDOFF.md](/Users/gduby/Documents/autoresearch-trading/v2/HANDOFF.md)
6. the current artifact and trace files

Never let memory outrank the written repo.

## Current Canonical State

- Dataset: `v2/data.pt`
- Sidecars: `v2/data_sidecars/*.pt`
- Dataset version: `v4_exact_chain`
- Dataset fingerprint: `46f2d184e186496f`
- Official baseline artifact: `exp_106`
- Current live code: `exp_106` baseline behavior in [v2/train.py](/Users/gduby/Documents/autoresearch-trading/v2/train.py) plus the morning policy in [v2/core/policy.py](/Users/gduby/Documents/autoresearch-trading/v2/core/policy.py)
- Official trade window: bars `60` through `120`
- Official scored runs: `exp_074` through `exp_078`, `exp_099`, `exp_104`, `exp_106`
- Rejected screens after the baseline lock: `exp_107` through `exp_110`

## What The Agent Must Remember

### Stable truths

- Oracle replay is achievable; the harness itself is not the blocker.
- Contract features alone do not explain oracle contract selection.
- The model’s remaining problem is not just win rate. It is a deeper gate and cross-side scoring problem.
- The morning window is real and load-bearing.
- The dataset audit found anomalies, but not enough overlap with bad trace days to justify a dataset migration.

### Current baseline diagnosis

From `exp_106` promote traces:

- gate accuracy: `21.2%`
- selection accuracy: `6.8%`
- average oracle minus model gap: about `0.346`
- promote replay direction mix: `191C / 0P`

The key trace finding after deeper analysis:

- when the model traded and the oracle also wanted a call, the model was often serviceable
- when the model traded and the oracle wanted a put, the model was catastrophic
- `exp_110` confirmed this failure surface is real, because a direct calibration term broke the all-call collapse but overcorrected into puts

Interpretation:

- the project’s active problem is cross-side calibration and selection quality, not generic “needs more temporal context”

## Current Hypothesis Status

### Families that are dead or parked

- explicit direction head family (`exp_100` to `exp_103`)
- pure Kronos-style temporal embeddings as a standalone fix (`exp_108`)
- pure flow dropout as a standalone fix (`exp_109`)
- policy-window-only supervision as a standalone fix (`exp_107`)
- first-pass cross-side margin calibration at the tested weight (`exp_110`)

### Families that remain alive

- gentler cross-side calibration ideas that preserve full exact-chain competition
- trace-targeted selection improvements that do not add a separate direction head
- audit-driven dataset escalation only if the explicit trigger is crossed later

### Families that are deferred

- stop/contract-filter policy queue
- dataset rebuild or relabel
- broad architecture reinvention
- anything that needs archive context by default

## Approved Mutable Surface

Default loop:

- [v2/train.py](/Users/gduby/Documents/autoresearch-trading/v2/train.py)
- [v2/core/policy.py](/Users/gduby/Documents/autoresearch-trading/v2/core/policy.py)

Approved expanded surface for the current block:

- [v2/train.py](/Users/gduby/Documents/autoresearch-trading/v2/train.py)
- [v2/replay.py](/Users/gduby/Documents/autoresearch-trading/v2/replay.py)
- [v2/core/data_integrity.py](/Users/gduby/Documents/autoresearch-trading/v2/core/data_integrity.py)
- [v2/ops/pre_run_gate.py](/Users/gduby/Documents/autoresearch-trading/v2/ops/pre_run_gate.py)

Use the expanded surface only when the hypothesis cannot be tested honestly inside the two-file loop.

## Autonomous Improvement Loop

Run this loop recursively, but do not skip steps.

### 1. Re-anchor

- Read the four startup docs listed at the top of this file.
- Run `git status --short`.
- Confirm that [v2/train.py](/Users/gduby/Documents/autoresearch-trading/v2/train.py) is on the baseline unless you are in the middle of an experiment.
- Confirm [v2/results.tsv](/Users/gduby/Documents/autoresearch-trading/v2/results.tsv) has only official runs.

### 2. Rebuild local understanding from written evidence

- Read the latest section of [v2/lab_notebook.md](/Users/gduby/Documents/autoresearch-trading/v2/lab_notebook.md).
- Inspect the current official trace source:
  - [v2/artifacts/replay_traces.csv](/Users/gduby/Documents/autoresearch-trading/v2/artifacts/replay_traces.csv)
- Summarize, in writing, the dominant failure mode before proposing any new code.

### 3. Choose exactly one hypothesis

A valid hypothesis:

- targets one failure surface
- predicts what metric or trace pattern should change
- can be falsified by a 1-fold screen
- does not rely on hidden memory

An invalid hypothesis:

- bundles multiple unrelated ideas
- changes the task definition
- depends on a future dataset rebuild
- ignores the last trace

### 4. Define success before editing

Write down:

- expected direction-balance change
- expected gate or drawdown change
- what would count as evidence to continue the family even if the score stays negative
- what would make the family dead

### 5. Edit

- Prefer the smallest honest implementation.
- Use `apply_patch`.
- If you expand the mutable surface, update the live docs in the same change set.

### 6. Run the local gate

- `python3 -m v2.ops.pre_run_gate --data v2/data.pt`

Do not spend GPU until this passes.

### 7. Screen first

- `./v2/ops/deploy.sh run_screen exp_NNN`

Judge the result against:

- score
- hard gates
- direction balance
- baseline comparisons
- whether the targeted trace failure mode improved

### 8. Decide what to do next

If the screen is clearly bad:

- revert the code
- log the result
- update the handoff state
- choose a different hypothesis

If the screen is still bad but clearly improves the exact targeted failure mode:

- it is allowed to continue the family once more
- write down exactly why the family remains alive

If the screen looks promotable:

- run `./v2/ops/deploy.sh run_one exp_NNN`
- generate the mandatory promote trace
- only then decide keep/revert

### 9. Externalize learning

After every experiment, update:

- [v2/lab_notebook.md](/Users/gduby/Documents/autoresearch-trading/v2/lab_notebook.md)
- [v2/HANDOFF.md](/Users/gduby/Documents/autoresearch-trading/v2/HANDOFF.md) if the stable understanding changed
- [v2/program.md](/Users/gduby/Documents/autoresearch-trading/v2/program.md) if the live protocol changed
- this file if the autonomous operating method should change

Do not keep important conclusions only in model context.

## Recursive Self-Improvement Rules

These are the meta-rules for improving your own research behavior.

### Improve hypothesis quality, not just model code

After each experiment, ask:

- Did I target the right failure surface?
- Did I predict the sign of the change correctly?
- Did I choose the smallest honest intervention?
- Did I learn something reusable even if the score did not improve?

If the answer is no, improve the hypothesis selection process before editing more code.

### Retire bad families aggressively

Retire a family when:

- two or more attempts reproduce the same failure mode without a meaningful secondary improvement
- the family improves one metric by simply flipping collapse from one side to the other
- the family requires changing the mission boundary to look good

Example:

- `exp_110` means “cross-side calibration exists as a real failure surface”
- it does **not** mean “the tested formulation is good”

### Prefer evidence compression

A self-improving agent should reduce uncertainty over time.

That means:

- convert repeated findings into stable docs
- shrink the search space when a family dies
- make the next agent’s starting context cleaner than yours

### Never recurse into abstraction for its own sake

Do not create elaborate planning taxonomies when a concrete next experiment is available.
The point of recursion here is tighter diagnosis and better decisions, not meta-ceremony.

## Improve The Entire Project, Not Just The Model

The project is a research machine, not only a model file.
If the machine is weak, better hypotheses will still be wasted.

Project-level improvement is valid and encouraged when it preserves the evidence base and increases future research quality.

### The project has five improvement layers

#### 1. Truth layer

Goal:

- make sure data, sidecars, replay semantics, and official evidence stay defensible

Current opportunities:

- classify the 9 recurring short-session sidecar schema-break dates
- keep improving anomaly reporting so dataset migration decisions are explicit rather than emotional
- make it easier to compare audit findings with worst trace days

Good project improvements:

- better anomaly summaries
- stronger overlap reports between audits and traces
- clearer documentation of exchange-calendar edge cases

Bad project improvements:

- rebuilding the dataset “just to see”
- mutating the evidence contract without a written migration decision

#### 2. Research operating system layer

Goal:

- make the experiment loop faster, safer, and harder to misread

Current opportunities:

- improve the handoff quality after each loop
- reduce ambiguity between official results and screens
- make experiment-family retirement decisions more explicit

Good project improvements:

- stronger handoff templates
- experiment-family summaries
- better status reporting from current docs and results

#### 3. Observability layer

Goal:

- make failure modes cheaper to diagnose

Current opportunities:

- automate trace aggregation by side, time bucket, and skip reason
- save standard trace-derived summaries next to the raw CSV
- make side-calibration failures visible without ad hoc notebook work

Good project improvements:

- a lightweight trace-analysis helper under `v2/analysis/` or `v2/ops/`
- standard summaries for:
  - traded-bar oracle positivity
  - same-side vs opposite-side delta gap
  - gate-skip missed winners
  - occupancy effects from `in_position`, `cooldown`, and `loss_cap`

This layer is especially valuable right now because the project’s bottleneck is diagnosis quality.

#### 4. Deployment and reliability layer

Goal:

- reduce wasted time and cost from Akash failures

Current opportunities:

- make lease-state checking more explicit when `.deploy-state` is stale
- surface low-ACT balance conditions before boot attempts fail
- improve remote status visibility during long uploads and screens

Good project improvements:

- a helper that verifies the saved lease is still active before using it
- clearer wallet/deposit guidance in the operator docs
- better remote-progress summaries in deploy tooling

#### 5. Documentation and memory layer

Goal:

- make the repo smarter across sessions, not just the current model context

Current opportunities:

- promote repeated findings from the notebook into durable docs faster
- keep the autonomous handoff aligned with the live protocol
- compress repeated experiment lessons into “dead family” summaries

Good project improvements:

- structured “what changed / what died / what remains alive” updates
- keeping the handoff smaller and more precise over time, not noisier
- adding new docs only when they reduce ambiguity instead of multiplying it

## When Project Improvements Should Take Priority

Do project-level work before another experiment when any of these is true:

- the next hypothesis is not yet clear from traces
- repeated screens are failing for the same reason and diagnosis quality is poor
- deployment friction is consuming meaningful time or budget
- docs and live code are starting to drift
- a small tooling improvement would make several future experiments cheaper or more truthful

Do **not** hide from modeling work by drifting into generic cleanup.
Project improvements must pay back into research quality.

## Preferred Whole-Project Improvement Backlog

These are the best current project-level improvements, ordered by expected leverage.

### Priority A — Trace tooling

Build a standard trace post-processor that outputs:

- side-match rates on traded bars
- same-side vs opposite-side delta gaps
- missed oracle winners by skip reason
- bucketed summaries by `bar_of_day`
- worst dates by average `delta_pnl`

Why:

- this directly supports the current `exp_111` decision
- it reduces repeated manual analysis
- it improves the next hypothesis more than another generic ablation would

### Priority B — Audit interpretation

Build a small calendar-aware classifier for the recurring schema-break dates so the project can distinguish:

- harmless short-session artifacts
- real sidecar breakage

Why:

- the audit track already exists
- the current weak point is interpretation, not raw detection

### Priority C — Deployment resilience

Improve Akash handling so the system can:

- detect closed leases earlier
- warn when ACT balance is too low for the default deposit
- guide the agent toward the lowest-risk recovery path

Why:

- this directly affected the last session
- failed deployment recovery is pure waste

### Priority D — Family memory

Add a compact way to record:

- what each experiment family tried
- why it died or stayed alive
- what exact failure mode repeated

Why:

- this prevents the agent from rediscovering the same bad family under a new label

## A Second Loop: Project Improvement Loop

Run this separately from the experiment loop when needed.

### 1. Identify the bottleneck class

Choose one:

- data truth
- observability
- deployment reliability
- documentation memory
- model training

Do not say “the project needs improvement” in general.
Name the bottleneck class.

### 2. Define the payoff

Write one sentence:

- “This project improvement will make future experiments more truthful / cheaper / easier to diagnose by doing X.”

If you cannot state the payoff, do not do the work.

### 3. Make the smallest useful improvement

Examples:

- one trace summarizer
- one deployment health check
- one audit interpretation helper
- one handoff compression pass

Avoid broad refactors unless the system is actually blocked.

### 4. Validate the improvement

Project improvements should have a check too:

- does the pre-run gate still pass?
- does the new tool produce useful output on current artifacts?
- does the doc reduce ambiguity instead of adding noise?
- would the next agent genuinely act better because this now exists?

### 5. Record the leverage

Write down:

- what future confusion or wasted work this change should prevent
- what new question it unlocks
- whether it changes the next experiment choice

## How To Decide Between Model Work And Project Work

Ask these in order:

1. Is the current next experiment already obvious from existing traces?
2. Would a small project improvement materially change that choice?
3. Is the project currently losing more time to diagnosis/infrastructure than to model iteration?

Decision rule:

- if the next experiment is obvious, do the experiment
- if the next experiment is unclear because tooling or interpretation is weak, improve the project first
- if infrastructure is failing, fix infrastructure before spending more GPU

## Current Best Guess For `exp_111`

The leading options are:

1. continue the cross-side family, but with a gentler formulation
2. retire the family and move to a different trace-targeted selection idea

If continuing the family, prefer something softer than the `exp_110` hard margin-style push. Examples of valid directions:

- lower-weight side calibration
- calibration only on clear-label bars
- side calibration tied to oracle soft mass rather than a hard side-best margin
- a symmetric penalty that discourages one side from dominating without forcing a full flip

If you continue this family, the experiment must explicitly explain why it should avoid the `35C / 127P` overcorrection from `exp_110`.

## Dataset Escalation Rule

Do not rebuild or replace `v4_exact_chain` unless one of these happens:

- at least `5` raw-input sessions are flagged by the audit trigger
- flagged dates materially overlap the worst trace days from the current candidate or baseline

Until then:

- audits are evidence
- audits are not permission to rewrite the dataset

## Akash Operational Notes

All training is remote on Akash H100.

Use:

- `./v2/ops/deploy.sh boot`
- `./v2/ops/deploy.sh start`
- `./v2/ops/deploy.sh run_screen exp_NNN`
- `./v2/ops/deploy.sh run_one exp_NNN`
- `./v2/ops/deploy.sh status`

Operational lesson from the last session:

- a saved `.deploy-state` may point to a closed lease
- if SSH refuses, verify the lease state on-chain before assuming the node is just slow
- if the wallet is low on ACT, the default boot deposit may fail; the last successful recovery used `DEPOSIT_ACT=4 ./v2/ops/deploy.sh boot`

Do not hide deployment failures. Log them as operational facts.

## Stop Conditions

Stop the autonomous loop when any of these fires:

- session limits in [v2/program.md](/Users/gduby/Documents/autoresearch-trading/v2/program.md)
- the next hypothesis cannot be justified from current traces
- the repo state is no longer trustworthy
- the evidence says a whole family should be retired and no next family has been selected yet

When stopping, produce a clean written handoff rather than one more speculative experiment.

## Minimal End-Of-Session Checklist

Before handing off, confirm:

- [ ] live code is either the kept baseline or the current intentional experiment state
- [ ] unkept code changes are reverted
- [ ] [v2/results.tsv](/Users/gduby/Documents/autoresearch-trading/v2/results.tsv) contains only official runs
- [ ] [v2/lab_notebook.md](/Users/gduby/Documents/autoresearch-trading/v2/lab_notebook.md) records every screen and its decision
- [ ] [v2/HANDOFF.md](/Users/gduby/Documents/autoresearch-trading/v2/HANDOFF.md) reflects stable current understanding
- [ ] the latest trace-backed failure mode is written down
- [ ] the next experiment is either clearly proposed or clearly deferred

## Current One-Paragraph Reality

The project has a trustworthy exact-chain baseline and a real morning-window signal, but the live scorer still fails mainly on cross-side calibration and selection quality. Kronos-inspired standalone architecture ideas did not help much; the audit tooling did. `exp_110` proved that side calibration is the right failure surface, but its first implementation simply flipped collapse from calls to puts. The next agent should either design a gentler calibration experiment for `exp_111` or explicitly retire that family before moving on.
