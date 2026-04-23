# Claude Continuous Research Prompt — v3 Trading System

Use this prompt to run a bounded continuous research loop on the current
`v3` trading stack.

## Prompt

You are working in `/Users/gduby/Documents/autoresearch-trading`.

Your job is to run a continuous hypothesis -> implementation ->
experiment -> evaluation loop on the `v3` trading system, but **only**
inside the current honest research regime.

Your goal is **not** to chase flashy PF or invent a new stack every
cycle. Your goal is to improve the real walk-forward trade quality of the
current `v3` system while preserving attribution.

## Read First

Before changing anything, read:

- [/Users/gduby/Documents/autoresearch-trading/v3/HANDOFF.md](/Users/gduby/Documents/autoresearch-trading/v3/HANDOFF.md)
- [/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)
- [/Users/gduby/Documents/autoresearch-trading/v3/layer3/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer3/README.md)
- [/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_gpu_promotion_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_gpu_promotion_2026_04_22.md)
- [/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_side_contrastive_and_layer3_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_side_contrastive_and_layer3_2026_04_22.md)

Calibrate yourself to these truths before you start:

- `V0 + time-stop` is still the honest PF champion at `1.132`
- the unified policy is **not shelved**
- unified-entry DD is materially better than `V0`
- the provisional composed-stack champion is now:
  `simulated-L3 per-seed oracle + objective-consistent unified entry + Layer-3 robust 0.90`
- side-contrastive is **not** a de-biasing workstream here; do not optimize for lower call share
- `prior_window_robust --robust-slack 0.90` is the current honest default calibrator
- oracle-peak, fixed-horizon, and global margin-offset rescue targets were all falsified as generic fixes
- threshold sweeps in Layer-3 are still exploratory unless calibrated on prior validation only

## Scope

Stay focused on these modules unless a narrow dependency requires a small
change elsewhere:

- `/Users/gduby/Documents/autoresearch-trading/v3/layer2`
- `/Users/gduby/Documents/autoresearch-trading/v3/layer3`
- `/Users/gduby/Documents/autoresearch-trading/v3/reference`
- `/Users/gduby/Documents/autoresearch-trading/v3/HANDOFF.md`

Do **not** roam into RL, archived branches, or unrelated `v2` work unless
you have a concrete dependency you can justify in one sentence.

## Current Best Workstream

Prioritize in this order:

1. Lock and preserve the provisional composed-stack champion when evaluating new work
2. Explain or reduce the mean-vs-floor tradeoff between the promoted stack and the iteration-2 challenger
3. Only after 1 and 2: broader architecture changes or larger-context training

Do **not** jump to bar-level multi-entry decisions yet. That would muddy
attribution while champion lock and outer-loop target design are still
unsettled.

Do **not** start RL or agent-style training. That is explicitly out of
scope.

## Ground Rules

1. The `13`-window rolling harness is the only promotion truth.
2. Smoke and dev runs are for hypothesis screening, plumbing, and fast ablations.
3. Keep one contract, current Layer-0 rails, and the fixed `09:45–11:30 ET` execution window unless a change is directly part of the hypothesis.
4. Teachers remain feature-only priors in the champion path.
5. Preserve attribution. Change one meaningful thing at a time.
6. Do not silently conflate baselines:
   - `V0 + time-stop`: PF reference
   - Layer-2.5 patience-gated: DD trade-quality reference
7. Do not call the unified architecture "signal-limited" unless side discrimination and exit composition have both been tested honestly.
8. Do not frame "reduce call share" as success. A side knob is only useful if it improves the composed stack's PF floor and DD.
9. Do not spend loops on global positive `decision_margin` offsets unless a new diagnosis makes the mechanism materially different from the falsified iteration-2 rescue pass.

## Required Loop

Repeat this cycle:

1. Form one narrow hypothesis.
2. State the expected mechanism in one or two sentences.
3. Make the smallest code change that tests it.
4. Run the minimum honest evaluation needed.
5. Compare against the correct baseline.
6. Record what happened in a new dated note under `v3/reference/`.
7. Update `v3/HANDOFF.md` if the result materially changes the repo's current belief.
8. Commit the cycle before moving on.

If a hypothesis fails, keep the learning and move on. Do not keep grinding
the same failed idea unless the next attempt is materially different.

## Commands You Should Actually Use

### Basic safety / verification

```bash
git status --short
.venv/bin/python -m py_compile \
  v3/layer2/unified_policy.py \
  v3/layer2/train_unified_policy.py \
  v3/layer3/common.py \
  v3/layer3/train_rolling.py
```

### Re-export the canonical action-surface dataset only if schema or labels changed

```bash
.venv/bin/python -m v3.layer2.export_action_surface_dataset
```

### Unified-policy smoke test

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier smoke \
  --device cpu \
  --run-dir v3/artifacts/layer2_unified_policy_smoke_loop
```

### Unified-policy dev rolling run, single seed

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier dev \
  --device cpu \
  --seed 42 \
  --run-dir v3/artifacts/layer2_unified_policy_dev_loop_seed42
```

### Example simulated-L3 dev rerun

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier dev \
  --device cpu \
  --seed 42 \
  --run-dir v3/artifacts/layer2_unified_policy_simL3_seed42 \
  --utility-target simulated_l3 \
  --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed42_fp.npz
```

If you are testing a new composed target family, keep the old champion as
the comparison baseline and keep the evaluation honest through Layer-3.

### Layer-3 on Layer-2.5 entries

```bash
.venv/bin/python -m v3.layer3.train_rolling \
  --out-dir v3/artifacts/layer3_rolling_entry_patience_loop
```

### Layer-3 on unified-policy chosen trades

```bash
.venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/v3_unified_promo_001/seed_42/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_loop_seed42
```

If you generate a new unified-policy run that should feed Layer-3, point
`--chosen-trades` at that run's `seed_XX/chosen_trades.pkl`.

## How to Compare Results

For unified entry-only runs, compare against:

- PF baseline: `V0 + time-stop = 1.132`
- DD context:
  - `V0 + time-stop = 95.9%` cold-start artifact
  - Layer-2.5 patience-gated baseline = `21.4%`

For unified + Layer-3 experiments, compare against the **same entry set held
to time-stop** before making any broader claim.

Do not compare a Layer-3 result directly against unrelated entry policies
without stating that the entry set changed.

For current composed-stack work, compare against:

- provisional champion:
  `simulated-L3 per-seed oracle + objective-consistent unified entry + Layer-3 robust 0.90`
- current high-mean challenger:
  iteration-2 simulated-L3 outer loop

Do not call a new branch a promotion if it only improves the mean while
regressing the PF floor.

## Commit Discipline

You must create regular rollback points.

Before long work, create or switch to a dedicated branch if needed:

```bash
git checkout -b claude/v3-continuous-loop
```

After every completed hypothesis cycle:

1. stage only relevant files
2. commit with a clear message

Example:

```bash
git add v3/layer2 v3/layer3 v3/reference v3/HANDOFF.md
git commit -m "v3 loop: tune side contrastive weight and evaluate rolling seed 42"
```

Commit frequency rules:

- commit after every successful experiment cycle
- commit after any meaningful code refactor before starting a long run
- if a run fails but the code change is still useful and stable, commit it separately as a plumbing commit
- do **not** amend old commits
- do **not** rebase, reset `--hard`, or clean unrelated files

The worktree is dirty. Be careful:

- never revert or stage unrelated user changes
- do not touch the weird duplicated archive files unless explicitly told to
- stage only the files in your current write set

## Stop Conditions

Stop and summarize instead of thrashing if any of these happen:

- three consecutive hypotheses fail without changing the repo's belief
- the next step would require changing entry regime and exit regime at the same time
- you need GPU for something that has not cleared CPU dev gates
- you are about to change dataset schema and training logic and evaluation logic all in one cycle

## What a Good Cycle Looks Like

A good cycle usually ends with:

- one code change
- one experiment artifact directory
- one reference note in `v3/reference/`
- one concise `v3/HANDOFF.md` update if warranted
- one git commit

Keep going until you reach a clean stopping point, then summarize:

- hypothesis
- change made
- exact command(s) run
- artifact path
- result vs baseline
- whether the repo's current belief changed
