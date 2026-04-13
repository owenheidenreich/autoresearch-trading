# Agent Directives

## First Steps

1. Read `v2/HANDOFF.md` — current state, live code, what to trust.
2. Read `v2/docs/founder_intent.md` — founder voice, standards, anti-goals.
3. Read `v2/program.md` — the definitive protocol.
4. Read `v2/COMMANDS.md` — what the human can ask you to do.
5. Before making architecture decisions, read `docs/domain/` — 0DTE options domain knowledge. Understand the instrument.

## Project Structure

This project has two ML approaches to SPX 0DTE trading, with shared infrastructure:

- `supervised/` (aliased as `v2/`) — supervised learning, 144 experiments, promoted model
- `rl/` (aliased as `v3/`) — reinforcement learning (PPO), experimental
- `shared/` — data infrastructure facade used by both approaches
- `docs/` — all documentation including domain knowledge
- `output/` → `supervised/output/` — trade visualizations, equity curves
- `analysis/` → `supervised/analysis/` — harness eval, policy sweeps
- `models/` → `supervised/models/` — active model checkpoints
- `plots/` — visualization scripts (plot_trades.py, plot_progress.py)

**Symlinks for import compatibility:** `v2` → `supervised`, `v3` → `rl`. All Python imports use `v2.*` / `v3.*` and resolve through these symlinks.

## Key File Map (read this before searching the codebase)

**Training & model (supervised):**
- `supervised/train.py` — model architecture (`TradingModel`), `forward()`, `compute_loss()`, training loop
- `supervised/core/policy.py` — `DecisionPolicy` dataclass (stops, targets, trade window, trailing exit params)

**Trade simulation (NOT in replay.py):**
- `supervised/core/simulator.py` — `simulate_trade()`, `TRAILING_TIERS`, stop/TP/trailing exit logic, MFE tracking, spread cost model
- `supervised/core/schema.py` — `TradeIntent`, `SimulatedTrade`, `OpenPosition` dataclasses

**Replay & evaluation:**
- `supervised/replay.py` — orchestrates replay: loads model, runs inference, calls `simulator.simulate_trade()`, computes baselines, collects traces
- `supervised/core/metrics.py` — score formula, hard gates, baseline computation

**Data pipeline:**
- `supervised/core/chain_data.py` — `CONTRACT_FEATURE_FIELDS` (15 features), `build_contract_row()`, `padded_snapshot()`
- `supervised/pipeline/compute_features.py` — `bs_greeks_vec()` (Black-Scholes greeks), 47 context features
- `supervised/pipeline/build_v2_dataset.py` — builds `data.pt` + sidecar `.pt` files, oracle label computation

**Operations:**
- `supervised/ops/deploy.sh` — GPU lifecycle: boot/start/run_screen/run_one/stop
- `supervised/ops/model_manage.py` — keep/revert promoted model

**Shared data layer (used by both approaches):**
- `shared/chain_data.py` — contract representation, sidecar loading (re-exports from `supervised.core.chain_data`)
- `shared/features.py` — feature constants and indices (re-exports from `supervised.core.features`)
- `shared/market_data.py` — raw cache paths and loading (re-exports from `supervised.pipeline.build_v2_dataset`)
- `shared/option_math.py` — Black-Scholes IV, ATM finding, flow features (re-exports from `supervised.pipeline.compute_features`)

**Domain knowledge (read for trading context):**
- `docs/domain/` — 0DTE Greeks, dealer mechanics, Pickles practitioner journal, volatility trading theory

**Full layout:** `supervised/LAYOUT.md`

## Experiment Pipeline (read this before running anything)

An **experiment** (exp_NNN) means: code change → commit → train from scratch on GPU → evaluate. A local replay is NOT an experiment.

**The pipeline:**
1. **Edit** `supervised/train.py` and/or `supervised/core/policy.py` (the mutable surface)
2. **Compile check**: `python -m py_compile supervised/train.py`
3. **Commit** the code change with the experiment ID in the message
4. **Pre-GPU gate**: `python3 -m v2.ops.pre_run_gate --data v2/data.pt`
5. **Boot GPU**: `./supervised/ops/deploy.sh boot` then `./supervised/ops/deploy.sh start`
6. **Screen** (1-fold): `./supervised/ops/deploy.sh run_screen exp_NNN`
7. **If screening passes**, run official (5-fold): `./supervised/ops/deploy.sh run_one exp_NNN`
8. **Post-run checklist** (see COMMANDS.md "Post-Run Checklists"): validate, trace, keep/revert, plot, analyze, document, commit

**What local replay is for:**
- Validating a promoted model: `python3 -m v2.replay --model v2/models/model.pt --mask promote`
- Generating traces: add `--traces` flag
- Comparing baselines: baselines are computed automatically during replay
- Policy sweeps (testing a policy change without retraining): useful for quick directional signal, but NOT an official result

**Critical: policy changes still require a full GPU run.** Changing `policy.py` affects how trades are executed during evaluation, but the model was trained under the OLD policy. For a fair evaluation, you must retrain from scratch so the training labels and evaluation policy are consistent. A local replay with changed policy is a useful preview but cannot be promoted.

**Label consistency warning:** Oracle labels in `supervised/data_sidecars/` are computed by `build_v2_dataset.py` using `simulate_trade()` with whatever policy was active at build time. If you change trailing tiers, stop percentages, or trade windows in `policy.py` or `simulator.py`, the training labels no longer match the evaluation policy. This is acceptable for small policy tweaks (the model learns general contract quality, not policy-specific P&L), but large policy changes may require a sidecar rebuild.

## Hard Rules

- **Default mutable surface:** `supervised/train.py` and `supervised/core/policy.py`.
- **Expanded surface is allowed only when the active protocol/hypothesis requires it.** For the current side-collapse reset this includes `supervised/core/metrics.py`, `supervised/replay.py`, `supervised/core/data_integrity.py`, `supervised/ops/pre_run_gate.py`, and the live docs that must stay in sync.
- **One hypothesis per experiment.** Coordinated edits are allowed when they are inseparable parts of the same hypothesis.
- **Promotion is score-gated.** Score = `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Must also beat all four baselines. Traces may justify continuing a hypothesis family, but not promoting it.
- **Direction mix is diagnostic, not a hard score gate.** Always report call count, put count, minority share, and direction balance.
- **Every experiment trains from scratch.** No warm-starting.
- **Log everything** in `supervised/results.tsv` and `supervised/lab_notebook.md`.
- **All training runs on Akash H100 GPU, never locally.**
- **Run harness eval before GPU spend.**
- **Do not read `archive/` as default context.** Only consult it when the human explicitly asks for historical context.

## Code Quality

- Before reporting a task complete, run `python -m py_compile <file>` on every changed file.
- Commit every time a change is made.
- Before editing a file, re-read it. After editing, verify the change applied.
- One hypothesis per experiment. Do not bundle unrelated changes.

## Context Management

- After 10+ messages, re-read any file before editing.
- If delegation is explicitly allowed, parallelize independent work.
- File reads are capped at 2,000 lines. Use offset/limit for larger files.

## Live Documentation

See `docs/README.md` for the full index of current, accurate documentation.
