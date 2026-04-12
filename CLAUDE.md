# Agent Directives

## First Steps

1. Read `v2/HANDOFF.md` — current state and trust boundaries.
2. Read `v2/docs/founder_intent.md` — founder voice, standards, anti-goals.
3. Read `v2/program.md` — the definitive protocol.
4. Read `v2/COMMANDS.md` — what the human can ask you to do.

## Key File Map (read this before searching the codebase)

**Training & model:**
- `v2/train.py` — model architecture (`TradingModel`), `forward()`, `compute_loss()`, training loop
- `v2/core/policy.py` — `DecisionPolicy` dataclass (stops, targets, trade window, trailing exit params)

**Trade simulation (NOT in replay.py):**
- `v2/core/simulator.py` — `simulate_trade()`, `TRAILING_TIERS`, stop/TP/trailing exit logic, MFE tracking, spread cost model
- `v2/core/schema.py` — `TradeIntent`, `SimulatedTrade`, `OpenPosition` dataclasses

**Replay & evaluation:**
- `v2/replay.py` — orchestrates replay: loads model, runs inference, calls `simulator.simulate_trade()`, computes baselines, collects traces
- `v2/core/metrics.py` — score formula, hard gates, baseline computation

**Data pipeline:**
- `v2/core/chain_data.py` — `CONTRACT_FEATURE_FIELDS` (15 features), `build_contract_row()`, `padded_snapshot()`
- `v2/pipeline/compute_features.py` — `bs_greeks_vec()` (Black-Scholes greeks), 47 context features
- `v2/pipeline/build_v2_dataset.py` — builds `data.pt` + sidecar `.pt` files, oracle label computation

**Operations:**
- `v2/ops/deploy.sh` — GPU lifecycle: boot/start/run_screen/run_one/stop
- `v2/ops/model_manage.py` — keep/revert promoted model

**Full layout:** `v2/LAYOUT.md`

## Hard Rules

- **Default mutable surface:** `v2/train.py` and `v2/core/policy.py`.
- **Expanded surface is allowed only when the active protocol/hypothesis requires it.** For the current side-collapse reset this includes `v2/core/metrics.py`, `v2/replay.py`, `v2/core/data_integrity.py`, `v2/ops/pre_run_gate.py`, and the live docs that must stay in sync.
- **One hypothesis per experiment.** Coordinated edits are allowed when they are inseparable parts of the same hypothesis.
- **Promotion is score-gated.** Score = `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Must also beat all four baselines. Traces may justify continuing a hypothesis family, but not promoting it.
- **Direction mix is diagnostic, not a hard score gate.** Always report call count, put count, minority share, and direction balance.
- **Every experiment trains from scratch.** No warm-starting.
- **Log everything** in `v2/results.tsv` and `v2/lab_notebook.md`.
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

See `v2/docs/README.md` for the full index of current, accurate documentation.
