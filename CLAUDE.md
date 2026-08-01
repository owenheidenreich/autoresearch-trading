# Agent Directives

> **Protocol101 Path-D status (2026-08-01):** Read
> [`v4/docs/protocol101/PATH_D_CURRENT_STATE.md`](v4/docs/protocol101/PATH_D_CURRENT_STATE.md)
> before Protocol101 model, data, training, validation, or paper work. Path D is
> in design repair; Phase-1 training/backtest and learned-trader paper readiness
> are not authorized.

## First Steps

1. Read `v2/ART2_LOOP.md` — the canonical hill-climbing protocol. This is the operating loop definition.
2. Read `v2/docs/founder_intent.md` — founder voice, standards, anti-goals.
3. Read `v2/docs/current_state.md` — current system state snapshot.
4. Read `v2/COMMANDS.md` — what the human can ask you to do.
5. Before making architecture decisions, read `v2/docs/domain/` — 0DTE options domain knowledge. Understand the instrument.

## Document Precedence (when docs disagree)

1. **Code** (`metrics.py`, `config.py`, `simulator.py`, etc.) — ground truth
2. `v2/ART2_LOOP.md` — operating loop definition
3. `v2/docs/current_state.md` — current system state
4. `v2/docs/evaluator.md` — scoring and evaluation rules
5. `v2/COMMANDS.md` — command reference
6. `v2/PIPELINE.md` — system map
7. `v2/program.md` — historical protocol (reference only, not current operating truth)
8. Everything else — reference/history

## Naming

`v2/` is the **current canonical system**. The name is historical (it replaced a v1 prototype). The GitHub repo references "v4 exact chain" which describes the *data schema version*, not a separate system. There is only one active system and it lives in `v2/`.

## Project Structure

```
root/
├── CLAUDE.md              ← you are here
├── ARCHIVE_POLICY.md      ← explains archive/ vs archive_quarantine/
├── v2/                    ← the working system (all code, data, docs)
│   ├── train.py           ← model architecture, training loop
│   ├── replay.py          ← evaluation, baselines, traces
│   ├── plot_trades.py     ← generates trades.html, equity.html, trades.csv
│   ├── plot_progress.py   ← generates progress.png
│   ├── core/              ← policy, simulator, metrics, schema, features, chain_data
│   ├── ops/               ← deploy.sh, monitor.py, model_manage.py, pre_run_gate.py
│   ├── pipeline/          ← build_v2_dataset.py, compute_features.py, download_full_chain.py
│   ├── analysis/          ← harness_eval.py, policy_sweep.py
│   ├── docs/              ← all documentation including domain knowledge
│   ├── data.pt            ← canonical dataset
│   ├── data_sidecars/     ← per-day contract snapshots and labels
│   ├── models/            ← active model checkpoints
│   ├── artifacts/         ← experiment artifacts (exp_NNN/)
│   ├── output/            ← trades.html, equity.html, progress.png, trades.csv
│   ├── ART2_LOOP.md, COMMANDS.md, program.md, PIPELINE.md
│   ├── lab_notebook.md, results.tsv
│   └── __init__.py
└── archive/               ← historical reference only, do not read unless asked
```

## Key File Map

**Training & model:**
- `v2/train.py` — model architecture (`TradingModel`), `forward()`, `compute_loss()`, training loop
- `v2/core/policy.py` — `DecisionPolicy` dataclass (stops, targets, trade window, trailing exit params)

**Trade simulation (NOT in replay.py):**
- `v2/core/simulator.py` — `simulate_trade()`, `TRAILING_TIERS`, stop/TP/trailing exit logic, MFE tracking, spread cost model
- `v2/core/schema.py` — `TradeIntent`, `SimulatedTrade` dataclasses

**Replay & evaluation:**
- `v2/replay.py` — orchestrates replay: loads model, runs inference, calls `simulator.simulate_trade()`, computes baselines, collects traces
- `v2/core/metrics.py` — score formula, hard gates, baseline computation

**Stage contracts:**
- `v2/core/config.py` — `RuntimeConfig` (single source of truth for shared constants)
- `v2/core/eval_report.py` — `EvalReport` (durable eval artifact with stored trades)

**Data pipeline:**
- `v2/core/chain_data.py` — `CONTRACT_FEATURE_FIELDS` (22 features), `build_contract_row()`, `padded_snapshot()`
- `v2/pipeline/compute_features.py` — `bs_greeks_vec()` (Black-Scholes greeks + charm), 52 context features
- `v2/pipeline/build_v2_dataset.py` — builds `data.pt` + sidecar `.pt` files, oracle label computation

**Visualization:**
- `v2/plot_trades.py` — generates trades.html, equity.html, trades.csv
- `v2/plot_progress.py` — generates progress.png
- `v2/ops/monitor.py` — live monitoring dashboard

**Operations:**
- `v2/ops/deploy.sh` — GPU lifecycle: boot/start/run_screen/run_one/stop
- `v2/ops/model_manage.py` — keep/revert promoted model
- `v2/ops/health.py` — pipeline health checks: `python -m v2.ops.health`

**Domain knowledge (read for trading context):**
- `v2/docs/domain/` — 0DTE Greeks, dealer mechanics, Pickles practitioner journal, volatility trading theory

## Experiment Pipeline

An **experiment** (exp_NNN) means: code change → commit → train from scratch on GPU → evaluate. A local replay is NOT an experiment.

1. **Edit** `v2/train.py` and/or `v2/core/policy.py` (the mutable surface)
2. **Compile check**: `python -m py_compile v2/train.py`
3. **Commit** the code change with the experiment ID in the message
4. **Pre-GPU gate**: `python3 -m v2.ops.pre_run_gate --data v2/data.pt`
5. **Boot GPU**: `./v2/ops/deploy.sh boot` then `./v2/ops/deploy.sh start`
6. **Screen** (1-fold): `./v2/ops/deploy.sh run_screen exp_NNN`
7. **If screening passes**, run official (5-fold): `./v2/ops/deploy.sh run_one exp_NNN`
8. **Post-run checklist** (see COMMANDS.md): validate, trace, keep/revert, plot, analyze, document, commit

**Local replay** (not an experiment):
- Validate model: `python3 -m v2.replay --model v2/models/model.pt --mask promote`
- Policy sweeps: modify policy, replay, compare — useful signal but cannot be promoted

**Label consistency warning:** Oracle labels in `v2/data_sidecars/` are computed with the policy active at build time. Small policy tweaks are acceptable (model learns general contract quality). Large policy changes may require a sidecar rebuild.

## Hard Rules

- **Default mutable surface:** `v2/train.py` and `v2/core/policy.py`.
- **One hypothesis per experiment.** No bundling unrelated changes.
- **Promotion is score-gated.** Must beat all four baselines with no gate failure.
- **Every experiment trains from scratch.** No warm-starting.
- **Log everything** in `v2/results.tsv` and `v2/lab_notebook.md`.
- **All training runs on Akash H100 GPU, never locally.**
- **Run harness eval before GPU spend.**
- **Do not read `archive/` as default context.** Only consult when explicitly asked.

## Code Quality

- Run `python -m py_compile <file>` on every changed file before reporting complete.
- Commit every time a change is made.
- Re-read files before editing (especially after 10+ messages).

## Live Documentation

See `v2/docs/README.md` for the full documentation index.
