# CLAUDE.md — Agent Directives

This file is loaded into every Claude Code session. Keep it short. Detail lives in the linked live docs.

## Mission

Build a **trustworthy exact-chain research system** for SPX 0DTE long options. Not a live trading bot. Not yet.

The things we refuse to fake:
- Treating bad data as good because a score looks exciting.
- Treating stale, pre-reset, or incompatible scores as live evidence.
- Relying on model "memory" when information should be written down.
- Letting documentation drift until nobody knows what's real.

If a result cannot be defended from **current data, current code, and current artifacts**, it is not evidence. See [v2/docs/founder_intent.md](v2/docs/founder_intent.md) — when trade-offs are unclear, that file wins over convenience.

## Start every session by reading

1. [v2/ART2_LOOP.md](v2/ART2_LOOP.md) — canonical hill-climbing protocol (preconditions → training → validation → promotion). The operating loop definition.
2. [v2/docs/current_state.md](v2/docs/current_state.md) — what IS true right now (dataset, evaluator, phase).
3. [v2/COMMANDS.md](v2/COMMANDS.md) — the exact shell commands behind user requests.
4. [v2/docs/founder_intent.md](v2/docs/founder_intent.md) — non-negotiable standards.

Before any architecture or modeling decision, also read [v2/docs/domain/](v2/docs/domain/) — understand the instrument before changing how we trade it.

## Document precedence (when docs disagree)

1. **Code** (`v2/core/metrics.py`, `v2/core/config.py`, `v2/core/simulator.py`, `v2/train.py`) — ground truth
2. [v2/ART2_LOOP.md](v2/ART2_LOOP.md) — operating loop
3. [v2/docs/current_state.md](v2/docs/current_state.md) — current state snapshot
4. [v2/docs/evaluator.md](v2/docs/evaluator.md) — scoring & evaluation
5. [v2/COMMANDS.md](v2/COMMANDS.md) — command reference
6. [v2/PIPELINE.md](v2/PIPELINE.md) — system map
7. [v2/program.md](v2/program.md) — historical protocol (reference only)
8. Everything else — reference/history

If a doc contradicts the code, the code wins. Flag the drift.

## Repo layout

`v2/` is the **only live system.** The "v2" name is historical (it replaced a v1 prototype). "v4 exact chain" refers to the *data schema version*, not a separate system.

```
root/
├── CLAUDE.md              ← this file
├── ARCHIVE_POLICY.md      ← archive/ vs archive_quarantine/ rules
├── v2/                    ← the working system
│   ├── train.py, train_seq.py, train_awac.py
│   ├── replay.py, seq_agent.py, collect_trajectories.py
│   ├── plot_trades.py, plot_progress.py
│   ├── core/              ← policy, simulator, metrics, schema, features, chain_data, config
│   ├── ops/               ← deploy.sh, health.py, model_manage.py, pre_run_gate.py, monitor.py
│   ├── pipeline/          ← build_v2_dataset.py, compute_features.py, download_full_chain.py
│   ├── analysis/          ← harness_eval.py, policy_sweep.py, analyze_losses.py, …
│   ├── docs/              ← all live documentation (incl. domain/)
│   ├── data.pt, data_sidecars/, data.pt.sha256
│   ├── models/, artifacts/, runs/, trajectories/, build_reports/
│   ├── output/            ← trades.html, equity.html, progress.png, trades.csv
│   ├── ART2_LOOP.md, COMMANDS.md, PIPELINE.md, program.md, HANDOFF.md
│   └── lab_notebook.md, results.tsv
├── archive/               ← historical — DO NOT READ unless explicitly asked
└── archive_quarantine/    ← see ARCHIVE_POLICY.md
```

## Key file map (by purpose)

**Training & policy**
- [v2/train.py](v2/train.py) — `TradingModel`, `forward()`, `compute_loss()`, supervised loop
- [v2/train_seq.py](v2/train_seq.py) — sequential BC stage
- [v2/train_awac.py](v2/train_awac.py) — AWAC RL stage (current regime)
- [v2/core/policy.py](v2/core/policy.py) — `DecisionPolicy` (stops, TPs, trade window, trailing)

**Trade simulation & replay**
- [v2/core/simulator.py](v2/core/simulator.py) — `simulate_trade()`, `TRAILING_TIERS`, MFE, spread cost model
- [v2/core/schema.py](v2/core/schema.py) — `TradeIntent`, `SimulatedTrade`
- [v2/replay.py](v2/replay.py) — loads model, runs inference, calls simulator, baselines, traces
- [v2/core/metrics.py](v2/core/metrics.py) — v4.0 dollar-weighted PF/Sortino score, hard gates, baselines

**Contracts & config**
- [v2/core/config.py](v2/core/config.py) — `RuntimeConfig` (single source of truth for shared constants)
- [v2/core/eval_report.py](v2/core/eval_report.py) — `EvalReport` (durable eval artifact with stored trades)
- [v2/core/artifact_kind.py](v2/core/artifact_kind.py) — `cv_eval` / `fold_checkpoint` / `final_train`

**Data pipeline**
- [v2/core/chain_data.py](v2/core/chain_data.py) — `CONTRACT_FEATURE_FIELDS` (22 features), `build_contract_row()`, `padded_snapshot()`
- [v2/pipeline/compute_features.py](v2/pipeline/compute_features.py) — `bs_greeks_vec()`, 52 context features
- [v2/pipeline/build_v2_dataset.py](v2/pipeline/build_v2_dataset.py) — builds `data.pt` + sidecars, oracle labels
- [v2/core/data_integrity.py](v2/core/data_integrity.py) — manifest/feature/sidecar audits, side-bias audit

**Operations**
- [v2/ops/deploy.sh](v2/ops/deploy.sh) — GPU lifecycle (boot/start/stop/status and all `run_*` variants)
- [v2/ops/health.py](v2/ops/health.py) — pipeline health check (`python3 -m v2.ops.health`)
- [v2/ops/pre_run_gate.py](v2/ops/pre_run_gate.py) — pre-GPU integrity gate
- [v2/ops/model_manage.py](v2/ops/model_manage.py) — keep/revert (refuses non-`final_train` artifacts)

## Experiment pipeline (current — post 2026-04-17 harness repair)

An **experiment** means: code change → commit → screen → (if justified) CV → (if passed) final-train → keep/revert. A local replay is **not** an experiment.

The old single-shot `run_screen` / `run_one` flow has been replaced by a scope-separated pipeline. Screens debug; CV selects configs; `final_train` produces the only promotable artifact.

1. **Edit** `v2/train.py` and/or `v2/core/policy.py` (default mutable surface).
2. **Compile check:** `python -m py_compile v2/train.py v2/core/policy.py`.
3. **Harness eval + pre-run gate:**
   `python3 -m v2.analysis.harness_eval --data v2/data.pt`
   `python3 -m v2.ops.pre_run_gate --data v2/data.pt`
4. **Commit** the code change with the experiment ID.
5. **Boot GPU:** `./v2/ops/deploy.sh boot` then `./v2/ops/deploy.sh start`.
6. **Screen** (no artifact):
   - Parity debug: `./v2/ops/deploy.sh run_screen_latest exp_NNN` (fold 4 only)
   - Regime triage: `./v2/ops/deploy.sh run_screen_mini exp_NNN` (folds 0, 2, 4)
7. **Full CV** (emits a `CV_EVAL` artifact — NOT deployable):
   `./v2/ops/deploy.sh run_cv exp_NNN`
8. **Final train** (emits a `FINAL_TRAIN` — the only kind `keep` accepts):
   `./v2/ops/deploy.sh run_final_train exp_NNN`
9. **Validate → trace → keep/revert → plot → analyze → lab-notebook → commit.** Full post-run checklist lives in [v2/COMMANDS.md](v2/COMMANDS.md#post-run-checklists). Do not report results to the user until every step is complete.

**Label consistency:** oracle labels in `v2/data_sidecars/` are computed with the policy active at build time. Small policy tweaks are acceptable; large changes require a sidecar rebuild.

## Hard rules

- **Default mutable surface:** `v2/train.py` and `v2/core/policy.py`. Expanded surfaces are enumerated in [v2/docs/current_state.md](v2/docs/current_state.md); don't silently widen it.
- **One hypothesis per experiment.** No bundling.
- **Every experiment trains from scratch.** No warm-starting.
- **Promotion is score-gated AND trace-reviewed.** Must beat all four baselines with no gate failure. Degenerate behavior (all-one-side, single-exit-type) gets reverted even if the score is green.
- **Only `final_train` artifacts are deployable.** `cv_eval` and `fold_checkpoint` cannot be kept.
- **All training runs on Akash H100. Never train locally.**
- **Log every run** in `v2/lab_notebook.md` and (for official runs) `v2/results.tsv`.
- **Harness eval before any GPU spend.** This is non-negotiable.
- **Commit after every meaningful change.** Include the experiment ID in the message.
- **Do not read `archive/` by default.** Only when explicitly asked.
- **Use `deploy.sh` commands; don't manually SSH/SCP to replicate them.** If one fails, re-run it — don't open-code the steps.

## Anti-patterns (things that have bitten this project)

- **Reward hacking** — optimizing the score without a root-cause for the behavior change. Find *why* first.
- **Regime gating / difficulty weighting** — the model must learn from data like a trader would. No hand-tuned regime switches.
- **Silent policy/sidecar drift** — if you change labeling or schema, rebuild the dataset; do not run on stale artifacts.
- **Mock databases / mock sidecars** — audit real artifacts; no stubs in the promotion path.
- **Spamming GPU status checks** — once per minute maximum when waiting on training.
- **Skipping harness eval** before deploying. Don't.
- **Claiming completion before plots + lab notebook + commit.** Incomplete work presented as done is worse than no work.

## Code quality

- Run `python -m py_compile <file>` on every file you change before reporting complete.
- Use dedicated tools (Read, Edit, Grep, Glob) over Bash for file ops.
- Re-read files before editing them after long conversations — state drifts.
- Don't add speculative helpers, backwards-compat shims, or comments explaining *what* well-named code already says.

## When to stop and ask

Mandatory pause conditions (from [v2/ART2_LOOP.md](v2/ART2_LOOP.md)): evaluator fingerprint changed, dataset fingerprint changed, docs disagree on score/gates/stack, last promotion packet is incomplete, health checks failing, or 6 consecutive no-improve experiments. Do not start the next experiment under any of these — fix the foundation first.

## Live documentation index

Full index: [v2/docs/README.md](v2/docs/README.md).

- Scoring & gates → [v2/docs/evaluator.md](v2/docs/evaluator.md)
- Data structure & integrity → [v2/docs/data_contract.md](v2/docs/data_contract.md)
- Feature schema (52 ctx) → [v2/docs/feature_schema.md](v2/docs/feature_schema.md)
- Sidecar labels → [v2/docs/labeling.md](v2/docs/labeling.md)
- Training internals → [v2/docs/how_training_works.md](v2/docs/how_training_works.md)
- Open questions → [v2/docs/open_questions.md](v2/docs/open_questions.md)
- Durable decisions → [v2/docs/decision_log.md](v2/docs/decision_log.md)
- Incidents & post-mortems → [v2/docs/incidents/](v2/docs/incidents/)
