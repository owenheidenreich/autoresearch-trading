# Context Bundle — SPX 0DTE Trading Research System

Generated 2026-04-15 after pipeline integrity fix + post-reduction tightening.

---

## 1. Repo Tree

```
root/
├── CLAUDE.md                          # Agent bootstrap (mandatory read)
├── ARCHIVE_POLICY.md                  # Explains archive/ vs archive_quarantine/
├── .env                               # API keys
├── .gitignore
├── .python-version
├── pyproject.toml                     # Dependencies
├── uv.lock
├── archive/                           # Historical v1/v3 (read-only)
├── archive_quarantine/                # Quarantined stale files (2026-04-15)
│   ├── ARCHIVE_MANIFEST.md
│   ├── artifacts/                     # 85 old experiment bundles
│   ├── context_bundle/                # Stale code copies
│   ├── docs/                          # Stale session handoff docs
│   ├── live/                          # Broken IBKR module (imports training.*)
│   └── models/                        # 32 stale model checkpoints
│
└── v2/                                # === THE ACTIVE SYSTEM ===
    ├── __init__.py
    │
    ├── core/                          # Shared contracts + logic
    │   ├── __init__.py
    │   ├── config.py                  # RuntimeConfig (single source of truth) [NEW]
    │   ├── schema.py                  # TradeIntent, SimulatedTrade
    │   ├── policy.py                  # DecisionPolicy, DEFAULT_POLICY
    │   ├── features.py                # FEATURE_NAMES (52), normalization
    │   ├── chain_data.py              # CONTRACT_FEATURE_FIELDS (22), sidecars
    │   ├── simulator.py               # simulate_trade(), spread cost model
    │   ├── metrics.py                 # compute_metrics() (DOLLAR-WEIGHTED PF) [FIXED]
    │   ├── eval_report.py             # EvalReport with stored trades [NEW]
    │   ├── env.py                     # TradingEnv (RL), adaptive spread [FIXED]
    │   ├── labels.py                  # Oracle labeler
    │   ├── candidates.py              # Dynamic contract candidates
    │   ├── walkforward.py             # Walk-forward CV harness
    │   ├── trajectory_buffer.py       # AWAC trajectory dataset
    │   ├── dataset_fingerprint.py     # Deterministic fingerprinting
    │   ├── data_integrity.py          # Data validation
    │   └── decision_trace.py          # Per-bar decision logging
    │
    ├── pipeline/                      # Data acquisition + build
    │   ├── __init__.py
    │   ├── download_full_chain.py     # Raw data from Polygon
    │   ├── compute_features.py        # 52 context features
    │   ├── build_v2_dataset.py        # data.pt + sidecars [UPDATED: config_fingerprint]
    │   └── sources/__init__.py
    │
    ├── train.py                       # Supervised TradingModel [UPDATED: validation + env capture]
    ├── train_seq.py                   # Sequential agent BC
    ├── train_awac.py                  # AWAC offline RL
    ├── seq_agent.py                   # SequentialAgent architecture
    ├── collect_trajectories.py        # Offline trajectory collection
    ├── replay.py                      # Replay evaluation [FIXED: qty, EvalReport]
    ├── plot_trades.py                 # trades.html, equity.html
    ├── plot_progress.py               # progress.png
    │
    ├── ops/                           # Operations + automation
    │   ├── __init__.py
    │   ├── deploy.sh                  # Akash GPU lifecycle [UPDATED: exclude archive_quarantine]
    │   ├── health.py                  # Pipeline health checks [NEW]
    │   ├── pre_run_gate.py            # Pre-GPU safety gate
    │   ├── artifact.py                # Artifact bundle management
    │   ├── run_experiment_wf.py       # 5-fold walk-forward experiment
    │   ├── autoresearch.py            # Bounded overnight loop
    │   ├── autoresearch_config.json   # Experiment queue config
    │   ├── model_manage.py            # Keep/revert promoted model
    │   ├── monitor.py                 # Live dashboard
    │   ├── preflight.py               # Remote GPU validation
    │   ├── status_report.py           # Deployment diagnostics
    │   ├── lease_check.py             # Akash lease management
    │   ├── check_docs_stale.sh        # Doc staleness hook
    │   ├── deploy-autoresearch.yaml   # Akash YAML
    │   └── requirements-gpu.txt       # GPU pip requirements
    │
    ├── analysis/                      # Evaluation utilities
    │   ├── __init__.py
    │   ├── harness_eval.py            # Regression test harness
    │   ├── policy_sweep.py            # Policy parameter sweeps
    │   ├── frontier_study.py          # Agent frontier analysis
    │   ├── fold_diagnosis.py          # Per-fold diagnostics
    │   ├── flip_day_study.py          # Side-flip forensics
    │   ├── analyze_losses.py          # Losing-day analysis
    │   ├── behavioral_report.py       # Trade behavior diagnostics
    │   └── grid_study.py              # Hyperparameter grid search
    │
    ├── data.pt                        # Canonical dataset (169 MB, 382K bars)
    ├── data.pt.sha256                 # Dataset integrity hash
    ├── data_sidecars/                 # 986 per-day contract snapshots (1.9 GB)
    ├── models/
    │   ├── model.pt                   # Current promoted checkpoint
    │   └── README.md
    ├── artifacts/                     # (empty — exp_146 archived)
    ├── trajectories/                  # AWAC offline RL data (2.7 GB)
    ├── traj_k4_fold0/                 # K4 trajectory variant (407 MB)
    ├── output/                        # Generated HTML/PNG/CSV (regenerated)
    ├── state/                         # Loop state (transient)
    ├── harness_eval/suite.json        # Regression test cases
    ├── results.tsv                    # Experiment ledger
    ├── lab_notebook.md                # Experiment narrative
    │
    ├── docs/                          # Documentation
    │   ├── README.md                  # Doc index
    │   ├── founder_intent.md          # Non-negotiable standards
    │   ├── how_training_works.md      # Current architecture
    │   ├── current_state.md           # Research position
    │   ├── open_questions.md          # Active questions
    │   ├── decision_log.md            # Key decisions
    │   ├── data_contract.md           # Dataset schema
    │   ├── feature_schema.md          # Feature definitions
    │   ├── evaluator.md               # Evaluator docs
    │   ├── labeling.md                # Label scheme
    │   └── domain/                    # 0DTE domain knowledge (5 files)
    │
    ├── HANDOFF.md                     # Session bootstrap
    ├── COMMANDS.md                    # Command reference
    ├── KNOWN_ISSUES.md                # Active bugs and stale artifacts
    ├── CRITICAL_PF_BUG_INVESTIGATION.md  # Detailed PF bug diagnosis (ref'd by KNOWN_ISSUES)
    ├── CONTEXT_BUNDLE.md              # This file
    └── program.md                     # Operating protocol
```

---

## 2. Canonical Commands

| Stage | Command |
|-------|---------|
| Dataset build | `python -m v2.pipeline.build_v2_dataset` |
| Supervised training | `python -m v2.train --data v2/data.pt` |
| Sequential BC | `python -m v2.train_seq` |
| Trajectory collection | `python -m v2.collect_trajectories` |
| AWAC training | `python -m v2.train_awac` |
| Replay / eval | `python -m v2.replay --model v2/models/model.pt --mask promote` |
| Sequential replay | `python -m v2.replay --sequential --seq-model v2/models/seq_agent.pt` |
| Trade plots | `python -m v2.plot_trades` |
| Progress plot | `python -m v2.plot_progress` |
| Harness eval | `python -m v2.analysis.harness_eval --data v2/data.pt` |
| Pre-GPU gate | `python -m v2.ops.pre_run_gate --data v2/data.pt` |
| 5-fold experiment | `python -m v2.ops.run_experiment_wf exp_NNN` |
| GPU deploy | `./v2/ops/deploy.sh boot / start / run_screen exp_NNN / run_one exp_NNN` |
| Autoresearch | `python -m v2.ops.autoresearch` |
| Keep/revert model | `python -m v2.ops.model_manage keep exp_NNN` |
| Health check (full) | `python -m v2.ops.health` |
| Health check (quick) | `python -m v2.ops.health quick` |
| Policy sweep | `python -m v2.analysis.policy_sweep` |

---

## 3. Source-of-Truth File List

| Artifact | Path | Role |
|----------|------|------|
| **Dataset** | `v2/data.pt` | Canonical training/eval data (382K bars, 52 features, 986 days) |
| **Sidecars** | `v2/data_sidecars/*.pt` | Per-day contract chain snapshots (22 contract features) |
| **Promoted model** | `v2/models/model.pt` | Current promoted checkpoint (stale — needs re-evaluation) |
| **Results ledger** | `v2/results.tsv` | Official experiment outcomes |
| **Lab notebook** | `v2/lab_notebook.md` | Experiment narrative and analysis |
| **Runtime config** | `v2/core/config.py` | Single source of truth for shared constants |
| **Schema** | `v2/core/schema.py` | TradeIntent + SimulatedTrade (the universal contract) |
| **Policy** | `v2/core/policy.py` | DecisionPolicy (stops, targets, trailing, risk limits) |
| **Feature contract** | `v2/core/features.py` | FEATURE_NAMES (52 features, canonical order) |
| **Contract features** | `v2/core/chain_data.py` | CONTRACT_FEATURE_FIELDS (22 fields) |
| **Score formula** | `v2/core/metrics.py` | compute_score() — now dollar-weighted PF |
| **Spread cost** | `v2/core/simulator.py` | _compute_spread_cost() — adaptive model |
| **Eval reports** | `v2/output/eval_report.json` | Durable eval artifacts with stored trades |
| **Harness tests** | `v2/harness_eval/suite.json` | Regression test cases |

---

## 4. Active Docs List

| Doc | Path | Purpose | Referenced by |
|-----|------|---------|---------------|
| HANDOFF.md | `v2/HANDOFF.md` | Session bootstrap | CLAUDE.md (mandatory) |
| COMMANDS.md | `v2/COMMANDS.md` | Command reference | CLAUDE.md (mandatory) |
| program.md | `v2/program.md` | Operating protocol | CLAUDE.md (mandatory) |
| KNOWN_ISSUES.md | `v2/KNOWN_ISSUES.md` | Active bugs, stale artifacts | CLAUDE.md (mandatory) |
| PF Bug Investigation | `v2/CRITICAL_PF_BUG_INVESTIGATION.md` | Detailed PF bug diagnosis | Referenced by KNOWN_ISSUES |
| ARCHIVE_POLICY.md | `ARCHIVE_POLICY.md` | Archive semantics | Root-level, CLAUDE.md |
| Founder intent | `v2/docs/founder_intent.md` | Non-negotiable standards | CLAUDE.md, pre_run_gate |
| How training works | `v2/docs/how_training_works.md` | Architecture description | pre_run_gate (content check) |
| Current state | `v2/docs/current_state.md` | Research position | docs/README.md |
| Open questions | `v2/docs/open_questions.md` | Active questions | pre_run_gate (existence check) |
| Decision log | `v2/docs/decision_log.md` | Key decisions | pre_run_gate (existence check) |
| Data contract | `v2/docs/data_contract.md` | Dataset schema | docs/README.md |
| Feature schema | `v2/docs/feature_schema.md` | Feature definitions | docs/README.md |
| Evaluator | `v2/docs/evaluator.md` | Score formula docs | docs/README.md |
| Labeling | `v2/docs/labeling.md` | Label scheme | docs/README.md |
| 0DTE domain | `v2/docs/domain/*.md` | Options domain knowledge | CLAUDE.md (domain ref) |

---

## 5. Sample Artifact Layout

When a 5-fold experiment completes, the artifact bundle contains:

```
v2/artifacts/exp_NNN/
├── manifest.json              # Metadata + fingerprints + score + promotion status
│   {
│     "experiment_id": "exp_NNN",
│     "git_sha": "abc1234",
│     "timestamp": "2026-04-16T03:45:00",
│     "score": 0.823,
│     "promoted": true,
│     "dataset_fingerprint": "b422dcbce1d15cfc",
│     "evaluator_fingerprint": "e45320ccab...",
│     "policy_fingerprint": "7f3a2b...",
│     "model_fingerprint": "d9e1f2...",
│     "config_fingerprint": "a1b2c3...",          # [NEW]
│     "hyperparams": {
│       "lookback": 30, "d_model": 96, "depth": 3,
│       "n_heads": 4, "dropout": 0.05,
│       "contract_features": 22, "max_contracts_per_bar": 100
│     },
│     "env_overrides": {                           # [NEW]
│       "SOFT_TEMP": "0.15", "GATE_W": "0.4"
│     },
│     "checkpoint_epoch": 18,
│     "checkpoint_val_loss": 0.342
│   }
│
├── model.pt                   # Trained checkpoint (state_dict + hyperparams)
├── policy.json                # Frozen DecisionPolicy snapshot
├── train.py.snapshot          # Source code at training time
├── policy.py.snapshot         # Source code at training time
│
└── eval_report.json           # [NEW] Full evaluation with stored trades
    {
      "report_id": "a1b2c3d4",
      "report_timestamp": "2026-04-16T04:00:00",
      "experiment_id": "exp_NNN",
      "evaluator_fingerprint": "e45320ccab...",
      "dollar_pf": 1.24,
      "dollar_net_pnl": 1842.50,
      "pct_pf": 1.48,
      "score": 0.823,
      "gate_failure": null,
      "beats_all_baselines": true,
      "baselines": {"random": 0.12, "atm_always": 0.31, ...},
      "trades": [ ... 761 SimulatedTrade dicts ... ],
      "daily_equity_curve": [10000.0, 10125.0, 9980.0, ...],
      "metrics": { ... full ReplayMetrics snapshot ... },
      "per_fold": [ ... 5 fold metric dicts ... ]
    }
```

### Key properties of the new artifact layout:
- **Stored trades** enable re-scoring without re-replaying
- **config_fingerprint** detects when model was trained under different constants
- **evaluator_fingerprint** detects when score formula has changed since evaluation
- **env_overrides** captures exactly which training vars were set
- **daily_equity_curve** enables plot regeneration from report without replay
