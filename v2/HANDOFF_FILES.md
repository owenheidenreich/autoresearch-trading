# Active Files

## Core agent path
- `v2/core/env.py` — TradingEnv, session state (13-dim), decay penalty, late-session constraints
- `v2/seq_agent.py` — SequentialAgent (frozen encoder + session_proj + policy/value heads)
- `v2/train_seq.py` — BC training, REINFORCE with side-balance + entry cost + KL anchor
- `v2/replay.py` — replay_sequential(), --sequential CLI, --zero-state ablation, behavioral report
- `v2/train.py` — TradingModel (frozen encoder), contract scoring heads

## Session state features (13-dim)
```
[0]  bar_of_session_norm
[1]  in_position
[2]  bars_held_norm
[3]  unrealized_pnl_frac
[4]  position_mfe_frac
[5]  position_mae_frac
[6]  day_pnl_frac
[7]  num_trades_today_norm
[8]  num_stops_today_norm
[9]  bars_since_last_trade
[10] best_score_this_bar
[11] position_side
[12] best_call_score - best_put_score  ← NEW (Side13)
```

## Environment constraints (configurable via env vars)
- `ENV_LATE_ENTRY_BAR=89` — block new entries after bar 89
- `ENV_LATE_EXIT_BAR=95` — force-close underwater zombies (MFE<2%) after bar 95
- `ENV_DECAY_COEFF=0.001` — per-bar holding penalty for stagnating losers

## RL reward components
- Escalating entry cost: `RL_ENTRY_COST=0.015` + `RL_ENTRY_ESCALATION=0.015` per prior entry
- Side-imbalance penalty: `RL_SIDE_IMBALANCE=0.03` (episode-end, ≥2 entries)
- KL anchor to BC: `RL_KL_COEFF=0.03`
- Behavioral band checkpoint selection: entries/day in [1.0, 3.0], flips < 1.5

## Oracle (BC teacher)
- `v2/train_seq.py` `_compute_oracle_actions()` with thesis-persistence rules:
  - Side-commitment lockout: `ORACLE_SIDE_LOCKOUT=5` bars
  - Daily entry cap: `ORACLE_MAX_ENTRIES=4`

## Analysis and diagnostics
- `v2/analysis/behavioral_report.py` — auto-prints after sequential replay
- `v2/analysis/frontier_study.py` — multi-agent 5-fold comparison
- `v2/analysis/flip_day_study.py` — flip-day forensic classification
- `v2/analysis/grid_study.py` — side × decay coefficient grid

## Key model checkpoints
- `v2/models/model_trained_encoder.pt` — frozen encoder (22 contract features, epoch 2)
- `v2/models/seq_agent_side13.pt` — BC checkpoint (13-dim state, epoch 20)
- `v2/models/seq_agent_side13_rl.pt` — **current best** (RL epoch 5, PF 0.963)
- `v2/models/seq_agent_persistent.pt` — BC with oracle persistence rules
- `v2/models/grid_s03_d001.pt` — previous best (12-dim, PF 0.920 with bar89/95)

## Data
- `v2/data.pt` — 382,920 bars, 986 days, 52 context features, 22 contract features
- `v2/data_sidecars/` — per-day contract snapshots with path library labels
