# v3 Pure-RL Documentation Index

This directory is the live documentation surface for the `v3/` pure reinforcement-learning reset.

`v2/` remains the historical exact-chain supervised baseline. It is still useful context and a comparison point, but it is not the live research target for `v3`.

## Start Here

- [project_overview.md](project_overview.md) — plain-English overview of what `v3` is and how it differs from `v2`
- [../program.md](../program.md) — operating protocol for the pure-RL phase

## Core Ideas

- `v3` trains from replayed outcomes, not oracle imitation labels
- the agent can trade all day
- the agent controls contract choice, risk sizing, and active position management
- the environment keeps only physical constraints: exact visible contracts, fills, costs, size-aware slippage, one position max, and mandatory EOD flat

## Runtime Surface

- `v3/build_market_state.py` — build the derived multiscale market-state cache
- `v3/train.py` — PPO training loop
- `v3/replay.py` — deterministic replay and evaluation
- `v3/core/env.py` — exact-chain RL environment
- `v3/core/market_state.py` — market-state cache build/load logic
- `v3/core/schema.py` — runtime contracts
- `v3/core/metrics.py` — v3 evaluation contract
- `v3/core/artifact.py` — artifact save/load helpers
- `v3/ops/run_experiment.py` — one-command local experiment runner
- `v3/ops/pre_run_gate.py` — readiness checks before a real run
- `v3/ops/status_report.py` — operator-facing dataset and launch status
- `v3/ops/deploy.sh` — Akash GPU wrapper for `v3`

## Readiness Flow

1. Build the market-state cache once per dataset fingerprint:
   `python3 -m v3.build_market_state --data v2/data.pt --output v3/data/market_state_v1.pt`
2. `python3 -m v3.ops.pre_run_gate --data v2/data.pt --market-state v3/data/market_state_v1.pt`
3. `python3 -m v3.ops.status_report --data v2/data.pt --market-state v3/data/market_state_v1.pt`
3. Local smoke when needed:
   `python3 -m v3.train --data v2/data.pt --market-state v3/data/market_state_v1.pt --device cpu --updates 1 --rollout-days 1 --eval-interval 1 --max-eval-days 1 --checkpoint /tmp/v3_smoke.pt`
4. Real experiment on GPU:
   `./v3/ops/deploy.sh run_one v3_exp_001 --updates 10 --rollout-days 8 --eval-interval 2`
