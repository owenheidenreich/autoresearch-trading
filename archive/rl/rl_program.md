# V3 Program

This is the operating protocol for the pure-RL reset.

## Mission

Build a dynamic SPX 0DTE research system where the model learns:

- whether to trade
- which exact contract to trade
- how much risk to allocate
- how to manage the position after entry

The model learns from replayed outcomes under the exact-chain environment, not from oracle imitation targets.

## Phase Boundary

- `v2/` remains the historical supervised exact-chain baseline.
- `v3/` is the live research surface for dynamic pure RL.
- `shared/` is the data infrastructure facade — v3 imports data primitives from here, not from v2 directly.
- `v2/results.tsv` remains historical evidence only.
- `v3/results.tsv` is the official result log for the RL phase.

## Environment Law

The v3 environment keeps only physical constraints:

- exact visible contracts only
- next-bar execution
- commissions and size-aware slippage
- integer quantity
- one live position at a time
- same-contract-only resizing while a position is open
- mandatory end-of-day flatten

The environment does not hardcode session windows, cooldowns, or daily loss caps.

## Multiscale Memory

The live v3 observation contract is now multiscale:

- `context_1m`: trailing `90 x 47` trusted one-minute features from `v2/data.pt`
- `context_5m`: trailing causal 5-minute bucket context from the derived v3 market-state cache
- `session_state`: explicit intraday anchors like opening range, initial balance, session high/low distance, and time-of-day volume pressure
- `trade_state`: explicit live trade lifecycle memory like bars held, MFE/MAE, add/reduce counts, and entry anchor context

The derived market-state cache is versioned independently as `v3_market_state_v1`.

## Training Loop

1. Implement one environment/model/training change.
2. Train the PPO policy on day episodes from the train split.
3. Replay deterministically on held-out days.
4. Save checkpoint, traces, metrics, and baselines in a v3 artifact.
5. Promote only if the new replay score is positive, drawdown stays within the v3 limit, and the run beats the v3 baselines.

## Operator Workflow

1. Build the market-state cache:
   `python3 -m v3.build_market_state --data v2/data.pt --output v3/data/market_state_v1.pt`
2. Run the readiness gate:
   `python3 -m v3.ops.pre_run_gate --data v2/data.pt --market-state v3/data/market_state_v1.pt`
3. Review the current surface:
   `python3 -m v3.ops.status_report --data v2/data.pt --market-state v3/data/market_state_v1.pt`
4. Optional local smoke:
   `python3 -m v3.train --data v2/data.pt --market-state v3/data/market_state_v1.pt --device cpu --updates 1 --rollout-days 1 --eval-interval 1 --max-eval-days 1 --checkpoint /tmp/v3_smoke.pt`
5. Launch a real GPU experiment:
   `./v3/ops/deploy.sh run_one v3_exp_001 --updates 10 --rollout-days 8 --eval-interval 2`
6. Replay the resulting artifact deterministically:
   `python3 -m v3.replay --checkpoint v3/artifacts/v3_exp_001 --data v2/data.pt --market-state v3/data/market_state_v1.pt --mask promote_mask --with-baselines`
