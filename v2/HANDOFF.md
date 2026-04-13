# Handoff: Current v2 State

Read this file, then [docs/founder_intent.md](docs/founder_intent.md), then [program.md](program.md), then [docs/current_state.md](docs/current_state.md).

For domain knowledge: [docs/domain/](docs/domain/) contains 0DTE options knowledge and practitioner trading journals. Read these to understand the instrument before making architecture decisions.

## Current Model

**exp_144** — promoted 2026-04-12. Best result in project history.

| Metric | Value |
|--------|-------|
| Score | 0.547 |
| PF | 1.356 |
| DD | 8.2% |
| Sortino | 7.36 |
| Net P&L | +$4,771 |
| Final Equity | $14,771 |
| WR | 39.1% |
| Trades | 184 (158C/26P) |
| Selection Accuracy | 8.7% |
| Gate Accuracy | 21.7% |

Folds: `[0.176, -0.200, -0.200, -0.200, 3.160]` — 3 folds hit -0.200 floor. All 4 baselines also fail on those folds (hard market regimes, not model failure).

## What Made It Profitable

Three consecutive execution improvements, no model architecture changes since exp_139:

1. **Contract feature normalization** (exp_139): Per-bar z-score of 11 continuous contract features + Greek sign flip for puts. Fixed 340,000x scale mismatch where strike dominated contract_proj. First profitable model.
2. **Breakeven trailing trigger 0.30 → 0.15** (exp_140): Locks breakeven earlier on trades that reach +15% unrealized. Eliminated 96% of whipsaw losses.
3. **Cooldown bars 5 → 3** (exp_144): Faster re-entry after stop-loss. The 5-bar cooldown was blocking 172 profitable bars.

## Live Code

- `v2/train.py` — TradingModel: encoder + contract_proj + call/put score heads + put_bias + no_trade_head
- `v2/core/policy.py` — DecisionPolicy: stop=30%, target=50%, hold=120, trailing exit, breakeven=0.15, cooldown=3, window bars 60-105
- `v2/core/simulator.py` — simulate_trade(), TRAILING_TIERS, _build_trailing_tiers()
- `v2/replay.py` — replay_validation(), baselines, traces

## Dataset

- `v2/data.pt` — v4_exact_chain, fingerprint 46f2d184e186496f, 986 days, 47 features
- `v2/data_sidecars/*.pt` — per-day contract snapshots and labels
- Labels computed with: stop=30%, target=50%, hold=120, TRAILING_TIERS (breakeven at 0.30 original)
- Trade window in labels: bars 30-270 (broader than eval window 60-105)

## What To Trust

- `v2/data.pt` and `v2/data_sidecars/` as the canonical dataset
- `v2/results.tsv` as official scored runs
- `v2/lab_notebook.md` as the experiment log
- `v2/artifacts/exp_144/` as the current promoted artifact
- `v2/models/model.pt` as the production model

## What Not To Trust

- Any pre-exact-chain score (before exp_074)
- `v2/state/inner_loop_state.json` — stale, references exp_008 from old GPU session
- Experiment conclusions from previous Claude sessions — form your own from the code and data
- Archive docs that reference specific feature priorities or model changes — these are opinions, not facts

## Abandoned Approaches

- Hierarchical direction head (exp_100-103): direction doesn't decompose, 52% accuracy = random
- Direction-conditioned KL (exp_103): uncalibrated cross-direction scores
- Auxiliary side-calibration losses (exp_111-113): dir_acc stuck at random
- Pairwise ranking loss (exp_121): collapsed to 85% puts
- SOFT_TEMP=0.05 (exp_141): great direction balance but call selection collapsed
- direction_proj replacing put_bias (exp_142): improved weak folds but degraded fold 4
- direction_proj as residual (exp_143): same pattern
