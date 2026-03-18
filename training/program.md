# Autoresearch Trading Contract (Foundation Phase)

This file is the strict operating contract for autonomous loop experiments.
If anything else conflicts with this file, this file wins.

## Mission
Improve the training objective in `train.py` for SPX 0DTE option trading while preserving mechanical reliability.

## Scope Lock
- Foundation phase is locked to **60 features**.
- Model contract is locked to **two-head** outputs.
- Active trading semantics are locked to 6-direction entries + contextual exit.
- No migration work (64+Charm) is allowed in this phase.

## Model Contract (Required)
- Two-head architecture:
  - Gate head: `[NO_TRADE, TRADE]`
  - Direction head: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]`
- 8 effective actions:
  - `DO_NOTHING`
  - `BUY_CALL_ATM`, `BUY_CALL_OTM5`, `BUY_CALL_OTM10`
  - `BUY_PUT_ATM`, `BUY_PUT_OTM5`, `BUY_PUT_OTM10`
  - `EXIT`

## Data Contract (Required)
`data.pt` must include the two-head/OTM target fields and option price arrays required by training and replay:
- Targets:
  - `call_pnl`, `put_pnl`
  - `exit_call_label`, `exit_put_label`
  - `otm5_call_pnl`, `otm5_put_pnl`
  - `otm10_call_pnl`, `otm10_put_pnl`
- Prices:
  - `atm_call_prices`, `atm_put_prices`
  - `otm5_call_prices`, `otm5_put_prices`
  - `otm10_call_prices`, `otm10_put_prices`

Missing required fields must fail fast. No silent fallback behavior.

## Architecture Lock
Do not change tensor-shape-defining architecture values during this phase:
- `D_MODEL`
- `DEPTH`
- `N_HEADS`

Reason: warm-start compatibility with `best_model.pt` must be preserved.

## Safety + Runtime Constraints
- No `torch.compile`.
- No `DataParallel` / `DistributedDataParallel`.
- No `torch.jit.trace` / `torch.jit.script`.
- Keep memory/runtime within the configured budget.
- Keep changes focused: one coherent hypothesis per experiment.

## Output Metrics Contract (Required Keys)
The training script output must include these parseable metric keys:
- `score:`
- `profit_factor:`
- `trades_per_day:`
- `trade_sharpe:`
- `stop_loss_rate:`
- `worst_chunk_pf:`

## Reliability Policy
A candidate can be kept only if:
- Score improves over current best, and
- No contract violation, and
- No critical anomaly flags.

Near-tie improvements require stability confirmation from guard-band metrics.

## Autonomy Rules
- Modify `train.py` only.
- Preserve parseable output format.
- Avoid broad rewrites unless strongly justified by experiment evidence.
- Prefer small, testable deltas that can be reverted cleanly.
