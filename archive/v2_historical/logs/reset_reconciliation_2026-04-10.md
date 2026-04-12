# Exact-Chain Reset Reconciliation — 2026-04-10

This note explains the mixed state that was removed from the live `v2/` path during the reset.

## Archived Live-State Inputs

- `archive/v2_historical/logs/results_pre_reset_mixed_state_2026-04-10.tsv`
- `archive/v2_historical/logs/lab_notebook_pre_reset_mixed_state_2026-04-10.md`

## Unresolved Code-Only States

- `exp_088` exists as commit `06169e2` (`multiplicative context-contract interaction in score_head`)
- `exp_089` exists as commit `9e15a0e` (`L2 score regularization`)

Neither experiment has an authoritative scored result in the live notebook or `results.tsv`.
They are preserved in git history and archive context, but they must not be treated as scored evidence.

## Moved Out Of The Live v2 Path

- `v2/ops/fast_sweep.py`
- `v2/ops/run_loop.sh`
- `v2/docs/PLAN-codex-docs-train-restart.md`

## Live Baseline After Reset

The live baseline was restored to the `exp_080`-style training stack:

- gate BCE
- soft KL selection
- `SOFT_TEMP=0.20`
- no direct PnL regression
- no auxiliary side head
- no gate reweighting
- no score regularization
