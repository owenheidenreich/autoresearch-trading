# Protocol 125: Protocol101 Pre-Tuesday Readiness Pack

No paid market data was downloaded. No live broker endpoint was called. No orders were placed. No model was trained.

- Decision: `ready_for_tuesday_no_order_live_shadow_only`
- Source-of-truth equity: `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/equity.html`
- Trade overlay: `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.html`
- Tuesday checklist: `v4/promotion/PROTOCOL_101_TUESDAY_LIVE_SHADOW_CHECKLIST.md`

## Replay Realism

- Paper starting cash: `$10,000`
- Trades: `1028`
- Skipped trades: `0`
- Unaffordable skips: `0`
- Ending equity: `$329,050`
- Replay PnL: `$319,050`
- Max drawdown: `$-2,790`
- Worst day PnL: `$-2,080`
- Max known buying power: `$3,520`
- Premium coverage: `100.0%`
- Worst intratrade MAE: `$-1,840`
- Path coverage: `100.0%`
- Top 20 trades share of net PnL: `19.1%`
- Top day share of net PnL: `2.9%`
- Single-trade majority: `False`
- Top-20 majority: `False`

## Execution Skepticism

- Protocol 114 decision: `fragile_needs_more_data`
- Protocol 114 next hypothesis: `prioritize 1s/tick/live-shadow validation because edge is timing-sensitive`
- Protocol 118 historical shadow decision: `pass_historical_no_order_shadow_rehearsal_live_capture_next`
- Protocol 123 order-state decision: `pass_10000_order_state_rehearsal_live_data_pending`
- Protocol 124 live-data parity decision: `blocked_live_subscriptions_delayed_plumbing_passed`

## Shadow Event Contract

- Event schema: `v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness/protocol101_shadow_event_schema.json`
- JSONL examples: `v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness/protocol101_shadow_event_examples.jsonl`
- Events: `2056`
- Max open positions: `1`
- Verification status: `pass`

## Visual Inspection Targets

- Target file: `v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness/visual_inspection_targets.csv`
- Daily PnL file: `v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness/daily_pnl.csv`
- Best day: `2026-03-03` `$9,270`
- Worst day: `2024-10-31` `$-2,080`

Use `trades.html` Trade # jump with these rows first:

| category | rank | trade # | session | side | premium | pnl | exit |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| top_winner | 1 | 187 | 2024-12-20 | CALL | $3,430 | $4,140 | target |
| top_winner | 2 | 886 | 2026-02-17 | CALL | $3,490 | $3,780 | target |
| top_winner | 3 | 199 | 2024-12-27 | PUT | $3,500 | $3,660 | target |
| top_winner | 4 | 934 | 2026-03-05 | PUT | $3,370 | $3,650 | target |
| top_winner | 5 | 822 | 2026-01-21 | CALL | $2,710 | $3,620 | target |
| top_winner | 6 | 842 | 2026-01-29 | PUT | $3,300 | $3,580 | target |
| top_winner | 7 | 364 | 2025-08-22 | CALL | $2,960 | $3,390 | target |
| top_winner | 8 | 859 | 2026-02-05 | CALL | $3,220 | $3,260 | target |
| top_winner | 9 | 673 | 2025-11-20 | PUT | $3,050 | $3,120 | target |
| top_winner | 10 | 133 | 2024-11-21 | CALL | $2,920 | $3,090 | target |
| worst_loser | 1 | 929 | 2026-03-04 | PUT | $3,500 | $-1,840 | hard_stop |
| worst_loser | 2 | 947 | 2026-03-10 | PUT | $3,130 | $-1,680 | hard_stop |
| worst_loser | 3 | 468 | 2025-09-25 | PUT | $3,210 | $-1,610 | hard_stop |
| worst_loser | 4 | 734 | 2025-12-11 | PUT | $3,340 | $-1,560 | sequence_residual_override |
| worst_loser | 5 | 950 | 2026-03-11 | PUT | $3,260 | $-1,530 | protocol054_fallback |
| worst_loser | 6 | 204 | 2024-12-30 | PUT | $2,750 | $-1,530 | hard_stop |
| worst_loser | 7 | 862 | 2026-02-05 | PUT | $2,820 | $-1,480 | hard_stop |
| worst_loser | 8 | 317 | 2025-08-07 | CALL | $3,320 | $-1,420 | sequence_residual_override |
| worst_loser | 9 | 966 | 2026-03-16 | PUT | $3,190 | $-1,410 | protocol054_fallback |
| worst_loser | 10 | 388 | 2025-09-02 | CALL | $3,310 | $-1,390 | sequence_residual_override |
| highest_premium | 1 | 839 | 2026-01-28 | PUT | $3,520 | $310 | sequence_residual_override |
| highest_premium | 2 | 851 | 2026-02-03 | PUT | $3,520 | $500 | protocol054_fallback |
| highest_premium | 3 | 331 | 2025-08-12 | CALL | $3,520 | $360 | sequence_residual_override |
| highest_premium | 4 | 9 | 2024-10-02 | CALL | $3,520 | $1,790 | protocol054_fallback |
| highest_premium | 5 | 290 | 2025-07-30 | PUT | $3,520 | $830 | sequence_residual_override |
| highest_premium | 6 | 266 | 2025-07-21 | CALL | $3,510 | $250 | sequence_residual_override |
| highest_premium | 7 | 497 | 2025-10-06 | CALL | $3,510 | $250 | sequence_residual_override |
| highest_premium | 8 | 646 | 2025-11-17 | CALL | $3,510 | $140 | sequence_residual_override |
| highest_premium | 9 | 637 | 2025-11-14 | CALL | $3,510 | $1,200 | sequence_residual_override |
| highest_premium | 10 | 863 | 2026-02-05 | CALL | $3,510 | $500 | sequence_residual_override |
| highest_starting_cash_usage | 1 | 3 | 2024-10-01 | CALL | $3,400 | $-460 | protocol054_fallback |
| highest_starting_cash_usage | 2 | 2 | 2024-10-01 | CALL | $3,100 | $300 | protocol054_fallback |
| highest_starting_cash_usage | 3 | 4 | 2024-10-01 | CALL | $2,640 | $2,430 | protocol054_fallback |
| highest_starting_cash_usage | 4 | 1 | 2024-10-01 | PUT | $2,480 | $-50 | sequence_residual_override |
| highest_starting_cash_usage | 5 | 9 | 2024-10-02 | CALL | $3,520 | $1,790 | protocol054_fallback |
| highest_starting_cash_usage | 6 | 10 | 2024-10-02 | CALL | $3,420 | $100 | protocol054_fallback |
| highest_starting_cash_usage | 7 | 7 | 2024-10-01 | CALL | $2,620 | $350 | sequence_residual_override |
| highest_starting_cash_usage | 8 | 8 | 2024-10-02 | CALL | $2,540 | $620 | sequence_residual_override |
| highest_starting_cash_usage | 9 | 17 | 2024-10-04 | CALL | $3,230 | $1,280 | protocol054_fallback |
| highest_starting_cash_usage | 10 | 15 | 2024-10-04 | CALL | $3,250 | $-870 | protocol054_fallback |

## Next Gate

On Tuesday, run Protocol101 no-order live shadow capture first. Compare emitted JSONL to protocol101_shadow_event_schema.json and keep paper orders disabled until live feature/quote parity passes.
