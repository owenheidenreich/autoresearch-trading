# Protocol 134: Sizing Attribution

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_sizing_attribution_research_candidate_not_live`
- Policy: `lower_two_contract_threshold`
- Incremental PnL vs one contract: `$34,950`
- Scaled trades: `184`
- Skipped trades: `64`
- Scaled positive fraction: `85.3%`
- Top day share of positive incremental PnL: `6.4%`
- Top month share of positive incremental PnL: `24.3%`

## Concentration

- Top day: `{'session': '2025-09-17', 'incremental_pnl': 4890.0}`
- Top month: `{'month': '2025-09', 'incremental_pnl': 18550.0}`
- Skipped baseline PnL given up: `$26,260`

## Outputs

- Attribution rows: `v4/audit/autoresearch/v4_aplus_hypothesis_134_protocol101_sizing_attribution/attribution_rows.csv`
- Group summary: `v4/audit/autoresearch/v4_aplus_hypothesis_134_protocol101_sizing_attribution/group_summary.csv`

## Next Gate

Keep the sizing policy as an offline research candidate. Next test should add split-by-split equity curves and paper-account visualization, not live multi-contract trading.
