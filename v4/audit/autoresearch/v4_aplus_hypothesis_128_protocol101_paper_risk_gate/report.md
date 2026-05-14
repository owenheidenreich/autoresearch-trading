# Protocol 128: Protocol101 Paper Account Risk Gate

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_paper_risk_gate_overlay_ready_for_live_shadow`
- Paper starting cash: `$10,000`
- IBKR reserve excluded from trading capital: `$500`
- Max contracts for initial paper trading: `1`
- Max known premium: `min($4,000, 40% equity)`
- Daily new-entry stop: `$-750` realized PnL

## Replay Invariants

- Trades: `1028`
- Max concurrent positions: `1`
- Zero unaffordable trades: `True`
- Zero premium-cap violations: `True`
- All SPXW PM contracts: `True`
- All flat by close: `True`

## Risk Gate Overlay

- Passed rows: `993` / `1028`
- Hard block rows: `0`
- Daily-loss protective-only rows: `35`
- Reason counts: `{'daily_loss_stop': 35}`

## Outputs

- Risk rows: `v4/audit/autoresearch/v4_aplus_hypothesis_128_protocol101_paper_risk_gate/risk_gate_rows.csv`
- Summary: `v4/audit/autoresearch/v4_aplus_hypothesis_128_protocol101_paper_risk_gate/summary.json`

## Next Gate

Use this same risk gate in Tuesday's no-order shadow stream. Do not enable any paper-order endpoint until the live schema, quote freshness, and order-state rehearsal all pass.
