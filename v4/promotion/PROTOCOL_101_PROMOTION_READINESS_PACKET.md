# Protocol 101 Promotion-Readiness Packet

## Candidate

- Name: Protocol 101 sequential event policy with causal short-history features
- Status: research promotion candidate, not paper/live approved
- Freeze manifest: `v4/promotion/PROTOCOL_101_FREEZE.json`
- Protocol 102 report: `v4/audit/autoresearch/v4_aplus_hypothesis_102_protocol101_readiness/report.md`
- Incremental paid data cost: `$0`

## Evidence

- q3_2025: median PnL `59130`, PF `3.779`, baseline margin `1340`, +0.25 stress `46380`.
- q4_2025: median PnL `92460`, PF `5.899`, baseline margin `140`, +0.25 stress `77760`.
- q1_2026: median PnL `95010`, PF `3.714`, baseline margin `5010`, +0.25 stress `82560`.
- march_2026: median PnL `41790`, PF `3.420`, baseline margin `2690`, +0.25 stress `36290`.

## External Stress

- q4_2024 temporal-regime stress: median PnL `58280`, PF `3.460`, strict serial baseline margin `1290`, +0.25 stress `47730`.
- This is supportive only, not a chronological promotion gate, because frozen 2025-trained artifacts were applied backward to Q4 2024.
- q4_2024 attribution: aggregate model PnL was `261080` versus strict serial `269630`, with seed margins `3690`, `5380`, `-2140`, `-16770`, and `1290`. The largest drag was flat threshold/no-entry (`-25110`), especially missed post-open call winners; same-minute contract swaps were net positive (`14540`).
- frozen-seed ensemble check: rejected as a replacement. It improved Q4 2024 seed 4 but failed the registered strict-serial gate in Q4 2025 and March 2026, so Protocol 101 remains the frozen candidate.

## Blockers Before Paper Trading

- Protocol 101 has not been validated on broader chronological locked periods beyond the current Q3/Q4/Q1/March gate; Q4 2024 is only a temporal-regime stress audit.
- No 1s/tick path audit has been run for Protocol 101 selected trades.
- No Protocol 101 no-order live shadow capture exists.
- IBKR live market-data entitlement/cash-settlement blockers remain unresolved from prior live attempts.
- Protocol 101 Q4 margin over strict baseline is thin, so broader data is required before paper/live confidence.
- Protocol 101 still has seed-level abstention fragility: Q4 2024 baseline-only winners >= `1000` totaled `83780` across `54` trades.
- Q3 2024 data-batch preflight is ready, but the actual download remains blocked until explicit user approval. Existing local coverage is continuous from `2024-10-01` through `2026-03-31`; Q3 2024 would add `64` expected sessions.
- Paid-data guardrail is now enforced in code for the Q3 path: ThetaData `index_history_ohlc` and Databento `timeseries.get_range` downloads require the exact approval text from the Q3 2024 manifest.

## Decision

Protocol 101 is ready for broader locked historical validation. It is not ready for broker-connected paper trading.
