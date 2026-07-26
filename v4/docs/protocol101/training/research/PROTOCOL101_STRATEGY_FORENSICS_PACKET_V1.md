# AUDIT_PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1

What is this: research-only Protocol101 strategy-forensics packet
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_strategy_forensics_packet_complete_model_training_still_blocked`

## Headline

- Trades: `1028`
- PnL: `$319,050`
- Win rate: `0.761`
- Profit factor: `4.421`
- Median premium: `$2,750`
- Median duration: `10.0` minutes
- Hard-stop trades: `11` for `$-14,470`
- Early-MFE-then-loss trades: `212`

## Response Analysis

The pasted response is directionally right: Protocol101 should be treated as an interrogated trading hypothesis, not a final model. This packet turns the memo into executable diagnostics where current artifacts are sufficient and explicit blockers where they are not.

## Top Trade Archetypes

| segment | right | time | moneyness | premium | exit | trades | pnl | win rate | PF |
|---|---|---|---|---|---|---:|---:|---:|---:|
| q4_2025 | C | post_open_morning | ITM | 3000_3500 | sequence_residual_override | 74 | $20,210 | 0.797 | 6.677 |
| q4_2025 | P | post_open_morning | ITM | 3000_3500 | sequence_residual_override | 39 | $14,430 | 0.692 | 3.559 |
| q1_2026 | C | post_open_morning | ITM | 3000_3500 | sequence_residual_override | 24 | $12,690 | 0.833 | 13.441 |
| q3_2025 | C | post_open_morning | ITM | 2500_3000 | sequence_residual_override | 39 | $11,550 | 0.974 | 116.500 |
| q3_2025 | C | post_open_morning | ITM | 3000_3500 | sequence_residual_override | 63 | $11,330 | 0.841 | 3.239 |
| q1_2026 | P | post_open_morning | ITM | 3000_3500 | sequence_residual_override | 18 | $10,780 | 0.944 | 21.731 |
| q3_2025 | C | post_open_morning | ITM | 2000_2500 | sequence_residual_override | 26 | $10,360 | 1.000 | inf |
| q4_2025 | C | post_open_morning | ITM | 2500_3000 | sequence_residual_override | 33 | $8,410 | 0.636 | 5.621 |
| q1_2026 | C | post_open_morning | ITM | 3000_3500 | protocol054_fallback | 31 | $8,230 | 0.710 | 3.267 |
| q4_2025 | P | post_open_morning | ITM | 2000_2500 | sequence_residual_override | 16 | $7,580 | 0.875 | 35.455 |
| q1_2026 | P | post_open_morning | ITM | 3000_3500 | target | 2 | $7,230 | 1.000 | inf |
| q1_2026 | C | post_open_morning | ITM | 3000_3500 | target | 2 | $7,040 | 1.000 | inf |

## Hard-Stop Finding

Hard-stop rows are small in the seed-1 paper replay but severe: `11` rows for `$-14,470`. They are high-priority because they are interpretable and may point to either pre-entry rejection or path-based lifecycle management.

Worst hard-stop rows:

| session | decision_time | side | premium | pnl | MFE | MAE | early MFE loss |
|---|---|---|---:|---:|---:|---:|---|
| 2026-03-04 | 2026-03-04T15:03:00+00:00 | P | $3,500 | $-1,840 | $680 | $-1,840 | True |
| 2026-03-10 | 2026-03-10T14:15:00+00:00 | P | $3,130 | $-1,680 | $330 | $-1,680 | True |
| 2025-09-25 | 2025-09-25T14:00:00+00:00 | P | $3,210 | $-1,610 | $-40 | $-1,610 | False |
| 2024-12-30 | 2024-12-30T15:27:00+00:00 | P | $2,750 | $-1,530 | $30 | $-1,530 | True |
| 2026-02-05 | 2026-02-05T19:01:00+00:00 | P | $2,820 | $-1,480 | $250 | $-1,480 | True |
| 2026-02-04 | 2026-02-04T15:01:00+00:00 | P | $2,450 | $-1,350 | $190 | $-1,350 | True |
| 2026-01-02 | 2026-01-02T15:25:00+00:00 | P | $2,430 | $-1,230 | $170 | $-1,230 | True |
| 2026-02-11 | 2026-02-11T15:29:00+00:00 | P | $2,060 | $-1,030 | $130 | $-1,030 | True |

## Losing-Day Finding

| day | segment | trades | pnl | win rate | hard stops | early-MFE losses |
|---|---|---:|---:|---:|---:|---:|
| 2024-10-31 | q4_2024_external | 4 | $-2,080 | 0.000 | 0 | 4 |
| 2026-03-04 | q1_2026 | 5 | $-1,980 | 0.400 | 1 | 2 |
| 2025-12-11 | q4_2025 | 2 | $-1,540 | 0.500 | 0 | 0 |
| 2025-11-26 | q4_2025 | 2 | $-1,420 | 0.000 | 0 | 1 |
| 2024-11-22 | q4_2024_external | 5 | $-1,300 | 0.400 | 0 | 3 |
| 2025-09-25 | q3_2025 | 3 | $-1,200 | 0.333 | 1 | 0 |
| 2024-12-12 | q4_2024_external | 2 | $-1,190 | 0.000 | 0 | 2 |
| 2026-03-16 | q1_2026 | 5 | $-1,140 | 0.600 | 0 | 1 |
| 2026-01-08 | q1_2026 | 2 | $-990 | 0.000 | 1 | 2 |
| 2025-12-16 | q4_2025 | 2 | $-950 | 0.000 | 0 | 1 |

## Score Calibration Proxy

Spearman score-margin vs PnL: `-0.0571`
Spearman score-margin vs MFE: `-0.1595`

This is only a selected-trade calibration proxy. A full calibration test needs rejected candidates and same-event candidate rankings.

## Claim-By-Claim Verdict

| claim | verdict | next test |
|---|---|---|
| Protocol101 is a trading hypothesis, not the final model. | `supported` | Keep Protocol101 as paper default while running strategy-forensics diagnostics before new model training. |
| Execution realism is the first gate. | `supported_blocker` | PROTOCOL101_EXECUTION_REALISM_BY_ARCHETYPE_V1 plus stratified paper/no-order fill collection. |
| Protocol101 strategy identity is blurry: scalp, runner, or hybrid. | `partially_answered` | Attach post-exit paths and classify scalp, runner, failed runner, giveback winner, and giveback loser states. |
| Hard-stop losses are a high-EV diagnostic target. | `supported` | PROTOCOL101_HARD_STOP_AUTOPSY_V1 to separate pre-entry rejection signatures from path-management failures. |
| Protocol101 exits may be early, late, or accidentally right. | `proxy_only` | PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1 using bid paths from actual exit to forced flat. |
| Protocol101 may be too narrow. | `plausible_not_proven` | Matched rejected-candidate audit using full_surface_action_advantage.parquet joined to Protocol101 baseline actions, with no future/path columns used for similarity. |
| Calls and puts may be separate strategies. | `supported_for_audit` | Side-specific strategy audit by side/time/premium/moneyness/exit/path archetype. |
| Protocol101 may block better later Protocol101 opportunities. | `unanswered_blocked` | Build a counterfactual-flat Protocol101 event replay that scores every event as if no position were open, then compare later feasible entries inside actual open intervals. |
| Protocol101 score may not calibrate realized value. | `concerning_proxy` | PROTOCOL101_SCORE_RELIABILITY_V1 with rejected candidates and same-event rankings. |
| Current diagnostic splits are not final promotion evidence. | `supported_blocker` | Build strategy matrix, CSCV/PBO diagnostics, and freeze a packet before untouched holdout scoring. |

## Internal Slot-Cost Status

Status: `blocked_missing_counterfactual_flat_protocol101_actions`

The available baseline attachment records deployed Protocol101 state. During open intervals rows are marked holding, so it cannot reveal which Protocol101 entries would have fired if the account were flat.

## Generated Outputs

- Summary: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/summary.json`
- Trade atlas: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/trade_archetype_cube.csv`
- Hard-stop autopsy: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/hard_stop_autopsy.csv`
- Losing-day autopsy: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/losing_day_autopsy.csv`
- Runner/giveback proxy: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/runner_giveback_proxy.csv`
- Score calibration: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/score_calibration.csv`
- Timing fragility by archetype: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/execution_fragility_by_archetype.csv`
- Narrowness proxy: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/protocol101_narrowness_proxy.csv`
- Experiment backlog: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/experiment_backlog.csv`
- Response claim analysis: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/response_claim_analysis.csv`
