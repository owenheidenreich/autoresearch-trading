# Evidence Index

All paths are repo-relative from `/Users/gduby/Documents/autoresearch-trading`.

## Protocol101 Core

| Artifact | Path | Use |
|---|---|---|
| Protocol101 experiment report | `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/report.md` | Strict serial gate and split metrics |
| Protocol101 readiness packet | `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md` | Promotion checks and blockers |
| Protocol101 freeze | `v4/promotion/PROTOCOL_101_FREEZE.json` | Frozen artifact manifest |
| Live entry bridge | `v4/live/protocol101_entry.py` | Runtime feature/candidate construction |
| Live entry wrapper | `v4/live/protocol101_live_entry.py` | Live-facing protocol helper |
| Event-set model | `v4/model/serial_opportunity.py` | Wait/candidate event policy mechanics |

## Protocol101 Trade Behavior

| Artifact | Path | Use |
|---|---|---|
| Trade chart report | `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/report.md` | Paper-account lens and concentration |
| Trade log | `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv` | Source for trade behavior audit |
| Equity chart | `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/equity.html` | Source-of-truth visual paper curve |
| All seed trades | `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/research_all_seed_trades.csv` | Research ensemble before paper-account filtering |

## Falsification And Timing

| Artifact | Path | Use |
|---|---|---|
| Skeptical falsification | `v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification/report.md` | Random/baseline/delay stress |
| Targeted high-res validation | `v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_validation/report.md` | High-resolution coverage |
| Timing fragility hardening | `v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/report.md` | Delay-second stress and future freshness rule |
| Capital realism | `v4/audit/autoresearch/v4_aplus_hypothesis_122_protocol101_capital_realism/report.md` | Buying-power realism |
| Account-aware sizing | `v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation/report.md` | Sizing validation |
| Multi-contract decision | `v4/audit/autoresearch/v4_aplus_hypothesis_154_protocol101_multi_contract_promotion_decision/report.md` | Multi-contract research-only decision |

## Strong Research Challengers

| Protocol | Path | Why Important |
|---|---|---|
| Protocol194 | `v4/audit/autoresearch/v4_aplus_hypothesis_194_full_action_surface_edge_5seed_confirmation/report.md` | Full-action surface-edge challenger beat Protocol101 in replay |
| Protocol195 | `v4/audit/autoresearch/v4_aplus_hypothesis_195_protocol194_timing_fragility/report.md` | Shows Protocol194 timing fragility |
| Protocol215 | `v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/report.md` | Repaired-history full-action challenger |
| Protocol240 | `v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/report.md` | Strong premium-blend challenger |
| Protocol248 | `v4/audit/autoresearch/v4_aplus_hypothesis_248_challenger_failure_surface/report.md` | Clean quality/drawdown comparison vs Protocol101 |
| Protocol260 | `v4/audit/autoresearch/v4_aplus_hypothesis_260_policy_router_reliability_priors/report.md` | Router between Protocol101 and challenger streams |
| Protocol265 | `v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/report.md` | Baseline-anchored continuation research |

## Unified / Foundation Work

| Artifact | Path | Use |
|---|---|---|
| Engineer response | `v4/docs/engineer-response.md` | External guidance that shaped foundation review |
| Foundation hardening review | `v4/audit/autoresearch/foundation_hardening_review/report.md` | Implementation audit and blockers |
| Unified serial game | `v4/model/unified_serial_game.py` | Wait/enter/hold/exit contract |
| Protocol276 replay | `v4/audit/autoresearch/v4_aplus_hypothesis_276_integrated_entry_lifecycle_serial_replay/report.md` | Integrated replay failure |
| Protocol276 failure attribution | `v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution/report.md` | Root-cause attribution |
| Learned-defer handoff | `v4/docs/ML_ENGINEER_HANDOFF_LEARNED_DEFER_NEURAL_POLICY_2026_05_24.md` | Neural/slot-cost handoff |
| Learned-defer validation packet | `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/report.md` | Frozen challenger research-only validation |

## Governance And Process

| Artifact | Path | Use |
|---|---|---|
| Research ledger | `v4/ledger/RESEARCH_LEDGER.md` | Protocol chronology |
| Hypothesis-to-promotion process | `v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md` | Required process discipline |
| Naming guide | `v4/docs/NAMING_GUIDE.md` | Protocol/artifact naming |
| Protocol101 daily paper trading | `v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md` | Operational default workflow |

