# Protocol101 Paper Trading Readiness Cartography

Date: 2026-05-25

Iteration: `ITER-paper-readiness-state-packet`

Primary role: Codebase Cartographer, with readiness scorecard output

Primary system section: Section 4 live paper trading, with Section 1 governance and Section 3 model gates as dependencies

Mutation scope used: new read-only assessment artifacts under `research_ops/iterations/ITER-paper-readiness-state-packet/`

Forbidden actions honored: no broker scripts, no IBKR connection, no `placeOrder`, no launchd commands, no runtime flag mutation, no paper order enablement mutation, no training, no threshold tuning, no challenger promotion, no paid data download, no artifact cleanup.

## Executive Finding

Protocol101 has a substantial guarded paper-trading implementation, but the current repo state does not prove confirmed end-to-end IBKR paper trading. The strongest evidence is that the system can connect in prior sessions, collect live SPX/VIX/SPXW market data, build live context, score the Protocol051 surface, run Protocol101 candidate diagnostics, and produce repeated no-entry decisions. The strongest missing evidence is a real Protocol101 BUY/SELL intent progressing through paper dry-run, guarded paper submission, fill/cancel/timeout logging, and lifecycle exit reconstruction.

The correct next step is not model tweaking. The current readiness state supports no-order observation and observability repair. Paper-submit should remain paused or avoided operationally until the missing evidence gates are closed by an explicit separate run plan.

## Current Control Plane

Current default control: `PAPER_DEFAULT_PROTOCOL101`

Current scheduled paper runtime: guarded persistent Protocol101 paper session, currently configured toward `paper-submit` in the launchd plist and shell wrapper.

Current runtime paper guard flag:

- `v4/runtime/protocol101_paper_order_enablement.json`
- `paper_orders_enabled: true`
- `real_money_trading: false`
- `scope: ibkr_paper_account_only_protocol101_one_contract`
- `required_env: V4_ALLOW_IBKR_PAPER_ORDERS=YES`
- `account_id_redacted: null`

Important interpretation: the runtime flag being enabled is not evidence that end-to-end paper trading works. It only means one of the required paper-order gates is configured as enabled. Confirmed paper trading still requires current account, quote, context, guard, broker-status, fill/cancel, and lifecycle evidence.

## Primary Artifacts Identified

Protocol101 entry artifact:

- Manifest: `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/manifest.json`
- Model/scaler directory: same artifact directory
- Observed threshold in live logs and summaries: `-1.3651819953918456`
- Runtime loader: `v4/live/protocol101_entry.py`

Protocol051 surface artifact:

- Manifest: `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/manifest.json`
- Entry model variant: `surface_structure_aplus_side_value_rank`
- Runtime loader and scorer: `v4/live/protocol051_live_entry.py`

Lifecycle artifact:

- Manifest: `v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/model_artifacts/train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026/seed_1/manifest.json`
- Lifecycle runtime: `v4/live/protocol066_lifecycle.py`
- Persistent bridge use: `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`

Paper guard and executor:

- Guard: `v4/live/ibkr_paper_guard.py`
- Executor: `v4/live/ibkr_paper_executor.py`
- Paper trade log schema: `v4/live/paper_trade_log.py`

## Operational Path Map

Daily monitor:

- `v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py`
- Shell wrapper: `v4/ops/ibkr/run_protocol101_daily_monitor.sh`
- Launchd plist: `v4/ops/launchd/com.autoresearch.protocol101.daily-monitor.plist`

One-shot live entry paper bridge:

- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- Default mode: `intent-shadow`
- Handles live context, option ladder, Protocol101 decision, optional dry-run or paper-submit, and lifecycle exit intent for an existing position.

Persistent paper trader:

- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
- Default mode: `paper-submit`
- Runs session loop, refreshes SPXW ladder, builds live rows, scores Protocol051, evaluates Protocol101, applies paper guard, and logs session artifacts.

Paper session wrapper:

- `v4/ops/ibkr/run_protocol101_paper_session.sh`
- `PROTOCOL101_SESSION_KIND` defaults to `persistent`
- `PROTOCOL101_ENTRY_BRIDGE_MODE` defaults to `paper-submit`
- Sets `PROTOCOL101_ENABLE_PAPER_ORDERS=YES`, `PROTOCOL101_ACKNOWLEDGE_PAPER_LOSS=YES`, and `V4_ALLOW_IBKR_PAPER_ORDERS=YES` unless overridden.

Launchd paper session:

- `v4/ops/launchd/com.autoresearch.protocol101.paper-session.plist`
- Scheduled for 06:30 PT
- Environment sets persistent `paper-submit` with paper-order enablement and acknowledgement flags.

Preflight:

- `v4/ops/ibkr/run_protocol101_paper_preflight.sh`
- Would run `wait_for_ibkr_api.py --run-entitlement-probe` if executed. It was mapped only, not run.

## Stale or Conflicting Documents

Current docs point in different directions:

- `v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md` describes a guarded daily paper-submit path.
- `v4/ops/launchd/com.autoresearch.protocol101.paper-session.plist` and `v4/ops/ibkr/run_protocol101_paper_session.sh` are configured toward guarded persistent `paper-submit`.
- `v4/promotion/PROTOCOL_101_TUESDAY_PAPER_SESSION_RUNBOOK.md` still says the morning job starts `no-order-shadow` by default and requires an explicit paper-order gate.
- `v4/promotion/PROTOCOL_101_TUESDAY_LIVE_SHADOW_CHECKLIST.md` says not to place paper orders until live-data parity passes and user approves.
- `v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md` is older and says Protocol101 is not paper/live approved.
- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` is the best current high-level synthesis found, but it still reports no observed paper orders or fills.

Readiness impact: Stage 0 can identify the control plane, but stale docs create operational risk. The packet treats launchd and shell configs as the current configured default, while treating older promotion runbooks as stale safety guidance that still matters until superseded.

## Evidence Map

Live market data entitlement evidence:

- `v4/audit/ibkr_live_data_entitlements/summary.json`
- `v4/audit/ibkr_live_data_entitlements/report.md`
- Status: prior entitlement probe passed with live SPX/VIX and SPXW NBBO evidence; no orders.

Latest persistent paper-submit observation:

- Log: `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl`
- Summary: `v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader/2026-05-21/20260522T021051Z_23b3c2c0/summary.json`
- Status: 320 model decisions, 320 candidate sets, 325 market snapshots, zero enter intents, zero submitted orders, zero fills, zero broker endpoint rows. Trade log validation passed. Persistent reconnect failures were also recorded.

Daily monitor observation:

- `v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/20260522T021241Z_ebef6f6d/summary.json`
- Status: monitor ready, entitlement startup pass, trade log valid, but live capture pass false and live parity ready false. Zero broker rows, zero orders, zero fills.

One-shot intent-shadow observation:

- `v4/logs/paper_trading/2026-05-19/protocol101_intent-shadow_bridge_verify_2026-05-19.jsonl`
- `v4/audit/autoresearch/v4_aplus_hypothesis_158_protocol101_live_entry_paper_bridge/2026-05-19/20260520T011859Z_8d023ed8/summary.json`
- Status: connected to IBKR paper in prior run, built 42-contract chain, produced no-entry decisions, zero broker calls.

Paper dry-run evidence:

- Synthetic executor smoke: `v4/logs/paper_trading/2026-05-14/protocol142_executor_smoke.jsonl`
- Test coverage: `v4/tests/test_protocol142_paper_executor.py`
- Status: guard/executor dry-run can log a paper-order intent without calling broker endpoint. Missing: real Protocol101 live BUY/SELL intent dry-run.

Model-improvement gate evidence:

- `v4/audit/autoresearch/section3_model_experiment_preflight/summary.json`
- `v4/audit/autoresearch/unified_neural_training_readiness/summary.json`
- `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json`
- `v4/audit/autoresearch/live_no_order_full_action_parity_readiness/summary.json`
- `v4/audit/autoresearch/untouched_holdout_availability/summary.json`
- Status: model improvement is blocked by missing fill/execution observations, untouched holdout availability, live full-action parity, and training readiness gates.

## Stage Status Summary

Stage 0, repo/control readiness: PASS

The current artifact/control plane is identifiable, including Protocol101, Protocol051 surface, lifecycle artifact, paper guard, runtime flag, and current configured default. Stale/conflicting docs are identified.

Stage 1, preflight readiness: UNKNOWN

Requirements and scripts are mapped, and prior entitlement evidence exists. Current-day preflight was not run, current IBKR session/account state is unconfirmed, and account id evidence is redacted or missing.

Stage 2, no-order observation readiness: UNKNOWN

Prior logs show live quotes, context accumulation, surface scoring, Protocol101 candidate diagnostics, and no-entry decisions. However, no nonzero candidate set or entry decision was observed, persistent logs do not preserve enough quote-age detail for every decision, and live parity remains false in the monitor summary.

Stage 3, paper-dry-run readiness: NOT TESTED

The executor dry-run mechanism is covered by tests and one synthetic smoke log, but no real Protocol101 live BUY/SELL intent reached paper-dry-run.

Stage 4, paper-submit readiness: BLOCKED

The code has guardrails for one-contract DU/paper orders and no real-money trading, but there is no observed Protocol101 paper submission, fill, cancel, timeout, or broker-status row. Current logs show zero broker endpoint rows and zero fills.

Stage 5, lifecycle/exit readiness: UNKNOWN

Lifecycle code exists and open-position detection is mapped, but no open Protocol101 paper position, exit intent, forced-flat action, or exit fill/cancel evidence was found.

Stage 6, observability/reconstruction readiness: FAIL

Logs are useful but incomplete for full reconstruction. Missing or inconsistently present fields include raw quote timestamps, raw logits, candidate set hashes, feature vector hashes, action masks, selected-action metadata for full-action parity, latency, broker status, and lifecycle state.

Stage 7, evidence required before model tweaking: BLOCKED

Model tweaking, retraining, hill-climbing, and challenger work are not justified until operational truth gates pass: dry-run/submit/fill/cancel/lifecycle evidence, full reconstruction fields, untouched holdout availability, live no-order full-action parity, and fill/execution observation thresholds.

## CEO Recommendation

Recommendation: D. Pause paper-submit and repair observability.

Operating mode while repairing: A. Continue no-order observation only.

Do not start Protocol101 model improvement work yet. The system can be studied and instrumented, but confirmed end-to-end IBKR paper trading has not been proven.

