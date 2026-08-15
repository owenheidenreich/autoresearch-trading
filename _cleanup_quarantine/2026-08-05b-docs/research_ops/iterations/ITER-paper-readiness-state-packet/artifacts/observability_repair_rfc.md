# Protocol101 Paper Observability Repair RFC

Date: 2026-05-25

Iteration: `ITER-paper-readiness-state-packet`

Status: proposed

Decision requested: approve observability repair work before any paper-submit or model-improvement work.

Does this change the paper-trading default: no

Broker endpoint called: no

Paid data required: no

Model training requested: no

Runtime or launchd mutation requested: no

## Purpose

Protocol101 has guarded paper-trading code, but the current evidence does not prove confirmed end-to-end IBKR paper trading. The blocker is not only that no orders or fills were observed. The deeper blocker is that current logs are not complete enough to reconstruct every decision from raw market observation through model decision, guard result, order intent, broker outcome, lifecycle state, and exit result.

This RFC defines the minimum observability contract required before moving from no-order observation to paper-dry-run, paper-submit, or model improvement.

## Scope

In scope:

- Logging and audit contract for Protocol101 no-order observation.
- Logging and audit contract for paper-dry-run.
- Logging and audit contract for guarded one-contract IBKR paper-submit, when separately approved later.
- Logging and audit contract for open-position lifecycle and exit.
- Fail-closed readiness checks for missing reconstruction fields.

Out of scope:

- Calling IBKR.
- Submitting paper orders.
- Changing runtime flags.
- Changing launchd.
- Training or tuning models.
- Promoting challengers.
- Changing Protocol101 trading behavior.
- Changing thresholds, candidate filters, or guard policy except where a later implementation RFC explicitly requests it.

## Hypothesis

Hypothesis: If Protocol101 logs a complete reconstruction envelope for every decision and fail-closes when required fields are missing, then the team can safely distinguish between implementation existence, no-order operational evidence, paper-dry-run evidence, paper-submit evidence, lifecycle evidence, and model-improvement readiness.

Failure mode addressed: The system currently risks treating code paths or partial logs as proof that end-to-end paper trading works.

Assumptions reduced:

- Whether quote freshness is real and reconstructable.
- Whether a no-entry decision is explained by market state, candidate filters, model logits, or missing data.
- Whether a dry-run or submit intent passed the guard for the right reasons.
- Whether any broker endpoint call happened before guard pass.
- Whether lifecycle exit state can be reconstructed after entry.

Expected failure mode if wrong: The audit may still pass incomplete sessions, or logs may become large but not reconstructable. That is why the acceptance checks below must be fail-closed and field-specific.

## Baseline

Primary baseline:

- `research_ops/iterations/ITER-paper-readiness-state-packet/artifacts/paper_trading_readiness_packet.md`
- `research_ops/iterations/ITER-paper-readiness-state-packet/artifacts/paper_trading_readiness_scorecard.csv`
- `research_ops/iterations/ITER-paper-readiness-state-packet/artifacts/missing_evidence_list.md`

Observed baseline:

- Stage 0 PASS.
- Stage 1 UNKNOWN.
- Stage 2 UNKNOWN.
- Stage 3 NOT TESTED.
- Stage 4 BLOCKED.
- Stage 5 UNKNOWN.
- Stage 6 FAIL.
- Stage 7 BLOCKED.

Latest operational baseline:

- `v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl`
- 320 model decisions.
- 320 candidate sets.
- 325 market snapshots.
- Zero enter intents.
- Zero submitted orders.
- Zero fills.
- Zero broker endpoint rows.

## Proposed Observability Contract

Every Protocol101 paper-readiness event must belong to a session envelope:

- `run_id`
- `session_id`
- `event_id`
- `event_type`
- `mode`
- `decision_mode`
- `paper_trading`
- `real_money`
- `live_orders_enabled`
- `broker_order_endpoint_called`
- `timestamp_utc`
- `received_timestamp_utc`
- `decision_timestamp_utc`, when a model decision is made
- `source_script`
- `source_git_sha`, if available
- `artifact_ids`
- `runtime_flag_digest`, not raw sensitive content
- `account_id_redacted`, never raw account id

Every market-data event used by a decision must log:

- `spx_price`
- `vix_price`
- `spx_market_data_type`
- `vix_market_data_type`
- `spx_raw_quote_timestamp_utc`
- `vix_raw_quote_timestamp_utc`
- `spx_quote_age_ms`
- `vix_quote_age_ms`
- `context_start_timestamp_utc`
- `context_rows`
- `context_age_ms`
- `context_ready`

Every option quote used by candidate generation must log:

- `underlying_symbol`
- `trading_class`
- `expiry`
- `settlement`
- `strike`
- `right`
- `exchange`
- `bid`
- `ask`
- `bid_size`
- `ask_size`
- `spread`
- `mid`
- `market_data_type`
- `raw_quote_timestamp_utc`
- `received_timestamp_utc`
- `quote_age_ms`
- `delta`
- `gamma`
- `theta`
- `vega`
- `implied_volatility`
- `quote_valid`
- `quote_reject_reason`

Every candidate-set event must log:

- `candidate_set_hash`
- `feature_vector_hash`
- `eligible_token_count`
- `valid_score_count`
- `candidate_count`
- `rejected_count`
- `top_rejected_contracts`
- `candidate_contracts`
- `filter_reasons`
- `min_edge`
- `max_edge`
- `surface_score_summary`
- `feature_missing_columns`
- `feature_imputed_columns`

Every Protocol101 model decision must log:

- `action_mask`
- `raw_logits`
- `wait_logit`
- `candidate_logits`
- `selected_action`
- `selected_contract`
- `selected_score`
- `selected_margin`
- `threshold`
- `threshold_source`
- `decision_reason`
- `no_entry_reason`
- `latency_ms`

Every guard event must log:

- `guard_passed`
- `guard_block_reasons`
- `permission_enable_flag_present`
- `permission_ack_flag_present`
- `permission_env_present`
- `account_prefix_ok`
- `paper_account_confirmed`
- `real_money_false_confirmed`
- `quantity_ok`
- `one_open_position_ok`
- `quote_freshness_ok`
- `context_freshness_ok`
- `affordability_ok`
- `account_cash`
- `account_equity`
- `open_positions`
- `premium_required`
- `guard_config_digest`

Every order-intent event must log:

- `intent_id`
- `intent_source_decision_event_id`
- `dry_run`
- `action`
- `side`
- `quantity`
- `limit_price`
- `reference_bid`
- `reference_ask`
- `reference_quote_age_ms`
- `contract_payload`
- `order_payload`
- `would_submit`
- `submit_allowed`

Every broker-status event, if paper-submit is later approved, must log:

- `broker_order_endpoint_called`
- `broker_call_timestamp_utc`
- `broker_order_id`
- `broker_status`
- `filled_quantity`
- `remaining_quantity`
- `average_fill_price`
- `last_fill_price`
- `cancel_requested`
- `cancel_timestamp_utc`
- `timeout_timestamp_utc`
- `final_status`
- `broker_latency_ms`
- `submit_to_fill_ms`
- `submit_to_cancel_ms`

Every lifecycle event must log:

- `position_detected`
- `position_contract_payload`
- `position_quantity`
- `position_avg_cost`
- `position_source`
- `runtime_state_hash`
- `lifecycle_feature_vector_hash`
- `lifecycle_raw_scores`
- `lifecycle_action`
- `exit_intent_id`
- `forced_flat_due`
- `forced_flat_triggered`
- `disconnect_while_holding`
- `final_position_state`

## Fail-Closed Audit Rules

The audit must mark a run not ready if any of the following are true:

- Any no-order or dry-run row has `broker_order_endpoint_called=true`.
- Any paper-submit row has `real_money=true`.
- Any broker submit occurs without a prior guard pass in the same intent chain.
- Any order quantity exceeds one contract.
- Any BUY intent occurs while open-position count is already at or above one.
- Any selected option quote lacks raw quote timestamp or quote age.
- Any decision lacks candidate set hash, feature vector hash, raw logits, margin, threshold, or selected-action metadata.
- Any no-entry decision lacks a no-entry reason.
- Any guard block lacks structured block reasons.
- Any submitted order lacks final broker status, fill, cancel, or timeout.
- Any entry fill lacks subsequent lifecycle state.
- Any held position lacks exit, forced-flat, or explicit still-open state.
- Any raw account id is logged.
- Any required field is missing, null, or placeholder when the event type requires it.

## Stage Acceptance Criteria

Stage 1 preflight can pass only when current connection, paper account, entitlement, env flag, runtime flag, and quote access evidence are present in the same dated packet.

Stage 2 no-order observation can pass only when SPX/VIX/SPXW quotes, context, option ladder, surface score, candidate set, Protocol101 decision, and full reconstruction fields are logged with zero broker endpoint calls.

Stage 3 paper-dry-run can pass only when a real Protocol101 BUY/SELL intent is validated by the guard with `dry_run=true`, `broker_order_endpoint_called=false`, and a complete would-submit payload.

Stage 4 paper-submit can pass only after separate approval and only when one-contract paper order evidence includes guard pass, broker endpoint call after guard pass, order id, status, fill/cancel/timeout, account affordability, fresh quote/context, and no real-money path.

Stage 5 lifecycle can pass only when a filled or held position is detected, lifecycle state is logged, exit or forced-flat intent is generated, and the final state is reconstructable.

Stage 6 observability can pass only when every decision and order chain can be reconstructed from durable logs.

Stage 7 model-improvement readiness can pass only after Stages 1 through 6 pass and Section 3 model gates are reopened.

## Likely Implementation Touch Points

This RFC does not implement code. If approved later, likely touch points are:

- `v4/live/paper_trade_log.py`
- `v4/live/protocol101_entry.py`
- `v4/live/protocol101_live_entry.py`
- `v4/live/ibkr_paper_guard.py`
- `v4/live/ibkr_paper_executor.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
- `v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py`
- `v4/tests/test_protocol142_paper_executor.py`
- `v4/tests/test_protocol158_live_entry_paper_bridge.py`
- `v4/tests/test_protocol160_persistent_paper_trader.py`
- `v4/tests/test_protocol157_daily_ops_monitor.py`

Implementation should be logging/audit first and should not change model decisions, thresholds, candidate filters, order sizing, or trading behavior.

## Proposed Work Plan

1. Add schema-level required fields by event type.
2. Add stable event ids and intent-chain ids.
3. Add candidate-set and feature-vector hashes.
4. Add raw quote timestamp and quote-age propagation into candidate and decision rows.
5. Add raw logits and action-mask logging.
6. Add structured guard-pass and guard-block logging.
7. Add broker status and lifecycle reconstruction fields for future approved paper-submit runs.
8. Add monitor/audit checks that fail closed on missing fields.
9. Add tests for no-order, dry-run, submit, and lifecycle reconstruction contracts using fakes only.
10. Produce a new no-order observation readiness packet after implementation.

## Safety

Paid data approval needed: no.

Broker/runtime risk: none for this RFC. Future implementation should use fake IBKR objects in tests and no-order logs for validation unless a later run plan explicitly approves broker-connected activity.

Paper default unchanged: yes.

Real-money risk: the audit must fail closed if `real_money` is true or unknown.

## Decision Requested

Approve this RFC for implementation as a logging and audit repair project.

Do not approve paper-submit from this RFC alone.

Do not approve model tweaking, hill-climbing, retraining, threshold tuning, or challenger promotion from this RFC alone.

