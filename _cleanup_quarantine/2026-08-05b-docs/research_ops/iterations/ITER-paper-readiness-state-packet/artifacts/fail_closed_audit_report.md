# Fail-Closed Audit Report

Date: 2026-05-25

Audit source: fake and fixture logs only.

Broker scripts run: no

IBKR called: no

Paper orders submitted: no

## Audit Layer Implemented

Function: `validate_observability_contract(rows)` in `v4/live/paper_trade_log.py`

Purpose: mark Protocol101 paper-readiness logs as `not_ready` unless required reconstruction fields are present and safety invariants hold.

This is stricter than the existing `validate_trade_log(rows)` compatibility validator.

## Fail-Closed Cases Covered By Tests

- Real-money true or unknown fails.
- No-order or dry-run rows with `broker_order_endpoint_called=true` fail.
- Submitted-order fixture without a prior guard pass fails.
- Quantity above one contract fails.
- Selected quote without raw timestamp fails.
- Selected quote without quote age fails.
- No-entry model decision without a structured no-entry reason fails.
- Raw account id leakage fails.
- Fake broker-status and lifecycle rows require reconstruction fields.

## Passing Fake Chains Covered By Tests

- Complete no-order fixture: market snapshot, candidate set, model decision, risk gate.
- Fake submit chain: prior guard pass followed by one submitted order fixture.
- Fake broker status plus lifecycle state chain.

## Test Command

`python3 -m pytest -q v4/tests/test_protocol101_observability_contract.py v4/tests/test_paper_trade_log.py v4/tests/test_protocol142_paper_executor.py v4/tests/test_protocol158_live_entry_paper_bridge.py v4/tests/test_protocol160_persistent_paper_trader.py v4/tests/test_protocol157_daily_ops_monitor.py v4/tests/test_protocol244_paper_runtime_shell.py v4/tests/test_protocol148_150_live_ops.py v4/tests/test_protocol155_live_timing_evidence.py`

Result: `59 passed in 1.52s`

## Interpretation

The audit layer is implemented and fake-verified. It does not prove live IBKR paper trading works. It only proves that future fake/no-order/dry-run/submit/lifecycle logs can be judged fail-closed against the reconstruction contract.

