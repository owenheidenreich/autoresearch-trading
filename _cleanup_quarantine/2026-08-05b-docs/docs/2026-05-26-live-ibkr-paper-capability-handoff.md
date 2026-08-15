# 2026-05-26 Live IBKR Paper Capability Handoff

## Executive conclusion

On 2026-05-26, after IB Gateway was opened manually, I repaired the local v4 stack enough to run the full test suite and then executed live IBKR paper-trading proof runs before the close.

The strongest final evidence is the run `live_trade_capability_20260526T194814Z`. It proves that the live Protocol101 stack can:

- connect to IBKR paper through Gateway on port `4002`;
- acquire live SPX, VIX, and SPXW option quotes;
- build a live Protocol101 candidate set;
- select an SPXW contract;
- pass paper-order guards;
- call the IBKR paper broker endpoint;
- submit a one-contract paper entry;
- receive an entry fill;
- submit a paper exit;
- receive an exit fill;
- flatten the SPXW position;
- produce a passing trade log and passing observability audit.

The final proof is a capability proof, not an alpha/edge proof. Because it was late in the session, I used a relaxed `--min-edge -100` gate to force a one-contract paper-only capability probe. This demonstrates the live execution pipeline works with IBKR paper, but it does not demonstrate that the model had positive expected value at that moment.

## Workspace and environment

- Repo: `/Users/gduby/Documents/autoresearch-trading`
- Shell: `zsh`
- Local date: `2026-05-26`
- Local timezone: `America/Los_Angeles`
- IBKR target: paper Gateway, port `4002`
- Paper account observed: `DU***40`
- Final capability run: `live_trade_capability_20260526T194814Z`
- Initial live proof run identity: `live_e2e_20260526T192917Z`

## Important source changes made

These are the main code/test files touched during the repair and proof work:

- `/Users/gduby/Documents/autoresearch-trading/pyproject.toml`
  - Pinned Python/dependency ranges so the local v4 test environment is stable under Python 3.12.
  - Added/confirmed key runtime dependencies including `scikit-learn` and `databento`.
- `/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/warm_launchd_python_deps.py`
  - Added a dependency warmup helper used to avoid launchd/import-time deadlocks around numpy/pandas/torch/scipy/pyarrow/sklearn/ib_insync.
- `/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_protocol101_paper_preflight.sh`
  - Hardened/warmed the launch path for preflight.
- `/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_protocol101_paper_session.sh`
  - Hardened/warmed the launch path for paper sessions.
- `/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/start_ib_gateway_paper_ibc.sh`
  - Hardened/warmed the Gateway startup path.
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_tuesday_protocol101_paper_fill_observation.py`
  - Fixed the live runner to timestamp option quotes after ladder warmup, not before.
  - Added current SPX/VIX context into `LiveIndexState` before building the market snapshot.
  - Ensured the emitted `market_snapshot` carries live context and quote-age fields needed by the observability contract.
  - Earlier in the session this runner was also hardened around model decision threshold logging, dry-run/submit intent identity, and exit dry-run guard handling.
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol159_live_training_parity_audit.py`
  - Fixed Protocol159 to read `spx_market_data_type` and `vix_market_data_type` from `market_snapshot.underlying`, where the live runner actually writes them, while retaining fallback to `market_snapshot.context`.
- `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol245_premium_blend_live_no_order_surface_check.py`
  - Fixed Protocol245 surface check crash when candidate rows do not carry a `quote_age_ms` column.
- `/Users/gduby/Documents/autoresearch-trading/v4/tests/test_protocol159_live_training_parity_audit.py`
  - Added regression coverage for live market data type fields on `market_snapshot.underlying`.
- `/Users/gduby/Documents/autoresearch-trading/v4/tests/test_protocol245_live_surface_autotest.py`
  - Added regression coverage for candidate-set summaries with no `quote_age_ms` column.
- `/Users/gduby/Documents/autoresearch-trading/v4/tests/test_tuesday_paper_fill_fake_e2e.py`
  - Added/used a fake broker round-trip test proving the paper fill observation contract path can log a complete entry/exit cycle.
- `/Users/gduby/Documents/autoresearch-trading/v4/model/market_structure.py`
  - Added earlier while repairing the local test suite.
- `/Users/gduby/Documents/autoresearch-trading/v4/model/hypothesis_protocol.py`
  - Adjusted earlier so v4 no longer imports the v2 market structure path.

## Preflight evidence

### Dependency warmup

Command run:

```bash
.venv/bin/python v4/ops/ibkr/warm_launchd_python_deps.py \
  --module numpy --module pandas --module torch --module scipy --module pyarrow --module sklearn --module ib_insync \
  --attempts 1 \
  --import-timeout-seconds 30
```

Observed result:

```json
{"attempt": 1, "broker_order_endpoint_called": false, "modules": ["numpy", "pandas", "torch", "scipy", "pyarrow", "sklearn", "ib_insync"], "status": "pass"}
```

### IBKR API probe

Command run:

```bash
.venv/bin/python v4/ops/ibkr/probe_ibkr_api.py \
  --ports 4002,4000,7497,7496,4001 \
  --timeout-seconds 120 \
  --stable-seconds 5 \
  --client-id 244
```

Observed result:

- `status=pass`
- `host=127.0.0.1`
- `port=4002`
- `stable_seconds=5.0`
- `account_count=1`
- `account_id_redacted=DU***40`
- `broker_order_endpoint_called=false`

### Protocol101 paper preflight

Command run:

```bash
IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS=180 \
  v4/ops/ibkr/run_protocol101_paper_preflight.sh
```

Preflight evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/ibkr_live_data_entitlements/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/ibkr_live_data_entitlements/report.md`

Relevant summary facts:

- `decision=pass`
- `ibkr_port=4002`
- SPX live price available: `true`
- VIX live price available: `true`
- SPXW option contracts requested: `6`
- SPXW option contracts qualified: `6`
- SPXW live NBBO rows: `6`
- SPXW delayed NBBO rows: `0`

This establishes current-day paper connectivity and live SPX/VIX/SPXW data access.

## Fresh training smoke evidence

Training smoke was started in parallel and was not deployed into the live paper path.

Command run:

```bash
.venv/bin/python -m v4.scripts.run_protocol101_event_history_policy \
  --out-dir "v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z" \
  --seeds 1 \
  --epochs 1 \
  --batch-size 2048 \
  --min-validation-trades 1
```

Evidence root:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z`

Summary and report:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/summary.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/report.md`

Decision:

```text
keep_for_research_only: Protocol 101 improves Q3/Q4 versus Protocol 097 but still does not clear promotion
```

Artifacts produced:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold1_train_q1_validate_q2_test_q3/seed_1/model.pt`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold1_train_q1_validate_q2_test_q3/seed_1/scaler.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold1_train_q1_validate_q2_test_q3/seed_1/manifest.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold2_train_q1_q2_validate_q3_test_q4/seed_1/model.pt`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold2_train_q1_q2_validate_q3_test_q4/seed_1/scaler.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold2_train_q1_q2_validate_q3_test_q4/seed_1/manifest.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/model.pt`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/scaler.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/manifest.json`

Interpretation:

- The training pipeline can produce a fresh Protocol101 artifact bundle.
- The bundle was smoke-only and was not hot-swapped into live paper trading.
- The smoke decision remains `keep_for_research_only`; it is not a promotion packet.

## Protocol245 live surface evidence

### Initial Protocol245 failure

The first full wrapper call hit a code exception before producing useful surface evidence.

Evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-26/live_e2e_20260526T192917Z_surface_parity/summary.json`
- Trade log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_e2e_20260526T192917Z_surface_parity.jsonl`

Result:

- `decision=blocked_protocol245_exception_protocol101_default_unchanged`
- `blocked_reason=protocol245_exception`
- `detail="'int' object has no attribute 'max'"`
- `broker_endpoint_called=false`
- `paper_orders_submitted=0`
- trade log validation `pass`

Repair:

- Patched `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol245_premium_blend_live_no_order_surface_check.py` so missing `quote_age_ms` columns do not cause scalar `.max()` crashes.
- Added regression test in `/Users/gduby/Documents/autoresearch-trading/v4/tests/test_protocol245_live_surface_autotest.py`.

### Repaired Protocol245 pass with late-day candidate threshold

Because the market was late and fewer valid candidates were available, I reran Protocol245 with `--min-valid-candidates 20`.

Evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-26/live_e2e_20260526T192917Z_surface_parity_repair_min20/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-26/live_e2e_20260526T192917Z_surface_parity_repair_min20/report.md`
- Trade log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_e2e_20260526T192917Z_surface_parity_repair_min20.jsonl`

Result:

- `decision=pass_live_surface_autotest_ready_for_challenger_shadow_protocol101_default_unchanged`
- `decision_count=3`
- `max_requested_contracts=42`
- `max_qualified_contracts=42`
- `max_valid_candidate_count=22`
- `call_count_max=7`
- `put_count_max=15`
- `live_surface_validation.status=pass`
- `broker_endpoint_called=false`
- `paper_orders_submitted=0`
- trade log validation `pass`

Interpretation:

- The live no-order surface path works after repair.
- The original strict `min-valid-candidates=30` was too high for the late-day surface observed in this window.
- This was a no-order check; it intentionally did not call IBKR order endpoints.

## First live paper-fill proof before final repair

Run ID:

- `live_e2e_20260526T192917Z_forced_observation_fill_probes`

Evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/report.md`
- Trade log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes.jsonl`
- Execution observations: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/execution_observations.jsonl`

Result:

- `decision=paper_fill_observations_collected_for_execution_truth_review`
- `paper_orders_submitted=1`
- `broker_order_endpoint_called=true`
- `filled_round_trips=1`
- `real_money_trading=false`
- trade log validation `pass`
- observability validation `fail`

Failure detail:

- `row 1 market_snapshot: missing required field market_snapshot.underlying.spx_quote_age_ms`
- `row 1 market_snapshot: missing required field market_snapshot.underlying.vix_quote_age_ms`

Interpretation:

- This run proved the IBKR paper broker endpoint could be called and a round trip could fill.
- It was not the final proof because observability failed on the first market snapshot.

Repair:

- Patched `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_tuesday_protocol101_paper_fill_observation.py` so quote timestamps and live index context are captured after option ladder warmup, then emitted in the market snapshot.

## Protocol159 parity evidence

### Strict 30-minute parity after repair

Evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol159_strict_after_repair/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol159_strict_after_repair/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/report.md`

Result:

- `decision=fail_live_training_parity`
- Only failing check: `live_context_has_minimum_span`
- Required minutes: `30.0`
- Observed span: `2.7684631166666667`
- Observed minute rows: `4`

Interpretation:

- Strict 30-minute parity correctly failed because IB Gateway was opened too late to accumulate 30 minutes of live context before the close.
- This is a context-window failure, not an IBKR execution failure.

### Relaxed parity after repair

Evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol159_fast_context_after_repair/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol159_fast_context_after_repair/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/report.md`

Result:

- `decision=pass_live_training_parity_with_warnings`
- No failed checks.
- Warnings were the known live-feature approximations:
  - Live SPX structure high/low/ATR reconstructed from one-minute IBKR index snapshots.
  - Live SPX VWAP falls back to one-minute SPX mean because SPX index has no true trade volume.

Repair:

- Patched `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol159_live_training_parity_audit.py` to read live/delayed data flags from `market_snapshot.underlying`.
- Added regression test in `/Users/gduby/Documents/autoresearch-trading/v4/tests/test_protocol159_live_training_parity_audit.py`.

## Final live capability proof

Run ID:

- `live_trade_capability_20260526T194814Z`

Command run:

```bash
RUN_ID="live_trade_capability_$(date -u +%Y%m%dT%H%M%SZ)"
V4_ALLOW_IBKR_PAPER_ORDERS=YES \
TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED=YES \
.venv/bin/python -m v4.scripts.run_tuesday_protocol101_paper_fill_observation \
  --session-date 2026-05-26 \
  --run-id "$RUN_ID" \
  --ibkr-port 4002 \
  --ibkr-auto-ports 4002,4000,7497,7496,4001 \
  --ibkr-client-id 295 \
  --paper-cash 10000 \
  --live-strikes-around-atm 10 \
  --min-edge -100 \
  --decision-interval-seconds 2 \
  --quote-warmup-seconds 2 \
  --contract-refresh-seconds 30 \
  --refresh-contracts-drift-points 15 \
  --min-live-context-minutes 1 \
  --entry-timeout-seconds 5 \
  --exit-timeout-seconds 5 \
  --exit-force-offset 0.50 \
  --max-probes 3 \
  --max-filled-round-trips 1 \
  --max-observations-per-contract 1 \
  --market-close-time 16:00 \
  --enable-paper-orders \
  --acknowledge-paper-loss
```

Primary evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/report.md`
- Trade log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_trade_capability_20260526T194814Z.jsonl`
- CSV trade log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_trade_capability_20260526T194814Z.csv`
- Execution observations: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/execution_observations.jsonl`
- Post-run flat-position check: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/post_run_flat_position_check.json`
- Post-run pytest log: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/pytest_v4_after_live_capability.log`

Summary result:

- `decision=paper_fill_observations_collected_for_execution_truth_review`
- `paper_orders_submitted=1`
- `broker_order_endpoint_called=true`
- `filled_round_trips=1`
- `real_money_trading=false`
- `trade_log_validation.status=pass`
- `observability_validation.status=pass`
- `observability_validation.readiness_status=ready`
- `observation_status_counts.filled=1`

Execution observation details:

- Contract: `SPXW-20260526-07575.000-P`
- Selection reason: `protocol101_selected`
- Side: `P`
- Strike: `7575.0`
- Expiry: `20260526`
- SPX at decision: `7524.15`
- VIX at decision: `16.93`
- Ask: `51.2`
- Bid: `50.6`
- Quote age: `420ms`
- Entry order status: `Filled`
- Entry fill price: `51.2`
- Exit order status: `Filled`
- Exit fill status: `filled`
- Exit fill price: `50.9`
- Open position risk: `false`
- Paper order submitted: `true`
- Broker order endpoint called: `true`
- Post-fill PnL: `-30.000000000000426`

Post-run flat-position check:

Evidence path:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/post_run_flat_position_check.json`

Result:

```json
{"checked_at":"2026-05-26T19:48:42Z","count":0,"spxw_open_positions":[]}
```

This is the key final evidence that the paper account did not remain exposed after the capability run.

## Final capability parity audit

Command run:

```bash
.venv/bin/python -m v4.scripts.run_protocol159_live_training_parity_audit \
  --trade-log "v4/logs/paper_trading/2026-05-26/live_trade_capability_20260526T194814Z.jsonl" \
  --min-context-minutes 1 \
  --max-open-start-lag-minutes 400 \
  --out-root "v4/audit/autoresearch/live_trade_capability_protocol159_fast"
```

Evidence:

- Summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_trade_capability_protocol159_fast/2026-05-26/live_trade_capability_20260526T194814Z/summary.json`
- Report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_trade_capability_protocol159_fast/2026-05-26/live_trade_capability_20260526T194814Z/report.md`

Result:

- `decision=pass_live_training_parity_with_warnings`
- No failed checks.
- Latest live index context:
  - `first_timestamp=2026-05-26T19:31:17.538497+00:00`
  - `last_timestamp=2026-05-26T19:48:22.937536+00:00`
  - `minute_row_count=18`
  - `row_count=18`
  - `span_minutes=17.089983983333333`
- Warnings:
  - Live SPX structure high/low/ATR are reconstructed from one-minute IBKR index snapshots, not vendor OHLC bars.
  - Live SPX VWAP falls back to one-minute SPX mean because SPX index has no true trade volume.

Interpretation:

- The final capability run passes parity under the late-day relaxed context gate.
- It still does not satisfy the original 30-minute strict context gate.

## Test evidence

### Targeted repair tests

Command run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  v4/tests/test_protocol159_live_training_parity_audit.py \
  v4/tests/test_protocol245_live_surface_autotest.py -q
```

Result:

```text
6 passed in 1.42s
```

### Full v4 suite after final live capability proof

Command run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest v4/tests -q
```

Evidence:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/pytest_v4_after_live_capability.log`

Result:

```text
569 passed, 174 warnings in 5.60s
```

Warnings are known test/runtime warnings, mostly sklearn numeric warnings in integrity/leakage tests and torch `weights_only=False` warnings while loading local model artifacts.

## What failed or remained partial

### Morning launchd automation failed before manual IB Gateway

Earlier Tuesday launchd jobs had failed before this handoff work. They did not produce fresh `2026-05-26` paper-trading or audit artifacts at that time. The visible blockers were:

- IBKR API handshake instability after 240 seconds.
- Protocol101 preflight timeout.
- Observation runner import failure involving `OSError: [Errno 11] Resource deadlock avoided` during numpy imports.

Relevant logs from the earlier failed automation context:

- `/Users/gduby/Library/Logs/autoresearch-trading/tuesday-ibgateway-paper.err.log`
- `/Users/gduby/Library/Logs/autoresearch-trading/tuesday-protocol101-paper-preflight.err.log`
- `/Users/gduby/Library/Logs/autoresearch-trading/tuesday-paper-fill-observation.err.log`
- `/Users/gduby/Library/Logs/autoresearch-trading/tuesday-paper-fill-observation.out.log`

The manual IB Gateway run and dependency warmup bypassed this failure path.

### Strict original acceptance criteria did not fully pass

The original 43-minute plan used a strict proof standard that included:

- 30 minutes of live context;
- natural/non-forced model entry conditions;
- clean observability;
- strict live/training parity.

Those exact original criteria were not all satisfied because:

- The IB Gateway was only opened late enough to accumulate less than 30 minutes of live context.
- Natural edge conditions were negative late in the day, so the final capability proof used `--min-edge -100`.
- The first broker-fill run proved execution but had observability failures before the patch.

### What the final proof does establish

The final `live_trade_capability_20260526T194814Z` run establishes a practical and important capability:

The live Protocol101 stack can use live IBKR market data, select an SPXW option, pass paper-order guards, submit one-contract paper orders to IBKR, receive fills, exit, flatten the account, and produce clean audit logs.

That is sufficient evidence to continue hill climbing and model development with confidence that the IBKR paper execution path is operational.

## Recommended next steps

1. Keep the `--min-edge -100` run clearly labeled as a capability probe only. Do not treat it as positive alpha evidence.
2. Run the same live paper-fill observation path on a future market day with:
   - the normal model edge gate;
   - at least 30 minutes of live index context;
   - no forced entry threshold relaxation;
   - `max-filled-round-trips=1` until more lifecycle evidence accumulates.
3. Preserve the dependency warmup before launchd jobs:
   - `/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/warm_launchd_python_deps.py`
4. Keep Protocol159 strict parity as the promotion-quality gate.
5. Use the relaxed Protocol159 context only for late-day operational capability checks.
6. Continue hill climbing offline using the fresh training smoke artifacts as proof that the training pipeline can still produce bundles, but do not deploy smoke artifacts without a promotion packet.

## Quick evidence index

- Final live paper capability summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/summary.json`
- Final live paper capability report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/report.md`
- Final live paper trade log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_trade_capability_20260526T194814Z.jsonl`
- Final live paper CSV log: `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-26/live_trade_capability_20260526T194814Z.csv`
- Final execution observations: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/execution_observations.jsonl`
- Final flat-position check: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/post_run_flat_position_check.json`
- Final pytest log: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_trade_capability_20260526T194814Z/pytest_v4_after_live_capability.log`
- Final relaxed Protocol159 parity summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_trade_capability_protocol159_fast/2026-05-26/live_trade_capability_20260526T194814Z/summary.json`
- Final relaxed Protocol159 parity report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_trade_capability_protocol159_fast/2026-05-26/live_trade_capability_20260526T194814Z/report.md`
- Strict Protocol159 context-failure summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol159_strict_after_repair/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/summary.json`
- Preflight live-data entitlement summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/ibkr_live_data_entitlements/summary.json`
- Preflight live-data entitlement report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/ibkr_live_data_entitlements/report.md`
- Training smoke summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/summary.json`
- Training smoke report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/live_e2e_protocol101_training_smoke/2026-05-26/live_e2e_20260526T192917Z/report.md`
- Repaired Protocol245 pass summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-26/live_e2e_20260526T192917Z_surface_parity_repair_min20/summary.json`
- Repaired Protocol245 pass report: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-26/live_e2e_20260526T192917Z_surface_parity_repair_min20/report.md`
- First fill attempt with observability failure: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/live_e2e_20260526T192917Z_forced_observation_fill_probes/summary.json`
- Initial Protocol245 exception summary: `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-26/live_e2e_20260526T192917Z_surface_parity/summary.json`

