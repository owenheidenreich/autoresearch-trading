# IBKR Gateway Testing Handoff

Last updated: 2026-05-25 PT  
Repo: `/Users/gduby/Documents/autoresearch-trading`

This handoff describes the work done to prepare the project for the Tuesday
2026-05-26 IBKR Gateway / IBKR paper observation test.

## Purpose

The project is blocked on truthfulness of the trading game, especially execution
realism. The Tuesday test is designed to collect bounded IBKR paper-only
fill/cancel/timeout observations for Protocol101 selected or near-selected SPXW
0DTE candidates.

This is not model training, not threshold tuning, not a strategy promotion, not
live-money trading, and not a production-default change.

## Current Installed Launchd Schedule

All times are America/Los_Angeles.

| Time | LaunchAgent | Purpose | Current loaded state |
|---:|---|---|---|
| 06:05 | `com.autoresearch.tuesday.ibgateway.paper` | Start IB Gateway paper through IBC. | loaded, not running, runs=0 |
| 06:20 | `com.autoresearch.tuesday.protocol101.paper-preflight` | Wait for IBKR API and entitlement probes. | loaded, not running, runs=0 |
| 06:31 | `com.autoresearch.tuesday.paper-fill-observation` | Run live surface check, then bounded paper fill probes. | loaded, not running, runs=0 |
| 13:15 | `com.autoresearch.tuesday.evidence-postsession` | Regenerate readiness/evidence packets. | loaded, not running, runs=0 |
| 13:35 | `com.autoresearch.tuesday.ibgateway.paper-shutdown` | Stop the paper Gateway stack. | loaded, not running, runs=0 |

The old overlapping jobs were disabled:

- `com.autoresearch.protocol101.paper-session`
- `com.autoresearch.premiumblend.no-order-surface-check`
- `com.autoresearch.tuesday.no-order-evidence`

## Paper-Submit Approval Scope

The user approved paper-submit observations for the Tuesday execution test.
That approval was encoded narrowly:

- `V4_ALLOW_IBKR_PAPER_ORDERS=YES`
- `TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED=YES`
- `PROTOCOL101_ENTRY_BRIDGE_MODE=paper-submit`
- `PROTOCOL101_ENABLE_PAPER_ORDERS=YES`
- `PROTOCOL101_ACKNOWLEDGE_PAPER_LOSS=YES`

The approval is bounded by the Tuesday runner caps:

| Cap | Default |
|---|---:|
| Max probes | `8` |
| Max filled round trips | `3` |
| Max observations per contract | `2` |
| Entry timeout | `12s` |
| Exit timeout | `12s` |
| Forced exit limit offset | `$0.50` |
| Quantity | `1` contract |
| Paper cash assumption | `$10,000` |

Do not submit additional orders manually during the post-session review.

## Key Files Added Or Changed

| Path | Purpose |
|---|---|
| `docs/TUESDAY_NO_ORDER_EVIDENCE_COLLECTION_PLAN.md` | Human-readable Tuesday paper-fill observation plan. |
| `v4/scripts/run_tuesday_protocol101_paper_fill_observation.py` | Broker-connected IBKR paper-only fill probe runner. |
| `v4/ops/ibkr/run_tuesday_paper_fill_observation_collection.sh` | Date-guarded launch wrapper for Tuesday. |
| `v4/ops/launchd/com.autoresearch.tuesday.paper-fill-observation.plist` | LaunchAgent for the Tuesday paper observation run. |
| `v4/ops/launchd/install_tuesday_no_order_evidence.sh` | Installer for the Tuesday launchd stack; name is stale, behavior is paper-fill observation. |
| `v4/scripts/run_tuesday_no_order_evidence_packet.py` | Consolidates Tuesday outputs into a review packet. |
| `v4/scripts/run_protocol272_fill_model_readiness.py` | Adds an execution-truth packet gate before the 30-fill calibration gate. |
| `v4/scripts/run_unified_neural_training_readiness.py` | Renames/adjusts the fill gate to execution realism fill evidence. |
| `v4/tests/test_tuesday_no_order_evidence_launchd.py` | Static tests for schedule, approval flags, caps, and packet behavior. |

## Runner Behavior

The paper-fill runner does the following:

1. Connects to IBKR paper via Gateway.
2. Requests SPX and VIX context plus an SPXW 0DTE option ladder.
3. Runs the existing Protocol051 surface scorer.
4. Builds the current Protocol101 candidate frame from live quotes.
5. Prefers the Protocol101 selected candidate when it exists.
6. Otherwise probes a near-selected/top-edge candidate that is fresh, affordable,
   non-crossed, and within per-contract caps.
7. Submits a one-contract BUY limit at current ask in the IBKR paper account.
8. Waits for fill/cancel/timeout.
9. If filled, immediately submits a one-contract SELL paper exit at bid.
10. If the first exit does not fill, tries a forced exit with a `$0.50` lower
    limit offset.
11. Stops at max probes, max filled round trips, market close, or open-position
    risk.

The purpose is to collect execution observations, not maximize PnL.

## Expected Tuesday Outputs

Main paper trade logs:

- `v4/logs/paper_trading/2026-05-26/tuesday_paper_fill_observation_2026-05-26_surface_parity.jsonl`
- `v4/logs/paper_trading/2026-05-26/tuesday_paper_fill_observation_2026-05-26_protocol101_fill_probes.jsonl`

Paper-fill observation audit:

- `v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/tuesday_paper_fill_observation_2026-05-26_protocol101_fill_probes/summary.json`
- `v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/tuesday_paper_fill_observation_2026-05-26_protocol101_fill_probes/report.md`
- `v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26/tuesday_paper_fill_observation_2026-05-26_protocol101_fill_probes/execution_observations.jsonl`

Post-session evidence packet:

- `v4/audit/autoresearch/tuesday_no_order_evidence_packet/summary.json`
- `v4/audit/autoresearch/tuesday_no_order_evidence_packet/report.md`

Readiness packets regenerated post-session:

- `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json`
- `v4/audit/autoresearch/live_no_order_full_action_parity_readiness/summary.json`
- `v4/audit/autoresearch/untouched_holdout_availability/summary.json`
- `v4/audit/autoresearch/unified_neural_training_readiness/summary.json`
- `v4/audit/autoresearch/section3_model_experiment_preflight/summary.json`
- `v4/audit/autoresearch/project_section_readiness/summary.json`

## Gate Change: 30 Fills Is No Longer The First Useful Gate

Before this work, the fill gate only recognized `30` observations for a
calibrated stochastic fill model. That is too high for Protocol101's usual
daily trade count.

The new structure is:

| Gate | Requirement | Meaning |
|---|---:|---|
| Execution-truth packet | `8` execution observations and `1` filled round trip | Enough to start answering whether stale quotes/fill assumptions are obviously broken. |
| Calibrated stochastic fill model | `30` observations | Enough to consider a simple empirical fill/slippage model, still requiring separate validation. |

Until the larger calibration gate passes, replay should keep conservative
ask-entry/bid-exit stress assumptions.

## Verification Already Run

Safe checks run before handoff:

```bash
python3 -m pytest -q \
  v4/tests/test_execution_observation_contract.py \
  v4/tests/test_fill_model_readiness_bounds.py \
  v4/tests/test_holdout_availability.py \
  v4/tests/test_live_no_order_parity_readiness.py \
  v4/tests/test_neural_training_readiness.py \
  v4/tests/test_tuesday_no_order_evidence_launchd.py \
  v4/tests/test_protocol245_live_surface_autotest.py \
  v4/tests/test_protocol160_persistent_paper_trader.py \
  v4/tests/test_protocol142_paper_executor.py \
  v4/tests/test_project_sections.py \
  v4/tests/test_model_experiment_preflight.py
```

Result:

```text
45 passed in 1.94s
```

IBC credential readiness was also checked previously:

```text
decision: pass_ibc_credentials_ready_for_cold_start_rehearsal
ibc_installed: true
username_present: true
password_present: true
runtime_config_status: written
broker_order_endpoint_called: false
market_data_endpoint_called: false
```

## Tuesday Human Dependency

If IBKR Gateway requires 2FA, the human must approve it. The automation can
start Gateway and wait for API readiness, but it cannot approve IBKR Mobile/2FA.

## Post-Session Review Checklist

After 13:15 PT on Tuesday, inspect without submitting more orders:

1. Launchd status:

```bash
for label in \
  com.autoresearch.tuesday.ibgateway.paper \
  com.autoresearch.tuesday.protocol101.paper-preflight \
  com.autoresearch.tuesday.paper-fill-observation \
  com.autoresearch.tuesday.evidence-postsession \
  com.autoresearch.tuesday.ibgateway.paper-shutdown
do
  echo "$label"
  launchctl print "gui/$UID/$label" | rg "state =|runs =|last exit"
done
```

2. Paper observation summary:

```bash
find v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26 -name summary.json -print
```

3. Execution observation rows:

```bash
find v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation/2026-05-26 -name execution_observations.jsonl -print -exec wc -l {} \;
```

4. Consolidated evidence packet:

```bash
python3 -m v4.scripts.run_tuesday_no_order_evidence_packet --session-date 2026-05-26
```

5. Fill readiness:

```bash
python3 -m v4.scripts.run_protocol272_fill_model_readiness --skip-ledger
```

6. Training readiness:

```bash
python3 -m v4.scripts.run_unified_neural_training_readiness --skip-ledger
```

Do not train, tune thresholds, score protected holdout data, submit additional
orders, or call broker endpoints during the post-session review.

## Failure Modes To Check

| Failure mode | Where to look |
|---|---|
| Gateway did not start | `~/Library/Logs/autoresearch-trading/tuesday-ibgateway-paper.*.log` |
| Missing credentials | `v4/audit/autoresearch/v4_aplus_hypothesis_146_ibc_credential_readiness/summary.json` |
| 2FA not approved | Gateway/preflight logs under `~/Library/Logs/autoresearch-trading/` |
| No market data / entitlement issue | `tuesday-protocol101-paper-preflight.*.log`, Protocol245 summary |
| No candidates | paper-fill summary and `candidate_set` rows in paper logs |
| Stale quotes | `execution_observations.jsonl`, `quote_age_ms`, `raw_quote_timestamp`, `received_timestamp` |
| Open position risk | paper-fill summary `decision`, `open_position_risk` rows |
| Unattributed broker rows | `v4/audit/autoresearch/tuesday_no_order_evidence_packet/summary.json` |
| No fill/cancel/timeout evidence | Protocol272 fill readiness summary |

## Current Readiness As Of Handoff

Current summaries still block model hill climbing because Tuesday has not run:

- `section_1_2_decision`: `section_1_2_ready`
- `model_hill_climb_decision`: `model_hill_climb_blocked_until_truth_gates_pass`
- Fill readiness: `blocked_insufficient_fill_observations_keep_stress_replay`
- Execution observations: `0`
- Round-trip fills: `0`
- Challenge blockers: execution realism fill evidence, untouched holdout data availability, live no-order full-action parity

## Important Safety Notes

- Real-money trading remains disabled.
- Operational/default strategy remains unchanged.
- This test is IBKR paper-only.
- The Tuesday approval is bounded to the installed runner and caps above.
- The post-session audit sets `V4_ALLOW_IBKR_PAPER_ORDERS=NO`.
- Any work beyond the bounded Tuesday observation caps needs fresh human approval.
- Do not treat Tuesday paper fills as proof of profitability; they are evidence
  about execution realism and quote/fill truth.
