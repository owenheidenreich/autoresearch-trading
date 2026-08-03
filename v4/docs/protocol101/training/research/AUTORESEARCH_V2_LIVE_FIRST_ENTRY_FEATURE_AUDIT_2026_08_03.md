# Autoresearch v2 live-first entry feature audit

Date: 2026-08-03

Status: `FOUNDATION_DESIGN_AND_GATES_BUILT_NOT_FIT_READY`

## Executive answer

Claude is right about the central causal failure and wrong only if “we do not
need a live session” is read as a complete train/live-parity conclusion.

The unavailable ThetaData close was provable offline. We did not need live OPRA
to discover that one-minute look-ahead. But the live session answered different,
load-bearing questions that static schema documentation cannot settle for this
system:

1. Actual gateway and local receipt latency for our subscription and machine.
2. Mid-minute cold-start behavior and the need to discard the first interval.
3. Daily instrument-ID remapping and the need for raw OSI identity.
4. Same-day definition additions that change the candidate universe.
5. Which optional schemas actually emit useful SPXW 0DTE records.
6. Sparse-bar, no-update, trade-side, and full-universe throughput behavior.

Therefore the correct synthesis is:

- Use offline semantics and replay to prove causal clocks and deterministic
  value identity.
- Use bounded live shadow captures to measure availability, warm-up, universe,
  missingness, delivery latency, reconnects, and load.
- Train only through the exact shared adapter that satisfies both.

No model fit is authorized by this audit.

## Response to Claude's other claims

### Correct and adopted

- The former `+$540` confirmation is invalid as a live claim. All 445,063 fitted
  rows used unavailable SPX context, and causal-clock decisions reproduced only
  18.44% of the complete cases.
- Waiting 60 seconds is not a repair. It pairs a later SPX close with an older
  option market and changes the trading game.
- The replacement must use the SPX bar representing `[t-60s,t)`, not the bar
  stamped `t` that represents `[t,t+60s)`.
- The spent confirmation set is development evidence now. A causal replacement
  needs a fresh confirmation epoch.
- A causal rerun may honestly return `NO_INCREMENTAL_EDGE`.

### Corrected or made more precise

- “Features 13–18 were already causal” is too broad. The three option-ladder
  features were decision-local, and `is_call` was causal. Delta and gamma were
  recomputed from the leaked SPX spot, so their fitted values were not causal.
- “Just rebuild and rerun autoresearch” is premature. First freeze the live
  source contract, receipt cutoff, identity law, missingness law, execution
  arrival, and label origin. Otherwise the project can produce another valid
  backtest of a different live game.
- A live session is not a “burn” in the holdout sense. It is model-free systems
  identification. It must not be used to tune model economics, but its latency,
  schema, completeness, and failure observations are required foundation data.

## What the expanded live capture established

The second owner-authorized capture subscribed to the 510 current-session SPXW
0DTE raw symbols for 120 seconds across CMBP-1, TCBBO, trades, OHLCV-1m,
statistics, and status. It touched no model, holdout, broker, or order path.

| Observation | Result | Training/runtime consequence |
|---|---:|---|
| CMBP-1 | 221,549 rows; 510 symbols | Full-universe event ingestion was about 1,838 rows/sec. Combined-stream local p99 lag exceeded 3 seconds. Use a bounded ladder or isolated ingestion and measure it again before making these model features. |
| Trades | 4,174 rows; 100 symbols | Tick trade features exist live, but the historical Path-D corpus does not own trades. New historical substrate is required. |
| TCBBO | 4,174 rows; exact one-to-one keys with trades | It supplies BBO-at-trade for quote-rule classification. It is not currently owned historically. |
| Native trade side | `N` for 4,174/4,174 | Do not call native aggressor side an alpha feature. Buy/sell flow requires a separately tested TCBBO quote rule. |
| OHLCV-1m | 157 rows: 92 then 65 symbols out of 510 | Bars are sparse. A healthy missing exact-contract minute means zero traded volume, not “carry the prior minute.” Zero can be finalized only after the frozen receipt cutoff. |
| Statistics | 0 intraday rows | Open interest is session-static/pre-open at most, never an intraday 90-second carry. |
| Status | 0 event rows in the window | Status is event-driven and belongs in readiness/health guards, with a replay/fault test—not as learned alpha. |

The multi-second CMBP tail is a property of this full-universe combined capture,
not proof of one isolated root cause. It may combine gateway batching, Python
callback pressure, and subscription breadth. The safe conclusion is simply
that this topology is not fit-ready.

## Equality standard for every entry feature

A train/live twin is equal only if all seven dimensions match:

1. **Source:** same native schema or the same raw events feeding one shared
   adapter.
2. **Event interval:** both rows represent the same closed market interval.
3. **Availability:** the row was actually received by the frozen emission
   cutoff.
4. **Universe and identity:** same current-session definitions, raw OSI key,
   candidate filters, and strike ladder.
5. **Missing/carry law:** no-update, zero, missing, and stale are represented
   identically; reconnect resets warm-up.
6. **Computation:** one implementation and constants produce historical and
   live values.
7. **Action game:** entry quote, arrival latency, fill accounting, and label
   horizon start from the same clocks.

Matching columns is necessary but not sufficient.

## Entry feature surface

The executable catalog is
`v4/research/autoresearch_v2/entry_live_feature_catalog.py`. The compiler now
requires its exact contract IDs. A nonempty sentence is rejected.

| Family | Historical data | Live evidence | Current status | Recommended use |
|---|---|---|---|---|
| Contract geometry and clock | Owned definitions/calendar | Definitions observed; daily ID churn measured | `PENDING_SHARED_ADAPTER` | First safe base after its definition/calendar receipts: right, strike, expiry/close clocks, day/early-close. Moneyness is not in this class because it needs spot. |
| Native OPRA CBBO-1m quote/size | 251 sessions owned | Schema, cadence, sizes, and latency observed | `PENDING_SAME_SESSION_REPLAY` | Highest-priority quote, spread, size imbalance, and microprice family. |
| Cross-sectional 0DTE ladder | CBBO-1m + definitions owned | Near-ATM ladder and same-day additions observed | `PENDING_SHARED_ADAPTER` | Straddle, put/call ratio, skew, curvature, depth/liquidity ranks. |
| OPRA CBBO-1s rolling | 251 sessions owned | Live observed | `PENDING_SAME_SESSION_REPLAY` | Short returns, spread/imbalance history, update count, quote age. |
| OPRA OHLCV-1m | 251 sessions owned | Live sparse behavior observed | `PENDING_SPARSE_ZERO_ADAPTER` | Volume/range only after dense zero-finalization and receipt proofs. |
| OPRA TCBBO/trade flow | Not owned | Live observed | `NEEDS_HISTORICAL_SUBSTRATE` | Defer unless a bounded cost/benefit case justifies tick-history acquisition. |
| OPRA CMBP-1 event flow | Not owned | Live observed with high full-universe load | `NEEDS_HISTORICAL_SUBSTRATE` | Defer; if acquired, use a bounded near-ATM ladder and isolated ingestion. |
| OPRA-implied spot | CBBO-1m ladder owned | Parents observed; existing parity prototype | `PENDING_SHARED_ADAPTER` | Strong same-vendor replacement/ablation for Theta SPX context. |
| OPRA-implied vol/skew | CBBO-1m + definitions owned | Parents observed | `PENDING_CAUSAL_PARENT` | ATM IV, skew, curvature, straddle regime after implied-spot proof. |
| ThetaData completed SPX/VIX | History owned | Live receipt timing not observed | `PENDING_LIVE_RECEIPT_PROOF` | Preserve as a separate feature-family ablation, not in the first fit-ready base. |
| Self-computed Greeks | Historical inputs owned | Calculation is live-computable | `PENDING_CAUSAL_PARENT` | Recompute only from causal spot, option price, contract, rate, and time. |
| ES/VX futures context | ES owned; VX incomplete | Live entitlement/clock unproved | `PENDING_COVERAGE_AND_LIVE` | Defer pending coverage and a separate live adapter. |
| Prior-day/session-static | Prior history owned | Pre-open replay unproved | `PENDING_PREOPEN_REPLAY` | OI, prior volume/close/range may be one frozen session value only. |
| Causal account state | Simulator state exists | Live ledger parity unproved | `PENDING_SERIAL_ADAPTER` | Later entry-policy state: prior entries/PnL/occupancy, never session-final totals. |
| Feed/status/missingness | Quality receipts exist | Partial live evidence | `GUARD_ONLY` | Block/WAIT inputs, not alpha features. |
| Intraday OI, future/session-final fields, stable daily IDs, full depth/queue | No matching causal source | None | `BARRED` | Never fit. |

## The recommended first causal feature base

Do not maximize the column count in one model. Maximize the number of separately
testable live-safe families.

The first fit-ready entry base should be built in this order:

1. Contract geometry and deterministic clock.
2. Native CBBO-1m quote/size features.
3. Atomic cross-sectional 0DTE ladder features.
4. OPRA-implied spot and volatility/skew features.
5. CBBO-1s rolling quote/liquidity features.
6. Causal self-computed Greeks from the selected spot parent.
7. Sparse-zero option volume only after its adapter passes.

ThetaData SPX/VIX should enter as a separately paired family after its live
receipt adapter is proved. This design lets autoresearch answer whether it adds
incremental edge over an OPRA-only base instead of making the whole foundation
depend on an unmeasured second vendor.

Trade/TBBO and CMBP event-flow families should not be purchased historically
until the owned-data families have been screened. Their live existence is
proved; their incremental value is not.

## Changes made to autoresearch_v2

The old compiler accepted any nonempty `live_twin` string. That was the escape
hatch through which the invalid clock passed.

It now requires:

- an exact executable catalog contract ID;
- exact feature membership, family, and availability clock;
- `FIT_READY` status; and
- alpha eligibility rather than a guard-only/barred classification.

The former signed-entry hypotheses use prose live-twin claims and now terminate
at compilation before any source decode or fit. The useful families remain
pending until their receipts are added.

## Next build sequence

1. Post-session, compare the captured 2026-08-03 CBBO live DBN to the same raw
   symbols/timestamps from the Historical API.
2. Collect bounded multi-session CBBO-1m/1s and OHLCV receipt distributions;
   isolate schemas so load is attributable.
3. With explicit authorization, capture ThetaData completed SPX/VIX receipt
   timing—or make the OPRA-implied-spot base the first independent path.
4. Freeze emission lag `L`, order-arrival latency, warm-up, reconnect, missing,
   and candidate-universe laws.
5. Build one event-driven adapter used both to materialize historical rows and
   to process live messages. Do not maintain a historical builder and a second
   live imitation.
6. Run golden-row equality, mutate-future, late-source, missing-row, sparse-zero,
   reconnect, definition-add/delete, early-close, and numeric edge-case tests.
7. Materialize a distinct development-only causal dataset. Do not open any
   protected confirmation epoch.
8. Run autoresearch family ablations in the order above, with cached OOF
   predictions, paired deltas, maxT correction, and strict serial replay.
9. Only a development survivor earns one fresh confirmation epoch.

Highest allowed claim: a live-first entry feature map and executable compiler
gate exist; the replacement foundation is not fit-ready and no edge is claimed.
