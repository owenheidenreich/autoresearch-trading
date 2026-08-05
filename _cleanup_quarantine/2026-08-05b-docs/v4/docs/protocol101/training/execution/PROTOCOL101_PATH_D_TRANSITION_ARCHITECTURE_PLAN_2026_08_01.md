# Protocol101 Path-D transition architecture plan

- Date: 2026-08-01
- Status: additive transition plan; offline steps 1–6 implemented, independent verification pending
- Decision clock: `received_timestamp_utc`

## 1. Purpose

Path D removes a structural dependency between model research and IBKR market
recordings. Historical and future decision inputs use the same vendor families:
Databento OPRA for SPXW options and ThetaData for SPX context. IBKR remains the
execution/account/safety plane. The architecture must establish that split
without changing the current paper default.

This plan incorporates the evidence in the Path-D transition plan, the cadence
adversarial review, the walking-skeleton learnings ledger, and current code:

- historical market features currently originate in
  `v4/dataset/spxw_0dte_neural.py`;
- live-style construction currently has a separate implementation in
  `v4/live/protocol101_live_entry.py`;
- the existing shared option-feature rules are in
  `v4/live/protocol101_feature_contract.py`;
- the legacy order-state scaffold lacks explicit unknown/reconciliation states;
  and
- the legacy paper executor waits and requests cancellation without providing
  the Path-D cancel-confirmation/reconciliation contract.

Those are evidence for an additive boundary, not authority to mutate the
legacy runtime.

## 2. Target dependency graph

```text
contracts (stdlib only)
  ^          ^             ^
  |          |             |
features   decision       risk
  ^          |             |
  |          +------ intent+governor decision
compat                     |
                           v
                    execution port
                           |
                   simulated executor

future, deferred:
Databento/ThetaData adapters -> canonical events
IBKR gateway adapter -> execution port
```

The decision package cannot see executors, gateways, IBKR objects, account
clients, or paper-runtime modules. The executor cannot submit without an
`ALLOW` `GovernorDecisionV1` bound to the exact semantic `intent_id` and broker
snapshot version.

## 3. Package topology

| Package | Responsibility | Allowed dependencies | Forbidden dependencies |
|---|---|---|---|
| `v4.path_d.contracts` | Strict wire types, schemas, IDs, executor port | Python standard library and relative contract modules | pandas, numpy, vendor SDKs, IBKR, legacy runtime |
| `v4.path_d.features` | Source-neutral numerical feature calculations | numpy; canonical values | vendor clients, broker state, execution |
| `v4.path_d.compat` | Temporary legacy DataFrame adapters and parity proof | features, legacy feature surface, pandas | broker and live submission |
| `v4.path_d.decision` | Received-clock state, snapshot, deterministic fixture intent | contracts, features | executor, broker, IBKR, paper runtime |
| `v4.path_d.risk` | Limits, preconditions, feed health, deterministic lifecycle safety, sole authorization | contracts | trained policy, vendor/broker client calls |
| `v4.path_d.execution` | Executor port implementations, lifecycle, virtual clock, latency bounds | contracts; offline pandas/pyarrow only in the evidence harness | live broker connection in steps 1–6 |
| `v4.path_d.observability` | Reserved transcript/report boundary | contracts | runtime mutation |
| `v4.path_d.runtime` | Reserved future composition root | none in steps 1–6 | current paper runtime integration |

## 4. Contract laws

All v1 object schemas set `additionalProperties: false`, use integer micros for
option prices, preserve the exact padded 21-character OSI symbol, and parse
unknown keys as errors. Golden fixtures cover:

- `ExecutionIntentV1`
- `GovernorDecisionV1`
- `ExecutionEventV1`
- `BrokerStateSnapshotV1`
- `CanonicalMarketEventV1`
- `FeatureSnapshotV1`

`ExecutorPort` is a standard-library `Protocol`.

### Semantic intent identity

`intent_id = sha256(canonical_json(semantic_payload))`.

Identity includes the schema and parent identity, origin, producer component,
strategy/artifact/feature identities, action, side, position effect, urgency,
quantity, reason, exact contract, price budget, decision clock and causal
watermarks, decision availability, validity deadline, position precondition,
and execution profile. It excludes `trace_id` and the measurement-only
model-start/model-finish/intent-emitted timestamps. Tests prove that changing a
measurement field leaves the ID stable while changing the hard limit changes
the ID.

`WAIT` and `HOLD` produce no intent. A `FORCED_FLAT` intent can originate only
from the governor.

## 5. Shared feature migration

The source-neutral builder implements the existing seven-element market vector
with the existing arithmetic and native float64 layout. The compatibility shim
adapts the existing historical DataFrame shape into canonical observations.
Tests compare both single-step and rolling-window outputs against the current
legacy functions using exact `tobytes(order="C")` and SHA-256 equality.

Legacy consumers remain untouched in this stage. Rewiring them before the
offline boundary is independently reviewed would create an unnecessary
behavior change and collide with current work in those files. A future
migration may redirect consumers one at a time only after parity remains green.

## 6. Offline decision and deterministic safety

Historical events are first represented as strict canonical events. Market
state accepts them only in nondecreasing `received_timestamp_utc` order and
selects only events available at the decision clock. The offline service
requires Databento option quotes plus ThetaData SPX context, builds a feature
snapshot, applies an explicitly deterministic exit fixture, and emits a
serialized close intent.

The governor then independently validates:

- exact broker snapshot version and position precondition;
- quantity, open-position, affordability, and daily-loss limits;
- broker connectivity;
- intent validity and age;
- option/SPX feed availability and age; and
- exact held-contract identity.

The lifecycle safety law is intentionally untrained: an upward-only floor from
the running maximum, a fixed maximum giveback, a fixed time stop, and holding
feed loss forced flat. The parameter defaults in this overlay are offline test
fixtures, not promotion-governance parameters.

## 7. Simulated execution and latency evidence

The simulated executor uses a virtual monotonic clock and logs every transition
as `ExecutionEventV1`. Covered paths are full fill, no fill, partial fill,
cancel confirmation, late fill after cancel request, reject, disconnect to
`UNKNOWN_RECONCILE`, and reconciliation to filled or cancelled.

The latency harness reads only the six owned paired sessions recorded in the
WS2 parity methodology. At 100/250/500/1000 ms it compares the Databento
decision BBO with the locally captured IBKR BBO at virtual arrival. CBBO and BBO
do not contain trades or queue position, so the harness reports a lower fill
bound of zero; a marketable observed BBO supplies only an upper bound of one.
No actual-fill claim is permitted.

## 8. Migration sequence

1. Orientation and module boundaries — implemented.
2. Strict contracts and golden fixtures — implemented.
3. Shared builder and compatibility parity — implemented additively.
4. Offline decision service — implemented.
5. Deterministic governor — implemented.
6. Simulated executor and owned-session latency bounds — implemented.
7. Live market/gateway adapters and runtime composition — deferred.
8. Any paper-runtime integration or default consideration — deferred and
   separately governed.

## 9. Stop condition

After the delta-scoped tests, E2E transcript, import-boundary audit, and latency
tables pass, stop for independent Claude verification. Do not connect to a
broker, contact a paid feed, train a model, or integrate with Protocol158/160.

