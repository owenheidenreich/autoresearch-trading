# Path-D module map

| Path-D module | New responsibility | Evidence/legacy reference | Migration status |
|---|---|---|---|
| `v4/path_d/contracts/_base.py` | Canonical JSON, strict parsing, semantic SHA-256 | Legacy IDs are spread across paper-log helpers | Additive |
| `v4/path_d/contracts/execution_intent.py` | Exact semantic intent and contract identity | `v4/live/ibkr_paper_guard.py` has a smaller paper intent | Additive; legacy untouched |
| `v4/path_d/contracts/governor_decision.py` | Sole-submit authorization receipt | Legacy permission/validation returns dictionaries | Additive |
| `v4/path_d/contracts/execution_event.py` | Strict transcript transition row | `v4/sim/order_state.py` has a smaller lifecycle | Additive |
| `v4/path_d/contracts/broker_state.py` | Versioned account/position precondition state | Current paper executor passes loose account values | Additive |
| `v4/path_d/contracts/market_event.py` | Databento/ThetaData canonical event | Vendor formats currently enter through separate ingestion paths | Additive |
| `v4/path_d/contracts/feature_snapshot.py` | Causal feature identity and watermarks | Current traces use ad hoc feature hashes | Additive |
| `v4/path_d/contracts/executor_port.py` | Dependency-inverted submit/reconcile API | No complete legacy reconcile port | Additive |
| `v4/path_d/features/source_neutral.py` | One shared market/option feature arithmetic surface | Historical `_market_features`; live `_market_features_from_live` | Additive control candidate |
| `v4/path_d/compat/legacy_features.py` | Legacy DataFrame adapter and parity seam | `v4/dataset/spxw_0dte_neural.py` | Temporary shim; no consumer rewiring |
| `v4/path_d/decision/market_state.py` | Received-clock canonical state | `v4/live/protocol101_capture_replay.py` establishes received-clock replay | Additive |
| `v4/path_d/decision/service.py` | Offline snapshot and deterministic fixture intent | No broker imports | Additive |
| `v4/path_d/risk/governor.py` | Deterministic limits/lifecycle safety/authorization | Current guards and lifecycle are separate | Additive |
| `v4/path_d/execution/state_machine.py` | Cancel race, unknown, reconcile lifecycle | `v4/sim/order_state.py` lacks unknown reconciliation | Additive |
| `v4/path_d/execution/simulated.py` | Virtual-clock executor and transcript | Legacy `NullSimulator`/instant-fill rehearsal are insufficient | Additive |
| `v4/path_d/execution/latency.py` | Six-day paired 100/250/500/1000 ms bounds | WS2 owned parity corpus | Additive offline evidence |
| `v4/path_d/execution/fake_gateway_stub.py` | Fake gateway seam only | Live IBKR adapter deferred | Additive test stub |
| `v4/scripts/run_path_d_offline_foundation.py` | Offline E2E evidence composition | No runtime scheduling | Additive offline script |
| Future `v4/path_d/execution/ibkr_adapter.py` | Real broker gateway adapter | Current paper executor | Deferred step 7 |
| Future Path-D runtime composition | Live adapters and guarded paper integration | Protocol158/160 | Deferred; current runtime untouched |

