# Path-D additive overlay

`v4.path_d` is the offline-first decision/execution boundary for the Path-D
redesign. It coexists with the legacy Protocol101 paper runtime; it does not
replace, register, schedule, or call it.

## Packages

- `contracts/`: standard-library-only strict v1 wire contracts, semantic IDs,
  JSON schemas, golden fixtures, and `ExecutorPort`.
- `features/`: source-neutral numerical feature calculations.
- `compat/`: temporary additive shims that prove parity with legacy builders.
- `decision/`: canonical received-clock market state and offline intent
  production. No executor, broker, or IBKR imports.
- `risk/`: deterministic limits, feed/position guards, simple lifecycle safety,
  and the sole submission authorization.
- `execution/`: virtual-clock simulator, full order state machine, fake gateway
  stub, transcript logging, and offline paired-quote latency bounds.
- `observability/`: reserved boundary for contract-based reporting.
- `runtime/`: reserved future composition root; intentionally empty in steps
  1–6.

## Import laws

```text
contracts -> Python standard library only
features  -> canonical values; no vendor/broker clients
compat    -> features + legacy data shapes
decision  -> contracts + features only
risk      -> contracts only
execution -> contracts; offline evidence readers only
```

Only a future `ibkr_adapter.py` may import `ib_insync` or `ibapi`. There is no
real adapter in this stage. `fake_gateway_stub.py` is an in-memory test seam.

## Clock and identity

The only decision clock is `received_timestamp_utc`, matching capture replay.
All option prices are integer USD-option-price micros. Contracts use exact
21-character padded OSI symbols. `ExecutionIntentV1.intent_id` hashes semantic
execution content and deliberately excludes trace and measurement-only timing.

## Offline evidence

Run:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_path_d_offline_foundation
```

This reads only local owned files and writes
`v4/audit/autoresearch/path_d_offline_foundation/`. It does not download data,
contact a broker, fit a model, or mutate runtime state.

