# SPX 0DTE v4 — Research Protocol

Clean-slate rebuild. v3 stays as-is in [../v3/](../v3/); v4 has no shared imports with v2 or v3.

**Canonical research protocol**: `/Users/gduby/.claude/plans/ok-well-this-just-declarative-puppy.md` (filename is historical; canonical title is `SPX_0DTE_v4_RESEARCH_PROTOCOL.md`).

## Governing principle

> v4 is not a neural-network rebuild. v4 is a causal execution-research system for SPX 0DTE long options. The model is subordinate to the simulator; the simulator is subordinate to point-in-time data; every result is subordinate to out-of-sample falsification. **The hero is the audit trail, not the model.**

## Current status

**Phase 0 build authorized.** Goal: build the minimum auditable v4 research substrate. No paid data, no model training, no dealer-flow conclusions, no neural networks, no live trading until Phase 0 exit criteria are met (see protocol Section 9).

## Directory layout

### Code packages (Python)
- [schema/](schema/) — schema dataclasses for raw / normalized / feature / label / audit rows
- [parser/](parser/) — canonical SPX/SPXW contract ID parser
- [ingest/](ingest/) — deterministic ingest (OptionsDX scaffold first); fingerprinting + hashing
- [checks/](checks/) — bid/ask sanity, timestamp monotonicity, duplicate-key, deterministic-rebuild
- [greeks/](greeks/) — Black-Scholes IV/Greek calculation; reconciliation against vendor Greeks
- [leakage/](leakage/) — planted-leak test, shuffled-label test, leak-detection CI
- [sim/](sim/) — minute-grain simulator skeleton + order-state-machine skeleton
- [tests/](tests/) — pytest suite

### Data directories (Parquet content gitignored)
- [raw/](raw/) — vendor-original messages, immutable, hash-stamped
- [normalized/](normalized/) — unified contract symbology; deterministic from raw
- [feature/](feature/) — causal features only (`is_live_reproducible=True`)
- [label/](label/) — future outcomes; never joined into features

### Markdown artifacts
- [docs/](docs/) — `DATA_CONTRACT.md`, `SIMULATOR_CARD.md`, etc.
- [audit/](audit/) — pipeline integrity reports, leakage audit results
- [ledger/](ledger/) — `RESEARCH_LEDGER.md` (append-only) + per-experiment files
- [promotion/](promotion/) — promotion-packet templates + instances

## Ground rules

1. **No imports from v2 or v3.** Clean slate. Verified by import-check in CI.
2. **Every feature row has `is_live_reproducible`.** If a feature could not exist live at decision_time, it does not belong in the feature layer.
3. **Deterministic rebuild.** Re-running ingest produces byte-identical normalized output.
4. **Leak-detection CI on every commit.** Planted-leak test must catch the planted leak; shuffled-label test must produce no edge.
5. **No paid data until Phase 0.5 vendor verification.** No model training until Phase 0 exit criteria green.

## Phase 0 ticket list

See protocol Section 9.4. Phase 0 is complete when all 15 tickets are merged AND `audit/PIPELINE_INTEGRITY_REPORT.md` is green.
