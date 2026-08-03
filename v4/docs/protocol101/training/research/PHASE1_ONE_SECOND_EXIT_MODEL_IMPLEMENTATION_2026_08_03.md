# Phase-1 One-Second Exit Model — Implementation Status

Date: 2026-08-03

## Scope

This implementation is development-only. It uses the owned OPRA `cbbo-1s`
corpus, never requests `cmbp-1`, never imports a broker/order path, never opens
the protected holdout, and does not alter the paper default.

## Implemented

- Environment/CLI-configured data, scratch, and artifact roots.
- External encrypted-APFS volume-name checks and a fail-closed 25% free-space
  rule.
- Checksum-preserving corpus relocation that leaves the original untouched.
- Hash-bound `OOFEntryReceiptV1`; full-fit entry artifacts are rejected.
- Exact session, raw-symbol, and instrument-ID CBBO-1s predicate filtering.
- One-second causal state rows using the corrected exit-47 live-twin contract
  plus 1/5/15/30/60-second P&L velocities.
- OPRA-bid executable P&L; IBKR quotes and portfolio P&L are absent.
- Self-computed IV/delta/gamma; vendor Greeks are absent.
- Features and future-derived A-ref/300-second diagnostic labels are written to
  separate partition trees.
- One-tick-through limits, retry-after-no-fill, zero-bid terminal value,
  0/1/2/5-second latency paths, $3 headline and $4 stress round-trip fees.
- Deterministic net -50% entry-premium catastrophic floor.
- HGB mean/q10/q50/q90 baseline and frozen action utility.
- Five expanding chronological fold construction with one-session embargoes.
- A no-order ThetaData completed-minute timing recorder and a separate shared
  OPRA/ThetaData emission-lag freezer.

## Current Gate State

| Gate | State | Evidence |
|---|---|---|
| Owned CBBO-1s corpus present | PASS | 251 session corpus at the current internal source root |
| Exact-contract predicate loader | PASS | Read-only smoke returned 23,098 rows, one symbol, one instrument, through 15:55 |
| Unit/integration tests | PASS | New tests plus existing Path-D/live-twin tests |
| External 4 TB SSD | BLOCKED | No external physical disk is mounted |
| Corpus relocation | BLOCKED | Requires the external encrypted APFS volume |
| ThetaData clock behavior | PASS | Correct rule confirmed: bar `t` is complete when bar `t+60s` appears |
| ThetaData lag sample power | PASS | Five valid completed-minute samples |
| Shared emission lag freeze | PASS | 2,336 ms, max of ThetaData p99 2,335.230 ms and Databento p99 319.521 ms |
| Repaired causal entry fit | BLOCKED | Shared lag and the remaining live-twin entry gates are not frozen |
| OOF entry receipts | BLOCKED | Cannot lawfully exist before repaired entry folds fit |
| Exit trajectory materialization | BLOCKED | Requires valid OOF entry receipts and relocated roots |
| Exit HGB training/evaluation | BLOCKED | Requires the trajectory dataset |
| Protected holdout | SEALED | Not opened by this work |

The five corrected no-order ThetaData samples ranged from 608.289 ms to
2,335.230 ms after the represented interval end. The frozen shared lag is
2,336 ms. This resolves the live timing measurement, but does not by itself
resolve the remaining multi-session entry live-twin gates.

## Storage Setup

After attaching the SSD, use Disk Utility to format it as encrypted APFS and
name it `AR_TRADING_DATA`. Then configure:

```bash
export AR_TRADING_DATA_ROOT=/Volumes/AR_TRADING_DATA/data
export AR_TRADING_SCRATCH_ROOT=/Volumes/AR_TRADING_DATA/scratch
export AR_TRADING_ARTIFACT_ROOT=/Volumes/AR_TRADING_DATA/artifacts
```

Preflight and relocate without deleting the source:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model \
  storage-preflight --create-roots

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model \
  relocate-corpus \
  --source /Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31
```

Do not remove the internal source corpus until the relocation reports
`COPIED_AND_VERIFIED` and the external manifest has been independently checked.

## Next Lawful Sequence

1. Attach and preflight the 4 TB SSD.
2. Relocate and checksum-verify the corpus.
3. Complete the remaining multi-session live-twin entry gates.
4. Fit the repaired expanding-fold entry HGB and emit immutable OOF receipts.
5. Build exact-contract one-second trajectory partitions.
6. Fit/evaluate the HGB exit policy on development folds only.
7. Freeze the full replay packet before any one-time holdout action.

Phase 2 `cmbp-1` acquisition remains unauthorized unless Phase 1 earns Tier T.
