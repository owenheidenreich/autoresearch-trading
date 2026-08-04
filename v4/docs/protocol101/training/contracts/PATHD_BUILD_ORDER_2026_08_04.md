# Path-D Build Order — certify first, train second

**Status: proposed architecture, owner-directed 2026-08-04.** This document fixes the ORDER in which the
system is built. It supersedes ad-hoc wave planning; every future wave must state which phase it belongs to.

---

## 1. The rule that generates the order

> **The model must not be able to tell the difference between live OPRA and historical training data.**

Everything below is a consequence. If a feature cannot be produced live with the same value at the same
decision timestamp, it must never enter training — because a model trained on it is not the model that will
run, and every hour spent evaluating it is measuring a system that does not exist.

## 2. Why the current order keeps failing

The project has trained first and certified afterwards, three times, and paid for it three times:

| Attempt | What happened | Cost |
|---|---|---|
| `signed18` | Trained, produced +$540/session, *then* the parity gate found a 60 s SPX look-ahead | The protected holdout, spent proving it wrong |
| June live bridge | Trained, then live produced **0 signals against 38 historical** | Weeks of parity forensics |
| Wave 3 (Claude, 2026-08-04) | Pre-registration specified delta/gamma/theta/vega — all `PENDING_CAUSAL_PARENT`, not admissible | Caught before running, but only by accident |

The failure is structural, not individual. **Nothing in the codebase prevents training on an uncertified
feature.** A catalog exists (`entry_live_feature_catalog.py`) with a `fit_status` per family and a
`lint_executable_live_twin` linter — but nothing calls it on the training path, so it documents the law
without enforcing it.

## 3. Current certification state — nothing is ready

All 16 families, from the catalog:

| Family | Features | Status |
|---|---|---|
| opra_cbbo1m_native | 11 | `PENDING_SAME_SESSION_REPLAY` |
| opra_cbbo1s_rolling | 12 | `PENDING_SAME_SESSION_REPLAY` |
| contract_clock | 8 | `PENDING_SHARED_ADAPTER` |
| opra_cbbo1m_cross_section | 11 | `PENDING_SHARED_ADAPTER` |
| opra_implied_spot | 8 | `PENDING_SHARED_ADAPTER` |
| opra_implied_volatility | 6 | `PENDING_CAUSAL_PARENT` |
| self_computed_greeks | 7 | `PENDING_CAUSAL_PARENT` |
| causal_account_state | 6 | `PENDING_SERIAL_ADAPTER` |
| opra_ohlcv1m_sparse | 4 | `PENDING_SPARSE_ZERO_ADAPTER` |
| session_static_prior_day | 4 | `PENDING_PREOPEN_REPLAY` |
| thetadata_completed_spx_vix | 18 | `PENDING_LIVE_RECEIPT_PROOF` |
| es_vx_completed_futures | 5 | `PENDING_COVERAGE_AND_LIVE` |
| opra_tcbbo_trade_flow | 7 | `NEEDS_HISTORICAL_SUBSTRATE` |
| opra_cmbp1_event_flow | 6 | `NEEDS_HISTORICAL_SUBSTRATE` |
| feed_health_guard | 8 | `GUARD_ONLY` |
| **barred_no_live_twin** | **10** | **`BARRED`** |

**Zero are `FIT_READY`.** The BARRED family includes **open interest** (`last_causal_open_interest`,
`intraday_open_interest`) — there is no live source with matching decision-time semantics, so it can never
be used — alongside the obvious label leakage (`future_mfe`, `future_pnl`, `oracle_action`,
`session_final_rank`).

## 4. The build order

### Phase 0 — Feature certification (produces the admission ledger)

Execute each family's `required_parity_receipts`. Output is a **signed admission ledger**: per feature,
`ADMITTED` / `BARRED`, with the receipt hash that earned it.

**Scope v1 deliberately to ONE feed.** Certify only the families that come from Databento OPRA or are
computed from it:

- `opra_cbbo1m_native` (11) · `opra_cbbo1s_rolling` (12) · `contract_clock` (8) ·
  `opra_cbbo1m_cross_section` (11) · `opra_implied_spot` (8) · `opra_implied_volatility` (6) ·
  `self_computed_greeks` (7) · `causal_account_state` (6) · `opra_ohlcv1m_sparse` (4)

≈ **73 features from a single decision plane.**

**Explicitly deferred, not abandoned:** ThetaData SPX/VIX (18) and ES/VX (5). Each adds a *second and third
live feed* to the decision path. Path-D exists precisely to escape multi-vendor parity work, and every time
a second feed crept back in, parity cost dominated the research. They may be certified later, as a separate
decision, after a one-feed model has shown something.

**Deferred on cost:** `tcbbo_trade_flow` and `cmbp1_event_flow` need paid historical substrate.

Most of Phase 0 is **offline**: same-session replay, golden-vector identity for the greeks (prove identical
math and inputs give identical numbers), shared adapter work. The only live-capture requirement in scope is
the OPRA same-session replay, and a live OPRA capture already exists
(`databento_live_opra_training_twin_2026_08_03`, 510 symbols, audited).

**Phase 0 exit gate:** ≥1 family `FIT_READY` with receipts, and the linter wired as a blocking check.

### Phase 1 — Make the admission law executable

**This is the single highest-leverage item in the plan.** Wire `lint_executable_live_twin` into the
training path so a feature outside the admission ledger raises rather than trains. A law nobody can violate
is worth more than a document everybody agrees with.

Same pattern as the prior-art check in `pathd_research_loop.py`: it stopped being advice the moment it
could refuse.

**Exit gate:** a test proving training *fails closed* on a non-admitted feature.

### Phase 2 — Execution certification (IBKR)

IBKR is execution-only; it supplies no alpha features. What it must certify is that a decision the model
makes can actually be filled: order preview → dry run → guarded paper submit, with fill/latency/rejection
evidence. This runs in parallel with Phase 0 because it shares no dependencies.

**Exit gate:** guarded paper round trips with recorded fills, slippage against the assumed fill law, and
rejection handling.

### Phase 3 — Train entry+wait and exit+hold on admitted features only

Only now. Two models, per Full Trader V2 §2.1:
- **flat:** `WAIT` or `BUY` one exact contract (42 slots)
- **open:** `HOLD` or `EXIT`

Architecture constraints already earned by prior evidence: shallow ranker inside a fixed game; detached side
head; recovery-aware exit penalty with a mandatory deterministic fallback; no score-thresholding; regime
gates at deployment only.

### Phase 4 — Autoresearch loop inside the constraint set

`run_wave` with bounded budget, semantic dedup, prior-art blocking, and the acceptance gate. Feature
variation happens **within the admitted ledger** — the search space is now provably live-executable, so a
positive result is a candidate rather than a parity autopsy.

### Phase 5 — Deploy decision

Forward live-paper confirmation, then a separate owner packet. The protected holdout is SPENT; fresh live
paper is the only out-of-sample left.

## 5. What this changes immediately

- **Wave 3 as frozen is suspended.** It specifies greeks that are `PENDING_CAUSAL_PARENT`. It becomes a
  Phase-3 candidate, to be re-frozen once the ledger admits its features. Its *analysis* — the premium-band
  confound, and that the 10:30 effect survives all five delta bands — stands and carries forward.
- **No training of any kind until Phase 0 produces a ledger and Phase 1 enforces it.**
- Every future wave declares its phase in its pre-registration.

## 6. The honest expectation

Phase 0 is unglamorous and produces no PnL. It is also the only thing that converts "we hope this works
live" into evidence, and the three failures in §2 each cost more than Phase 0 will.

It may also deliver a negative: if the OPRA-only set cannot be certified, or certifies to too few features
to carry a decision, that is a real finding about whether this architecture can be built at all — and far
cheaper to learn here than after another training campaign.

*Proposed by Claude Opus 5 — 2026-08-04, at owner direction. Requires owner sign-off before Phase 0 begins.*
