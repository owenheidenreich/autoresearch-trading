# Codex Goal — Phase 0: Feature Certification + Executable Admission Law (2026-08-04)

Governing architecture: [`PATHD_BUILD_ORDER_2026_08_04.md`](../contracts/PATHD_BUILD_ORDER_2026_08_04.md).
**No model training in this phase.** End with `STOP_FOR_CLAUDE_VERIFICATION`.

---

## 1. Objective

Produce a **signed admission ledger** — per feature, `ADMITTED` or `BARRED`, with the receipt hash that
earned it — and make it **executable**, so training fails closed on any feature not in it.

The rule this serves: *the model must not be able to tell the difference between live OPRA and historical
training data.* Three campaigns have been lost to training before certifying. This phase inverts that.

## 2. Scope — 9 families, 73 features, ONE feed

Certify only families sourced from Databento OPRA or computed from it:

| Family | n | Status today |
|---|---|---|
| `entry.contract_clock.v1` | 8 | `PENDING_SHARED_ADAPTER` |
| `entry.opra_cbbo1m_native.v1` | 11 | `PENDING_SAME_SESSION_REPLAY` |
| `entry.opra_cbbo1m_cross_section.v1` | 11 | `PENDING_SHARED_ADAPTER` |
| `entry.opra_cbbo1s_rolling.v1` | 12 | `PENDING_SAME_SESSION_REPLAY` |
| `entry.opra_ohlcv1m_sparse.v1` | 4 | `PENDING_SPARSE_ZERO_ADAPTER` |
| `entry.opra_implied_spot.v1` | 8 | `PENDING_SHARED_ADAPTER` |
| `entry.opra_implied_volatility.v1` | 6 | `PENDING_CAUSAL_PARENT` |
| `entry.self_computed_greeks.v1` | 7 | `PENDING_CAUSAL_PARENT` |
| `entry.causal_account_state.v1` | 6 | `PENDING_SERIAL_ADAPTER` |

**Out of scope, deferred not abandoned:** `thetadata_completed_spx_vix` (18) and `es_vx_completed_futures`
(5) — each adds a second/third live feed to the decision path. `tcbbo_trade_flow` and `cmbp1_event_flow`
need paid substrate. `feed_health_guard` is `GUARD_ONLY`. `barred_no_live_twin` (10) is permanently
excluded — including **open interest**, which has no live source with matching decision-time semantics.

**Why one feed is sufficient, not a compromise:** the greeks chain is entirely OPRA-derived —
`opra_implied_spot` (spot implied from put-call parity on OPRA quotes) → `opra_implied_volatility` →
`self_computed_greeks`. **No index feed is required to produce greeks.** Official SPX appears only in
`opra_implied_spot` receipt 3, "comparison to completed official SPX **for measurement only**" — a
validation reference, never a feature input. This is the architecture Path-D was designed around; keeping
it means one live subscription carries the entire decision plane.

## 3. Dependency order

**Tier 1 — no dependencies, do these first (six families, 49 features):**
`contract_clock` · `opra_cbbo1m_native` · `opra_cbbo1s_rolling` · `opra_ohlcv1m_sparse` ·
`opra_implied_spot` · `causal_account_state`

**Tier 2 — each depends on a Tier-1 family (three families, 24 features):**
`opra_cbbo1m_cross_section` (← native) · `opra_implied_volatility` (← implied_spot) ·
`self_computed_greeks` (← implied_spot, the "causal spot parent")

A Tier-2 family may not be certified before its parent. `self_computed_greeks` in particular cannot be
admitted on solver-hash and golden-vector evidence alone — its spot parent must be certified first.

## 4. The 27 receipts

Each family's `required_parity_receipts` is authoritative; read them from
`v4/research/autoresearch_v2/entry_live_feature_catalog.py`. Summarised:

- **Same-session value identity** (`cbbo1m_native`, `cbbo1s_rolling`) — live versus Historical-API values
  for the same session must match within a declared tolerance.
- **Receipt-latency distributions** (`cbbo1m_native`, `ohlcv1m_sparse`) — multi-session, so the availability
  clock is measured rather than assumed. **Do not reuse the 2,336 ms ThetaData figure for an OPRA family**;
  Databento OPRA CBBO p99 was measured at 319.5 ms in the same receipt, and conflating the two is the error
  that invalidated Wave 2's H3.
- **Implementation hashes** (`contract_clock`, `cbbo1s_rolling`, `implied_spot`, `implied_volatility`,
  `self_computed_greeks`) — the historical and live paths must run byte-identical code, proven by hash.
- **Golden-vector identity** (`self_computed_greeks`) — fixed inputs, fixed expected outputs, asserted on
  both paths.
- **Mutation / stability tests** (`implied_volatility`, `causal_account_state`) — including mutate-future
  invariance.
- **Ladder and universe identity** (`cbbo1m_cross_section`) — historical and live candidate universes must
  agree.
- **Sparse zero-fill invariance** (`ohlcv1m_sparse`) — per the live audit ruling,
  `SYNTHESIZE_ZERO_ONLY_AFTER_HEALTHY_FROZEN_CUTOFF_NEVER_CARRY`. Never carry a prior bar forward.

**Available substrate:** a live OPRA capture already exists —
`v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/` (510 SPXW 0DTE symbols, status
`AUDITED_NO_MODEL_NO_ORDER_LIVE_FEATURE_SURFACE`) with a feature-surface analysis. Use it before requesting
anything new. If a receipt genuinely needs a fresh capture, say so and stop — do not purchase.

## 5. Deliverable A — the admission ledger

`v4/research/pathd_feature_admission_ledger.py` plus a hashed JSON artifact:

```
schema_version, generated_at, ledger_sha256
features: [
  { name, family, contract_id, status: ADMITTED|BARRED,
    receipts: [{ name, path, sha256 }],
    availability_clock_ms,          # MEASURED for this family, never inherited
    tolerance,                      # declared numeric tolerance for identity
    barred_reason }                 # required when BARRED
]
```

Rules: a feature is `ADMITTED` only when **every** receipt for its family exists, hashes verify, and its
parent family is admitted. Anything else is `BARRED` with a reason. **Partial credit is not a status.**

## 6. Deliverable B — make the law executable (the Phase-0 exit gate)

A ledger nobody enforces is what we already have. Wire `lint_executable_live_twin` into the training path so
that constructing a feature matrix containing a non-admitted feature **raises**.

Required tests:
1. Training with an admitted feature set **succeeds**.
2. Training with a single non-admitted feature **raises**, naming the feature and its status.
3. Training with a `BARRED` feature (use `intraday_open_interest`) **raises**.
4. An empty or missing ledger **raises** — fail closed, never open.
5. A tampered ledger (`ledger_sha256` mismatch) **raises**.

Test 2 is the one that matters. It is the check that would have stopped Wave 3's greeks, and stopped
`signed18` before it cost the holdout.

## 7. Exit gate

Phase 0 is complete when: ≥1 family is `FIT_READY` with verified receipts; the ledger artifact exists and
its hash verifies; all five enforcement tests pass; and the per-family availability clock is **measured**
rather than inherited.

**A family that cannot be certified is a finding, not a failure.** Report it as `BARRED` with the receipt
that could not be produced. If the OPRA-only set certifies to too few features to carry a decision, that is
a real architectural result and must be stated plainly rather than worked around.

## 8. Hard stops

- **No model training, fitting, or hyperparameter search in this phase.**
- Protected 36-session firewall SPENT — never reopen; `holdout_open_count` stays 0.
- Do not modify `FILL_LAW`, the causal t−60s clock, the label law, or the OOF firewall.
- No paid data. No broker, order, live-submit, promotion, default, or runtime/launchd changes.
- **Do not weaken a receipt to reach `ADMITTED`.** Widening a tolerance to make identity pass is the same
  class of error as rewriting a frozen receipt to make a test pass.

## 9. Deliverable

The ledger artifact and module, the enforcement wiring and its five tests, a per-family certification report
(receipts produced, hashes, measured clocks, tolerances, and any family that failed with the reason), and an
updated status board. Then `STOP_FOR_CLAUDE_VERIFICATION`.

**Claude will verify:** re-derive the ledger hash; confirm each `ADMITTED` feature's receipts exist and
verify; confirm no Tier-2 family was admitted before its parent; confirm availability clocks were measured
per family rather than inherited from the ThetaData figure; and independently run the five enforcement
tests, especially that training fails closed.

*Prepared by Claude Opus 5 — 2026-08-04.*
