# Correction to Phase 0b — missing parent edge and clock semantics (2026-08-04)

**Apply before any further certification work.** This corrects an error in Claude's Phase 0 spec, not in
Codex's execution. Codex implemented the specified dependency graph faithfully; the graph was incomplete.

---

## 1. The error

`CODEX_PHASE0_FEATURE_CERTIFICATION_2026_08_04.md` §3 listed **`opra_implied_spot` under "Tier 1 — no
dependencies."** That is wrong. Its certified estimator consumes:

```
input_columns: ['ask', 'bid', 'raw_symbol', 'right', 'strike']
```

Those are OPRA quotes — the same substrate `entry.opra_cbbo1m_native.v1` describes, and whose multi-session
receipt-latency distribution is **not certified** (that is precisely why `cbbo1m_native` is barred).

So `PARENTS` is missing an edge:

```python
"entry.opra_implied_spot.v1": ("entry.opra_cbbo1m_native.v1",),
```

Because `implied_volatility` and `self_computed_greeks` already declare `implied_spot` as parent, adding
this one edge propagates correctly to all 21 Track-B features through machinery that already exists.

## 2. Why it matters — the clock is compute time, not availability

The implied-spot receipt is **honest about what it measured**:

```
availability_clock_kind: measured_local_shared_parity_adapter_compute_p99
local_adapter_compute_p99_ms: 4.898258
```

That is how long the estimator takes to run — 4.9 ms, and 0.0425 ms for the greeks. It is not *when the
feature is available relative to the decision timestamp*. A quantity derived from OPRA quotes cannot be
available before those quotes arrive, and Databento OPRA CBBO p99 was measured at **319.5 ms** in the shared
emission-lag receipt.

The ledger therefore records `availability_clock_ms: 4.898` for features whose true availability is
**unknown and certainly larger**. A model trained under that assumption would make decisions earlier than it
can live — the same failure class as the signed18 look-ahead, arriving from the opposite direction. Wave 2's
H3 assumed data was *slower* than measured; this records it as *faster*.

## 3. Required changes

**3.1 Add the parent edge** shown above. The existing parent rule will move the 21 Track-B features out of
`ADMITTED` automatically.

**3.2 Preserve the completed work.** The Track-B engineering is sound and verified — shared callable hash,
candidate-pair identity, official-SPX residuals with `official_spx_accepted_as_feature_input: False` and
mutation invariance. **Do not re-run or discard those receipts.** They remain valid; only the upstream clock
is missing. Record the blocked reason as parent-not-admitted so the distinction is visible: *the transform
is proven, the arrival time is not*.

**3.3 Make the clock field self-describing.** `availability_clock_ms` is currently a per-family number whose
meaning varies (definition replay warm-up for `contract_clock`, adapter compute p99 for Track B). Carry
`availability_clock_kind` into the ledger row alongside the value, and require for `ADMITTED` that the kind
is an **arrival/availability** measurement, not a compute measurement. A compute p99 may be recorded as an
additive component, never as the availability clock itself.

**3.4 Define composition.** When the parent certifies, the child's availability clock is
`parent_arrival_clock + own_compute_p99`, and that composed value is what an admitted row must carry.

## 4. Corrected certified state

| | Before | After correction |
|---|---|---|
| `ADMITTED` | 29/73 | **8/73** |
| Blocked on Track A | 44 | **65** |

The 8 remain `contract_clock` — calendar and contract geometry, no market observable. **Training remains
correctly blocked**, and Track A now gates 65 of 73 features rather than 38.

This is not a step backwards. The Track-B transform work is done and stays done; what changed is that the
ledger now tells the truth about what is still missing.

## 5. Verification after applying

- `PARENTS` contains the new edge; the Tier-2-before-parent test still passes.
- The 21 Track-B features report parent-not-admitted, with their receipts intact and re-verifiable.
- No `ADMITTED` row carries a compute-kind clock.
- Enforcement still fails closed: training raises on a non-admitted feature, on a barred feature, on a
  missing/empty ledger, and on a tampered hash.
- Regenerate the ledger and report the new `ledger_sha256`.

*Prepared by Claude Opus 5 — 2026-08-04. The error corrected here originated in Claude's Phase 0 spec.*
