# Codex Goal — Phase 0b Track A: Multi-Session Live OPRA Capture (2026-08-04)

**Cost question RESOLVED. Owner authorization still required — see §1 before connecting.**
No orders, no broker, no model. End with `STOP_FOR_CLAUDE_VERIFICATION`.

---

## 1. Authorization status

**Zero marginal cost is CONFIRMED** from the owner's Databento plan page:

> **OPRA · Equity options · Tier: Standard · $199/month · Status: Renews 2026-09-01 · Live data: Active**

This is a flat monthly subscription with live already active. The "usage-based rates" language on that page
applies to the *Available subscriptions* list — the datasets not subscribed to, where it reads "Subscribe to
activate live data." OPRA is past that gate. The grandfathered-metering concern raised in Phase 0b does not
apply.

**Do not connect until the owner has given explicit authorization in words.** A pricing page is evidence
about cost; it is not the authorization CLAUDE.md requires for a live market-data connection. If that
sentence is not in the conversation, stop and ask for it.

**Incidental, and it closes a question:** CME Globex MDP3.0 — the ES futures dataset — appears in the
*unsubscribed* list at $199/month. Live ES would be a second subscription. Combined with Protocol 028's
rejection of stitched ES VWAP and Wave 2's H3 failure, ES is closed on cost as well as evidence. Do not
propose it.

## 2. Why this unblocks almost everything

`entry.opra_cbbo1m_native.v1` is barred on **"multi-session local receipt-latency distribution."** Owned
evidence covers one session (2026-08-03). That single missing receipt currently blocks **65 of 73**
features, because after the 2026-08-04 correction the chain runs:

```
cbbo1m_native ──> cross_section
              └─> implied_spot ──> implied_volatility
                              └──> self_computed_greeks
```

Track B's transform receipts are already complete and verified. **Do not re-run them.** What they lack is
the arrival clock this capture supplies.

## 3. Capture design — declare before starting

- **≥5 regular sessions**, declared in the receipt *before* capture begins, not chosen after seeing the
  data. Include an elevated-volatility session if one occurs in the window.
- **No-order, no-broker, read-only stream.** Nothing in this task may touch IBKR or any order path.
- **Record per-message local receipt timestamps.** The measurement is *when data arrived locally relative
  to the interval it describes* — that is the whole point. A capture without local receipt times produces
  no usable receipt.
- **Universe:** the same current-session SPXW 0DTE symbol set as the 2026-08-03 capture (510 symbols) so
  the two are comparable.
- **Watch for tier limits.** The plan is Standard; if a concurrent-connection or symbol cap is hit, record
  it and report rather than trimming the universe silently.

## 4. Measure the clock from OPRA's own source

Three different latency figures now exist in this project and picking the wrong one has already invalidated
a wave:

| Figure | Source | Applies to |
|---|---|---|
| 2,336 ms | ThetaData completed-minute p99 | ThetaData families only |
| 30,603.667 ms | definition replay warm-up | `contract_clock` only |
| **319.5 ms** | **Databento OPRA CBBO p99** | the closest prior for OPRA — supersede it with your own multi-session measurement |

Wave 2's H3 applied the ThetaData figure to ES and was invalidated for it. **Measure per family, inherit
nothing.**

## 5. What the capture serves

- `opra_cbbo1m_native` — multi-session receipt-latency distribution; sparse-minute and freshness receipt;
  same-session live-versus-Historical-API value identity.
- `opra_cbbo1s_rolling` — same-session value identity; reconnect and no-update mutation tests.
- `opra_ohlcv1m_sparse` — OHLCV receipt-latency distribution; sparse zero-fill invariance under
  `SYNTHESIZE_ZERO_ONLY_AFTER_HEALTHY_FROZEN_CUTOFF_NEVER_CARRY` — **never carry a prior bar forward**;
  native OHLCV versus live-trade aggregation identity.
- `opra_cbbo1m_cross_section` — parent receipts, historical/live candidate-universe identity, atomic ladder
  completeness.

## 6. Then compose the Track-B clocks

Once `cbbo1m_native` is admitted with a measured arrival clock, the Track-B families compose:

```
child_availability = parent_arrival_p99 + own_local_compute_p99
```

with `availability_clock_kind = composed_parent_arrival_plus_local_compute` (already in
`ARRIVAL_CLOCK_KINDS`). Their existing receipts stand; only the clock field changes. A row may **never**
be admitted carrying a bare `measured_local_shared_parity_adapter_compute_p99`.

## 7. Exit gate

Every receipt produced and hash-verified; the ledger regenerated and its `ledger_sha256` recomputed; each
newly admitted family carrying an **arrival-kind** clock measured from its own source; and the mechanical
parent test still passing.

**Do not weaken a receipt to reach `ADMITTED`.** If live and Historical-API values do not match within the
declared tolerance, that is a finding about vendor consistency and must be reported as a bar, not tuned
away by widening the tolerance.

Expected on success: roughly **65 of 73** admitted, and training becomes possible for the first time.

## 8. Hard stops

- **No model training, fitting, or hyperparameter search.**
- Protected 36-session firewall SPENT — never reopen; `holdout_open_count` stays 0.
- Do not modify `FILL_LAW`, the causal t−60s clock, the label law, the OOF firewall, the `PARENTS` graph,
  or the enforcement path.
- **No paid downloads.** This is streaming capture under an active subscription; a historical range request
  is a different thing and is not authorized.
- No broker, orders, paper submit, promotion, default change, or runtime/launchd edits.

## 9. Deliverable

Per-family certification report (receipts, hashes, measured clocks with their kind, tolerances, and any
family that failed with the reason), the regenerated ledger with recomputed hash, the updated
`ADMITTED`/`BARRED` count, and an updated status board. Then `STOP_FOR_CLAUDE_VERIFICATION`.

**Claude will verify:** re-derive the ledger hash; confirm each newly admitted family's clock is
arrival-kind and measured from OPRA rather than inherited; confirm Track-B clocks are composed rather than
bare compute; confirm no tolerance was widened between Phase 0b and now; and re-run the enforcement tests
to confirm training still fails closed on a barred feature.

*Prepared by Claude Opus 5 — 2026-08-04.*
