# Path-D Phase-1 Provisional Fill / No-Fill / Latency Law — PROPOSAL

**STATUS: PROPOSAL (Claude writes → Codex reviews → owner signs).** NOT yet authority.
Defines the **frozen backtest** decision→order→fill/no-fill law used IDENTICALLY across
exit labels, serial replay, realized PnL, and occupancy accounting in Phase-1. Scope:
Tier-S feasibility backtest only. Constituent of the Path-D Phase-1 authority overlay.

**Explicitly provisional:** this is a *backtest execution model*, superseded by the Phase-2
production A1 order state machine (execution-policy contract). It is NOT live-fill parity
and may not support a paper-readiness or trusted-loss-control claim (A7 Tier-S boundary).
Its purpose is a single, conservative, self-consistent fill rule so `Q_exit`/`Q_hold`,
labels, and replayed PnL all agree.

## 1. Why a frozen provisional law is required now

FT2-60's `Q_exit`/`Q_hold` and the FT2-08 exit labels are only well-defined once the
decision→fill mapping is fixed. Leaving it open (v1's defect) makes labels, occupancy,
no-fills, PnL, and opportunity cost ambiguous. This contract freezes that mapping for
Phase-1.

## 2. The law (marketable-limit, conservative)

- **Order model:** a marketable limit priced one valid SPXW tick THROUGH the completed-
  second quote ($0.05 under $3, $0.10 at/above $3): entry BUY at ask+tick, exit SELL at
  bid−tick. (Mirrors the Path-2 review's broker-quote-anchored intent; backtest uses the
  Databento completed-second quote as the execution reference.)
- **Latency:** apply a preregistered decision→fill delay Δ (default from the latency sweep;
  entry at completed-minute `t+1`, exit at completed-second decision + Δ). Report a
  **sensitivity band** over Δ ∈ {0,1,2,5} s so no single Δ is load-bearing.
- **Slippage:** the fill price is the executable quote at (decision + Δ) under the order
  model; realized cross-feed slippage is taken from the latency-sweep distribution
  (median $0, p95 tail-by-Δ) — the ONLY place slippage enters (no double-count elsewhere).
- **Fees:** the frozen per-trade fee, charged once per fill (entry and exit each once).

## 3. No-fill, no-bid, boundaries

- **Exit no-fill:** if the marketable limit would not fill within the frozen window, the
  position **stays open** and the learned exit re-decides next second (the catastrophic
  floor still applies). Exit no-fills are logged and counted separately.
- **No-bid / locked / crossed:** valued per the authority's A2 convention (no-bid ⇒
  adverse/full-loss bound for downside labels; forced-flat realizes actual boundary state).
- **15:55 forced-flat:** mandatory; if a 15:54 exit committed, forced-flat does not
  re-consume the same boundary quote.
- **Occupancy release:** the slot is released at the actual simulated exit-fill time (drives
  the FT2-60 flat-slot opportunity term).

## 4. Identical-use requirement

The SAME law instance (same Δ policy, tick rule, slippage source, fee) is used for: exit
label computation, `Q_exit`/`Q_hold`, one-account serial replay, realized PnL, and
occupancy. A checker rule asserts a single shared fill-law hash across these consumers.

## 5. Conservatism + honesty

- Choose the **conservative** side on ambiguity (fills at the through-quote + latency tail,
  not optimistic mid).
- Report the Δ-sensitivity band; if the economic acceptance result flips across the band,
  route `owner_decision_required` (do not cherry-pick a favorable Δ).
- Highest claim: Tier-S feasibility. Live latency/fill parity is Phase-2 (measured on paper
  days), not asserted here.

## 6. Implementation + verification path (for Codex)

1. Review: is the law complete (fill, no-fill, no-bid, boundaries, occupancy), conservative,
   and internally consistent with FT2-60 v2 + FT2-08 exit? Any place slippage/fees could
   double-count? If blocking: STOP and report.
2. If sound: implement as a frozen Path-D-namespace contract; register its hash in the
   Path-D Phase-1 overlay; add the single-shared-fill-law-hash checker rule.
3. Claude verifies (single shared hash across consumers; Δ-sensitivity band reported;
   conservative direction; no double-count).
4. Owner signs (as part of the overlay).

No training, download, broker/recorder contact, live order, or runtime change is authorized.
