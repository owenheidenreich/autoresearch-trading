# G1 ES Direction Screen

**State: RELEASED 2026-08-05 by the measurement review (verdict B), under "large edge or stop".
Family FROZEN 2026-08-06.** Work started 2026-08-06 in parallel with the Track-A capture, which G1 does
not depend on: M1 and M3 read owned ES bars only, and per the
[gate-chain audit §2.4](../../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) no barred option feature
is on this screen's critical path.

## Purpose

Answer the project's only economic question: can a simple, causal policy predict ES direction over 15,
30, or 60 minutes strongly enough to beat 0.358 ES points per completed round trip?

The measurement review already bounds what an answer can mean. On 254 owned sessions the screen can
detect a **large** edge (roughly 2.2–4.0 net points/session, horizon-dependent) and cannot resolve a
merely cost-scale one. A negative result therefore means "no large edge here", never "no edge exists".

## Frozen family

The complete declaration is code, not prose:
[`v5/research/direction/family.py`](../../research/direction/family.py), content hash
`f43b92c2ed33a84e53732983999fdf889d31453cfe8663e3e09af1f182d8db49`.

That hash was reissued from `5ec8b5a4…` on 2026-08-06 when the machine's home directory was renamed and
the declared corpus root stopped resolving. Exactly one of 213 declaration fields changed — the path —
and no outcome had been inspected. Both hashes and the proof: [re-freeze record](REFREEZE_2026_08_06.md).

- **M1** — first-five-minute acceptance, taken when opening volume is at or above the expanding median
  of strictly earlier sessions.
- **M3** — non-roll overnight gap.
- **JOINT** — the single predeclared interaction: the gap, taken only when the first five minutes
  confirm its sign.
- Each with both directions and horizons 15/30/60, giving **18 members**. That is the multiplicity the
  surrogate campaign controls.

Eligible index, verified from the bars on 2026-08-06 using session dates and instrument ids only:
**254** sessions for M1, **249** for M3 and joint after excluding the first session and the four contract
rolls (2025-09-22, 2025-12-22, 2026-03-23, 2026-06-19).

Decision at 09:35:00 ET on bars 09:30–09:34; entry at the **close** of the 09:35 bar, which is
deliberately a minute later than the decision because the ES emission lag is UNCERTIFIED.

## Order of work

1. ~~Freeze the family, clock, economics, folds, comparator rule, and surrogate gates.~~ **Done
   2026-08-06**, before any outcome was inspected.
2. Build the session loader, the per-member replay, and the family economic gate.
3. Build matched surrogates, the bounded-path shared-term fixture, and the injected-effect fixture.
4. Run the 1,000-campaign known-answer study **without inspecting real-policy economics**: false pass
   ≤ 5.0% with Wilson upper ≤ 7.5%, fixture ≤ 5.0%, injected MDE-size effect recovered ≥ 80%.
5. Repair the null **at most once** if that gate fails, then refreeze.
6. Recompute the MDE from the frozen realized occupancy, before reading the economic outcome.
7. One raw economic replay. Report `PASS`, `NO_LARGE_EDGE`, or `UNDERPOWERED` honestly.

## Pass consequence

Only an exact 60-minute pass may reopen one locked option-dollar replay. It does not authorize model
training, threshold tuning, data purchases, paper trading, or promotion. If no member passes, the stop
finding goes into the do-not-retest ledger and the option programme stays closed.
