# Path D adversarial verdict

**Verdict: conditional GO for Path D as the lead learned-trader architecture; NO-GO for paper trading or an authority amendment today.**

Path D correctly eliminates the largest model risk in Path B/B-prime: the model would train and infer from Databento OPRA rather than learning on Databento and then seeing IBKR features live. But it does not eliminate cross-vendor risk—it concentrates it in the execution bridge, where failures are asymmetric and potentially more damaging.

The existing minute-cadence authority remains controlling. A1 currently requires next-minute ask/bid fills, not a sub-minute order window. [Authority §2.4](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:194)

## What the evidence actually proves

The quote evidence is encouraging, but it does not prove executable slippage is tolerable:

- Ordinary cross-vendor bid differences were small: weighted MAE was approximately `0.0218` option points, or `$2.18` per contract. Daily p95 differences were approximately `$10–$20` per contract.
- B-prime independently re-derives `0.8897` exit-within-five-seconds agreement, `0.99856` state agreement, and median/p95 exit-price differences of `$0/$20` per contract. [B-prime report](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ws2_bprime_falsification/bprime_report.md:22)
- But one day’s p95 quoted exit difference was `$116`, and its p95 timing difference was 144 seconds.
- Those are vendor-quote/policy comparisons—not Databento intent → network latency → IBKR order → actual fill observations.
- The current execution packet contains only 10 fill observations, reports a 60% fill rate, and has no usable median latency. [Fill readiness evidence](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json:117)

Therefore: **normal-condition reconciliation looks promising; tail execution tolerance is UNKNOWN.** Whether `$20` p95 is acceptable depends on the eventual model’s lower-confidence expected edge. It could be harmless to a large edge or completely consume a thin one.

## Severity-ranked findings

| Severity | Finding | Consequence |
|---|---|---|
| **Critical** | A protective EXIT can become an unplanned HOLD if the market moves through the limit before IBKR receives it. | A stop that does not fill is worse than ordinary slippage; loss continues while the model believes it exited or intended to exit. |
| **Critical** | Databento loss while holding has no safe “abstain” interpretation. | Flat-state outage can block entry; open-position outage requires an independent broker-facing risk governor and forced-flat path. |
| **High** | The proposed A1 replacement is currently a phrase, not a complete order state machine. | Training may assume fills that the live cancel/reprice logic cannot reproduce. |
| **High** | Path D is not actually single-vendor for the full feature vector. | Canonical SPX price action remains mandatory; Databento does not currently provide official SPX/VIX index values. |
| **High** | Current live OPRA entitlement is unproven. | Local evidence shows `license_not_found_unauthorized` on July 24; current entitlement status is `UNKNOWN`. [Local license evidence](/Users/gduby/Documents/autoresearch-trading/v4/audit/protocol101_sync_evidence_2026_07_24_databento_downloads.jsonl:29) |
| **Medium** | Intraday listings, instrument mappings, and reconnect recovery require a new Databento state machine. | A missing or wrongly mapped 0DTE strike could create the wrong candidate universe or order the wrong contract. |

Fast moves, thin or disappearing books, zero bids, wide spreads, feed divergence, local backlog, model latency, broker-routing latency, and cancel/replace races are the conditions most likely to break reconciliation.

## Recommended execution model

Use a **broker-quote-anchored, tick-aware, bounded marketable-limit policy**.

Do not set the order limit directly from Databento. Databento should determine the intent—contract, BUY/SELL, urgency, and adverse-price budget. Immediately before submission, the execution adapter should:

1. Resolve the exact Databento OSI identity to an IBKR-qualified SPXW contract and verify the returned local symbol.
2. Read a fresh IBKR executable quote solely for execution.
3. Reject stale, missing, locked/crossed, or contract-mismatched quotes.
4. Submit a marketable limit through the IBKR quote by one valid tick:
   - BUY: current IBKR ask plus one tick, subject to the entry budget.
   - SELL: current IBKR bid minus one tick, subject to the frozen exit-urgency tier.
5. Use IOC if confirmed supported for this route, or a very short timer with cancel acknowledgement.
6. Requote only a frozen number of times within a frozen total window.
7. Reconcile fills, positions, and open orders before issuing another intent.

SPX/SPXW minimum ticks are ordinarily `$0.05` below `$3` and `$0.10` at or above `$3`; arbitrary cent rounding is not enough. [Cboe SPX specification](https://cdn.cboe.com/resources/spx/spx-fact-sheet.pdf)

Why not the alternatives:

- **Market order:** strongest fill probability, but uncapped tail price and poor historical reproducibility.
- **Patient limit at Databento bid/ask:** creates severe fill-selection bias and is unsafe for protective exits.
- **One-shot marketable limit:** better, but still vulnerable to fast moves.
- **Broker algo:** potentially useful later, but proprietary behavior is harder to reproduce historically.

The current executor is not this policy. It submits `LMT/DAY`, waits—normally 15 seconds—and requests cancellation if unfilled. It does not wait for cancel confirmation before returning. [Executor configuration](/Users/gduby/Documents/autoresearch-trading/v4/live/ibkr_paper_executor.py:21), [cancel behavior](/Users/gduby/Documents/autoresearch-trading/v4/live/ibkr_paper_executor.py:214). That is not sufficient for bounded repricing because a late fill can race the next order.

## A1 change assessment

Changing A1 from next-minute fills to an intent-driven bounded window is **sound and necessary in principle**, but the proposed wording is not complete enough to sign.

The amendment must freeze:

- Databento decision-availability time and local receipt time.
- Model-completion, order-send, broker-ack, fill, cancel-request, and cancel-confirmation clocks.
- Exact IBKR quote-freshness requirement.
- Contract identity and valid tick calculation.
- Initial limit, retry count, repricing rule, total window, and maximum adverse collar.
- Entry no-fill behavior: remain flat; no phantom trade.
- Exit no-fill behavior: position remains open; invoke the next urgency/risk state.
- Late fill after cancel and disconnect-after-submit reconciliation.
- Partial fills, even if quantity one makes them initially rare.
- No-bid, locked/crossed, forced-flat, and 15:55 boundary behavior.
- Fees and D48/D49 accounting.
- The emergency rule when a protective exit exceeds its normal collar.

Every training label, serial replay, realized P&L calculation, floor result, and live order must use that same state machine. Trusted sub-minute labels require raw event-path evidence; `cbbo-1s` cannot prove that an intra-second limit would have filled.

The historical side should replay a virtual broker adapter at several preregistered latency/slippage rungs, then validate those rungs prospectively against IBKR paper evidence. Exact fill parity is impossible from Databento alone; what can be required is **identical order policy plus conservatively calibrated execution uncertainty**.

## ThetaData

ThetaData cannot simply be removed from the current critical path.

The signed feature floor requires canonical SPX price action for VWAP, momentum, moneyness, and internal delta/gamma. [Authority feature floor](/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md:261) Databento states that it does not provide official index values such as SPX; OPRA provides SPX options, not the cash index itself. [Databento index-coverage statement](https://databento.com/stocks)

The current 17-feature alpha list is non-VIX, so VIX may eventually be removable from the model plane. But SPX is not. ThetaData as a particular company is replaceable; the synchronized historical/live official-SPX dependency is not. Removing or deriving SPX from options would be a new feature contract requiring causal validation.

The cleanest Path D model plane may therefore be:

- Databento historical/live OPRA for options.
- ThetaData historical/live—or another single matched source—for SPX context.
- IBKR only for execution quote, account, orders, fills, and safety.

## Databento operations and entitlement

Databento supports heartbeat detection, automatic reconnect policies, reconnect-gap callbacks, and up to 24 hours of intraday replay. A recovered Path D process must resubscribe, replay from the last committed `ts_recv`, deduplicate records, rebuild quote age, and wait for replay completion plus ladder completeness before permitting decisions. [Databento live recovery documentation](https://databento.com/docs/api-reference-live/client/add-reconnect-callback)

For symbols:

- Consume point-in-time definitions and `SymbolMappingMsg` records.
- Key model state by exact padded OSI, not reusable numeric instrument IDs alone.
- Subscribe to a controlled ladder plus held contracts; avoid consuming the entire OPRA feed unnecessarily.
- Handle new intraday strikes and unresolved mappings explicitly.
- Build the model candidate universe from Databento; use IBKR only to qualify the selected contract for execution.

Databento specifically warns that OPRA volume can backlog clients and notes that intraday listings are delivered through timestamped definitions/mappings. [Databento OPRA conventions](https://databento.com/docs/examples/options/equity-options-introduction/opra)

Entitlement cannot be inferred from this repository. OPRA generally treats an individual using data solely for personal/family investment activity as nonprofessional; business/entity use, redistribution, or securities-industry status can make the subscriber professional. [OPRA subscriber criteria](https://www.opraplan.com/faqs) Databento currently lists personal OPRA display/non-display licensing separately from commercial use, including algorithmic non-display use. [Databento licensing requirements](https://databento.com/docs/knowledge-base/portal/billing/manage-monthly-limit)

The owner must complete the portal attestation. The local account did not have live OPRA entitlement on July 24.

## Cheapest decisive test

Run a quarantined **Path D Execution-Bridge Skeleton**, not another model experiment:

1. Free offline rung: replay the six paired days with Databento-generated intents and IBKR received-clock quotes under 100/250/500/1,000 ms latency rungs, valid tick rounding, bounded limits, no-fill, and cancel/retry state.
2. One owner-authorized live rung: three Databento-live + IBKR-paper sessions using deterministic scheduled fixtures, not model-selected trades. Collect at least the already-required 30 fill observations across calls/puts, `$3–$8` premiums, quiet/fast periods, and late day.

The architecture fails cheaply if there is any wrong-contract order, duplicate/late fill, unreconstructable cancel, stale-feed entry, or open-position outage without deterministic risk handling. Economic acceptance must compare p95/p99 execution cost and no-fill opportunity cost against a preregistered fraction of the eventual model’s lower-confidence expected edge—not against an arbitrary `$0.05` or `$20`.

## Comparative disposition

1. **Path D:** lead target architecture, conditional on execution-bridge and entitlement proof.
2. **Path A:** cleanest same-vendor benchmark, but delayed by insufficient IBKR history.
3. **Path C:** pragmatic interim concept, but its one-second floor is itself an authority change and is not currently authorized.
4. **Path B/B-prime:** remains unproven and is inferior to D because it retains cross-vendor model inputs.

The most important thing Path D gets wrong is treating “IBKR for execution only” as a thin adapter. **Execution is a second causal policy.** It decides whether, when, and at what price the model’s intent becomes a position or ceases to be one. Until that policy is specified, replayed, and prospectively measured, Path D has solved model parity but not the trading game.

No files, authority, data, runtime, broker, or recorder state were changed.