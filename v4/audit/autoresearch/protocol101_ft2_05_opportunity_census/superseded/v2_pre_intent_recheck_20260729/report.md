# Protocol101 FT2-05 Opportunity Census

## Owner Memo

**Outcome: `feasible`.** executable post-fee opportunities exist; FT2-05 cannot green-light training

This is a label-side map of the governed game, not evidence that a profitable
model exists. Across `45` development-only sessions, the fixed $10,000
D48 reference mask left `15,183` distinct executable minutes
with at least one contract, or about `337.4` minutes per
day. A deliberately loose 25%-tail quality screen plus positive conservative
upside left about `240.4` qualifying minutes per day. This is a
curve diagnostic, not a selected guardrail.

The hindsight `best_session` selector made `$341,796.00` under one-account
simulator-v5 semantics, D48, D49, fees, and the daily stop. P5 using the same
hindsight exit rule made `$156,830.00`. These are ceilings: they prove paths
exist, not that present-time features can identify them. The median entry
friction across premium bands was `5.3%` of premium.

The primary remaining-session paths contained `2,690,689` no-bid
minutes out of `31,579,138` marks (`8.52%`). Those minutes are
valued at full loss rather than erased. Across the preregistered MNAR
no-bid-excluded sensitivity, the minimum path-metric rank correlation was
`0.9793`. This is label-surface stability, not candidate-policy
stability; FT2-10/11 must still verify the latter.

Compared with preserved FT2-05 v1, v2 removes
`11,111`
candidate rows and one embargo session. The transparent best-session oracle
changed by `$-35,365.00` and
`-5` trades. This impact combines the
next-minute fill law, no-bid law, exact-boundary flat, embargo exclusion, and
D49 soft-close; it must not be attributed to any one change in isolation.

The 45-session cluster calculation estimates that a best-session improvement
would need to be roughly `$62,495.47` in total PnL to achieve conventional
95%/80% detection under the observed session variance. FT2-11 must refine this
before paid training.

Where D48-reference opportunities live:

- Market phase: `[{"count": 51425, "market_phase": "lunch", "share": 0.35940175420204773}, {"count": 41762, "market_phase": "afternoon", "share": 0.29186846979068387}, {"count": 35939, "market_phase": "primary_morning", "share": 0.25117238005381415}, {"count": 13959, "market_phase": "europe_close_transition", "share": 0.09755739595345424}]`.
- Moneyness: `[{"count": 85109, "moneyness_band": "wing", "share": 0.5342284322587124}, {"count": 50541, "moneyness_band": "near", "share": 0.3172454052425429}, {"count": 23662, "moneyness_band": "atm", "share": 0.1485261624987446}]`.
- Premium band: `[{"count": 78096, "premium_band": "small_1_3", "share": 0.490207893943959}, {"count": 47804, "premium_band": "medium_3_8", "share": 0.30006528070704025}, {"count": 33412, "premium_band": "cheap_le_1", "share": 0.2097268253490007}]`.

**Regime warning:** this census sees only January through early March 2025.
April 2025 volatility is in fold-1 outer test, and May-June 2025 is protected
holdout. No conclusion here may be generalized to those regimes. Every headline
table is therefore also stratified by month and the governed SPX
regular-session realized-range proxy.

The only allowed interpretation of `feasible` is: the census characterizes the
opportunity surface of the governed game on the 45-session window, and FT2-08
design work may start. It does not authorize design acceptance, training,
selection, holdout access, shadow operation, or paper trading.

## Frozen Scope

- Node: `FT2-05-OPPORTUNITY-CENSUS`.
- Product-contract SHA-256: `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`.
- Sessions: exactly 45 from the repaired FT2-04 manifest.
- Horizons: `(3, 5, 10, 20, 45, 90, 'remaining_session')`.
- Entry/exit accounting: BUY decision at `t` fills at the `t+1` executable ask;
  EXIT decision at `v` fills at the `v+1` executable bid; $3 round trip.
- No-bid accounting: full-loss path state; exact 15:55 forced-flat boundary.
- D49 soft-close: sub-$1 contracts cannot keep the lane open once remaining
  daily budget cannot fund a $1 contract plus fees.
- Hard safety: D48 premium cap, D49 remaining loss budget, one open position,
  15:30 entry cutoff, 15:55 forced flat.
- Volatility proxy: session SPX realized range in basis points.
- No model was fitted or scored.

## Threshold-Independent Surface

The packet contains the complete family distributions, Pareto frontier,
guardrail curves, excluded-winner curves, and oracle-assisted trade-rate curves.
Guardrail levels are percentile screens across `0%, 10%, 20%, 25%, 30%, 40%,
50%`; none is selected here.

## Oracle And P5 Ceilings

All nine transparent variants are reported: seven best-achievable-bid horizons,
hold-to-forced-flat, and first-real-profit. P5 controls direction and nearest
eligible-under-cap strike only; each comparison uses the same exit oracle to
isolate its entry choice.

## Friction, Power, And Regimes

Premium-band friction includes spread crossing plus the $3 fee. MDE tables use
session clusters and separately show cluster-adjusted trade-count projections.
Regime headline tables split every replay by month and SPX range tercile.

## Compute

- Label rows: `460,937`.
- D48-reference rows: `159,312`.
- Wall time: `0.261` hours.
- CPU time: `0.260` core-hours.
- Resume checkpoints: one Parquet file and summary per session.

## Route

Terminal outcome is `feasible`. The node stops here. FT2-08 was not started.
