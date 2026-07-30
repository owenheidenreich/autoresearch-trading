# Protocol101 FT2-05 Opportunity Census

## Owner Memo

**Outcome: `feasible`.** executable post-fee opportunities exist; FT2-05 cannot green-light training

This is a label-side map of the governed game, not evidence that a profitable
model exists. Across `46` development-only sessions, the fixed $10,000
D48 reference mask left `15,350` distinct executable minutes
with at least one contract, or about `333.7` minutes per
day. A deliberately loose 25%-tail quality screen plus positive conservative
upside left about `241.0` qualifying minutes per day. This is a
curve diagnostic, not a selected guardrail.

The hindsight `best_session` selector made `$377,161.00` under one-account
simulator-v5 semantics, D48, D49, fees, and the daily stop. P5 using the same
hindsight exit rule made `$117,454.00`. These are ceilings: they prove paths
exist, not that present-time features can identify them. The median entry
friction across premium bands was `5.3%` of premium.

The 46-session cluster calculation estimates that a best-session improvement
would need to be roughly `$91,691.39` in total PnL to achieve conventional
95%/80% detection under the observed session variance. FT2-11 must refine this
before paid training.

Where D48-reference opportunities live:

- Market phase: `[{"count": 51320, "market_phase": "lunch", "share": 0.35876571172909416}, {"count": 42128, "market_phase": "afternoon", "share": 0.2945066621925814}, {"count": 35700, "market_phase": "primary_morning", "share": 0.24957006836961537}, {"count": 13898, "market_phase": "europe_close_transition", "share": 0.09715755770870908}]`.
- Moneyness: `[{"count": 86106, "moneyness_band": "wing", "share": 0.5385832681782643}, {"count": 50389, "moneyness_band": "near", "share": 0.31517748240813137}, {"count": 23380, "moneyness_band": "atm", "share": 0.1462392494136044}]`.
- Premium band: `[{"count": 78565, "premium_band": "small_1_3", "share": 0.4914151681000782}, {"count": 48612, "premium_band": "medium_3_8", "share": 0.3040625488663018}, {"count": 32698, "premium_band": "cheap_le_1", "share": 0.20452228303362002}]`.

**Regime warning:** this census sees only January through early March 2025.
April 2025 volatility is in fold-1 outer test, and May-June 2025 is protected
holdout. No conclusion here may be generalized to those regimes. Every headline
table is therefore also stratified by month and the governed SPX
regular-session realized-range proxy.

The only allowed interpretation of `feasible` is: the census characterizes the
opportunity surface of the governed game on the 46-session window, and FT2-08
design work may start. It does not authorize design acceptance, training,
selection, holdout access, shadow operation, or paper trading.

## Frozen Scope

- Node: `FT2-05-OPPORTUNITY-CENSUS`.
- Product-contract SHA-256: `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`.
- Sessions: exactly 46 from the repaired FT2-04 manifest.
- Horizons: `(3, 5, 10, 20, 45, 90, 'remaining_session')`.
- Entry/exit accounting: executable ask to executable bid, $3 round trip.
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

- Label rows: `472,048`.
- D48-reference rows: `159,875`.
- Wall time: `0.244` hours.
- CPU time: `0.244` core-hours.
- Resume checkpoints: one Parquet file and summary per session.

## Route

Terminal outcome is `feasible`. The node stops here. FT2-08 was not started.
