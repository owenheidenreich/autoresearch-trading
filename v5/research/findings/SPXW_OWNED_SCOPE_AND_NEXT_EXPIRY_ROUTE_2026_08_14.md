# Historical scope audit — the proposed next-expiry route was rejected

**Owner correction:** the next-expiry debit-vertical recommendation in this finding is withdrawn. The
active model may trade only a single long SPXW 0DTE call or put. The inventory and attribution results
below remain valid historical evidence; the route and decision request do not.

**What changed for the bot:** the rejected MES route is gone, and the remaining owned-data SPXW
structure ideas have now been screened without opening another invalid economic cell. The only current
route that changes the game without changing the owner's instrument universe is next-listed-expiry
SPXW, preferably as a defined-risk debit vertical.

## The owned quote inventory is genuinely 0DTE-only

The complete source audit mapped every quoted instrument identifier to its daily definition on all
**251** CBBO-1m sessions from 2025-08-01 through 2026-07-31:

- sessions with a quoted future-expiration contract: **0/251**;
- future-expiration quoted instruments: **0**; and
- same-day-only quote sessions: **251/251**.

The definition files contain future expirations, but definitions are contract metadata, not prices. They
cannot support a longer-tenor simulation.

## Why the frozen selector was not translated into a debit vertical

The audit inspected no future price and no P&L. It asked only whether each of the compact policy's 35
frozen selected contracts had an exact live farther-OTM short leg at entry and remained within the $500
risk cap.

| Width | Live/risk-eligible pairs | Folds represented | Median max entry loss incl. fees |
|---:|---:|---:|---:|
| 5 | 15 | 2/5 | $131.16 |
| 10 | 14 | 2/5 | $201.16 |
| 15 | 11 | 2/5 | $251.16 |
| 20 | 8 | 2/5 | $281.16 |
| 25 | 0 | 0/5 | — |
| 30 | 0 | 0/5 | — |

Every audited width is incapable of the required four-of-five chronology before outcomes. Opening P&L
would therefore create another unsatisfiable test. It was correctly refused.

## The failed iron fly does not hide a profitable two-leg half

The already-spent 15:00 five-point iron-fly result was decomposed at the **same entries and same frozen
exits**. This is failure attribution, not a newly selected strategy. Component sums reproduce the primary
to less than `5e-13` dollars.

| Frozen component | Touch net / session | Positive folds | Midpoint gross / session | Positive folds |
|---|---:|---:|---:|---:|
| Call credit spread | -$31.02 | 0/5 | -$0.75 | 3/5 |
| Put credit spread | -$29.56 | 1/5 | +$1.94 | 2/5 |

Removing two legs does not reveal stable premium. The put midpoint is a $1.94 point estimate that fails
chronology and becomes -$29.56 at touch. Neither side can be promoted or retested from this post-run
decomposition.

## Highest-value in-scope acquisition

Acquire only the **nearest listed SPXW expiration strictly after each session** over the already-used
1,045-session 2022-06-01 through 2026-07-31 range:

- OPRA `definition` to resolve exact next-expiry symbols;
- OPRA `cbbo-1m` for only those symbols;
- no ES, futures, ETF, statistics, trade-print, broker, paper or live data; and
- policy context derived only from SPXW or SPX.

The intended game is one 10-point next-expiry debit vertical, held no more than 120 minutes, with entry
debit plus two measured option round-trip fees capped at $500. Before any outcome, the new corpus must
prove entry-pair coverage in at least four chronological folds and measure its effective sample size.
Only then may a roughly 29–50 parameter causal enter-versus-wait model be declared.

This is mechanically different from the failed 0DTE long option: the short leg caps premium at risk,
while the extra day removes the same-session terminal cliff from the 60–120-minute decision. It remains a
hypothesis, not a superiority claim.

## Cost and exact owner decision

No vendor was contacted. Using 212 already-recorded Databento CBBO cost estimates, the measured weighted
rate is `1.490116119e-7` dollars per row. The 251 owned definition files show a median **476** next-expiry
contracts versus **478** quoted 0DTE contracts. Extrapolated to 1,045 sessions:

- next-expiry CBBO-1m: **$29.50**;
- definitions: **$30.31**;
- combined estimate: **$59.81**; and
- recommended hard cap: **$100**.

The one remaining Tier-1 decision is authorization for a read-only exact cost preflight followed by the
scoped download only if the vendor estimate is at or below $100. A fresh cost quote could differ; the
downloader must stop before purchase above the cap.

## Evidence

- quote/vertical attainability receipt:
  `v5/work/entry-exit-attribution/SPXW_OWNED_QUOTE_SCOPE_AUDIT_V1.json`
- iron-fly component receipt:
  `v5/work/entry-exit-attribution/IRON_FLY_LEG_ATTRIBUTION_V1.json`
- acquisition estimate:
  `v5/work/entry-exit-attribution/SPXW_NEXT_EXPIRY_ACQUISITION_ESTIMATE_V1.json`
- focused tests:
  `v5/tests/test_spxw_owned_quote_scope.py`,
  `v5/tests/test_defined_risk_iron_fly_leg_attribution.py`, and
  `v5/tests/test_spxw_next_expiry_acquisition.py`
