# SPXW 0DTE Data Downloads And Prototype Build

This is the first paid-data pilot. It is intentionally small: prove that
quote-realistic SPXW 0DTE data improves on v3 before buying a full historical
backfill.

## Pilot Window

Use `2026-01-02` through `2026-03-31`.

## Exact Data To Download First

| Priority | Source | Dataset / Product | Symbols | Granularity | Required? | Store under |
|---|---|---|---|---|---|---|
| 1 | Databento | `OPRA.PILLAR`, schema `definition` | `SPXW.OPT` | point-in-time definitions | Required | `data/raw/databento/opra_spxw_definition/` |
| 2 | Databento | `OPRA.PILLAR`, schema `cbbo-1m` | filtered SPXW 0DTE raw symbols | 1 minute | Required | `data/raw/databento/opra_spxw_cbbo_1m/` |
| 3 | Databento | `OPRA.PILLAR`, schema `ohlcv-1m` | same raw symbols | 1 minute | Strongly recommended | `data/raw/databento/opra_spxw_ohlcv_1m/` |
| 4 | Databento | `OPRA.PILLAR`, schema `statistics` | same raw symbols | daily/stat messages | Recommended | `data/raw/databento/opra_spxw_statistics/` |
| 5 | Cboe DataShop, Cboe Global Indices Feed, or ThetaData | SPX index values | `SPX` | 1 minute or better | Required | `data/raw/index/spx_1m/` |
| 6 | Cboe DataShop, Cboe Global Indices Feed, or ThetaData | VIX index values | `VIX` | 1 minute or better | Required | `data/raw/index/vix_1m/` |
| 7 | Databento audit slice | `OPRA.PILLAR`, schema `tcbbo` or `cbbo-1s` | ATM +/- `$50`, 10 selected days | tick/trade-sampled or 1 second | Audit only | `data/raw/audit/opra_spxw_tcbbo_or_cbbo_1s/` |

If licensed SPX/VIX index bars are not available yet, use the option-derived
context path first:

```bash
python -m v4.scripts.build_derived_context --session 2026-01-02
```

This derives SPX context from SPXW put-call parity and a volatility-regime
context from same-chain near-ATM 0DTE IV. It is causal and uses only data we
already bought. True SPX/VIX index bars are still useful promotion-grade
evidence, but they are not a blocker for the low-cost three-month pilot
download.

The code can also create explicit Databento futures proxies with
`v4/scripts/download_databento_context_proxies.py`: ES futures as SPX context
and VX futures as VIX context. These files carry `is_proxy = True` and should
only be used for pipeline smoke testing.

Promotion-grade research should rebuild the neural rows with official or
live-equivalent SPX/VIX bars:

```bash
python -m v4.scripts.build_databento_neural_dataset \
  --start-date 2025-01-02 \
  --end-date 2025-03-31 \
  --context-mode official \
  --official-spx-dir data/raw/vendor_official/spx_1m \
  --official-vix-dir data/raw/vendor_official/vix_1m \
  --processed-dir data/processed/spxw_0dte_neural_q1_2025_official_context \
  --normalized-dir v4/normalized_official_context \
  --summary-out v4/audit/q1_2025_official_context_neural_build_summary.json
```

The official context files may be CSV, JSONL, or Parquet. They must include a
timestamp column and a close/value column; OHLCV columns are used when present.
The builder writes canonical stamped copies to `data/raw/index/spx_1m/` and
`data/raw/index/vix_1m/` with `is_official_index_data = True`.

## Databento Download Sequence

Use `v4.ingest.databento_opra` as the implementation surface. It accepts a
Databento historical client object but does not import the paid SDK, so tests
remain runnable without credentials.

```python
from databento import Historical
from v4.ingest.databento_opra import (
    fetch_0dte_definitions,
    fetch_cbbo_1m,
    fetch_option_ohlcv_1m,
    fetch_open_interest,
)

client = Historical("YOUR_DATABENTO_KEY")
defs = fetch_0dte_definitions(client, "2026-01-02")
symbols = defs["raw_symbol"].tolist()
cbbo = fetch_cbbo_1m(client, "2026-01-02", symbols)
ohlcv = fetch_option_ohlcv_1m(client, "2026-01-02", symbols)
oi = fetch_open_interest(client, "2026-01-02", symbols)
```

Definitions are filtered to:

```text
root == SPXW
expiration == trading date
strike % 5 == 0
PM-settled weekly/daily contracts only
```

Do not include `SPX` root rows. When standard monthly AM contracts are listed
on the same calendar date, they are outside this prototype universe.

## Index Bars

Load SPX and VIX one-minute bars separately:

```python
from v4.ingest.index_bars import load_spx_1m, load_vix_1m

spx = load_spx_1m("data/raw/index/spx_1m/2026-01-02.csv")
vix = load_vix_1m("data/raw/index/vix_1m/2026-01-02.csv")
```

Accepted input can be CSV, JSONL, or Parquet with either full OHLCV fields or a
timestamp plus index value. The normalized columns are:

```text
event_time, symbol, open, high, low, close, volume
```

## Normalize One Day

```python
from v4.ingest.databento_opra import normalize_spxw_0dte_day

normalized = normalize_spxw_0dte_day(
    defs,
    cbbo,
    ohlcv_1m=ohlcv,
    statistics=oi,
    index_bars=spx,
)
```

Important rules enforced by the normalized path:

- `cbbo-1m` is quote data, not full trade-flow data.
- `cbbo-1m.price` and `cbbo-1m.size` may be retained as last-sale snapshot
  fields, but `size` is never promoted to minute volume.
- `option_ohlcv_volume` and `volume` are populated only from `ohlcv-1m`.
- open interest comes only from `statistics` rows where `stat_type == 9`.
- Databento does not provide IV/Greeks for this workflow. Compute them from
  SPX, strike, time-to-close, bid/ask mid, and documented rate assumptions.

## Build Neural Decision Rows

```python
from v4.dataset.spxw_0dte_neural import build_neural_dataset

rows = build_neural_dataset(normalized, spx, vix)
```

Each row represents one decision minute and contains:

```text
market_window       # SPX, VIX, VWAP-like anchor, OMAR, range, momentum
option_ladder       # ATM +/- $50, $5 strikes, calls and puts
position_state      # flat for the first entry-quality prototype
candidate_mask      # executable one-contract long call/put candidates
labels_net_pnl      # entry ask, exit bid, no broker commission
labels_mid_pnl      # audit comparison only; never the training execution label
```

Labels must use executable prices:

```text
entry = ask
exit = bid
broker commission excluded from primary labels
mandatory flat before close
no mid fills for training labels
```

## Acceptance Checks Before Training

- every tradable row is `SPXW`
- every strike is `$5` aligned
- no AM-settled `SPX` contracts enter the dataset
- `CBBO-1m` last-sale size is not used as volume
- OI only comes from `statistics.stat_type == 9`
- feature timestamps are `<= decision_time`
- labels change materially when using bid/ask instead of mid
- the 10-day high-resolution audit confirms `cbbo-1m` is acceptable for this
  prototype's stop/target labels

## Economic Stop Rule

Do not buy the full 2022-present backfill until this three-month pilot shows a
measurable improvement over v3 under bid/ask execution. If `ohlcv-1m` or the
high-resolution audit slice is too expensive, continue with definitions +
`cbbo-1m`, but remove volume-sensitive features from the prototype.

For the first two downloaded days, OPRA definitions + `cbbo-1m` + `ohlcv-1m` +
statistics averaged about `$0.43/day`. At 63 weekdays for
`2026-01-02` through `2026-03-31`, the working estimate is about `$27` before
the high-resolution audit slice. A 10-day `cbbo-1s` ATM +/- `$50` audit is
estimated near `$1.40` from the first two sample days. Prefer `cbbo-1s` over
`tcbbo` for the audit unless trade-sampled records are explicitly required.
