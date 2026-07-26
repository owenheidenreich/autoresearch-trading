# Protocol101 Vendor Overlap Inventory

- Status: `pass`
- Evidence grade: `data_plane_only`
- Labels used: `false`
- PnL used: `false`
- Strategy metrics used: `false`

## Product Coverage

- `databento/opra_spxw_definition` (option): files=`834`, sessions=`417`, formats=`{'dbn.zst': 417, 'parquet': 417}`
- `databento/opra_spxw_cbbo_1m` (option): files=`834`, sessions=`417`, formats=`{'dbn.zst': 417, 'parquet': 417}`
- `databento/opra_spxw_ohlcv_1m` (option): files=`834`, sessions=`417`, formats=`{'dbn.zst': 417, 'parquet': 417}`
- `databento/opra_spxw_statistics` (option): files=`834`, sessions=`417`, formats=`{'dbn.zst': 417, 'parquet': 417}`
- `thetadata/spx_1m` (index): files=`418`, sessions=`418`, formats=`{'parquet': 418}`
- `thetadata/vix_1m` (index): files=`428`, sessions=`428`, formats=`{'parquet': 428}`

## Conclusion

- `option_side_pairing_available_from_owned_databento`: `True`
- `option_side_pairing_available_from_owned_thetadata`: `False`
- `index_side_pairing_available_from_owned_thetadata`: `True`
- `requires_thetadata_option_quote_sample_for_td_option_diff`: `True`

## Interpretation

- This is an inventory only. It does not fit deltas, labels, PnL, or strategy performance.
- If ThetaData option quote files are absent, TD option-side diffing requires a separately approved small option-quote sample.
