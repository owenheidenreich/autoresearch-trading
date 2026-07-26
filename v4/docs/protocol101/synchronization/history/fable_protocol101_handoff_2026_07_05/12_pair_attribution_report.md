# Protocol101 Fair-Contract Pair Attribution

## Decision

- Status: `fail_not_paper_ready`
- Candidate: `attempt107`
- Broker endpoint called: `false`
- Paper-submit allowed: `false`
- Model training executed here: `false`
- Threshold tuning executed here: `false`

## Main Finding

Attempt107 still fails paired cross-vendor paper-readiness: same-input replay is deterministic, but candidate scores and rankings are sensitive to IBKR-vs-historical feature deltas at threshold-crossing moments.

## Cases

### july1_score_ceiling_action_drift

- Session: `2026-07-01`
- Decision timestamp: `2026-07-01T14:36:00+00:00`
- Live action: `enter` selected `SPXW-20260701-07510.000-P` score `25.337100982666016`
- Historical action: `wait` selected `None` score `55.5837516784668`
- Reason inspected: Historical score crosses max_score_ceiling=50 while IBKR score remains below the ceiling.

| Contract | Live Score | Historical Score | Delta | Top Group Swaps |
|---|---:|---:|---:|---|
| `SPXW-20260701-07510.000-P` | 25.3371 | 55.5838 | +30.2467 | option +30.25, environment +0.00, market_delta +0.00, market_last +0.00 |

Top live scores:

- `SPXW-20260701-07515.000-P` score `52.855892181396484` offset `25.0`
- `SPXW-20260701-07520.000-P` score `51.15100860595703` offset `30.0`
- `SPXW-20260701-07490.000-P` score `32.69255828857422` offset `0.0`
- `SPXW-20260701-07495.000-P` score `25.723106384277344` offset `5.0`
- `SPXW-20260701-07510.000-P` score `25.337100982666016` offset `20.0`

Top historical scores:

- `SPXW-20260701-07510.000-P` score `55.5837516784668` offset `20.0`
- `SPXW-20260701-07520.000-P` score `50.80788040161133` offset `30.0`
- `SPXW-20260701-07515.000-P` score `50.087154388427734` offset `25.0`
- `SPXW-20260701-07490.000-P` score `28.265777587890625` offset `0.0`
- `SPXW-20260701-07485.000-P` score `26.94477653503418` offset `-5.0`

### july2_selected_contract_ranking_drift

- Session: `2026-07-02`
- Decision timestamp: `2026-07-02T13:57:00+00:00`
- Live action: `enter` selected `SPXW-20260702-07545.000-P` score `45.347900390625`
- Historical action: `enter` selected `SPXW-20260702-07520.000-P` score `43.98440170288086`
- Reason inspected: Both feeds enter at the same minute, but cross-vendor feature drift flips the selected put.

| Contract | Live Score | Historical Score | Delta | Top Group Swaps |
|---|---:|---:|---:|---|
| `SPXW-20260702-07545.000-P` | 45.3479 | 43.4790 | -1.8689 | option -1.87, environment +0.00, market_delta +0.00, market_last +0.00 |
| `SPXW-20260702-07520.000-P` | 44.0285 | 43.9844 | -0.0441 | option -0.04, environment +0.00, market_delta +0.00, market_last +0.00 |

Top live scores:

- `SPXW-20260702-07555.000-C` score `90.02082061767578` offset `25.0`
- `SPXW-20260702-07550.000-C` score `88.61724853515625` offset `20.0`
- `SPXW-20260702-07540.000-C` score `81.42164611816406` offset `10.0`
- `SPXW-20260702-07545.000-C` score `75.11268615722656` offset `15.0`
- `SPXW-20260702-07560.000-C` score `74.78388214111328` offset `30.0`

Top historical scores:

- `SPXW-20260702-07540.000-C` score `103.42879486083984` offset `10.0`
- `SPXW-20260702-07575.000-C` score `74.78388214111328` offset `45.0`
- `SPXW-20260702-07570.000-C` score `73.93158721923828` offset `40.0`
- `SPXW-20260702-07550.000-C` score `73.16629028320312` offset `20.0`
- `SPXW-20260702-07535.000-C` score `71.35456848144531` offset `5.0`

## Interpretation

- If a single vendor-side feature group moves a score across the ceiling or flips the top rank, the candidate is data-plane-sensitive at that timestamp.
- This audit does not tune the threshold or edit the candidate. It only identifies whether the current failure is likely repairable feature semantics or candidate brittleness.

## Outputs

- `feature_diffs_csv`: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/feature_diffs.csv`
- `group_swaps_csv`: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/group_swaps.csv`
- `rankings_csv`: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/rankings.csv`
- `summary_json`: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/summary.json`
- `report_md`: `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/report.md`
