# Databento Raw Consistency Audit

Windows audited: Q3 2025, Q4 2025, Q1 2026, and combined July 2025 through March 2026.

| Window | Expected Sessions | Complete | Missing Files | Zero/Unreadable | First | Last |
|---|---:|---|---:|---:|---|---|
| q3_2025 | 64 | True | 0 | 0 | 2025-07-01 | 2025-09-30 |
| q4_2025 | 64 | True | 0 | 0 | 2025-10-01 | 2025-12-31 |
| q1_2026 | 61 | True | 0 | 0 | 2026-01-02 | 2026-03-31 |
| combined_collected | 189 | True | 0 | 0 | 2025-07-01 | 2026-03-31 |

## Schema Rows

| Window | Schema | Sessions | Rows | Min Rows | Max Rows |
|---|---|---:|---:|---:|---:|
| q3_2025 | definition | 64 | 1017830 | 15052 | 17002 |
| q3_2025 | cbbo-1m | 64 | 10069430 | 98055 | 338452 |
| q3_2025 | ohlcv-1m | 64 | 1607059 | 14631 | 38179 |
| q3_2025 | statistics | 64 | 174114 | 2228 | 4296 |
| q4_2025 | definition | 64 | 1103280 | 16172 | 18554 |
| q4_2025 | cbbo-1m | 64 | 11594235 | 86039 | 370332 |
| q4_2025 | ohlcv-1m | 64 | 1901889 | 10318 | 49059 |
| q4_2025 | statistics | 64 | 186590 | 2196 | 4926 |
| q1_2026 | definition | 61 | 1088932 | 17020 | 20030 |
| q1_2026 | cbbo-1m | 61 | 11140845 | 144681 | 433530 |
| q1_2026 | ohlcv-1m | 61 | 2026868 | 22899 | 48362 |
| q1_2026 | statistics | 61 | 181526 | 2278 | 5444 |

## Q3 2025 Download Cost

Total estimated Databento spend: `$26.7890`

- `cbbo-1m`: `$1.5005`
- `definition`: `$1.7063`
- `ohlcv-1m`: `$23.4681`
- `statistics`: `$0.1142`

## Holiday / Closed Sessions Excluded

- `2025-07-04` Independence Day
- `2025-09-01` Labor Day
- `2025-11-27` Thanksgiving
- `2025-12-25` Christmas
- `2026-01-01` New Year's Day
- `2026-01-19` Martin Luther King Jr. Day
- `2026-02-16` Presidents Day
