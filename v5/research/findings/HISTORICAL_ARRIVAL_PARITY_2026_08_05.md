# Historical arrival parity — the training corpus contains no latency

**Meaning for the bot:** the owned training corpus says every option quote is available the instant its
minute ends. Live, the same quotes arrive about a fifth of a second later, sometimes half a second. A
model fitted directly on these rows would be trained under a timing assumption the live system cannot
satisfy, and its historical results would not be reproducible live.

This is a dated measurement finding, not a status page. Current state and authorization come only from
[`v5/STATUS.md`](../../STATUS.md).

## 1. What was measured

Every base session file in the Path-D normalized corpus was read and compared field by field.

| Quantity | Result |
|---|---|
| Session files read | **251** (all of them; `_official_context` sidecars excluded) |
| Rows read | **47,707,186** — matches the [data-quality report](../../../v4/audit/autoresearch/protocol101_pathd_data_acquisition/data_quality_report.json) |
| `max abs(receive_time - event_time)` across every row | **0.0 seconds** |
| `max abs(quote_age_ms)` across every row | **0 ms** |
| Sessions containing any non-zero arrival lag | **0** |

The column `timestamp_source` reads `databento_cbbo_1m_ts_recv` on every row, so the stored arrival stamp
is nominally a receive timestamp. Its value is nevertheless exactly the interval boundary. The Historical
API stamps a completed CBBO-1m bar at the close of its interval; the corpus preserved that stamp and no
real arrival time was ever recorded.

## 2. What live arrival actually looks like

From the only paired live capture,
[`attempt002/capture_summary.json`](../../../v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/attempt002/capture_summary.json).
`local_receipt - ts_recv` is the full leg from exchange receipt to this machine.

| Stream | p50 | p99 | max |
|---|---:|---:|---:|
| CBBO-1m (`rtype=193`) — the option feature stream | **226.932 ms** | **319.521 ms** | 319.543 ms |
| CBBO-1s (`rtype=192`) — the executable entry stream | 93.985 ms | 319.517 ms | **440.879 ms** |
| Gateway leg only (`ts_out - ts_recv`), CBBO-1m | 38.706 ms | 54.493 ms | 54.518 ms |

An earlier unsplit capture,
[`attempt001`](../../../v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/attempt001/capture_summary.json),
recorded a p99 of 462.410 ms and a max of 463.304 ms across both streams combined.

**The gap is the whole finding: historical says 0 ms, live measured 227 ms at the median.**

## 3. Why value identity did not catch this

The same-session comparison proved all 914 live CBBO-1m rows matched their Historical API counterparts
after decoding. That test compared **values**. It did not compare **arrival times**, because the
historical side has no arrival time to compare. Value identity and timing identity are separate claims,
and only the first is evidenced.

## 4. What must happen before any fit

1. Arrival must be injected into historical rows rather than assumed. `simulate_historical_arrival` in
   [`training_twin.py`](../training_twin.py) stamps `received_at_ns = interval_end_ns + L` and
   `assert_no_zero_lag` refuses a frame that still claims instantaneous arrival.
2. `L` must come from a signed latency receipt for the **same source family**. The only frozen constant
   available is 2,336 ms derived from five ThetaData samples, which is the wrong stream and too thin. It
   is recorded as `UNCERTIFIED` in [`knobs.py`](../knobs.py) and cannot authorize a fit.
3. A Track-A OPRA capture must re-derive `L` across multiple sessions. Two sessions support an observed
   envelope, not a population worst case.

Until step 2 is satisfied the machinery correctly refuses to produce a training matrix. That refusal is
the intended behaviour, not a bug.

## 5. What this does not say

- It does not say the corpus values are wrong. They matched the Historical API exactly.
- It does not say prior negative results were caused by this. Those campaigns lost money by margins far
  larger than a quarter-second of staleness would explain.
- It does not establish a worst-case latency. Percentiles here come from one capture session.

## 6. Reproduction

```bash
./.venv/bin/python - <<'PY'
import glob, pandas as pd
fs = sorted(f for f in glob.glob(
    '/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized/*.parquet')
    if 'official_context' not in f)
worst = 0.0
for f in fs:
    d = pd.read_parquet(f, columns=['event_time', 'receive_time'])
    worst = max(worst, float((d['receive_time'] - d['event_time']).dt.total_seconds().abs().max()))
print(len(fs), 'sessions, max arrival lag', worst, 'seconds')
PY
```

Read-only on owned local data. No model was fitted, no capture started, no vendor or broker contacted.
