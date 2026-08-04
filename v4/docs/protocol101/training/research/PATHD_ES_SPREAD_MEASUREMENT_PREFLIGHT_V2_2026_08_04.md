# ES BBO-1s Spread Measurement — Corrected Preflight (2026-08-04)

**Status: READY TO SPEND — blocked only on the owner-supplied approval string.**

Supersedes [`PATHD_ES_SPREAD_MEASUREMENT_PREFLIGHT_2026_08_03.md`](PATHD_ES_SPREAD_MEASUREMENT_PREFLIGHT_2026_08_03.md),
which paused because the manifest's fixed UTC window did not match its stated RTH purpose. That manifest is
left unchanged and unspent.

**Authorization:** owner, 2026-08-04 — *"proceed to option A"*, reaffirmed after I recommended against it.
Both the recommendation and the override are recorded in
[`PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md`](../contracts/PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md).

---

## What was wrong, and what changed

The 2026-08-03 manifest declared **one** fixed window, `13:30–20:00 UTC`. That is correct RTH on the 14 EDT
sessions, but on **2025-11-19, 2025-12-08, 2025-12-23, 2026-01-13, 2026-01-30 and 2026-02-18** it resolves
to **08:30–15:00 EST** — adding a premarket hour and **omitting the closing RTH hour**, which is the period
most likely to matter for an elevated-volatility conditional spread.

The superseding manifest
(`v4/audit/autoresearch/protocol101_pathd_data_acquisition/paid_data_approval_manifest_es_bbo1s_2026_08_04.json`)
declares **explicit DST-aware per-session windows**. All six winter sessions now resolve to `14:30Z`, and
the runner's validator confirms every one of the 20 windows is exactly 09:30–16:00 America/New_York and
exactly 6.5 hours.

## Read-only cost verification — nothing spent

`metadata.get_cost` only. **No `timeseries.get_range` call was made and no bytes were downloaded.**

| | |
|---|---|
| Dataset / symbol / schema | `GLBX.MDP3` / `ES.FUT` (`stype_in=parent`) / `bbo-1s` |
| Sessions | 20 |
| **Exact cost** | **$1.478328** |
| Cap | $3.00 — **PASS** |
| Prior corrected estimate (2026-08-03) | $1.478327 — reproduces to the sixth decimal |
| `GLBX.MDP3` listed for this key | yes |

Per-session costs range $0.056876 (2025-12-23, a half day) to $0.096264 (2026-03-09).

## What this buys

The Path-D feasibility hurdle for ES currently rests on an **assumed** 1-tick spread. That assumption is
load-bearing: **at a 2-tick spread ES's advantage over passive 0DTE options disappears entirely.** The
runner measures a duration-weighted spread both unconditionally and conditional on elevated realized
volatility, and converts it to a friction number and a mean-payoff hurdle.

**It measures the denominator only.** As recorded in the decision memo, the numerator — the size of the
omar effect — is currently `NOT_IDENTIFIED`, so this measurement cannot on its own establish that any ES
strategy is viable. It is worth having regardless: it converts a standing assumption into a durable
measurement for $1.48, and that number stays valid whatever happens to the signal question.

## The remaining step is the owner's

The paid-data guard requires the **exact** approval string via `V4_PAID_DATA_APPROVAL_TEXT`, supplied by a
human at execution time. The manifest states plainly that Claude drafted it and **must not** also supply
that text — the control's whole value is that a person provides it. I have therefore stopped here.

```bash
export V4_PAID_DATA_APPROVAL_TEXT="Owner authorizes GLBX.MDP3 ES.FUT bbo-1s for the 20 named sessions at DST-aware 09:30-16:00 America/New_York windows up to USD 3.00 to measure ES spread; no other paid data."

PYTHONPATH=. python -m v4.scripts.measure_pathd_es_bbo1s_spread \
  --manifest v4/audit/autoresearch/protocol101_pathd_data_acquisition/paid_data_approval_manifest_es_bbo1s_2026_08_04.json \
  --output-dir /Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_bbo_1s_measurement_2026_08_04
```

The runner re-checks the cost against the cap, re-invokes the guard immediately before **every** one of the
20 range calls, refuses a pre-existing output directory, and writes an acquisition receipt with per-file
sha256, byte counts and row counts.

*Signed: Claude Opus 5 — 2026-08-04.*
