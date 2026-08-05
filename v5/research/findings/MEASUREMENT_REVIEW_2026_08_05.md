# Measurement Review — can 254 sessions support any useful conclusion?

**Dated research finding, 2026-08-05. This completes STATUS job 2.**

## Meaning for the bot

**Verdict B: the owned year can only measure a large directional edge — roughly 2 to 4 ES points
per session depending on horizon — and cannot resolve an edge that merely covers the 0.358-point
cost. Both limits in the review brief are real, and at first order they are slightly
*conservative*, not optimistic: measured day-to-day correlation is mildly negative, so independence
is not flattering the numbers.** Under the owner's accepted "large edge or stop" framing, the G1
direction screen is released: it hunts a large edge, and if none shows, the programme stops rather
than chasing something the instruments cannot measure. No design available today escapes either
limit without abandoning the frozen once-daily mechanism family or waiting years.

A term of art used throughout: the **minimum detectable effect (MDE)** is the smallest true average
edge an experiment would detect with 80% probability at one-sided 95% confidence. An edge smaller
than the MDE can exist and still produce a null result.

## 1. Are the limits correctly derived? Yes — every number reproduces from raw data

All numbers below were recomputed this session directly from the owned ES 1-minute bars
(`~/.autoresearch-trading/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m`, 261 files,
254 non-empty sessions) by [`analysis.py`](../../work/measurement-review/analysis.py); the raw
output is [`analysis_receipt.json`](../../work/measurement-review/analysis_receipt.json).

| Quantity | Brief / audit claim | Recomputed | Match |
|---|---:|---:|---|
| Non-empty sessions | 254 | 254 | exact |
| Session SD, 15/30/60 min at the 09:35 slot | 13.486 / 18.099 / 24.807 | 13.486 / 18.099 / 24.807 | exact |
| MDE at 1 trade/session | 2.104 points | 2.104 | exact |
| Detectable per trade at 6 / 26 trades/session | 0.859 / 0.413 | 0.859 / 0.413 | exact |
| Forward sessions for a 0.5-point edge, 15 min | 4,498 (17.85 y) | 4,500 (17.86 y) | within rounding (≤ 2 sessions) |
| Owned download range (contaminates the proposed 2024-08→2025-07 firewall) | 2025-01-02 → 2026-03-31 | 2025-01-02 → 2026-03-31, 311 sessions, $2.59 est. | exact |

The audit's slightly larger per-session MDEs (2.236 / 2.938 / 4.041) add the 4-of-5 fold-sign rule
on top of the plain lower bound; the direction of that adjustment is correct and it only makes
detection harder. I did not independently rerun its 8,000-resample campaign; the plain-bound
numbers above bracket it from below.

**The independence assumptions were attacked, as the brief asked, and they survive at first
order.** Across sessions, the lag-1 autocorrelation of the fixed-slot moves is **−0.040 / −0.106 /
−0.112** at 15/30/60 minutes — mildly negative, meaning treating sessions as independent slightly
*understates* the effective sample (effective N ≈ 275–318 rather than 254). Within a session, the
lag-1 autocorrelation of non-overlapping 15-minute returns is **−0.055**, so the 1/√k
trade-frequency scaling is also not undermined at first order. Caveat marked plainly: lag-1 is not
a full dependence test — volatility clustering and longer-range structure are not excluded — and
the fixed-slot variance remains a proxy until a specific policy's occupancy is frozen and its MDE
recomputed, which the G1 plan already requires.

## 2. Does any design escape Limit 1 (detection)? No design that exists; one arithmetic near-miss

Sessions needed to detect a **cost-scale** (0.358/trade) edge, from the measured 15-minute
variance, assuming independence:

| Trades per session | Sessions needed | Years at 252/yr |
|---:|---:|---:|
| 1 (the frozen M1/M3 design) | 8,778 | **34.8** |
| 6 | 1,463 | 5.8 |
| 26 (physical 15-minute maximum) | 338 | 1.3 |

- **Higher frequency** is the only arithmetic escape, and it is out of reach honestly: the frozen
  M1/M3 family decides once per day at 09:35; a 26-trade/session design is a scalper, which the
  owner has explicitly rejected as the product vision; no such mechanism exists or is proposed
  (the brief forbids proposing one); and even at the physical maximum the owned year detects
  0.413/trade against a 0.358 bar — a 1.15× margin resting entirely on the independence assumption.
- **Buying more history at the frozen design** needs ~35 years of sessions. ES 1-minute history is
  cheap in dollars (the owned 311 sessions cost ~$2.59; an exact quote is **UNKNOWN** and requires
  owner-authorized vendor contact), but a 1990s–2000s market is not evidence about this market, and
  historical detection would still not be forward confirmation (Limit 2).
- **A lower-variance instrument or structure**: session variance is the market's, not a design
  choice. The one real lever is horizon — 15-minute variance (13.486) is roughly half of 60-minute
  (24.807), so the 15-minute screen has the best MDE-to-cost ratio. Noted tension: only a
  **60-minute** G1 pass reopens the option wrapper (G2), and 60 minutes is the worst-measured
  horizon. A 15-minute-only pass would leave the option product closed under ledger row 181.
- **A better signal-to-noise target** (hit rate instead of points): detecting 55% directional
  accuracy needs only ~620 trades, which is why the G1 plan's binned-economics diagnostic is
  useful as a *screen*. It cannot replace the economic gate — accuracy on small moves with losses
  on large ones is the exact failure row 181 documents.
- **Pooling correlated instruments** (ES/NQ/RTY): with cross-correlation near 0.9, three
  instruments are worth ≈ 3/(1+0.9·2) ≈ 1.07 effective copies — a ~7% gain. Ruled out. (The exact
  correlation on owned data is **UNKNOWN** — no NQ/RTY data is owned — but no plausible value
  changes the conclusion.)
- **Accepting a larger minimum edge as the design target** is the one escape that works, and it is
  verdict B: the owned year *can* detect 2.1–3.9 points/session (plain bound; 2.2–4.0 with the
  fold rule). That is the test G1 runs.

## 3. Does any design escape Limit 2 (confirmation)? Partially — by starting the clock now

Verified: the holdout is spent (opened once 2026-08-02, recorded in STATUS §6 and the stand-down
record); the proposed 2024-08→2025-07 replacement is contaminated by the owned 2025-01→2026-03
download; the forward-confirmation table reproduces (see §1). The "one year to multiple decades"
estimate the brief flagged as unverified is **confirmed**: ~1.1 years for a 2-point 15-minute
edge, ~4.5 years for 1 point, ~18 years for 0.5 — before fold-sign and serial penalties.

- **Forward reservation, starting now, is the real escape** — it costs nothing and every
  undeclared day is confirmation calendar permanently lost. The owner decided 2026-08-05 to
  declare it; the draft declaration is
  [`FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md`](../../governance/FORWARD_CONFIRMATION_RESERVATION_2026_08_06.md).
- **A genuinely unused historical range** remains possible in principle but requires a provenance
  audit proving non-use plus an owner-authorized vendor quote (price **UNKNOWN**), and it tests
  regime robustness, not the current regime. Secondary at best.
- **Sequential (always-valid) testing** spends confirmation continuously instead of once and stops
  early when the true edge exceeds the declared one — worth adopting *inside* G8's
  pre-registration — but it does not change the (SD/edge)² scaling. No order-of-magnitude escape.
- **Paper as ongoing evidence** is the same mathematics under a different name.

## 4. The honest recommendation

Proceed to G1 under "large edge or stop," exactly as the owner accepted:

1. **G1 is released.** It tests the only claim the data can support: a large (≥ ~2.2–4.0
   points/session, horizon-dependent, recomputed after occupancy freeze) directional edge in the
   frozen M1/M3 family. A pass is meaningful and a fail is a clean stop — the sample is adequate
   for *that* question, and the mild negative autocorrelation means the test is not flattered by
   its independence assumptions.
2. **Do not spend effort making a cost-scale edge measurable.** Every route there (35 years of
   history, a 26-trade scalper, pooling) is either unaffordable in calendar, contradicts the
   product vision, or yields ~7% — and none produces forward confirmation anyway.
3. **The reservation declaration and the G8 sequential option are the only Limit-2 mitigations
   worth carrying.** Both are drafted; both wait on the owner's signature where marked.

## What this does not say

It does not say a cost-scale edge is absent — it says this instrument cannot see one, which is a
statement about the measuring device. It does not authorize training, capture changes, data
purchase, vendor or broker contact, or holdout access. It closes no mechanism: rows 177–183 stand
untouched.

---

*Read-only provenance: every number above reproduces by running
`./.venv/bin/python v5/work/measurement-review/analysis.py` (reads owned parquet only, writes one
JSON receipt beside itself) and the receipt-range check on
`v4/audit/databento_es_vwap_downloads.jsonl`. No model was fitted, no capture touched, no vendor or
broker contacted, no reserved evidence opened.*
