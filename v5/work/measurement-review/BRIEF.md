# Fable Review — is this project measurable at all?

**One question. Do not expand the scope.** This is a viability review of a constraint, not a research
campaign, not a strategy proposal, and not a request for new mechanisms.

---

## The question

> **Given the two limits below, is there any design under which this project can reach a confirmed,
> tradable edge? If yes, name it. If no, say so plainly.**

Answer in that order: (1) are the limits correctly derived, (2) does any design escape them, (3) what is
the honest recommendation to the owner.

## Why you and not another wave

These limits are not about a strategy. They constrain **G1, G4, G5 and G8 simultaneously** and they
determine whether the project's stated goal is reachable in principle. Every prior review here has assessed
one hypothesis. This one assesses whether the measuring instrument can resolve what we are looking for.

## Limit 1 — detection

254 owned ES sessions. Measured round-trip friction **0.358 ES points**. Smallest edge detectable at 80%
power, one-sided 95%, as a function of trades per session:

| Trades/session | Detectable points/trade | vs cost bar |
|---:|---:|---:|
| 1 | 2.104 | 5.9× |
| 6 (frozen session cap) | 0.859 | 2.4× |
| 26 (theoretical 15-min max) | 0.413 | 1.2× |

Inputs verified twice, independently: session standard deviations 13.486 / 18.099 / 24.807 points at
15/30/60 minutes reproduce exactly from
`~/.autoresearch-trading/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m`. The table assumes
trades within a session are **independent**, which they are not, so these are optimistic floors.

**Claim to test: even at the maximum physically available trade frequency, the owned year cannot resolve an
edge that merely covers costs.**

## Limit 2 — confirmation

- The protected 36-session holdout is **SPENT** (opened once, 2026-08-02, for `signed18`, which was then
  invalidated by a 60-second look-ahead).
- The proposed replacement range (2024-08 → 2025-07) is **already contaminated** — owned ES data runs
  2025-01-02 → 2026-03-31 per `v4/audit/databento_es_vwap_downloads.jsonl`. No clean range is currently
  identified.
- Forward paper confirmation is estimated at **roughly one year to multiple decades**, depending on effect
  size. (This estimate is Codex's and is **not** independently verified — check it.)

**Claim to test: there is currently no confirmation path that completes on a useful timescale.**

## What a good answer addresses

1. **Are the limits real?** Attack the arithmetic, the independence assumption, the choice of session as
   the resampling unit, and whether the fixed-slot variance is the right proxy for a signal-conditioned
   strategy's variance.
2. **Does any design escape Limit 1?** Candidates worth ruling in or out: higher trade frequency; a lower
   variance instrument or structure; a target with better signal-to-noise than signed price change;
   pooling correlated instruments; a shorter horizon with proportionally lower cost; accepting a larger
   minimum edge as the design target rather than a cost-scale one.
3. **Does any design escape Limit 2?** Candidates: buying a genuinely unused historical range and the exact
   cost; reserving forward data starting now while research continues on the old data; a sequential or
   always-valid test that spends confirmation budget continuously rather than once; accepting paper
   operation as ongoing evidence rather than a one-time gate.
4. **If neither escapes**, say whether the honest recommendation is to stop, to change instrument or
   product, or to continue with an explicitly reduced claim — and what that reduced claim would be.

## Constraints on your answer

- **The five prior negatives stand** and are not under review. Rows 177–183 of the do-not-retest ledger are
  closed; do not propose anything they close.
- **Do not propose new mechanisms.** M1 and M3 are frozen; this review is about measurability.
- Plain English per `CLAUDE.md` §2, definitions on first use, every claim tied to a file or a number.
- Where you cannot resolve something, write `UNKNOWN` rather than estimating.
- No training, capture, vendor or broker contact, holdout access, or runtime change.

## Context

- [`v5/STATUS.md`](../../STATUS.md) §3 and §7 — the current state and gate chain.
- [Gate-chain audit](../../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) — the full audit.
- [Programme restart record](../../governance/PROGRAM_RESTART_RECORD_2026_08_05.md)
  — what binds and what is superseded.
- Owner's goal, unchanged: an automated bot that buys SPX 0DTE calls and puts and makes money.

*Written by: Claude Opus 5 — 2026-08-05.*
