# Codex Goal — Measure the ES Spread (owner-authorized paid download, 2026-08-03)

**Why you and not Claude:** Claude has no Databento credential in its environment (`.env` holds only
Anthropic and Polygon keys; no `~/.databento/` config, no keychain entry). Your adversarial review ran a
successful `metadata.get_cost` probe, so you have working credentials. This task is outsourced to you for
that reason, and because it is the long-running half of the work.

## Why this matters

The ES branch of the Path-D feasibility comparison rests on a single **assumed** constant: $17.00 round-trip
friction, derived from "ES is ~always 1 tick wide" plus ~$4.50 commissions. You already quantified how
load-bearing that is:

| ES spread | Friction | 15m | 30m | 60m |
|---|---|---|---|---|
| 1 tick | $17.00 | 54.25% | 52.96% | 52.06% |
| **2 ticks** | **$29.50** | **57.38%** | **55.13%** | **53.58%** |
| 4 ticks | $54.50 | 63.63% | 59.48% | 56.61% |

**At 2 ticks, ES's advantage over passive 0DTE options disappears entirely.** The entire "ES is a credible
branch" position depends on a number nobody has measured. This task measures it.

## Authorization — read before spending anything

The owner authorized this in conversation on 2026-08-03. The governance record is
`v4/audit/autoresearch/protocol101_pathd_data_acquisition/paid_data_approval_manifest_es_bbo1s_2026_08_03.json`.

**Note carefully:** the *pre-existing* manifest (`paid_data_approval_manifest.json`) authorizes ES/VX
**ohlcv-1m only** and lists forbidden as *"any other paid item"* — **`bbo-1s` is outside it.** That is why a
new, narrower manifest exists. Do not edit the original.

Claude drafted the new manifest but deliberately did **not** supply the approval text —
`V4_PAID_DATA_APPROVAL_TEXT` must be set by the owner or by you under owner direction, because the guard's
only protection is a human providing it at execution time.

**Scope, and nothing beyond it:** `GLBX.MDP3`, `ES.FUT` (`stype_in=parent`), schema **`bbo-1s`**, the
**20 named RTH sessions** (13:30–20:00 UTC), cap **USD 3.00**, estimated **USD 1.46**.

## Steps

1. **Cost check first.** Run `metadata.get_cost` for the exact request. **Abort if it exceeds USD 3.00.**
   Record the returned figure.
2. **Call `require_paid_data_approval`** (`v4/checks/paid_data_guard.py`) with the new manifest immediately
   before the download endpoint, exactly as the guard's docstring specifies.
3. **Download** only the authorized scope. Write a receipt next to the data: actual cost, bytes received,
   session count, request parameters, timestamp.
4. **Measure the spread.** For each session, over RTH:
   - distribution of `ask_px - bid_px` in ticks (0.25 pt = $12.50) — report the **share at 1 tick, 2 ticks,
     3+**, plus median and mean;
   - **time-weighted** spread, not sample-weighted — a spread that is 1 tick for 99% of the session but
     blows out during the moves you would actually trade is not a 1-tick market;
   - spread conditional on **elevated 15-minute realized volatility**, since that is when a directional
     rule would fire. This is the number that matters, and it may differ sharply from the unconditional one.
5. **Recompute the ES hurdle with MEASURED friction**, replacing the assumed $17.00. Use the
   **mean-payoff** screen (not the median — the median-in-an-EV-formula error is what produced the
   withdrawn 116.2% figure). Apply your roll-day exclusion from A2.
6. **Report whether ES still beats passive 0DTE** (which sits at ~53.9–56.6% depending on estimator).

## Hard stops

- Do **not** exceed the authorized scope: no `mbp-1`, no `tbbo`, no extra sessions, no widened window,
  no other symbol. Acquiring more sessions requires a **new** manifest.
- Do **not** open the protected 36-session firewall — `holdout_open_count` stays 0.
- Do **not** modify `FILL_LAW`, the causal t−60s clock, the label law, or the OOF firewall.
- No broker/order/live/paper-submit, no promotion, no runtime-flag/launchd/plist edits.
- **This does not authorize the feasibility gate itself.** That is separately pre-registered and frozen in
  `PATHD_MATCHED_FEASIBILITY_PREREGISTRATION_2026_08_03.md`; it runs as its own task and must not be folded
  into this one.

## Deliverable

A short results document: measured tick-width distribution (unconditional and volatility-conditional,
time-weighted), the resulting friction constant, the recomputed ES hurdle at 15/30/60m, and a one-line
verdict — **does ES remain a credible branch, or does the measurement close it?** Update the feasibility
study's ES section and the roadmap status board. Include the cost receipt. End with
`STOP_FOR_CLAUDE_VERIFICATION`.

**Claude will verify:** actual spend against the cap and the manifest, that the guard was genuinely invoked
rather than bypassed, that the scope was not exceeded, that the spread is time-weighted rather than
sample-weighted, and that the recomputed hurdle uses the mean-payoff screen with roll days excluded.

*Prepared by Claude Opus 5 — 2026-08-03.*
