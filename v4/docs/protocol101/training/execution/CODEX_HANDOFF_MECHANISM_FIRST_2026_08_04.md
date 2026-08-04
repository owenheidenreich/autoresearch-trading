# Codex Handoff — Mechanism-First, Entry+Exit Compatible (2026-08-04)

**Read this, then answer §3 BEFORE writing any code.** The plan in
[`PATHD_MECHANISM_FIRST_MASTER_PLAN_2026_08_04.md`](PATHD_MECHANISM_FIRST_MASTER_PLAN_2026_08_04.md)
is drafted but **not frozen**. Your first job is to attack it, not to execute it.

---

## 1. Where we are, in one page

**Five independent negatives closed the directional programme today.** 0DTE long premium is negative-EV
before any cost (−$13.00/trade gross, 5/5 folds). The 18-feature entry contract has zero ranking power.
The April-validated exit repair failed 6/6. The outcome shape is incompatible with the signed Charter
(23–69% big-loss vs a 2% cap). And `omar` — the last survivor of sixty screened members — closed on **raw
economics**: −0.324 points per trade, no null involved.

**What is proven and costs nothing to hold:** the IBKR paper execution plane (guarded round trip on
`DU***40`; contract identity `SPXW  260804C07775000` matches the training corpus OSI format exactly); the
research machinery (gate, loop, four-box, maxT, prior-art blocking); **measured ES friction of $17.9176 =
0.358 points per trade**, from 1,095,818 quote states at a 1.0397-tick spread; and a full year of owned
data (261 sessions GLBX ES OHLCV-1m, 251 sessions OPRA SPXW, 505 SPX 1m, 338 VIX 1m).

**What is spent:** the 36-session protected holdout, opened once on 2026-08-02 for `signed18` — which was
then invalidated by a 60-second look-ahead. **There is no confirmation firewall.** Do not let any plan
depend on one.

**The single lesson that cost the most:** the number that closed `omar` was computable from owned data with
no surrogate machinery, and it sat in a committed artifact for a day while three studies argued about an
information coefficient. **Compute the economics on raw data first.** Use nulls only to explain a number
that is already interesting.

**The second lesson:** this programme designed nulls one at a time, each repairing the last one's flaw
after seeing the data. That is the multiple-comparison error moved up one level — a search over nulls until
one produced a result. **Any null must pass a known-answer gate before its verdict is believed.**

## 2. What "a compatible model" means, precisely

The owner's requirement is a model that can **enter and exit profitably** — one system, not two artifacts
that happen to run in sequence. This project already failed at exactly this:

- The exit model is **degenerate**: `learned_exit_index == 0` in **980/1031 (95.1%)** trajectories,
  identical to `exit_immediate` in 982/1031. Sealed models reproduce with zero error — it is not a
  serialization bug, it is a mis-calibrated policy.
- It **amputates the convex tail**: learned-exit p99 **$86** and top-decile contribution **+$2,331**,
  versus hold-to-close **$3,422** and **+$212,841**. The entire reason to hold a long option was thrown
  away.
- Cause: it was fit against its own target, with no reference to what the entry was buying. Only **17.46%**
  of first-state labels were positive — nearly all defense, no upside capture.

**Non-negotiable properties of any entry+exit system you propose:**

1. **One objective that rewards both loss truncation and tail capture** (the Stage-2 contract requires
   both). An exit that improves mean PnL while collapsing p99 is a failure.
2. **A deterministic catastrophic floor is MANDATORY.** P073 and P087 both *rejected* removing it; 26.7% of
   exits depended on it. The learned policy chooses only among non-catastrophic states.
3. **Validated under one-account serial occupancy.** April's warning: composed curves were repeatedly
   single-window or overlap artifacts.
4. **The composed system clears the full acceptance gate again.** It does not inherit a component's tier.

## 3. Questions to answer BEFORE writing code

Answer these in a written response. Several are places I expect the plan is wrong.

> **⚠ ANSWERED AND CORRECTED 2026-08-04.** Both reviewers found my Q1 premises false and I verified
> it myself. **VIX has entered models** — `vix_level` / `vix_change_5m` / `vix_change_15m` are in
> `v4/research/lean_autoresearch/harness.py`; that S4/S5 campaign returned `NULL_NO_NEW_ENTRY_EDGE`. I
> checked only the canonical Stage-1 path and generalised to "any model". **The 10:01 boundary is a
> feature-warmup constraint**, not an arbitrary exclusion (`history_minutes=30`, `momentum_15m` needs 15
> prior bars). **The overnight-gap mechanism is untestable** — the ES corpus is RTH-only, 390 rows,
> 09:30→15:59, zero pre-09:30 bars. **Owned data was overstated** — 251 official SPX and 251 official VIX,
> not 505/338, of which 36 are the spent holdout. **And the instrument was never declared**, which is the
> root defect: the 0.358 bar is ES futures while §2's exit language is options. See §3/§3c of the master
> plan. Five of six mechanisms die; one survives.

**Q1 — Is the mechanism list right?** §3 of the master plan declares six. I verified two claims today:
every screen and campaign ran **10:00/10:01 → 15:00/15:20** (`pathd_phase1_entry.py:380`,
`minute_of_session` 31..350), so the opening 30 minutes and closing 40 minutes are untested; and **VIX has
never entered any model** (declared at index 1 of the market feature names, hardcoded `float("nan")` in
`official_spx_market_window_from_rows`, absent from `feature_matrix`'s `required_market`). **Verify both
independently.** Then: what mechanism with a real economic reason is missing, and is any of the six
mis-specified?

**Q2 — Is the survival bar wrong, and is it too low?** The plan uses 1.5 × 0.358 = 0.537 points per trade.
**I think this is a hole in my own plan.** One account holds one position; a mechanism firing 132 times per
session cannot take them all, so a per-trade average overstates what is harvestable, and the binding
constraint is per-*session* return under serial occupancy plus opportunity cost against the best
alternative signal. What is the right bar, and should the screen report per-session-under-occupancy rather
than per-trade?

**Q3 — Can entry and exit be trained jointly, and under what objective?** This is the core question. Given
§2, propose the objective and say how it avoids both failure modes: the always-exit degeneracy (P029–031
signature, >50% first-step exits) and its mirror, the 98%-label disaster. Cite the specific mechanism, and
run `prior_art_check` on it first.

**Q4 — What should the trader graph be?** `PROTOCOL101_FULL_TRADER_GRAPH_V2.json` has 47 nodes and 104
edges, each carrying `independent_acceptance`, `owner_gate`, `model_training` and `purpose`. Most of it was
built for the 0DTE class that is now closed. Which nodes survive a mechanism-first flow, which are dead,
and what **typed input/output contract and acceptance test** does each surviving node carry? The graph —
not a pile of scripts — should be the thing that gets validated. The exit went degenerate unnoticed
precisely because its acceptance test never referenced the entry's output distribution.

**Q5 — What is the smallest wave that answers the question?** Declare hypotheses, budget, controls. The
declared count **is** the maxT family. Budget is hard; exhausting it without `TIER_A` ends the wave as
`NO_EDGE`, which is a successful outcome.

**Q6 — Where does this plan fail?** Adversarially. What would you change?

## 4. Principles that are not optional

**Graph engineering.** Every node declares its input contract, output contract, and an independent
acceptance test. Edges are typed; no implicit data flow between stages. Reachability from the reset node
and declared terminal states. A node whose acceptance test cannot fail is not a test. Where a downstream
node consumes an upstream distribution — the exit consuming entry outputs — its acceptance test must
reference that distribution explicitly.

**Autoresearch loops.** Use `v4/research/pathd_research_loop.py`: `Hypothesis` (mechanism drives both the
prior-art search and the semantic hash, so renaming does not evade dedup), `WaveSpec` (`validate()` refuses
more hypotheses than the budget), `run_wave`, `Registry`, `prior_art_check` / `blocked_by_prior_art`.
Plus `v4/research/autoresearch_v2/` — `statistics.py` (paired summaries, session-blocked maxT),
`screens.py`, `causality.py`, and the semantic registry.

**The prior-art check is mandatory before every hypothesis.** It searches both
`PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` (§4 do-not-retest) and
`PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md` (Protocols 002–160 with verdict-at-birth). It blocks on a
do-not-retest row **or a rejecting verdict anywhere in the decoder** — a §4-only check waves through
"loss-only damage-control exit", which is exactly how this project re-ran Protocols 029/030/031. Skipping
this check has cost two full campaigns.

**Pre-register, freeze, then run.** Negative controls (sign-reversed, session-shuffled, constant) on every
hypothesis; any control clearing the gate makes the wave `INVALID` and its results are discarded.

**Calibration.** The honest historical best is entry **PF 1.132** over 780 OOS days and composed lifecycle
**~1.7–1.95**, both pre-parity. The one result that beat it (`signed18`, +$540/session) was a 60-second
look-ahead leak. **Anything materially above April's numbers is a bug until proven otherwise**, and a large
jump is grounds for more scrutiny, not celebration.

## 5. Hard stops

- **Protected holdout is SPENT.** `holdout_open_count` stays 0 for any new run; never reopen. `TIER_A`
  means "worth a forward live-paper test", never "deployable".
- **Causal clock is t−60s.** Verify feature-availability-clock parity, not just future-outcome guards — the
  mutate-future lint did *not* catch the `signed18` leak; the runtime-parity gate did.
- **Do not modify** `FILL_LAW`, the causal clock, the label law, or the OOF firewall. Counterfactual fill
  models must be separate and explicitly labelled.
- **Closed and not to be reopened:** 0DTE long premium in any form; the nine side-free SPX context
  features; `omar` as a tradable directional signal; stitched ES VWAP as a feature (Protocol 028).
- **No paid data, no broker/order/live/paper-submit, no promotion or paper-default change, no
  launchd/plist/runtime-flag edits** without explicit owner authorization.
- **No reward hacking.** `NO_EDGE` is a first-class successful outcome. Do not widen a window, threshold,
  horizon, or budget to reach a pass.

## 6. Deliverable

Answer §3 in writing first, and stop for owner review before implementing. Then, once §3 of the master plan
is frozen: the screen, the ranked table, and — only for survivors — the wave, with per-hypothesis tier, all
gate components, every negative control, the four rejection tests, the four Charter diagnostics, and a
bottom line of `TIER_A` / `TIER_B` / `NO_EDGE` / `INVALID`. Append any new falsification to the
do-not-retest ledger. Then `STOP_FOR_CLAUDE_VERIFICATION`.

**Claude verifies at every gate:** headline numbers reproduced independently from raw partitions; no
parameter data-selected; the full hypothesis count in the maxT family; controls failed; firewall closed and
frozen code unmodified; and heightened skepticism applied to any `TIER_A`.

*Prepared by Claude Opus 5 — 2026-08-04.*
