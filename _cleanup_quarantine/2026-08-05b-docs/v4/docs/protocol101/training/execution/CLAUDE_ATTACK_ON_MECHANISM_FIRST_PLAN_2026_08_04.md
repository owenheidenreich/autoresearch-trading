# Attack on the Mechanism-First Master Plan — answers to §3 (2026-08-04)

Response to [`CODEX_HANDOFF_MECHANISM_FIRST_2026_08_04.md`](CODEX_HANDOFF_MECHANISM_FIRST_2026_08_04.md).
No code written. No data purchased. No model trained. Read-only verification only.

**Bottom line: the plan has one defect that makes everything downstream of it ambiguous — the traded
instrument is never declared, and the three artifacts it rests on describe two different instruments.
Of the six mechanisms, one is untestable on owned data, one is forbidden by a frozen contract, one is
blocked pending a receipt, and two carry prior negatives the status column calls "NEVER TESTED".**

---

## Q1 — Is the mechanism list right?

### Both claims verified, and both are narrower than stated

**Window claim: CONFIRMED, with an important refinement.**
[`pathd_phase1_entry.py:380`](../../../../research/pathd_phase1_entry.py#L380) filters
`minute_of_session` to `[31, 350]` = **10:01–15:20 ET**. The opening 31 and closing 40 minutes carry no
entry rows.

The refinement matters: this is not an arbitrary cutoff, it is a **feature-warmup constraint**.
`official_spx_market_window_from_rows` ([`pathd_entry_features.py:101`](../../../../research/pathd_entry_features.py#L101))
takes `history_minutes=30` and computes `momentum_15m` only when `selected_index >= 15`
([line 214](../../../../research/pathd_entry_features.py#L214)). Minute 31 is the first boundary at which
the whole signed-17 context is warm. So **M1 is not "flip the filter" — the existing feature kernel
cannot produce features before 10:01 at all.** M1 requires a new warmup law, a new NaN policy, and its
own parity story. That is a materially larger piece of work than the plan's Day 1–2 budget implies.

The window is also a **six-block structure**, not a flat range
([`pathd_phase1_entry.py:734-748`](../../../../research/pathd_phase1_entry.py#L734)): blocks 31-89,
90-149, 150-209, 210-269, 270-329, 330-350, with one selection per block. This matters for Q2 — the
machinery already contains an occupancy proxy.

Also: the "every campaign ended at 15:00/15:20" claim is true of **entry** only. Graph node
`FT2-28-LIFECYCLE-ROW-BUILD` builds **15:31–15:55 ET** open-state decision rows. The closing window is
untested for entry, not untouched by the programme.

Data coverage is **not** the limitation — verified on the corpus at
`/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw`:

| Partition | Sessions | Time span (ET) |
|---|---|---|
| `opra_spxw_cbbo_1m` | 251 | 09:30:02 → 16:00:00 |
| `index/spx_1m` official | **251** (+254 proxy) | 09:30 → 16:00 |
| `index/vix_1m` official | **251** (+87 proxy VX) | 09:31 → 16:00 |
| `glbx_es_ohlcv_1m` | 261 | **13:30–19:59 UTC = 09:30–15:59 ET, RTH only** |

**VIX claim: CONFIRMED, and it is stronger than stated.** `vix_close` is declared at index 1 of
`_OFFICIAL_SPX_MARKET_FEATURE_NAMES` ([line 22](../../../../research/pathd_entry_features.py#L22)) and
written as literal `float("nan")` at
[line 221](../../../../research/pathd_entry_features.py#L221). It is absent from `required_market` in
[`protocol101_canonical_stage1_contract.py:177`](../../../../model/protocol101_canonical_stage1_contract.py#L177).

But it is not merely unused: **`vix_change` is a `QUARANTINED_ALPHA_TOKEN`**
([line 63](../../../../model/protocol101_canonical_stage1_contract.py#L63)), enforced by
`assert_model_alpha_firewall` → `assert_alpha_feature_names`
([line 366](../../../../model/protocol101_canonical_stage1_contract.py#L366)), which rejects any feature
name containing that substring. **M4 as written cannot be run without modifying a frozen contract the
plan's own §6 says stays frozen.** Either M4 is screen-only and never becomes a model feature, or the
owner must explicitly amend the firewall. Silently editing it would be the exact failure the governance
exists to prevent.

### Three corrections to the mechanism table

**M3 (overnight gap) is not testable on owned data — drop it.** The ES corpus is RTH-only (390 rows,
09:30–15:59 ET, 261 sessions). There are no overnight bars anywhere in the corpus. Testing M3 requires a
paid purchase, which is blocked. Pre-commit now that M3 is **dropped, not proxied** — substituting a
proxy series here is how a data contract quietly breaks.

**M4's "338 owned sessions" is 251.** 338 files = 251 `official_vix` + 87 `proxy_vx_fut`. Same inflation
in the SPX count: 505 = 251 official + 254 `proxy_es_fut`. The kernel's row invariants
([lines 142-149](../../../../research/pathd_entry_features.py#L142)) reject anything that is not
`is_official_index_data=True` from `thetadata_index_history_ohlc`. The honest usable overlap for every
mechanism is **251 sessions** — and 36 of those are the spent holdout (see Q6).

**M1 and M2 are not "NEVER TESTED" as mechanisms.** The prior-art documents contain:

- *V1A hand-coded opening-structure reversion* — **failed at boundary**, n=109, mean −0.354%, PF 0.918,
  with two controls. *V1B* (V1A + IV/VRP gates) — **failed**, PF 0.975.
- *Late-session Pickles/ORC trigger tournament* — **0/8 won**; every detector's MFE20 was 0.3–1.2 bps
  below the control's +7.20 bps; 0/22 direction matches.
- *Eight late-session soft features (W2a)* — **gate failed**, PF 0.722, DD 97%, direction accuracy 0.496.

These were hand-coded triggers scored on 0DTE option PnL, so they do not close an ES-underlying version.
But the status column should read **"never tested on this instrument and target"**, not "NEVER TESTED".
Overstating novelty is how a search gets re-run.

### The defect that matters most: the instrument is undeclared

The survival bar is **ES futures friction**. Verified in
`v4/audit/autoresearch/pathd_es_spread_measurement_2026_08_04/spread_measurement_result.json`:
`$4.50 commissions + 1.0734 elevated-RV ticks × $12.50 = $17.9176`, ÷ $50/point = 0.358 **ES** points.
The instrument is `ES.c.0`, GLBX.MDP3, from `bbo-1s`.

Meanwhile:

- **§2 of the handoff specifies the exit in option terms** — convex tail, p99 $86 vs hold-to-close
  $3,422, top-decile $2,331 vs $212,841, catastrophic floor. **ES has a linear payoff and no expiry.**
  None of that transfers.
- **The proven execution plane is options** — `SPXW  260804C07775000` on `DU***40`. Nothing in the record
  proves an ES futures round trip on that account: different product, different permissions, different
  margin.
- **The do-not-retest ledger closes the option route outright.** Row 180: *"Path-D 0DTE long-premium
  class: buying SPXW 0DTE premium at minute cadence — ANY features, model, exit policy, or execution
  style. CLOSED, structural."*

So the plan is, unavoidably, an **ES futures** plan — and the ledger explicitly sanctions that as one of
the three genuinely-new routes (*"a different instrument (ES 15–60 min, hurdle 52–54%)"*). That is fine.
But then §2's exit specification is written for the wrong instrument, and the "execution plane costs
nothing to hold" claim does not cover it. **Declare the instrument in §3 before freezing anything.**

### What is missing

The one mechanism with a measured economic reason **from our own corpus** is absent: **short premium.**
The strongest empirical fact the programme produced is that buying 0DTE premium is −$13.00/trade gross
with all friction removed, 5/5 folds — the variance risk premium, rediscovered on owned data with the
sign in the seller's favour. The ledger itself names it as genuinely new (*"a different position
structure (defined-risk short premium, longer tenor)"*).

I am **not** proposing to add it to this wave — it needs a new charter (the big-loss profile inverts) and
multi-leg data we do not own. I am flagging that the six declared mechanisms are all price-pattern
hypotheses of the kind that has failed sixty times here, while the one measured economic asymmetry in the
record is not on the list. If M1/M2/M5 return NO_EDGE, that is the next thing to design, not a seventh
pattern.

---

## Q2 — Is the survival bar wrong?

**Yes, and your instinct is right — but it is wrong in three ways, not one.**

### (a) The unit is wrong — use per-session net under occupancy

Plain arithmetic: an RTH session is 390 minutes. Non-overlapping capacity is **26 / 13 / 6** trades at
15 / 30 / 60-minute horizons. A mechanism firing 132×/session can take at most 20% of its own signals at
h=15, and far fewer if the firings cluster (which, for clock-driven mechanisms, they will by construction
— that is what M1 and M2 *are*).

The machinery already solves this once: the six fixed blocks with one selection per block cap a session
at 6 entries. That is a defensible, already-frozen occupancy rule, and I would reuse it rather than
invent one.

**Recommendation:** keep the 1.5× per-trade number as a cheap triage filter (it kills obvious losers in
minutes, which is the plan's virtue), but make the **survival decision** per-session:

> A mechanism survives if, under a declared one-position rule, per-session net dollars are positive with
> a session-bootstrap LCB > 0 **and** the sign is stable in ≥4/5 folds.

That bar cannot be reward-hacked by trade-count inflation, it is the statistic the four-box already
computes, and it does not require inventing an arbitrary dollar threshold today.

### (b) The bar is too low in accuracy terms — state the hurdle out loud

From the same measurement artifact, mean absolute ES move is $300.28 / $421.98 / $597.09 at 15/30/60 min.
Required directional accuracy for a gross target `T` is `0.5 + T/(2·M)`:

| Horizon | At 1.0× bar ($17.92) | At the plan's 1.5× bar ($26.88) |
|---|---|---|
| 15 min | 52.98% | **54.48%** |
| 30 min | 52.12% | **53.19%** |
| 60 min | 51.50% | **52.25%** |

A sustained **54.5% 15-minute directional hit rate on ES** would be a larger result than anything in this
project's honest record. By the plan's own calibration rule, **a pass at 15 minutes should be treated as
a bug before it is treated as an edge.** Put that sentence in the plan, so the Day-2 reaction to a
survivor is scrutiny rather than a training run.

### (c) There is no comparator, and ES has drift

The bar is absolute. The April machinery always scored against a best comparator. The feasibility study
records ES drift of +$0.13 to +$4.87 over these horizons on 261 sessions of a rising market. A long-biased
clock mechanism can clear an absolute bar on drift alone. **Require the screen to report drift-adjusted
excess** — each mechanism against buy-and-hold-for-h on the same boundaries — and to report the
comparator, not just the mechanism.

### (d) One credit and one caveat on the friction number

Credit: using the **elevated-RV** quartile spread (1.0734 ticks) rather than the unconditional 1.0397 is
the right conservative choice, since any volatility-selective mechanism trades in exactly that regime.

Caveat: $17.92 assumes a 1-lot marketable fill at touch with no queue, no impact, and no adverse
selection. At 1 lot in ES that is defensible, but it is a **floor**, not a central estimate. Label it as
a floor in the plan so nobody later reads a 1.1× result as "clears friction".

---

## Q3 — Can entry and exit be trained jointly, and under what objective?

**`prior_art_check` result: BLOCKED.** 49 hits, 4 blocking. The two that bind:

- Distillation row 180 (do-not-retest): the 0DTE long-premium class is closed for **any exit policy**.
- Farm lineage 031: *"Loss-only damage-control exit — Reject / exit-loop stop"*; 059: *"Require one-bar
  confirmation before a giveback exit — Reject / third lifecycle miss"*. Plus the Wave-1 Phase-1 exit
  repair family, `NO_EDGE` over 1,031 trajectories, all six arms losing to their comparator.

So: **a jointly trained entry+exit on SPXW 0DTE is forbidden outright.** On ES, the question reopens —
but the §2 framing does not survive the instrument change. There is no convex tail to amputate and no
expiry; "loss truncation vs tail capture" collapses into the stop/trail trade-off, which is a
well-trodden and mostly negative area.

### My answer: joint *selection*, not joint *training* — this wave

Both failure modes you named share one cause: **an exit head fitted against its own label.** Always-exit
degeneracy (17.46% positive first-state labels → the model learns "never hold") and the 98%-label
disaster are the two ends of the same mis-specification. The fix is not a better exit label. It is to
stop fitting one until an entry exists that is worth exiting well.

Proposal:

1. **Declare a small deterministic exit family** — time-stop at h; deterministic catastrophic floor at
   k×ATR (mandatory, per §2 item 2, and per P073/P087 both rejecting its removal with 26.7% of exits
   depending on it); one trailing variant. Three members, declared, in the maxT family.
2. **Train only the entry.**
3. **Evaluate the entry under each fixed exit under serial occupancy.** The "joint system" is the
   (entry, exit) *pair* chosen by outer evaluation.
4. The composed pair re-clears the full gate; it inherits no tier.

This is honest about the record: this project has never produced a learned exit that beat a deterministic
one, across P029/030/031/059 and the six-arm Wave-1 repair. Spending a budget-constrained wave on a
seventh attempt is the sunk-cost move.

### If the owner wants a learned exit anyway

Then the only objective that avoids both degeneracies is one where the exit is **trained against the
entry's realized forward-path distribution** and scored by the *same session-level net objective as the
entry* — not a per-trajectory exit label — with the deterministic floor applied first so the learned
policy chooses only among non-catastrophic states. And it must carry this acceptance test, which is the
one that was missing:

> The learned exit's **exit-time and payoff distribution** is compared against hold-to-horizon on the
> same trajectories. Preservation of p99 and top-decile contribution below a declared threshold is a
> **failure**, not a diagnostic.

That test, and not a better label, is what would have caught 95.1% first-step exits on the day it
happened.

---

## Q4 — What should the trader graph be?

### Graph V2 is not prunable — its product contract is the dead object

`PROTOCOL101_FULL_TRADER_GRAPH_V2.json` declares
`product_contract.instrument = "SPXW_0DTE"`, `action_universe = "WAIT plus 42 contract slots"`,
`ask_entry_bid_exit_accounting`, `SPXW_0DTE_eligibility`. **That is precisely the class closed as
structural by ledger row 180.** You cannot prune V2 into a mechanism-first flow; you need Graph V3 whose
product contract names the instrument (per Q1) and whose action universe is that instrument's.

### Two Phase-F nodes are dead resources, and the graph cannot legitimately terminate

`FT2-91-PROTECTED-HOLDOUT` and `FT2-90-FRESH-CONFIRMATION` are both `owner_gate=true` terminal-path nodes.
Verified: `holdout_access_receipt.json` in
`v4/audit/autoresearch/autoresearch_v2_entry_model_confirmation_2026_08_02_attempt001/` reads
`holdout_open_count: 1`, `status: ACCESS_COMPLETE`, `result_status: CONFIRMED_EDGE` — the result later
invalidated by the 60-second look-ahead. **The graph's only confirmation path runs through a spent
resource.** V3 must mark these `DEAD`, not `pending`, and replace them with a forward live-paper node
whose acceptance is calendar time, not compute.

### Survives / dies

**Survives (typed, retained):** `FT2-08` data/tensor/label contract, `FT2-10` entry science contract,
`FT2-11` evidence statistics contract, every `independent_acceptance` node (`20, 26, 31, 51, 70, 74, 80,
81`), every `owner_gate` (`21, 52, 62, 95`), `FT2-82` complete-system freeze.

`FT2-80-FOUR-BOX-COMBINED-AUDIT` — *"Boxes A–D under strict serial economics; the full trader must beat…"*
— is the most valuable node in the graph and the **only** place occupancy is enforced. Everything in Q2
should be pushed into its contract rather than reinvented in the screen.

**Dies with the 0DTE class:** `FT2-05` opportunity census, `FT2-25` full-ladder feature admission,
`FT2-28` lifecycle rows (15:31–15:55, 0DTE open-state), `FT2-30` entry harness pilot, `FT2-92` IBKR
decision shadow as specified — all keyed to the 42-slot SPXW ladder.

### The node the graph does not have

There is **no mechanism-economics screen node**, and the master plan does not run through the graph at
all — which contradicts "the graph, not a pile of scripts, is the thing that gets validated". V3 needs:

```
NODE  V3-05-MECHANISM-ECONOMICS-SCREEN
in    {raw partition manifest + sha, mechanism spec, occupancy rule, comparator}
out   {per-session net under occupancy, per-fold sign vector, trade count,
       comparator delta, capacity ratio (fired ÷ takeable)}
acc   negative controls (sign-reversed, session-shuffled, constant) all fail
      AND a known-answer test on synthetic data with a planted edge of declared
      size recovers it within tolerance
```

Note the second half. The plan requires a known-answer gate on **nulls** but not on the **screen itself**
— the same class of error one level up. A screen that cannot demonstrate it would detect a planted edge
is not a screen.

### The coupling law

Every node consuming an upstream distribution declares it as a **named typed input**, and its acceptance
test asserts against that named input. Concretely, in V3 the exit node's input type is
`entry_forward_path_distribution`, and its acceptance test includes the p99/top-decile preservation
comparison from Q3. A node whose acceptance test does not reference its declared upstream inputs fails
schema validation. That rule, mechanically enforced, is what makes the 95.1% failure impossible to repeat
silently.

---

## Q5 — What is the smallest wave that answers this?

Given Q1, the six-mechanism family does not survive contact:

| | Status |
|---|---|
| M1 opening range | Testable, but needs a new feature warmup law (cost understated) |
| M2 closing hour | Testable; cheapest of the six |
| M3 overnight gap | **Untestable — no overnight bars owned. Drop.** |
| M4 VIX regime | **Blocked by the frozen alpha firewall.** Needs owner amendment or screen-only routing |
| M5 calendar | Testable; nearly free once M1/M2 boundaries exist |
| M6 SPX–ES basis | **Blocked pending an ES clock receipt**, and `ES.c.0` is a stitched continuous series with 4 rolls in-corpus (2025-09-22, 2025-12-22, 2026-03-23, 2026-06-19) — the same stitched object P028 rejected, with roll discontinuities of calendar-spread size. Basis is also dominated by the deterministic carry term, which the spec does not remove. |

One cheap unblock worth doing regardless: the 20 owned `glbx_es_bbo_1s_measurement_2026_08_04` sessions
can now ground the **GLBX ES completed-minute timing receipt** whose absence produced
`BLOCKED_CLOCK_OR_DATA_CONTRACT`. That is a derivation from data we already own, not a purchase. It does
not resurrect M6 this week, but it retires a standing blocker for free.

**My recommendation — one family, not six:**

> **Family: ES intraday clock structure.** M1, M2 and M5 are all deterministic functions of the session
> clock and the calendar. Testing them as three families triples the maxT count for no scientific gain.
> Declare **one** family: clock-and-calendar structure on ES, 3 horizons × 2 directions = **6 declared
> members**, session-blocked maxT over 6, occupancy rule = the six frozen blocks, comparator =
> buy-and-hold-for-h, controls = sign-reversed / session-shuffled / constant.
>
> **Budget: 2 days. Training only if per-session LCB > 0 and sign-stable ≥4/5 folds. Otherwise
> `NO_EDGE`, and that is Monday's deliverable.**

That fits the calendar honestly. Six mechanisms × three horizons in two days, with M1 needing new warmup
machinery, does not.

---

## Q6 — Where does this plan fail?

Ranked by how much damage each does.

1. **Undeclared instrument.** The bar is ES, §2's exit spec is options, the proven execution plane is
   options, and the ledger closes options. Whichever way this resolves, at least one of the three
   artifacts the plan rests on is invalid for the work being done. *Fix: declare the instrument in §3
   before freezing; rewrite §2's exit properties for it.*
2. **The screen's own selection sits outside the maxT family.** §4 says no p-values on pass 1; §5 declares
   the family at pass 2. But the search happens in pass 1 — 6 mechanisms × 3 horizons × direction is ≥36
   implicit comparisons, and only the survivor is carried forward. This is the multiple-comparison error
   at exactly the level the handoff warns about for nulls. *Fix: the declared family includes every
   screened member, not the survivors.*
3. **Deadline pressure with no marginal-result rule.** Days 3–5 are pre-reserved for training. A
   0.55-point survivor is indistinguishable from noise, and the plan has no rule that stops it.
   *Fix: pre-commit that a survivor below 2× the bar, or with LCB ≤ 0, is `NO_EDGE` regardless of
   remaining calendar.*
4. **Pooled-mean pass rule.** §4 makes fold instability *visible* but not *binding*. One volatile month
   can carry a pooled mean over 251 sessions. *Fix: per-fold sign stability is part of the pass rule,
   declared now.*
5. **M3's missing data creates a widening incentive.** On Day 1 the discovery that no overnight bars exist
   invites either a paid purchase (blocked) or a proxy substitution (silent data-contract break).
   *Fix: pre-commit M3 as dropped, not proxied.*
6. **M4 requires editing a frozen contract.** `vix_change` is a quarantined token. *Fix: route M4 as
   screen-only and never a model feature, or obtain an explicit owner amendment. Do not edit silently.*
7. **The holdout sessions are inside the corpus.** 251 OPRA sessions = 215 development + the 36 spent
   holdout. The plan never says which are used. *Fix: declare 215; if the 36 are reported at all, they
   are reported separately and gate nothing.*
8. **The owned-data claims are inflated** (505 SPX / 338 VIX are really 251 official each). If the screen
   silently picks up `proxy_es_fut` / `proxy_vx_fut` files it changes the data contract mid-flight.
   *Fix: the screen's partition manifest filters on `is_official_index_data` and records the file count.*
9. **No comparator.** See Q2(c) — ES drift can carry a long-biased mechanism over an absolute bar.
10. **Friction is a floor, not a central estimate.** 1-lot, at touch, no queue, no impact.
11. **The plan does not run through the graph**, so the claim that the graph is what gets validated is not
    true of this week's work. Either add the screen node (Q4) or drop the claim.

### What I would change, in three sentences

Declare the instrument as ES and rewrite §2's exit properties for a linear payoff. Collapse M1/M2/M5 into
one six-member clock-and-calendar family, drop M3, route M4 around the firewall or defer it, and defer M6
behind the ES clock receipt. Make the survival test per-session net under the six-block occupancy rule
with an LCB and 4/5 fold sign-stability, and pre-commit that anything below 2× the bar is `NO_EDGE` no
matter what day it is.

---

*Prepared by Claude Opus 5 — 2026-08-04. Verification was read-only: no purchase, no broker contact, no
training, no runtime mutation, holdout untouched.*
