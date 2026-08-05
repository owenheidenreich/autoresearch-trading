# Codex Handoff — Research the Route Past Every Gate

**To: Codex (GPT-5.6, extra-high reasoning). From: Claude Opus 5, 2026-08-05.**
**Task class: research and adversarial review. Produce no model, run no capture, spend no money.**

---

## 0. Read these first, in this order

1. [`STATUS.md`](../../../../STATUS.md) — the only status page. Gate chain G1–G9 lives there.
2. [`PATHD_PROGRAMME_RESTART_RECORD_2026_08_05.md`](../contracts/PATHD_PROGRAMME_RESTART_RECORD_2026_08_05.md)
   — the programme is REOPENED; what is superseded and what still binds.
3. [`PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`](../history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md)
   — the do-not-retest ledger. **Rows 177–183 are the live constraints.** Anything you propose that a row
   closes is dead on arrival, and proposing it is the single most common failure in this project.
4. [`CLAUDE.md`](../../../../CLAUDE.md) — ground rules, including the three signature tiers.
5. [`PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md`](../contracts/PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md)
   §6 and §7 — the restart conditions and the durable lessons. Both still bind.

## 1. The task

**For each gate G1–G9, research what it would actually take to get past it, and rank the whole set by
leverage.** This is the most ambitious ask the project has made: not a plan for one wave, but the route
through the entire chain, with the dead ends identified before we walk into them.

The deliverable is one document. For every gate, answer:

- **What specifically blocks it** — mechanism, not vibe.
- **What would count as passing** — a preregisterable criterion with a number, not "looks good."
- **The cheapest experiment that could pass or fail it**, and its cost in time, money, and calendar days.
- **The most likely way we fool ourselves here**, and the control that would catch it.
- **What it costs if we get it wrong** — which gates downstream become invalid.

Rank by leverage: *(probability the gate is passable) × (what passing unlocks) ÷ (cost to find out)*.
Say plainly which gates you think are impassable. **A well-argued "this chain has no viable route" is a
successful deliverable** and worth more than an optimistic sequence.

## 2. Non-negotiable context

**The protected holdout is SPENT.** Opened once on 2026-08-02 for `signed18`, which was then invalidated
by a 60-second look-ahead. There is no historical confirmation firewall. Any design that assumes one is
invalid.

**Three standing traps, all of which have already caught this project:**

1. **The shared-term artifact** (ledger row 183). A feature that is a function of price `P(t)` and a target
   `P(t+h) − P(t)` share the `−P(t)` term and correlate for *any* bounded path. A session-shuffle control
   cannot detect it — shuffling destroys the pairing the artifact lives in. Matched surrogates with no
   predictability by construction reproduced **78.5%** of a `+0.522` headline.
2. **Searching over nulls.** Nulls were designed one at a time, each repairing the last after seeing the
   data. That is multiple comparisons moved up a level. **A null must pass a known-answer gate before its
   verdict is believed.** The gate caught a null that manufactured a monotone surrogate (ρ = 1.000) where
   the real data is flat (−0.507), and with it a p-value of 1e-06 that was measuring its own defect.
3. **Statistics before economics.** The number that closed `omar` (−0.324 points/trade) needed one pass
   over owned data and sat computable for a day while three studies argued about a correlation
   coefficient. **Raw economics first, always.**

**Measured costs** (do not re-derive, do not assume): ES round-trip friction **$17.9176 = 0.358 ES
points**; ES spread 1.0397 ticks (1.0734 top volatility quartile); SPX option round trip $3.08.

**Owned data**: 254 usable ES OHLCV-1m sessions (2025-08-01 → 2026-07-31, RTH-only 09:30–15:59, 390 bars,
7 empty files, `ES.c.0` stitched with 4 in-corpus rolls); 251 Databento OPRA SPXW CBBO-1m/1s; 251 official
SPX 1m and 251 official VIX 1m (of which **36 are the spent holdout**); 20 sessions ES bbo-1s.

## 3. Attack my premises — two have already failed audit

I am not asking for agreement. In the last two days, **four** premises in this project's plans were found
false, two by an adversarial reviewer and two by me:

| Premise | Verdict |
|---|---|
| "116.2% win rate required, therefore impossible" | **REFUTED** — median-in-an-EV-formula artefact; the mean-payoff screen gives 80.19% |
| "ES is the only reachable instrument" | **REFUTED** |
| "VIX has never entered any model" | **FALSE** — `vix_level`, `vix_change_5m`, `vix_change_15m` are in `lean_autoresearch/harness.py`; that campaign returned `NULL_NO_NEW_ENTRY_EDGE` |
| "Overnight gap untestable — corpus is RTH-only" | **FALSE** — the gap is today's 09:30 open minus yesterday's 15:59 close, both RTH bars. Computable on **253 of 254** sessions; median \|gap\| **19.25 points** against a 0.358-point bar |

Assume more are wrong. **Verify every load-bearing claim in STATUS.md and in §4 below against code and
data before you build on it, and report what you find false.** That is the highest-value thing you can do.

## 4. Per-gate research questions

### G1 — Direction (the gate that blocks everything)

*Can we predict SPX/ES direction over 15–60 minutes well enough to clear 0.358 points?*

- **Is the candidate list right?** My frozen list is two mechanisms: **M1** opening range 09:30–10:00, and
  **M3** overnight gap (reinstated above). Each carries a stated economic reason. Is there a mechanism with
  a real economic story that is neither in the ledger nor already refuted? Propose at most two more, each
  with its reason stated in one sentence that survives being said out loud. **Check the ledger first.**
- **M1 needs a new warmup law.** The 09:30–10:00 window was never tested because `history_minutes=30` and
  `momentum_15m` needs 15 prior bars, so the kernel structurally cannot emit features before 10:01. That is
  an implementation accident, not a finding. Design a feature set computable from ≤5 bars of session
  history. What is causally available at 09:35 — prior-session bars, opening range, gap, volume against a
  session-relative baseline? Does anything in that set escape trap 1?
- **M3 needs roll handling.** On the 4 `ES.c.0` roll dates the gap contains the roll spread, which is a
  contract artifact, not a market move. Exclude, adjust, or use a different series? Note Protocol 028
  rejected stitched ES VWAP *as a feature*; is a gap computed across a roll boundary the same objection?
- **The primary statistic.** I claim it must be **net points per calendar session under a frozen
  one-account serial allocator** (causal selection, frozen tie-breaks, one position, no overlap, flat by
  close), with a session-bootstrap lower bound and 4-of-5 fold sign stability — because per-trade means
  overstate what one account can harvest. Attack this. Is there a better primary statistic?
- **Power, computed before anything runs.** With 254 sessions and 5 folds, what is the minimum detectable
  effect at 80% power? **The previous campaign died `UNDERPOWERED` because the power condition tested the
  learned subset (60 sessions, 3 folds), not the full index.** If the MDE exceeds the plausible effect
  size, G1 is unanswerable on owned data and we must know that *now*, not after the screen.
- **Surrogate design.** Specify the matched surrogate (own realized volatility, no predictability by
  construction) and the known-answer gate it must pass first.

### G2 — Option wrapper re-opens

Ledger row 181 permits re-entry on *"demonstrated directional skill on the underlying."* **That phrase is
not a criterion — it cannot be preregistered as written.** Convert it into an exact, falsifiable bar.

- How much directional skill, at what horizon, with what confidence, sustained over how many sessions?
- Row 181 also says friction is **4.68% of a $565 average premium** and the class is negative-EV *before
  costs* at minute cadence. The feasibility study found the same option held **60 minutes** clears a far
  lower hurdle (53.9–57.2% vs 80.2–116.2%). **Does directional skill plus a 60-minute hold actually escape
  row 181's structural closure, or does theta eat it regardless?** Derive it — this decides whether the
  owner's stated product is reachable at all.
- If the answer is no, say so. The honest alternative (ES futures directly) is a different product from
  what the owner wants, and he deserves to know which one he is getting.

### G3 — Feature certification

- The capture is now **two sessions** (08-06, 08-07), four windows, two open bursts. The admitted
  availability clock is `max p99 across all declared windows and sessions`. **Is a max-of-n estimator
  defensible at n=2?** What does extreme-value theory say about estimating a worst-case arrival latency
  from two stressed samples, and what safety margin should be added? Current basis is 5 samples spanning
  608–2335 ms, a 3.8× spread.
- **Which of the 65 unblocked features actually serve G1?** They are OPRA option-surface features. If G1 is
  a direction question on ES/SPX, are these features on the critical path at all, or do they only serve the
  option wrapper at G2? Be blunt if the capture is not on the critical path.
- The 6 account-state features have passing parity receipts but no arrival clock. What is the cheapest way
  to measure fill-notification latency?

### G4 — Train

- Given whatever survives G1, what model class is justified? April's honest calibration was a shallow
  ranker (RF, 200 trees, depth 5, leaf 20 → PF 1.291). **The owner wants a neural network.** At what sample
  size and signal-to-noise does a network beat a shallow ranker here, and what would we need to see at G1
  to justify one? Answer with a number, not a preference.
- The 18-feature contract had **zero** ranking power (deciles: top −$15.37, middle −$10.79, bottom
  −$16.32 — unordered, not weakly ordered). What diagnostic run at G1 would predict ranking power *before*
  a training run is spent?

### G5 — Validation replay

- **I found a defect in the gate; verify and fix it.** Condition 7 requires all 8 fee/latency cells
  directionally positive, but the metric is `delta = learned − comparator` and both legs carry the same
  `fee_per_side`, so the fee term **cancels exactly**. Confirmed empirically: the `fee=3.00` and `fee=4.00`
  rows are bit-identical in `replay.json`. **The 8-cell grid is 4 latency tests run twice; fee robustness
  is not being measured at all.** Design a genuine fee-stress test on an absolute metric.
- Are the other 6 conditions individually sound, or does another one fail to test what it claims?

### G6 — Runtime parity

Built and runnable (`v4/research/autoresearch_v2/runtime_decision_parity.py`); demands 100% per-decision
match. It caught `signed18` at 18.44%. What realistically causes sub-100% match for a *correctly* built
model — float non-determinism, tie-breaking, quote-age boundaries, reconnects? Which are acceptable and
which are bugs? Define the float tolerance before we need it.

### G7 / G8 — Live shadow and guarded paper

- **The power question that could sink the whole plan.** Fresh live paper is now the *only* out-of-sample
  test. If G1 delivers an edge of size X points/session, **how many paper sessions are needed to confirm it
  at 80% power?** If that is 200 sessions, the plan is a year long and the owner must know today. Compute
  this as a function of X and report the curve.
- Design the pre-registration: session count, minimum detectable effect, practical bar, and stopping rule
  — all frozen *before* the first paper session.

### G9 — Real money

Out of scope. State only what evidence the eventual owner packet would require.

### Cross-cutting — rebuilding the confirmation firewall

The single structural fix available. Two routes:

- **Acquire an unused historical range** (e.g. ES 1m, 2024-08 → 2025-07) never seen by any campaign, and
  reserve it untouched. **Price it exactly** — the ES spread measurement cost $1.47, so this may be a few
  dollars. Do not download; produce the cost and the exact scope.
- **Reserve forward sessions**, which costs calendar time.

Compare them honestly: an earlier period tests regime robustness; forward data tests non-staleness. Which
is the better firewall for a mechanism-based claim, and can both be used without double-dipping?

## 5. Deliverable

One markdown document at
`v4/docs/protocol101/training/research/PATHD_GATE_CHAIN_RESEARCH_2026_08_05.md`, containing:

1. **Ranked leverage table** across all gates — the headline.
2. Per-gate answers to the five questions in §1.
3. **Refutations section** — every claim of mine you found false, with evidence. Expected to be non-empty.
4. **The impassability verdict** — which gates you believe cannot be passed, and what evidence would change
   your mind.
5. **A one-week concrete sequence**, if one exists. If none does, say that instead.

Plain English per `CLAUDE.md` §2. Define terms on first use. Every claim cites a file, receipt, or
measurement. Where evidence is missing, write `UNKNOWN`.

## 6. Hard stops

No model training or fitting. No broker or vendor connection. No paid download — **price it, do not buy
it.** No launchd, runtime-flag, paper-default, or promotion changes. Do not open the protected holdout
(`holdout_open_count` must stay at 1 and no new access). Do not modify the causal clock, `FILL_LAW`, the
label law, or the out-of-fold firewall. Do not weaken any gate to make a result fit — if a document and the
code disagree, the code wins and the document gets fixed.

Reading data and code, computing on owned local data, and static analysis are all in scope and encouraged.
The gap check in §3 is the standard: a single decisive query beat a paragraph of reasoning.

*Signed: Claude Opus 5 — 2026-08-05.*
