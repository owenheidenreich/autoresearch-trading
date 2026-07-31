# FT2-21 Design-Approval Bundle — Cohesive Narrative (DRAFT)

**STATUS: OWNER-SIGNED — FT2-21 DESIGN APPROVED (2026-07-30).** The delta
review is clean (§3), the pilot findings are folded in (§8), and the owner has
signed the design approval (§12). Amendments D54-D60 are adopted. The owner's
reserved pre-training step is the Walking-Skeleton dry-run (separate plan).
Formal incorporation of D54-D60 into the binding consolidated authority (new
authority hash, receipt re-pin) and the Graph V2 FT2-21 state update are the
next governed Codex step. Binding docs remain the consolidated authority and
Graph V2 JSON.

Prepared 2026-07-30 (Fable), folding two independent streams into one story:
the internal delta-review of the repaired design, and the external ChatGPT
Deep Research second opinion.

---

## 1. What signing FT2-21 does

FT2-21 is the owner gate that ends **Phase A (design)** and unlocks **Phase B
(machinery)** — the first work that builds real tensors from real data. It
does **not** authorize training, GPU spend, paid downloads, broker contact, or
paper trading; each of those remains behind its own later gate. Signing says:
"the design contracts are sound enough to start building the machine that will
one day test them."

## 2. The journey to here, in one paragraph

The three design contracts (data/tensor/label, entry science, evidence
statistics) went through three independent adversarial reviews that took the
finding count 28 → 17 → 6 — a design being sanded smooth, not one falling
apart. The third review ended in a rules-mandated STOP with 5 residuals; the
owner authorized one scoped fix round (Codex), which closed all 5, rebuilt the
opportunity census to v4, and was independently Fable-verified. A follow-on
reseal corrected one clerical drift (an authority document referencing an old
graph hash) and added two new consistency-checker guards so that class of
error cannot recur. In parallel, the owner commissioned an external
second-opinion review from ChatGPT Deep Research, deliberately framed to be
fair (name strengths, not just faults) and actionable (every finding must
carry a fix). This bundle folds both results together.

## 3. What the delta review confirmed — DONE, CLEAN (2026-07-30)

The delta-scoped 3-seat fresh-seat review ran and returned
**`DELTA_REVIEW_CLEAN`** (packet:
`protocol101_ft2_20_delta_scoped_review_attempt002`). Fable-verified:
- Three independent, isolated seats (trading-realism, ML-statistics,
  live-parity) each returned `DELTA_REVIEW_CLEAN`; the controller read no
  seat output until all three completed.
- Delta-scoped to the diff manifest (baseline commit `a7602fdc`); the frozen
  ~95% stayed out of scope.
- Reviewed the current authority `d115b953` (verified — not a stale version).
- Two in-scope findings (B2, D1) were bounded-fixed and re-reviewed clean.
- Routing: `STOP_FOR_OWNER_DECISION` — the review deliberately did NOT
  self-approve FT2-21. Design is cleared; approval remains the owner's.

**Owner-directed sequencing change (2026-07-30):** the 1-second exit-realism
pilot (D55) now runs BEFORE FT2-21 signature, not after, because its findings
could change the design being approved (see D55/D57). This bundle is therefore
HELD in draft until the pilot completes and its findings are folded in.

## 4. The independent second opinion (ChatGPT Deep Research)

**Verdict: no FATAL findings. The premise survived external review.** The
reviewer — literature-grounded and explicitly instructed to be balanced —
concluded the project is "not obviously unsound" but "trying to find a narrow
positive island in a market that is, on average, unfavorable to long option
buyers," and recommended proceeding in disciplined phases rather than stopping.

**Three strengths it told us to protect:**
1. **Claim discipline** — the project rigorously separates "design complete,"
   "component accepted," "paper ready," and "real-money ready." Options
   backtests fail as often from claim inflation as from modeling error; this
   design resists that.
2. **Economically coherent trader identity** — no win-rate optimization, no
   mechanical scalping, no hard profit cap; it correctly relies on rare convex
   winners to pay for long-premium carry (the Pickles profile).
3. **Right causal direction** — next-minute fills, rejected-fill accounting,
   complete-ladder masks, separate entry/exit evidence, and formal live-parity
   gates are the correct conservatism for this domain.

**The one genuinely valuable reframe:** the variance-risk-premium literature
shows long index options bleed most *overnight*, while the *intraday*
component is roughly neutral-to-positive. This trader is intraday-only with
forced-flat at 15:55 — so the charter's "no overnight" rule is not just risk
hygiene, it places the strategy in the one window where the structural
headwind is weakest. That is the economic reason the game is plausibly
playable.

**Structural (non-fatal) findings, all fixable:** adverse base-rate economics
(edge must be selective and real after costs); minute data adequate for entry
but too coarse to validate exits; underpowered/one-regime corpus; model class
possibly large for the sample; feature set possibly too sparse; governance's
marginal value now falling relative to empirical evidence. None kills the
approach; all convert to the amendments below.

## 5. Adopted amendments (D54-D57) — what actually changes

These are the owner-adopted changes folded into the design at FT2-21. They
add gates and evidence steps; they do **not** reopen the frozen contracts.

**D54 — Name the Entry Economic No-Harm Gate (G-ENTRY-FEASIBILITY).**
Before any neural GPU spend, a simple-model entry candidate (heuristic or
gradient-boosted tree, per the signed model-family ladder) must show, on
governed 5-fold serial replay with full costs, a result whose adjusted 95% CI
does not show it worse than the P5 benchmark. Pass unlocks the neural
campaign; fail routes to `no_genuine_entry_signal` and stops the spend. This
elevates existing component-freeze + MDE-before-spend logic into one
un-skippable checkpoint — the cheapest possible defense against spending on a
model the data can't support.

**D55 — One 1-second pilot resolves both exit-realism and forward-fill.**
Confirmed empirically: the current minute corpus cannot detect vendor
forward-fills (the staleness columns are inert; cheap wings are 99.8%
identical minute-to-minute), and it cannot see intra-minute exit dynamics.
Confirmed via Databento (see §5 corrected table): the 1-second subsample
starts 2025-02-20, full quote-events (`cmbp-1`) reach back to 2023-03-28, and
NOTHING sub-minute exists before 2023-03-28. Therefore **minute data is the
permanent training substrate for regime breadth back to 2022; sub-minute data
is a validation/exit-calibration instrument** on top of it (2023-03-28+ via
`cmbp-1`, or 2025-02-20+ via the cheaper `cbbo-1s` already owned). The pilot
ran on 30 recent sessions to measure minute-compression distortion and the
forward-fill rate (results in §8). **Cost: ~$24 (30 sessions), spent under the
$30 cap.**

**D56 — Preregister Tier-1 feature-admission candidates.** Add, through the
existing admission ladder (parity gate first), features that raise trading
signal without reintroducing vendor drift: **calendar/event flags** (FOMC,
CPI, OPEX, half-days — deterministic, ~zero parity risk) and **prior-day
levels** (prior close, prior high/low, overnight gap — index-derived, low
risk). These directly address the "feature set too sparse" finding and the
owner's own prior-day-context instinct. Term-structure/VIX features stay
deferred (already quarantined for insufficient paired history).

**D57 — 2022-2024 minute backfill: DEFERRED pending the D55 pilot
(owner decision 2026-07-30).** Not purchased yet, deliberately, for three
reasons: (a) the 1-second pilot may reveal something about our training/exit
methods we want to change before committing to more minute data; (b) we may
decide to use sub-minute data for the *entry* model too, not just exit
validation; (c) the sub-minute vs minute history tradeoff (corrected below)
may change the substrate strategy. The decision is revisited only after the
pilot's findings land. **Cost preserved for when we decide (exact,
cost-estimate-verified): $41 (quotes + definitions) to $347 (all four
schemas, corpus-identical), for ~3-4× the current corpus across the 2022 bear
/ 2023 recovery / 2024 normal regimes.** Still the cheapest lever against the
underpowered/one-regime findings, and — per the correction below — minute is
the ONLY resolution that reaches 2022 at all.

**Data-substrate reality (CORRECTED 2026-07-30 after direct Databento API
verification; supersedes an earlier incorrect "no sub-minute before 2025"
claim):**

| Resolution | Earliest available | 0DTE cost/session | Full-window cost & /month |
|---|---|---|---|
| Minute (`cbbo-1m`) | pre-2022 (years back) | ~$0.028 | 2025-02-20→now: **~$10 total, $0.58/mo** |
| Full quote-events (`cmbp-1`, finer than 1s) | **2023-03-28** | ~$1.74 | ~$1,300 for 2023-03-28→2024 (0DTE) |
| 1-second subsample (`cbbo-1s`) | 2025-02-20 | ~$0.91 | 2025-02-20→now: **~$328 total, $19.04/mo** |

Corrected conclusions: (1) sub-minute quote data DOES exist back to
**2023-03-28** via `cmbp-1` — not 2025-only; only the convenient 1-second
subsample starts 2025-02-20. (2) NOTHING sub-minute exists before 2023-03-28,
so the earliest history can only ever be minute resolution — the 2022 backfill
(D57) is not made redundant by any sub-minute option. (3) 1-second costs ~33×
minute per session. (4) For a longer sub-minute exit-calibration window than
the 2025+ data already owned, `cmbp-1` from 2023-03-28 (~$1,300 0DTE) is the
option; it is finer than 1-second and reaches ~2 years further back.

**Active new spend right now: the D55 pilot only, $24.12 one-time.** The
backfill's $41-347 is deferred, not committed.

## 6. What is explicitly NOT changing

- The three design contracts' frozen content (the triple-reviewed ~95%).
- The signed parity gate: no real training until train-vs-live feature parity
  is certified — this stands independent of everything above.
- The Charter, G4 survival rules, product contract, and all prior owner
  decisions (D01-D53).
- The staged-gate philosophy: FT2-21 unlocks machinery, nothing further.

## 7. Sign-off meaning

Approving FT2-21 (once §3 is real) means: the design is accepted as sound;
the D54-D57 amendments are adopted; Phase B machinery construction may begin;
and the data-acquisition decisions (D55 pilot, D57 backfill) are authorized to
proceed to their own execution gates. It does not start training, spend GPU,
or touch a broker.

**Highest allowed claim after signing:** "Protocol101 Full Trader Graph V2
design is owner-approved; Phase B machinery is unlocked; two small,
cost-verified data acquisitions are authorized to proceed."

---

## 8. Pilot findings folded in (D58) + premium-band evidence

### D55 pilot — resolved (2026-07-30, $24 spent, under cap)
- **Forward-fill: benign** (0.067%). The Point-3 concern is empirically a
  non-issue; the inert staleness columns don't matter because there is almost
  nothing to detect.
- **Minute fill assumption: unbiased** (mean −0.03 pts, ~50/50 optimistic).
  The backtest P&L is not inflated by a fantasy fill — it adds noise, not a
  favorable lie. High-value for trust.
- **Exit/floor NOT validatable on minute data alone** (74% pooled hidden
  intra-minute dip; $34-141/contract p95 floor-trigger gap). Concentrated in
  expensive contracts; small on cheap wings.

### D58 (NEW) — binding requirement on the future lifecycle (exit/floor) design
Does NOT reopen the frozen entry contracts. When the lifecycle contract
(Phase D / FT2-60 series) is written, it MUST: build its exit/floor labels
from sub-minute data (the owned 2025+ `cbbo-1s`, or `cmbp-1` back to
2023-03-28 for a longer window — corrected data table in §5); and it must NOT
claim intraminute floor protection from minute paths. Plus one report-only
entry-side check: confirm 1s-corrected early-drawdown labels do not materially
change entry contract *ranking* (expected benign — fills are unbiased and
cheap-band gaps are tiny — but verify).

### Premium-band profitability evidence (census oracle, existing data)
The hindsight opportunity concentrates in the **$3-8 premium band** (oracle
677 trades, mean $180, 36% RoP), not the cheap wings (0 oracle trades ≤$1;
P5 loses in $1-3). The current $500 cap admits only the lower half of the
profitable band on a $10k account. **Evidence tilts toward widening D48 toward
~10% to capture $3-8**, tempered by: it is hindsight; $3-8 carries the largest
intra-minute gaps (couples to D58); and a single 10% trade can exceed the 5%
worst-day target if it gaps before the exit acts.

### D48 disposition (owner decision 2026-07-30)
Keep D48 at **5% into the first training run**; treat "widen toward ~10% to
capture $3-8" as the leading hypothesis. The entry-feasibility gate (D54) must
report *realized* band-profitability from the trained model so the cap is
decided on actual model behavior, not hindsight. "Preferred band" is a
reported/learned outcome, never hardcoded. "Rarely held to zero" is explicitly
the exit model's mandate (loss truncation), not an entry-cap job.

---

## 9. Path to first training (the stop line)

After FT2-21 is signed, these are the ordered gates to the first training run.
The owner has reserved a step before committing to full-scale training; the
STOP line is marked.

1. **FT2-21 signature** (owner) — unlocks Phase B. *No training.*
2. **Phase B machinery (Codex Goals, on EXISTING data — no downloads):**
   FT2-28 open-state lifecycle rows; FT2-30 harness pilot (build real tensors
   on a small sample, publish measured actuals for the §3-arithmetic
   estimates, run the fp16 equivalence + identity-continuity tests); FT2-31
   independent machinery acceptance. *No training — machinery only.*
3. **Parity gate** — train-vs-live feature parity certification. HARD gate,
   stands independent of all the above; must pass before any real training.
4. **D54 entry-feasibility gate** — simple-model (heuristic/GBT) entry vs P5
   on serial replay with full costs, incl. the realized band-profitability
   report. Pass unlocks neural spend; fail stops it. *This is the last gate
   before neural training.*
5. **>>> STOP LINE <<<** — "ready to begin training." The owner's reserved
   pre-training step happens here. Nothing past this runs without explicit
   owner authorization: no GPU tranche, no neural campaign, no paid data.

Everything through step 4 uses data already owned and requires no purchase.
The first *paid* actions (GPU tranche, any backfill/1s calibration buy) live
past the stop line, behind their own owner gates.

---

## 10. D59 (NEW) — Tick vs 1-second: one representation, derived and certified

**Owner concern (2026-07-30):** training must not mix "tick" and "1-second"
data as if they were two different things.

**Resolution — they are NOT two things.** 1-second data is coarsened tick
data (the last quote in each 1-second interval). So the handling is:

1. **Standardize on ONE canonical sub-minute representation: 1-second.** It is
   the resolution the D55 pilot ran at, the resolution exit/floor labels are
   built at, and finer-than-1s is not tradeable live (reaction + routing
   latency is ~seconds). Sub-second detail would be precision the trader can
   never act on.
2. **Acquire `cmbp-1` (tick) only, and downsample to 1-second in-house** with
   the vendor's own rule (last consolidated BBO per 1-second interval). Raw
   ticks are never a training input; everything becomes uniform 1-second
   first, so no representation-mixing ever enters training.
3. **Certify the downsampler against ground truth.** On the 2025-02-20+ overlap
   we own BOTH the derived 1-second (from tick) and Databento's official
   `cbbo-1s` (the 30-session D55 pilot). Prove the in-house downsampler
   reproduces the official `cbbo-1s` bars exactly (or within a frozen
   tolerance) before trusting the derived 1-second for the 2023-03-28 →
   2025-02-19 period where no official 1-second exists. Verify, don't assume.
4. **Retain the raw tick as a superset** in case finer detail is ever wanted.

**Cost-neutral:** `cmbp-1` (~$0.86/session) ≈ `cbbo-1s` (~$0.91/session), so
buying tick-for-the-whole-window-and-downsampling costs the same as the
messier stitch of official-1s (2025+) onto tick (2023-2025), and is cleaner
and provable. This supersedes any earlier "just buy cbbo-1s" phrasing.

---

## 11. D60 (NEW, owner-requested) — "Replay & Visualize" is a formal process step

Every accepted candidate in the REAL pipeline (not only the walking-skeleton
dry-run) must produce, on its validation replay under the serial simulator, an
owner-facing visual evidence set: the **equity curve**, the **trades plotted
on the SPX price graph**, and the **four-bucket Pickles outcome distribution**.
Rationale: the visual read is decision-relevant owner evidence and the plumbing
already exists (`equity.html` / `trades.csv` from the census sanity path; the
trade-chart exporter). This was implicit before; D60 names it as a required
step. See the Walking-Skeleton plan
(`PROTOCOL101_WALKING_SKELETON_DRYRUN_PLAN_2026_07_30.md`), which exercises it
first as Stage 3.

---

## 12. Owner signature — FT2-21 design approval

By signing, the owner approves the Protocol101 Full Trader Graph V2 design as
sound enough to begin machinery, adopts amendments D54-D60, and authorizes the
Walking-Skeleton dry-run as the reserved pre-training step. This does NOT
authorize training, GPU spend, paid downloads, broker contact, or paper orders
— each remains behind its own later gate. It directs the next governed step:
formally incorporate D54-D60 into the consolidated authority and record the
FT2-21 approval in the Graph V2 state.

Owner signature: **Owen Heidenreich**   Date: **July 30, 2026**
