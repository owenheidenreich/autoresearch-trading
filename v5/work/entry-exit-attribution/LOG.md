# Attribution rounds

> **Goal reset, 2026-08-14.** The owner clarified that the intended four roles were morning entry/exit
> (09:30–12:45) and afternoon entry/exit (12:46–16:00), embedded in a complete causal $10,000 trader—not
> four small time windows and not an entry/exit attribution sweep. The active goal is now
> [`GOAL.md`](GOAL.md), and the active implementation plan is [`PLAN.md`](PLAN.md). The completed rounds
> below remain immutable prior evidence and design input. They are not tests of the corrected
> full-ladder, OTM-to-ITM, 60–120-minute policy.
>
> The corrected run starts a new round sequence below the historical record. Its admissible new
> information is the within-day candle sequence and contemporaneous cross-sectional option-ladder
> surface. Reusing only the old scalar 30-minute population would violate the do-not-retest closure.

### Autonomous loop cycle 1 — V5 attribution and successor design, 2026-08-14

- **Outcome-blind diagnosis:** all 423,053 V5 OOF scores decompose into a minute-common state offset plus
  a contract-only node term. Checkpoint reconstruction error is `2.38e-7`; stored prediction error is
  `7.63e-6`. The strict preregistered all-minute argmax identity missed on 20/47,622 floating-point near
  ties and is not relabelled a pass; source algebra and mutation tests establish the narrower structural
  identity.
- **What V5 learned:** static option geometry (within-minute node correlation: vega +.851, absolute delta
  +.788, moneyness +.718) and a chart/clock-dependent activation level. Its magnitude loss never used
  WAIT or EXIT outputs, so those heads received no gradients.
- **Why activation failed:** three fold cutoffs exceeded every scored minute; all 24 activated sessions
  first crossed at 09:35–09:39. This is calibration/opening-score collapse, not an opening-time edge.
- **One change:** replace magnitude-plus-cutoff with a 48-parameter state×contract action-value form whose
  trained actions are WAIT and the current affordable contracts. Focused tests: 15 passed.
- **Gate result:** refused as designed. The architecture is outside the hashed family and the proposed
  `serial_action_advantage_120m` label is outside the signed magnitude reopening. No protected control
  file was edited.
- **Next:** one narrow owner re-ruling, then one frozen action-value fit. A non-positive spread-free result
  or less than 4/5 chronology closes the long selector and routes to the strategic different-game packet.

### Autonomous loop cycle 2 — executable action-value target, 2026-08-14

- **Frozen question:** for one $10,000 trade, does buying this exact contract at the ask and liquidating
  through the existing 120-minute bid/settlement/fee law beat preserving the slot for every later action?
- **V1 failed safely:** 1,964 of 79,218 decision minutes have no eligible contract row. The builder stopped
  before a diagnostic or output. V2 preserves them as explicit WAIT-only states; no outcome choice moved.
- **Population:** 243 sessions, 79,218 minute values and 698,231 contract values. Receipt and all three
  generated artifact hashes re-verify. No model, selector or reserved session.
- **Scale:** median Q(enter) -$173; median Q(wait) +$1,437; median A(enter) -$1,315. Only 5,026 contract
  actions (0.72%) beat waiting. Every session has a positive hindsight ceiling, median +$2,567.
- **Interpretation:** this is the intended rare-timing problem and an oracle ceiling, not evidence of an
  edge. The target stays frozen despite its sparsity.
- **Next:** finish the no-fit target alignment/loss plumbing and call the unchanged gate. Only a signed
  narrow re-ruling may open the single compact fit.

One entry per round: the cells in the common currency `(n, p, W, L)`, both main
effects, the interaction, and the **one** change made in response. Format is in
[`PLAN.md` §5](PLAN.md#5-phases).

---

### Phase 0 — ceiling map, 2026-08-14. No fitting.

- **Question:** is there headroom in either half worth modelling?
- **Population:** 143,434 candidates, 4,445,404 path rows, 251 quote sessions,
  entry at the ask, exit at the bid, counts matched at 2,247.
- **Result (bid / mid):**

| | net$ | headroom |
|---|---:|---:|
| baseline (random entry, 5m clock) | −19.5 / +2.5 | — |
| entry ceiling (oracle contract, 5m clock) | +240.6 / +283.8 | **+260 / +281** |
| exit ceiling (random entry, oracle exit) | +327.0 / +351.6 | **+346 / +349** |
| joint ceiling | +986.0 / +1,042.1 | **+1,006 / +1,040** |

- **Verdict: GO on both halves.** Neither ceiling is small, at either pricing, so
  the gate does not stop the job. The ceilings are **super-additive** — 260 + 346
  = 606 against a joint 1,006 — so the halves are complementary in principle and
  a negative interaction in practice would be a fixable defect rather than an
  inherent property.
- **Already informative:** the two halves reach their ceilings by different
  mechanisms. Perfect entry works by **collapsing L** (158 → 12) and perfect exit
  by **stretching W** (208 → 482) while only partly trimming L (158 → 94).
- **Receipts:** `v4/audit/autoresearch/factorial_phase0_2026_08_14/receipt_bid.json`,
  `receipt_mid.json`

---

### Round 1 — first full factorial, 2026-08-14

- **Hypothesis:** the fitted halves each contribute, and the combination is worth
  about their sum. Falsified by a large interaction in either direction.
- **The one change:** none — this is the reference configuration.
- **Cells** (151 scored sessions, counts matched at 1,351, entry at ask / exit at bid):

| entry | exit | n | p | W | L | net$ | CI |
|---|---|---:|---:|---:|---:|---:|---|
| random | clock | 1,351 | 36.9% | 207.8 | 158.4 | −23.1 | [−35.8, −9.5] |
| random | model | 1,351 | 36.8% | 225.4 | 181.1 | −31.5 | [−46.5, −16.0] |
| model | clock | 1,351 | 44.0% | 274.9 | 243.2 | −15.0 | [−33.7, +3.3] |
| model | model | 1,351 | 45.1% | 289.6 | 267.9 | −16.6 | [−39.4, +5.2] |

- **entry effect +$8.2 · exit effect −$8.4 · INTERACTION +$6.7**
- **Predicted from parts −$27.3, measured −$16.6.** Both decompositions agree:
  the halves are mildly **complementary**, not fighting.
- **What each half moved:**

| | p | W | L |
|---|---:|---:|---:|
| entry | **+7.1pp** | +67.1 | **+84.7** |
| exit | −0.1pp | +17.6 | **+22.6** |
| combined | +8.1pp | +81.7 | +109.5 |

- **Diagnosis, which is the point of the exercise:**
  - **The entry has real skill and a named defect.** It lifts the hit rate 7.1
    points — the largest honest entry effect this project has measured — but
    enlarges the average loser by $84.7, which eats almost all of it. It is
    selecting trades that win more often *and lose bigger*.
  - **The exit has no skill and a named defect.** It moves `p` by −0.1pp, as an
    exit must, but raises `W` by $17.6 and `L` by $22.6 — it stretches both tails
    because it simply holds longer (9.5 min against 5). It has no trimming
    ability at all, and the oracle exit shows trimming is where the value is
    (L 158 → 94).
- **Verdict:** neither half is repaired by touching the other. Two independent
  repairs are indicated.
- **Next change (one only):** the entry's label. It currently predicts *whether*
  a trade clears its round trip, which is indifferent to how badly the losers
  lose; a label carrying loss magnitude is the direct attack on the +$84.7.
- **Receipt:** `v4/audit/autoresearch/factorial_round1_2026_08_14/receipt_bid.json`

---

### Round 2 — the two "magic times", preregistered, 2026-08-14

Owner hypothesis, named **before** any test: decisive moves cluster at **10:00
("magic time")** and **13:30 ("algo")**, and the two windows are different games
because theta and gamma differ. Two named minutes is a preregistered test of two
things, not a search over seventy-two, which is the strongest form this project
has run.

**A blocking defect found first.** Every dataset built before this round started
at **10:35**. `path_features` demanded 61 minutes of prior history for the
60-minute lookback, and the session opens at 09:30, so the first scoreable minute
was 10:30 in principle and 10:35 in practice. **The 09:30-10:30 window
contributed zero candidates to the conditional-drift census, the selective
policy, the exit study and the factorial.** Fixed with adaptive lookbacks that use
whatever history exists and record `history_minutes`; the fixed form is retained
as the default so every earlier table still reproduces. Rebuilt: 707,171 → 851,567
candidates.

**The named minutes, against their own local neighbourhoods** (the day trends, so
the day average is the wrong control):

| | n | p | net$ | CI |
|---|---:|---:|---:|---|
| **10:00 exactly** | 4,361 | 36.7% | **−$36.7** | [−67.4, −3.5] |
| 09:55 + 10:05 | 8,662 | 37.5% | −$27.7 | [−55.8, +2.8] |
| 09:35–10:30 excl. 10:00 | 47,295 | 38.4% | −$25.6 | [−44.3, −3.9] |
| **13:30 exactly** | 2,472 | 35.6% | **−$37.7** | [−58.1, −14.0] |
| 13:25 + 13:35 | 4,916 | 35.1% | −$22.6 | [−48.7, +5.9] |

**Both fail.** Neither minute beats the minutes either side of it; both are
slightly worse. As *unconditional* entry timing, the magic-time claim is dead.

**What the claim actually was, and what this does not test.** "S3 will play a
strong support, long at 1330 algo" is a timing filter *on a setup*, not a
standalone signal. This refutes "buy at 13:30 regardless"; it says nothing about
whether 13:30 is a good moment to act on a level. That is a model question, and
it is the one the factorial can answer.

**The greeks reasoning is confirmed and measured:**

| | premium | theta/price | gamma$ | spread |
|---|---:|---:|---:|---:|
| morning 09:35–10:30 | $1,233 | −0.00209 | 4,521 | $17.2 |
| afternoon 13:00–14:00 | $755 | **−0.00504** | **7,904** | $14.8 |

Afternoon decay runs **2.4x** faster and carries **1.75x** the gamma per dollar
on a contract costing **40% less**. The two windows are genuinely different
instruments, exactly as claimed.

**And the windows differ in how far they are from paying:**

| window | p | break-even p | gap |
|---|---:|---:|---:|
| morning 09:35–10:30 | 38.3% | 41.0% | **+2.7pp** |
| midday 10:35–12:55 | 37.7% | 40.6% | +2.9pp |
| afternoon 13:00–14:00 | 34.2% | 39.4% | **+5.2pp** |
| late 14:05–15:30 | 32.4% | 35.6% | +3.2pp |

**Verdict:** the magic *minutes* are noise; the magic *windows* are real as
structure. Morning needs 2.7 points of hit rate to break even, afternoon needs
5.2 — nearly double the ask. That is the evidence for splitting models by window,
and it says the morning is where to spend the effort.

**Next change (one only):** run the factorial separately on the morning and
afternoon windows and compare the `(p, W, L)` signatures. If the entry effect
lives in a different component in each, two models are justified; if not, one
model with time-of-day as a feature keeps all 251 sessions and is strictly better.

---

### Round 3 — the exit is conditionally powerful and unconditionally zero, 2026-08-14

Owner correction: a 30-minute label truncates a 60-minute move; averaging every
candidate at 10:00 hides the sessions where a move actually began; and an exit
holding 7.9 minutes cannot monetise a trend at all. **The entry should look for
the start of a sustained move and the exit should hold it without leaving early.**

**A trailing-stop family was added** — exit when the price falls a declared
fraction from its own running peak. This is the first exit family in the project
that *can* ride rather than only leave early, and it is declared, not fitted.
(Distinct from the closed −30% stop, which measured from the entry and fired
inside normal noise; a trail gives back a fraction of a gain instead.)

**On a random entry the trails lose**, at both price sources: at the mid, hold-5m
−$0.2 against trail-25% −$17.8, trail-35% −$14.9. Paired, the fitted rule beats
trail-25% by +$14.8 [+0.1, +28.7].

**That does not refute the thesis, and the reason is a defect in this packet.**
[`PLAN.md` §5 Phase 3](PLAN.md#5-phases) fits the exit on *random* entries to
isolate it. That is correct only for exits whose value is entry-**independent**.
A trailing stop's entire value is conditional on the entry having caught a move,
so the exit-alone arm structurally understates it. **Phase 3 must gain an arm
that scores each exit against the oracle entry as well as the random one.**

**Measured conditionally** (grouping on realised excursion is selection on
outcome — a diagnostic ceiling, not a policy):

| population | share | hold 5m | hold 30m | hold 60m | trail 25% | oracle |
|---|---:|---:|---:|---:|---:|---:|
| MFE < 10% (no move) | 26.6% | **−214** | −476 | −598 | −329 | −11 |
| MFE 10–25% | 17.5% | −59 | −300 | −421 | −212 | +191 |
| MFE 25–50% | 19.6% | +31 | −100 | −208 | −108 | +411 |
| **MFE > 50% (big move)** | **36.7%** | +126 | +484 | **+675** | +313 | +1295 |

**The exit's correct behaviour is entirely conditional.** Hold long when a move
materialises (+$675 at 60 minutes), cut fast when it does not (−$214 at 5 minutes
against −$598 at 60). Averaged over a random entry these cancel almost exactly —
which is why every exit study this project has run measured about zero. **Every
one of them was unconditional, and unconditional is the single condition under
which this effect disappears.**

**The trap check passes.** Big movers are not cheap contracts: mean premium
$1,096 against $1,103 for the rest, medians both $1,000. The percentage-excursion
failure mode does not apply.

**What the entry would have to deliver:**

| exit | big movers | rest | base rate | break-even precision | gap |
|---|---:|---:|---:|---:|---:|
| hold 30m | +$484 | −$313 | 36.7% | 39.3% | **+2.6pp** |
| hold 60m | +$675 | −$430 | 36.7% | 38.9% | **+2.2pp** |

An entry that identifies "this becomes a big mover" at **38.9%** precision — 2.2
points over the base rate — makes a sixty-minute hold break even. Round 1's entry
model demonstrated **+7.1pp** of hit rate on a different label, roughly three
times the required margin.

**Verdict:** the thesis is supported and quantified. The exit is not null; it is
conditional, and it has been measured in the one regime where conditionality
cancels.

**Next change (one only):** relabel the entry to predict **MFE above a
threshold** rather than profit at 30 minutes, and score it through a **long**
clock. Design note for the packet: *the fixed exit an entry is scored through
must match the thesis under test* — a 5-minute clock asks "find trades that pop",
a 60-minute clock asks "find trades that trend". Same model, different question.

---

### Corrected goal, Phase A — coverage and the inclusion bug, 2026-08-14. No fitting.

- **Goal:** causal per-day trader with the whole developing candle/ladder state, morning and afternoon
  entry/exit roles, OTM-to-10/20/30-ITM labels over 60/90/120 minutes, and a $10,000 replay.
- **Clock correction:** `t` is an Eastern minute boundary. State sees the completed ES bar `[t-1,t)` and
  ladder snapshot `t`, then acts at the displayed `t` touch. The four-state router is explicit and the
  opening regime owns its exit across 12:46.
- **Attempt 001 defect:** excluded 38 sessions if 09:35 had no affordable contract or ES missed any minute.
  No-contract days are required abstention evidence, so that was a post-entry inclusion error.
- **Attempt 002:** 251 quote sessions inventoried; **243 complete primary episodes**. Eight exclusions are
  four empty ES files, two material full-day quote gaps and two early-close/partial clocks. Thirty included
  days have no eligible 09:35 contract and remain episodes.
- **Coverage:** 47,707,186 quote rows; median eligible contracts 8 at 10:00 and 10 at 13:30. Historical
  vendor greeks absent; recomputation is required. Quote age is zero everywhere and not treated as observed
  arrival latency.
- **Receipts:** `causal_day_trader_coverage_2026_08_14_attempt002/receipt.json`; attempt 001 is preserved and
  named as superseded.

### Corrected goal, Phase B — full causal episode/ladder dataset, 2026-08-14. No fitting.

- Family frozen before outcomes: `DECLARATION.json`, SHA-256
  `6414759333b11d4054eb3cfb1de4e09f990db09951895d80cb64575faf8f629a`.
- Built **94,770 candles, 93,798 minute states, 1,880,427 actual live ladder rows, 698,231 affordable OTM
  candidate contracts and 79,218 all-minute atlas rows** over 243 sessions.
- Candidate-level 30-ITM prevalence: **1.57% / 2.93% / 4.21%** at 60/90/120 minutes. Every failure and every
  blocked later bid remains in the table.
- Receipt: `causal_day_dataset_2026_08_14/receipt.json`.

### Corrected goal, Phase C — every-day key-time atlas, 2026-08-14. No fitting.

- 10:00: any eligible contract reaches 30 ITM by 60m **9.88%**, local 09:50–10:10 rate **8.99%**;
  corrected difference +0.88pp [−2.08,+4.34]. By 120m: 19.75% vs 20.35%, −0.60pp
  [−4.51,+3.74].
- 13:30: 30 ITM by 60m **5.76%**, local 13:20–13:40 rate **5.10%**; +0.66pp
  [−2.09,+3.61]. By 120m: 16.87% vs 16.28%, +0.60pp [−2.84,+4.14].
- **Verdict:** the paths exist; neither exact minute exceeds its local structure. The clocks are diagnostic
  areas, never entry triggers.
- A surprising 10:00 fixed-clock mean was audited over all six named-time x clock cells. Including no-trade
  days, 10:00 is +$29.94/+55.52/+75.70 per session at 60/90/120, but every corrected interval includes
  zero; 13:30 is negative except a +$3.43 120m point estimate whose zero-recovery sensitivity is −$24.23.
  The 10:00 sign is put-driven, not a cheap-contract or missing-exit artifact. It is a random-contract
  diagnostic, not a causal selector or $10,000 policy.
- Receipts: `causal_day_atlas_2026_08_14/receipt.json` and
  `causal_day_atlas_profit_audit_2026_08_14/receipt.json`.

### Corrected goal, Phase D — simulator built; fitting refused, 2026-08-14.

- Deterministic simulator: one position, one exact contract, ask-in/bid-out, fee once, 1/2/3 caps,
  origin-owned exits, first-later-bid pending liquidation, 120-minute maximum and separately labelled
  cash-settlement/zero-recovery terminal paths. Ten known-answer tests and 18 real no-trade smoke cells.
- Terminal source is unresolved. The 16:00 underlying snapshot exists but is not certified as official
  settlement, so 16,485/71,136/124,002 candidate-clock rows at 60/90/120 remain blocked rather than dropped.
- **No model was fit.** `STATUS.md` still blocks option training until G1 passes; the later quote-policy
  ledger row closes another long-side selective fit; neural comparison additionally needs 1,140 sessions
  and 20 sessions per parameter against 243 here. A fail-closed policy gate records all three.
- Receipts: `causal_day_simulator_2026_08_14/receipt.json` and
  `causal_day_model_block_2026_08_14/receipt.json`.

### Corrected goal, Phase E — terminal settlement resolved, 2026-08-14. No fitting.

- Audited the separately owned `raw/index/spx_1m/{session}.official_spx.parquet` source on every included
  session. All 243 files are official, non-derived, non-proxy SPX; their final 16:00 timestamp equals the
  normalized same-day SPXW PM settlement timestamp, and their close exactly equals the aligned terminal
  underlying. Zero mismatches.
- Rebuilt dataset and atlas as v2 with terminal intrinsic accounting. A no-bid-at-close row is labelled
  `validated_cash_settlement`, never an executable bid; its bid field remains missing and a zero-recovery
  sensitivity remains paired. The previously blocked populations—16,485/71,136/124,002 rows at
  60/90/120 minutes—are now all accounted for without dropping one.
- The corrected 13:30 fixed-clock means are −$34.77/−$27.52/−$24.23 at 60/90/120 minutes. The old +$3.43
  120-minute diagnostic had excluded 101 terminal no-bid contracts. All six named-time corrected
  intervals still contain zero. The 10:00 values are unchanged.
- Receipts: `causal_day_terminal_settlement_2026_08_14/receipt.json`,
  `causal_day_dataset_settlement_validated_2026_08_14/receipt.json`,
  `causal_day_atlas_settlement_validated_2026_08_14/receipt.json` and
  `causal_day_atlas_profit_settlement_validated_2026_08_14/receipt.json`.

### Corrected goal, Phase F — unfitted policy interfaces and exact causal bridge, 2026-08-14.

- Implemented one common PyTorch forward contract for five declared, **unfitted** alternatives: shallow
  joint (500 parameters), shallow shared four-head (544), neural joint (1,076), neural shared four-head
  (1,120) and conditional four-independent specialists (4,216).
- The exact tensorizer sees at most 120 completed candles, the contemporaneous bounded ladder, account,
  position and clock state, and the explicit routed role. Future candles and later ladder snapshots cannot
  change the current tensor; missing values require a future fold-fitted imputation rather than being
  silently filled from the full corpus.
- The shared neural design would require 22,400 sessions under the frozen 20-sessions-per-parameter rule;
  the independent design would require 84,320. The owned foundation has 243. No weights, thresholds or
  economics were produced.
- Receipt: `causal_day_architecture_interfaces_2026_08_14_attempt002/receipt.json`. Attempt 001 is preserved
  and superseded because it did not bind the tensorizer source and exact feature contract.

### Corrected goal, Phase G — replay/accounting plumbing and final simulator receipt, 2026-08-14. No fitting.

- The policy now sees only the live bounded entry ladder. A held contract remains separately visible after
  moving deep ITM/outside the entry band. Stateful policies must implement and receive an episode reset.
- Replay capture records every considered contract, selection flag, optional probability, action, reason,
  diagnostics, fill, spread, fee and account state. The fixed report contract covers abstention, trade
  frequency, premium risk, capital use, MFE/MAE, holding time, payoff, drawdown and $10,000-account return;
  planned loss remains explicitly unknown unless a policy declares it.
- Four examples selected only by first/middle/last/no-candidate session structure exercise the export. The
  fixed known-answer policy produced −$293.08 (2025-08-01), −$938.08 (2026-01-20), +$1,136.92
  (2026-07-30) and no trade (2025-10-14). These are implementation replays, **not** model-skill evidence
  and not the required out-of-fold model win/loss/abstention replays.
- Receipts: `causal_day_simulator_2026_08_14_attempt003/receipt.json` and
  `causal_day_replay_plumbing_2026_08_14_attempt003/receipt.json`. Earlier attempts remain immutable.

### Completion boundary

All non-fit deliverables that can be completed under current governance are present. The policy comparison,
out-of-fold model replays, complete learned-policy economics and empirical one-versus-four conclusion still
require fitting and remain refused by the independent G1 and later do-not-retest gates; neural candidates
also fail the frozen absolute-session and sessions-per-parameter requirements. The project must describe
that state as **foundation complete, fit-dependent answer blocked**, not profitable, complete or promoted.

### Completion-audit correction — whole day and whole chain, 2026-08-14. No fitting.

- The first completion claim was too broad. Two tested interfaces were internally consistent but narrower
  than the activated goal: the tensorizer kept only the latest 120 candles, which discarded the open by
  afternoon, and the simulator showed the policy only the ±25-point ladder used by the compact atlas.
- Corrected before any policy fit in `DECLARATION_V2.json`, SHA-256
  `811cb21d2c5bab3c2f18b58833d7e9043ed1621d0c1baf722ab8bf199ca516b4`. V1 remains immutable evidence;
  V2 is an explicitly outcome-neutral scope repair, not a post-fit retry.
- The policy observation now retains every completed candle from 09:30—5 at 09:35, 30 at 10:00, 240 at
  13:30 and 330 at 15:00—and every contemporaneous live two-sided chain row. Entry actions remain
  separately masked to the declared affordable `[-25, 0)` OTM band.
- Real-data proof on first/middle/last sessions across those four minutes: **229–365 full live contracts**
  per snapshot, of which **209–345** lie outside the entry band and remain context. The observation count
  equals the source live-two-sided count in all 12 cells; calls and puts are both present. Missing volume
  and unsolved greeks use value-plus-observed-flag channels instead of making the actual chain impossible
  to tensorize.
- Architecture dimensions are now 18 candle and 23 ladder channels. Parameter counts are shallow joint
  676, shallow four-head 720, neural joint 1,252, neural four-head 1,296 and independent 4,920. The shared
  neural form would require 25,920 sessions under the frozen ratio, against 243.
- Receipts: `causal_day_observation_contract_2026_08_14/receipt.json`,
  `causal_day_architecture_interfaces_2026_08_14_attempt003/receipt.json` and
  `causal_day_model_block_2026_08_14_attempt003/receipt.json`.

### Completion-audit correction — complete trade-path reporting, 2026-08-14. No fitting.

- The simulator trade ledger now records entry OTM depth, maximum/final ITM depth, OTM-to-ITM conversion,
  first crossing/time-to-cross, signed underlying MFE/MAE and final underlying state in addition to option
  MFE/MAE, fills, spread, fees and account state.
- The reporting contract now includes every corresponding aggregate plus a deterministic session-unit
  bootstrap interval corrected for the supplied declared family size.
- Full-chain known-answer replays capture **108,252–126,831 considered contract-minutes per day**, replacing
  the 7,691–7,701 near-band rows in attempt 003. Their P&L is unchanged because the fixed action law is
  unchanged; they remain implementation examples, not model evidence.
- Receipts: `causal_day_simulator_2026_08_14_attempt005/receipt.json` and
  `causal_day_replay_plumbing_2026_08_14_attempt005/receipt.json`. Attempts 004 and earlier are preserved
  and superseded for the final observation/report contract.

### Revised completion boundary

The non-fit foundation now matches the full activated observation and reporting scope. The overall goal is
still **not complete**: no chronological policy was fit, so architecture comparison, matched controls,
model-selected win/loss/abstention replays, behavioral attribution, complete family economics and the
one-versus-four answer do not exist. The same G1 and later do-not-retest prohibitions remain the limiting
condition; neural sample rules add a third block. No model result may be inferred from the model-free atlas
or known-answer fixtures.

- Requirement-by-requirement audit: D1 coverage, D2 simulator, D4 model-free atlas and D7 durable records
  are proven; D3 architecture comparison, D5 model replays, D6 policy economics and V1–V3 fitted validation
  remain unachieved. Receipt: `causal_day_completion_audit_2026_08_14_attempt002/receipt.json`. Attempt 001
  is preserved and superseded because indexing its own path changed a record hash after issuance.

### Fit continuation — evidence re-ruling and width-3 preflight, 2026-08-14. No fit yet.

- The owner signed `CAUSAL_DAY_FIT_RERULING_2026_08_14.md`, lifting the suspension for the exact
  `itm_depth_magnitude` label at 60/90/120 minutes on `causal_day_quote_243` and no wider.
- Effective information is **2,879–7,557 observations** by integrated autocorrelation and **590–1,016**
  by design effect. The generous fit budget is **377 parameters**; the conservative design-effect budget
  is **29–50** and admits none of the declared family.
- `causal_day_hidden_size` is frozen at **3**. `DECLARATION_V3.json` remains immutable width-8 history and
  is not executable under the new budget; it will be superseded by generated `DECLARATION_V4.json`.
- Preflight built the models through `computed_parameter_counts()` and obtained 226 / 245 / 322 / 341 for
  shallow joint / shallow four-head / neural joint / neural four-head. `load_reopening()` plus
  `assert_fit_permitted()` returned **PERMITTED** for all four architectures at all three horizons with
  all seven kill conditions. No gate, knob, status or ledger file was changed for this preflight.
- The decisive rule remains first: if gross mid-to-mid P&L per trade is **≤ 0**, the reopening is spent
  as a negative result. No re-labelling, operating-point search, exit fit or retry follows.

### Fit continuation — V4 conclusion void because of a selector specification defect, 2026-08-14.

- Generated `DECLARATION_V4.json` from the actual canonical width-3 models. Counts are 226 / 245 / 322 /
  341; V3 remains immutable width-8 history. The generous budget is 377 parameters and the conservative
  design-effect budget remains 29–50, admitting none of the family.
- The pre-economics audit passed all 61 feature rows, the future-mutation probe, the no-post-entry-filter
  and reservation assertions, the whole-chain equality proof, and every immutable cache file: 243
  sessions, 79,218 entry minutes, 26,808,322 whole-chain nodes and 698,231 eligible actions.
- Completed all **24** declared chronological fits: four architectures × three horizons × real/shuffled.
  Every member produced 423,053 out-of-fold contract predictions over the same 150 scored sessions. The
  fit receipt records `economics_read=false` and `threshold_tuned=false`.
- The V4 selector was defective by construction. The label was clipped at 30 points and normalized by
  30, while activation required a prediction >=30 points, exactly the target-support ceiling. A
  smooth-L1 conditional-mean regression can learn the ranking while remaining below that boundary; the
  threshold therefore required a degenerate ceiling case or overshoot rather than a high rank.
- Across 423,053 predictions the maximum score was **29.8785** and zero rows reached 30. Consequently no
  trade was selected and **no mid-to-mid economic result was measured**. The prior negative interpretation
  is void; this is not an operating-point retry because the defect follows from the declaration's label
  arithmetic without reading P&L.
- Status: `INCONCLUSIVE_SPECIFICATION_DEFECT`. The 24 fits, predictions and feature audit remain valid;
  only the selector and its economic interpretation are superseded. The reopening is not spent and no
  negative Job 39 result belongs in the do-not-retest ledger.
- V5 will freeze a causal rank selector from strictly earlier fold-training sessions, retain the same
  architecture, width, label, horizon, folds, trade cap and seven kill conditions, and add a pre-run gate
  refusing any absolute threshold at or above the label clip ceiling. Kill #1 binds only after that
  corrected selector actually measures gross mid-to-mid P&L.
- Receipts: `causal_day_fit_feature_audit_2026_08_14_attempt001/receipt.json`,
  `causal_day_magnitude_fit_2026_08_14_attempt001/receipt.json`, and
  immutable-but-void `causal_day_magnitude_primary_economics_2026_08_14_attempt001/receipt.json`.

### Fit continuation — corrected causal rank policy fails chronological stability, 2026-08-14.

- Issued an immutable interpretation correction for V4. Status is
  `INCONCLUSIVE_SPECIFICATION_DEFECT`; its negative conclusion is void, the 24 fits/predictions remain
  valid, and the reopening was not spent by the defective zero-trade selector.
- Generated self-hashed `DECLARATION_V5.json`. It reuses the unchanged width-3
  `neural_four_head`/120-minute fit at its computed 341 parameters, retains all seven kill conditions and
  the original folds/cap/risk law, and records `load_reopening()` + `assert_fit_permitted()` as
  `PERMITTED`. The owner-controlled policy-fit gate was not edited.
- Added a selector-specific pre-run gate that refuses every absolute prediction threshold at or above
  the label clip ceiling. V5 instead targets two high-score minutes per training session, freezes the
  resulting fold cutoff from strictly earlier sessions, and walks each score day causally. Whole-day
  top-N on a scored session is forbidden.
- The pre-economics ranking audit reproduced real learning over 423,053 OOF rows: correlation **+0.3300**;
  all ten actual-depth deciles monotone from **−10.28 to +6.50 points**; predictions >=25 at **14,103 real
  versus 0 shuffled**; score SD **11.44 versus 5.11**. The diagnostic read no P&L or bid price.
- The corrected selector took **24 trades on 24 days**. Kill #1 passed at **+$137.81 mean gross
  mid-to-mid per trade** (median −$117.50). The exact regime/side/delta/premium-matched control was
  **−$133.23** over 24 complete matches; the separately calibrated shuffled policy was **+$68.51** over
  37 trades. Both pooled point comparisons pass.
- Chronology fails: 22 real trades came from fold 2, two from fold 3, and folds 1/4/5 abstained. Absolute
  gross is positive in **1/5 folds**; both paired deltas are positive in **2/5**, below 4/5. Retaining the
  declared 648-member family, corrected one-sided session-block lower bounds are **−$108.75** absolute,
  **−$44.78** versus matched and **−$172.59** versus shuffled.
- Status: `NEGATIVE_CONTROL_OR_CHRONOLOGY_KILL_FAILED`. This is the corrected V5 result, not a revival of
  the void V4 conclusion. Bid economics and exit fitting remain unread; no alternate rank target, cutoff,
  seed, architecture or horizon follows. Generous budget 377; conservative budget 29–50 admits none.
- Receipts: `causal_day_magnitude_v4_specification_correction_2026_08_14_attempt001/receipt.json`,
  `causal_day_rank_selector_calibration_2026_08_14_attempt001/receipt.json`, and
  `causal_day_magnitude_corrected_primary_economics_2026_08_14_attempt001/receipt.json`.

### Goal reset — autonomous breakthrough loop, 2026-08-14.

- The owner replaced the one-shot Job 39 prompt with a persistent research objective: choose and execute
  the highest-information authorized project action, record it, and continue without using the owner as
  an agent/task relay or routine method approver.
- Significance is an exit criterion, not a search instruction. Every cycle predeclares its family and
  controls; failed branches close instead of spawning adjacent threshold, seed or architecture retries.
- The first cycle is outcome-blind attribution of V5's fold concentration and 09:35 entry collapse. Its
  purpose is to choose among a conservative-budget compact sequential policy, a quantified independent-
  data route or a decision-grade different-trading-game pivot. It cannot rescue V5 economically.
- Local analysis, code, tests, simulation and applicable-gate-permitted fitting on owned pre-2026-08-06
  data are authorized. Spending, external contact, reserved evidence, broker/paper/live actions, runtime
  mutation, promotion and gate loosening remain owner-only.

### Breakthrough loop cycle 1 — compact sequential target ready; fit refused, 2026-08-14.

- Outcome-blind attribution proved V5 was an additive option-geometry magnitude sensor, not a trader:
  chart state added the same offset to every contract, WAIT and EXIT received no training gradient, and
  all activations first crossed at 09:35--09:39.
- Implemented a 48-parameter compact successor with state×side, state×moneyness and an explicitly trained
  WAIT action. This fits the conservative 29--50 parameter evidence budget.
- Built the frozen one-trade `serial_action_advantage_120m` surface on all 243 sessions: 79,218 decision
  minutes, 698,231 exact contract actions and 1,964 explicit WAIT-only minutes. Only 0.72% of actions beat
  preserving the slot for the best later action, making selectivity part of the target rather than a
  post-fit threshold.
- Built the identical real/shuffled fit path and issued self-hashed
  `ACTION_VALUE_FIT_DECLARATION_V1.json`. The unchanged gate refused the unregistered
  `compact_interaction_entry` architecture before target-session loading, optimizer construction or any
  output. No fit or economic result exists.

### Breakthrough loop cycle 2 — one defined-risk translation fails, 2026-08-14.

- Preregistered one member, not a grid: a 15:00 five-point ATM iron fly, exit at the first complete
  four-leg touch at/after 15:15 or validated settlement, four measured fees, one trade/session and
  entry-defined loss no greater than $500. The 33-member correction retains the 32 prior two-sided cells.
- All 243 sessions traded. Mean gross midpoint P&L was **+$1.19/session**, but aggressive four-leg touch
  plus fees was **-$60.57**; corrected lower bound **-$64.66**; winning sessions **0/243**; positive
  chronological blocks **0/5**. Worst realised loss was -$127.32 and largest declared maximum loss
  $227.32, so account risk passed while economics failed.
- Hashes reverified. A post-run audit voided only the reported long-straddle secondary control because its
  algebra did not use the actual long ask/bid path. The primary iron fly and naked-short control are not
  affected; the primary independently fails every required criterion.
- The declaration forbids a nearby width/time/horizon retry. The exact defined-risk structure closes.
  The remaining highest-information action is the already frozen 48-parameter action-value fit, requiring
  one narrow owner-controlled gate re-ruling and no other routine choice.
- Final verification after the cycle: **787 v5 tests passed**; `v5/ops/check_project.py` reports both the
  project and repository structure healthy.

### Breakthrough loop cycle 3 — complete inference/economics boundary sealed, 2026-08-14. No fit.

- A pre-fit audit caught one structural inference ambiguity. The learned WAIT estimate could be negative
  even though doing nothing is always worth $0, which could force a negative contract. Inference now uses
  `max(0, predicted Q(wait))` and enters only when a current contract strictly exceeds it. This is a
  feasibility floor, not a tuned threshold, and adds no parameter.
- Implemented and tested the causal one-trade walk, deterministic outcome-independent tie-break, explicit
  abstention, and a composition control matching exact session/regime/side before nearest minute, delta
  and premium. Outcome mutation cannot change the match.
- Implemented the complete outcome firewall: all-session gross midpoint primary first; only if positive,
  open bid-net economics, the matched control and identically fitted shuffled policy. The final policy must
  clear corrected 649-family lower bounds and 4/5 fold signs for absolute bid net and both paired deltas.
- Added an explicit $10,000 risk kill: with no proven early stop, both selected premium plus fee and worst
  realised loss must remain within $500. Affordability alone is not called healthy risk.
- Issued `ACTION_VALUE_EVALUATION_DECLARATION_V1.json`, self-hash
  `33ef2e0d96d73370ab529f91fd3b3fb47ad7f9a9208fc4a12920c5c3e0d08bef`.
- Invoking the evaluator still fails at the unchanged owner gate on
  `compact_interaction_entry`, before the nonexistent fit receipt or any prediction/outcome is opened.
  Fit and evaluation output/evidence paths are all absent. The exact 48-parameter experiment is now ready
  end to end; the narrow architecture/label re-ruling is the only remaining in-scope condition.
- Final alternative audit: 189 owned pre-cutoff sub-minute sessions are Tier-S held-position exit
  trajectories and cannot answer entry; no macro-event corpus or longer-tenor ladder is registered; ES is
  closed as underpowered and needs roughly 988 sessions even for a four-point detection floor; the exact
  defined-risk translation is closed by its result. No other safe owned-data action can supersede the fit.
- Final boundary verification: both fit/evaluation declaration self-hashes and all implementation hashes
  reproduce; **797 v5 tests passed**; the project/repository checker is green; all four declared fit and
  economics output/evidence paths remain absent.

### Breakthrough loop cycle 3 specification correction — V2 packet signable, 2026-08-14. No fit.

- Registered `compact_interaction_entry` in the canonical architecture builder and gate family. V2 now
  obtains **48** only from `computed_parameter_counts()['compact_interaction_entry']`; tests require the
  canonical and direct built-model counts to agree. This is within the conservative 29–50 range.
- Added a reusable selector-attainability guard to the fit gate. Absolute thresholds at or above a clipped
  support ceiling are refused; relative selectors need a finite built-model firing witness. The action-
  value target and outputs are unclipped, and a canonical witness produces ENTER $1,000 versus WAIT $0.
- Superseded the non-signable V1 artifacts with self-hashed `ACTION_VALUE_FIT_DECLARATION_V2.json` and
  `ACTION_VALUE_EVALUATION_DECLARATION_V2.json`. Both carry the canonical family counts and the exact
  attainability proof. V1 remains on disk as history.
- Settled reopening accounting: defective V4 did not spend Job 39; corrected V5 did, when its measured
  result failed the control/chronology kill. `serial_action_advantage_120m` is therefore a new signed
  scope widening, not an unspent-label continuation.
- Both V2 commands passed declaration/count/attainability preflight and were refused by the current signed
  scope before fit or economics. All V2 fit/economic output and evidence paths remain absent. The exact
  proposed signature tuple is `COMPACT_INTERACTION_SUCCESSOR_V2.md`.

### Scope signed; activation-cycle defect caught before fit, 2026-08-14. No fit.

- Owen Heidenreich signed `CAUSAL_DAY_ACTION_VALUE_SCOPE_RERULING_2026_08_14.md` on 2026-08-14 for the
  exact V2 declaration hashes. The signed bytes are preserved unchanged.
- A post-sign activation audit found a cryptographic dependency cycle: V2 hashes the gate implementation;
  that gate still recognizes only `itm_depth_magnitude`; changing it to recognize the signed action-value
  label invalidates V2 before authorization is checked. Regenerated declarations would not match the two
  exact hashes the signed ruling names.
- No gate was bypassed, no signed document was edited after signature, and no fit/economics path was
  created. `SCOPE_ACTIVATION_DEFECT_V1.md` proves all three execution routes and freezes the smallest
  correction: an owner-authorized mechanical V3 reseal with an executable zero-semantic-diff assertion.
- Implemented that assertion before seeking the correction. The complete V2 research-law projection is
  pinned at `3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627`.
  Nine tests prove mechanical seal/path changes pass, while mutations to the seed, horizon, selector
  floor, trade cap, multiplicity family or $10,000 risk limit fail; stale fit/evaluation linkage also
  fails. No candidate V3 exists and the current owner boundary is unchanged.

### Breakthrough loop cycle 4 — compact action-value policy fails spread-free, 2026-08-14.

- The owner authorized the mechanical-only activation needed to execute the already signed scope. Issued
  `CAUSAL_DAY_ACTION_VALUE_SCOPE_ACTIVATION_2026_08_14.md`, registered only the signed architecture/label/
  horizon/corpus tuple, and generated V3 declarations. The executable reseal proves zero differences from
  the V2 research-law projection at
  `3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627`.
- The exact 48-parameter `compact_interaction_entry` real fit and identical shuffled-label null completed
  over 423,053 out-of-fold candidate rows. The fit receipt records no economics read and no operating-point
  tuning. Its self-hash and all six referenced artifacts reproduce.
- The frozen causal selector took 35 trades across 150 scored sessions and abstained on 115. Midpoint-only
  gross was **-$15.93/trade**, **-$3.72/scored session**, median **-$52.50**, win rate **14.29%**, worst
  **-$282.50**. This is a measured negative: the attainable $0 WAIT floor fired, unlike Job 39 V4's void
  ceiling threshold.
- Kill condition 1 failed. The evaluator correctly left executable bid economics, composition-matched and
  shuffled-policy controls, corrected confidence bounds, 4/5 chronology and account-risk results unread.
  The receipt records `bid_economics_read=false`, no reserved sessions and no promotion/order.
- Branch consequence: the declared long-0DTE selector is closed without a neighboring threshold, seed,
  horizon, architecture or subgroup retry. The next active action is the strategic different-game packet,
  because this policy loses even after removing the spread.

### Breakthrough loop cycle 5 — strategic breakthrough selects MES, 2026-08-14.

- Compared every remaining game against the binding mechanism. Longer-tenor and Micro-E-mini options keep
  option premium and implied-volatility pricing; short premium retains the measured catastrophic tail;
  the tested four-leg defined-risk translation is closed. One MES future uniquely preserves the desired
  10–30-point S&P move while removing premium, theta, expiry, Greeks and strike selection.
- Encoded and tested the translation in `mes_strategic_pivot.py` (7 focused tests). CME's $5/point and
  $1.25 tick plus the measured 1.0734-tick elevated-volatility ES spread and published IBKR MES fees imply
  0.508 points/$2.54 round trip. The future research law charges 0.55 points/$2.75 until exact MES BBO
  replaces the explicitly labelled ES proxy.
- One MES with a five-point stop translates a 10/20/30-point target to $47.25/$97.25/$147.25 net against
  -$27.75 at the stop. A stop is not a maximum-loss guarantee, so a future test retains the -$500 observed
  worst-trade kill, one position, at most two trades/day and no overnight holding.
- The existing power receipt requires 3,807/952/423 sessions to resolve a one/two/three-point net edge at
  60 minutes under the simple one-hypothesis calculation. Roughly 1,800–1,830 MES sessions exist from its
  2019-05-06 launch to the cutoff, so this route can test a genuinely large ~2-point edge but cannot prove
  a cost-scale sliver. Controls, correction and 4/5 chronology remain stricter.
- Actual on-disk vendor estimates scale to $150.52 for 1,830 OHLCV-1m plus BBO-1s sessions. The exact owner
  change is therefore finite: a MES research charter and an up-to-$200 data authorization, with read-only
  cost preflight and no broker/paper/live action. No purchase was made.
- Strategic breakthrough route B is satisfied by `MES_STRATEGIC_BREAKTHROUGH_V1.md` and self-hashed receipt
  `mes_strategic_pivot_2026_08_14/receipt.json`. It is a decision-grade game change, not a profitable-policy
  claim.

### Owner correction — MES rejected; SPXW/SPX only, 2026-08-14.

- Owen Heidenreich instructed: **“not approved. SPXW and SPX only.”** The MES packet and receipt are
  preserved as rejected historical analysis. They do not satisfy route B and may not be executed.
- Updated the active goal boundary: no futures, MES, ES, ETFs or other traded instrument. Pre-existing
  evidence may mention ES, but active policy inputs must derive only from SPXW or SPX.
- Read-only source audit on 2025-08-01, 2025-12-08 and 2026-07-31 found 39–41 expirations in each daily
  definition file, but every one of 500/442/958 quoted contract IDs mapped only to that session's expiry.
  The owned quote corpus is therefore 0DTE-only; definitions do not substitute for longer-tenor prices.
- The active next action is an outcome-blind two-leg SPXW 0DTE structure feasibility audit. If no
  cost/payoff geometry can clear before outcomes, the remaining in-scope route is a quantified request for
  longer-tenor SPXW/SPX quotes, not another model retry.

### Breakthrough loop cycle 6 — SPXW-only route audit, 2026-08-14.

- Tightened the owner boundary to SPXW/SPX inputs only. ES may remain in historical evidence but cannot
  enter the active policy as a feature or traded route.
- Audited all 251 owned source CBBO sessions against their definitions. Every quoted instrument expires on
  the same session; zero future-expiry contracts have prices in the corpus.
- Audited the compact policy's frozen 35 entries without reading outcomes. Exact 5–20 point debit-vertical
  pairs exist for only 8–15 entries and only folds 1–2; 25/30-point pairs do not exist. Four-of-five
  chronology is impossible for every width, so no vertical P&L was opened.
- Decomposed the already-spent iron fly at identical frozen entries and exits. Call/put credit-spread halves
  lose **-$31.02/-$29.56 per session** at touch with **0/5 and 1/5** positive folds. Midpoint values are
  -$0.75/+1.94 and unstable at 3/5 and 2/5. The split reproduces the primary to below `5e-13` dollars.
- Estimated the smallest genuinely new in-scope corpus from recorded costs only: 1,045 sessions of
  next-listed-expiry SPXW definitions plus CBBO-1m, **$59.81** expected and **$100** proposed hard cap.
  No vendor was contacted and no data was downloaded. The exact owner request is in
  `SPXW_NEXT_EXPIRY_ROUTE_V1.md`.

### Owner correction — long single-leg SPXW 0DTE only, 2026-08-14.

- Owen Heidenreich instructed: **“its a long call or long put SPXW model only.”** This rejects every
  futures, longer-tenor, short-premium and multileg route. The next-expiry debit-vertical request is
  withdrawn without vendor contact or download.
- Froze the exact action space: WAIT or buy one SPXW 0DTE call/put while flat; HOLD or sell that same long
  contract while holding. SPX may be causal context. No writing, second leg, other expiry or other product.
- Replaced the invalid data request with the same-game unlock. The external 0DTE OHLCV store contains
  **794** non-empty sessions from 2022-06-01 through 2025-07-31 that lack full-ladder CBBO.
- Costs already recorded across 420 sessions estimate the missing same-day definitions at **$23.03** and
  CBBO-1m at **$21.56**, **$44.59 combined**, with a proposed **$75 hard cap** after exact preflight. No
  vendor was contacted and no data was downloaded.

### Breakthrough loop cycle 7 — same-game data decision and evidence-sized lifecycle, 2026-08-14.

- Predeclared and ran a no-fit decision-value projection from existing receipts. Applying the observed
  243/251 completeness rate to 794 older non-empty sessions yields 768 additional complete episodes and
  **1,011 total**; the all-complete ceiling is 1,037. Measured design effects project **2,455–4,228**
  effective observations at observed completeness, or a conservative **122–211 parameter** budget
  (**125–216** if every non-empty day is complete).
- This is about **4.2x** the present evidence but does not admit an existing open shared model: the
  smallest is `shallow_joint` at 226 parameters and requires at least 1,081 complete sessions even under
  the optimistic design-effect rate. The closed 48-parameter action-value model is not retried.
- Built one genuinely new prerequisite rather than buying data speculatively. The unfitted
  `compact_shared_lifecycle` model has a computed **120 parameters**, one shared three-value state,
  whole-ladder mean/max context, state×side and state×moneyness entry terms, trained WAIT, and a shared
  HOLD/SELL head carrying position origin. It fits the worst projected 122-parameter budget with two
  parameters of headroom.
- Nine real feature-only cells—first/middle/last sessions at 10:00, 13:30 and 15:00—preserve hundreds of
  context contracts beyond eligible entry actions, emit finite WAIT/entry/HOLD/SELL scores and are
  invariant to masked future candles/contracts. Cached target arrays and economics were never accessed.
- Froze the post-acquisition law before a vendor call: deterministic 40% training prefix and five
  chronological blocks, maximum two trades/day, nested out-of-fold entry trajectories for exit training,
  frozen entry weights before the exit head fits, identical shuffled path, composition control, $500
  risk kills and midpoint-first evaluation. Real target construction is forbidden until acquisition,
  a fresh declaration and the applicable owner gate all permit it.
- No vendor contact, download, spend, fit, new economics, reserved session, order or promotion occurred.
  The single remaining Tier-1 request is the exact same-day SPXW definition+CBBO preflight and capped
  purchase already specified.

### Owner decision — preflight and acquisition deferred, 2026-08-15.

- The handoff state was independently re-verified before the decision was posed: 839 tests pass, the
  project checker is green, and `computed_parameter_count()` returns 120 from the built module's actual
  trainable weights against the 122-parameter worst-case budget.
- Owen Heidenreich answered **"Not now"** to the single Tier-1 request (exact-cost same-day SPXW
  definition+CBBO preflight and capped acquisition, ~$44.59 estimated, $75 hard cap).
- The packet remains decision-ready and unchanged. No vendor was contacted, nothing was downloaded and
  no money was spent. Every other route stays closed by the ledger and the frozen scope, so no further
  work is startable until the owner either authorizes the acquisition or changes the scope.

### External adversarial review of the frozen lifecycle protocol, 2026-08-15.

- The owner ran two external reviews (ChatGPT Pro max-effort referee on a 10-document packet; a separate
  literature Deep Research). Reports live at `v5/work/entry-exit-attribution/external-review/chatgpt-research/8-15-26/`. Claims were verified
  against the repository before acceptance; nothing below is taken on the reviewer's word.
- **CONFIRMED — the capacity budget is compared against data no fit ever sees.** The 122-parameter
  worst-case budget scales from the full 1,011-session backfill, but the protocol's own chronology trains
  the first outer fit on only the 404-session prefix. Under the project's own linear scaling the per-fit
  conservative budgets are ~49/63/78/93/107 across the five outer fits. Verified split of the built
  module: the entry+shared phase trains **96** parameters (exceeds at least the first three per-fit
  budgets), the exit head **24** (its trajectory-level ESS was never measured). Whole-module (120,
  the project's row-43 convention) exceeds all five. **This changes the pending Tier-1 decision:** the
  $75-capped backfill does not license the built model if per-fit budgets bind, and SPXW daily 0DTE
  history (mid-2022 onward) is too short to ever satisfy them at this size.
- **CONFIRMED — no research-exposure firewall on the score blocks.** Chronological sorting places the
  heavily-reused 251-session quote corpus (2025-08 onward) inside the late score blocks; the protocol
  protects blocks only from entry/exit fitting, not from prior design exposure.
- **CONFIRMED — the serial risk law conflicts with the daily breaker.** Two legal $500 losses make a
  10% day against the unresolved row-21 5% breaker conflict.
- **CONFIRMED — STATUS §2 still carried the falsified "only calendar time" sentence.** Fixed in place
  with a pointer to the 2026-08-14 header correction.
- The referee's 13 amendment drafts (exposure ledger, end-to-end known-answer power gate, per-fit
  capacity gate, quote admissibility law, decision clock, duration-matched exit control, clustered
  session-level inference, dependence-preserving null, separate policy/attribution comparators,
  semantic freeze, divergence-register gate, closure-scope limit, serial daily breaker) are recorded
  for triage; none is adopted yet.
- **Deep Research (literature) verdict:** no published evidence through 2026-08 of a positive-edge
  occupant in exactly this box (retail, long-only, single-leg SPX 0DTE, minute data); documented edges
  are seller-side variance/gamma/jump premia, spread/queue capture and hedged relative value, each
  violating a frozen constraint. Two usable nuggets: (1) the SEC DERA 0DTE study measures ~half the
  quoted-spread cost via 1-second passive-then-cross execution, which conflicts with job 30's measured
  $60-86 adverse selection at the midpoint and may be a resolution/contract-regime difference worth a
  declared re-measurement; (2) the same study documents an endemic OPRA trade/quote sequencing defect,
  independently corroborating this project's print-artifact finding.
- No protocol edit, fit, purchase or vendor contact occurred. Next: owner steer on the capacity-law
  question, then a protocol V2 amendment draft and a local known-answer learning-power campaign.

### Known-answer capacity campaign V1 — SPECIFICATION DEFECT, superseded by V2, 2026-08-15.

- V1 (declaration `d8cd12f1…`) completed all 480 trials and is voided as a capacity measurement by its
  own evidence: medium-edge recovery plateaued at **23%** at both 650 and 890 training sessions, and a
  controlled diagnostic at n=404/medium moved from **0.00 to 36.04 USD/minute** (oracle 78.37) when the
  training budget went from the frozen 60 epochs to 180. The optimization budget binds before sample
  size, so V1 measured the optimizer. Its receipt at
  `v4/audit/autoresearch/capacity_known_answer_2026_08_15/receipt.json` is preserved untouched.
- Two V1 results carry forward as valid: the planted edge is representable by the real architecture
  (hand-built reference weights reach 81–87% of oracle at both effect sizes), and null discipline is
  perfect at n≥404 (clean rate 1.00 in every null cell; 0.88 at n=243).
- V2 (declaration `2cf5c2a1…`) replaces the fixed budget with outcome-blind convergence training
  (plateau of epoch-mean training loss, tolerance 1e-4, patience 20, cap 400 epochs) and reruns the
  identical grid, seeds and decision rule. Receipt path `receipt_v2.json`; the runner now refuses to
  overwrite an existing receipt. 847 tests green.

### Capacity campaign V2 complete — the backfill cannot power the 120-parameter fit, 2026-08-15.

- All 480 V2 trials completed under convergence training. Null discipline perfect: 0 entries across
  160 null trials at every size. Recovery of the small planted edge: 23%/23%/28%/40% at
  243/404/650/890 training sessions against the declared 80% — no tested size qualifies, and 890 is
  the largest prefix the proposed backfill can produce. Medium edge peaks at 70% (Wilson lower 57%)
  and never reaches requirement. Receipt `receipt_v2.json`; finding
  `research/findings/CAPACITY_KNOWN_ANSWER_2026_08_15.md`; PLAN item 16 updated to MEASURED.
- Consequences recorded: the row-43 full-corpus budget convention is falsified for this pipeline; the
  20-observations-per-parameter rule measured optimistic; the pending $75 backfill request loses its
  stated purpose until a design passes its own known-answer rehearsal. The failure mode is one-sided
  (silent misses, never fake passes), which is the referee's false-negative mechanism 1 measured.
- Drafted `PROTOCOL_V2_AMENDMENT_DRAFT_2026_08_15.md` (A1 known-answer power gate, A2 capacity by
  measurement, A3 exposure firewall, A4 serial breaker, A5 duration-matched exit control, A6 quote
  admissibility and decision clock, A7 clustered session-level estimand, A8 semantic freeze and
  closure scope). Awaiting owner signature; nothing adopted, no fit, no purchase, no vendor contact.
