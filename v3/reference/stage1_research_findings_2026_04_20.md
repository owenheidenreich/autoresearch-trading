# Stage 1 Research Findings — 2026-04-20

> Research trail: [decay_analysis_2026_04_20.md](decay_analysis_2026_04_20.md) → [attribution_full_2026-04-20.txt](attribution_full_2026-04-20.txt) → **[this doc, #3]** → [omar_findings_2026_04_20.md](omar_findings_2026_04_20.md). See [INDEX.md](INDEX.md) for the full reading order.

## Scope

Full-dataset Stage 1 attribution run across 986 sessions, with two policy exit
baselines (time-stop = hold to session end; stop/target = −35% stop / +60%
target / time-stop whichever first). Raw output at
`v3/reference/attribution_full_2026-04-20.txt`. Miss-analysis script at
`v3/analysis/miss_analysis.py`.

## Headline results

### Feasibility (confirms earlier)

All four diagnostics within envelope across every stratum. Coverage 99.3%,
cap_bind_severity mean 34.6% / p95 45.3%, suppression 0.9%, low_delta_forcing
0.2%. Rails are not the bottleneck.

### Attribution — teacher outcomes on 986 oracle bars

| Outcome               | n   | %     |
|-----------------------|-----|-------|
| guardrail_suppression | 0   | 0.0%  |
| abstention            | 467 | 47.4% |
| side_error            | 402 | 40.8% |
| **entered_right**     | 117 | **11.9%** |

Teachers catch the session-best bar in the right direction **11.9% of the
time**. The other 88% is split roughly evenly between "teacher didn't trigger"
and "teacher triggered wrong direction."

### Dollar totals (on 117 right-direction entries only)

| Line                                  | $ total     | $ per trade |
|---------------------------------------|-------------|-------------|
| Oracle session PnL                    | $2,502,356  | $21,388     |
| Exit-headroom on selected contract    | $342,576    | $2,928      |
| Time-stop baseline (hold to end)      | $206,969    | $1,769      |
| Stop/target baseline (−35/+60)        | $87,138     | $744        |

Note: oracle session PnL is the session's hindsight-best trade, not
constrained to the teacher's selection. The other three are measured on the
teacher-selected contract at the oracle entry bar.

### Gaps (positive = room for improvement)

| Gap                         | $ total    | Interpretation                         |
|-----------------------------|------------|----------------------------------------|
| selection_gap               | $0         | teacher's contract tiebreak matches oracle's |
| exit_gap vs time-stop       | $135,607   | hindsight exit beats naive hold         |
| exit_gap vs stop/target     | $255,438   | hindsight exit beats mechanical rule    |
| **stop/target − time-stop** | **−$119,831** | **mechanical rule is WORSE than holding** |

## Three sharp findings

### Finding 1 — The fixed stop/target rule actively hurts PnL

On the 117 right-direction entries, the −35%/+60% mechanical rule realizes
$87k vs $207k for naive hold-to-end — a **$120k cost** from applying the
mechanical rule. This is the decay-analysis prediction playing out at scale:
5-9% of −60%-drawdown contracts recover, plus many profitable trades briefly
spike past +60% before settling much higher.

**Implication:** Layer 3 exit policy should NOT use fixed stop/target rules.
Even naive hold-to-end beats −35/+60 by $120k. The state-dependent exit
learned by a model has $135k more headroom on top of naive hold.

### Finding 2 — Contract selection is *usually* correct, but the $0 attribution finding was a measurement artifact

The attribution reporter's `selection_gap = $0` across all 117 right-direction
entries looked too clean, and it was. The mechanism:

- Selection_gap is only computed on `entered_right` bars — bars where the
  teacher *matched* the oracle's session-best trade in both bar and direction.
- Oracle bars are, by definition, clean directional moves — they're the
  session's best trade, meaning the underlying moved decisively one way.
- On clean directional moves, highest-delta-under-cap captures the most
  dollars per contract. Gamma/theta score also increases monotonically with
  delta, so the teacher's tiebreak picks the highest-delta passing contract.
- Therefore the teacher's pick and the oracle's pick are the *same contract*
  100% of the time on oracle bars, trivially. Not a finding — a tautology of
  the bar selection.

The honest picture comes from `selection_quality.py`, which measures the same
gap across ALL 44,877 teacher-entered bars (not just oracle bars).

**Honest selection-quality stats (full cache, Moderate rails):**

| Stratum              | n      | mean gap | median | p95    | max     | %>$1 | %>$50 |
|----------------------|--------|----------|--------|--------|---------|------|-------|
| all                  | 44,877 | $6.69    | $0.00  | $48.80 | $1,399  | 14%  | 5%    |
| teacher_orc          | 36,835 | $6.38    | $0.00  | $46.27 | $1,399  | 13%  | 5%    |
| teacher_failed_break | 8,042  | $8.14    | $0.00  | $59.49 | $923    | 15%  | 6%    |
| direction_call       | 23,716 | $5.97    | $0.00  | $42.63 | $1,399  | 12%  | 4%    |
| direction_put        | 21,161 | $7.50    | $0.00  | $56.44 | $387    | 15%  | 6%    |

Reading the numbers:
- **Median gap is $0** — on more than half of entries, gamma/theta does pick
  the forward-PnL-maximizing contract. The rule works most of the time.
- **14% of entries have a positive gap** — ~1 in 7 trades would have been
  better on a different strike.
- **Total selection leakage ≈ $300k** across the full cache (mean × entries).
  Comparable in magnitude to the exit-gap totals on oracle-bar entries.
- **FailedBreak is slightly worse at selection** ($8.14 vs $6.38 mean).
  Plausibly because FB triggers inside the range where direction is less
  decided, weakening the correlation between gamma/theta and forward PnL.
- **Put selection is slightly worse than call selection** ($7.50 vs $5.97).
  Smaller effect; worth noting but not urgent.

**Corrected implication:** The `gamma_dollar / theta_to_premium` tiebreak is a
good rule, not a perfect one. ~86% of entries get it right. The remaining 14%
represent a real but second-order opportunity — good candidate for a learned
Layer-3 selection rule later, but not a near-term priority. The near-term
bottleneck is still entry logic (88% of sessions miss the oracle bar entirely
via abstention or side error).

### Finding 3 — Teacher windows and conditions miss two distinct failure modes

The miss-analysis script pulls feature values at each oracle bar and splits
them by outcome group. Key columns:

```
feature              entered_right   abstention   side_error
oracle_minute               36           86           55
inside_f15 (median)          0            1            0
%neither_orc_nor_fb_gate    11%          65%           18%
```

Interpretation:

- **Abstention oracle bars are LATE in the session (median minute 86 ≈ 10:56 ET)
  and have close INSIDE the first-15 range** (inside_f15 median = 1). 65% of
  these bars have neither teacher's gate condition met at all.
  - These are late-session moves that don't match the "first-15 breakout +
    VWAP agree" pattern. Our teachers are designed for early-session
    directional continuation; they simply don't have a playbook for late
    moves from inside consolidation.

- **Side-error oracle bars are mid-window (median minute 55 ≈ 10:25 ET) and
  usually DO have the ORC gate met** (82% of side-errors fit an ORC call or
  put gate), but the direction the gate indicates is the opposite of what
  oracle says was actually profitable.
  - These are false-breakout cases: price breaks the first-15 range with
    VWAP agreement, but the end-of-session move is actually in the OPPOSITE
    direction. The ORC "break + VWAP-trend" combination does not reliably
    predict the session's ultimate direction.

### Finding 4 — FailedBreak teacher is almost irrelevant

Across all three outcome groups, the FailedBreak teacher's gate fires on only
5-9% of oracle bars. It does correctly identify some side-error setups (9.2%
of them in the call direction, 8.7% put), but as a standalone playbook it
contributes little to entered_right entries. Its lookback (5 bars) and
its "inside range after recent break" trigger are narrow.

## Implications for teacher rework

### Rework priority order

1. **Add a late-session / consolidation-break playbook.** Roughly half of all
   oracle-best bars fall in the 10:45-11:30 window inside a settled range.
   The current ORC and FailedBreak teachers do not handle this. A new teacher
   that watches for secondary breakouts off a post-opening consolidation
   would capture a large chunk of the 47% abstention rate.

2. **Strengthen directional inference in ORC.** On 82% of side-error bars,
   ORC's gate condition IS met but for the wrong direction. The
   "break + VWAP trend" combination is not enough. Candidate additions (do
   NOT hardcode as gates in Layer 0 — log them as inputs first):
   - Volume-by-direction divergence (did the breakout bar have balanced or
     one-sided flow?)
   - Prior-day close position
   - IV skew (put-call skew has directional information)
   - Time since the break occurred (late breaks in a day often reverse)

3. **Drop or merge the FailedBreak teacher.** At 5-9% gate-firing rate on
   oracle bars, it's not pulling its weight as a standalone playbook. Either
   merge its logic into ORC (as a "reject counter-trend" filter) or keep
   it as an experimental playbook with a wider lookback and extra signal
   requirements.

### Non-priorities (based on the evidence)

- Don't rework contract selection *yet*. Median selection gap is $0 and the
  tail is modest — 14% of entries have a positive gap, ~$300k total leakage
  across the full cache. Real but second-order. Layer-3 learned selection
  can revisit this later.
- Don't tighten Layer 0 rails (feasibility is fully in envelope).
- Don't add fixed stop-loss exit rules to the policy baseline — they cost
  money against simple holding.

## What this sets up for Stage 2

The attribution + miss-analysis infrastructure is reusable. Each new teacher
variant gets measured against the same 986-session oracle ceiling, with
directly comparable abstention / side_error / right-direction counts. The
question becomes mechanical: does the new teacher reduce abstention + side
error counts, or not?

Specifically: a successful teacher rework should produce one or both of:
- lower abstention count (teacher catches more oracle bars)
- lower side_error count (when it catches, it gets direction right)

Everything else stays fixed — the rails, the oracle, the contract-selection
tiebreak, the exit baselines.
