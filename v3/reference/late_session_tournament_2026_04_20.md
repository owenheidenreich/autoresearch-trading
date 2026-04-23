# Late-Session Tournament — W4 Verdict (2026-04-20)

**Workstream:** W4 first-pass tournament (2 detectors × 4 families) from the
layered implementation plan.

**Verdict:** **NO QUALIFIER.** The plan's stopping rule triggers. The late-
session regime is Layer-2-only for now. Capture it via the soft features
in W2 (post-G1); do **not** ship a late-session teacher in this cycle.

## Question

For the late-session regime (`minute ∈ [40, 120]` ∧ `inside_first15` ∧
squeeze `last10_range ≤ 1.0 × OMAR`), which detector + family combination
produces a directional edge usable as a teacher?

## Tournament

Two detectors × four families = 8 cells. Same data (986-day cache), same
20-bar forward-move horizon, same router K=3.

**Detectors**
- `A_legacy` (5-gate / narrow): adds the `sigma_pos` direction gate
  (≤ +0.5 for calls, ≥ −0.5 for puts) and OMAR retest (close within
  0.5 × OMAR.range of high/low/mid) on top of the base
  late-session-window + inside-first15 + squeeze gates. This is the
  universe used by `late_session_fakeout_split.py`; the 22/22 reverse-
  match evidence belongs only here.
- `B_wide` (3-gate / research): late-session window + inside-first15 +
  squeeze. `sigma_pos` / OMAR distance / VP context are router-input
  candidates only, not detector gates. Per Codex's Experiment 5 framing
  (matched-direction controls), this is the honest universe to test
  trigger-level edge on.

**Families**
1. `immediate`: enter at trigger bar in break direction.
2. `confirm_1bar`: enter at trigger+1 if still outside last10 range.
3. `failed_break_reversal`: enter at first re-entry within K bars,
   OPPOSITE direction.
4. `router`: at trigger+K, decide CLEAN vs FAKEOUT. CLEAN → continuation,
   FAKEOUT → reversal at the re-entry bar.

**Qualifying bar (per plan)**
- triggers/day in [0.3, 3.0]
- direction-match rate at oracle coincidences ≥ 60%
- `mfe20` median edge over matched control ≥ +3 bps

Matched control = random late-session inside-first15 bars × random
direction. Measured: `mfe20` median = **+7.20 bps**.

## Results

Headline (full table in [test8_late_session_tournament.py](../analysis/test8_late_session_tournament.py) output):

| Detector | Family | n_trig | per_day | n_oracle | dir_match | mfe20 med | edge vs ctrl | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A_legacy | immediate            | 1034 | 1.05 | 22 | **0/22 (0.0%)** | +6.61 | −0.59 | FAIL |
| A_legacy | confirm_1bar         |  691 | 0.70 | 11 | 0/11 (0.0%)     | +6.60 | −0.61 | FAIL |
| A_legacy | failed_break_reversal|  557 | 0.56 |  0 | n/a             | +6.05 | −1.16 | FAIL |
| A_legacy | router (pooled)      | 1034 | 1.05 |  8 | 0/8  (0.0%)     | +6.37 | −0.83 | FAIL |
| A_legacy | router_clean         |  477 | 0.48 |  8 | 0/8  (0.0%)     | +6.87 | −0.33 | (info) |
| A_legacy | router_fakeout       |  557 | 0.56 |  0 | n/a             | +6.05 | −1.16 | (info) |
| B_wide   | immediate            | 2328 | 2.36 | 44 | **0/44 (0.0%)** | +6.89 | −0.31 | FAIL |
| B_wide   | confirm_1bar         | 1673 | 1.70 | 31 | 0/31 (0.0%)     | +6.77 | −0.44 | FAIL |
| B_wide   | failed_break_reversal| 1122 | 1.14 |  0 | n/a             | +6.49 | −0.72 | FAIL |
| B_wide   | router (pooled)      | 2328 | 2.36 | 17 | 0/17 (0.0%)     | +6.61 | −0.60 | FAIL |
| B_wide   | router_clean         | 1206 | 1.22 | 17 | 0/17 (0.0%)     | +6.93 | −0.28 | (info) |
| B_wide   | router_fakeout       | 1122 | 1.14 |  0 | n/a             | +6.49 | −0.72 | (info) |

Every cell that has oracle coincidences has 0% direction match (the same
phenomenon `late_session_fakeout_split.py` and `test2_nr10_teacher_dryrun.py`
already surfaced — restated here on the wider universe). Every cell's
`mfe20` median is **below** matched-control random.

## Why nothing cleared

Three structural problems, in order of severity.

1. **Squeeze + inside-first15 anti-selects for forward volatility.**
   Random late-session inside-first15 bars produce `mfe20` median +7.20 bps.
   Trigger families operating on that same window (which adds the
   last10-range squeeze gate and a fresh break) produce 6.0–6.9 bps.
   The squeeze gate selects bars with low recent realized volatility, and
   the 20-bar forward window inherits that low-volatility regime. The
   filter is removing volatility, not adding edge.

2. **Delayed-entry families lose oracle coincidences mechanically.** The
   opportunity oracle picks a single best entry bar per day. `immediate`
   enters at the trigger bar, so coincidences happen when the trigger bar
   IS the oracle bar (22 cases on Detector A; 44 on Detector B).
   `confirm_1bar` / `failed_break_reversal` / `router_clean` enter at
   trigger+1 / first re-entry / trigger+K, which mostly are NOT the
   oracle bar. So those families have very few oracle coincidences (0–11
   on Detector A, 0–31 on Detector B), and `direction_match` is undefined
   or 0% by mechanical sample-size, not by signal.

3. **The 22/22 reverse-match was a within-cohort property, not a trigger
   property.** `late_session_fakeout_split.py` reported all 22 oracle
   coincidences had break direction OPPOSITE oracle direction. The
   failed-break reversal family enters in the OPPOSITE direction — but
   at re-entry bars (trigger+1..trigger+3), not the oracle bar itself.
   Its `mfe20` (+6.05 bps) is still BELOW matched control. The reversal
   direction is correct in spirit; the entry timing dilutes the signal
   into the surrounding noise. There's an oracle bar where the right
   direction was clear; reversing AROUND it does not recover the move.

## Implications for the plan

- **W4 stops here.** No detector/family combination cleared the
  full-sample qualifying bar. Walk-forward (test9) cannot redeem cells
  that fail in aggregate, so it is not run.
- **Late-session regime is Layer-2-only.** The features that locate this
  regime (`omar_retest_dist_norm`, `last10_range_over_omar`,
  `inside_first15`, `late_window_40_120_flag`, etc.) are still real
  cohort-level structure, and they will be available to the v2 model
  when W2 unblocks (post-G1). The model can learn to weight them in
  ways the rule-based families cannot.
- **`break-then-VWAP-reclaim reversal` and `second-test continuation`
  are NOT revisited.** They were intentionally dropped from the
  first-pass tournament to avoid a winner-by-chance fishing expedition.
  Per the plan, they are revisited only if a first-pass family wins.
  None did. They stay dropped.
- **No new dry-run test scripts.** Inventing more trigger families on
  the same data is exactly the failure mode the plan's stopping rule
  exists to prevent. The next late-session research should come from a
  different question (e.g., a different conditioning event, a different
  forward-move target, or the Layer-2 model's residuals — not another
  trigger variant).

## What this does NOT change

- A1 (ORC + sigma_pos hard gate) stays shipped and validated. The
  side_error 402 → 176 / entered_right 117 → 84 result is independent
  of W4.
- W2 still runs once G1 (integer-contract reconciliation) clears. The
  proposed 8 features absorb the late-session structure into the model
  layer.
- W3 (localization vs trigger_quality reporters) is the methodology that
  caught this cleanly. Without the matched-control denominator and the
  forward-move-edge bar, this verdict would have looked like "0/22
  direction match, must be a measurement bug." With them, it is the
  verdict the data actually supports: the late-session regime is real
  but not trigger-addressable in any of the four templates tested.

## Code

- [test8_late_session_tournament.py](../analysis/test8_late_session_tournament.py)
  — the tournament script. 2 × 4 grid + router branch breakdown. Reusable
  by re-running on a different oracle index or detector definition; not
  re-run in this cycle.
- [late_session_trigger_families.py](../analysis/late_session_trigger_families.py)
  — Codex's prior partial answer. test8 supersedes it (same families, both
  detectors, plus router and stopping bar).
- [late_session_fakeout_split.py](../analysis/late_session_fakeout_split.py)
  — the 22/22 reverse-match evidence. Restated and explained above; the
  reversal direction is correct on the cohort but does not transfer to
  trigger-level edge.

## Reproducing

```
python -m v3.analysis.test8_late_session_tournament
```
Reads the latest `attribution_full_*.txt` for the oracle index. Output
matches this doc on the current 986-day cache (post-A1 baseline).
