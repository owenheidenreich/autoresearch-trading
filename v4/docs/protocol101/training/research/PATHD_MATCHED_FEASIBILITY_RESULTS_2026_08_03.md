# Matched Execution-Aware Feasibility Gate — Results (2026-08-03)

## Verdict

**`INVALID` — discard the family result.** All nine primary hypotheses are individually
`NOT_FEASIBLE`, but the frozen precedence is decisive: the H2-30 sign-reversed negative control clears
the unadjusted gate. It earns +$13.78 per opportunity, is positive in 5/5 folds, has one-sided
`p=0.00981`, and remains positive after removing any single session or any weekday. An accepted negative
control invalidates the run; it does not promote the reversed rule.

Therefore **neither branch warrants a forward live-paper test from this run**, and this invalid run cannot
close either branch permanently. A corrected gate would require a new pre-registration; the frozen
document and these results must remain intact.

## Safety and frozen scope

- No model was fit and no parameter was selected from the results.
- Only the 166 development OOF sessions were assigned folds; the protected firewall was not opened.
- No paid data, broker, paper-submit, promotion, runtime flag, or launch configuration was used.
- All nine frozen tests were included in one 20,000-replicate session-blocked sign-flip maxT family.
- Four ES roll changes were identified. The change session and adjacent sessions were excluded exactly as
  in Codex A2: 12 sessions named in the machine result.

Machine result:
`/Volumes/AR_TRADING_DATA/reports/pathd_matched_feasibility_2026_08_03/matched_feasibility_result.json`
with result SHA-256
`6c0984122e8e4d4a86622cac0582c6082706997bb542bbe936d8c7451411e7b8`.
A second complete execution produced the same hash.

## Operational interpretation fixed before result inspection

The frozen prose did not fully specify executable mechanics. The runner records these deterministic
interpretations in its result:

- 0DTE primary fill: post at the actionable arrival bid; fill at the first offer at least one option tick
  below that bid inside 60 seconds; no fill earns $0 per opportunity; hold from actual fill time; force
  flat at 15:55 ET; charge $3 round trip.
- H1 uses exactly 10:30–10:59 ET; H2 uses all candidate times.
- ES uses strict nonoverlapping trades: observe one prior horizon, choose momentum/reversion, hold one
  horizon, then make the next decision. Primary friction is the frozen two-tick $29.50 requirement.
- Net EV is pooled mean dollars per opportunity. Significance uses session means. “Not concentrated”
  means the mean remains positive after removing each session and after removing each weekday.
- The 0DTE session-shuffle is structurally invariant because H1 is a fixed wall-clock mask and H2 is an
  all-time mask. H2's constant control is also identical to H2. These weaknesses were reported, not
  repaired after freeze.

## Sanity check

The required H1 source-population check reproduces exactly from `oof_scores.parquet`:

| Candidates | Sessions | Gross mean | Positive folds | Fold means |
|---:|---:|---:|---:|---|
| 10,063 | 120 | +$5.9927 | 4/5 | +$9.84 / +$7.56 / −$17.29 / +$30.81 / +$6.41 |

The primary one-tick-penetration fill rate on the full candidate population is 65.38%. This is lower than
the earlier 73.92% measured on the 878 control trajectories and confirms that the control-population fill
constant was not portable.

## Nine primary tests

`EV` is pooled mean per opportunity. `Session EV` is the unweighted mean of session means used for the
t-test/maxT statistic. The difference is material for H1 because sessions contain unequal candidate
counts.

| Test | n | sessions | EV | Session EV | +folds | raw p | maxT p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H1 0DTE 30m | 10,063 | 120 | +$14.05 | −$18.50 | 1/5 | 0.8265 | 1.0000 | NOT_FEASIBLE |
| H1 0DTE 60m | 10,063 | 120 | +$39.63 | −$3.76 | 3/5 | 0.5454 | 0.99935 | NOT_FEASIBLE |
| H2 0DTE 30m | 156,950 | 166 | −$17.70 | −$17.04 | 0/5 | 0.9988 | 1.0000 | NOT_FEASIBLE |
| H2 0DTE 60m | 156,950 | 166 | −$13.58 | −$11.00 | 1/5 | 0.8065 | 1.0000 | NOT_FEASIBLE |
| H3 ES momentum 15m | 3,770 | 158 | −$39.05 | −$39.07 | 0/5 | >0.9999 | 1.0000 | NOT_FEASIBLE |
| H3 ES momentum 30m | 1,728 | 158 | −$29.73 | −$29.72 | 2/5 | 0.9624 | 1.0000 | NOT_FEASIBLE |
| H3 ES momentum 60m | 784 | 158 | −$30.28 | −$28.40 | 1/5 | 0.8304 | 1.0000 | NOT_FEASIBLE |
| H4 ES reversion 15m | 3,770 | 158 | −$19.03 | −$18.99 | 0/5 | 0.9947 | 1.0000 | NOT_FEASIBLE |
| H4 ES reversion 30m | 1,728 | 158 | −$28.59 | −$28.60 | 2/5 | 0.9565 | 1.0000 | NOT_FEASIBLE |

H1's positive pooled numbers do not satisfy the gate: the session-blocked means are negative, fold
stability fails, maxT fails, and both concentration checks fail.

## ES full friction band

No ES rule is positive even at the optimistic one-tick assumption:

| Test | 1 tick | 2 ticks (primary) | 3 ticks | 4 ticks |
|---|---:|---:|---:|---:|
| H3 momentum 15m | −$26.75 | −$39.05 | −$51.36 | −$63.66 |
| H3 momentum 30m | −$17.37 | −$29.73 | −$42.09 | −$54.44 |
| H3 momentum 60m | −$17.96 | −$30.28 | −$42.61 | −$54.93 |
| H4 reversion 15m | −$6.72 | −$19.03 | −$31.33 | −$43.63 |
| H4 reversion 30m | −$16.23 | −$28.59 | −$40.94 | −$53.30 |

The pending ES spread purchase is consequently **not a dependency of this gate**. Collapsing the band to
a measured point cannot make any preregistered ES rule positive because all five already fail at one tick.

## Negative controls

Each cell is pooled EV; `*` marks the control that clears the unadjusted gate.

| Test | Sign-reversed | Session-shuffled | Constant |
|---|---:|---:|---:|
| H1 30m | −$17.96 | +$14.05 | −$17.70 |
| H1 60m | −$43.55 | +$39.63 | −$13.58 |
| H2 30m | **+$13.78\*** | −$17.70 | −$17.70 |
| H2 60m | +$9.66 | −$13.58 | −$13.58 |
| H3 15m | −$19.03 | −$22.66 | −$29.90 |
| H3 30m | −$28.59 | −$33.39 | −$30.55 |
| H3 60m | −$27.89 | −$36.50 | −$35.75 |
| H4 15m | −$39.05 | −$42.84 | −$29.90 |
| H4 30m | −$29.73 | −$24.13 | −$30.55 |

The H2-30 reversed control has session-mean EV +$13.00, fold means
+$14.09/+$19.61/+$10.75/+$9.10/+$13.64, and raw p=0.00981. It is only a falsifier: it reuses a passive-buy
fill event and negates the subsequent price move, so it is not evidence of an executable passive short.

## Bottom line

**Family result: `INVALID`. Individual result: 9/9 `NOT_FEASIBLE`. Forward-test routing: neither branch.**

The run cannot honestly say “both branches close” because its own negative-control precedence discards the
family. It does say that none of the nine frozen rules earned a forward test, and that all five ES rules
lose even under the one-tick friction lower bound.

STOP_FOR_CLAUDE_VERIFICATION
