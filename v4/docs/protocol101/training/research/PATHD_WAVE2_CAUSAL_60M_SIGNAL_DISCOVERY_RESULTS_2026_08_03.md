# Path-D Wave 2 — Causal 60-Minute Signal Discovery Results

## Verdict

**`NO_SIGNAL`**. None of the three frozen non-fitted scores earned the model-training gate. The result is
valid: all nine negative-control replays failed, every causal/identity/reproduction audit passed, and the
spent historical firewall remained closed.

Machine packet:

`/Volumes/AR_TRADING_DATA/artifacts/pathd_wave2_signal_discovery_2026_08_03_retry2/results.json`

Semantic result SHA-256:

`9c3549ed52e6dc373f3419bacd4bdc2c690d1539af226aea2a11e5c26bd3e095`

No model was fit. No paid data, broker, paper-order, runtime, launch, promotion, or default path was used.

## Frozen scope and source population

- Pre-registration: `PATHD_WAVE2_CAUSAL_60M_SIGNAL_DISCOVERY_PREREGISTRATION_2026_08_03.md`.
- Evaluated population: exactly 156,950 governed candidates across the 166 sessions in the five
  entry-v2 OOF test manifests: 34 / 33 / 33 / 33 / 33 sessions by fold.
- No row from the spent 36-session firewall was decoded (`protected_holdout_opened=false`).
- Score coverage after causal history requirements:
  - surface continuation: 80,962 rows / 161 sessions / five folds;
  - option microstructure: 140,122 rows / 163 sessions / five folds;
  - cross-market alignment: 119,417 rows / 155 sessions / five folds.
- W2-H03 excluded eight ES roll, adjacent-roll, or empty-partition sessions. Filled rows without an
  actionable exit observation within two seconds of the exact 60-minute target were unavailable.
- Primary economics used one contract, one position, one chronological $10,000 account, affordability,
  a 5% realized daily stop, passive one-tick-penetration entry, actual fill time, exact 60-minute bid
  exit, and $1.50 per filled side.

Two initial executions returned `BLOCKED_CLOCK_OR_DATA_CONTRACT` before replay or inference. They are
preserved as audit evidence. The first exposed an empty ES roll-day partition; the second exposed stale
and early-close rows whose stored exit was not a true 60-minute observation. Both repairs were mechanical,
were frozen before any economic result, and did not change a score, sign, weight, threshold, horizon, or
acceptance gate.

## Primary family

The best preregistered comparator was the passive 10:30–10:59 ET 60-minute rule: +$7,053 total over 119
filled trades. It is only a benchmark, not an accepted hypothesis; the replay also reports a $9,329
maximum drawdown from $10,000 and it received none of the Wave-2 statistical or risk gates. Passive
all-time lost $9,833; not trading was $0.

| Member | Net PnL | Delta vs best | +folds | 95% LCB | maxT p | Filled trades / sessions / folds | Top decile | Big-loss share | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| W2-H01 full-ladder surface continuation | -$9,821 | -$16,874 | 2/5 | -$227.41 | 0.962752 | 117 / 72 / 3 | not best; -$27.13 vs bottom -$26.62 | 59.83% | FAIL |
| W2-H02 option microstructure continuation | -$9,856 | -$16,909 | 1/5 | -$259.53 | 0.907205 | 267 / 134 / 5 | not best; -$9.76 vs bottom -$14.12 | 58.43% | FAIL |
| W2-H03 cross-market alignment | -$9,904 | -$16,957 | 2/5 | -$248.30 | 0.931903 | 113 / 63 / 3 | not best; -$13.76 vs bottom -$4.17 | 58.41% | FAIL |

All three failed the central economic gate, the four-of-five fold gate, positive bootstrap LCB, maxT,
decile ordering, and the charter's 2% big-loss cap. Surface and cross-market also failed the all-five-fold
filled-trade requirement after their accounts lost affordability. The positive-delta concentration checks
passed, but cannot rescue negative total paired deltas.

## Negative controls

No control was accepted, so the family is not `INVALID`.

| Member | Sign-reversed net / delta | Session-shuffled net / delta | Constant net / delta |
|---|---:|---:|---:|
| W2-H01 | -$9,957 / -$17,010 | -$9,886 / -$16,939 | -$9,833 / -$16,886 |
| W2-H02 | +$1,146 / -$5,907 | -$9,720 / -$16,773 | -$9,833 / -$16,886 |
| W2-H03 | -$9,735 / -$16,788 | -$270 / -$7,323 | -$9,833 / -$16,886 |

The reversed microstructure score was positive versus $0, but it still lost the best comparator by
$5,907, had only 2/5 positive paired folds, a -$194.20 bootstrap LCB, and raw one-sided p=0.647. It is a
failed control, not a new candidate.

## Charter diagnostics and 30-minute diagnostic

The realized/peak harvest ratios were negative for every member. Maximum drawdowns were $10,943,
$11,241, and $11,335 respectively; the one-account affordability guard left ending equities of $179,
$144, and $96. These are economic failures, not near misses.

At 30 minutes all three members lost to not trading, were positive in 0/5 folds, and had negative
bootstrap LCBs:

| Member | 30-minute net | Best comparator | +folds | 95% LCB |
|---|---:|---|---:|---:|
| W2-H01 | -$9,869 | not trading | 0/5 | -$101.44 |
| W2-H02 | -$9,839 | not trading | 0/5 | -$87.55 |
| W2-H03 | -$9,721 | not trading | 0/5 | -$93.34 |

## Audits and outputs

- Causal-clock violations: 0 / 0 / 0 by member.
- Future-outcome mutation: pass.
- Candidate identity: 156,950 rows = 156,950 unique UIDs; complete one-to-one outcome join.
- Persisted candidate reproduction seal: matched
  `751a5041f67d6a8431710b26608eb9ab907d4c1ce9886acf9c723a25d54d5c99`.
- Forbidden state before/after: unchanged.
- SSD post-run preflight: pass; 98.12% free and allocation below the 150 GB cap.

The packet contains the freeze receipt, feature-lineage/clock manifest, 42 MB scored candidate dataset,
replay attempts/trades, maxT packet, controls, side-effect audit, JSON result, and human report. No model-
training goal was drafted because the gate did not pass.

## Do-not-retest ruling

Do not rerun these exact surface-continuation, option-microstructure-continuation, or SPX/ES/VIX
cross-market composites on the entry-v2 OOF corpus with new weights, signs, thresholds, adjacent time
windows, or seeds. A future entry study requires a materially different causal observable, economic
target, position structure, or trading game and a new bounded preregistration. This result does not
authorize model training.

STOP_FOR_CLAUDE_VERIFICATION
